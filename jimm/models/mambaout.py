"""MambaOut in flax nnx, NHWC. Mirrors timm.models.mambaout.

Gated CNN blocks: LayerNorm, a linear expansion split into a gate and a value
whose last ``conv_ratio * dim`` channels pass through a 7x7 depthwise
convolution, then a projection back to ``dim``. The head pools, normalizes,
and applies an MLP (fc, GELU, LayerNorm) before the classifier.
"""

import jax.numpy as jnp
from flax import nnx

from ..layers import ClassifierMixin, DropPath, gelu, global_pool_nhwc
from ..registry import _cfg, register_model

_init = nnx.initializers.truncated_normal(0.02)


def _layer_norm(dim, *, rngs):
    return nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)


def _conv3x3_s2(in_chs, out_chs, *, rngs):
    return nnx.Conv(
        in_chs,
        out_chs,
        (3, 3),
        strides=(2, 2),
        padding=((1, 1), (1, 1)),
        kernel_init=_init,
        rngs=rngs,
    )


class GatedConvBlock(nnx.Module):
    def __init__(
        self, dim, expansion_ratio=8 / 3, kernel=7, conv_ratio=1.0, drop_path=0.0, *, rngs
    ):
        hidden = int(expansion_ratio * dim)
        conv_chs = int(conv_ratio * dim)
        self.split = (hidden, 2 * hidden - conv_chs)
        self.norm = _layer_norm(dim, rngs=rngs)
        self.fc1 = nnx.Linear(dim, hidden * 2, kernel_init=_init, rngs=rngs)
        self.conv = nnx.Conv(
            conv_chs,
            conv_chs,
            (kernel, kernel),
            padding=((kernel // 2, kernel // 2),) * 2,
            feature_group_count=conv_chs,
            kernel_init=_init,
            rngs=rngs,
        )
        self.fc2 = nnx.Linear(hidden, dim, kernel_init=_init, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        y = self.fc1(self.norm(x))
        a, b = self.split
        g, i, c = y[..., :a], y[..., a:b], y[..., b:]
        y = self.fc2(gelu(g) * jnp.concatenate([i, self.conv(c)], axis=-1))
        return x + self.drop_path(y)


class MambaOutStage(nnx.Module):
    def __init__(self, dim, dim_out, depth, downsample=False, dpr=None, *, rngs):
        if downsample:
            self.downsample = _conv3x3_s2(dim, dim_out, rngs=rngs)
            self.downsample_norm = _layer_norm(dim_out, rngs=rngs)
        else:
            self.downsample = self.downsample_norm = None
        dpr = dpr or [0.0] * depth
        self.blocks = nnx.List([GatedConvBlock(dim_out, drop_path=r, rngs=rngs) for r in dpr])

    def __call__(self, x):
        if self.downsample is not None:
            x = self.downsample_norm(self.downsample(x))
        for blk in self.blocks:
            x = blk(x)
        return x


class MambaOut(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        dims=(96, 192, 384, 576),
        depths=(3, 3, 9, 3),
        head_mlp_ratio=4,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.stem_conv1 = _conv3x3_s2(in_chans, dims[0] // 2, rngs=rngs)
        self.stem_norm1 = _layer_norm(dims[0] // 2, rngs=rngs)
        self.stem_conv2 = _conv3x3_s2(dims[0] // 2, dims[0], rngs=rngs)
        self.stem_norm2 = _layer_norm(dims[0], rngs=rngs)
        total = sum(depths)
        dpr = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]
        stages, prev = [], dims[0]
        for i, (dim, depth) in enumerate(zip(dims, depths)):
            stage_dpr = dpr[sum(depths[:i]) : sum(depths[: i + 1])]
            stages.append(
                MambaOutStage(prev, dim, depth, downsample=i > 0, dpr=stage_dpr, rngs=rngs)
            )
            prev = dim
        self.stages = nnx.List(stages)
        # timm MlpHead: pool, LayerNorm, fc, GELU, LayerNorm, dropout, fc. Pre-logits
        # features (and num_features) have the hidden width, timm's head_hidden_size.
        hidden = int(head_mlp_ratio * prev)
        self.num_features = hidden
        self.head_norm = _layer_norm(prev, rngs=rngs)
        self.head_fc1 = nnx.Linear(prev, hidden, kernel_init=_init, rngs=rngs)
        self.head_norm2 = _layer_norm(hidden, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = (
            nnx.Linear(hidden, num_classes, kernel_init=_init, rngs=rngs)
            if num_classes > 0
            else None
        )

    def forward_features(self, x):
        x = gelu(self.stem_norm1(self.stem_conv1(x)))
        x = self.stem_norm2(self.stem_conv2(x))
        for stage in self.stages:
            x = stage(x)
        return x

    def forward_head(self, x):
        x = self.head_norm(global_pool_nhwc(x, self.global_pool))
        x = self.head_drop(self.head_norm2(gelu(self.head_fc1(x))))
        return self.fc(x) if self.fc is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_CFGS = {
    "mambaout_femto": ((48, 96, 192, 288), (3, 3, 9, 3)),
    "mambaout_kobe": ((48, 96, 192, 288), (3, 3, 15, 3)),
    "mambaout_tiny": ((96, 192, 384, 576), (3, 3, 9, 3)),
    "mambaout_small": ((96, 192, 384, 576), (3, 4, 27, 3)),
    "mambaout_base": ((128, 256, 512, 768), (3, 4, 27, 3)),
}


def _make(name):
    dims, depths = _CFGS[name]

    def entry(**kwargs):
        model = MambaOut(dims, depths, **kwargs)
        model.default_cfg = _cfg(crop_pct=1.0, interpolation="bicubic")
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
