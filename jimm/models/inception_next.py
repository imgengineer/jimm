"""InceptionNeXt in flax nnx, NHWC. Mirrors timm.models.inception_next (MetaNeXt).

Each block splits its channels between an identity branch and depthwise 3x3,
1xK, and Kx1 convolutions, then applies BatchNorm, a 1x1-conv MLP, and layer
scale. Stages downsample with BatchNorm and a 2x2 stride-2 convolution; the
head is an MLP (fc, GELU, LayerNorm, fc).
"""

import jax.numpy as jnp
from flax import nnx

from ..layers import BatchNorm, ClassifierMixin, DropPath, Mlp, global_pool_nhwc
from ..registry import _cfg, register_model

_init = nnx.initializers.truncated_normal(0.02)


def _dwconv(chs, kernel, *, rngs):
    return nnx.Conv(
        chs,
        chs,
        kernel,
        padding=tuple((k // 2, k // 2) for k in kernel),
        feature_group_count=chs,
        kernel_init=_init,
        rngs=rngs,
    )


class InceptionDWConv2d(nnx.Module):
    def __init__(self, chs, square_kernel=3, band_kernel=11, branch_ratio=0.125, *, rngs):
        gc = int(chs * branch_ratio)
        self.split = (chs - 3 * gc, chs - 2 * gc, chs - gc)
        self.dwconv_hw = _dwconv(gc, (square_kernel, square_kernel), rngs=rngs)
        self.dwconv_w = _dwconv(gc, (1, band_kernel), rngs=rngs)
        self.dwconv_h = _dwconv(gc, (band_kernel, 1), rngs=rngs)

    def __call__(self, x):
        a, b, c = self.split
        return jnp.concatenate(
            [
                x[..., :a],
                self.dwconv_hw(x[..., a:b]),
                self.dwconv_w(x[..., b:c]),
                self.dwconv_h(x[..., c:]),
            ],
            axis=-1,
        )


class MetaNeXtBlock(nnx.Module):
    def __init__(self, dim, mlp_ratio=4.0, ls_init_value=1e-6, drop_path=0.0, mixer=None, *, rngs):
        self.token_mixer = InceptionDWConv2d(dim, **(mixer or {}), rngs=rngs)
        self.norm = BatchNorm(dim, epsilon=1e-5, rngs=rngs)
        self.mlp = Mlp(dim, int(mlp_ratio * dim), kernel_init=_init, rngs=rngs)
        self.gamma = nnx.Param(jnp.full((dim,), ls_init_value)) if ls_init_value else None
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        y = self.mlp(self.norm(self.token_mixer(x)))
        if self.gamma is not None:
            y = y * self.gamma[...]
        return x + self.drop_path(y)


class MetaNeXtStage(nnx.Module):
    def __init__(
        self, in_chs, out_chs, depth, stride=2, mlp_ratio=4.0, dpr=None, mixer=None, *, rngs
    ):
        if stride > 1:
            self.downsample_norm = BatchNorm(in_chs, epsilon=1e-5, rngs=rngs)
            self.downsample = nnx.Conv(
                in_chs,
                out_chs,
                (2, 2),
                strides=(2, 2),
                padding="VALID",
                kernel_init=_init,
                rngs=rngs,
            )
        else:
            self.downsample_norm = self.downsample = None
        dpr = dpr or [0.0] * depth
        self.blocks = nnx.List(
            [MetaNeXtBlock(out_chs, mlp_ratio, drop_path=r, mixer=mixer, rngs=rngs) for r in dpr]
        )

    def __call__(self, x):
        if self.downsample is not None:
            x = self.downsample(self.downsample_norm(x))
        for blk in self.blocks:
            x = blk(x)
        return x


class InceptionNeXt(ClassifierMixin, nnx.Module):
    _classifier_attr = "fc"

    def __init__(
        self,
        dims=(96, 192, 384, 768),
        depths=(3, 3, 9, 3),
        mlp_ratios=(4, 4, 4, 3),
        mixer=None,
        head_mlp_ratio=3,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.stem = nnx.Conv(
            in_chans, dims[0], (4, 4), strides=(4, 4), padding="VALID", kernel_init=_init, rngs=rngs
        )
        self.stem_norm = BatchNorm(dims[0], epsilon=1e-5, rngs=rngs)
        total = sum(depths)
        dpr = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]
        stages, prev = [], dims[0]
        for i, (dim, depth) in enumerate(zip(dims, depths)):
            stage_dpr = dpr[sum(depths[:i]) : sum(depths[: i + 1])]
            stages.append(
                MetaNeXtStage(
                    prev, dim, depth, 2 if i else 1, mlp_ratios[i], stage_dpr, mixer, rngs=rngs
                )
            )
            prev = dim
        self.stages = nnx.List(stages)
        # timm MlpClassifierHead: pool, fc, GELU, LayerNorm, dropout, fc. Pre-logits
        # features (and num_features) have the hidden width, timm's head_hidden_size.
        hidden = int(head_mlp_ratio * prev)
        self.num_features = hidden
        self.head_fc1 = nnx.Linear(prev, hidden, kernel_init=_init, rngs=rngs)
        self.head_norm = nnx.LayerNorm(hidden, epsilon=1e-6, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = (
            nnx.Linear(hidden, num_classes, kernel_init=_init, rngs=rngs)
            if num_classes > 0
            else None
        )

    def forward_features(self, x):
        x = self.stem_norm(self.stem(x))
        for stage in self.stages:
            x = stage(x)
        return x

    def forward_head(self, x):
        x = global_pool_nhwc(x, self.global_pool)
        x = self.head_drop(self.head_norm(nnx.gelu(self.head_fc1(x), approximate=False)))
        return self.fc(x) if self.fc is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_ATTO_MIXER = dict(band_kernel=9, branch_ratio=0.25)
_CFGS = {
    "inception_next_atto": ((40, 80, 160, 320), (2, 2, 6, 2), _ATTO_MIXER),
    "inception_next_tiny": ((96, 192, 384, 768), (3, 3, 9, 3), None),
    "inception_next_small": ((96, 192, 384, 768), (3, 3, 27, 3), None),
    "inception_next_base": ((128, 256, 512, 1024), (3, 3, 27, 3), None),
}


def _make(name):
    dims, depths, mixer = _CFGS[name]

    def entry(**kwargs):
        model = InceptionNeXt(dims, depths, mixer=mixer, **kwargs)
        model.default_cfg = _cfg(interpolation="bicubic")
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
