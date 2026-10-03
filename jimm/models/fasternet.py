"""FasterNet in flax nnx, NHWC. Mirrors timm.models.fasternet.

Each block mixes space with a partial convolution (a 3x3 convolution over the
first quarter of the channels, the rest passed through) and channels with a
BatchNorm 1x1-conv MLP, added to the input. Stages after the first start
with a 2x2 strided convolution that doubles the width; the head pools,
projects to 1,280 channels, and classifies.
"""

import jax.numpy as jnp
from flax import nnx

from ..layers import BatchNorm, ClassifierMixin, DropPath, gelu, global_pool_nhwc
from ..registry import _cfg, register_model

_init = nnx.initializers.truncated_normal(0.02)


def _conv(in_chs, out_chs, kernel, stride=1, padding="VALID", *, rngs):
    return nnx.Conv(
        in_chs,
        out_chs,
        (kernel, kernel),
        strides=(stride, stride),
        padding=padding,
        use_bias=False,
        kernel_init=_init,
        rngs=rngs,
    )


class PartialConv3(nnx.Module):
    def __init__(self, dim, n_div, *, rngs):
        self.dim_conv3 = dim // n_div
        self.partial_conv3 = _conv(
            self.dim_conv3, self.dim_conv3, 3, padding=((1, 1), (1, 1)), rngs=rngs
        )

    def __call__(self, x):
        x1 = self.partial_conv3(x[..., : self.dim_conv3])
        return jnp.concatenate([x1, x[..., self.dim_conv3 :]], axis=-1)


class MLPBlock(nnx.Module):
    def __init__(self, dim, n_div, mlp_ratio, drop_path, ls_init, act, *, rngs):
        hidden = int(dim * mlp_ratio)
        self.spatial_mixing = PartialConv3(dim, n_div, rngs=rngs)
        self.mlp_fc1 = _conv(dim, hidden, 1, rngs=rngs)
        self.mlp_norm = BatchNorm(hidden, epsilon=1e-5, rngs=rngs)
        self.mlp_fc2 = _conv(hidden, dim, 1, rngs=rngs)
        self.act = act
        self.layer_scale = nnx.Param(jnp.full((dim,), ls_init)) if ls_init > 0 else None
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        y = self.spatial_mixing(x)
        y = self.mlp_fc2(self.act(self.mlp_norm(self.mlp_fc1(y))))
        if self.layer_scale is not None:
            y = self.layer_scale[...] * y
        return x + self.drop_path(y)


class PatchMerging(nnx.Module):
    def __init__(self, dim, patch_size=2, *, rngs):
        self.reduction = _conv(dim, 2 * dim, patch_size, patch_size, rngs=rngs)
        self.norm = BatchNorm(2 * dim, epsilon=1e-5, rngs=rngs)

    def __call__(self, x):
        return self.norm(self.reduction(x))


class FasterNetStage(nnx.Module):
    def __init__(self, dim, depth, n_div, mlp_ratio, drop_path, ls_init, act, merge, *, rngs):
        self.downsample = PatchMerging(dim // 2, rngs=rngs) if merge else None
        self.blocks = nnx.List(
            [
                MLPBlock(dim, n_div, mlp_ratio, drop_path[i], ls_init, act, rngs=rngs)
                for i in range(depth)
            ]
        )

    def __call__(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class FasterNet(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        embed_dim=96,
        depths=(1, 2, 8, 2),
        mlp_ratio=2.0,
        n_div=4,
        feature_dim=1280,
        layer_scale_init_value=0.0,
        act="relu",
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.1,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.act = gelu if act == "gelu" else nnx.relu
        self.patch_embed_proj = _conv(in_chans, embed_dim, 4, 4, rngs=rngs)
        self.patch_embed_norm = BatchNorm(embed_dim, epsilon=1e-5, rngs=rngs)
        # timm calculate_drop_path_rates(stagewise=True): linear over all blocks, split by stage.
        total = sum(depths)
        rates = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]
        stages = []
        for i, depth in enumerate(depths):
            start = sum(depths[:i])
            stages.append(
                FasterNetStage(
                    embed_dim * 2**i,
                    depth,
                    n_div,
                    mlp_ratio,
                    rates[start : start + depth],
                    layer_scale_init_value,
                    self.act,
                    merge=i > 0,
                    rngs=rngs,
                )
            )
        self.stages = nnx.List(stages)
        # The head's 1x1 convolution runs on pooled features, i.e. a bias-free linear layer.
        self.conv_head = nnx.Linear(
            embed_dim * 2 ** (len(depths) - 1),
            feature_dim,
            use_bias=False,
            kernel_init=_init,
            rngs=rngs,
        )
        self.num_features = feature_dim
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = (
            nnx.Linear(feature_dim, num_classes, kernel_init=_init, rngs=rngs)
            if num_classes > 0
            else None
        )

    def forward_features(self, x):
        x = self.patch_embed_norm(self.patch_embed_proj(x))
        for stage in self.stages:
            x = stage(x)
        return x

    def forward_head(self, x):
        x = self.act(self.conv_head(global_pool_nhwc(x, self.global_pool)))
        x = self.head_drop(x)
        return self.fc(x) if self.fc is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_CFGS = {  # embed_dim, depths, drop_path_rate, activation
    "fasternet_t0": (40, (1, 2, 8, 2), 0.0, "gelu"),
    "fasternet_t1": (64, (1, 2, 8, 2), 0.02, "gelu"),
    "fasternet_t2": (96, (1, 2, 8, 2), 0.05, "relu"),
    "fasternet_s": (128, (1, 2, 13, 2), 0.1, "relu"),
    "fasternet_m": (144, (3, 4, 18, 3), 0.2, "relu"),
    "fasternet_l": (192, (3, 4, 18, 3), 0.3, "relu"),
}


def _make(name):
    embed_dim, depths, drop_path_rate, act = _CFGS[name]

    def entry(**kwargs):
        kwargs.setdefault("drop_path_rate", drop_path_rate)
        model = FasterNet(embed_dim, depths, act=act, **kwargs)
        model.default_cfg = _cfg(crop_pct=1.0, interpolation="bicubic")
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
