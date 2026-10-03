"""FocalNet in flax nnx, NHWC. Mirrors timm.models.focalnet.

Focal modulation replaces self-attention: a 1x1 convolution produces a
query, a context map, and per-level gates; the context passes through a
stack of depthwise convolutions with growing kernels (plus a global average)
whose gated outputs are summed, projected by a 1x1 convolution, and
multiplied into the query. Blocks are pre-norm (post-norm with layer scale
for the large models) with 1x1-conv MLPs; strided convolutions with
LayerNorm embed patches and downsample between stages.
"""

import jax.numpy as jnp
from flax import nnx

from ..layers import ClassifierMixin, DropPath, Mlp, gelu, global_pool_nhwc
from ..registry import _cfg, register_model

_init = nnx.initializers.truncated_normal(0.02)


class FocalModulation(nnx.Module):
    def __init__(
        self,
        dim,
        focal_window,
        focal_level,
        focal_factor=2,
        use_post_norm=False,
        normalize_modulator=False,
        drop=0.0,
        *,
        rngs,
    ):
        self.focal_level = focal_level
        self.normalize_modulator = normalize_modulator
        self.f = nnx.Linear(dim, 2 * dim + focal_level + 1, kernel_init=_init, rngs=rngs)
        self.h = nnx.Linear(dim, dim, kernel_init=_init, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, kernel_init=_init, rngs=rngs)
        self.drop = nnx.Dropout(drop, rngs=rngs)
        layers = []
        for k in range(focal_level):
            kernel = focal_factor * k + focal_window
            pad = kernel // 2
            layers.append(
                nnx.Conv(
                    dim,
                    dim,
                    (kernel, kernel),
                    padding=((pad, pad), (pad, pad)),
                    feature_group_count=dim,
                    use_bias=False,
                    kernel_init=_init,
                    rngs=rngs,
                )
            )
        self.focal_layers = nnx.List(layers)
        self.norm = nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs) if use_post_norm else None

    def __call__(self, x):
        C = x.shape[-1]
        x = self.f(x)
        q, ctx, gates = x[..., :C], x[..., C : 2 * C], x[..., 2 * C :]
        ctx_all = 0
        for level, layer in enumerate(self.focal_layers):
            ctx = gelu(layer(ctx))
            ctx_all = ctx_all + ctx * gates[..., level : level + 1]
        ctx_global = gelu(jnp.mean(ctx, axis=(1, 2), keepdims=True))
        ctx_all = ctx_all + ctx_global * gates[..., self.focal_level :]
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)
        x = q * self.h(ctx_all)
        if self.norm is not None:
            x = self.norm(x)
        return self.drop(self.proj(x))


class FocalNetBlock(nnx.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        focal_level=1,
        focal_window=3,
        use_post_norm=False,
        use_post_norm_in_modulation=False,
        normalize_modulator=False,
        layerscale_value=None,
        drop=0.0,
        drop_path=0.0,
        *,
        rngs,
    ):
        self.use_post_norm = use_post_norm
        self.norm1 = nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)
        self.modulation = FocalModulation(
            dim,
            focal_window,
            focal_level,
            use_post_norm=use_post_norm_in_modulation,
            normalize_modulator=normalize_modulator,
            drop=drop,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop, kernel_init=_init, rngs=rngs)
        if layerscale_value is not None:
            self.ls1 = nnx.Param(jnp.full((dim,), layerscale_value))
            self.ls2 = nnx.Param(jnp.full((dim,), layerscale_value))
        else:
            self.ls1 = self.ls2 = None
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        # Pre-norm blocks normalize inputs; post-norm blocks normalize branch outputs instead.
        if self.use_post_norm:
            y = self.norm1(self.modulation(x))
        else:
            y = self.modulation(self.norm1(x))
        if self.ls1 is not None:
            y = self.ls1[...] * y
        x = x + self.drop_path(y)
        y = self.norm2(self.mlp(x)) if self.use_post_norm else self.mlp(self.norm2(x))
        if self.ls2 is not None:
            y = self.ls2[...] * y
        return x + self.drop_path(y)


class Downsample(nnx.Module):
    def __init__(self, in_chs, out_chs, stride=4, overlap=False, *, rngs):
        kernel, pad = stride, 0
        if overlap:
            kernel, pad = (7, 2) if stride == 4 else (3, 1)
        self.proj = nnx.Conv(
            in_chs,
            out_chs,
            (kernel, kernel),
            strides=(stride, stride),
            padding=((pad, pad), (pad, pad)),
            kernel_init=_init,
            rngs=rngs,
        )
        self.norm = nnx.LayerNorm(out_chs, epsilon=1e-5, rngs=rngs)

    def __call__(self, x):
        return self.norm(self.proj(x))


class FocalNetStage(nnx.Module):
    def __init__(
        self,
        dim,
        out_dim,
        depth,
        mlp_ratio=4.0,
        downsample=True,
        focal_level=1,
        focal_window=1,
        use_overlap_down=False,
        use_post_norm=False,
        use_post_norm_in_modulation=False,
        normalize_modulator=False,
        layerscale_value=None,
        drop=0.0,
        drop_path=None,
        *,
        rngs,
    ):
        self.downsample = (
            Downsample(dim, out_dim, 2, use_overlap_down, rngs=rngs) if downsample else None
        )
        drop_path = drop_path or [0.0] * depth
        self.blocks = nnx.List(
            [
                FocalNetBlock(
                    out_dim,
                    mlp_ratio,
                    focal_level,
                    focal_window,
                    use_post_norm,
                    use_post_norm_in_modulation,
                    normalize_modulator,
                    layerscale_value,
                    drop,
                    drop_path[i],
                    rngs=rngs,
                )
                for i in range(depth)
            ]
        )

    def __call__(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class FocalNet(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"

    def __init__(
        self,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        mlp_ratio=4.0,
        focal_levels=(2, 2, 2, 2),
        focal_windows=(3, 3, 3, 3),
        use_overlap_down=False,
        use_post_norm=False,
        use_post_norm_in_modulation=False,
        normalize_modulator=False,
        layerscale_value=None,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        proj_drop_rate=0.0,
        drop_path_rate=0.1,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        dims = [embed_dim * 2**i for i in range(len(depths))]
        self.stem = Downsample(in_chans, dims[0], 4, use_overlap_down, rngs=rngs)
        total = sum(depths)
        dpr = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]
        layers, in_dim = [], dims[0]
        for i, (dim, depth) in enumerate(zip(dims, depths)):
            start = sum(depths[:i])
            layers.append(
                FocalNetStage(
                    in_dim,
                    dim,
                    depth,
                    mlp_ratio,
                    i > 0,
                    focal_levels[i],
                    focal_windows[i],
                    use_overlap_down,
                    use_post_norm,
                    use_post_norm_in_modulation,
                    normalize_modulator,
                    layerscale_value,
                    proj_drop_rate,
                    dpr[start : start + depth],
                    rngs=rngs,
                )
            )
            in_dim = dim
        self.layers = nnx.List(layers)
        self.num_features = dims[-1]
        self.norm = nnx.LayerNorm(self.num_features, epsilon=1e-5, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = (
            nnx.Linear(self.num_features, num_classes, kernel_init=_init, rngs=rngs)
            if num_classes > 0
            else None
        )

    def forward_features(self, x):
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    def forward_head(self, x):
        x = self.head_drop(global_pool_nhwc(x, self.global_pool))
        return self.head(x) if self.head is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_LARGE = dict(
    depths=(2, 2, 18, 2), use_post_norm=True, use_overlap_down=True, layerscale_value=1e-4
)
_CFGS = {
    "focalnet_tiny_srf": dict(depths=(2, 2, 6, 2), embed_dim=96),
    "focalnet_small_srf": dict(depths=(2, 2, 18, 2), embed_dim=96),
    "focalnet_base_srf": dict(depths=(2, 2, 18, 2), embed_dim=128),
    "focalnet_tiny_lrf": dict(depths=(2, 2, 6, 2), embed_dim=96, focal_levels=(3, 3, 3, 3)),
    "focalnet_small_lrf": dict(depths=(2, 2, 18, 2), embed_dim=96, focal_levels=(3, 3, 3, 3)),
    "focalnet_base_lrf": dict(depths=(2, 2, 18, 2), embed_dim=128, focal_levels=(3, 3, 3, 3)),
    "focalnet_large_fl3": dict(
        **_LARGE, embed_dim=192, focal_levels=(3, 3, 3, 3), focal_windows=(5, 5, 5, 5)
    ),
    "focalnet_large_fl4": dict(**_LARGE, embed_dim=192, focal_levels=(4, 4, 4, 4)),
    "focalnet_xlarge_fl3": dict(
        **_LARGE, embed_dim=256, focal_levels=(3, 3, 3, 3), focal_windows=(5, 5, 5, 5)
    ),
    "focalnet_xlarge_fl4": dict(**_LARGE, embed_dim=256, focal_levels=(4, 4, 4, 4)),
    "focalnet_huge_fl3": dict(
        **_LARGE,
        embed_dim=352,
        focal_levels=(3, 3, 3, 3),
        focal_windows=(3, 3, 3, 3),
        use_post_norm_in_modulation=True,
    ),
    "focalnet_huge_fl4": dict(
        **_LARGE, embed_dim=352, focal_levels=(4, 4, 4, 4), use_post_norm_in_modulation=True
    ),
}


def _make(name):
    cfg = _CFGS[name]
    # The large and xlarge ImageNet-22k checkpoints use 384x384 inputs.
    size, crop = (384, 1.0) if "large" in name else (224, 0.9)

    def entry(**kwargs):
        model = FocalNet(**{**cfg, **kwargs})
        model.default_cfg = _cfg(input_size=(3, size, size), crop_pct=crop, interpolation="bicubic")
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
