"""PoolFormer (v1) in flax nnx, NHWC. Mirrors timm's MetaFormer poolformer_* models."""

import jax.numpy as jnp
from flax import nnx

from ..layers import ClassifierMixin, DropPath, gelu, global_pool_nhwc
from ..registry import _cfg, register_model

_init = nnx.initializers.truncated_normal(0.02)


def _group_norm(dim, *, rngs):
    # timm GroupNorm1: one group over (H, W, C), PyTorch's default epsilon.
    return nnx.GroupNorm(dim, num_groups=1, epsilon=1e-5, rngs=rngs)


class PoolFormerBlock(nnx.Module):
    def __init__(
        self, dim, pool_size=3, mlp_ratio=4.0, drop_path=0.0, layer_scale_init=1e-5, *, rngs
    ):
        self.pool_size = pool_size
        self.norm1 = _group_norm(dim, rngs=rngs)
        self.norm2 = _group_norm(dim, rngs=rngs)
        self.mlp_fc1 = nnx.Linear(dim, int(dim * mlp_ratio), kernel_init=_init, rngs=rngs)
        self.mlp_fc2 = nnx.Linear(int(dim * mlp_ratio), dim, kernel_init=_init, rngs=rngs)
        self.scale1 = nnx.Param(jnp.full((dim,), layer_scale_init))
        self.scale2 = nnx.Param(jnp.full((dim,), layer_scale_init))
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        # Token mixer: average pooling (padding excluded) minus identity.
        y = self.norm1(x)
        p = self.pool_size // 2
        pooled = nnx.avg_pool(
            y,
            (self.pool_size, self.pool_size),
            strides=(1, 1),
            padding=((p, p), (p, p)),
            count_include_pad=False,
        )
        x = x + self.drop_path(self.scale1[...] * (pooled - y))
        y = self.mlp_fc2(gelu(self.mlp_fc1(self.norm2(x))))
        return x + self.drop_path(self.scale2[...] * y)


class PoolFormer(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        layers,
        embed_dims,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.0,
        layer_scale_init=1e-5,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dims[-1]
        dpr = [drop_path_rate * i / max(sum(layers) - 1, 1) for i in range(sum(layers))]
        patches, stages, k = [], [], 0
        for i, (n, dim) in enumerate(zip(layers, embed_dims)):
            # Stem: 7x7 stride-4 conv (padding 2); later stages: 3x3 stride-2 conv (padding 1).
            kernel, stride, pad = (3, 2, 1) if i else (7, 4, 2)
            patches.append(
                nnx.Conv(
                    embed_dims[i - 1] if i else in_chans,
                    dim,
                    (kernel, kernel),
                    strides=(stride, stride),
                    padding=((pad, pad), (pad, pad)),
                    kernel_init=_init,
                    rngs=rngs,
                )
            )
            blocks = []
            for _ in range(n):
                blocks.append(
                    PoolFormerBlock(
                        dim, drop_path=dpr[k], layer_scale_init=layer_scale_init, rngs=rngs
                    )
                )
                k += 1
            stages.append(nnx.List(blocks))
        self.patches = nnx.List(patches)
        self.stages = nnx.List(stages)
        # timm MetaFormer head: global pool, LayerNorm, then the classifier.
        self.head_norm = nnx.LayerNorm(self.num_features, epsilon=1e-6, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = (
            nnx.Linear(self.num_features, num_classes, kernel_init=_init, rngs=rngs)
            if num_classes > 0
            else None
        )

    def forward_features(self, x):
        for patch, stage in zip(self.patches, self.stages):
            x = patch(x)
            for blk in stage:
                x = blk(x)
        return x

    def forward_head(self, x):
        x = self.head_norm(global_pool_nhwc(x, self.global_pool))
        x = self.head_drop(x)
        return self.fc(x) if self.fc is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


def _poolformer(layers, dims, **kwargs):
    model = PoolFormer(layers, dims, **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9, interpolation="bicubic")
    return model


@register_model
def poolformer_s12(**kwargs):
    return _poolformer([2, 2, 6, 2], [64, 128, 320, 512], **kwargs)


@register_model
def poolformer_s24(**kwargs):
    return _poolformer([4, 4, 12, 4], [64, 128, 320, 512], **kwargs)


@register_model
def poolformer_s36(**kwargs):
    return _poolformer([6, 6, 18, 6], [64, 128, 320, 512], layer_scale_init=1e-6, **kwargs)


@register_model
def poolformer_m36(**kwargs):
    return _poolformer([6, 6, 18, 6], [96, 192, 384, 768], layer_scale_init=1e-6, **kwargs)


@register_model
def poolformer_m48(**kwargs):
    return _poolformer([8, 8, 24, 8], [96, 192, 384, 768], layer_scale_init=1e-6, **kwargs)
