"""PiT (Pooling-based Vision Transformer) in flax nnx, NHWC. Mirrors timm.models.pit.

An overlapping patch embedding with a learned 2D position embedding feeds three
transformer stages. Between stages, a depthwise convolution with channel
multiplier 2 pools the patch tokens and a linear layer widens the class
token(s); the head reads the normalized class token, averaged with the
distillation token's head for the distilled variants.
"""

import jax.numpy as jnp
from flax import nnx

from ..layers import ClassifierMixin
from ..registry import _cfg, register_model
from .vision_transformer import Block

_init = nnx.initializers.truncated_normal(0.02)


class Pooling(nnx.Module):
    def __init__(self, in_dim, out_dim, stride=2, *, rngs):
        pad = stride // 2
        self.conv = nnx.Conv(
            in_dim,
            out_dim,
            (stride + 1, stride + 1),
            strides=(stride, stride),
            padding=((pad, pad), (pad, pad)),
            feature_group_count=in_dim,
            rngs=rngs,
        )
        self.fc = nnx.Linear(in_dim, out_dim, rngs=rngs)

    def __call__(self, x, cls_tokens):
        return self.conv(x), self.fc(cls_tokens)


class Transformer(nnx.Module):
    def __init__(self, base_dim, depth, heads, mlp_ratio, pool=None, dpr=None, *, rngs):
        embed_dim = base_dim * heads
        self.pool = pool
        dpr = dpr or [0.0] * depth
        self.blocks = nnx.List(
            [Block(embed_dim, heads, mlp_ratio, True, 0.0, r, rngs=rngs) for r in dpr]
        )

    def __call__(self, x, cls_tokens):
        if self.pool is not None:
            x, cls_tokens = self.pool(x, cls_tokens)
        B, H, W, C = x.shape
        n = cls_tokens.shape[1]
        tokens = jnp.concatenate([cls_tokens, x.reshape(B, H * W, C)], axis=1)
        for blk in self.blocks:
            tokens = blk(tokens)
        return tokens[:, n:].reshape(B, H, W, C), tokens[:, :n]


class PoolingVisionTransformer(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"
    _default_global_pool = "token"

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride=8,
        base_dims=(48, 48, 48),
        depth=(2, 6, 4),
        heads=(2, 4, 8),
        mlp_ratio=4.0,
        distilled=False,
        num_classes=1000,
        in_chans=3,
        global_pool="token",
        drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        embed_dim = base_dims[0] * heads[0]
        self.patch_embed = nnx.Conv(
            in_chans,
            embed_dim,
            (patch_size, patch_size),
            strides=(stride, stride),
            padding="VALID",
            rngs=rngs,
        )
        grid = (img_size - patch_size) // stride + 1
        self.pos_embed = nnx.Param(_init(rngs.params(), (1, grid, grid, embed_dim)))
        self.cls_token = nnx.Param(_init(rngs.params(), (1, 2 if distilled else 1, embed_dim)))
        # timm calculate_drop_path_rates(stagewise=True): linear over all blocks, split by stage.
        total = sum(depth)
        rates = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]
        stages, prev = [], embed_dim
        for i, (base_dim, d, h) in enumerate(zip(base_dims, depth, heads)):
            dim = base_dim * h
            pool = Pooling(prev, dim, rngs=rngs) if i else None
            dpr = rates[sum(depth[:i]) : sum(depth[: i + 1])]
            stages.append(Transformer(base_dim, d, h, mlp_ratio, pool, dpr, rngs=rngs))
            prev = dim
        self.transformers = nnx.List(stages)
        self.norm = nnx.LayerNorm(prev, epsilon=1e-6, rngs=rngs)
        self.num_features = prev
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = nnx.Linear(prev, num_classes, rngs=rngs) if num_classes > 0 else None
        self.head_dist = (
            nnx.Linear(prev, num_classes, rngs=rngs) if distilled and num_classes > 0 else None
        )

    def reset_classifier(self, num_classes, global_pool=None):
        distilled = self.cls_token.shape[1] == 2
        super().reset_classifier(num_classes, global_pool or "token")
        if distilled:
            self.head_dist = (
                nnx.Linear(self.num_features, num_classes, rngs=nnx.Rngs(0))
                if num_classes > 0
                else None
            )

    def forward_features(self, x):
        x = self.patch_embed(x) + self.pos_embed[...]
        cls_tokens = jnp.broadcast_to(self.cls_token[...], (x.shape[0], *self.cls_token.shape[1:]))
        for stage in self.transformers:
            x, cls_tokens = stage(x, cls_tokens)
        return self.norm(cls_tokens)

    def forward_head(self, x):
        if self.cls_token.shape[1] == 2:
            x, x_dist = self.head_drop(x[:, 0]), self.head_drop(x[:, 1])
            if self.head is None:
                return x
            # Distilled models average the class and distillation heads.
            return (self.head(x) + self.head_dist(x_dist)) / 2
        x = self.head_drop(x[:, 0])
        return self.head(x) if self.head is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_CFGS = {  # patch_size, stride, base_dims, depth, heads
    "pit_b_224": (14, 7, (64, 64, 64), (3, 6, 4), (4, 8, 16)),
    "pit_s_224": (16, 8, (48, 48, 48), (2, 6, 4), (3, 6, 12)),
    "pit_xs_224": (16, 8, (48, 48, 48), (2, 6, 4), (2, 4, 8)),
    "pit_ti_224": (16, 8, (32, 32, 32), (2, 6, 4), (2, 4, 8)),
}


def _make(name, distilled=False):
    patch_size, stride, base_dims, depth, heads = _CFGS[name.replace("_distilled", "")]

    def entry(**kwargs):
        model = PoolingVisionTransformer(
            patch_size=patch_size,
            stride=stride,
            base_dims=base_dims,
            depth=depth,
            heads=heads,
            distilled=distilled,
            **kwargs,
        )
        model.default_cfg = _cfg(crop_pct=0.9, interpolation="bicubic")
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
    register_model(_make(_name.replace("_224", "_distilled_224"), distilled=True))
