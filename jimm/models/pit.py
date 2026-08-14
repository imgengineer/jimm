"""PiT (Pooling-based Vision Transformer) in flax nnx. Mirrors timm.models.pit."""
import jax.numpy as jnp
from flax import nnx

from ..layers import DropPath, Mlp
from ..registry import register_model, _cfg
from .vision_transformer import Attention


class PiTBlock(nnx.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, drop_path=0.0, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = Attention(dim, num_heads, rngs=rngs)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop, rngs=rngs)

    def __call__(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PoolingLayer(nnx.Module):
    """Depthwise conv 3x3 stride 2 on tokens (cls token stays), then dim projection."""

    def __init__(self, dim_in, dim_out, *, rngs):
        self.dw = nnx.Conv(dim_in, dim_in, (3, 3), strides=(2, 2), feature_group_count=dim_in, rngs=rngs)
        self.pw = nnx.Conv(dim_in, dim_out, (1, 1), rngs=rngs)
        self.norm1 = nnx.LayerNorm(dim_in, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim_out, rngs=rngs)
        self.cls_fc = nnx.Linear(dim_in, dim_out, rngs=rngs)

    def __call__(self, x, H, W):
        cls, tokens = x[:, :1], x[:, 1:]
        B = x.shape[0]
        t = self.norm1(tokens).reshape(B, H, W, -1)
        t = self.pw(self.dw(t))
        tokens = t.reshape(B, -1, t.shape[-1])
        cls = self.cls_fc(self.norm1(cls))
        return jnp.concatenate([cls, self.norm2(tokens)], axis=1), t.shape[1], t.shape[2]


class PiT(nnx.Module):
    default_cfg: dict = {}

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 global_pool="", embed_dim=384, depth=(2, 6, 4), num_heads=6, mlp_ratio=4.0,
                 drop_rate=0.0, drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        heads = [num_heads * 2**i for i in range(len(depth))]
        dims = [embed_dim * 2**i for i in range(len(depth))]
        self.num_features = dims[-1]
        self.patch_embed = nnx.Conv(in_chans, embed_dim, (patch_size, patch_size),
                                    strides=(patch_size, patch_size), rngs=rngs)
        n = (img_size // patch_size) ** 2
        self.cls_token = nnx.Param(jnp.zeros((1, 1, embed_dim)))
        self.pos_embed = nnx.Param(jnp.zeros((1, n + 1, embed_dim)))
        dpr = [drop_path_rate * i / max(sum(depth) - 1, 1) for i in range(sum(depth))]
        stages, k = [], 0
        for i, d in enumerate(depth):
            stages.append(nnx.List([PiTBlock(dims[i], heads[i], mlp_ratio, drop_rate,
                                             dpr[k + j], rngs=rngs) for j in range(d)]))
            k += d
        self.stages = nnx.List(stages)
        self.pools = nnx.List([PoolingLayer(dims[i], dims[i + 1], rngs=rngs)
                               for i in range(len(depth) - 1)])
        self.norm = nnx.LayerNorm(dims[-1], rngs=rngs)
        self.head = nnx.Linear(dims[-1], num_classes, rngs=rngs) if num_classes > 0 else None
        self.res0 = img_size // patch_size

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).reshape(B, -1, self.patch_embed.out_features)
        x = jnp.concatenate([jnp.broadcast_to(self.cls_token.value, (B, 1, x.shape[-1])), x], axis=1)
        H = W = self.res0
        for i, stage in enumerate(self.stages):
            for blk in stage:
                x = blk(x)
            if i < len(self.pools):
                x, H, W = self.pools[i](x, H, W)
        return self.norm(x)

    def forward_head(self, x):
        x = x[:, 0] if self.global_pool == "" else jnp.mean(x[:, 1:], axis=1)
        return self.head(x) if self.head is not None else x

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes, self.global_pool = num_classes, global_pool
        if num_classes > 0 and self.head is None:
            raise RuntimeError("cannot re-add classifier to a num_classes=0 model")
        self.head = nnx.Linear(self.num_features, num_classes, rngs=nnx.Rngs(0)) if num_classes > 0 else None

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


def _pit(embed_dim, depth, num_heads, **kwargs):
    model = PiT(embed_dim=embed_dim, depth=depth, num_heads=num_heads, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def pit_ti_224(**kwargs):
    return _pit(256, (2, 6, 4), 4, **kwargs)


@register_model
def pit_xs_224(**kwargs):
    return _pit(384, (2, 6, 4), 6, **kwargs)


@register_model
def pit_s_224(**kwargs):
    return _pit(384, (2, 9, 4), 6, **kwargs)


@register_model
def pit_b_224(**kwargs):
    return _pit(512, (3, 11, 4), 8, **kwargs)
