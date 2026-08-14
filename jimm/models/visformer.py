"""Visformer in flax nnx. Mirrors timm.models.visformer (conv stem + BN-in-transformer stages)."""
import jax.numpy as jnp
from flax import nnx

from ..layers import ConvBNAct, DropPath, Mlp, ClassifierMixin
from ..registry import register_model, _cfg
from .vision_transformer import Attention

class VisBlock(nnx.Module):
    """Transformer block with BatchNorm (visformer style) on token dim."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, drop_path=0.0, *, rngs):
        self.norm1 = nnx.BatchNorm(dim, rngs=rngs)
        self.attn = Attention(dim, num_heads, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)
        self.norm2 = nnx.BatchNorm(dim, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop, rngs=rngs)

    def __call__(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))

class Visformer(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"
    default_cfg: dict = {}

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 global_pool="avg", embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
                 drop_rate=0.0, drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dim
        # conv stem: 2x conv3x3 s2 then patch-embed-ish conv
        self.stem = nnx.List([
            ConvBNAct(in_chans, embed_dim // 2, 3, 2, rngs=rngs),
            ConvBNAct(embed_dim // 2, embed_dim, 3, 2, rngs=rngs)])
        n = (img_size // 4) ** 2
        self.pos_embed = nnx.Param(jnp.zeros((1, n, embed_dim)))
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nnx.List([VisBlock(embed_dim, num_heads, mlp_ratio, drop_rate,
                                         dpr[i], rngs=rngs) for i in range(depth)])
        self.norm = nnx.BatchNorm(embed_dim, rngs=rngs)
        self.head = nnx.Linear(embed_dim, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        for layer in self.stem:
            x = layer(x)
        x = x.reshape(x.shape[0], -1, self.num_features)
        x = x + self.pos_embed.value
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward_head(self, x):
        x = jnp.mean(x, axis=1)
        return self.head(x) if self.head is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

@register_model
def visformer_tiny(**kwargs):
    model = Visformer(embed_dim=192, depth=12, num_heads=3, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def visformer_small(**kwargs):
    model = Visformer(embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg()
    return model
