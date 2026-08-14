"""CaiT in flax nnx. Mirrors timm.models.cait (LayerScale + class-attention stage)."""
import jax.numpy as jnp
from flax import nnx

from ..layers import DropPath, Mlp, PatchEmbed, ClassifierMixin
from ..registry import register_model, _cfg
from .vision_transformer import Attention

class LayerScaleBlock(nnx.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, drop_path=0.0,
                 init_values=1e-5, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = Attention(dim, num_heads, rngs=rngs)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop, rngs=rngs)
        self.gamma1 = nnx.Param(init_values * jnp.ones(dim))
        self.gamma2 = nnx.Param(init_values * jnp.ones(dim))

    def __call__(self, x):
        x = x + self.drop_path(self.gamma1.value * self.attn(self.norm1(x)))
        return x + self.drop_path(self.gamma2.value * self.mlp(self.norm2(x)))

class ClassAttentionBlock(nnx.Module):
    """Attention from cls token to patch tokens only (CaiT class-attention)."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop_path=0.0, init_values=1e-5, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nnx.Linear(dim, dim, rngs=rngs)
        self.k = nnx.Linear(dim, dim, rngs=rngs)
        self.v = nnx.Linear(dim, dim, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), rngs=rngs)
        self.drop_path = DropPath(drop_path)
        self.gamma1 = nnx.Param(init_values * jnp.ones(dim))
        self.gamma2 = nnx.Param(init_values * jnp.ones(dim))

    def __call__(self, x):
        cls = x[:, :1]
        tokens = self.norm1(x)
        B, N, C = tokens.shape
        q = self.q(cls).reshape(B, 1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k(tokens).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v(tokens).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        attn = nnx.softmax(q @ k.transpose(0, 1, 3, 2) * self.scale, axis=-1)
        cls_out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, 1, C)
        cls = cls + self.drop_path(self.gamma1.value * self.proj(cls_out))
        cls = cls + self.drop_path(self.gamma2.value * self.mlp(self.norm2(cls)))
        return jnp.concatenate([cls, x[:, 1:]], axis=1)

class CaiT(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"
    _default_global_pool = ""
    default_cfg: dict = {}

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 global_pool="", embed_dim=384, depth=12, num_heads=8, mlp_ratio=4.0,
                 depth_token_only=2, drop_rate=0.0, drop_path_rate=0.0, init_values=1e-5, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, rngs=rngs)
        n = self.patch_embed.num_patches
        self.pos_embed = nnx.Param(jnp.zeros((1, n, embed_dim)))
        self.cls_token = nnx.Param(jnp.zeros((1, 1, embed_dim)))
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nnx.List([LayerScaleBlock(embed_dim, num_heads, mlp_ratio, drop_rate,
                                                dpr[i], init_values, rngs=rngs)
                                for i in range(depth)])
        self.blocks_token_only = nnx.List([
            ClassAttentionBlock(embed_dim, num_heads, mlp_ratio, 0.0, init_values, rngs=rngs)
            for _ in range(depth_token_only)])
        self.norm = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.head = nnx.Linear(embed_dim, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).reshape(B, -1, self.num_features)
        x = x + self.pos_embed.value
        for blk in self.blocks:
            x = blk(x)
        x = jnp.concatenate([jnp.broadcast_to(self.cls_token.value, (B, 1, self.num_features)), x], axis=1)
        for blk in self.blocks_token_only:
            x = blk(x)
        return self.norm(x)

    def forward_head(self, x):
        x = x[:, 0]
        return self.head(x) if self.head is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

def _cait(embed_dim, depth, num_heads, depth_token_only=2, **kwargs):
    model = CaiT(embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                 depth_token_only=depth_token_only, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def cait_xxs24_224(**kwargs):
    return _cait(192, 24, 4, **kwargs)

@register_model
def cait_xs24_224(**kwargs):
    return _cait(288, 24, 6, **kwargs)

@register_model
def cait_s24_224(**kwargs):
    return _cait(384, 24, 8, **kwargs)

@register_model
def cait_m36_224(**kwargs):
    return _cait(768, 36, 16, **kwargs)
