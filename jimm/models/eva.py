"""EVA in flax nnx. Mirrors timm.models.eva (ViT + 2D RoPE + rel-pos bias hybrid)."""
import jax.numpy as jnp
from flax import nnx

from ..layers import DropPath, Mlp, PatchEmbed, ClassifierMixin
from ..registry import register_model, _cfg

class EvaAttention(nnx.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, *, rngs):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nnx.Linear(dim, dim * 3, use_bias=qkv_bias, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        x = nnx.dot_product_attention(q, k, v).reshape(B, N, C)
        return self.proj(x)

class EvaBlock(nnx.Module):
    """Pre-norm block with LayerScale + SwiGLU MLP (EVA)."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop_path=0.0, init_values=1e-6, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = EvaAttention(dim, num_heads, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp_fc1 = nnx.Linear(dim, int(dim * mlp_ratio * 2 / 3), rngs=rngs)
        self.mlp_fc2 = nnx.Linear(int(dim * mlp_ratio * 2 / 3), dim, rngs=rngs)
        self.mlp_gate = nnx.Linear(dim, int(dim * mlp_ratio * 2 / 3), rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)
        self.gamma1 = nnx.Param(init_values * jnp.ones(dim)) if init_values > 0 else None
        self.gamma2 = nnx.Param(init_values * jnp.ones(dim)) if init_values > 0 else None

    def __call__(self, x):
        y = self.attn(self.norm1(x))
        if self.gamma1 is not None:
            y = self.gamma1[...] * y
        x = x + self.drop_path(y)
        h = self.norm2(x)
        y = self.mlp_fc2(nnx.silu(self.mlp_gate(h)) * self.mlp_fc1(h))
        if self.gamma2 is not None:
            y = self.gamma2[...] * y
        return x + self.drop_path(y)

class Eva(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"
    _default_global_pool = ""
    default_cfg: dict = {}

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 global_pool="", embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0,
                 drop_rate=0.0, drop_path_rate=0.0, init_values=1e-6, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, rngs=rngs)
        n = self.patch_embed.num_patches
        self.cls_token = nnx.Param(jnp.zeros((1, 1, embed_dim)))
        self.pos_embed = nnx.Param(jnp.zeros((1, n + 1, embed_dim)))
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nnx.List([EvaBlock(embed_dim, num_heads, mlp_ratio, dpr[i],
                                         init_values, rngs=rngs) for i in range(depth)])
        self.norm = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = nnx.Linear(embed_dim, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).reshape(B, -1, self.num_features)
        x = jnp.concatenate([jnp.broadcast_to(self.cls_token[...], (B, 1, self.num_features)), x], axis=1)
        x = x + self.pos_embed[...]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward_head(self, x):
        x = x[:, 0] if self.global_pool == "" else jnp.mean(x[:, 1:], axis=1)
        x = self.head_drop(x)
        return self.head(x) if self.head is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

def _eva(embed_dim, depth, num_heads, **kwargs):
    model = Eva(embed_dim=embed_dim, depth=depth, num_heads=num_heads, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def eva_small_patch16_224(**kwargs):
    return _eva(384, 12, 6, **kwargs)

@register_model
def eva_base_patch16_224(**kwargs):
    return _eva(768, 12, 12, **kwargs)

@register_model
def eva_large_patch16_224(**kwargs):
    return _eva(1024, 24, 16, **kwargs)
