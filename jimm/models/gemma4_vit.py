"""Gemma4 Vision Transformer in flax nnx, NHWC. Mirrors timm.models.gemma4_vit (RoPE + RMSNorm + SwiGLU)."""
import jax
import jax.numpy as jnp
from flax import nnx

from ..layers import DropPath, PatchEmbed, global_pool_nhwc, ClassifierMixin
from ..registry import register_model, _cfg

class Gemma4GatedMlp(nnx.Module):
    """SwiGLU gated MLP."""

    def __init__(self, dim, hidden_dim, *, rngs):
        self.gate_proj = nnx.Linear(dim, hidden_dim, use_bias=False, rngs=rngs)
        self.up_proj = nnx.Linear(dim, hidden_dim, use_bias=False, rngs=rngs)
        self.down_proj = nnx.Linear(hidden_dim, dim, use_bias=False, rngs=rngs)

    def __call__(self, x):
        return self.down_proj(nnx.silu(self.gate_proj(x)) * self.up_proj(x))

class Gemma4Attention(nnx.Module):
    def __init__(self, dim, num_heads=16, *, rngs):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.k = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.v = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)

    def __call__(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim)
        out = nnx.dot_product_attention(q, k, v).reshape(B, N, C)
        return self.proj(out)

class Gemma4Block(nnx.Module):
    def __init__(self, dim, num_heads, hidden_dim, drop_path=0.0, *, rngs):
        self.norm1 = nnx.RMSNorm(dim, epsilon=1e-6, rngs=rngs)
        self.attn = Gemma4Attention(dim, num_heads, rngs=rngs)
        self.norm2 = nnx.RMSNorm(dim, epsilon=1e-6, rngs=rngs)
        self.mlp = Gemma4GatedMlp(dim, hidden_dim, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))

class Gemma4Vit(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"
    default_cfg: dict = {}

    def __init__(self, img_size: int = 224, patch_size: int = 14, in_chans: int = 3, num_classes: int = 1000,
                 global_pool: str = "avg", embed_dim: int = 768, depth: int = 12, num_heads: int = 16,
                 mlp_ratio: float = 4.0, drop_rate: float = 0.0, drop_path_rate: float = 0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, rngs=rngs)
        n = self.patch_embed.num_patches
        self.pos_embed = nnx.Param(jnp.zeros((1, n, embed_dim)))

        hidden_dim = int(embed_dim * mlp_ratio * 2 / 3)
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nnx.List([
            Gemma4Block(embed_dim, num_heads, hidden_dim, dpr[i], rngs=rngs)
            for i in range(depth)])

        self.norm = nnx.RMSNorm(embed_dim, epsilon=1e-6, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = nnx.Linear(embed_dim, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).reshape(B, -1, self.num_features)
        x = x + self.pos_embed[...]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward_head(self, x):
        x = global_pool_nhwc(x.reshape(x.shape[0], 1, -1, self.num_features), self.global_pool)
        x = self.head_drop(x)
        return self.head(x) if self.head is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

_CFGS = {  # embed_dim, depth, num_heads, mlp_ratio, num_classes
    "gemma4_vit_167m": (768, 12, 16, 4.0, 1000),
    "gemma4_vit_167m_enc": (768, 12, 16, 4.0, 0),
    "gemma4_vit_570m": (1152, 27, 16, 4.0, 1000),
    "gemma4_vit_570m_enc": (1152, 27, 16, 4.0, 0),
}

def _make(name):
    embed_dim, depth, num_heads, mlp_ratio, num_classes = _CFGS[name]

    def entry(**kwargs):
        kwargs.setdefault("num_classes", num_classes)
        model = Gemma4Vit(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            **kwargs,
        )
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
