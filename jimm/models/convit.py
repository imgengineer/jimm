"""ConViT in flax nnx. Mirrors timm.models.convit (GPSA: gated positional self-attention)."""
import jax
import jax.numpy as jnp
from flax import nnx

from ..layers import DropPath, Mlp, PatchEmbed, ClassifierMixin
from ..registry import register_model, _cfg

class GPSA(nnx.Module):
    """Gated positional self-attention (ConViT): content attn + gated positional attn."""

    def __init__(self, dim, num_heads, grid, *, rngs):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nnx.Linear(dim, dim * 3, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)
        self.gate = nnx.Param(jnp.ones((num_heads, 1, 1)))
        self.pos_proj = nnx.Linear(2, num_heads, rngs=rngs)
        gh = gw = grid
        coords = jnp.stack(jnp.meshgrid(jnp.arange(gh), jnp.arange(gw), indexing="ij"))
        cf = coords.reshape(2, -1)  # (2, gh*gw)
        rel = (cf[:, :, None] - cf[:, None, :]).transpose(1, 2, 0) / gh  # (gh*gw, gh*gw, 2)
        # pad a zero row/col for the cls token
        rel = jnp.pad(rel, ((1, 0), (1, 0), (0, 0)))
        self.rel_coords = nnx.Variable(rel)  # (N, N, 2) with cls

    def __call__(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        content = q @ k.transpose(0, 1, 3, 2) * self.scale
        pos = self.pos_proj(self.rel_coords[...]).transpose(2, 0, 1)  # (heads, N, N)
        gate = nnx.sigmoid(self.gate[...])
        attn = nnx.softmax(gate * pos[None] + (1 - gate) * content, axis=-1)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(x)

import jax  # noqa: E402

class ConViTBlock(nnx.Module):
    def __init__(self, dim, num_heads, grid, use_gpsa, drop_path=0.0, *, rngs):
        from .vision_transformer import Attention
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = GPSA(dim, num_heads, grid, rngs=rngs) if use_gpsa \
            else Attention(dim, num_heads, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = Mlp(dim, dim * 4, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))

class ConViT(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"
    _default_global_pool = ""
    default_cfg: dict | None = None

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 global_pool="", embed_dim=432, depth=12, num_heads=9, gpsa_depth=10,
                 drop_rate=0.0, drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, rngs=rngs)
        n = self.patch_embed.num_patches
        self.grid = img_size // patch_size
        self.cls_token = nnx.Param(jnp.zeros((1, 1, embed_dim)))
        self.pos_embed = nnx.Param(jnp.zeros((1, n + 1, embed_dim)))
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nnx.List([
            ConViTBlock(embed_dim, num_heads, self.grid, i < gpsa_depth, dpr[i], rngs=rngs)
            for i in range(depth)])
        self.norm = nnx.LayerNorm(embed_dim, rngs=rngs)
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
        x = x[:, 0]
        return self.head(x) if self.head is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

@register_model
def convit_tiny(**kwargs):
    model = ConViT(embed_dim=432, depth=12, num_heads=9, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def convit_small(**kwargs):
    model = ConViT(embed_dim=576, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def convit_base(**kwargs):
    model = ConViT(embed_dim=768, depth=12, num_heads=16, **kwargs)
    model.default_cfg = _cfg()
    return model
