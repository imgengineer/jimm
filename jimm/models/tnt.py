"""TNT (Transformer-iN-Transformer) in flax nnx. Mirrors timm.models.tnt."""
import jax.numpy as jnp
from flax import nnx

from ..layers import DropPath, Mlp, PatchEmbed
from ..registry import register_model, _cfg
from .vision_transformer import Attention


class TNTBlock(nnx.Module):
    """outer token attention + inner pixel-level attention per patch."""

    def __init__(self, dim, num_heads, inner_dim, inner_heads, drop_path=0.0, *, rngs):
        self.norm_out = nnx.LayerNorm(dim, rngs=rngs)
        self.outer_attn = Attention(dim, num_heads, rngs=rngs)
        self.norm_out2 = nnx.LayerNorm(dim, rngs=rngs)
        self.outer_mlp = Mlp(dim, dim * 4, rngs=rngs)
        self.norm_in = nnx.LayerNorm(inner_dim, rngs=rngs)
        self.inner_attn = Attention(inner_dim, inner_heads, rngs=rngs)
        self.norm_in2 = nnx.LayerNorm(inner_dim, rngs=rngs)
        self.inner_mlp = Mlp(inner_dim, inner_dim * 4, rngs=rngs)
        self.proj = nnx.Linear(inner_dim, dim, rngs=rngs)
        self.drop_path = DropPath(drop_path)

    def __call__(self, tokens, inner, grid):
        # outer (token) attention on cls+patch tokens
        tokens = tokens + self.drop_path(self.outer_attn(self.norm_out(tokens)))
        tokens = tokens + self.drop_path(self.outer_mlp(self.norm_out2(tokens)))
        # inner (pixel) attention within each patch
        inner = inner + self.drop_path(self.inner_attn(self.norm_in(inner)))
        inner = inner + self.drop_path(self.inner_mlp(self.norm_in2(inner)))
        # project inner back into patch tokens (inner is per-patch: (B, G*G, inner_dim))
        B = tokens.shape[0]
        inner_agg = self.proj(inner)
        tokens = tokens + jnp.concatenate(
            [jnp.zeros((B, 1, tokens.shape[-1]), tokens.dtype), inner_agg], axis=1)
        return tokens, inner


class TNT(nnx.Module):
    default_cfg: dict = {}

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 global_pool="", embed_dim=384, depth=12, num_heads=6, inner_dim=24,
                 inner_heads=4, drop_rate=0.0, drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, rngs=rngs)
        n = self.patch_embed.num_patches
        self.grid = img_size // patch_size
        self.cls_token = nnx.Param(jnp.zeros((1, 1, embed_dim)))
        self.pos_embed = nnx.Param(jnp.zeros((1, n + 1, embed_dim)))
        # inner pixel embedding: patch is patch_size x patch_size, split into inner patches
        self.inner_embed = nnx.Linear(patch_size * patch_size * in_chans, inner_dim, rngs=rngs)
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nnx.List([TNTBlock(embed_dim, num_heads, inner_dim, inner_heads,
                                         dpr[i], rngs=rngs) for i in range(depth)])
        self.norm = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate)
        self.head = nnx.Linear(embed_dim, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        B = x.shape[0]
        patches = self.patch_embed(x)  # (B, G, G, embed)
        G = patches.shape[1]
        tokens = patches.reshape(B, -1, self.num_features)
        tokens = jnp.concatenate([jnp.broadcast_to(self.cls_token.value, (B, 1, self.num_features)),
                                  tokens], axis=1)
        tokens = tokens + self.pos_embed.value
        # inner: raw patch pixels -> inner_dim
        B2, H, W, C = x.shape
        p = self.patch_embed.patch_size
        inner = x.reshape(B, G, p, G, p, C).transpose(0, 1, 3, 2, 4, 5).reshape(B, G * G, p * p * C)
        inner = self.inner_embed(inner)
        for blk in self.blocks:
            tokens, inner = blk(tokens, inner, G)
        return self.norm(tokens)

    def forward_head(self, x):
        x = x[:, 0] if self.global_pool == "" else jnp.mean(x[:, 1:], axis=1)
        x = self.head_drop(x)
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


@register_model
def tnt_tiny_patch16_224(**kwargs):
    model = TNT(embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def tnt_small_patch16_224(**kwargs):
    model = TNT(embed_dim=384, depth=12, num_heads=6, inner_dim=48, **kwargs)
    model.default_cfg = _cfg()
    return model
