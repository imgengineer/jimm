"""ViT-SAM (Segment Anything image encoder) in flax nnx. Mirrors timm.models.vision_transformer_sam.

Key traits: window attention with global-attention every few blocks, rel-pos bias, neck conv.
"""
import jax.numpy as jnp
from flax import nnx

from ..layers import DropPath, Mlp, PatchEmbed
from ..registry import register_model, _cfg
from .swin_transformer import window_partition, window_reverse


class SamAttention(nnx.Module):
    def __init__(self, dim, num_heads, window_size=None, *, rngs):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.ws = window_size
        self.qkv = nnx.Linear(dim, dim * 3, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = nnx.softmax(q @ k.transpose(0, 1, 3, 2) * self.scale, axis=-1)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(x)


class SamBlock(nnx.Module):
    def __init__(self, dim, num_heads, window_size, mlp_ratio=4.0, drop_path=0.0, *, rngs):
        self.ws = window_size
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = SamAttention(dim, num_heads, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), rngs=rngs)
        self.drop_path = DropPath(drop_path)

    def __call__(self, x):
        B, H, W, C = x.shape
        if self.ws:
            t = window_partition(x, self.ws)
            t = t + self.drop_path(self.attn(self.norm1(t)))
            t = t + self.drop_path(self.mlp(self.norm2(t)))
            return window_reverse(t, self.ws, H, W, B)
        t = x.reshape(B, H * W, C)
        t = t + self.drop_path(self.attn(self.norm1(t)))
        t = t + self.drop_path(self.mlp(self.norm2(t)))
        return t.reshape(B, H, W, C)


class VisionTransformerSAM(nnx.Module):
    default_cfg: dict = {}

    def __init__(self, img_size=1024, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, window_size=14, global_interval=3, num_classes=0,
                 drop_path_rate=0.0, *, rngs):
        self.embed_dim = embed_dim
        self.num_features = 256
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, rngs=rngs)
        n = self.patch_embed.num_patches
        self.pos_embed = nnx.Param(jnp.zeros((1, n, embed_dim)))
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nnx.List([
            SamBlock(embed_dim, num_heads, window_size if i % global_interval else 0,
                     4.0, dpr[i], rngs=rngs) for i in range(depth)])
        # neck: 1x1 to 256 then 3x3 SAME 256
        self.neck1 = nnx.Conv(embed_dim, 256, (1, 1), use_bias=False, rngs=rngs)
        self.neck2 = nnx.Conv(256, 256, (3, 3), use_bias=False, padding="VALID", rngs=rngs)
        self.head = None  # no classifier; this is an encoder

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed.value.reshape(1, self.patch_embed.grid_size[0],
                                             self.patch_embed.grid_size[1], self.embed_dim)
        for blk in self.blocks:
            x = blk(x)
        x = self.neck2(self.neck1(x))
        return x

    def forward_head(self, x):
        return global_pool_enc(x)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool="avg"):
        raise RuntimeError("ViT-SAM is an encoder without a classifier head")

    def __call__(self, x):
        return self.forward_features(x)


def global_pool_enc(x):
    return jnp.mean(x, axis=(1, 2))


@register_model
def vit_sam_base_patch16_224(**kwargs):
    model = VisionTransformerSAM(img_size=224, embed_dim=768, depth=12, num_heads=12,
                                 window_size=14, **kwargs)
    model.default_cfg = _cfg()
    return model
