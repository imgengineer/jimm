"""Vision Transformer in flax nnx, NHWC input. Mirrors timm.models.vision_transformer."""
import jax.numpy as jnp
from flax import nnx

from ..layers import DropPath, Mlp, PatchEmbed
from ..registry import register_model, _cfg


class Attention(nnx.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, drop=0.0, *, rngs):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nnx.Linear(dim, dim * 3, use_bias=qkv_bias, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)
        self.drop = nnx.Dropout(drop)

    def __call__(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = nnx.softmax(q @ k.transpose(0, 1, 3, 2) * self.scale, axis=-1)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.drop(self.proj(x))


class Block(nnx.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0,
                 drop_path=0.0, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = Attention(dim, num_heads, qkv_bias, drop, rngs=rngs)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop, rngs=rngs)

    def __call__(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class VisionTransformer(nnx.Module):
    default_cfg: dict = {}

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 global_pool="", embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 qkv_bias=True, drop_rate=0.0, drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, rngs=rngs)
        n = self.patch_embed.num_patches
        self.cls_token = nnx.Param(jnp.zeros((1, 1, embed_dim)))
        self.pos_embed = nnx.Param(jnp.zeros((1, n + 1, embed_dim)))
        self.pos_drop = nnx.Dropout(drop_rate)
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nnx.List([Block(embed_dim, num_heads, mlp_ratio, qkv_bias,
                                      drop_rate, dpr[i], rngs=rngs) for i in range(depth)])
        self.norm = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate)
        self.head = nnx.Linear(embed_dim, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).reshape(B, -1, self.num_features)
        x = jnp.concatenate([jnp.broadcast_to(self.cls_token.value, (B, 1, self.num_features)), x], axis=1)
        x = self.pos_drop(x + self.pos_embed.value)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

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


def _vit(img_size, patch_size, embed_dim, depth, num_heads, **kwargs):
    model = VisionTransformer(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
                              depth=depth, num_heads=num_heads, **kwargs)
    model.default_cfg = _cfg(input_size=(3, img_size, img_size))
    return model


@register_model
def vit_tiny_patch16_224(**kwargs):
    return _vit(224, 16, 192, 12, 3, **kwargs)


@register_model
def vit_small_patch16_224(**kwargs):
    return _vit(224, 16, 384, 12, 6, **kwargs)


@register_model
def vit_base_patch16_224(**kwargs):
    return _vit(224, 16, 768, 12, 12, **kwargs)


@register_model
def vit_large_patch16_224(**kwargs):
    return _vit(224, 16, 1024, 24, 16, **kwargs)


# DeiT: same architecture as ViT, different training recipe (timm registers both)
@register_model
def deit_tiny_patch16_224(**kwargs):
    return _vit(224, 16, 192, 12, 3, **kwargs)


@register_model
def deit_small_patch16_224(**kwargs):
    return _vit(224, 16, 384, 12, 6, **kwargs)


@register_model
def deit_base_patch16_224(**kwargs):
    return _vit(224, 16, 768, 12, 12, **kwargs)


# BEiT v1 / DeiT-III: ViT architecture, different pretraining/recipes
@register_model
def beit_base_patch16_224(**kwargs):
    return _vit(224, 16, 768, 12, 12, **kwargs)


@register_model
def beit_large_patch16_224(**kwargs):
    return _vit(224, 16, 1024, 24, 16, **kwargs)


@register_model
def deit3_small_patch16_224(**kwargs):
    return _vit(224, 16, 384, 12, 6, **kwargs)


@register_model
def deit3_base_patch16_224(**kwargs):
    return _vit(224, 16, 768, 12, 12, **kwargs)


@register_model
def deit3_large_patch16_224(**kwargs):
    return _vit(224, 16, 1024, 24, 16, **kwargs)
