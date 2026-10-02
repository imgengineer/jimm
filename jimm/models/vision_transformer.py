"""Vision Transformer in flax nnx, NHWC input. Mirrors timm.models.vision_transformer."""

import jax.numpy as jnp
from flax import nnx

from ..attention import dot_product_attention
from ..layers import ClassifierMixin, DropPath, Mlp, PatchEmbed
from ..registry import _cfg, register_model


class Attention(nnx.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, drop=0.0, *, rngs):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nnx.Linear(dim, dim * 3, use_bias=qkv_bias, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)
        self.drop = nnx.Dropout(drop, rngs=rngs)

    def __call__(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        out = dot_product_attention(q, k, v)
        x = out.reshape(B, N, C)
        return self.drop(self.proj(x))


class Block(nnx.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        drop_path=0.0,
        init_values=None,
        *,
        rngs,
    ):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = Attention(dim, num_heads, qkv_bias, drop, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop, rngs=rngs)
        # timm LayerScale (DeiT-III); absent for the original ViT.
        self.gamma1 = nnx.Param(jnp.full((dim,), init_values)) if init_values else None
        self.gamma2 = nnx.Param(jnp.full((dim,), init_values)) if init_values else None

    def __call__(self, x):
        y = self.attn(self.norm1(x))
        x = x + self.drop_path(y if self.gamma1 is None else self.gamma1[...] * y)
        y = self.mlp(self.norm2(x))
        return x + self.drop_path(y if self.gamma2 is None else self.gamma2[...] * y)


class VisionTransformer(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"
    _default_global_pool = ""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        global_pool="",
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        drop_path_rate=0.0,
        init_values=None,
        no_embed_class=False,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dim
        self.no_embed_class = no_embed_class
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, rngs=rngs)
        n = self.patch_embed.num_patches
        self.cls_token = nnx.Param(jnp.zeros((1, 1, embed_dim)))
        # DeiT-III adds position embeddings to patch tokens only, then prepends the class token.
        self.pos_embed = nnx.Param(jnp.zeros((1, n if no_embed_class else n + 1, embed_dim)))
        self.pos_drop = nnx.Dropout(drop_rate, rngs=rngs)
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nnx.List(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    drop_rate,
                    dpr[i],
                    init_values,
                    rngs=rngs,
                )
                for i in range(depth)
            ]
        )
        self.norm = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = nnx.Linear(embed_dim, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).reshape(B, -1, self.num_features)
        cls_token = jnp.broadcast_to(self.cls_token[...], (B, 1, self.num_features))
        if self.no_embed_class:
            x = jnp.concatenate([cls_token, x + self.pos_embed[...]], axis=1)
        else:
            x = jnp.concatenate([cls_token, x], axis=1) + self.pos_embed[...]
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward_head(self, x):
        x = x[:, 0] if self.global_pool == "" else jnp.mean(x[:, 1:], axis=1)
        x = self.head_drop(x)
        return self.head(x) if self.head is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


def _vit(img_size, patch_size, embed_dim, depth, num_heads, **kwargs):
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        **kwargs,
    )
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


# BEiT v1: approximated by the ViT architecture
@register_model
def beit_base_patch16_224(**kwargs):
    return _vit(224, 16, 768, 12, 12, **kwargs)


@register_model
def beit_large_patch16_224(**kwargs):
    return _vit(224, 16, 1024, 24, 16, **kwargs)


def _deit3(embed_dim, depth, num_heads, **kwargs):
    model = _vit(
        224, 16, embed_dim, depth, num_heads, init_values=1e-6, no_embed_class=True, **kwargs
    )
    model.default_cfg = _cfg(crop_pct=0.9, interpolation="bicubic")
    return model


@register_model
def deit3_small_patch16_224(**kwargs):
    return _deit3(384, 12, 6, **kwargs)


@register_model
def deit3_base_patch16_224(**kwargs):
    return _deit3(768, 12, 12, **kwargs)


@register_model
def deit3_large_patch16_224(**kwargs):
    return _deit3(1024, 24, 16, **kwargs)
