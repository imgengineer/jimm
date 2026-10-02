"""CaiT in flax nnx. Mirrors timm.models.cait (LayerScale + class-attention stage)."""

import jax
import jax.numpy as jnp
from flax import nnx

from ..attention import dot_product_attention
from ..features import _select_features
from ..layers import ClassifierMixin, DropPath, Mlp, PatchEmbed
from ..registry import _cfg, register_model


def _mix_heads(x, linear):
    """Apply ``linear`` across the head axis of (B, heads, Nq, Nk) attention maps.

    Written as broadcast multiply-adds so XLA fuses the mixing with the softmax;
    a Linear over a trailing four-wide head axis ran 1.5x slower in training.
    """
    kernel = linear.kernel[...].astype(x.dtype)
    out = linear.bias[...].astype(x.dtype)[None, :, None, None]
    for h in range(x.shape[1]):
        out = out + x[:, h, None] * kernel[h][None, :, None, None]
    return out


class TalkingHeadAttn(nnx.Module):
    """Self-attention whose logits and weights are mixed across heads (talking heads)."""

    def __init__(self, dim, num_heads, qkv_bias=True, *, rngs):
        self.num_heads = num_heads
        self.qkv = nnx.Linear(dim, dim * 3, use_bias=qkv_bias, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)
        self.proj_l = nnx.Linear(num_heads, num_heads, rngs=rngs)
        self.proj_w = nnx.Linear(num_heads, num_heads, rngs=rngs)

    def __call__(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        logits = jnp.einsum("bqhd,bkhd->bhqk", q * (C // self.num_heads) ** -0.5, k)
        logits = _mix_heads(
            logits.astype(jnp.promote_types(logits.dtype, jnp.float32)), self.proj_l
        )
        weights = _mix_heads(jax.nn.softmax(logits, axis=-1), self.proj_w).astype(v.dtype)
        return self.proj(jnp.einsum("bhqk,bkhd->bqhd", weights, v).reshape(B, N, C))


class LayerScaleBlock(nnx.Module):
    def __init__(
        self, dim, num_heads, mlp_ratio=4.0, drop=0.0, drop_path=0.0, init_values=1e-5, *, rngs
    ):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = TalkingHeadAttn(dim, num_heads, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop, rngs=rngs)
        self.gamma1 = nnx.Param(init_values * jnp.ones(dim))
        self.gamma2 = nnx.Param(init_values * jnp.ones(dim))

    def __call__(self, x):
        x = x + self.drop_path(self.gamma1[...] * self.attn(self.norm1(x)))
        return x + self.drop_path(self.gamma2[...] * self.mlp(self.norm2(x)))


class ClassAttentionBlock(nnx.Module):
    """Attention from cls token to patch tokens only (CaiT class-attention)."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop_path=0.0, init_values=1e-5, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q = nnx.Linear(dim, dim, rngs=rngs)
        self.k = nnx.Linear(dim, dim, rngs=rngs)
        self.v = nnx.Linear(dim, dim, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)
        self.gamma1 = nnx.Param(init_values * jnp.ones(dim))
        self.gamma2 = nnx.Param(init_values * jnp.ones(dim))

    def __call__(self, x):
        cls = x[:, :1]
        tokens = self.norm1(x)
        B, N, C = tokens.shape
        q = self.q(tokens[:, :1]).reshape(B, 1, self.num_heads, self.head_dim)
        k = self.k(tokens).reshape(B, N, self.num_heads, self.head_dim)
        v = self.v(tokens).reshape(B, N, self.num_heads, self.head_dim)
        cls_out = dot_product_attention(q, k, v).reshape(B, 1, C)
        cls = cls + self.drop_path(self.gamma1[...] * self.proj(cls_out))
        cls = cls + self.drop_path(self.gamma2[...] * self.mlp(self.norm2(cls)))
        return jnp.concatenate([cls, x[:, 1:]], axis=1)


class CaiT(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"
    _default_global_pool = ""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        global_pool="",
        embed_dim=384,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        depth_token_only=2,
        drop_rate=0.0,
        drop_path_rate=0.0,
        init_values=1e-5,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, rngs=rngs)
        n = self.patch_embed.num_patches
        self.pos_embed = nnx.Param(jnp.zeros((1, n, embed_dim)))
        self.cls_token = nnx.Param(jnp.zeros((1, 1, embed_dim)))
        # timm CaiT uses the same stochastic depth rate in every block.
        self.blocks = nnx.List(
            [
                LayerScaleBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    drop_rate,
                    drop_path_rate,
                    init_values,
                    rngs=rngs,
                )
                for _ in range(depth)
            ]
        )
        self.blocks_token_only = nnx.List(
            [
                ClassAttentionBlock(embed_dim, num_heads, mlp_ratio, 0.0, init_values, rngs=rngs)
                for _ in range(depth_token_only)
            ]
        )
        self.norm = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.head = nnx.Linear(embed_dim, num_classes, rngs=rngs) if num_classes > 0 else None

    def _forward_features(self, x, intermediates=None):
        B = x.shape[0]
        x = self.patch_embed(x).reshape(B, -1, self.num_features)
        x = x + self.pos_embed[...]
        for blk in self.blocks:
            x = blk(x)
            if intermediates is not None:
                intermediates.append(x)
        x = jnp.concatenate(
            [jnp.broadcast_to(self.cls_token[...], (B, 1, self.num_features)), x], axis=1
        )
        for blk in self.blocks_token_only:
            x = blk(x)
            if intermediates is not None:
                intermediates.append(x)
        return self.norm(x)

    def forward_features(self, x):
        return self._forward_features(x)

    def forward_intermediates(self, x, out_indices=None):
        """Patch/class-attention block outputs, followed by normalized features."""
        features = []
        features.append(self._forward_features(x, features))
        return _select_features(features, out_indices)

    def forward_head(self, x):
        x = x[:, 0]
        return self.head(x) if self.head is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


def _cait(embed_dim, depth, num_heads, depth_token_only=2, **kwargs):
    model = CaiT(
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        depth_token_only=depth_token_only,
        **kwargs,
    )
    model.default_cfg = _cfg(crop_pct=1.0, interpolation="bicubic")
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
    return _cait(768, 36, 16, init_values=1e-6, **kwargs)
