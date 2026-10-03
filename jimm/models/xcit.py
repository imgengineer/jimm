"""XCiT in flax nnx, NHWC. Mirrors timm.models.xcit.

A stack of strided 3x3 conv + BatchNorm layers embeds patches, a Fourier
positional encoding is added, and each block applies cross-covariance
attention (over channels), a local patch interaction module (two depthwise
3x3 convolutions around GELU and BatchNorm), and an MLP, each with layer
scale. Two class-attention blocks then update a class token that the head
classifies.
"""

import jax.numpy as jnp
from flax import nnx

from ..attention import dot_product_attention
from ..features import _select_features
from ..layers import BatchNorm, ClassifierMixin, DropPath, Mlp, gelu
from ..registry import _cfg, register_model
from ._conv import ConvNormAct
from .edgenext import CrossCovarianceAttn, PositionalEncodingFourier

_init = nnx.initializers.truncated_normal(0.02)


class ConvPatchEmbed(nnx.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768, *, rngs):
        if patch_size not in (8, 16):
            raise ValueError("XCiT patch embedding supports patch sizes 8 and 16")
        divisors = (8, 4, 2, 1) if patch_size == 16 else (4, 2, 1)
        chs = [in_chans, *(embed_dim // d for d in divisors)]
        self.proj = nnx.List(
            [ConvNormAct(chs[i], chs[i + 1], 3, 2, rngs=rngs) for i in range(len(chs) - 1)]
        )

    def __call__(self, x):
        for i, layer in enumerate(self.proj):
            x = layer(gelu(x) if i else x)
        return x


class LPI(nnx.Module):
    """Local patch interaction: depthwise 3x3 conv, GELU, BatchNorm, depthwise 3x3 conv."""

    def __init__(self, dim, kernel=3, *, rngs):
        pad = kernel // 2
        conv = dict(padding=((pad, pad), (pad, pad)), feature_group_count=dim, rngs=rngs)
        self.conv1 = nnx.Conv(dim, dim, (kernel, kernel), **conv)
        self.bn = BatchNorm(dim, epsilon=1e-5, rngs=rngs)
        self.conv2 = nnx.Conv(dim, dim, (kernel, kernel), **conv)

    def __call__(self, x, H, W):
        B, N, C = x.shape
        x = self.conv2(self.bn(gelu(self.conv1(x.reshape(B, H, W, C)))))
        return x.reshape(B, N, C)


class XCABlock(nnx.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        drop_path=0.0,
        eta=1.0,
        *,
        rngs,
    ):
        self.norm1 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.attn = CrossCovarianceAttn(dim, num_heads, qkv_bias, rngs=rngs)
        self.norm3 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.local_mp = LPI(dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop, kernel_init=_init, rngs=rngs)
        self.gamma1 = nnx.Param(jnp.full((dim,), eta))
        self.gamma3 = nnx.Param(jnp.full((dim,), eta))
        self.gamma2 = nnx.Param(jnp.full((dim,), eta))
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x, H, W):
        x = x + self.drop_path(self.gamma1[...] * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma3[...] * self.local_mp(self.norm3(x), H, W))
        return x + self.drop_path(self.gamma2[...] * self.mlp(self.norm2(x)))


class ClassAttn(nnx.Module):
    """timm CaiT class attention: only the class token queries."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, *, rngs):
        self.num_heads = num_heads
        self.q = nnx.Linear(dim, dim, use_bias=qkv_bias, kernel_init=_init, rngs=rngs)
        self.k = nnx.Linear(dim, dim, use_bias=qkv_bias, kernel_init=_init, rngs=rngs)
        self.v = nnx.Linear(dim, dim, use_bias=qkv_bias, kernel_init=_init, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, kernel_init=_init, rngs=rngs)

    def __call__(self, x):
        B, N, C = x.shape
        h = self.num_heads
        q = self.q(x[:, :1]).reshape(B, 1, h, C // h)
        k = self.k(x).reshape(B, N, h, C // h)
        v = self.v(x).reshape(B, N, h, C // h)
        return self.proj(dot_product_attention(q, k, v).reshape(B, 1, C))


class ClassAttentionBlock(nnx.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        eta=1.0,
        tokens_norm=False,
        *,
        rngs,
    ):
        self.norm1 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.attn = ClassAttn(dim, num_heads, qkv_bias, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop, kernel_init=_init, rngs=rngs)
        self.gamma1 = nnx.Param(jnp.full((dim,), eta))
        self.gamma2 = nnx.Param(jnp.full((dim,), eta))
        self.tokens_norm = tokens_norm

    def __call__(self, x):
        # As in timm, the normalized patch tokens are added to the patch tokens, the class
        # token MLP's residual is the normalized class token, and patch tokens are doubled.
        x_norm = self.norm1(x)
        x = x + self.gamma1[...] * jnp.concatenate([self.attn(x_norm), x_norm[:, 1:]], axis=1)
        if self.tokens_norm:
            x = self.norm2(x)
        else:
            x = jnp.concatenate([self.norm2(x[:, :1]), x[:, 1:]], axis=1)
        cls = self.gamma2[...] * self.mlp(x[:, :1])
        return x + jnp.concatenate([cls, x[:, 1:]], axis=1)


class Xcit(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"
    _default_global_pool = "token"

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        cls_attn_layers=2,
        use_pos_embed=True,
        eta=1.0,
        tokens_norm=False,
        num_classes=1000,
        in_chans=3,
        global_pool="token",
        drop_rate=0.0,
        pos_drop_rate=0.0,
        proj_drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        if img_size % patch_size:
            raise ValueError("patch_size should divide the image size evenly")
        self.num_classes, self.global_pool = num_classes, global_pool
        self.patch_embed = ConvPatchEmbed(patch_size, in_chans, embed_dim, rngs=rngs)
        self.cls_token = nnx.Param(_init(rngs.params(), (1, 1, embed_dim)))
        self.pos_embed = (
            PositionalEncodingFourier(dim=embed_dim, rngs=rngs) if use_pos_embed else None
        )
        self.pos_drop = nnx.Dropout(pos_drop_rate, rngs=rngs)
        # XCiT uses the same drop path rate in every block.
        self.blocks = nnx.List(
            [
                XCABlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    proj_drop_rate,
                    drop_path_rate,
                    eta,
                    rngs=rngs,
                )
                for _ in range(depth)
            ]
        )
        self.cls_attn_blocks = nnx.List(
            [
                ClassAttentionBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    drop_rate,
                    eta,
                    tokens_norm,
                    rngs=rngs,
                )
                for _ in range(cls_attn_layers)
            ]
        )
        self.norm = nnx.LayerNorm(embed_dim, epsilon=1e-6, rngs=rngs)
        self.num_features = embed_dim
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = (
            nnx.Linear(embed_dim, num_classes, kernel_init=_init, rngs=rngs)
            if num_classes > 0
            else None
        )

    def _forward_features(self, x, intermediates=None):
        x = self.patch_embed(x)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        if self.pos_embed is not None:
            x = x + self.pos_embed(H, W).reshape(1, H * W, C)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x, H, W)
            if intermediates is not None:
                intermediates.append(x)
        x = jnp.concatenate([jnp.broadcast_to(self.cls_token[...], (B, 1, C)), x], axis=1)
        for blk in self.cls_attn_blocks:
            x = blk(x)
        return self.norm(x)

    def forward_features(self, x):
        return self._forward_features(x)

    def forward_intermediates(self, x, out_indices=None):
        """Patch tokens after each XCA block, followed by the normalized output tokens."""
        features = []
        features.append(self._forward_features(x, features))
        return _select_features(features, out_indices)

    def forward_head(self, x):
        if self.global_pool:
            x = jnp.mean(x[:, 1:], axis=1) if self.global_pool == "avg" else x[:, 0]
        x = self.head_drop(x)
        return self.head(x) if self.head is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


XCiT = Xcit  # jimm's earlier name for the class

_SIZES = {  # embed_dim, depth, num_heads, eta, tokens_norm
    "nano_12": (128, 12, 4, 1.0, False),
    "tiny_12": (192, 12, 4, 1.0, True),
    "small_12": (384, 12, 8, 1.0, True),
    "tiny_24": (192, 24, 4, 1e-5, True),
    "small_24": (384, 24, 8, 1e-5, True),
    "medium_24": (512, 24, 8, 1e-5, True),
    "large_24": (768, 24, 16, 1e-5, True),
}


def _make(size, patch_size, img_size):
    embed_dim, depth, num_heads, eta, tokens_norm = _SIZES[size]

    def entry(**kwargs):
        model = Xcit(
            img_size,
            patch_size,
            embed_dim,
            depth,
            num_heads,
            eta=eta,
            tokens_norm=tokens_norm,
            **kwargs,
        )
        model.default_cfg = _cfg(
            input_size=(3, img_size, img_size),
            crop_pct=1.0,
            interpolation="bicubic",
            fixed_input_size=True,
        )
        return model

    entry.__name__ = f"xcit_{size}_p{patch_size}_{img_size}"
    return entry


for _size in _SIZES:
    for _patch in (16, 8):
        for _img in (224, 384):
            register_model(_make(_size, _patch, _img))
