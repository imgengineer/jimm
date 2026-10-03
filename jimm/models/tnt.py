"""TNT (Transformer iN Transformer) in flax nnx, NHWC. Mirrors timm.models.tnt.

Each 16x16 patch is embedded as a 4x4 grid of "pixel" tokens by a 7x7
stride-4 conv (applied per patch, or for the legacy variant to the whole
image and then unfolded) plus a learned pixel position embedding. Pixel
tokens are flattened, normalized and projected into patch tokens with a class
token and position embedding. Each block runs an inner transformer over the
pixel tokens of every patch, adds their normalized projection to the patch
tokens, then runs an outer transformer over the patch tokens.
"""

import jax.numpy as jnp
from flax import nnx

from ..attention import dot_product_attention
from ..features import _select_features
from ..layers import ClassifierMixin, DropPath, Mlp
from ..registry import _cfg, register_model

_trunc = nnx.initializers.truncated_normal(0.02)
_LINEAR = dict(kernel_init=_trunc, bias_init=nnx.initializers.zeros)


def _ln(dim, *, rngs):
    return nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)


def _mlp(dim, hidden, drop, rngs):
    return Mlp(dim, hidden, drop, **_LINEAR, rngs=rngs)


class Attention(nnx.Module):
    """Attention with a fused q/k projection and a separate v projection."""

    def __init__(self, dim, num_heads, qkv_bias=False, proj_drop=0.0, *, rngs):
        self.num_heads = num_heads
        self.qk = nnx.Linear(dim, dim * 2, use_bias=qkv_bias, **_LINEAR, rngs=rngs)
        self.v = nnx.Linear(dim, dim, use_bias=qkv_bias, **_LINEAR, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, **_LINEAR, rngs=rngs)
        self.proj_drop = nnx.Dropout(proj_drop, rngs=rngs)

    def __call__(self, x):
        B, N, C = x.shape
        h = self.num_heads
        qk = self.qk(x).reshape(B, N, 2, h, C // h)
        v = self.v(x).reshape(B, N, h, C // h)
        x = dot_product_attention(qk[:, :, 0], qk[:, :, 1], v).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Block(nnx.Module):
    def __init__(
        self, dim, dim_out, num_pixel, num_heads_in, num_heads_out, mlp_ratio, qkv_bias, drop,
        drop_path, legacy, *, rngs,
    ):  # fmt: skip
        self.norm_in = _ln(dim, rngs=rngs)
        self.attn_in = Attention(dim, num_heads_in, qkv_bias, drop, rngs=rngs)
        self.norm_mlp_in = _ln(dim, rngs=rngs)
        self.mlp_in = _mlp(dim, dim * 4, drop, rngs)
        self.legacy = legacy
        if legacy:
            self.norm1_proj = _ln(dim, rngs=rngs)
            self.proj = nnx.Linear(dim * num_pixel, dim_out, **_LINEAR, rngs=rngs)
            self.norm2_proj = None
        else:
            self.norm1_proj = _ln(dim * num_pixel, rngs=rngs)
            self.proj = nnx.Linear(dim * num_pixel, dim_out, use_bias=False, **_LINEAR, rngs=rngs)
            self.norm2_proj = _ln(dim_out, rngs=rngs)
        self.norm_out = _ln(dim_out, rngs=rngs)
        self.attn_out = Attention(dim_out, num_heads_out, qkv_bias, drop, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)
        self.norm_mlp = _ln(dim_out, rngs=rngs)
        self.mlp = _mlp(dim_out, int(dim_out * mlp_ratio), drop, rngs)

    def __call__(self, pixel_embed, patch_embed):
        pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
        pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
        B, N, _ = patch_embed.shape
        if self.legacy:
            add = self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))
        else:
            add = self.norm2_proj(self.proj(self.norm1_proj(pixel_embed.reshape(B, N - 1, -1))))
        patch_embed = jnp.concatenate([patch_embed[:, :1], patch_embed[:, 1:] + add], axis=1)
        patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
        patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        return pixel_embed, patch_embed


class PixelEmbed(nnx.Module):
    def __init__(self, patch_size, in_chans, in_dim, stride, legacy, *, rngs):
        self.patch_size, self.legacy = patch_size, legacy
        self.new_patch_size = -(-patch_size // stride)
        self.proj = nnx.Conv(
            in_chans, in_dim, (7, 7), strides=stride, padding=((3, 3), (3, 3)), rngs=rngs
        )

    def __call__(self, x, pixel_pos):
        B, H, W, C = x.shape
        p, q = self.patch_size, self.new_patch_size
        gh, gw = H // p, W // p
        if self.legacy:
            x = self.proj(x)
            x = x.reshape(B, gh, q, gw, q, -1).transpose(0, 1, 3, 2, 4, 5)
            x = x.reshape(B * gh * gw, q, q, -1)
        else:
            x = x.reshape(B, gh, p, gw, p, C).transpose(0, 1, 3, 2, 4, 5)
            x = self.proj(x.reshape(B * gh * gw, p, p, C))
        x = x + pixel_pos
        return x.reshape(B * gh * gw, q * q, -1), gh * gw


class TNT(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"
    _default_global_pool = "token"

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        global_pool="token",
        embed_dim=768,
        inner_dim=48,
        depth=12,
        num_heads_inner=4,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        first_stride=4,
        legacy=False,
        drop_rate=0.0,
        proj_drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        assert global_pool in ("", "token", "avg")
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dim
        self.pixel_embed = PixelEmbed(
            patch_size, in_chans, inner_dim, first_stride, legacy, rngs=rngs
        )
        q = self.pixel_embed.new_patch_size
        num_patches = (img_size // patch_size) ** 2
        num_pixel = q * q
        self.num_patches = num_patches
        self.norm1_proj = _ln(num_pixel * inner_dim, rngs=rngs)
        self.proj = nnx.Linear(num_pixel * inner_dim, embed_dim, **_LINEAR, rngs=rngs)
        self.norm2_proj = _ln(embed_dim, rngs=rngs)
        self.cls_token = nnx.Param(_trunc(rngs.params(), (1, 1, embed_dim)))
        self.patch_pos = nnx.Param(_trunc(rngs.params(), (1, num_patches + 1, embed_dim)))
        self.pixel_pos = nnx.Param(_trunc(rngs.params(), (1, q, q, inner_dim)))
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nnx.List(
            [
                Block(
                    inner_dim,
                    embed_dim,
                    num_pixel,
                    num_heads_inner,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    proj_drop_rate,
                    dpr[i],
                    legacy,
                    rngs=rngs,
                )  # fmt: skip
                for i in range(depth)
            ]
        )
        self.norm = _ln(embed_dim, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = self._make_head(num_classes, rngs)

    def _make_head(self, num_classes, rngs):
        if num_classes <= 0:
            return None
        return nnx.Linear(self.num_features, num_classes, **_LINEAR, rngs=rngs)

    def reset_classifier(self, num_classes, global_pool=None):
        if global_pool is not None:
            assert global_pool in ("", "token", "avg")
            self.global_pool = global_pool
        self.num_classes = num_classes
        self.head = self._make_head(num_classes, nnx.Rngs(0))

    def _forward_features(self, x, intermediates=None):
        B = x.shape[0]
        pixel_embed, n = self.pixel_embed(x, self.pixel_pos[...])
        patch = self.norm2_proj(self.proj(self.norm1_proj(pixel_embed.reshape(B, n, -1))))
        cls = jnp.broadcast_to(self.cls_token[...], (B, 1, patch.shape[-1]))
        patch = jnp.concatenate([cls, patch], axis=1) + self.patch_pos[...]
        for blk in self.blocks:
            pixel_embed, patch = blk(pixel_embed, patch)
            if intermediates is not None:
                intermediates.append(patch)
        return self.norm(patch)

    def forward_features(self, x):
        return self._forward_features(x)

    def forward_intermediates(self, x, out_indices=None):
        """Patch-token block outputs, followed by normalized features."""
        features = []
        features.append(self._forward_features(x, features))
        return _select_features(features, out_indices)

    def forward_head(self, x):
        if self.global_pool == "avg":
            x = x[:, 1:].mean(axis=1)
        elif self.global_pool == "token":
            x = x[:, 0]
        x = self.head_drop(x)
        return self.head(x) if self.head is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_CFGS = {
    "tnt_s_legacy_patch16_224": dict(embed_dim=384, inner_dim=24, num_heads=6, legacy=True),
    "tnt_s_patch16_224": dict(embed_dim=384, inner_dim=24, num_heads=6),
    "tnt_b_patch16_224": dict(embed_dim=640, inner_dim=40, num_heads=10),
}


def _make(name):
    cfg = _CFGS[name]

    def entry(**kwargs):
        model = TNT(**{**cfg, **kwargs})
        model.default_cfg = _cfg(
            crop_pct=0.9,
            interpolation="bicubic",
            fixed_input_size=True,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
