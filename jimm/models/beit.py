"""BEiT in flax nnx. Mirrors timm.models.beit.

Relative position bias in every block (with extra entries for the class token),
query and value biases only, layer scale, no absolute position embedding, and a
mean-pooled LayerNorm head.
"""

import math
from functools import lru_cache

import jax.numpy as jnp
import numpy as np
from flax import nnx

from ..attention import dot_product_attention
from ..layers import ClassifierMixin, DropPath, PatchEmbed, gelu
from ..registry import _cfg, register_model


def _trunc_normal(std):
    return nnx.initializers.truncated_normal(std)


@lru_cache
def _relative_position_index(window):
    """(N + 1, N + 1) table index; the last three rows of the table serve the class token."""
    wh, ww = window
    num_relative = (2 * wh - 1) * (2 * ww - 1) + 3
    grid = np.stack(np.meshgrid(np.arange(wh), np.arange(ww), indexing="ij")).reshape(2, -1)
    rel = (grid[:, :, None] - grid[:, None, :]).transpose(1, 2, 0)
    index = np.zeros((wh * ww + 1,) * 2, dtype=np.int32)
    index[1:, 1:] = (rel[..., 0] + wh - 1) * (2 * ww - 1) + rel[..., 1] + ww - 1
    index[0, 0:] = num_relative - 3  # class token to patches
    index[0:, 0] = num_relative - 2  # patches to class token
    index[0, 0] = num_relative - 1
    return index


class Attention(nnx.Module):
    def __init__(
        self, dim, num_heads, window_size, qkv_bias=True, drop=0.0, proj_std=0.02, *, rngs
    ):
        self.num_heads = num_heads
        self.window_size = tuple(window_size)
        self.qkv = nnx.Linear(
            dim, dim * 3, use_bias=False, kernel_init=_trunc_normal(0.02), rngs=rngs
        )
        # Query and value biases only; the key bias is fixed at zero.
        self.q_bias = nnx.Param(jnp.zeros(dim)) if qkv_bias else None
        self.v_bias = nnx.Param(jnp.zeros(dim)) if qkv_bias else None
        wh, ww = self.window_size
        num_relative = (2 * wh - 1) * (2 * ww - 1) + 3
        self.relative_position_bias_table = nnx.Param(jnp.zeros((num_relative, num_heads)))
        self.proj = nnx.Linear(dim, dim, kernel_init=_trunc_normal(proj_std), rngs=rngs)
        self.drop = nnx.Dropout(drop, rngs=rngs)

    def __call__(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        if self.q_bias is not None:
            q_bias = self.q_bias[...]
            bias = jnp.concatenate([q_bias, jnp.zeros_like(q_bias), self.v_bias[...]])
            qkv = qkv + bias.astype(qkv.dtype)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        table = self.relative_position_bias_table[...]
        rel_bias = table[_relative_position_index(self.window_size)].transpose(2, 0, 1)[None]
        rel_bias = rel_bias.astype(jnp.promote_types(table.dtype, jnp.float32))
        x = dot_product_attention(q, k, v, bias=rel_bias).reshape(B, N, C)
        return self.drop(self.proj(x))


class Mlp(nnx.Module):
    def __init__(self, dim, hidden_dim, drop=0.0, fc2_std=0.02, *, rngs):
        self.fc1 = nnx.Linear(dim, hidden_dim, kernel_init=_trunc_normal(0.02), rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, dim, kernel_init=_trunc_normal(fc2_std), rngs=rngs)
        self.drop = nnx.Dropout(drop, rngs=rngs)

    def __call__(self, x):
        return self.drop(self.fc2(self.drop(gelu(self.fc1(x)))))


class Block(nnx.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        init_values=None,
        layer_id=0,
        *,
        rngs,
    ):
        # timm fix_init_weight: output projections shrink with depth.
        out_std = 0.02 / math.sqrt(2.0 * (layer_id + 1))
        self.norm1 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.attn = Attention(dim, num_heads, window_size, drop=drop, proj_std=out_std, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop, fc2_std=out_std, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)
        self.gamma1 = nnx.Param(jnp.full((dim,), init_values)) if init_values else None
        self.gamma2 = nnx.Param(jnp.full((dim,), init_values)) if init_values else None

    def __call__(self, x):
        y = self.attn(self.norm1(x))
        x = x + self.drop_path(y if self.gamma1 is None else self.gamma1[...] * y)
        y = self.mlp(self.norm2(x))
        return x + self.drop_path(y if self.gamma2 is None else self.gamma2[...] * y)


class Beit(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        global_pool="avg",
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        init_values=None,
        head_init_scale=0.001,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, rngs=rngs)
        grid = self.patch_embed.grid_size
        self.cls_token = nnx.Param(_trunc_normal(0.02)(rngs.params(), (1, 1, embed_dim)))
        self.pos_drop = nnx.Dropout(drop_rate, rngs=rngs)
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nnx.List(
            [
                Block(
                    embed_dim,
                    num_heads,
                    grid,
                    mlp_ratio,
                    drop_rate,
                    dpr[i],
                    init_values,
                    layer_id=i,
                    rngs=rngs,
                )
                for i in range(depth)
            ]
        )
        # Mean pooling normalizes the pooled features (fc_norm) instead of every token.
        use_fc_norm = global_pool == "avg"
        self.norm = None if use_fc_norm else nnx.LayerNorm(embed_dim, epsilon=1e-6, rngs=rngs)
        self.fc_norm = nnx.LayerNorm(embed_dim, epsilon=1e-6, rngs=rngs) if use_fc_norm else None
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = (
            nnx.Linear(
                embed_dim,
                num_classes,
                kernel_init=_trunc_normal(0.02 * head_init_scale),
                rngs=rngs,
            )
            if num_classes > 0
            else None
        )

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).reshape(B, -1, self.num_features)
        cls_token = jnp.broadcast_to(self.cls_token[...], (B, 1, self.num_features))
        x = self.pos_drop(jnp.concatenate([cls_token, x], axis=1))
        for blk in self.blocks:
            x = blk(x)
        return x if self.norm is None else self.norm(x)

    def forward_head(self, x):
        if self.global_pool:
            x = jnp.mean(x[:, 1:], axis=1) if self.global_pool == "avg" else x[:, 0]
        if self.fc_norm is not None:
            x = self.fc_norm(x)
        x = self.head_drop(x)
        return self.head(x) if self.head is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


def _beit(img_size, embed_dim, depth, num_heads, init_values, **kwargs):
    model = Beit(
        img_size=img_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        init_values=init_values,
        **kwargs,
    )
    model.default_cfg = _cfg(
        input_size=(3, img_size, img_size),
        crop_pct=0.9 if img_size == 224 else 1.0,
        interpolation="bicubic",
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )
    return model


@register_model
def beit_base_patch16_224(**kwargs):
    return _beit(224, 768, 12, 12, 0.1, **kwargs)


@register_model
def beit_base_patch16_384(**kwargs):
    return _beit(384, 768, 12, 12, 0.1, **kwargs)


@register_model
def beit_large_patch16_224(**kwargs):
    return _beit(224, 1024, 24, 16, 1e-5, **kwargs)


@register_model
def beit_large_patch16_384(**kwargs):
    return _beit(384, 1024, 24, 16, 1e-5, **kwargs)
