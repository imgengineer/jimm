"""Swin Transformer V2 in flax nnx, NHWC. Mirrors timm.models.swin_transformer_v2.

Scaled cosine window attention with a continuous relative position bias (an MLP
over log-spaced coordinates), "res-post-norm" blocks that normalize each branch
before the residual sum, and patch merging that normalizes after the reduction.
"""

import math
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from ..attention import dot_product_attention
from ..layers import ClassifierMixin, DropPath, Mlp
from ..registry import _cfg, register_model
from .swin_transformer import window_partition, window_reverse

_init = nnx.initializers.truncated_normal(0.02)


def _layer_norm(dim, *, rngs):
    return nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)


@lru_cache
def _relative_positions(window, pretrained_window):
    """Log-spaced relative coordinates (2Wh-1, 2Ww-1, 2) and the (N, N) table index."""
    wh, ww = window
    coords = np.stack(
        np.meshgrid(
            np.arange(-(wh - 1), wh, dtype=np.float32),
            np.arange(-(ww - 1), ww, dtype=np.float32),
            indexing="ij",
        ),
        axis=-1,
    )
    scale = pretrained_window if pretrained_window[0] > 0 else window
    coords /= np.array([scale[0] - 1, scale[1] - 1], dtype=np.float32)
    coords *= 8  # normalize to [-8, 8], then log-space
    table = np.sign(coords) * np.log2(np.abs(coords) + 1.0) / 3.0
    grid = np.stack(np.meshgrid(np.arange(wh), np.arange(ww), indexing="ij")).reshape(2, -1)
    rel = (grid[:, :, None] - grid[:, None, :]).transpose(1, 2, 0)
    index = (rel[..., 0] + wh - 1) * (2 * ww - 1) + rel[..., 1] + ww - 1
    return table.astype(np.float32), index


class WindowAttention(nnx.Module):
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        drop=0.0,
        pretrained_window_size=(0, 0),
        *,
        rngs,
    ):
        self.window_size = tuple(window_size)
        self.pretrained_window_size = tuple(pretrained_window_size)
        self.num_heads = num_heads
        self.logit_scale = nnx.Param(jnp.full((num_heads, 1, 1), math.log(10.0)))
        # timm cpb_mlp: Linear(2, 512) -> ReLU -> Linear(512, heads, no bias)
        self.cpb_fc1 = nnx.Linear(2, 512, kernel_init=_init, rngs=rngs)
        self.cpb_fc2 = nnx.Linear(512, num_heads, use_bias=False, kernel_init=_init, rngs=rngs)
        self.qkv = nnx.Linear(dim, dim * 3, use_bias=False, kernel_init=_init, rngs=rngs)
        # Query and value biases only; the key bias is fixed at zero.
        self.q_bias = nnx.Param(jnp.zeros(dim)) if qkv_bias else None
        self.v_bias = nnx.Param(jnp.zeros(dim)) if qkv_bias else None
        self.proj = nnx.Linear(dim, dim, kernel_init=_init, rngs=rngs)
        self.drop = nnx.Dropout(drop, rngs=rngs)

    def __call__(self, x, mask=None):
        B, N, C = x.shape
        H, D = self.num_heads, C // self.num_heads
        qkv = self.qkv(x)
        if self.q_bias is not None:
            q_bias = self.q_bias[...]
            bias = jnp.concatenate([q_bias, jnp.zeros_like(q_bias), self.v_bias[...]])
            qkv = qkv + bias.astype(qkv.dtype)
        qkv = qkv.reshape(B, N, 3, H, D)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        # Cosine attention (F.normalize, eps 1e-12) with a clamped learnable temperature.
        q = q / jnp.maximum(jnp.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
        k = k / jnp.maximum(jnp.linalg.norm(k, axis=-1, keepdims=True), 1e-12)
        scale = jnp.exp(jnp.minimum(self.logit_scale[...], math.log(100.0))).reshape(1, 1, H, 1)
        # dot_product_attention divides logits by sqrt(head_dim); fold that into q.
        q = q * (scale * math.sqrt(D)).astype(q.dtype)
        table, index = _relative_positions(self.window_size, self.pretrained_window_size)
        rel_bias = self.cpb_fc2(nnx.relu(self.cpb_fc1(jnp.asarray(table))))
        rel_bias = rel_bias.reshape(-1, H)[index].transpose(2, 0, 1)  # (heads, N, N)
        bias = 16 * jax.nn.sigmoid(rel_bias.astype(jnp.promote_types(rel_bias.dtype, jnp.float32)))
        if mask is not None:
            nW = mask.shape[0]
            window_bias = jnp.broadcast_to(
                mask[None, :, None, :, :], (B // nW, nW, 1, N, N)
            ).reshape(B, 1, N, N)
            bias = bias[None] + window_bias
        x = dot_product_attention(q, k, v, bias=bias).reshape(B, N, C)
        return self.drop(self.proj(x))


class SwinTransformerV2Block(nnx.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=8,
        shift=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        drop_path=0.0,
        pretrained_window_size=0,
        *,
        rngs,
    ):
        if min(input_resolution) <= window_size:
            # timm: windows no larger than the map, and no shift when one window covers it.
            shift = 0
            window_size = min(input_resolution)
        self.ws, self.shift = window_size, shift
        self.attn = WindowAttention(
            dim,
            (window_size, window_size),
            num_heads,
            qkv_bias,
            drop,
            (pretrained_window_size, pretrained_window_size),
            rngs=rngs,
        )
        self.norm1 = _layer_norm(dim, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop, rngs=rngs)
        self.norm2 = _layer_norm(dim, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)
        attn_mask = None
        if shift > 0:
            H, W = input_resolution
            img_mask = jnp.zeros((1, H, W, 1))
            bands = (slice(0, -window_size), slice(-window_size, -shift), slice(-shift, None))
            for i, hs in enumerate(bands):
                for j, ws_ in enumerate(bands):
                    img_mask = img_mask.at[:, hs, ws_, :].set(3 * i + j)
            mask_windows = window_partition(img_mask, window_size).reshape(-1, window_size**2)
            diff = mask_windows[:, None, :] - mask_windows[:, :, None]
            attn_mask = jnp.where(diff != 0, -100.0, 0.0)
        # nnx.Variable: raw array attributes break nnx.cached_partial graph flattening
        self.attn_mask = nnx.Variable(attn_mask) if attn_mask is not None else None

    def _attn(self, x):
        B, H, W, C = x.shape
        if self.shift > 0:
            x = jnp.roll(x, (-self.shift, -self.shift), axis=(1, 2))
        mask = self.attn_mask[...] if self.attn_mask is not None else None
        x = self.attn(window_partition(x, self.ws), mask)
        x = window_reverse(x, self.ws, H, W, B)
        if self.shift > 0:
            x = jnp.roll(x, (self.shift, self.shift), axis=(1, 2))
        return x

    def __call__(self, x):
        x = x + self.drop_path(self.norm1(self._attn(x)))
        return x + self.drop_path(self.norm2(self.mlp(x)))


class PatchMerging(nnx.Module):
    """2x2 patch merging; SwinV2 normalizes after the linear reduction."""

    def __init__(self, dim, *, rngs):
        self.reduction = nnx.Linear(4 * dim, 2 * dim, use_bias=False, kernel_init=_init, rngs=rngs)
        self.norm = _layer_norm(2 * dim, rngs=rngs)

    def __call__(self, x):
        x0, x1 = x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :]
        x2, x3 = x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]
        return self.norm(self.reduction(jnp.concatenate([x0, x1, x2, x3], axis=-1)))


class SwinTransformerV2Stage(nnx.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        downsample=False,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=None,
        pretrained_window_size=0,
        *,
        rngs,
    ):
        self.downsample = PatchMerging(dim, rngs=rngs) if downsample else None
        if downsample:
            dim, input_resolution = 2 * dim, tuple(r // 2 for r in input_resolution)
        drop_path = drop_path or [0.0] * depth
        self.blocks = nnx.List(
            [
                SwinTransformerV2Block(
                    dim,
                    input_resolution,
                    num_heads,
                    window_size,
                    shift=0 if i % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=drop_path[i],
                    pretrained_window_size=pretrained_window_size,
                    rngs=rngs,
                )
                for i in range(depth)
            ]
        )

    def __call__(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class SwinTransformerV2(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"

    def __init__(
        self,
        img_size=256,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        global_pool="avg",
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=8,
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        pretrained_window_sizes=(0, 0, 0, 0),
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dim * 2 ** (len(depths) - 1)
        self.patch_embed_conv = nnx.Conv(
            in_chans,
            embed_dim,
            (patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            rngs=rngs,
        )
        self.patch_norm = _layer_norm(embed_dim, rngs=rngs)
        res = img_size // patch_size
        dpr = [drop_path_rate * i / max(sum(depths) - 1, 1) for i in range(sum(depths))]
        self.stages = nnx.List(
            [
                SwinTransformerV2Stage(
                    embed_dim * 2 ** max(i - 1, 0),
                    (res // 2 ** max(i - 1, 0),) * 2,
                    depths[i],
                    num_heads[i],
                    window_size,
                    downsample=i > 0,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                    pretrained_window_size=pretrained_window_sizes[i],
                    rngs=rngs,
                )
                for i in range(len(depths))
            ]
        )
        self.norm = _layer_norm(self.num_features, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = (
            nnx.Linear(self.num_features, num_classes, kernel_init=_init, rngs=rngs)
            if num_classes > 0
            else None
        )

    def forward_features(self, x):
        x = self.patch_norm(self.patch_embed_conv(x))
        for stage in self.stages:
            x = stage(x)
        return self.norm(x)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_CFGS = {
    "swinv2_tiny_window8_256": (256, 8, 96, (2, 2, 6, 2), (3, 6, 12, 24)),
    "swinv2_tiny_window16_256": (256, 16, 96, (2, 2, 6, 2), (3, 6, 12, 24)),
    "swinv2_small_window8_256": (256, 8, 96, (2, 2, 18, 2), (3, 6, 12, 24)),
    "swinv2_small_window16_256": (256, 16, 96, (2, 2, 18, 2), (3, 6, 12, 24)),
    "swinv2_base_window8_256": (256, 8, 128, (2, 2, 18, 2), (4, 8, 16, 32)),
    "swinv2_base_window16_256": (256, 16, 128, (2, 2, 18, 2), (4, 8, 16, 32)),
    "swinv2_base_window12_192": (192, 12, 128, (2, 2, 18, 2), (4, 8, 16, 32)),
    "swinv2_large_window12_192": (192, 12, 192, (2, 2, 18, 2), (6, 12, 24, 48)),
}


def _make(name):
    img_size, window_size, embed_dim, depths, num_heads = _CFGS[name]

    def entry(**kwargs):
        model = SwinTransformerV2(
            img_size=img_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            **kwargs,
        )
        model.default_cfg = _cfg(
            input_size=(3, img_size, img_size), crop_pct=0.9, interpolation="bicubic"
        )
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
