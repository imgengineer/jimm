"""Swin Transformer V2 (Christoph Reich's implementation) in flax nnx, NHWC.

Mirrors timm.models.swin_transformer_v2_cr: scaled cosine window attention with a
meta-MLP position bias over log-spaced offsets, res-post-norm blocks whose norm
scales start at ``init_values``, extra main-branch norms (every ``extra_norm_period``
blocks and at the end of the network), and no separate final norm.
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

_xavier = nnx.initializers.xavier_uniform()


def _qkv_init(key, shape, dtype=jnp.float32):
    # timm treats q, k, and v as separate (dim, dim) projections for Xavier limits.
    limit = math.sqrt(6.0 / (shape[0] + shape[1] // 3))
    return jax.random.uniform(key, shape, dtype, -limit, limit)


def _layer_norm(dim, scale_init=nnx.initializers.ones, *, rngs):
    return nnx.LayerNorm(dim, epsilon=1e-5, scale_init=scale_init, rngs=rngs)


@lru_cache
def _log_relative_coordinates(window):
    """sign(d) * log(1 + |d|) for every (query, key) pair in a window: (N * N, 2)."""
    wh, ww = window
    grid = np.stack(
        np.meshgrid(np.arange(wh, dtype=np.float32), np.arange(ww, dtype=np.float32), indexing="ij")
    ).reshape(2, -1)
    rel = (grid[:, :, None] - grid[:, None, :]).transpose(1, 2, 0).reshape(-1, 2)
    return np.sign(rel) * np.log(1.0 + np.abs(rel))


class WindowMultiHeadAttention(nnx.Module):
    def __init__(self, dim, num_heads, window_size, drop=0.0, meta_hidden_dim=384, *, rngs):
        self.num_heads = num_heads
        self.window_size = tuple(window_size)
        self.qkv = nnx.Linear(dim, dim * 3, kernel_init=_qkv_init, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, kernel_init=_xavier, rngs=rngs)
        # timm meta_mlp: Linear(2, 384) -> ReLU -> Dropout(0.125) -> Linear(384, heads)
        self.meta_fc1 = nnx.Linear(2, meta_hidden_dim, kernel_init=_xavier, rngs=rngs)
        self.meta_drop = nnx.Dropout(0.125, rngs=rngs)
        self.meta_fc2 = nnx.Linear(meta_hidden_dim, num_heads, kernel_init=_xavier, rngs=rngs)
        self.logit_scale = nnx.Param(jnp.full((num_heads,), math.log(10.0)))
        self.drop = nnx.Dropout(drop, rngs=rngs)

    def __call__(self, x, mask=None):
        B, N, C = x.shape
        H, D = self.num_heads, C // self.num_heads
        qkv = self.qkv(x).reshape(B, N, 3, H, D)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q / jnp.maximum(jnp.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
        k = k / jnp.maximum(jnp.linalg.norm(k, axis=-1, keepdims=True), 1e-12)
        scale = jnp.exp(jnp.minimum(self.logit_scale[...], math.log(100.0))).reshape(1, 1, H, 1)
        # dot_product_attention divides logits by sqrt(head_dim); fold that into q.
        q = q * (scale * math.sqrt(D)).astype(q.dtype)
        coords = jnp.asarray(_log_relative_coordinates(self.window_size))
        rel_bias = self.meta_fc2(self.meta_drop(nnx.relu(self.meta_fc1(coords))))
        bias = rel_bias.T.reshape(H, N, N)
        bias = bias.astype(jnp.promote_types(bias.dtype, jnp.float32))
        if mask is not None:
            nW = mask.shape[0]
            window_bias = jnp.broadcast_to(
                mask[None, :, None, :, :], (B // nW, nW, 1, N, N)
            ).reshape(B, 1, N, N)
            bias = bias[None] + window_bias
        x = dot_product_attention(q, k, v, bias=bias).reshape(B, N, C)
        return self.drop(self.proj(x))


class SwinTransformerV2CrBlock(nnx.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift=0,
        mlp_ratio=4.0,
        init_values=0.0,
        drop=0.0,
        drop_path=0.0,
        extra_norm=False,
        *,
        rngs,
    ):
        if min(input_resolution) <= window_size:
            # timm: windows no larger than the map, and no shift when one window covers it.
            shift = 0
            window_size = min(input_resolution)
        self.ws, self.shift = window_size, shift
        self.attn = WindowMultiHeadAttention(dim, num_heads, (window_size,) * 2, drop, rngs=rngs)
        # timm initializes the residual-branch norm scales to init_values (0 by default).
        branch_init = (
            nnx.initializers.ones if init_values is None else nnx.initializers.constant(init_values)
        )
        self.norm1 = _layer_norm(dim, branch_init, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop, rngs=rngs)
        self.norm2 = _layer_norm(dim, branch_init, rngs=rngs)
        self.norm3 = _layer_norm(dim, rngs=rngs) if extra_norm else None
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
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x if self.norm3 is None else self.norm3(x)


class PatchMerging(nnx.Module):
    def __init__(self, dim, *, rngs):
        self.norm = _layer_norm(4 * dim, rngs=rngs)
        self.reduction = nnx.Linear(
            4 * dim, 2 * dim, use_bias=False, kernel_init=_xavier, rngs=rngs
        )

    def __call__(self, x):
        x0, x1 = x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :]
        x2, x3 = x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]
        return self.reduction(self.norm(jnp.concatenate([x0, x1, x2, x3], axis=-1)))


class SwinTransformerV2CrStage(nnx.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        downscale=False,
        mlp_ratio=4.0,
        init_values=0.0,
        drop=0.0,
        drop_path=None,
        extra_norm_period=0,
        extra_norm_stage=False,
        *,
        rngs,
    ):
        self.downsample = PatchMerging(dim, rngs=rngs) if downscale else None
        if downscale:
            dim, input_resolution = 2 * dim, tuple(r // 2 for r in input_resolution)
        drop_path = drop_path or [0.0] * depth

        def extra_norm(index):
            i = index + 1
            if extra_norm_period and i % extra_norm_period == 0:
                return True
            return extra_norm_stage and i == depth

        self.blocks = nnx.List(
            [
                SwinTransformerV2CrBlock(
                    dim,
                    input_resolution,
                    num_heads,
                    window_size,
                    shift=0 if i % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                    drop=drop,
                    drop_path=drop_path[i],
                    extra_norm=extra_norm(i),
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


class SwinTransformerV2Cr(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: str = "avg",
        embed_dim: int = 96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size: int | None = None,
        mlp_ratio: float = 4.0,
        init_values: float | None = 0.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        extra_norm_period: int = 0,
        extra_norm_stage: bool = False,
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
        window_size = window_size or res // 8  # timm's default window_ratio of 8
        dpr = [drop_path_rate * i / max(sum(depths) - 1, 1) for i in range(sum(depths))]
        self.stages = nnx.List(
            [
                SwinTransformerV2CrStage(
                    embed_dim * 2 ** max(i - 1, 0),
                    (res // 2 ** max(i - 1, 0),) * 2,
                    depths[i],
                    num_heads[i],
                    window_size,
                    downscale=i > 0,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                    drop=drop_rate,
                    drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                    extra_norm_period=extra_norm_period,
                    # The last stage ends with a main-branch norm instead of a final norm.
                    extra_norm_stage=extra_norm_stage or i == len(depths) - 1,
                    rngs=rngs,
                )
                for i in range(len(depths))
            ]
        )
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = (
            nnx.Linear(
                self.num_features,
                num_classes,
                kernel_init=nnx.initializers.zeros,
                rngs=rngs,
            )
            if num_classes > 0
            else None
        )

    def forward_features(self, x):
        x = self.patch_norm(self.patch_embed_conv(x))
        for stage in self.stages:
            x = stage(x)
        return x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_T, _S, _B = (
    (96, (2, 2, 6, 2), (3, 6, 12, 24)),
    (96, (2, 2, 18, 2), (3, 6, 12, 24)),
    (128, (2, 2, 18, 2), (4, 8, 16, 32)),
)
_L = (192, (2, 2, 18, 2), (6, 12, 24, 48))
_H224, _H384 = (352, (2, 2, 18, 2), (8, 16, 32, 64)), (352, (2, 2, 18, 2), (11, 22, 44, 88))
_G = (512, (2, 2, 42, 2), (16, 32, 64, 128))
_NS, _EXTRA = {"extra_norm_stage": True}, {"extra_norm_period": 6}
# name: ((embed_dim, depths, num_heads), image size, overrides) — timm registrations; the
# window is 1/32 of the image size.
_CFGS = {
    "swinv2_cr_tiny_384": (_T, 384, {}),
    "swinv2_cr_tiny_224": (_T, 224, {}),
    "swinv2_cr_tiny_ns_224": (_T, 224, _NS),
    "swinv2_cr_small_384": (_S, 384, {}),
    "swinv2_cr_small_224": (_S, 224, {}),
    "swinv2_cr_small_ns_224": (_S, 224, _NS),
    "swinv2_cr_small_ns_256": (_S, 256, _NS),
    "swinv2_cr_base_384": (_B, 384, {}),
    "swinv2_cr_base_224": (_B, 224, {}),
    "swinv2_cr_base_ns_224": (_B, 224, _NS),
    "swinv2_cr_large_384": (_L, 384, {}),
    "swinv2_cr_large_224": (_L, 224, {}),
    "swinv2_cr_huge_384": (_H384, 384, _EXTRA),
    "swinv2_cr_huge_224": (_H224, 224, _EXTRA),
    "swinv2_cr_giant_384": (_G, 384, _EXTRA),
    "swinv2_cr_giant_224": (_G, 224, _EXTRA),
}


def _make(name):
    (dim, depths, heads), size, overrides = _CFGS[name]
    cfg = dict(img_size=size, embed_dim=dim, depths=depths, num_heads=heads, **overrides)
    ev = {"crop_pct": 0.9} if size == 224 else {"input_size": (3, size, size), "crop_pct": 1.0}

    def entry(**kwargs):
        model = SwinTransformerV2Cr(**dict(cfg, **kwargs))
        model.default_cfg = _cfg(interpolation="bicubic", **ev)
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
