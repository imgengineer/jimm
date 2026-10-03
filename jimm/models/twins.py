"""Twins (PCPVT and SVT) in flax nnx, NHWC. Mirrors timm.models.twins.

Each stage embeds patches with a strided convolution and LayerNorm, applies a
positional encoding generator (a residual depthwise 3x3 convolution) after
its first block, and runs pre-norm transformer blocks. PCPVT blocks use
global attention whose keys and values come from a strided-convolution
subsampled map; SVT alternates locally grouped attention within 7x7 windows
with that global subsampled attention. The head averages the normalized
tokens.
"""

import math

import jax.numpy as jnp
from flax import nnx

from ..attention import dot_product_attention
from ..layers import ClassifierMixin, DropPath, Mlp
from ..registry import _cfg, register_model

_init = nnx.initializers.truncated_normal(0.02)


def _conv_init(kernel, out_chs, groups=1):
    # timm: normal(0, sqrt(2 / fan_out)), fan_out = k * k * out_channels / groups
    return nnx.initializers.normal(math.sqrt(2.0 / (kernel * kernel * out_chs // groups)))


class LocallyGroupedAttn(nnx.Module):
    def __init__(self, dim, num_heads=8, ws=7, *, rngs):
        self.num_heads, self.ws = num_heads, ws
        self.qkv = nnx.Linear(dim, 3 * dim, kernel_init=_init, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, kernel_init=_init, rngs=rngs)

    def __call__(self, x, size):
        B, N, C = x.shape
        H, W = size
        ws = self.ws
        pad_b, pad_r = -H % ws, -W % ws
        # timm zero-pads the normalized tokens, so padded positions take part as keys.
        x = jnp.pad(x.reshape(B, H, W, C), ((0, 0), (0, pad_b), (0, pad_r), (0, 0)))
        nh, nw = (H + pad_b) // ws, (W + pad_r) // ws
        x = x.reshape(B, nh, ws, nw, ws, C).transpose(0, 1, 3, 2, 4, 5)
        qkv = self.qkv(x.reshape(B * nh * nw, ws * ws, C))
        qkv = qkv.reshape(B * nh * nw, ws * ws, 3, self.num_heads, C // self.num_heads)
        x = dot_product_attention(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2])
        x = x.reshape(B, nh, nw, ws, ws, C).transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, nh * ws, nw * ws, C)[:, :H, :W]
        return self.proj(x.reshape(B, N, C))


class GlobalSubSampleAttn(nnx.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=1, *, rngs):
        self.num_heads = num_heads
        self.q = nnx.Linear(dim, dim, kernel_init=_init, rngs=rngs)
        self.kv = nnx.Linear(dim, 2 * dim, kernel_init=_init, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, kernel_init=_init, rngs=rngs)
        if sr_ratio > 1:
            self.sr = nnx.Conv(
                dim,
                dim,
                (sr_ratio, sr_ratio),
                strides=(sr_ratio, sr_ratio),
                padding="VALID",
                kernel_init=_conv_init(sr_ratio, dim),
                rngs=rngs,
            )
            self.norm = nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)
        else:
            self.sr = self.norm = None

    def __call__(self, x, size):
        B, N, C = x.shape
        h = self.num_heads
        q = self.q(x).reshape(B, N, h, C // h)
        if self.sr is not None:
            x = self.sr(x.reshape(B, *size, C))
            x = self.norm(x.reshape(B, -1, C))
        kv = self.kv(x).reshape(B, -1, 2, h, C // h)
        x = dot_product_attention(q, kv[:, :, 0], kv[:, :, 1])
        return self.proj(x.reshape(B, N, C))


class Block(nnx.Module):
    def __init__(
        self, dim, num_heads, mlp_ratio=4.0, drop=0.0, drop_path=0.0, sr_ratio=1, ws=1, *, rngs
    ):
        self.norm1 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.attn = (
            GlobalSubSampleAttn(dim, num_heads, sr_ratio, rngs=rngs)
            if ws == 1
            else LocallyGroupedAttn(dim, num_heads, ws, rngs=rngs)
        )
        self.norm2 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop, kernel_init=_init, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x, size):
        x = x + self.drop_path(self.attn(self.norm1(x), size))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchEmbed(nnx.Module):
    def __init__(self, patch_size, in_chans, embed_dim, *, rngs):
        self.proj = nnx.Conv(
            in_chans,
            embed_dim,
            (patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            kernel_init=_conv_init(patch_size, embed_dim),
            rngs=rngs,
        )
        self.norm = nnx.LayerNorm(embed_dim, epsilon=1e-5, rngs=rngs)

    def __call__(self, x):
        x = self.proj(x)
        B, H, W, C = x.shape
        return self.norm(x.reshape(B, H * W, C)), (H, W)


class PosConv(nnx.Module):
    """Positional encoding generator: a residual depthwise 3x3 convolution."""

    def __init__(self, dim, *, rngs):
        self.proj = nnx.Conv(
            dim,
            dim,
            (3, 3),
            padding=((1, 1), (1, 1)),
            feature_group_count=dim,
            kernel_init=_conv_init(3, dim, dim),
            rngs=rngs,
        )

    def __call__(self, x, size):
        B, N, C = x.shape
        x = x.reshape(B, *size, C)
        return (self.proj(x) + x).reshape(B, N, C)


class Twins(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"

    def __init__(
        self,
        patch_size=4,
        embed_dims=(64, 128, 256, 512),
        num_heads=(1, 2, 4, 8),
        mlp_ratios=(4, 4, 4, 4),
        depths=(3, 4, 6, 3),
        sr_ratios=(8, 4, 2, 1),
        wss=None,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        pos_drop_rate=0.0,
        proj_drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        embeds, prev = [], in_chans
        for i, dim in enumerate(embed_dims):
            embeds.append(PatchEmbed(patch_size if i == 0 else 2, prev, dim, rngs=rngs))
            prev = dim
        self.patch_embeds = nnx.List(embeds)
        self.pos_drop = nnx.Dropout(pos_drop_rate, rngs=rngs)
        total = sum(depths)
        dpr = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]
        stages, cur = [], 0
        for k, (dim, depth) in enumerate(zip(embed_dims, depths)):
            stages.append(
                nnx.List(
                    [
                        Block(
                            dim,
                            num_heads[k],
                            mlp_ratios[k],
                            proj_drop_rate,
                            dpr[cur + i],
                            sr_ratios[k],
                            # SVT alternates local (even blocks) and global attention.
                            ws=1 if wss is None or i % 2 == 1 else wss[k],
                            rngs=rngs,
                        )
                        for i in range(depth)
                    ]
                )
            )
            cur += depth
        self.blocks = nnx.List(stages)
        self.pos_block = nnx.List([PosConv(dim, rngs=rngs) for dim in embed_dims])
        self.num_features = embed_dims[-1]
        self.norm = nnx.LayerNorm(self.num_features, epsilon=1e-6, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = (
            nnx.Linear(self.num_features, num_classes, kernel_init=_init, rngs=rngs)
            if num_classes > 0
            else None
        )

    def forward_features(self, x):
        for i, (embed, blocks, pos_blk) in enumerate(
            zip(self.patch_embeds, self.blocks, self.pos_block)
        ):
            x, size = embed(x)
            x = self.pos_drop(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                if j == 0:
                    x = pos_blk(x, size)
            if i < len(self.blocks) - 1:
                x = x.reshape(x.shape[0], *size, -1)
        return self.norm(x)

    def forward_head(self, x):
        if self.global_pool == "avg":
            x = jnp.mean(x, axis=1)
        x = self.head_drop(x)
        return self.head(x) if self.head is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_PCPVT = dict(embed_dims=(64, 128, 320, 512), num_heads=(1, 2, 5, 8), mlp_ratios=(8, 8, 4, 4))
_SVT = dict(mlp_ratios=(4, 4, 4, 4), wss=(7, 7, 7, 7))
_CFGS = {
    "twins_pcpvt_small": dict(**_PCPVT, depths=(3, 4, 6, 3)),
    "twins_pcpvt_base": dict(**_PCPVT, depths=(3, 4, 18, 3)),
    "twins_pcpvt_large": dict(**_PCPVT, depths=(3, 8, 27, 3)),
    "twins_svt_small": dict(
        **_SVT, embed_dims=(64, 128, 256, 512), num_heads=(2, 4, 8, 16), depths=(2, 2, 10, 4)
    ),
    "twins_svt_base": dict(
        **_SVT, embed_dims=(96, 192, 384, 768), num_heads=(3, 6, 12, 24), depths=(2, 2, 18, 2)
    ),
    "twins_svt_large": dict(
        **_SVT, embed_dims=(128, 256, 512, 1024), num_heads=(4, 8, 16, 32), depths=(2, 2, 18, 2)
    ),
}


def _make(name):
    cfg = _CFGS[name]

    def entry(**kwargs):
        model = Twins(**{**cfg, **kwargs})
        model.default_cfg = _cfg(crop_pct=0.9, interpolation="bicubic", fixed_input_size=True)
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
