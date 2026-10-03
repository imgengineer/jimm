"""DaViT in flax nnx, NHWC. Mirrors timm.models.davit.

Every stage alternates spatial blocks (self-attention within 7x7 windows)
and channel blocks (attention between channels across all tokens), each
preceded by a residual depthwise 3x3 convolutional position encoding and
followed by a second one and an MLP. A 7x7 stride-4 stem and 2x2 strided
convolutions after LayerNorm downsample; the head pools, normalizes, and
classifies. The ``_fl`` variants are Florence-2's image towers (12x12
windows, 3x3 downsampling, and dynamically scaled channel attention).
"""

import jax
import jax.numpy as jnp
from flax import nnx

from ..attention import dot_product_attention
from ..layers import ClassifierMixin, DropPath, Mlp, global_pool_nhwc
from ..registry import _cfg, register_model

_init = nnx.initializers.truncated_normal(0.02)


def _pad_to(x, multiple):
    H, W = x.shape[1:3]
    return jnp.pad(x, ((0, 0), (0, -H % multiple), (0, -W % multiple), (0, 0)))


class ConvPosEnc(nnx.Module):
    def __init__(self, dim, kernel=3, *, rngs):
        pad = kernel // 2
        self.proj = nnx.Conv(
            dim,
            dim,
            (kernel, kernel),
            padding=((pad, pad), (pad, pad)),
            feature_group_count=dim,
            rngs=rngs,
        )

    def __call__(self, x):
        return x + self.proj(x)


class Stem(nnx.Module):
    def __init__(self, in_chs, out_chs, eps=1e-5, *, rngs):
        self.conv = nnx.Conv(
            in_chs, out_chs, (7, 7), strides=(4, 4), padding=((3, 3), (3, 3)), rngs=rngs
        )
        self.norm = nnx.LayerNorm(out_chs, epsilon=eps, rngs=rngs)

    def __call__(self, x):
        return self.norm(self.conv(_pad_to(x, 4)))


class Downsample(nnx.Module):
    def __init__(self, in_chs, out_chs, kernel=2, eps=1e-5, *, rngs):
        self.norm = nnx.LayerNorm(in_chs, epsilon=eps, rngs=rngs)
        self.even_k = kernel % 2 == 0
        pad = 0 if self.even_k else kernel // 2
        self.kernel = kernel
        self.conv = nnx.Conv(
            in_chs,
            out_chs,
            (kernel, kernel),
            strides=(2, 2),
            padding=((pad, pad), (pad, pad)),
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.norm(x)
        if self.even_k:
            x = _pad_to(x, self.kernel)
        return self.conv(x)


class ChannelAttention(nnx.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, v2=False, *, rngs):
        self.num_heads, self.v2 = num_heads, v2
        self.qkv = nnx.Linear(dim, 3 * dim, use_bias=qkv_bias, kernel_init=_init, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, kernel_init=_init, rngs=rngs)

    def __call__(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        if self.v2:
            # Channel-by-channel query/key products, scaled by the token count; outputs mix values.
            attn = jnp.einsum("bnhi,bnhj->bhij", q * N**-0.5, k)
            out = jnp.einsum("bhij,bnhj->bnhi", jax.nn.softmax(attn, axis=-1), v)
        else:
            # timm's original channel attention: key/value products, outputs mix queries.
            attn = jnp.einsum("bnhi,bnhj->bhij", k * (C // self.num_heads) ** -0.5, v)
            out = jnp.einsum("bhij,bnhj->bnhi", jax.nn.softmax(attn, axis=-1), q)
        return self.proj(out.reshape(B, N, C))


class WindowAttention(nnx.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, *, rngs):
        self.num_heads = num_heads
        self.qkv = nnx.Linear(dim, 3 * dim, use_bias=qkv_bias, kernel_init=_init, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, kernel_init=_init, rngs=rngs)

    def __call__(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        x = dot_product_attention(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2])
        return self.proj(x.reshape(B, N, C))


class SpatialBlock(nnx.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        eps=1e-5,
        *,
        rngs,
    ):
        self.window_size = window_size
        self.cpe1 = ConvPosEnc(dim, rngs=rngs)
        self.norm1 = nnx.LayerNorm(dim, epsilon=eps, rngs=rngs)
        self.attn = WindowAttention(dim, num_heads, qkv_bias, rngs=rngs)
        self.cpe2 = ConvPosEnc(dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, epsilon=eps, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), kernel_init=_init, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        B, H, W, C = x.shape
        ws = self.window_size
        shortcut = self.cpe1(x)
        y = _pad_to(self.norm1(shortcut), ws)
        nh, nw = y.shape[1] // ws, y.shape[2] // ws
        y = y.reshape(B, nh, ws, nw, ws, C).transpose(0, 1, 3, 2, 4, 5)
        y = self.attn(y.reshape(B * nh * nw, ws * ws, C))
        y = y.reshape(B, nh, nw, ws, ws, C).transpose(0, 1, 3, 2, 4, 5)
        x = shortcut + self.drop_path(y.reshape(B, nh * ws, nw * ws, C)[:, :H, :W])
        x = self.cpe2(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class ChannelBlock(nnx.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        v2=False,
        eps=1e-5,
        *,
        rngs,
    ):
        self.cpe1 = ConvPosEnc(dim, rngs=rngs)
        self.norm1 = nnx.LayerNorm(dim, epsilon=eps, rngs=rngs)
        self.attn = ChannelAttention(dim, num_heads, qkv_bias, v2, rngs=rngs)
        self.cpe2 = ConvPosEnc(dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, epsilon=eps, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), kernel_init=_init, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        B, H, W, C = x.shape
        x = self.cpe1(x)
        x = x + self.drop_path(self.attn(self.norm1(x).reshape(B, H * W, C)).reshape(B, H, W, C))
        x = self.cpe2(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class DaVitStage(nnx.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        depth=1,
        downsample=True,
        num_heads=3,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rates=None,
        down_kernel_size=2,
        channel_attn_v2=False,
        eps=1e-5,
        *,
        rngs,
    ):
        self.downsample = (
            Downsample(in_chs, out_chs, down_kernel_size, eps, rngs=rngs) if downsample else None
        )
        drop_path_rates = drop_path_rates or [0.0] * depth
        self.blocks = nnx.List(
            [
                nnx.List(
                    [
                        SpatialBlock(
                            out_chs,
                            num_heads,
                            window_size,
                            mlp_ratio,
                            qkv_bias,
                            drop_path_rates[i],
                            eps,
                            rngs=rngs,
                        ),
                        ChannelBlock(
                            out_chs,
                            num_heads,
                            mlp_ratio,
                            qkv_bias,
                            drop_path_rates[i],
                            channel_attn_v2,
                            eps,
                            rngs=rngs,
                        ),
                    ]
                )
                for i in range(depth)
            ]
        )

    def __call__(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        for pair in self.blocks:
            for blk in pair:
                x = blk(x)
        return x


class DaVit(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"

    def __init__(
        self,
        depths=(1, 1, 3, 1),
        embed_dims=(96, 192, 384, 768),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_eps=1e-5,
        down_kernel_size=2,
        channel_attn_v2=False,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.stem = Stem(in_chans, embed_dims[0], norm_eps, rngs=rngs)
        # timm calculate_drop_path_rates(stagewise=True): linear over all blocks, split by stage.
        total = sum(depths)
        rates = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]
        stages, in_chs = [], embed_dims[0]
        for i, (dim, depth) in enumerate(zip(embed_dims, depths)):
            start = sum(depths[:i])
            stages.append(
                DaVitStage(
                    in_chs,
                    dim,
                    depth,
                    i > 0,
                    num_heads[i],
                    window_size,
                    mlp_ratio,
                    qkv_bias,
                    rates[start : start + depth],
                    down_kernel_size,
                    channel_attn_v2,
                    norm_eps,
                    rngs=rngs,
                )
            )
            in_chs = dim
        self.stages = nnx.List(stages)
        self.num_features = embed_dims[-1]
        self.head_norm = nnx.LayerNorm(self.num_features, epsilon=norm_eps, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = (
            nnx.Linear(self.num_features, num_classes, kernel_init=_init, rngs=rngs)
            if num_classes > 0
            else None
        )

    def forward_features(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x

    def forward_head(self, x):
        x = self.head_drop(self.head_norm(global_pool_nhwc(x, self.global_pool)))
        return self.head(x) if self.head is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_FL = dict(window_size=12, down_kernel_size=3, channel_attn_v2=True)
_CFGS = {
    "davit_tiny": dict(
        depths=(1, 1, 3, 1), embed_dims=(96, 192, 384, 768), num_heads=(3, 6, 12, 24)
    ),
    "davit_small": dict(
        depths=(1, 1, 9, 1), embed_dims=(96, 192, 384, 768), num_heads=(3, 6, 12, 24)
    ),
    "davit_base": dict(
        depths=(1, 1, 9, 1), embed_dims=(128, 256, 512, 1024), num_heads=(4, 8, 16, 32)
    ),
    "davit_large": dict(
        depths=(1, 1, 9, 1), embed_dims=(192, 384, 768, 1536), num_heads=(6, 12, 24, 48)
    ),
    "davit_huge": dict(
        depths=(1, 1, 9, 1), embed_dims=(256, 512, 1024, 2048), num_heads=(8, 16, 32, 64)
    ),
    "davit_giant": dict(
        depths=(1, 1, 12, 3), embed_dims=(384, 768, 1536, 3072), num_heads=(12, 24, 48, 96)
    ),
    "davit_base_fl": dict(
        depths=(1, 1, 9, 1), embed_dims=(128, 256, 512, 1024), num_heads=(4, 8, 16, 32), **_FL
    ),
    "davit_huge_fl": dict(
        depths=(1, 1, 9, 1), embed_dims=(256, 512, 1024, 2048), num_heads=(8, 16, 32, 64), **_FL
    ),
}


def _make(name):
    cfg = _CFGS[name]

    def entry(**kwargs):
        model = DaVit(**{**cfg, **kwargs})
        # Florence-2 towers default to 768x768 inputs.
        size = 768 if name.endswith("_fl") else 224
        model.default_cfg = _cfg(input_size=(3, size, size), crop_pct=0.95, interpolation="bicubic")
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
