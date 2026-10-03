"""CSATv2 in flax nnx, NHWC. Mirrors timm.models.csatv2.

The stem works in the JPEG frequency domain: the (ImageNet-normalized) input
is mapped back to 0-255 RGB, converted to YCbCr, split into 8x8 blocks,
transformed by an orthonormal 2D DCT-II, reordered in zigzag order,
normalized with fixed per-coefficient statistics and projected per channel by
1x1 convs (Y gets 3/4 of the width). Four stages of ConvNeXt-V2-style blocks
(depthwise 7x7, LayerNorm, GRN MLP) follow, each block's output scaled by a
spatial attention map: channel mean and max pooled to 7x7, a 7x7 conv, a
one-channel transformer whose token attention, as in timm, mixes across the
batch, and align-corners bilinear upsampling. The last two stages end with
transformer blocks (depthwise position conv, 8x32-dim attention, MLP).
"""

import math
from functools import reduce

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from ..attention import dot_product_attention
from ..layers import ClassifierMixin, DropPath, gelu
from ..registry import _cfg, register_model

_trunc = nnx.initializers.truncated_normal(0.02)
_INIT = dict(kernel_init=_trunc, bias_init=nnx.initializers.zeros)


def _zigzag(rows, cols):
    dia = [[] for _ in range(rows + cols - 1)]
    for i in range(rows):
        for j in range(cols):
            s = i + j
            if s % 2 == 0:
                dia[s].insert(0, i * cols + j)
            else:
                dia[s].append(i * cols + j)
    return [i for d in dia for i in d]


def _dct_matrix(k):
    """Orthonormal DCT-II matrix ``D[u, n]``, as timm builds it via FFT."""
    n = np.arange(k)
    u = n[:, None]
    scale = np.where(u == 0, math.sqrt(1 / k), math.sqrt(2 / k))
    return (np.cos(np.pi * (2 * n + 1) * u / (2 * k)) * scale).astype(np.float32)


def _split_out_chs(out_chs, ratio=(24, 4, 4)):
    g = reduce(math.gcd, ratio)
    r = [x // g for x in ratio]
    unit = out_chs // sum(r)
    return tuple(x * unit for x in r)


def _adaptive_avg_matrix(size, out):
    m = np.zeros((out, size), np.float32)
    for i in range(out):
        start, end = (i * size) // out, -(-((i + 1) * size) // out)
        m[i, start:end] = 1.0 / (end - start)
    return m


def _align_corners_matrix(src, dst):
    m = np.zeros((dst, src), np.float32)
    for i in range(dst):
        pos = i * (src - 1) / (dst - 1) if dst > 1 else 0.0
        lo = min(int(math.floor(pos)), src - 1)
        hi = min(lo + 1, src - 1)
        frac = pos - lo
        m[i, lo] += 1 - frac
        m[i, hi] += frac
    return m


class LearnableDct2d(nnx.Module):
    def __init__(self, kernel_size=8, out_chs=32, *, rngs):
        self.k = kernel_size
        y_ch, cb_ch, cr_ch = _split_out_chs(out_chs)
        kk = kernel_size**2

        def conv(c):
            return nnx.Conv(kk, c, (1, 1), **_INIT, rngs=rngs)

        self.conv_y, self.conv_cb, self.conv_cr = conv(y_ch), conv(cb_ch), conv(cr_ch)
        self.dct = nnx.Variable(jnp.asarray(_dct_matrix(kernel_size)))
        self.permutation = nnx.Variable(jnp.asarray(_zigzag(kernel_size, kernel_size)))
        self.mean = nnx.Variable(
            jnp.asarray(np.array(_DCT_MEAN.split(), np.float32).reshape(3, kk))
        )
        self.var = nnx.Variable(jnp.asarray(np.array(_DCT_VAR.split(), np.float32).reshape(3, kk)))

    def __call__(self, x):
        B, H, W, _ = x.shape
        k = self.k
        mean = jnp.asarray((0.485, 0.456, 0.406), x.dtype)
        std = jnp.asarray((0.229, 0.224, 0.225), x.dtype)
        x = (x * std + mean) * 255
        r, g, b = x[..., 0], x[..., 1], x[..., 2]
        y = r * 0.299 + g * 0.587 + b * 0.114
        x = jnp.stack([y, 0.564 * (b - y) + 128, 0.713 * (r - y) + 128], axis=1)  # [B, 3, H, W]
        x = x.reshape(B, 3, H // k, k, W // k, k).transpose(0, 2, 4, 1, 3, 5)
        d = self.dct[...].astype(x.dtype)
        x = jnp.einsum("...ij,ui,vj->...uv", x, d, d)  # D X D^T per block
        x = x.reshape(B, H // k, W // k, 3, k * k)[..., self.permutation[...]]
        x = (x - self.mean[...]) / (self.var[...] ** 0.5 + 1e-8)
        return jnp.concatenate(
            [self.conv_y(x[..., 0, :]), self.conv_cb(x[..., 1, :]), self.conv_cr(x[..., 2, :])],
            axis=-1,
        )


class GlobalResponseNorm(nnx.Module):
    def __init__(self, dim):
        self.gamma = nnx.Param(jnp.zeros(dim))
        self.beta = nnx.Param(jnp.zeros(dim))

    def __call__(self, x):
        gx = jnp.sqrt(jnp.sum(x**2, axis=(1, 2), keepdims=True))
        nx = gx / (jnp.mean(gx, axis=-1, keepdims=True) + 1e-6)
        return x + self.beta[...] + self.gamma[...] * (x * nx)


class PosConv(nnx.Module):
    def __init__(self, chs, *, rngs):
        self.proj = nnx.Conv(
            chs, chs, (3, 3), padding=((1, 1), (1, 1)), feature_group_count=chs, **_INIT, rngs=rngs
        )

    def __call__(self, x):
        return self.proj(x) + x


class Mlp(nnx.Module):
    def __init__(self, dim, hidden, out, *, rngs):
        self.fc1 = nnx.Linear(dim, hidden, **_INIT, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, out, **_INIT, rngs=rngs)

    def __call__(self, x):
        return self.fc2(gelu(self.fc1(x)))


class SpatialTransformerBlock(nnx.Module):
    """One-channel transformer on the 7x7 map; timm's token attention runs across the batch."""

    def __init__(self, *, rngs):
        self.pos_embed = PosConv(1, rngs=rngs)
        self.norm1 = nnx.LayerNorm(1, epsilon=1e-5, rngs=rngs)
        self.qkv = nnx.Linear(1, 3, use_bias=False, **_INIT, rngs=rngs)
        self.norm2 = nnx.LayerNorm(1, epsilon=1e-5, rngs=rngs)
        self.mlp = Mlp(1, 4, 1, rngs=rngs)

    def __call__(self, x):
        B, H, W, _ = x.shape
        t = self.pos_embed(self.norm1(x))
        qkv = self.qkv(t).reshape(B, H * W, 3)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]  # [B, HW]
        attn = jax.nn.softmax(q @ k.T, axis=-1)  # [B, B]
        x = x + (attn @ v).reshape(B, H, W, 1)
        return x + self.mlp(self.norm2(x))


class SpatialAttention(nnx.Module):
    def __init__(self, *, rngs):
        self.conv = nnx.Conv(2, 1, (7, 7), padding=((3, 3), (3, 3)), **_INIT, rngs=rngs)
        self.attn = SpatialTransformerBlock(rngs=rngs)

    def __call__(self, x):
        x = jnp.concatenate([x.mean(axis=-1, keepdims=True), x.max(axis=-1, keepdims=True)], -1)
        ph = jnp.asarray(_adaptive_avg_matrix(x.shape[1], 7), x.dtype)
        pw = jnp.asarray(_adaptive_avg_matrix(x.shape[2], 7), x.dtype)
        x = jnp.einsum("ih,jw,bhwc->bijc", ph, pw, x)
        return self.attn(self.conv(x))


class Block(nnx.Module):
    def __init__(self, dim, drop_path=0.0, *, rngs):
        self.dwconv = nnx.Conv(
            dim, dim, (7, 7), padding=((3, 3), (3, 3)), feature_group_count=dim, **_INIT, rngs=rngs
        )
        self.norm = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.pwconv1 = nnx.Linear(dim, 4 * dim, **_INIT, rngs=rngs)
        self.grn = GlobalResponseNorm(4 * dim)
        self.pwconv2 = nnx.Linear(4 * dim, dim, **_INIT, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)
        self.attn = SpatialAttention(rngs=rngs)

    def __call__(self, x):
        shortcut = x
        x = self.pwconv2(self.grn(gelu(self.pwconv1(self.norm(self.dwconv(x))))))
        a = self.attn(x)
        uh = jnp.asarray(_align_corners_matrix(a.shape[1], x.shape[1]), x.dtype)
        uw = jnp.asarray(_align_corners_matrix(a.shape[2], x.shape[2]), x.dtype)
        x = x * jnp.einsum("hi,wj,bijc->bhwc", uh, uw, a)
        return shortcut + self.drop_path(x)


class Attention(nnx.Module):
    def __init__(self, dim, num_heads=8, head_dim=32, *, rngs):
        self.num_heads, self.head_dim = num_heads, head_dim
        inner = num_heads * head_dim
        self.qkv = nnx.Linear(dim, inner * 3, use_bias=False, **_INIT, rngs=rngs)
        self.proj = nnx.Linear(inner, dim, **_INIT, rngs=rngs)

    def __call__(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        out = dot_product_attention(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2])
        return self.proj(out.reshape(B, N, -1))


class TransformerBlock(nnx.Module):
    def __init__(self, dim, drop_path=0.0, *, rngs):
        self.pos_embed = PosConv(dim, rngs=rngs)
        self.norm1 = nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)
        self.attn = Attention(dim, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)
        self.mlp = Mlp(dim, 4 * dim, dim, rngs=rngs)

    def __call__(self, x):
        B, H, W, C = x.shape
        t = self.pos_embed(self.norm1(x)).reshape(B, H * W, C)
        x = x + self.drop_path(self.attn(t)).reshape(B, H, W, C)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class CSATv2(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        dims=(32, 72, 168, 386),
        depths=(2, 2, 8, 6),
        transformer_depths=(0, 0, 2, 2),
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.0,
        transformer_drop_path=False,
        *,
        rngs,
    ):
        assert in_chans == 3, "the DCT stem expects RGB input"
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = dims[-1]
        total = (
            sum(depths)
            if transformer_drop_path
            else sum(d - t for d, t in zip(depths, transformer_depths))
        )
        rates = iter(np.linspace(0, drop_path_rate, total).tolist() if total else [])
        self.stem_dct = LearnableDct2d(8, dims[0], rngs=rngs)
        stages = []
        for i, (dim, depth, t_depth) in enumerate(zip(dims, depths, transformer_depths)):
            layers = []
            if i > 0:
                layers.append(
                    nnx.Conv(
                        dims[i - 1], dim, (2, 2), strides=2, padding="VALID", **_INIT, rngs=rngs
                    )
                )
            layers += [Block(dim, next(rates), rngs=rngs) for _ in range(depth - t_depth)]
            layers += [
                TransformerBlock(dim, next(rates) if transformer_drop_path else 0.0, rngs=rngs)
                for _ in range(t_depth)
            ]
            if i < len(depths) - 1:
                layers.append(nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs))
            stages.append(nnx.List(layers))
        self.stages = nnx.List(stages)
        self.norm = nnx.LayerNorm(dims[-1], epsilon=1e-6, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = self._make_fc(num_classes, rngs)

    def _make_fc(self, num_classes, rngs):
        if num_classes <= 0:
            return None
        return nnx.Linear(self.num_features, num_classes, **_INIT, rngs=rngs)

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.global_pool = global_pool if global_pool is not None else self.global_pool
        self.fc = self._make_fc(num_classes, nnx.Rngs(0))

    def forward_features(self, x):
        x = self.stem_dct(x)
        for stage in self.stages:
            for layer in stage:
                x = layer(x)
        return x

    def forward_head(self, x):
        if self.global_pool == "avg":
            x = x.mean(axis=(1, 2))
        elif self.global_pool == "max":
            x = x.max(axis=(1, 2))
        x = self.head_drop(self.norm(x))
        return self.fc(x) if self.fc is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_CFGS = {
    "csatv2": (dict(), dict(input_size=(3, 512, 512), interpolation="bilinear")),
    "csatv2_21m": (
        dict(dims=(48, 96, 224, 448), depths=(3, 3, 10, 8), transformer_depths=(0, 0, 4, 3)),
        dict(input_size=(3, 640, 640), interpolation="bicubic"),
    ),
}


def _make(name):
    arch, cfg = _CFGS[name]

    def entry(**kwargs):
        model = CSATv2(**{**arch, **kwargs})
        model.default_cfg = _cfg(crop_pct=1.0, **cfg)
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))

# Per-coefficient DCT statistics (Y, Cb, Cr rows of 64 zigzag-ordered values) from timm.

_DCT_MEAN = (
    "932.42657 -0.0026 0.33415 -0.0284 3e-05 -0.02792 -0.00183 6e-05 0.00032 0.03402 -0.00571 "
    "0.0002 6e-05 -0.00038 -0.00558 -0.00116 -0.0 -0.00047 -8e-05 -0.0003 0.00942 0.00161 "
    "-9e-05 -6e-05 -0.00014 -0.00035 1e-05 -0.0022 0.00033 -2e-05 -3e-05 -0.0002 7e-05 -0.0 "
    "5e-05 0.00293 -4e-05 6e-05 0.00019 4e-05 6e-05 -0.00015 -2e-05 7e-05 0.0001 -4e-05 8e-05 "
    "0.0 8e-05 -1e-05 0.00015 2e-05 7e-05 3e-05 4e-05 -1e-05 4e-05 -0.0 2e-05 -0.0 -8e-05 -0.0 "
    "-3e-05 3e-05 962.34735 -0.00428 0.09835 0.00152 -9e-05 0.00312 -0.00141 -1e-05 -0.00013 "
    "0.0105 0.00065 6e-05 -0.0 3e-05 0.00264 0.0 1e-05 7e-05 -6e-05 3e-05 0.00341 0.00163 4e-05 "
    "3e-05 -1e-05 8e-05 -0.0 0.0009 0.00018 -6e-05 -1e-05 7e-05 -3e-05 -1e-05 6e-05 0.00084 "
    "-0.0 -1e-05 0.0 4e-05 -1e-05 -2e-05 0.0 1e-05 2e-05 1e-05 4e-05 0.00011 0.0 -3e-05 0.00011 "
    "-2e-05 1e-05 1e-05 1e-05 1e-05 -7e-05 -3e-05 1e-05 0.0 1e-05 2e-05 1e-05 0.0 1053.16101 "
    "-0.00213 -0.09207 0.00186 0.00013 0.00034 -0.00119 2e-05 0.00011 -0.00984 0.00046 -7e-05 "
    "-1e-05 -5e-05 0.0018 0.00042 2e-05 -0.0001 4e-05 3e-05 -0.00301 0.00125 -2e-05 -3e-05 "
    "-1e-05 -1e-05 -1e-05 0.00056 0.00021 1e-05 -1e-05 2e-05 -1e-05 -1e-05 5e-05 -0.0007 -2e-05 "
    "-2e-05 5e-05 -4e-05 -0.0 2e-05 -2e-05 1e-05 0.0 -3e-05 4e-05 7e-05 1e-05 0.0 0.00013 -0.0 "
    "0.0 2e-05 -0.0 -1e-05 -4e-05 -3e-05 0.0 1e-05 -1e-05 1e-05 -0.0 0.0 "
)

_DCT_VAR = (
    "270372.375 6287.10645 5974.94043 1653.10889 1463.91748 1832.58997 755.92468 692.41528 "
    "648.57184 641.46881 285.79288 301.621 380.43405 349.84027 374.15891 190.3096 190.76746 "
    "221.64578 200.82646 145.87979 126.92046 62.14622 67.75562 102.42001 129.74922 130.04631 "
    "103.12189 97.76417 53.17402 54.81048 73.48712 81.04342 69.351 49.06024 33.96053 37.03279 "
    "20.48858 24.9483 33.90822 44.54912 47.56363 40.0316 30.43313 22.63899 26.53739 26.57114 "
    "21.84404 17.41557 15.18253 10.69678 11.24111 12.97229 15.08971 15.31646 8.90409 7.44213 "
    "6.66096 6.97719 4.17834 3.83882 4.51073 2.36646 2.41363 1.48266 18839.21094 321.70932 "
    "300.15259 77.4783 76.02293 89.04748 33.99642 34.74807 32.12333 28.19588 12.04675 14.26871 "
    "18.45779 16.59588 15.67892 7.37718 8.56312 10.28946 9.41013 6.6909 5.16453 2.55186 3.03073 "
    "4.66765 5.85418 5.74644 4.33702 3.66948 1.95107 2.26034 3.0638 3.50705 3.06359 2.19284 "
    "1.54454 1.5786 0.97078 1.13941 1.48653 1.89996 1.95544 1.6495 1.24754 0.93677 1.09267 "
    "1.09516 0.94163 0.78966 0.72489 0.50841 0.50909 0.55664 0.63111 0.64125 0.38847 0.33378 "
    "0.30918 0.33463 0.20875 0.19298 0.21903 0.1338 0.13444 0.09554 17127.39844 292.81421 "
    "271.45209 66.64056 63.60253 76.35437 28.06587 27.84831 25.96656 23.6037 9.99173 11.34992 "
    "14.46955 12.92553 12.69353 5.91537 6.60187 7.90891 7.32825 5.32785 4.2966 2.13459 2.44135 "
    "3.66021 4.50335 4.38959 3.34888 2.97181 1.60633 1.7701 2.35118 2.69018 2.38189 1.74596 "
    "1.26014 1.31684 0.79327 0.92046 1.1767 1.47609 1.50914 1.28725 0.99898 0.74832 0.85736 "
    "0.858 0.74663 0.63508 0.58748 0.41098 0.41121 0.44663 0.50277 0.51519 0.31729 0.27336 "
    "0.25399 0.27241 0.17353 0.16255 0.1844 0.11602 0.11511 0.0845 "
)
