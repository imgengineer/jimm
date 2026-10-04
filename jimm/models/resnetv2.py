"""Pre-activation ResNet V2 (including BiT) in flax nnx, NHWC. Mirrors timm.models.resnetv2.

Blocks apply norm + activation before each convolution (basic 3x3-3x3 or
1x1-3x3-1x1 bottleneck), with projection shortcuts taken from the
pre-activated input (strided 1x1 conv, or 2x2 average pool + 1x1 conv for
the "d"/"t" models). Normalization is BatchNorm, GroupNorm, EvoNorm-S0 or
Filter Response Norm + TLU; BiT models use weight-standardized convolutions,
GroupNorm, width multipliers and a zero-padded stem pool. Stems are a 7x7
conv or a three-conv deep/tiered stem; a final norm + activation precedes
pooling and the classifier.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from ..layers import BatchNorm, ClassifierMixin, DropPath, make_divisible
from ..registry import _cfg, register_model


class StdConv2d(nnx.Module):
    """Weight-standardized convolution (per output channel, eps 1e-8) with PyTorch padding."""

    def __init__(self, in_chs, out_chs, kernel, stride=1, *, rngs):
        init = nnx.initializers.lecun_normal()
        self.kernel = nnx.Param(init(rngs.params(), (kernel, kernel, in_chs, out_chs)))
        self.stride, self.pad = stride, ((stride - 1) + (kernel - 1)) // 2

    def __call__(self, x):
        w = self.kernel[...]
        mean = w.mean(axis=(0, 1, 2), keepdims=True)
        var = w.var(axis=(0, 1, 2), keepdims=True)
        w = ((w - mean) * jax.lax.rsqrt(var + 1e-8)).astype(x.dtype)
        p = self.pad
        return jax.lax.conv_general_dilated(
            x, w, (self.stride, self.stride), ((p, p), (p, p)),
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )  # fmt: skip


def _conv(in_chs, out_chs, kernel, stride=1, std=False, *, rngs):
    if std:
        return StdConv2d(in_chs, out_chs, kernel, stride, rngs=rngs)
    p = ((stride - 1) + (kernel - 1)) // 2
    return nnx.Conv(
        in_chs, out_chs, (kernel, kernel), strides=stride, padding=((p, p), (p, p)),
        use_bias=False, rngs=rngs,
    )  # fmt: skip


class EvoNorm2dS0(nnx.Module):
    """``x * sigmoid(v * x) / group_std(x)``, then an affine transform (32 groups, eps 1e-5)."""

    def __init__(self, chs, groups=32, eps=1e-5):
        self.scale = nnx.Param(jnp.ones(chs))
        self.bias = nnx.Param(jnp.zeros(chs))
        self.v = nnx.Param(jnp.ones(chs))
        self.groups, self.eps = groups, eps

    def __call__(self, x):
        B, H, W, C = x.shape
        g = x.astype(jnp.promote_types(x.dtype, jnp.float32)).reshape(B, H, W, self.groups, -1)
        std = jnp.sqrt(g.var(axis=(1, 2, 4), keepdims=True) + self.eps)
        std = jnp.broadcast_to(std, g.shape).reshape(B, H, W, C).astype(x.dtype)
        x = x * jax.nn.sigmoid(x * self.v[...]) / std
        return x * self.scale[...] + self.bias[...]


class FilterResponseNormTlu2d(nnx.Module):
    """Filter response normalization (per-channel spatial RMS, eps 1e-5) with a learned TLU."""

    def __init__(self, chs, eps=1e-5):
        self.scale = nnx.Param(jnp.ones(chs))
        self.bias = nnx.Param(jnp.zeros(chs))
        self.tau = nnx.Param(jnp.zeros(chs))
        self.eps = eps

    def __call__(self, x):
        acc = x.astype(jnp.promote_types(x.dtype, jnp.float32))
        rms = jax.lax.rsqrt(jnp.mean(acc * acc, axis=(1, 2), keepdims=True) + self.eps)
        x = x * rms.astype(x.dtype) * self.scale[...] + self.bias[...]
        return jnp.maximum(x, self.tau[...])


class NormAct(nnx.Module):
    """Norm followed by ReLU; EvoNorm and FRN carry their own nonlinearity."""

    def __init__(self, chs, kind="bn", *, rngs):
        self.kind = kind
        if kind == "bn":
            self.norm = BatchNorm(chs, epsilon=1e-5, rngs=rngs)
        elif kind == "gn":
            self.norm = nnx.GroupNorm(chs, num_groups=32, epsilon=1e-5, rngs=rngs)
        elif kind == "evos":
            self.norm = EvoNorm2dS0(chs)
        else:
            self.norm = FilterResponseNormTlu2d(chs)

    def __call__(self, x):
        x = self.norm(x)
        return nnx.relu(x) if self.kind in ("bn", "gn") else x


def _avg_pool_ceil(x, s):
    """``AvgPool2d(2, s, ceil_mode=True, count_include_pad=False)``."""
    B, H, W, C = x.shape
    oh, ow = -(-(H - 2) // s) + 1, -(-(W - 2) // s) + 1
    ph, pw = max((oh - 1) * s + 2 - H, 0), max((ow - 1) * s + 2 - W, 0)
    pad = ((0, 0), (0, ph), (0, pw), (0, 0))
    window, strides = (1, 2, 2, 1), (1, s, s, 1)
    total = jax.lax.reduce_window(jnp.pad(x, pad), 0.0, jax.lax.add, window, strides, "VALID")
    ones = jnp.pad(jnp.ones((1, H, W, 1), x.dtype), pad)
    return total / jax.lax.reduce_window(ones, 0.0, jax.lax.add, window, strides, "VALID")


class Downsample(nnx.Module):
    def __init__(self, in_chs, out_chs, stride, avg_down, std, *, rngs):
        self.pool_stride = stride if avg_down and stride > 1 else 0
        self.conv = _conv(in_chs, out_chs, 1, 1 if avg_down else stride, std, rngs=rngs)

    def __call__(self, x):
        if self.pool_stride:
            x = _avg_pool_ceil(x, self.pool_stride)
        return self.conv(x)


class PreActBasic(nnx.Module):
    def __init__(self, in_chs, out_chs, stride, first, avg_down, kind, std, dpr, *, rngs):
        mid = make_divisible(out_chs * 1.0)
        need_proj = first and (stride != 1 or in_chs != out_chs)
        self.downsample = (
            Downsample(in_chs, out_chs, stride, avg_down, std, rngs=rngs) if need_proj else None
        )
        self.norm1 = NormAct(in_chs, kind, rngs=rngs)
        self.conv1 = _conv(in_chs, mid, 3, stride, std, rngs=rngs)
        self.norm2 = NormAct(mid, kind, rngs=rngs)
        self.conv2 = _conv(mid, out_chs, 3, 1, std, rngs=rngs)
        self.drop_path = DropPath(dpr, rngs=rngs)

    def __call__(self, x):
        x_preact = self.norm1(x)
        shortcut = x if self.downsample is None else self.downsample(x_preact)
        y = self.conv2(self.norm2(self.conv1(x_preact)))
        return self.drop_path(y) + shortcut


class PreActBottleneck(nnx.Module):
    def __init__(self, in_chs, out_chs, stride, first, avg_down, kind, std, dpr, *, rngs):
        mid = make_divisible(out_chs * 0.25)
        self.downsample = (
            Downsample(in_chs, out_chs, stride, avg_down, std, rngs=rngs) if first else None
        )
        self.norm1 = NormAct(in_chs, kind, rngs=rngs)
        self.conv1 = _conv(in_chs, mid, 1, 1, std, rngs=rngs)
        self.norm2 = NormAct(mid, kind, rngs=rngs)
        self.conv2 = _conv(mid, mid, 3, stride, std, rngs=rngs)
        self.norm3 = NormAct(mid, kind, rngs=rngs)
        self.conv3 = _conv(mid, out_chs, 1, 1, std, rngs=rngs)
        self.drop_path = DropPath(dpr, rngs=rngs)

    def __call__(self, x):
        x_preact = self.norm1(x)
        shortcut = x if self.downsample is None else self.downsample(x_preact)
        y = self.conv1(x_preact)
        y = self.conv2(self.norm2(y))
        y = self.conv3(self.norm3(y))
        return self.drop_path(y) + shortcut


class ResNetStage(nnx.Module):
    def __init__(self, block, in_chs, out_chs, stride, depth, avg_down, kind, std, dprs, *, rngs):
        self.blocks = nnx.List(
            [
                block(
                    in_chs if i == 0 else out_chs,
                    out_chs,
                    stride if i == 0 else 1,
                    i == 0,
                    avg_down,
                    kind,
                    std,
                    dprs[i],
                    rngs=rngs,
                )  # fmt: skip
                for i in range(depth)
            ]
        )

    def __call__(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class Stem(nnx.Module):
    def __init__(self, in_chans, out_chs, stem_type, kind, std, *, rngs):
        self.stem_type = stem_type
        if "deep" in stem_type or "tiered" in stem_type:
            c0, c1 = (
                (3 * out_chs // 8, out_chs // 2) if "tiered" in stem_type else (out_chs // 2,) * 2
            )
            self.conv1 = _conv(in_chans, c0, 3, 2, std, rngs=rngs)
            self.norm1 = NormAct(c0, kind, rngs=rngs)
            self.conv2 = _conv(c0, c1, 3, 1, std, rngs=rngs)
            self.norm2 = NormAct(c1, kind, rngs=rngs)
            self.conv3 = _conv(c1, out_chs, 3, 1, std, rngs=rngs)
            self.conv = None
        else:
            self.conv = _conv(in_chans, out_chs, 7, 2, std, rngs=rngs)

    def __call__(self, x):
        if self.conv is not None:
            x = self.conv(x)
        else:
            x = self.conv3(self.norm2(self.conv2(self.norm1(self.conv1(x)))))
        if "fixed" in self.stem_type:
            # ConstantPad2d(1, 0) + MaxPool2d(3, 2): pads with zeros, not -inf.
            x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
            return jax.lax.reduce_window(
                x, -jnp.inf, jax.lax.max, (1, 3, 3, 1), (1, 2, 2, 1), "VALID"
            )
        return nnx.max_pool(x, (3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))


class ResNetV2(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        layers,
        channels=(256, 512, 1024, 2048),
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        width_factor=1,
        stem_type="",
        avg_down=False,
        basic=False,
        norm="bn",
        std_conv=False,
        drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        stem_chs = make_divisible(64 * width_factor)
        self.stem = Stem(in_chans, stem_chs, stem_type, norm, std_conv, rngs=rngs)
        block = PreActBasic if basic else PreActBottleneck
        total = sum(layers)
        dpr = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]
        stages, prev, k = [], stem_chs, 0
        for i, (d, c) in enumerate(zip(layers, channels)):
            out = make_divisible(c * width_factor)
            stages.append(
                ResNetStage(
                    block,
                    prev,
                    out,
                    1 if i == 0 else 2,
                    d,
                    avg_down,
                    norm,
                    std_conv,
                    dpr[k : k + d],
                    rngs=rngs,
                )  # fmt: skip
            )
            prev, k = out, k + d
        self.stages = nnx.List(stages)
        self.num_features = prev
        self.norm = NormAct(prev, norm, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = self._make_fc(num_classes, rngs)

    def _make_fc(self, num_classes, rngs):
        if num_classes <= 0:
            return None
        return nnx.Linear(
            self.num_features, num_classes, kernel_init=nnx.initializers.normal(0.01), rngs=rngs
        )

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.global_pool = global_pool if global_pool is not None else self.global_pool
        self.fc = self._make_fc(num_classes, nnx.Rngs(0))

    def forward_features(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.norm(x)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_BASIC = dict(channels=(64, 128, 256, 512), basic=True)
_D = dict(stem_type="deep", avg_down=True)
_BIT = dict(stem_type="fixed", norm="gn", std_conv=True)
_L50, _L101, _L152 = (3, 4, 6, 3), (3, 4, 23, 3), (3, 8, 36, 3)
_ARCHS = {  # layers, kwargs, input size, crop_pct, interpolation
    "resnetv2_18": ((2, 2, 2, 2), _BASIC, 224, 0.9, "bicubic"),
    "resnetv2_18d": ((2, 2, 2, 2), dict(_BASIC, **_D), 224, 0.9, "bicubic"),
    "resnetv2_34": (_L50, _BASIC, 224, 0.9, "bicubic"),
    "resnetv2_34d": (_L50, dict(_BASIC, **_D), 224, 0.9, "bicubic"),
    "resnetv2_50": (_L50, {}, 224, 0.95, "bicubic"),
    "resnetv2_50d": (_L50, _D, 224, 0.875, "bicubic"),
    "resnetv2_50t": (_L50, dict(stem_type="tiered", avg_down=True), 224, 0.875, "bicubic"),
    "resnetv2_101": (_L101, {}, 224, 0.95, "bicubic"),
    "resnetv2_101d": (_L101, _D, 224, 0.875, "bicubic"),
    "resnetv2_152": (_L152, {}, 224, 0.875, "bicubic"),
    "resnetv2_152d": (_L152, _D, 224, 0.875, "bicubic"),
    "resnetv2_50d_gn": (_L50, dict(_D, norm="gn"), 224, 0.95, "bicubic"),
    "resnetv2_50d_evos": (_L50, dict(_D, norm="evos"), 224, 0.95, "bicubic"),
    "resnetv2_50d_frn": (_L50, dict(_D, norm="frn"), 224, 0.875, "bicubic"),
    "resnetv2_50x1_bit": (_L50, dict(_BIT, width_factor=1), 224, 0.875, "bicubic"),
    "resnetv2_50x3_bit": (_L50, dict(_BIT, width_factor=3), 448, 1.0, "bilinear"),
    "resnetv2_101x1_bit": (_L101, dict(_BIT, width_factor=1), 448, 1.0, "bilinear"),
    "resnetv2_101x3_bit": (_L101, dict(_BIT, width_factor=3), 448, 1.0, "bilinear"),
    "resnetv2_152x2_bit": (_L152, dict(_BIT, width_factor=2), 224, 0.875, "bicubic"),
    "resnetv2_152x4_bit": (_L152, dict(_BIT, width_factor=4), 480, 1.0, "bilinear"),
}


def _make(name):
    layers, kw, size, crop, interp = _ARCHS[name]
    extra = dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    def entry(**kwargs):
        model = ResNetV2(layers, **{**kw, **kwargs})
        model.default_cfg = _cfg(
            input_size=(3, size, size), crop_pct=crop, interpolation=interp, **extra
        )
        return model

    entry.__name__ = name
    return entry


for _name in _ARCHS:
    register_model(_make(_name))
