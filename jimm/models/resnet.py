"""ResNet family in flax nnx, NHWC. Mirrors timm.models.resnet.

Covers timm's stems (7x7, "deep" three-conv and "deep_tiered", optionally a
strided conv instead of the max pool as in ResNet-RS), shortcuts (strided
1x1 conv, or 2x2 average pool + 1x1 conv for the "d" variants), basic and
bottleneck (ResNeXt grouped, wide) blocks, squeeze-excite and ECA channel
attention, anti-aliased downsampling (2x2 average pool or 3x3 blur pool)
and BatchNorm or GroupNorm.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx  # pyright: ignore[reportMissingImports]

from ..layers import BatchNorm, ClassifierMixin, DropPath, SqueezeExcite, make_divisible
from ..registry import _cfg, register_model


def _norm(chs, norm="bn", *, rngs):
    if norm == "gn":
        return nnx.GroupNorm(chs, num_groups=32, epsilon=1e-5, rngs=rngs)
    return BatchNorm(chs, rngs=rngs)


def _avg_pool_ceil(x, s):
    """``AvgPool2d(2, s, ceil_mode=True, count_include_pad=False)`` (kernel 2)."""
    B, H, W, C = x.shape
    if s == 2 and H % 2 == 0 and W % 2 == 0:
        return x.reshape(B, H // 2, 2, W // 2, 2, C).mean(axis=(2, 4))
    oh, ow = -(-(H - 2) // s) + 1, -(-(W - 2) // s) + 1
    ph, pw = max((oh - 1) * s + 2 - H, 0), max((ow - 1) * s + 2 - W, 0)
    pad = ((0, 0), (0, ph), (0, pw), (0, 0))
    window, strides = (1, 2, 2, 1), (1, s, s, 1)
    total = jax.lax.reduce_window(jnp.pad(x, pad), 0.0, jax.lax.add, window, strides, "VALID")
    ones = jnp.pad(jnp.ones((1, H, W, 1), x.dtype), pad)
    return total / jax.lax.reduce_window(ones, 0.0, jax.lax.add, window, strides, "VALID")


class BlurPool(nnx.Module):
    """timm BlurPool2d: reflect-padded depthwise [1, 2, 1] binomial blur with stride."""

    def __init__(self, stride=2):
        self.stride = stride

    def __call__(self, x):
        C = x.shape[-1]
        k = np.array([1.0, 2.0, 1.0]) / 4.0
        filt = jnp.asarray(np.outer(k, k), x.dtype)[:, :, None, None] * jnp.ones(
            (1, 1, 1, C), x.dtype
        )
        x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="reflect")
        return jax.lax.conv_general_dilated(
            x, filt, (self.stride, self.stride), "VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"), feature_group_count=C,
        )  # fmt: skip


def _avg_pool_2x2(x):
    """``AvgPool2d(2)``: 2x2 windows with stride 2, dropping an odd last row/column."""
    window = (1, 2, 2, 1)
    return jax.lax.reduce_window(x, 0.0, jax.lax.add, window, window, "VALID") / 4.0


class AvgPoolAA(nnx.Module):
    def __call__(self, x):
        return _avg_pool_2x2(x)


def _aa(aa, stride):
    return AvgPoolAA() if aa == "avg" else BlurPool(stride)


class EcaModule(nnx.Module):
    """Efficient channel attention: a 1D conv across pooled channels (timm EcaModule)."""

    def __init__(self, chs, gamma=2, beta=1, *, rngs):
        t = int(abs(math.log(chs, 2) + beta) / gamma)
        k = max(t if t % 2 else t + 1, 3)
        self.conv = nnx.Conv(1, 1, (k,), padding=(((k - 1) // 2,) * 2,), use_bias=False, rngs=rngs)

    def __call__(self, x):
        s = x.mean(axis=(1, 2))[:, :, None]  # [B, C, 1]
        s = self.conv(s)[:, :, 0]
        return x * jax.nn.sigmoid(s)[:, None, None, :]


def _attn(attn, chs, se_ratio, rngs):
    if attn == "se":
        rd = make_divisible(chs * se_ratio, 8, round_limit=0.0)
        return SqueezeExcite(chs, rd_channels=rd, rngs=rngs)
    if attn == "eca":
        return EcaModule(chs, rngs=rngs)
    return None


class Downsample(nnx.Module):
    def __init__(self, in_chs, out_chs, stride, kernel=1, avg_down=False, norm="bn", *, rngs):
        # timm downsample_conv: stride-1 projections stay 1x1.
        self.avg_stride = stride if avg_down and stride > 1 else 0
        if avg_down:
            kernel, conv_stride = 1, 1
        else:
            kernel, conv_stride = (1 if stride == 1 else kernel), stride
        pad = (conv_stride - 1 + kernel - 1) // 2
        self.conv = nnx.Conv(
            in_chs,
            out_chs,
            kernel_size=(kernel, kernel),
            strides=(conv_stride, conv_stride),
            padding=((pad, pad), (pad, pad)),
            use_bias=False,
            rngs=rngs,
        )
        self.bn = _norm(out_chs, norm, rngs=rngs)

    def __call__(self, x):
        if self.avg_stride:
            x = _avg_pool_ceil(x, self.avg_stride)
        return self.bn(self.conv(x))


def _shortcut(in_chs, out_chs, stride, down_kernel, avg_down, norm, rngs):
    if stride != 1 or in_chs != out_chs:
        return Downsample(in_chs, out_chs, stride, down_kernel, avg_down, norm, rngs=rngs)
    return None


class BasicBlock(nnx.Module):
    expansion = 1

    def __init__(
        self,
        in_chs,
        chs,
        stride=1,
        drop_path_rate=0.0,
        se=False,
        groups=1,
        base_width=64,
        reduce_first=1,
        down_kernel=1,
        attn=None,
        se_ratio=1 / 16,
        aa=None,
        avg_down=False,
        norm="bn",
        *,
        rngs,
    ):  # groups/base_width unused, kept for uniform block signature
        out_chs = chs * self.expansion
        first = chs // reduce_first
        use_aa = aa is not None and stride == 2
        s = 1 if use_aa else stride
        self.conv1 = nnx.Conv(
            in_chs, first, (3, 3), strides=(s, s), padding=1, use_bias=False, rngs=rngs
        )
        self.bn1 = _norm(first, norm, rngs=rngs)
        self.aa = _aa(aa, stride) if use_aa else None
        self.conv2 = nnx.Conv(first, out_chs, (3, 3), padding=1, use_bias=False, rngs=rngs)
        self.bn2 = _norm(out_chs, norm, rngs=rngs)
        self.se = _attn("se" if se else attn, out_chs, se_ratio, rngs)
        self.shortcut = _shortcut(in_chs, out_chs, stride, down_kernel, avg_down, norm, rngs)
        self.drop_path = DropPath(drop_path_rate, rngs=rngs)

    def __call__(self, x):
        y = nnx.relu(self.bn1(self.conv1(x)))
        if self.aa is not None:
            y = self.aa(y)
        y = self.bn2(self.conv2(y))
        if self.se is not None:
            y = self.se(y)
        sc = x if self.shortcut is None else self.shortcut(x)
        return nnx.relu(self.drop_path(y) + sc)


class Bottleneck(nnx.Module):
    expansion = 4

    def __init__(
        self,
        in_chs,
        chs,
        stride=1,
        drop_path_rate=0.0,
        se=False,
        groups=1,
        base_width=64,
        reduce_first=1,
        down_kernel=1,
        attn=None,
        se_ratio=1 / 16,
        aa=None,
        avg_down=False,
        norm="bn",
        *,
        rngs,
    ):
        out_chs = chs * self.expansion
        mid = int(math.floor(chs * (base_width / 64)) * groups)
        first = mid // reduce_first
        use_aa = aa is not None and stride == 2
        s = 1 if use_aa else stride
        self.conv1 = nnx.Conv(in_chs, first, (1, 1), use_bias=False, rngs=rngs)
        self.bn1 = _norm(first, norm, rngs=rngs)
        self.conv2 = nnx.Conv(
            first,
            mid,
            (3, 3),
            strides=(s, s),
            padding=1,
            use_bias=False,
            feature_group_count=groups,
            rngs=rngs,
        )
        self.bn2 = _norm(mid, norm, rngs=rngs)
        self.aa = _aa(aa, stride) if use_aa else None
        self.conv3 = nnx.Conv(mid, out_chs, (1, 1), use_bias=False, rngs=rngs)
        self.bn3 = _norm(out_chs, norm, rngs=rngs)
        self.se = _attn("se" if se else attn, out_chs, se_ratio, rngs)
        self.shortcut = _shortcut(in_chs, out_chs, stride, down_kernel, avg_down, norm, rngs)
        self.drop_path = DropPath(drop_path_rate, rngs=rngs)

    def __call__(self, x):
        y = nnx.relu(self.bn1(self.conv1(x)))
        y = nnx.relu(self.bn2(self.conv2(y)))
        if self.aa is not None:
            y = self.aa(y)
        y = self.bn3(self.conv3(y))
        if self.se is not None:
            y = self.se(y)
        sc = x if self.shortcut is None else self.shortcut(x)
        return nnx.relu(self.drop_path(y) + sc)


class ResNet(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.0,
        se=False,
        groups=1,
        base_width=64,
        deep_stem=False,
        stem_width=64,
        reduce_first=1,
        down_kernel=1,
        stem_type=None,
        avg_down=False,
        attn=None,
        se_ratio=1 / 16,
        aa=None,
        replace_stem_pool=False,
        norm="bn",
        channels=(64, 128, 256, 512),
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        blocks_fn = list(block) if isinstance(block, (list, tuple)) else [block] * len(channels)
        stem_type = stem_type if stem_type is not None else ("deep" if deep_stem else "")
        deep = "deep" in stem_type
        chs = 2 * stem_width if deep else 64
        if deep:
            # timm "deep" stem: three 3x3 convolutions, the first strided.
            s0, s1 = (
                (3 * (stem_width // 4), stem_width) if "tiered" in stem_type else (stem_width,) * 2
            )
            self.conv1 = nnx.Sequential(
                nnx.Conv(
                    in_chans, s0, (3, 3), strides=(2, 2), padding=1, use_bias=False, rngs=rngs
                ),
                _norm(s0, norm, rngs=rngs),
                nnx.relu,
                nnx.Conv(s0, s1, (3, 3), padding=1, use_bias=False, rngs=rngs),
                _norm(s1, norm, rngs=rngs),
                nnx.relu,
                nnx.Conv(s1, chs, (3, 3), padding=1, use_bias=False, rngs=rngs),
            )
        else:
            self.conv1 = nnx.Conv(
                in_chans,
                chs,
                (7, 7),
                strides=(2, 2),
                padding=[(3, 3), (3, 3)],
                use_bias=False,
                rngs=rngs,
            )
        self.bn1 = _norm(chs, norm, rngs=rngs)
        self.aa = aa
        if replace_stem_pool:
            self.maxpool = nnx.Sequential(
                nnx.Conv(chs, chs, (3, 3), strides=(2, 2), padding=1, use_bias=False, rngs=rngs),
                _norm(chs, norm, rngs=rngs),
                nnx.relu,
            )
        else:
            self.maxpool = None
        total = sum(layers)
        dpr = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]
        stages, k = [], 0
        for i, (n, width, blk) in enumerate(zip(layers, channels, blocks_fn)):
            stride = 1 if i == 0 else 2
            blocks = []
            for j in range(n):
                blocks.append(
                    blk(
                        chs,
                        width,
                        stride if j == 0 else 1,
                        dpr[k],
                        se=se,
                        groups=groups,
                        base_width=base_width,
                        reduce_first=reduce_first,
                        down_kernel=down_kernel,
                        attn=attn,
                        se_ratio=se_ratio,
                        aa=aa,
                        avg_down=avg_down,
                        norm=norm,
                        rngs=rngs,
                    )
                )
                chs = width * blk.expansion
                k += 1
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.num_features = chs
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def _stem_pool(self, x):
        if self.maxpool is not None:
            return self.maxpool(x)
        if self.aa == "avg":
            return _avg_pool_2x2(x)
        if self.aa == "blur":
            x = nnx.max_pool(x, (3, 3), strides=(1, 1), padding=((1, 1), (1, 1)))
            return BlurPool(2)(x)
        # PyTorch padding: Flax's SAME pads strided windows on one side only.
        return nnx.max_pool(x, (3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))

    def forward_features(self, x):
        x = nnx.relu(self.bn1(self.conv1(x)))
        x = self._stem_pool(x)
        for stage in self.stages:
            for blk in stage:
                x = blk(x)
        return x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_B, _BN = BasicBlock, Bottleneck
_D = dict(stem_width=32, stem_type="deep", avg_down=True)
_T = dict(stem_width=32, stem_type="deep_tiered", avg_down=True)
_C = dict(stem_width=32, stem_type="deep")
_S = dict(stem_width=64, stem_type="deep")
_X32 = dict(groups=32, base_width=4)
_RS = dict(_D, replace_stem_pool=True, attn="se", se_ratio=0.25)
_L18, _L34, _L26 = (2, 2, 2, 2), (3, 4, 6, 3), (2, 2, 2, 2)
_L50, _L101, _L152, _L200 = (3, 4, 6, 3), (3, 4, 23, 3), (3, 8, 36, 3), (3, 24, 36, 3)
_ARCHS = {
    "resnet10t": (_B, (1, 1, 1, 1), _T),
    "resnet14t": (_BN, (1, 1, 1, 1), _T),
    "resnet18": (_B, _L18, {}),
    "resnet18d": (_B, _L18, _D),
    "resnet34": (_B, _L34, {}),
    "resnet34d": (_B, _L34, _D),
    "resnet26": (_BN, _L26, {}),
    "resnet26t": (_BN, _L26, _T),
    "resnet26d": (_BN, _L26, _D),
    "resnet50": (_BN, _L50, {}),
    "resnet50c": (_BN, _L50, _C),
    "resnet50d": (_BN, _L50, _D),
    "resnet50s": (_BN, _L50, _S),
    "resnet50t": (_BN, _L50, _T),
    "resnet101": (_BN, _L101, {}),
    "resnet101c": (_BN, _L101, _C),
    "resnet101d": (_BN, _L101, _D),
    "resnet101s": (_BN, _L101, _S),
    "resnet152": (_BN, _L152, {}),
    "resnet152c": (_BN, _L152, _C),
    "resnet152d": (_BN, _L152, _D),
    "resnet152s": (_BN, _L152, _S),
    "resnet200": (_BN, _L200, {}),
    "resnet200d": (_BN, _L200, _D),
    "wide_resnet50_2": (_BN, _L50, dict(base_width=128)),
    "wide_resnet101_2": (_BN, _L101, dict(base_width=128)),
    "resnet50_gn": (_BN, _L50, dict(norm="gn")),
    "resnext50_32x4d": (_BN, _L50, _X32),
    "resnext50d_32x4d": (_BN, _L50, dict(_X32, **_D)),
    "resnext101_32x4d": (_BN, _L101, _X32),
    "resnext101_32x8d": (_BN, _L101, dict(groups=32, base_width=8)),
    "resnext101_32x16d": (_BN, _L101, dict(groups=32, base_width=16)),
    "resnext101_32x32d": (_BN, _L101, dict(groups=32, base_width=32)),
    "resnext101_64x4d": (_BN, _L101, dict(groups=64, base_width=4)),
    "ecaresnet26t": (_BN, _L26, dict(_T, attn="eca")),
    "ecaresnet50d": (_BN, _L50, dict(_D, attn="eca")),
    "ecaresnet50t": (_BN, _L50, dict(_T, attn="eca")),
    "ecaresnetlight": (_BN, (1, 1, 11, 3), dict(stem_width=32, avg_down=True, attn="eca")),
    "ecaresnet101d": (_BN, _L101, dict(_D, attn="eca")),
    "ecaresnet200d": (_BN, _L200, dict(_D, attn="eca")),
    "ecaresnet269d": (_BN, (3, 30, 48, 8), dict(_D, attn="eca")),
    "ecaresnext26t_32x4d": (_BN, _L26, dict(_X32, attn="eca", **_T)),
    "ecaresnext50t_32x4d": (_BN, _L26, dict(_X32, attn="eca", **_T)),
    "seresnet18": (_B, _L18, dict(attn="se")),
    "seresnet34": (_B, _L34, dict(attn="se")),
    "seresnet50": (_BN, _L50, dict(attn="se")),
    "seresnet50t": (_BN, _L50, dict(_T, attn="se")),
    "seresnet101": (_BN, _L101, dict(attn="se")),
    "seresnet152": (_BN, _L152, dict(attn="se")),
    "seresnet152d": (_BN, _L152, dict(_D, attn="se")),
    "seresnet200d": (_BN, _L200, dict(_D, attn="se")),
    "seresnet269d": (_BN, (3, 30, 48, 8), dict(_D, attn="se")),
    "seresnext26d_32x4d": (_BN, _L26, dict(_X32, attn="se", **_D)),
    "seresnext26t_32x4d": (_BN, _L26, dict(_X32, attn="se", **_T)),
    "seresnext50_32x4d": (_BN, _L50, dict(_X32, attn="se")),
    "seresnext101_32x4d": (_BN, _L101, dict(_X32, attn="se")),
    "seresnext101_32x8d": (_BN, _L101, dict(groups=32, base_width=8, attn="se")),
    "seresnext101d_32x8d": (_BN, _L101, dict(groups=32, base_width=8, attn="se", **_D)),
    "seresnext101_64x4d": (_BN, _L101, dict(groups=64, base_width=4, attn="se")),
    "resnetblur18": (_B, _L18, dict(aa="blur")),
    "resnetblur50": (_BN, _L50, dict(aa="blur")),
    "resnetblur50d": (_BN, _L50, dict(_D, aa="blur")),
    "resnetblur101d": (_BN, _L101, dict(_D, aa="blur")),
    "resnetaa34d": (_B, _L34, dict(_D, aa="avg")),
    "resnetaa50": (_BN, _L50, dict(aa="avg")),
    "resnetaa50d": (_BN, _L50, dict(_D, aa="avg")),
    "resnetaa101d": (_BN, _L101, dict(_D, aa="avg")),
    "seresnetaa50d": (_BN, _L50, dict(_D, aa="avg", attn="se")),
    "seresnextaa101d_32x8d": (
        _BN,
        _L101,
        dict(_D, groups=32, base_width=8, aa="avg", attn="se"),
    ),  # fmt: skip
    "seresnextaa201d_32x8d": (
        _BN,
        (3, 24, 36, 4),
        dict(_D, stem_width=64, groups=32, base_width=8, aa="avg", attn="se"),
    ),  # fmt: skip
    "resnetrs50": (_BN, _L50, _RS),
    "resnetrs101": (_BN, _L101, _RS),
    "resnetrs152": (_BN, _L152, _RS),
    "resnetrs200": (_BN, _L200, _RS),
    "resnetrs270": (_BN, (4, 29, 53, 4), _RS),
    "resnetrs350": (_BN, (4, 36, 72, 4), _RS),
    "resnetrs420": (_BN, (4, 44, 87, 4), _RS),
    "test_resnet": (
        (_B, _B, _BN, _B),
        (1, 1, 1, 1),
        dict(stem_width=16, stem_type="deep", avg_down=True, channels=(32, 48, 48, 96)),
    ),  # fmt: skip
}


def _make(name, cfg):
    block, layers, kwargs_fixed = _ARCHS[name]

    def entry(**kwargs):
        model = ResNet(block, layers, **{**kwargs_fixed, **kwargs})
        model.default_cfg = _cfg(**cfg)
        return model

    entry.__name__ = name
    return entry


def _register(cfgs):
    for name, cfg in cfgs.items():
        register_model(_make(name, cfg))


from ._resnet_cfgs import RESNET_CFGS  # noqa: E402  (timm default configs per variant)

_register(RESNET_CFGS)
