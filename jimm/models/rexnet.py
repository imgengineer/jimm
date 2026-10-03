"""ReXNet in flax nnx, NHWC. Mirrors timm.models.rexnet.

MobileNetV2-style linear bottlenecks whose widths grow linearly block by block,
squeeze-excite with BatchNorm, ReLU6 after the depthwise convolution, and a
residual added to the first ``in_chs`` output channels only.
"""

import math

import jax.numpy as jnp
from flax import nnx

from ..layers import BatchNorm, ClassifierMixin, DropPath, make_divisible
from ..registry import _cfg, register_model
from ._conv import ConvNormAct


class SEWithNorm(nnx.Module):
    def __init__(self, chs, rd_chs, *, rngs):
        self.fc1 = nnx.Linear(chs, rd_chs, rngs=rngs)
        self.bn = BatchNorm(rd_chs, epsilon=1e-5, rngs=rngs)
        self.fc2 = nnx.Linear(rd_chs, chs, rngs=rngs)

    def __call__(self, x):
        s = jnp.mean(x, axis=(1, 2), keepdims=True)
        return x * nnx.sigmoid(self.fc2(nnx.relu(self.bn(self.fc1(s)))))


class LinearBottleneck(nnx.Module):
    def __init__(
        self, in_chs, out_chs, stride, exp_ratio=1.0, se_ratio=0.0, ch_div=1, drop_path=0.0, *, rngs
    ):
        self.in_chs = in_chs
        self.use_shortcut = stride == 1 and in_chs <= out_chs
        if exp_ratio != 1.0:
            dw_chs = make_divisible(round(in_chs * exp_ratio), ch_div)
            self.conv_exp = ConvNormAct(in_chs, dw_chs, act=nnx.silu, rngs=rngs)
        else:
            dw_chs = in_chs
            self.conv_exp = None
        self.conv_dw = ConvNormAct(dw_chs, dw_chs, 3, stride, dw_chs, rngs=rngs)
        self.se = (
            SEWithNorm(dw_chs, make_divisible(int(dw_chs * se_ratio), ch_div), rngs=rngs)
            if se_ratio > 0
            else None
        )
        self.conv_pwl = ConvNormAct(dw_chs, out_chs, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        shortcut = x
        if self.conv_exp is not None:
            x = self.conv_exp(x)
        x = self.conv_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.conv_pwl(nnx.relu6(x))
        if self.use_shortcut:
            x = self.drop_path(x)
            x = jnp.concatenate([x[..., : self.in_chs] + shortcut, x[..., self.in_chs :]], axis=-1)
        return x


def _block_cfg(width_mult, depth_mult, initial_chs=16, final_chs=180, se_ratio=1 / 12, ch_div=1):
    layers = [math.ceil(n * depth_mult) for n in (1, 2, 2, 3, 3, 5)]
    strides = sum([[s] + [1] * (n - 1) for s, n in zip((1, 2, 2, 2, 1, 2), layers)], [])
    exp_ratios = [1] * layers[0] + [6] * sum(layers[1:])
    se_ratios = [0.0] * (layers[0] + layers[1]) + [se_ratio] * sum(layers[2:])
    num_blocks = sum(layers)
    base_chs = initial_chs / width_mult if width_mult < 1.0 else initial_chs
    out_chs = []
    for _ in range(num_blocks):
        out_chs.append(make_divisible(round(base_chs * width_mult), ch_div))
        base_chs += final_chs / num_blocks
    return list(zip(out_chs, exp_ratios, strides, se_ratios))


class RexNet(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        width_mult=1.0,
        depth_mult=1.0,
        ch_div=1,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.2,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        stem_base = 32 / width_mult if width_mult < 1.0 else 32
        stem_chs = make_divisible(round(stem_base * width_mult), ch_div)
        self.stem = ConvNormAct(in_chans, stem_chs, 3, 2, act=nnx.silu, rngs=rngs)
        cfg = _block_cfg(width_mult, depth_mult, ch_div=ch_div)
        blocks, prev = [], stem_chs
        for i, (chs, exp_ratio, stride, se_ratio) in enumerate(cfg):
            dpr = drop_path_rate * i / max(len(cfg) - 1, 1)
            blocks.append(
                LinearBottleneck(prev, chs, stride, exp_ratio, se_ratio, ch_div, dpr, rngs=rngs)
            )
            prev = chs
        pen_chs = make_divisible(1280 * width_mult, ch_div)
        blocks.append(ConvNormAct(prev, pen_chs, act=nnx.silu, rngs=rngs))
        self.features = nnx.List(blocks)
        self.num_features = pen_chs
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(pen_chs, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.stem(x)
        for blk in self.features:
            x = blk(x)
        return x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_CFGS = {  # width_mult, ch_div
    "rexnet_100": (1.0, 1),
    "rexnet_130": (1.3, 1),
    "rexnet_150": (1.5, 1),
    "rexnet_200": (2.0, 1),
    "rexnet_300": (3.0, 1),
    "rexnetr_100": (1.0, 8),
    "rexnetr_130": (1.3, 8),
    "rexnetr_150": (1.5, 8),
    "rexnetr_200": (2.0, 8),
    "rexnetr_300": (3.0, 16),
}


def _make(name):
    width_mult, ch_div = _CFGS[name]

    def entry(**kwargs):
        model = RexNet(width_mult, ch_div=ch_div, **kwargs)
        model.default_cfg = _cfg(interpolation="bicubic")
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
