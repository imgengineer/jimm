"""SelecSLS in flax nnx, NHWC. Mirrors timm.models.selecsls.

Each block applies a (possibly strided) 3x3 convolution followed by two 1x1 +
3x3 pairs whose 3x3 convolutions halve the width, then fuses the three
outputs with a 1x1 convolution. Later blocks of a stage also concatenate the
output of the stage's first block. A head of plain convolutions, two of them
strided, follows the blocks.
"""

import jax.numpy as jnp
from flax import nnx

from ..layers import ClassifierMixin
from ..registry import _cfg, register_model
from ._conv import ConvNormAct


def _conv_bn(in_chs, out_chs, kernel=3, stride=1, *, rngs):
    return ConvNormAct(in_chs, out_chs, kernel, stride, act=nnx.relu, rngs=rngs)


class SelecSlsBlock(nnx.Module):
    def __init__(self, in_chs, skip_chs, mid_chs, out_chs, is_first, stride, *, rngs):
        self.is_first = is_first
        self.conv1 = _conv_bn(in_chs, mid_chs, 3, stride, rngs=rngs)
        self.conv2 = _conv_bn(mid_chs, mid_chs, 1, rngs=rngs)
        self.conv3 = _conv_bn(mid_chs, mid_chs // 2, 3, rngs=rngs)
        self.conv4 = _conv_bn(mid_chs // 2, mid_chs, 1, rngs=rngs)
        self.conv5 = _conv_bn(mid_chs, mid_chs // 2, 3, rngs=rngs)
        self.conv6 = _conv_bn(2 * mid_chs + (0 if is_first else skip_chs), out_chs, 1, rngs=rngs)

    def __call__(self, x, skip=None):
        d1 = self.conv1(x)
        d2 = self.conv3(self.conv2(d1))
        d3 = self.conv5(self.conv4(d2))
        if self.is_first:
            out = self.conv6(jnp.concatenate([d1, d2, d3], axis=-1))
            return out, out
        return self.conv6(jnp.concatenate([d1, d2, d3, skip], axis=-1)), skip


class SelecSls(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        features,
        head,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.stem = _conv_bn(in_chans, 32, 3, 2, rngs=rngs)
        self.features = nnx.List([SelecSlsBlock(*args, rngs=rngs) for args in features])
        self.head = nnx.List([_conv_bn(*args, rngs=rngs) for args in head])
        self.num_features = head[-1][1]
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x, skip = self.stem(x), None
        for blk in self.features:
            x, skip = blk(x, skip)
        for layer in self.head:
            x = layer(x)
        return x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


# in_chs, skip_chs, mid_chs, out_chs, is_first, stride
_FEATURES = {
    "selecsls42": [
        (32, 0, 64, 64, True, 2),
        (64, 64, 64, 128, False, 1),
        (128, 0, 144, 144, True, 2),
        (144, 144, 144, 288, False, 1),
        (288, 0, 304, 304, True, 2),
        (304, 304, 304, 480, False, 1),
    ],
    "selecsls60": [
        (32, 0, 64, 64, True, 2),
        (64, 64, 64, 128, False, 1),
        (128, 0, 128, 128, True, 2),
        (128, 128, 128, 128, False, 1),
        (128, 128, 128, 288, False, 1),
        (288, 0, 288, 288, True, 2),
        (288, 288, 288, 288, False, 1),
        (288, 288, 288, 288, False, 1),
        (288, 288, 288, 416, False, 1),
    ],
    "selecsls84": [
        (32, 0, 64, 64, True, 2),
        (64, 64, 64, 144, False, 1),
        (144, 0, 144, 144, True, 2),
        *[(144, 144, 144, 144, False, 1)] * 3,
        (144, 144, 144, 304, False, 1),
        (304, 0, 304, 304, True, 2),
        *[(304, 304, 304, 304, False, 1)] * 4,
        (304, 304, 304, 512, False, 1),
    ],
}


def _head(in_chs, mid_chs, b=False, last_kernel=1):
    # in_chs, out_chs, kernel, stride; the ``b`` variants end with 1,024 channels.
    last = (1280, 1024) if b else (1024, 1280)
    return [
        (in_chs, mid_chs, 3, 2),
        (mid_chs, 1024, 3, 1),
        (1024, last[0], 3, 2),
        (last[0], last[1], last_kernel, 1),
    ]


_CFGS = {
    "selecsls42": (_FEATURES["selecsls42"], _head(480, 960)),
    "selecsls42b": (_FEATURES["selecsls42"], _head(480, 960, b=True)),
    "selecsls60": (_FEATURES["selecsls60"], _head(416, 756)),
    "selecsls60b": (_FEATURES["selecsls60"], _head(416, 756, b=True)),
    "selecsls84": (_FEATURES["selecsls84"], _head(512, 960, last_kernel=3)),
}


def _make(name):
    features, head = _CFGS[name]

    def entry(**kwargs):
        model = SelecSls(features, head, **kwargs)
        model.default_cfg = _cfg()
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
