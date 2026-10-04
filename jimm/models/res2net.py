"""Res2Net and Res2NeXt in flax nnx, NHWC. Mirrors timm.models.res2net.

timm builds these as ResNets with ``Bottle2neck`` blocks: the bottleneck's 3x3 conv is
split into ``scale`` channel groups processed hierarchically, each group adding the
previous group's output in stride-1 blocks. The ``d`` variants use the deep stem and
average-pool shortcuts.
"""

import math

import jax.numpy as jnp  # pyright: ignore[reportMissingImports]
from flax import nnx  # pyright: ignore[reportMissingImports]

from ..layers import BatchNorm, DropPath
from ..registry import _cfg, register_model
from .resnet import Downsample, ResNet


class Res2NetBottleneck(nnx.Module):
    expansion = 4
    scale = 4

    def __init__(
        self, in_chs, chs, stride=1, drop_path_rate=0.0, *, groups=1, base_width=26,
        avg_down=False, rngs, **_,
    ):  # fmt: skip
        out_chs = chs * self.expansion
        width = int(math.floor(chs * base_width / 64)) * groups
        mid = width * self.scale
        self.stride = stride
        # timm "is_first": strided or projecting blocks skip the cascade and pool the last chunk.
        self.is_first = stride > 1 or in_chs != out_chs
        self.conv1 = nnx.Conv(in_chs, mid, (1, 1), use_bias=False, rngs=rngs)
        self.bn1 = BatchNorm(mid, rngs=rngs)
        self.convs = nnx.List(
            [
                nnx.Conv(
                    width,
                    width,
                    (3, 3),
                    strides=(stride, stride),
                    padding=1,
                    feature_group_count=groups,
                    use_bias=False,
                    rngs=rngs,
                )  # fmt: skip
                for _ in range(self.scale - 1)
            ]
        )
        self.bns = nnx.List([BatchNorm(width, rngs=rngs) for _ in range(self.scale - 1)])
        self.conv3 = nnx.Conv(mid, out_chs, (1, 1), use_bias=False, rngs=rngs)
        self.bn3 = BatchNorm(out_chs, rngs=rngs)
        self.shortcut = (
            Downsample(in_chs, out_chs, stride, avg_down=avg_down, rngs=rngs)
            if (stride != 1 or in_chs != out_chs)
            else None
        )
        self.drop_path = DropPath(drop_path_rate, rngs=rngs)

    def __call__(self, x):
        y = nnx.relu(self.bn1(self.conv1(x)))
        # timm Bottle2neck: convs on chunks[:-1]; the last chunk passes through, average
        # pooled (3x3, padded, stride of the block) in a stage's first block, which also
        # skips the hierarchical add.
        chunks = jnp.split(y, self.scale, axis=-1)
        out, prev = [], None
        for conv, bn, c in zip(self.convs, self.bns, chunks[:-1]):
            prev = nnx.relu(bn(conv(c if (prev is None or self.is_first) else prev + c)))
            out.append(prev)
        last = chunks[-1]
        if self.is_first:
            last = nnx.avg_pool(
                last, (3, 3), strides=(self.stride, self.stride), padding=((1, 1), (1, 1))
            )
        out.append(last)
        y = jnp.concatenate(out, axis=-1)
        y = self.bn3(self.conv3(y))
        sc = x if self.shortcut is None else self.shortcut(x)
        return nnx.relu(self.drop_path(y) + sc)


def _block(scale):
    return type(f"Res2NetBottleneck{scale}s", (Res2NetBottleneck,), {"scale": scale})


def Res2Net(layers, scale=4, base_width=26, **kwargs):
    """A timm Res2Net: ``ResNet`` with ``scale``-way hierarchical bottlenecks."""
    return ResNet(_block(scale), layers, base_width=base_width, **kwargs)


_D = dict(stem_type="deep", stem_width=32, avg_down=True)
# name: (layers, scale, base width, extra ResNet arguments)
_CFGS = {
    "res2net50_26w_4s": ((3, 4, 6, 3), 4, 26, {}),
    "res2net101_26w_4s": ((3, 4, 23, 3), 4, 26, {}),
    "res2net50_26w_6s": ((3, 4, 6, 3), 6, 26, {}),
    "res2net50_26w_8s": ((3, 4, 6, 3), 8, 26, {}),
    "res2net50_48w_2s": ((3, 4, 6, 3), 2, 48, {}),
    "res2net50_14w_8s": ((3, 4, 6, 3), 8, 14, {}),
    "res2next50": ((3, 4, 6, 3), 4, 4, dict(groups=8)),
    "res2net50d": ((3, 4, 6, 3), 4, 26, _D),
    "res2net101d": ((3, 4, 23, 3), 4, 26, _D),
}


def _make(name):
    layers, scale, base_width, extra = _CFGS[name]

    def entry(**kwargs):
        model = Res2Net(layers, scale, base_width, **{**extra, **kwargs})
        model.default_cfg = _cfg()
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
