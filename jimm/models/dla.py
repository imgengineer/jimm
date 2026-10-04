"""DLA (Deep Layer Aggregation) in flax nnx, NHWC. Mirrors timm.models.dla.

Each level is a tree of residual blocks whose outputs, together with the
max-pooled level input ("level root" for levels 3-5), are aggregated by a
1x1 root convolution.
"""

import math

import jax.numpy as jnp
from flax import nnx

from ..layers import BatchNorm, ClassifierMixin
from ..registry import _cfg, register_model
from ._conv import ConvNormAct


class DlaBasic(nnx.Module):
    def __init__(self, in_chs, out_chs, stride=1, cardinality=1, base_width=64, *, rngs):
        self.conv1 = ConvNormAct(in_chs, out_chs, 3, stride, act=nnx.relu, rngs=rngs)
        self.conv2 = ConvNormAct(out_chs, out_chs, 3, rngs=rngs)

    def __call__(self, x, shortcut=None):
        shortcut = x if shortcut is None else shortcut
        return nnx.relu(self.conv2(self.conv1(x)) + shortcut)


class DlaBottleneck(nnx.Module):
    expansion = 2

    def __init__(self, in_chs, out_chs, stride=1, cardinality=1, base_width=64, *, rngs):
        mid = int(math.floor(out_chs * (base_width / 64)) * cardinality) // self.expansion
        self.conv1 = ConvNormAct(in_chs, mid, act=nnx.relu, rngs=rngs)
        self.conv2 = ConvNormAct(mid, mid, 3, stride, cardinality, act=nnx.relu, rngs=rngs)
        self.conv3 = ConvNormAct(mid, out_chs, rngs=rngs)

    def __call__(self, x, shortcut=None):
        shortcut = x if shortcut is None else shortcut
        return nnx.relu(self.conv3(self.conv2(self.conv1(x))) + shortcut)


class DlaBottle2neck(nnx.Module):
    """Res2Net bottleneck for DLA: the 3x3 conv runs hierarchically over ``scale`` channel
    groups (the first group alone in strided blocks, which pool the untouched last group)."""

    expansion = 2

    def __init__(self, in_chs, out_chs, stride=1, cardinality=8, base_width=4, scale=4, *, rngs):
        self.is_first, self.scale = stride > 1, scale
        mid = int(math.floor(out_chs * (base_width / 64)) * cardinality) // self.expansion
        self.stride = stride
        self.conv1 = ConvNormAct(in_chs, mid * scale, act=nnx.relu, rngs=rngs)
        self.convs = nnx.List(
            [
                nnx.Conv(
                    mid,
                    mid,
                    (3, 3),
                    strides=stride,
                    padding=1,
                    feature_group_count=cardinality,
                    use_bias=False,
                    rngs=rngs,
                )  # fmt: skip
                for _ in range(max(1, scale - 1))
            ]
        )
        self.bns = nnx.List([BatchNorm(mid, rngs=rngs) for _ in range(max(1, scale - 1))])
        self.conv3 = ConvNormAct(mid * scale, out_chs, rngs=rngs)

    def __call__(self, x, shortcut=None):
        shortcut = x if shortcut is None else shortcut
        chunks = jnp.split(self.conv1(x), self.scale, axis=-1)
        out, prev = [], None
        for conv, bn, c in zip(self.convs, self.bns, chunks):
            prev = nnx.relu(bn(conv(c if (prev is None or self.is_first) else prev + c)))
            out.append(prev)
        if self.scale > 1:
            last = chunks[-1]
            if self.is_first:
                s = self.stride
                last = nnx.avg_pool(last, (3, 3), strides=(s, s), padding=((1, 1), (1, 1)))
            out.append(last)
        return nnx.relu(self.conv3(jnp.concatenate(out, axis=-1)) + shortcut)


class DlaRoot(nnx.Module):
    def __init__(self, in_chs, out_chs, shortcut=False, *, rngs):
        self.conv = ConvNormAct(in_chs, out_chs, rngs=rngs)
        self.shortcut = shortcut

    def __call__(self, children):
        x = self.conv(jnp.concatenate(children, axis=-1))
        if self.shortcut:
            x = x + children[0]
        return nnx.relu(x)


class DlaTree(nnx.Module):
    def __init__(
        self,
        levels,
        block,
        in_chs,
        out_chs,
        stride=1,
        cardinality=1,
        base_width=64,
        level_root=False,
        root_dim=0,
        root_shortcut=False,
        *,
        rngs,
    ):
        root_dim = root_dim or 2 * out_chs
        if level_root:
            root_dim += in_chs
        self.stride, self.level_root = stride, level_root
        cargs = dict(cardinality=cardinality, base_width=base_width)
        if levels == 1:
            self.tree1 = block(in_chs, out_chs, stride, **cargs, rngs=rngs)
            self.tree2 = block(out_chs, out_chs, 1, **cargs, rngs=rngs)
            self.project = ConvNormAct(in_chs, out_chs, rngs=rngs) if in_chs != out_chs else None
            self.root = DlaRoot(root_dim, out_chs, root_shortcut, rngs=rngs)
        else:
            cargs["root_shortcut"] = root_shortcut
            self.tree1 = DlaTree(levels - 1, block, in_chs, out_chs, stride, **cargs, rngs=rngs)
            self.tree2 = DlaTree(
                levels - 1, block, out_chs, out_chs, root_dim=root_dim + out_chs, **cargs, rngs=rngs
            )
            self.project = self.root = None

    def __call__(self, x, shortcut=None, children=None):
        children = [] if children is None else children
        s = self.stride
        bottom = nnx.max_pool(x, (s, s), strides=(s, s)) if s > 1 else x
        shortcut = bottom if self.project is None else self.project(bottom)
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, shortcut)
        if self.root is not None:
            return self.root([self.tree2(x1), x1] + children)
        children.append(x1)
        return self.tree2(x1, None, children)


class DLA(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        levels,
        channels,
        block,
        cardinality=1,
        base_width=64,
        shortcut_root=False,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.base_layer = ConvNormAct(in_chans, channels[0], 7, act=nnx.relu, rngs=rngs)
        self.level0 = nnx.List(
            [
                ConvNormAct(channels[0], channels[0], 3, act=nnx.relu, rngs=rngs)
                for _ in range(levels[0])
            ]
        )
        self.level1 = nnx.List(
            [
                ConvNormAct(
                    channels[0] if i == 0 else channels[1],
                    channels[1],
                    3,
                    2 if i == 0 else 1,
                    act=nnx.relu,
                    rngs=rngs,
                )
                for i in range(levels[1])
            ]
        )
        cargs = dict(cardinality=cardinality, base_width=base_width, root_shortcut=shortcut_root)
        self.level2 = DlaTree(levels[2], block, channels[1], channels[2], 2, **cargs, rngs=rngs)
        self.level3, self.level4, self.level5 = (
            DlaTree(
                levels[i],
                block,
                channels[i - 1],
                channels[i],
                2,
                level_root=True,
                **cargs,
                rngs=rngs,
            )
            for i in (3, 4, 5)
        )
        self.num_features = channels[5]
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(channels[5], num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.base_layer(x)
        for layer in (*self.level0, *self.level1):
            x = layer(x)
        return self.level5(self.level4(self.level3(self.level2(x))))

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_SMALL = (16, 32, 64, 64, 128, 256)
_LARGE = (16, 32, 128, 256, 512, 1024)
_CFGS = {
    "dla34": ((1, 1, 1, 2, 2, 1), (16, 32, 64, 128, 256, 512), DlaBasic, {}),
    "dla46_c": ((1, 1, 1, 2, 2, 1), _SMALL, DlaBottleneck, {}),
    "dla46x_c": ((1, 1, 1, 2, 2, 1), _SMALL, DlaBottleneck, dict(cardinality=32, base_width=4)),
    "dla60x_c": ((1, 1, 1, 2, 3, 1), _SMALL, DlaBottleneck, dict(cardinality=32, base_width=4)),
    "dla60": ((1, 1, 1, 2, 3, 1), _LARGE, DlaBottleneck, {}),
    "dla60x": ((1, 1, 1, 2, 3, 1), _LARGE, DlaBottleneck, dict(cardinality=32, base_width=4)),
    "dla102": ((1, 1, 1, 3, 4, 1), _LARGE, DlaBottleneck, dict(shortcut_root=True)),
    "dla102x": (
        (1, 1, 1, 3, 4, 1),
        _LARGE,
        DlaBottleneck,
        dict(cardinality=32, base_width=4, shortcut_root=True),
    ),
    "dla102x2": (
        (1, 1, 1, 3, 4, 1),
        _LARGE,
        DlaBottleneck,
        dict(cardinality=64, base_width=4, shortcut_root=True),
    ),
    "dla169": ((1, 1, 2, 3, 5, 1), _LARGE, DlaBottleneck, dict(shortcut_root=True)),
    "dla60_res2net": (
        (1, 1, 1, 2, 3, 1),
        _LARGE,
        DlaBottle2neck,
        dict(cardinality=1, base_width=28),
    ),
    "dla60_res2next": (
        (1, 1, 1, 2, 3, 1),
        _LARGE,
        DlaBottle2neck,
        dict(cardinality=8, base_width=4),
    ),
}


def _make(name):
    levels, channels, block, fixed = _CFGS[name]

    def entry(**kwargs):
        model = DLA(levels, channels, block, **fixed, **kwargs)
        model.default_cfg = _cfg()
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
