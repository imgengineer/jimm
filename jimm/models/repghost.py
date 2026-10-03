"""RepGhostNet in flax nnx, NHWC. Mirrors timm.models.repghost.

RepGhost modules replace GhostNet's channel concatenation with a
re-parameterizable sum: a 1x1 convolution feeds a depthwise 3x3 convolution
whose output is added to a BatchNorm of its input (the fusion branch).
Bottlenecks expand with one module, optionally downsample with a strided
depthwise convolution and apply hard-sigmoid squeeze-excite, and project with
a second module. The head pools, projects to 1,280 channels with ReLU, and
classifies.
"""

from flax import nnx

from ..layers import BatchNorm, ClassifierMixin, SqueezeExcite, global_pool_nhwc, make_divisible
from ..registry import _cfg, register_model
from ._conv import ConvNormAct


class RepGhostModule(nnx.Module):
    def __init__(self, in_chs, out_chs, relu=True, reparam=True, *, rngs):
        self.relu = relu
        self.primary_conv = ConvNormAct(
            in_chs, out_chs, 1, act=nnx.relu if relu else None, rngs=rngs
        )
        self.fusion_bn = BatchNorm(out_chs, epsilon=1e-5, rngs=rngs) if reparam else None
        self.cheap_operation = ConvNormAct(out_chs, out_chs, 3, groups=out_chs, rngs=rngs)

    def __call__(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        if self.fusion_bn is not None:
            x2 = x2 + self.fusion_bn(x1)
        return nnx.relu(x2) if self.relu else x2


class RepGhostBottleneck(nnx.Module):
    def __init__(
        self, in_chs, mid_chs, out_chs, kernel=3, stride=1, se_ratio=0.0, reparam=True, *, rngs
    ):
        self.ghost1 = RepGhostModule(in_chs, mid_chs, True, reparam, rngs=rngs)
        self.conv_dw = (
            ConvNormAct(mid_chs, mid_chs, kernel, stride, groups=mid_chs, rngs=rngs)
            if stride > 1
            else None
        )
        self.se = (
            SqueezeExcite(
                mid_chs,
                rd_channels=make_divisible(mid_chs * se_ratio, 4),
                gate=nnx.hard_sigmoid,
                rngs=rngs,
            )
            if se_ratio > 0
            else None
        )
        self.ghost2 = RepGhostModule(mid_chs, out_chs, False, reparam, rngs=rngs)
        self.shortcut = (
            None
            if in_chs == out_chs and stride == 1
            else nnx.List(
                [
                    ConvNormAct(in_chs, in_chs, kernel, stride, groups=in_chs, rngs=rngs),
                    ConvNormAct(in_chs, out_chs, 1, rngs=rngs),
                ]
            )
        )

    def __call__(self, x):
        y = self.ghost1(x)
        if self.conv_dw is not None:
            y = self.conv_dw(y)
        if self.se is not None:
            y = self.se(y)
        y = self.ghost2(y)
        if self.shortcut is not None:
            for layer in self.shortcut:
                x = layer(x)
        return y + x


# Stages of (kernel, expansion, out_chs, se_ratio, stride) blocks.
_STAGES = [
    [(3, 8, 16, 0, 1)],
    [(3, 24, 24, 0, 2)],
    [(3, 36, 24, 0, 1)],
    [(5, 36, 40, 0.25, 2)],
    [(5, 60, 40, 0.25, 1)],
    [(3, 120, 80, 0, 2)],
    [
        (3, 100, 80, 0, 1),
        (3, 120, 80, 0, 1),
        (3, 120, 80, 0, 1),
        (3, 240, 112, 0.25, 1),
        (3, 336, 112, 0.25, 1),
    ],
    [(5, 336, 160, 0.25, 2)],
    [(5, 480, 160, 0, 1), (5, 480, 160, 0.25, 1), (5, 480, 160, 0, 1), (5, 480, 160, 0.25, 1)],
]


class RepGhostNet(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        width=1.0,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.2,
        reparam=True,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        stem_chs = make_divisible(16 * width, 4)
        self.conv_stem = ConvNormAct(in_chans, stem_chs, 3, 2, act=nnx.relu, rngs=rngs)
        stages, prev = [], stem_chs
        for cfg in _STAGES:
            blocks = []
            for kernel, exp, chs, se_ratio, stride in cfg:
                out = make_divisible(chs * width, 4)
                mid = make_divisible(exp * width, 4)
                blocks.append(
                    RepGhostBottleneck(prev, mid, out, kernel, stride, se_ratio, reparam, rngs=rngs)
                )
                prev = out
            stages.append(nnx.List(blocks))
        self.blocks = nnx.List(stages)
        pool_dim = make_divisible(_STAGES[-1][-1][1] * width * 2, 4)
        self.conv_last = ConvNormAct(prev, pool_dim, 1, act=nnx.relu, rngs=rngs)
        # timm's 1x1 head convolution runs on pooled features, i.e. a biased linear layer.
        self.conv_head = nnx.Linear(pool_dim, 1280, rngs=rngs)
        self.num_features = 1280
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(1280, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.conv_stem(x)
        for stage in self.blocks:
            for blk in stage:
                x = blk(x)
        return self.conv_last(x)

    def forward_head(self, x):
        x = nnx.relu(self.conv_head(global_pool_nhwc(x, self.global_pool)))
        x = self.head_drop(x)
        return self.fc(x) if self.fc is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_WIDTHS = {
    "repghostnet_050": 0.5,
    "repghostnet_058": 0.58,
    "repghostnet_080": 0.8,
    "repghostnet_100": 1.0,
    "repghostnet_111": 1.11,
    "repghostnet_130": 1.3,
    "repghostnet_150": 1.5,
    "repghostnet_200": 2.0,
}


def _make(name):
    width = _WIDTHS[name]

    def entry(**kwargs):
        model = RepGhostNet(width, **kwargs)
        model.default_cfg = _cfg(interpolation="bicubic")
        return model

    entry.__name__ = name
    return entry


for _name in _WIDTHS:
    register_model(_make(_name))
