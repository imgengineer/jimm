"""HRNet in flax nnx, NHWC. Mirrors timm.models.hrnet with the classification head.

Each high-resolution module owns its branches and fusion layers; the head
increases every branch with a bottleneck, merges them by strided convolutions,
and projects to 2048 features before pooling.
"""

import jax.numpy as jnp
from flax import nnx

from ..layers import ClassifierMixin
from ..registry import _cfg, register_model
from ._conv import ConvNormAct


class BasicBlock(nnx.Module):
    expansion = 1

    def __init__(self, in_chs, chs, stride=1, *, rngs):
        self.conv1 = ConvNormAct(in_chs, chs, 3, stride, act=nnx.relu, rngs=rngs)
        self.conv2 = ConvNormAct(chs, chs, 3, rngs=rngs)
        self.downsample = (
            ConvNormAct(in_chs, chs, 1, stride, rngs=rngs) if stride != 1 or in_chs != chs else None
        )

    def __call__(self, x):
        shortcut = x if self.downsample is None else self.downsample(x)
        return nnx.relu(self.conv2(self.conv1(x)) + shortcut)


class Bottleneck(nnx.Module):
    expansion = 4

    def __init__(self, in_chs, chs, stride=1, *, rngs):
        out_chs = chs * self.expansion
        self.conv1 = ConvNormAct(in_chs, chs, 1, act=nnx.relu, rngs=rngs)
        self.conv2 = ConvNormAct(chs, chs, 3, stride, act=nnx.relu, rngs=rngs)
        self.conv3 = ConvNormAct(chs, out_chs, 1, rngs=rngs)
        self.downsample = (
            ConvNormAct(in_chs, out_chs, 1, stride, rngs=rngs)
            if stride != 1 or in_chs != out_chs
            else None
        )

    def __call__(self, x):
        shortcut = x if self.downsample is None else self.downsample(x)
        return nnx.relu(self.conv3(self.conv2(self.conv1(x))) + shortcut)


def _make_layer(block, in_chs, chs, num_blocks, stride=1, *, rngs):
    layers = [block(in_chs, chs, stride, rngs=rngs)]
    layers += [block(chs * block.expansion, chs, rngs=rngs) for _ in range(1, num_blocks)]
    return nnx.List(layers)


def _run(layers, x):
    for layer in layers:
        x = layer(x)
    return x


def _upsample_nearest(x, factor):
    batch, height, width, chs = x.shape
    x = jnp.broadcast_to(x[:, :, None, :, None], (batch, height, factor, width, factor, chs))
    return x.reshape(batch, height * factor, width * factor, chs)


class HighResolutionModule(nnx.Module):
    """Parallel resolution branches followed by all-to-all fusion."""

    def __init__(self, block, num_blocks, in_chs, chs, *, rngs):
        self.branches = nnx.List(
            [
                _make_layer(block, branch_in, branch_chs, blocks, rngs=rngs)
                for branch_in, branch_chs, blocks in zip(in_chs, chs, num_blocks)
            ]
        )
        widths = [c * block.expansion for c in chs]
        self.out_chs = widths
        fuse_layers = []
        for i in range(len(widths) if len(widths) > 1 else 0):
            row = []
            for j, width in enumerate(widths):
                if j > i:  # 1x1 projection, then nearest upsampling by 2 ** (j - i)
                    row.append(ConvNormAct(width, widths[i], 1, rngs=rngs))
                elif j == i:
                    row.append(None)
                else:  # (i - j) stride-two 3x3 convolutions
                    steps = []
                    for k in range(i - j):
                        last = k == i - j - 1
                        steps.append(
                            ConvNormAct(
                                width,
                                widths[i] if last else width,
                                3,
                                2,
                                act=None if last else nnx.relu,
                                rngs=rngs,
                            )
                        )
                    row.append(nnx.List(steps))
            fuse_layers.append(nnx.List(row))
        self.fuse_layers = nnx.List(fuse_layers)

    def __call__(self, xs):
        xs = [_run(branch, x) for branch, x in zip(self.branches, xs)]
        if len(self.fuse_layers) == 0:
            return xs
        outs = []
        for i, row in enumerate(self.fuse_layers):
            y = None
            for j, (layer, x) in enumerate(zip(row, xs)):
                if layer is None:
                    z = x
                elif j > i:
                    z = _upsample_nearest(layer(x), 2 ** (j - i))
                else:
                    z = _run(layer, x)
                y = z if y is None else y + z
            outs.append(nnx.relu(y))
        return outs


def _make_transition(pre_chs, cur_chs, *, rngs):
    layers = []
    for i, width in enumerate(cur_chs):
        if i < len(pre_chs):
            layers.append(
                ConvNormAct(pre_chs[i], width, 3, act=nnx.relu, rngs=rngs)
                if width != pre_chs[i]
                else None
            )
        else:
            steps = []
            for j in range(i + 1 - len(pre_chs)):
                out_chs = width if j == i - len(pre_chs) else pre_chs[-1]
                steps.append(ConvNormAct(pre_chs[-1], out_chs, 3, 2, act=nnx.relu, rngs=rngs))
            layers.append(nnx.List(steps))
    return nnx.List(layers)


class HRNet(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        stage1,
        stages,
        stem_width=64,
        head_conv_bias=True,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.stem = nnx.List(
            [
                ConvNormAct(in_chans, stem_width, 3, 2, act=nnx.relu, rngs=rngs),
                ConvNormAct(stem_width, 64, 3, 2, act=nnx.relu, rngs=rngs),
            ]
        )
        stage1_blocks, stage1_chs = stage1
        self.layer1 = _make_layer(Bottleneck, 64, stage1_chs, stage1_blocks, rngs=rngs)
        pre_chs = [stage1_chs * Bottleneck.expansion]
        transitions, stage_modules = [], []
        for num_modules, num_blocks, chs in stages:
            transitions.append(_make_transition(pre_chs, list(chs), rngs=rngs))
            modules, in_chs = [], list(chs)
            for _ in range(num_modules):
                modules.append(HighResolutionModule(BasicBlock, num_blocks, in_chs, chs, rngs=rngs))
                in_chs = modules[-1].out_chs
            stage_modules.append(nnx.List(modules))
            pre_chs = in_chs
        self.transitions = nnx.List(transitions)
        self.stages = nnx.List(stage_modules)

        head_chs = (32, 64, 128, 256)
        self.incre_modules = nnx.List(
            [
                _make_layer(Bottleneck, chs, head, 1, rngs=rngs)
                for chs, head in zip(pre_chs, head_chs)
            ]
        )
        self.downsamp_modules = nnx.List(
            [
                ConvNormAct(
                    head_chs[i] * 4,
                    head_chs[i + 1] * 4,
                    3,
                    2,
                    act=nnx.relu,
                    use_bias=head_conv_bias,
                    rngs=rngs,
                )
                for i in range(len(head_chs) - 1)
            ]
        )
        self.num_features = 2048
        self.final_layer = ConvNormAct(
            head_chs[-1] * 4, self.num_features, 1, act=nnx.relu, use_bias=head_conv_bias, rngs=rngs
        )
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = _run(self.layer1, _run(self.stem, x))
        ys = [x]
        for transition, modules in zip(self.transitions, self.stages):
            # New branches and changed widths derive from the lowest-resolution output.
            xs = []
            for i, layer in enumerate(transition):
                if layer is None:
                    xs.append(ys[i])
                else:
                    xs.append(_run(layer, ys[-1]) if isinstance(layer, nnx.List) else layer(ys[-1]))
            for module in modules:
                xs = module(xs)
            ys = xs
        y = _run(self.incre_modules[0], ys[0])
        for incre, down, x in zip(self.incre_modules[1:], self.downsamp_modules, ys[1:]):
            y = _run(incre, x) + down(y)
        return self.final_layer(y)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


# timm configurations: stage1 (blocks, channels); stages 2-4 (modules, blocks, channels);
# extra arguments (the _ssld models drop the head conv biases); eval config overrides.
_CFGS = {
    "hrnet_w18": (
        (4, 64),
        ((1, (4, 4), (18, 36)), (4, (4, 4, 4), (18, 36, 72)), (3, (4, 4, 4, 4), (18, 36, 72, 144))),
        {},
        {"crop_pct": 0.95},
    ),
    "hrnet_w18_small": (
        (1, 32),
        ((1, (2, 2), (16, 32)), (1, (2, 2, 2), (16, 32, 64)), (1, (2, 2, 2, 2), (16, 32, 64, 128))),
        {},
        {"interpolation": "bicubic"},
    ),
    "hrnet_w18_small_v2": (
        (2, 64),
        ((1, (2, 2), (18, 36)), (3, (2, 2, 2), (18, 36, 72)), (2, (2, 2, 2, 2), (18, 36, 72, 144))),
        {},
        {"interpolation": "bicubic"},
    ),
    "hrnet_w18_ssld": (
        (4, 64),
        ((1, (4, 4), (18, 36)), (4, (4, 4, 4), (18, 36, 72)), (3, (4, 4, 4, 4), (18, 36, 72, 144))),
        {"head_conv_bias": False},
        {"crop_pct": 0.95, "test_input_size": (3, 288, 288)},
    ),
    "hrnet_w30": (
        (4, 64),
        (
            (1, (4, 4), (30, 60)),
            (4, (4, 4, 4), (30, 60, 120)),
            (3, (4, 4, 4, 4), (30, 60, 120, 240)),
        ),
        {},
        {},
    ),
    "hrnet_w32": (
        (4, 64),
        (
            (1, (4, 4), (32, 64)),
            (4, (4, 4, 4), (32, 64, 128)),
            (3, (4, 4, 4, 4), (32, 64, 128, 256)),
        ),
        {},
        {},
    ),
    "hrnet_w40": (
        (4, 64),
        (
            (1, (4, 4), (40, 80)),
            (4, (4, 4, 4), (40, 80, 160)),
            (3, (4, 4, 4, 4), (40, 80, 160, 320)),
        ),
        {},
        {},
    ),
    "hrnet_w44": (
        (4, 64),
        (
            (1, (4, 4), (44, 88)),
            (4, (4, 4, 4), (44, 88, 176)),
            (3, (4, 4, 4, 4), (44, 88, 176, 352)),
        ),
        {},
        {},
    ),
    "hrnet_w48": (
        (4, 64),
        (
            (1, (4, 4), (48, 96)),
            (4, (4, 4, 4), (48, 96, 192)),
            (3, (4, 4, 4, 4), (48, 96, 192, 384)),
        ),
        {},
        {},
    ),
    "hrnet_w48_ssld": (
        (4, 64),
        (
            (1, (4, 4), (48, 96)),
            (4, (4, 4, 4), (48, 96, 192)),
            (3, (4, 4, 4, 4), (48, 96, 192, 384)),
        ),
        {"head_conv_bias": False},
        {"crop_pct": 0.95, "test_input_size": (3, 288, 288)},
    ),
    "hrnet_w64": (
        (4, 64),
        (
            (1, (4, 4), (64, 128)),
            (4, (4, 4, 4), (64, 128, 256)),
            (3, (4, 4, 4, 4), (64, 128, 256, 512)),
        ),
        {},
        {},
    ),
}


def _make(name):
    stage1, stages, extra, ev = _CFGS[name]

    def entry(**kwargs):
        model = HRNet(stage1, stages, **{**extra, **kwargs})
        model.default_cfg = _cfg(**ev)
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
