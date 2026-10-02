"""TResNet in flax nnx, NHWC. Mirrors timm.models.tresnet.

SpaceToDepth stem, anti-aliased (BlurPool) downsampling, SE in the first three
stages, and LeakyReLU blocks with ReLU after each residual sum.
"""

from functools import partial

import jax.numpy as jnp
from flax import nnx

from ..layers import ClassifierMixin, DropPath, SqueezeExcite
from ..registry import _cfg, register_model
from ._conv import ConvNormAct

_block_act = partial(nnx.leaky_relu, negative_slope=1e-3)


def space_to_depth(x, block_size=4):
    """(B,H,W,C) -> (B,H/4,W/4,16C), channels ordered (row, column, C) as in timm."""
    b, h, w, c = x.shape
    x = x.reshape(b, h // block_size, block_size, w // block_size, block_size, c)
    x = x.transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(b, h // block_size, w // block_size, block_size * block_size * c)


def blur_pool(x):
    """timm BlurPool2d: [1, 2, 1] binomial blur with reflect padding, then stride two.

    The separable filter is applied as strided slices, which XLA fuses into one pass.
    """
    x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="reflect")
    h, w = x.shape[1] - 2, x.shape[2] - 2
    x = (x[:, 0:h:2] + 2 * x[:, 1 : h + 1 : 2] + x[:, 2 : h + 2 : 2]) * 0.25
    return (x[:, :, 0:w:2] + 2 * x[:, :, 1 : w + 1 : 2] + x[:, :, 2 : w + 2 : 2]) * 0.25


class Downsample(nnx.Module):
    """Average pool (ceil mode, no padding counted) before the 1x1 projection, as in timm."""

    def __init__(self, in_chs, out_chs, stride, *, rngs):
        self.stride = stride
        self.conv = ConvNormAct(in_chs, out_chs, rngs=rngs)

    def __call__(self, x):
        if self.stride == 2:
            h, w = x.shape[1:3]
            x = nnx.avg_pool(
                x, (2, 2), strides=(2, 2), padding=((0, h % 2), (0, w % 2)), count_include_pad=False
            )
        return self.conv(x)


class BasicBlock(nnx.Module):
    expansion = 1

    def __init__(self, in_chs, planes, stride=1, use_se=True, drop_path_rate=0.0, *, rngs):
        self.stride = stride
        self.conv1 = ConvNormAct(in_chs, planes, 3, act=_block_act, rngs=rngs)
        # Zero-initialized BN scale makes each residual branch start as identity.
        self.conv2 = ConvNormAct(planes, planes, 3, bn_weight_init=0.0, rngs=rngs)
        self.se = (
            SqueezeExcite(planes, rd_channels=max(planes // 4, 64), rngs=rngs) if use_se else None
        )
        self.downsample = (
            Downsample(in_chs, planes, stride, rngs=rngs)
            if stride != 1 or in_chs != planes
            else None
        )
        self.drop_path = DropPath(drop_path_rate, rngs=rngs)

    def __call__(self, x):
        shortcut = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        if self.stride == 2:
            out = blur_pool(out)
        out = self.conv2(out)
        if self.se is not None:
            out = self.se(out)
        return nnx.relu(self.drop_path(out) + shortcut)


class Bottleneck(nnx.Module):
    expansion = 4

    def __init__(self, in_chs, planes, stride=1, use_se=True, drop_path_rate=0.0, *, rngs):
        out_chs = planes * self.expansion
        self.stride = stride
        self.conv1 = ConvNormAct(in_chs, planes, 1, act=_block_act, rngs=rngs)
        self.conv2 = ConvNormAct(planes, planes, 3, act=_block_act, rngs=rngs)
        self.se = (
            SqueezeExcite(planes, rd_channels=max(out_chs // 8, 64), rngs=rngs) if use_se else None
        )
        self.conv3 = ConvNormAct(planes, out_chs, 1, bn_weight_init=0.0, rngs=rngs)
        self.downsample = (
            Downsample(in_chs, out_chs, stride, rngs=rngs)
            if stride != 1 or in_chs != out_chs
            else None
        )
        self.drop_path = DropPath(drop_path_rate, rngs=rngs)

    def __call__(self, x):
        shortcut = x if self.downsample is None else self.downsample(x)
        out = self.conv2(self.conv1(x))
        if self.stride == 2:
            out = blur_pool(out)
        if self.se is not None:
            out = self.se(out)
        out = self.conv3(out)
        return nnx.relu(self.drop_path(out) + shortcut)


class TResNet(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        layers,
        width_factor=1.0,
        v2=False,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        planes = int(64 * width_factor)
        if v2:
            planes = planes // 8 * 8
        self.conv1 = ConvNormAct(in_chans * 16, planes, 3, act=nnx.leaky_relu, rngs=rngs)
        dpr = [drop_path_rate * i / max(sum(layers) - 1, 1) for i in range(sum(layers))]
        first = Bottleneck if v2 else BasicBlock
        stage_cfgs = (
            (first, planes, 1, True),
            (first, planes * 2, 2, True),
            (Bottleneck, planes * 4, 2, True),
            (Bottleneck, planes * 8, 2, False),
        )
        chs, idx, stages = planes, 0, []
        for (block, stage_planes, stride, use_se), depth in zip(stage_cfgs, layers):
            blocks = []
            for i in range(depth):
                blocks.append(
                    block(
                        chs,
                        stage_planes,
                        stride if i == 0 else 1,
                        use_se=use_se,
                        drop_path_rate=dpr[idx],
                        rngs=rngs,
                    )
                )
                chs, idx = stage_planes * block.expansion, idx + 1
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.num_features = chs
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = (
            nnx.Linear(
                self.num_features,
                num_classes,
                kernel_init=nnx.initializers.normal(0.01),
                rngs=rngs,
            )
            if num_classes > 0
            else None
        )

    def forward_features(self, x):
        x = self.conv1(space_to_depth(x))
        for stage in self.stages:
            for blk in stage:
                x = blk(x)
        return x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


def _tresnet(layers, **kwargs):
    model = TResNet(layers, **kwargs)
    model.default_cfg = _cfg(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
    return model


@register_model
def tresnet_m(**kwargs):
    return _tresnet([3, 4, 11, 3], **kwargs)


@register_model
def tresnet_l(**kwargs):
    return _tresnet([4, 5, 18, 3], width_factor=1.2, **kwargs)


@register_model
def tresnet_xl(**kwargs):
    return _tresnet([4, 5, 24, 3], width_factor=1.3, **kwargs)


@register_model
def tresnet_v2_l(**kwargs):
    return _tresnet([3, 4, 23, 3], v2=True, **kwargs)
