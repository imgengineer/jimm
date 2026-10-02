"""Selective Kernel ResNets in flax nnx, NHWC. Mirrors timm.models.sknet.

timm's variant runs a 3x3 path and a dilated 3x3 path (on split input channels
by default) and mixes them with a softmax over paths from a BN bottleneck.
"""

import math

import jax.numpy as jnp
from flax import nnx

from ..layers import BatchNorm, ClassifierMixin, DropPath, make_divisible
from ..registry import _cfg, register_model
from ._conv import ConvNormAct


class SelectiveKernel(nnx.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        stride=1,
        groups=1,
        rd_ratio=1 / 16,
        rd_divisor=8,
        split_input=True,
        *,
        rngs,
    ):
        # timm keep_3x3: kernel sizes (3, 5) become 3x3 convs with dilations (1, 2).
        self.split_input = split_input
        path_in = in_chs // 2 if split_input else in_chs
        groups = min(out_chs, groups)
        self.paths = nnx.List(
            [
                ConvNormAct(
                    path_in, out_chs, 3, stride, groups, act=nnx.relu, dilation=d, rngs=rngs
                )
                for d in (1, 2)
            ]
        )
        attn_chs = make_divisible(out_chs * rd_ratio, rd_divisor)
        self.fc_reduce = nnx.Linear(out_chs, attn_chs, use_bias=False, rngs=rngs)
        self.norm = BatchNorm(attn_chs, epsilon=1e-5, rngs=rngs)
        self.fc_select = nnx.Linear(attn_chs, out_chs * 2, use_bias=False, rngs=rngs)

    def __call__(self, x):
        if self.split_input:
            half = x.shape[-1] // 2
            paths = [op(x[..., i * half : (i + 1) * half]) for i, op in enumerate(self.paths)]
        else:
            paths = [op(x) for op in self.paths]
        s = jnp.mean(paths[0] + paths[1], axis=(1, 2))
        s = self.fc_select(nnx.relu(self.norm(self.fc_reduce(s))))
        attn = nnx.softmax(s.reshape(s.shape[0], 2, -1), axis=1).astype(paths[0].dtype)
        return paths[0] * attn[:, None, None, 0] + paths[1] * attn[:, None, None, 1]


class Downsample(nnx.Module):
    """timm 1x1 conv + BN shortcut, optionally after a 2x2 average pool (ResNet-D)."""

    def __init__(self, in_chs, out_chs, stride, avg_down=False, *, rngs):
        self.pool = avg_down and stride > 1
        self.conv = ConvNormAct(in_chs, out_chs, 1, 1 if avg_down else stride, rngs=rngs)

    def __call__(self, x):
        if self.pool:
            h, w = x.shape[1:3]
            x = nnx.avg_pool(
                x, (2, 2), strides=(2, 2), padding=((0, h % 2), (0, w % 2)), count_include_pad=False
            )
        return self.conv(x)


class SelectiveKernelBasic(nnx.Module):
    expansion = 1

    def __init__(
        self, in_chs, planes, stride=1, downsample=None, drop_path_rate=0.0, *, rngs, **sk
    ):
        self.conv1 = SelectiveKernel(in_chs, planes, stride, rngs=rngs, **sk)
        self.conv2 = ConvNormAct(planes, planes, 3, rngs=rngs)
        self.downsample = downsample
        self.drop_path = DropPath(drop_path_rate, rngs=rngs)

    def __call__(self, x):
        shortcut = x if self.downsample is None else self.downsample(x)
        return nnx.relu(self.drop_path(self.conv2(self.conv1(x))) + shortcut)


class SelectiveKernelBottleneck(nnx.Module):
    expansion = 4

    def __init__(
        self,
        in_chs,
        planes,
        stride=1,
        downsample=None,
        drop_path_rate=0.0,
        groups=1,
        base_width=64,
        *,
        rngs,
        **sk,
    ):
        width = int(math.floor(planes * (base_width / 64)) * groups)
        self.conv1 = ConvNormAct(in_chs, width, 1, act=nnx.relu, rngs=rngs)
        self.conv2 = SelectiveKernel(width, width, stride, groups, rngs=rngs, **sk)
        self.conv3 = ConvNormAct(width, planes * self.expansion, 1, rngs=rngs)
        self.downsample = downsample
        self.drop_path = DropPath(drop_path_rate, rngs=rngs)

    def __call__(self, x):
        shortcut = x if self.downsample is None else self.downsample(x)
        x = self.conv3(self.conv2(self.conv1(x)))
        return nnx.relu(self.drop_path(x) + shortcut)


class SKNet(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        block,
        layers,
        sk_kwargs=None,
        groups=1,
        base_width=64,
        deep_stem=False,
        avg_down=False,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        if deep_stem:  # timm stem_type="deep", stem_width=32
            self.stem = nnx.List(
                [
                    ConvNormAct(in_chans, 32, 3, 2, act=nnx.relu, rngs=rngs),
                    ConvNormAct(32, 32, 3, act=nnx.relu, rngs=rngs),
                    ConvNormAct(32, 64, 3, act=nnx.relu, rngs=rngs),
                ]
            )
        else:
            self.stem = nnx.List([ConvNormAct(in_chans, 64, 7, 2, act=nnx.relu, rngs=rngs)])
        block_kwargs = dict(sk_kwargs or {})
        if block is SelectiveKernelBottleneck:
            block_kwargs.update(groups=groups, base_width=base_width)
        dpr = [drop_path_rate * i / max(sum(layers) - 1, 1) for i in range(sum(layers))]
        chs, idx, stages = 64, 0, []
        for i, depth in enumerate(layers):
            planes, stride = 64 * 2**i, 1 if i == 0 else 2
            out_chs = planes * block.expansion
            blocks = []
            for j in range(depth):
                s = stride if j == 0 else 1
                downsample = (
                    Downsample(chs, out_chs, s, avg_down, rngs=rngs)
                    if j == 0 and (s != 1 or chs != out_chs)
                    else None
                )
                blocks.append(
                    block(chs, planes, s, downsample, dpr[idx], rngs=rngs, **block_kwargs)
                )
                chs, idx = out_chs, idx + 1
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.num_features = chs
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(chs, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        for layer in self.stem:
            x = layer(x)
        x = nnx.max_pool(x, (3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))
        for stage in self.stages:
            for blk in stage:
                x = blk(x)
        return x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


def _sknet(block, layers, **kwargs):
    model = SKNet(block, layers, **kwargs)
    model.default_cfg = _cfg(interpolation="bicubic")
    return model


_BASIC_SK = dict(rd_ratio=1 / 8, rd_divisor=16)


@register_model
def skresnet18(**kwargs):
    return _sknet(SelectiveKernelBasic, [2, 2, 2, 2], sk_kwargs=_BASIC_SK, **kwargs)


@register_model
def skresnet34(**kwargs):
    return _sknet(SelectiveKernelBasic, [3, 4, 6, 3], sk_kwargs=_BASIC_SK, **kwargs)


@register_model
def skresnet50(**kwargs):
    return _sknet(SelectiveKernelBottleneck, [3, 4, 6, 3], **kwargs)


@register_model
def skresnet50d(**kwargs):
    return _sknet(SelectiveKernelBottleneck, [3, 4, 6, 3], deep_stem=True, avg_down=True, **kwargs)


@register_model
def skresnet101(**kwargs):
    return _sknet(SelectiveKernelBottleneck, [3, 4, 23, 3], **kwargs)


@register_model
def skresnext50_32x4d(**kwargs):
    return _sknet(
        SelectiveKernelBottleneck,
        [3, 4, 6, 3],
        sk_kwargs=dict(rd_ratio=1 / 16, rd_divisor=32, split_input=False),
        groups=32,
        base_width=4,
        **kwargs,
    )
