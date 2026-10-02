"""ResNeSt (Split-Attention) in flax nnx, NHWC. Mirrors timm.models.resnest."""

import jax.numpy as jnp
from flax import nnx

from ..layers import BatchNorm, ClassifierMixin, DropPath, make_divisible
from ..registry import _cfg, register_model
from ._conv import ConvNormAct


def _avg_pool_3x3_s2(x):
    # PyTorch AvgPool2d(3, 2, padding=1), which counts the zero padding.
    return nnx.avg_pool(x, (3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))


class SplitAttn(nnx.Module):
    """Radix-2 split attention: a grouped 3x3 conv whose branches are mixed by a softmax."""

    def __init__(self, chs, radix=2, rd_ratio=0.25, *, rngs):
        self.radix = radix
        self.conv = ConvNormAct(chs, chs * radix, 3, groups=radix, act=nnx.relu, rngs=rngs)
        attn_chs = make_divisible(chs * radix * rd_ratio, 8, min_value=32)
        self.fc1 = nnx.Linear(chs, attn_chs, rngs=rngs)
        self.bn1 = BatchNorm(attn_chs, epsilon=1e-5, rngs=rngs)
        self.fc2 = nnx.Linear(attn_chs, chs * radix, rngs=rngs)

    def __call__(self, x):
        x = self.conv(x)
        B, H, W, _ = x.shape
        x = x.reshape(B, H, W, self.radix, -1)
        gap = jnp.mean(x.sum(axis=3), axis=(1, 2))
        attn = self.fc2(nnx.relu(self.bn1(self.fc1(gap)))).reshape(B, self.radix, -1)
        attn = nnx.softmax(attn, axis=1).astype(x.dtype)
        return (x * attn[:, None, None]).sum(axis=3)


class Downsample(nnx.Module):
    """timm downsample_avg: ceil-mode 2x2 average pool (when strided), 1x1 conv, BN."""

    def __init__(self, in_chs, out_chs, stride, *, rngs):
        self.stride = stride
        self.conv = ConvNormAct(in_chs, out_chs, rngs=rngs)

    def __call__(self, x):
        if self.stride > 1:
            h, w = x.shape[1:3]
            x = nnx.avg_pool(
                x, (2, 2), strides=(2, 2), padding=((0, h % 2), (0, w % 2)), count_include_pad=False
            )
        return self.conv(x)


class ResNeStBottleneck(nnx.Module):
    expansion = 4

    def __init__(self, in_chs, planes, stride=1, downsample=None, drop_path_rate=0.0, *, rngs):
        self.conv1 = ConvNormAct(in_chs, planes, act=nnx.relu, rngs=rngs)
        self.conv2 = SplitAttn(planes, rngs=rngs)
        # timm avd (avd_first=False): strided blocks average-pool after the split attention.
        self.avd = stride > 1
        self.conv3 = ConvNormAct(planes, planes * self.expansion, bn_weight_init=0.0, rngs=rngs)
        self.downsample = downsample
        # Stochastic depth drops the residual branch (timm applies it to the block input).
        self.drop_path = DropPath(drop_path_rate, rngs=rngs)

    def __call__(self, x):
        out = self.conv2(self.conv1(x))
        if self.avd:
            out = _avg_pool_3x3_s2(out)
        out = self.conv3(out)
        shortcut = x if self.downsample is None else self.downsample(x)
        return nnx.relu(self.drop_path(out) + shortcut)


class ResNeSt(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        layers,
        stem_width=32,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = 512 * ResNeStBottleneck.expansion
        # timm deep stem: three 3x3 convs, the last widening to 2 * stem_width.
        self.stem = nnx.List(
            [
                ConvNormAct(in_chans, stem_width, 3, 2, act=nnx.relu, rngs=rngs),
                ConvNormAct(stem_width, stem_width, 3, act=nnx.relu, rngs=rngs),
                ConvNormAct(stem_width, stem_width * 2, 3, act=nnx.relu, rngs=rngs),
            ]
        )
        dpr = [drop_path_rate * i / max(sum(layers) - 1, 1) for i in range(sum(layers))]
        chs, stages, k = stem_width * 2, [], 0
        for i, depth in enumerate(layers):
            planes, stride = 64 * 2**i, 1 if i == 0 else 2
            out_chs = planes * ResNeStBottleneck.expansion
            blocks = []
            for j in range(depth):
                s = stride if j == 0 else 1
                downsample = (
                    Downsample(chs, out_chs, s, rngs=rngs)
                    if j == 0 and (s != 1 or chs != out_chs)
                    else None
                )
                blocks.append(ResNeStBottleneck(chs, planes, s, downsample, dpr[k], rngs=rngs))
                chs, k = out_chs, k + 1
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

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


def _resnest(layers, stem_width=32, img_size=224, **kwargs):
    model = ResNeSt(layers, stem_width=stem_width, **kwargs)
    model.default_cfg = _cfg(input_size=(3, img_size, img_size))
    return model


@register_model
def resnest14d(**kwargs):
    return _resnest([1, 1, 1, 1], **kwargs)


@register_model
def resnest50d(**kwargs):
    return _resnest([3, 4, 6, 3], **kwargs)


@register_model
def resnest101e(**kwargs):
    return _resnest([3, 4, 23, 3], stem_width=64, img_size=256, **kwargs)
