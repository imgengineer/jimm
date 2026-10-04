"""ResNeSt (Split-Attention) in flax nnx, NHWC. Mirrors timm.models.resnest."""

import jax.numpy as jnp
from flax import nnx

from ..layers import BatchNorm, ClassifierMixin, DropPath, make_divisible
from ..registry import _cfg, register_model
from ._conv import ConvNormAct


class SplitAttn(nnx.Module):
    """timm ``SplitAttn``: a grouped 3x3 conv with ``radix`` branches per cardinal group,
    mixed by a softmax over the branches (a sigmoid gate when ``radix`` is 1)."""

    def __init__(self, chs, radix=2, groups=1, stride=1, rd_ratio=0.25, *, rngs):
        self.radix, self.groups = radix, groups
        self.conv = ConvNormAct(
            chs, chs * radix, 3, stride, groups=groups * radix, act=nnx.relu, rngs=rngs
        )
        attn_chs = make_divisible(chs * radix * rd_ratio, 8, min_value=32)
        self.fc1 = nnx.Conv(chs, attn_chs, (1, 1), feature_group_count=groups, rngs=rngs)
        self.bn1 = BatchNorm(attn_chs, epsilon=1e-5, rngs=rngs)
        self.fc2 = nnx.Conv(attn_chs, chs * radix, (1, 1), feature_group_count=groups, rngs=rngs)

    def __call__(self, x):
        x = self.conv(x)
        B, H, W, RC = x.shape
        x = x.reshape(B, H, W, self.radix, RC // self.radix)
        gap = jnp.mean(x.sum(axis=3), axis=(1, 2), keepdims=True)
        attn = self.fc2(nnx.relu(self.bn1(self.fc1(gap)))).reshape(B, RC)
        if self.radix > 1:
            # fc2 channels are (group, radix, c); the softmax runs over radix per group.
            attn = attn.reshape(B, self.groups, self.radix, -1).transpose(0, 2, 1, 3)
            attn = nnx.softmax(attn, axis=1).reshape(B, self.radix, -1)
        else:
            attn = nnx.sigmoid(attn).reshape(B, 1, -1)
        return (x * attn[:, None, None].astype(x.dtype)).sum(axis=3)


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

    def __init__(
        self, in_chs, planes, stride=1, downsample=None, drop_path_rate=0.0, radix=2,
        cardinality=1, base_width=64, avd_first=False, *, rngs,
    ):  # fmt: skip
        width = int(planes * base_width / 64) * cardinality
        self.conv1 = ConvNormAct(in_chs, width, act=nnx.relu, rngs=rngs)
        # timm avd: strided blocks move the stride to a 3x3 average pool, before
        # (avd_first) or after the split attention.
        self.avd_stride, self.avd_first = stride if stride > 1 else 0, avd_first
        self.conv2 = SplitAttn(width, radix, cardinality, rngs=rngs)
        self.conv3 = ConvNormAct(width, planes * self.expansion, bn_weight_init=0.0, rngs=rngs)
        self.downsample = downsample
        # Stochastic depth drops the residual branch (timm applies it to the block input).
        self.drop_path = DropPath(drop_path_rate, rngs=rngs)

    def _avd(self, x):
        s = self.avd_stride
        return nnx.avg_pool(x, (3, 3), strides=(s, s), padding=((1, 1), (1, 1)))

    def __call__(self, x):
        out = self.conv1(x)
        if self.avd_stride and self.avd_first:
            out = self._avd(out)
        out = self.conv2(out)
        if self.avd_stride and not self.avd_first:
            out = self._avd(out)
        out = self.conv3(out)
        shortcut = x if self.downsample is None else self.downsample(x)
        return nnx.relu(self.drop_path(out) + shortcut)


class ResNeSt(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        layers,
        stem_width=32,
        radix=2,
        cardinality=1,
        base_width=64,
        avd_first=False,
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
                blocks.append(
                    ResNeStBottleneck(
                        chs,
                        planes,
                        s,
                        downsample,
                        dpr[k],
                        radix,
                        cardinality,
                        base_width,
                        avd_first,
                        rngs=rngs,
                    )  # fmt: skip
                )
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


_CFGS = {
    # name: (layers, stem width, image size, crop, interpolation, block arguments)
    "resnest14d": ((1, 1, 1, 1), 32, 224, 0.875, "bilinear", {}),
    "resnest26d": ((2, 2, 2, 2), 32, 224, 0.875, "bilinear", {}),
    "resnest50d": ((3, 4, 6, 3), 32, 224, 0.875, "bilinear", {}),
    "resnest101e": ((3, 4, 23, 3), 64, 256, 0.875, "bilinear", {}),
    "resnest200e": ((3, 24, 36, 3), 64, 320, 0.909, "bicubic", {}),
    "resnest269e": ((3, 30, 48, 8), 64, 416, 0.928, "bicubic", {}),
    "resnest50d_4s2x40d": (
        (3, 4, 6, 3), 32, 224, 0.875, "bicubic",
        dict(radix=4, cardinality=2, base_width=40, avd_first=True),
    ),
    "resnest50d_1s4x24d": (
        (3, 4, 6, 3), 32, 224, 0.875, "bicubic",
        dict(radix=1, cardinality=4, base_width=24, avd_first=True),
    ),
}  # fmt: skip


def _make(name):
    layers, stem_width, size, crop, interp, block = _CFGS[name]

    def entry(**kwargs):
        model = ResNeSt(layers, stem_width=stem_width, **{**block, **kwargs})
        model.default_cfg = _cfg(input_size=(3, size, size), crop_pct=crop, interpolation=interp)
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
