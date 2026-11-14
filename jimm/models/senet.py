"""SENet-154 and the legacy SE-ResNet/SE-ResNeXt models in flax nnx, NHWC.

``senet154`` mirrors timm's ``senet154`` (timm.models.resnet). The ``legacy_*``
entries mirror timm.models.senet, whose basic/blocks place the projecting
stride differently from the ResNet module and use a wider squeeze-excite
reduction.
"""

import jax.numpy as jnp
from flax import nnx

from ..layers import BatchNorm, ClassifierMixin
from ..registry import _cfg, register_model


class SEModule(nnx.Module):
    """Squeeze-excite with a fixed ``channels // reduction`` bottleneck."""

    def __init__(self, channels, reduction, *, rngs):
        rd = channels // reduction
        self.fc1 = nnx.Linear(channels, rd, rngs=rngs)
        self.fc2 = nnx.Linear(rd, channels, rngs=rngs)

    def __call__(self, x):
        s = jnp.mean(x, axis=(1, 2), keepdims=True)
        return x * nnx.sigmoid(self.fc2(nnx.relu(self.fc1(s))))


def _conv_bn(in_chs, out_chs, kernel, stride, padding, *, rngs):
    return nnx.Sequential(
        nnx.Conv(
            in_chs,
            out_chs,
            (kernel, kernel),
            strides=(stride, stride),
            padding=((padding, padding),) * 2,
            use_bias=False,
            rngs=rngs,
        ),
        BatchNorm(out_chs, rngs=rngs),
    )


class SEResNetBlock(nnx.Module):
    expansion = 1

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None, *, rngs):
        self.conv1 = nnx.Conv(
            inplanes,
            planes,
            (3, 3),
            strides=(stride, stride),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            rngs=rngs,
        )
        self.bn1 = BatchNorm(planes, rngs=rngs)
        self.conv2 = nnx.Conv(
            planes,
            planes,
            (3, 3),
            padding=((1, 1), (1, 1)),
            feature_group_count=groups,
            use_bias=False,
            rngs=rngs,
        )
        self.bn2 = BatchNorm(planes, rngs=rngs)
        self.se = SEModule(planes, reduction, rngs=rngs)
        self.downsample = downsample

    def __call__(self, x):
        shortcut = x if self.downsample is None else self.downsample(x)
        out = nnx.relu(self.bn1(self.conv1(x)))
        out = nnx.relu(self.bn2(self.conv2(out)))
        return nnx.relu(self.se(out) + shortcut)


class SEResNetBottleneck(nnx.Module):
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None, *, rngs):
        # The projecting stride is on the first 1x1 conv, as in Cadene's SENet.
        self.conv1 = nnx.Conv(
            inplanes,
            planes,
            (1, 1),
            strides=(stride, stride),
            use_bias=False,
            rngs=rngs,
        )
        self.bn1 = BatchNorm(planes, rngs=rngs)
        self.conv2 = nnx.Conv(
            planes,
            planes,
            (3, 3),
            padding=((1, 1), (1, 1)),
            feature_group_count=groups,
            use_bias=False,
            rngs=rngs,
        )
        self.bn2 = BatchNorm(planes, rngs=rngs)
        self.conv3 = nnx.Conv(planes, planes * 4, (1, 1), use_bias=False, rngs=rngs)
        self.bn3 = BatchNorm(planes * 4, rngs=rngs)
        self.se = SEModule(planes * 4, reduction, rngs=rngs)
        self.downsample = downsample

    def __call__(self, x):
        shortcut = x if self.downsample is None else self.downsample(x)
        out = nnx.relu(self.bn1(self.conv1(x)))
        out = nnx.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return nnx.relu(self.se(out) + shortcut)


class SEResNeXtBottleneck(SEResNetBottleneck):
    expansion = 4

    def __init__(
        self, inplanes, planes, groups, reduction, stride=1, downsample=None, base_width=4, *, rngs
    ):
        width = int(planes * (base_width / 64)) * groups
        self.conv1 = nnx.Conv(inplanes, width, (1, 1), use_bias=False, rngs=rngs)
        self.bn1 = BatchNorm(width, rngs=rngs)
        self.conv2 = nnx.Conv(
            width,
            width,
            (3, 3),
            strides=(stride, stride),
            padding=((1, 1), (1, 1)),
            feature_group_count=groups,
            use_bias=False,
            rngs=rngs,
        )
        self.bn2 = BatchNorm(width, rngs=rngs)
        self.conv3 = nnx.Conv(width, planes * 4, (1, 1), use_bias=False, rngs=rngs)
        self.bn3 = BatchNorm(planes * 4, rngs=rngs)
        self.se = SEModule(planes * 4, reduction, rngs=rngs)
        self.downsample = downsample

    def __call__(self, x):
        shortcut = x if self.downsample is None else self.downsample(x)
        out = nnx.relu(self.bn1(self.conv1(x)))
        out = nnx.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return nnx.relu(self.se(out) + shortcut)


class SEBottleneck(SEResNetBottleneck):
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None, *, rngs):
        self.conv1 = nnx.Conv(inplanes, planes * 2, (1, 1), use_bias=False, rngs=rngs)
        self.bn1 = BatchNorm(planes * 2, rngs=rngs)
        self.conv2 = nnx.Conv(
            planes * 2,
            planes * 4,
            (3, 3),
            strides=(stride, stride),
            padding=((1, 1), (1, 1)),
            feature_group_count=groups,
            use_bias=False,
            rngs=rngs,
        )
        self.bn2 = BatchNorm(planes * 4, rngs=rngs)
        self.conv3 = nnx.Conv(planes * 4, planes * 4, (1, 1), use_bias=False, rngs=rngs)
        self.bn3 = BatchNorm(planes * 4, rngs=rngs)
        self.se = SEModule(planes * 4, reduction, rngs=rngs)
        self.downsample = downsample

    def __call__(self, x):
        shortcut = x if self.downsample is None else self.downsample(x)
        out = nnx.relu(self.bn1(self.conv1(x)))
        out = nnx.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return nnx.relu(self.se(out) + shortcut)


class SENet(ClassifierMixin, nnx.Module):
    """timm's Cadene SENet implementation, used by the legacy_* registrations."""

    def __init__(
        self,
        block,
        layers,
        groups,
        reduction,
        num_classes=1000,
        in_chans=3,
        inplanes=64,
        input_3x3=False,
        downsample_kernel_size=1,
        downsample_padding=0,
        global_pool="avg",
        drop_rate=0.2,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.inplanes = inplanes
        if input_3x3:
            self.stem = nnx.List(
                [
                    _conv_bn(in_chans, 64, 3, 2, 1, rngs=rngs),
                    _conv_bn(64, 64, 3, 1, 1, rngs=rngs),
                    _conv_bn(64, inplanes, 3, 1, 1, rngs=rngs),
                ]
            )
        else:
            self.stem = nnx.List([_conv_bn(in_chans, inplanes, 7, 2, 3, rngs=rngs)])
        self.layer1, self.inplanes = self._make_layer(
            block, 64, layers[0], groups, reduction, 1, 1, 0, rngs=rngs
        )
        self.layer2, self.inplanes = self._make_layer(
            block,
            128,
            layers[1],
            groups,
            reduction,
            2,
            downsample_kernel_size,
            downsample_padding,
            rngs=rngs,
        )
        self.layer3, self.inplanes = self._make_layer(
            block,
            256,
            layers[2],
            groups,
            reduction,
            2,
            downsample_kernel_size,
            downsample_padding,
            rngs=rngs,
        )
        self.layer4, self.inplanes = self._make_layer(
            block,
            512,
            layers[3],
            groups,
            reduction,
            2,
            downsample_kernel_size,
            downsample_padding,
            rngs=rngs,
        )
        self.num_features = 512 * block.expansion
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def _make_layer(
        self, block, planes, blocks, groups, reduction, stride, down_kernel, down_padding, *, rngs
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = _conv_bn(
                self.inplanes,
                planes * block.expansion,
                down_kernel,
                stride,
                down_padding,
                rngs=rngs,
            )
        layers = [block(self.inplanes, planes, groups, reduction, stride, downsample, rngs=rngs)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction, rngs=rngs))
        return nnx.List(layers), self.inplanes

    def forward_features(self, x):
        for conv_bn in self.stem:
            x = nnx.relu(conv_bn(x))
        # timm pool0: MaxPool2d(3, stride=2, ceil_mode=True), no padding.
        x = nnx.max_pool(x, (3, 3), strides=(2, 2), padding="VALID")
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            for blk in layer:
                x = blk(x)
        return x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


def _legacy_senet(cfg_name, block, layers, groups, reduction, **kwargs):
    model = SENet(block, layers, groups, reduction, **kwargs)
    model.default_cfg = _cfg(
        crop_pct=0.875,
        interpolation="bilinear"
        if cfg_name not in ("legacy_seresnet18", "legacy_seresnext26_32x4d")
        else "bicubic",
    )
    return model


@register_model
def legacy_seresnet18(**kwargs):
    return _legacy_senet("legacy_seresnet18", SEResNetBlock, (2, 2, 2, 2), 1, 16, **kwargs)


@register_model
def legacy_seresnet34(**kwargs):
    return _legacy_senet("legacy_seresnet34", SEResNetBlock, (3, 4, 6, 3), 1, 16, **kwargs)


@register_model
def legacy_seresnet50(**kwargs):
    return _legacy_senet("legacy_seresnet50", SEResNetBottleneck, (3, 4, 6, 3), 1, 16, **kwargs)


@register_model
def legacy_seresnet101(**kwargs):
    return _legacy_senet("legacy_seresnet101", SEResNetBottleneck, (3, 4, 23, 3), 1, 16, **kwargs)


@register_model
def legacy_seresnet152(**kwargs):
    return _legacy_senet("legacy_seresnet152", SEResNetBottleneck, (3, 8, 36, 3), 1, 16, **kwargs)


@register_model
def legacy_senet154(**kwargs):
    return _legacy_senet(
        "legacy_senet154",
        SEBottleneck,
        (3, 8, 36, 3),
        64,
        16,
        inplanes=128,
        input_3x3=True,
        downsample_kernel_size=3,
        downsample_padding=1,
        **kwargs,
    )


@register_model
def legacy_seresnext26_32x4d(**kwargs):
    return _legacy_senet(
        "legacy_seresnext26_32x4d", SEResNeXtBottleneck, (2, 2, 2, 2), 32, 16, **kwargs
    )


@register_model
def legacy_seresnext50_32x4d(**kwargs):
    return _legacy_senet(
        "legacy_seresnext50_32x4d", SEResNeXtBottleneck, (3, 4, 6, 3), 32, 16, **kwargs
    )


@register_model
def legacy_seresnext101_32x4d(**kwargs):
    return _legacy_senet(
        "legacy_seresnext101_32x4d", SEResNeXtBottleneck, (3, 4, 23, 3), 32, 16, **kwargs
    )
