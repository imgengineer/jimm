"""ResNet / ResNeXt / SE-ResNet in flax nnx, NHWC. Mirrors timm.models.resnet."""
import jax.numpy as jnp
from flax import nnx

from ..layers import DropPath, SqueezeExcite, global_pool_nhwc
from ..registry import register_model, _cfg


class Downsample(nnx.Module):
    def __init__(self, in_chs, out_chs, stride, *, rngs):
        self.conv = nnx.Conv(in_chs, out_chs, kernel_size=(1, 1), strides=(stride, stride),
                             use_bias=False, rngs=rngs)
        self.bn = nnx.BatchNorm(out_chs, rngs=rngs)

    def __call__(self, x):
        return self.bn(self.conv(x))


class BasicBlock(nnx.Module):
    expansion = 1

    def __init__(self, in_chs, chs, stride=1, drop_path_rate=0.0, se=False,
                 groups=1, base_width=64, *, rngs):  # groups/base_width unused, kept for uniform block signature
        out_chs = chs * self.expansion
        self.conv1 = nnx.Conv(in_chs, chs, (3, 3), strides=(stride, stride), use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(chs, rngs=rngs)
        self.conv2 = nnx.Conv(chs, out_chs, (3, 3), use_bias=False, rngs=rngs)
        self.bn2 = nnx.BatchNorm(out_chs, rngs=rngs)
        self.se = SqueezeExcite(out_chs, rngs=rngs) if se else None
        self.shortcut = Downsample(in_chs, out_chs, stride, rngs=rngs) if (stride != 1 or in_chs != out_chs) else None
        self.drop_path = DropPath(drop_path_rate)

    def __call__(self, x):
        y = nnx.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.se is not None:
            y = self.se(y)
        sc = x if self.shortcut is None else self.shortcut(x)
        return nnx.relu(y + self.drop_path(sc))


class Bottleneck(nnx.Module):
    expansion = 4

    def __init__(self, in_chs, chs, stride=1, drop_path_rate=0.0, se=False,
                 groups=1, base_width=64, *, rngs):
        out_chs = chs * self.expansion
        mid = chs * base_width * groups // 64
        self.conv1 = nnx.Conv(in_chs, mid, (1, 1), use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(mid, rngs=rngs)
        self.conv2 = nnx.Conv(mid, mid, (3, 3), strides=(stride, stride), use_bias=False,
                              feature_group_count=groups, rngs=rngs)
        self.bn2 = nnx.BatchNorm(mid, rngs=rngs)
        self.conv3 = nnx.Conv(mid, out_chs, (1, 1), use_bias=False, rngs=rngs)
        self.bn3 = nnx.BatchNorm(out_chs, rngs=rngs)
        self.se = SqueezeExcite(out_chs, rngs=rngs) if se else None
        self.shortcut = Downsample(in_chs, out_chs, stride, rngs=rngs) if (stride != 1 or in_chs != out_chs) else None
        self.drop_path = DropPath(drop_path_rate)

    def __call__(self, x):
        y = nnx.relu(self.bn1(self.conv1(x)))
        y = nnx.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        if self.se is not None:
            y = self.se(y)
        sc = x if self.shortcut is None else self.shortcut(x)
        return nnx.relu(y + self.drop_path(sc))


class ResNet(nnx.Module):
    default_cfg: dict = {}

    def __init__(self, block: type[BasicBlock] | type[Bottleneck], layers, num_classes=1000, in_chans=3, global_pool="avg",
                 drop_rate=0.0, drop_path_rate=0.0, se=False, groups=1, base_width=64, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = 512 * block.expansion
        self.conv1 = nnx.Conv(in_chans, 64, (7, 7), strides=(2, 2), padding=[(3, 3), (3, 3)],
                              use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(64, rngs=rngs)
        dpr = [drop_path_rate * i / max(sum(layers) - 1, 1) for i in range(sum(layers))]
        chs, stages, k = 64, [], 0
        for i, (n, stride) in enumerate(zip(layers, [1, 2, 2, 2])):
            width = 64 * 2**i
            blocks = []
            for j in range(n):
                blocks.append(block(chs, width, stride if j == 0 else 1, dpr[k],
                                    se=se, groups=groups, base_width=base_width, rngs=rngs))
                chs = width * block.expansion
                k += 1
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.head_drop = nnx.Dropout(drop_rate)
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = nnx.max_pool(nnx.relu(self.bn1(self.conv1(x))), (3, 3), strides=(2, 2), padding="SAME")
        for stage in self.stages:
            for blk in stage:
                x = blk(x)
        return x

    def forward_head(self, x):
        x = global_pool_nhwc(x, self.global_pool)
        x = self.head_drop(x)
        return self.fc(x) if self.fc is not None else x

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool="avg"):
        self.num_classes, self.global_pool = num_classes, global_pool
        if num_classes > 0 and self.fc is None:
            raise RuntimeError("cannot re-add classifier to a num_classes=0 model")
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=nnx.Rngs(0)) if num_classes > 0 else None

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def resnet18(**kwargs):
    return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


@register_model
def resnet34(**kwargs):
    return _resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


@register_model
def resnet50(**kwargs):
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


@register_model
def resnet101(**kwargs):
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


@register_model
def resnet152(**kwargs):
    return _resnet(Bottleneck, [3, 8, 36, 3], **kwargs)


@register_model
def resnext50_32x4d(**kwargs):
    return _resnet(Bottleneck, [3, 4, 6, 3], groups=32, base_width=4, **kwargs)


@register_model
def resnext101_32x8d(**kwargs):
    return _resnet(Bottleneck, [3, 4, 23, 3], groups=32, base_width=8, **kwargs)


@register_model
def seresnet50(**kwargs):
    return _resnet(Bottleneck, [3, 4, 6, 3], se=True, **kwargs)


@register_model
def seresnext50_32x4d(**kwargs):
    return _resnet(Bottleneck, [3, 4, 6, 3], se=True, groups=32, base_width=4, **kwargs)
