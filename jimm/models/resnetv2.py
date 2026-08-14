"""Pre-activation ResNet (ResNetV2) in flax nnx, NHWC. Mirrors timm.models.resnetv2."""
from flax import nnx

from ..layers import global_pool_nhwc
from ..registry import register_model, _cfg


class PreActBottleneck(nnx.Module):
    expansion = 4

    def __init__(self, in_chs, chs, stride=1, *, rngs):
        out_chs = chs * self.expansion
        self.bn1 = nnx.BatchNorm(in_chs, rngs=rngs)
        self.conv1 = nnx.Conv(in_chs, chs, (1, 1), use_bias=False, rngs=rngs)
        self.bn2 = nnx.BatchNorm(chs, rngs=rngs)
        self.conv2 = nnx.Conv(chs, chs, (3, 3), strides=(stride, stride), use_bias=False, rngs=rngs)
        self.bn3 = nnx.BatchNorm(chs, rngs=rngs)
        self.conv3 = nnx.Conv(chs, out_chs, (1, 1), use_bias=False, rngs=rngs)
        self.short_conv = nnx.Conv(in_chs, out_chs, (1, 1), strides=(stride, stride),
                                   use_bias=False, rngs=rngs) if (stride != 1 or in_chs != out_chs) else None

    def __call__(self, x):
        x_preact = nnx.relu(self.bn1(x))
        sc = x_preact if self.short_conv is None else self.short_conv(x_preact)
        y = self.conv1(x_preact)
        y = self.conv2(nnx.relu(self.bn2(y)))
        y = self.conv3(nnx.relu(self.bn3(y)))
        return y + sc


class ResNetV2(nnx.Module):
    default_cfg: dict = {}

    def __init__(self, layers, num_classes=1000, in_chans=3, global_pool="avg",
                 drop_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = 512 * PreActBottleneck.expansion
        self.conv1 = nnx.Conv(in_chans, 64, (7, 7), strides=(2, 2), padding=[(3, 3), (3, 3)],
                              use_bias=False, rngs=rngs)
        chs, stages = 64, []
        for i, (n, stride) in enumerate(zip(layers, [1, 2, 2, 2])):
            width = 64 * 2**i
            blocks = []
            for j in range(n):
                blocks.append(PreActBottleneck(chs, width, stride if j == 0 else 1, rngs=rngs))
                chs = width * PreActBottleneck.expansion
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.norm = nnx.BatchNorm(self.num_features, rngs=rngs)  # final pre-act norm
        self.head_drop = nnx.Dropout(drop_rate)
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = nnx.max_pool(self.conv1(x), (3, 3), strides=(2, 2), padding="SAME")
        for stage in self.stages:
            for blk in stage:
                x = blk(x)
        return nnx.relu(self.norm(x))  # final pre-activation

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


def _resnetv2(layers, **kwargs):
    model = ResNetV2(layers, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def resnetv2_50(**kwargs):
    return _resnetv2([3, 4, 6, 3], **kwargs)


@register_model
def resnetv2_101(**kwargs):
    return _resnetv2([3, 4, 23, 3], **kwargs)


@register_model
def resnetv2_152(**kwargs):
    return _resnetv2([3, 8, 36, 3], **kwargs)
