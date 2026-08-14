"""VGG in flax nnx, NHWC. Mirrors timm.models.vgg (head: 7x7 pool -> MLP classifier)."""
from flax import nnx

from ..layers import ClassifierMixin
from ..registry import register_model, _cfg


class ConvBlock(nnx.Module):
    def __init__(self, in_chs, chs, use_bn, *, rngs):
        self.conv = nnx.Conv(in_chs, chs, (3, 3), use_bias=not use_bn, rngs=rngs)
        self.bn = nnx.BatchNorm(chs, rngs=rngs) if use_bn else None

    def __call__(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return nnx.relu(x)


class VGG(ClassifierMixin, nnx.Module):
    default_cfg: dict = {}

    def __init__(self, cfg, use_bn=True, num_classes=1000, in_chans=3, drop_rate=0.0, *, rngs):
        self.num_classes = num_classes
        blocks, chs = [], in_chans
        for v in cfg:
            if v == "M":
                blocks.append("M")
            else:
                blocks.append(ConvBlock(chs, v, use_bn, rngs=rngs))
                chs = v
        self.features = nnx.List(blocks)
        self.num_features = 4096
        self.pre_fc = nnx.Linear(512 * 7 * 7, 4096, rngs=rngs)  # assumes 224 input (5 maxpools)
        self.fc2 = nnx.Linear(4096, 4096, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate)
        self.fc = nnx.Linear(4096, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        for b in self.features:
            x = nnx.max_pool(x, (2, 2), strides=(2, 2)) if b == "M" else b(x)
        return x

    def forward_head(self, x):
        x = x.reshape(x.shape[0], -1)
        x = nnx.relu(self.pre_fc(x))
        x = self.head_drop(x)
        x = nnx.relu(self.fc2(x))
        x = self.head_drop(x)
        return self.fc(x) if self.fc is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_CFGS = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(name, use_bn, **kwargs):
    model = VGG(_CFGS[name], use_bn, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vgg11_bn(**kwargs):
    return _vgg("vgg11", True, **kwargs)


@register_model
def vgg13_bn(**kwargs):
    return _vgg("vgg13", True, **kwargs)


@register_model
def vgg16_bn(**kwargs):
    return _vgg("vgg16", True, **kwargs)


@register_model
def vgg19_bn(**kwargs):
    return _vgg("vgg19", True, **kwargs)
