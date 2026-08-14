"""MobileNetV3 in flax nnx, NHWC. Mirrors timm.models.mobilenetv3 / torchvision."""
from flax import nnx

from ..layers import SqueezeExcite, global_pool_nhwc, hswish, ClassifierMixin
from ..registry import register_model, _cfg
from .mobilenetv2 import ConvBN, relu6

_ACT = {"relu": relu6, "hswish": hswish}

class InvertedResidualV3(nnx.Module):
    def __init__(self, in_chs, out_chs, kernel, stride, expand, se, act, *, rngs):
        mid = int(in_chs * expand)
        self.use_residual = stride == 1 and in_chs == out_chs
        self.act = _ACT[act]
        self.expand = ConvBN(in_chs, mid, rngs=rngs) if mid != in_chs else None
        self.dw = nnx.Conv(mid, mid, (kernel, kernel), strides=(stride, stride), use_bias=False,
                           feature_group_count=mid, rngs=rngs)
        self.bn1 = nnx.BatchNorm(mid, rngs=rngs)
        self.se = SqueezeExcite(mid, 0.25, rngs=rngs) if se else None
        self.pw = nnx.Conv(mid, out_chs, (1, 1), use_bias=False, rngs=rngs)
        self.bn2 = nnx.BatchNorm(out_chs, rngs=rngs)

    def __call__(self, x):
        y = x if self.expand is None else self.act(self.expand(x))
        y = self.act(self.bn1(self.dw(y)))
        if self.se is not None:
            y = self.se(y)
        y = self.bn2(self.pw(y))
        return x + y if self.use_residual else y

# (kernel, expand, in, out, se, act, stride)
LARGE_CFG = [
    (3, 1, 16, 16, 0, "relu", 1), (3, 4, 16, 24, 0, "relu", 2), (3, 3, 24, 24, 0, "relu", 1),
    (5, 3, 24, 40, 1, "relu", 2), (5, 3, 40, 40, 1, "relu", 1), (5, 3, 40, 40, 1, "relu", 1),
    (3, 6, 40, 80, 0, "hswish", 2), (3, 2.5, 80, 80, 0, "hswish", 1), (3, 2.3, 80, 80, 0, "hswish", 1),
    (3, 2.3, 80, 80, 0, "hswish", 1), (3, 6, 80, 112, 1, "hswish", 1), (3, 6, 112, 112, 1, "hswish", 1),
    (5, 6, 112, 160, 1, "hswish", 2), (5, 6, 160, 160, 1, "hswish", 1), (5, 6, 160, 160, 1, "hswish", 1),
]
SMALL_CFG = [
    (3, 1, 16, 16, 1, "relu", 2), (3, 4.5, 16, 24, 0, "relu", 2), (3, 3.67, 24, 24, 0, "relu", 1),
    (5, 4, 24, 40, 1, "hswish", 2), (5, 6, 40, 40, 1, "hswish", 1), (5, 6, 40, 40, 1, "hswish", 1),
    (5, 3, 40, 48, 1, "hswish", 1), (5, 3, 48, 48, 1, "hswish", 1),
    (5, 6, 48, 96, 1, "hswish", 2), (5, 6, 96, 96, 1, "hswish", 1), (5, 6, 96, 96, 1, "hswish", 1),
]

class MobileNetV3(ClassifierMixin, nnx.Module):
    default_cfg: dict = {}

    def __init__(self, cfg, head_chs, head_mid, num_classes=1000, in_chans=3,
                 global_pool="avg", drop_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.conv1 = nnx.Conv(in_chans, 16, (3, 3), strides=(2, 2), use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(16, rngs=rngs)
        self.blocks = nnx.List([InvertedResidualV3(i, o, k, s, e, se, a, rngs=rngs)
                                for k, e, i, o, se, a, s in cfg])
        self.conv_head = nnx.Conv(cfg[-1][3], head_chs, (1, 1), use_bias=False, rngs=rngs)
        self.bn_head = nnx.BatchNorm(head_chs, rngs=rngs)
        self.num_features = head_mid
        self.fc1 = nnx.Linear(head_chs, head_mid, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate)
        self.fc = nnx.Linear(head_mid, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = hswish(self.bn1(self.conv1(x)))
        for blk in self.blocks:
            x = blk(x)
        return hswish(self.bn_head(self.conv_head(x)))

    def forward_head(self, x):
        x = global_pool_nhwc(x, self.global_pool)
        x = hswish(self.fc1(x))
        x = self.head_drop(x)
        return self.fc(x) if self.fc is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

@register_model
def mobilenetv3_large_100(**kwargs):
    model = MobileNetV3(LARGE_CFG, 960, 1280, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def mobilenetv3_small_100(**kwargs):
    model = MobileNetV3(SMALL_CFG, 576, 1024, **kwargs)
    model.default_cfg = _cfg()
    return model
