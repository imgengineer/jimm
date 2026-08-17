"""Xception in flax nnx, NHWC. Mirrors timm.models.xception (aligned with the keras ref impl)."""
from flax import nnx

from ..layers import ClassifierMixin
from ..registry import register_model, _cfg

class SeparableConv(nnx.Module):
    """3x3 depthwise -> BN -> 1x1 pointwise -> BN (no activation in between)."""

    def __init__(self, in_chs, out_chs, stride=1, *, rngs):
        self.dw = nnx.Conv(in_chs, in_chs, (3, 3), strides=(stride, stride), use_bias=False,
                           feature_group_count=in_chs, rngs=rngs)
        self.bn_dw = nnx.BatchNorm(in_chs, rngs=rngs)
        self.pw = nnx.Conv(in_chs, out_chs, (1, 1), use_bias=False, rngs=rngs)
        self.bn_pw = nnx.BatchNorm(out_chs, rngs=rngs)

    def __call__(self, x):
        return self.bn_pw(self.pw(self.bn_dw(self.dw(x))))

class XceptionBlock(nnx.Module):
    """relu-sep x repeats, optional maxpool stride 2, residual shortcut."""

    def __init__(self, in_chs, out_chs, repeats, stride=1, *, rngs):
        self.reps = nnx.List([SeparableConv(in_chs if i == 0 else out_chs, out_chs, rngs=rngs)
                              for i in range(repeats)])
        self.pool = stride == 2
        self.shortcut = nnx.Sequential(
            nnx.Conv(in_chs, out_chs, (1, 1), strides=(stride, stride), use_bias=False, rngs=rngs),
            nnx.BatchNorm(out_chs, rngs=rngs)) if (stride == 2 or in_chs != out_chs) else None

    def __call__(self, x):
        y = x
        for sep in self.reps:
            y = sep(nnx.relu(y))
        if self.pool:
            y = nnx.max_pool(y, (3, 3), strides=(2, 2), padding="SAME")
        sc = x if self.shortcut is None else self.shortcut(x)
        return y + sc

class Xception(ClassifierMixin, nnx.Module):
    default_cfg: dict | None = None

    def __init__(self, num_classes=1000, in_chans=3, global_pool="avg", drop_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.conv1 = nnx.Conv(in_chans, 32, (3, 3), strides=(2, 2), use_bias=False, padding="VALID", rngs=rngs)
        self.bn1 = nnx.BatchNorm(32, rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, (3, 3), use_bias=False, padding="VALID", rngs=rngs)
        self.bn2 = nnx.BatchNorm(64, rngs=rngs)
        self.block1 = XceptionBlock(64, 128, 2, stride=2, rngs=rngs)
        self.block2 = XceptionBlock(128, 256, 2, stride=2, rngs=rngs)
        self.block3 = XceptionBlock(256, 728, 2, stride=2, rngs=rngs)
        self.middle = nnx.List([XceptionBlock(728, 728, 3, rngs=rngs) for _ in range(8)])
        self.block12 = XceptionBlock(728, 1024, 2, stride=2, rngs=rngs)
        self.conv3 = SeparableConv(1024, 1536, rngs=rngs)
        self.conv4 = SeparableConv(1536, 2048, rngs=rngs)
        self.num_features = 2048
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(2048, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = nnx.relu(self.bn1(self.conv1(x)))
        x = nnx.relu(self.bn2(self.conv2(x)))
        x = self.block3(self.block2(self.block1(x)))
        for blk in self.middle:
            x = blk(x)
        x = self.block12(x)
        x = nnx.relu(self.conv3(x))
        return nnx.relu(self.conv4(x))

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

@register_model
def xception(**kwargs):
    model = Xception(**kwargs)
    model.default_cfg = _cfg(input_size=(3, 299, 299))
    return model
