"""Aligned Xception (wider, GELU-ish variant) in flax nnx, NHWC. Mirrors timm.models.xception_aligned."""
from flax import nnx

from ..layers import global_pool_nhwc
from ..registry import register_model, _cfg
from .xception import SeparableConv, XceptionBlock


class XceptionAligned(nnx.Module):
    default_cfg: dict = {}

    def __init__(self, num_classes=1000, in_chans=3, global_pool="avg", drop_rate=0.0,
                 widths=(64, 128, 256, 728, 1024, 1536, 2048), middle_blocks=8, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.conv1 = nnx.Conv(in_chans, widths[0], (3, 3), strides=(2, 2), use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(widths[0], rngs=rngs)
        self.conv2 = nnx.Conv(widths[0], widths[1], (3, 3), use_bias=False, rngs=rngs)
        self.bn2 = nnx.BatchNorm(widths[1], rngs=rngs)
        self.block1 = XceptionBlock(widths[1], widths[2], 2, stride=2, rngs=rngs)
        self.block2 = XceptionBlock(widths[2], widths[3], 2, stride=2, rngs=rngs)
        self.block3 = XceptionBlock(widths[3], widths[3], 2, stride=2, rngs=rngs)
        self.middle = nnx.List([XceptionBlock(widths[3], widths[3], 3, rngs=rngs)
                                for _ in range(middle_blocks)])
        self.block12 = XceptionBlock(widths[3], widths[4], 2, stride=2, rngs=rngs)
        self.conv3 = SeparableConv(widths[4], widths[5], rngs=rngs)
        self.conv4 = SeparableConv(widths[5], widths[6], rngs=rngs)
        self.num_features = widths[6]
        self.head_drop = nnx.Dropout(drop_rate)
        self.fc = nnx.Linear(widths[6], num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = nnx.relu(self.bn1(self.conv1(x)))
        x = nnx.relu(self.bn2(self.conv2(x)))
        x = self.block3(self.block2(self.block1(x)))
        for blk in self.middle:
            x = blk(x)
        x = self.block12(x)
        x = nnx.relu(self.conv3(x))
        return nnx.relu(self.conv4(x))

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


@register_model
def xception41(**kwargs):
    model = XceptionAligned(widths=(32, 64, 128, 256, 728, 1536, 2048), middle_blocks=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 299, 299))
    return model


@register_model
def xception65(**kwargs):
    model = XceptionAligned(**kwargs)
    model.default_cfg = _cfg(input_size=(3, 299, 299))
    return model
