"""MNASNet-B1 / Single-Path NASNet in flax nnx, NHWC. Mirrors timm's EfficientNet builder.

Blocks use ReLU; channels round to multiples of 8 and the 1280-channel head is not
scaled by the width multiplier.
"""

from flax import nnx

from ..layers import BatchNorm, ClassifierMixin, make_divisible
from ..registry import _cfg, register_model


class SepConv(nnx.Module):
    """3x3 depthwise -> 1x1 pointwise (mnasnet stage-0 block)."""

    def __init__(self, in_chs, out_chs, kernel=3, stride=1, *, rngs):
        self.dw = nnx.Conv(
            in_chs,
            in_chs,
            (kernel, kernel),
            strides=(stride, stride),
            padding=kernel // 2,
            use_bias=False,
            feature_group_count=in_chs,
            rngs=rngs,
        )
        self.bn1 = BatchNorm(in_chs, rngs=rngs)
        self.pw = nnx.Conv(in_chs, out_chs, (1, 1), use_bias=False, rngs=rngs)
        self.bn2 = BatchNorm(out_chs, rngs=rngs)

    def __call__(self, x):
        return self.bn2(self.pw(nnx.relu(self.bn1(self.dw(x)))))


class MBBlock(nnx.Module):
    def __init__(self, in_chs, out_chs, kernel, stride, expand, *, rngs):
        mid = make_divisible(in_chs * expand)
        self.use_residual = stride == 1 and in_chs == out_chs
        self.expand = nnx.Conv(in_chs, mid, (1, 1), use_bias=False, rngs=rngs)
        self.bn0 = BatchNorm(mid, rngs=rngs)
        self.dw = nnx.Conv(
            mid,
            mid,
            (kernel, kernel),
            strides=(stride, stride),
            padding=kernel // 2,
            use_bias=False,
            feature_group_count=mid,
            rngs=rngs,
        )
        self.bn1 = BatchNorm(mid, rngs=rngs)
        self.pw = nnx.Conv(mid, out_chs, (1, 1), use_bias=False, rngs=rngs)
        self.bn2 = BatchNorm(out_chs, rngs=rngs)

    def __call__(self, x):
        y = nnx.relu(self.bn0(self.expand(x)))
        y = nnx.relu(self.bn1(self.dw(y)))
        y = self.bn2(self.pw(y))
        return x + y if self.use_residual else y


# (type, kernel, stride, expand, out, repeats) rows from timm's arch definitions.
MNASNET_CFG = [  # mnasnet_b1
    ("sep", 3, 1, 1, 16, 1),
    ("mb", 3, 2, 3, 24, 3),
    ("mb", 5, 2, 3, 40, 3),
    ("mb", 5, 2, 6, 80, 3),
    ("mb", 3, 1, 6, 96, 2),
    ("mb", 5, 2, 6, 192, 4),
    ("mb", 3, 1, 6, 320, 1),
]
SPNAS_CFG = [
    ("sep", 3, 1, 1, 16, 1),
    ("mb", 3, 2, 3, 24, 3),
    ("mb", 5, 2, 6, 40, 1),
    ("mb", 3, 1, 3, 40, 3),
    ("mb", 5, 2, 6, 80, 1),
    ("mb", 3, 1, 3, 80, 3),
    ("mb", 5, 1, 6, 96, 1),
    ("mb", 5, 1, 3, 96, 3),
    ("mb", 5, 2, 6, 192, 4),
    ("mb", 3, 1, 6, 320, 1),
]


class MNASNet(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        cfg=MNASNET_CFG,
        width_mult=1.0,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        stem = make_divisible(32 * width_mult)
        self.conv1 = nnx.Conv(
            in_chans, stem, (3, 3), strides=(2, 2), padding=1, use_bias=False, rngs=rngs
        )
        self.bn1 = BatchNorm(stem, rngs=rngs)
        blocks, chs = [], stem
        for kind, k, s, e, out, n in cfg:
            out = make_divisible(out * width_mult)
            for j in range(n):
                if kind == "sep":
                    blocks.append(SepConv(chs, out, k, s if j == 0 else 1, rngs=rngs))
                else:
                    blocks.append(MBBlock(chs, out, k, s if j == 0 else 1, e, rngs=rngs))
                chs = out
        self.blocks = nnx.List(blocks)
        head = 1280
        self.conv_head = nnx.Conv(chs, head, (1, 1), use_bias=False, rngs=rngs)
        self.bn_head = BatchNorm(head, rngs=rngs)
        self.num_features = head
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(head, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = nnx.relu(self.bn1(self.conv1(x)))
        for blk in self.blocks:
            x = blk(x)
        return nnx.relu(self.bn_head(self.conv_head(x)))

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


def _mnasnet(cfg, width_mult, **kwargs):
    model = MNASNet(cfg, width_mult, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def mnasnet_050(**kwargs):
    return _mnasnet(MNASNET_CFG, 0.5, **kwargs)


@register_model
def mnasnet_100(**kwargs):
    return _mnasnet(MNASNET_CFG, 1.0, **kwargs)


@register_model
def mnasnet_140(**kwargs):
    return _mnasnet(MNASNET_CFG, 1.4, **kwargs)


@register_model
def spnasnet_100(**kwargs):
    return _mnasnet(SPNAS_CFG, 1.0, **kwargs)
