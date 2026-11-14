"""Xception in flax nnx, NHWC. Mirrors timm.models.xception (aligned with the keras ref impl)."""

from flax import nnx

from ..layers import BatchNorm, ClassifierMixin
from ..registry import _cfg, register_model


class SeparableConv(nnx.Module):
    """3x3 depthwise -> 1x1 pointwise; BatchNorm is applied by the enclosing block."""

    def __init__(self, in_chs, out_chs, kernel=3, stride=1, padding=1, *, rngs):
        self.dw = nnx.Conv(
            in_chs,
            in_chs,
            (kernel, kernel),
            strides=(stride, stride),
            padding=((padding, padding),) * 2,
            use_bias=False,
            feature_group_count=in_chs,
            rngs=rngs,
        )
        self.pw = nnx.Conv(in_chs, out_chs, (1, 1), use_bias=False, rngs=rngs)

    def __call__(self, x):
        return self.pw(self.dw(x))


class XceptionBlock(nnx.Module):
    """relu-sep-bn x repeats, optional maxpool stride 2, residual shortcut."""

    def __init__(
        self,
        in_chs,
        out_chs,
        repeats,
        stride=1,
        start_with_relu=True,
        grow_first=True,
        *,
        rngs,
    ):
        self.start_with_relu = start_with_relu
        self.pool = stride != 1
        self.shortcut = (
            nnx.Sequential(
                nnx.Conv(
                    in_chs, out_chs, (1, 1), strides=(stride, stride), use_bias=False, rngs=rngs
                ),
                BatchNorm(out_chs, rngs=rngs),
            )
            if (stride != 1 or in_chs != out_chs)
            else None
        )
        convs, norms = [], []
        for i in range(repeats):
            if grow_first:
                inc = in_chs if i == 0 else out_chs
                outc = out_chs
            else:
                inc = in_chs
                outc = in_chs if i < repeats - 1 else out_chs
            convs.append(SeparableConv(inc, outc, rngs=rngs))
            norms.append(BatchNorm(outc, rngs=rngs))
        self.convs = nnx.List(convs)
        self.norms = nnx.List(norms)

    def __call__(self, x):
        y = x
        for i, (sep, norm) in enumerate(zip(self.convs, self.norms)):
            if i > 0 or self.start_with_relu:
                y = nnx.relu(y)
            y = norm(sep(y))
        if self.pool:
            y = nnx.max_pool(y, (3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))
        sc = x if self.shortcut is None else self.shortcut(x)
        return y + sc


class Xception(ClassifierMixin, nnx.Module):
    def __init__(self, num_classes=1000, in_chans=3, global_pool="avg", drop_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.conv1 = nnx.Conv(
            in_chans, 32, (3, 3), strides=(2, 2), use_bias=False, padding="VALID", rngs=rngs
        )
        self.bn1 = BatchNorm(32, rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, (3, 3), use_bias=False, padding="VALID", rngs=rngs)
        self.bn2 = BatchNorm(64, rngs=rngs)
        self.block1 = XceptionBlock(64, 128, 2, stride=2, start_with_relu=False, rngs=rngs)
        self.block2 = XceptionBlock(128, 256, 2, stride=2, rngs=rngs)
        self.block3 = XceptionBlock(256, 728, 2, stride=2, rngs=rngs)
        self.middle = nnx.List([XceptionBlock(728, 728, 3, rngs=rngs) for _ in range(8)])
        self.block12 = XceptionBlock(728, 1024, 2, stride=2, grow_first=False, rngs=rngs)
        self.conv3 = SeparableConv(1024, 1536, rngs=rngs)
        self.bn3 = BatchNorm(1536, rngs=rngs)
        self.conv4 = SeparableConv(1536, 2048, rngs=rngs)
        self.bn4 = BatchNorm(2048, rngs=rngs)
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
        x = nnx.relu(self.bn3(self.conv3(x)))
        return nnx.relu(self.bn4(self.conv4(x)))

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


@register_model
def xception(**kwargs):
    model = Xception(**kwargs)
    model.default_cfg = _cfg(input_size=(3, 299, 299))
    return model


@register_model
def legacy_xception(**kwargs):
    model = Xception(**kwargs)
    model.default_cfg = _cfg(
        input_size=(3, 299, 299),
        crop_pct=0.8975,
        interpolation="bicubic",
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )
    return model
