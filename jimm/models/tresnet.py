"""TResNet in flax nnx, NHWC. Mirrors timm.models.tresnet (SpaceToDepth stem + SE bottlenecks)."""
import jax.numpy as jnp
from flax import nnx

from ..layers import ConvBNAct, SqueezeExcite, global_pool_nhwc
from ..registry import register_model, _cfg


def space_to_depth(x, block_size=2):
    """(B,H,W,C) -> (B,H/2,W/2,4C)."""
    b, h, w, c = x.shape
    x = x.reshape(b, h // block_size, block_size, w // block_size, block_size, c)
    return x.transpose(0, 1, 3, 2, 4, 5).reshape(b, h // block_size, w // block_size, -1)


class TResNetBasic(nnx.Module):
    def __init__(self, in_chs, out_chs, stride, *, rngs):
        self.conv1 = ConvBNAct(in_chs, out_chs, 3, stride, rngs=rngs)
        self.conv2 = ConvBNAct(out_chs, out_chs, 3, act="identity", rngs=rngs)
        self.shortcut = ConvBNAct(in_chs, out_chs, 1, stride, act="identity", rngs=rngs) \
            if (stride != 1 or in_chs != out_chs) else None

    def __call__(self, x):
        y = self.conv2(self.conv1(x))
        sc = x if self.shortcut is None else self.shortcut(x)
        return nnx.relu(y + sc)


class TResNetBottleneck(nnx.Module):
    expansion = 4

    def __init__(self, in_chs, chs, stride, *, rngs):
        out_chs = chs * self.expansion
        self.conv1 = ConvBNAct(in_chs, chs, 1, rngs=rngs)
        self.conv2 = ConvBNAct(chs, chs, 3, stride, rngs=rngs)
        self.conv3 = ConvBNAct(chs, out_chs, 1, act="identity", rngs=rngs)
        self.se = SqueezeExcite(out_chs, rngs=rngs, rd_ratio=0.0625)
        self.shortcut = ConvBNAct(in_chs, out_chs, 1, stride, act="identity", rngs=rngs) \
            if (stride != 1 or in_chs != out_chs) else None

    def __call__(self, x):
        y = self.conv3(self.conv2(self.conv1(x)))
        y = self.se(y)
        sc = x if self.shortcut is None else self.shortcut(x)
        return nnx.relu(y + sc)


class TResNet(nnx.Module):
    default_cfg: dict = {}

    def __init__(self, layers, widths=(64, 128, 256, 512), num_classes=1000, in_chans=3,
                 global_pool="avg", drop_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        # SpaceToDepth stem: 224 -> 112 with 12 chs, then 1x1 conv to 64
        self.stem_s2d = lambda x: space_to_depth(x, 2)
        self.stem_conv = ConvBNAct(in_chans * 4, 64, 1, rngs=rngs)
        self.num_features = widths[-1] * TResNetBottleneck.expansion
        chs, stages = 64, []
        for i, (n, w) in enumerate(zip(layers, widths)):
            block = TResNetBasic if i == 0 else TResNetBottleneck
            stride = 1 if i == 0 else 2
            blocks = []
            for j in range(n):
                blocks.append(block(chs, w, stride if j == 0 else 1, rngs=rngs))
                chs = w if i == 0 else w * TResNetBottleneck.expansion
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.head_drop = nnx.Dropout(drop_rate)
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.stem_conv(self.stem_s2d(x))
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


def _tresnet(layers, **kwargs):
    model = TResNet(layers, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def tresnet_m(**kwargs):
    return _tresnet([3, 4, 11, 3], **kwargs)


@register_model
def tresnet_l(**kwargs):
    return _tresnet([4, 5, 18, 3], **kwargs)


@register_model
def tresnet_xl(**kwargs):
    return _tresnet([4, 5, 24, 3], **kwargs)
