"""LCNetV2 in flax nnx, NHWC. Mirrors timm.models.lcnet (v2 depthwise-sep blocks + SE)."""
import jax.numpy as jnp
from flax import nnx

from ..layers import ConvBNAct, SqueezeExcite, global_pool_nhwc, hswish
from ..registry import register_model, _cfg


class LCBlock(nnx.Module):
    """dw 3x3/5x5 + pw 1x1, optional SE; stride-2 variant doubles via two branches."""

    def __init__(self, in_chs, out_chs, kernel=5, stride=1, se=False, *, rngs):
        self.stride = stride
        self.dw = ConvBNAct(in_chs, in_chs, kernel, stride, groups=in_chs, act="hswish", rngs=rngs)
        self.se = SqueezeExcite(in_chs, rd_ratio=0.25, rngs=rngs) if se else None
        self.pw = ConvBNAct(in_chs, out_chs, 1, act="hswish", rngs=rngs)

    def __call__(self, x):
        y = self.dw(x)
        if self.se is not None:
            y = self.se(y)
        return self.pw(y)


# (kernel, out, se, stride, repeats)
LCNETV2_CFG = [
    (5, 64, 0, 1, 1), (5, 64, 0, 2, 1), (5, 128, 0, 1, 1), (5, 128, 0, 2, 1),
    (5, 256, 0, 1, 2), (5, 256, 0, 2, 1),
    (5, 512, 0, 1, 1), (5, 512, 1, 1, 1), (5, 512, 0, 1, 1), (5, 512, 1, 1, 1),
    (5, 512, 0, 2, 1),
    (5, 512, 0, 1, 1), (5, 512, 1, 1, 1), (5, 512, 0, 1, 1), (5, 512, 1, 1, 1),
]


class LCNetV2(nnx.Module):
    default_cfg: dict = {}

    def __init__(self, width_mult=1.0, num_classes=1000, in_chans=3, global_pool="avg",
                 drop_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        stem = max(int(32 * width_mult), 8)
        self.conv1 = ConvBNAct(in_chans, stem, 3, 2, act="hswish", rngs=rngs)
        blocks, chs = [], stem
        for k, c, se, s, n in LCNETV2_CFG:
            out = max(int(c * width_mult), 8)
            for j in range(n):
                blocks.append(LCBlock(chs, out, k, s if j == 0 else 1, bool(se), rngs=rngs))
                chs = out
        self.blocks = nnx.List(blocks)
        head = max(int(1024 * width_mult), 8)
        self.conv_head = ConvBNAct(chs, head, 1, act="hswish", rngs=rngs)
        self.num_features = head
        self.head_drop = nnx.Dropout(drop_rate)
        self.fc = nnx.Linear(head, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.conv1(x)
        for blk in self.blocks:
            x = blk(x)
        return self.conv_head(x)

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
def lcnetv2_050(**kwargs):
    model = LCNetV2(0.5, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def lcnetv2_100(**kwargs):
    model = LCNetV2(1.0, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def lcnetv2_150(**kwargs):
    model = LCNetV2(1.5, **kwargs)
    model.default_cfg = _cfg()
    return model
