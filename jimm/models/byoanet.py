"""ByoaNet in flax nnx, NHWC. Mirrors timm.models.byoanet (RegNet-ish with varying blocks)."""
import jax.numpy as jnp
from flax import nnx

from ..layers import ConvBNAct, SqueezeExcite, global_pool_nhwc
from ..registry import register_model, _cfg


class ByoaBlock(nnx.Module):
    """1x1 -> dw 3x3 -> SE -> 1x1 bottleneck (byoanet default block)."""

    def __init__(self, in_chs, out_chs, stride, expand=1.0, se=True, groups=1, *, rngs):
        mid = int(out_chs * expand)
        self.conv1 = ConvBNAct(in_chs, mid, 1, act="silu", rngs=rngs)
        self.dw = ConvBNAct(mid, mid, 3, stride, groups=groups, act="silu", rngs=rngs)
        self.se = SqueezeExcite(mid, rd_ratio=0.25, rngs=rngs) if se else None
        self.pw = ConvBNAct(mid, out_chs, 1, act="identity", rngs=rngs)
        self.shortcut = ConvBNAct(in_chs, out_chs, 1, stride, act="identity", rngs=rngs) \
            if (stride != 1 or in_chs != out_chs) else None

    def __call__(self, x):
        y = self.dw(self.conv1(x))
        if self.se is not None:
            y = self.se(y)
        y = self.pw(y)
        sc = x if self.shortcut is None else self.shortcut(x)
        return y + sc


_CFGS = {  # (widths, depths, expand, stride)
    "byoanet_pv1": ((24, 56, 152, 288, 608), (1, 1, 4, 7, 1), 1.0, (1, 2, 2, 2, 2)),
    "byoanet_pv2": ((32, 72, 192, 384, 768), (1, 1, 4, 8, 1), 1.0, (1, 2, 2, 2, 2)),
    "byoanet_s": ((24, 56, 160, 304, 608), (2, 3, 7, 2, 1), 1.0, (1, 2, 2, 2, 2)),
}


class ByoaNet(nnx.Module):
    default_cfg: dict = {}

    def __init__(self, widths, depths, expand, strides, num_classes=1000, in_chans=3,
                 global_pool="avg", drop_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.stem = ConvBNAct(in_chans, widths[0], 3, 2, act="silu", rngs=rngs)
        stages, chs = [], widths[0]
        for w, d, s in zip(widths[1:], depths, strides):
            blocks = []
            for j in range(d):
                blocks.append(ByoaBlock(chs, w, s if j == 0 else 1, expand, rngs=rngs))
                chs = w
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.num_features = widths[-1]
        self.head_drop = nnx.Dropout(drop_rate)
        self.fc = nnx.Linear(widths[-1], num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.stem(x)
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


def _make(name):
    widths, depths, expand, strides = _CFGS[name]

    def entry(**kwargs):
        model = ByoaNet(widths, depths, expand, strides, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
