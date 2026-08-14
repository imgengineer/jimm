"""ByobNet (BYO-Build-a-Network) in flax nnx, NHWC. Mirrors timm.models.byobnet."""
from flax import nnx

from ..layers import ConvBNAct, SqueezeExcite, ClassifierMixin
from ..registry import register_model, _cfg

class ByobBlock(nnx.Module):
    def __init__(self, in_chs, out_chs, stride, expand=1.0, se=True, groups=1, *, rngs):
        mid = int(out_chs * expand)
        self.conv1 = ConvBNAct(in_chs, mid, 1, act="silu", rngs=rngs)
        self.dw = ConvBNAct(mid, mid, 3, stride, groups=mid if groups > 1 else 1, act="silu", rngs=rngs)
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

_CFGS = {  # widths, depths, expand, strides
    "byobnet_s": ((24, 56, 160, 304, 608), (2, 3, 7, 2, 1), 1.0, (1, 2, 2, 2, 2)),
    "byobnet_m": ((32, 72, 192, 384, 768), (2, 3, 8, 3, 1), 1.0, (1, 2, 2, 2, 2)),
    "byobnet_l": ((40, 88, 232, 464, 928), (2, 3, 9, 3, 1), 1.0, (1, 2, 2, 2, 2)),
}

class ByobNet(ClassifierMixin, nnx.Module):
    default_cfg: dict = {}

    def __init__(self, widths, depths, expand, strides, num_classes=1000, in_chans=3,
                 global_pool="avg", drop_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.stem = nnx.List([
            ConvBNAct(in_chans, widths[0], 3, 2, act="silu", rngs=rngs),
            ConvBNAct(widths[0], widths[0], 3, 1, act="silu", rngs=rngs)])
        stages, chs = [], widths[0]
        for w, d, s in zip(widths[1:], depths, strides):
            blocks = []
            for j in range(d):
                blocks.append(ByobBlock(chs, w, s if j == 0 else 1, expand, rngs=rngs))
                chs = w
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.num_features = widths[-1]
        self.head_drop = nnx.Dropout(drop_rate)
        self.fc = nnx.Linear(widths[-1], num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        for layer in self.stem:
            x = layer(x)
        for stage in self.stages:
            for blk in stage:
                x = blk(x)
        return x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

def _make(name):
    widths, depths, expand, strides = _CFGS[name]

    def entry(**kwargs):
        model = ByobNet(widths, depths, expand, strides, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
