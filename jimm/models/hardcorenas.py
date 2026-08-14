"""HardCoReNAS (RegNet-style NAS) in flax nnx, NHWC. Mirrors timm.models.hardcorenas."""
from flax import nnx

from ..layers import ConvBNAct, ClassifierMixin
from ..registry import register_model, _cfg

class HardBlock(nnx.Module):
    """bottleneck 1x1 -> dw 3x3 (groups) -> 1x1, residual."""

    def __init__(self, in_chs, out_chs, mid_chs, stride, groups, *, rngs):
        self.conv1 = ConvBNAct(in_chs, mid_chs, 1, act="silu", rngs=rngs)
        self.conv2 = ConvBNAct(mid_chs, mid_chs, 3, stride, groups=groups, act="silu", rngs=rngs)
        self.conv3 = ConvBNAct(mid_chs, out_chs, 1, act="identity", rngs=rngs)
        self.shortcut = ConvBNAct(in_chs, out_chs, 1, stride, act="identity", rngs=rngs) \
            if (stride != 1 or in_chs != out_chs) else None

    def __call__(self, x):
        y = self.conv3(self.conv2(self.conv1(x)))
        sc = x if self.shortcut is None else self.shortcut(x)
        return nnx.relu(y + sc)

# (in->mid->out, stride, groups, repeats) per stage entry
_CFGS = {
    "hardcorenas_a": [
        (16, 16, 1, 1, 1), (24, 24, 2, 1, 1), (24, 24, 1, 1, 1),
        (40, 40, 2, 8, 1), (40, 40, 1, 8, 2), (80, 80, 2, 8, 1), (80, 80, 1, 8, 2),
        (96, 96, 1, 8, 1), (192, 192, 2, 8, 1), (192, 192, 1, 8, 2), (192, 192, 1, 8, 1),
        (376, 376, 1, 8, 1),
    ],
    "hardcorenas_f": [
        (16, 16, 1, 1, 1), (24, 24, 2, 1, 1), (24, 24, 1, 1, 1),
        (40, 40, 2, 8, 1), (40, 40, 1, 8, 2), (80, 80, 2, 8, 1), (80, 80, 1, 8, 2),
        (112, 112, 1, 8, 1), (224, 224, 2, 8, 1), (224, 224, 1, 8, 2), (224, 224, 1, 8, 1),
        (480, 480, 1, 8, 1),
    ],
}

class HardCoReNAS(ClassifierMixin, nnx.Module):
    default_cfg: dict = {}

    def __init__(self, cfg, num_classes=1000, in_chans=3, global_pool="avg", drop_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.stem = ConvBNAct(in_chans, 32, 3, 2, act="silu", rngs=rngs)
        blocks, chs = [], 32
        for c, out, s, g, n in cfg:
            for j in range(n):
                blocks.append(HardBlock(chs, out, c, s if j == 0 else 1, g, rngs=rngs))
                chs = out
        self.blocks = nnx.List(blocks)
        self.num_features = cfg[-1][1]
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        return x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

def _make(name):
    cfg = _CFGS[name]

    def entry(**kwargs):
        model = HardCoReNAS(cfg, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
