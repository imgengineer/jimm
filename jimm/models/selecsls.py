"""SelecSLS in flax nnx, NHWC. Mirrors timm.models.selecsls (dual 3x3+1x1 selective blocks)."""
from flax import nnx

from ..layers import ConvBNAct, ClassifierMixin
from ..registry import register_model, _cfg

class SelecSLSBlock(nnx.Module):
    """conv1x1 -> conv3x3 -> conv1x1 with residual; selective downsample variants."""

    def __init__(self, in_chs, out_chs, stride=1, *, rngs):
        mid = out_chs // 2
        self.conv1 = ConvBNAct(in_chs, mid, 1, rngs=rngs)
        self.conv2 = ConvBNAct(mid, mid, 3, stride, rngs=rngs)
        self.conv3 = ConvBNAct(mid, out_chs, 1, act="identity", rngs=rngs)
        self.shortcut = ConvBNAct(in_chs, out_chs, 1, stride, act="identity", rngs=rngs) \
            if (stride != 1 or in_chs != out_chs) else None

    def __call__(self, x):
        y = self.conv3(self.conv2(self.conv1(x)))
        sc = x if self.shortcut is None else self.shortcut(x)
        return nnx.relu(y + sc)

class SelecSLS(ClassifierMixin, nnx.Module):
    default_cfg: dict = {}

    def __init__(self, channels=(64, 128, 288, 544), depths=(2, 3, 7, 3), num_classes=1000,
                 in_chans=3, global_pool="avg", drop_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.stem = nnx.List([
            ConvBNAct(in_chans, 32, 3, 2, rngs=rngs),
            ConvBNAct(32, 64, 3, 2, rngs=rngs)])
        stages, chs = [], 64
        for i, (c, d) in enumerate(zip(channels, depths)):
            blocks = []
            for j in range(d):
                blocks.append(SelecSLSBlock(chs, c, 2 if j == 0 else 1, rngs=rngs))
                chs = c
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.num_features = channels[-1]
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(channels[-1], num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        for layer in self.stem:
            x = layer(x)
        for stage in self.stages:
            for blk in stage:
                x = blk(x)
        return x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

_CFGS = {
    "selecsls42": ((64, 128, 288, 544), (2, 3, 7, 3)),
    "selecsls60": ((64, 128, 288, 544), (3, 4, 10, 4)),
}

def _make(name):
    channels, depths = _CFGS[name]

    def entry(**kwargs):
        model = SelecSLS(channels, depths, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
