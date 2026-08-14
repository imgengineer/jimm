"""RegNet (X and Y) in flax nnx, NHWC. Mirrors timm.models.regnet / torchvision.

Stage widths/depths/groups are generated with the RegNet design-space algorithm
(same as torchvision's BlockParams.from_init_params), so configs match exactly.
"""
import math

from flax import nnx

from ..layers import SqueezeExcite, ClassifierMixin
from ..registry import register_model, _cfg

def _quantize(v, q):
    return int(round(v / q) * q)  # timm quantize_float (timm default group_min_ratio=0)

def gen_cfg(depth, w0, wa, wm, group_width):
    """Returns (widths, depths, groups) per stage, torchvision-exact."""
    QUANT = 8
    widths_cont = [i * wa + w0 for i in range(depth)]
    capacity = [round(math.log(w / w0) / math.log(wm)) for w in widths_cont]
    block_widths = [int(round(w0 * wm ** c / QUANT)) * QUANT for c in capacity]
    widths, depths = [], []  # merge stages on RAW block widths first (torchvision order)
    for w in block_widths:
        if widths and w == widths[-1]:
            depths[-1] += 1
        else:
            widths.append(w)
            depths.append(1)
    groups = [min(group_width, w) for w in widths]  # group WIDTH, bottleneck_multiplier == 1.0
    widths = [_quantize(w, g) for w, g in zip(widths, groups)]
    groups = [w // g for w, g in zip(widths, groups)]
    return widths, depths, groups

class RegNetBlock(nnx.Module):
    """Bottleneck with group conv (and optional SE for RegNetY)."""

    def __init__(self, in_chs, out_chs, stride, groups, se_ratio=0.0, *, rngs):
        mid = out_chs  # bottleneck multiplier 1 for RegNet
        self.conv1 = nnx.Conv(in_chs, mid, (1, 1), use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(mid, rngs=rngs)
        self.conv2 = nnx.Conv(mid, mid, (3, 3), strides=(stride, stride), use_bias=False,
                              feature_group_count=groups, rngs=rngs)
        self.bn2 = nnx.BatchNorm(mid, rngs=rngs)
        self.se = SqueezeExcite(mid, rngs=rngs, rd_ratio=se_ratio) if se_ratio > 0 else None
        self.conv3 = nnx.Conv(mid, out_chs, (1, 1), use_bias=False, rngs=rngs)
        self.bn3 = nnx.BatchNorm(out_chs, rngs=rngs)
        self.shortcut = nnx.Sequential(
            nnx.Conv(in_chs, out_chs, (1, 1), strides=(stride, stride), use_bias=False, rngs=rngs),
            nnx.BatchNorm(out_chs, rngs=rngs)) if (stride != 1 or in_chs != out_chs) else None

    def __call__(self, x):
        y = nnx.relu(self.bn1(self.conv1(x)))
        y = nnx.relu(self.bn2(self.conv2(y)))
        if self.se is not None:
            y = self.se(y)
        y = self.bn3(self.conv3(y))
        sc = x if self.shortcut is None else self.shortcut(x)
        return nnx.relu(y + sc)

class RegNet(ClassifierMixin, nnx.Module):
    default_cfg: dict = {}

    def __init__(self, widths, depths, groups, num_classes=1000, in_chans=3,
                 global_pool="avg", drop_rate=0.0, se_ratio=0.0, stem_chs=32, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.stem_conv = nnx.Conv(in_chans, stem_chs, (3, 3), strides=(2, 2), use_bias=False, rngs=rngs)
        self.stem_bn = nnx.BatchNorm(stem_chs, rngs=rngs)
        stages, chs = [], stem_chs
        for w, d, g in zip(widths, depths, groups):
            blocks = []
            for j in range(d):
                blocks.append(RegNetBlock(chs, w, 2 if j == 0 else 1, g, se_ratio, rngs=rngs))
                chs = w
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.num_features = widths[-1]
        self.head_drop = nnx.Dropout(drop_rate)
        self.fc = nnx.Linear(widths[-1], num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = nnx.relu(self.stem_bn(self.stem_conv(x)))
        for stage in self.stages:
            for blk in stage:
                x = blk(x)
        return x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

# name: (depth, w0, wa, wm, group_width) — RegNet paper / torchvision params
_CFGS = {
    "regnetx_002": (13, 24, 36.44, 2.49, 8),
    "regnetx_004": (22, 24, 24.48, 2.54, 16),
    "regnetx_008": (16, 56, 35.73, 2.28, 16),
    "regnetx_016": (18, 80, 34.01, 2.25, 24),
    "regnetx_032": (25, 88, 26.31, 2.25, 48),
    "regnety_004": (16, 48, 27.89, 2.09, 8),
    "regnety_008": (14, 56, 38.84, 2.4, 16),
    "regnety_016": (27, 48, 20.71, 2.65, 24),
    "regnety_032": (21, 80, 42.63, 2.66, 24),
}

def _make(name):
    depth, w0, wa, wm, gw = _CFGS[name]
    widths, depths, groups = gen_cfg(depth, w0, wa, wm, gw)
    se = 0.25 if name.startswith("regnety") else 0.0

    def entry(**kwargs):
        model = RegNet(widths, depths, groups, se_ratio=se, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
