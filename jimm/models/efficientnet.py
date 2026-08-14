"""EfficientNet (B0-B7) in flax nnx, NHWC. Mirrors timm.models.efficientnet / torchvision."""
import math

from jax.nn import silu
from flax import nnx

from ..layers import DropPath, SqueezeExcite, ClassifierMixin
from ..registry import register_model, _cfg
from .mobilenetv2 import ConvBN

class MBConv(nnx.Module):
    def __init__(self, in_chs, out_chs, kernel, stride, expand, drop_path=0.0, *, rngs):
        mid = in_chs * expand
        self.use_residual = stride == 1 and in_chs == out_chs
        self.drop_path = DropPath(drop_path)
        self.expand = ConvBN(in_chs, mid, rngs=rngs) if expand != 1 else None
        self.dw = nnx.Conv(mid, mid, (kernel, kernel), strides=(stride, stride), use_bias=False,
                           feature_group_count=mid, rngs=rngs)
        self.bn1 = nnx.BatchNorm(mid, rngs=rngs)
        self.se = SqueezeExcite(mid, rngs=rngs, rd_ratio=0.25)
        self.pw = nnx.Conv(mid, out_chs, (1, 1), use_bias=False, rngs=rngs)
        self.bn2 = nnx.BatchNorm(out_chs, rngs=rngs)

    def __call__(self, x):
        y = x if self.expand is None else silu(self.expand(x))
        y = silu(self.bn1(self.dw(y)))
        y = self.se(y)
        y = self.bn2(self.pw(y))
        return x + self.drop_path(y) if self.use_residual else y

# (kernel, expand, out_chs, repeats, stride) — B0 base
BASE_CFG = [
    (3, 1, 16, 1, 1), (3, 6, 24, 2, 2), (5, 6, 40, 2, 2), (3, 6, 80, 3, 2),
    (5, 6, 112, 3, 1), (5, 6, 192, 4, 2), (3, 6, 320, 1, 1),
]

# name: (width_mult, depth_mult, img_size, drop_rate)
_VARIANTS = {
    "efficientnet_b0": (1.0, 1.0, 224, 0.2),
    "efficientnet_b1": (1.0, 1.1, 240, 0.2),
    "efficientnet_b2": (1.1, 1.2, 260, 0.3),
    "efficientnet_b3": (1.2, 1.4, 300, 0.3),
    "efficientnet_b4": (1.4, 1.8, 380, 0.4),
    "efficientnet_b5": (1.6, 2.2, 456, 0.4),
    "efficientnet_b6": (1.8, 2.6, 528, 0.5),
    "efficientnet_b7": (2.0, 3.1, 600, 0.5),
    # TinyNet ( EfficientNet scaled with r != 1 )
    "tinynet_a": (0.86, 1.0, 192, 0.2),
    "tinynet_b": (0.84, 0.75, 188, 0.2),
    "tinynet_c": (0.825, 0.54, 184, 0.2),
    "tinynet_d": (0.68, 0.71, 152, 0.2),
    "tinynet_e": (0.475, 0.51, 106, 0.2),
}

def _round_width(c, mult):
    return max(8, int(c * mult + 4) // 8 * 8)

def _round_depth(n, mult):
    return int(math.ceil(n * mult))

class EfficientNet(ClassifierMixin, nnx.Module):
    default_cfg: dict = {}

    def __init__(self, width_mult=1.0, depth_mult=1.0, num_classes=1000, in_chans=3,
                 global_pool="avg", drop_rate=0.2, drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        stem = _round_width(32, width_mult)
        self.conv_stem = nnx.Conv(in_chans, stem, (3, 3), strides=(2, 2), use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(stem, rngs=rngs)
        total = sum(_round_depth(n, depth_mult) for _, _, _, n, _ in BASE_CFG)
        dpr = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]
        blocks, chs = [], stem
        for k, e, c, n, s in BASE_CFG:
            out = _round_width(c, width_mult)
            for j in range(_round_depth(n, depth_mult)):
                blocks.append(MBConv(chs, out, k, s if j == 0 else 1, e, dpr[len(blocks)], rngs=rngs))
                chs = out
        self.blocks = nnx.List(blocks)
        head = _round_width(1280, width_mult)
        self.conv_head = nnx.Conv(chs, head, (1, 1), use_bias=False, rngs=rngs)
        self.bn_head = nnx.BatchNorm(head, rngs=rngs)
        self.num_features = head
        self.head_drop = nnx.Dropout(drop_rate)
        self.fc = nnx.Linear(head, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = silu(self.bn1(self.conv_stem(x)))
        for blk in self.blocks:
            x = blk(x)
        return silu(self.bn_head(self.conv_head(x)))

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

def _make(name):
    w, d, img, drop = _VARIANTS[name]

    def entry(**kwargs):
        kwargs.setdefault("drop_rate", drop)
        model = EfficientNet(w, d, **kwargs)
        model.default_cfg = _cfg(input_size=(3, img, img))
        return model
    entry.__name__ = name
    return entry

for _name in _VARIANTS:
    register_model(_make(_name))
