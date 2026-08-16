"""EfficientNet (B0-B7) in flax nnx, NHWC. Mirrors timm.models.efficientnet / torchvision."""
import math

import jax.numpy as jnp
from jax.nn import silu
from flax import nnx

from ..layers import DropPath, ClassifierMixin
from ..registry import register_model, _cfg

class SqueezeExciteEff(nnx.Module):
    def __init__(self, in_chs, exp_chs, se_ratio=0.25, *, rngs):
        try:
            rd_chs = max(1, int(in_chs * se_ratio))
        except Exception:
            rd_chs = 1
        self.conv_reduce = nnx.Conv(exp_chs, rd_chs, (1, 1), use_bias=True, rngs=rngs)
        self.conv_expand = nnx.Conv(rd_chs, exp_chs, (1, 1), use_bias=True, rngs=rngs)

    def __call__(self, x):
        s = jnp.mean(x, axis=(1, 2), keepdims=True)
        s = silu(self.conv_reduce(s))
        s = nnx.sigmoid(self.conv_expand(s))
        return x * s


class MBConv(nnx.Module):
    def __init__(self, in_chs, out_chs, kernel, stride, expand, drop_path=0.0, *, rngs):
        mid = in_chs * expand
        self.has_expand = expand != 1
        self.use_residual = stride == 1 and in_chs == out_chs
        self.drop_path = DropPath(drop_path, rngs=rngs)
        if self.has_expand:
            self.expand = nnx.Conv(in_chs, mid, (1, 1), use_bias=False, rngs=rngs)
            self.bn1 = nnx.BatchNorm(mid, rngs=rngs)
        self.dw = nnx.Conv(mid, mid, (kernel, kernel), strides=(stride, stride), use_bias=False,
                           feature_group_count=mid, rngs=rngs)
        self.bn2 = nnx.BatchNorm(mid, rngs=rngs)
        self.se = SqueezeExciteEff(in_chs, mid, 0.25, rngs=rngs)
        self.pw = nnx.Conv(mid, out_chs, (1, 1), use_bias=False, rngs=rngs)
        self.bn3 = nnx.BatchNorm(out_chs, rngs=rngs)

    def __call__(self, x):
        y = silu(self.bn1(self.expand(x))) if self.has_expand else x
        y = silu(self.bn2(self.dw(y)))
        y = self.se(y)
        y = self.bn3(self.pw(y))
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
    if not mult:
        return c
    c_val = c * mult
    try:
        new_c = max(8, int(c_val + 4) // 8 * 8)
    except Exception:
        new_c = 8
    if new_c < 0.9 * c_val:
        new_c += 8
    return new_c

def _round_depth(n, mult):
    try:
        return int(math.ceil(n * mult))
    except Exception:
        return n

class EfficientNet(ClassifierMixin, nnx.Module):
    default_cfg: dict = {}

    def __init__(self, width_mult=1.0, depth_mult=1.0, channel_multiplier=None, depth_multiplier=None,
                 num_classes=1000, in_chans=3, global_pool="avg", drop_rate=0.2, drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        width_mult = channel_multiplier if channel_multiplier is not None else width_mult
        depth_mult = depth_multiplier if depth_multiplier is not None else depth_mult
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
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
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
