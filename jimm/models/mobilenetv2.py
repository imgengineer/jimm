"""MobileNetV2 in flax nnx, NHWC. Mirrors timm.models.mobilenetv2 / torchvision."""
import jax.numpy as jnp
from flax import nnx

from ..layers import ClassifierMixin
from ..registry import register_model, _cfg

def relu6(x):
    return jnp.minimum(jnp.maximum(x, 0), 6)

class ConvBN(nnx.Module):
    """1x1 conv + BN (expansion layer)."""

    def __init__(self, in_chs, out_chs, *, rngs):
        self.conv = nnx.Conv(in_chs, out_chs, (1, 1), use_bias=False, rngs=rngs)
        self.bn = nnx.BatchNorm(out_chs, rngs=rngs)

    def __call__(self, x):
        return self.bn(self.conv(x))

class InvertedResidual(nnx.Module):
    def __init__(self, in_chs, out_chs, stride, expand_ratio, *, rngs):
        mid = in_chs * expand_ratio
        self.use_residual = stride == 1 and in_chs == out_chs
        self.expand = ConvBN(in_chs, mid, rngs=rngs) if expand_ratio != 1 else None
        self.dw = nnx.Conv(mid, mid, (3, 3), strides=(stride, stride), use_bias=False,
                           feature_group_count=mid, rngs=rngs)
        self.bn1 = nnx.BatchNorm(mid, rngs=rngs)
        self.pw = nnx.Conv(mid, out_chs, (1, 1), use_bias=False, rngs=rngs)
        self.bn2 = nnx.BatchNorm(out_chs, rngs=rngs)

    def __call__(self, x):
        y = x if self.expand is None else relu6(self.expand(x))
        y = relu6(self.bn1(self.dw(y)))
        y = self.bn2(self.pw(y))
        return x + y if self.use_residual else y

def round_chs(c, mult):
    try:
        return max(8, int(c * mult + 4) // 8 * 8)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid channel multiplier: {mult!r}") from exc

class MobileNetV2(ClassifierMixin, nnx.Module):
    default_cfg: dict | None = None

    # (expand, out_chs, repeats, stride)
    CFG = ((1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2),
           (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1))

    def __init__(self, width_mult=1.0, num_classes=1000, in_chans=3, global_pool="avg",
                 drop_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        stem = round_chs(32, width_mult)
        self.conv1 = nnx.Conv(in_chans, stem, (3, 3), strides=(2, 2), use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(stem, rngs=rngs)
        blocks, chs = [], stem
        for t, c, n, s in self.CFG:
            out = round_chs(c, width_mult)
            for j in range(n):
                blocks.append(InvertedResidual(chs, out, s if j == 0 else 1, t, rngs=rngs))
                chs = out
        self.blocks = nnx.List(blocks)
        head = round_chs(1280, width_mult) if width_mult > 1.0 else 1280
        self.conv_head = nnx.Conv(chs, head, (1, 1), use_bias=False, rngs=rngs)
        self.bn_head = nnx.BatchNorm(head, rngs=rngs)
        self.num_features = head
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(head, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = relu6(self.bn1(self.conv1(x)))
        for blk in self.blocks:
            x = blk(x)
        return relu6(self.bn_head(self.conv_head(x)))

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

def _mbv2(width_mult, **kwargs):
    model = MobileNetV2(width_mult, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def mobilenetv2_050(**kwargs):
    return _mbv2(0.5, **kwargs)

@register_model
def mobilenetv2_100(**kwargs):
    return _mbv2(1.0, **kwargs)

@register_model
def mobilenetv2_140(**kwargs):
    return _mbv2(1.4, **kwargs)
