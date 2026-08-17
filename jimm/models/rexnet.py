"""ReXNet in flax nnx, NHWC. Mirrors timm.models.rexnet (swish + SE inverted residuals)."""
import jax
from flax import nnx

from ..layers import SqueezeExcite, ClassifierMixin
from ..registry import register_model, _cfg
from .mobilenetv2 import ConvBN, round_chs

class ReXBlock(nnx.Module):
    def __init__(self, in_chs, out_chs, stride, expand, use_se, *, rngs):
        try:
            mid = int(round(in_chs * expand))
        except Exception:
            mid = in_chs * 6
        self.use_residual = stride == 1 and in_chs == out_chs
        self.expand = ConvBN(in_chs, mid, rngs=rngs) if expand != 1 else None
        self.dw = nnx.Conv(mid, mid, (3, 3), strides=(stride, stride), use_bias=False,
                           feature_group_count=mid, rngs=rngs)
        self.bn1 = nnx.BatchNorm(mid, rngs=rngs)
        self.se = SqueezeExcite(mid, rngs=rngs, rd_ratio=1 / 12) if use_se else None
        self.pw = nnx.Conv(mid, out_chs, (1, 1), use_bias=False, rngs=rngs)
        self.bn2 = nnx.BatchNorm(out_chs, rngs=rngs)

    def __call__(self, x):
        y = x if self.expand is None else nnx.silu(self.expand(x))
        y = nnx.silu(self.bn1(self.dw(y)))
        if self.se is not None:
            y = self.se(y)
        y = self.bn2(self.pw(y))
        return x + y if self.use_residual else y

# (expand, out, repeats, stride, se)
REXNET_CFG = [
    (1, 16, 1, 1, False),
    (6, 24, 2, 2, False),
    (6, 32, 3, 2, False),
    (6, 64, 4, 2, True),
    (6, 96, 3, 1, True),
    (6, 160, 3, 2, True),
    (6, 320, 1, 1, True),
]

class ReXNet(ClassifierMixin, nnx.Module):
    default_cfg: dict | None = None

    def __init__(self, width_mult=1.0, num_classes=1000, in_chans=3, global_pool="avg",
                 drop_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        stem = round_chs(32, width_mult)
        self.conv1 = nnx.Conv(in_chans, stem, (3, 3), strides=(2, 2), use_bias=False,
                              padding="VALID", rngs=rngs)
        self.bn1 = nnx.BatchNorm(stem, rngs=rngs)
        blocks, chs = [], stem
        for e, c, n, s, se in REXNET_CFG:
            out = round_chs(c, width_mult)
            for j in range(n):
                blocks.append(ReXBlock(chs, out, s if j == 0 else 1, e, se, rngs=rngs))
                chs = out
        self.blocks = nnx.List(blocks)
        head = round_chs(1280, width_mult)
        self.conv_head = nnx.Conv(chs, head, (1, 1), use_bias=False, rngs=rngs)
        self.bn_head = nnx.BatchNorm(head, rngs=rngs)
        self.num_features = head
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(head, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = nnx.silu(self.bn1(self.conv1(x)))
        for blk in self.blocks:
            x = blk(x)
        return nnx.silu(self.bn_head(self.conv_head(x)))

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

def _rexnet(width_mult, **kwargs):
    model = ReXNet(width_mult, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def rexnet_100(**kwargs):
    return _rexnet(1.0, **kwargs)

@register_model
def rexnet_130(**kwargs):
    return _rexnet(1.3, **kwargs)

@register_model
def rexnet_150(**kwargs):
    return _rexnet(1.5, **kwargs)

@register_model
def rexnet_200(**kwargs):
    return _rexnet(2.0, **kwargs)
