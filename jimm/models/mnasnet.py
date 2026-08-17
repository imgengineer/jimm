"""MNASNet / Single-Path NASNet in flax nnx, NHWC. Mirrors timm.models.mnasnet."""
from flax import nnx

from ..layers import relu6, ClassifierMixin
from ..registry import register_model, _cfg
from .mobilenetv2 import round_chs

class SepConv(nnx.Module):
    """3x3 depthwise -> 1x1 pointwise (mnasnet stage-0 block)."""

    def __init__(self, in_chs, out_chs, kernel=3, stride=1, *, rngs):
        self.dw = nnx.Conv(in_chs, in_chs, (kernel, kernel), strides=(stride, stride),
                           use_bias=False, feature_group_count=in_chs, rngs=rngs)
        self.bn1 = nnx.BatchNorm(in_chs, rngs=rngs)
        self.pw = nnx.Conv(in_chs, out_chs, (1, 1), use_bias=False, rngs=rngs)
        self.bn2 = nnx.BatchNorm(out_chs, rngs=rngs)

    def __call__(self, x):
        return self.bn2(self.pw(relu6(self.bn1(self.dw(x)))))

class MBBlock(nnx.Module):
    def __init__(self, in_chs, out_chs, kernel, stride, expand, *, rngs):
        mid = in_chs * expand
        self.use_residual = stride == 1 and in_chs == out_chs
        self.expand = nnx.Conv(in_chs, mid, (1, 1), use_bias=False, rngs=rngs)
        self.bn0 = nnx.BatchNorm(mid, rngs=rngs)
        self.dw = nnx.Conv(mid, mid, (kernel, kernel), strides=(stride, stride), use_bias=False,
                           feature_group_count=mid, rngs=rngs)
        self.bn1 = nnx.BatchNorm(mid, rngs=rngs)
        self.pw = nnx.Conv(mid, out_chs, (1, 1), use_bias=False, rngs=rngs)
        self.bn2 = nnx.BatchNorm(out_chs, rngs=rngs)

    def __call__(self, x):
        y = relu6(self.bn0(self.expand(x)))
        y = relu6(self.bn1(self.dw(y)))
        y = self.bn2(self.pw(y))
        return x + y if self.use_residual else y

# (type, kernel, expand, out, repeats, stride)
# mnasnet-a1: sep stage stride 1; the 96-channel MB stage keeps resolution (stride 1)
MNASNET_CFG = [
    ("sep", 3, 16, 1, 1, 1),
    ("mb", 3, 24, 3, 2, 3),
    ("mb", 5, 40, 3, 2, 6),
    ("mb", 5, 80, 3, 2, 6),
    ("mb", 3, 96, 2, 1, 6),
    ("mb", 5, 192, 4, 2, 6),
    ("mb", 5, 320, 1, 1, 6),
]
# spnasnet: same MB config, no sep stage-0 nuance (identical structure family)
SPNAS_CFG = MNASNET_CFG

class MNASNet(ClassifierMixin, nnx.Module):
    default_cfg: dict | None = None

    def __init__(self, cfg=MNASNET_CFG, width_mult=1.0, num_classes=1000, in_chans=3,
                 global_pool="avg", drop_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        stem = round_chs(32, width_mult)
        self.conv1 = nnx.Conv(in_chans, stem, (3, 3), strides=(2, 2), use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(stem, rngs=rngs)
        blocks, chs = [], stem
        for item in cfg:
            kind, k, out, n, s, e = item
            out = round_chs(out, width_mult)
            for j in range(n):
                if kind == "sep":
                    blocks.append(SepConv(chs, out, k, s if j == 0 else 1, rngs=rngs))
                else:
                    blocks.append(MBBlock(chs, out, k, s if j == 0 else 1, e, rngs=rngs))
                chs = out
        self.blocks = nnx.List(blocks)
        head = round_chs(1280, width_mult)
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

def _mnasnet(cfg, width_mult, **kwargs):
    model = MNASNet(cfg, width_mult, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def mnasnet_050(**kwargs):
    return _mnasnet(MNASNET_CFG, 0.5, **kwargs)

@register_model
def mnasnet_100(**kwargs):
    return _mnasnet(MNASNET_CFG, 1.0, **kwargs)

@register_model
def mnasnet_140(**kwargs):
    return _mnasnet(MNASNET_CFG, 1.4, **kwargs)

@register_model
def spnasnet_100(**kwargs):
    return _mnasnet(SPNAS_CFG, 1.0, **kwargs)
