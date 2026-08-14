"""RepGhostNet in flax nnx, NHWC. Mirrors timm.models.repghost (ghost bottlenecks, reparam-style)."""
import jax.numpy as jnp
from flax import nnx

from ..layers import ConvBNAct, SqueezeExcite, ClassifierMixin
from ..registry import register_model, _cfg

class RepGhostModule(nnx.Module):
    """1x1 conv half + dw 3x3 cheap half (reparam ghost), concat."""

    def __init__(self, in_chs, out_chs, *, rngs):
        half = -(-out_chs // 2)
        self.out_chs = out_chs
        self.conv1 = nnx.Conv(in_chs, half, (1, 1), use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(half, rngs=rngs)
        self.dw = nnx.Conv(half, half, (3, 3), use_bias=False, feature_group_count=half, rngs=rngs)
        self.bn2 = nnx.BatchNorm(half, rngs=rngs)

    def __call__(self, x):
        x1 = self.bn1(self.conv1(x))
        x2 = self.bn2(self.dw(x1))
        return jnp.concatenate([x1, x2], axis=-1)[..., : self.out_chs]

class RepGhostBottleneck(nnx.Module):
    def __init__(self, in_chs, mid_chs, out_chs, stride, se, *, rngs):
        self.ghost1 = RepGhostModule(in_chs, mid_chs, rngs=rngs)
        self.dw = ConvBNAct(mid_chs, mid_chs, 3, stride, groups=mid_chs, act="identity", rngs=rngs) \
            if stride == 2 else None
        self.se = SqueezeExcite(mid_chs, rd_ratio=0.25, rngs=rngs) if se else None
        self.ghost2 = RepGhostModule(mid_chs, out_chs, rngs=rngs)
        self.use_sc = stride == 2 or in_chs != out_chs
        if self.use_sc:
            self.sc_dw = ConvBNAct(in_chs, in_chs, 3, stride, groups=in_chs, act="identity", rngs=rngs)
            self.sc_pw = ConvBNAct(in_chs, out_chs, 1, act="identity", rngs=rngs)

    def __call__(self, x):
        y = self.ghost1(x)
        if self.dw is not None:
            y = self.dw(y)
        if self.se is not None:
            y = self.se(y)
        y = self.ghost2(y)
        if self.use_sc:
            return y + self.sc_pw(self.sc_dw(x))
        return y + x

# (kernel, exp, out, se, stride, repeats)
REP_CFG = [
    (3, 16, 16, 0, 1, 1), (3, 48, 24, 0, 2, 1), (3, 72, 24, 0, 1, 1),
    (5, 72, 40, 1, 2, 1), (5, 120, 40, 1, 1, 1),
    (3, 240, 80, 0, 2, 1), (3, 200, 80, 0, 1, 1), (3, 184, 80, 0, 1, 1), (3, 184, 80, 0, 1, 1),
    (3, 480, 112, 1, 1, 1), (3, 672, 112, 1, 1, 1),
    (5, 672, 160, 1, 2, 1), (5, 960, 160, 0, 1, 1), (5, 960, 160, 1, 1, 1),
    (5, 960, 160, 0, 1, 1), (5, 960, 160, 1, 1, 1),
]

class RepGhostNet(ClassifierMixin, nnx.Module):
    default_cfg: dict = {}

    def __init__(self, width_mult=1.0, num_classes=1000, in_chans=3, global_pool="avg",
                 drop_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        stem = max(int(16 * width_mult), 8)
        self.conv1 = nnx.Conv(in_chans, stem, (3, 3), strides=(2, 2), use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(stem, rngs=rngs)
        blocks, chs = [], stem
        for k, e, c, se, s, n in REP_CFG:
            out = max(int(c * width_mult), 8)
            mid = max(int(e * width_mult), 8)
            for j in range(n):
                blocks.append(RepGhostBottleneck(chs, mid, out, s if j == 0 else 1, se, rngs=rngs))
                chs = out
        self.blocks = nnx.List(blocks)
        head = max(int(960 * width_mult), 8)
        self.conv_head = ConvBNAct(chs, head, 1, rngs=rngs)
        self.num_features = head
        self.head_drop = nnx.Dropout(drop_rate)
        self.fc = nnx.Linear(head, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = nnx.relu(self.bn1(self.conv1(x)))
        for blk in self.blocks:
            x = blk(x)
        return self.conv_head(x)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

def _repghost(width_mult, **kwargs):
    model = RepGhostNet(width_mult, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def repghostnet_050(**kwargs):
    return _repghost(0.5, **kwargs)

@register_model
def repghostnet_100(**kwargs):
    return _repghost(1.0, **kwargs)

@register_model
def repghostnet_130(**kwargs):
    return _repghost(1.3, **kwargs)
