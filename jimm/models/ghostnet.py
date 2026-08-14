"""GhostNet in flax nnx, NHWC. Mirrors timm.models.ghostnet."""
import jax.numpy as jnp
from flax import nnx

from ..layers import ConvBNAct, SqueezeExcite, global_pool_nhwc
from ..registry import register_model, _cfg


class GhostModule(nnx.Module):
    """Cheap half conv + 5x5 depthwise on half, concatenated."""

    def __init__(self, in_chs, out_chs, *, rngs):
        half = -(-out_chs // 2)  # ceil; cheap branch same width, output sliced to out_chs
        self.out_chs = out_chs
        self.conv1 = nnx.Conv(in_chs, half, (1, 1), use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(half, rngs=rngs)
        self.dw = nnx.Conv(half, half, (5, 5), use_bias=False, feature_group_count=half, rngs=rngs)
        self.bn2 = nnx.BatchNorm(half, rngs=rngs)

    def __call__(self, x):
        x1 = nnx.relu(self.bn1(self.conv1(x)))
        x2 = nnx.relu(self.bn2(self.dw(x1)))
        return jnp.concatenate([x1, x2], axis=-1)[..., : self.out_chs]


class GhostBottleneck(nnx.Module):
    def __init__(self, in_chs, mid_chs, out_chs, kernel, stride, se, *, rngs):
        self.use_shortcut_conv = stride == 2 or in_chs != out_chs
        self.ghost1 = GhostModule(in_chs, mid_chs, rngs=rngs)
        self.dw = ConvBNAct(mid_chs, mid_chs, kernel, stride, groups=mid_chs, act="identity", rngs=rngs) if stride == 2 else None
        self.se = SqueezeExcite(mid_chs, rngs=rngs, rd_ratio=0.25) if se else None
        self.ghost2 = GhostModule(mid_chs, out_chs, rngs=rngs)
        if self.use_shortcut_conv:
            self.sc_dw = ConvBNAct(in_chs, in_chs, kernel, stride, groups=in_chs, act="identity", rngs=rngs)
            self.sc_pw = ConvBNAct(in_chs, out_chs, 1, act="identity", rngs=rngs)

    def __call__(self, x):
        y = self.ghost1(x)
        if self.dw is not None:
            y = self.dw(y)
        if self.se is not None:
            y = self.se(y)
        y = self.ghost2(y)  # linear (no act) per paper
        if self.use_shortcut_conv:
            return y + self.sc_pw(self.sc_dw(x))
        return y + x


# (kernel, exp, out, se, stride, repeats)
GHOSTNET_CFG = [
    (3, 16, 16, 0, 1, 1),
    (3, 48, 24, 0, 2, 1), (3, 72, 24, 0, 1, 1),
    (5, 72, 40, 1, 2, 1), (5, 120, 40, 1, 1, 1),
    (3, 240, 80, 0, 2, 1),
    (3, 200, 80, 0, 1, 1), (3, 184, 80, 0, 1, 1), (3, 184, 80, 0, 1, 1),
    (3, 480, 112, 1, 1, 1), (3, 672, 112, 1, 1, 1),
    (5, 672, 160, 1, 2, 1),
    (5, 960, 160, 0, 1, 1), (5, 960, 160, 1, 1, 1), (5, 960, 160, 0, 1, 1), (5, 960, 160, 1, 1, 1),
]


class GhostNet(nnx.Module):
    default_cfg: dict = {}

    def __init__(self, width_mult=1.0, num_classes=1000, in_chans=3, global_pool="avg",
                 drop_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        stem = max(int(16 * width_mult), 8)
        self.conv1 = nnx.Conv(in_chans, stem, (3, 3), strides=(2, 2), use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(stem, rngs=rngs)
        blocks, chs = [], stem
        for k, e, c, se, s, n in GHOSTNET_CFG:
            out = max(int(c * width_mult), 8)
            mid = max(int(e * width_mult), 8)
            for j in range(n):
                blocks.append(GhostBottleneck(chs, mid, out, k, s if j == 0 else 1, se, rngs=rngs))
                chs = out
        self.blocks = nnx.List(blocks)
        head = max(int(960 * width_mult), 8)
        self.conv_head = ConvBNAct(chs, head, 1, rngs=rngs)
        self.num_features = 1280
        self.head_fc1 = nnx.Linear(head, 1280, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate)
        self.fc = nnx.Linear(1280, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = nnx.relu(self.bn1(self.conv1(x)))
        for blk in self.blocks:
            x = blk(x)
        return self.conv_head(x)

    def forward_head(self, x):
        x = global_pool_nhwc(x, self.global_pool)
        x = nnx.relu(self.head_fc1(x))
        x = self.head_drop(x)
        return self.fc(x) if self.fc is not None else x

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool="avg"):
        self.num_classes, self.global_pool = num_classes, global_pool
        if num_classes > 0 and self.fc is None:
            raise RuntimeError("cannot re-add classifier to a num_classes=0 model")
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=nnx.Rngs(0)) if num_classes > 0 else None

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


def _ghostnet(width_mult, **kwargs):
    model = GhostNet(width_mult, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def ghostnet_050(**kwargs):
    return _ghostnet(0.5, **kwargs)


@register_model
def ghostnet_100(**kwargs):
    return _ghostnet(1.0, **kwargs)


@register_model
def ghostnet_130(**kwargs):
    return _ghostnet(1.3, **kwargs)
