"""MobileNetV5 in flax nnx, NHWC. Mirrors timm.models.mobilenetv5 (Multi-Scale Fusion Inverted Residuals)."""
import jax.numpy as jnp
from flax import nnx

from ..layers import ConvBNAct, SqueezeExcite, ClassifierMixin
from ..registry import register_model, _cfg

class V5Block(nnx.Module):
    """Inverted residual with multi-scale kernels (3x3 and 5x5) and SE."""

    def __init__(self, in_chs, out_chs, stride=1, expand=4, se=True, *, rngs):
        mid = in_chs * expand
        self.conv1 = ConvBNAct(in_chs, mid, 1, act="silu", rngs=rngs)
        self.dw3 = ConvBNAct(mid // 2, mid // 2, 3, stride, groups=mid // 2, act="silu", rngs=rngs)
        self.dw5 = ConvBNAct(mid // 2, mid // 2, 5, stride, groups=mid // 2, act="silu", rngs=rngs)
        self.se = SqueezeExcite(mid, rd_ratio=0.25, rngs=rngs) if se else None
        self.pw = ConvBNAct(mid, out_chs, 1, act="identity", rngs=rngs)
        self.use_sc = stride == 1 and in_chs == out_chs

    def __call__(self, x):
        h = self.conv1(x)
        h1, h2 = jnp.split(h, 2, axis=-1)
        h = jnp.concatenate([self.dw3(h1), self.dw5(h2)], axis=-1)
        if self.se is not None:
            h = self.se(h)
        y = self.pw(h)
        return x + y if self.use_sc else y

class MobileNetV5(ClassifierMixin, nnx.Module):
    default_cfg: dict | None = None

    def __init__(self, channels=(32, 64, 128, 256, 512), depths=(2, 2, 6, 2),
                 num_classes: int = 1000, in_chans: int = 3, global_pool: str = "avg",
                 drop_rate: float = 0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = channels[-1]

        self.stem = ConvBNAct(in_chans, channels[0], 3, 2, act="silu", rngs=rngs)
        stages, chs = [], channels[0]
        for i, (c, d) in enumerate(zip(channels[1:], depths)):
            blocks = []
            for j in range(d):
                blocks.append(V5Block(chs, c, 2 if j == 0 else 1, 4, se=(i >= 2), rngs=rngs))
                chs = c
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)

        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.stem(x)
        for stage in self.stages:
            for blk in stage:
                x = blk(x)
        return x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

_CFGS = {
    "mobilenetv5_300m": dict(channels=(32, 64, 128, 256, 512), depths=(2, 2, 6, 2)),
    "mobilenetv5_300m_enc": dict(channels=(32, 64, 128, 256, 512), depths=(2, 2, 6, 2), num_classes=0),
    "mobilenetv5_base": dict(channels=(48, 96, 192, 384, 768), depths=(3, 3, 9, 3)),
}

def _make(name):
    cfg = _CFGS[name]

    def entry(**kwargs):
        model = MobileNetV5(**dict(cfg, **kwargs))
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
