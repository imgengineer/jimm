"""FasterNet in flax nnx, NHWC. Mirrors timm.models.fasternet (partial conv blocks)."""
import jax.numpy as jnp
from flax import nnx

from ..layers import DropPath, ClassifierMixin
from ..registry import register_model, _cfg

class PartialConv(nnx.Module):
    """Conv on the first p_chs channels only (FasterNet)."""

    def __init__(self, dim, p_chs, *, rngs):
        self.p_chs = p_chs
        self.conv = nnx.Conv(p_chs, p_chs, (3, 3), use_bias=False, rngs=rngs)
        self.bn = nnx.BatchNorm(p_chs, rngs=rngs)

    def __call__(self, x):
        y = nnx.relu(self.bn(self.conv(x[..., : self.p_chs])))
        return jnp.concatenate([y, x[..., self.p_chs:]], axis=-1)

class FasterNetBlock(nnx.Module):
    def __init__(self, dim, expand=2, drop_path=0.0, *, rngs):
        self.pconv = PartialConv(dim, dim // 4, rngs=rngs)
        self.fc1 = nnx.Linear(dim, dim * expand, rngs=rngs)
        self.fc2 = nnx.Linear(dim * expand, dim, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        y = self.pconv(x)
        y = self.fc2(nnx.relu(self.fc1(y)))
        return x + self.drop_path(y)

class FasterNet(ClassifierMixin, nnx.Module):
    default_cfg: dict = {}

    def __init__(self, channels=(64, 128, 256, 512), depths=(1, 2, 8, 2), num_classes=1000,
                 in_chans=3, global_pool="avg", drop_rate=0.0, drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = channels[-1]
        self.stem = nnx.Conv(in_chans, channels[0], (4, 4), strides=(4, 4), use_bias=False, rngs=rngs)
        self.stem_bn = nnx.BatchNorm(channels[0], rngs=rngs)
        dpr = [drop_path_rate * i / max(sum(depths) - 1, 1) for i in range(sum(depths))]
        stages, k = [], 0
        for i, (c, d) in enumerate(zip(channels, depths)):
            blocks = []
            if i > 0:
                blocks.append(("down", c))
            for _ in range(d):
                blocks.append(FasterNetBlock(c, 2, dpr[k], rngs=rngs))
                k += 1
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.downsamples = nnx.List([
            nnx.Sequential(nnx.BatchNorm(channels[i], rngs=rngs),
                           nnx.Conv(channels[i], channels[i + 1], (2, 2), strides=(2, 2),
                                    use_bias=False, rngs=rngs))
            for i in range(3)])
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(channels[-1], num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = nnx.relu(self.stem_bn(self.stem(x)))
        for i, stage in enumerate(self.stages):
            if i > 0:
                x = self.downsamples[i - 1](x)
            for blk in stage:
                if isinstance(blk, tuple):
                    continue
                x = blk(x)
        return x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

_CFGS = {
    "fasternet_t0": ((48, 96, 192, 384), (1, 2, 8, 2)),
    "fasternet_t1": ((64, 128, 256, 512), (1, 2, 8, 2)),
    "fasternet_s": ((96, 192, 384, 768), (1, 2, 13, 2)),
}

def _make(name):
    channels, depths = _CFGS[name]

    def entry(**kwargs):
        model = FasterNet(channels, depths, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
