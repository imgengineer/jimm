"""RDNet in flax nnx, NHWC. Mirrors timm.models.rdnet (dense connection blocks)."""
import jax.numpy as jnp
from flax import nnx

from ..layers import DropPath, ClassifierMixin
from ..registry import register_model, _cfg

class RDBlock(nnx.Module):
    """dense: dw 7x7 on growing concat -> 1x1 mix each step."""

    def __init__(self, dim, growth=64, layers=4, drop_path=0.0, *, rngs):
        self.layers = layers
        self.dws = nnx.List([nnx.Conv(dim + i * growth, dim + i * growth, (7, 7),
                                      feature_group_count=dim + i * growth, rngs=rngs)
                             for i in range(layers)])
        self.mixs = nnx.List([nnx.Linear(dim + i * growth, growth, rngs=rngs) for i in range(layers)])
        self.norm = nnx.LayerNorm(dim + layers * growth, rngs=rngs)
        self.out_proj = nnx.Linear(dim + layers * growth, dim, rngs=rngs)
        self.drop_path = DropPath(drop_path)

    def __call__(self, x):
        feats = x
        for dw, mix in zip(self.dws, self.mixs):
            y = mix(nnx.gelu(dw(feats)))
            feats = jnp.concatenate([feats, y], axis=-1)
        return x + self.drop_path(self.out_proj(self.norm(feats)))

class RDNet(ClassifierMixin, nnx.Module):
    default_cfg: dict = {}

    def __init__(self, channels=(96, 192, 384, 768), depths=(2, 2, 6, 2), growth=64,
                 num_classes=1000, in_chans=3, global_pool="avg", drop_rate=0.0,
                 drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = channels[-1]
        self.stem = nnx.Conv(in_chans, channels[0], (4, 4), strides=(4, 4), rngs=rngs)
        self.stem_norm = nnx.LayerNorm(channels[0], rngs=rngs)
        dpr = [drop_path_rate * i / max(sum(depths) - 1, 1) for i in range(sum(depths))]
        stages, k = [], 0
        for i, (c, d) in enumerate(zip(channels, depths)):
            blocks = [RDBlock(c, growth, 4, dpr[k + j], rngs=rngs) for j in range(d)]
            k += d
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.downsamples = nnx.List([
            nnx.Sequential(nnx.LayerNorm(channels[i], rngs=rngs),
                           nnx.Conv(channels[i], channels[i + 1], (2, 2), strides=(2, 2), rngs=rngs))
            for i in range(3)])
        self.head_norm = nnx.LayerNorm(channels[-1], rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate)
        self.fc = nnx.Linear(channels[-1], num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.stem_norm(self.stem(x))
        for i, stage in enumerate(self.stages):
            if i > 0:
                x = self.downsamples[i - 1](x)
            for blk in stage:
                x = blk(x)
        return self.head_norm(x)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

_CFGS = {
    "rdnet_tiny": ((96, 192, 384, 768), (2, 2, 6, 2), 64),
    "rdnet_small": ((96, 192, 384, 768), (2, 4, 12, 4), 64),
    "rdnet_base": ((128, 256, 512, 1024), (2, 4, 12, 4), 96),
}

def _make(name):
    channels, depths, growth = _CFGS[name]

    def entry(**kwargs):
        model = RDNet(channels, depths, growth, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
