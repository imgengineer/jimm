"""EdgeNeXt in flax nnx, NHWC. Mirrors timm.models.edgenext (split depthwise transpose encoder)."""
import jax.numpy as jnp
from flax import nnx

from ..layers import DropPath, global_pool_nhwc
from ..registry import register_model, _cfg


class SDTEBlock(nnx.Module):
    """Split-Depthwise-Transpose encoder: dw conv on channel-split parts + pointwise MLP."""

    def __init__(self, dim, expand=4, drop_path=0.0, *, rngs):
        self.dw1 = nnx.Conv(dim // 2, dim // 2, (3, 3), feature_group_count=dim // 2, rngs=rngs)
        self.dw2 = nnx.Conv(dim // 4, dim // 4, (5, 5), feature_group_count=dim // 4, rngs=rngs)
        self.pw = nnx.Conv(dim, dim, (1, 1), rngs=rngs)
        self.norm = nnx.LayerNorm(dim, rngs=rngs)
        self.fc1 = nnx.Linear(dim, dim * expand, rngs=rngs)
        self.fc2 = nnx.Linear(dim * expand, dim, rngs=rngs)
        self.drop_path = DropPath(drop_path)

    def __call__(self, x):
        x1, x2, x3 = jnp.split(x, [x.shape[-1] // 2, 3 * x.shape[-1] // 4], axis=-1)
        x1 = self.dw1(x1)
        x2 = self.dw2(x2)
        y = self.pw(jnp.concatenate([x1, x2, x3], axis=-1))
        x = x + self.drop_path(y)
        return x + self.drop_path(self.fc2(nnx.gelu(self.fc1(self.norm(x)))))


class EdgeNeXt(nnx.Module):
    default_cfg: dict = {}

    def __init__(self, channels=(48, 96, 160, 304), depths=(3, 3, 9, 3), num_classes=1000,
                 in_chans=3, global_pool="avg", drop_rate=0.0, drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = channels[-1]
        self.stem = nnx.Conv(in_chans, channels[0], (4, 4), strides=(4, 4), rngs=rngs)
        self.stem_norm = nnx.LayerNorm(channels[0], rngs=rngs)
        dpr = [drop_path_rate * i / max(sum(depths) - 1, 1) for i in range(sum(depths))]
        stages, k = [], 0
        for i, (c, d) in enumerate(zip(channels, depths)):
            blocks = [SDTEBlock(c, 4, dpr[k + j], rngs=rngs) for j in range(d)]
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

    def forward_head(self, x):
        x = global_pool_nhwc(x, self.global_pool)
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


_CFGS = {
    "edgenext_xx_small": ((48, 96, 160, 304), (3, 3, 9, 3)),
    "edgenext_x_small": ((64, 128, 256, 512), (3, 3, 9, 3)),
    "edgenext_small": ((80, 160, 296, 552), (3, 3, 9, 3)),
}


def _make(name):
    channels, depths = _CFGS[name]

    def entry(**kwargs):
        model = EdgeNeXt(channels, depths, **kwargs)
        model.default_cfg = _cfg(input_size=(3, 256, 256))
        return model
    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
