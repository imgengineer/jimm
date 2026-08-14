"""MambaOut in flax nnx, NHWC. Mirrors timm.models.mambaout (ConvNeXt-style, no SSM)."""
import jax.numpy as jnp
from flax import nnx

from ..layers import DropPath, global_pool_nhwc
from ..registry import register_model, _cfg


class MambaOutBlock(nnx.Module):
    """dw 7x7 -> norm -> 1x1 -> gelu -> 1x1 (ConvNeXt-ish token mixer)."""

    def __init__(self, dim, mlp_ratio=4.0, drop_path=0.0, *, rngs):
        self.dw = nnx.Conv(dim, dim, (7, 7), feature_group_count=dim, rngs=rngs)
        self.norm = nnx.LayerNorm(dim, rngs=rngs)
        self.fc1 = nnx.Linear(dim, int(dim * mlp_ratio), rngs=rngs)
        self.fc2 = nnx.Linear(int(dim * mlp_ratio), dim, rngs=rngs)
        self.drop_path = DropPath(drop_path)

    def __call__(self, x):
        y = self.dw(x)
        y = self.norm(y)
        y = self.fc2(nnx.gelu(self.fc1(y)))
        return x + self.drop_path(y)


class MambaOut(nnx.Module):
    default_cfg: dict = {}

    def __init__(self, channels=(96, 192, 384, 768), depths=(3, 3, 9, 3), num_classes=1000,
                 in_chans=3, global_pool="avg", drop_rate=0.0, drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = channels[-1]
        self.stem = nnx.Conv(in_chans, channels[0], (4, 4), strides=(4, 4), rngs=rngs)
        self.stem_norm = nnx.LayerNorm(channels[0], rngs=rngs)
        dpr = [drop_path_rate * i / max(sum(depths) - 1, 1) for i in range(sum(depths))]
        stages, k = [], 0
        for i, (c, d) in enumerate(zip(channels, depths)):
            blocks = [MambaOutBlock(c, 4.0, dpr[k + j], rngs=rngs) for j in range(d)]
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
    "mambaout_femto": ((48, 96, 192, 384), (3, 3, 9, 3)),
    "mambaout_kobe": ((64, 128, 256, 512), (3, 3, 9, 3)),
    "mambaout_tiny": ((96, 192, 384, 768), (3, 3, 9, 3)),
}


def _make(name):
    channels, depths = _CFGS[name]

    def entry(**kwargs):
        model = MambaOut(channels, depths, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
