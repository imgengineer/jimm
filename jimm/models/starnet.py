"""StarNet in flax nnx, NHWC. Mirrors timm.models.starnet (elementwise-mul blocks)."""
import jax.numpy as jnp
from flax import nnx

from ..layers import ConvBNAct, DropPath, global_pool_nhwc
from ..registry import register_model, _cfg


class StarBlock(nnx.Module):
    """dw 7x7 -> two 1x1 branches -> elementwise mul -> 1x1 (star operation)."""

    def __init__(self, dim, mlp_ratio=4, drop_path=0.0, *, rngs):
        self.dw = nnx.Conv(dim, dim, (7, 7), use_bias=False, feature_group_count=dim, rngs=rngs)
        self.bn_dw = nnx.BatchNorm(dim, rngs=rngs)
        self.f1 = nnx.Conv(dim, dim * mlp_ratio, (1, 1), rngs=rngs)
        self.f2 = nnx.Conv(dim, dim * mlp_ratio, (1, 1), rngs=rngs)
        self.g = nnx.Conv(dim * mlp_ratio, dim, (1, 1), use_bias=False, rngs=rngs)
        self.bn = nnx.BatchNorm(dim, rngs=rngs)
        self.drop_path = DropPath(drop_path)

    def __call__(self, x):
        y = self.bn_dw(self.dw(x))
        y = self.f1(y) * nnx.relu(self.f2(y))
        y = self.bn(self.g(y))
        return x + self.drop_path(y)


class StarNet(nnx.Module):
    default_cfg: dict = {}

    def __init__(self, channels=(48, 96, 192, 384), depths=(3, 3, 12, 5), num_classes=1000,
                 in_chans=3, global_pool="avg", drop_rate=0.0, drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = channels[-1]
        self.stem = nnx.List([ConvBNAct(in_chans, channels[0] // 2, 3, 2, rngs=rngs),
                              ConvBNAct(channels[0] // 2, channels[0], 3, 2, rngs=rngs)])
        dpr = [drop_path_rate * i / max(sum(depths) - 1, 1) for i in range(sum(depths))]
        stages, chs, k = [], channels[0], 0
        for i, (c, d) in enumerate(zip(channels, depths)):
            blocks = []
            if i > 0:
                blocks.append(ConvBNAct(chs, c, 3, 2, act="identity", rngs=rngs))
            for _ in range(d):
                blocks.append(StarBlock(c, 4, dpr[k], rngs=rngs))
                k += 1
            stages.append(nnx.List(blocks))
            chs = c
        self.stages = nnx.List(stages)
        self.head_drop = nnx.Dropout(drop_rate)
        self.fc = nnx.Linear(channels[-1], num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        for layer in self.stem:
            x = layer(x)
        for stage in self.stages:
            for blk in stage:
                x = blk(x)
        return x

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
    "starnet_s050": ((32, 64, 128, 256), (1, 1, 3, 1)),
    "starnet_s1": ((48, 96, 192, 384), (2, 2, 6, 2)),
    "starnet_s2": ((64, 128, 256, 512), (2, 3, 8, 3)),
}


def _make(name):
    channels, depths = _CFGS[name]

    def entry(**kwargs):
        model = StarNet(channels, depths, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
