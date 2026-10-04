"""DenseNet in flax nnx, NHWC. Mirrors timm.models.densenet."""

import jax.numpy as jnp
from flax import nnx

from ..layers import BatchNorm, ClassifierMixin
from ..registry import _cfg, register_model
from ._efficientnet import BlurPool


class DenseLayer(nnx.Module):
    def __init__(self, in_chs, growth_rate, bn_size=4, *, rngs):
        mid = bn_size * growth_rate
        self.bn1 = BatchNorm(in_chs, rngs=rngs)
        self.conv1 = nnx.Conv(in_chs, mid, (1, 1), use_bias=False, rngs=rngs)
        self.bn2 = BatchNorm(mid, rngs=rngs)
        self.conv2 = nnx.Conv(mid, growth_rate, (3, 3), use_bias=False, rngs=rngs)

    def __call__(self, x):
        y = self.conv1(nnx.relu(self.bn1(x)))
        y = self.conv2(nnx.relu(self.bn2(y)))
        return jnp.concatenate([x, y], axis=-1)


class Transition(nnx.Module):
    def __init__(self, in_chs, out_chs, *, rngs):
        self.bn = BatchNorm(in_chs, rngs=rngs)
        self.conv = nnx.Conv(in_chs, out_chs, (1, 1), use_bias=False, rngs=rngs)

    def __call__(self, x):
        x = self.conv(nnx.relu(self.bn(x)))
        return nnx.avg_pool(x, (2, 2), strides=(2, 2))


class DenseNet(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        growth_rate,
        block_config,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        bn_size=4,
        stem_type="",
        aa_layer=None,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        stem_chs = 2 * growth_rate
        self.deep_stem = "deep" in stem_type
        if self.deep_stem:
            # timm deep stem: three 3x3 convs (the first strided) with norm-act after each.
            self.conv0 = nnx.Conv(
                in_chans, growth_rate, (3, 3), strides=2, padding=1, use_bias=False, rngs=rngs
            )
            self.norm0 = BatchNorm(growth_rate, rngs=rngs)
            self.conv1 = nnx.Conv(
                growth_rate, growth_rate, (3, 3), padding=1, use_bias=False, rngs=rngs
            )
            self.norm1 = BatchNorm(growth_rate, rngs=rngs)
            self.conv2 = nnx.Conv(
                growth_rate, stem_chs, (3, 3), padding=1, use_bias=False, rngs=rngs
            )
            self.norm2 = BatchNorm(stem_chs, rngs=rngs)
        else:
            self.conv0 = nnx.Conv(
                in_chans, stem_chs, (7, 7), strides=(2, 2), padding=[(3, 3), (3, 3)],
                use_bias=False, rngs=rngs,
            )  # fmt: skip
            self.norm0 = BatchNorm(stem_chs, rngs=rngs)
        # Anti-aliased stem pool (timm aa_layer, stem only): stride-1 max pool, then a blur
        # pool with stride 2.
        self.aa = BlurPool(2, "reflect") if aa_layer == "blur" else None
        stages, chs = [], stem_chs
        for i, n in enumerate(block_config):
            layers = []
            for _ in range(n):
                layers.append(DenseLayer(chs, growth_rate, bn_size, rngs=rngs))
                chs += growth_rate
            stages.append(nnx.List(layers))
            if i != len(block_config) - 1:
                stages.append(Transition(chs, chs // 2, rngs=rngs))
                chs //= 2
        self.stages = nnx.List(stages)
        self.norm5 = BatchNorm(chs, rngs=rngs)
        self.num_features = chs
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(chs, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = nnx.relu(self.norm0(self.conv0(x)))
        if self.deep_stem:
            x = nnx.relu(self.norm1(self.conv1(x)))
            x = nnx.relu(self.norm2(self.conv2(x)))
        if self.aa is not None:
            x = self.aa(nnx.max_pool(x, (3, 3), strides=(1, 1), padding=((1, 1), (1, 1))))
        else:
            x = nnx.max_pool(x, (3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))
        for stage in self.stages:
            x = stage(x) if isinstance(stage, Transition) else _run_dense(stage, x)
        return nnx.relu(self.norm5(x))

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


def _run_dense(layers, x):
    for layer in layers:
        x = layer(x)
    return x


_CFGS = {
    # name: (growth rate, block config, extra arguments, test input size)
    "densenet121": (32, (6, 12, 24, 16), {}, 288),
    "densenet161": (48, (6, 12, 36, 24), {}, None),
    "densenet169": (32, (6, 12, 32, 32), {}, None),
    "densenet201": (32, (6, 12, 48, 32), {}, None),
    "densenet264d": (48, (6, 12, 64, 48), dict(stem_type="deep"), None),
    "densenetblur121d": (32, (6, 12, 24, 16), dict(stem_type="deep", aa_layer="blur"), 288),
}


def _make(name):
    growth_rate, block_config, extra, test = _CFGS[name]
    ev = {"test_input_size": (3, test, test)} if test else {}

    def entry(**kwargs):
        model = DenseNet(growth_rate, block_config, **{**extra, **kwargs})
        model.default_cfg = _cfg(interpolation="bicubic", **ev)
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
