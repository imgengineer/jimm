"""DenseNet in flax nnx, NHWC. Mirrors timm.models.densenet."""
import jax.numpy as jnp
from flax import nnx

from ..layers import global_pool_nhwc
from ..registry import register_model, _cfg


class DenseLayer(nnx.Module):
    def __init__(self, in_chs, growth_rate, bn_size=4, *, rngs):
        mid = bn_size * growth_rate
        self.bn1 = nnx.BatchNorm(in_chs, rngs=rngs)
        self.conv1 = nnx.Conv(in_chs, mid, (1, 1), use_bias=False, rngs=rngs)
        self.bn2 = nnx.BatchNorm(mid, rngs=rngs)
        self.conv2 = nnx.Conv(mid, growth_rate, (3, 3), use_bias=False, rngs=rngs)

    def __call__(self, x):
        y = self.conv1(nnx.relu(self.bn1(x)))
        y = self.conv2(nnx.relu(self.bn2(y)))
        return jnp.concatenate([x, y], axis=-1)


class Transition(nnx.Module):
    def __init__(self, in_chs, out_chs, *, rngs):
        self.bn = nnx.BatchNorm(in_chs, rngs=rngs)
        self.conv = nnx.Conv(in_chs, out_chs, (1, 1), use_bias=False, rngs=rngs)

    def __call__(self, x):
        x = self.conv(nnx.relu(self.bn(x)))
        return nnx.avg_pool(x, (2, 2), strides=(2, 2))


class DenseNet(nnx.Module):
    default_cfg: dict = {}

    def __init__(self, growth_rate, block_config, num_classes=1000, in_chans=3,
                 global_pool="avg", drop_rate=0.0, bn_size=4, stem_chs=64, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.conv0 = nnx.Conv(in_chans, stem_chs, (7, 7), strides=(2, 2), padding=[(3, 3), (3, 3)],
                              use_bias=False, rngs=rngs)
        self.norm0 = nnx.BatchNorm(stem_chs, rngs=rngs)
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
        self.norm5 = nnx.BatchNorm(chs, rngs=rngs)
        self.num_features = chs
        self.head_drop = nnx.Dropout(drop_rate)
        self.fc = nnx.Linear(chs, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = nnx.max_pool(nnx.relu(self.norm0(self.conv0(x))), (3, 3), strides=(2, 2), padding="SAME")
        for stage in self.stages:
            x = stage(x) if isinstance(stage, Transition) else _run_dense(stage, x)
        return nnx.relu(self.norm5(x))

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


def _run_dense(layers, x):
    for layer in layers:
        x = layer(x)
    return x


def _densenet(growth_rate, block_config, **kwargs):
    model = DenseNet(growth_rate, block_config, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def densenet121(**kwargs):
    return _densenet(32, [6, 12, 24, 16], **kwargs)


@register_model
def densenet169(**kwargs):
    return _densenet(32, [6, 12, 32, 32], **kwargs)


@register_model
def densenet201(**kwargs):
    return _densenet(32, [6, 12, 48, 32], **kwargs)
