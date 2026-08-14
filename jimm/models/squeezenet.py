"""SqueezeNet in flax nnx, NHWC. Mirrors timm.models.squeezenet."""
from flax import nnx

from ..layers import ClassifierMixin
from ..registry import register_model, _cfg

class Fire(nnx.Module):
    def __init__(self, in_chs, squeeze, expand, *, rngs):
        self.squeeze = nnx.Conv(in_chs, squeeze, (1, 1), rngs=rngs)
        self.expand1x1 = nnx.Conv(squeeze, expand, (1, 1), rngs=rngs)
        self.expand3x3 = nnx.Conv(squeeze, expand, (3, 3), rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.squeeze(x))
        import jax.numpy as jnp
        return jnp.concatenate([nnx.relu(self.expand1x1(x)), nnx.relu(self.expand3x3(x))], axis=-1)

class SqueezeNet(ClassifierMixin, nnx.Module):
    default_cfg: dict = {}

    def __init__(self, version="1_0", num_classes=1000, in_chans=3, global_pool="avg",
                 drop_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        if version == "1_0":
            self.conv1 = nnx.Conv(in_chans, 96, (7, 7), strides=(2, 2), rngs=rngs)
            cfg = [(96, 16, 64), (128, 16, 64), (128, 32, 128), "M",
                   (256, 32, 128), "M", (256, 48, 192), (384, 48, 192),
                   (384, 64, 256), "M", (512, 64, 256)]
        else:
            self.conv1 = nnx.Conv(in_chans, 64, (3, 3), strides=(2, 2), rngs=rngs)
            cfg = [(64, 16, 64), (128, 16, 64), "M", (128, 32, 128), (256, 32, 128), "M",
                   (256, 48, 192), (384, 48, 192), (384, 64, 256), (512, 64, 256), "M"]
        blocks = []
        for item in cfg:
            if item == "M":
                blocks.append("M")
            else:
                in_chs, sq, ex = item
                blocks.append(Fire(in_chs, sq, ex, rngs=rngs))
        self.features = nnx.List(blocks)
        self.num_features = 512
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(512, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = nnx.max_pool(nnx.relu(self.conv1(x)), (3, 3), strides=(2, 2), padding="SAME")
        for b in self.features:
            x = nnx.max_pool(x, (3, 3), strides=(2, 2), padding="SAME") if b == "M" else b(x)
        return x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

@register_model
def squeezenet1_0(**kwargs):
    model = SqueezeNet("1_0", **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def squeezenet1_1(**kwargs):
    model = SqueezeNet("1_1", **kwargs)
    model.default_cfg = _cfg()
    return model
