"""Inception-ResNet-v2 in flax nnx, NHWC. Mirrors timm.models.inception_resnet_v2 exactly."""
import jax.numpy as jnp
from flax import nnx

from ..layers import ConvBNAct, ClassifierMixin
from ..registry import register_model, _cfg

def _pool3(x):
    return nnx.max_pool(x, (3, 3), strides=(2, 2), padding="VALID")

class Mixed5b(nnx.Module):
    def __init__(self, *, rngs):
        self.branch0 = ConvBNAct(192, 96, 1, rngs=rngs)
        self.branch1 = nnx.List([ConvBNAct(192, 48, 1, rngs=rngs), ConvBNAct(48, 64, 5, rngs=rngs)])
        self.branch2 = nnx.List([ConvBNAct(192, 64, 1, rngs=rngs), ConvBNAct(64, 96, 3, rngs=rngs),
                                 ConvBNAct(96, 96, 3, rngs=rngs)])
        self.branch3 = ConvBNAct(192, 64, 1, rngs=rngs)

    def __call__(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1[1](self.branch1[0](x))
        x2 = self.branch2[2](self.branch2[1](self.branch2[0](x)))
        x3 = self.branch3(nnx.avg_pool(x, (3, 3), strides=(1, 1), padding="SAME"))
        return jnp.concatenate([x0, x1, x2, x3], axis=-1)

class Block35(nnx.Module):
    def __init__(self, scale=0.17, *, rngs):
        self.scale = scale
        self.branch0 = ConvBNAct(320, 32, 1, rngs=rngs)
        self.branch1 = nnx.List([ConvBNAct(320, 32, 1, rngs=rngs), ConvBNAct(32, 32, 3, rngs=rngs)])
        self.branch2 = nnx.List([ConvBNAct(320, 32, 1, rngs=rngs), ConvBNAct(32, 48, 3, rngs=rngs),
                                 ConvBNAct(48, 64, 3, rngs=rngs)])
        self.conv2d = nnx.Conv(128, 320, (1, 1), rngs=rngs)

    def __call__(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1[1](self.branch1[0](x))
        x2 = self.branch2[2](self.branch2[1](self.branch2[0](x)))
        out = self.conv2d(jnp.concatenate([x0, x1, x2], axis=-1))
        return nnx.relu(x + out * self.scale)

class Mixed6a(nnx.Module):
    def __init__(self, *, rngs):
        self.branch0 = ConvBNAct(320, 384, 3, 2, padding="VALID", rngs=rngs)
        self.branch1 = nnx.List([ConvBNAct(320, 256, 1, rngs=rngs), ConvBNAct(256, 256, 3, rngs=rngs),
                                 ConvBNAct(256, 384, 3, 2, padding="VALID", rngs=rngs)])

    def __call__(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1[2](self.branch1[1](self.branch1[0](x)))
        return jnp.concatenate([x0, x1, _pool3(x)], axis=-1)

class Block17(nnx.Module):
    def __init__(self, scale=0.10, *, rngs):
        self.scale = scale
        self.branch0 = ConvBNAct(1088, 192, 1, rngs=rngs)
        self.branch1 = nnx.List([ConvBNAct(1088, 128, 1, rngs=rngs),
                                 ConvBNAct(128, 160, (1, 7), rngs=rngs),
                                 ConvBNAct(160, 192, (7, 1), rngs=rngs)])
        self.conv2d = nnx.Conv(384, 1088, (1, 1), rngs=rngs)

    def __call__(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1[2](self.branch1[1](self.branch1[0](x)))
        out = self.conv2d(jnp.concatenate([x0, x1], axis=-1))
        return nnx.relu(x + out * self.scale)

class Mixed7a(nnx.Module):
    def __init__(self, *, rngs):
        self.branch0 = nnx.List([ConvBNAct(1088, 256, 1, rngs=rngs),
                                 ConvBNAct(256, 384, 3, 2, padding="VALID", rngs=rngs)])
        self.branch1 = nnx.List([ConvBNAct(1088, 256, 1, rngs=rngs),
                                 ConvBNAct(256, 288, 3, 2, padding="VALID", rngs=rngs)])
        self.branch2 = nnx.List([ConvBNAct(1088, 256, 1, rngs=rngs), ConvBNAct(256, 288, 3, rngs=rngs),
                                 ConvBNAct(288, 320, 3, 2, padding="VALID", rngs=rngs)])

    def __call__(self, x):
        x0 = self.branch0[1](self.branch0[0](x))
        x1 = self.branch1[1](self.branch1[0](x))
        x2 = self.branch2[2](self.branch2[1](self.branch2[0](x)))
        return jnp.concatenate([x0, x1, x2, _pool3(x)], axis=-1)

class Block8(nnx.Module):
    def __init__(self, scale=0.20, no_relu=False, *, rngs):
        self.scale, self.no_relu = scale, no_relu
        self.branch0 = ConvBNAct(2080, 192, 1, rngs=rngs)
        self.branch1 = nnx.List([ConvBNAct(2080, 192, 1, rngs=rngs),
                                 ConvBNAct(192, 224, (1, 3), rngs=rngs),
                                 ConvBNAct(224, 256, (3, 1), rngs=rngs)])
        self.conv2d = nnx.Conv(448, 2080, (1, 1), rngs=rngs)

    def __call__(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1[2](self.branch1[1](self.branch1[0](x)))
        out = self.conv2d(jnp.concatenate([x0, x1], axis=-1))
        x = x + out * self.scale
        return x if self.no_relu else nnx.relu(x)

class InceptionResNetV2(ClassifierMixin, nnx.Module):
    default_cfg: dict | None = None

    def __init__(self, num_classes=1000, in_chans=3, global_pool="avg", drop_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.conv2d_1a = ConvBNAct(in_chans, 32, 3, 2, padding="VALID", rngs=rngs)
        self.conv2d_2a = ConvBNAct(32, 32, 3, padding="VALID", rngs=rngs)
        self.conv2d_2b = ConvBNAct(32, 64, 3, rngs=rngs)
        self.conv2d_3b = ConvBNAct(64, 80, 1, rngs=rngs)
        self.conv2d_4a = ConvBNAct(80, 192, 3, padding="VALID", rngs=rngs)
        self.mixed_5b = Mixed5b(rngs=rngs)
        self.repeat = nnx.List([Block35(rngs=rngs) for _ in range(10)])
        self.mixed_6a = Mixed6a(rngs=rngs)
        self.repeat_1 = nnx.List([Block17(rngs=rngs) for _ in range(20)])
        self.mixed_7a = Mixed7a(rngs=rngs)
        self.repeat_2 = nnx.List([Block8(rngs=rngs) for _ in range(9)])
        self.block8 = Block8(no_relu=True, rngs=rngs)
        self.conv2d_7b = ConvBNAct(2080, 1536, 1, rngs=rngs)
        self.num_features = 1536
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(1536, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.conv2d_2b(self.conv2d_2a(self.conv2d_1a(x)))
        x = _pool3(x)
        x = self.conv2d_4a(self.conv2d_3b(x))
        x = _pool3(x)
        x = self.mixed_5b(x)
        for blk in self.repeat:
            x = blk(x)
        x = self.mixed_6a(x)
        for blk in self.repeat_1:
            x = blk(x)
        x = self.mixed_7a(x)
        for blk in self.repeat_2:
            x = blk(x)
        x = self.block8(x)
        return self.conv2d_7b(x)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

@register_model
def inception_resnet_v2(**kwargs):
    model = InceptionResNetV2(**kwargs)
    model.default_cfg = _cfg(input_size=(3, 299, 299))
    return model
