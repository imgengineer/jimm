"""InceptionV3 in flax nnx, NHWC. Mirrors timm.models.inception_v3 (aux classifier included)."""
import jax.numpy as jnp
from flax import nnx

from ..layers import ConvBNAct, ClassifierMixin
from ..registry import register_model, _cfg

class InceptionA(nnx.Module):  # 35x35 grid
    def __init__(self, in_chs, pool_chs, *, rngs):
        self.b1 = ConvBNAct(in_chs, 64, 1, rngs=rngs)
        self.b2 = nnx.List([ConvBNAct(in_chs, 48, 1, rngs=rngs), ConvBNAct(48, 64, 5, rngs=rngs)])
        self.b3 = nnx.List([ConvBNAct(in_chs, 64, 1, rngs=rngs), ConvBNAct(64, 96, 3, rngs=rngs),
                            ConvBNAct(96, 96, 3, rngs=rngs)])
        self.b4 = ConvBNAct(in_chs, pool_chs, 1, rngs=rngs)

    def __call__(self, x):
        b2 = self.b2[1](self.b2[0](x))
        b3 = self.b3[2](self.b3[1](self.b3[0](x)))
        b4 = self.b4(nnx.avg_pool(x, (3, 3), strides=(1, 1), padding="SAME"))
        return jnp.concatenate([self.b1(x), b2, b3, b4], axis=-1)

class InceptionB(nnx.Module):  # 35 -> 17 reduction (VALID padding, per original)
    def __init__(self, in_chs, *, rngs):
        self.b1 = ConvBNAct(in_chs, 384, 3, 2, padding="VALID", rngs=rngs)
        self.b2 = nnx.List([ConvBNAct(in_chs, 64, 1, rngs=rngs), ConvBNAct(64, 96, 3, rngs=rngs),
                            ConvBNAct(96, 96, 3, 2, padding="VALID", rngs=rngs)])

    def __call__(self, x):
        b2 = self.b2[2](self.b2[1](self.b2[0](x)))
        b3 = nnx.max_pool(x, (3, 3), strides=(2, 2), padding="VALID")
        return jnp.concatenate([self.b1(x), b2, b3], axis=-1)

class InceptionC(nnx.Module):  # 17x17 grid
    def __init__(self, in_chs, ch7, *, rngs):
        self.b1 = ConvBNAct(in_chs, 192, 1, rngs=rngs)
        self.b2 = nnx.List([ConvBNAct(in_chs, ch7, 1, rngs=rngs),
                            ConvBNAct(ch7, ch7, (1, 7), rngs=rngs), ConvBNAct(ch7, 192, (7, 1), rngs=rngs)])
        self.b3 = nnx.List([ConvBNAct(in_chs, ch7, 1, rngs=rngs), ConvBNAct(ch7, ch7, (7, 1), rngs=rngs),
                            ConvBNAct(ch7, ch7, (1, 7), rngs=rngs), ConvBNAct(ch7, ch7, (7, 1), rngs=rngs),
                            ConvBNAct(ch7, 192, (1, 7), rngs=rngs)])
        self.b4 = ConvBNAct(in_chs, 192, 1, rngs=rngs)

    def __call__(self, x):
        b2 = self.b2[2](self.b2[1](self.b2[0](x)))
        b3 = self.b3[4](self.b3[3](self.b3[2](self.b3[1](self.b3[0](x)))))
        b4 = self.b4(nnx.avg_pool(x, (3, 3), strides=(1, 1), padding="SAME"))
        return jnp.concatenate([self.b1(x), b2, b3, b4], axis=-1)

class InceptionD(nnx.Module):  # 17 -> 8 reduction (VALID padding, per original)
    def __init__(self, in_chs, *, rngs):
        self.b1 = nnx.List([ConvBNAct(in_chs, 192, 1, rngs=rngs),
                            ConvBNAct(192, 320, 3, 2, padding="VALID", rngs=rngs)])
        self.b2 = nnx.List([ConvBNAct(in_chs, 192, 1, rngs=rngs), ConvBNAct(192, 192, (1, 7), rngs=rngs),
                            ConvBNAct(192, 192, (7, 1), rngs=rngs),
                            ConvBNAct(192, 192, 3, 2, padding="VALID", rngs=rngs)])

    def __call__(self, x):
        b1 = self.b1[1](self.b1[0](x))
        b2 = self.b2[3](self.b2[2](self.b2[1](self.b2[0](x))))
        b3 = nnx.max_pool(x, (3, 3), strides=(2, 2), padding="VALID")
        return jnp.concatenate([b1, b2, b3], axis=-1)

class InceptionE(nnx.Module):  # 8x8 grid
    def __init__(self, in_chs, *, rngs):
        self.b1 = ConvBNAct(in_chs, 320, 1, rngs=rngs)
        self.b2a = nnx.List([ConvBNAct(in_chs, 384, 1, rngs=rngs), ConvBNAct(384, 384, (1, 3), rngs=rngs)])
        self.b2b = nnx.List([ConvBNAct(in_chs, 384, 1, rngs=rngs), ConvBNAct(384, 384, (3, 1), rngs=rngs)])
        self.b3a = nnx.List([ConvBNAct(in_chs, 448, 1, rngs=rngs), ConvBNAct(448, 384, 3, rngs=rngs),
                             ConvBNAct(384, 384, (1, 3), rngs=rngs)])
        self.b3b = nnx.List([ConvBNAct(in_chs, 448, 1, rngs=rngs), ConvBNAct(448, 384, 3, rngs=rngs),
                             ConvBNAct(384, 384, (3, 1), rngs=rngs)])
        self.b4 = ConvBNAct(in_chs, 192, 1, rngs=rngs)

    def __call__(self, x):
        b2a = self.b2a[1](self.b2a[0](x))
        b2b = self.b2b[1](self.b2b[0](x))
        b3a = self.b3a[2](self.b3a[1](self.b3a[0](x)))
        b3b = self.b3b[2](self.b3b[1](self.b3b[0](x)))
        b4 = self.b4(nnx.avg_pool(x, (3, 3), strides=(1, 1), padding="SAME"))
        return jnp.concatenate([self.b1(x), b2a, b2b, b3a, b3b, b4], axis=-1)

class InceptionV3(ClassifierMixin, nnx.Module):
    default_cfg: dict = {}

    def __init__(self, num_classes=1000, in_chans=3, global_pool="avg", drop_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.stem = nnx.List([
            ConvBNAct(in_chans, 32, 3, 2, padding="VALID", rngs=rngs),   # 299 -> 149
            ConvBNAct(32, 32, 3, padding="VALID", rngs=rngs),            # 149 -> 147
            ConvBNAct(32, 64, 3, rngs=rngs),                             # SAME 147
        ])
        self.pool0 = lambda x: nnx.max_pool(x, (3, 3), strides=(2, 2), padding="VALID")
        self.conv1 = ConvBNAct(64, 80, 1, padding="VALID", rngs=rngs)    # 73 -> 73
        self.conv2 = ConvBNAct(80, 192, 3, padding="VALID", rngs=rngs)   # 73 -> 71
        self.mixed_5b = InceptionA(192, 32, rngs=rngs)
        self.mixed_5c = InceptionA(256, 64, rngs=rngs)
        self.mixed_5d = InceptionA(288, 64, rngs=rngs)
        self.mixed_6a = InceptionB(288, rngs=rngs)
        self.mixed_6b = InceptionC(768, 128, rngs=rngs)
        self.mixed_6c = InceptionC(768, 160, rngs=rngs)
        self.mixed_6d = InceptionC(768, 160, rngs=rngs)
        self.mixed_6e = InceptionC(768, 192, rngs=rngs)
        self.aux_fc = nnx.Linear(768, num_classes, rngs=rngs) if num_classes > 0 else None
        self.mixed_7a = InceptionD(768, rngs=rngs)
        self.mixed_7b = InceptionE(1280, rngs=rngs)
        self.mixed_7c = InceptionE(2048, rngs=rngs)
        self.num_features = 2048
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(2048, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        for layer in self.stem:
            x = layer(x)
        x = self.pool0(x)
        x = self.conv2(self.conv1(x))
        x = self.pool0(x)
        x = self.mixed_5d(self.mixed_5c(self.mixed_5b(x)))
        x = self.mixed_6a(x)
        x = self.mixed_6e(self.mixed_6d(self.mixed_6c(self.mixed_6b(x))))
        x = self.mixed_7a(x)
        return self.mixed_7c(self.mixed_7b(x))

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

@register_model
def inception_v3(**kwargs):
    model = InceptionV3(**kwargs)
    model.default_cfg = _cfg(input_size=(3, 299, 299), crop_pct=0.875)
    return model
