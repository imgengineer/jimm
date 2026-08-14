"""InceptionV4 in flax nnx, NHWC. Mirrors timm.models.inception_v4 (uniform 3x3 stem + A/B/C blocks)."""
import jax.numpy as jnp
from flax import nnx

from ..layers import ConvBNAct, ClassifierMixin
from ..registry import register_model, _cfg

def _pool3(x):
    return nnx.max_pool(x, (3, 3), strides=(2, 2), padding="VALID")

class InceptionA(nnx.Module):
    def __init__(self, in_chs, *, rngs):
        self.b1 = ConvBNAct(in_chs, 96, 1, rngs=rngs)
        self.b2 = nnx.List([ConvBNAct(in_chs, 64, 1, rngs=rngs), ConvBNAct(64, 96, 3, rngs=rngs)])
        self.b3 = nnx.List([ConvBNAct(in_chs, 64, 1, rngs=rngs), ConvBNAct(64, 96, 3, rngs=rngs),
                            ConvBNAct(96, 96, 3, rngs=rngs)])
        self.b4 = ConvBNAct(in_chs, 96, 1, rngs=rngs)

    def __call__(self, x):
        b2 = self.b2[1](self.b2[0](x))
        b3 = self.b3[2](self.b3[1](self.b3[0](x)))
        b4 = self.b4(nnx.avg_pool(x, (3, 3), strides=(1, 1), padding="SAME"))
        return jnp.concatenate([self.b1(x), b2, b3, b4], axis=-1)

class ReductionA(nnx.Module):  # 35 -> 17 (or 71 -> 35)
    def __init__(self, in_chs, k, l, m, n, *, rngs):
        self.b1 = ConvBNAct(in_chs, n, 3, 2, padding="VALID", rngs=rngs)
        self.b2 = nnx.List([ConvBNAct(in_chs, k, 1, rngs=rngs), ConvBNAct(k, l, 3, rngs=rngs),
                            ConvBNAct(l, m, 3, 2, padding="VALID", rngs=rngs)])

    def __call__(self, x):
        b2 = self.b2[2](self.b2[1](self.b2[0](x)))
        return jnp.concatenate([self.b1(x), b2, _pool3(x)], axis=-1)

class InceptionB(nnx.Module):
    def __init__(self, in_chs, *, rngs):
        self.b1 = ConvBNAct(in_chs, 384, 1, rngs=rngs)
        self.b2 = nnx.List([ConvBNAct(in_chs, 192, 1, rngs=rngs), ConvBNAct(192, 224, (1, 7), rngs=rngs),
                            ConvBNAct(224, 256, (7, 1), rngs=rngs)])
        self.b3 = nnx.List([ConvBNAct(in_chs, 192, 1, rngs=rngs), ConvBNAct(192, 192, (7, 1), rngs=rngs),
                            ConvBNAct(192, 224, (1, 7), rngs=rngs), ConvBNAct(224, 224, (7, 1), rngs=rngs),
                            ConvBNAct(224, 256, (1, 7), rngs=rngs)])
        self.b4 = ConvBNAct(in_chs, 128, 1, rngs=rngs)

    def __call__(self, x):
        b2 = self.b2[2](self.b2[1](self.b2[0](x)))
        b3 = self.b3[4](self.b3[3](self.b3[2](self.b3[1](self.b3[0](x)))))
        b4 = self.b4(nnx.avg_pool(x, (3, 3), strides=(1, 1), padding="SAME"))
        return jnp.concatenate([self.b1(x), b2, b3, b4], axis=-1)

class ReductionB(nnx.Module):  # 17 -> 8
    def __init__(self, in_chs, *, rngs):
        self.b1 = nnx.List([ConvBNAct(in_chs, 192, 1, rngs=rngs),
                            ConvBNAct(192, 192, 3, 2, padding="VALID", rngs=rngs)])
        self.b2 = nnx.List([ConvBNAct(in_chs, 256, 1, rngs=rngs), ConvBNAct(256, 256, (1, 7), rngs=rngs),
                            ConvBNAct(256, 320, (7, 1), rngs=rngs),
                            ConvBNAct(320, 320, 3, 2, padding="VALID", rngs=rngs)])

    def __call__(self, x):
        b1 = self.b1[1](self.b1[0](x))
        b2 = self.b2[3](self.b2[2](self.b2[1](self.b2[0](x))))
        return jnp.concatenate([b1, b2, _pool3(x)], axis=-1)

class InceptionC(nnx.Module):
    def __init__(self, in_chs, *, rngs):
        self.b1 = ConvBNAct(in_chs, 256, 1, rngs=rngs)
        self.b2a = ConvBNAct(in_chs, 384, 1, rngs=rngs)
        self.b2a1 = ConvBNAct(384, 256, (1, 3), rngs=rngs)
        self.b2a2 = ConvBNAct(384, 256, (3, 1), rngs=rngs)
        self.b3a = nnx.List([ConvBNAct(in_chs, 384, 1, rngs=rngs), ConvBNAct(384, 448, (3, 1), rngs=rngs),
                             ConvBNAct(448, 512, (1, 3), rngs=rngs)])
        self.b3a1 = ConvBNAct(512, 256, (1, 3), rngs=rngs)
        self.b3a2 = ConvBNAct(512, 256, (3, 1), rngs=rngs)
        self.b4 = ConvBNAct(in_chs, 256, 1, rngs=rngs)

    def __call__(self, x):
        b2a = self.b2a1(self.b2a(x))
        b2b = self.b2a2(self.b2a(x))
        b3h = self.b3a[2](self.b3a[1](self.b3a[0](x)))
        b3a = self.b3a1(b3h)
        b3b = self.b3a2(b3h)
        b4 = self.b4(nnx.avg_pool(x, (3, 3), strides=(1, 1), padding="SAME"))
        return jnp.concatenate([self.b1(x), b2a, b2b, b3a, b3b, b4], axis=-1)

class InceptionV4(ClassifierMixin, nnx.Module):
    default_cfg: dict = {}

    def __init__(self, num_classes=1000, in_chans=3, global_pool="avg", drop_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.stem = nnx.List([  # 299 -> 35
            ConvBNAct(in_chans, 32, 3, 2, padding="VALID", rngs=rngs),
            ConvBNAct(32, 32, 3, padding="VALID", rngs=rngs),
            ConvBNAct(32, 64, 3, rngs=rngs),
        ])
        # mixed 3a/4a/5a stem blocks (inception-v4 specific)
        self.s3a_b1 = lambda x: _pool3(x)
        self.s3a_b2 = ConvBNAct(64, 96, 3, 2, padding="VALID", rngs=rngs)
        self.s4a_b1 = nnx.List([ConvBNAct(160, 64, 1, rngs=rngs),
                                ConvBNAct(64, 96, 3, padding="VALID", rngs=rngs)])
        self.s4a_b2 = nnx.List([ConvBNAct(160, 64, 1, rngs=rngs), ConvBNAct(64, 64, (7, 1), rngs=rngs),
                                ConvBNAct(64, 64, (1, 7), rngs=rngs),
                                ConvBNAct(64, 96, 3, padding="VALID", rngs=rngs)])
        self.s5a_b1 = ConvBNAct(192, 192, 3, 2, padding="VALID", rngs=rngs)
        self.s5a_b2 = lambda x: _pool3(x)
        self.mixed_a = nnx.List([InceptionA(384, rngs=rngs) for _ in range(4)])
        self.red_a = ReductionA(384, 192, 224, 256, 384, rngs=rngs)
        self.mixed_b = nnx.List([InceptionB(1024, rngs=rngs) for _ in range(7)])
        self.red_b = ReductionB(1024, rngs=rngs)
        self.mixed_c = nnx.List([InceptionC(1536, rngs=rngs) for _ in range(3)])
        self.num_features = 1536
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(1536, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        for layer in self.stem:
            x = layer(x)
        x = jnp.concatenate([self.s3a_b1(x), self.s3a_b2(x)], axis=-1)
        x = jnp.concatenate([self.s4a_b1[1](self.s4a_b1[0](x)),
                             self.s4a_b2[3](self.s4a_b2[2](self.s4a_b2[1](self.s4a_b2[0](x))))], axis=-1)
        x = jnp.concatenate([self.s5a_b1(x), self.s5a_b2(x)], axis=-1)
        for blk in self.mixed_a:
            x = blk(x)
        x = self.red_a(x)
        for blk in self.mixed_b:
            x = blk(x)
        x = self.red_b(x)
        for blk in self.mixed_c:
            x = blk(x)
        return x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

@register_model
def inception_v4(**kwargs):
    model = InceptionV4(**kwargs)
    model.default_cfg = _cfg(input_size=(3, 299, 299))
    return model
