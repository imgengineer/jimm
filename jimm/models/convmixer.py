"""ConvMixer in flax nnx, NHWC. Mirrors timm.models.convmixer."""

from flax import nnx

from ..layers import BatchNorm, ClassifierMixin, gelu
from ..registry import _cfg, register_model


class ConvMixerBlock(nnx.Module):
    """Residual depthwise mixing, then pointwise mixing; timm activates before each norm."""

    def __init__(self, dim, kernel=9, act=gelu, *, rngs):
        self.dw = nnx.Conv(dim, dim, (kernel, kernel), feature_group_count=dim, rngs=rngs)
        self.bn1 = BatchNorm(dim, rngs=rngs)
        self.pw = nnx.Conv(dim, dim, (1, 1), rngs=rngs)
        self.bn2 = BatchNorm(dim, rngs=rngs)
        self.act = act

    def __call__(self, x):
        x = x + self.bn1(self.act(self.dw(x)))
        return self.bn2(self.act(self.pw(x)))


class ConvMixer(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        dim=1536,
        depth=20,
        patch_size=7,
        kernel=9,
        act=gelu,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = dim
        self.stem = nnx.Conv(
            in_chans, dim, (patch_size, patch_size), strides=(patch_size, patch_size), rngs=rngs
        )
        self.stem_bn = BatchNorm(dim, rngs=rngs)
        self.act = act
        self.blocks = nnx.List([ConvMixerBlock(dim, kernel, act, rngs=rngs) for _ in range(depth)])
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(dim, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.stem_bn(self.act(self.stem(x)))
        for blk in self.blocks:
            x = blk(x)
        return x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


@register_model
def convmixer_768_32(**kwargs):
    model = ConvMixer(768, 32, patch_size=7, kernel=7, act=nnx.relu, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def convmixer_1024_20(**kwargs):
    model = ConvMixer(1024, 20, patch_size=14, kernel=9, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def convmixer_1536_20(**kwargs):
    model = ConvMixer(1536, 20, patch_size=7, kernel=9, **kwargs)
    model.default_cfg = _cfg()
    return model
