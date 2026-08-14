"""ConvMixer in flax nnx, NHWC. Mirrors timm.models.convmixer."""
from flax import nnx

from ..layers import global_pool_nhwc
from ..registry import register_model, _cfg


class ConvMixerBlock(nnx.Module):
    def __init__(self, dim, kernel=9, *, rngs):
        self.dw = nnx.Conv(dim, dim, (kernel, kernel), feature_group_count=dim, rngs=rngs)
        self.bn1 = nnx.BatchNorm(dim, rngs=rngs)
        self.pw = nnx.Conv(dim, dim, (1, 1), rngs=rngs)
        self.bn2 = nnx.BatchNorm(dim, rngs=rngs)

    def __call__(self, x):
        y = nnx.gelu(self.bn1(self.dw(x)))
        x = x + y
        return nnx.gelu(self.bn2(self.pw(x)))


class ConvMixer(nnx.Module):
    default_cfg: dict = {}

    def __init__(self, dim=1536, depth=20, patch_size=7, kernel=9, num_classes=1000,
                 in_chans=3, global_pool="avg", *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = dim
        self.stem = nnx.Conv(in_chans, dim, (patch_size, patch_size), strides=(patch_size, patch_size), rngs=rngs)
        self.stem_bn = nnx.BatchNorm(dim, rngs=rngs)
        self.blocks = nnx.List([ConvMixerBlock(dim, kernel, rngs=rngs) for _ in range(depth)])
        self.fc = nnx.Linear(dim, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = nnx.gelu(self.stem_bn(self.stem(x)))
        for blk in self.blocks:
            x = blk(x)
        return x

    def forward_head(self, x):
        x = global_pool_nhwc(x, self.global_pool)
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


@register_model
def convmixer_768_32(**kwargs):
    model = ConvMixer(768, 32, patch_size=7, kernel=9, **kwargs)
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
