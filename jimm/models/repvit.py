"""RepViT in flax nnx, NHWC. Mirrors timm.models.repvit.

RepVGG-style depthwise token mixers (3x3 conv + BN, 1x1 depthwise conv, and
identity, then BN), squeeze-excite in every other block, 1x1-conv MLPs, and a
BatchNorm + Linear classifier averaged with a distillation head.
"""

from flax import nnx

from ..layers import (
    BatchNorm,
    ClassifierMixin,
    SqueezeExcite,
    gelu,
    global_pool_nhwc,
    make_divisible,
)
from ..registry import _cfg, register_model


class ConvNorm(nnx.Module):
    def __init__(self, in_dim, out_dim, kernel=1, stride=1, groups=1, bn_weight_init=1.0, *, rngs):
        pad = (kernel - 1) // 2
        self.c = nnx.Conv(
            in_dim,
            out_dim,
            (kernel, kernel),
            strides=(stride, stride),
            padding=((pad, pad), (pad, pad)),
            feature_group_count=groups,
            use_bias=False,
            rngs=rngs,
        )
        self.bn = BatchNorm(
            out_dim, epsilon=1e-5, scale_init=nnx.initializers.constant(bn_weight_init), rngs=rngs
        )

    def __call__(self, x):
        return self.bn(self.c(x))


class RepVggDw(nnx.Module):
    def __init__(self, dim, kernel, legacy=False, *, rngs):
        self.conv = ConvNorm(dim, dim, kernel, groups=dim, rngs=rngs)
        if legacy:
            self.conv1 = ConvNorm(dim, dim, 1, groups=dim, rngs=rngs)
            self.bn = None
        else:
            self.conv1 = nnx.Conv(dim, dim, (1, 1), feature_group_count=dim, rngs=rngs)
            self.bn = BatchNorm(dim, epsilon=1e-5, rngs=rngs)

    def __call__(self, x):
        x = self.conv(x) + self.conv1(x) + x
        return x if self.bn is None else self.bn(x)


class RepVitMlp(nnx.Module):
    def __init__(self, dim, hidden, *, rngs):
        self.conv1 = ConvNorm(dim, hidden, rngs=rngs)
        self.conv2 = ConvNorm(hidden, dim, bn_weight_init=0.0, rngs=rngs)

    def __call__(self, x):
        return self.conv2(gelu(self.conv1(x)))


class RepViTBlock(nnx.Module):
    def __init__(self, dim, mlp_ratio, kernel, use_se, legacy=False, *, rngs):
        self.token_mixer = RepVggDw(dim, kernel, legacy, rngs=rngs)
        # timm SqueezeExcite (SEModule): rd = make_divisible(dim / 4, 8), ReLU, sigmoid.
        self.se = (
            SqueezeExcite(dim, rd_channels=make_divisible(dim * 0.25, 8), rngs=rngs)
            if use_se
            else None
        )
        self.channel_mixer = RepVitMlp(dim, dim * mlp_ratio, rngs=rngs)

    def __call__(self, x):
        x = self.token_mixer(x)
        if self.se is not None:
            x = self.se(x)
        return x + self.channel_mixer(x)


class RepVitDownsample(nnx.Module):
    def __init__(self, in_dim, mlp_ratio, out_dim, kernel, legacy=False, *, rngs):
        self.pre_block = RepViTBlock(in_dim, mlp_ratio, kernel, False, legacy, rngs=rngs)
        self.spatial_downsample = ConvNorm(in_dim, in_dim, kernel, 2, groups=in_dim, rngs=rngs)
        self.channel_downsample = ConvNorm(in_dim, out_dim, rngs=rngs)
        self.ffn = RepVitMlp(out_dim, out_dim * mlp_ratio, rngs=rngs)

    def __call__(self, x):
        x = self.channel_downsample(self.spatial_downsample(self.pre_block(x)))
        return x + self.ffn(x)


class RepVitStage(nnx.Module):
    def __init__(self, in_dim, out_dim, depth, mlp_ratio, kernel, downsample, legacy, *, rngs):
        self.downsample = (
            RepVitDownsample(in_dim, mlp_ratio, out_dim, kernel, legacy, rngs=rngs)
            if downsample
            else None
        )
        # Squeeze-excite in every other block, starting with the first.
        self.blocks = nnx.List(
            [
                RepViTBlock(out_dim, mlp_ratio, kernel, j % 2 == 0, legacy, rngs=rngs)
                for j in range(depth)
            ]
        )

    def __call__(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class NormLinear(nnx.Module):
    def __init__(self, in_dim, out_dim, *, rngs):
        self.bn = BatchNorm(in_dim, epsilon=1e-5, rngs=rngs)
        self.l = nnx.Linear(
            in_dim, out_dim, kernel_init=nnx.initializers.truncated_normal(0.02), rngs=rngs
        )

    def __call__(self, x):
        return self.l(self.bn(x))


class RepVit(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"

    def __init__(
        self,
        embed_dim=(48, 96, 192, 384),
        depth=(2, 2, 14, 2),
        mlp_ratio=2,
        kernel_size=3,
        legacy=False,
        distillation=True,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.stem1 = ConvNorm(in_chans, embed_dim[0] // 2, 3, 2, rngs=rngs)
        self.stem2 = ConvNorm(embed_dim[0] // 2, embed_dim[0], 3, 2, rngs=rngs)
        stages, in_dim = [], embed_dim[0]
        for i, (dim, d) in enumerate(zip(embed_dim, depth)):
            stages.append(
                RepVitStage(in_dim, dim, d, mlp_ratio, kernel_size, i > 0, legacy, rngs=rngs)
            )
            in_dim = dim
        self.stages = nnx.List(stages)
        self.num_features = in_dim
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = NormLinear(in_dim, num_classes, rngs=rngs) if num_classes > 0 else None
        self.head_dist = (
            NormLinear(in_dim, num_classes, rngs=rngs) if distillation and num_classes > 0 else None
        )

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.global_pool = global_pool if global_pool is not None else self.global_pool
        rngs = nnx.Rngs(0)
        self.head = (
            NormLinear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None
        )
        if self.head_dist is not None:
            self.head_dist = (
                NormLinear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None
            )

    def forward_features(self, x):
        x = self.stem2(gelu(self.stem1(x)))
        for stage in self.stages:
            x = stage(x)
        return x

    def forward_head(self, x):
        x = self.head_drop(global_pool_nhwc(x, self.global_pool))
        if self.head is None:
            return x
        if self.head_dist is None:
            return self.head(x)
        return (self.head(x) + self.head_dist(x)) / 2

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_CFGS = {
    "repvit_m0_9": ((48, 96, 192, 384), (2, 2, 14, 2), False),
    "repvit_m1_0": ((56, 112, 224, 448), (2, 2, 14, 2), False),
    "repvit_m1_1": ((64, 128, 256, 512), (2, 2, 12, 2), False),
    "repvit_m1_5": ((64, 128, 256, 512), (4, 4, 24, 4), False),
    "repvit_m2_3": ((80, 160, 320, 640), (6, 6, 34, 2), False),
    "repvit_m1": ((48, 96, 192, 384), (2, 2, 14, 2), True),
    "repvit_m2": ((64, 128, 256, 512), (2, 2, 12, 2), True),
    "repvit_m3": ((64, 128, 256, 512), (4, 4, 18, 2), True),
}


def _make(name):
    embed_dim, depth, legacy = _CFGS[name]

    def entry(**kwargs):
        model = RepVit(embed_dim, depth, legacy=legacy, **kwargs)
        model.default_cfg = _cfg(crop_pct=0.95, interpolation="bicubic")
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
