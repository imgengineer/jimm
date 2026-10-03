"""PP-HGNet and PP-HGNetV2 in flax nnx, NHWC. Mirrors timm.models.hgnet.

Each block chains 3x3 convolutions (1x1 + depthwise "light" convolutions in
later V2 stages) and aggregates the block input and every intermediate
output with a 1x1 convolution, followed by sigmoid squeeze-excite (V1) or a
second 1x1 convolution (V2). Blocks after the first in a stage are residual,
and stages after the first start with a strided depthwise convolution. The
head pools, projects to 2,048 channels with ReLU, and classifies. V2 models
B0-B3 scale and shift every activation with a learnable affine pair.
"""

import jax.numpy as jnp
from flax import nnx

from ..layers import BatchNorm, ClassifierMixin, DropPath, global_pool_nhwc
from ..registry import _cfg, register_model


class LearnableAffineBlock(nnx.Module):
    def __init__(self):
        self.scale = nnx.Param(jnp.ones((1,)))
        self.bias = nnx.Param(jnp.zeros((1,)))

    def __call__(self, x):
        return self.scale[...] * x + self.bias[...]


class ConvBNAct(nnx.Module):
    def __init__(
        self, in_chs, out_chs, kernel, stride=1, groups=1, use_act=True, use_lab=False, *, rngs
    ):
        pad = (stride - 1 + kernel - 1) // 2
        self.conv = nnx.Conv(
            in_chs,
            out_chs,
            (kernel, kernel),
            strides=(stride, stride),
            padding=((pad, pad), (pad, pad)),
            feature_group_count=groups,
            use_bias=False,
            rngs=rngs,
        )
        self.bn = BatchNorm(out_chs, epsilon=1e-5, rngs=rngs)
        self.use_act = use_act
        self.lab = LearnableAffineBlock() if use_act and use_lab else None

    def __call__(self, x):
        x = self.bn(self.conv(x))
        if self.use_act:
            x = nnx.relu(x)
        return x if self.lab is None else self.lab(x)


class LightConvBNAct(nnx.Module):
    def __init__(self, in_chs, out_chs, kernel, use_lab=False, *, rngs):
        self.conv1 = ConvBNAct(in_chs, out_chs, 1, use_act=False, use_lab=use_lab, rngs=rngs)
        self.conv2 = ConvBNAct(out_chs, out_chs, kernel, groups=out_chs, use_lab=use_lab, rngs=rngs)

    def __call__(self, x):
        return self.conv2(self.conv1(x))


class EseModule(nnx.Module):
    def __init__(self, chs, *, rngs):
        self.conv = nnx.Linear(chs, chs, rngs=rngs)

    def __call__(self, x):
        return x * nnx.sigmoid(self.conv(jnp.mean(x, axis=(1, 2), keepdims=True)))


class StemV1(nnx.Module):
    def __init__(self, stem_chs, *, rngs):
        self.stem = nnx.List(
            [
                ConvBNAct(stem_chs[i], stem_chs[i + 1], 3, 2 if i == 0 else 1, rngs=rngs)
                for i in range(len(stem_chs) - 1)
            ]
        )

    def __call__(self, x):
        for conv in self.stem:
            x = conv(x)
        return nnx.max_pool(x, (3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))


class StemV2(nnx.Module):
    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False, *, rngs):
        self.stem1 = ConvBNAct(in_chs, mid_chs, 3, 2, use_lab=use_lab, rngs=rngs)
        self.stem2a = ConvBNAct(mid_chs, mid_chs // 2, 2, use_lab=use_lab, rngs=rngs)
        self.stem2b = ConvBNAct(mid_chs // 2, mid_chs, 2, use_lab=use_lab, rngs=rngs)
        self.stem3 = ConvBNAct(mid_chs * 2, mid_chs, 3, 2, use_lab=use_lab, rngs=rngs)
        self.stem4 = ConvBNAct(mid_chs, out_chs, 1, use_lab=use_lab, rngs=rngs)

    def __call__(self, x):
        # timm zero-pads the right and bottom before each 2x2 convolution and the pool.
        pad = ((0, 0), (0, 1), (0, 1), (0, 0))
        x = jnp.pad(self.stem1(x), pad)
        x2 = self.stem2b(jnp.pad(self.stem2a(x), pad))
        x1 = nnx.max_pool(x, (2, 2), strides=(1, 1))
        x = self.stem3(jnp.concatenate([x1, x2], axis=-1))
        return self.stem4(x)


class HighPerfGpuBlock(nnx.Module):
    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        layer_num,
        kernel=3,
        residual=False,
        light_block=False,
        use_lab=False,
        agg="ese",
        drop_path=0.0,
        *,
        rngs,
    ):
        self.residual = residual
        layers = []
        for i in range(layer_num):
            chs = in_chs if i == 0 else mid_chs
            layers.append(
                LightConvBNAct(chs, mid_chs, kernel, use_lab, rngs=rngs)
                if light_block
                else ConvBNAct(chs, mid_chs, kernel, use_lab=use_lab, rngs=rngs)
            )
        self.layers = nnx.List(layers)
        total = in_chs + layer_num * mid_chs
        if agg == "se":
            self.aggregation = nnx.List(
                [
                    ConvBNAct(total, out_chs // 2, 1, use_lab=use_lab, rngs=rngs),
                    ConvBNAct(out_chs // 2, out_chs, 1, use_lab=use_lab, rngs=rngs),
                ]
            )
        else:
            self.aggregation = nnx.List(
                [
                    ConvBNAct(total, out_chs, 1, use_lab=use_lab, rngs=rngs),
                    EseModule(out_chs, rngs=rngs),
                ]
            )
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        feats = [x]
        y = x
        for layer in self.layers:
            y = layer(y)
            feats.append(y)
        y = jnp.concatenate(feats, axis=-1)
        for layer in self.aggregation:
            y = layer(y)
        return self.drop_path(y) + x if self.residual else y


class HighPerfGpuStage(nnx.Module):
    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        block_num,
        downsample,
        light_block,
        kernel,
        layer_num,
        use_lab=False,
        agg="ese",
        drop_path=None,
        *,
        rngs,
    ):
        self.downsample = (
            ConvBNAct(in_chs, in_chs, 3, 2, groups=in_chs, use_act=False, rngs=rngs)
            if downsample
            else None
        )
        drop_path = drop_path or [0.0] * block_num
        self.blocks = nnx.List(
            [
                HighPerfGpuBlock(
                    in_chs if i == 0 else out_chs,
                    mid_chs,
                    out_chs,
                    layer_num,
                    kernel,
                    residual=i > 0,
                    light_block=light_block,
                    use_lab=use_lab,
                    agg=agg,
                    drop_path=drop_path[i],
                    rngs=rngs,
                )
                for i in range(block_num)
            ]
        )

    def __call__(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class HighPerfGpuNet(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        stem_type,
        stem_chs,
        stages,
        use_lab=False,
        head_hidden_size=2048,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        if stem_type == "v2":
            self.stem = StemV2(in_chans, *stem_chs, use_lab=use_lab, rngs=rngs)
        else:
            self.stem = StemV1([in_chans, *stem_chs], rngs=rngs)
        # timm calculate_drop_path_rates(stagewise=True): linear over all blocks, split by stage.
        depths = [cfg[3] for cfg in stages]
        rates = [drop_path_rate * i / max(sum(depths) - 1, 1) for i in range(sum(depths))]
        self.stages = nnx.List(
            [
                HighPerfGpuStage(
                    *cfg,
                    use_lab=use_lab,
                    agg="ese" if stem_type == "v1" else "se",
                    drop_path=rates[sum(depths[:i]) : sum(depths[: i + 1])],
                    rngs=rngs,
                )
                for i, cfg in enumerate(stages)
            ]
        )
        # The head projects pooled features to ``head_hidden_size`` before classifying.
        self.last_conv = nnx.Linear(stages[-1][2], head_hidden_size, use_bias=False, rngs=rngs)
        self.head_lab = LearnableAffineBlock() if use_lab else None
        self.num_features = head_hidden_size
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(head_hidden_size, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x

    def forward_head(self, x):
        x = nnx.relu(self.last_conv(global_pool_nhwc(x, self.global_pool)))
        if self.head_lab is not None:
            x = self.head_lab(x)
        x = self.head_drop(x)
        return self.fc(x) if self.fc is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


# Stages: in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel, layer_num.
_CFGS = {
    "hgnet_tiny": (
        "v1",
        (48, 48, 96),
        [
            (96, 96, 224, 1, False, False, 3, 5),
            (224, 128, 448, 1, True, False, 3, 5),
            (448, 160, 512, 2, True, False, 3, 5),
            (512, 192, 768, 1, True, False, 3, 5),
        ],
        False,
    ),
    "hgnet_small": (
        "v1",
        (64, 64, 128),
        [
            (128, 128, 256, 1, False, False, 3, 6),
            (256, 160, 512, 1, True, False, 3, 6),
            (512, 192, 768, 2, True, False, 3, 6),
            (768, 224, 1024, 1, True, False, 3, 6),
        ],
        False,
    ),
    "hgnet_base": (
        "v1",
        (96, 96, 160),
        [
            (160, 192, 320, 1, False, False, 3, 7),
            (320, 224, 640, 2, True, False, 3, 7),
            (640, 256, 960, 3, True, False, 3, 7),
            (960, 288, 1280, 2, True, False, 3, 7),
        ],
        False,
    ),
    "hgnetv2_b0": (
        "v2",
        (16, 16),
        [
            (16, 16, 64, 1, False, False, 3, 3),
            (64, 32, 256, 1, True, False, 3, 3),
            (256, 64, 512, 2, True, True, 5, 3),
            (512, 128, 1024, 1, True, True, 5, 3),
        ],
        True,
    ),
    "hgnetv2_b1": (
        "v2",
        (24, 32),
        [
            (32, 32, 64, 1, False, False, 3, 3),
            (64, 48, 256, 1, True, False, 3, 3),
            (256, 96, 512, 2, True, True, 5, 3),
            (512, 192, 1024, 1, True, True, 5, 3),
        ],
        True,
    ),
    "hgnetv2_b2": (
        "v2",
        (24, 32),
        [
            (32, 32, 96, 1, False, False, 3, 4),
            (96, 64, 384, 1, True, False, 3, 4),
            (384, 128, 768, 3, True, True, 5, 4),
            (768, 256, 1536, 1, True, True, 5, 4),
        ],
        True,
    ),
    "hgnetv2_b3": (
        "v2",
        (24, 32),
        [
            (32, 32, 128, 1, False, False, 3, 5),
            (128, 64, 512, 1, True, False, 3, 5),
            (512, 128, 1024, 3, True, True, 5, 5),
            (1024, 256, 2048, 1, True, True, 5, 5),
        ],
        True,
    ),
    "hgnetv2_b4": (
        "v2",
        (32, 48),
        [
            (48, 48, 128, 1, False, False, 3, 6),
            (128, 96, 512, 1, True, False, 3, 6),
            (512, 192, 1024, 3, True, True, 5, 6),
            (1024, 384, 2048, 1, True, True, 5, 6),
        ],
        False,
    ),
    "hgnetv2_b5": (
        "v2",
        (32, 64),
        [
            (64, 64, 128, 1, False, False, 3, 6),
            (128, 128, 512, 2, True, False, 3, 6),
            (512, 256, 1024, 5, True, True, 5, 6),
            (1024, 512, 2048, 2, True, True, 5, 6),
        ],
        False,
    ),
    "hgnetv2_b6": (
        "v2",
        (48, 96),
        [
            (96, 96, 192, 2, False, False, 3, 6),
            (192, 192, 512, 3, True, False, 3, 6),
            (512, 384, 1024, 6, True, True, 5, 6),
            (1024, 768, 2048, 3, True, True, 5, 6),
        ],
        False,
    ),
}


def _make(name):
    stem_type, stem_chs, stages, use_lab = _CFGS[name]

    def entry(**kwargs):
        model = HighPerfGpuNet(stem_type, stem_chs, stages, use_lab, **kwargs)
        model.default_cfg = _cfg(crop_pct=0.965, interpolation="bicubic")
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
