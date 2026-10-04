"""VoVNet in flax nnx, NHWC. Mirrors timm.models.vovnet.

One-shot aggregation (OSA) blocks chain 3x3 convolutions (separable ones for
the ``_dw`` variants) and concatenate the block input with every
intermediate output before a 1x1 convolution. The V2 (``ese_``) models add
residuals to all but the first block of a stage and effective squeeze-excite
(one 1x1 convolution with a hard-sigmoid gate) to the last block. Stages
after the first start with a ceil-mode 3x3 max pool.
"""

import jax.numpy as jnp
from flax import nnx

from ..layers import BatchNorm, ClassifierMixin, DropPath
from ..registry import _cfg, register_model
from ._conv import ConvNormAct
from ._efficientnet import EvoNorm2dS0
from .nfnet import EcaModule


def _max_pool_ceil(x, kernel=3, stride=2):
    """PyTorch ``MaxPool2d(kernel, stride, ceil_mode=True)``: pad the end only as needed."""
    pads = []
    for size in x.shape[1:3]:
        out = -(-(size - kernel) // stride) + 1
        if (out - 1) * stride >= size:  # windows must start inside the input
            out -= 1
        pads.append((0, max((out - 1) * stride + kernel - size, 0)))
    return nnx.max_pool(x, (kernel, kernel), strides=(stride, stride), padding=pads)


class EvoConvNormAct(nnx.Module):
    """Conv followed by timm's ``evonorms0`` norm-act (EvoNorm-S0, 32 groups)."""

    def __init__(self, in_chs, out_chs, kernel, stride, *, rngs):
        p = kernel // 2
        self.conv = nnx.Conv(
            in_chs, out_chs, (kernel, kernel), strides=stride, padding=((p, p), (p, p)),
            use_bias=False, rngs=rngs,
        )  # fmt: skip
        self.norm = EvoNorm2dS0(out_chs, out_chs // 32)

    def __call__(self, x):
        return self.norm(self.conv(x))


def _conv(in_chs, out_chs, kernel=1, stride=1, norm="bn", *, rngs):
    if norm == "evos":
        return EvoConvNormAct(in_chs, out_chs, kernel, stride, rngs=rngs)
    return ConvNormAct(in_chs, out_chs, kernel, stride, act=nnx.relu, rngs=rngs)


class SeparableConvNormAct(nnx.Module):
    """Depthwise 3x3 and pointwise convolutions, then BatchNorm and ReLU."""

    def __init__(self, in_chs, out_chs, stride=1, *, rngs):
        self.conv_dw = nnx.Conv(
            in_chs,
            in_chs,
            (3, 3),
            strides=(stride, stride),
            padding=((1, 1), (1, 1)),
            feature_group_count=in_chs,
            use_bias=False,
            rngs=rngs,
        )
        self.conv_pw = nnx.Conv(in_chs, out_chs, (1, 1), use_bias=False, rngs=rngs)
        self.norm = BatchNorm(out_chs, epsilon=1e-5, rngs=rngs)

    def __call__(self, x):
        return nnx.relu(self.norm(self.conv_pw(self.conv_dw(x))))


class EffectiveSE(nnx.Module):
    def __init__(self, chs, *, rngs):
        self.fc = nnx.Linear(chs, chs, rngs=rngs)

    def __call__(self, x):
        return x * nnx.hard_sigmoid(self.fc(jnp.mean(x, axis=(1, 2), keepdims=True)))


class OsaBlock(nnx.Module):
    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        layers,
        residual=False,
        depthwise=False,
        attn=None,
        drop_path=0.0,
        norm="bn",
        *,
        rngs,
    ):
        self.residual = residual
        self.conv_reduction = (
            _conv(in_chs, mid_chs, norm=norm, rngs=rngs)
            if depthwise and in_chs != mid_chs
            else None
        )
        convs, chs = [], in_chs
        for _ in range(layers):
            convs.append(
                SeparableConvNormAct(mid_chs, mid_chs, rngs=rngs)
                if depthwise
                else _conv(chs, mid_chs, 3, norm=norm, rngs=rngs)
            )
            chs = mid_chs
        self.conv_mid = nnx.List(convs)
        self.conv_concat = _conv(in_chs + layers * mid_chs, out_chs, norm=norm, rngs=rngs)
        if attn == "eca":
            self.attn = EcaModule(out_chs, rngs=rngs)
        else:
            self.attn = EffectiveSE(out_chs, rngs=rngs) if attn else None
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        feats = [x]
        y = x if self.conv_reduction is None else self.conv_reduction(x)
        for conv in self.conv_mid:
            y = conv(y)
            feats.append(y)
        y = self.conv_concat(jnp.concatenate(feats, axis=-1))
        if self.attn is not None:
            y = self.attn(y)
        # timm applies drop path to every block output, residual or not.
        y = self.drop_path(y)
        return y + x if self.residual else y


class OsaStage(nnx.Module):
    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        blocks,
        layers,
        downsample,
        residual,
        depthwise,
        attn,
        drop_path_rates,
        norm="bn",
        *,
        rngs,
    ):
        self.downsample = downsample
        self.blocks = nnx.List(
            [
                OsaBlock(
                    in_chs if i == 0 else out_chs,
                    mid_chs,
                    out_chs,
                    layers,
                    residual=residual and i > 0,
                    depthwise=depthwise,
                    attn=attn if i == blocks - 1 else None,
                    drop_path=drop_path_rates[i],
                    norm=norm,
                    rngs=rngs,
                )
                for i in range(blocks)
            ]
        )

    def __call__(self, x):
        if self.downsample:
            x = _max_pool_ceil(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class VovNet(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        stem_chs,
        stage_conv_chs,
        stage_out_chs,
        layer_per_block,
        block_per_stage,
        residual,
        depthwise,
        attn,
        norm="bn",
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool

        def conv3x3(in_chs, out_chs, stride):
            if depthwise:
                return SeparableConvNormAct(in_chs, out_chs, stride, rngs=rngs)
            return _conv(in_chs, out_chs, 3, stride, norm, rngs=rngs)

        self.stem = nnx.List(
            [
                _conv(in_chans, stem_chs[0], 3, 2, norm, rngs=rngs),
                conv3x3(stem_chs[0], stem_chs[1], 1),
                conv3x3(stem_chs[1], stem_chs[2], 2),
            ]
        )
        # timm calculate_drop_path_rates(stagewise=True): linear over all blocks, split by stage.
        total = sum(block_per_stage)
        rates = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]
        in_chs = [stem_chs[-1], *stage_out_chs[:-1]]
        stages = []
        for i in range(4):
            start = sum(block_per_stage[:i])
            stages.append(
                OsaStage(
                    in_chs[i],
                    stage_conv_chs[i],
                    stage_out_chs[i],
                    block_per_stage[i],
                    layer_per_block,
                    downsample=i > 0,  # the stride-4 stem already downsamples stage 0
                    residual=residual,
                    depthwise=depthwise,
                    attn=attn,
                    drop_path_rates=rates[start : start + block_per_stage[i]],
                    norm=norm,
                    rngs=rngs,
                )
            )
        self.stages = nnx.List(stages)
        self.num_features = stage_out_chs[-1]
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        for layer in self.stem:
            x = layer(x)
        for stage in self.stages:
            x = stage(x)
        return x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_WIDE = dict(stage_conv_chs=(128, 160, 192, 224), stage_out_chs=(256, 512, 768, 1024))
_SLIM = dict(stage_conv_chs=(64, 80, 96, 112), stage_out_chs=(112, 256, 384, 512))
_V1 = dict(residual=False, depthwise=False, attn=None)
_V2 = dict(residual=True, depthwise=False, attn="ese")
_V2_DW = dict(residual=True, depthwise=True, attn="ese")
_CFGS = {
    "vovnet39a": dict(**_WIDE, **_V1, layer_per_block=5, block_per_stage=(1, 1, 2, 2)),
    "vovnet57a": dict(**_WIDE, **_V1, layer_per_block=5, block_per_stage=(1, 1, 4, 3)),
    "ese_vovnet19b_slim_dw": dict(
        **_SLIM, **_V2_DW, stem_chs=(64, 64, 64), layer_per_block=3, block_per_stage=(1, 1, 1, 1)
    ),
    "ese_vovnet19b_dw": dict(
        **_WIDE, **_V2_DW, stem_chs=(64, 64, 64), layer_per_block=3, block_per_stage=(1, 1, 1, 1)
    ),
    "ese_vovnet19b_slim": dict(**_SLIM, **_V2, layer_per_block=3, block_per_stage=(1, 1, 1, 1)),
    # timm defines but does not register this configuration.
    "ese_vovnet19b": dict(**_WIDE, **_V2, layer_per_block=3, block_per_stage=(1, 1, 1, 1)),
    "ese_vovnet39b": dict(**_WIDE, **_V2, layer_per_block=5, block_per_stage=(1, 1, 2, 2)),
    "ese_vovnet57b": dict(**_WIDE, **_V2, layer_per_block=5, block_per_stage=(1, 1, 4, 3)),
    "ese_vovnet99b": dict(**_WIDE, **_V2, layer_per_block=5, block_per_stage=(1, 3, 9, 3)),
    "eca_vovnet39b": dict(
        **_WIDE, **{**_V2, "attn": "eca"}, layer_per_block=5, block_per_stage=(1, 1, 2, 2)
    ),
    "ese_vovnet39b_evos": dict(
        **_WIDE, **_V2, layer_per_block=5, block_per_stage=(1, 1, 2, 2), norm="evos"
    ),
}


def _make(name):
    cfg = {"stem_chs": (64, 64, 128), **_CFGS[name]}

    def entry(**kwargs):
        model = VovNet(**cfg, **kwargs)
        model.default_cfg = _cfg(interpolation="bicubic")
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
