"""RDNet in flax nnx, NHWC. Mirrors timm.models.rdnet (DenseNets Reloaded).

Each dense stage optionally compresses its input with a LayerNorm + conv
transition, then every block reads the concatenation of all earlier features
and contributes ``growth_rate`` new channels.
"""

import jax.numpy as jnp
from flax import nnx

from ..layers import ClassifierMixin, DropPath, gelu, global_pool_nhwc
from ..registry import _cfg, register_model

_conv_init = nnx.initializers.kaiming_normal()


def _layer_norm(chs, *, rngs):
    return nnx.LayerNorm(chs, epsilon=1e-6, rngs=rngs)


class EffectiveSE(nnx.Module):
    """timm EffectiveSEModule: one full-width 1x1 projection with a hard-sigmoid gate."""

    def __init__(self, chs, *, rngs):
        self.fc = nnx.Linear(chs, chs, kernel_init=_conv_init, rngs=rngs)

    def __call__(self, x):
        return x * nnx.hard_sigmoid(self.fc(jnp.mean(x, axis=(1, 2), keepdims=True)))


class DenseBlock(nnx.Module):
    """Depthwise 7x7, LayerNorm, 1x1 expand, GELU, 1x1 to ``growth_rate`` channels."""

    def __init__(
        self,
        in_chs,
        growth_rate,
        ese=False,
        drop_path_rate=0.0,
        ls_init_value=1e-6,
        bottleneck_width_ratio=4.0,
        *,
        rngs,
    ):
        inter_chs = int(in_chs * bottleneck_width_ratio / 8) * 8
        self.conv_dw = nnx.Conv(
            in_chs,
            in_chs,
            (7, 7),
            padding=3,
            feature_group_count=in_chs,
            kernel_init=_conv_init,
            rngs=rngs,
        )
        self.norm = _layer_norm(in_chs, rngs=rngs)
        self.pw1 = nnx.Linear(in_chs, inter_chs, kernel_init=_conv_init, rngs=rngs)
        self.pw2 = nnx.Linear(inter_chs, growth_rate, kernel_init=_conv_init, rngs=rngs)
        self.se = EffectiveSE(growth_rate, rngs=rngs) if ese else None
        self.gamma = nnx.Param(jnp.full((growth_rate,), ls_init_value))
        self.drop_path = DropPath(drop_path_rate, rngs=rngs)

    def __call__(self, x):
        x = self.pw2(gelu(self.pw1(self.norm(self.conv_dw(x)))))
        if self.se is not None:
            x = self.se(x)
        # Keep the compute dtype so concatenated features stay BF16 under AMP.
        return self.drop_path(x * self.gamma[...].astype(x.dtype))


class DenseStage(nnx.Module):
    def __init__(
        self,
        in_chs,
        num_blocks,
        growth_rate,
        transition_chs=None,
        stride=1,
        ese=False,
        drop_path_rates=None,
        *,
        rngs,
    ):
        if transition_chs is None:
            self.norm = self.conv = None
            chs = in_chs
        else:
            self.norm = _layer_norm(in_chs, rngs=rngs)
            self.conv = nnx.Conv(
                in_chs,
                transition_chs,
                (stride, stride),
                strides=(stride, stride),
                padding="VALID",
                kernel_init=_conv_init,
                rngs=rngs,
            )
            chs = transition_chs
        drop_path_rates = drop_path_rates or [0.0] * num_blocks
        blocks = []
        for rate in drop_path_rates:
            blocks.append(DenseBlock(chs, growth_rate, ese, rate, rngs=rngs))
            chs += growth_rate
        self.blocks = nnx.List(blocks)
        self.out_chs = chs

    def __call__(self, x):
        if self.conv is not None:
            x = self.conv(self.norm(x))
        features = [x]
        for blk in self.blocks:
            features.append(blk(jnp.concatenate(features, axis=-1)))
        return jnp.concatenate(features, axis=-1)


class RDNet(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        num_init_features=64,
        growth_rates=(64, 104, 128, 128, 128, 128, 224),
        is_downsample_block=(False, True, True, False, False, False, True),
        num_plain_stages=2,
        num_blocks=3,
        transition_compression_ratio=0.5,
        patch_size=4,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.stem = nnx.Conv(
            in_chans,
            num_init_features,
            (patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            kernel_init=_conv_init,
            rngs=rngs,
        )
        self.stem_norm = _layer_norm(num_init_features, rngs=rngs)
        total = num_blocks * len(growth_rates)
        dpr = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]
        chs, stages = num_init_features, []
        for i, growth_rate in enumerate(growth_rates):
            stage = DenseStage(
                chs,
                num_blocks,
                growth_rate,
                transition_chs=int(chs * transition_compression_ratio / 8) * 8 if i else None,
                stride=2 if is_downsample_block[i] else 1,
                # timm "BlockESE" blocks follow the leading plain "Block" stages.
                ese=i >= num_plain_stages,
                drop_path_rates=dpr[i * num_blocks : (i + 1) * num_blocks],
                rngs=rngs,
            )
            stages.append(stage)
            chs = stage.out_chs
        self.stages = nnx.List(stages)
        self.num_features = chs
        # timm NormMlpClassifierHead: global pool, then LayerNorm, dropout, and fc.
        self.head_norm = _layer_norm(chs, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(chs, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.stem_norm(self.stem(x))
        for stage in self.stages:
            x = stage(x)
        return x

    def forward_head(self, x):
        x = global_pool_nhwc(x, self.global_pool)
        x = self.head_norm(x)
        x = self.head_drop(x)
        return self.fc(x) if self.fc is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_DOWNSAMPLE_11 = (False, True, True, False, False, False, False, False, False, True, False)

_CFGS = {
    "rdnet_tiny": (
        64,
        (64, 104, 128, 128, 128, 128, 224),
        (False, True, True, False, False, False, True),
    ),
    "rdnet_small": (72, (64, 128) + (128,) * 7 + (240,) * 2, _DOWNSAMPLE_11),
    "rdnet_base": (120, (96, 128) + (168,) * 7 + (336,) * 2, _DOWNSAMPLE_11),
    "rdnet_large": (
        144,
        (128, 192) + (256,) * 8 + (360,) * 2,
        (False, True, True) + (False,) * 7 + (True, False),
    ),
}


def _make(name):
    num_init_features, growth_rates, is_downsample_block = _CFGS[name]

    def entry(**kwargs):
        model = RDNet(num_init_features, growth_rates, is_downsample_block, **kwargs)
        model.default_cfg = _cfg(crop_pct=0.9, interpolation="bicubic")
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
