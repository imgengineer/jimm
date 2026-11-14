"""GhostNet in flax nnx, NHWC. Mirrors timm.models.ghostnet."""

import jax
import jax.numpy as jnp
from flax import nnx

from ..layers import (
    BatchNorm,
    ClassifierMixin,
    ConvBNAct,
    SqueezeExcite,
    global_pool_nhwc,
    make_divisible,
)
from ..registry import _cfg, register_model


class GhostModule(nnx.Module):
    """Primary 1x1 conv on half the channels plus a cheap 3x3 depthwise branch."""

    def __init__(self, in_chs, out_chs, use_act=True, *, rngs):
        half = -(-out_chs // 2)  # ceil; cheap branch same width, output sliced to out_chs
        self.out_chs = out_chs
        self.use_act = use_act
        self.conv1 = nnx.Conv(in_chs, half, (1, 1), use_bias=False, rngs=rngs)
        self.bn1 = BatchNorm(half, rngs=rngs)
        self.dw = nnx.Conv(half, half, (3, 3), use_bias=False, feature_group_count=half, rngs=rngs)
        self.bn2 = BatchNorm(half, rngs=rngs)

    def __call__(self, x):
        act = nnx.relu if self.use_act else (lambda y: y)
        x1 = act(self.bn1(self.conv1(x)))
        x2 = act(self.bn2(self.dw(x1)))
        return jnp.concatenate([x1, x2], axis=-1)[..., : self.out_chs]


class GhostModuleV2(nnx.Module):
    """Ghost module with GhostNetV2's DFC attention gate from a cheap 2x-downsampled branch."""

    def __init__(self, in_chs, out_chs, stride=1, *, rngs):
        self.out_chs = out_chs
        half = -(-out_chs // 2)
        self.primary = ConvBNAct(in_chs, half, 1, stride, rngs=rngs)
        self.cheap = ConvBNAct(half, half, 3, groups=half, rngs=rngs)
        self.short_conv = nnx.Sequential(
            ConvBNAct(in_chs, out_chs, 1, stride, act="identity", rngs=rngs),
            ConvBNAct(out_chs, out_chs, (1, 5), groups=out_chs, act="identity", rngs=rngs),
            ConvBNAct(out_chs, out_chs, (5, 1), groups=out_chs, act="identity", rngs=rngs),
        )

    def __call__(self, x):
        res = self.short_conv(nnx.avg_pool(x, (2, 2), strides=(2, 2), padding="VALID"))
        x1 = self.primary(x)
        x2 = self.cheap(x1)
        out = jnp.concatenate([x1, x2], axis=-1)[..., : self.out_chs]
        gate = jax.image.resize(jax.nn.sigmoid(res), out.shape[:3] + (self.out_chs,), "nearest")
        return out * gate


class GhostModuleV3(nnx.Module):
    """GhostNetV3 re-parameterizable ghost module (training form), optional DFC gate."""

    def __init__(self, in_chs, out_chs, kernel=1, stride=1, mode="original", use_act=True, *, rngs):
        self.out_chs = out_chs
        self.mode = mode
        half = -(-out_chs // 2)
        self.primary = nnx.List(
            [ConvBNAct(in_chs, half, kernel, stride, act="identity", rngs=rngs) for _ in range(3)]
        )
        self.cheap = nnx.List(
            [ConvBNAct(half, half, 3, groups=half, act="identity", rngs=rngs) for _ in range(3)]
        )
        self.cheap_scale = ConvBNAct(half, half, 1, groups=half, act="identity", rngs=rngs)
        self.cheap_skip = BatchNorm(half, rngs=rngs)
        self.use_act = use_act
        self.short_conv = (
            nnx.Sequential(
                ConvBNAct(in_chs, out_chs, 1, stride, act="identity", rngs=rngs),
                ConvBNAct(out_chs, out_chs, (1, 5), groups=out_chs, act="identity", rngs=rngs),
                ConvBNAct(out_chs, out_chs, (5, 1), groups=out_chs, act="identity", rngs=rngs),
            )
            if mode == "shortcut"
            else None
        )

    def __call__(self, x):
        x1 = self.primary[0](x)
        for conv in self.primary[1:]:
            x1 = x1 + conv(x)
        if self.use_act:
            x1 = nnx.relu(x1)
        x2 = self.cheap_scale(x1) + self.cheap_skip(x1)
        for conv in self.cheap:
            x2 = x2 + conv(x1)
        if self.use_act:
            x2 = nnx.relu(x2)
        out = jnp.concatenate([x1, x2], axis=-1)
        if self.mode != "shortcut":
            return out
        res = self.short_conv(nnx.avg_pool(x, (2, 2), strides=(2, 2), padding="VALID"))
        gate = jax.image.resize(jax.nn.sigmoid(res), out.shape[:3] + (self.out_chs,), "nearest")
        return out[..., : self.out_chs] * gate


class GhostBottleneck(nnx.Module):
    def __init__(
        self, in_chs, mid_chs, out_chs, kernel, stride, se, version="v1", mode="original", *, rngs
    ):
        self.use_shortcut_conv = stride == 2 or in_chs != out_chs
        if version == "v2" and mode == "attn":
            self.ghost1 = GhostModuleV2(in_chs, mid_chs, rngs=rngs)
        else:
            self.ghost1 = GhostModule(in_chs, mid_chs, rngs=rngs)
        self.dw = (
            ConvBNAct(mid_chs, mid_chs, kernel, stride, groups=mid_chs, act="identity", rngs=rngs)
            if stride == 2
            else None
        )
        self.se = (
            SqueezeExcite(
                mid_chs,
                rd_channels=make_divisible(mid_chs * 0.25, 4),
                gate=nnx.hard_sigmoid,
                rngs=rngs,
            )
            if se
            else None
        )
        self.ghost2 = GhostModule(mid_chs, out_chs, use_act=False, rngs=rngs)
        if self.use_shortcut_conv:
            self.sc_dw = ConvBNAct(
                in_chs, in_chs, kernel, stride, groups=in_chs, act="identity", rngs=rngs
            )
            self.sc_pw = ConvBNAct(in_chs, out_chs, 1, act="identity", rngs=rngs)

    def __call__(self, x):
        y = self.ghost1(x)
        if self.dw is not None:
            y = self.dw(y)
        if self.se is not None:
            y = self.se(y)
        y = self.ghost2(y)  # linear projection
        if self.use_shortcut_conv:
            return y + self.sc_pw(self.sc_dw(x))
        return y + x


class GhostBottleneckV3(nnx.Module):
    """GhostNetV3 bottleneck: re-parameterizable depthwise branches in the first ghost
    module and the strided depthwise stage, with a plain ghost projection."""

    def __init__(self, in_chs, mid_chs, out_chs, kernel, stride, se, mode="original", *, rngs):
        self.use_shortcut_conv = stride == 2 or in_chs != out_chs
        self.ghost1 = GhostModuleV3(in_chs, mid_chs, mode=mode, rngs=rngs)
        if stride > 1:
            self.dw_scale = ConvBNAct(
                mid_chs, mid_chs, 1, stride, groups=mid_chs, act="identity", rngs=rngs
            )
            self.dw_convs = nnx.List(
                [
                    ConvBNAct(
                        mid_chs, mid_chs, kernel, stride, groups=mid_chs, act="identity", rngs=rngs
                    )
                    for _ in range(3)
                ]
            )
        else:
            self.dw_scale = self.dw_convs = None
        self.se = (
            SqueezeExcite(
                mid_chs,
                rd_channels=make_divisible(mid_chs * 0.25, 4),
                gate=nnx.hard_sigmoid,
                rngs=rngs,
            )
            if se
            else None
        )
        self.ghost2 = GhostModuleV3(mid_chs, out_chs, mode="original", use_act=False, rngs=rngs)
        if self.use_shortcut_conv:
            self.sc_dw = ConvBNAct(
                in_chs, in_chs, kernel, stride, groups=in_chs, act="identity", rngs=rngs
            )
            self.sc_pw = ConvBNAct(in_chs, out_chs, 1, act="identity", rngs=rngs)

    def __call__(self, x):
        y = self.ghost1(x)
        if self.dw_convs is not None:
            dw = self.dw_scale(y)
            for conv in self.dw_convs:
                dw = dw + conv(y)
            y = dw
        if self.se is not None:
            y = self.se(y)
        y = self.ghost2(y)
        if self.use_shortcut_conv:
            return y + self.sc_pw(self.sc_dw(x))
        return y + x


# (kernel, exp, out, se, stride, repeats)
GHOSTNET_CFG = [
    (3, 16, 16, 0, 1, 1),
    (3, 48, 24, 0, 2, 1),
    (3, 72, 24, 0, 1, 1),
    (5, 72, 40, 1, 2, 1),
    (5, 120, 40, 1, 1, 1),
    (3, 240, 80, 0, 2, 1),
    (3, 200, 80, 0, 1, 1),
    (3, 184, 80, 0, 1, 1),
    (3, 184, 80, 0, 1, 1),
    (3, 480, 112, 1, 1, 1),
    (3, 672, 112, 1, 1, 1),
    (5, 672, 160, 1, 2, 1),
    (5, 960, 160, 0, 1, 1),
    (5, 960, 160, 1, 1, 1),
    (5, 960, 160, 0, 1, 1),
    (5, 960, 160, 1, 1, 1),
]


class GhostNet(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        width_mult=1.0,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        version="v1",
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        # timm rounds every width to a multiple of 4.
        stem = make_divisible(16 * width_mult, 4)
        self.conv1 = nnx.Conv(
            in_chans, stem, (3, 3), strides=(2, 2), padding=1, use_bias=False, rngs=rngs
        )
        self.bn1 = BatchNorm(stem, rngs=rngs)
        blocks, chs = [], stem
        layer_idx = 0
        for k, e, c, se, s, n in GHOSTNET_CFG:
            out = make_divisible(c * width_mult, 4)
            mid = make_divisible(e * width_mult, 4)
            for j in range(n):
                # timm enables the DFC gate / re-parameterized branches after the first
                # two bottleneck layers.
                if version == "v2":
                    mode = "attn" if layer_idx > 1 else "original"
                    blocks.append(
                        GhostBottleneck(
                            chs,
                            mid,
                            out,
                            k,
                            s if j == 0 else 1,
                            se,
                            version="v2",
                            mode=mode,
                            rngs=rngs,
                        )
                    )
                elif version == "v3":
                    mode = "shortcut" if layer_idx > 1 else "original"
                    blocks.append(
                        GhostBottleneckV3(
                            chs, mid, out, k, s if j == 0 else 1, se, mode=mode, rngs=rngs
                        )
                    )
                else:
                    blocks.append(
                        GhostBottleneck(chs, mid, out, k, s if j == 0 else 1, se, rngs=rngs)
                    )
                chs = out
                layer_idx += 1
        self.blocks = nnx.List(blocks)
        head = make_divisible(960 * width_mult, 4)
        self.conv_head = ConvBNAct(chs, head, 1, rngs=rngs)
        self.num_features = 1280
        self.head_fc1 = nnx.Linear(head, 1280, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(1280, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = nnx.relu(self.bn1(self.conv1(x)))
        for blk in self.blocks:
            x = blk(x)
        return self.conv_head(x)

    def forward_head(self, x):
        x = global_pool_nhwc(x, self.global_pool)
        x = nnx.relu(self.head_fc1(x))
        x = self.head_drop(x)
        return self.fc(x) if self.fc is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


def _ghostnet(width_mult, version="v1", **kwargs):
    model = GhostNet(width_mult, version=version, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def ghostnet_050(**kwargs):
    return _ghostnet(0.5, **kwargs)


@register_model
def ghostnet_100(**kwargs):
    return _ghostnet(1.0, **kwargs)


@register_model
def ghostnet_130(**kwargs):
    return _ghostnet(1.3, **kwargs)


@register_model
def ghostnetv2_100(**kwargs):
    return _ghostnet(1.0, version="v2", **kwargs)


@register_model
def ghostnetv2_130(**kwargs):
    return _ghostnet(1.3, version="v2", **kwargs)


@register_model
def ghostnetv2_160(**kwargs):
    return _ghostnet(1.6, version="v2", **kwargs)


@register_model
def ghostnetv3_050(**kwargs):
    return _ghostnet(0.5, version="v3", **kwargs)


@register_model
def ghostnetv3_100(**kwargs):
    return _ghostnet(1.0, version="v3", **kwargs)


@register_model
def ghostnetv3_130(**kwargs):
    return _ghostnet(1.3, version="v3", **kwargs)


@register_model
def ghostnetv3_160(**kwargs):
    return _ghostnet(1.6, version="v3", **kwargs)
