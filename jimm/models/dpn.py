"""DPN (Dual Path Network) in flax nnx, NHWC. Mirrors timm.models.dpn.

Blocks are pre-activation (BatchNorm with epsilon 1e-3, ReLU, then a bias-free
convolution). Each block adds to a residual path and appends ``inc`` channels to
a densely connected path; the two paths travel concatenated as
``[residual | dense]``. The stem and the final BatchNorm on the feature map use
the model activation (ReLU, or SiLU for dpn48b); timm's ``fc_act_layer="elu"``
does not take effect.
"""

from functools import partial

import jax.numpy as jnp
from flax import nnx

from ..layers import BatchNorm, ClassifierMixin
from ..registry import _cfg, register_model

_IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
_IMAGENET_DPN_STD = (1 / (0.0167 * 255),) * 3


def _bn(chs, *, rngs):
    return BatchNorm(chs, epsilon=1e-3, rngs=rngs)


def _conv(in_chs, out_chs, kernel, stride=1, groups=1, *, rngs):
    pad = ((stride - 1) + (kernel - 1)) // 2  # timm get_padding
    return nnx.Conv(
        in_chs,
        out_chs,
        (kernel, kernel),
        strides=(stride, stride),
        padding=((pad, pad), (pad, pad)),
        feature_group_count=groups,
        use_bias=False,
        rngs=rngs,
    )


class BnAct(nnx.Module):
    def __init__(self, chs, act, *, rngs):
        self.norm = _bn(chs, rngs=rngs)
        self.act = act

    def __call__(self, x):
        return self.act(self.norm(x))


class BnActConv(nnx.Module):
    """timm BnActConv2d: BatchNorm and activation on the input, then the convolution."""

    def __init__(self, in_chs, out_chs, kernel, stride=1, groups=1, act=nnx.relu, *, rngs):
        self.norm = _bn(in_chs, rngs=rngs)
        self.conv = _conv(in_chs, out_chs, kernel, stride, groups, rngs=rngs)
        self.act = act

    def __call__(self, x):
        return self.conv(self.act(self.norm(x)))


class DualPathBlock(nnx.Module):
    def __init__(self, in_chs, r, bw, inc, groups, block_type="normal", b=False, *, rngs):
        self.bw = bw
        stride = 2 if block_type == "down" else 1
        conv = partial(BnActConv, rngs=rngs)
        self.c1x1_w = conv(in_chs, bw + 2 * inc, 1, stride) if block_type != "normal" else None
        self.c1x1_a = conv(in_chs, r, 1)
        self.c3x3_b = conv(r, r, 3, stride, groups)
        if b:
            self.c1x1_c = BnAct(r, nnx.relu, rngs=rngs)
            self.c1x1_c1 = _conv(r, bw, 1, rngs=rngs)
            self.c1x1_c2 = _conv(r, inc, 1, rngs=rngs)
        else:
            self.c1x1_c = conv(r, bw + inc, 1)
            self.c1x1_c1 = self.c1x1_c2 = None

    def __call__(self, x):
        x_s = x if self.c1x1_w is None else self.c1x1_w(x)
        y = self.c1x1_c(self.c3x3_b(self.c1x1_a(x)))
        if self.c1x1_c1 is not None:
            out1, out2 = self.c1x1_c1(y), self.c1x1_c2(y)
        else:
            out1, out2 = y[..., : self.bw], y[..., self.bw :]
        resid = x_s[..., : self.bw] + out1
        return jnp.concatenate([resid, x_s[..., self.bw :], out2], axis=-1)


class DPN(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        k_sec,
        inc_sec,
        k_r,
        groups,
        small=False,
        num_init_features=64,
        b=False,
        act="relu",
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        act = {"relu": nnx.relu, "silu": nnx.silu}[act]
        bw_factor = 1 if small else 4
        kernel = 3 if small else 7
        self.conv1_1 = nnx.Sequential(
            _conv(in_chans, num_init_features, kernel, 2, rngs=rngs),
            BnAct(num_init_features, act, rngs=rngs),
        )
        stages, in_chs = [], num_init_features
        for i, (k, inc) in enumerate(zip(k_sec, inc_sec)):
            bw = 64 * bw_factor * 2**i
            r = (k_r * bw) // (64 * bw_factor)
            blocks = [
                DualPathBlock(
                    in_chs, r, bw, inc, groups, "proj" if i == 0 else "down", b, rngs=rngs
                )
            ]
            in_chs = bw + 3 * inc
            for _ in range(1, k):
                blocks.append(DualPathBlock(in_chs, r, bw, inc, groups, "normal", b, rngs=rngs))
                in_chs += inc
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.num_features = in_chs
        self.final_norm = BnAct(in_chs, act, rngs=rngs)  # timm conv5_bn_ac
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(in_chs, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.conv1_1(x)
        x = nnx.max_pool(x, (3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))
        for stage in self.stages:
            for blk in stage:
                x = blk(x)
        return self.final_norm(x)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_CFGS = {  # k_sec, inc_sec, k_r, groups, small, num_init_features, b, act
    "dpn48b": ((3, 4, 6, 3), (16, 32, 32, 64), 128, 32, True, 10, True, "silu"),
    "dpn68": ((3, 4, 12, 3), (16, 32, 32, 64), 128, 32, True, 10, False, "relu"),
    "dpn68b": ((3, 4, 12, 3), (16, 32, 32, 64), 128, 32, True, 10, True, "relu"),
    "dpn92": ((3, 4, 20, 3), (16, 32, 24, 128), 96, 32, False, 64, False, "relu"),
    "dpn98": ((3, 6, 20, 3), (16, 32, 32, 128), 160, 40, False, 96, False, "relu"),
    "dpn107": ((4, 8, 20, 3), (20, 64, 64, 128), 200, 50, False, 128, False, "relu"),
    "dpn131": ((4, 8, 28, 3), (16, 32, 32, 128), 160, 40, False, 128, False, "relu"),
}


def _make(name):
    k_sec, inc_sec, k_r, groups, small, nif, b, act = _CFGS[name]

    def entry(**kwargs):
        model = DPN(k_sec, inc_sec, k_r, groups, small, nif, b, act, **kwargs)
        if name in ("dpn48b", "dpn68b"):  # timm's default tags for these use ImageNet statistics
            model.default_cfg = _cfg(
                crop_pct=0.95 if name == "dpn68b" else 0.875, interpolation="bicubic"
            )
        else:
            model.default_cfg = _cfg(
                interpolation="bicubic", mean=_IMAGENET_DPN_MEAN, std=_IMAGENET_DPN_STD
            )
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
