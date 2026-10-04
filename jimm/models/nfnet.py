"""NFNet (normalizer-free networks) in flax nnx, NHWC. Mirrors timm.models.nfnet.

Convolutions standardize their weights per output channel and scale them by a
learned gain (scaled weight standardization), so no activation normalization
is needed. Pre-activation bottleneck blocks scale their input by 1/sqrt of the
expected variance and add ``alpha`` times the residual branch (grouped 3x3
convolutions and squeeze-excite or ECA); average pooling and 1x1 convolutions
form strided shortcuts. Activations are scaled by fixed gammas that preserve
variance. ``dm_nfnet`` variants follow DeepMind's checkpoints: TensorFlow SAME
padding and a learned skip gain that starts the residual branches at zero.
"""

import math

import jax
import jax.numpy as jnp
from flax import nnx

from ..layers import ClassifierMixin, DropPath, global_pool_nhwc, make_divisible
from ..registry import _cfg, register_model

_NONLIN_GAMMA = {"gelu": 1.7015043497085571, "silu": 1.7881293296813965, "relu": 1.7139588594436646}
_ACTS = {"gelu": lambda x: nnx.gelu(x, approximate=False), "silu": nnx.silu, "relu": nnx.relu}


class ScaledStdConv(nnx.Module):
    """Conv with scaled weight standardization: ``gain * gamma * (w - mean) / (std * sqrt(fan_in))``."""

    def __init__(
        self,
        in_chs,
        out_chs,
        kernel=3,
        stride=1,
        groups=1,
        gamma=1.0,
        eps=1e-6,
        gain_init=1.0,
        same_padding=False,
        *,
        rngs,
    ):
        self.stride, self.groups, self.eps = stride, groups, eps
        self.dtype = None
        fan_in = kernel * kernel * (in_chs // groups)
        self.scale = gamma * fan_in**-0.5
        if same_padding:
            self.padding = "SAME"
        else:
            pad = (stride - 1 + kernel - 1) // 2
            self.padding = ((pad, pad), (pad, pad))
        self.kernel = nnx.Param(
            nnx.initializers.variance_scaling(1.0, "fan_in", "normal")(
                rngs.params(), (kernel, kernel, in_chs // groups, out_chs)
            )
        )
        self.bias = nnx.Param(jnp.zeros(out_chs))
        self.gain = nnx.Param(jnp.full((out_chs,), gain_init))

    def __call__(self, x):
        w = self.kernel[...]
        mean = jnp.mean(w, axis=(0, 1, 2), keepdims=True)
        var = jnp.var(w, axis=(0, 1, 2), keepdims=True)
        w = (w - mean) * jax.lax.rsqrt(var + self.eps) * (self.gain[...] * self.scale)
        dtype = self.dtype or jnp.result_type(x, w)
        return jax.lax.conv_general_dilated(
            x.astype(dtype),
            w.astype(dtype),
            (self.stride, self.stride),
            self.padding,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=self.groups,
        ) + self.bias[...].astype(dtype)


class SEModule(nnx.Module):
    def __init__(self, chs, rd_ratio=0.5, rd_divisor=8, *, rngs):
        rd = make_divisible(chs * rd_ratio, rd_divisor, round_limit=0.0)
        self.fc1 = nnx.Linear(chs, rd, rngs=rngs)
        self.fc2 = nnx.Linear(rd, chs, rngs=rngs)

    def __call__(self, x):
        s = self.fc2(nnx.relu(self.fc1(jnp.mean(x, axis=(1, 2), keepdims=True))))
        return x * nnx.sigmoid(s)


class EcaModule(nnx.Module):
    """Efficient channel attention: a 1D convolution across pooled channels."""

    def __init__(self, chs, gamma=2, beta=1, *, rngs):
        t = int(abs(math.log(chs, 2) + beta) / gamma)
        self.kernel_size = max(t if t % 2 else t + 1, 3)
        self.conv = nnx.Conv(
            1,
            1,
            (self.kernel_size,),
            padding=(((self.kernel_size - 1) // 2,) * 2,),
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, x):
        y = self.conv(jnp.mean(x, axis=(1, 2))[..., None])
        return x * nnx.sigmoid(y[:, None, None, :, 0])


class NormFreeBlock(nnx.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        stride=1,
        alpha=1.0,
        beta=1.0,
        bottle_ratio=0.25,
        group_size=None,
        ch_div=8,
        attn="se",
        attn_kwargs=None,
        attn_gain=2.0,
        skipinit=False,
        reg=False,
        extra_conv=True,
        act=None,
        conv=None,
        drop_path=0.0,
        *,
        rngs,
    ):
        # RegNet-style blocks size the bottleneck from the input and attend before conv3.
        mid = make_divisible((in_chs if reg else out_chs) * bottle_ratio, ch_div)
        groups = 1 if not group_size else mid // group_size
        if group_size and group_size % ch_div == 0:
            mid = group_size * groups
        self.alpha, self.beta, self.attn_gain, self.act = alpha, beta, attn_gain, act
        self.stride = stride
        self.downsample = (
            conv(in_chs, out_chs, 1, rngs=rngs) if in_chs != out_chs or stride != 1 else None
        )
        self.conv1 = conv(in_chs, mid, 1, rngs=rngs)
        self.conv2 = conv(mid, mid, 3, stride, groups, rngs=rngs)
        self.conv2b = conv(mid, mid, 3, 1, groups, rngs=rngs) if extra_conv else None
        self.conv3 = conv(mid, out_chs, 1, gain_init=1.0 if skipinit else 0.0, rngs=rngs)
        attn_chs = mid if reg else out_chs
        if attn == "se":
            attn_layer = SEModule(attn_chs, **(attn_kwargs or {}), rngs=rngs)
        elif attn == "eca":
            attn_layer = EcaModule(attn_chs, rngs=rngs)
        else:
            attn_layer = None
        self.attn = attn_layer if reg else None
        self.attn_last = None if reg else attn_layer
        self.skipinit_gain = nnx.Param(jnp.zeros(())) if skipinit else None
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        out = self.act(x) * self.beta
        shortcut = x
        if self.downsample is not None:
            if self.stride > 1:
                # Ceil-mode 2x2 average pooling that ignores padding, as timm's DownsampleAvg.
                h, w = out.shape[1:3]
                pooled = nnx.avg_pool(
                    out,
                    (2, 2),
                    strides=(2, 2),
                    padding=((0, h % 2), (0, w % 2)),
                    count_include_pad=False,
                )
            else:
                pooled = out
            shortcut = self.downsample(pooled)
        out = self.conv1(out)
        out = self.conv2(self.act(out))
        if self.conv2b is not None:
            out = self.conv2b(self.act(out))
        if self.attn is not None:
            out = self.attn_gain * self.attn(out)
        out = self.conv3(self.act(out))
        if self.attn_last is not None:
            out = self.attn_gain * self.attn_last(out)
        out = self.drop_path(out)
        if self.skipinit_gain is not None:
            out = out * self.skipinit_gain[...]
        return out * self.alpha + shortcut


class NormFreeNet(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        depths=(1, 2, 6, 3),
        channels=(256, 512, 1536, 1536),
        stem_chs=128,
        group_size=128,
        bottle_ratio=0.5,
        feat_mult=2.0,
        alpha=0.2,
        act_layer="gelu",
        attn="se",
        attn_kwargs=None,
        dm=False,
        std_conv_eps=1e-5,
        stem_type="deep_quad",
        extra_conv=True,
        reg=False,
        width_factor=1.0,
        num_features=None,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        gamma = _NONLIN_GAMMA[act_layer]
        base_act = _ACTS[act_layer]
        # DeepMind models scale activations; timm's own models fold gamma into the convs.
        if dm:
            self.act = lambda x: base_act(x) * gamma
            conv_gamma = 1.0
        else:
            self.act = base_act
            conv_gamma = gamma

        def conv(in_chs, out_chs, kernel, stride=1, groups=1, gain_init=1.0, *, rngs):
            return ScaledStdConv(
                in_chs,
                out_chs,
                kernel,
                stride,
                groups,
                conv_gamma,
                std_conv_eps,
                gain_init,
                dm,
                rngs=rngs,
            )

        stem_chs = make_divisible((stem_chs or channels[0]) * width_factor, 8)
        if stem_type == "deep_quad":
            stem = [stem_chs // 8, stem_chs // 4, stem_chs // 2, stem_chs]
            strides = (2, 1, 1, 2)
            kernels = (3, 3, 3, 3)
        else:  # timm "3x3" or "7x7_pool": one stride-2 conv (then a 3x3 max pool)
            stem, strides, kernels = [stem_chs], (2,), (7 if "7x7" in stem_type else 3,)
        self.stem = nnx.List(
            [
                conv(c_in, c_out, k, s, rngs=rngs)
                for c_in, c_out, k, s in zip([in_chans, *stem[:-1]], stem, kernels, strides)
            ]
        )
        self.stem_pool = "pool" in stem_type
        stem_stride = 4 if stem_type == "deep_quad" or self.stem_pool else 2
        # timm calculate_drop_path_rates(stagewise=True): linear over all blocks, split by stage.
        total = sum(depths)
        rates = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]
        stages, prev, expected_var, k = [], stem_chs, 1.0, 0
        for i, depth in enumerate(depths):
            blocks = []
            for j in range(depth):
                out = make_divisible(channels[i] * width_factor, 8)
                stride = (1 if i == 0 and stem_stride > 2 else 2) if j == 0 else 1
                blocks.append(
                    NormFreeBlock(
                        prev,
                        out,
                        stride,
                        alpha,
                        1.0 / expected_var**0.5,
                        1.0 if reg and i == 0 and j == 0 else bottle_ratio,
                        group_size,
                        attn=attn,
                        attn_kwargs=attn_kwargs,
                        skipinit=dm,
                        reg=reg,
                        extra_conv=extra_conv,
                        act=self.act,
                        conv=conv,
                        drop_path=rates[k],
                        rngs=rngs,
                    )
                )
                if j == 0:
                    expected_var = 1.0  # reset after the first block of each stage
                expected_var += alpha**2
                prev, k = out, k + 1
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        if num_features is None:
            num_features = int(channels[-1] * feat_mult)
        if num_features:
            self.num_features = make_divisible(width_factor * num_features, 8)
            self.final_conv = conv(prev, self.num_features, 1, rngs=rngs)
        else:
            self.num_features, self.final_conv = prev, None
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = (
            nnx.Linear(
                self.num_features, num_classes, kernel_init=nnx.initializers.normal(0.01), rngs=rngs
            )
            if num_classes > 0
            else None
        )

    def forward_features(self, x):
        for i, conv in enumerate(self.stem):
            x = conv(x)
            if i < len(self.stem) - 1:
                x = self.act(x)
        if self.stem_pool:
            x = nnx.max_pool(x, (3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))
        for stage in self.stages:
            for blk in stage:
                x = blk(x)
        if self.final_conv is not None:
            x = self.final_conv(x)
        return self.act(x)

    def forward_head(self, x):
        x = self.head_drop(global_pool_nhwc(x, self.global_pool))
        return self.fc(x) if self.fc is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_DEPTHS = {
    "f0": (1, 2, 6, 3),
    "f1": (2, 4, 12, 6),
    "f2": (3, 6, 18, 9),
    "f3": (4, 8, 24, 12),
    "f4": (5, 10, 30, 15),
    "f5": (6, 12, 36, 18),
    "f6": (7, 14, 42, 21),
    "f7": (8, 16, 48, 24),
}
_F_SIZES = {"f0": 192, "f1": 224, "f2": 256, "f3": 320, "f4": 384, "f5": 416, "f6": 448, "f7": 480}
_L = dict(feat_mult=1.5, group_size=64, bottle_ratio=0.25, act_layer="silu")
_CFGS = {
    **{f"nfnet_{f}": (dict(depths=d), _F_SIZES[f], 0.9) for f, d in _DEPTHS.items()},
    **{
        f"dm_nfnet_{f}": (dict(depths=d, dm=True), _F_SIZES[f], crop)
        for (f, d), crop in zip(_DEPTHS.items(), (0.9, 0.91, 0.92, 0.94, 0.951, 0.954, 0.956))
    },
    "nfnet_l0": (dict(depths=_DEPTHS["f0"], **_L, attn_kwargs=dict(rd_ratio=0.25)), 224, 1.0),
    "eca_nfnet_l0": (dict(depths=_DEPTHS["f0"], **_L, attn="eca"), 224, 1.0),
    "eca_nfnet_l1": (dict(depths=_DEPTHS["f1"], **{**_L, "feat_mult": 2.0}, attn="eca"), 256, 1.0),
    "eca_nfnet_l2": (dict(depths=_DEPTHS["f2"], **{**_L, "feat_mult": 2.0}, attn="eca"), 320, 1.0),
    "eca_nfnet_l3": (dict(depths=_DEPTHS["f3"], **{**_L, "feat_mult": 2.0}, attn="eca"), 352, 1.0),
}


_NFRES = dict(
    channels=(256, 512, 1024, 2048), stem_type="7x7_pool", stem_chs=64, bottle_ratio=0.25,
    group_size=None, act_layer="relu", attn=None, extra_conv=False, num_features=0,
)  # fmt: skip
_SE16 = dict(attn="se", attn_kwargs=dict(rd_ratio=1 / 16))


def _nfreg(depths, channels=(48, 104, 208, 440)):
    return dict(
        depths=depths, channels=channels, stem_type="3x3", stem_chs=None, group_size=8,
        width_factor=0.75, bottle_ratio=2.25, num_features=1280 * channels[-1] // 440, reg=True,
        attn_kwargs=dict(rd_ratio=0.5), extra_conv=False, act_layer="silu",
    )  # fmt: skip


# name: (constructor args, image size, crop, test size) — NF-ResNets and NF-RegNets.
_NF_CFGS = {
    "nf_regnet_b0": (_nfreg((1, 3, 6, 6)), 192, 0.9, 256),
    "nf_regnet_b1": (_nfreg((2, 4, 7, 7)), 256, 0.9, 288),
    "nf_regnet_b2": (_nfreg((2, 4, 8, 8), (56, 112, 232, 488)), 240, 0.9, 272),
    "nf_regnet_b3": (_nfreg((2, 5, 9, 9), (56, 128, 248, 528)), 288, 0.9, 320),
    "nf_regnet_b4": (_nfreg((2, 6, 11, 11), (64, 144, 288, 616)), 320, 0.9, 384),
    "nf_regnet_b5": (_nfreg((3, 7, 14, 14), (80, 168, 336, 704)), 384, 0.9, 456),
    "nf_resnet26": (dict(depths=(2, 2, 2, 2), **_NFRES), 224, 0.9, None),
    "nf_resnet50": (dict(depths=(3, 4, 6, 3), **_NFRES), 256, 0.94, 288),
    "nf_resnet101": (dict(depths=(3, 4, 23, 3), **_NFRES), 224, 0.9, None),
    "nf_seresnet26": (dict(depths=(2, 2, 2, 2), **{**_NFRES, **_SE16}), 224, 0.9, None),
    "nf_seresnet50": (dict(depths=(3, 4, 6, 3), **{**_NFRES, **_SE16}), 224, 0.9, None),
    "nf_seresnet101": (dict(depths=(3, 4, 23, 3), **{**_NFRES, **_SE16}), 224, 0.9, None),
    "nf_ecaresnet26": (dict(depths=(2, 2, 2, 2), **{**_NFRES, "attn": "eca"}), 224, 0.9, None),
    "nf_ecaresnet50": (dict(depths=(3, 4, 6, 3), **{**_NFRES, "attn": "eca"}), 224, 0.9, None),
    "nf_ecaresnet101": (dict(depths=(3, 4, 23, 3), **{**_NFRES, "attn": "eca"}), 224, 0.9, None),
    "test_nfnet": (
        dict(
            depths=(1, 1, 1, 1), channels=(32, 64, 96, 128), feat_mult=1.5, group_size=8,
            bottle_ratio=0.25, attn_kwargs=dict(rd_ratio=0.25, rd_divisor=8), act_layer="silu",
        ),
        160, 0.95, None,
    ),
}  # fmt: skip


def _make(name):
    if name in _NF_CFGS:
        cfg, size, crop, test = _NF_CFGS[name]
    else:
        (cfg, size, crop), test = _CFGS[name], None
    ev = {"test_input_size": (3, test, test)} if test else {}
    if name == "test_nfnet":
        ev.update(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    def entry(**kwargs):
        model = NormFreeNet(**{**cfg, **kwargs})
        model.default_cfg = _cfg(
            input_size=(3, size, size), crop_pct=crop, interpolation="bicubic", **ev
        )
        return model

    entry.__name__ = name
    return entry


for _name in (*_CFGS, *_NF_CFGS):
    register_model(_make(_name))
