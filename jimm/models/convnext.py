"""ConvNeXt and ConvNeXt-V2 in flax nnx, NHWC. Mirrors timm.models.convnext.

One configurable model covers every timm variant: patch or overlapping stems, LayerNorm,
RMSNorm or SimpleNorm, layer scale or Global Response Norm (V2) MLPs, per-stage kernel
sizes, a norm-first head and the MLP (``pre_logits``) head of the CLIP models.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from ..features import _select_features
from ..layers import ClassifierMixin, DropPath, gelu, global_pool_nhwc, make_divisible
from ..registry import _cfg, register_model

_ACTS = {"gelu": gelu, "gelu_tanh": nnx.gelu, "silu": nnx.silu}


class SimpleNorm(nnx.Module):
    """timm SimpleNorm: scale by the (unbiased) channel variance without centering."""

    def __init__(self, dim, epsilon=1e-6):
        self.scale = nnx.Param(jnp.ones(dim))
        self.eps = epsilon

    def __call__(self, x):
        return x * jax.lax.rsqrt(jnp.var(x, axis=-1, keepdims=True, ddof=1) + self.eps) * self.scale


def _norm(kind, dim, eps, *, rngs):
    if kind == "rmsnorm":
        return nnx.RMSNorm(dim, epsilon=eps, rngs=rngs)
    if kind == "simplenorm":
        return SimpleNorm(dim, eps)
    return nnx.LayerNorm(dim, epsilon=eps, rngs=rngs)


def _conv(in_chs, out_chs, kernel, stride=1, pad=0, groups=1, bias=True, *, rngs):
    return nnx.Conv(
        in_chs, out_chs, (kernel, kernel), strides=stride, padding=((pad, pad), (pad, pad)),
        feature_group_count=groups, use_bias=bias, rngs=rngs,
    )  # fmt: skip


class GRN(nnx.Module):
    """Global Response Normalization (ConvNeXt V2), timm ``GlobalResponseNorm``."""

    def __init__(self, dim, eps=1e-6):
        self.scale = nnx.Param(jnp.zeros(dim))
        self.bias = nnx.Param(jnp.zeros(dim))
        self.eps = eps

    def __call__(self, x):
        gx = jnp.sqrt(jnp.sum(jnp.square(x), axis=(1, 2), keepdims=True))
        nx = gx / (jnp.mean(gx, axis=-1, keepdims=True) + self.eps)
        return x + (self.bias[...] + self.scale[...] * (x * nx))


class Mlp(nnx.Module):
    def __init__(self, dim, hidden, act, use_grn, bias, *, rngs):
        self.fc1 = nnx.Linear(dim, hidden, use_bias=bias, rngs=rngs)
        self.act = act
        self.grn = GRN(hidden) if use_grn else None
        self.fc2 = nnx.Linear(hidden, dim, use_bias=bias, rngs=rngs)

    def __call__(self, x):
        x = self.act(self.fc1(x))
        if self.grn is not None:
            x = self.grn(x)
        return self.fc2(x)


class ConvNeXtBlock(nnx.Module):
    def __init__(self, dim, kernel, cfg, drop_path=0.0, *, rngs):
        bias = cfg["conv_bias"]
        self.conv_dw = _conv(dim, dim, kernel, pad=kernel // 2, groups=dim, bias=bias, rngs=rngs)
        self.norm = _norm(cfg["norm_layer"], dim, cfg["norm_eps"], rngs=rngs)
        act = _ACTS[cfg["act_layer"]]
        self.mlp = Mlp(dim, int(cfg["mlp_ratio"] * dim), act, cfg["use_grn"], bias, rngs=rngs)
        ls = cfg["ls_init_value"]
        self.gamma = nnx.Param(ls * jnp.ones(dim)) if ls is not None else None
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        y = self.mlp(self.norm(self.conv_dw(x)))
        if self.gamma is not None:
            y = self.gamma[...] * y
        return x + self.drop_path(y)


class ConvNeXtStage(nnx.Module):
    def __init__(self, in_chs, out_chs, kernel, stride, depth, cfg, dpr, *, rngs):
        if in_chs != out_chs or stride > 1:
            self.downsample = nnx.Sequential(
                _norm(cfg["norm_layer"], in_chs, cfg["norm_eps"], rngs=rngs),
                _conv(in_chs, out_chs, 2 if stride > 1 else 1, stride, bias=cfg["conv_bias"], rngs=rngs),
            )  # fmt: skip
        else:
            self.downsample = None
        self.blocks = nnx.List(
            [ConvNeXtBlock(out_chs, kernel, cfg, dpr[i], rngs=rngs) for i in range(depth)]
        )

    def __call__(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        for blk in self.blocks:
            x = blk(x)
        return x


_DEFAULTS = dict(
    kernel_sizes=7, ls_init_value=1e-6, stem_type="patch", patch_size=4, head_norm_first=False,
    head_hidden_size=None, conv_bias=True, use_grn=False, act_layer="gelu", norm_layer="layernorm",
    norm_eps=None, mlp_ratio=4,
)  # fmt: skip


class ConvNeXt(ClassifierMixin, nnx.Module):
    """timm ``ConvNeXt``; keyword arguments follow timm's constructor (see ``_DEFAULTS``)."""

    def __init__(
        self,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
        **overrides,
    ):
        cfg = {**_DEFAULTS, **overrides}
        cfg["norm_layer"] = cfg["norm_layer"].removesuffix("2d")
        cfg["norm_eps"] = cfg["norm_eps"] or 1e-6
        self.num_classes, self.global_pool = num_classes, global_pool
        norm = cfg["norm_layer"]
        ks = cfg["kernel_sizes"]
        ks = (ks,) * 4 if isinstance(ks, int) else tuple(ks)
        stem, bias = cfg["stem_type"], cfg["conv_bias"]
        if stem == "patch":
            p = cfg["patch_size"]
            self.stem_conv = _conv(in_chans, dims[0], p, p, bias=bias, rngs=rngs)
            self.stem_conv2, stem_stride = None, p
        else:
            mid = make_divisible(dims[0] // 2) if "tiered" in stem else dims[0]
            self.stem_conv = _conv(in_chans, mid, 3, 2, 1, bias=bias, rngs=rngs)
            self.stem_conv2, stem_stride = _conv(mid, dims[0], 3, 2, 1, bias=bias, rngs=rngs), 4
        self.stem_act = _ACTS[cfg["act_layer"]] if "act" in stem else None
        self.stem_norm = _norm(norm, dims[0], cfg["norm_eps"], rngs=rngs)
        total = sum(depths)
        dpr = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]
        stages, prev, k = [], dims[0], 0
        for i in range(4):
            stride = 2 if stem_stride == 2 or i > 0 else 1
            stages.append(
                ConvNeXtStage(prev, dims[i], ks[i], stride, depths[i], cfg, dpr[k:], rngs=rngs)
            )
            prev, k = dims[i], k + depths[i]
        self.stages = nnx.List(stages)
        self.num_features = prev
        eps = cfg["norm_eps"]
        if cfg["head_norm_first"]:
            self.norm_pre, self.head_norm = _norm(norm, prev, eps, rngs=rngs), None
        else:
            self.norm_pre, self.head_norm = None, _norm(norm, prev, eps, rngs=rngs)
        hidden = cfg["head_hidden_size"]
        self.pre_logits = nnx.Linear(prev, hidden, rngs=rngs) if hidden else None
        self.head_hidden_size = hidden or prev
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = (
            nnx.Linear(self.head_hidden_size, num_classes, rngs=rngs) if num_classes > 0 else None
        )

    def reset_classifier(self, num_classes, global_pool=None):
        if global_pool is not None:
            self.global_pool = global_pool
        self.num_classes = num_classes
        self.fc = (
            nnx.Linear(self.head_hidden_size, num_classes, rngs=nnx.Rngs(0))
            if num_classes > 0
            else None
        )

    def _stem(self, x):
        x = self.stem_conv(x)
        if self.stem_conv2 is not None:
            if self.stem_act is not None:
                x = self.stem_act(x)
            x = self.stem_conv2(x)
        return self.stem_norm(x)

    def forward_intermediates(self, x, out_indices=None):
        """Feature maps after the stem and after each stage."""
        feats = [self._stem(x)]
        for stage in self.stages:
            feats.append(stage(feats[-1]))
        return _select_features(feats, out_indices)

    def forward_features(self, x):
        x = self._stem(x)
        for stage in self.stages:
            x = stage(x)
        return x if self.norm_pre is None else self.norm_pre(x)

    def forward_head(self, x, pre_logits=False):
        x = global_pool_nhwc(x, self.global_pool)
        if self.head_norm is not None:
            x = self.head_norm(x)
        if self.pre_logits is not None:
            x = gelu(self.pre_logits(x))
        x = self.head_drop(x)
        if pre_logits or self.fc is None:
            return x
        return self.fc(x)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_CLIP = {"mean": (0.48145466, 0.4578275, 0.40821073), "std": (0.26862954, 0.26130258, 0.27577711)}
_HALF = {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}
_T288 = {"test_input_size": (3, 288, 288)}
_V2 = {"use_grn": True, "ls_init_value": None}
_ATTO, _FEMTO = ((2, 2, 6, 2), (40, 80, 160, 320)), ((2, 2, 6, 2), (48, 96, 192, 384))
_PICO, _NANO = ((2, 2, 6, 2), (64, 128, 256, 512)), ((2, 2, 8, 2), (80, 160, 320, 640))
_ZEPTO = ((2, 2, 4, 2), (32, 64, 128, 256))
_TINY, _SMALL = ((3, 3, 9, 3), (96, 192, 384, 768)), ((3, 3, 27, 3), (96, 192, 384, 768))
_BASE, _LARGE = ((3, 3, 27, 3), (128, 256, 512, 1024)), ((3, 3, 27, 3), (192, 384, 768, 1536))

# name: ((depths, dims), constructor overrides, eval cfg) — recorded from timm 1.0.29.
# ``conv_mlp`` only changes timm's tensor layout, so it is omitted.
_CFGS = {
    "convnext_atto": (_ATTO, {}, {"crop_pct": 0.875, **_T288}),
    "convnext_atto_ols": (_ATTO, {"stem_type": "overlap_tiered"}, {"crop_pct": 0.875, **_T288}),
    "convnext_atto_rms": (_ATTO, {"norm_layer": "rmsnorm2d"}, {"crop_pct": 0.875, "test_input_size": (3, 256, 256)}),
    "convnext_femto": (_FEMTO, {}, {"crop_pct": 0.875, **_T288}),
    "convnext_femto_ols": (_FEMTO, {"stem_type": "overlap_tiered"}, {"crop_pct": 0.875, **_T288}),
    "convnext_pico": (_PICO, {}, {"crop_pct": 0.875, **_T288}),
    "convnext_pico_ols": (_PICO, {"stem_type": "overlap_tiered"}, {"crop_pct": 0.95, **_T288}),
    "convnext_nano": (_NANO, {}, {"crop_pct": 0.95, **_T288}),
    "convnext_nano_ols": (_NANO, {"stem_type": "overlap"}, {"crop_pct": 0.95, **_T288}),
    "convnext_zepto_rms": (_ZEPTO, {"norm_layer": "simplenorm"}, {"crop_pct": 0.875, **_HALF}),
    "convnext_zepto_rms_ols": (
        _ZEPTO, {"norm_layer": "simplenorm", "stem_type": "overlap_act"}, {"crop_pct": 0.9, **_HALF},
    ),
    "convnext_tiny": (_TINY, {}, {"crop_pct": 0.95, **_T288}),
    "convnext_tiny_hnf": (_TINY, {"head_norm_first": True}, {"crop_pct": 0.95, **_T288}),
    "convnext_small": (_SMALL, {}, {"crop_pct": 0.95, **_T288}),
    "convnext_base": (_BASE, {}, {"crop_pct": 0.875, **_T288}),
    "convnext_large": (_LARGE, {}, {"crop_pct": 0.875, **_T288}),
    "convnext_large_mlp": (
        _LARGE, {"head_hidden_size": 1536}, {"input_size": (3, 320, 320), "crop_pct": 1.0, **_CLIP},
    ),
    "convnext_xlarge": (((3, 3, 27, 3), (256, 512, 1024, 2048)), {}, {"crop_pct": 0.875, **_T288}),
    "convnext_xxlarge": (
        ((3, 4, 30, 3), (384, 768, 1536, 3072)), {"norm_eps": 1e-5},
        {"input_size": (3, 256, 256), "crop_pct": 1.0, **_CLIP},
    ),
    "convnextv2_atto": (_ATTO, _V2, {"crop_pct": 0.875, **_T288}),
    "convnextv2_femto": (_FEMTO, _V2, {"crop_pct": 0.875, **_T288}),
    "convnextv2_pico": (_PICO, _V2, {"crop_pct": 0.875, **_T288}),
    "convnextv2_nano": (_NANO, _V2, {"crop_pct": 0.875, **_T288}),
    "convnextv2_tiny": (_TINY, _V2, {"crop_pct": 0.875, **_T288}),
    "convnextv2_small": (_SMALL, _V2, {"crop_pct": 0.875}),
    "convnextv2_base": (_BASE, _V2, {"crop_pct": 0.875, **_T288}),
    "convnextv2_large": (_LARGE, _V2, {"crop_pct": 0.875, **_T288}),
    "convnextv2_huge": (
        ((3, 3, 27, 3), (352, 704, 1408, 2816)), _V2, {"input_size": (3, 384, 384), "crop_pct": 1.0},
    ),
    "test_convnext": (
        ((1, 2, 4, 2), (24, 32, 48, 64)), {"norm_eps": 1e-5, "act_layer": "gelu_tanh"},
        {"input_size": (3, 160, 160), "crop_pct": 0.95, **_HALF},
    ),
    "test_convnext2": (
        ((1, 1, 1, 1), (32, 64, 96, 128)), {"norm_eps": 1e-5, "act_layer": "gelu_tanh"},
        {"input_size": (3, 160, 160), "crop_pct": 0.95, **_HALF},
    ),
    "test_convnext3": (
        ((1, 1, 1, 1), (32, 64, 96, 128)),
        {"norm_eps": 1e-5, "kernel_sizes": (7, 5, 5, 3), "act_layer": "silu"},
        {"input_size": (3, 160, 160), "crop_pct": 0.95, **_HALF},
    ),
}  # fmt: skip


def _make(name):
    (depths, dims), overrides, ev = _CFGS[name]

    def entry(**kwargs):
        model = ConvNeXt(depths, dims, **{**overrides, **kwargs})
        model.default_cfg = _cfg(interpolation="bicubic", **ev)
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
