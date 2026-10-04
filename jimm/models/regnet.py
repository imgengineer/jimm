"""RegNet (X, Y, V, Z) in flax nnx, NHWC. Mirrors timm.models.regnet.

Stage widths/depths/groups are generated with timm's ``generate_regnet`` (float32
arithmetic, as torch) and ``adjust_widths_groups_comp``, so configs match exactly.
RegNetV uses pre-activation bottlenecks; RegNetZ uses inverted (``bottle_ratio`` 4)
bottlenecks with a linear output and no shortcut where the shape changes.
"""

import numpy as np
from flax import nnx

from ..layers import BatchNorm, ClassifierMixin, DropPath, SqueezeExcite, make_divisible
from ..registry import _cfg, register_model
from .byobnet import _avg2, _conv

_DEFAULTS = dict(
    bottle_ratio=1.0, se_ratio=0.0, group_min_ratio=0.0, stem_width=32, downsample="conv1x1",
    linear_out=False, preact=False, num_features=0, act="relu", norm="batchnorm",
)  # fmt: skip


def generate_regnet(wa, w0, wm, depth, group_size, quant=8):
    """timm ``generate_regnet``: per-block widths, merged into (widths, depths) per stage."""
    f32 = np.float32
    widths_cont = np.arange(depth, dtype=f32) * f32(wa) + f32(w0)
    exps = np.round(np.log(widths_cont / f32(w0)) / f32(np.log(wm)))
    widths = (np.round(f32(w0) * np.power(f32(wm), exps) / f32(quant)) * f32(quant)).astype(int)
    stage_widths, stage_depths = np.unique(widths, return_counts=True)
    return stage_widths.tolist(), stage_depths.tolist(), [group_size] * len(stage_widths)


def adjust_widths_groups(widths, bottle_ratio, group_sizes, min_ratio=0.0):
    """timm ``adjust_widths_groups_comp``: make bottleneck widths divisible by the group size."""
    bot = [int(w * bottle_ratio) for w in widths]
    group_sizes = [min(g, b) for g, b in zip(group_sizes, bot)]
    if min_ratio:
        bot = [make_divisible(b, g) for b, g in zip(bot, group_sizes)]
    else:
        bot = [int(round(b / g) * g) for b, g in zip(bot, group_sizes)]
    return [int(b / bottle_ratio) for b in bot], group_sizes


def gen_cfg(depth, w0, wa, wm, group_size, group_min_ratio=0.0, bottle_ratio=1.0):
    """Returns (widths, depths, group_sizes) per stage."""
    widths, depths, gs = generate_regnet(wa, w0, wm, depth, group_size)
    widths, gs = adjust_widths_groups(widths, bottle_ratio, gs, group_min_ratio)
    return widths, depths, gs


def _norm(kind, chs, *, rngs):
    if kind == "batchnorm":
        return BatchNorm(chs, rngs=rngs)
    return nnx.GroupNorm(chs, num_groups=chs // 16, epsilon=1e-5, rngs=rngs)  # group size 16


class Shortcut(nnx.Module):
    """timm ``downsample_conv`` / ``downsample_avg``; no norm in pre-activation blocks."""

    def __init__(self, kind, in_chs, out_chs, stride, norm, preact, *, rngs):
        self.pool_stride = stride if kind == "avg" and stride > 1 else 0
        conv_stride = 1 if kind == "avg" else stride
        self.conv = _conv(in_chs, out_chs, 1, conv_stride, rngs=rngs)
        self.bn = None if preact else _norm(norm, out_chs, rngs=rngs)

    def __call__(self, x):
        if self.pool_stride:
            x = _avg2(x, self.pool_stride, ceil=True)
        x = self.conv(x)
        return x if self.bn is None else self.bn(x)


def _shortcut(kind, in_chs, out_chs, stride, norm, preact, *, rngs):
    """Returns (module, use_residual) as timm's ``create_shortcut``."""
    if in_chs != out_chs or stride != 1:
        if not kind:
            return None, False
        return Shortcut(kind, in_chs, out_chs, stride, norm, preact, rngs=rngs), True
    return None, True


def _se(chs, in_chs, se_ratio, act, *, rngs):
    if not se_ratio:
        return None
    return SqueezeExcite(chs, rd_channels=int(round(in_chs * se_ratio)), act=act, rngs=rngs)


class RegNetBlock(nnx.Module):
    """timm ``Bottleneck``: conv-norm-act x2, optional SE, conv-norm, residual, act."""

    def __init__(self, in_chs, out_chs, stride, group_size, cfg, drop_path=0.0, *, rngs):
        act, norm = getattr(nnx, cfg["act"]), cfg["norm"]
        mid = int(round(out_chs * cfg["bottle_ratio"]))
        self.act = act
        self.conv1 = _conv(in_chs, mid, 1, rngs=rngs)
        self.bn1 = _norm(norm, mid, rngs=rngs)
        self.conv2 = _conv(mid, mid, 3, stride, groups=mid // group_size, rngs=rngs)
        self.bn2 = _norm(norm, mid, rngs=rngs)
        self.se = _se(mid, in_chs, cfg["se_ratio"], act, rngs=rngs)
        self.conv3 = _conv(mid, out_chs, 1, rngs=rngs)
        self.bn3 = _norm(norm, out_chs, rngs=rngs)
        self.shortcut, self.residual = _shortcut(
            cfg["downsample"], in_chs, out_chs, stride, norm, False, rngs=rngs
        )
        self.linear_out = cfg["linear_out"]
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.act(self.bn2(self.conv2(y)))
        if self.se is not None:
            y = self.se(y)
        y = self.bn3(self.conv3(y))
        if self.residual:
            y = self.drop_path(y) + (x if self.shortcut is None else self.shortcut(x))
        return y if self.linear_out else self.act(y)


class RegNetPreBlock(nnx.Module):
    """timm ``PreBottleneck``: norm-act before each conv; the shortcut taps the first norm."""

    def __init__(self, in_chs, out_chs, stride, group_size, cfg, drop_path=0.0, *, rngs):
        act, norm = getattr(nnx, cfg["act"]), cfg["norm"]
        mid = int(round(out_chs * cfg["bottle_ratio"]))
        self.act = act
        self.norm1 = _norm(norm, in_chs, rngs=rngs)
        self.conv1 = _conv(in_chs, mid, 1, rngs=rngs)
        self.norm2 = _norm(norm, mid, rngs=rngs)
        self.conv2 = _conv(mid, mid, 3, stride, groups=mid // group_size, rngs=rngs)
        self.se = _se(mid, in_chs, cfg["se_ratio"], act, rngs=rngs)
        self.norm3 = _norm(norm, mid, rngs=rngs)
        self.conv3 = _conv(mid, out_chs, 1, rngs=rngs)
        self.shortcut, self.residual = _shortcut(
            cfg["downsample"], in_chs, out_chs, stride, norm, True, rngs=rngs
        )
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        x = self.act(self.norm1(x))
        y = self.conv1(x)
        y = self.conv2(self.act(self.norm2(y)))
        if self.se is not None:
            y = self.se(y)
        y = self.conv3(self.act(self.norm3(y)))
        if self.residual:
            y = self.drop_path(y) + (x if self.shortcut is None else self.shortcut(x))
        return y


class RegNet(ClassifierMixin, nnx.Module):
    """timm ``RegNet``; ``cfg`` holds the RegNetCfg fields (see ``_DEFAULTS``)."""

    def __init__(
        self,
        cfg,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        cfg = {**_DEFAULTS, **cfg}
        self.num_classes, self.global_pool = num_classes, global_pool
        act, preact, stem_chs = getattr(nnx, cfg["act"]), cfg["preact"], cfg["stem_width"]
        self.act, self.preact = act, preact
        self.stem_conv = _conv(in_chans, stem_chs, 3, 2, rngs=rngs)
        self.stem_bn = None if preact else _norm(cfg["norm"], stem_chs, rngs=rngs)
        widths, depths, group_sizes = gen_cfg(
            cfg["depth"], cfg["w0"], cfg["wa"], cfg["wm"], cfg["group_size"],
            cfg["group_min_ratio"], cfg["bottle_ratio"],
        )  # fmt: skip
        dpr = np.linspace(0, drop_path_rate, sum(depths)).tolist()
        block = RegNetPreBlock if preact else RegNetBlock
        stages, chs, k = [], stem_chs, 0
        for w, d, g in zip(widths, depths, group_sizes):
            blocks = []
            for j in range(d):
                blocks.append(block(chs, w, 2 if j == 0 else 1, g, cfg, dpr[k], rngs=rngs))
                chs, k = w, k + 1
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        if cfg["num_features"]:
            self.final_conv = _conv(chs, cfg["num_features"], 1, rngs=rngs)
            self.final_bn = _norm(cfg["norm"], cfg["num_features"], rngs=rngs)
            self.final_act, self.num_features = True, cfg["num_features"]
        else:
            self.final_conv = self.final_bn = None
            self.final_act, self.num_features = cfg["linear_out"] or preact, chs
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.stem_conv(x)
        if self.stem_bn is not None:
            x = self.act(self.stem_bn(x))
        for stage in self.stages:
            for blk in stage:
                x = blk(x)
        if self.final_conv is not None:
            x = self.final_bn(self.final_conv(x))
        return self.act(x) if self.final_act else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_X, _Y = {}, {"se_ratio": 0.25}
_TV = {"group_min_ratio": 0.9}
_V = {"se_ratio": 0.25, "preact": True, "act": "silu"}
_Z = {"bottle_ratio": 4.0, "se_ratio": 0.25, "downsample": None, "linear_out": True, "act": "silu"}
_BI = {"interpolation": "bicubic"}
_TV2 = {**_BI, "crop_pct": 0.965}
_RA = {**_BI, "crop_pct": 0.95, "test_input_size": (3, 288, 288)}
_SEER = {**_BI, "input_size": (3, 384, 384), "crop_pct": 1.0}
_Z256 = {**_BI, "input_size": (3, 256, 256), "crop_pct": 1.0, "test_input_size": (3, 320, 320)}

# name: ((depth, w0, wa, wm, group_size), RegNetCfg overrides, eval cfg) — timm model_cfgs
_CFGS = {
    "regnetx_002": ((13, 24, 36.44, 2.49, 8), _X, _BI),
    "regnetx_004": ((22, 24, 24.48, 2.54, 16), _X, _BI),
    "regnetx_004_tv": ((22, 24, 24.48, 2.54, 16), _TV, _TV2),
    "regnetx_006": ((16, 48, 36.97, 2.24, 24), _X, _BI),
    "regnetx_008": ((16, 56, 35.73, 2.28, 16), _X, _TV2),
    "regnetx_016": ((18, 80, 34.01, 2.25, 24), _X, _TV2),
    "regnetx_032": ((25, 88, 26.31, 2.25, 48), _X, _TV2),
    "regnetx_040": ((23, 96, 38.65, 2.43, 40), _X, _BI),
    "regnetx_064": ((17, 184, 60.83, 2.07, 56), _X, _BI),
    "regnetx_080": ((23, 80, 49.56, 2.88, 120), _X, _TV2),
    "regnetx_120": ((19, 168, 73.36, 2.37, 112), _X, _BI),
    "regnetx_160": ((22, 216, 55.59, 2.1, 128), _X, _TV2),
    "regnetx_320": ((23, 320, 69.86, 2.0, 168), _X, _TV2),
    "regnety_002": ((13, 24, 36.44, 2.49, 8), _Y, _BI),
    "regnety_004": ((16, 48, 27.89, 2.09, 8), _Y, _TV2),
    "regnety_006": ((15, 48, 32.54, 2.32, 16), _Y, _BI),
    "regnety_008": ((14, 56, 38.84, 2.4, 16), _Y, _BI),
    "regnety_008_tv": ((14, 56, 38.84, 2.4, 16), {**_Y, **_TV}, _TV2),
    "regnety_016": ((27, 48, 20.71, 2.65, 24), _Y, _TV2),
    "regnety_032": ((21, 80, 42.63, 2.66, 24), _Y, _RA),
    "regnety_040": ((22, 96, 31.41, 2.24, 64), _Y, _RA),
    "regnety_064": ((25, 112, 33.22, 2.27, 72), _Y, _RA),
    "regnety_080": ((17, 192, 76.82, 2.19, 56), _Y, _RA),
    "regnety_080_tv": ((17, 192, 76.82, 2.19, 56), {**_Y, **_TV}, _TV2),
    "regnety_120": ((19, 168, 73.36, 2.37, 112), _Y, _RA),
    "regnety_160": ((18, 200, 106.23, 2.48, 112), _Y, _RA),
    "regnety_320": ((20, 232, 115.89, 2.53, 232), _Y, _TV2),
    "regnety_640": ((20, 352, 147.48, 2.4, 328), _Y, _SEER),
    "regnety_1280": ((27, 456, 160.83, 2.52, 264), _Y, _SEER),
    "regnety_2560": ((27, 640, 230.83, 2.53, 373), _Y, _SEER),
    "regnety_040_sgn": ((22, 96, 31.41, 2.24, 64), {**_Y, "act": "silu", "norm": "groupnorm"}, _RA),
    "regnetv_040": ((22, 96, 31.41, 2.24, 64), _V, _RA),
    "regnetv_064": ((25, 112, 33.22, 2.27, 72), {**_V, "downsample": "avg"}, _RA),
    "regnetz_005": ((21, 16, 10.7, 2.51, 4), {**_Z, "num_features": 1024}, _RA),
    "regnetz_040": ((28, 48, 14.5, 2.226, 8), _Z, _Z256),
    "regnetz_040_h": ((28, 48, 14.5, 2.226, 8), {**_Z, "num_features": 1536}, _Z256),
}


def _make(name):
    (depth, w0, wa, wm, gs), overrides, ev = _CFGS[name]
    cfg = dict(depth=depth, w0=w0, wa=wa, wm=wm, group_size=gs, **overrides)

    def entry(**kwargs):
        model = RegNet(cfg, **kwargs)
        model.default_cfg = _cfg(**ev)
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
