"""HardCoReNAS in flax nnx, NHWC. Mirrors timm.models.hardcorenas.

The searched architectures are MobileNetV3 networks: a 32-channel stem,
inverted residual blocks with hard-swish (ReLU for ``nre`` blocks) and
optional hard-sigmoid squeeze-excite, a 960-channel 1x1 convolution, and a
1,280-wide head after pooling. timm's block strings are decoded into
:class:`~jimm.models.mobilenetv3.MobileNetV3` block tuples.
"""

from ..registry import _cfg, register_model
from .mobilenetv3 import MobileNetV3


def _decode(arch_def, in_chs=32):
    """timm block strings (``ir_r1_k5_s2_e3_c24_nre_se0.25``) -> MobileNetV3 tuples
    ``(kernel, expand, in, out, se, act, stride)``; ``ds`` blocks have no expansion."""
    cfg = []
    for block in (b for stage in arch_def for b in stage):
        _, *ops = block.split("_")
        args = {"e": 1.0, "se": 0, "act": "hswish"}
        for op in ops:
            if op == "nre":
                args["act"] = "relu"
            elif op.startswith("se"):
                args["se"] = 1  # timm ratio 0.25 of the expanded width
            else:
                args[op[0]] = float(op[1:])
        assert args["r"] == 1
        out = int(args["c"])
        cfg.append(
            (int(args["k"]), args["e"], in_chs, out, args["se"], args["act"], int(args["s"]))
        )
        in_chs = out
    return cfg


_STEM = [["ds_r1_k3_s1_e1_c16_nre"]]
_ARCHS = {
    "hardcorenas_a": [
        ["ir_r1_k5_s2_e3_c24_nre", "ir_r1_k5_s1_e3_c24_nre_se0.25"],
        ["ir_r1_k5_s2_e3_c40_nre", "ir_r1_k5_s1_e6_c40_nre_se0.25"],
        ["ir_r1_k5_s2_e6_c80_se0.25", "ir_r1_k5_s1_e6_c80_se0.25"],
        ["ir_r1_k5_s1_e6_c112_se0.25", "ir_r1_k5_s1_e6_c112_se0.25"],
        ["ir_r1_k5_s2_e6_c192_se0.25", "ir_r1_k5_s1_e6_c192_se0.25"],
    ],
    "hardcorenas_b": [
        ["ir_r1_k5_s2_e3_c24_nre", "ir_r1_k5_s1_e3_c24_nre_se0.25", "ir_r1_k3_s1_e3_c24_nre"],
        ["ir_r1_k5_s2_e3_c40_nre", "ir_r1_k5_s1_e3_c40_nre", "ir_r1_k5_s1_e3_c40_nre"],
        ["ir_r1_k5_s2_e3_c80", "ir_r1_k5_s1_e3_c80", "ir_r1_k3_s1_e3_c80", "ir_r1_k3_s1_e3_c80"],
        [
            "ir_r1_k5_s1_e3_c112",
            "ir_r1_k3_s1_e3_c112",
            "ir_r1_k3_s1_e3_c112",
            "ir_r1_k3_s1_e3_c112",
        ],
        ["ir_r1_k5_s2_e6_c192_se0.25", "ir_r1_k5_s1_e6_c192_se0.25", "ir_r1_k3_s1_e3_c192_se0.25"],
    ],
    "hardcorenas_c": [
        ["ir_r1_k5_s2_e3_c24_nre", "ir_r1_k5_s1_e3_c24_nre_se0.25"],
        [
            "ir_r1_k5_s2_e3_c40_nre",
            "ir_r1_k5_s1_e3_c40_nre",
            "ir_r1_k5_s1_e3_c40_nre",
            "ir_r1_k5_s1_e3_c40_nre",
        ],
        [
            "ir_r1_k5_s2_e4_c80",
            "ir_r1_k5_s1_e6_c80_se0.25",
            "ir_r1_k3_s1_e3_c80",
            "ir_r1_k3_s1_e3_c80",
        ],
        [
            "ir_r1_k5_s1_e6_c112_se0.25",
            "ir_r1_k3_s1_e3_c112",
            "ir_r1_k3_s1_e3_c112",
            "ir_r1_k3_s1_e3_c112",
        ],
        ["ir_r1_k5_s2_e6_c192_se0.25", "ir_r1_k5_s1_e6_c192_se0.25", "ir_r1_k3_s1_e3_c192_se0.25"],
    ],
    "hardcorenas_d": [
        ["ir_r1_k5_s2_e3_c24_nre_se0.25", "ir_r1_k5_s1_e3_c24_nre_se0.25"],
        [
            "ir_r1_k5_s2_e3_c40_nre_se0.25",
            "ir_r1_k5_s1_e4_c40_nre_se0.25",
            "ir_r1_k3_s1_e3_c40_nre_se0.25",
        ],
        [
            "ir_r1_k5_s2_e4_c80_se0.25",
            "ir_r1_k3_s1_e3_c80_se0.25",
            "ir_r1_k3_s1_e3_c80_se0.25",
            "ir_r1_k3_s1_e3_c80_se0.25",
        ],
        [
            "ir_r1_k3_s1_e4_c112_se0.25",
            "ir_r1_k5_s1_e4_c112_se0.25",
            "ir_r1_k3_s1_e3_c112_se0.25",
            "ir_r1_k5_s1_e3_c112_se0.25",
        ],
        [
            "ir_r1_k5_s2_e6_c192_se0.25",
            "ir_r1_k5_s1_e6_c192_se0.25",
            "ir_r1_k5_s1_e6_c192_se0.25",
            "ir_r1_k3_s1_e6_c192_se0.25",
        ],
    ],
    "hardcorenas_e": [
        ["ir_r1_k5_s2_e3_c24_nre_se0.25", "ir_r1_k5_s1_e3_c24_nre_se0.25"],
        [
            "ir_r1_k5_s2_e6_c40_nre_se0.25",
            "ir_r1_k5_s1_e4_c40_nre_se0.25",
            "ir_r1_k5_s1_e4_c40_nre_se0.25",
            "ir_r1_k3_s1_e3_c40_nre_se0.25",
        ],
        ["ir_r1_k5_s2_e4_c80_se0.25", "ir_r1_k3_s1_e6_c80_se0.25"],
        [
            "ir_r1_k5_s1_e6_c112_se0.25",
            "ir_r1_k5_s1_e6_c112_se0.25",
            "ir_r1_k5_s1_e6_c112_se0.25",
            "ir_r1_k5_s1_e3_c112_se0.25",
        ],
        [
            "ir_r1_k5_s2_e6_c192_se0.25",
            "ir_r1_k5_s1_e6_c192_se0.25",
            "ir_r1_k5_s1_e6_c192_se0.25",
            "ir_r1_k3_s1_e6_c192_se0.25",
        ],
    ],
    "hardcorenas_f": [
        ["ir_r1_k5_s2_e3_c24_nre_se0.25", "ir_r1_k5_s1_e3_c24_nre_se0.25"],
        ["ir_r1_k5_s2_e6_c40_nre_se0.25", "ir_r1_k5_s1_e6_c40_nre_se0.25"],
        [
            "ir_r1_k5_s2_e6_c80_se0.25",
            "ir_r1_k5_s1_e6_c80_se0.25",
            "ir_r1_k3_s1_e3_c80_se0.25",
            "ir_r1_k3_s1_e3_c80_se0.25",
        ],
        [
            "ir_r1_k3_s1_e6_c112_se0.25",
            "ir_r1_k5_s1_e6_c112_se0.25",
            "ir_r1_k5_s1_e6_c112_se0.25",
            "ir_r1_k3_s1_e3_c112_se0.25",
        ],
        [
            "ir_r1_k5_s2_e6_c192_se0.25",
            "ir_r1_k5_s1_e6_c192_se0.25",
            "ir_r1_k3_s1_e6_c192_se0.25",
            "ir_r1_k3_s1_e6_c192_se0.25",
        ],
    ],
}


def _make(name):
    cfg = _decode(_STEM + _ARCHS[name])

    def entry(**kwargs):
        # timm's final ``cn_r1_k1_s1_c960`` block is MobileNetV3's 960-channel head conv.
        model = MobileNetV3(cfg, 960, 1280, stem_chs=32, **kwargs)
        model.default_cfg = _cfg()
        return model

    entry.__name__ = name
    return entry


for _name in _ARCHS:
    register_model(_make(_name))
