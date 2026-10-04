"""MaxViT, CoAtNet, CoAtNeXt and MaxxViT in flax nnx, NHWC. Mirrors timm.models.maxxvit.

Registers every timm configuration of the family (rw, rmlp and tf MaxViT,
rw/rmlp/tf-style CoAtNet, CoAtNeXt, MaxxViT and MaxxViT-V2) from the
configs in ``_maxxvit_cfgs``.
"""

from ..registry import _cfg, register_model
from ._maxxvit import MaxxVit
from ._maxxvit_cfgs import MAXXVIT_CFGS


def _make(name):
    base, conv, transformer, ev = MAXXVIT_CFGS[name]
    size = ev["img_size"]

    def entry(**kwargs):
        model = MaxxVit(
            **{
                **base,
                "conv_overrides": conv,
                "transformer_overrides": transformer,
                "img_size": size,
                **kwargs,
            }
        )
        extra = {k: v for k, v in ev.items() if k != "img_size"}
        model.default_cfg = _cfg(
            input_size=(3, size, size), interpolation="bicubic", fixed_input_size=True, **extra
        )
        return model

    entry.__name__ = name
    return entry


for _name in MAXXVIT_CFGS:
    register_model(_make(_name))
