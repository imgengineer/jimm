"""MaxViT in flax nnx, NHWC. Mirrors timm.models.maxxvit "rw" MaxViT configurations.

Each block is an MBConv followed by window and grid partition attention with
relative-position bias. ``maxvit_small_rw_224`` and ``maxvit_base_rw_224`` have no
timm counterpart; they scale the same rw blocks.
"""

from ..registry import _cfg, register_model
from ._maxxvit import ConvCfg, MaxxVit, TransformerCfg

# rw MaxViT: stride in the depthwise conv, SiLU SE at 1/16, no output biases.
_CONV = ConvCfg(stride_mode="dw", attn_ratio=1 / 16)
_TRANSFORMER = TransformerCfg()

# embed widths, depths, stem widths, input size
_CFGS = {
    "maxvit_pico_rw_256": ((32, 64, 128, 256), (2, 2, 5, 2), (24, 32), 256),
    "maxvit_nano_rw_256": ((64, 128, 256, 512), (1, 2, 3, 1), (32, 64), 256),
    "maxvit_tiny_rw_224": ((64, 128, 256, 512), (2, 2, 5, 2), (32, 64), 224),
    "maxvit_small_rw_224": ((64, 128, 256, 512), (2, 2, 13, 2), (32, 64), 224),
    "maxvit_base_rw_224": ((96, 192, 384, 768), (2, 6, 14, 2), (32, 64), 224),
}


def _make(name):
    embed_dim, depths, stem_width, size = _CFGS[name]

    def entry(**kwargs):
        kwargs.setdefault("img_size", size)
        model = MaxxVit(embed_dim, depths, ("M",) * 4, stem_width, _CONV, _TRANSFORMER, **kwargs)
        model.default_cfg = _cfg(
            input_size=(3, size, size),
            crop_pct=0.95,
            interpolation="bicubic",
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
