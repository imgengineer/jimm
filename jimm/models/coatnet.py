"""CoAtNet in flax nnx, NHWC. Mirrors timm.models.maxxvit "rw" CoAtNet configurations.

Two MBConv stages precede two transformer stages that attend across the whole
feature map with relative-position bias.
"""

from ..registry import _cfg, register_model
from ._maxxvit import ConvCfg, MaxxVit, TransformerCfg

# rw CoAtNet: pre-norm activation, ReLU SE at 1/4, no MBConv output biases.
_CFGS = {
    "coatnet_0_rw_224": dict(
        embed_dim=(96, 192, 384, 768),
        depths=(2, 3, 7, 2),
        stem_width=(32, 64),
        conv_cfg=ConvCfg(stride_mode="pool", pre_norm_act=True, attn_early=True, se_act="relu"),
        transformer_cfg=TransformerCfg(shortcut_bias=False),
    ),
    "coatnet_1_rw_224": dict(
        embed_dim=(96, 192, 384, 768),
        depths=(2, 6, 14, 2),
        stem_width=(32, 64),
        conv_cfg=ConvCfg(stride_mode="dw", pre_norm_act=True, attn_early=True, se_act="relu"),
        transformer_cfg=TransformerCfg(shortcut_bias=False),
    ),
    "coatnet_2_rw_224": dict(
        embed_dim=(128, 256, 512, 1024),
        depths=(2, 6, 14, 2),
        stem_width=(64, 128),
        conv_cfg=ConvCfg(stride_mode="dw", pre_norm_act=True, se_act="silu"),
        transformer_cfg=TransformerCfg(),
    ),
}


def _make(name):
    def entry(**kwargs):
        model = MaxxVit(block_types=("C", "C", "T", "T"), **_CFGS[name], **kwargs)
        model.default_cfg = _cfg(
            crop_pct=0.95, interpolation="bicubic", mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
        )
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
