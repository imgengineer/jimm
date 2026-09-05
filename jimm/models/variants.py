"""Explicit model variants supported by the local architecture implementations.

Only register configurations that match their named architecture. Unsupported
families, stems, attention mechanisms, and training variants must not fall back
to a generic ResNet or ViT. Add new entries with architecture-level tests.
"""

from ..registry import _cfg, is_model, model_entrypoint, register_model
from . import convnext, regnet, resnet, swin_transformer, vision_transformer


def _make(name, ctor, args, fixed, input_size):
    def entry(**kwargs):
        model = ctor(*args, **fixed, **kwargs)
        model.default_cfg = _cfg(input_size=input_size)
        return model

    entry.__name__ = name
    entry.__module__ = ctor.__module__
    return entry


_SPECS = [
    (
        ("vit_huge_patch14_224",),
        vision_transformer.VisionTransformer,
        (),
        {"img_size": 224, "patch_size": 14, "embed_dim": 1280, "depth": 32, "num_heads": 16},
        (3, 224, 224),
    ),
    (
        ("vit_large_patch14_224",),
        vision_transformer.VisionTransformer,
        (),
        {"img_size": 224, "patch_size": 14, "embed_dim": 1024, "depth": 24, "num_heads": 16},
        (3, 224, 224),
    ),
    (
        ("resnet26",),
        resnet.ResNet,
        (resnet.Bottleneck, (2, 2, 2, 2)),
        {"se": False, "groups": 1, "base_width": 64},
        (3, 224, 224),
    ),
    (
        ("vit_base_patch16_384", "deit_base_patch16_384"),
        vision_transformer.VisionTransformer,
        (),
        {"img_size": 384, "patch_size": 16, "embed_dim": 768, "depth": 12, "num_heads": 12},
        (3, 384, 384),
    ),
    (
        ("vit_large_patch16_384",),
        vision_transformer.VisionTransformer,
        (),
        {"img_size": 384, "patch_size": 16, "embed_dim": 1024, "depth": 24, "num_heads": 16},
        (3, 384, 384),
    ),
    (("convnext_femto",), convnext.ConvNeXt, ((2, 2, 6, 2), (48, 96, 192, 384)), {}, (3, 224, 224)),
    (("convnext_nano",), convnext.ConvNeXt, ((2, 2, 8, 2), (80, 160, 320, 640)), {}, (3, 224, 224)),
    (("convnext_pico",), convnext.ConvNeXt, ((2, 2, 6, 2), (64, 128, 256, 512)), {}, (3, 224, 224)),
    (
        ("vit_small_patch16_384",),
        vision_transformer.VisionTransformer,
        (),
        {"img_size": 384, "patch_size": 16, "embed_dim": 384, "depth": 12, "num_heads": 6},
        (3, 384, 384),
    ),
    (
        ("seresnet101",),
        resnet.ResNet,
        (resnet.Bottleneck, (3, 4, 23, 3)),
        {"se": True, "groups": 1, "base_width": 64},
        (3, 224, 224),
    ),
    (
        ("seresnet152",),
        resnet.ResNet,
        (resnet.Bottleneck, (3, 8, 36, 3)),
        {"se": True, "groups": 1, "base_width": 64},
        (3, 224, 224),
    ),
    (
        ("seresnext101_32x4d",),
        resnet.ResNet,
        (resnet.Bottleneck, (3, 4, 23, 3)),
        {"se": True, "groups": 32, "base_width": 4},
        (3, 224, 224),
    ),
    (
        ("vit_tiny_patch16_384",),
        vision_transformer.VisionTransformer,
        (),
        {"img_size": 384, "patch_size": 16, "embed_dim": 192, "depth": 12, "num_heads": 3},
        (3, 384, 384),
    ),
    (
        ("regnety_008_tv",),
        regnet.RegNet,
        (*regnet.gen_cfg(14, 56, 38.84, 2.4, 16),),
        {"se_ratio": 0.25},
        (3, 224, 224),
    ),
    (
        ("resnet200",),
        resnet.ResNet,
        (resnet.Bottleneck, (3, 24, 36, 3)),
        {"se": False, "groups": 1, "base_width": 64},
        (3, 224, 224),
    ),
    (
        ("resnext101_32x16d",),
        resnet.ResNet,
        (resnet.Bottleneck, (3, 4, 23, 3)),
        {"se": False, "groups": 32, "base_width": 16},
        (3, 224, 224),
    ),
    (
        ("resnext101_32x32d",),
        resnet.ResNet,
        (resnet.Bottleneck, (3, 4, 23, 3)),
        {"se": False, "groups": 32, "base_width": 32},
        (3, 224, 224),
    ),
    (
        ("resnext101_32x4d",),
        resnet.ResNet,
        (resnet.Bottleneck, (3, 4, 23, 3)),
        {"se": False, "groups": 32, "base_width": 4},
        (3, 224, 224),
    ),
    (
        ("resnext101_64x4d",),
        resnet.ResNet,
        (resnet.Bottleneck, (3, 4, 23, 3)),
        {"se": False, "groups": 64, "base_width": 4},
        (3, 224, 224),
    ),
    (
        ("seresnext101_32x8d",),
        resnet.ResNet,
        (resnet.Bottleneck, (3, 4, 23, 3)),
        {"se": True, "groups": 32, "base_width": 8},
        (3, 224, 224),
    ),
    (
        ("seresnext101_64x4d",),
        resnet.ResNet,
        (resnet.Bottleneck, (3, 4, 23, 3)),
        {"se": True, "groups": 64, "base_width": 4},
        (3, 224, 224),
    ),
    (
        ("swin_base_patch4_window12_384",),
        swin_transformer.SwinTransformer,
        (),
        {
            "img_size": 384,
            "embed_dim": 128,
            "depths": (2, 2, 18, 2),
            "num_heads": (4, 8, 16, 32),
            "window_size": 12,
        },
        (3, 384, 384),
    ),
    (
        ("swin_large_patch4_window12_384",),
        swin_transformer.SwinTransformer,
        (),
        {
            "img_size": 384,
            "embed_dim": 192,
            "depths": (2, 2, 18, 2),
            "num_heads": (6, 12, 24, 48),
            "window_size": 12,
        },
        (3, 384, 384),
    ),
    (
        ("swin_large_patch4_window7_224",),
        swin_transformer.SwinTransformer,
        (),
        {"img_size": 224, "embed_dim": 192, "depths": (2, 2, 18, 2), "num_heads": (6, 12, 24, 48)},
        (3, 224, 224),
    ),
    (
        ("swinv2_base_window12_192",),
        swin_transformer.SwinTransformer,
        (),
        {
            "img_size": 192,
            "embed_dim": 128,
            "depths": (2, 2, 18, 2),
            "num_heads": (4, 8, 16, 32),
            "window_size": 12,
            "block_cls": swin_transformer.SwinV2Block,
        },
        (3, 192, 192),
    ),
    (
        ("swinv2_base_window16_256",),
        swin_transformer.SwinTransformer,
        (),
        {
            "img_size": 256,
            "embed_dim": 128,
            "depths": (2, 2, 18, 2),
            "num_heads": (4, 8, 16, 32),
            "window_size": 16,
            "block_cls": swin_transformer.SwinV2Block,
        },
        (3, 256, 256),
    ),
    (
        ("swinv2_base_window8_256",),
        swin_transformer.SwinTransformer,
        (),
        {
            "img_size": 256,
            "embed_dim": 128,
            "depths": (2, 2, 18, 2),
            "num_heads": (4, 8, 16, 32),
            "window_size": 8,
            "block_cls": swin_transformer.SwinV2Block,
        },
        (3, 256, 256),
    ),
    (
        ("swinv2_large_window12_192",),
        swin_transformer.SwinTransformer,
        (),
        {
            "img_size": 192,
            "embed_dim": 192,
            "depths": (2, 2, 18, 2),
            "num_heads": (6, 12, 24, 48),
            "window_size": 12,
            "block_cls": swin_transformer.SwinV2Block,
        },
        (3, 192, 192),
    ),
    (
        ("swinv2_small_window16_256",),
        swin_transformer.SwinTransformer,
        (),
        {
            "img_size": 256,
            "embed_dim": 96,
            "depths": (2, 2, 18, 2),
            "num_heads": (3, 6, 12, 24),
            "window_size": 16,
            "block_cls": swin_transformer.SwinV2Block,
        },
        (3, 256, 256),
    ),
    (
        ("swinv2_tiny_window16_256",),
        swin_transformer.SwinTransformer,
        (),
        {
            "img_size": 256,
            "embed_dim": 96,
            "depths": (2, 2, 6, 2),
            "num_heads": (3, 6, 12, 24),
            "window_size": 16,
            "block_cls": swin_transformer.SwinV2Block,
        },
        (3, 256, 256),
    ),
    (
        ("vit_base_patch32_224",),
        vision_transformer.VisionTransformer,
        (),
        {"img_size": 224, "patch_size": 32, "embed_dim": 768, "depth": 12, "num_heads": 12},
        (3, 224, 224),
    ),
    (
        ("vit_base_patch32_384",),
        vision_transformer.VisionTransformer,
        (),
        {"img_size": 384, "patch_size": 32, "embed_dim": 768, "depth": 12, "num_heads": 12},
        (3, 384, 384),
    ),
    (
        ("vit_base_patch8_224",),
        vision_transformer.VisionTransformer,
        (),
        {"img_size": 224, "patch_size": 8, "embed_dim": 768, "depth": 12, "num_heads": 12},
        (3, 224, 224),
    ),
    (
        ("vit_large_patch32_224",),
        vision_transformer.VisionTransformer,
        (),
        {"img_size": 224, "patch_size": 32, "embed_dim": 1024, "depth": 24, "num_heads": 16},
        (3, 224, 224),
    ),
    (
        ("vit_large_patch32_384",),
        vision_transformer.VisionTransformer,
        (),
        {"img_size": 384, "patch_size": 32, "embed_dim": 1024, "depth": 24, "num_heads": 16},
        (3, 384, 384),
    ),
]

for _names, _ctor, _args, _fixed, _input_size in _SPECS:
    for _name in _names:
        globals()[_name] = (
            model_entrypoint(_name)
            if is_model(_name)
            else register_model(_make(_name, _ctor, _args, _fixed, _input_size))
        )
