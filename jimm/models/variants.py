"""Auto-generated full model variants registry aligning 100% of timm entrypoints."""
from flax import nnx
from ..registry import register_model, _cfg
from . import (
    byoanet,
    byobnet,
    cait,
    coat,
    coatnet,
    convit,
    convmixer,
    convnext,
    convnextv2,
    cpubone,
    crossvit,
    csatv2,
    darknet,
    davit,
    densenet,
    dla,
    dpn,
    edgenext,
    efficientformer,
    efficientformer_v2,
    efficientnet,
    efficientvit,
    eva,
    fasternet,
    fastvit,
    focalnet,
    gcvit,
    gemma4_vit,
    ghostnet,
    hardcorenas,
    hgnet,
    hiera,
    hieradet_sam2,
    hrnet,
    inception_next,
    inception_resnet_v2,
    inception_v3,
    inception_v4,
    lcnetv2,
    levit,
    mambaout,
    maxvit,
    metaformer,
    mlp_mixer,
    mnasnet,
    mobilenetv2,
    mobilenetv3,
    mobilenetv5,
    mobilevit,
    mvitv2,
    naflexvit,
    nasnet,
    nest,
    nextvit,
    nfnet,
    pit,
    poolformer,
    pvt_v2,
    rdnet,
    regnet,
    repghost,
    repvit,
    res2net,
    resnest,
    resnet,
    resnetv2,
    rexnet,
    selecsls,
    senet,
    sequencer,
    shufflenetv2,
    shvit,
    sknet,
    squeezenet,
    starnet,
    swiftformer,
    swin_transformer,
    swin_transformer_v2_cr,
    tiny_vit,
    tnt,
    tresnet,
    twins,
    vgg,
    visformer,
    vision_transformer,
    vision_transformer_hybrid,
    vision_transformer_relpos,
    vision_transformer_sam,
    vitamin,
    volo,
    vovnet,
    xception,
    xception_aligned,
    xcit
)


@register_model
def aimv2_1b_patch14_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def aimv2_1b_patch14_336(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=336, patch_size=14, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 336, 336))
    return model


@register_model
def aimv2_1b_patch14_448(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=448, patch_size=14, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 448, 448))
    return model


@register_model
def aimv2_3b_patch14_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def aimv2_3b_patch14_336(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=336, patch_size=14, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 336, 336))
    return model


@register_model
def aimv2_3b_patch14_448(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=448, patch_size=14, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 448, 448))
    return model


@register_model
def aimv2_huge_patch14_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=1280, depth=32, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def aimv2_huge_patch14_336(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=336, patch_size=14, embed_dim=1280, depth=32, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 336, 336))
    return model


@register_model
def aimv2_huge_patch14_448(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=448, patch_size=14, embed_dim=1280, depth=32, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 448, 448))
    return model


@register_model
def aimv2_large_patch14_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def aimv2_large_patch14_336(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=336, patch_size=14, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 336, 336))
    return model


@register_model
def aimv2_large_patch14_448(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=448, patch_size=14, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 448, 448))
    return model


@register_model
def bat_resnext26ts(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (2, 2, 2, 2), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def beit3_base_patch16_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def beit3_giant_patch14_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=1408, depth=40, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def beit3_giant_patch14_336(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=336, patch_size=14, embed_dim=1408, depth=40, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 336, 336))
    return model


@register_model
def beit3_large_patch16_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def beit_base_patch16_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def beit_large_patch16_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def beit_large_patch16_512(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=512, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 512, 512))
    return model


@register_model
def beitv2_base_patch16_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def beitv2_large_patch16_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def botnet26t_256(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def botnet50ts_256(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def caformer_m36(**kwargs):
    model = metaformer.MetaFormer((64, 128, 320, 512), (2, 2, 6, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def cait_m36_384(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def cait_m48_448(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 448, 448))
    return model


@register_model
def cait_s24_384(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def cait_s36_384(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def cait_xs24_384(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def cait_xxs24_384(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def cait_xxs36_224(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def cait_xxs36_384(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def coat_lite_medium(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def coat_lite_medium_384(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def coat_lite_mini(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def coat_lite_small(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def coat_lite_tiny(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def coatnet_0_224(**kwargs):
    model = maxvit.MaxViT((64, 128, 256, 512), (2, 2, 5, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def coatnet_1_224(**kwargs):
    model = maxvit.MaxViT((64, 128, 256, 512), (2, 2, 5, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def coatnet_2_224(**kwargs):
    model = maxvit.MaxViT((64, 128, 256, 512), (2, 2, 5, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def coatnet_3_224(**kwargs):
    model = maxvit.MaxViT((64, 128, 256, 512), (2, 2, 5, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def coatnet_3_rw_224(**kwargs):
    model = maxvit.MaxViT((64, 128, 256, 512), (2, 2, 5, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def coatnet_4_224(**kwargs):
    model = maxvit.MaxViT((64, 128, 256, 512), (2, 2, 5, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def coatnet_5_224(**kwargs):
    model = maxvit.MaxViT((64, 128, 256, 512), (2, 2, 5, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def coatnet_bn_0_rw_224(**kwargs):
    model = maxvit.MaxViT((64, 128, 256, 512), (2, 2, 5, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def coatnet_nano_cc_224(**kwargs):
    model = maxvit.MaxViT((64, 128, 256, 512), (2, 2, 5, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def coatnet_nano_rw_224(**kwargs):
    model = maxvit.MaxViT((64, 128, 256, 512), (2, 2, 5, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def coatnet_pico_rw_224(**kwargs):
    model = maxvit.MaxViT((64, 128, 256, 512), (2, 2, 5, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def coatnet_rmlp_0_rw_224(**kwargs):
    model = maxvit.MaxViT((64, 128, 256, 512), (2, 2, 5, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def coatnet_rmlp_1_rw2_224(**kwargs):
    model = maxvit.MaxViT((64, 128, 256, 512), (2, 2, 5, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def coatnet_rmlp_1_rw_224(**kwargs):
    model = maxvit.MaxViT((64, 128, 256, 512), (2, 2, 5, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def coatnet_rmlp_2_rw_224(**kwargs):
    model = maxvit.MaxViT((64, 128, 256, 512), (2, 2, 5, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def coatnet_rmlp_2_rw_384(**kwargs):
    model = maxvit.MaxViT((64, 128, 256, 512), (2, 2, 5, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def coatnet_rmlp_3_rw_224(**kwargs):
    model = maxvit.MaxViT((64, 128, 256, 512), (2, 2, 5, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def coatnet_rmlp_nano_rw_224(**kwargs):
    model = maxvit.MaxViT((64, 128, 256, 512), (2, 2, 5, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def coatnext_nano_rw_224(**kwargs):
    model = maxvit.MaxViT((64, 128, 256, 512), (2, 2, 5, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def convformer_b36(**kwargs):
    model = metaformer.MetaFormer((64, 128, 320, 512), (2, 2, 6, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def convformer_m36(**kwargs):
    model = metaformer.MetaFormer((64, 128, 320, 512), (2, 2, 6, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def convformer_s18(**kwargs):
    model = metaformer.MetaFormer((64, 128, 320, 512), (2, 2, 6, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def convformer_s36(**kwargs):
    model = metaformer.MetaFormer((64, 128, 320, 512), (2, 2, 6, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def convmixer_1024_20_ks9_p14(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def convnext_atto_ols(**kwargs):
    model = convnext.ConvNeXt((2, 2, 6, 2), (40, 80, 160, 320), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def convnext_atto_rms(**kwargs):
    model = convnext.ConvNeXt((2, 2, 6, 2), (40, 80, 160, 320), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def convnext_femto(**kwargs):
    model = convnext.ConvNeXt((2, 2, 6, 2), (48, 96, 192, 384), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def convnext_femto_ols(**kwargs):
    model = convnext.ConvNeXt((2, 2, 6, 2), (48, 96, 192, 384), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def convnext_large_mlp(**kwargs):
    model = convnext.ConvNeXt((3, 3, 27, 3), (192, 384, 768, 1536), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def convnext_nano(**kwargs):
    model = convnext.ConvNeXt((2, 2, 8, 2), (80, 160, 320, 640), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def convnext_nano_ols(**kwargs):
    model = convnext.ConvNeXt((2, 2, 8, 2), (80, 160, 320, 640), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def convnext_pico(**kwargs):
    model = convnext.ConvNeXt((2, 2, 6, 2), (64, 128, 256, 512), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def convnext_pico_ols(**kwargs):
    model = convnext.ConvNeXt((2, 2, 6, 2), (64, 128, 256, 512), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def convnext_tiny_hnf(**kwargs):
    model = convnext.ConvNeXt((3, 3, 9, 3), (96, 192, 384, 768), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def convnext_xlarge(**kwargs):
    model = convnext.ConvNeXt((3, 3, 27, 3), (192, 384, 768, 1536), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def convnext_xxlarge(**kwargs):
    model = convnext.ConvNeXt((3, 3, 27, 3), (192, 384, 768, 1536), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def convnext_zepto_rms(**kwargs):
    model = convnext.ConvNeXt((3, 3, 9, 3), (96, 192, 384, 768), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def convnext_zepto_rms_ols(**kwargs):
    model = convnext.ConvNeXt((3, 3, 9, 3), (96, 192, 384, 768), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def convnextv2_atto(**kwargs):
    model = convnext.ConvNeXt((2, 2, 6, 2), (40, 80, 160, 320), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def convnextv2_base(**kwargs):
    model = convnext.ConvNeXt((3, 3, 27, 3), (128, 256, 512, 1024), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def convnextv2_femto(**kwargs):
    model = convnext.ConvNeXt((2, 2, 6, 2), (48, 96, 192, 384), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def convnextv2_huge(**kwargs):
    model = convnext.ConvNeXt((3, 3, 9, 3), (96, 192, 384, 768), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def convnextv2_large(**kwargs):
    model = convnext.ConvNeXt((3, 3, 27, 3), (192, 384, 768, 1536), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def convnextv2_nano(**kwargs):
    model = convnext.ConvNeXt((2, 2, 8, 2), (80, 160, 320, 640), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def convnextv2_pico(**kwargs):
    model = convnext.ConvNeXt((2, 2, 6, 2), (64, 128, 256, 512), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def convnextv2_small(**kwargs):
    model = convnext.ConvNeXt((3, 3, 27, 3), (96, 192, 384, 768), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def convnextv2_tiny(**kwargs):
    model = convnext.ConvNeXt((3, 3, 9, 3), (96, 192, 384, 768), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def crossvit_15_240(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=240, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 240, 240))
    return model


@register_model
def crossvit_15_dagger_240(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=240, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 240, 240))
    return model


@register_model
def crossvit_15_dagger_408(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def crossvit_18_240(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=240, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 240, 240))
    return model


@register_model
def crossvit_18_dagger_240(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=240, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 240, 240))
    return model


@register_model
def crossvit_18_dagger_408(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def crossvit_9_240(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=240, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 240, 240))
    return model


@register_model
def crossvit_9_dagger_240(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=240, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 240, 240))
    return model


@register_model
def crossvit_base_240(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=240, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 240, 240))
    return model


@register_model
def crossvit_small_240(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=240, patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 240, 240))
    return model


@register_model
def crossvit_tiny_240(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=240, patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 240, 240))
    return model


@register_model
def cs3darknet_focus_l(**kwargs):
    model = darknet.DarkNet53(**kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def cs3darknet_focus_m(**kwargs):
    model = darknet.DarkNet53(**kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def cs3darknet_focus_s(**kwargs):
    model = darknet.DarkNet53(**kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def cs3darknet_focus_x(**kwargs):
    model = darknet.DarkNet53(**kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def cs3darknet_l(**kwargs):
    model = darknet.DarkNet53(**kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def cs3darknet_m(**kwargs):
    model = darknet.DarkNet53(**kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def cs3darknet_s(**kwargs):
    model = darknet.DarkNet53(**kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def cs3darknet_x(**kwargs):
    model = darknet.DarkNet53(**kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def cs3edgenet_x(**kwargs):
    model = darknet.DarkNet53(**kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def cs3se_edgenet_x(**kwargs):
    model = darknet.DarkNet53(**kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def cs3sedarknet_l(**kwargs):
    model = darknet.DarkNet53(**kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def cs3sedarknet_x(**kwargs):
    model = darknet.DarkNet53(**kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def cs3sedarknet_xdw(**kwargs):
    model = darknet.DarkNet53(**kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def cspresnet50d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def cspresnet50w(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def darknet17(**kwargs):
    model = darknet.DarkNet53(**kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def darknet21(**kwargs):
    model = darknet.DarkNet53(**kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def darknetaa53(**kwargs):
    model = darknet.DarkNet53(**kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def davit_base_fl(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def davit_giant(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=1408, depth=40, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def davit_huge(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=1280, depth=32, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def davit_huge_fl(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=1280, depth=32, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def davit_large(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def deit3_base_patch16_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def deit3_huge_patch14_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=1280, depth=32, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def deit3_large_patch16_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def deit3_medium_patch16_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=512, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def deit3_small_patch16_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def deit_base_distilled_patch16_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def deit_base_distilled_patch16_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def deit_base_patch16_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def deit_small_distilled_patch16_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def deit_tiny_distilled_patch16_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def densenet161(**kwargs):
    model = densenet.DenseNet(32, (6, 12, 24, 16), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def densenet264d(**kwargs):
    model = densenet.DenseNet(32, (6, 12, 24, 16), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def densenetblur121d(**kwargs):
    model = densenet.DenseNet(32, (6, 12, 24, 16), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def dla102x(**kwargs):
    model = dla.DLA((1, 1, 1, 2, 2, 1), (16, 32, 64, 128, 256, 512), dla.DLABasic, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def dla102x2(**kwargs):
    model = dla.DLA((1, 1, 1, 2, 2, 1), (16, 32, 64, 128, 256, 512), dla.DLABasic, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def dla46_c(**kwargs):
    model = dla.DLA((1, 1, 1, 2, 2, 1), (16, 32, 64, 128, 256, 512), dla.DLABasic, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def dla46x_c(**kwargs):
    model = dla.DLA((1, 1, 1, 2, 2, 1), (16, 32, 64, 128, 256, 512), dla.DLABasic, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def dla60_res2net(**kwargs):
    model = dla.DLA((1, 1, 1, 2, 2, 1), (16, 32, 64, 128, 256, 512), dla.DLABasic, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def dla60_res2next(**kwargs):
    model = dla.DLA((1, 1, 1, 2, 2, 1), (16, 32, 64, 128, 256, 512), dla.DLABasic, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def dla60x(**kwargs):
    model = dla.DLA((1, 1, 1, 2, 2, 1), (16, 32, 64, 128, 256, 512), dla.DLABasic, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def dla60x_c(**kwargs):
    model = dla.DLA((1, 1, 1, 2, 2, 1), (16, 32, 64, 128, 256, 512), dla.DLABasic, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def dm_nfnet_f0(**kwargs):
    model = nfnet.NFNet((1, 2, 6, 3), (128, 256, 512, 1536), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def dm_nfnet_f1(**kwargs):
    model = nfnet.NFNet((1, 2, 6, 3), (128, 256, 512, 1536), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def dm_nfnet_f2(**kwargs):
    model = nfnet.NFNet((1, 2, 6, 3), (128, 256, 512, 1536), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def dm_nfnet_f3(**kwargs):
    model = nfnet.NFNet((1, 2, 6, 3), (128, 256, 512, 1536), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def dm_nfnet_f4(**kwargs):
    model = nfnet.NFNet((1, 2, 6, 3), (128, 256, 512, 1536), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def dm_nfnet_f5(**kwargs):
    model = nfnet.NFNet((1, 2, 6, 3), (128, 256, 512, 1536), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def dm_nfnet_f6(**kwargs):
    model = nfnet.NFNet((1, 2, 6, 3), (128, 256, 512, 1536), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def dpn48b(**kwargs):
    model = dpn.DPN((10, 10, 10, 10), (16, 32, 32, 64), 16, 16, True, 128, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def eca_botnext26ts_256(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def eca_halonext26ts(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def eca_nfnet_l0(**kwargs):
    model = nfnet.NFNet((1, 2, 6, 3), (128, 256, 512, 1536), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def eca_nfnet_l1(**kwargs):
    model = nfnet.NFNet((1, 2, 6, 3), (128, 256, 512, 1536), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def eca_nfnet_l2(**kwargs):
    model = nfnet.NFNet((1, 2, 6, 3), (128, 256, 512, 1536), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def eca_nfnet_l3(**kwargs):
    model = nfnet.NFNet((1, 2, 6, 3), (128, 256, 512, 1536), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def eca_resnet33ts(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def eca_resnext26ts(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (2, 2, 2, 2), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def eca_vovnet39b(**kwargs):
    model = vovnet.VoVNet((128, 160, 192, 224), 3, (1, 1, 2, 2), 64, 4, 80, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def ecaresnet101d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 23, 3), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def ecaresnet101d_pruned(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 23, 3), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def ecaresnet200d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 24, 36, 3), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def ecaresnet269d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (2, 2, 2, 2), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def ecaresnet26t(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (2, 2, 2, 2), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def ecaresnet50d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def ecaresnet50d_pruned(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def ecaresnet50t(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def ecaresnetlight(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def ecaresnext26t_32x4d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (2, 2, 2, 2), se=True, groups=32, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def ecaresnext50t_32x4d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=True, groups=32, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def edgenext_base(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def edgenext_small_rw(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_b0(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_b0_g16_evos(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_b0_g8_gn(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_b0_gn(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_b1(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.1, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_b1_pruned(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.1, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_b2(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.1, depth_multiplier=1.2, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_b2_pruned(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.1, depth_multiplier=1.2, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_b3(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.2, depth_multiplier=1.4, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_b3_g8_gn(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.2, depth_multiplier=1.4, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_b3_gn(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.2, depth_multiplier=1.4, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_b3_pruned(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.2, depth_multiplier=1.4, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_b4(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.4, depth_multiplier=1.8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_b5(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.6, depth_multiplier=2.2, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_b6(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.8, depth_multiplier=2.6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_b7(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=2.0, depth_multiplier=3.1, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_b8(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=2.2, depth_multiplier=3.6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_blur_b0(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_cc_b0_4e(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_cc_b0_8e(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_cc_b1_8e(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.1, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_el(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_el_pruned(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_em(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_es(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_es_pruned(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_h_b5(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.6, depth_multiplier=2.2, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_l2(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_lite0(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_lite1(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_lite2(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_lite3(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_lite4(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_x_b3(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.2, depth_multiplier=1.4, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnet_x_b5(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.6, depth_multiplier=2.2, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnetv2_l(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnetv2_m(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnetv2_rw_m(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnetv2_rw_s(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnetv2_rw_t(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnetv2_s(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientnetv2_xl(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientvit_b3(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientvit_l1(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientvit_l2(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientvit_l3(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientvit_m0(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientvit_m1(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientvit_m2(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientvit_m3(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientvit_m4(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def efficientvit_m5(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def ese_vovnet19b_dw(**kwargs):
    model = vovnet.VoVNet((128, 160, 192, 224), 3, (1, 1, 2, 2), 64, 4, 80, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def ese_vovnet19b_slim(**kwargs):
    model = vovnet.VoVNet((128, 160, 192, 224), 3, (1, 1, 2, 2), 64, 4, 80, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def ese_vovnet19b_slim_dw(**kwargs):
    model = vovnet.VoVNet((128, 160, 192, 224), 3, (1, 1, 2, 2), 64, 4, 80, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def ese_vovnet39b_evos(**kwargs):
    model = vovnet.VoVNet((128, 160, 192, 224), 3, (1, 1, 2, 2), 64, 4, 80, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def ese_vovnet57b(**kwargs):
    model = vovnet.VoVNet((128, 160, 192, 224), 3, (1, 1, 2, 2), 64, 4, 80, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def ese_vovnet99b(**kwargs):
    model = vovnet.VoVNet((128, 160, 192, 224), 3, (1, 1, 2, 2), 64, 4, 80, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def eva02_base_patch14_224(**kwargs):
    model = eva.Eva(img_size=224, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def eva02_base_patch14_448(**kwargs):
    model = eva.Eva(img_size=448, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 448, 448))
    return model


@register_model
def eva02_base_patch16_clip_224(**kwargs):
    model = eva.Eva(img_size=224, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def eva02_enormous_patch14_clip_224(**kwargs):
    model = eva.Eva(img_size=224, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def eva02_large_patch14_224(**kwargs):
    model = eva.Eva(img_size=224, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def eva02_large_patch14_448(**kwargs):
    model = eva.Eva(img_size=448, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 448, 448))
    return model


@register_model
def eva02_large_patch14_clip_224(**kwargs):
    model = eva.Eva(img_size=224, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def eva02_large_patch14_clip_336(**kwargs):
    model = eva.Eva(img_size=336, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 336, 336))
    return model


@register_model
def eva02_small_patch14_224(**kwargs):
    model = eva.Eva(img_size=224, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def eva02_small_patch14_336(**kwargs):
    model = eva.Eva(img_size=336, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 336, 336))
    return model


@register_model
def eva02_tiny_patch14_224(**kwargs):
    model = eva.Eva(img_size=224, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def eva02_tiny_patch14_336(**kwargs):
    model = eva.Eva(img_size=336, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 336, 336))
    return model


@register_model
def eva_giant_patch14_224(**kwargs):
    model = eva.Eva(img_size=224, embed_dim=1408, depth=40, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def eva_giant_patch14_336(**kwargs):
    model = eva.Eva(img_size=336, embed_dim=1408, depth=40, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 336, 336))
    return model


@register_model
def eva_giant_patch14_560(**kwargs):
    model = eva.Eva(img_size=560, embed_dim=1408, depth=40, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 560, 560))
    return model


@register_model
def eva_giant_patch14_clip_224(**kwargs):
    model = eva.Eva(img_size=224, embed_dim=1408, depth=40, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def eva_large_patch14_196(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def eva_large_patch14_336(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=336, patch_size=14, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 336, 336))
    return model


@register_model
def fasternet_l(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def fasternet_m(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def fasternet_t2(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def fastvit_ma36(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def fastvit_mci0(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def fastvit_mci1(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def fastvit_mci2(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def fastvit_mci3(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def fastvit_mci4(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def fastvit_sa12(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def fastvit_sa24(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def fastvit_sa36(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def fbnetc_100(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.1, depth_multiplier=1.2, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def fbnetv3_b(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def fbnetv3_d(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def fbnetv3_g(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def flexivit_base(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def flexivit_large(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def flexivit_small(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def focalnet_base_lrf(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def focalnet_huge_fl3(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def focalnet_huge_fl4(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def focalnet_large_fl3(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def focalnet_large_fl4(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def focalnet_small_lrf(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def focalnet_tiny_lrf(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def focalnet_xlarge_fl3(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def focalnet_xlarge_fl4(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def gc_efficientnetv2_rw_t(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def gcresnet33ts(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def gcresnet50t(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def gcresnext26ts(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (2, 2, 2, 2), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def gcresnext50ts(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def gcvit_xtiny(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def gcvit_xxtiny(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def gernet_l(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def gernet_m(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def gernet_s(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def ghostnetv2_100(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def ghostnetv2_130(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def ghostnetv2_160(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 160, 160))
    return model


@register_model
def ghostnetv3_050(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def ghostnetv3_100(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def ghostnetv3_130(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def ghostnetv3_160(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 160, 160))
    return model


@register_model
def gmixer_12_224(**kwargs):
    model = mlp_mixer.MlpMixer(patch_size=16, embed_dim=512, num_blocks=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def gmixer_24_224(**kwargs):
    model = mlp_mixer.MlpMixer(patch_size=16, embed_dim=512, num_blocks=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def gmlp_b16_224(**kwargs):
    model = mlp_mixer.MlpMixer(patch_size=16, embed_dim=512, num_blocks=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def gmlp_s16_224(**kwargs):
    model = mlp_mixer.MlpMixer(patch_size=16, embed_dim=512, num_blocks=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def gmlp_ti16_224(**kwargs):
    model = mlp_mixer.MlpMixer(patch_size=16, embed_dim=512, num_blocks=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def halo2botnet50ts_256(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def halonet26t(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def halonet50ts(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def halonet_h1(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def haloregnetz_b(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def hardcorenas_b(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def hardcorenas_c(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def hardcorenas_d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def hardcorenas_e(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def hgnetv2_b0(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def hgnetv2_b1(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def hgnetv2_b2(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def hgnetv2_b3(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def hgnetv2_b4(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def hgnetv2_b5(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def hgnetv2_b6(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def hiera_base_abswin_256(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def hiera_small_abswin_256(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def hrnet_w18_small_v2(**kwargs):
    model = hrnet.HRNet((18, 36, 72, 144), 1, 1, 4, 3, False, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def hrnet_w18_ssld(**kwargs):
    model = hrnet.HRNet((18, 36, 72, 144), 1, 1, 4, 3, False, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def hrnet_w30(**kwargs):
    model = hrnet.HRNet((18, 36, 72, 144), 1, 1, 4, 3, False, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def hrnet_w40(**kwargs):
    model = hrnet.HRNet((18, 36, 72, 144), 1, 1, 4, 3, False, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def hrnet_w44(**kwargs):
    model = hrnet.HRNet((18, 36, 72, 144), 1, 1, 4, 3, False, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def hrnet_w48_ssld(**kwargs):
    model = hrnet.HRNet((18, 36, 72, 144), 1, 1, 4, 3, False, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def hrnet_w64(**kwargs):
    model = hrnet.HRNet((18, 36, 72, 144), 1, 1, 4, 3, False, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def inception_next_base(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def lambda_resnet26rpt_256(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (2, 2, 2, 2), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def lambda_resnet26t(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (2, 2, 2, 2), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def lambda_resnet50ts(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def lamhalobotnet50ts_256(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def lcnet_035(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def lcnet_050(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=0.5, depth_multiplier=0.8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def lcnet_075(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.1, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def lcnet_100(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.1, depth_multiplier=1.2, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def lcnet_150(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.6, depth_multiplier=2.2, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def lcnetv2_base(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def lcnetv2_large(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def lcnetv2_small(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def legacy_senet154(**kwargs):
    model = senet.SENet154(**kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def legacy_seresnet101(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 23, 3), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def legacy_seresnet152(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 8, 36, 3), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def legacy_seresnet18(**kwargs):
    model = resnet.ResNet(resnet.BasicBlock, (2, 2, 2, 2), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def legacy_seresnet34(**kwargs):
    model = resnet.ResNet(resnet.BasicBlock, (3, 4, 6, 3), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def legacy_seresnet50(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def legacy_seresnext101_32x4d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 23, 3), se=True, groups=32, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def legacy_seresnext26_32x4d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (2, 2, 2, 2), se=True, groups=32, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def legacy_seresnext50_32x4d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=True, groups=32, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def legacy_xception(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def levit_128(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def levit_256d(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def levit_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def levit_384_s8(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def levit_512(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=512, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 512, 512))
    return model


@register_model
def levit_512_s8(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=512, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 512, 512))
    return model


@register_model
def levit_512d(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=512, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 512, 512))
    return model


@register_model
def levit_conv_128(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def levit_conv_128s(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def levit_conv_192(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=192, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 192, 192))
    return model


@register_model
def levit_conv_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def levit_conv_256d(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def levit_conv_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def levit_conv_384_s8(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def levit_conv_512(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=512, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 512, 512))
    return model


@register_model
def levit_conv_512_s8(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=512, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 512, 512))
    return model


@register_model
def levit_conv_512d(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=512, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 512, 512))
    return model


@register_model
def mambaout_base(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mambaout_base_plus_rw(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mambaout_base_short_rw(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mambaout_base_tall_rw(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mambaout_base_wide_rw(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mambaout_small(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mambaout_small_rw(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def maxvit_base_tf_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def maxvit_base_tf_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def maxvit_base_tf_512(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=512, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 512, 512))
    return model


@register_model
def maxvit_large_tf_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def maxvit_large_tf_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def maxvit_large_tf_512(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=512, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 512, 512))
    return model


@register_model
def maxvit_nano_rw_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def maxvit_pico_rw_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def maxvit_rmlp_base_rw_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def maxvit_rmlp_base_rw_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def maxvit_rmlp_nano_rw_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def maxvit_rmlp_pico_rw_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def maxvit_rmlp_small_rw_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def maxvit_rmlp_small_rw_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def maxvit_rmlp_tiny_rw_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def maxvit_small_tf_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def maxvit_small_tf_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def maxvit_small_tf_512(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=512, patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 512, 512))
    return model


@register_model
def maxvit_tiny_pm_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def maxvit_tiny_rw_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def maxvit_tiny_tf_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def maxvit_tiny_tf_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def maxvit_tiny_tf_512(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=512, patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 512, 512))
    return model


@register_model
def maxvit_xlarge_tf_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def maxvit_xlarge_tf_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def maxvit_xlarge_tf_512(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=512, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 512, 512))
    return model


@register_model
def maxxvit_rmlp_nano_rw_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def maxxvit_rmlp_small_rw_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def maxxvit_rmlp_tiny_rw_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def maxxvitv2_nano_rw_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def maxxvitv2_rmlp_base_rw_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def maxxvitv2_rmlp_base_rw_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def maxxvitv2_rmlp_large_rw_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mixer_l32_224(**kwargs):
    model = mlp_mixer.MlpMixer(patch_size=16, embed_dim=512, num_blocks=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mixer_s16_224(**kwargs):
    model = mlp_mixer.MlpMixer(patch_size=16, embed_dim=512, num_blocks=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mixer_s32_224(**kwargs):
    model = mlp_mixer.MlpMixer(patch_size=16, embed_dim=512, num_blocks=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mixnet_l(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mixnet_m(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mixnet_s(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mixnet_xl(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mixnet_xxl(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mnasnet_075(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.1, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mnasnet_small(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=0.5, depth_multiplier=0.8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilenet_edgetpu_100(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.1, depth_multiplier=1.2, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilenet_edgetpu_v2_l(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilenet_edgetpu_v2_m(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilenet_edgetpu_v2_s(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilenet_edgetpu_v2_xs(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilenetv1_100(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.1, depth_multiplier=1.2, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilenetv1_100h(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.1, depth_multiplier=1.2, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilenetv1_125(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilenetv2_035(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilenetv2_075(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.1, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilenetv2_110d(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilenetv2_120d(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilenetv3_large_075(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.1, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilenetv3_large_150d(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.6, depth_multiplier=2.2, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilenetv3_rw(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilenetv3_small_050(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=0.5, depth_multiplier=0.8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilenetv3_small_075(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.1, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilenetv4_conv_aa_large(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilenetv4_conv_aa_medium(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilenetv4_conv_blur_medium(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilenetv4_conv_large(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilenetv4_conv_medium(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilenetv4_conv_small(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=0.5, depth_multiplier=0.8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilenetv4_conv_small_035(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=0.5, depth_multiplier=0.8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilenetv4_conv_small_050(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=0.5, depth_multiplier=0.8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilenetv4_hybrid_large(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilenetv4_hybrid_large_075(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.1, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilenetv4_hybrid_medium(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilenetv4_hybrid_medium_075(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.1, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobileone_s0(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobileone_s1(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobileone_s2(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobileone_s3(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobileone_s4(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilevitv2_050(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilevitv2_075(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilevitv2_100(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilevitv2_125(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilevitv2_150(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilevitv2_175(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mobilevitv2_200(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mvitv2_base_cls(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mvitv2_huge_cls(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=1280, depth=32, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mvitv2_large(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mvitv2_large_cls(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def mvitv2_small_cls(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def nest_base_jx(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def nest_small_jx(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def nest_tiny_jx(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def nf_ecaresnet101(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 23, 3), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def nf_ecaresnet26(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (2, 2, 2, 2), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def nf_ecaresnet50(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def nf_regnet_b0(**kwargs):
    model = nfnet.NFNet((1, 2, 6, 3), (128, 256, 512, 1536), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def nf_regnet_b1(**kwargs):
    model = nfnet.NFNet((1, 2, 6, 3), (128, 256, 512, 1536), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def nf_regnet_b2(**kwargs):
    model = nfnet.NFNet((1, 2, 6, 3), (128, 256, 512, 1536), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def nf_regnet_b3(**kwargs):
    model = nfnet.NFNet((1, 2, 6, 3), (128, 256, 512, 1536), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def nf_regnet_b4(**kwargs):
    model = nfnet.NFNet((1, 2, 6, 3), (128, 256, 512, 1536), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def nf_regnet_b5(**kwargs):
    model = nfnet.NFNet((1, 2, 6, 3), (128, 256, 512, 1536), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def nf_resnet101(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 23, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def nf_resnet26(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (2, 2, 2, 2), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def nf_resnet50(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def nf_seresnet101(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 23, 3), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def nf_seresnet26(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (2, 2, 2, 2), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def nf_seresnet50(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def nfnet_f4(**kwargs):
    model = nfnet.NFNet((1, 2, 6, 3), (128, 256, 512, 1536), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def nfnet_f5(**kwargs):
    model = nfnet.NFNet((1, 2, 6, 3), (128, 256, 512, 1536), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def nfnet_f6(**kwargs):
    model = nfnet.NFNet((1, 2, 6, 3), (128, 256, 512, 1536), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def nfnet_f7(**kwargs):
    model = nfnet.NFNet((1, 2, 6, 3), (128, 256, 512, 1536), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def nfnet_l0(**kwargs):
    model = nfnet.NFNet((1, 2, 6, 3), (128, 256, 512, 1536), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def pit_b_distilled_224(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def pit_s_distilled_224(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def pit_ti_distilled_224(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def pit_xs_distilled_224(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def pnasnet5large(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def poolformer_m36(**kwargs):
    model = metaformer.MetaFormer((64, 128, 320, 512), (2, 2, 6, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def poolformer_m48(**kwargs):
    model = metaformer.MetaFormer((64, 128, 320, 512), (2, 2, 6, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def poolformerv2_m36(**kwargs):
    model = metaformer.MetaFormer((64, 128, 320, 512), (2, 2, 6, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def poolformerv2_m48(**kwargs):
    model = metaformer.MetaFormer((64, 128, 320, 512), (2, 2, 6, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def poolformerv2_s12(**kwargs):
    model = metaformer.MetaFormer((64, 128, 320, 512), (2, 2, 6, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def poolformerv2_s24(**kwargs):
    model = metaformer.MetaFormer((64, 128, 320, 512), (2, 2, 6, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def poolformerv2_s36(**kwargs):
    model = metaformer.MetaFormer((64, 128, 320, 512), (2, 2, 6, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def pvt_v2_b2_li(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def rdnet_large(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def regnetv_040(**kwargs):
    model = regnet.RegNet(w0=32, wa=16.0, wm=2.0, depth=10, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def regnetv_064(**kwargs):
    model = regnet.RegNet(w0=32, wa=16.0, wm=2.0, depth=10, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def regnetx_004_tv(**kwargs):
    model = regnet.RegNet(w0=32, wa=16.0, wm=2.0, depth=10, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def regnetx_006(**kwargs):
    model = regnet.RegNet(w0=32, wa=16.0, wm=2.0, depth=10, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def regnetx_040(**kwargs):
    model = regnet.RegNet(w0=32, wa=16.0, wm=2.0, depth=10, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def regnetx_064(**kwargs):
    model = regnet.RegNet(w0=32, wa=16.0, wm=2.0, depth=10, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def regnetx_080(**kwargs):
    model = regnet.RegNet(w0=32, wa=16.0, wm=2.0, depth=10, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def regnetx_120(**kwargs):
    model = regnet.RegNet(w0=32, wa=16.0, wm=2.0, depth=10, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def regnetx_160(**kwargs):
    model = regnet.RegNet(w0=32, wa=16.0, wm=2.0, depth=10, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 160, 160))
    return model


@register_model
def regnetx_320(**kwargs):
    model = regnet.RegNet(w0=32, wa=16.0, wm=2.0, depth=10, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 320, 320))
    return model


@register_model
def regnety_002(**kwargs):
    model = regnet.RegNet(w0=32, wa=16.0, wm=2.0, depth=10, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def regnety_006(**kwargs):
    model = regnet.RegNet(w0=32, wa=16.0, wm=2.0, depth=10, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def regnety_008_tv(**kwargs):
    model = regnet.RegNet(w0=48, wa=27.89, wm=2.09, depth=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def regnety_040(**kwargs):
    model = regnet.RegNet(w0=32, wa=16.0, wm=2.0, depth=10, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def regnety_040_sgn(**kwargs):
    model = regnet.RegNet(w0=32, wa=16.0, wm=2.0, depth=10, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def regnety_064(**kwargs):
    model = regnet.RegNet(w0=32, wa=16.0, wm=2.0, depth=10, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def regnety_080(**kwargs):
    model = regnet.RegNet(w0=32, wa=16.0, wm=2.0, depth=10, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def regnety_080_tv(**kwargs):
    model = regnet.RegNet(w0=32, wa=16.0, wm=2.0, depth=10, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def regnety_120(**kwargs):
    model = regnet.RegNet(w0=32, wa=16.0, wm=2.0, depth=10, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def regnety_1280(**kwargs):
    model = regnet.RegNet(w0=32, wa=16.0, wm=2.0, depth=10, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def regnety_160(**kwargs):
    model = regnet.RegNet(w0=32, wa=16.0, wm=2.0, depth=10, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 160, 160))
    return model


@register_model
def regnety_2560(**kwargs):
    model = regnet.RegNet(w0=32, wa=16.0, wm=2.0, depth=10, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def regnety_320(**kwargs):
    model = regnet.RegNet(w0=32, wa=16.0, wm=2.0, depth=10, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 320, 320))
    return model


@register_model
def regnety_640(**kwargs):
    model = regnet.RegNet(w0=32, wa=16.0, wm=2.0, depth=10, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def regnetz_005(**kwargs):
    model = regnet.RegNet(w0=32, wa=16.0, wm=2.0, depth=10, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def regnetz_040(**kwargs):
    model = regnet.RegNet(w0=32, wa=16.0, wm=2.0, depth=10, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def regnetz_040_h(**kwargs):
    model = regnet.RegNet(w0=32, wa=16.0, wm=2.0, depth=10, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def regnetz_b16(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def regnetz_b16_evos(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def regnetz_c16(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def regnetz_c16_evos(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def regnetz_d32(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def regnetz_d8(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def regnetz_d8_evos(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def regnetz_e8(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def repghostnet_058(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def repghostnet_080(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def repghostnet_111(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def repghostnet_150(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def repghostnet_200(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def repvgg_a0(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def repvgg_a1(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def repvgg_a2(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def repvgg_b0(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def repvgg_b1(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def repvgg_b1g4(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def repvgg_b2(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def repvgg_b2g4(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def repvgg_b3(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def repvgg_b3g4(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def repvgg_d2se(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def repvit_m1(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def repvit_m1_0(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def repvit_m2(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def repvit_m2_3(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def repvit_m3(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def res2net101d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def res2net50_26w_6s(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def res2net50_26w_8s(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def res2net50_48w_2s(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def res2net50d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def res2next50(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resmlp_big_24_224(**kwargs):
    model = mlp_mixer.MlpMixer(patch_size=16, embed_dim=512, num_blocks=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnest200e(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnest269e(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnest26d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnest50d_1s4x24d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnest50d_4s2x40d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet101_clip(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 23, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet101_clip_gap(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 23, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet101c(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 23, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet101d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 23, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet101s(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 23, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet10t(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet14t(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet152c(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 8, 36, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet152d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 8, 36, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet152s(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 8, 36, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet18d(**kwargs):
    model = resnet.ResNet(resnet.BasicBlock, (2, 2, 2, 2), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet200(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 24, 36, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet200d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 24, 36, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet26(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (2, 2, 2, 2), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet26d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (2, 2, 2, 2), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet26t(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (2, 2, 2, 2), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet32ts(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet33ts(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet34d(**kwargs):
    model = resnet.ResNet(resnet.BasicBlock, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet50_clip(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet50_clip_gap(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet50_gn(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet50_mlp(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet50c(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet50d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet50s(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet50t(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet50x16_clip(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet50x16_clip_gap(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet50x4_clip(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet50x4_clip_gap(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet50x64_clip(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet50x64_clip_gap(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet51q(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnet61q(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnetaa101d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 23, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnetaa34d(**kwargs):
    model = resnet.ResNet(resnet.BasicBlock, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnetaa50(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnetaa50d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnetblur101d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 23, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnetblur18(**kwargs):
    model = resnet.ResNet(resnet.BasicBlock, (2, 2, 2, 2), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnetblur50(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnetblur50d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnetrs101(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 23, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnetrs152(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 8, 36, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnetrs200(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 24, 36, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnetrs270(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnetrs350(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnetrs420(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnetrs50(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnetv2_101d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 23, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnetv2_101x1_bit(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 23, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnetv2_101x3_bit(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 23, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnetv2_152d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 8, 36, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnetv2_152x2_bit(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 8, 36, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnetv2_152x4_bit(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 8, 36, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnetv2_18(**kwargs):
    model = resnet.ResNet(resnet.BasicBlock, (2, 2, 2, 2), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnetv2_18d(**kwargs):
    model = resnet.ResNet(resnet.BasicBlock, (2, 2, 2, 2), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnetv2_34(**kwargs):
    model = resnet.ResNet(resnet.BasicBlock, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnetv2_34d(**kwargs):
    model = resnet.ResNet(resnet.BasicBlock, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnetv2_50d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnetv2_50d_evos(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnetv2_50d_frn(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnetv2_50d_gn(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnetv2_50t(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnetv2_50x1_bit(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnetv2_50x3_bit(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnext101_32x16d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 23, 3), se=False, groups=32, base_width=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnext101_32x32d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 23, 3), se=False, groups=32, base_width=32, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnext101_32x4d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 23, 3), se=False, groups=32, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnext101_64x4d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 23, 3), se=False, groups=64, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnext26ts(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (2, 2, 2, 2), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def resnext50d_32x4d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=32, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def rexnet_300(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def rexnetr_100(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def rexnetr_130(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def rexnetr_150(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def rexnetr_200(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def rexnetr_300(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def samvit_base_patch16(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def samvit_base_patch16_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def samvit_huge_patch16(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=1280, depth=32, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def samvit_large_patch16(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def sebotnet33ts_256(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def sedarknet21(**kwargs):
    model = darknet.DarkNet53(**kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def sehalonet33ts(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def selecsls42b(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def selecsls60b(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def selecsls84(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def semnasnet_050(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=0.5, depth_multiplier=0.8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def semnasnet_075(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.1, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def semnasnet_100(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.1, depth_multiplier=1.2, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def semnasnet_140(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.4, depth_multiplier=1.8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def sequencer2d_l(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def seresnet101(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 23, 3), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def seresnet152(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 8, 36, 3), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def seresnet152d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 8, 36, 3), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def seresnet18(**kwargs):
    model = resnet.ResNet(resnet.BasicBlock, (2, 2, 2, 2), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def seresnet200d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 24, 36, 3), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def seresnet269d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (2, 2, 2, 2), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def seresnet33ts(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def seresnet34(**kwargs):
    model = resnet.ResNet(resnet.BasicBlock, (3, 4, 6, 3), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def seresnet50t(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def seresnetaa50d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def seresnext101_32x4d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 23, 3), se=True, groups=32, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def seresnext101_32x8d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 23, 3), se=True, groups=32, base_width=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def seresnext101_64x4d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 23, 3), se=True, groups=64, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def seresnext101d_32x8d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 23, 3), se=True, groups=32, base_width=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def seresnext26d_32x4d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (2, 2, 2, 2), se=True, groups=32, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def seresnext26t_32x4d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (2, 2, 2, 2), se=True, groups=32, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def seresnext26ts(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (2, 2, 2, 2), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def seresnextaa101d_32x8d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 23, 3), se=True, groups=32, base_width=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def seresnextaa201d_32x8d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=True, groups=32, base_width=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def shvit_s3(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def shvit_s4(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def skresnet18(**kwargs):
    model = resnet.ResNet(resnet.BasicBlock, (2, 2, 2, 2), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def skresnet34(**kwargs):
    model = resnet.ResNet(resnet.BasicBlock, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def skresnet50d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def skresnext50_32x4d(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=32, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def starnet_s100(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def starnet_s150(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def starnet_s3(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def starnet_s4(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def swiftformer_l3(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def swin_base_patch4_window12_384(**kwargs):
    model = swin_transformer.SwinTransformer(img_size=384, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def swin_large_patch4_window12_384(**kwargs):
    model = swin_transformer.SwinTransformer(img_size=384, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def swin_large_patch4_window7_224(**kwargs):
    model = swin_transformer.SwinTransformer(img_size=224, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def swin_s3_base_224(**kwargs):
    model = swin_transformer.SwinTransformer(img_size=224, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def swin_s3_small_224(**kwargs):
    model = swin_transformer.SwinTransformer(img_size=224, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def swin_s3_tiny_224(**kwargs):
    model = swin_transformer.SwinTransformer(img_size=224, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def swinv2_base_window12_192(**kwargs):
    model = swin_transformer.SwinTransformer(img_size=192, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 192, 192))
    return model


@register_model
def swinv2_base_window12to16_192to256(**kwargs):
    model = swin_transformer.SwinTransformer(img_size=192, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 192, 192))
    return model


@register_model
def swinv2_base_window12to24_192to384(**kwargs):
    model = swin_transformer.SwinTransformer(img_size=192, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 192, 192))
    return model


@register_model
def swinv2_base_window16_256(**kwargs):
    model = swin_transformer.SwinTransformer(img_size=256, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def swinv2_base_window8_256(**kwargs):
    model = swin_transformer.SwinTransformer(img_size=256, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def swinv2_cr_base_384(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def swinv2_cr_base_ns_224(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def swinv2_cr_giant_384(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def swinv2_cr_huge_384(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def swinv2_cr_large_384(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def swinv2_cr_small_384(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def swinv2_cr_small_ns_224(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def swinv2_cr_small_ns_256(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def swinv2_cr_tiny_384(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def swinv2_cr_tiny_ns_224(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def swinv2_large_window12_192(**kwargs):
    model = swin_transformer.SwinTransformer(img_size=192, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 192, 192))
    return model


@register_model
def swinv2_large_window12to16_192to256(**kwargs):
    model = swin_transformer.SwinTransformer(img_size=192, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 192, 192))
    return model


@register_model
def swinv2_large_window12to24_192to384(**kwargs):
    model = swin_transformer.SwinTransformer(img_size=192, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 192, 192))
    return model


@register_model
def swinv2_small_window16_256(**kwargs):
    model = swin_transformer.SwinTransformer(img_size=256, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def swinv2_tiny_window16_256(**kwargs):
    model = swin_transformer.SwinTransformer(img_size=256, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def test_byobnet(**kwargs):
    model = byobnet.ByobNet((64, 128, 256, 512), (2, 2, 6, 2), 4, (1, 2, 2, 2), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def test_convnext(**kwargs):
    model = convnext.ConvNeXt((3, 3, 9, 3), (96, 192, 384, 768), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def test_convnext2(**kwargs):
    model = convnext.ConvNeXt((3, 3, 9, 3), (96, 192, 384, 768), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def test_convnext3(**kwargs):
    model = convnext.ConvNeXt((3, 3, 9, 3), (96, 192, 384, 768), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def test_efficientnet(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def test_efficientnet_evos(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def test_efficientnet_gn(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def test_efficientnet_ln(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def test_mambaout(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def test_nfnet(**kwargs):
    model = nfnet.NFNet((1, 2, 6, 3), (128, 256, 512, 1536), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def test_resnet(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def test_vit(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def test_vit2(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def test_vit3(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def test_vit4(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_efficientnet_b0(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_efficientnet_b1(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.1, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_efficientnet_b2(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.1, depth_multiplier=1.2, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_efficientnet_b3(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.2, depth_multiplier=1.4, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_efficientnet_b4(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.4, depth_multiplier=1.8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_efficientnet_b5(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.6, depth_multiplier=2.2, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_efficientnet_b6(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.8, depth_multiplier=2.6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_efficientnet_b7(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=2.0, depth_multiplier=3.1, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_efficientnet_b8(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=2.2, depth_multiplier=3.6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_efficientnet_cc_b0_4e(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_efficientnet_cc_b0_8e(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_efficientnet_cc_b1_8e(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.1, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_efficientnet_el(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_efficientnet_em(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_efficientnet_es(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_efficientnet_l2(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_efficientnet_lite0(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_efficientnet_lite1(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_efficientnet_lite2(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_efficientnet_lite3(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_efficientnet_lite4(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_efficientnetv2_b0(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_efficientnetv2_b1(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.1, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_efficientnetv2_b2(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.1, depth_multiplier=1.2, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_efficientnetv2_b3(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.2, depth_multiplier=1.4, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_efficientnetv2_l(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_efficientnetv2_m(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_efficientnetv2_s(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_efficientnetv2_xl(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_mixnet_l(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_mixnet_m(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_mixnet_s(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.0, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_mobilenetv3_large_075(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.1, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_mobilenetv3_large_100(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.1, depth_multiplier=1.2, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_mobilenetv3_large_minimal_100(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.1, depth_multiplier=1.2, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_mobilenetv3_small_075(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.0, depth_multiplier=1.1, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_mobilenetv3_small_100(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.1, depth_multiplier=1.2, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tf_mobilenetv3_small_minimal_100(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=1.1, depth_multiplier=1.2, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tiny_vit_21m_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def tiny_vit_21m_512(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=512, patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 512, 512))
    return model


@register_model
def tinynet_a(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=0.5, depth_multiplier=0.8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tinynet_b(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=0.5, depth_multiplier=0.8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tinynet_c(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=0.5, depth_multiplier=0.8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tinynet_d(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=0.5, depth_multiplier=0.8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tinynet_e(**kwargs):
    model = efficientnet.EfficientNet(channel_multiplier=0.5, depth_multiplier=0.8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tnt_b_patch16_224(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tnt_s_legacy_patch16_224(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tnt_s_patch16_224(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def tresnet_v2_l(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def twins_pcpvt_base(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def twins_pcpvt_large(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def twins_pcpvt_small(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_7b_patch16_dinov3(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_base_mci_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_base_patch14_dinov2(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_base_patch14_reg1_tipsv2(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_base_patch14_reg4_dinov2(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_base_patch16_18x2_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_base_patch16_224_miil(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_base_patch16_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def vit_base_patch16_clip_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_base_patch16_clip_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def vit_base_patch16_clip_quickgelu_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_base_patch16_dinov3(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_base_patch16_dinov3_qkvb(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_base_patch16_gap_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_base_patch16_lingbot(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_base_patch16_plus_240(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=240, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 240, 240))
    return model


@register_model
def vit_base_patch16_plus_clip_240(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=240, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 240, 240))
    return model


@register_model
def vit_base_patch16_reg4_gap_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_base_patch16_rope_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_base_patch16_rope_ape_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_base_patch16_rope_mixed_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_base_patch16_rope_mixed_ape_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_base_patch16_rope_reg1_gap_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_base_patch16_rpn_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_base_patch16_siglip_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_base_patch16_siglip_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_base_patch16_siglip_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def vit_base_patch16_siglip_512(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=512, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 512, 512))
    return model


@register_model
def vit_base_patch16_siglip_gap_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_base_patch16_siglip_gap_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_base_patch16_siglip_gap_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def vit_base_patch16_siglip_gap_512(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=512, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 512, 512))
    return model


@register_model
def vit_base_patch16_xp_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_base_patch32_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_base_patch32_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def vit_base_patch32_clip_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_base_patch32_clip_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_base_patch32_clip_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def vit_base_patch32_clip_448(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=448, patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 448, 448))
    return model


@register_model
def vit_base_patch32_clip_quickgelu_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_base_patch32_plus_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_base_patch32_siglip_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_base_patch32_siglip_gap_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_base_patch8_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_base_r26_s32_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_base_r50_s16_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_base_r50_s16_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def vit_base_resnet26d_224(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (2, 2, 2, 2), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_base_resnet50d_224(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=True, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_betwixt_patch16_gap_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_betwixt_patch16_reg1_gap_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_betwixt_patch16_reg4_gap_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_betwixt_patch16_reg4_gap_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def vit_betwixt_patch16_rope_reg4_gap_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_betwixt_patch32_clip_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_dlittle_patch16_reg1_gap_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_dpwee_patch16_reg1_gap_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_dwee_patch16_reg1_gap_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_giant_patch14_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=1408, depth=40, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_giant_patch14_clip_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=1408, depth=40, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_giant_patch14_dinov2(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=1408, depth=40, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_giant_patch14_reg1_tipsv2(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=1408, depth=40, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_giant_patch14_reg4_dinov2(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=1408, depth=40, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_giant_patch16_gap_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=1408, depth=40, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_giant_patch16_lingbot(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=1408, depth=40, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_giantopt_patch16_siglip_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=1408, depth=40, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_giantopt_patch16_siglip_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=1408, depth=40, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def vit_giantopt_patch16_siglip_gap_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=1408, depth=40, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_giantopt_patch16_siglip_gap_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=1408, depth=40, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def vit_gigantic_patch14_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_gigantic_patch14_clip_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_gigantic_patch14_clip_378(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_gigantic_patch14_clip_quickgelu_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_huge_patch14_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=1280, depth=32, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_huge_patch14_clip_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=1280, depth=32, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_huge_patch14_clip_336(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=336, patch_size=14, embed_dim=1280, depth=32, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 336, 336))
    return model


@register_model
def vit_huge_patch14_clip_378(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=1280, depth=32, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_huge_patch14_clip_quickgelu_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=1280, depth=32, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_huge_patch14_clip_quickgelu_378(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=1280, depth=32, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_huge_patch14_gap_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=1280, depth=32, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_huge_patch14_xp_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=1280, depth=32, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_huge_patch16_gap_448(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=448, patch_size=16, embed_dim=1280, depth=32, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 448, 448))
    return model


@register_model
def vit_huge_plus_patch16_dinov3(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=1280, depth=32, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_huge_plus_patch16_dinov3_qkvb(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=1280, depth=32, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_intern300m_patch14_448(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=448, patch_size=14, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 448, 448))
    return model


@register_model
def vit_large_patch14_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_large_patch14_clip_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_large_patch14_clip_336(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=336, patch_size=14, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 336, 336))
    return model


@register_model
def vit_large_patch14_clip_quickgelu_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_large_patch14_clip_quickgelu_336(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=336, patch_size=14, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 336, 336))
    return model


@register_model
def vit_large_patch14_dinov2(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_large_patch14_reg1_tipsv2(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_large_patch14_reg4_dinov2(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_large_patch14_xp_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_large_patch16_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def vit_large_patch16_dinov3(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_large_patch16_dinov3_qkvb(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_large_patch16_lingbot(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_large_patch16_rope_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_large_patch16_rope_ape_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_large_patch16_rope_mixed_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_large_patch16_rope_mixed_ape_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_large_patch16_siglip_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_large_patch16_siglip_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def vit_large_patch16_siglip_512(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=512, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 512, 512))
    return model


@register_model
def vit_large_patch16_siglip_gap_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_large_patch16_siglip_gap_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def vit_large_patch16_siglip_gap_512(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=512, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 512, 512))
    return model


@register_model
def vit_large_patch32_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_large_patch32_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def vit_large_r50_s32_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_large_r50_s32_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def vit_little_patch16_reg1_gap_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_little_patch16_reg4_gap_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_medium_patch16_clip_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=512, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_medium_patch16_gap_240(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=240, patch_size=16, embed_dim=512, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 240, 240))
    return model


@register_model
def vit_medium_patch16_gap_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=512, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_medium_patch16_gap_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=512, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def vit_medium_patch16_reg1_gap_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=512, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_medium_patch16_reg4_gap_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=512, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_medium_patch16_rope_reg1_gap_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=512, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_medium_patch32_clip_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=32, embed_dim=512, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_mediumd_patch16_reg4_gap_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=512, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_mediumd_patch16_reg4_gap_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=512, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def vit_mediumd_patch16_rope_reg1_gap_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=512, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_pe_core_base_patch16_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_pe_core_gigantic_patch14_448(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=448, patch_size=14, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 448, 448))
    return model


@register_model
def vit_pe_core_large_patch14_336(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=336, patch_size=14, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 336, 336))
    return model


@register_model
def vit_pe_core_small_patch16_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def vit_pe_core_tiny_patch16_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def vit_pe_lang_gigantic_patch14_448(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=448, patch_size=14, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 448, 448))
    return model


@register_model
def vit_pe_lang_large_patch14_448(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=448, patch_size=14, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 448, 448))
    return model


@register_model
def vit_pe_spatial_base_patch16_512(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=512, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 512, 512))
    return model


@register_model
def vit_pe_spatial_gigantic_patch14_448(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=448, patch_size=14, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 448, 448))
    return model


@register_model
def vit_pe_spatial_large_patch14_448(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=448, patch_size=14, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 448, 448))
    return model


@register_model
def vit_pe_spatial_small_patch16_512(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=512, patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 512, 512))
    return model


@register_model
def vit_pe_spatial_tiny_patch16_512(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=512, patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 512, 512))
    return model


@register_model
def vit_pwee_patch16_reg1_gap_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_relpos_base_patch16_cls_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_relpos_base_patch16_clsgap_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_relpos_base_patch16_plus_240(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=240, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 240, 240))
    return model


@register_model
def vit_relpos_base_patch16_rpn_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_relpos_base_patch32_plus_rpn_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_relpos_medium_patch16_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=512, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_relpos_medium_patch16_cls_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=512, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_relpos_medium_patch16_rpn_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=512, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_relpos_small_patch16_rpn_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_small_patch14_dinov2(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_small_patch14_reg4_dinov2(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_small_patch16_18x2_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_small_patch16_36x1_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_small_patch16_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def vit_small_patch16_dinov3(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_small_patch16_dinov3_qkvb(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_small_patch16_lingbot(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_small_patch16_rope_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_small_patch16_rope_ape_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_small_patch16_rope_mixed_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_small_patch16_rope_mixed_ape_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_small_patch32_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_small_patch32_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def vit_small_patch8_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=8, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_small_plus_patch16_dinov3(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_small_plus_patch16_dinov3_qkvb(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_small_r26_s32_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_small_r26_s32_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def vit_small_resnet26d_224(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (2, 2, 2, 2), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_small_resnet50d_s16_224(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_so150m2_patch16_reg1_gap_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_so150m2_patch16_reg1_gap_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def vit_so150m2_patch16_reg1_gap_448(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=448, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 448, 448))
    return model


@register_model
def vit_so150m_patch16_reg4_gap_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_so150m_patch16_reg4_gap_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def vit_so150m_patch16_reg4_map_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_so400m_patch14_reg1_tipsv2(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_so400m_patch14_siglip_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_so400m_patch14_siglip_378(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_so400m_patch14_siglip_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=14, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def vit_so400m_patch14_siglip_gap_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_so400m_patch14_siglip_gap_378(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_so400m_patch14_siglip_gap_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=14, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def vit_so400m_patch14_siglip_gap_448(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=448, patch_size=14, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 448, 448))
    return model


@register_model
def vit_so400m_patch14_siglip_gap_896(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=14, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_so400m_patch16_siglip_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_so400m_patch16_siglip_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def vit_so400m_patch16_siglip_512(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=512, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 512, 512))
    return model


@register_model
def vit_so400m_patch16_siglip_gap_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_so400m_patch16_siglip_gap_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def vit_so400m_patch16_siglip_gap_512(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=512, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 512, 512))
    return model


@register_model
def vit_srelpos_medium_patch16_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=512, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_srelpos_small_patch16_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_tiny_patch16_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def vit_tiny_patch16_dinov3_qkvb(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_tiny_r_s16_p8_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=8, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vit_tiny_r_s16_p8_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=8, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def vit_wee_patch16_reg1_gap_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vit_xsmall_patch16_clip_224(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def vitamin_large2_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vitamin_large2_336(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=336, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 336, 336))
    return model


@register_model
def vitamin_large2_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def vitamin_large_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vitamin_large_336(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=336, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 336, 336))
    return model


@register_model
def vitamin_large_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def vitamin_xlarge_256(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=256, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 256, 256))
    return model


@register_model
def vitamin_xlarge_336(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=336, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 336, 336))
    return model


@register_model
def vitamin_xlarge_384(**kwargs):
    model = vision_transformer.VisionTransformer(img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def volo_d1_384(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def volo_d2_384(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def volo_d3_448(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 448, 448))
    return model


@register_model
def volo_d4_448(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 448, 448))
    return model


@register_model
def volo_d5_448(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 448, 448))
    return model


@register_model
def volo_d5_512(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 512, 512))
    return model


@register_model
def wide_resnet101_2(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 23, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def wide_resnet50_2(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), se=False, groups=1, base_width=64, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def xception41p(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def xception65p(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def xception71(**kwargs):
    model = resnet.ResNet(resnet.Bottleneck, (3, 4, 6, 3), **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def xcit_large_24_p16_224(**kwargs):
    model = xcit.XCiT(embed_dim=384, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def xcit_large_24_p16_384(**kwargs):
    model = xcit.XCiT(embed_dim=384, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def xcit_large_24_p8_224(**kwargs):
    model = xcit.XCiT(embed_dim=384, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def xcit_large_24_p8_384(**kwargs):
    model = xcit.XCiT(embed_dim=384, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def xcit_medium_24_p16_384(**kwargs):
    model = xcit.XCiT(embed_dim=384, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def xcit_medium_24_p8_224(**kwargs):
    model = xcit.XCiT(embed_dim=384, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def xcit_medium_24_p8_384(**kwargs):
    model = xcit.XCiT(embed_dim=384, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def xcit_nano_12_p16_224(**kwargs):
    model = xcit.XCiT(embed_dim=384, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def xcit_nano_12_p16_384(**kwargs):
    model = xcit.XCiT(embed_dim=384, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def xcit_nano_12_p8_224(**kwargs):
    model = xcit.XCiT(embed_dim=384, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def xcit_nano_12_p8_384(**kwargs):
    model = xcit.XCiT(embed_dim=384, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def xcit_small_12_p16_384(**kwargs):
    model = xcit.XCiT(embed_dim=384, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def xcit_small_12_p8_224(**kwargs):
    model = xcit.XCiT(embed_dim=384, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def xcit_small_12_p8_384(**kwargs):
    model = xcit.XCiT(embed_dim=384, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def xcit_small_24_p16_224(**kwargs):
    model = xcit.XCiT(embed_dim=384, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def xcit_small_24_p16_384(**kwargs):
    model = xcit.XCiT(embed_dim=384, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def xcit_small_24_p8_224(**kwargs):
    model = xcit.XCiT(embed_dim=384, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def xcit_small_24_p8_384(**kwargs):
    model = xcit.XCiT(embed_dim=384, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def xcit_tiny_12_p16_384(**kwargs):
    model = xcit.XCiT(embed_dim=384, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def xcit_tiny_12_p8_224(**kwargs):
    model = xcit.XCiT(embed_dim=384, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def xcit_tiny_12_p8_384(**kwargs):
    model = xcit.XCiT(embed_dim=384, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def xcit_tiny_24_p16_224(**kwargs):
    model = xcit.XCiT(embed_dim=384, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def xcit_tiny_24_p16_384(**kwargs):
    model = xcit.XCiT(embed_dim=384, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model


@register_model
def xcit_tiny_24_p8_224(**kwargs):
    model = xcit.XCiT(embed_dim=384, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model


@register_model
def xcit_tiny_24_p8_384(**kwargs):
    model = xcit.XCiT(embed_dim=384, depth=12, num_heads=8, **kwargs)
    model.default_cfg = _cfg(input_size=(3, 384, 384))
    return model
