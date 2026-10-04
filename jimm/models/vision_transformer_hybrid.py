"""Hybrid ViTs (CNN feature maps as patch tokens) in flax nnx, NHWC. Mirrors timm.models.vision_transformer_hybrid.

The patch embedding runs a CNN backbone and projects its feature map to the
transformer width with a (patch-sized) strided conv: BiT-style ResNetV2 stems
and stages (weight-standardized convs with TF "same" padding, GroupNorm +
ReLU, non-pre-activation bottlenecks) for the R26/R50 models, timm ResNet-D
trunks for the resnet26d/50d models, or a three-conv stem for MobileCLIP's
vit_base_mci_224. A standard class-token ViT follows.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from ..layers import BatchNorm, ClassifierMixin, gelu, make_divisible
from ..registry import _cfg, register_model
from .resnet import Bottleneck as ResNetBottleneck
from .resnet import ResNet
from .vision_transformer import Block

_trunc = nnx.initializers.truncated_normal(0.02)


class StdConv2dSame(nnx.Module):
    """Weight-standardized conv (eps 1e-8) with TensorFlow "SAME" padding."""

    def __init__(self, in_chs, out_chs, kernel, stride=1, *, rngs):
        init = nnx.initializers.lecun_normal()
        self.kernel = nnx.Param(init(rngs.params(), (kernel, kernel, in_chs, out_chs)))
        self.stride = stride

    def __call__(self, x):
        w = self.kernel[...]
        mean = w.mean(axis=(0, 1, 2), keepdims=True)
        var = w.var(axis=(0, 1, 2), keepdims=True)
        w = ((w - mean) * jax.lax.rsqrt(var + 1e-8)).astype(x.dtype)
        return jax.lax.conv_general_dilated(
            x, w, (self.stride, self.stride), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
        )


class GroupNormAct(nnx.Module):
    def __init__(self, chs, act=True, *, rngs):
        self.norm = nnx.GroupNorm(chs, num_groups=32, epsilon=1e-5, rngs=rngs)
        self.act = act

    def __call__(self, x):
        x = self.norm(x)
        return nnx.relu(x) if self.act else x


class DownsampleConv(nnx.Module):
    def __init__(self, in_chs, out_chs, stride, *, rngs):
        self.conv = StdConv2dSame(in_chs, out_chs, 1, stride, rngs=rngs)
        self.norm = GroupNormAct(out_chs, act=False, rngs=rngs)

    def __call__(self, x):
        return self.norm(self.conv(x))


class BitBottleneck(nnx.Module):
    """ResNetV2 non-pre-activation bottleneck (timm resnetv2.Bottleneck)."""

    def __init__(self, in_chs, out_chs, stride, *, rngs):
        mid = make_divisible(out_chs * 0.25)
        self.downsample = (
            DownsampleConv(in_chs, out_chs, stride, rngs=rngs)
            if stride != 1 or in_chs != out_chs
            else None
        )
        self.conv1 = StdConv2dSame(in_chs, mid, 1, rngs=rngs)
        self.norm1 = GroupNormAct(mid, rngs=rngs)
        self.conv2 = StdConv2dSame(mid, mid, 3, stride, rngs=rngs)
        self.norm2 = GroupNormAct(mid, rngs=rngs)
        self.conv3 = StdConv2dSame(mid, out_chs, 1, rngs=rngs)
        self.norm3 = GroupNormAct(out_chs, act=False, rngs=rngs)

    def __call__(self, x):
        sc = x if self.downsample is None else self.downsample(x)
        y = self.norm1(self.conv1(x))
        y = self.norm2(self.conv2(y))
        return nnx.relu(self.norm3(self.conv3(y)) + sc)


class BitStage(nnx.Module):
    def __init__(self, in_chs, out_chs, stride, depth, *, rngs):
        self.blocks = nnx.List(
            [
                BitBottleneck(
                    in_chs if i == 0 else out_chs, out_chs, stride if i == 0 else 1, rngs=rngs
                )
                for i in range(depth)
            ]
        )

    def __call__(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class BitStem(nnx.Module):
    def __init__(self, in_chans, out_chs=64, *, rngs):
        self.conv = StdConv2dSame(in_chans, out_chs, 7, 2, rngs=rngs)
        self.norm = GroupNormAct(out_chs, rngs=rngs)

    def __call__(self, x):
        x = self.norm(self.conv(x))
        # create_pool2d('max', 3, 2, padding='same'): TF-style padding with -inf.
        return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 3, 3, 1), (1, 2, 2, 1), "SAME")


class ResNetV2Backbone(nnx.Module):
    """timm ResNetV2(preact=False, stem_type='same', StdConv2dSame) without its head."""

    def __init__(self, layers, in_chans=3, *, rngs):
        self.stem = BitStem(in_chans, rngs=rngs)
        chs, stages = 64, []
        for i, depth in enumerate(layers):
            out = 256 * 2**i
            stages.append(BitStage(chs, out, 1 if i == 0 else 2, depth, rngs=rngs))
            chs = out
        self.stages = nnx.List(stages)
        self.num_features = chs

    def __call__(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x


class ResNetTrunk(nnx.Module):
    """timm ResNet-D features_only trunk up to ``num_stages`` stages."""

    def __init__(self, layers, num_stages, in_chans=3, *, rngs):
        self.net = ResNet(
            ResNetBottleneck,
            layers[:num_stages],
            num_classes=0,
            in_chans=in_chans,
            stem_width=32,
            stem_type="deep",
            avg_down=True,
            channels=(64, 128, 256, 512)[:num_stages],
            rngs=rngs,
        )
        self.num_features = self.net.num_features

    def __call__(self, x):
        return self.net.forward_features(x)


class ConvNormAct(nnx.Module):
    def __init__(self, in_chs, out_chs, kernel, stride, last, *, rngs):
        self.conv = nnx.Conv(
            in_chs, out_chs, (kernel, kernel), strides=stride, padding="VALID", use_bias=last,
            rngs=rngs,
        )  # fmt: skip
        self.bn = None if last else BatchNorm(out_chs, rngs=rngs)

    def __call__(self, x):
        x = self.conv(x)
        return x if self.bn is None else gelu(self.bn(x))


class ConvStem(nnx.Module):
    """MobileCLIP's three-conv stem (kernels/strides 4, 2, 2; the last conv biased, no norm)."""

    def __init__(self, in_chans, channels, *, rngs):
        c0, c1, c2 = channels
        self.layers = nnx.List(
            [
                ConvNormAct(in_chans, c0, 4, 4, False, rngs=rngs),
                ConvNormAct(c0, c1, 2, 2, False, rngs=rngs),
                ConvNormAct(c1, c2, 2, 2, True, rngs=rngs),
            ]
        )
        self.num_features = c2

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class HybridEmbed(nnx.Module):
    def __init__(self, backbone, embed_dim, patch_size=1, proj=True, *, rngs):
        self.backbone = backbone
        p = (patch_size, patch_size)
        self.proj = (
            nnx.Conv(backbone.num_features, embed_dim, p, strides=p, padding="VALID", rngs=rngs)
            if proj
            else None
        )

    def __call__(self, x):
        x = self.backbone(x)
        if self.proj is not None:
            x = self.proj(x)
        B, H, W, C = x.shape
        return x.reshape(B, H * W, C)


class HybridVisionTransformer(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"
    _default_global_pool = "token"

    def __init__(
        self,
        backbone,
        img_size=224,
        reduction=16,
        patch_size=1,
        proj=True,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        no_embed_class=False,
        num_classes=1000,
        global_pool="token",
        drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dim
        self.no_embed_class = no_embed_class
        self.patch_embed = HybridEmbed(backbone, embed_dim, patch_size, proj, rngs=rngs)
        n = (img_size // reduction // patch_size) ** 2
        self.cls_token = nnx.Param(nnx.initializers.normal(1e-6)(rngs.params(), (1, 1, embed_dim)))
        self.pos_embed = nnx.Param(
            _trunc(rngs.params(), (1, n if no_embed_class else n + 1, embed_dim))
        )
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nnx.List(
            [
                Block(embed_dim, num_heads, mlp_ratio, True, 0.0, dpr[i], rngs=rngs)
                for i in range(depth)
            ]
        )
        self.norm = nnx.LayerNorm(embed_dim, epsilon=1e-6, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = self._make_head(num_classes, rngs)

    def _make_head(self, num_classes, rngs):
        return nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.global_pool = global_pool if global_pool is not None else self.global_pool
        self.head = self._make_head(num_classes, nnx.Rngs(0))

    def forward_features(self, x):
        x = self.patch_embed(x)
        B, _, C = x.shape
        cls = jnp.broadcast_to(self.cls_token[...], (B, 1, C))
        if self.no_embed_class:
            x = jnp.concatenate([cls, x + self.pos_embed[...]], axis=1)
        else:
            x = jnp.concatenate([cls, x], axis=1) + self.pos_embed[...]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward_head(self, x):
        if self.global_pool == "avg":
            x = x[:, 1:].mean(axis=1)
        elif self.global_pool == "token":
            x = x[:, 0]
        x = self.head_drop(x)
        return self.head(x) if self.head is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_INCEPTION = dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
_SMALL = dict(embed_dim=384, depth=12, num_heads=6)
_BASE = dict(embed_dim=768, depth=12, num_heads=12)
_LARGE = dict(embed_dim=1024, depth=24, num_heads=16)
_CFGS = {  # backbone spec, reduction, vit kwargs, input size, cfg extras
    "vit_tiny_r_s16_p8_224": (("bit", ()), 4, dict(patch_size=8, embed_dim=192, depth=12, num_heads=3), 224, _INCEPTION),
    "vit_tiny_r_s16_p8_384": (("bit", ()), 4, dict(patch_size=8, embed_dim=192, depth=12, num_heads=3), 384, _INCEPTION),
    "vit_small_r26_s32_224": (("bit", (2, 2, 2, 2)), 32, _SMALL, 224, _INCEPTION),
    "vit_small_r26_s32_384": (("bit", (2, 2, 2, 2)), 32, _SMALL, 384, _INCEPTION),
    "vit_base_r26_s32_224": (("bit", (2, 2, 2, 2)), 32, _BASE, 224, _INCEPTION),
    "vit_base_r50_s16_224": (("bit", (3, 4, 9)), 16, _BASE, 224, _INCEPTION),
    "vit_base_r50_s16_384": (("bit", (3, 4, 9)), 16, _BASE, 384, _INCEPTION),
    "vit_large_r50_s32_224": (("bit", (3, 4, 6, 3)), 32, _LARGE, 224, _INCEPTION),
    "vit_large_r50_s32_384": (("bit", (3, 4, 6, 3)), 32, _LARGE, 384, _INCEPTION),
    "vit_small_resnet26d_224": (("resnet", (2, 2, 2, 2), 4), 32, dict(embed_dim=768, depth=8, num_heads=8, mlp_ratio=3), 224, {}),
    "vit_small_resnet50d_s16_224": (("resnet", (3, 4, 6, 3), 3), 16, dict(embed_dim=768, depth=8, num_heads=8, mlp_ratio=3), 224, {}),
    "vit_base_resnet26d_224": (("resnet", (2, 2, 2, 2), 4), 32, _BASE, 224, {}),
    "vit_base_resnet50d_224": (("resnet", (3, 4, 6, 3), 4), 32, _BASE, 224, {}),
    "vit_base_mci_224": (("mci",), 16, dict(_BASE, proj=False, no_embed_class=True), 224, dict(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))),
}  # fmt: skip


def _backbone(spec, in_chans, rngs):
    kind = spec[0]
    if kind == "bit":
        layers = spec[1]
        if not layers:
            stem = BitStem(in_chans, rngs=rngs)
            stem.num_features = 64
            return stem
        return ResNetV2Backbone(layers, in_chans, rngs=rngs)
    if kind == "resnet":
        return ResNetTrunk(spec[1], spec[2], in_chans, rngs=rngs)
    return ConvStem(in_chans, (192, 192, 768), rngs=rngs)


def _make(name):
    spec, reduction, vit, size, extra = _CFGS[name]

    def entry(in_chans=3, rngs=None, **kwargs):
        rngs = rngs if rngs is not None else nnx.Rngs(0)
        backbone = _backbone(spec, in_chans, rngs)
        model = HybridVisionTransformer(
            backbone, img_size=size, reduction=reduction, **{**vit, **kwargs}, rngs=rngs
        )
        model.default_cfg = _cfg(
            input_size=(3, size, size),
            crop_pct=0.9 if size == 224 else 1.0,
            interpolation="bicubic",
            fixed_input_size=True,
            **extra,
        )
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
