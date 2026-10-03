"""ViTamin in flax nnx, NHWC. Mirrors timm.models.vitamin.

A convolutional embedding (3x3 stride-2 stem with LayerNorm2d + GELU, two
stages of pre-LayerNorm2d MBConv blocks with average-pool shortcuts, and a
LayerNorm2d + strided 3x3 conv) reduces the image 16x; its tokens get a
learned position embedding and run through pre-norm ViT blocks whose MLP is a
GeGLU with its own input LayerNorm. Tokens are average-pooled and normalized
before the classifier.
"""

import math

import jax
import jax.numpy as jnp
from flax import nnx

from ..layers import ClassifierMixin, DropPath, gelu, make_divisible
from ..registry import _cfg, register_model
from .vision_transformer import Attention


def _conv(in_chs, out_chs, kernel=1, stride=1, groups=1, *, rngs):
    pad = kernel // 2
    fan_out = kernel * kernel * out_chs // groups
    return nnx.Conv(
        in_chs,
        out_chs,
        (kernel, kernel),
        strides=(stride, stride),
        padding=((pad, pad), (pad, pad)),
        feature_group_count=groups,
        kernel_init=nnx.initializers.normal(math.sqrt(2.0 / fan_out)),
        rngs=rngs,
    )


def _ln(dim, *, rngs):
    return nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)


def _avg_pool_3x3_s2(x):
    """``AvgPool2d(3, 2, padding=1, count_include_pad=False)``."""
    window, strides, pad = (1, 3, 3, 1), (1, 2, 2, 1), ((0, 0), (1, 1), (1, 1), (0, 0))
    total = jax.lax.reduce_window(x, 0.0, jax.lax.add, window, strides, pad)
    ones = jnp.ones((1, *x.shape[1:3], 1), x.dtype)
    count = jax.lax.reduce_window(ones, 0.0, jax.lax.add, window, strides, pad)
    return total / count


class Stem(nnx.Module):
    def __init__(self, in_chs, out_chs, *, rngs):
        self.conv1 = _conv(in_chs, out_chs, 3, 2, rngs=rngs)
        self.norm1 = _ln(out_chs, rngs=rngs)
        self.conv2 = _conv(out_chs, out_chs, 3, rngs=rngs)

    def __call__(self, x):
        return self.conv2(gelu(self.norm1(self.conv1(x))))


class Downsample2d(nnx.Module):
    def __init__(self, dim, dim_out, *, rngs):
        self.expand = _conv(dim, dim_out, rngs=rngs) if dim != dim_out else None

    def __call__(self, x):
        x = _avg_pool_3x3_s2(x)
        return self.expand(x) if self.expand is not None else x


class MbConvLNBlock(nnx.Module):
    def __init__(self, in_chs, out_chs, stride=1, drop_path=0.0, expand_ratio=4.0, *, rngs):
        mid = make_divisible(out_chs * expand_ratio)
        if stride == 2:
            self.shortcut = Downsample2d(in_chs, out_chs, rngs=rngs)
        elif in_chs != out_chs:
            self.shortcut = _conv(in_chs, out_chs, rngs=rngs)
        else:
            self.shortcut = None
        self.pre_norm = _ln(in_chs, rngs=rngs)
        self.conv1_1x1 = _conv(in_chs, mid, rngs=rngs)
        self.conv2_kxk = _conv(mid, mid, 3, stride, groups=mid, rngs=rngs)
        self.conv3_1x1 = _conv(mid, out_chs, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        shortcut = self.shortcut(x) if self.shortcut is not None else x
        x = gelu(self.conv1_1x1(self.pre_norm(x)))
        x = self.conv3_1x1(gelu(self.conv2_kxk(x)))
        return self.drop_path(x) + shortcut


class StridedConv(nnx.Module):
    def __init__(self, in_chs, embed_dim, *, rngs):
        self.proj = nnx.Conv(
            in_chs, embed_dim, (3, 3), strides=2, padding=((1, 1), (1, 1)), rngs=rngs
        )
        self.norm = _ln(in_chs, rngs=rngs)

    def __call__(self, x):
        return self.proj(self.norm(x))


class MbConvStages(nnx.Module):
    def __init__(self, embed_dims, depths, stem_width, in_chans=3, *, rngs):
        self.stem = Stem(in_chans, stem_width, rngs=rngs)
        stages = []
        for s, dim in enumerate(embed_dims[:2]):
            in_chs = embed_dims[s - 1] if s > 0 else stem_width
            blocks = [
                MbConvLNBlock(in_chs if d == 0 else dim, dim, 2 if d == 0 else 1, rngs=rngs)
                for d in range(depths[s])
            ]
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.pool = StridedConv(embed_dims[1], embed_dims[2], rngs=rngs)

    def __call__(self, x):
        x = self.stem(x)
        for stage in self.stages:
            for blk in stage:
                x = blk(x)
        return self.pool(x)


class HybridEmbed(nnx.Module):
    def __init__(self, backbone):
        self.backbone = backbone

    def __call__(self, x):
        x = self.backbone(x)
        B, H, W, C = x.shape
        return x.reshape(B, H * W, C)


_trunc = nnx.initializers.truncated_normal(0.02)


class GeGluMlp(nnx.Module):
    def __init__(self, dim, hidden, *, rngs):
        init = dict(kernel_init=nnx.initializers.xavier_uniform(), rngs=rngs)
        self.norm = _ln(dim, rngs=rngs)
        self.w0 = nnx.Linear(dim, hidden, **init)
        self.w1 = nnx.Linear(dim, hidden, **init)
        self.w2 = nnx.Linear(hidden, dim, **init)

    def __call__(self, x):
        x = self.norm(x)
        return self.w2(gelu(self.w0(x)) * self.w1(x))


class Block(nnx.Module):
    def __init__(self, dim, num_heads, mlp_ratio, drop_path=0.0, *, rngs):
        self.norm1 = _ln(dim, rngs=rngs)
        self.attn = Attention(dim, num_heads, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)
        self.norm2 = _ln(dim, rngs=rngs)
        self.mlp = GeGluMlp(dim, int(dim * mlp_ratio), rngs=rngs)

    def __call__(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class ViTamin(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"

    def __init__(
        self,
        conv_dims=(64, 128),
        conv_depths=(2, 4),
        stem_width=64,
        img_size=224,
        embed_dim=384,
        depth=14,
        num_heads=6,
        mlp_ratio=2.0,
        pos_embed=True,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        pos_drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dim
        backbone = MbConvStages(
            (*conv_dims, embed_dim), conv_depths, stem_width, in_chans, rngs=rngs
        )
        self.patch_embed = HybridEmbed(backbone)
        n = (img_size // 16) ** 2
        self.pos_embed = nnx.Param(_trunc(rngs.params(), (1, n, embed_dim))) if pos_embed else None
        self.pos_drop = nnx.Dropout(pos_drop_rate, rngs=rngs)
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nnx.List(
            [Block(embed_dim, num_heads, mlp_ratio, dpr[i], rngs=rngs) for i in range(depth)]
        )
        self.fc_norm = _ln(embed_dim, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = self._make_head(num_classes, rngs)

    def _make_head(self, num_classes, rngs):
        if num_classes <= 0:
            return None
        return nnx.Linear(self.num_features, num_classes, kernel_init=_trunc, rngs=rngs)

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.global_pool = global_pool if global_pool is not None else self.global_pool
        self.head = self._make_head(num_classes, nnx.Rngs(0))

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed[...]
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        return x

    def forward_head(self, x):
        if self.global_pool == "avg":
            x = x.mean(axis=1)
        x = self.head_drop(self.fc_norm(x))
        return self.head(x) if self.head is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_S = dict(conv_dims=(64, 128), stem_width=64, embed_dim=384, depth=14, num_heads=6)
_B = dict(conv_dims=(128, 256), stem_width=128, embed_dim=768, depth=14, num_heads=12)
_L = dict(conv_dims=(160, 320), stem_width=160, embed_dim=1024, depth=31, num_heads=16)
_XL = dict(
    conv_dims=(192, 384), stem_width=192, embed_dim=1152, depth=32, num_heads=16, pos_embed=False
)
_CFGS = {
    "vitamin_small_224": (_S, 224),
    "vitamin_base_224": (_B, 224),
    "vitamin_large_224": (_L, 224),
    "vitamin_large_256": (_L, 256),
    "vitamin_large_336": (_L, 336),
    "vitamin_large_384": (_L, 384),
    "vitamin_large2_224": (_L, 224),
    "vitamin_large2_256": (_L, 256),
    "vitamin_large2_336": (_L, 336),
    "vitamin_large2_384": (_L, 384),
    "vitamin_xlarge_256": (_XL, 256),
    "vitamin_xlarge_336": (_XL, 336),
    "vitamin_xlarge_384": (_XL, 384),
}


def _make(name):
    arch, size = _CFGS[name]

    def entry(**kwargs):
        model = ViTamin(**{**arch, "img_size": size, **kwargs})
        model.default_cfg = _cfg(
            input_size=(3, size, size),
            crop_pct=0.9 if size == 224 else 1.0,
            interpolation="bicubic",
            fixed_input_size=True,
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
