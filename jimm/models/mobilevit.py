"""MobileViT in flax nnx, NHWC. Mirrors timm.models.mobilevit (ByobNet-based v1 models).

Inverted residual blocks (1x1 expansion, depthwise 3x3, linear 1x1 projection,
residual only when the shape is unchanged) are followed in the last three
stages by MobileViT blocks: a local 3x3 convolution and a 1x1 projection,
transformer layers over sequences formed by the same pixel position of every
2x2 patch, a 1x1 projection back, and a 3x3 convolution fusing the result with
the block input. A 1x1 convolution widens the features before pooling.
"""

import math

import jax
import jax.numpy as jnp
from flax import nnx

from ..attention import dot_product_attention
from ..layers import ClassifierMixin, DropPath, global_pool_nhwc, make_divisible
from ..registry import _cfg, register_model
from ._conv import ConvNormAct


def _conv_norm_act(in_chs, out_chs, kernel=1, stride=1, groups=1, act=True, *, rngs):
    return ConvNormAct(
        in_chs, out_chs, kernel, stride, groups, act=nnx.silu if act else None, rngs=rngs
    )


class InvertedResidual(nnx.Module):
    """timm ByobNet bottleneck with ``bottle_in=True``, depthwise middle conv, linear output."""

    def __init__(self, in_chs, out_chs, stride=1, bottle_ratio=4.0, drop_path=0.0, *, rngs):
        mid = make_divisible(in_chs * bottle_ratio)
        self.has_shortcut = in_chs == out_chs and stride == 1
        self.conv1_1x1 = _conv_norm_act(in_chs, mid, rngs=rngs)
        self.conv2_kxk = _conv_norm_act(mid, mid, 3, stride, groups=mid, rngs=rngs)
        self.conv3_1x1 = _conv_norm_act(mid, out_chs, act=False, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        y = self.drop_path(self.conv3_1x1(self.conv2_kxk(self.conv1_1x1(x))))
        # timm's empty downsample type drops the shortcut when the shape changes.
        return x + y if self.has_shortcut else y


class TransformerBlock(nnx.Module):
    """Pre-norm ViT block with SiLU MLP."""

    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, drop_path=0.0, *, rngs):
        self.num_heads = num_heads
        self.norm1 = nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)
        self.qkv = nnx.Linear(dim, 3 * dim, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)
        self.fc1 = nnx.Linear(dim, int(dim * mlp_ratio), rngs=rngs)
        self.fc2 = nnx.Linear(int(dim * mlp_ratio), dim, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        B, N, C = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        y = dot_product_attention(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2])
        x = x + self.drop_path(self.proj(y.reshape(B, N, C)))
        return x + self.drop_path(self.fc2(nnx.silu(self.fc1(self.norm2(x)))))


class MobileVitBlock(nnx.Module):
    def __init__(
        self,
        chs,
        transformer_dim,
        transformer_depth=2,
        patch_size=2,
        num_heads=4,
        mlp_ratio=2.0,
        drop_path=0.0,
        *,
        rngs,
    ):
        self.patch_size = patch_size
        self.conv_kxk = _conv_norm_act(chs, chs, 3, rngs=rngs)
        self.conv_1x1 = nnx.Conv(chs, transformer_dim, (1, 1), use_bias=False, rngs=rngs)
        self.transformer = nnx.List(
            [
                TransformerBlock(transformer_dim, num_heads, mlp_ratio, drop_path, rngs=rngs)
                for _ in range(transformer_depth)
            ]
        )
        self.norm = nnx.LayerNorm(transformer_dim, epsilon=1e-5, rngs=rngs)
        self.conv_proj = _conv_norm_act(transformer_dim, chs, rngs=rngs)
        self.conv_fusion = _conv_norm_act(2 * chs, chs, 3, rngs=rngs)

    def __call__(self, x):
        shortcut = x
        x = self.conv_1x1(self.conv_kxk(x))
        B, H, W, C = x.shape
        p = self.patch_size
        new_h, new_w = math.ceil(H / p) * p, math.ceil(W / p) * p
        if (new_h, new_w) != (H, W):
            x = jax.image.resize(x, (B, new_h, new_w, C), "bilinear", antialias=False)
        nh, nw = new_h // p, new_w // p
        # Sequences of patches, one per pixel position within the p x p patch.
        x = x.reshape(B, nh, p, nw, p, C).transpose(0, 2, 4, 1, 3, 5).reshape(B * p * p, nh * nw, C)
        for blk in self.transformer:
            x = blk(x)
        x = self.norm(x)
        x = x.reshape(B, p, p, nh, nw, C).transpose(0, 3, 1, 4, 2, 5).reshape(B, new_h, new_w, C)
        if (new_h, new_w) != (H, W):
            x = jax.image.resize(x, (B, H, W, C), "bilinear", antialias=False)
        x = self.conv_proj(x)
        return self.conv_fusion(jnp.concatenate([shortcut, x], axis=-1))


class LinearSelfAttention(nnx.Module):
    """timm's linear self-attention: softmax over patches, key-weighted context vector."""

    def __init__(self, dim, attn_drop=0.0, proj_drop=0.0, *, rngs):
        self.dim = dim
        self.qkv = nnx.Linear(dim, 1 + 2 * dim, rngs=rngs)
        self.attn_drop = nnx.Dropout(attn_drop, rngs=rngs)
        self.out_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.out_drop = nnx.Dropout(proj_drop, rngs=rngs)

    def __call__(self, x):
        # x: (B, patch_area, num_patches, C)
        qkv = self.qkv(x)
        query, key, value = qkv[..., :1], qkv[..., 1 : 1 + self.dim], qkv[..., 1 + self.dim :]
        scores = self.attn_drop(jax.nn.softmax(query, axis=-1))
        context = jnp.sum(key * scores, axis=-2, keepdims=True)
        return self.out_drop(self.out_proj(nnx.relu(value) * context))


class LinearTransformerBlock(nnx.Module):
    """Pre-norm block with MobileViTv2 linear self-attention and a 1x1-conv MLP."""

    def __init__(self, dim, mlp_ratio=2.0, drop=0.0, attn_drop=0.0, drop_path=0.0, *, rngs):
        self.norm1 = nnx.GroupNorm(dim, num_groups=1, epsilon=1e-5, rngs=rngs)
        self.attn = LinearSelfAttention(dim, attn_drop, drop, rngs=rngs)
        self.drop_path1 = DropPath(drop_path, rngs=rngs)
        self.norm2 = nnx.GroupNorm(dim, num_groups=1, epsilon=1e-5, rngs=rngs)
        hidden = int(dim * mlp_ratio)
        self.fc1 = nnx.Linear(dim, hidden, rngs=rngs)
        self.drop = nnx.Dropout(drop, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, dim, rngs=rngs)
        self.drop_path2 = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        y = self.fc2(self.drop(nnx.silu(self.fc1(self.norm2(x)))))
        return x + self.drop_path2(y)


class MobileVitV2Block(nnx.Module):
    """A depthwise 3x3 conv, a 1x1 projection, linear-attention transformer blocks,
    a single-group GroupNorm, and a 1x1 projection with BatchNorm (no shortcut)."""

    def __init__(
        self,
        chs,
        transformer_depth=2,
        patch_size=2,
        bottle_ratio=0.5,
        mlp_ratio=2.0,
        drop_path=0.0,
        *,
        rngs,
    ):
        self.patch_size = patch_size
        tdim = make_divisible(bottle_ratio * chs)
        self.conv_kxk = _conv_norm_act(chs, chs, 3, groups=chs, rngs=rngs)
        self.conv_1x1 = nnx.Conv(chs, tdim, (1, 1), use_bias=False, rngs=rngs)
        self.transformer = nnx.List(
            [
                LinearTransformerBlock(tdim, mlp_ratio, drop_path=drop_path, rngs=rngs)
                for _ in range(transformer_depth)
            ]
        )
        self.norm = nnx.GroupNorm(tdim, num_groups=1, epsilon=1e-5, rngs=rngs)
        self.conv_proj = _conv_norm_act(tdim, chs, act=False, rngs=rngs)

    def __call__(self, x):
        B, H, W, _ = x.shape
        p = self.patch_size
        new_h, new_w = math.ceil(H / p) * p, math.ceil(W / p) * p
        if (new_h, new_w) != (H, W):
            x = jax.image.resize(x, (B, new_h, new_w, x.shape[-1]), "bilinear", antialias=False)
        x = self.conv_1x1(self.conv_kxk(x))
        C = x.shape[-1]
        nh, nw = new_h // p, new_w // p
        x = x.reshape(B, nh, p, nw, p, C).transpose(0, 2, 4, 1, 3, 5).reshape(B, p * p, nh * nw, C)
        for blk in self.transformer:
            x = blk(x)
        x = self.norm(x)
        x = x.reshape(B, p, p, nh, nw, C).transpose(0, 3, 1, 4, 2, 5).reshape(B, new_h, new_w, C)
        if (new_h, new_w) != (H, W):
            x = jax.image.resize(x, (B, H, W, C), "bilinear", antialias=False)
        return self.conv_proj(x)


class MobileVit(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        stages,
        stem_chs=16,
        num_features=640,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        """``stages``: (depth, out_chs, stride, bottle_ratio, transformer_dim, transformer_depth);
        a transformer_dim of None means an inverted-residual-only stage."""
        self.num_classes, self.global_pool = num_classes, global_pool
        self.stem = _conv_norm_act(in_chans, stem_chs, 3, 2, rngs=rngs)
        out, prev = [], stem_chs
        for depth, chs, stride, br, t_dim, t_depth in stages:
            blocks = []
            for i in range(depth):
                blocks.append(
                    InvertedResidual(
                        prev, chs, stride if i == 0 else 1, br, drop_path_rate, rngs=rngs
                    )
                )
                prev = chs
            if t_dim is not None:
                blocks.append(
                    MobileVitBlock(chs, t_dim, t_depth, drop_path=drop_path_rate, rngs=rngs)
                )
            out.append(nnx.List(blocks))
        self.stages = nnx.List(out)
        self.final_conv = _conv_norm_act(prev, num_features, rngs=rngs)
        self.num_features = num_features
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.stem(x)
        for stage in self.stages:
            for blk in stage:
                x = blk(x)
        return self.final_conv(x)

    def forward_head(self, x):
        x = self.head_drop(global_pool_nhwc(x, self.global_pool))
        return self.fc(x) if self.fc is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


class MobileVitV2(ClassifierMixin, nnx.Module):
    """MobileViTv2: inverted-residual stages with linear-attention transformer blocks."""

    def __init__(
        self,
        stages,
        stem_chs=32,
        num_features=0,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        """``stages``: (depth, out_chs, stride, bottle_ratio, transformer_depth); a
        transformer depth of 0 means an inverted-residual-only stage."""
        self.num_classes, self.global_pool = num_classes, global_pool
        self.stem = _conv_norm_act(in_chans, stem_chs, 3, 2, rngs=rngs)
        out, prev = [], stem_chs
        for depth, chs, stride, br, t_depth in stages:
            blocks = []
            for i in range(depth):
                blocks.append(
                    InvertedResidual(
                        prev, chs, stride if i == 0 else 1, br, drop_path_rate, rngs=rngs
                    )
                )
                prev = chs
            if t_depth:
                blocks.append(MobileVitV2Block(chs, t_depth, drop_path=drop_path_rate, rngs=rngs))
            out.append(nnx.List(blocks))
        self.stages = nnx.List(out)
        if num_features:
            self.final_conv = _conv_norm_act(prev, num_features, rngs=rngs)
            self.num_features = num_features
        else:
            self.final_conv = None
            self.num_features = prev
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.stem(x)
        for stage in self.stages:
            for blk in stage:
                x = blk(x)
        return self.final_conv(x) if self.final_conv is not None else x

    def forward_head(self, x):
        x = self.head_drop(global_pool_nhwc(x, self.global_pool))
        return self.fc(x) if self.fc is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


def _stages(chs, t_dims, br):
    return [
        (1, chs[0], 1, br, None, None),
        (3, chs[1], 2, br, None, None),
        (1, chs[2], 2, br, t_dims[0], 2),
        (1, chs[3], 2, br, t_dims[1], 4),
        (1, chs[4], 2, br, t_dims[2], 3),
    ]


_CFGS = {
    "mobilevit_xxs": (_stages((16, 24, 48, 64, 80), (64, 80, 96), 2.0), 320),
    "mobilevit_xs": (_stages((32, 48, 64, 80, 96), (96, 120, 144), 4.0), 384),
    "mobilevit_s": (_stages((32, 64, 96, 128, 160), (144, 192, 240), 4.0), 640),
}


def _v2_stages(multiplier):
    chs = tuple(int(c * multiplier) for c in (64, 128, 256, 384, 512))
    return (
        (1, chs[0], 1, 2.0, 0),
        (2, chs[1], 2, 2.0, 0),
        (1, chs[2], 2, 2.0, 2),
        (1, chs[3], 2, 2.0, 4),
        (1, chs[4], 2, 2.0, 3),
    ), int(32 * multiplier)


_V2_CFGS = {
    f"mobilevitv2_{name}": _v2_stages(mult)
    for name, mult in (
        ("050", 0.50),
        ("075", 0.75),
        ("100", 1.0),
        ("125", 1.25),
        ("150", 1.50),
        ("175", 1.75),
        ("200", 2.0),
    )
}


def _make(name):
    stages, num_features = _CFGS[name]

    def entry(**kwargs):
        model = MobileVit(stages, num_features=num_features, **kwargs)
        model.default_cfg = _cfg(
            input_size=(3, 256, 256),
            crop_pct=0.9,
            interpolation="bicubic",
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
        )
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))


def _make_v2(name):
    stages, stem_chs = _V2_CFGS[name]

    def entry(**kwargs):
        model = MobileVitV2(stages, stem_chs=stem_chs, **kwargs)
        model.default_cfg = _cfg(
            input_size=(3, 256, 256),
            crop_pct=0.888,
            interpolation="bicubic",
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
        )
        return model

    entry.__name__ = name
    return entry


for _name in _V2_CFGS:
    register_model(_make_v2(_name))
