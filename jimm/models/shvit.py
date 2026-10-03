"""SHViT (Single-Head Vision Transformer) in flax nnx, NHWC. Mirrors timm.models.shvit.

A four-conv stride-16 stem feeds three stages of blocks: a residual
depthwise 3x3 conv + BatchNorm, then (except in the first stage) single-head
attention over a slice of the channels after a single-group GroupNorm, with
the remaining channels passed through, and a residual 1x1-conv FFN. Between
stages, residual depthwise and FFN blocks surround a patch merging block
(1x1 conv, strided depthwise conv, squeeze-excite, 1x1 conv). The head pools
and applies BatchNorm and a linear classifier.
"""

import jax.numpy as jnp
from flax import nnx

from ..attention import dot_product_attention
from ..layers import BatchNorm, ClassifierMixin, global_pool_nhwc, make_divisible
from ..registry import _cfg, register_model
from ._conv import ConvNormAct


def _conv_norm(in_chs, out_chs, kernel=1, stride=1, groups=1, bn_weight_init=1.0, *, rngs):
    return ConvNormAct(
        in_chs, out_chs, kernel, stride, groups, bn_weight_init=bn_weight_init, rngs=rngs
    )


class SqueezeExcite(nnx.Module):
    def __init__(self, chs, rd_ratio=0.25, *, rngs):
        rd = make_divisible(chs * rd_ratio, 8)
        self.fc1 = nnx.Linear(chs, rd, rngs=rngs)
        self.fc2 = nnx.Linear(rd, chs, rngs=rngs)

    def __call__(self, x):
        s = self.fc2(nnx.relu(self.fc1(jnp.mean(x, axis=(1, 2), keepdims=True))))
        return x * nnx.sigmoid(s)


class FFN(nnx.Module):
    def __init__(self, dim, hidden, *, rngs):
        self.pw1 = _conv_norm(dim, hidden, rngs=rngs)
        self.pw2 = _conv_norm(hidden, dim, bn_weight_init=0.0, rngs=rngs)

    def __call__(self, x):
        return x + self.pw2(nnx.relu(self.pw1(x)))


class DWConv(nnx.Module):
    """Residual depthwise 3x3 conv + BatchNorm."""

    def __init__(self, dim, bn_weight_init=1.0, *, rngs):
        self.conv = _conv_norm(dim, dim, 3, groups=dim, bn_weight_init=bn_weight_init, rngs=rngs)

    def __call__(self, x):
        return x + self.conv(x)


class PatchMerging(nnx.Module):
    def __init__(self, dim, out_dim, *, rngs):
        hid = 4 * dim
        self.conv1 = _conv_norm(dim, hid, rngs=rngs)
        self.conv2 = _conv_norm(hid, hid, 3, 2, groups=hid, rngs=rngs)
        self.se = SqueezeExcite(hid, rngs=rngs)
        self.conv3 = _conv_norm(hid, out_dim, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.conv2(nnx.relu(self.conv1(x))))
        return self.conv3(self.se(x))


class SHSA(nnx.Module):
    """Single-head self-attention over the first ``pdim`` channels."""

    def __init__(self, dim, qk_dim, pdim, *, rngs):
        self.qk_dim, self.pdim = qk_dim, pdim
        self.pre_norm = nnx.GroupNorm(pdim, num_groups=1, epsilon=1e-5, rngs=rngs)
        self.qkv = _conv_norm(pdim, 2 * qk_dim + pdim, rngs=rngs)
        self.proj = _conv_norm(dim, dim, bn_weight_init=0.0, rngs=rngs)

    def __call__(self, x):
        B, H, W, C = x.shape
        x1, x2 = x[..., : self.pdim], x[..., self.pdim :]
        qkv = self.qkv(self.pre_norm(x1)).reshape(B, H * W, 1, -1)
        q, k, v = jnp.split(qkv, [self.qk_dim, 2 * self.qk_dim], axis=-1)
        x1 = dot_product_attention(q, k, v).reshape(B, H, W, self.pdim)
        return x + self.proj(nnx.relu(jnp.concatenate([x1, x2], axis=-1)))


class BasicBlock(nnx.Module):
    def __init__(self, dim, qk_dim, pdim, use_attn, *, rngs):
        self.conv = DWConv(dim, bn_weight_init=0.0, rngs=rngs)
        self.mixer = SHSA(dim, qk_dim, pdim, rngs=rngs) if use_attn else None
        self.ffn = FFN(dim, 2 * dim, rngs=rngs)

    def __call__(self, x):
        x = self.conv(x)
        if self.mixer is not None:
            x = self.mixer(x)
        return self.ffn(x)


class StageBlock(nnx.Module):
    def __init__(self, prev_dim, dim, qk_dim, pdim, use_attn, depth, *, rngs):
        if prev_dim != dim:
            self.downsample = nnx.List(
                [
                    DWConv(prev_dim, rngs=rngs),
                    FFN(prev_dim, 2 * prev_dim, rngs=rngs),
                    PatchMerging(prev_dim, dim, rngs=rngs),
                    DWConv(dim, rngs=rngs),
                    FFN(dim, 2 * dim, rngs=rngs),
                ]
            )
        else:
            self.downsample = None
        self.blocks = nnx.List(
            [BasicBlock(dim, qk_dim, pdim, use_attn, rngs=rngs) for _ in range(depth)]
        )

    def __call__(self, x):
        if self.downsample is not None:
            for layer in self.downsample:
                x = layer(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class NormLinear(nnx.Module):
    def __init__(self, dim, num_classes, *, rngs):
        self.bn = BatchNorm(dim, epsilon=1e-5, rngs=rngs)
        self.l = nnx.Linear(
            dim, num_classes, kernel_init=nnx.initializers.truncated_normal(0.02), rngs=rngs
        )

    def __call__(self, x):
        return self.l(self.bn(x))


class SHViT(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"

    def __init__(
        self,
        embed_dim=(128, 256, 384),
        partial_dim=(32, 64, 96),
        qk_dim=(16, 16, 16),
        depth=(1, 2, 3),
        types=("s", "s", "s"),
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        stem = embed_dim[0]
        chs = [in_chans, stem // 8, stem // 4, stem // 2, stem]
        self.patch_embed = nnx.List(
            [_conv_norm(chs[i], chs[i + 1], 3, 2, rngs=rngs) for i in range(4)]
        )
        stages, prev = [], stem
        for i, dim in enumerate(embed_dim):
            stages.append(
                StageBlock(
                    prev, dim, qk_dim[i], partial_dim[i], types[i] == "s", depth[i], rngs=rngs
                )
            )
            prev = dim
        self.stages = nnx.List(stages)
        self.num_features = embed_dim[-1]
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = (
            NormLinear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None
        )

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.global_pool = global_pool if global_pool is not None else self.global_pool
        self.head = (
            NormLinear(self.num_features, num_classes, rngs=nnx.Rngs(0))
            if num_classes > 0
            else None
        )

    def forward_features(self, x):
        for i, conv in enumerate(self.patch_embed):
            x = conv(x)
            if i < 3:
                x = nnx.relu(x)
        for stage in self.stages:
            x = stage(x)
        return x

    def forward_head(self, x):
        x = self.head_drop(global_pool_nhwc(x, self.global_pool))
        return self.head(x) if self.head is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_TYPES = ("i", "s", "s")
_CFGS = {  # embed_dim, depth, partial_dim
    "shvit_s1": ((128, 224, 320), (2, 4, 5), (32, 48, 68)),
    "shvit_s2": ((128, 308, 448), (2, 4, 5), (32, 66, 96)),
    "shvit_s3": ((192, 352, 448), (3, 5, 5), (48, 75, 96)),
    "shvit_s4": ((224, 336, 448), (4, 7, 6), (48, 72, 96)),
}


def _make(name):
    embed_dim, depth, partial_dim = _CFGS[name]

    def entry(**kwargs):
        model = SHViT(embed_dim, partial_dim, depth=depth, types=_TYPES, **kwargs)
        model.default_cfg = _cfg(interpolation="bicubic")
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
