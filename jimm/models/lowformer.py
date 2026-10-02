"""LowFormer in Flax NNX, adapted from timm 1.0.30 (NHWC).

Reference: https://github.com/huggingface/pytorch-image-models/blob/v1.0.30/timm/models/lowformer.py
Copyright 2026 Ryan Hou & Ross Wightman. JAX adaptation for jimm.
"""

import jax.numpy as jnp
from flax import nnx

from ..features import _select_features
from ..layers import ClassifierMixin, DropPath, gelu, global_pool_nhwc, hswish
from ..registry import _cfg, register_model
from ._conv import ConvNormAct, ConvTranspose


class MBConv(nnx.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        stride=1,
        expand_ratio=4,
        expand_groups=1,
        use_bias=False,
        act_layer=hswish,
        fused=False,
        *,
        rngs,
    ):
        mid = round(in_chs * expand_ratio)
        if fused:
            self.expand = ConvNormAct(
                in_chs,
                mid,
                3,
                stride,
                groups=expand_groups,
                use_bias=use_bias,
                act=act_layer,
                rngs=rngs,
            )
            self.depthwise = None
        else:
            self.expand = ConvNormAct(
                in_chs,
                mid,
                groups=expand_groups,
                norm=False,
                use_bias=use_bias,
                act=act_layer,
                rngs=rngs,
            )
            self.depthwise = ConvNormAct(
                mid,
                mid,
                3,
                stride,
                groups=mid,
                norm=False,
                use_bias=use_bias,
                act=act_layer,
                rngs=rngs,
            )
        self.project = ConvNormAct(mid, out_chs, groups=1 if fused else expand_groups, rngs=rngs)

    def __call__(self, x):
        x = self.expand(x)
        if self.depthwise is not None:
            x = self.depthwise(x)
        return self.project(x)


class Residual(nnx.Module):
    def __init__(self, module, shortcut=True, drop_path=0.0, *, rngs):
        self.module = module
        self.shortcut = shortcut
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        y = self.module(x)
        return x + self.drop_path(y) if self.shortcut else y


class ConvAttention(nnx.Module):
    """Strided depthwise QKV projection, attention, and learned convolutional upsampling."""

    def __init__(
        self,
        input_dim,
        head_dim_mul=0.5,
        att_stride=4,
        att_kernel=7,
        fuse_out_proj=False,
        *,
        rngs,
    ):
        self.num_heads = int(max(1, (input_dim * head_dim_mul) // 30))
        self.head_dim = int((input_dim // self.num_heads) * head_dim_mul)
        inner = self.num_heads * self.head_dim
        self.conv_proj = ConvNormAct(
            input_dim, input_dim, att_kernel, att_stride, groups=input_dim, rngs=rngs
        )
        self.pwise = nnx.Conv(input_dim, 3 * inner, (1, 1), use_bias=False, rngs=rngs)
        self.o_proj = None if fuse_out_proj else nnx.Conv(inner, input_dim, (1, 1), rngs=rngs)
        kernel, padding = (3, 1) if att_stride == 1 else (2 * att_stride, att_stride // 2)
        self.upsampling = ConvTranspose(
            inner if fuse_out_proj else input_dim,
            input_dim,
            kernel,
            att_stride,
            padding,
            groups=1 if fuse_out_proj else input_dim,
            rngs=rngs,
        )

    def __call__(self, x):
        height, width = x.shape[1:3]
        x = self.pwise(self.conv_proj(x))
        batch, h, w, _ = x.shape
        qkv = x.reshape(batch, h * w, self.num_heads, 3 * self.head_dim)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        x = nnx.dot_product_attention(q, k, v).reshape(batch, h, w, -1)
        if self.o_proj is not None:
            x = self.o_proj(x)
        return self.upsampling(x)[:, :height, :width]


class LowFormerBlock(nnx.Module):
    def __init__(
        self,
        dim,
        expand_ratio=4,
        fused_conv=False,
        expand_groups=1,
        attn=True,
        attn_mlp=True,
        attn_mlp_ratio=4,
        att_stride=1,
        proj_drop=0.0,
        drop_path=0.0,
        *,
        rngs,
    ):
        self.attn_norm = (
            nnx.GroupNorm(dim, num_groups=1, epsilon=1e-5, rngs=rngs) if attn and attn_mlp else None
        )
        self.attn = (
            ConvAttention(
                dim,
                att_stride=att_stride,
                att_kernel=5 if att_stride > 1 else 3,
                fuse_out_proj=fused_conv,
                rngs=rngs,
            )
            if attn
            else None
        )
        self.mlp = (
            nnx.Sequential(
                nnx.GroupNorm(dim, num_groups=1, epsilon=1e-5, rngs=rngs),
                nnx.Conv(dim, dim * attn_mlp_ratio, (1, 1), rngs=rngs),
                gelu,
                nnx.Conv(dim * attn_mlp_ratio, dim, (1, 1), rngs=rngs),
                nnx.Dropout(proj_drop, rngs=rngs),
            )
            if attn_mlp
            else None
        )
        self.local = MBConv(
            dim,
            dim,
            expand_ratio=expand_ratio,
            expand_groups=expand_groups,
            use_bias=True,
            fused=fused_conv and dim < 256,
            rngs=rngs,
        )
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        if self.attn is not None:
            y = self.attn_norm(x) if self.attn_norm is not None else x
            x = x + self.drop_path(self.attn(y))
        if self.mlp is not None:
            x = x + self.drop_path(self.mlp(x))
        return x + self.drop_path(self.local(x))


class LowFormer(ClassifierMixin, nnx.Module):
    _classifier_attr = "classifier"

    def __init__(
        self,
        width_list=(16, 32, 64, 128, 256),
        depth_list=(0, 1, 1, 3, 4),
        in_chans=3,
        num_classes=1000,
        global_pool="avg",
        head_widths=(1536, 1600),
        drop_rate=0.0,
        proj_drop_rate=0.0,
        drop_path_rate=0.0,
        expand_ratio=4,
        fused_conv=True,
        attn=True,
        attn_mlp=True,
        attn_mlp_ratio=4,
        stem_expand_ratio=2,
        downsample_expand_ratios=None,
        expand_groups=1,
        *,
        rngs,
    ):
        if len(width_list) != len(depth_list):
            raise ValueError("width_list and depth_list must have equal lengths")
        num_stages = len(width_list) - 1
        downsample_expand_ratios = downsample_expand_ratios or (expand_ratio,) * num_stages
        if len(downsample_expand_ratios) != num_stages:
            raise ValueError("one downsample expansion ratio is required per stage")
        self.num_classes, self.global_pool = num_classes, global_pool
        # jimm's num_features is the width returned after reset_classifier(0).
        self.num_features = self.head_hidden_size = head_widths[1]
        rates = [drop_path_rate * i / max(sum(depth_list) - 1, 1) for i in range(sum(depth_list))]
        stem = [ConvNormAct(in_chans, width_list[0], 3, 2, act=hswish, rngs=rngs)]
        for i in range(depth_list[0]):
            stem.append(
                Residual(
                    MBConv(
                        width_list[0],
                        width_list[0],
                        expand_ratio=stem_expand_ratio,
                        expand_groups=expand_groups,
                        fused=fused_conv,
                        rngs=rngs,
                    ),
                    drop_path=rates[i],
                    rngs=rngs,
                )
            )
        self.stem = nnx.Sequential(*stem)
        stages, offset, chs = [], depth_list[0], width_list[0]
        for i, (width, depth) in enumerate(zip(width_list[1:], depth_list[1:])):
            blocks = []
            if i >= 2:
                blocks.append(
                    MBConv(
                        chs,
                        width,
                        stride=2,
                        expand_ratio=downsample_expand_ratios[i],
                        expand_groups=expand_groups,
                        fused=fused_conv,
                        rngs=rngs,
                    )
                )
                chs = width
                for j in range(depth):
                    blocks.append(
                        LowFormerBlock(
                            width,
                            expand_ratio,
                            fused_conv,
                            expand_groups,
                            attn,
                            attn_mlp,
                            attn_mlp_ratio,
                            2 if i == 2 else 1,
                            proj_drop_rate,
                            rates[offset + j],
                            rngs=rngs,
                        )
                    )
            else:
                for j in range(depth):
                    blocks.append(
                        Residual(
                            MBConv(
                                chs,
                                width,
                                stride=2 if j == 0 else 1,
                                expand_ratio=downsample_expand_ratios[i]
                                if j == 0
                                else expand_ratio,
                                expand_groups=expand_groups,
                                fused=fused_conv,
                                rngs=rngs,
                            ),
                            shortcut=j != 0,
                            drop_path=rates[offset + j],
                            rngs=rngs,
                        )
                    )
                    chs = width
            stages.append(nnx.Sequential(*blocks))
            offset += depth
        self.stages = nnx.List(stages)
        self.in_conv = ConvNormAct(width_list[-1], head_widths[0], act=hswish, rngs=rngs)
        self.pre_classifier = nnx.Sequential(
            nnx.Linear(head_widths[0], head_widths[1], use_bias=False, rngs=rngs),
            nnx.LayerNorm(head_widths[1], epsilon=1e-5, rngs=rngs),
            hswish,
        )
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.classifier = (
            nnx.Linear(head_widths[1], num_classes, rngs=rngs) if num_classes > 0 else None
        )

    def forward_intermediates(self, x, out_indices=None):
        x, features = self.stem(x), []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return _select_features(features, out_indices)

    def forward_features(self, x):
        return self.forward_intermediates(x)[-1]

    def forward_head(self, x, pre_logits=False):
        x = self.pre_classifier(global_pool_nhwc(self.in_conv(x), self.global_pool))
        x = self.head_drop(x)
        return x if pre_logits or self.classifier is None else self.classifier(x)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_CFGS = {
    "lowformer_b0": dict(width_list=(16, 32, 64, 128, 256), depth_list=(0, 1, 1, 3, 4)),
    "lowformer_b1": dict(
        width_list=(16, 32, 64, 128, 256),
        depth_list=(0, 1, 1, 5, 5),
        downsample_expand_ratios=(6, 6, 6, 6),
    ),
    "lowformer_b15": dict(
        width_list=(20, 40, 80, 160, 320),
        depth_list=(0, 1, 1, 6, 6),
        head_widths=(2304, 2560),
        downsample_expand_ratios=(6, 6, 6, 6),
    ),
    "lowformer_b2": dict(
        width_list=(24, 48, 96, 192, 384),
        depth_list=(0, 1, 1, 6, 6),
        head_widths=(2304, 2560),
        downsample_expand_ratios=(6, 6, 6, 6),
    ),
    "lowformer_b3": dict(
        width_list=(32, 64, 128, 256, 512),
        depth_list=(1, 2, 3, 6, 6),
        stem_expand_ratio=4,
        downsample_expand_ratios=(6, 6, 6, 6),
    ),
    "lowformer_e1": dict(
        width_list=(20, 40, 80, 160, 320),
        depth_list=(0, 1, 1, 4, 4),
        head_widths=(2304, 2560),
        attn=False,
        attn_mlp=False,
        downsample_expand_ratios=(6, 6, 6, 6),
    ),
    "lowformer_e2": dict(
        width_list=(32, 64, 128, 256, 512),
        depth_list=(1, 2, 3, 4, 4),
        attn=False,
        attn_mlp=False,
        stem_expand_ratio=4,
        downsample_expand_ratios=(6, 6, 6, 6),
    ),
    "lowformer_e3": dict(
        width_list=(32, 64, 128, 256, 512),
        depth_list=(1, 2, 3, 6, 6),
        attn_mlp=False,
        stem_expand_ratio=4,
        downsample_expand_ratios=(6, 6, 6, 6),
    ),
}
_DEFAULT_CFG = _cfg(interpolation="bicubic", crop_pct=0.95)


def _make(name):
    def entry(**kwargs):
        model = LowFormer(**dict(_CFGS[name], **kwargs))
        model.default_cfg = dict(_DEFAULT_CFG)
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name), default_cfg=_DEFAULT_CFG)
