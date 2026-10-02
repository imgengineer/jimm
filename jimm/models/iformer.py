"""iFormer in Flax NNX, adapted from timm 1.0.30 (NHWC).

Reference: https://github.com/huggingface/pytorch-image-models/blob/v1.0.30/timm/models/iformer.py
Copyright (c) Chuanyang Zheng; Copyright 2026 Ryan Hou & Ross Wightman.
JAX adaptation for jimm.
"""

import jax.numpy as jnp
from flax import nnx

from ..features import _select_features
from ..layers import DropPath, gelu, global_pool_nhwc
from ..registry import _cfg, register_model
from ._conv import ConvNormAct


class SHMA(nnx.Module):
    """Single-head modulation attention with sigmoid value and output gates."""

    def __init__(self, dim, ratio=1, head_dim_reduction_ratio=2, window_size=0, *, rngs):
        dim_attn = dim // head_dim_reduction_ratio
        self.scale = dim_attn**-0.5
        self.window_size = window_size
        self.q = ConvNormAct(dim, dim_attn, rngs=rngs)
        self.k = ConvNormAct(dim, dim_attn, rngs=rngs)
        self.v_gate = ConvNormAct(dim, 2 * dim * ratio, rngs=rngs)
        self.proj = ConvNormAct(dim * ratio, dim, rngs=rngs)

    def __call__(self, x):
        batch, height, width, channels = x.shape
        ws = self.window_size
        if ws:
            x = jnp.pad(x, ((0, 0), (0, -height % ws), (0, -width % ws), (0, 0)))
            hp, wp = x.shape[1:3]
            x = x.reshape(batch, hp // ws, ws, wp // ws, ws, channels)
            x = x.transpose(0, 1, 3, 2, 4, 5).reshape(-1, ws, ws, channels)
        b, h, w, _ = x.shape
        v, gate = jnp.split(nnx.sigmoid(self.v_gate(x)), 2, axis=-1)
        q, k = self.q(x).reshape(b, h * w, -1), self.k(x).reshape(b, h * w, -1)
        attn = nnx.softmax((q * self.scale) @ k.swapaxes(-1, -2), axis=-1)
        y = (attn @ v.reshape(b, h * w, -1)).reshape(v.shape)
        y = self.proj(y * gate)
        if ws:
            y = y.reshape(batch, hp // ws, wp // ws, ws, ws, channels)
            y = y.transpose(0, 1, 3, 2, 4, 5).reshape(batch, hp, wp, channels)
            y = y[:, :height, :width]
        return y


class Residual(nnx.Module):
    def __init__(self, module, dim, drop_path=0.0, layer_scale=0.0, *, rngs):
        self.module = module
        self.gamma = nnx.Param(jnp.full((dim,), layer_scale)) if layer_scale > 0 else None
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        y = self.drop_path(self.module(x))
        if self.gamma is not None:
            y = y * self.gamma[...]
        return x + y


class IFormerStage(nnx.Module):
    def __init__(
        self,
        in_dim,
        dim,
        depth,
        num_attn,
        hdrr,
        conv_ratio,
        ffn_ratio,
        attn_ratio,
        rates,
        scale,
        *,
        rngs,
    ):
        if depth < 3 * num_attn:
            raise ValueError("each attention group requires three blocks")
        self.downsample = ConvNormAct(in_dim, dim, 3, 2, rngs=rngs) if in_dim != dim else None

        def conv_block(index):
            return Residual(
                nnx.Sequential(
                    ConvNormAct(dim, dim, 7, groups=dim, rngs=rngs),
                    ConvNormAct(dim, dim * conv_ratio, act=gelu, rngs=rngs),
                    ConvNormAct(dim * conv_ratio, dim, rngs=rngs),
                ),
                dim,
                rates[index],
                scale,
                rngs=rngs,
            )

        blocks = []
        if num_attn == 0:
            blocks = [conv_block(i) for i in range(depth)]
        else:
            num_conv = depth - 3 * num_attn
            prefix = max(num_conv - 1, 0)
            blocks.extend(conv_block(i) for i in range(prefix))
            for group in range(num_attn):
                offset = prefix + 3 * group
                blocks.extend(
                    [
                        Residual(ConvNormAct(dim, dim, 3, groups=dim, rngs=rngs), dim, rngs=rngs),
                        Residual(
                            SHMA(dim, attn_ratio, hdrr, rngs=rngs),
                            dim,
                            rates[offset + 1],
                            scale,
                            rngs=rngs,
                        ),
                        Residual(
                            nnx.Sequential(
                                ConvNormAct(dim, dim * ffn_ratio, act=gelu, rngs=rngs),
                                ConvNormAct(dim * ffn_ratio, dim, rngs=rngs),
                            ),
                            dim,
                            rates[offset + 2],
                            scale,
                            rngs=rngs,
                        ),
                    ]
                )
            if num_conv:
                blocks.append(conv_block(depth - 1))
        self.blocks = nnx.List(blocks)

    def __call__(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x


class NormLinear(nnx.Module):
    def __init__(self, dim, num_classes, *, rngs):
        self.bn = nnx.BatchNorm(dim, epsilon=1e-5, momentum=0.9, rngs=rngs)
        self.linear = nnx.Linear(dim, num_classes, rngs=rngs)

    def __call__(self, x):
        return self.linear(self.bn(x))


class IFormer(nnx.Module):
    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        global_pool="avg",
        dims=(32, 64, 128, 256),
        depths=(2, 2, 16, 6),
        attn_groups=(0, 0, 3, 2),
        attn_head_dim_reduction=(0, 0, 2, 4),
        conv_ratio=3,
        ffn_ratio=2,
        attn_ratio=1,
        drop_rate=0.0,
        drop_path_rate=0.0,
        layer_scale_init_value=0.0,
        distillation=False,
        *,
        rngs,
    ):
        if not len(dims) == len(depths) == len(attn_groups) == len(attn_head_dim_reduction):
            raise ValueError("stage configurations must have equal lengths")
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = dims[-1]
        self.distillation = distillation
        self.distilled_training = False
        self.deterministic = False
        self.stem = nnx.Sequential(
            ConvNormAct(in_chans, dims[0] // 2, 5, 2, act=gelu, rngs=rngs),
            ConvNormAct(dims[0] // 2, (dims[0] // 2) * 4, 5, 2, act=gelu, rngs=rngs),
            ConvNormAct((dims[0] // 2) * 4, dims[0], rngs=rngs),
        )
        total, offset, prev_dim = sum(depths), 0, dims[0]
        stages = []
        for dim, depth, groups, hdrr in zip(dims, depths, attn_groups, attn_head_dim_reduction):
            rates = [drop_path_rate * (offset + j) / max(total - 1, 1) for j in range(depth)]
            stages.append(
                IFormerStage(
                    prev_dim,
                    dim,
                    depth,
                    groups,
                    hdrr,
                    conv_ratio,
                    ffn_ratio,
                    attn_ratio,
                    rates,
                    layer_scale_init_value,
                    rngs=rngs,
                )
            )
            prev_dim, offset = dim, offset + depth
        self.stages = nnx.List(stages)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = NormLinear(dims[-1], num_classes, rngs=rngs) if num_classes > 0 else None
        self.head_dist = (
            NormLinear(dims[-1], num_classes, rngs=rngs)
            if distillation and num_classes > 0
            else None
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
        x = self.head_drop(global_pool_nhwc(x, self.global_pool))
        if pre_logits or self.head is None:
            return x
        logits = self.head(x)
        if self.head_dist is not None:
            distilled = self.head_dist(x)
            if not self.deterministic and self.distilled_training:
                return logits, distilled
            logits = (logits + distilled) / 2
        return logits

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        if global_pool is not None:
            self.global_pool = global_pool
        self.num_classes = num_classes
        self.head = (
            NormLinear(self.num_features, num_classes, rngs=nnx.Rngs(0))
            if num_classes > 0
            else None
        )
        self.head_dist = (
            NormLinear(self.num_features, num_classes, rngs=nnx.Rngs(1))
            if self.distillation and num_classes > 0
            else None
        )
        if self.deterministic:
            self.eval()

    def set_distilled_training(self, enable=True):
        self.distilled_training = enable

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_CFGS = {
    "iformer_t": dict(
        dims=(32, 64, 128, 256),
        depths=(2, 2, 16, 6),
        attn_groups=(0, 0, 3, 2),
        conv_ratio=3,
        ffn_ratio=2,
    ),
    "iformer_s": dict(
        dims=(32, 64, 176, 320),
        depths=(2, 2, 19, 6),
        attn_groups=(0, 0, 3, 2),
        conv_ratio=4,
        ffn_ratio=3,
    ),
    "iformer_m": dict(
        dims=(48, 96, 192, 384),
        depths=(2, 2, 22, 6),
        attn_groups=(0, 0, 4, 2),
        conv_ratio=4,
        ffn_ratio=3,
    ),
    "iformer_l": dict(
        dims=(48, 96, 256, 384),
        depths=(2, 2, 33, 6),
        attn_groups=(0, 0, 8, 2),
        conv_ratio=4,
        ffn_ratio=3,
    ),
    "iformer_l2": dict(
        dims=(64, 128, 256, 512),
        depths=(3, 3, 46, 9),
        attn_groups=(0, 0, 11, 3),
        conv_ratio=4,
        ffn_ratio=3,
    ),
    "iformer_h": dict(
        dims=(96, 192, 384, 768),
        depths=(5, 5, 60, 18),
        attn_groups=(0, 0, 15, 6),
        attn_head_dim_reduction=(0, 0, 1, 1),
        conv_ratio=4,
        ffn_ratio=4,
        layer_scale_init_value=1e-6,
    ),
}
for _base in ("iformer_m", "iformer_l", "iformer_l2"):
    _CFGS[_base + "_distilled"] = dict(_CFGS[_base], distillation=True)


def _make(name):
    def entry(**kwargs):
        model = IFormer(**dict(_CFGS[name], **kwargs))
        model.default_cfg = _cfg(interpolation="bicubic")
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name), default_cfg=_cfg(interpolation="bicubic"))
