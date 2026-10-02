"""EfficientViT (MIT) in flax nnx, NHWC. Mirrors timm.models.efficientvit_mit (B series).

Hard-swish MBConv stages lead into stages that alternate multi-scale ReLU linear
attention (LiteMLA, computed in float32) with MBConv blocks. The head applies a
1x1 convolution, pooling, and a LayerNorm MLP before the classifier.
"""

import jax.numpy as jnp
from flax import nnx

from ..features import _select_features
from ..layers import ClassifierMixin, global_pool_nhwc, hswish
from ..registry import _cfg, register_model
from ._conv import ConvNormAct


class DSConv(nnx.Module):
    """Residual depthwise-separable convolution in the stem."""

    def __init__(self, chs, act, *, rngs):
        self.depth_conv = ConvNormAct(chs, chs, 3, groups=chs, act=act, rngs=rngs)
        self.point_conv = ConvNormAct(chs, chs, 1, rngs=rngs)

    def __call__(self, x):
        return x + self.point_conv(self.depth_conv(x))


class MBConv(nnx.Module):
    """Inverted bottleneck; attention stages use biases and a single BatchNorm."""

    def __init__(self, in_chs, out_chs, stride, expand, act, fewer_norm, residual, *, rngs):
        mid = round(in_chs * expand)
        self.residual = residual
        self.inverted_conv = ConvNormAct(
            in_chs, mid, 1, norm=not fewer_norm, act=act, use_bias=fewer_norm, rngs=rngs
        )
        self.depth_conv = ConvNormAct(
            mid,
            mid,
            3,
            stride,
            groups=mid,
            norm=not fewer_norm,
            act=act,
            use_bias=fewer_norm,
            rngs=rngs,
        )
        self.point_conv = ConvNormAct(mid, out_chs, 1, rngs=rngs)

    def __call__(self, x):
        y = self.point_conv(self.depth_conv(self.inverted_conv(x)))
        return x + y if self.residual else y


class LiteMLA(nnx.Module):
    """Lightweight multi-scale linear attention with a ReLU kernel."""

    def __init__(self, chs, head_dim, scale=5, eps=1e-5, *, rngs):
        heads = chs // head_dim
        total = heads * head_dim
        self.head_dim, self.eps = head_dim, eps
        self.qkv = ConvNormAct(chs, 3 * total, 1, norm=False, rngs=rngs)
        # Depthwise spatial aggregation, then a 1x1 convolution within each head.
        self.aggreg = nnx.List(
            [
                nnx.List(
                    [
                        nnx.Conv(
                            3 * total,
                            3 * total,
                            (scale, scale),
                            padding=((scale // 2, scale // 2),) * 2,
                            feature_group_count=3 * total,
                            use_bias=False,
                            rngs=rngs,
                        ),
                        nnx.Conv(
                            3 * total,
                            3 * total,
                            (1, 1),
                            feature_group_count=3 * heads,
                            use_bias=False,
                            rngs=rngs,
                        ),
                    ]
                )
            ]
        )
        self.proj = ConvNormAct(2 * total, chs, 1, rngs=rngs)

    def __call__(self, x):
        batch, rows, cols, _ = x.shape
        qkv = self.qkv(x)
        multi = jnp.concatenate([qkv, self.aggreg[0][1](self.aggreg[0][0](qkv))], axis=-1)
        multi = multi.reshape(batch, rows * cols, -1, 3 * self.head_dim)
        q, k, v = jnp.split(multi.astype(jnp.float32), 3, axis=-1)
        # A constant value channel yields the normalizer of the linear attention.
        v = jnp.pad(v, ((0, 0), (0, 0), (0, 0), (0, 1)), constant_values=1.0)
        kv = jnp.einsum("bnhd,bnhe->bhde", nnx.relu(k), v)
        out = jnp.einsum("bnhd,bhde->bnhe", nnx.relu(q), kv)
        out = (out[..., :-1] / (out[..., -1:] + self.eps)).astype(qkv.dtype)
        return self.proj(out.reshape(batch, rows, cols, -1))


class EfficientVitBlock(nnx.Module):
    def __init__(self, chs, head_dim, expand, act, *, rngs):
        self.context = LiteMLA(chs, head_dim, rngs=rngs)
        self.local = MBConv(chs, chs, 1, expand, act, True, True, rngs=rngs)

    def __call__(self, x):
        return self.local(x + self.context(x))


class EfficientVit(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        widths,
        depths,
        head_dim,
        head_widths,
        expand_ratio=4.0,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        act = hswish
        self.stem_conv = ConvNormAct(in_chans, widths[0], 3, 2, act=act, rngs=rngs)
        self.stem_blocks = nnx.List([DSConv(widths[0], act, rngs=rngs) for _ in range(depths[0])])
        stages, in_chs = [], widths[0]
        for i, (width, depth) in enumerate(zip(widths[1:], depths[1:])):
            vit_stage = i >= 2
            blocks = [MBConv(in_chs, width, 2, expand_ratio, act, vit_stage, False, rngs=rngs)]
            if vit_stage:
                blocks += [
                    EfficientVitBlock(width, head_dim, expand_ratio, act, rngs=rngs)
                    for _ in range(depth)
                ]
            else:
                blocks += [
                    MBConv(width, width, 1, expand_ratio, act, False, True, rngs=rngs)
                    for _ in range(depth - 1)
                ]
            stages.append(nnx.List(blocks))
            in_chs = width
        self.stages = nnx.List(stages)
        self.head_in_conv = ConvNormAct(in_chs, head_widths[0], 1, act=act, rngs=rngs)
        self.head_fc1 = nnx.Linear(head_widths[0], head_widths[1], use_bias=False, rngs=rngs)
        self.head_norm = nnx.LayerNorm(head_widths[1], epsilon=1e-5, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.num_features = head_widths[1]
        self.fc = nnx.Linear(head_widths[1], num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_intermediates(self, x, out_indices=None):
        x = self.stem_conv(x)
        for block in self.stem_blocks:
            x = block(x)
        features = []
        for stage in self.stages:
            for block in stage:
                x = block(x)
            features.append(x)
        return _select_features(features, out_indices)

    def forward_features(self, x):
        return self.forward_intermediates(x)[-1]

    def forward_head(self, x, pre_logits=False):
        x = global_pool_nhwc(self.head_in_conv(x), self.global_pool)
        x = self.head_drop(hswish(self.head_norm(self.head_fc1(x))))
        return x if pre_logits or self.fc is None else self.fc(x)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


# timm configurations: stem and stage widths, depths, attention head width, head widths.
_CFGS = {
    "efficientvit_b0": ((8, 16, 32, 64, 128), (1, 2, 2, 2, 2), 16, (1024, 1280)),
    "efficientvit_b1": ((16, 32, 64, 128, 256), (1, 2, 3, 3, 4), 16, (1536, 1600)),
    "efficientvit_b2": ((24, 48, 96, 192, 384), (1, 3, 4, 4, 6), 32, (2304, 2560)),
}


def _make(name):
    widths, depths, head_dim, head_widths = _CFGS[name]

    def entry(**kwargs):
        model = EfficientVit(widths, depths, head_dim, head_widths, **kwargs)
        model.default_cfg = _cfg(crop_pct=0.95)
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
