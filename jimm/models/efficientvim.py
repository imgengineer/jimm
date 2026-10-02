"""EfficientViM in Flax NNX, adapted from timm 1.0.30 (NHWC/NLC).

Reference: https://github.com/huggingface/pytorch-image-models/blob/v1.0.30/timm/models/efficientvim.py
Copyright (c) 2024 MLVlab (MIT); Copyright 2025 Ross Wightman.
JAX adaptation for jimm; HSM-SSD uses standard JAX operations.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from ..features import _select_features
from ..layers import DropPath, SqueezeExcite
from ..registry import _cfg, register_model
from ._conv import ConvNormAct


class HSMSSD(nnx.Module):
    """Hidden-state mixer: spatial tokens -> gated hidden states -> spatial tokens."""

    def __init__(self, dim, ssd_expand=1.0, state_dim=64, a_init_range=(1.0, 16.0), *, rngs):
        inner = int(dim * ssd_expand)
        self.state_dim = state_dim
        self.BCdt_proj = ConvNormAct(dim, 3 * state_dim, norm=False, ndim=1, rngs=rngs)
        self.dw = ConvNormAct(
            3 * state_dim, 3 * state_dim, 3, groups=3 * state_dim, norm=False, rngs=rngs
        )
        self.hz_proj = ConvNormAct(dim, 2 * inner, norm=False, ndim=1, rngs=rngs)
        self.out_proj = ConvNormAct(inner, dim, norm=False, ndim=1, rngs=rngs)
        self.A = nnx.Param(
            jax.random.uniform(
                rngs.params(),
                (state_dim,),
                minval=a_init_range[0],
                maxval=a_init_range[1],
            )
        )
        self.D = nnx.Param(jnp.ones(1))

    def __call__(self, x):
        batch, height, width, dim = x.shape
        tokens = x.reshape(batch, height * width, dim)
        bcdt = self.BCdt_proj(tokens).reshape(batch, height, width, 3 * self.state_dim)
        b, c, dt = jnp.split(self.dw(bcdt).reshape(batch, height * width, -1), 3, axis=-1)
        a = nnx.softmax(dt + self.A[...], axis=1)
        hidden = (a * b).swapaxes(1, 2) @ tokens
        h, z = jnp.split(self.hz_proj(hidden), 2, axis=-1)
        hidden = self.out_proj(h * nnx.silu(z) + h * self.D[...])
        y = c @ hidden
        return y.reshape(batch, height, width, dim), hidden


class EfficientViMBlock(nnx.Module):
    def __init__(self, dim, mlp_ratio=4.0, ssd_expand=1.0, state_dim=64, drop_path=0.0, *, rngs):
        self.mixer = HSMSSD(dim, ssd_expand, state_dim, rngs=rngs)
        self.norm = nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)
        self.dwconv1 = ConvNormAct(dim, dim, 3, groups=dim, bn_weight_init=0, rngs=rngs)
        self.dwconv2 = ConvNormAct(dim, dim, 3, groups=dim, bn_weight_init=0, rngs=rngs)
        self.ffn = nnx.Sequential(
            ConvNormAct(dim, int(dim * mlp_ratio), act=nnx.relu, rngs=rngs),
            ConvNormAct(int(dim * mlp_ratio), dim, bn_weight_init=0, rngs=rngs),
        )
        self.alpha = nnx.Param(jnp.full((4, dim), 1e-4))
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        alpha = nnx.sigmoid(self.alpha[...])
        x = x + self.drop_path(alpha[0] * (self.dwconv1(x) - x))
        y, hidden = self.mixer(self.norm(x))
        x = x + self.drop_path(alpha[1] * (y - x))
        x = x + self.drop_path(alpha[2] * (self.dwconv2(x) - x))
        x = x + self.drop_path(alpha[3] * (self.ffn(x) - x))
        return x, hidden


class PatchMerging(nnx.Module):
    def __init__(self, in_dim, out_dim, ratio=4.0, *, rngs):
        hidden = int(out_dim * ratio)
        self.dwconv1 = ConvNormAct(in_dim, in_dim, 3, groups=in_dim, rngs=rngs)
        self.conv = nnx.Sequential(
            ConvNormAct(in_dim, hidden, act=nnx.relu, rngs=rngs),
            ConvNormAct(hidden, hidden, 3, 2, groups=hidden, act=nnx.relu, rngs=rngs),
            SqueezeExcite(hidden, 0.25, rngs=rngs),
            ConvNormAct(hidden, out_dim, rngs=rngs),
        )
        self.dwconv2 = ConvNormAct(out_dim, out_dim, 3, groups=out_dim, rngs=rngs)

    def __call__(self, x):
        x = self.conv(x + self.dwconv1(x))
        return x + self.dwconv2(x)


class EfficientViMStage(nnx.Module):
    def __init__(self, dim, out_dim, depth, mlp_ratio, ssd_expand, state_dim, rates, *, rngs):
        if depth < 1:
            raise ValueError("EfficientViM stages require at least one block")
        self.blocks = nnx.List(
            [
                EfficientViMBlock(dim, mlp_ratio, ssd_expand, state_dim, rate, rngs=rngs)
                for rate in rates
            ]
        )
        self.downsample = PatchMerging(dim, out_dim, rngs=rngs) if out_dim else None

    def __call__(self, x):
        for block in self.blocks:
            x, hidden = block(x)
        next_input = self.downsample(x) if self.downsample is not None else x
        return x, next_input, hidden


class EfficientViM(nnx.Module):
    """Three hidden-state heads and a final-map head with learned softmax fusion."""

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        embed_dim=(128, 256, 512),
        depths=(2, 2, 2),
        mlp_ratio=4.0,
        ssd_expand=1.0,
        state_dim=(49, 25, 9),
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.0,
        distillation=False,
        *,
        rngs,
    ):
        if not len(embed_dim) == len(depths) == len(state_dim):
            raise ValueError("stage configurations must have equal lengths")
        if global_pool not in ("avg", "") or (num_classes > 0 and not global_pool):
            raise ValueError("classification requires global_pool='avg'")
        self.num_classes, self.global_pool = num_classes, global_pool
        self.embed_dim = tuple(embed_dim)
        self.num_features = sum(embed_dim) + embed_dim[-1]
        self.distillation = distillation
        self.distilled_training = False
        self.deterministic = False
        dim = embed_dim[0]
        self.patch_embed = nnx.Sequential(
            ConvNormAct(in_chans, dim // 8, 3, 2, act=nnx.relu, rngs=rngs),
            ConvNormAct(dim // 8, dim // 4, 3, 2, act=nnx.relu, rngs=rngs),
            ConvNormAct(dim // 4, dim // 2, 3, 2, act=nnx.relu, rngs=rngs),
            ConvNormAct(dim // 2, dim, 3, 2, rngs=rngs),
        )
        stages, offset = [], 0
        for i, (dim, depth, state) in enumerate(zip(embed_dim, depths, state_dim)):
            rates = [drop_path_rate * (offset + j) / max(sum(depths) - 1, 1) for j in range(depth)]
            out_dim = embed_dim[i + 1] if i + 1 < len(depths) else 0
            stages.append(
                EfficientViMStage(
                    dim,
                    out_dim,
                    depth,
                    mlp_ratio,
                    ssd_expand,
                    state,
                    rates,
                    rngs=rngs,
                )
            )
            offset += depth
        self.stages = nnx.List(stages)
        self.norm = nnx.List(
            [nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs) for dim in (*embed_dim, embed_dim[-1])]
        )
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.weights = nnx.Param(jnp.ones(len(depths) + 1))
        self.heads = self._build_heads(num_classes, rngs)
        self.weights_dist = nnx.Param(jnp.ones(len(depths) + 1)) if distillation else None
        self.heads_dist = self._build_heads(num_classes, rngs) if distillation else None

    def _build_heads(self, num_classes, rngs):
        return (
            nnx.List(
                [
                    nnx.Linear(dim, num_classes, rngs=rngs)
                    for dim in (*self.embed_dim, self.embed_dim[-1])
                ]
            )
            if num_classes > 0
            else None
        )

    def forward_intermediates(self, x, out_indices=None):
        x, features = self.patch_embed(x), []
        for stage in self.stages:
            feature, x, _ = stage(x)
            features.append(feature)
        return _select_features(features, out_indices)

    def forward_features(self, x):
        x, features = self.patch_embed(x), []
        for i, stage in enumerate(self.stages):
            _, x, hidden = stage(x)
            features.append(self.norm[i](hidden))
        features.append(self.norm[-1](x).reshape(x.shape[0], -1, x.shape[-1]))
        return features

    def forward_head(self, x, pre_logits=False):
        if not self.global_pool:
            return x
        pooled = [self.head_drop(jnp.mean(feature, axis=1)) for feature in x]
        if pre_logits or self.heads is None:
            return jnp.concatenate(pooled, axis=-1)
        weights = nnx.softmax(self.weights[...])
        logits = sum(weights[i] * head(pooled[i]) for i, head in enumerate(self.heads))
        if self.heads_dist is not None:
            weights_dist = nnx.softmax(self.weights_dist[...])
            distilled = sum(
                weights_dist[i] * head(pooled[i]) for i, head in enumerate(self.heads_dist)
            )
            if not self.deterministic and self.distilled_training:
                return logits, distilled
            logits = (logits + distilled) / 2
        return logits

    def get_classifier(self):
        return self.heads

    def reset_classifier(self, num_classes, global_pool=None):
        pool = self.global_pool if global_pool is None else global_pool
        if pool not in ("avg", "") or (num_classes > 0 and not pool):
            raise ValueError("classification requires global_pool='avg'")
        self.num_classes, self.global_pool = num_classes, pool
        self.heads = self._build_heads(num_classes, nnx.Rngs(0))
        self.heads_dist = self._build_heads(num_classes, nnx.Rngs(1)) if self.distillation else None

    def set_distilled_training(self, enable=True):
        self.distilled_training = enable

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_CFGS = {
    "efficientvim_m1": dict(embed_dim=(128, 192, 320), depths=(2, 2, 2), state_dim=(49, 25, 9)),
    "efficientvim_m2": dict(embed_dim=(128, 256, 512), depths=(2, 2, 2), state_dim=(49, 25, 9)),
    "efficientvim_m3": dict(embed_dim=(224, 320, 512), depths=(2, 2, 2), state_dim=(49, 25, 9)),
    "efficientvim_m4": dict(embed_dim=(224, 320, 512), depths=(3, 4, 2), state_dim=(64, 32, 16)),
}
for _base in tuple(_CFGS):
    _CFGS[_base + "_dist"] = dict(_CFGS[_base], distillation=True)


def _default_cfg(name):
    size = 256 if name.startswith("efficientvim_m4") else 224
    return _cfg(input_size=(3, size, size), interpolation="bicubic")


def _make(name):
    def entry(**kwargs):
        model = EfficientViM(**dict(_CFGS[name], **kwargs))
        model.default_cfg = _default_cfg(name)
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name), default_cfg=_default_cfg(_name))
