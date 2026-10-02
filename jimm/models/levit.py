"""LeViT in flax nnx, NHWC input. Mirrors timm.models.levit (linear token mode).

Every linear projection is followed by BatchNorm, residual branches start at zero,
attention adds learned relative-position biases, and stages downsample with
attention over a subsampled query grid. Models average two distilled heads.
"""

from functools import lru_cache

import jax.numpy as jnp
import numpy as np
from flax import nnx

from ..attention import dot_product_attention
from ..features import _select_features
from ..layers import BatchNorm, DropPath, hswish
from ..registry import _cfg, register_model
from ._conv import ConvNormAct


@lru_cache(maxsize=None)
def _bias_index(resolution, stride=1):
    """Index ``(query, key)`` pairs by ``|dy| * width + |dx|`` on the token grid."""
    rows, cols = resolution
    keys = np.stack(np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")).reshape(2, -1)
    queries = np.stack(
        np.meshgrid(np.arange(0, rows, stride), np.arange(0, cols, stride), indexing="ij")
    ).reshape(2, -1)
    offsets = np.abs(queries[:, :, None] - keys[:, None, :])
    return offsets[0] * cols + offsets[1]


class LinearNorm(nnx.Module):
    """Bias-free linear projection followed by BatchNorm over all tokens."""

    def __init__(self, in_dim, out_dim, bn_weight_init=1.0, *, rngs):
        self.linear = nnx.Linear(in_dim, out_dim, use_bias=False, rngs=rngs)
        self.bn = BatchNorm(
            out_dim, scale_init=nnx.initializers.constant(bn_weight_init), rngs=rngs
        )

    def __call__(self, x):
        return self.bn(self.linear(x))


class NormLinear(nnx.Module):
    """BatchNorm, dropout, then the classifier projection."""

    def __init__(self, dim, num_classes, drop=0.0, *, rngs):
        self.bn = BatchNorm(dim, rngs=rngs)
        self.drop = nnx.Dropout(drop, rngs=rngs)
        self.linear = nnx.Linear(
            dim,
            num_classes,
            kernel_init=nnx.initializers.truncated_normal(0.02),
            rngs=rngs,
        )

    def __call__(self, x):
        return self.linear(self.drop(self.bn(x)))


class Attention(nnx.Module):
    def __init__(self, dim, key_dim, num_heads, attn_ratio, resolution, act, *, rngs):
        self.num_heads, self.key_dim = num_heads, key_dim
        self.val_dim = int(attn_ratio * key_dim)
        self.resolution = resolution
        self.act = act
        # Channels are grouped per head as [query, key, value], as in timm.
        self.qkv = LinearNorm(dim, num_heads * (2 * key_dim + self.val_dim), rngs=rngs)
        self.proj = LinearNorm(num_heads * self.val_dim, dim, bn_weight_init=0.0, rngs=rngs)
        self.attention_biases = nnx.Param(jnp.zeros((num_heads, resolution[0] * resolution[1])))

    def __call__(self, x):
        batch, tokens, _ = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, self.num_heads, -1)
        q, k, v = jnp.split(qkv, [self.key_dim, 2 * self.key_dim], axis=-1)
        bias = self.attention_biases[...][:, _bias_index(self.resolution)]
        x = dot_product_attention(q, k, v, bias=bias[None])
        return self.proj(self.act(x.reshape(batch, tokens, -1)))


class AttentionDownsample(nnx.Module):
    """Attention from a stride-two subsampled query grid to every key."""

    def __init__(self, in_dim, out_dim, key_dim, num_heads, attn_ratio, resolution, act, *, rngs):
        self.num_heads, self.key_dim = num_heads, key_dim
        self.val_dim = int(attn_ratio * key_dim)
        self.resolution = resolution
        self.act = act
        self.kv = LinearNorm(in_dim, num_heads * (key_dim + self.val_dim), rngs=rngs)
        self.q = LinearNorm(in_dim, num_heads * key_dim, rngs=rngs)
        self.proj = LinearNorm(num_heads * self.val_dim, out_dim, rngs=rngs)
        self.attention_biases = nnx.Param(jnp.zeros((num_heads, resolution[0] * resolution[1])))

    def __call__(self, x):
        batch, tokens, chs = x.shape
        kv = self.kv(x).reshape(batch, tokens, self.num_heads, -1)
        k, v = jnp.split(kv, [self.key_dim], axis=-1)
        rows, cols = self.resolution
        sub = x.reshape(batch, rows, cols, chs)[:, ::2, ::2].reshape(batch, -1, chs)
        q = self.q(sub).reshape(batch, sub.shape[1], self.num_heads, self.key_dim)
        bias = self.attention_biases[...][:, _bias_index(self.resolution, 2)]
        x = dot_product_attention(q, k, v, bias=bias[None])
        return self.proj(self.act(x.reshape(batch, sub.shape[1], -1)))


class LevitMlp(nnx.Module):
    def __init__(self, dim, hidden, act, drop=0.0, *, rngs):
        self.ln1 = LinearNorm(dim, hidden, rngs=rngs)
        self.act = act
        self.drop = nnx.Dropout(drop, rngs=rngs)
        self.ln2 = LinearNorm(hidden, dim, bn_weight_init=0.0, rngs=rngs)

    def __call__(self, x):
        return self.ln2(self.drop(self.act(self.ln1(x))))


class LevitBlock(nnx.Module):
    def __init__(
        self, dim, key_dim, num_heads, attn_ratio, mlp_ratio, resolution, act, drop_path, *, rngs
    ):
        self.attn = Attention(dim, key_dim, num_heads, attn_ratio, resolution, act, rngs=rngs)
        self.drop_path1 = DropPath(drop_path, rngs=rngs)
        self.mlp = LevitMlp(dim, int(dim * mlp_ratio), act, rngs=rngs)
        self.drop_path2 = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        x = x + self.drop_path1(self.attn(x))
        return x + self.drop_path2(self.mlp(x))


class LevitDownsample(nnx.Module):
    def __init__(self, in_dim, out_dim, key_dim, resolution, act, drop_path, *, rngs):
        self.attn_downsample = AttentionDownsample(
            in_dim, out_dim, key_dim, in_dim // key_dim, 4.0, resolution, act, rngs=rngs
        )
        self.mlp = LevitMlp(out_dim, out_dim * 2, act, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        x = self.attn_downsample(x)
        return x + self.drop_path(self.mlp(x))


class LevitStage(nnx.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        key_dim,
        depth,
        num_heads,
        attn_ratio,
        mlp_ratio,
        resolution,
        downsample,
        act,
        drop_path,
        *,
        rngs,
    ):
        self.downsample = (
            LevitDownsample(in_dim, out_dim, key_dim, resolution, act, drop_path, rngs=rngs)
            if downsample
            else None
        )
        if downsample:
            resolution = tuple((r - 1) // 2 + 1 for r in resolution)
        self.resolution = resolution
        self.blocks = nnx.List(
            [
                LevitBlock(
                    out_dim,
                    key_dim,
                    num_heads,
                    attn_ratio,
                    mlp_ratio,
                    resolution,
                    act,
                    drop_path,
                    rngs=rngs,
                )
                for _ in range(depth)
            ]
        )

    def __call__(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x


class Levit(nnx.Module):
    def __init__(
        self,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        embed_dim=(128, 256, 384),
        key_dim=16,
        depth=(2, 3, 4),
        num_heads=(4, 6, 8),
        attn_ratio=2.0,
        mlp_ratio=2.0,
        act=hswish,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.0,
        distillation=True,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dim[-1]
        self.drop_rate = drop_rate
        self.distillation = distillation
        self.distilled_training = False
        self.deterministic = False
        e = embed_dim[0]
        self.stem = nnx.List(
            [
                ConvNormAct(in_chans, e // 8, 3, 2, act=act, rngs=rngs),
                ConvNormAct(e // 8, e // 4, 3, 2, act=act, rngs=rngs),
                ConvNormAct(e // 4, e // 2, 3, 2, act=act, rngs=rngs),
                ConvNormAct(e // 2, e, 3, 2, rngs=rngs),
            ]
        )
        resolution = (img_size // 16, img_size // 16)
        stages, in_dim = [], e
        for i, (dim, d, heads) in enumerate(zip(embed_dim, depth, num_heads)):
            stage = LevitStage(
                in_dim,
                dim,
                key_dim,
                d,
                heads,
                attn_ratio,
                mlp_ratio,
                resolution,
                i > 0,
                act,
                drop_path_rate,
                rngs=rngs,
            )
            stages.append(stage)
            resolution, in_dim = stage.resolution, dim
        self.stages = nnx.List(stages)
        self.head = (
            NormLinear(self.num_features, num_classes, drop_rate, rngs=rngs)
            if num_classes > 0
            else None
        )
        self.head_dist = (
            NormLinear(self.num_features, num_classes, rngs=rngs)
            if distillation and num_classes > 0
            else None
        )

    def forward_intermediates(self, x, out_indices=None):
        for layer in self.stem:
            x = layer(x)
        batch, _, _, chs = x.shape
        x = x.reshape(batch, -1, chs)
        features = []
        for stage in self.stages:
            x = stage(x)
            rows, cols = stage.resolution
            features.append(x.reshape(batch, rows, cols, -1))
        return _select_features(features, out_indices)

    def forward_features(self, x):
        return self.forward_intermediates(x)[-1]

    def forward_head(self, x, pre_logits=False):
        x = jnp.mean(x, axis=(1, 2)) if self.global_pool == "avg" else x
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
            NormLinear(self.num_features, num_classes, self.drop_rate, rngs=nnx.Rngs(0))
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


# timm configurations: embed widths, key width, heads, and depth per stage.
_CFGS = {
    "levit_128s": dict(embed_dim=(128, 256, 384), key_dim=16, num_heads=(4, 6, 8), depth=(2, 3, 4)),
    "levit_192": dict(embed_dim=(192, 288, 384), key_dim=32, num_heads=(3, 5, 6), depth=(4, 4, 4)),
    "levit_256": dict(embed_dim=(256, 384, 512), key_dim=32, num_heads=(4, 6, 8), depth=(4, 4, 4)),
}


def _make(name):
    def entry(**kwargs):
        model = Levit(**_CFGS[name], **kwargs)
        model.default_cfg = _cfg(crop_pct=0.9, interpolation="bicubic")
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
