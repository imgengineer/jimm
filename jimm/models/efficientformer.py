"""EfficientFormer (v1) in flax nnx, NHWC. Mirrors timm.models.efficientformer.

A two-conv ReLU stem feeds four stages of PoolFormer-style blocks: average
pooling minus identity, then a 1x1-conv MLP with BatchNorm, both scaled by
layer scale. The last ``num_vit`` blocks of the final stage run on flattened
tokens as pre-norm transformer blocks whose attention adds learned relative
position biases over the fixed 7x7 grid. The head averages a class and a
distillation classifier applied to the mean token.
"""

import jax.numpy as jnp
from flax import nnx

from ..attention import dot_product_attention
from ..layers import BatchNorm, ClassifierMixin, DropPath, Mlp, gelu
from ..registry import _cfg, register_model
from .levit import _bias_index

_init = nnx.initializers.truncated_normal(0.02)


def _conv(in_chs, out_chs, kernel=1, stride=1, *, rngs):
    pad = kernel // 2
    return nnx.Conv(
        in_chs,
        out_chs,
        (kernel, kernel),
        strides=(stride, stride),
        padding=((pad, pad), (pad, pad)),
        rngs=rngs,
    )


class Attention(nnx.Module):
    def __init__(self, dim, key_dim=32, num_heads=8, attn_ratio=4, resolution=7, *, rngs):
        self.num_heads, self.key_dim = num_heads, key_dim
        self.resolution = (resolution, resolution)
        val_dim = int(attn_ratio * key_dim)
        # Channels are grouped per head as [query, key, value], as in timm.
        self.qkv = nnx.Linear(
            dim, num_heads * (2 * key_dim + val_dim), kernel_init=_init, rngs=rngs
        )
        self.proj = nnx.Linear(num_heads * val_dim, dim, kernel_init=_init, rngs=rngs)
        self.attention_biases = nnx.Param(jnp.zeros((num_heads, resolution * resolution)))

    def __call__(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, self.num_heads, -1)
        q, k, v = jnp.split(qkv, [self.key_dim, 2 * self.key_dim], axis=-1)
        bias = self.attention_biases[...][:, _bias_index(self.resolution)]
        x = dot_product_attention(q, k, v, bias=bias[None])
        return self.proj(x.reshape(B, N, -1))


class Stem4(nnx.Module):
    def __init__(self, in_chs, out_chs, *, rngs):
        self.conv1 = _conv(in_chs, out_chs // 2, 3, 2, rngs=rngs)
        self.norm1 = BatchNorm(out_chs // 2, epsilon=1e-5, rngs=rngs)
        self.conv2 = _conv(out_chs // 2, out_chs, 3, 2, rngs=rngs)
        self.norm2 = BatchNorm(out_chs, epsilon=1e-5, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.norm1(self.conv1(x)))
        return nnx.relu(self.norm2(self.conv2(x)))


class Downsample(nnx.Module):
    def __init__(self, in_chs, out_chs, *, rngs):
        self.conv = _conv(in_chs, out_chs, 3, 2, rngs=rngs)
        self.norm = BatchNorm(out_chs, epsilon=1e-5, rngs=rngs)

    def __call__(self, x):
        return self.norm(self.conv(x))


class ConvMlpWithNorm(nnx.Module):
    def __init__(self, dim, hidden, drop=0.0, *, rngs):
        self.fc1 = _conv(dim, hidden, rngs=rngs)
        self.norm1 = BatchNorm(hidden, epsilon=1e-5, rngs=rngs)
        self.fc2 = _conv(hidden, dim, rngs=rngs)
        self.norm2 = BatchNorm(dim, epsilon=1e-5, rngs=rngs)
        self.drop = nnx.Dropout(drop, rngs=rngs)

    def __call__(self, x):
        x = self.drop(gelu(self.norm1(self.fc1(x))))
        return self.drop(self.norm2(self.fc2(x)))


class MetaBlock1d(nnx.Module):
    """Pre-norm transformer block on flattened tokens."""

    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, drop_path=0.0, ls_init=1e-5, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)
        self.token_mixer = Attention(dim, rngs=rngs)
        self.ls1 = nnx.Param(jnp.full((dim,), ls_init))
        self.norm2 = nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop, kernel_init=_init, rngs=rngs)
        self.ls2 = nnx.Param(jnp.full((dim,), ls_init))
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        x = x + self.drop_path(self.ls1[...] * self.token_mixer(self.norm1(x)))
        return x + self.drop_path(self.ls2[...] * self.mlp(self.norm2(x)))


class MetaBlock2d(nnx.Module):
    """PoolFormer block: pooling minus identity, then a BatchNorm conv MLP."""

    def __init__(
        self, dim, pool_size=3, mlp_ratio=4.0, drop=0.0, drop_path=0.0, ls_init=1e-5, *, rngs
    ):
        self.pool_size = pool_size
        self.ls1 = nnx.Param(jnp.full((dim,), ls_init))
        self.mlp = ConvMlpWithNorm(dim, int(dim * mlp_ratio), drop, rngs=rngs)
        self.ls2 = nnx.Param(jnp.full((dim,), ls_init))
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        p = self.pool_size // 2
        pooled = nnx.avg_pool(
            x,
            (self.pool_size, self.pool_size),
            strides=(1, 1),
            padding=((p, p), (p, p)),
            count_include_pad=False,
        )
        x = x + self.drop_path(self.ls1[...] * (pooled - x))
        return x + self.drop_path(self.ls2[...] * self.mlp(x))


class EfficientFormerStage(nnx.Module):
    def __init__(
        self,
        dim,
        dim_out,
        depth,
        downsample=True,
        num_vit=0,
        pool_size=3,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=None,
        ls_init=1e-5,
        *,
        rngs,
    ):
        self.downsample = Downsample(dim, dim_out, rngs=rngs) if downsample else None
        drop_path = drop_path or [0.0] * depth
        # The last ``num_vit`` blocks are transformer blocks on flattened tokens.
        self.blocks = nnx.List(
            [
                MetaBlock1d(dim_out, mlp_ratio, drop, drop_path[j], ls_init, rngs=rngs)
                if j >= depth - num_vit
                else MetaBlock2d(
                    dim_out, pool_size, mlp_ratio, drop, drop_path[j], ls_init, rngs=rngs
                )
                for j in range(depth)
            ]
        )

    def __call__(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        for blk in self.blocks:
            if isinstance(blk, MetaBlock1d) and x.ndim == 4:
                x = x.reshape(x.shape[0], -1, x.shape[-1])
            x = blk(x)
        return x


class EfficientFormer(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"

    def __init__(
        self,
        depths=(3, 2, 6, 4),
        embed_dims=(48, 96, 224, 448),
        num_vit=0,
        mlp_ratio=4.0,
        pool_size=3,
        layer_scale_init_value=1e-5,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        proj_drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.stem = Stem4(in_chans, embed_dims[0], rngs=rngs)
        # timm calculate_drop_path_rates(stagewise=True): linear over all blocks, split by stage.
        total = sum(depths)
        rates = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]
        stages, prev = [], embed_dims[0]
        for i, (dim, depth) in enumerate(zip(embed_dims, depths)):
            stages.append(
                EfficientFormerStage(
                    prev,
                    dim,
                    depth,
                    downsample=i > 0,
                    num_vit=num_vit if i == len(depths) - 1 else 0,
                    pool_size=pool_size,
                    mlp_ratio=mlp_ratio,
                    drop=proj_drop_rate,
                    drop_path=rates[sum(depths[:i]) : sum(depths[: i + 1])],
                    ls_init=layer_scale_init_value,
                    rngs=rngs,
                )
            )
            prev = dim
        self.stages = nnx.List(stages)
        self.num_features = prev
        self.norm = nnx.LayerNorm(prev, epsilon=1e-5, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = self._linear(num_classes, rngs)
        self.head_dist = self._linear(num_classes, rngs)

    def _linear(self, num_classes, rngs):
        if num_classes <= 0:
            return None
        return nnx.Linear(self.num_features, num_classes, kernel_init=_init, rngs=rngs)

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.global_pool = global_pool if global_pool is not None else self.global_pool
        rngs = nnx.Rngs(0)
        self.head = self._linear(num_classes, rngs)
        self.head_dist = self._linear(num_classes, rngs)

    def forward_features(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        if x.ndim == 4:  # no transformer blocks: flatten for the token LayerNorm
            x = x.reshape(x.shape[0], -1, x.shape[-1])
        return self.norm(x)

    def forward_head(self, x):
        if self.global_pool == "avg":
            x = jnp.mean(x, axis=1)
        x = self.head_drop(x)
        if self.head is None:
            return x
        # timm averages the class and distillation heads outside distillation training.
        return (self.head(x) + self.head_dist(x)) / 2

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_CFGS = {  # depths, embed_dims, num_vit
    "efficientformer_l1": ((3, 2, 6, 4), (48, 96, 224, 448), 1),
    "efficientformer_l3": ((4, 4, 12, 6), (64, 128, 320, 512), 4),
    "efficientformer_l7": ((6, 6, 18, 8), (96, 192, 384, 768), 8),
}


def _make(name):
    depths, embed_dims, num_vit = _CFGS[name]

    def entry(**kwargs):
        model = EfficientFormer(depths, embed_dims, num_vit, **kwargs)
        model.default_cfg = _cfg(crop_pct=0.95, interpolation="bicubic", fixed_input_size=True)
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
