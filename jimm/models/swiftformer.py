"""SwiftFormer in flax nnx, NHWC. Mirrors timm.models.swiftformer.

Stages of convolutional encoders (depthwise 3x3 conv + BatchNorm, 1x1-conv
MLP, layer scale) end with a SwiftFormer block: a local representation
module, efficient additive attention (L2-normalized queries and keys pooled
into one global query by a learned vector instead of a token-by-token
attention map), and a BatchNorm 1x1-conv MLP. A two-conv stem and strided
3x3 convolutions downsample; the head averages classifier and distillation
outputs on the pooled, normalized features.
"""

import jax.numpy as jnp
from flax import nnx

from ..layers import BatchNorm, ClassifierMixin, DropPath, gelu, global_pool_nhwc
from ..registry import _cfg, register_model

_init = nnx.initializers.truncated_normal(0.02)


def _conv(in_chs, out_chs, kernel=1, stride=1, groups=1, *, rngs):
    pad = kernel // 2
    return nnx.Conv(
        in_chs,
        out_chs,
        (kernel, kernel),
        strides=(stride, stride),
        padding=((pad, pad), (pad, pad)),
        feature_group_count=groups,
        kernel_init=_init,
        rngs=rngs,
    )


def _l2_normalize(x, axis):
    return x / jnp.maximum(jnp.linalg.norm(x, axis=axis, keepdims=True), 1e-12)


class ConvEncoder(nnx.Module):
    """Depthwise 3x3 conv, BatchNorm, 1x1-conv MLP, and layer scale, with a residual.

    timm's LocalRepresentation is the same module with a hidden width equal to ``dim``.
    """

    def __init__(self, dim, hidden_dim, drop_path=0.0, ls_init=1.0, *, rngs):
        self.dwconv = _conv(dim, dim, 3, groups=dim, rngs=rngs)
        self.norm = BatchNorm(dim, epsilon=1e-5, rngs=rngs)
        self.pwconv1 = _conv(dim, hidden_dim, rngs=rngs)
        self.pwconv2 = _conv(hidden_dim, dim, rngs=rngs)
        self.layer_scale = nnx.Param(jnp.full((dim,), ls_init)) if ls_init is not None else None
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        y = self.pwconv2(gelu(self.pwconv1(self.norm(self.dwconv(x)))))
        if self.layer_scale is not None:
            y = self.layer_scale[...] * y
        return x + self.drop_path(y)


class Mlp(nnx.Module):
    def __init__(self, dim, hidden, drop=0.0, *, rngs):
        self.norm1 = BatchNorm(dim, epsilon=1e-5, rngs=rngs)
        self.fc1 = _conv(dim, hidden, rngs=rngs)
        self.fc2 = _conv(hidden, dim, rngs=rngs)
        self.drop = nnx.Dropout(drop, rngs=rngs)

    def __call__(self, x):
        x = self.drop(gelu(self.fc1(self.norm1(x))))
        return self.drop(self.fc2(x))


class EfficientAdditiveAttention(nnx.Module):
    def __init__(self, dim, *, rngs):
        self.scale = dim**-0.5
        self.to_query = nnx.Linear(dim, dim, kernel_init=_init, rngs=rngs)
        self.to_key = nnx.Linear(dim, dim, kernel_init=_init, rngs=rngs)
        self.w_g = nnx.Param(nnx.initializers.normal(1.0)(rngs.params(), (dim, 1)))
        self.proj = nnx.Linear(dim, dim, kernel_init=_init, rngs=rngs)
        self.final = nnx.Linear(dim, dim, kernel_init=_init, rngs=rngs)

    def __call__(self, x):
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        query = _l2_normalize(self.to_query(x), axis=-1)
        key = _l2_normalize(self.to_key(x), axis=-1)
        # One global query: tokens weighted by a learned vector, normalized over the tokens.
        attn = _l2_normalize(query @ self.w_g[...] * self.scale, axis=1)
        global_query = jnp.sum(attn * query, axis=1, keepdims=True)
        out = self.final(self.proj(global_query * key) + query)
        return out.reshape(B, H, W, C)


class SwiftFormerBlock(nnx.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, drop_path=0.0, ls_init=1e-5, *, rngs):
        self.local_representation = ConvEncoder(dim, dim, rngs=rngs)
        self.attn = EfficientAdditiveAttention(dim, rngs=rngs)
        self.linear = Mlp(dim, int(dim * mlp_ratio), drop, rngs=rngs)
        self.layer_scale_1 = nnx.Param(jnp.full((dim,), ls_init))
        self.layer_scale_2 = nnx.Param(jnp.full((dim,), ls_init))
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        x = self.local_representation(x)
        x = x + self.drop_path(self.layer_scale_1[...] * self.attn(x))
        return x + self.drop_path(self.layer_scale_2[...] * self.linear(x))


class SwiftFormerStage(nnx.Module):
    def __init__(
        self, in_dim, dim, depth, downsample, mlp_ratio, drop, drop_paths, ls_init, *, rngs
    ):
        self.downsample = (
            nnx.List([_conv(in_dim, dim, 3, 2, rngs=rngs), BatchNorm(dim, epsilon=1e-5, rngs=rngs)])
            if downsample
            else None
        )
        # Convolutional encoders, then one SwiftFormer block at the end of the stage.
        blocks = [
            ConvEncoder(dim, int(mlp_ratio * dim), drop_paths[i], rngs=rngs)
            for i in range(depth - 1)
        ]
        blocks.append(SwiftFormerBlock(dim, mlp_ratio, drop, drop_paths[-1], ls_init, rngs=rngs))
        self.blocks = nnx.List(blocks)

    def __call__(self, x):
        if self.downsample is not None:
            for layer in self.downsample:
                x = layer(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class SwiftFormer(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"

    def __init__(
        self,
        layers=(3, 3, 6, 4),
        embed_dims=(48, 56, 112, 220),
        mlp_ratio=4.0,
        downsamples=(False, True, True, True),
        layer_scale_init_value=1e-5,
        distillation=True,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        half = embed_dims[0] // 2
        self.stem = nnx.List(
            [
                _conv(in_chans, half, 3, 2, rngs=rngs),
                BatchNorm(half, epsilon=1e-5, rngs=rngs),
                _conv(half, embed_dims[0], 3, 2, rngs=rngs),
                BatchNorm(embed_dims[0], epsilon=1e-5, rngs=rngs),
            ]
        )
        total = sum(layers)
        stages, prev = [], embed_dims[0]
        for i, (depth, dim) in enumerate(zip(layers, embed_dims)):
            start = sum(layers[:i])
            dpr = [drop_path_rate * (start + j) / max(total - 1, 1) for j in range(depth)]
            stages.append(
                SwiftFormerStage(
                    prev,
                    dim,
                    depth,
                    downsamples[i],
                    mlp_ratio,
                    drop_rate,
                    dpr,
                    layer_scale_init_value,
                    rngs=rngs,
                )
            )
            prev = dim
        self.stages = nnx.List(stages)
        self.num_features = prev
        self.norm = BatchNorm(prev, epsilon=1e-5, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.distillation = distillation
        self.head = self._linear(num_classes, rngs)
        self.head_dist = self._linear(num_classes, rngs) if distillation else None

    def _linear(self, num_classes, rngs):
        if num_classes <= 0:
            return None
        return nnx.Linear(self.num_features, num_classes, kernel_init=_init, rngs=rngs)

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.global_pool = global_pool if global_pool is not None else self.global_pool
        rngs = nnx.Rngs(0)
        self.head = self._linear(num_classes, rngs)
        self.head_dist = self._linear(num_classes, rngs) if self.distillation else None

    def forward_features(self, x):
        for i, layer in enumerate(self.stem):
            x = layer(x)
            if i % 2:  # ReLU after each stem BatchNorm
                x = nnx.relu(x)
        for stage in self.stages:
            x = stage(x)
        return self.norm(x)

    def forward_head(self, x):
        x = self.head_drop(global_pool_nhwc(x, self.global_pool))
        if self.head is None:
            return x
        if self.head_dist is None:
            return self.head(x)
        # timm averages the class and distillation heads outside distillation training.
        return (self.head(x) + self.head_dist(x)) / 2

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_CFGS = {
    "swiftformer_xs": ((3, 3, 6, 4), (48, 56, 112, 220)),
    "swiftformer_s": ((3, 3, 9, 6), (48, 64, 168, 224)),
    "swiftformer_l1": ((4, 3, 10, 5), (48, 96, 192, 384)),
    "swiftformer_l3": ((4, 4, 12, 6), (64, 128, 320, 512)),
}


def _make(name):
    layers, embed_dims = _CFGS[name]

    def entry(**kwargs):
        model = SwiftFormer(layers, embed_dims, **kwargs)
        model.default_cfg = _cfg(crop_pct=0.95, interpolation="bicubic", fixed_input_size=True)
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
