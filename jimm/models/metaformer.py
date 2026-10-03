"""MetaFormer baselines (CAFormer, ConvFormer, PoolFormerV2) in flax nnx, NHWC.
Mirrors timm.models.metaformer; PoolFormer v1 lives in ``poolformer.py``.

Every block is ``res_scale * x + token_mixer(norm(x))`` followed by
``res_scale * x + mlp(norm(x))``, with bias-free norms and StarReLU MLPs. The
token mixer is average pooling minus identity (PoolFormerV2), an inverted
separable convolution with StarReLU and a 7x7 depthwise convolution
(ConvFormer), or that convolution in the first two stages and self-attention
in the last two (CAFormer). A 7x7 stride-4 stem and normalized 3x3 strided
convolutions downsample; the head pools, normalizes, and classifies with a
squared-ReLU MLP (or a linear layer for PoolFormerV2).
"""

import jax.numpy as jnp
from flax import nnx

from ..attention import dot_product_attention
from ..layers import ClassifierMixin, DropPath, global_pool_nhwc
from ..registry import _cfg, register_model

_init = nnx.initializers.truncated_normal(0.02)


def _norm(kind, dim, *, rngs):
    """Bias-free channel norm: LayerNorm or single-group GroupNorm, as in timm."""
    if kind == "group":
        return nnx.GroupNorm(dim, num_groups=1, epsilon=1e-6, use_bias=False, rngs=rngs)
    return nnx.LayerNorm(dim, epsilon=1e-6, use_bias=False, rngs=rngs)


def _conv(in_chs, out_chs, kernel, stride=1, pad=0, groups=1, use_bias=True, *, rngs):
    return nnx.Conv(
        in_chs,
        out_chs,
        (kernel, kernel),
        strides=(stride, stride),
        padding=((pad, pad), (pad, pad)),
        feature_group_count=groups,
        use_bias=use_bias,
        kernel_init=_init,
        rngs=rngs,
    )


class StarReLU(nnx.Module):
    """``scale * relu(x) ** 2 + bias`` with learnable scalars."""

    def __init__(self):
        self.scale = nnx.Param(jnp.ones((1,)))
        self.bias = nnx.Param(jnp.zeros((1,)))

    def __call__(self, x):
        return self.scale[...] * nnx.relu(x) ** 2 + self.bias[...]


class SepConv(nnx.Module):
    def __init__(self, dim, expansion_ratio=2, kernel=7, *, rngs):
        mid = int(expansion_ratio * dim)
        self.pwconv1 = _conv(dim, mid, 1, use_bias=False, rngs=rngs)
        self.act1 = StarReLU()
        self.dwconv = _conv(
            mid, mid, kernel, pad=kernel // 2, groups=mid, use_bias=False, rngs=rngs
        )
        self.pwconv2 = _conv(mid, dim, 1, use_bias=False, rngs=rngs)

    def __call__(self, x):
        return self.pwconv2(self.dwconv(self.act1(self.pwconv1(x))))


class Pooling(nnx.Module):
    def __init__(self, dim, pool_size=3, *, rngs):
        self.pool_size = pool_size

    def __call__(self, x):
        p = self.pool_size // 2
        pooled = nnx.avg_pool(
            x,
            (self.pool_size, self.pool_size),
            strides=(1, 1),
            padding=((p, p), (p, p)),
            count_include_pad=False,
        )
        return pooled - x


class Attention(nnx.Module):
    def __init__(self, dim, head_dim=32, *, rngs):
        self.num_heads = max(dim // head_dim, 1)
        self.head_dim = head_dim
        attn_dim = self.num_heads * head_dim
        self.qkv = nnx.Linear(dim, 3 * attn_dim, use_bias=False, kernel_init=_init, rngs=rngs)
        self.proj = nnx.Linear(attn_dim, dim, use_bias=False, kernel_init=_init, rngs=rngs)

    def __call__(self, x):
        B, H, W, C = x.shape
        qkv = self.qkv(x.reshape(B, H * W, C)).reshape(B, H * W, 3, self.num_heads, self.head_dim)
        y = dot_product_attention(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2])
        return self.proj(y.reshape(B, H * W, -1)).reshape(B, H, W, C)


_MIXERS = {"pool": Pooling, "conv": SepConv, "attn": Attention}


class MetaFormerBlock(nnx.Module):
    def __init__(self, dim, mixer, norm, res_scale=None, drop=0.0, drop_path=0.0, *, rngs):
        self.norm1 = _norm(norm, dim, rngs=rngs)
        self.token_mixer = _MIXERS[mixer](dim, rngs=rngs)
        self.norm2 = _norm(norm, dim, rngs=rngs)
        self.fc1 = _conv(dim, 4 * dim, 1, use_bias=False, rngs=rngs)
        self.act = StarReLU()
        self.fc2 = _conv(4 * dim, dim, 1, use_bias=False, rngs=rngs)
        self.drop = nnx.Dropout(drop, rngs=rngs)
        if res_scale is not None:
            self.res_scale1 = nnx.Param(jnp.full((dim,), res_scale))
            self.res_scale2 = nnx.Param(jnp.full((dim,), res_scale))
        else:
            self.res_scale1 = self.res_scale2 = None
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        r = x if self.res_scale1 is None else self.res_scale1[...] * x
        x = r + self.drop_path(self.token_mixer(self.norm1(x)))
        r = x if self.res_scale2 is None else self.res_scale2[...] * x
        y = self.drop(self.fc2(self.drop(self.act(self.fc1(self.norm2(x))))))
        return r + self.drop_path(y)


class MetaFormerStage(nnx.Module):
    def __init__(self, in_chs, out_chs, depth, mixer, norm, res_scale, drop, dp_rates, *, rngs):
        if in_chs == out_chs:
            self.downsample_norm = self.downsample = None
        else:
            self.downsample_norm = _norm("layer", in_chs, rngs=rngs)
            self.downsample = _conv(in_chs, out_chs, 3, 2, 1, rngs=rngs)
        self.blocks = nnx.List(
            [
                MetaFormerBlock(out_chs, mixer, norm, res_scale, drop, dp_rates[i], rngs=rngs)
                for i in range(depth)
            ]
        )

    def __call__(self, x):
        if self.downsample is not None:
            x = self.downsample(self.downsample_norm(x))
        for blk in self.blocks:
            x = blk(x)
        return x


class MlpHead(nnx.Module):
    def __init__(self, dim, num_classes, mlp_ratio=4, drop=0.0, *, rngs):
        hidden = int(mlp_ratio * dim)
        self.fc1 = nnx.Linear(dim, hidden, kernel_init=_init, rngs=rngs)
        self.norm = nnx.LayerNorm(hidden, epsilon=1e-6, rngs=rngs)
        self.drop = nnx.Dropout(drop, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, num_classes, kernel_init=_init, rngs=rngs)

    def __call__(self, x):
        # Squared ReLU between the layers.
        return self.fc2(self.drop(self.norm(nnx.relu(self.fc1(x)) ** 2)))


class MetaFormer(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"

    def __init__(
        self,
        depths=(2, 2, 6, 2),
        dims=(64, 128, 320, 512),
        token_mixers=("pool",) * 4,
        norm_layers=("layer",) * 4,
        res_scale_init_values=(None, None, 1.0, 1.0),
        use_mlp_head=True,
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
        self.use_mlp_head, self.drop_rate = use_mlp_head, drop_rate
        self.stem_conv = _conv(in_chans, dims[0], 7, 4, 2, rngs=rngs)
        self.stem_norm = _norm("layer", dims[0], rngs=rngs)
        # timm calculate_drop_path_rates(stagewise=True): linear over all blocks, split by stage.
        total = sum(depths)
        rates = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]
        stages, prev = [], dims[0]
        for i, (dim, depth) in enumerate(zip(dims, depths)):
            start = sum(depths[:i])
            stages.append(
                MetaFormerStage(
                    prev,
                    dim,
                    depth,
                    token_mixers[i],
                    norm_layers[i],
                    res_scale_init_values[i],
                    proj_drop_rate,
                    rates[start : start + depth],
                    rngs=rngs,
                )
            )
            prev = dim
        self.stages = nnx.List(stages)
        self.num_features = dims[-1]
        self.head_norm = nnx.LayerNorm(self.num_features, epsilon=1e-6, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate if use_mlp_head else 0.0, rngs=rngs)
        self.head = self._make_head(num_classes, rngs)

    def _make_head(self, num_classes, rngs):
        if num_classes <= 0:
            return None
        if self.use_mlp_head:
            return MlpHead(self.num_features, num_classes, drop=self.drop_rate, rngs=rngs)
        return nnx.Linear(self.num_features, num_classes, kernel_init=_init, rngs=rngs)

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.global_pool = global_pool if global_pool is not None else self.global_pool
        self.head = self._make_head(num_classes, nnx.Rngs(0))

    def forward_features(self, x):
        x = self.stem_norm(self.stem_conv(x))
        for stage in self.stages:
            x = stage(x)
        return x

    def forward_head(self, x):
        x = self.head_drop(self.head_norm(global_pool_nhwc(x, self.global_pool)))
        return self.head(x) if self.head is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_S18, _S36 = (3, 3, 9, 3), (3, 12, 18, 3)
_DIMS_S, _DIMS_M, _DIMS_B = (64, 128, 320, 512), (96, 192, 384, 576), (128, 256, 512, 768)
_CA = dict(token_mixers=("conv", "conv", "attn", "attn"))
_CONV = dict(token_mixers=("conv",) * 4)
_POOL2 = dict(token_mixers=("pool",) * 4, norm_layers=("group",) * 4, use_mlp_head=False)
_CFGS = {
    "caformer_s18": dict(**_CA, depths=_S18, dims=_DIMS_S),
    "caformer_s36": dict(**_CA, depths=_S36, dims=_DIMS_S),
    "caformer_m36": dict(**_CA, depths=_S36, dims=_DIMS_M),
    "caformer_b36": dict(**_CA, depths=_S36, dims=_DIMS_B),
    "convformer_s18": dict(**_CONV, depths=_S18, dims=_DIMS_S),
    "convformer_s36": dict(**_CONV, depths=_S36, dims=_DIMS_S),
    "convformer_m36": dict(**_CONV, depths=_S36, dims=_DIMS_M),
    "convformer_b36": dict(**_CONV, depths=_S36, dims=_DIMS_B),
    "poolformerv2_s12": dict(**_POOL2, depths=(2, 2, 6, 2), dims=_DIMS_S),
    "poolformerv2_s24": dict(**_POOL2, depths=(4, 4, 12, 4), dims=_DIMS_S),
    "poolformerv2_s36": dict(**_POOL2, depths=(6, 6, 18, 6), dims=_DIMS_S),
    "poolformerv2_m36": dict(**_POOL2, depths=(6, 6, 18, 6), dims=(96, 192, 384, 768)),
    "poolformerv2_m48": dict(**_POOL2, depths=(8, 8, 24, 8), dims=(96, 192, 384, 768)),
}


def _make(name):
    cfg = _CFGS[name]

    def entry(**kwargs):
        model = MetaFormer(**{**cfg, **kwargs})
        model.default_cfg = _cfg(crop_pct=1.0, interpolation="bicubic")
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
