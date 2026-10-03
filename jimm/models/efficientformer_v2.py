"""EfficientFormer-V2 in flax nnx, NHWC. Mirrors timm.models.efficientformer_v2.

Every block has a BatchNorm MLP (1x1 conv, depthwise 3x3 conv, 1x1 conv) with
layer scale; the last ``num_vit`` blocks of the third and fourth stages also
attend. Attention uses 1x1-conv queries, keys, and values, learned relative
position biases, talking-head mixing before and after the softmax, and a
depthwise 3x3 convolution of the values added to the output; in the third
stage it runs on a stride-2 map and is upsampled back. The last downsampling
adds attention whose queries pool the input locally and globally. The head
averages a classifier and a distillation classifier.
"""

import math

import jax
import jax.numpy as jnp
from flax import nnx

from ..layers import BatchNorm, ClassifierMixin, DropPath, gelu, global_pool_nhwc
from ..registry import _cfg, register_model
from .levit import _bias_index

_init = nnx.initializers.truncated_normal(0.02)


class ConvNorm(nnx.Module):
    def __init__(self, in_chs, out_chs, kernel=1, stride=1, groups=1, act=False, *, rngs):
        pad = (stride - 1 + kernel - 1) // 2
        self.conv = nnx.Conv(
            in_chs,
            out_chs,
            (kernel, kernel),
            strides=(stride, stride),
            padding=((pad, pad), (pad, pad)),
            feature_group_count=groups,
            rngs=rngs,
        )
        self.bn = BatchNorm(out_chs, epsilon=1e-5, rngs=rngs)
        self.act = act

    def __call__(self, x):
        x = self.bn(self.conv(x))
        return gelu(x) if self.act else x


def _heads_first(x, num_heads):
    """(B, H, W, heads * d) -> (B, heads, H * W, d), heads-major channels as in timm."""
    B, H, W, C = x.shape
    return x.reshape(B, H * W, num_heads, C // num_heads).transpose(0, 2, 1, 3)


class Attention2d(nnx.Module):
    def __init__(
        self, dim=384, key_dim=32, num_heads=8, attn_ratio=4, resolution=7, stride=None, *, rngs
    ):
        self.num_heads, self.key_dim = num_heads, key_dim
        if stride is not None:
            resolution = math.ceil(resolution / stride)
            self.stride_conv = ConvNorm(dim, dim, 3, stride, groups=dim, rngs=rngs)
        else:
            self.stride_conv = None
        self.stride = stride
        self.resolution = (resolution, resolution)
        self.dh = int(attn_ratio * key_dim) * num_heads
        kh = key_dim * num_heads
        self.q = ConvNorm(dim, kh, rngs=rngs)
        self.k = ConvNorm(dim, kh, rngs=rngs)
        self.v = ConvNorm(dim, self.dh, rngs=rngs)
        self.v_local = ConvNorm(self.dh, self.dh, 3, groups=self.dh, rngs=rngs)
        # 1x1 convolutions over the head axis of the attention map.
        self.talking_head1 = nnx.Linear(num_heads, num_heads, rngs=rngs)
        self.talking_head2 = nnx.Linear(num_heads, num_heads, rngs=rngs)
        self.proj = ConvNorm(self.dh, dim, rngs=rngs)
        self.attention_biases = nnx.Param(jnp.zeros((num_heads, resolution * resolution)))

    def _talk(self, layer, attn):
        return layer(attn.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)

    def __call__(self, x):
        B, H, W, C = x.shape
        if self.stride_conv is not None:
            x = self.stride_conv(x)
        q = _heads_first(self.q(x), self.num_heads)
        k = _heads_first(self.k(x), self.num_heads)
        v = self.v(x)
        v_local = self.v_local(v)
        v = _heads_first(v, self.num_heads)
        attn = (q @ k.swapaxes(-2, -1)) * self.key_dim**-0.5
        attn = attn + self.attention_biases[...][:, _bias_index(self.resolution)]
        attn = self._talk(self.talking_head1, attn)
        attn = self._talk(self.talking_head2, jax.nn.softmax(attn, axis=-1))
        h, w = self.resolution
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, h, w, self.dh) + v_local
        if self.stride is not None:
            x = jax.image.resize(x, (B, H, W, self.dh), "bilinear", antialias=False)
        return self.proj(gelu(x))


class LocalGlobalQuery(nnx.Module):
    def __init__(self, in_dim, out_dim, *, rngs):
        self.local = nnx.Conv(
            in_dim,
            in_dim,
            (3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
            feature_group_count=in_dim,
            rngs=rngs,
        )
        self.proj = ConvNorm(in_dim, out_dim, rngs=rngs)

    def __call__(self, x):
        # AvgPool2d(1, 2) is a stride-2 subsample.
        return self.proj(self.local(x) + x[:, ::2, ::2])


class Attention2dDownsample(nnx.Module):
    def __init__(
        self, dim=384, key_dim=16, num_heads=8, attn_ratio=4, resolution=7, out_dim=None, *, rngs
    ):
        self.num_heads, self.key_dim = num_heads, key_dim
        self.resolution = (resolution, resolution)
        self.resolution2 = (math.ceil(resolution / 2),) * 2
        self.dh = int(attn_ratio * key_dim) * num_heads
        kh = key_dim * num_heads
        self.q = LocalGlobalQuery(dim, kh, rngs=rngs)
        self.k = ConvNorm(dim, kh, rngs=rngs)
        self.v = ConvNorm(dim, self.dh, rngs=rngs)
        self.v_local = ConvNorm(self.dh, self.dh, 3, 2, groups=self.dh, rngs=rngs)
        self.proj = ConvNorm(self.dh, out_dim or dim, rngs=rngs)
        self.attention_biases = nnx.Param(jnp.zeros((num_heads, resolution * resolution)))

    def __call__(self, x):
        B = x.shape[0]
        q = _heads_first(self.q(x), self.num_heads)
        k = _heads_first(self.k(x), self.num_heads)
        v = self.v(x)
        v_local = self.v_local(v)
        v = _heads_first(v, self.num_heads)
        attn = (q @ k.swapaxes(-2, -1)) * self.key_dim**-0.5
        attn = attn + self.attention_biases[...][:, _bias_index(self.resolution, 2)]
        h, w = self.resolution2
        x = (jax.nn.softmax(attn, axis=-1) @ v).transpose(0, 2, 1, 3).reshape(B, h, w, self.dh)
        return self.proj(gelu(x + v_local))


class Downsample(nnx.Module):
    def __init__(self, in_chs, out_chs, resolution=7, use_attn=False, *, rngs):
        self.conv = ConvNorm(in_chs, out_chs, 3, 2, rngs=rngs)
        self.attn = (
            Attention2dDownsample(in_chs, resolution=resolution, out_dim=out_chs, rngs=rngs)
            if use_attn
            else None
        )

    def __call__(self, x):
        out = self.conv(x)
        return out if self.attn is None else self.attn(x) + out


class ConvMlpWithNorm(nnx.Module):
    def __init__(self, dim, hidden, drop=0.0, *, rngs):
        self.fc1 = ConvNorm(dim, hidden, act=True, rngs=rngs)
        self.mid = ConvNorm(hidden, hidden, 3, groups=hidden, act=True, rngs=rngs)
        self.fc2 = ConvNorm(hidden, dim, rngs=rngs)
        self.drop = nnx.Dropout(drop, rngs=rngs)

    def __call__(self, x):
        return self.drop(self.fc2(self.drop(self.mid(self.fc1(x)))))


class EfficientFormerV2Block(nnx.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        ls_init=1e-5,
        resolution=7,
        stride=None,
        use_attn=True,
        *,
        rngs,
    ):
        if use_attn:
            self.token_mixer = Attention2d(dim, resolution=resolution, stride=stride, rngs=rngs)
            self.ls1 = nnx.Param(jnp.full((dim,), ls_init))
        else:
            self.token_mixer = self.ls1 = None
        self.mlp = ConvMlpWithNorm(dim, int(dim * mlp_ratio), drop, rngs=rngs)
        self.ls2 = nnx.Param(jnp.full((dim,), ls_init))
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        if self.token_mixer is not None:
            x = x + self.drop_path(self.ls1[...] * self.token_mixer(x))
        return x + self.drop_path(self.ls2[...] * self.mlp(x))


class EfficientFormerV2Stage(nnx.Module):
    def __init__(
        self,
        dim,
        dim_out,
        depth,
        resolution=7,
        downsample=True,
        block_stride=None,
        downsample_use_attn=False,
        block_use_attn=False,
        num_vit=1,
        mlp_ratios=4.0,
        drop=0.0,
        drop_path=None,
        ls_init=1e-5,
        *,
        rngs,
    ):
        if downsample:
            self.downsample = Downsample(dim, dim_out, resolution, downsample_use_attn, rngs=rngs)
            dim, resolution = dim_out, math.ceil(resolution / 2)
        else:
            self.downsample = None
        if not isinstance(mlp_ratios, (tuple, list)):
            mlp_ratios = (mlp_ratios,) * depth
        drop_path = drop_path or [0.0] * depth
        self.blocks = nnx.List(
            [
                EfficientFormerV2Block(
                    dim,
                    mlp_ratios[i],
                    drop,
                    drop_path[i],
                    ls_init,
                    resolution,
                    block_stride,
                    use_attn=block_use_attn and i > depth - num_vit - 1,
                    rngs=rngs,
                )
                for i in range(depth)
            ]
        )

    def __call__(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class EfficientFormerV2(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"

    def __init__(
        self,
        depths=(2, 2, 6, 4),
        embed_dims=(32, 48, 96, 176),
        mlp_ratios=4,
        num_vit=0,
        layer_scale_init_value=1e-5,
        distillation=True,
        img_size=224,
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
        self.stem1 = ConvNorm(in_chans, embed_dims[0] // 2, 3, 2, act=True, rngs=rngs)
        self.stem2 = ConvNorm(embed_dims[0] // 2, embed_dims[0], 3, 2, act=True, rngs=rngs)
        if not isinstance(mlp_ratios, (tuple, list)):
            mlp_ratios = (mlp_ratios,) * len(depths)
        # timm calculate_drop_path_rates(stagewise=True): linear over all blocks, split by stage.
        total = sum(depths)
        rates = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]
        stages, prev, stride = [], embed_dims[0], 4
        for i, depth in enumerate(depths):
            start = sum(depths[:i])
            stages.append(
                EfficientFormerV2Stage(
                    prev,
                    embed_dims[i],
                    depth,
                    math.ceil(img_size / stride),
                    downsample=i > 0,
                    block_stride=2 if i == 2 else None,
                    downsample_use_attn=i >= 3,
                    block_use_attn=i >= 2,
                    num_vit=num_vit,
                    mlp_ratios=mlp_ratios[i],
                    drop=proj_drop_rate,
                    drop_path=rates[start : start + depth],
                    ls_init=layer_scale_init_value,
                    rngs=rngs,
                )
            )
            stride *= 2 if i > 0 else 1
            prev = embed_dims[i]
        self.stages = nnx.List(stages)
        self.num_features = embed_dims[-1]
        self.norm = BatchNorm(embed_dims[-1], epsilon=1e-5, rngs=rngs)
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
        x = self.stem2(self.stem1(x))
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


_CFGS = {  # depths, embed_dims, mlp ratios, num_vit, drop_path_rate
    "efficientformerv2_s0": (
        (2, 2, 6, 4),
        (32, 48, 96, 176),
        (4, 4, (4, 3, 3, 3, 4, 4), (4, 3, 3, 4)),
        2,
        0.0,
    ),
    "efficientformerv2_s1": (
        (3, 3, 9, 6),
        (32, 48, 120, 224),
        (4, 4, (4, 4, 3, 3, 3, 3, 4, 4, 4), (4, 4, 3, 3, 4, 4)),
        2,
        0.0,
    ),
    "efficientformerv2_s2": (
        (4, 4, 12, 8),
        (32, 64, 144, 288),
        (4, 4, (4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4), (4, 4, 3, 3, 3, 3, 4, 4)),
        4,
        0.02,
    ),
    "efficientformerv2_l": (
        (5, 5, 15, 10),
        (40, 80, 192, 384),
        (
            4,
            4,
            (4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4),
            (4, 4, 4, 3, 3, 3, 3, 4, 4, 4),
        ),
        6,
        0.1,
    ),
}


def _make(name):
    depths, dims, ratios, num_vit, dpr = _CFGS[name]

    def entry(**kwargs):
        kwargs.setdefault("drop_path_rate", dpr)
        model = EfficientFormerV2(depths, dims, ratios, num_vit, **kwargs)
        model.default_cfg = _cfg(crop_pct=0.95, interpolation="bicubic", fixed_input_size=True)
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
