"""TinyViT in flax nnx, NHWC. Mirrors timm.models.tiny_vit.

A two-conv stride-4 stem feeds an MBConv stage and three transformer stages.
Each transformer stage starts with patch merging (1x1 conv, strided depthwise
3x3 conv, 1x1 conv); its blocks attend within zero-padded windows, with a
LayerNorm inside the attention and learned relative position biases, then
apply a depthwise 3x3 convolution and a pre-norm MLP. The head pools,
normalizes, and classifies.
"""

import jax.numpy as jnp
from flax import nnx

from ..attention import dot_product_attention
from ..layers import ClassifierMixin, DropPath, Mlp, gelu, global_pool_nhwc
from ..registry import _cfg, register_model
from ._conv import ConvNormAct
from .levit import _bias_index

_init = nnx.initializers.truncated_normal(0.02)


def _conv_norm(in_chs, out_chs, kernel=1, stride=1, groups=1, bn_weight_init=1.0, *, rngs):
    return ConvNormAct(
        in_chs, out_chs, kernel, stride, groups, bn_weight_init=bn_weight_init, rngs=rngs
    )


class PatchEmbed(nnx.Module):
    def __init__(self, in_chs, out_chs, *, rngs):
        self.conv1 = _conv_norm(in_chs, out_chs // 2, 3, 2, rngs=rngs)
        self.conv2 = _conv_norm(out_chs // 2, out_chs, 3, 2, rngs=rngs)

    def __call__(self, x):
        return self.conv2(gelu(self.conv1(x)))


class MBConv(nnx.Module):
    def __init__(self, dim, expand_ratio=4.0, drop_path=0.0, *, rngs):
        mid = int(dim * expand_ratio)
        self.conv1 = _conv_norm(dim, mid, rngs=rngs)
        self.conv2 = _conv_norm(mid, mid, 3, groups=mid, rngs=rngs)
        self.conv3 = _conv_norm(mid, dim, bn_weight_init=0.0, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        y = gelu(self.conv2(gelu(self.conv1(x))))
        return gelu(x + self.drop_path(self.conv3(y)))


class PatchMerging(nnx.Module):
    def __init__(self, dim, out_dim, *, rngs):
        self.conv1 = _conv_norm(dim, out_dim, rngs=rngs)
        self.conv2 = _conv_norm(out_dim, out_dim, 3, 2, groups=out_dim, rngs=rngs)
        self.conv3 = _conv_norm(out_dim, out_dim, rngs=rngs)

    def __call__(self, x):
        return self.conv3(gelu(self.conv2(gelu(self.conv1(x)))))


class Attention(nnx.Module):
    def __init__(self, dim, num_heads, resolution, *, rngs):
        self.num_heads = num_heads
        self.resolution = (resolution, resolution)
        self.norm = nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)
        # Values are as wide as keys; channels are grouped per head as [q, k, v].
        self.qkv = nnx.Linear(dim, 3 * dim, kernel_init=_init, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, kernel_init=_init, rngs=rngs)
        self.attention_biases = nnx.Param(jnp.zeros((num_heads, resolution * resolution)))

    def __call__(self, x):
        B, N, C = x.shape
        qkv = self.qkv(self.norm(x)).reshape(B, N, self.num_heads, -1)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        bias = self.attention_biases[...][:, _bias_index(self.resolution)]
        x = dot_product_attention(q, k, v, bias=bias[None])
        return self.proj(x.reshape(B, N, C))


class TinyVitBlock(nnx.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        local_conv_size=3,
        *,
        rngs,
    ):
        self.window_size = window_size
        self.attn = Attention(dim, num_heads, window_size, rngs=rngs)
        self.local_conv = _conv_norm(dim, dim, local_conv_size, groups=dim, rngs=rngs)
        self.norm = nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop, kernel_init=_init, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        B, H, W, C = x.shape
        ws = self.window_size
        pad_b, pad_r = -H % ws, -W % ws
        # timm pads before the attention's LayerNorm, so padded tokens are attended to.
        y = jnp.pad(x, ((0, 0), (0, pad_b), (0, pad_r), (0, 0)))
        nh, nw = (H + pad_b) // ws, (W + pad_r) // ws
        y = y.reshape(B, nh, ws, nw, ws, C).transpose(0, 1, 3, 2, 4, 5)
        y = self.attn(y.reshape(B * nh * nw, ws * ws, C))
        y = y.reshape(B, nh, nw, ws, ws, C).transpose(0, 1, 3, 2, 4, 5)
        x = x + self.drop_path(y.reshape(B, nh * ws, nw * ws, C)[:, :H, :W])
        x = self.local_conv(x)
        return x + self.drop_path(self.mlp(self.norm(x)))


class ConvLayer(nnx.Module):
    def __init__(self, dim, depth, drop_path, expand_ratio=4.0, *, rngs):
        self.blocks = nnx.List(
            [MBConv(dim, expand_ratio, drop_path[i], rngs=rngs) for i in range(depth)]
        )

    def __call__(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class TinyVitStage(nnx.Module):
    def __init__(
        self,
        dim,
        out_dim,
        depth,
        num_heads,
        window_size,
        mlp_ratio,
        drop,
        drop_path,
        local_conv_size,
        *,
        rngs,
    ):
        self.downsample = PatchMerging(dim, out_dim, rngs=rngs)
        self.blocks = nnx.List(
            [
                TinyVitBlock(
                    out_dim,
                    num_heads,
                    window_size,
                    mlp_ratio,
                    drop,
                    drop_path[i],
                    local_conv_size,
                    rngs=rngs,
                )
                for i in range(depth)
            ]
        )

    def __call__(self, x):
        x = self.downsample(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class TinyVit(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        embed_dims=(96, 192, 384, 768),
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_sizes=(7, 7, 14, 7),
        mlp_ratio=4.0,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.1,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.patch_embed = PatchEmbed(in_chans, embed_dims[0], rngs=rngs)
        total = sum(depths)
        dpr = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]
        stages = [
            ConvLayer(embed_dims[0], depths[0], dpr[: depths[0]], mbconv_expand_ratio, rngs=rngs)
        ]
        for i in range(1, len(depths)):
            start = sum(depths[:i])
            stages.append(
                TinyVitStage(
                    embed_dims[i - 1],
                    embed_dims[i],
                    depths[i],
                    num_heads[i],
                    window_sizes[i],
                    mlp_ratio,
                    drop_rate,
                    dpr[start : start + depths[i]],
                    local_conv_size,
                    rngs=rngs,
                )
            )
        self.stages = nnx.List(stages)
        self.num_features = embed_dims[-1]
        self.head_norm = nnx.LayerNorm(self.num_features, epsilon=1e-5, rngs=rngs)
        # timm applies drop_rate inside the MLPs; its classifier head has no dropout.
        self.head_drop = nnx.Dropout(0.0, rngs=rngs)
        self.fc = (
            nnx.Linear(self.num_features, num_classes, kernel_init=_init, rngs=rngs)
            if num_classes > 0
            else None
        )

    def forward_features(self, x):
        x = self.patch_embed(x)
        for stage in self.stages:
            x = stage(x)
        return x

    def forward_head(self, x):
        x = self.head_drop(self.head_norm(global_pool_nhwc(x, self.global_pool)))
        return self.fc(x) if self.fc is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_CFGS = {  # embed_dims, num_heads, window_sizes, drop_path_rate, input size
    "tiny_vit_5m_224": ((64, 128, 160, 320), (2, 4, 5, 10), (7, 7, 14, 7), 0.0, 224),
    "tiny_vit_11m_224": ((64, 128, 256, 448), (2, 4, 8, 14), (7, 7, 14, 7), 0.1, 224),
    "tiny_vit_21m_224": ((96, 192, 384, 576), (3, 6, 12, 18), (7, 7, 14, 7), 0.2, 224),
    "tiny_vit_21m_384": ((96, 192, 384, 576), (3, 6, 12, 18), (12, 12, 24, 12), 0.1, 384),
    "tiny_vit_21m_512": ((96, 192, 384, 576), (3, 6, 12, 18), (16, 16, 32, 16), 0.1, 512),
}


def _make(name):
    embed_dims, num_heads, window_sizes, drop_path_rate, size = _CFGS[name]

    def entry(**kwargs):
        kwargs.setdefault("drop_path_rate", drop_path_rate)
        model = TinyVit(embed_dims, (2, 2, 6, 2), num_heads, window_sizes, **kwargs)
        model.default_cfg = _cfg(input_size=(3, size, size), crop_pct=0.95 if size == 224 else 1.0)
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
