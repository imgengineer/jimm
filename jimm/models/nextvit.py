"""Next-ViT in flax nnx, NHWC. Mirrors timm.models.nextvit.

A four-conv stride-4 stem feeds four stages of Next Convolution Blocks
(grouped 3x3 "multi-head convolutional attention" and a 1x1-conv MLP, both
residual after BatchNorm) and Next Transformer Blocks. A transformer block
runs efficient self-attention (keys and values average-pooled over
consecutive raster-order tokens, then BatchNorm) on 75% of its output
channels, projects the result to the remaining channels for a convolutional
attention branch, concatenates both and applies the MLP. Width changes and
2x downsampling (2x2 average pool) happen in each block's patch embedding.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from ..attention import dot_product_attention
from ..layers import BatchNorm, ClassifierMixin, DropPath, global_pool_nhwc
from ..registry import _cfg, register_model

_trunc = nnx.initializers.truncated_normal(0.02)


def _conv(in_chs, out_chs, kernel=1, stride=1, groups=1, use_bias=False, *, rngs):
    pad = kernel // 2
    return nnx.Conv(
        in_chs,
        out_chs,
        (kernel, kernel),
        strides=(stride, stride),
        padding=((pad, pad), (pad, pad)),
        feature_group_count=groups,
        use_bias=use_bias,
        kernel_init=_trunc,
        rngs=rngs,
    )


def _bn(chs, *, rngs):
    return BatchNorm(chs, epsilon=1e-5, momentum=0.9, rngs=rngs)


def _make_divisible(v, divisor, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    return new_v + divisor if new_v < 0.9 * v else new_v


def _avg_pool_2x2(x):
    """``AvgPool2d(2, 2, ceil_mode=True, count_include_pad=False)``."""
    B, H, W, C = x.shape
    ph, pw = H % 2, W % 2
    if not (ph or pw):
        return x.reshape(B, H // 2, 2, W // 2, 2, C).mean(axis=(2, 4))
    pad = ((0, 0), (0, ph), (0, pw), (0, 0))
    window = (1, 2, 2, 1)
    total = jax.lax.reduce_window(jnp.pad(x, pad), 0.0, jax.lax.add, window, window, "VALID")
    ones = jnp.pad(jnp.ones((1, H, W, 1), x.dtype), pad)
    count = jax.lax.reduce_window(ones, 0.0, jax.lax.add, window, window, "VALID")
    return total / count


class ConvNormAct(nnx.Module):
    def __init__(self, in_chs, out_chs, stride=1, *, rngs):
        self.conv = _conv(in_chs, out_chs, 3, stride, rngs=rngs)
        self.norm = _bn(out_chs, rngs=rngs)

    def __call__(self, x):
        return nnx.relu(self.norm(self.conv(x)))


class PatchEmbed(nnx.Module):
    def __init__(self, in_chs, out_chs, stride=1, *, rngs):
        self.stride = stride
        if stride == 2 or in_chs != out_chs:
            self.conv = _conv(in_chs, out_chs, rngs=rngs)
            self.norm = _bn(out_chs, rngs=rngs)
        else:
            self.conv = self.norm = None

    def __call__(self, x):
        if self.stride == 2:
            x = _avg_pool_2x2(x)
        if self.conv is not None:
            x = self.norm(self.conv(x))
        return x


class ConvAttention(nnx.Module):
    """Multi-head convolutional attention: grouped 3x3 conv (one group per head), BN, ReLU, 1x1."""

    def __init__(self, out_chs, head_dim, *, rngs):
        self.group_conv3x3 = _conv(out_chs, out_chs, 3, groups=out_chs // head_dim, rngs=rngs)
        self.norm = _bn(out_chs, rngs=rngs)
        self.projection = _conv(out_chs, out_chs, rngs=rngs)

    def __call__(self, x):
        return self.projection(nnx.relu(self.norm(self.group_conv3x3(x))))


class ConvMlp(nnx.Module):
    def __init__(self, dim, hidden, drop=0.0, *, rngs):
        self.fc1 = _conv(dim, hidden, use_bias=True, rngs=rngs)
        self.fc2 = _conv(hidden, dim, use_bias=True, rngs=rngs)
        self.drop = nnx.Dropout(drop, rngs=rngs)

    def __call__(self, x):
        return self.fc2(self.drop(nnx.relu(self.fc1(x))))


class NextConvBlock(nnx.Module):
    def __init__(self, in_chs, out_chs, stride, drop_path, drop, head_dim, *, rngs):
        self.patch_embed = PatchEmbed(in_chs, out_chs, stride, rngs=rngs)
        self.mhca = ConvAttention(out_chs, head_dim, rngs=rngs)
        self.attn_drop_path = DropPath(drop_path, rngs=rngs)
        self.norm = _bn(out_chs, rngs=rngs)
        self.mlp = ConvMlp(out_chs, int(out_chs * 3.0), drop, rngs=rngs)
        self.mlp_drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        x = self.patch_embed(x)
        x = x + self.attn_drop_path(self.mhca(x))
        return x + self.mlp_drop_path(self.mlp(self.norm(x)))


def _linear(din, dout, *, rngs):
    return nnx.Linear(din, dout, kernel_init=_trunc, bias_init=nnx.initializers.zeros, rngs=rngs)


class EfficientAttention(nnx.Module):
    """Self-attention whose keys and values come from tokens average-pooled ``sr_ratio**2`` at a time."""

    def __init__(self, dim, head_dim=32, sr_ratio=1, attn_drop=0.0, proj_drop=0.0, *, rngs):
        self.num_heads, self.head_dim = dim // head_dim, head_dim
        self.q = _linear(dim, dim, rngs=rngs)
        self.k = _linear(dim, dim, rngs=rngs)
        self.v = _linear(dim, dim, rngs=rngs)
        self.proj = _linear(dim, dim, rngs=rngs)
        self.attn_drop = attn_drop
        self.proj_drop = nnx.Dropout(proj_drop, rngs=rngs)
        self.n_ratio = sr_ratio**2
        self.norm = _bn(dim, rngs=rngs) if sr_ratio > 1 else None

    def __call__(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim)
        if self.norm is not None:
            # AvgPool1d over the raster-order token sequence, dropping any remainder.
            n = N // self.n_ratio
            x = x[:, : n * self.n_ratio].reshape(B, n, self.n_ratio, C).mean(axis=2)
            x = self.norm(x)
        k = self.k(x).reshape(B, -1, self.num_heads, self.head_dim)
        v = self.v(x).reshape(B, -1, self.num_heads, self.head_dim)
        x = dot_product_attention(q, k, v).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class NextTransformerBlock(nnx.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        drop_path,
        stride=1,
        sr_ratio=1,
        head_dim=32,
        mix_block_ratio=0.75,
        attn_drop=0.0,
        drop=0.0,
        *,
        rngs,
    ):
        self.mhsa_out_chs = _make_divisible(int(out_chs * mix_block_ratio), 32)
        self.mhca_out_chs = out_chs - self.mhsa_out_chs
        self.patch_embed = PatchEmbed(in_chs, self.mhsa_out_chs, stride, rngs=rngs)
        self.norm1 = _bn(self.mhsa_out_chs, rngs=rngs)
        self.e_mhsa = EfficientAttention(
            self.mhsa_out_chs, head_dim, sr_ratio, attn_drop, drop, rngs=rngs
        )
        self.mhsa_drop_path = DropPath(drop_path * mix_block_ratio, rngs=rngs)
        self.projection = PatchEmbed(self.mhsa_out_chs, self.mhca_out_chs, rngs=rngs)
        self.mhca = ConvAttention(self.mhca_out_chs, head_dim, rngs=rngs)
        self.mhca_drop_path = DropPath(drop_path * (1 - mix_block_ratio), rngs=rngs)
        self.norm2 = _bn(out_chs, rngs=rngs)
        self.mlp = ConvMlp(out_chs, int(out_chs * 2), drop, rngs=rngs)
        self.mlp_drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        x = self.patch_embed(x)
        B, H, W, C = x.shape
        out = self.e_mhsa(self.norm1(x).reshape(B, H * W, C))
        x = x + self.mhsa_drop_path(out).reshape(B, H, W, C)
        out = self.projection(x)
        out = out + self.mhca_drop_path(self.mhca(out))
        x = jnp.concatenate([x, out], axis=-1)
        return x + self.mlp_drop_path(self.mlp(self.norm2(x)))


class NextStage(nnx.Module):
    def __init__(
        self,
        in_chs,
        block_chs,
        block_types,
        stride,
        sr_ratio,
        mix_block_ratio,
        drop,
        attn_drop,
        drop_path,
        head_dim,
        *,
        rngs,
    ):
        blocks = []
        for i, (out_chs, kind) in enumerate(zip(block_chs, block_types)):
            s = stride if i == 0 else 1
            if kind == "conv":
                blk = NextConvBlock(in_chs, out_chs, s, drop_path[i], drop, head_dim, rngs=rngs)
            else:
                blk = NextTransformerBlock(
                    in_chs,
                    out_chs,
                    drop_path[i],
                    s,
                    sr_ratio,
                    head_dim,
                    mix_block_ratio,
                    attn_drop,
                    drop,
                    rngs=rngs,
                )
            blocks.append(blk)
            in_chs = out_chs
        self.blocks = nnx.List(blocks)

    def __call__(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class NextViT(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        depths=(3, 4, 10, 3),
        stem_chs=(64, 32, 64),
        strides=(1, 2, 2, 2),
        sr_ratios=(8, 4, 2, 1),
        head_dim=32,
        mix_block_ratio=0.75,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        stage_out_chs = [
            [96] * depths[0],
            [192] * (depths[1] - 1) + [256],
            [384, 384, 384, 384, 512] * (depths[2] // 5),
            [768] * (depths[3] - 1) + [1024],
        ]
        stage_types = [
            ["conv"] * depths[0],
            ["conv"] * (depths[1] - 1) + ["attn"],
            ["conv"] * 4 + ["attn"],
            ["conv"] * (depths[3] - 1) + ["attn"],
        ]
        stage_types[2] = stage_types[2] * (depths[2] // 5)
        self.stem = nnx.List(
            [
                ConvNormAct(in_chans, stem_chs[0], 2, rngs=rngs),
                ConvNormAct(stem_chs[0], stem_chs[1], rngs=rngs),
                ConvNormAct(stem_chs[1], stem_chs[2], rngs=rngs),
                ConvNormAct(stem_chs[2], stem_chs[2], 2, rngs=rngs),
            ]
        )
        total = sum(depths)
        dpr = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]
        stages, in_chs, k = [], stem_chs[-1], 0
        for i, depth in enumerate(depths):
            stages.append(
                NextStage(
                    in_chs,
                    stage_out_chs[i],
                    stage_types[i],
                    strides[i],
                    sr_ratios[i],
                    mix_block_ratio,
                    drop_rate,
                    attn_drop_rate,
                    dpr[k : k + depth],
                    head_dim,
                    rngs=rngs,
                )
            )
            in_chs, k = stage_out_chs[i][-1], k + depth
        self.stages = nnx.List(stages)
        self.num_features = in_chs
        self.norm = _bn(in_chs, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = self._make_fc(num_classes, rngs)

    def _make_fc(self, num_classes, rngs):
        return _linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.global_pool = global_pool if global_pool is not None else self.global_pool
        self.fc = self._make_fc(num_classes, nnx.Rngs(0))

    def forward_features(self, x):
        for conv in self.stem:
            x = conv(x)
        for stage in self.stages:
            x = stage(x)
        return self.norm(x)

    def forward_head(self, x):
        x = self.head_drop(global_pool_nhwc(x, self.global_pool))
        return self.fc(x) if self.fc is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_CFGS = {
    "nextvit_small": ((3, 4, 10, 3), 0.1),
    "nextvit_base": ((3, 4, 20, 3), 0.2),
    "nextvit_large": ((3, 4, 30, 3), 0.2),
}


def _make(name):
    depths, dpr = _CFGS[name]

    def entry(**kwargs):
        model = NextViT(depths, **{"drop_path_rate": dpr, **kwargs})
        model.default_cfg = _cfg(crop_pct=0.95, interpolation="bicubic")
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
