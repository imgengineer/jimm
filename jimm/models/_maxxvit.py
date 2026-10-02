"""MaxxViT blocks for timm's "rw" MaxViT and CoAtNet configurations, NHWC throughout.

Mirrors timm.models.maxxvit: pre-norm MBConv blocks, transformer blocks with
pooled shortcuts, window and grid partition attention, and relative-position
bias tables. Convolutions use PyTorch's symmetric padding.
"""

from functools import lru_cache
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from flax import nnx

from ..attention import dot_product_attention
from ..layers import BatchNorm, ClassifierMixin, DropPath, Mlp, SqueezeExcite, make_divisible


class ConvCfg(NamedTuple):
    stride_mode: str = "dw"  # where MBConv strides: "pool" before conv1 or "dw" in conv2
    pre_norm_act: bool = False
    output_bias: bool = False
    attn_early: bool = False  # SE between the depthwise conv and its norm
    se_act: str = "silu"
    attn_ratio: float = 0.25


class TransformerCfg(NamedTuple):
    dim_head: int = 32
    shortcut_bias: bool = True


_ACTS = {"relu": nnx.relu, "silu": nnx.silu}


@lru_cache(maxsize=None)
def _relative_position_index(window):
    rows, cols = window
    coords = np.stack(np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")).reshape(2, -1)
    offsets = (coords[:, :, None] - coords[:, None, :]).transpose(1, 2, 0)
    offsets[..., 0] += rows - 1
    offsets[..., 1] += cols - 1
    offsets[..., 0] *= 2 * cols - 1
    return offsets.sum(-1)


class RelPosBias(nnx.Module):
    """Learned bias per head for every relative offset inside a window."""

    def __init__(self, window, num_heads, *, rngs):
        self.window = tuple(window)
        size = (2 * window[0] - 1) * (2 * window[1] - 1)
        self.relative_position_bias_table = nnx.Param(
            nnx.initializers.truncated_normal(0.02)(rngs.params(), (size, num_heads))
        )

    def __call__(self):
        bias = self.relative_position_bias_table[...][_relative_position_index(self.window)]
        return jnp.transpose(bias, (2, 0, 1))[None]


class Attention(nnx.Module):
    """Head-first fused QKV attention over flattened tokens with relative bias."""

    def __init__(self, dim, dim_out, dim_head, window, *, rngs):
        self.num_heads, self.dim_head = dim // dim_head, dim_head
        self.qkv = nnx.Linear(dim, dim * 3, rngs=rngs)
        self.rel_pos = RelPosBias(window, self.num_heads, rngs=rngs)
        self.proj = nnx.Linear(dim, dim_out, rngs=rngs)

    def __call__(self, x):
        batch, tokens, _ = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, self.num_heads, 3, self.dim_head)
        q, k, v = qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :]
        x = dot_product_attention(q, k, v, bias=self.rel_pos())
        return self.proj(x.reshape(batch, tokens, -1))


def _avg_pool2(x):
    return nnx.avg_pool(x, (2, 2), strides=(2, 2))


class Downsample2d(nnx.Module):
    """2x2 average pooling, then a 1x1 projection when the width changes."""

    def __init__(self, dim, dim_out, bias=True, *, rngs):
        self.expand = (
            nnx.Conv(dim, dim_out, (1, 1), use_bias=bias, rngs=rngs) if dim != dim_out else None
        )

    def __call__(self, x):
        x = _avg_pool2(x)
        return x if self.expand is None else self.expand(x)


def _conv(in_chs, out_chs, kernel=1, stride=1, groups=1, bias=False, *, rngs):
    return nnx.Conv(
        in_chs,
        out_chs,
        (kernel, kernel),
        strides=(stride, stride),
        padding=((kernel // 2, kernel // 2),) * 2,
        feature_group_count=groups,
        use_bias=bias,
        rngs=rngs,
    )


class MbConvBlock(nnx.Module):
    """Pre-norm inverted bottleneck (1x1, depthwise 3x3, 1x1) with squeeze-excite."""

    def __init__(self, in_chs, out_chs, stride, cfg, drop_path, *, rngs):
        mid = make_divisible(in_chs * 4)  # rw models expand from the input width
        self.cfg = cfg
        self.pool = stride == 2 and cfg.stride_mode == "pool"
        self.shortcut = (
            Downsample2d(in_chs, out_chs, bias=cfg.output_bias, rngs=rngs) if stride == 2 else None
        )
        self.pre_norm = BatchNorm(in_chs, rngs=rngs)
        self.conv1_1x1 = _conv(in_chs, mid, rngs=rngs)
        self.norm1 = BatchNorm(mid, rngs=rngs)
        dw_stride = stride if cfg.stride_mode == "dw" else 1
        self.conv2_kxk = _conv(mid, mid, 3, dw_stride, groups=mid, rngs=rngs)
        self.se = SqueezeExcite(
            mid, rd_channels=int(cfg.attn_ratio * mid), act=_ACTS[cfg.se_act], rngs=rngs
        )
        self.norm2 = BatchNorm(mid, rngs=rngs)
        self.conv3_1x1 = _conv(mid, out_chs, bias=cfg.output_bias, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        shortcut = x if self.shortcut is None else self.shortcut(x)
        x = self.pre_norm(x)
        if self.cfg.pre_norm_act:
            x = nnx.silu(x)
        if self.pool:
            x = _avg_pool2(x)
        x = self.conv2_kxk(nnx.silu(self.norm1(self.conv1_1x1(x))))
        if self.cfg.attn_early:
            x = self.se(x)
        x = nnx.silu(self.norm2(x))
        if not self.cfg.attn_early:
            x = self.se(x)
        return self.drop_path(self.conv3_1x1(x)) + shortcut


class TransformerBlock2d(nnx.Module):
    """CoAtNet transformer block attending across the whole (pooled) feature map."""

    def __init__(self, dim, dim_out, stride, feat_size, cfg, drop_path, *, rngs):
        self.downsample = stride == 2
        self.shortcut = (
            Downsample2d(dim, dim_out, bias=cfg.shortcut_bias, rngs=rngs)
            if self.downsample
            else None
        )
        self.norm1 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.attn = Attention(dim, dim_out, cfg.dim_head, feat_size, rngs=rngs)
        self.drop_path1 = DropPath(drop_path, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim_out, epsilon=1e-6, rngs=rngs)
        self.mlp = Mlp(dim_out, dim_out * 4, rngs=rngs)
        self.drop_path2 = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        shortcut = x if self.shortcut is None else self.shortcut(x)
        y = self.norm1(x)
        if self.downsample:
            y = _avg_pool2(y)
        batch, rows, cols, chs = y.shape
        y = self.attn(y.reshape(batch, rows * cols, chs)).reshape(batch, rows, cols, -1)
        x = shortcut + self.drop_path1(y)
        return x + self.drop_path2(self.mlp(self.norm2(x)))


def _window_partition(x, size):
    batch, rows, cols, chs = x.shape
    x = x.reshape(batch, rows // size[0], size[0], cols // size[1], size[1], chs)
    return x.transpose(0, 1, 3, 2, 4, 5).reshape(-1, size[0] * size[1], chs)


def _window_reverse(windows, size, rows, cols):
    chs = windows.shape[-1]
    x = windows.reshape(-1, rows // size[0], cols // size[1], size[0], size[1], chs)
    return x.transpose(0, 1, 3, 2, 4, 5).reshape(-1, rows, cols, chs)


def _grid_partition(x, size):
    batch, rows, cols, chs = x.shape
    x = x.reshape(batch, size[0], rows // size[0], size[1], cols // size[1], chs)
    return x.transpose(0, 2, 4, 1, 3, 5).reshape(-1, size[0] * size[1], chs)


def _grid_reverse(windows, size, rows, cols):
    chs = windows.shape[-1]
    x = windows.reshape(-1, rows // size[0], cols // size[1], size[0], size[1], chs)
    return x.transpose(0, 3, 1, 4, 2, 5).reshape(-1, rows, cols, chs)


class PartitionAttention(nnx.Module):
    """Attention within local windows (block) or across a dilated grid, then an MLP."""

    def __init__(self, dim, partition, block, dim_head, drop_path, *, rngs):
        self.block, self.partition = block, tuple(partition)
        self.norm1 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.attn = Attention(dim, dim, dim_head, partition, rngs=rngs)
        self.drop_path1 = DropPath(drop_path, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.mlp = Mlp(dim, dim * 4, rngs=rngs)
        self.drop_path2 = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        _, rows, cols, _ = x.shape
        split, merge = (
            (_window_partition, _window_reverse) if self.block else (_grid_partition, _grid_reverse)
        )
        y = merge(self.attn(split(self.norm1(x), self.partition)), self.partition, rows, cols)
        x = x + self.drop_path1(y)
        return x + self.drop_path2(self.mlp(self.norm2(x)))


class MaxxVitBlock(nnx.Module):
    """MBConv, then window attention, then grid attention."""

    def __init__(
        self, dim, dim_out, stride, partition, conv_cfg, transformer_cfg, drop_path, *, rngs
    ):
        self.conv = MbConvBlock(dim, dim_out, stride, conv_cfg, drop_path, rngs=rngs)
        head = transformer_cfg.dim_head
        self.attn_block = PartitionAttention(dim_out, partition, True, head, drop_path, rngs=rngs)
        self.attn_grid = PartitionAttention(dim_out, partition, False, head, drop_path, rngs=rngs)

    def __call__(self, x):
        return self.attn_grid(self.attn_block(self.conv(x)))


class MaxxVit(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        embed_dim,
        depths,
        block_types,
        stem_width,
        conv_cfg,
        transformer_cfg,
        img_size=224,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dim[-1]
        self.stem = nnx.List(
            [
                _conv(in_chans, stem_width[0], 3, 2, rngs=rngs),
                BatchNorm(stem_width[0], rngs=rngs),
                _conv(stem_width[0], stem_width[1], 3, rngs=rngs),
            ]
        )
        feat = img_size // 2
        partition = (img_size // 32,) * 2  # timm partition_ratio: window = grid = size / 32
        rates = np.linspace(0.0, drop_path_rate, sum(depths)).tolist()
        stages, in_chs, offset = [], stem_width[1], 0
        for dim, depth, kind in zip(embed_dim, depths, block_types):
            feat = (feat - 1) // 2 + 1
            blocks = []
            for j in range(depth):
                stride, rate = 2 if j == 0 else 1, rates[offset + j]
                if kind == "C":
                    block = MbConvBlock(in_chs, dim, stride, conv_cfg, rate, rngs=rngs)
                elif kind == "T":
                    block = TransformerBlock2d(
                        in_chs, dim, stride, (feat, feat), transformer_cfg, rate, rngs=rngs
                    )
                else:
                    block = MaxxVitBlock(
                        in_chs, dim, stride, partition, conv_cfg, transformer_cfg, rate, rngs=rngs
                    )
                blocks.append(block)
                in_chs = dim
            stages.append(nnx.List(blocks))
            offset += depth
        self.stages = nnx.List(stages)
        self.norm = nnx.LayerNorm(self.num_features, epsilon=1e-6, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.stem[2](nnx.silu(self.stem[1](self.stem[0](x))))
        for stage in self.stages:
            for block in stage:
                x = block(x)
        return self.norm(x)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))
