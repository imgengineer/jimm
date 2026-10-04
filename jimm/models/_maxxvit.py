"""MaxxViT family (MaxViT, CoAtNet, CoAtNeXt, MaxxViT, MaxxViT-V2) in flax nnx, NHWC.

Mirrors timm.models.maxxvit and its configuration dataclasses: MBConv or
ConvNeXt conv blocks, CoAtNet transformer blocks over the whole (pooled)
feature map, MaxViT blocks (conv, window attention, grid attention) and
parallel window/grid attention, with learned, TF-style or MLP-generated
relative position bias. Convolutions use PyTorch symmetric padding, or
TensorFlow "SAME" padding for the tf ports.
"""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from ..attention import dot_product_attention
from ..layers import BatchNorm, ClassifierMixin, DropPath, SqueezeExcite, make_divisible
from ._maxxvit_cfgs import CONV_DEFAULTS, TRANSFORMER_DEFAULTS

_ACTS = {
    "gelu": lambda x: jax.nn.gelu(x, approximate=False),
    "gelu_tanh": lambda x: jax.nn.gelu(x, approximate=True),
    "silu": nnx.silu,
    "relu": nnx.relu,
    "tanh": jnp.tanh,
}


def conv_cfg(**overrides):
    cfg = {**CONV_DEFAULTS, **overrides}
    return cfg


def transformer_cfg(**overrides):
    return {**TRANSFORMER_DEFAULTS, **overrides}


def _norm(kind, chs, eps, *, rngs):
    if kind == "batchnorm2d":
        return BatchNorm(chs, epsilon=eps, rngs=rngs)
    return nnx.LayerNorm(chs, epsilon=eps, rngs=rngs)


def _conv(in_chs, out_chs, kernel=1, stride=1, groups=1, bias=True, padding="", *, rngs):
    if padding == "same":
        pad = "SAME"
    else:
        p = ((stride - 1) + (kernel - 1)) // 2
        pad = ((p, p), (p, p))
    return nnx.Conv(
        in_chs,
        out_chs,
        (kernel, kernel),
        strides=(stride, stride),
        padding=pad,
        feature_group_count=groups,
        use_bias=bias,
        rngs=rngs,
    )


def _pool(x, pool_type, padding=""):
    window2, strides2 = (1, 2, 2, 1), (1, 2, 2, 1)
    if pool_type == "max":
        return jax.lax.reduce_window(
            x, -jnp.inf, jax.lax.max, (1, 3, 3, 1), strides2, ((0, 0), (1, 1), (1, 1), (0, 0))
        )
    if pool_type == "max2":
        return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, window2, strides2, "VALID")
    if pool_type == "avg":
        pad = ((0, 0), (1, 1), (1, 1), (0, 0))
        total = jax.lax.reduce_window(x, 0.0, jax.lax.add, (1, 3, 3, 1), strides2, pad)
        ones = jnp.ones((1, *x.shape[1:3], 1), x.dtype)
        return total / jax.lax.reduce_window(ones, 0.0, jax.lax.add, (1, 3, 3, 1), strides2, pad)
    # avg2: 2x2 / stride 2; timm's 'same' variant zero-pads (counted) when sizes are odd.
    return (
        jax.lax.reduce_window(
            x, 0.0, jax.lax.add, window2, strides2, "SAME" if padding == "same" else "VALID"
        )
        / 4.0
    )


class Downsample2d(nnx.Module):
    def __init__(self, dim, dim_out, pool_type="avg2", padding="", bias=True, *, rngs):
        self.pool_type, self.padding = pool_type, padding
        self.expand = _conv(dim, dim_out, bias=bias, rngs=rngs) if dim != dim_out else None

    def __call__(self, x):
        x = _pool(x, self.pool_type, self.padding)
        return x if self.expand is None else self.expand(x)


# ----------------------------------------------------------------------------- relative position


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


class RelPosBiasTf(nnx.Module):
    """TF MaxViT relative bias: a (heads, 2h-1, 2w-1) table indexed by (x - i, y - j)."""

    def __init__(self, window, num_heads, *, rngs):
        self.window = tuple(window)
        h, w = window
        self.relative_position_bias_table = nnx.Param(
            nnx.initializers.normal(0.02)(rngs.params(), (num_heads, 2 * h - 1, 2 * w - 1))
        )

    def __call__(self):
        h, w = self.window
        ih = np.arange(h)[None, :] - np.arange(h)[:, None] + h - 1  # [i, x]
        iw = np.arange(w)[None, :] - np.arange(w)[:, None] + w - 1  # [j, y]
        t = self.relative_position_bias_table[...]
        bias = t[:, ih[:, None, :, None], iw[None, :, None, :]]  # [n, i, j, x, y]
        return bias.reshape(1, t.shape[0], h * w, h * w)


class RelPosMlp(nnx.Module):
    """timm RelPosMlp ('cr' mode): log-spaced relative coordinates through a ReLU MLP."""

    def __init__(self, window, num_heads, hidden_dim=512, *, rngs):
        self.window, self.num_heads = tuple(window), num_heads
        self.fc1 = nnx.Linear(2, hidden_dim, rngs=rngs)
        self.drop = nnx.Dropout(0.125, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, num_heads, rngs=rngs)
        h, w = window
        table = np.stack(
            np.meshgrid(np.arange(1 - h, h), np.arange(1 - w, w), indexing="ij"), axis=-1
        ).astype(np.float32)
        self.rel_coords_log = nnx.Variable(jnp.asarray(np.sign(table) * np.log1p(np.abs(table))))

    def __call__(self):
        coords = self.rel_coords_log[...]
        bias = self.fc2(self.drop(nnx.relu(self.fc1(coords)))).reshape(-1, self.num_heads)
        bias = bias[_relative_position_index(self.window)]
        return jnp.transpose(bias, (2, 0, 1))[None]


def _rel_pos(tcfg, window, num_heads, rngs):
    kind = tcfg["rel_pos_type"]
    if kind == "mlp":
        return RelPosMlp(window, num_heads, tcfg["rel_pos_dim"], rngs=rngs)
    if kind == "bias_tf":
        return RelPosBiasTf(window, num_heads, rngs=rngs)
    if kind == "bias":
        return RelPosBias(window, num_heads, rngs=rngs)
    return None


# ----------------------------------------------------------------------------- attention


class Attention(nnx.Module):
    """Fused-QKV attention over flattened tokens (timm Attention2d / AttentionCl)."""

    def __init__(self, dim, dim_out, dim_attn, dim_head, bias, head_first, rel_pos, *, rngs):
        self.num_heads, self.dim_head, self.head_first = dim_attn // dim_head, dim_head, head_first
        self.qkv = nnx.Linear(dim, dim_attn * 3, use_bias=bias, rngs=rngs)
        self.rel_pos = rel_pos
        self.proj = nnx.Linear(dim_attn, dim_out, use_bias=bias, rngs=rngs)

    def __call__(self, x):
        batch, tokens, _ = x.shape
        h, d = self.num_heads, self.dim_head
        if self.head_first:
            qkv = self.qkv(x).reshape(batch, tokens, h, 3, d)
            q, k, v = qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :]
        else:
            qkv = self.qkv(x).reshape(batch, tokens, 3, h, d)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        bias = self.rel_pos() if self.rel_pos is not None else None
        x = dot_product_attention(q, k, v, bias=bias)
        return self.proj(x.reshape(batch, tokens, -1))


def _attn_cl(dim, dim_out, tcfg, window, rngs):
    dim_attn = dim_out if tcfg["expand_first"] and dim_out > dim else dim
    rel = _rel_pos(tcfg, window, dim_attn // tcfg["dim_head"], rngs)
    return Attention(
        dim, dim_out, dim_attn, tcfg["dim_head"], tcfg["attn_bias"], tcfg["head_first"], rel,
        rngs=rngs,
    )  # fmt: skip


def _layer_scale(dim, value):
    return nnx.Param(jnp.full((dim,), value)) if value else None


def _scaled(x, gamma):
    return x if gamma is None else x * gamma[...]


class Mlp(nnx.Module):
    def __init__(self, dim, hidden, act, bias=True, *, rngs):
        self.fc1 = nnx.Linear(dim, hidden, use_bias=bias, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, dim, use_bias=bias, rngs=rngs)
        self.act = act

    def __call__(self, x):
        return self.fc2(_ACTS[self.act](self.fc1(x)))


# ----------------------------------------------------------------------------- conv blocks


class MbConvBlock(nnx.Module):
    """Pre-norm inverted bottleneck (1x1, depthwise kxk, 1x1) with squeeze-excite."""

    def __init__(self, in_chs, out_chs, stride, ccfg, drop_path, *, rngs):
        self.cfg = ccfg
        eps, kind, pad = ccfg["norm_eps"], ccfg["norm_layer"], ccfg["padding"]
        mid = make_divisible((out_chs if ccfg["expand_output"] else in_chs) * ccfg["expand_ratio"])
        groups = mid // ccfg["group_size"] if ccfg["group_size"] else 1
        self.shortcut = (
            Downsample2d(in_chs, out_chs, ccfg["pool_type"], pad, ccfg["output_bias"], rngs=rngs)
            if stride == 2
            else None
        )
        self.pool = stride == 2 and ccfg["stride_mode"] == "pool"
        s1 = stride if ccfg["stride_mode"] == "1x1" else 1
        s2 = stride if ccfg["stride_mode"] == "dw" else 1
        self.pre_norm = _norm(kind, in_chs, eps, rngs=rngs)
        self.conv1_1x1 = _conv(in_chs, mid, 1, s1, bias=False, rngs=rngs)
        self.norm1 = _norm(kind, mid, eps, rngs=rngs)
        self.conv2_kxk = _conv(mid, mid, ccfg["kernel_size"], s2, groups, False, pad, rngs=rngs)
        rd = int(ccfg["attn_ratio"] * (out_chs if ccfg["expand_output"] else mid))
        self.se = SqueezeExcite(mid, rd_channels=rd, act=_ACTS[ccfg["attn_act_layer"]], rngs=rngs)
        self.norm2 = _norm(kind, mid, eps, rngs=rngs)
        self.conv3_1x1 = _conv(mid, out_chs, bias=ccfg["output_bias"], rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        act = _ACTS[self.cfg["act_layer"]]
        shortcut = x if self.shortcut is None else self.shortcut(x)
        x = self.pre_norm(x)
        if self.cfg["pre_norm_act"]:
            x = act(x)
        if self.pool:
            x = _pool(x, self.cfg["downsample_pool_type"], self.cfg["padding"])
        x = self.conv2_kxk(act(self.norm1(self.conv1_1x1(x))))
        if self.cfg["attn_early"]:
            x = self.se(x)
        x = act(self.norm2(x))
        if not self.cfg["attn_early"]:
            x = self.se(x)
        return self.drop_path(self.conv3_1x1(x)) + shortcut


class ConvNeXtBlock(nnx.Module):
    def __init__(self, in_chs, out_chs, stride, ccfg, drop_path, *, rngs):
        self.cfg = ccfg
        if stride == 2:
            self.shortcut = Downsample2d(in_chs, out_chs, rngs=rngs)
        elif in_chs != out_chs:
            self.shortcut = _conv(in_chs, out_chs, bias=ccfg["output_bias"], rngs=rngs)
        else:
            self.shortcut = None
        self.pool = stride == 2 and ccfg["stride_mode"] == "pool"
        sdw = stride if ccfg["stride_mode"] == "dw" else 1
        self.conv_dw = _conv(in_chs, out_chs, 7, sdw, in_chs, ccfg["output_bias"], rngs=rngs)
        self.norm = _norm(ccfg["norm_layer"], out_chs, ccfg["norm_eps"], rngs=rngs)
        hidden = int(ccfg["expand_ratio"] * out_chs)
        self.mlp = Mlp(out_chs, hidden, ccfg["act_layer"], ccfg["output_bias"], rngs=rngs)
        self.ls = _layer_scale(out_chs, ccfg["init_values"])
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        shortcut = x if self.shortcut is None else self.shortcut(x)
        if self.pool:
            x = _pool(x, self.cfg["downsample_pool_type"])
        x = _scaled(self.mlp(self.norm(self.conv_dw(x))), self.ls)
        return self.drop_path(x) + shortcut


def _conv_block(in_chs, out_chs, stride, ccfg, drop_path, rngs):
    cls = ConvNeXtBlock if ccfg["block_type"] == "convnext" else MbConvBlock
    return cls(in_chs, out_chs, stride, ccfg, drop_path, rngs=rngs)


# ----------------------------------------------------------------------------- transformer blocks


class TransformerBlock2d(nnx.Module):
    """CoAtNet transformer block attending across the whole (pooled) feature map."""

    def __init__(self, dim, dim_out, stride, feat_size, tcfg, drop_path, *, rngs):
        self.tcfg = tcfg
        eps, kind = tcfg["norm_eps"], tcfg["norm_layer"]
        self.downsample = stride == 2
        self.shortcut = (
            Downsample2d(dim, dim_out, tcfg["pool_type"], bias=tcfg["shortcut_bias"], rngs=rngs)
            if self.downsample
            else None
        )
        self.norm1 = _norm(kind, dim, eps, rngs=rngs)
        dim_attn = dim_out if tcfg["expand_first"] else dim
        rel = _rel_pos(tcfg, feat_size, dim_attn // tcfg["dim_head"], rngs)
        self.attn = Attention(
            dim, dim_out, dim_attn, tcfg["dim_head"], tcfg["attn_bias"], tcfg["head_first"], rel,
            rngs=rngs,
        )  # fmt: skip
        self.ls1 = _layer_scale(dim_out, tcfg["init_values"])
        self.drop_path1 = DropPath(drop_path, rngs=rngs)
        self.norm2 = _norm(kind, dim_out, eps, rngs=rngs)
        self.mlp = Mlp(dim_out, int(dim_out * tcfg["expand_ratio"]), tcfg["act_layer"], rngs=rngs)
        self.ls2 = _layer_scale(dim_out, tcfg["init_values"])
        self.drop_path2 = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        shortcut = x if self.shortcut is None else self.shortcut(x)
        y = self.norm1(x)
        if self.downsample:
            y = _pool(y, self.tcfg["pool_type"])
        batch, rows, cols, chs = y.shape
        y = self.attn(y.reshape(batch, rows * cols, chs)).reshape(batch, rows, cols, -1)
        x = shortcut + self.drop_path1(_scaled(y, self.ls1))
        return x + self.drop_path2(_scaled(self.mlp(self.norm2(x)), self.ls2))


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

    def __init__(self, dim, partition, block, tcfg, drop_path, *, rngs):
        self.block, self.partition = block, tuple(partition)
        eps = tcfg["norm_eps"]
        self.norm1 = nnx.LayerNorm(dim, epsilon=eps, rngs=rngs)
        self.attn = _attn_cl(dim, dim, tcfg, partition, rngs)
        self.ls1 = _layer_scale(dim, tcfg["init_values"])
        self.drop_path1 = DropPath(drop_path, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, epsilon=eps, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * tcfg["expand_ratio"]), tcfg["act_layer"], rngs=rngs)
        self.ls2 = _layer_scale(dim, tcfg["init_values"])
        self.drop_path2 = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        _, rows, cols, _ = x.shape
        split, merge = (
            (_window_partition, _window_reverse) if self.block else (_grid_partition, _grid_reverse)
        )
        y = merge(self.attn(split(self.norm1(x), self.partition)), self.partition, rows, cols)
        x = x + self.drop_path1(_scaled(y, self.ls1))
        return x + self.drop_path2(_scaled(self.mlp(self.norm2(x)), self.ls2))


class ParallelPartitionAttention(nnx.Module):
    """Window and grid attention on the same input, each producing half the channels."""

    def __init__(self, dim, partition, tcfg, drop_path, *, rngs):
        self.partition = tuple(partition)
        eps = tcfg["norm_eps"]
        self.norm1 = nnx.LayerNorm(dim, epsilon=eps, rngs=rngs)
        self.attn_block = _attn_cl(dim, dim // 2, tcfg, partition, rngs)
        self.attn_grid = _attn_cl(dim, dim // 2, tcfg, partition, rngs)
        self.ls1 = _layer_scale(dim, tcfg["init_values"])
        self.drop_path1 = DropPath(drop_path, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, epsilon=eps, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * tcfg["expand_ratio"]), tcfg["act_layer"], rngs=rngs)
        self.ls2 = _layer_scale(dim, tcfg["init_values"])
        self.drop_path2 = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        _, rows, cols, _ = x.shape
        y = self.norm1(x)
        p = self.partition
        xw = _window_reverse(self.attn_block(_window_partition(y, p)), p, rows, cols)
        xg = _grid_reverse(self.attn_grid(_grid_partition(y, p)), p, rows, cols)
        x = x + self.drop_path1(_scaled(jnp.concatenate([xw, xg], axis=-1), self.ls1))
        return x + self.drop_path2(_scaled(self.mlp(self.norm2(x)), self.ls2))


class MaxxVitBlock(nnx.Module):
    """Conv block, then window attention (unless disabled), then grid attention."""

    def __init__(self, dim, dim_out, stride, partition, ccfg, tcfg, drop_path, *, rngs):
        self.conv = _conv_block(dim, dim_out, stride, ccfg, drop_path, rngs)
        self.attn_block = (
            None
            if tcfg["no_block_attn"]
            else PartitionAttention(dim_out, partition, True, tcfg, drop_path, rngs=rngs)
        )
        self.attn_grid = PartitionAttention(dim_out, partition, False, tcfg, drop_path, rngs=rngs)

    def __call__(self, x):
        x = self.conv(x)
        if self.attn_block is not None:
            x = self.attn_block(x)
        return self.attn_grid(x)


class ParallelMaxxVitBlock(nnx.Module):
    def __init__(self, dim, dim_out, stride, partition, ccfg, tcfg, drop_path, *, rngs):
        self.conv = nnx.List(
            [
                _conv_block(dim, dim_out, stride, ccfg, drop_path, rngs),
                _conv_block(dim_out, dim_out, 1, ccfg, drop_path, rngs),
            ]
        )
        self.attn = ParallelPartitionAttention(dim_out, partition, tcfg, drop_path, rngs=rngs)

    def __call__(self, x):
        for conv in self.conv:
            x = conv(x)
        return self.attn(x)


class Stem(nnx.Module):
    def __init__(self, in_chs, out_chs, ccfg, bias, *, rngs):
        out_chs = out_chs if isinstance(out_chs, (list, tuple)) else (out_chs, out_chs)
        pad = ccfg["padding"]
        self.act = ccfg["act_layer"]
        self.conv1 = _conv(in_chs, out_chs[0], 3, 2, bias=bias, padding=pad, rngs=rngs)
        self.norm1 = _norm(ccfg["norm_layer"], out_chs[0], ccfg["norm_eps"], rngs=rngs)
        self.conv2 = _conv(out_chs[0], out_chs[1], 3, 1, bias=bias, padding=pad, rngs=rngs)
        self.out_chs = out_chs[1]

    def __call__(self, x):
        return self.conv2(_ACTS[self.act](self.norm1(self.conv1(x))))


class MaxxVit(ClassifierMixin, nnx.Module):
    _classifier_attr = "fc"

    def __init__(
        self,
        embed_dim,
        depths,
        block_type,
        conv_overrides=None,
        transformer_overrides=None,
        stem_width=64,
        stem_bias=False,
        head_hidden_size=None,
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
        ccfg = conv_cfg(**(conv_overrides or {}))
        tcfg = transformer_cfg(**(transformer_overrides or {}))
        partition = tcfg["window_size"] or (img_size // tcfg["partition_ratio"],) * 2
        self.num_features = embed_dim[-1]
        self.stem = Stem(in_chans, stem_width, ccfg, stem_bias, rngs=rngs)
        feat = img_size // 2
        total = sum(depths)
        rates = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]
        stages, in_chs, offset = [], self.stem.out_chs, 0
        for dim, depth, kinds in zip(embed_dim, depths, block_type):
            feat = (feat - 1) // 2 + 1
            kinds = (kinds,) if isinstance(kinds, str) else tuple(kinds)
            kinds = kinds + (kinds[-1],) * (depth - len(kinds))
            blocks = []
            for j, kind in enumerate(kinds[:depth]):
                stride, rate = 2 if j == 0 else 1, rates[offset + j]
                if kind == "C":
                    block = _conv_block(in_chs, dim, stride, ccfg, rate, rngs)
                elif kind == "T":
                    block = TransformerBlock2d(
                        in_chs, dim, stride, (feat, feat), tcfg, rate, rngs=rngs
                    )
                elif kind == "M":
                    block = MaxxVitBlock(
                        in_chs, dim, stride, partition, ccfg, tcfg, rate, rngs=rngs
                    )
                else:
                    block = ParallelMaxxVitBlock(
                        in_chs, dim, stride, partition, ccfg, tcfg, rate, rngs=rngs
                    )
                blocks.append(block)
                in_chs = dim
            stages.append(nnx.List(blocks))
            offset += depth
        self.stages = nnx.List(stages)
        norm = _norm(tcfg["norm_layer"], self.num_features, tcfg["norm_eps"], rngs=rngs)
        if head_hidden_size:
            self.norm = None
            self.head_norm = norm
            self.pre_logits = nnx.Linear(self.num_features, head_hidden_size, rngs=rngs)
            self.num_features = head_hidden_size
        else:
            self.norm = norm
            self.head_norm = self.pre_logits = None
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = self._make_fc(num_classes, rngs)

    def _make_fc(self, num_classes, rngs):
        return nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.global_pool = global_pool if global_pool is not None else self.global_pool
        self.fc = self._make_fc(num_classes, nnx.Rngs(0))

    def forward_features(self, x):
        x = self.stem(x)
        for stage in self.stages:
            for block in stage:
                x = block(x)
        return x if self.norm is None else self.norm(x)

    def forward_head(self, x):
        if self.global_pool == "avg":
            x = x.mean(axis=(1, 2))
        elif self.global_pool == "max":
            x = x.max(axis=(1, 2))
        if self.head_norm is not None:
            x = jnp.tanh(self.pre_logits(self.head_norm(x)))
        x = self.head_drop(x)
        return self.fc(x) if self.fc is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))
