"""HieraDet (the SAM2 Hiera backbone) in flax nnx, NHWC. Mirrors timm.models.hieradet_sam2.

Windowed attention on zero-padded NHWC maps with global attention in selected
blocks, 2x2 max-pooled queries and shortcuts at stage transitions (each block
uses the window size of the previous block's stage, as in timm), and a
position embedding resized with PyTorch's bicubic kernel plus a tiled window
embedding.
"""

import math
from functools import lru_cache

import jax.numpy as jnp
import numpy as np
from flax import nnx

from ..attention import dot_product_attention
from ..layers import ClassifierMixin, DropPath, Mlp, global_pool_nhwc
from ..registry import _cfg, register_model


def _trunc_normal(std):
    return nnx.initializers.truncated_normal(std)


# timm init_weight_vit: truncated-normal weights and 0.02 Linear biases.
_bias_init = nnx.initializers.constant(0.02)


@lru_cache
def _bicubic_matrix(in_size, out_size):
    """PyTorch bicubic resize (align_corners=False, A=-0.75, clamped edges) as (out, in)."""
    a = -0.75

    def near(t):  # |t| <= 1
        return ((a + 2) * t - (a + 3)) * t * t + 1

    def far(t):  # 1 < |t| < 2
        return ((a * t - 5 * a) * t + 8 * a) * t - 4 * a

    weights = np.zeros((out_size, in_size))
    for o in range(out_size):
        src = in_size / out_size * (o + 0.5) - 0.5
        i0 = math.floor(src)
        t = src - i0
        for k, w in enumerate((far(t + 1), near(t), near(1 - t), far(2 - t))):
            weights[o, min(max(i0 - 1 + k, 0), in_size - 1)] += w
    return weights


def _max_pool_2x2(x):
    return nnx.max_pool(x, (2, 2), strides=(2, 2), padding="VALID")


def _window_partition(x, ws):
    B, H, W, C = x.shape
    x = x.reshape(B, H // ws, ws, W // ws, ws, C).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, ws, ws, C)


def _window_unpartition(windows, ws, hw):
    H, W = hw
    B = windows.shape[0] // (H * W // ws // ws)
    x = windows.reshape(B, H // ws, W // ws, ws, ws, -1).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(B, H, W, -1)


def _pad_to(size, window):
    return size + (-size) % window


class MultiScaleAttention(nnx.Module):
    def __init__(self, dim, dim_out, num_heads, q_pool=False, proj_std=0.02, *, rngs):
        self.num_heads, self.q_pool = num_heads, q_pool
        self.qkv = nnx.Linear(
            dim, dim_out * 3, kernel_init=_trunc_normal(0.02), bias_init=_bias_init, rngs=rngs
        )
        self.proj = nnx.Linear(
            dim_out, dim_out, kernel_init=_trunc_normal(proj_std), bias_init=_bias_init, rngs=rngs
        )

    def __call__(self, x):
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        if self.q_pool:
            q = _max_pool_2x2(q.reshape(B, H, W, -1))
            H, W = q.shape[1:3]
            q = q.reshape(B, H * W, self.num_heads, -1)
        return self.proj(dot_product_attention(q, k, v).reshape(B, H, W, -1))


class MultiScaleBlock(nnx.Module):
    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        mlp_ratio=4.0,
        q_pool=False,
        window_size=0,
        init_values=None,
        drop_path=0.0,
        layer_id=0,
        *,
        rngs,
    ):
        # timm fix_init_weight: output projections shrink with depth.
        out_std = 0.02 / math.sqrt(2.0 * (layer_id + 1))
        self.window_size, self.q_pool = window_size, q_pool
        self.norm1 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.proj = (
            nnx.Linear(
                dim, dim_out, kernel_init=_trunc_normal(0.02), bias_init=_bias_init, rngs=rngs
            )
            if dim != dim_out
            else None
        )
        self.attn = MultiScaleAttention(
            dim, dim_out, num_heads, q_pool=q_pool, proj_std=out_std, rngs=rngs
        )
        self.norm2 = nnx.LayerNorm(dim_out, epsilon=1e-6, rngs=rngs)
        self.mlp = Mlp(
            dim_out,
            int(dim_out * mlp_ratio),
            kernel_init=_trunc_normal(0.02),
            out_kernel_init=_trunc_normal(out_std),
            bias_init=_bias_init,
            rngs=rngs,
        )
        self.gamma1 = nnx.Param(jnp.full((dim_out,), init_values)) if init_values else None
        self.gamma2 = nnx.Param(jnp.full((dim_out,), init_values)) if init_values else None
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        shortcut = x
        x = self.norm1(x)
        if self.proj is not None:
            shortcut = self.proj(x)
            if self.q_pool:
                shortcut = _max_pool_2x2(shortcut)
        ws = self.window_size
        H, W = x.shape[1:3]
        if ws:
            # Zero-pad to whole windows; padded tokens stay visible as keys, as in timm.
            x = jnp.pad(x, ((0, 0), (0, (-H) % ws), (0, (-W) % ws), (0, 0)))
            x = _window_partition(x, ws)
        x = self.attn(x)
        if ws:
            if self.q_pool:
                ws //= 2
            H, W = shortcut.shape[1:3]
            x = _window_unpartition(x, ws, (_pad_to(H, ws), _pad_to(W, ws)))[:, :H, :W]
        x = x if self.gamma1 is None else self.gamma1[...] * x
        x = shortcut + self.drop_path(x)
        y = self.mlp(self.norm2(x))
        return x + self.drop_path(y if self.gamma2 is None else self.gamma2[...] * y)


class HieraDet(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        global_pool="avg",
        embed_dim=96,
        num_heads=1,
        q_pool=3,
        stages=(2, 3, 16, 3),
        global_pos_size=(7, 7),
        window_spec=(8, 4, 14, 7),
        global_att_blocks=(12, 16, 20),
        init_values=None,
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.patch_embed = nnx.Conv(
            in_chans,
            embed_dim,
            (7, 7),
            strides=(4, 4),
            padding=((3, 3), (3, 3)),
            kernel_init=_trunc_normal(0.02),
            rngs=rngs,
        )
        self.pos_embed = nnx.Param(
            _trunc_normal(0.02)(rngs.params(), (1, *global_pos_size, embed_dim))
        )
        self.pos_embed_window = nnx.Param(
            _trunc_normal(0.02)(rngs.params(), (1, window_spec[0], window_spec[0], embed_dim))
        )
        stage_ends = [sum(stages[: i + 1]) - 1 for i in range(len(stages))]
        q_pool_blocks = [end + 1 for end in stage_ends[:-1]][:q_pool]
        depth = sum(stages)
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        blocks, cur_stage = [], 0
        for i in range(depth):
            dim_out = embed_dim
            window_size = window_spec[cur_stage]  # lags one block behind the stage
            if global_att_blocks is not None and i in global_att_blocks:
                window_size = 0
            if i - 1 in stage_ends:
                dim_out, num_heads, cur_stage = embed_dim * 2, num_heads * 2, cur_stage + 1
            blocks.append(
                MultiScaleBlock(
                    embed_dim,
                    dim_out,
                    num_heads,
                    mlp_ratio,
                    q_pool=i in q_pool_blocks,
                    window_size=window_size,
                    init_values=init_values,
                    drop_path=dpr[i],
                    layer_id=i,
                    rngs=rngs,
                )
            )
            embed_dim = dim_out
        self.blocks = nnx.List(blocks)
        self.num_features = embed_dim
        # timm ClNormMlpClassifierHead: pool, LayerNorm, dropout, and a zero-initialized fc.
        self.head_norm = nnx.LayerNorm(embed_dim, epsilon=1e-6, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = (
            nnx.Linear(embed_dim, num_classes, kernel_init=nnx.initializers.zeros, rngs=rngs)
            if num_classes > 0
            else None
        )

    def _pos_embed(self, x):
        h, w = x.shape[1:3]
        pos = self.pos_embed[...]
        rows = jnp.asarray(_bicubic_matrix(pos.shape[1], h), dtype=pos.dtype)
        cols = jnp.asarray(_bicubic_matrix(pos.shape[2], w), dtype=pos.dtype)
        pos = jnp.einsum("oh,bhwc,pw->bopc", rows, pos, cols)
        win = self.pos_embed_window[...]
        pos = pos + jnp.tile(win, (1, h // win.shape[1], w // win.shape[2], 1))
        return x + pos

    def forward_features(self, x):
        x = self._pos_embed(self.patch_embed(x))
        for blk in self.blocks:
            x = blk(x)
        return x

    def forward_head(self, x):
        x = self.head_norm(global_pool_nhwc(x, self.global_pool))
        x = self.head_drop(x)
        return self.fc(x) if self.fc is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_CFGS = {
    "sam2_hiera_tiny": (dict(stages=(1, 2, 7, 2), global_att_blocks=(5, 7, 9)), 896),
    "sam2_hiera_small": (dict(stages=(1, 2, 11, 2), global_att_blocks=(7, 10, 13)), 896),
    "sam2_hiera_base_plus": (dict(embed_dim=112, num_heads=2, global_pos_size=(14, 14)), 896),
    "sam2_hiera_large": (
        dict(
            embed_dim=144,
            num_heads=2,
            stages=(2, 6, 36, 4),
            global_att_blocks=(23, 33, 43),
            window_spec=(8, 4, 16, 8),
        ),
        1024,
    ),
    "hieradet_small": (
        dict(
            stages=(1, 2, 11, 2),
            global_att_blocks=(7, 10, 13),
            window_spec=(8, 4, 16, 8),
            init_values=1e-5,
        ),
        256,
    ),
}


def _make(name):
    cfg, img_size = _CFGS[name]

    def entry(**kwargs):
        model = HieraDet(**dict(cfg, **kwargs))
        model.default_cfg = _cfg(
            input_size=(3, img_size, img_size), crop_pct=1.0, interpolation="bicubic"
        )
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
