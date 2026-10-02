"""Hiera in flax nnx. Mirrors timm.models.hiera (Hierarchical Vision Transformer).

Patch tokens are "unrolled" once per stage transition, so each 2x2 query pooling
is a max over the leading axis of the flattened sequence and every 8x8 mask unit
is a contiguous run of tokens. The first two stages attend within mask units,
the rest globally; as in timm, each block uses the attention type of the
previous block's stage, so stage-transition blocks keep the earlier type.
"""

import math

from flax import nnx

from ..attention import dot_product_attention
from ..layers import ClassifierMixin, DropPath, Mlp, global_pool_nhwc
from ..registry import _cfg, register_model


def _trunc_normal(std):
    return nnx.initializers.truncated_normal(std)


# timm init_weight_vit: truncated-normal weights and 0.02 Linear biases.
_bias_init = nnx.initializers.constant(0.02)


def _unroll(x, size, num_unrolls, stride=(2, 2)):
    """(B, H*W, C) raster tokens -> order (sy_1, sx_1, ..., sy_k, sx_k, H / 2^k, W / 2^k)."""
    B, _, C = x.shape
    sy, sx = stride
    x = x.reshape(B, *size, C)
    for _ in range(num_unrolls):
        h, w = x.shape[1] // sy, x.shape[2] // sx
        x = x.reshape(-1, h, sy, w, sx, C).transpose(0, 2, 4, 1, 3, 5).reshape(-1, h, w, C)
    return x.reshape(B, -1, C)


class MaskUnitAttention(nnx.Module):
    """Mask-unit (windowed) or global attention, optionally max-pooling the queries."""

    def __init__(
        self,
        dim,
        dim_out,
        heads,
        q_stride=1,
        window_size=0,
        use_mask_unit_attn=False,
        proj_std=0.02,
        *,
        rngs,
    ):
        self.heads, self.dim_out = heads, dim_out
        self.q_stride, self.window_size = q_stride, window_size
        self.use_mask_unit_attn = use_mask_unit_attn
        self.qkv = nnx.Linear(
            dim, 3 * dim_out, kernel_init=_trunc_normal(0.02), bias_init=_bias_init, rngs=rngs
        )
        self.proj = nnx.Linear(
            dim_out, dim_out, kernel_init=_trunc_normal(proj_std), bias_init=_bias_init, rngs=rngs
        )

    def __call__(self, x):
        B, N, _ = x.shape
        H, D = self.heads, self.dim_out // self.heads
        qkv = self.qkv(x)
        if self.use_mask_unit_attn:
            # Token n = t * num_windows + w: windows are the fastest-varying axis.
            nw = N // (self.q_stride * self.window_size)
            qkv = qkv.reshape(B, -1, nw, 3, H, D).transpose(0, 2, 1, 3, 4, 5)
            q, k, v = qkv[:, :, :, 0], qkv[:, :, :, 1], qkv[:, :, :, 2]
            if self.q_stride > 1:
                q = q.reshape(B, nw, self.q_stride, -1, H, D).max(axis=2)
            tq, tk = q.shape[2], k.shape[2]
            out = dot_product_attention(
                q.reshape(B * nw, tq, H, D),
                k.reshape(B * nw, tk, H, D),
                v.reshape(B * nw, tk, H, D),
            )
            out = out.reshape(B, nw, tq, H * D).transpose(0, 2, 1, 3).reshape(B, -1, H * D)
        else:
            qkv = qkv.reshape(B, N, 3, H, D)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            if self.q_stride > 1:
                q = q.reshape(B, self.q_stride, -1, H, D).max(axis=1)
            out = dot_product_attention(q, k, v).reshape(B, -1, H * D)
        return self.proj(out)


class HieraBlock(nnx.Module):
    def __init__(
        self,
        dim,
        dim_out,
        heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        q_stride=1,
        window_size=0,
        use_mask_unit_attn=False,
        layer_id=0,
        *,
        rngs,
    ):
        # timm fix_init_weight: output projections shrink with depth.
        out_std = 0.02 / math.sqrt(2.0 * (layer_id + 1))
        self.q_stride = q_stride
        self.norm1 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.proj = (
            nnx.Linear(
                dim, dim_out, kernel_init=_trunc_normal(0.02), bias_init=_bias_init, rngs=rngs
            )
            if dim != dim_out
            else None
        )
        self.attn = MaskUnitAttention(
            dim,
            dim_out,
            heads,
            q_stride,
            window_size,
            use_mask_unit_attn,
            proj_std=out_std,
            rngs=rngs,
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
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        x_norm = self.norm1(x)
        if self.proj is not None:
            # Expand channels, then max-pool the shortcut like the queries.
            x = self.proj(x_norm)
            x = x.reshape(x.shape[0], self.q_stride, -1, x.shape[-1]).max(axis=1)
        x = x + self.drop_path(self.attn(x_norm))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class Hiera(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        img_size=224,
        in_chans=3,
        embed_dim=96,
        num_heads=1,
        num_classes=1000,
        global_pool="avg",
        stages=(2, 3, 16, 3),
        q_pool=3,
        q_stride=(2, 2),
        mask_unit_size=(8, 8),
        mask_unit_attn=(True, True, False, False),
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.tokens_shape = (img_size // 4, img_size // 4)
        self.num_unrolls = len(stages) - 1
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
            _trunc_normal(0.02)(rngs.params(), (1, math.prod(self.tokens_shape), embed_dim))
        )
        stage_ends = [sum(stages[: i + 1]) - 1 for i in range(len(stages))]
        q_pool_blocks = [end + 1 for end in stage_ends[:q_pool]]
        flat_q_stride, flat_mu_size = math.prod(q_stride), math.prod(mask_unit_size)
        depth = sum(stages)
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        blocks, cur_stage = [], 0
        for i in range(depth):
            dim_out = embed_dim
            use_mask_unit_attn = mask_unit_attn[cur_stage]  # lags one block behind the stage
            if i - 1 in stage_ends:
                dim_out, num_heads, cur_stage = embed_dim * 2, num_heads * 2, cur_stage + 1
                if i in q_pool_blocks:
                    flat_mu_size //= flat_q_stride
            blocks.append(
                HieraBlock(
                    embed_dim,
                    dim_out,
                    num_heads,
                    mlp_ratio,
                    dpr[i],
                    q_stride=flat_q_stride if i in q_pool_blocks else 1,
                    window_size=flat_mu_size,
                    use_mask_unit_attn=use_mask_unit_attn,
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

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).reshape(B, -1, self.pos_embed.shape[-1]) + self.pos_embed[...]
        x = _unroll(x, self.tokens_shape, self.num_unrolls)
        for blk in self.blocks:
            x = blk(x)
        # All unrolls are consumed by the q-pooling stages: tokens are back in raster order.
        h, w = (s // 2**self.num_unrolls for s in self.tokens_shape)
        return x.reshape(B, h, w, x.shape[-1])

    def forward_head(self, x):
        x = self.head_norm(global_pool_nhwc(x, self.global_pool))
        x = self.head_drop(x)
        return self.fc(x) if self.fc is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_CFGS = {
    "hiera_tiny_224": dict(embed_dim=96, num_heads=1, stages=(1, 2, 7, 2)),
    "hiera_small_224": dict(embed_dim=96, num_heads=1, stages=(1, 2, 11, 2)),
    "hiera_base_224": dict(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3)),
    "hiera_base_plus_224": dict(embed_dim=112, num_heads=2, stages=(2, 3, 16, 3)),
    "hiera_large_224": dict(embed_dim=144, num_heads=2, stages=(2, 6, 36, 4)),
    "hiera_huge_224": dict(embed_dim=256, num_heads=4, stages=(2, 6, 36, 4)),
}


def _make(name):
    cfg = _CFGS[name]

    def entry(**kwargs):
        model = Hiera(**dict(cfg, **kwargs))
        model.default_cfg = _cfg(crop_pct=0.9, interpolation="bicubic")
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
