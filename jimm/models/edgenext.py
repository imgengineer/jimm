"""EdgeNeXt in flax nnx, NHWC. Mirrors timm.models.edgenext.

Convolutional encoder blocks (depthwise kxk convolution, LayerNorm, MLP,
layer scale) are followed in later stages by split depth-wise transpose
attention blocks: channels are split into groups processed by a cascade of
depthwise 3x3 convolutions (Res2Net style), then cross-covariance attention
(attention over channels with L2-normalized queries and keys and a learned
per-head temperature), optionally with a Fourier positional encoding, and an
MLP. Stages after the first downsample with LayerNorm and a 2x2 strided
convolution; the head pools, normalizes, and classifies.
"""

import math

import jax
import jax.numpy as jnp
from flax import nnx

from ..layers import ClassifierMixin, DropPath, Mlp, global_pool_nhwc
from ..registry import _cfg, register_model

_init = nnx.initializers.truncated_normal(0.02)


def _dw_conv(in_chs, out_chs, kernel, stride=1, use_bias=True, *, rngs):
    pad = (stride - 1 + kernel - 1) // 2
    return nnx.Conv(
        in_chs,
        out_chs,
        (kernel, kernel),
        strides=(stride, stride),
        padding=((pad, pad), (pad, pad)),
        feature_group_count=in_chs,
        use_bias=use_bias,
        kernel_init=_init,
        rngs=rngs,
    )


def _l2_normalize(x, axis):
    # torch.nn.functional.normalize: x / max(||x||, 1e-12)
    return x / jnp.maximum(jnp.linalg.norm(x, axis=axis, keepdims=True), 1e-12)


class PositionalEncodingFourier(nnx.Module):
    def __init__(self, hidden_dim=32, dim=768, temperature=10000.0, *, rngs):
        self.hidden_dim, self.temperature = hidden_dim, temperature
        self.token_projection = nnx.Linear(2 * hidden_dim, dim, kernel_init=_init, rngs=rngs)

    def __call__(self, h, w):
        eps, scale = 1e-6, 2 * math.pi
        y = jnp.arange(1, h + 1, dtype=jnp.float32) / (h + eps) * scale
        x = jnp.arange(1, w + 1, dtype=jnp.float32) / (w + eps) * scale
        dim_t = jnp.arange(self.hidden_dim, dtype=jnp.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        def encode(pos):  # sin on even, cos on odd features, interleaved
            pos = pos[:, None] / dim_t
            return jnp.stack([jnp.sin(pos[:, 0::2]), jnp.cos(pos[:, 1::2])], axis=-1).reshape(
                pos.shape[0], -1
            )

        pos_y = jnp.broadcast_to(encode(y)[:, None], (h, w, self.hidden_dim))
        pos_x = jnp.broadcast_to(encode(x)[None], (h, w, self.hidden_dim))
        return self.token_projection(jnp.concatenate([pos_y, pos_x], axis=-1))


class ConvBlock(nnx.Module):
    def __init__(
        self,
        dim,
        dim_out=None,
        kernel=7,
        stride=1,
        conv_bias=True,
        expand_ratio=4.0,
        ls_init_value=1e-6,
        drop_path=0.0,
        *,
        rngs,
    ):
        dim_out = dim_out or dim
        self.shortcut_after_dw = stride > 1 or dim != dim_out
        self.conv_dw = _dw_conv(dim, dim_out, kernel, stride, conv_bias, rngs=rngs)
        self.norm = nnx.LayerNorm(dim_out, epsilon=1e-6, rngs=rngs)
        self.mlp = Mlp(dim_out, int(expand_ratio * dim_out), kernel_init=_init, rngs=rngs)
        self.gamma = nnx.Param(jnp.full((dim_out,), ls_init_value)) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        shortcut = x
        x = self.conv_dw(x)
        if self.shortcut_after_dw:
            shortcut = x
        x = self.mlp(self.norm(x))
        if self.gamma is not None:
            x = self.gamma[...] * x
        return shortcut + self.drop_path(x)


class CrossCovarianceAttn(nnx.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, *, rngs):
        self.num_heads = num_heads
        self.temperature = nnx.Param(jnp.ones((num_heads, 1, 1)))
        self.qkv = nnx.Linear(dim, 3 * dim, use_bias=qkv_bias, kernel_init=_init, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, kernel_init=_init, rngs=rngs)

    def __call__(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        # Attention between channels: queries and keys are normalized over the tokens.
        q, k = _l2_normalize(q, axis=1), _l2_normalize(k, axis=1)
        attn = jnp.einsum("bnhd,bnhe->bhde", q, k) * self.temperature[...]
        attn = jax.nn.softmax(attn, axis=-1)
        x = jnp.einsum("bhde,bnhe->bnhd", attn, v)
        return self.proj(x.reshape(B, N, C))


class SplitTransposeBlock(nnx.Module):
    def __init__(
        self,
        dim,
        num_scales=1,
        num_heads=8,
        expand_ratio=4.0,
        use_pos_emb=True,
        conv_bias=True,
        qkv_bias=True,
        ls_init_value=1e-6,
        drop_path=0.0,
        *,
        rngs,
    ):
        width = math.ceil(dim / num_scales)
        self.width = width
        self.convs = nnx.List(
            [
                _dw_conv(width, width, 3, use_bias=conv_bias, rngs=rngs)
                for _ in range(max(1, num_scales - 1))
            ]
        )
        self.pos_embd = PositionalEncodingFourier(dim=dim, rngs=rngs) if use_pos_emb else None
        self.norm_xca = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.gamma_xca = nnx.Param(jnp.full((dim,), ls_init_value)) if ls_init_value > 0 else None
        self.xca = CrossCovarianceAttn(dim, num_heads, qkv_bias, rngs=rngs)
        self.norm = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.mlp = Mlp(dim, int(expand_ratio * dim), kernel_init=_init, rngs=rngs)
        self.gamma = nnx.Param(jnp.full((dim,), ls_init_value)) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        shortcut = x
        B, H, W, C = x.shape
        # torch.chunk: equal splits of ``width`` channels, the last one possibly smaller.
        splits = jnp.split(x, [self.width * (i + 1) for i in range(len(self.convs))], axis=-1)
        outs, sp = [], splits[0]
        for i, conv in enumerate(self.convs):
            if i > 0:
                sp = sp + splits[i]
            sp = conv(sp)
            outs.append(sp)
        x = jnp.concatenate([*outs, splits[-1]], axis=-1).reshape(B, H * W, C)
        if self.pos_embd is not None:
            x = x + self.pos_embd(H, W).reshape(1, H * W, C)
        y = self.xca(self.norm_xca(x))
        x = x + self.drop_path(y if self.gamma_xca is None else self.gamma_xca[...] * y)
        x = self.mlp(self.norm(x.reshape(B, H, W, C)))
        if self.gamma is not None:
            x = self.gamma[...] * x
        return shortcut + self.drop_path(x)


class EdgeNeXtStage(nnx.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        stride=2,
        depth=2,
        num_global_blocks=1,
        num_heads=4,
        scales=2,
        kernel=7,
        expand_ratio=4.0,
        use_pos_emb=False,
        downsample_block=False,
        conv_bias=True,
        ls_init_value=1.0,
        drop_path_rates=None,
        *,
        rngs,
    ):
        if downsample_block or stride == 1:
            self.downsample = None
        else:
            self.downsample = nnx.List(
                [
                    nnx.LayerNorm(in_chs, epsilon=1e-6, rngs=rngs),
                    nnx.Conv(
                        in_chs,
                        out_chs,
                        (2, 2),
                        strides=(2, 2),
                        padding="VALID",
                        use_bias=conv_bias,
                        kernel_init=_init,
                        rngs=rngs,
                    ),
                ]
            )
            in_chs = out_chs
        drop_path_rates = drop_path_rates or [0.0] * depth
        blocks = []
        for i in range(depth):
            if i < depth - num_global_blocks:
                blocks.append(
                    ConvBlock(
                        in_chs,
                        out_chs,
                        kernel,
                        stride if downsample_block and i == 0 else 1,
                        conv_bias,
                        expand_ratio,
                        ls_init_value,
                        drop_path_rates[i],
                        rngs=rngs,
                    )
                )
            else:
                blocks.append(
                    SplitTransposeBlock(
                        in_chs,
                        scales,
                        num_heads,
                        expand_ratio,
                        use_pos_emb,
                        conv_bias,
                        ls_init_value=ls_init_value,
                        drop_path=drop_path_rates[i],
                        rngs=rngs,
                    )
                )
            in_chs = out_chs
        self.blocks = nnx.List(blocks)

    def __call__(self, x):
        if self.downsample is not None:
            for layer in self.downsample:
                x = layer(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class EdgeNeXt(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"

    def __init__(
        self,
        dims=(24, 48, 88, 168),
        depths=(3, 3, 9, 3),
        global_block_counts=(0, 1, 1, 1),
        kernel_sizes=(3, 5, 7, 9),
        heads=(8, 8, 8, 8),
        d2_scales=(2, 2, 3, 4),
        use_pos_emb=(False, True, False, False),
        ls_init_value=1e-6,
        expand_ratio=4.0,
        downsample_block=False,
        conv_bias=True,
        stem_type="patch",
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        kernel, pad = (4, 0) if stem_type == "patch" else (9, 4)
        self.stem_conv = nnx.Conv(
            in_chans,
            dims[0],
            (kernel, kernel),
            strides=(4, 4),
            padding=((pad, pad), (pad, pad)),
            use_bias=conv_bias,
            kernel_init=_init,
            rngs=rngs,
        )
        self.stem_norm = nnx.LayerNorm(dims[0], epsilon=1e-6, rngs=rngs)
        # timm calculate_drop_path_rates(stagewise=True): linear over all blocks, split by stage.
        total = sum(depths)
        rates = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]
        stages, in_chs = [], dims[0]
        for i in range(4):
            start = sum(depths[:i])
            stages.append(
                EdgeNeXtStage(
                    in_chs,
                    dims[i],
                    stride=2 if i > 0 else 1,
                    depth=depths[i],
                    num_global_blocks=global_block_counts[i],
                    num_heads=heads[i],
                    scales=d2_scales[i],
                    kernel=kernel_sizes[i],
                    expand_ratio=expand_ratio,
                    use_pos_emb=use_pos_emb[i],
                    downsample_block=downsample_block,
                    conv_bias=conv_bias,
                    ls_init_value=ls_init_value,
                    drop_path_rates=rates[start : start + depths[i]],
                    rngs=rngs,
                )
            )
            in_chs = dims[i]
        self.stages = nnx.List(stages)
        self.num_features = dims[-1]
        self.head_norm = nnx.LayerNorm(self.num_features, epsilon=1e-6, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = (
            nnx.Linear(self.num_features, num_classes, kernel_init=_init, rngs=rngs)
            if num_classes > 0
            else None
        )

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


_CFGS = {
    "edgenext_xx_small": dict(depths=(2, 2, 6, 2), dims=(24, 48, 88, 168), heads=(4, 4, 4, 4)),
    "edgenext_x_small": dict(depths=(3, 3, 9, 3), dims=(32, 64, 100, 192), heads=(4, 4, 4, 4)),
    "edgenext_small": dict(depths=(3, 3, 9, 3), dims=(48, 96, 160, 304)),
    "edgenext_base": dict(depths=(3, 3, 9, 3), dims=(80, 160, 288, 584)),
    "edgenext_small_rw": dict(
        depths=(3, 3, 9, 3),
        dims=(48, 96, 192, 384),
        downsample_block=True,
        conv_bias=False,
        stem_type="overlap",
    ),
}


def _make(name):
    cfg = _CFGS[name]

    def entry(**kwargs):
        model = EdgeNeXt(**{**cfg, **kwargs})
        model.default_cfg = _cfg(input_size=(3, 256, 256), crop_pct=0.9, interpolation="bicubic")
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
