"""Global Context ViT in flax nnx, NHWC. Mirrors timm.models.gcvit.

Stages alternate local window attention with global-query attention, where
the queries come from a feature block (MBConv + max pooling) that shrinks the
stage's feature map to the window size, so every window attends with image
level context. Both use Swin-style relative position biases. MBConv blocks
with squeeze-excite and strided convolutions between LayerNorms downsample.
"""

import math

import jax.numpy as jnp
import numpy as np
from flax import nnx

from ..layers import ClassifierMixin, DropPath, Mlp, gelu, global_pool_nhwc, make_divisible
from ..registry import _cfg, register_model

_init = nnx.initializers.truncated_normal(0.02)


def _relative_position_index(ws):
    coords = np.stack(np.meshgrid(np.arange(ws), np.arange(ws), indexing="ij")).reshape(2, -1)
    rel = (coords[:, :, None] - coords[:, None, :]).transpose(1, 2, 0) + (ws - 1)
    return rel[..., 0] * (2 * ws - 1) + rel[..., 1]


class SEModule(nnx.Module):
    """timm SEModule with bias-free 1x1 convolutions and GELU."""

    def __init__(self, chs, *, rngs):
        rd = make_divisible(chs * 0.25, 8, round_limit=0.0)
        self.fc1 = nnx.Linear(chs, rd, use_bias=False, rngs=rngs)
        self.fc2 = nnx.Linear(rd, chs, use_bias=False, rngs=rngs)

    def __call__(self, x):
        s = self.fc2(gelu(self.fc1(jnp.mean(x, axis=(1, 2), keepdims=True))))
        return x * nnx.sigmoid(s)


class MbConvBlock(nnx.Module):
    def __init__(self, chs, *, rngs):
        self.conv_dw = nnx.Conv(
            chs,
            chs,
            (3, 3),
            padding=((1, 1), (1, 1)),
            feature_group_count=chs,
            use_bias=False,
            rngs=rngs,
        )
        self.se = SEModule(chs, rngs=rngs)
        self.conv_pw = nnx.Conv(chs, chs, (1, 1), use_bias=False, rngs=rngs)

    def __call__(self, x):
        return x + self.conv_pw(self.se(gelu(self.conv_dw(x))))


def _max_pool(x):
    return nnx.max_pool(x, (3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))


class Downsample2d(nnx.Module):
    def __init__(self, dim, dim_out=None, eps=1e-5, *, rngs):
        dim_out = dim_out or dim
        self.norm1 = nnx.LayerNorm(dim, epsilon=eps, rngs=rngs)
        self.conv_block = MbConvBlock(dim, rngs=rngs)
        self.reduction = nnx.Conv(
            dim,
            dim_out,
            (3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(dim_out, epsilon=eps, rngs=rngs)

    def __call__(self, x):
        return self.norm2(self.reduction(self.conv_block(self.norm1(x))))


class FeatureBlock(nnx.Module):
    """MBConv blocks, each followed by max pooling while ``levels`` reductions remain."""

    def __init__(self, dim, levels=0, *, rngs):
        self.reductions = levels
        self.blocks = nnx.List([MbConvBlock(dim, rngs=rngs) for _ in range(max(1, levels))])

    def __call__(self, x):
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i < self.reductions:
                x = _max_pool(x)
        return x


class Stem(nnx.Module):
    def __init__(self, in_chs, out_chs, eps=1e-5, *, rngs):
        self.conv1 = nnx.Conv(
            in_chs, out_chs, (3, 3), strides=(2, 2), padding=((1, 1), (1, 1)), rngs=rngs
        )
        self.down = Downsample2d(out_chs, eps=eps, rngs=rngs)

    def __call__(self, x):
        return self.down(self.conv1(x))


class WindowAttentionGlobal(nnx.Module):
    def __init__(self, dim, num_heads, window_size, use_global=True, qkv_bias=True, *, rngs):
        self.num_heads, self.window_size, self.use_global = num_heads, window_size, use_global
        self.relative_position_bias_table = nnx.Param(
            _init(rngs.params(), ((2 * window_size - 1) ** 2, num_heads))
        )
        width = 2 * dim if use_global else 3 * dim
        self.qkv = nnx.Linear(dim, width, use_bias=qkv_bias, kernel_init=_init, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, kernel_init=_init, rngs=rngs)

    def __call__(self, x, q_global=None):
        B, N, C = x.shape
        h, d = self.num_heads, C // self.num_heads
        if self.use_global and q_global is not None:
            kv = self.qkv(x).reshape(B, N, 2, h, d)
            k, v = kv[:, :, 0], kv[:, :, 1]
            # timm tiles the per-image global queries over the whole window batch, so window
            # i uses the query of image i % batch.
            q = jnp.tile(q_global, (B // q_global.shape[0], 1, 1, 1)).reshape(B, N, h, d)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, h, d)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        attn = jnp.einsum("bqhd,bkhd->bhqk", q * d**-0.5, k)
        bias = self.relative_position_bias_table[...][_relative_position_index(self.window_size)]
        attn = nnx.softmax(attn + bias.transpose(2, 0, 1)[None], axis=-1)
        return self.proj(jnp.einsum("bhqk,bkhd->bqhd", attn, v).reshape(B, N, C))


class GlobalContextVitBlock(nnx.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        use_global=True,
        qkv_bias=True,
        layer_scale=None,
        drop=0.0,
        drop_path=0.0,
        eps=1e-5,
        *,
        rngs,
    ):
        self.window_size = window_size
        self.norm1 = nnx.LayerNorm(dim, epsilon=eps, rngs=rngs)
        self.attn = WindowAttentionGlobal(
            dim, num_heads, window_size, use_global, qkv_bias, rngs=rngs
        )
        self.norm2 = nnx.LayerNorm(dim, epsilon=eps, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop, kernel_init=_init, rngs=rngs)
        if layer_scale is not None:
            self.ls1 = nnx.Param(jnp.full((dim,), layer_scale))
            self.ls2 = nnx.Param(jnp.full((dim,), layer_scale))
        else:
            self.ls1 = self.ls2 = None
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def _window_attn(self, x, q_global):
        B, H, W, C = x.shape
        ws = self.window_size
        nh, nw = H // ws, W // ws
        x = x.reshape(B, nh, ws, nw, ws, C).transpose(0, 1, 3, 2, 4, 5)
        x = self.attn(x.reshape(B * nh * nw, ws * ws, C), q_global)
        x = x.reshape(B, nh, nw, ws, ws, C).transpose(0, 1, 3, 2, 4, 5)
        return x.reshape(B, H, W, C)

    def __call__(self, x, q_global=None):
        y = self._window_attn(self.norm1(x), q_global)
        if self.ls1 is not None:
            y = self.ls1[...] * y
        x = x + self.drop_path(y)
        y = self.mlp(self.norm2(x))
        if self.ls2 is not None:
            y = self.ls2[...] * y
        return x + self.drop_path(y)


class GlobalContextVitStage(nnx.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        feat_size,
        window_size,
        downsample=True,
        stage_norm=False,
        mlp_ratio=4.0,
        qkv_bias=True,
        layer_scale=None,
        drop=0.0,
        drop_path=None,
        eps=1e-5,
        *,
        rngs,
    ):
        if downsample:
            self.downsample = Downsample2d(dim, 2 * dim, eps, rngs=rngs)
            dim, feat_size = 2 * dim, feat_size // 2
        else:
            self.downsample = None
        self.dim = dim
        levels = int(math.log2(feat_size / window_size))
        self.global_block = FeatureBlock(dim, levels, rngs=rngs)
        drop_path = drop_path or [0.0] * depth
        self.blocks = nnx.List(
            [
                GlobalContextVitBlock(
                    dim,
                    num_heads,
                    window_size,
                    mlp_ratio,
                    use_global=i % 2 != 0,
                    qkv_bias=qkv_bias,
                    layer_scale=layer_scale,
                    drop=drop,
                    drop_path=drop_path[i],
                    eps=eps,
                    rngs=rngs,
                )
                for i in range(depth)
            ]
        )
        self.norm = nnx.LayerNorm(dim, epsilon=eps, rngs=rngs) if stage_norm else None

    def __call__(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        global_query = self.global_block(x)
        for blk in self.blocks:
            x = blk(x, global_query)
        return x if self.norm is None else self.norm(x)


class GlobalContextVit(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        img_size=224,
        window_ratio=(32, 32, 16, 32),
        embed_dim=64,
        depths=(3, 4, 19, 5),
        num_heads=(2, 4, 8, 16),
        mlp_ratio=3.0,
        qkv_bias=True,
        layer_scale=None,
        norm_eps=1e-5,
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
        feat_size = img_size // 4
        window_sizes = [img_size // r for r in window_ratio]
        self.stem = Stem(in_chans, embed_dim, norm_eps, rngs=rngs)
        # timm calculate_drop_path_rates(stagewise=True): linear over all blocks, split by stage.
        total = sum(depths)
        rates = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]
        stages = []
        for i, depth in enumerate(depths):
            scale = 2 ** max(i - 1, 0)
            start = sum(depths[:i])
            stages.append(
                GlobalContextVitStage(
                    embed_dim * scale,
                    depth,
                    num_heads[i],
                    feat_size // scale,
                    window_sizes[i],
                    downsample=i != 0,
                    stage_norm=i == len(depths) - 1,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    layer_scale=layer_scale,
                    drop=proj_drop_rate,
                    drop_path=rates[start : start + depth],
                    eps=norm_eps,
                    rngs=rngs,
                )
            )
        self.stages = nnx.List(stages)
        self.num_features = stages[-1].dim
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = (
            nnx.Linear(self.num_features, num_classes, kernel_init=_init, rngs=rngs)
            if num_classes > 0
            else None
        )

    def forward_features(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x

    def forward_head(self, x):
        x = self.head_drop(global_pool_nhwc(x, self.global_pool))
        return self.fc(x) if self.fc is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_CFGS = {
    "gcvit_xxtiny": dict(depths=(2, 2, 6, 2), num_heads=(2, 4, 8, 16)),
    "gcvit_xtiny": dict(depths=(3, 4, 6, 5), num_heads=(2, 4, 8, 16)),
    "gcvit_tiny": dict(depths=(3, 4, 19, 5), num_heads=(2, 4, 8, 16)),
    "gcvit_small": dict(
        depths=(3, 4, 19, 5), num_heads=(3, 6, 12, 24), embed_dim=96, mlp_ratio=2, layer_scale=1e-5
    ),
    "gcvit_base": dict(
        depths=(3, 4, 19, 5), num_heads=(4, 8, 16, 32), embed_dim=128, mlp_ratio=2, layer_scale=1e-5
    ),
}


def _make(name):
    cfg = _CFGS[name]

    def entry(**kwargs):
        model = GlobalContextVit(**{**cfg, **kwargs})
        model.default_cfg = _cfg(interpolation="bicubic", fixed_input_size=True)
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
