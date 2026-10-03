"""NesT (Nested Hierarchical Transformer) in flax nnx, NHWC. Mirrors timm.models.nest.

A 4x4 patch embedding is split into 16, 4, and finally 1 non-overlapping
blocks of 14x14 tokens across three levels. Each level adds a learned
position embedding per block and runs transformer layers that attend only
within a block; between levels, block aggregation (a 3x3 convolution,
LayerNorm, and a strided 3x3 max pool) halves the resolution. The head
averages the normalized feature map. The ``_jx`` variants pad the pooling
like TensorFlow's SAME.
"""

from flax import nnx

from ..attention import dot_product_attention
from ..layers import ClassifierMixin, DropPath, Mlp, global_pool_nhwc
from ..registry import _cfg, register_model

_init = nnx.initializers.truncated_normal(0.02)


class Attention(nnx.Module):
    """Self-attention within each image block of a (B, T, N, C) input."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, drop=0.0, *, rngs):
        self.num_heads = num_heads
        self.qkv = nnx.Linear(dim, 3 * dim, use_bias=qkv_bias, kernel_init=_init, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, kernel_init=_init, rngs=rngs)
        self.drop = nnx.Dropout(drop, rngs=rngs)

    def __call__(self, x):
        B, T, N, C = x.shape
        qkv = self.qkv(x).reshape(B * T, N, 3, self.num_heads, C // self.num_heads)
        x = dot_product_attention(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2])
        # timm (after the original Jax code) orders the merged channels as (head_dim, heads).
        x = x.reshape(B, T, N, self.num_heads, C // self.num_heads).swapaxes(-1, -2)
        return self.drop(self.proj(x.reshape(B, T, N, C)))


class TransformerLayer(nnx.Module):
    def __init__(
        self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0, drop_path=0.0, *, rngs
    ):
        self.norm1 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.attn = Attention(dim, num_heads, qkv_bias, drop, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop, kernel_init=_init, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class ConvPool(nnx.Module):
    def __init__(self, in_chs, out_chs, pad_type="", *, rngs):
        self.pad_type = pad_type
        self.conv = nnx.Conv(
            in_chs, out_chs, (3, 3), padding=((1, 1), (1, 1)), kernel_init=_init, rngs=rngs
        )
        self.norm = nnx.LayerNorm(out_chs, epsilon=1e-6, rngs=rngs)

    def __call__(self, x):
        x = self.norm(self.conv(x))
        # PyTorch pads both sides; TensorFlow SAME pads the extra row and column at the end.
        padding = "SAME" if self.pad_type == "same" else ((1, 1), (1, 1))
        return nnx.max_pool(x, (3, 3), strides=(2, 2), padding=padding)


def blockify(x, block_size):
    B, H, W, C = x.shape
    gh, gw = H // block_size, W // block_size
    x = x.reshape(B, gh, block_size, gw, block_size, C).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(B, gh * gw, block_size * block_size, C)


def deblockify(x, block_size):
    B, T, _, C = x.shape
    grid = int(round(T**0.5))
    x = x.reshape(B, grid, grid, block_size, block_size, C).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(B, grid * block_size, grid * block_size, C)


class NestLevel(nnx.Module):
    def __init__(
        self,
        num_blocks,
        block_size,
        seq_length,
        num_heads,
        depth,
        embed_dim,
        prev_embed_dim=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        drop_path=None,
        pad_type="",
        *,
        rngs,
    ):
        self.block_size = block_size
        self.pos_embed = nnx.Param(_init(rngs.params(), (1, num_blocks, seq_length, embed_dim)))
        self.pool = (
            ConvPool(prev_embed_dim, embed_dim, pad_type, rngs=rngs)
            if prev_embed_dim is not None
            else None
        )
        drop_path = drop_path or [0.0] * depth
        self.transformer_encoder = nnx.List(
            [
                TransformerLayer(
                    embed_dim, num_heads, mlp_ratio, qkv_bias, drop, drop_path[i], rngs=rngs
                )
                for i in range(depth)
            ]
        )

    def __call__(self, x):
        if self.pool is not None:
            x = self.pool(x)
        x = blockify(x, self.block_size) + self.pos_embed[...]
        for layer in self.transformer_encoder:
            x = layer(x)
        return deblockify(x, self.block_size)


class Nest(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=4,
        num_levels=3,
        embed_dims=(128, 256, 512),
        num_heads=(4, 8, 16),
        depths=(2, 2, 20),
        mlp_ratio=4.0,
        qkv_bias=True,
        pad_type="",
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.5,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        grid = img_size // patch_size
        num_blocks = [4 ** (num_levels - 1 - i) for i in range(num_levels)]
        self.block_size = grid // int(round(num_blocks[0] ** 0.5))
        seq_length = grid * grid // num_blocks[0]
        self.patch_embed = nnx.Conv(
            in_chans,
            embed_dims[0],
            (patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            kernel_init=_init,
            rngs=rngs,
        )
        # timm calculate_drop_path_rates(stagewise=True): linear over all layers, split by level.
        total = sum(depths)
        rates = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]
        levels, prev = [], None
        for i in range(num_levels):
            start = sum(depths[:i])
            levels.append(
                NestLevel(
                    num_blocks[i],
                    self.block_size,
                    seq_length,
                    num_heads[i],
                    depths[i],
                    embed_dims[i],
                    prev,
                    mlp_ratio,
                    qkv_bias,
                    drop_rate,
                    rates[start : start + depths[i]],
                    pad_type,
                    rngs=rngs,
                )
            )
            prev = embed_dims[i]
        self.levels = nnx.List(levels)
        self.num_features = embed_dims[-1]
        self.norm = nnx.LayerNorm(self.num_features, epsilon=1e-6, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = (
            nnx.Linear(self.num_features, num_classes, kernel_init=_init, rngs=rngs)
            if num_classes > 0
            else None
        )

    def forward_features(self, x):
        x = self.patch_embed(x)
        for level in self.levels:
            x = level(x)
        return self.norm(x)

    def forward_head(self, x):
        x = self.head_drop(global_pool_nhwc(x, self.global_pool))
        return self.fc(x) if self.fc is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_CFGS = {  # embed_dims, num_heads, depths
    "nest_base": ((128, 256, 512), (4, 8, 16), (2, 2, 20)),
    "nest_small": ((96, 192, 384), (3, 6, 12), (2, 2, 20)),
    "nest_tiny": ((96, 192, 384), (3, 6, 12), (2, 2, 8)),
}


def _make(name, pad_type=""):
    embed_dims, num_heads, depths = _CFGS[name.removesuffix("_jx")]

    def entry(**kwargs):
        kwargs.setdefault("pad_type", pad_type)
        model = Nest(embed_dims=embed_dims, num_heads=num_heads, depths=depths, **kwargs)
        model.default_cfg = _cfg(interpolation="bicubic", fixed_input_size=True)
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
    register_model(_make(f"{_name}_jx", pad_type="same"))
