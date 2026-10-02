"""Visformer in flax nnx, NHWC. Mirrors timm.models.visformer.

A convolutional stem and attention-free spatial-MLP stage at 1/8 resolution precede
two BatchNorm transformer stages at 1/16 and 1/32, each entered through a strided
patch embedding with learned positional embeddings.
"""

import jax.numpy as jnp
from flax import nnx

from ..attention import dot_product_attention
from ..layers import BatchNorm, ClassifierMixin, DropPath, gelu
from ..registry import _cfg, register_model


class SpatialMlp(nnx.Module):
    """1x1 MLP, optionally with a grouped 3x3 convolution between the projections."""

    def __init__(self, dim, hidden, spatial_conv, group=8, drop=0.0, *, rngs):
        if spatial_conv:
            hidden = dim * 2 if group >= 2 else dim * 5 // 6
        self.conv1 = nnx.Conv(dim, hidden, (1, 1), use_bias=False, rngs=rngs)
        self.conv2 = (
            nnx.Conv(
                hidden,
                hidden,
                (3, 3),
                padding=((1, 1), (1, 1)),
                feature_group_count=group,
                use_bias=False,
                rngs=rngs,
            )
            if spatial_conv
            else None
        )
        self.conv3 = nnx.Conv(hidden, dim, (1, 1), use_bias=False, rngs=rngs)
        self.drop = nnx.Dropout(drop, rngs=rngs)

    def __call__(self, x):
        x = self.drop(gelu(self.conv1(x)))
        if self.conv2 is not None:
            x = gelu(self.conv2(x))
        return self.drop(self.conv3(x))


class Attention(nnx.Module):
    def __init__(self, dim, num_heads, head_dim_ratio=1.0, *, rngs):
        self.num_heads = num_heads
        self.head_dim = round(dim // num_heads * head_dim_ratio)
        self.qkv = nnx.Linear(dim, self.head_dim * num_heads * 3, use_bias=False, rngs=rngs)
        self.proj = nnx.Linear(self.head_dim * num_heads, dim, use_bias=False, rngs=rngs)

    def __call__(self, x):
        batch, rows, cols, _ = x.shape
        # timm orders QKV channels as (q/k/v, head, head dim).
        qkv = self.qkv(x).reshape(batch, rows * cols, 3, self.num_heads, self.head_dim)
        x = dot_product_attention(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2])
        return self.proj(x.reshape(batch, rows, cols, -1))


class Block(nnx.Module):
    def __init__(
        self,
        dim,
        num_heads,
        head_dim_ratio,
        mlp_ratio,
        attn,
        spatial_conv,
        group,
        drop_path,
        *,
        rngs,
    ):
        self.norm1 = BatchNorm(dim, rngs=rngs) if attn else None
        self.attn = Attention(dim, num_heads, head_dim_ratio, rngs=rngs) if attn else None
        self.norm2 = BatchNorm(dim, rngs=rngs)
        self.mlp = SpatialMlp(dim, int(dim * mlp_ratio), spatial_conv, group, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        if self.attn is not None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchEmbed(nnx.Module):
    """Strided convolution followed by BatchNorm."""

    def __init__(self, in_chs, dim, patch, *, rngs):
        self.proj = nnx.Conv(in_chs, dim, (patch, patch), strides=(patch, patch), rngs=rngs)
        self.norm = BatchNorm(dim, rngs=rngs)

    def __call__(self, x):
        return self.norm(self.proj(x))


class Visformer(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"

    def __init__(
        self,
        img_size=224,
        init_channels=32,
        embed_dim=384,
        depth=(7, 4, 4),
        num_heads=6,
        mlp_ratio=4.0,
        group=8,
        attn_stage="011",
        spatial_conv="100",
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dim * 2
        self.stem_conv = nnx.Conv(
            in_chans,
            init_channels,
            (7, 7),
            strides=(2, 2),
            padding=((3, 3), (3, 3)),
            use_bias=False,
            rngs=rngs,
        )
        self.stem_norm = BatchNorm(init_channels, rngs=rngs)
        dims = (embed_dim // 2, embed_dim, embed_dim * 2)
        patches = (4, 2, 2)
        sizes = (img_size // 8, img_size // 16, img_size // 32)
        rates = [drop_path_rate * i / max(sum(depth) - 1, 1) for i in range(sum(depth))]
        embeds, stages, in_chs, offset = [], [], init_channels, 0
        for i, (dim, patch, size, n) in enumerate(zip(dims, patches, sizes, depth)):
            embeds.append(PatchEmbed(in_chs, dim, patch, rngs=rngs))
            setattr(self, f"pos_embed{i + 1}", nnx.Param(jnp.zeros((1, size, size, dim))))
            stages.append(
                nnx.List(
                    [
                        Block(
                            dim,
                            num_heads,
                            0.5 if i == 0 else 1.0,
                            mlp_ratio,
                            attn_stage[i] == "1",
                            spatial_conv[i] == "1",
                            group,
                            rates[offset + j],
                            rngs=rngs,
                        )
                        for j in range(n)
                    ]
                )
            )
            in_chs, offset = dim, offset + n
        self.patch_embeds = nnx.List(embeds)
        self.stages = nnx.List(stages)
        self.pos_drop = nnx.Dropout(0.0, rngs=rngs)
        self.norm = BatchNorm(self.num_features, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = (
            nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None
        )

    def forward_features(self, x):
        x = nnx.relu(self.stem_norm(self.stem_conv(x)))
        for i, (embed, stage) in enumerate(zip(self.patch_embeds, self.stages)):
            x = self.pos_drop(embed(x) + getattr(self, f"pos_embed{i + 1}")[...])
            for block in stage:
                x = block(x)
        return self.norm(x)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_CFGS = {  # timm: stem width, embed width, heads
    "visformer_tiny": dict(init_channels=16, embed_dim=192, num_heads=3),
    "visformer_small": dict(init_channels=32, embed_dim=384, num_heads=6),
}


def _make(name):
    def entry(**kwargs):
        model = Visformer(**_CFGS[name], **kwargs)
        model.default_cfg = _cfg(crop_pct=0.9, interpolation="bicubic")
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
