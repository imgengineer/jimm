"""ConViT in flax nnx, NHWC. Mirrors timm.models.convit.

The first ``local_up_to_layer`` blocks use gated positional self-attention
(GPSA): per head, a learned gate mixes content attention with a softmax over
a linear function of each token pair's relative offset and squared distance,
initialized to attend locally. The class token joins only before the
remaining blocks, which use standard self-attention; the head classifies the
normalized class token.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from ..attention import dot_product_attention
from ..layers import ClassifierMixin, DropPath, Mlp, PatchEmbed
from ..registry import _cfg, register_model

_init = nnx.initializers.truncated_normal(0.02)


def _rel_indices(grid):
    """(N, N, 3) key-minus-query x offset, y offset, and squared distance (timm order)."""
    ind = np.arange(grid)[None, :] - np.arange(grid)[:, None]
    indx = np.tile(ind, (grid, grid))
    indy = np.repeat(np.repeat(ind, grid, axis=0), grid, axis=1)
    return np.stack([indx, indy, indx**2 + indy**2], axis=-1).astype(np.float32)


class GPSA(nnx.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, locality_strength=1.0, *, rngs):
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qk = nnx.Linear(dim, 2 * dim, use_bias=qkv_bias, kernel_init=_init, rngs=rngs)
        # timm local_init: values start as the identity and positional attention as local.
        self.v = nnx.Linear(
            dim,
            dim,
            use_bias=qkv_bias,
            kernel_init=lambda key, shape, dtype=jnp.float32: jnp.eye(*shape, dtype=dtype),
            rngs=rngs,
        )
        self.proj = nnx.Linear(dim, dim, kernel_init=_init, rngs=rngs)
        local = _local_kernel(num_heads, locality_strength)
        self.pos_proj = nnx.Linear(
            3,
            num_heads,
            kernel_init=lambda key, shape, dtype=jnp.float32: jnp.asarray(local, dtype),
            rngs=rngs,
        )
        self.gating_param = nnx.Param(jnp.ones((num_heads,)))

    def __call__(self, x):
        B, N, C = x.shape
        h = self.num_heads
        qk = self.qk(x).reshape(B, N, 2, h, C // h)
        q, k = qk[:, :, 0], qk[:, :, 1]
        patch = jnp.einsum("bqhd,bkhd->bhqk", q, k).astype(jnp.float32) * self.scale
        grid = int(round(N**0.5))
        pos = self.pos_proj(jnp.asarray(_rel_indices(grid), x.dtype)).transpose(2, 0, 1)
        gate = jax.nn.sigmoid(self.gating_param[...].astype(jnp.float32))[None, :, None, None]
        attn = (1.0 - gate) * jax.nn.softmax(patch, axis=-1) + gate * jax.nn.softmax(
            pos.astype(jnp.float32), axis=-1
        )
        attn = (attn / jnp.sum(attn, axis=-1, keepdims=True)).astype(x.dtype)
        v = self.v(x).reshape(B, N, h, C // h)
        return self.proj(jnp.einsum("bhqk,bkhd->bqhd", attn, v).reshape(B, N, C))


def _local_kernel(num_heads, locality_strength):
    """Flax (3, heads) kernel of timm's GPSA.local_init positional projection."""
    kernel = np.zeros((3, num_heads), np.float32)
    size = int(num_heads**0.5)
    center = (size - 1) / 2 if size % 2 == 0 else size // 2
    for h1 in range(size):
        for h2 in range(size):
            pos = h1 + size * h2
            kernel[2, pos] = -1
            kernel[1, pos] = 2 * (h1 - center)
            kernel[0, pos] = 2 * (h2 - center)
    return kernel * locality_strength


class MHSA(nnx.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, *, rngs):
        self.num_heads = num_heads
        self.qkv = nnx.Linear(dim, 3 * dim, use_bias=qkv_bias, kernel_init=_init, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, kernel_init=_init, rngs=rngs)

    def __call__(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        x = dot_product_attention(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2])
        return self.proj(x.reshape(B, N, C))


class Block(nnx.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        drop_path=0.0,
        use_gpsa=True,
        locality_strength=1.0,
        *,
        rngs,
    ):
        self.norm1 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.attn = (
            GPSA(dim, num_heads, qkv_bias, locality_strength, rngs=rngs)
            if use_gpsa
            else MHSA(dim, num_heads, qkv_bias, rngs=rngs)
        )
        self.norm2 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop, kernel_init=_init, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class ConVit(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"
    _default_global_pool = "token"

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        embed_dim=48,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        local_up_to_layer=3,
        locality_strength=1.0,
        use_pos_embed=True,
        num_classes=1000,
        in_chans=3,
        global_pool="token",
        drop_rate=0.0,
        pos_drop_rate=0.0,
        proj_drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        embed_dim *= num_heads  # timm specifies the per-head width
        self.local_up_to_layer = local_up_to_layer
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, rngs=rngs)
        self.cls_token = nnx.Param(_init(rngs.params(), (1, 1, embed_dim)))
        self.pos_embed = (
            nnx.Param(_init(rngs.params(), (1, self.patch_embed.num_patches, embed_dim)))
            if use_pos_embed
            else None
        )
        self.pos_drop = nnx.Dropout(pos_drop_rate, rngs=rngs)
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nnx.List(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    proj_drop_rate,
                    dpr[i],
                    use_gpsa=i < local_up_to_layer,
                    locality_strength=locality_strength,
                    rngs=rngs,
                )
                for i in range(depth)
            ]
        )
        self.norm = nnx.LayerNorm(embed_dim, epsilon=1e-6, rngs=rngs)
        self.num_features = embed_dim
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = (
            nnx.Linear(embed_dim, num_classes, kernel_init=_init, rngs=rngs)
            if num_classes > 0
            else None
        )

    def forward_features(self, x):
        x = self.patch_embed(x)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        if self.pos_embed is not None:
            x = x + self.pos_embed[...]
        x = self.pos_drop(x)
        cls = jnp.broadcast_to(self.cls_token[...], (B, 1, C))
        for i, blk in enumerate(self.blocks):
            if i == self.local_up_to_layer:
                x = jnp.concatenate([cls, x], axis=1)
            x = blk(x)
        return self.norm(x)

    def forward_head(self, x):
        if self.global_pool:
            x = jnp.mean(x[:, 1:], axis=1) if self.global_pool == "avg" else x[:, 0]
        x = self.head_drop(x)
        return self.head(x) if self.head is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


def _make(name, num_heads):
    def entry(**kwargs):
        model = ConVit(
            embed_dim=48, num_heads=num_heads, local_up_to_layer=10, locality_strength=1.0, **kwargs
        )
        model.default_cfg = _cfg(fixed_input_size=True)
        return model

    entry.__name__ = name
    return entry


for _name, _heads in (("convit_tiny", 4), ("convit_small", 9), ("convit_base", 16)):
    register_model(_make(_name, _heads))
