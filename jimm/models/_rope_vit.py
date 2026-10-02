"""Dynamic patches and axial half-rotation RoPE shared by Qwen3 and DeepSeek ViTs.

Adapted for JAX from timm 1.0.30, Copyright 2019 Ross Wightman.
"""

import jax.numpy as jnp
from flax import nnx

from ..attention import dot_product_attention


class DynamicPatchEmbed(nnx.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, dynamic_img_pad=False, *, rngs):
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        self.patch_size = patch_size
        self.dynamic_img_pad = dynamic_img_pad
        self.proj = nnx.Conv(
            in_chans,
            embed_dim,
            (patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            rngs=rngs,
        )

    def __call__(self, x):
        height, width = x.shape[1:3]
        if self.dynamic_img_pad:
            x = jnp.pad(
                x, ((0, 0), (0, -height % self.patch_size), (0, -width % self.patch_size), (0, 0))
            )
        elif height % self.patch_size or width % self.patch_size:
            raise ValueError("image height and width must be divisible by patch_size")
        return self.proj(x)


def axial_rope(height, width, head_dim, temperature=10000.0):
    """Integer patch coordinates, axis frequency blocks, and half-rotation layout."""
    bands = temperature ** (-jnp.arange(head_dim // 4, dtype=jnp.float32) / (head_dim // 4))
    h, w = jnp.meshgrid(
        jnp.arange(height, dtype=jnp.float32), jnp.arange(width, dtype=jnp.float32), indexing="ij"
    )
    phase = jnp.concatenate((h[..., None] * bands, w[..., None] * bands), axis=-1)
    phase = jnp.tile(phase.reshape(height * width, head_dim // 2), (1, 2))
    return jnp.sin(phase), jnp.cos(phase)


def apply_rope(x, rope):
    sin, cos = rope
    calc = x.astype(jnp.float32)
    first, second = jnp.split(calc, 2, axis=-1)
    rotated = jnp.concatenate((-second, first), axis=-1)
    return (calc * cos[:, None, :] + rotated * sin[:, None, :]).astype(x.dtype)


def resample_pos_embed_grid(pos_embed, grid_size, new_size):
    """Bilinear position interpolation with align_corners=True, matching Qwen."""
    height, width = new_size
    if new_size == (grid_size, grid_size):
        return pos_embed
    dtype = pos_embed.dtype
    grid = pos_embed.astype(jnp.float32).reshape(1, grid_size, grid_size, -1)
    rows = jnp.linspace(0, grid_size - 1, height)
    cols = jnp.linspace(0, grid_size - 1, width)
    r0, c0 = jnp.floor(rows).astype(jnp.int32), jnp.floor(cols).astype(jnp.int32)
    r1, c1 = jnp.minimum(r0 + 1, grid_size - 1), jnp.minimum(c0 + 1, grid_size - 1)
    rw, cw = (rows - r0)[None, :, None, None], (cols - c0)[None, None, :, None]
    grid = grid[:, r0] * (1 - rw) + grid[:, r1] * rw
    grid = grid[:, :, c0] * (1 - cw) + grid[:, :, c1] * cw
    return grid.reshape(1, height * width, -1).astype(dtype)


class RopeAttention(nnx.Module):
    def __init__(self, dim, num_heads, proj_drop=0.0, attn_drop=0.0, *, rngs):
        if dim % num_heads or (dim // num_heads) % 4:
            raise ValueError("axial RoPE requires a head dimension divisible by four")
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.qkv = nnx.Linear(dim, 3 * dim, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)
        self.proj_drop = nnx.Dropout(proj_drop, rngs=rngs)
        self.attn_drop = nnx.Dropout(attn_drop, rngs=rngs)
        self.attn_drop_rate = attn_drop

    def __call__(self, x, rope):
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q, k = apply_rope(q, rope), apply_rope(k, rope)
        if self.attn_drop_rate and not self.attn_drop.deterministic:
            q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
            attn = nnx.softmax((q * self.head_dim**-0.5) @ k.swapaxes(-1, -2), axis=-1)
            x = (self.attn_drop(attn) @ v).transpose(0, 2, 1, 3)
        else:
            x = dot_product_attention(q, k, v)
        return self.proj_drop(self.proj(x.reshape(batch, tokens, dim)))
