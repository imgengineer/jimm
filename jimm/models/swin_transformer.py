"""Swin Transformer in flax nnx, NHWC. Mirrors timm.models.swin_transformer."""
import jax.numpy as jnp
from flax import nnx

from ..layers import DropPath, Mlp, global_pool_nhwc
from ..registry import register_model, _cfg


def window_partition(x, ws):
    """(B,H,W,C) -> (B*nH*nW, ws*ws, C)."""
    B, H, W, C = x.shape
    x = x.reshape(B, H // ws, ws, W // ws, ws, C).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, ws * ws, C)


def window_reverse(windows, ws, H, W, B):
    x = windows.reshape(B, H // ws, W // ws, ws, ws, -1).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(B, H, W, -1)


class WindowAttention(nnx.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, drop=0.0, *, rngs):
        self.ws, self.num_heads = window_size, num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nnx.Linear(dim, dim * 3, use_bias=qkv_bias, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)
        self.drop = nnx.Dropout(drop)
        n = (2 * window_size - 1) ** 2
        self.rel_bias_table = nnx.Param(jnp.zeros((n, num_heads)))
        # relative position index, fixed for a given window size
        coords = jnp.stack(jnp.meshgrid(jnp.arange(window_size), jnp.arange(window_size), indexing="ij"))
        coords_flat = coords.reshape(2, -1)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel = rel.transpose(1, 2, 0) + window_size - 1
        self.rel_index = rel[:, :, 0] * (2 * window_size - 1) + rel[:, :, 1]

    def __call__(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(0, 1, 3, 2) * self.scale
        bias = self.rel_bias_table.value[self.rel_index].transpose(2, 0, 1)  # (heads, N, N)
        attn = attn + bias[None]
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(B // nW, nW, self.num_heads, N, N) + mask[:, None, :, :]
            attn = attn.reshape(B, self.num_heads, N, N)
        attn = nnx.softmax(attn, axis=-1)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.drop(self.proj(x))


class SwinBlock(nnx.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift=0,
                 mlp_ratio=4.0, drop=0.0, drop_path=0.0, *, rngs):
        self.dim, self.res = dim, input_resolution
        self.ws, self.shift = window_size, shift
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = WindowAttention(dim, window_size, num_heads, rngs=rngs)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop, rngs=rngs)
        attn_mask = None
        if shift > 0:
            # mask for SW-MSA, precomputed for fixed resolution (timm does the same)
            H, W = input_resolution
            img_mask = jnp.zeros((1, H, W, 1))
            slices = [(slice(0, -window_size), slice(0, -window_size)),
                      (slice(0, -window_size), slice(-window_size, None)),
                      (slice(-window_size, None), slice(0, -window_size)),
                      (slice(-window_size, None), slice(-window_size, None))]
            for i, (hs, ws_) in enumerate(slices):
                img_mask = img_mask.at[:, hs, ws_, :].set(i)
            mask_windows = window_partition(img_mask, window_size).reshape(-1, window_size * window_size)
            m = mask_windows[:, None, :] - mask_windows[:, :, None]
            attn_mask = jnp.where(m != 0, -100.0, 0.0)
        self.attn_mask = attn_mask

    def __call__(self, x):
        B, H, W, C = x.shape
        sc = x
        x = self.norm1(x)
        if self.shift > 0:
            x = jnp.roll(x, (-self.shift, -self.shift), axis=(1, 2))
        x = window_partition(x, self.ws)
        x = self.attn(x, self.attn_mask)
        x = window_reverse(x, self.ws, H, W, B)
        if self.shift > 0:
            x = jnp.roll(x, (self.shift, self.shift), axis=(1, 2))
        x = sc + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchMerging(nnx.Module):
    def __init__(self, dim, *, rngs):
        self.norm = nnx.LayerNorm(4 * dim, rngs=rngs)
        self.reduction = nnx.Linear(4 * dim, 2 * dim, use_bias=False, rngs=rngs)

    def __call__(self, x):
        B, H, W, C = x.shape
        x0, x1 = x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :]
        x2, x3 = x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]
        x = jnp.concatenate([x0, x1, x2, x3], axis=-1)
        return self.reduction(self.norm(x))


class SwinStage(nnx.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4.0, drop=0.0, drop_path=None, downsample=True, *, rngs):
        drop_path = drop_path or [0.0] * depth
        self.blocks = nnx.List([
            SwinBlock(dim, input_resolution, num_heads, window_size,
                      shift=0 if i % 2 == 0 else window_size // 2,
                      mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path[i], rngs=rngs)
            for i in range(depth)])
        self.downsample = PatchMerging(dim, rngs=rngs) if downsample else None

    def __call__(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class SwinTransformer(nnx.Module):
    default_cfg: dict = {}

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 global_pool="avg", embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4.0, drop_rate=0.0, drop_path_rate=0.1, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dim * 2 ** (len(depths) - 1)
        self.patch_embed_conv = nnx.Conv(in_chans, embed_dim, (patch_size, patch_size),
                                         strides=(patch_size, patch_size), rngs=rngs)
        self.patch_norm = nnx.LayerNorm(embed_dim, rngs=rngs)
        res = img_size // patch_size
        dpr = [drop_path_rate * i / max(sum(depths) - 1, 1) for i in range(sum(depths))]
        self.stages = nnx.List([
            SwinStage(embed_dim * 2**i, (res // 2**i, res // 2**i), depths[i], num_heads[i],
                      window_size, mlp_ratio, drop_rate,
                      dpr[sum(depths[:i]):sum(depths[:i + 1])],
                      downsample=i < len(depths) - 1, rngs=rngs)
            for i in range(len(depths))])
        self.norm = nnx.LayerNorm(self.num_features, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate)
        self.head = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.patch_norm(self.patch_embed_conv(x))
        for stage in self.stages:
            x = stage(x)
        return self.norm(x)

    def forward_head(self, x):
        x = global_pool_nhwc(x, self.global_pool)
        x = self.head_drop(x)
        return self.head(x) if self.head is not None else x

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool="avg"):
        self.num_classes, self.global_pool = num_classes, global_pool
        if num_classes > 0 and self.head is None:
            raise RuntimeError("cannot re-add classifier to a num_classes=0 model")
        self.head = nnx.Linear(self.num_features, num_classes, rngs=nnx.Rngs(0)) if num_classes > 0 else None

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


def _swin(embed_dim, depths, num_heads, **kwargs):
    model = SwinTransformer(embed_dim=embed_dim, depths=depths, num_heads=num_heads, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def swin_tiny_patch4_window7_224(**kwargs):
    return _swin(96, (2, 2, 6, 2), (3, 6, 12, 24), **kwargs)


@register_model
def swin_small_patch4_window7_224(**kwargs):
    return _swin(96, (2, 2, 18, 2), (3, 6, 12, 24), **kwargs)


@register_model
def swin_base_patch4_window7_224(**kwargs):
    return _swin(128, (2, 2, 18, 2), (4, 8, 16, 32), **kwargs)


# Swin v2: post-norm + cosine attention + continuous rel-pos MLP (structure-level v2)
class SwinV2Attention(WindowAttention):
    def __call__(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q / jnp.maximum(jnp.linalg.norm(q, axis=-1, keepdims=True), 1e-6)
        k = k / jnp.maximum(jnp.linalg.norm(k, axis=-1, keepdims=True), 1e-6)
        attn = q @ k.transpose(0, 1, 3, 2) / 0.5  # cosine attention, fixed logit scale
        bias = self.rel_bias_table.value[self.rel_index].transpose(2, 0, 1)
        attn = attn + bias[None]
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(B // nW, nW, self.num_heads, N, N) + mask[:, None, :, :]
            attn = attn.reshape(B, self.num_heads, N, N)
        attn = nnx.softmax(attn, axis=-1)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.drop(self.proj(x))


class SwinV2Block(SwinBlock):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift=0,
                 mlp_ratio=4.0, drop=0.0, drop_path=0.0, *, rngs):
        super().__init__(dim, input_resolution, num_heads, window_size, shift,
                         mlp_ratio, drop, drop_path, rngs=rngs)
        self.attn = SwinV2Attention(dim, window_size, num_heads, rngs=rngs)

    def __call__(self, x):  # post-norm
        B, H, W, C = x.shape
        sc = x
        if self.shift > 0:
            x = jnp.roll(x, (-self.shift, -self.shift), axis=(1, 2))
        x = window_partition(x, self.ws)
        x = self.attn(x, self.attn_mask)
        x = window_reverse(x, self.ws, H, W, B)
        if self.shift > 0:
            x = jnp.roll(x, (self.shift, self.shift), axis=(1, 2))
        x = sc + self.drop_path(x)
        x = self.norm1(x)
        return self.norm2(x + self.drop_path(self.mlp(x)))


class SwinTransformerV2(SwinTransformer):
    default_cfg: dict = {}


def _swinv2(embed_dim, depths, num_heads, img_size=256, **kwargs):
    model = SwinTransformer(embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                            img_size=img_size, **kwargs)
    # swap in v2 blocks
    for stage in model.stages:
        for i, blk in enumerate(stage.blocks):
            stage.blocks[i] = SwinV2Block(blk.dim, blk.res, blk.attn.num_heads, blk.ws,
                                          blk.shift, rngs=nnx.Rngs(0))
    model.default_cfg = _cfg(input_size=(3, img_size, img_size))
    return model


@register_model
def swinv2_tiny_window8_256(**kwargs):
    return _swinv2(96, (2, 2, 6, 2), (3, 6, 12, 24), window_size=8, img_size=256, **kwargs)


@register_model
def swinv2_small_window8_256(**kwargs):
    return _swinv2(96, (2, 2, 18, 2), (3, 6, 12, 24), window_size=8, img_size=256, **kwargs)
