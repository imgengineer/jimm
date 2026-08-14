"""Swin Transformer V2 (Channel Restricted / CR) in flax nnx, NHWC. Mirrors timm.models.swin_transformer_v2_cr."""
import jax.numpy as jnp
from flax import nnx

from ..layers import DropPath, Mlp, ClassifierMixin
from ..registry import register_model, _cfg
from .swin_transformer import WindowAttention, window_partition, window_reverse

class SwinV2CrAttention(WindowAttention):
    def __call__(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q / jnp.maximum(jnp.linalg.norm(q, axis=-1, keepdims=True), 1e-6)
        k = k / jnp.maximum(jnp.linalg.norm(k, axis=-1, keepdims=True), 1e-6)
        attn = q @ k.transpose(0, 1, 3, 2) / 0.5
        bias = self.rel_bias_table.value[self.rel_index].transpose(2, 0, 1)
        attn = attn + bias[None]
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(B // nW, nW, self.num_heads, N, N) + mask[:, None, :, :]
            attn = attn.reshape(B, self.num_heads, N, N)
        attn = nnx.softmax(attn, axis=-1)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.drop(self.proj(x))

class SwinV2CrBlock(nnx.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift=0,
                 mlp_ratio=4.0, drop=0.0, drop_path=0.0, *, rngs):
        self.dim, self.res = dim, input_resolution
        self.ws, self.shift = window_size, shift
        self.attn = SwinV2CrAttention(dim, window_size, num_heads, rngs=rngs)
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

        attn_mask = None
        if shift > 0:
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

class SwinTransformerV2CrStage(nnx.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4.0, drop=0.0, drop_path=None, downsample=True, *, rngs):
        drop_path = drop_path or [0.0] * depth
        self.blocks = nnx.List([
            SwinV2CrBlock(dim, input_resolution, num_heads, window_size,
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

class SwinTransformerV2Cr(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"
    default_cfg: dict = {}

    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 3, num_classes: int = 1000,
                 global_pool: str = "avg", embed_dim: int = 96, depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24), window_size: int = 7, mlp_ratio: float = 4.0,
                 drop_rate: float = 0.0, drop_path_rate: float = 0.1, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dim * 2 ** (len(depths) - 1)
        self.patch_embed_conv = nnx.Conv(in_chans, embed_dim, (patch_size, patch_size),
                                         strides=(patch_size, patch_size), rngs=rngs)
        self.patch_norm = nnx.LayerNorm(embed_dim, rngs=rngs)
        res = img_size // patch_size
        dpr = [drop_path_rate * i / max(sum(depths) - 1, 1) for i in range(sum(depths))]
        self.stages = nnx.List([
            SwinTransformerV2CrStage(embed_dim * 2**i, (res // 2**i, res // 2**i), depths[i], num_heads[i],
                                    window_size, mlp_ratio, drop_rate,
                                    dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                    downsample=i < len(depths) - 1, rngs=rngs)
            for i in range(len(depths))])
        self.norm = nnx.LayerNorm(self.num_features, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.patch_norm(self.patch_embed_conv(x))
        for stage in self.stages:
            x = stage(x)
        return self.norm(x)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

_CFGS = {
    "swinv2_cr_tiny_224": dict(embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), window_size=7, img_size=224),
    "swinv2_cr_small_224": dict(embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), window_size=7, img_size=224),
    "swinv2_cr_base_224": dict(embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), window_size=7, img_size=224),
    "swinv2_cr_large_224": dict(embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48), window_size=7, img_size=224),
    "swinv2_cr_huge_224": dict(embed_dim=352, depths=(2, 2, 18, 2), num_heads=(11, 22, 44, 88), window_size=7, img_size=224),
    "swinv2_cr_giant_224": dict(embed_dim=512, depths=(2, 2, 42, 2), num_heads=(16, 32, 64, 128), window_size=7, img_size=224),
}

def _make(name):
    cfg = _CFGS[name]

    def entry(**kwargs):
        model = SwinTransformerV2Cr(**dict(cfg, **kwargs))
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
