"""MViT v2 in flax nnx. Mirrors timm.models.mvitv2 (pooled attention with rel-pos bias)."""
import jax.numpy as jnp
from flax import nnx

from ..layers import DropPath, Mlp, PatchEmbed, ClassifierMixin
from ..registry import register_model, _cfg

class PooledAttention(nnx.Module):
    """Attention with pooled K/V (spatial stride) + 2D relative position bias."""

    def __init__(self, dim, num_heads, pool_stride=1, grid=14, *, rngs):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.pool_stride = pool_stride
        self.qkv = nnx.Linear(dim, dim * 3, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)
        grid = (grid, grid) if isinstance(grid, int) else tuple(grid)
        self.grid = grid
        gh, gw = grid
        self.rel_bias = nnx.Param(jnp.zeros(((2 * gh - 1) * (2 * gw - 1), num_heads)))
        coords = jnp.stack(jnp.meshgrid(jnp.arange(gh), jnp.arange(gw), indexing="ij"))
        cf = coords.reshape(2, -1)
        rel = (cf[:, :, None] - cf[:, None, :]).transpose(1, 2, 0) + jnp.array([gh - 1, gw - 1])
        # nnx.Variable: raw array attributes break nnx.cached_partial graph flattening
        self.rel_index = nnx.Variable(rel[:, :, 0] * (2 * gw - 1) + rel[:, :, 1])  # (gh*gw, gh*gw)

    def __call__(self, x, H, W):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        if self.pool_stride > 1:
            k = self._pool(k.transpose(0, 2, 1, 3), H, W).transpose(0, 2, 1, 3)
            v = self._pool(v.transpose(0, 2, 1, 3), H, W).transpose(0, 2, 1, 3)
        bias = None
        # ponytail: rel-pos bias applied only for same-resolution attn (skipped when K/V pooled)
        if self.pool_stride == 1 and k.shape[1] == self.rel_index[...].shape[0]:
            bias = self.rel_bias[...][self.rel_index[...]].transpose(2, 0, 1)
        x = nnx.dot_product_attention(q, k, v, bias=bias).reshape(B, N, C)
        return self.proj(x)

    def _pool(self, t, H, W):
        # (B, heads, N, hd) -> pool on spatial -> back
        B, h, N, d = t.shape
        t = t.transpose(0, 2, 1, 3).reshape(B, H, W, h * d)
        t = nnx.avg_pool(t, (self.pool_stride, self.pool_stride),
                         strides=(self.pool_stride, self.pool_stride), padding="SAME")
        return t.reshape(B, -1, h, d).transpose(0, 2, 1, 3)

class MViTBlock(nnx.Module):
    def __init__(self, dim, num_heads, pool_stride, grid, mlp_ratio=4.0, drop_path=0.0, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = PooledAttention(dim, num_heads, pool_stride, grid, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        return x + self.drop_path(self.mlp(self.norm2(x)))

class MViTv2(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"
    default_cfg: dict | None = None

    def __init__(self, img_size=224, in_chans=3, num_classes=1000, global_pool="avg",
                 embed_dim=96, depths=(2, 3, 16, 3), num_heads=(1, 2, 4, 8),
                 pool_strides=(1, 2, 2, 2), drop_rate=0.0, drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        # MViT stem: 7x7 conv stride 4 (overlapping), not patch_size-strided
        self.patch_embed = nnx.Conv(in_chans, embed_dim, (7, 7), strides=(4, 4), rngs=rngs)
        stages, chs, k = [], embed_dim, 0
        dims = [embed_dim * 2**i for i in range(len(depths))]
        self.num_features = dims[-1]
        dpr = [drop_path_rate * i / max(sum(depths) - 1, 1) for i in range(sum(depths))]
        self.stage_projs = nnx.List([
            nnx.Linear(chs if i == 0 else dims[i - 1], dims[i], rngs=rngs) for i in range(len(depths))])
        for i, (d, h) in enumerate(zip(depths, num_heads)):
            grid = img_size // 4 // 2**i
            blocks = [MViTBlock(dims[i], h, pool_strides[i], (grid, grid), 4.0, dpr[k + j], rngs=rngs)
                      for j in range(d)]
            k += d
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.norm = nnx.LayerNorm(dims[-1], rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = nnx.Linear(dims[-1], num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        B, H, W, C = x.shape
        for i, stage in enumerate(self.stages):
            if i > 0:
                # downsample tokens 2x before projecting to the next stage width
                t = x.reshape(B, H, W, -1)
                t = nnx.avg_pool(t, (2, 2), strides=(2, 2), padding="SAME")
                B, H, W, C = t.shape
                x = t.reshape(B, H * W, C)
            else:
                x = x.reshape(B, H * W, C)
            x = self.stage_projs[i](x)
            for blk in stage:
                x = blk(x, H, W)
        return self.norm(x).reshape(B, H, W, -1)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

_CFGS = {  # embed_dim, depths, num_heads, pool_strides
    "mvitv2_tiny": (96, (1, 2, 5, 2), (1, 2, 4, 8), (1, 2, 2, 2)),
    "mvitv2_small": (96, (2, 4, 11, 2), (1, 2, 4, 8), (1, 2, 2, 2)),
    "mvitv2_base": (96, (2, 6, 18, 2), (1, 2, 4, 8), (1, 2, 2, 2)),
}

def _make(name):
    embed_dim, depths, heads, ps = _CFGS[name]

    def entry(**kwargs):
        model = MViTv2(embed_dim=embed_dim, depths=depths, num_heads=heads, pool_strides=ps, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
