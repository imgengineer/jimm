"""LeViT in flax nnx, NHWC. Mirrors timm.models.levit (conv stem + attention with subsampling)."""
import jax.numpy as jnp
from flax import nnx

from ..layers import hswish, ClassifierMixin
from ..registry import register_model, _cfg

class LevitAttention(nnx.Module):
    def __init__(self, dim, num_heads, key_dim=16, *, rngs):
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.scale = key_dim ** -0.5
        self.q = nnx.Linear(dim, num_heads * key_dim, rngs=rngs)
        self.k = nnx.Linear(dim, num_heads * key_dim, rngs=rngs)
        self.v = nnx.Linear(dim, num_heads * key_dim, rngs=rngs)
        self.proj = nnx.Linear(num_heads * key_dim, dim, rngs=rngs)
        # per-head attention bias (LeViT hallmark), fixed for a given resolution
        self.attn_bias = nnx.Param(jnp.zeros((num_heads, 1, 1)))

    def __call__(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.key_dim).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, self.key_dim).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, self.key_dim).transpose(0, 2, 1, 3)
        attn = nnx.softmax(q @ k.transpose(0, 1, 3, 2) * self.scale + self.attn_bias[...], axis=-1)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, -1)
        return self.proj(x)

class LevitBlock(nnx.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2.0, key_dim=16, *, rngs):
        self.attn = LevitAttention(dim, num_heads, key_dim, rngs=rngs)
        self.mlp_fc1 = nnx.Linear(dim, int(dim * mlp_ratio), rngs=rngs)
        self.mlp_fc2 = nnx.Linear(int(dim * mlp_ratio), dim, rngs=rngs)

    def __call__(self, x):
        x = x + self.attn(x)
        return x + self.mlp_fc2(hswish(self.mlp_fc1(x)))

class SubsampleStage(nnx.Module):
    """Attention-based stride-2 subsampling between LeViT stages."""

    def __init__(self, in_dim, out_dim, num_heads, key_dim=16, *, rngs):
        self.attn = LevitAttention(in_dim, num_heads, key_dim, rngs=rngs)
        self.proj = nnx.Linear(in_dim, out_dim, rngs=rngs)  # attn output is in_dim wide
        self.in_dim, self.out_dim = in_dim, out_dim

    def __call__(self, x, H, W):
        B = x.shape[0]
        t = self.attn(x)
        t = t.reshape(B, H, W, -1)
        t = nnx.avg_pool(t, (2, 2), strides=(2, 2), padding="SAME")
        H2, W2 = t.shape[1], t.shape[2]
        t = self.proj(t.reshape(B, -1, t.shape[-1]))
        return t, H2, W2

class LeViT(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"
    default_cfg: dict | None = None

    def __init__(self, img_size=224, in_chans=3, num_classes=1000, global_pool="avg",
                 embed_dims=(128, 256, 384), depths=(4, 4, 4), num_heads=(4, 8, 12),
                 key_dim=16, drop_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dims[-1]
        # conv stem: 4x conv3x3 s2 with BN -> 14x14 tokens at embed_dims[0]
        stem, chs = [], in_chans
        for i, out in enumerate([embed_dims[0] // 8, embed_dims[0] // 4,
                                 embed_dims[0] // 2, embed_dims[0]]):
            stem.append(nnx.Conv(chs, out, (3, 3), strides=(2, 2), use_bias=False, rngs=rngs))
            stem.append(nnx.BatchNorm(out, rngs=rngs))
            chs = out
        self.stem = nnx.List(stem)
        self.stages = nnx.List([
            nnx.List([LevitBlock(dim, h, 2.0, key_dim, rngs=rngs) for _ in range(d)])
            for dim, d, h in zip(embed_dims, depths, num_heads)])
        self.subsamples = nnx.List([
            SubsampleStage(embed_dims[i], embed_dims[i + 1], num_heads[i + 1], key_dim, rngs=rngs)
            for i in range(len(embed_dims) - 1)])
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = nnx.Linear(embed_dims[-1], num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        for layer in self.stem:
            x = layer(x)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        for i, stage in enumerate(self.stages):
            for blk in stage:
                x = blk(x)
            if i < len(self.subsamples):
                x, H, W = self.subsamples[i](x, H, W)
        return x.reshape(B, H, W, -1)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

_CFGS = {  # embed_dims, depths, num_heads
    "levit_128s": ((128, 256, 384), (2, 3, 4), (4, 8, 12)),
    "levit_192": ((192, 288, 384), (3, 3, 4), (4, 8, 12)),
    "levit_256": ((256, 384, 512), (4, 4, 4), (8, 12, 16)),
}

def _make(name):
    embed_dims, depths, heads = _CFGS[name]

    def entry(**kwargs):
        model = LeViT(embed_dims=embed_dims, depths=depths, num_heads=heads, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
