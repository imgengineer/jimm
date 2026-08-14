"""GCViT in flax nnx, NHWC. Mirrors timm.models.gcvit (global-context self-attention)."""
import jax.numpy as jnp
from flax import nnx

from ..layers import DropPath, Mlp, ClassifierMixin
from ..registry import register_model, _cfg
from .swin_transformer import window_partition, window_reverse
from .vision_transformer import Attention as _Attn

class GlobalQuery(nnx.Module):
    """Small CNN producing a global context vector for the stage (pooled to one token)."""

    def __init__(self, dim, *, rngs):
        self.pool = nnx.Conv(dim, dim, (3, 3), strides=(2, 2), feature_group_count=dim, rngs=rngs)
        self.proj = nnx.Conv(dim, dim, (1, 1), rngs=rngs)

    def __call__(self, x):
        y = self.proj(self.pool(x))
        return jnp.mean(y, axis=(1, 2), keepdims=True)  # (B,1,1,C)

class GCBlock(nnx.Module):
    """Window attention with a global context vector injected into the tokens."""

    def __init__(self, dim, num_heads, window_size, drop_path=0.0, *, rngs):
        self.ws = window_size
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = _Attn(dim, num_heads, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * 4), rngs=rngs)
        self.drop_path = DropPath(drop_path)
        self.global_query = GlobalQuery(dim, rngs=rngs)

    def __call__(self, x):
        B, H, W, C = x.shape
        gq = self.global_query(x)  # (B,1,1,C)
        t = window_partition(x + gq, self.ws)  # (B*nW, ws*ws, C), global ctx broadcast to all tokens
        t = t + self.drop_path(self.attn(self.norm1(t)))
        t = t + self.drop_path(self.mlp(self.norm2(t)))
        return window_reverse(t, self.ws, H, W, B)

class GCViT(ClassifierMixin, nnx.Module):
    default_cfg: dict = {}

    def __init__(self, channels=(96, 192, 384, 768), depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, num_classes=1000, in_chans=3, global_pool="avg",
                 drop_rate=0.0, drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = channels[-1]
        self.stem = nnx.Conv(in_chans, channels[0], (4, 4), strides=(4, 4), rngs=rngs)
        dpr = [drop_path_rate * i / max(sum(depths) - 1, 1) for i in range(sum(depths))]
        stages, k = [], 0
        for i, (c, d, h) in enumerate(zip(channels, depths, num_heads)):
            blocks = [GCBlock(c, h, window_size, dpr[k + j], rngs=rngs) for j in range(d)]
            k += d
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.downsamples = nnx.List([
            nnx.Sequential(nnx.LayerNorm(channels[i], rngs=rngs),
                           nnx.Conv(channels[i], channels[i + 1], (2, 2), strides=(2, 2), rngs=rngs))
            for i in range(3)])
        self.norm = nnx.LayerNorm(channels[-1], rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate)
        self.fc = nnx.Linear(channels[-1], num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            if i > 0:
                x = self.downsamples[i - 1](x)
            for blk in stage:
                x = blk(x)
        return self.norm(x)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

_CFGS = {
    "gcvit_tiny": ((96, 192, 384, 768), (2, 2, 6, 2), (3, 6, 12, 24)),
    "gcvit_small": ((96, 192, 384, 768), (2, 2, 18, 2), (3, 6, 12, 24)),
    "gcvit_base": ((128, 256, 512, 1024), (2, 2, 18, 2), (4, 8, 16, 32)),
}

def _make(name):
    channels, depths, heads = _CFGS[name]

    def entry(**kwargs):
        model = GCViT(channels, depths, heads, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
