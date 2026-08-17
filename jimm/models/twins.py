"""Twins-SVT in flax nnx, NHWC. Mirrors timm.models.twins (locally-grouped + global attention)."""
from flax import nnx

from ..layers import DropPath, Mlp, ClassifierMixin
from ..registry import register_model, _cfg
from .swin_transformer import window_partition, window_reverse

class GroupedAttention(nnx.Module):
    """Attention within non-overlapping windows (local) or full (global)."""

    def __init__(self, dim, num_heads, window_size=None, *, rngs):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.ws = window_size
        self.qkv = nnx.Linear(dim, dim * 3, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(self, x):
        B, H, W, C = x.shape
        if self.ws is None:
            t = x.reshape(B, H * W, C)
            t = self._attn(t)
            return t.reshape(B, H, W, C)
        t = window_partition(x, self.ws)
        t = self._attn(t)
        return window_reverse(t, self.ws, H, W, B)

    def _attn(self, t):
        B, N, C = t.shape
        qkv = self.qkv(t).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        out = nnx.dot_product_attention(q, k, v).reshape(B, N, C)
        return self.proj(out)

class TwinsBlock(nnx.Module):
    """locally-grouped (window) attention then global attention (Twins SVT)."""

    def __init__(self, dim, num_heads, window_size, drop_path=0.0, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.local_attn = GroupedAttention(dim, num_heads, window_size, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.global_attn = GroupedAttention(dim, num_heads, None, rngs=rngs)
        self.norm3 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * 4), rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        x = x + self.drop_path(self.local_attn(self.norm1(x)))
        x = x + self.drop_path(self.global_attn(self.norm2(x)))
        return x + self.drop_path(self.mlp(self.norm3(x)))

class Twins(ClassifierMixin, nnx.Module):
    default_cfg: dict | None = None

    def __init__(self, channels=(64, 128, 256, 512), depths=(2, 2, 6, 2), num_heads=(2, 4, 8, 16),
                 window_size=7, num_classes=1000, in_chans=3, global_pool="avg",
                 drop_rate=0.0, drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = channels[-1]
        dpr = [drop_path_rate * i / max(sum(depths) - 1, 1) for i in range(sum(depths))]
        patches, stages, k = [], [], 0
        for i, (c, d, h) in enumerate(zip(channels, depths, num_heads)):
            patches.append(nnx.Conv(in_chans if i == 0 else channels[i - 1], c,
                                    (4, 4) if i == 0 else (2, 2),
                                    strides=(4, 4) if i == 0 else (2, 2), rngs=rngs))
            blocks = [TwinsBlock(c, h, window_size, dpr[k + j], rngs=rngs) for j in range(d)]
            k += d
            stages.append(nnx.List(blocks))
        self.patches = nnx.List(patches)
        self.stages = nnx.List(stages)
        self.norm = nnx.LayerNorm(channels[-1], rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(channels[-1], num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        for patch, stage in zip(self.patches, self.stages):
            x = patch(x)
            for blk in stage:
                x = blk(x)
        return self.norm(x)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

_CFGS = {
    "twins_svt_small": ((64, 128, 256, 512), (2, 2, 8, 4), (2, 4, 8, 16)),
    "twins_svt_base": ((64, 128, 256, 512), (2, 2, 18, 4), (2, 4, 8, 16)),
    "twins_svt_large": ((128, 256, 512, 1024), (2, 2, 18, 4), (4, 8, 16, 32)),
}

def _make(name):
    channels, depths, heads = _CFGS[name]

    def entry(**kwargs):
        model = Twins(channels, depths, heads, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
