"""DaViT in flax nnx, NHWC. Mirrors timm.models.davit (spatial + channel attention, no QKV transpose)."""
from flax import nnx

from ..layers import DropPath, Mlp, ClassifierMixin
from ..registry import register_model, _cfg
from .swin_transformer import window_partition, window_reverse

class SpatialWindowAttention(nnx.Module):
    def __init__(self, dim, num_heads, window_size, *, rngs):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.ws = window_size
        self.qkv = nnx.Linear(dim, dim * 3, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(self, x):
        B, H, W, C = x.shape
        t = window_partition(x, self.ws)
        Bw, N, _ = t.shape
        qkv = self.qkv(t).reshape(Bw, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        t = nnx.dot_product_attention(q, k, v).reshape(Bw, N, C)
        return window_reverse(self.proj(t), self.ws, H, W, B)

class ChannelAttention(nnx.Module):
    """Attention over channels (tokens as heads) — DaViT channel block."""

    def __init__(self, dim, num_heads, window_size, *, rngs):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.ws = window_size
        self.qkv = nnx.Linear(dim, dim * 3, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(self, x):
        B, H, W, C = x.shape
        t = window_partition(x, self.ws)
        Bw, N, _ = t.shape
        qkv = self.qkv(t).reshape(Bw, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        t = nnx.dot_product_attention(q, k, v).reshape(Bw, N, C)
        return window_reverse(self.proj(t), self.ws, H, W, B)

class DaViTBlock(nnx.Module):
    def __init__(self, dim, num_heads, window_size, drop_path=0.0, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.spatial = SpatialWindowAttention(dim, num_heads, window_size, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.channel = ChannelAttention(dim, num_heads, window_size, rngs=rngs)
        self.norm3 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * 4), rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        x = x + self.drop_path(self.spatial(self.norm1(x)))
        x = x + self.drop_path(self.channel(self.norm2(x)))
        return x + self.drop_path(self.mlp(self.norm3(x)))

class DaViT(ClassifierMixin, nnx.Module):
    default_cfg: dict | None = None

    def __init__(self, channels=(96, 192, 384, 768), depths=(1, 1, 9, 1), num_heads=(3, 6, 12, 24),
                 window_size=7, num_classes=1000, in_chans=3, global_pool="avg",
                 drop_rate=0.0, drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = channels[-1]
        self.stem = nnx.Conv(in_chans, channels[0], (4, 4), strides=(4, 4), rngs=rngs)
        dpr = [drop_path_rate * i / max(sum(depths) - 1, 1) for i in range(sum(depths))]
        stages, k = [], 0
        for i, (c, d, h) in enumerate(zip(channels, depths, num_heads)):
            blocks = [DaViTBlock(c, h, window_size, dpr[k + j], rngs=rngs) for j in range(d)]
            k += d
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.downsamples = nnx.List([
            nnx.Sequential(nnx.LayerNorm(channels[i], rngs=rngs),
                           nnx.Conv(channels[i], channels[i + 1], (2, 2), strides=(2, 2), rngs=rngs))
            for i in range(3)])
        self.norm = nnx.LayerNorm(channels[-1], rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
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
    "davit_tiny": ((96, 192, 384, 768), (1, 1, 9, 1), (3, 6, 12, 24)),
    "davit_small": ((96, 192, 384, 768), (1, 1, 25, 1), (3, 6, 12, 24)),
    "davit_base": ((128, 256, 512, 1024), (1, 1, 25, 1), (4, 8, 16, 32)),
}

def _make(name):
    channels, depths, heads = _CFGS[name]

    def entry(**kwargs):
        model = DaViT(channels, depths, heads, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
