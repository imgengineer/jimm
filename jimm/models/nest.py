"""NeST (Nested ViT) in flax nnx. Mirrors timm.models.nest (nested local attention + aggregation)."""
from flax import nnx

from ..layers import DropPath, Mlp, ClassifierMixin
from ..registry import register_model, _cfg
from .swin_transformer import window_partition, window_reverse
from .vision_transformer import Attention

class NestBlock(nnx.Module):
    def __init__(self, dim, num_heads, window_size, drop_path=0.0, *, rngs):
        self.ws = window_size
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = Attention(dim, num_heads, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * 4), rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        B, H, W, C = x.shape
        t = window_partition(x, self.ws)
        t = t + self.drop_path(self.attn(self.norm1(t)))
        t = t + self.drop_path(self.mlp(self.norm2(t)))
        return window_reverse(t, self.ws, H, W, B)

class Nest(ClassifierMixin, nnx.Module):
    default_cfg: dict | None = None

    def __init__(self, channels=(96, 192, 384, 768), depths=(2, 2, 8, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, num_classes=1000, in_chans=3, global_pool="avg",
                 drop_rate=0.0, drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = channels[-1]
        self.stem = nnx.Conv(in_chans, channels[0], (4, 4), strides=(4, 4), rngs=rngs)
        dpr = [drop_path_rate * i / max(sum(depths) - 1, 1) for i in range(sum(depths))]
        stages, k = [], 0
        for i, (c, d, h) in enumerate(zip(channels, depths, num_heads)):
            blocks = [NestBlock(c, h, window_size, dpr[k + j], rngs=rngs) for j in range(d)]
            k += d
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        # aggregation between stages: 2x2 pool + conv to next width
        self.aggregates = nnx.List([
            nnx.Sequential(nnx.LayerNorm(channels[i], rngs=rngs),
                           lambda x: nnx.avg_pool(x, (2, 2), strides=(2, 2), padding="SAME"),
                           nnx.Conv(channels[i], channels[i + 1], (1, 1), rngs=rngs))
            for i in range(3)])
        self.norm = nnx.LayerNorm(channels[-1], rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(channels[-1], num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            if i > 0:
                x = self.aggregates[i - 1](x)
            for blk in stage:
                x = blk(x)
        return self.norm(x)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

_CFGS = {
    "nest_tiny": ((96, 192, 384, 768), (2, 2, 8, 2), (3, 6, 12, 24)),
    "nest_small": ((96, 192, 384, 768), (2, 2, 18, 2), (3, 6, 12, 24)),
    "nest_base": ((128, 256, 512, 1024), (2, 2, 18, 2), (4, 8, 16, 32)),
}

def _make(name):
    channels, depths, heads = _CFGS[name]

    def entry(**kwargs):
        model = Nest(channels, depths, heads, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
