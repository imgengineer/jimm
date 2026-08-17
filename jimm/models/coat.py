"""CoaT (co-scale conv-attention) in flax nnx. Mirrors timm.models.coat."""
from flax import nnx

from ..layers import DropPath, Mlp, ClassifierMixin
from ..registry import register_model, _cfg
from .vision_transformer import Attention

class CoaTBlock(nnx.Module):
    def __init__(self, dim, num_heads, drop_path=0.0, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = Attention(dim, num_heads, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * 4), rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))

class CoaT(ClassifierMixin, nnx.Module):
    default_cfg: dict | None = None

    def __init__(self, channels=(192, 384, 768, 768), depths=(2, 2, 3, 2), num_heads=(6, 12, 24, 24),
                 img_size=224, in_chans=3, num_classes=1000, global_pool="avg",
                 drop_rate=0.0, drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = channels[-1]
        # conv patch embeds per stage
        patches, stages = [], []
        dpr = [drop_path_rate * i / max(sum(depths) - 1, 1) for i in range(sum(depths))]
        k = 0
        for i, (c, d, h) in enumerate(zip(channels, depths, num_heads)):
            patches.append(nnx.Conv(in_chans if i == 0 else channels[i - 1], c,
                                    (4, 4) if i == 0 else (2, 2),
                                    strides=(4, 4) if i == 0 else (2, 2), rngs=rngs))
            blocks = [CoaTBlock(c, h, dpr[k + j], rngs=rngs) for j in range(d)]
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
            B, H, W, C = x.shape
            t = x.reshape(B, H * W, C)
            for blk in stage:
                t = blk(t)
            x = t.reshape(B, H, W, C)
        return self.norm(x)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

_CFGS = {
    "coat_tiny": ((192, 384, 768, 768), (2, 2, 3, 2), (6, 12, 24, 24)),
    "coat_small": ((192, 384, 768, 768), (2, 2, 6, 2), (6, 12, 24, 24)),
    "coat_mini": ((192, 384, 768, 768), (2, 2, 2, 2), (6, 12, 24, 24)),
}

def _make(name):
    channels, depths, heads = _CFGS[name]

    def entry(**kwargs):
        model = CoaT(channels, depths, heads, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
