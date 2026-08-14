"""Hiera in flax nnx, NHWC. Mirrors timm.models.hiera (Hierarchical Vision Transformer)."""
from flax import nnx

from ..layers import DropPath, Mlp, ClassifierMixin
from ..registry import register_model, _cfg

class HieraBlock(nnx.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop_path=0.0, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nnx.Linear(dim, dim * 3, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), rngs=rngs)
        self.drop_path = DropPath(drop_path)

    def __call__(self, x):
        B, N, C = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(B, N, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = nnx.softmax(q @ k.transpose(0, 1, 3, 2) * self.scale, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = x + self.drop_path(self.proj(out))
        return x + self.drop_path(self.mlp(self.norm2(x)))

class Hiera(ClassifierMixin, nnx.Module):
    default_cfg: dict = {}

    def __init__(self, embed_dim=96, num_heads=1, stages=(1, 2, 7, 2),
                 mlp_ratio=4.0, num_classes=1000, in_chans=3, global_pool="avg",
                 drop_rate=0.0, drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        
        self.stem = nnx.Conv(in_chans, embed_dim, (7, 7), strides=(4, 4), padding="SAME", rngs=rngs)
        
        dims = [embed_dim * (2 ** i) for i in range(len(stages))]
        heads = [num_heads * (2 ** i) for i in range(len(stages))]
        self.num_features = dims[-1]

        dpr = [drop_path_rate * i / max(sum(stages) - 1, 1) for i in range(sum(stages))]
        stages_list = []
        k = 0
        for i, (d, dim, h) in enumerate(zip(stages, dims, heads)):
            stage_blocks = [HieraBlock(dim, h, mlp_ratio, dpr[k + j], rngs=rngs) for j in range(d)]
            k += d
            stages_list.append(nnx.List(stage_blocks))
        self.stages = nnx.List(stages_list)

        self.downsamples = nnx.List([
            nnx.Sequential(
                nnx.Conv(dims[i], dims[i + 1], (2, 2), strides=(2, 2), rngs=rngs)
            ) for i in range(len(stages) - 1)
        ])

        self.norm = nnx.LayerNorm(self.num_features, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate)
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            if i > 0:
                x = self.downsamples[i - 1](x)
            B, H, W, C = x.shape
            x = x.reshape(B, H * W, C)
            for blk in stage:
                x = blk(x)
            x = x.reshape(B, H, W, C)
        return self.norm(x)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

_CFGS = {
    "hiera_tiny_224": dict(embed_dim=96, num_heads=1, stages=(1, 2, 7, 2)),
    "hiera_small_224": dict(embed_dim=96, num_heads=1, stages=(1, 2, 11, 2)),
    "hiera_base_224": dict(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3)),
    "hiera_base_plus_224": dict(embed_dim=112, num_heads=2, stages=(2, 3, 16, 3)),
    "hiera_large_224": dict(embed_dim=144, num_heads=2, stages=(2, 6, 36, 4)),
    "hiera_huge_224": dict(embed_dim=256, num_heads=4, stages=(2, 6, 36, 4)),
}

def _make(name):
    cfg = _CFGS[name]

    def entry(**kwargs):
        model = Hiera(**dict(cfg, **kwargs))
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
