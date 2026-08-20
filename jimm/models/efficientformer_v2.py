"""EfficientFormer-V2 in flax nnx, NHWC. Mirrors timm.models.efficientformer_v2."""
from flax import nnx

from ..layers import DropPath, ClassifierMixin
from ..registry import register_model, _cfg

class ConvMlp(nnx.Module):
    """Conv MLP with depthwise 3x3 in between."""

    def __init__(self, in_chs, hidden_chs, out_chs, *, rngs):
        self.fc1 = nnx.Conv(in_chs, hidden_chs, (1, 1), use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(hidden_chs, rngs=rngs)
        self.dw = nnx.Conv(hidden_chs, hidden_chs, (3, 3), feature_group_count=hidden_chs, use_bias=False, rngs=rngs)
        self.bn2 = nnx.BatchNorm(hidden_chs, rngs=rngs)
        self.fc2 = nnx.Conv(hidden_chs, out_chs, (1, 1), use_bias=False, rngs=rngs)
        self.bn3 = nnx.BatchNorm(out_chs, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.bn1(self.fc1(x)))
        x = nnx.relu(self.bn2(self.dw(x)))
        return self.bn3(self.fc2(x))

class Attention2d(nnx.Module):
    def __init__(self, dim, key_dim=16, num_heads=8, *, rngs):
        self.num_heads = num_heads
        self.head_dim = key_dim
        self.q = nnx.Linear(dim, num_heads * key_dim, rngs=rngs)
        self.k = nnx.Linear(dim, num_heads * key_dim, rngs=rngs)
        self.v = nnx.Linear(dim, num_heads * key_dim, rngs=rngs)
        self.proj = nnx.Linear(num_heads * key_dim, dim, rngs=rngs)

    def __call__(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim)
        out = nnx.dot_product_attention(q, k, v).reshape(B, N, -1)
        return self.proj(out)

class EfficientFormerV2Block(nnx.Module):
    def __init__(self, dim, mlp_ratio=4.0, is_vit=False, drop_path=0.0, *, rngs):
        self.is_vit = is_vit
        self.drop_path = DropPath(drop_path, rngs=rngs)
        if is_vit:
            self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
            self.attn = Attention2d(dim, rngs=rngs)
            self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
            self.mlp_fc1 = nnx.Linear(dim, int(dim * mlp_ratio), rngs=rngs)
            self.mlp_fc2 = nnx.Linear(int(dim * mlp_ratio), dim, rngs=rngs)
        else:
            self.mlp = ConvMlp(dim, int(dim * mlp_ratio), dim, rngs=rngs)

    def __call__(self, x):
        if self.is_vit:
            B, H, W, C = x.shape
            t = x.reshape(B, H * W, C)
            t = t + self.drop_path(self.attn(self.norm1(t)))
            t = t + self.drop_path(self.mlp_fc2(nnx.gelu(self.mlp_fc1(self.norm2(t)))))
            return t.reshape(B, H, W, C)
        else:
            return x + self.drop_path(self.mlp(x))

class EfficientFormerV2(ClassifierMixin, nnx.Module):
    default_cfg: dict | None = None

    def __init__(self, depths=(2, 2, 6, 2), embed_dims=(32, 48, 96, 176), num_vit=2,
                 mlp_ratios=(4, 4, 4, 4), num_classes=1000, in_chans=3, global_pool="avg",
                 drop_rate=0.0, drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dims[-1]

        # Stem (4x downsample)
        self.stem = nnx.List([
            nnx.Conv(in_chans, embed_dims[0] // 2, (3, 3), strides=(2, 2), use_bias=False, rngs=rngs),
            nnx.BatchNorm(embed_dims[0] // 2, rngs=rngs),
            nnx.Conv(embed_dims[0] // 2, embed_dims[0], (3, 3), strides=(2, 2), use_bias=False, rngs=rngs),
            nnx.BatchNorm(embed_dims[0], rngs=rngs),
        ])

        dpr = [drop_path_rate * i / max(sum(depths) - 1, 1) for i in range(sum(depths))]
        stages, k = [], 0
        for i, (c, d, mr) in enumerate(zip(embed_dims, depths, mlp_ratios)):
            blocks = []
            for j in range(d):
                is_vit = (i >= 2) and (j >= d - num_vit)
                blocks.append(EfficientFormerV2Block(c, mr, is_vit, dpr[k], rngs=rngs))
                k += 1
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)

        self.downsamples = nnx.List([
            nnx.Sequential(
                nnx.Conv(embed_dims[i], embed_dims[i + 1], (3, 3), strides=(2, 2), use_bias=False, rngs=rngs),
                nnx.BatchNorm(embed_dims[i + 1], rngs=rngs)
            ) for i in range(len(embed_dims) - 1)
        ])

        self.norm = nnx.BatchNorm(self.num_features, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = nnx.relu(self.stem[1](self.stem[0](x)))
        x = nnx.relu(self.stem[3](self.stem[2](x)))
        for i, stage in enumerate(self.stages):
            if i > 0:
                x = self.downsamples[i - 1](x)
            for blk in stage:
                x = blk(x)
        return self.norm(x)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

_CFGS = {
    "efficientformerv2_s0": dict(depths=(2, 2, 6, 2), embed_dims=(32, 48, 96, 176), num_vit=2),
    "efficientformerv2_s1": dict(depths=(3, 3, 9, 3), embed_dims=(32, 48, 120, 224), num_vit=2),
    "efficientformerv2_s2": dict(depths=(3, 3, 15, 3), embed_dims=(36, 64, 144, 288), num_vit=4),
    "efficientformerv2_l": dict(depths=(5, 5, 15, 5), embed_dims=(48, 96, 192, 384), num_vit=6),
}

def _make(name):
    cfg = _CFGS[name]

    def entry(**kwargs):
        model = EfficientFormerV2(**dict(cfg, **kwargs))
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
