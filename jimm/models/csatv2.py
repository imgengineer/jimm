"""CSATv2 in flax nnx, NHWC. Mirrors timm.models.csatv2 (Cascaded Spatial Attention Transformer)."""
from flax import nnx

from ..layers import ConvBNAct, DropPath, Mlp, ClassifierMixin
from ..registry import register_model, _cfg
from .vision_transformer import Attention

class SpatialAttention(nnx.Module):
    """Spatial attention using 7x7 depthwise convolution + gating."""

    def __init__(self, dim, *, rngs):
        self.dw = nnx.Conv(dim, dim, (7, 7), feature_group_count=dim, use_bias=False, rngs=rngs)
        self.bn = nnx.BatchNorm(dim, rngs=rngs)
        self.proj = nnx.Conv(dim, dim, (1, 1), rngs=rngs)

    def __call__(self, x):
        return self.proj(nnx.relu(self.bn(self.dw(x))))

class SpatialTransformerBlock(nnx.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop_path=0.0, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.spatial = SpatialAttention(dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), rngs=rngs)
        self.drop_path = DropPath(drop_path)

    def __call__(self, x):
        x = x + self.drop_path(self.spatial(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))

class TransformerBlock(nnx.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, drop_path=0.0, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = Attention(dim, num_heads, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), rngs=rngs)
        self.drop_path = DropPath(drop_path)

    def __call__(self, x):
        B, H, W, C = x.shape
        t = x.reshape(B, H * W, C)
        t = t + self.drop_path(self.attn(self.norm1(t)))
        t = t + self.drop_path(self.mlp(self.norm2(t)))
        return t.reshape(B, H, W, C)

class CSATv2(ClassifierMixin, nnx.Module):
    default_cfg: dict = {}

    def __init__(self, dims=(48, 96, 224, 448), depths=(3, 3, 9, 3), num_classes=1000,
                 in_chans=3, global_pool="avg", drop_rate=0.0, drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = dims[-1]

        # Stem
        self.stem = nnx.List([
            ConvBNAct(in_chans, dims[0] // 2, 3, 2, rngs=rngs),
            ConvBNAct(dims[0] // 2, dims[0], 3, 2, rngs=rngs),
        ])

        dpr = [drop_path_rate * i / max(sum(depths) - 1, 1) for i in range(sum(depths))]
        stages, k = [], 0
        for i, (dim, d) in enumerate(zip(dims, depths)):
            blocks = []
            for j in range(d):
                if i < 2:
                    blocks.append(SpatialTransformerBlock(dim, 4.0, dpr[k], rngs=rngs))
                else:
                    blocks.append(TransformerBlock(dim, 8, 4.0, dpr[k], rngs=rngs))
                k += 1
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)

        self.downsamples = nnx.List([
            ConvBNAct(dims[i], dims[i + 1], 3, 2, act="identity", rngs=rngs)
            for i in range(len(dims) - 1)
        ])

        self.norm = nnx.LayerNorm(self.num_features, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate)
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        for layer in self.stem:
            x = layer(x)
        for i, stage in enumerate(self.stages):
            if i > 0:
                x = self.downsamples[i - 1](x)
            for blk in stage:
                x = blk(x)
        return self.norm(x)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

@register_model
def csatv2(**kwargs):
    model = CSATv2(dims=(48, 96, 224, 448), depths=(3, 3, 9, 3), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def csatv2_21m(**kwargs):
    model = CSATv2(dims=(64, 128, 256, 512), depths=(3, 4, 12, 3), **kwargs)
    model.default_cfg = _cfg()
    return model
