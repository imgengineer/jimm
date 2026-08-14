"""EfficientFormer (v1) in flax nnx. Mirrors timm.models.efficientformer."""
from flax import nnx

from ..layers import DropPath, PatchEmbed, ClassifierMixin
from ..registry import register_model, _cfg
from .vision_transformer import Attention

class ConvBlock(nnx.Module):
    """dw 3x3 + two 1x1 (pointwise MLP), with BN, residual."""

    def __init__(self, dim, mlp_ratio=4, drop_path=0.0, *, rngs):
        self.dw = nnx.Conv(dim, dim, (3, 3), use_bias=False, feature_group_count=dim, rngs=rngs)
        self.bn0 = nnx.BatchNorm(dim, rngs=rngs)
        self.fc1 = nnx.Conv(dim, dim * mlp_ratio, (1, 1), rngs=rngs)
        self.bn1 = nnx.BatchNorm(dim * mlp_ratio, rngs=rngs)
        self.fc2 = nnx.Conv(dim * mlp_ratio, dim, (1, 1), rngs=rngs)
        self.bn2 = nnx.BatchNorm(dim, rngs=rngs)
        self.drop_path = DropPath(drop_path)

    def __call__(self, x):
        y = self.bn0(self.dw(x))
        y = self.bn2(self.fc2(self.bn1(self.fc1(y))))
        return x + self.drop_path(y)

class AttnBlock(nnx.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, drop_path=0.0, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = Attention(dim, num_heads, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp_fc1 = nnx.Linear(dim, dim * mlp_ratio, rngs=rngs)
        self.mlp_fc2 = nnx.Linear(dim * mlp_ratio, dim, rngs=rngs)
        self.drop_path = DropPath(drop_path)

    def __call__(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp_fc2(nnx.gelu(self.mlp_fc1(self.norm2(x)))))

class EfficientFormer(ClassifierMixin, nnx.Module):
    default_cfg: dict = {}

    def __init__(self, channels=(48, 96, 224, 448), depths=(3, 2, 6, 4),
                 num_attn_blocks=(0, 0, 2, 2), num_classes=1000, in_chans=3,
                 global_pool="avg", drop_rate=0.0, drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = channels[-1]
        # stem: two conv3x3 s2
        self.stem = nnx.List([
            nnx.Conv(in_chans, channels[0] // 2, (3, 3), strides=(2, 2), use_bias=False, rngs=rngs),
            nnx.BatchNorm(channels[0] // 2, rngs=rngs),
            nnx.Conv(channels[0] // 2, channels[0], (3, 3), strides=(2, 2), use_bias=False, rngs=rngs),
            nnx.BatchNorm(channels[0], rngs=rngs)])
        dpr = [drop_path_rate * i / max(sum(depths) - 1, 1) for i in range(sum(depths))]
        stages, k = [], 0
        for i, (c, d, na) in enumerate(zip(channels, depths, num_attn_blocks)):
            blocks = []
            for j in range(d):
                if j >= d - na:  # last na blocks are attention (on tokens)
                    blocks.append(AttnBlock(c, 8, 4, dpr[k], rngs=rngs))
                else:
                    blocks.append(ConvBlock(c, 4, dpr[k], rngs=rngs))
                k += 1
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.downsamples = nnx.List([
            nnx.Sequential(nnx.Conv(channels[i], channels[i + 1], (3, 3), strides=(2, 2),
                                    use_bias=False, rngs=rngs),
                           nnx.BatchNorm(channels[i + 1], rngs=rngs))
            for i in range(3)])
        self.head_drop = nnx.Dropout(drop_rate)
        self.fc = nnx.Linear(channels[-1], num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        for layer in self.stem:
            x = layer(x)
        for i, stage in enumerate(self.stages):
            if i > 0:
                x = self.downsamples[i - 1](x)
            B, H, W, C = x.shape
            for blk in stage:
                if isinstance(blk, AttnBlock):
                    t = blk(x.reshape(B, H * W, C))
                    x = t.reshape(B, H, W, C)
                else:
                    x = blk(x)
        return x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

_CFGS = {  # channels, depths, num_attn_blocks
    "efficientformer_l1": ((48, 96, 224, 448), (3, 2, 6, 4), (0, 0, 2, 2)),
    "efficientformer_l3": ((64, 128, 320, 512), (4, 4, 12, 6), (0, 0, 4, 4)),
    "efficientformer_l7": ((96, 192, 384, 768), (6, 6, 18, 8), (0, 0, 6, 6)),
}

def _make(name):
    channels, depths, na = _CFGS[name]

    def entry(**kwargs):
        model = EfficientFormer(channels, depths, na, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
