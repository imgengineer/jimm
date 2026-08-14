"""NextViT in flax nnx, NHWC. Mirrors timm.models.nextvit (conv blocks + transformer blocks)."""
from flax import nnx

from ..layers import ConvBNAct, DropPath, hswish, ClassifierMixin
from ..registry import register_model, _cfg
from .vision_transformer import Attention

class ConvBlock(nnx.Module):
    """1x1 expand -> dw 3x3 -> 1x1 project, residual."""

    def __init__(self, in_chs, out_chs, stride, expand=4, drop_path=0.0, *, rngs):
        mid = in_chs * expand
        self.conv1 = ConvBNAct(in_chs, mid, 1, act="hswish", rngs=rngs)
        self.dw = ConvBNAct(mid, mid, 3, stride, groups=mid, act="hswish", rngs=rngs)
        self.pw = ConvBNAct(mid, out_chs, 1, act="identity", rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)
        self.shortcut = ConvBNAct(in_chs, out_chs, 1, stride, act="identity", rngs=rngs) \
            if (stride != 1 or in_chs != out_chs) else None

    def __call__(self, x):
        y = self.pw(self.dw(self.conv1(x)))
        sc = x if self.shortcut is None else self.shortcut(x)
        return self.drop_path(y) + sc

class TransformerBlock(nnx.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2.0, drop_path=0.0, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = Attention(dim, num_heads, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.fc1 = nnx.Linear(dim, int(dim * mlp_ratio), rngs=rngs)
        self.fc2 = nnx.Linear(int(dim * mlp_ratio), dim, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.fc2(hswish(self.fc1(self.norm2(x)))))

class NextViT(ClassifierMixin, nnx.Module):
    default_cfg: dict = {}

    def __init__(self, channels=(96, 192, 384, 768), depths=(2, 3, 8, 3), tf_depths=(0, 1, 2, 2),
                 num_classes=1000, in_chans=3, global_pool="avg", drop_rate=0.0,
                 drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = channels[-1]
        self.stem = nnx.List([
            ConvBNAct(in_chans, 32, 3, 2, act="hswish", rngs=rngs),
            ConvBNAct(32, 64, 3, 2, act="hswish", rngs=rngs),
            ConvBNAct(64, channels[0], 3, 1, act="hswish", rngs=rngs)])
        dpr = [drop_path_rate * i / max(sum(depths) - 1, 1) for i in range(sum(depths))]
        stages, chs, k = [], channels[0], 0
        for i, (c, d, td) in enumerate(zip(channels, depths, tf_depths)):
            blocks = []
            for j in range(d):
                blocks.append(ConvBlock(chs, c, 2 if (j == 0 and i > 0) else 1, 4, dpr[k], rngs=rngs))
                chs = c
                k += 1
            for _ in range(td):
                blocks.append(TransformerBlock(c, 8, 2.0, drop_path_rate, rngs=rngs))
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(channels[-1], num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        for layer in self.stem:
            x = layer(x)
        for stage in self.stages:
            B = None
            for blk in stage:
                if isinstance(blk, TransformerBlock):
                    B, H, W, C = x.shape
                    x = blk(x.reshape(B, H * W, C)).reshape(B, H, W, C)
                else:
                    x = blk(x)
        return x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

_CFGS = {
    "nextvit_small": ((96, 192, 384, 768), (2, 3, 8, 3), (0, 1, 2, 2)),
    "nextvit_base": ((96, 256, 512, 1024), (2, 3, 10, 3), (0, 1, 2, 2)),
    "nextvit_large": ((96, 256, 512, 1024), (3, 4, 12, 3), (0, 1, 3, 2)),
}

def _make(name):
    channels, depths, tf = _CFGS[name]

    def entry(**kwargs):
        model = NextViT(channels, depths, tf, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
