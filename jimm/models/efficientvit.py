"""EfficientViT (MSRA) in flax nnx, NHWC. Mirrors timm.models.efficientvit_msra (MBConv + lightweight attn)."""
from flax import nnx

from ..layers import ConvBNAct, SqueezeExcite, ClassifierMixin
from ..registry import register_model, _cfg

class EViTMBConv(nnx.Module):
    def __init__(self, in_chs, out_chs, stride, expand=4, se=False, *, rngs):
        mid = in_chs * expand
        self.conv1 = ConvBNAct(in_chs, mid, 1, act="silu", rngs=rngs)
        self.dw = ConvBNAct(mid, mid, 3, stride, groups=mid, act="silu", rngs=rngs)
        self.se = SqueezeExcite(mid, rd_ratio=0.25, rngs=rngs) if se else None
        self.pw = ConvBNAct(mid, out_chs, 1, act="identity", rngs=rngs)
        self.shortcut = ConvBNAct(in_chs, out_chs, 1, stride, act="identity", rngs=rngs) \
            if (stride != 1 or in_chs != out_chs) else None

    def __call__(self, x):
        y = self.dw(self.conv1(x))
        if self.se is not None:
            y = self.se(y)
        y = self.pw(y)
        sc = x if self.shortcut is None else self.shortcut(x)
        return y + sc

class EViTAttention(nnx.Module):
    """Lightweight attention: single-head, ReLU-based (EfficientViT)."""

    def __init__(self, dim, *, rngs):
        self.scale = dim ** -0.5
        self.qkv = nnx.Linear(dim, dim * 3, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).transpose(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = nnx.relu(q) @ nnx.relu(k).transpose(0, 2, 1) * self.scale
        x = (attn @ v)
        return self.proj(x)

class EViTBlock(nnx.Module):
    def __init__(self, dim, drop_path=0.0, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = EViTAttention(dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.fc1 = nnx.Linear(dim, dim * 2, rngs=rngs)
        self.fc2 = nnx.Linear(dim * 2, dim, rngs=rngs)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.fc2(nnx.gelu(self.fc1(self.norm2(x))))

class EfficientViT(ClassifierMixin, nnx.Module):
    default_cfg: dict = {}

    def __init__(self, channels=(64, 128, 192, 256), depths=(1, 2, 3, 4), attn_depths=(0, 0, 1, 2),
                 num_classes=1000, in_chans=3, global_pool="avg", drop_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.stem = nnx.List([
            ConvBNAct(in_chans, 32, 3, 2, act="silu", rngs=rngs),
            ConvBNAct(32, channels[0], 3, 2, act="silu", rngs=rngs)])
        stages, chs = [], channels[0]
        for i, (c, d, ad) in enumerate(zip(channels, depths, attn_depths)):
            blocks = []
            for j in range(d):
                blocks.append(EViTMBConv(chs, c, 2 if (j == 0 and i > 0) else 1, 4, i >= 2, rngs=rngs))
                chs = c
            for _ in range(ad):
                blocks.append(EViTBlock(c, rngs=rngs))
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.num_features = channels[-1]
        self.head_drop = nnx.Dropout(drop_rate)
        self.fc = nnx.Linear(channels[-1], num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        for layer in self.stem:
            x = layer(x)
        for stage in self.stages:
            B = None
            for blk in stage:
                if isinstance(blk, EViTBlock):
                    B, H, W, C = x.shape
                    x = blk(x.reshape(B, H * W, C)).reshape(B, H, W, C)
                else:
                    x = blk(x)
        return x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

_CFGS = {  # channels, depths, attn_depths
    "efficientvit_b0": ((32, 64, 128, 256), (1, 2, 2, 2), (0, 0, 0, 1)),
    "efficientvit_b1": ((64, 128, 192, 256), (1, 2, 3, 4), (0, 0, 1, 2)),
    "efficientvit_b2": ((96, 192, 384, 512), (1, 3, 4, 4), (0, 0, 1, 2)),
}

def _make(name):
    channels, depths, ad = _CFGS[name]

    def entry(**kwargs):
        model = EfficientViT(channels, depths, ad, **kwargs)
        model.default_cfg = _cfg(input_size=(3, 256, 256))
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
