"""ViTAMIN (Vision Models Are Trending towards INterpretable Multiscale Transformers) in flax nnx, NHWC. Mirrors timm.models.vitamin."""
import jax.numpy as jnp
from flax import nnx

from ..layers import ConvBNAct, DropPath, global_pool_nhwc
from ..registry import register_model, _cfg
from .vision_transformer import Attention


class MbConvLNBlock(nnx.Module):
    """MBConv with LayerNorm."""

    def __init__(self, in_chs, out_chs, stride=1, expand=4, *, rngs):
        mid = in_chs * expand
        self.conv1 = nnx.Conv(in_chs, mid, (1, 1), use_bias=False, rngs=rngs)
        self.norm1 = nnx.LayerNorm(mid, rngs=rngs)
        self.dw = nnx.Conv(mid, mid, (3, 3), strides=(stride, stride), feature_group_count=mid, use_bias=False, rngs=rngs)
        self.norm2 = nnx.LayerNorm(mid, rngs=rngs)
        self.pw = nnx.Conv(mid, out_chs, (1, 1), use_bias=False, rngs=rngs)
        self.norm3 = nnx.LayerNorm(out_chs, rngs=rngs)
        self.shortcut = ConvBNAct(in_chs, out_chs, 1, stride, act="identity", rngs=rngs) \
            if (stride != 1 or in_chs != out_chs) else None

    def __call__(self, x):
        y = nnx.gelu(self.norm1(self.conv1(x)))
        y = nnx.gelu(self.norm2(self.dw(y)))
        y = self.norm3(self.pw(y))
        sc = x if self.shortcut is None else self.shortcut(x)
        return y + sc


class GeGluMlp(nnx.Module):
    """GeGLU MLP."""

    def __init__(self, dim, hidden_dim, *, rngs):
        self.fc1 = nnx.Linear(dim, hidden_dim * 2, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, dim, rngs=rngs)

    def __call__(self, x):
        h1, h2 = jnp.split(self.fc1(x), 2, axis=-1)
        return self.fc2(nnx.gelu(h1) * h2)


class VitBlock(nnx.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop_path=0.0, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = Attention(dim, num_heads, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = GeGluMlp(dim, int(dim * mlp_ratio * 2 / 3), rngs=rngs)
        self.drop_path = DropPath(drop_path)

    def __call__(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class ViTAMIN(nnx.Module):
    default_cfg: dict = {}

    def __init__(self, channels=(64, 128, 256, 512), conv_depths=(2, 2), vit_depths=(6, 2),
                 num_heads=(8, 16), num_classes=1000, in_chans=3, global_pool="avg",
                 drop_rate=0.0, drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = channels[-1]

        # Stem: Conv 3x3 s2
        self.stem = nnx.List([
            ConvBNAct(in_chans, channels[0] // 2, 3, 2, rngs=rngs),
            ConvBNAct(channels[0] // 2, channels[0], 3, 2, rngs=rngs)])

        # Stage 0..1: MBConv
        self.conv_stages = nnx.List([
            nnx.List([MbConvLNBlock(channels[0], channels[0], 1, rngs=rngs) for _ in range(conv_depths[0])]),
            nnx.List([MbConvLNBlock(channels[0] if j == 0 else channels[1], channels[1], 2 if j == 0 else 1, rngs=rngs)
                      for j in range(conv_depths[1])])
        ])

        # Stage 2..3: ViT blocks
        self.down2 = ConvBNAct(channels[1], channels[2], 3, 2, act="identity", rngs=rngs)
        self.vit_stage2 = nnx.List([VitBlock(channels[2], num_heads[0], rngs=rngs) for _ in range(vit_depths[0])])
        self.down3 = ConvBNAct(channels[2], channels[3], 3, 2, act="identity", rngs=rngs)
        self.vit_stage3 = nnx.List([VitBlock(channels[3], num_heads[1], rngs=rngs) for _ in range(vit_depths[1])])

        self.norm = nnx.LayerNorm(channels[-1], rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate)
        self.fc = nnx.Linear(channels[-1], num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        for layer in self.stem:
            x = layer(x)
        for stage in self.conv_stages:
            for blk in stage:
                x = blk(x)
        x = self.down2(x)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        for blk in self.vit_stage2:
            x = blk(x)
        x = x.reshape(B, H, W, C)
        x = self.down3(x)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        for blk in self.vit_stage3:
            x = blk(x)
        return self.norm(x).reshape(B, H, W, -1)

    def forward_head(self, x):
        x = global_pool_nhwc(x, self.global_pool)
        x = self.head_drop(x)
        return self.fc(x) if self.fc is not None else x

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool="avg"):
        self.num_classes, self.global_pool = num_classes, global_pool
        if num_classes > 0 and self.fc is None:
            raise RuntimeError("cannot re-add classifier to a num_classes=0 model")
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=nnx.Rngs(0)) if num_classes > 0 else None

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_CFGS = {
    "vitamin_small_224": ((64, 128, 256, 512), (2, 2), (6, 2), (8, 16)),
    "vitamin_base_224": ((80, 160, 320, 640), (2, 2), (8, 2), (10, 20)),
    "vitamin_large_224": ((96, 192, 384, 768), (2, 2), (12, 2), (12, 24)),
    "vitamin_large2_224": ((128, 256, 512, 1024), (2, 2), (16, 2), (16, 32)),
}


def _make(name):
    channels, conv_depths, vit_depths, heads = _CFGS[name]

    def entry(**kwargs):
        model = ViTAMIN(channels, conv_depths, vit_depths, heads, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
