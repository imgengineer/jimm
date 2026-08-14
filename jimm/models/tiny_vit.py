"""TinyViT in flax nnx, NHWC. Mirrors timm.models.tiny_vit (MBConv + window attention)."""
import jax.numpy as jnp
from flax import nnx

from ..layers import ConvBNAct, DropPath, Mlp, global_pool_nhwc
from ..registry import register_model, _cfg
from .mobilenetv2 import InvertedResidual
from .swin_transformer import WindowAttention, window_partition, window_reverse


class TinyViTWindowBlock(nnx.Module):
    def __init__(self, dim, num_heads, window_size, drop_path=0.0, *, rngs):
        self.ws = window_size
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = WindowAttention(dim, window_size, num_heads, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = Mlp(dim, int(dim * 4), rngs=rngs)
        self.drop_path = DropPath(drop_path)

    def __call__(self, x):
        B, H, W, C = x.shape
        t = window_partition(x, self.ws)
        t = t + self.drop_path(self.attn(self.norm1(t)))
        t = t + self.drop_path(self.mlp(self.norm2(t)))
        return window_reverse(t, self.ws, H, W, B)


class TinyViT(nnx.Module):
    default_cfg: dict = {}

    def __init__(self, channels=(96, 192, 384, 576), depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 18),
                 window_size=7, num_classes=1000, in_chans=3, global_pool="avg",
                 drop_rate=0.0, drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = channels[-1]
        self.stem = nnx.List([
            ConvBNAct(in_chans, channels[0] // 2, 3, 2, rngs=rngs),
            ConvBNAct(channels[0] // 2, channels[0], 3, 2, rngs=rngs)])
        # stage 0: MBConv; stages 1-3: window attention
        dpr = [drop_path_rate * i / max(sum(depths) - 1, 1) for i in range(sum(depths))]
        self.conv_stage = nnx.List([InvertedResidual(channels[0], channels[0], 1, 4, rngs=rngs)
                                    for _ in range(depths[0])])
        attn_stages, k = [], depths[0]
        for i in range(1, 4):
            c, d, h = channels[i], depths[i], num_heads[i]
            blocks = [TinyViTWindowBlock(c, h, window_size, dpr[k + j], rngs=rngs) for j in range(d)]
            k += d
            attn_stages.append(nnx.List(blocks))
        self.attn_stages = nnx.List(attn_stages)
        self.downsamples = nnx.List([
            ConvBNAct(channels[i], channels[i + 1], 3, 2, rngs=rngs) for i in range(3)])
        self.norm = nnx.LayerNorm(channels[-1], rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate)
        self.fc = nnx.Linear(channels[-1], num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        for layer in self.stem:
            x = layer(x)
        for blk in self.conv_stage:
            x = blk(x)
        for i, stage in enumerate(self.attn_stages):
            x = self.downsamples[i](x)
            for blk in stage:
                x = blk(x)
        return self.norm(x)

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
    "tiny_vit_5m_224": ((64, 128, 256, 320), (2, 2, 4, 2), (2, 4, 8, 10)),
    "tiny_vit_11m_224": ((64, 128, 256, 448), (2, 2, 6, 2), (2, 4, 8, 14)),
    "tiny_vit_21m_224": ((96, 192, 384, 576), (2, 2, 6, 2), (3, 6, 12, 18)),
}


def _make(name):
    channels, depths, heads = _CFGS[name]

    def entry(**kwargs):
        model = TinyViT(channels, depths, heads, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
