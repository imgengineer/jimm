"""FastViT in flax nnx, NHWC. Mirrors timm.models.fastvit (RepMixer + attention stages)."""
import jax.numpy as jnp
from flax import nnx

from ..layers import DropPath, global_pool_nhwc
from ..registry import register_model, _cfg
from .vision_transformer import Attention


class RepMixerBlock(nnx.Module):
    """dw 7x7 residual token mixer + pointwise MLP."""

    def __init__(self, dim, mlp_ratio=3.0, drop_path=0.0, *, rngs):
        self.dw = nnx.Conv(dim, dim, (7, 7), use_bias=False, feature_group_count=dim, rngs=rngs)
        self.bn = nnx.BatchNorm(dim, rngs=rngs)
        self.mlp1 = nnx.Conv(dim, int(dim * mlp_ratio), (1, 1), rngs=rngs)
        self.mlp_bn1 = nnx.BatchNorm(int(dim * mlp_ratio), rngs=rngs)
        self.mlp2 = nnx.Conv(int(dim * mlp_ratio), dim, (1, 1), rngs=rngs)
        self.mlp_bn2 = nnx.BatchNorm(dim, rngs=rngs)
        self.drop_path = DropPath(drop_path)

    def __call__(self, x):
        y = x + self.bn(self.dw(x))  # RepMixer: add before act
        y = nnx.relu(y)
        y = self.mlp_bn2(self.mlp2(self.mlp_bn1(self.mlp1(y))))
        return x + self.drop_path(y)


class AttnStage(nnx.Module):
    def __init__(self, dim, num_heads, depth, drop_path=0.0, *, rngs):
        self.blocks = nnx.List([])
        blocks = []
        for _ in range(depth):
            blocks.append(nnx.List([
                nnx.LayerNorm(dim, rngs=rngs),
                Attention(dim, num_heads, rngs=rngs),
                nnx.LayerNorm(dim, rngs=rngs),
                nnx.Linear(dim, dim * 2, rngs=rngs),
                nnx.Linear(dim * 2, dim, rngs=rngs),
                DropPath(drop_path)]))
        self.blocks = nnx.List(blocks)

    def __call__(self, x):
        B, H, W, C = x.shape
        t = x.reshape(B, H * W, C)
        for norm1, attn, norm2, fc1, fc2, dp in self.blocks:
            t = t + dp(attn(norm1(t)))
            t = t + dp(fc2(nnx.gelu(fc1(norm2(t)))))
        return t.reshape(B, H, W, C)


class FastViT(nnx.Module):
    default_cfg: dict = {}

    def __init__(self, channels=(48, 96, 192, 384), depths=(2, 2, 6, 2), attn_depths=(0, 0, 0, 2),
                 num_classes=1000, in_chans=3, global_pool="avg", drop_rate=0.0,
                 drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = channels[-1]
        self.stem = nnx.List([
            nnx.Conv(in_chans, channels[0] // 2, (3, 3), strides=(2, 2), use_bias=False, rngs=rngs),
            nnx.BatchNorm(channels[0] // 2, rngs=rngs),
            nnx.Conv(channels[0] // 2, channels[0], (3, 3), strides=(2, 2), use_bias=False, rngs=rngs),
            nnx.BatchNorm(channels[0], rngs=rngs)])
        dpr = [drop_path_rate * i / max(sum(depths) - 1, 1) for i in range(sum(depths))]
        stages, k = [], 0
        for i, (c, d, ad) in enumerate(zip(channels, depths, attn_depths)):
            blocks: list = [RepMixerBlock(c, 3.0, dpr[k + j], rngs=rngs) for j in range(d)]
            k += d
            if ad:
                blocks.append(AttnStage(c, 8, ad, drop_path_rate, rngs=rngs))
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.downsamples = nnx.List([
            nnx.Sequential(nnx.Conv(channels[i], channels[i], (3, 3), strides=(2, 2),
                                    feature_group_count=channels[i], use_bias=False, rngs=rngs),
                           nnx.BatchNorm(channels[i], rngs=rngs),
                           nnx.Conv(channels[i], channels[i + 1], (1, 1), use_bias=False, rngs=rngs),
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
            for blk in stage:
                x = blk(x)
        return x

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
    "fastvit_t8": ((48, 96, 192, 384), (2, 2, 4, 2), (0, 0, 0, 1)),
    "fastvit_t12": ((64, 128, 256, 512), (2, 2, 6, 2), (0, 0, 0, 1)),
    "fastvit_s12": ((64, 128, 256, 512), (2, 2, 8, 2), (0, 0, 1, 1)),
}


def _make(name):
    channels, depths, ad = _CFGS[name]

    def entry(**kwargs):
        model = FastViT(channels, depths, ad, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
