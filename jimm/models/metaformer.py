"""MetaFormer (CAFormer) in flax nnx, NHWC. Mirrors timm.models.metaformer (conv + attention metaformers)."""
import jax.numpy as jnp
from flax import nnx

from ..layers import DropPath, ClassifierMixin
from ..registry import register_model, _cfg
from .vision_transformer import Attention

class ConvFormerBlock(nnx.Module):
    """dw 3x3 token mixer + MLP, layer scale, residual."""

    def __init__(self, dim, mlp_ratio=4.0, drop_path=0.0, layer_scale=1e-5, *, rngs):
        self.dw = nnx.Conv(dim, dim, (3, 3), feature_group_count=dim, rngs=rngs)
        self.norm = nnx.LayerNorm(dim, rngs=rngs)
        self.fc1 = nnx.Linear(dim, int(dim * mlp_ratio), rngs=rngs)
        self.fc2 = nnx.Linear(int(dim * mlp_ratio), dim, rngs=rngs)
        self.scale1 = nnx.Param(layer_scale * jnp.ones(dim))
        self.scale2 = nnx.Param(layer_scale * jnp.ones(dim))
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        x = x + self.drop_path(self.scale1[...] * self.dw(x))
        return x + self.drop_path(self.scale2[...] * self.fc2(nnx.gelu(self.fc1(self.norm(x)))))

class AttnFormerBlock(nnx.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop_path=0.0, layer_scale=1e-5, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = Attention(dim, num_heads, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.fc1 = nnx.Linear(dim, int(dim * mlp_ratio), rngs=rngs)
        self.fc2 = nnx.Linear(int(dim * mlp_ratio), dim, rngs=rngs)
        self.scale1 = nnx.Param(layer_scale * jnp.ones(dim))
        self.scale2 = nnx.Param(layer_scale * jnp.ones(dim))
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        x = x + self.drop_path(self.scale1[...] * self.attn(self.norm1(x)))
        return x + self.drop_path(self.scale2[...] * self.fc2(nnx.gelu(self.fc1(self.norm2(x)))))

class MetaFormer(ClassifierMixin, nnx.Module):
    default_cfg: dict | None = None

    def __init__(self, channels=(64, 128, 320, 512), depths=(3, 3, 9, 3), attn_from=2,
                 num_classes=1000, in_chans=3, global_pool="avg", drop_rate=0.0,
                 drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = channels[-1]
        self.stem = nnx.Conv(in_chans, channels[0], (7, 7), strides=(4, 4), rngs=rngs)
        dpr = [drop_path_rate * i / max(sum(depths) - 1, 1) for i in range(sum(depths))]
        stages, k = [], 0
        for i, (c, d) in enumerate(zip(channels, depths)):
            blocks = []
            for j in range(d):
                if i >= attn_from:
                    blocks.append(AttnFormerBlock(c, 8, 4.0, dpr[k], rngs=rngs))
                else:
                    blocks.append(ConvFormerBlock(c, 4.0, dpr[k], rngs=rngs))
                k += 1
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.downsamples = nnx.List([
            nnx.Conv(channels[i], channels[i + 1], (3, 3), strides=(2, 2), rngs=rngs)
            for i in range(3)])
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(channels[-1], num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            if i > 0:
                x = self.downsamples[i - 1](x)
            B, H, W, C = x.shape
            for blk in stage:
                if isinstance(blk, AttnFormerBlock):
                    x = blk(x.reshape(B, H * W, C)).reshape(B, H, W, C)
                else:
                    x = blk(x)
        return x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

_CFGS = {  # channels, depths, attn_from
    "caformer_s18": ((64, 128, 320, 512), (3, 3, 9, 3), 2),
    "caformer_s36": ((64, 128, 320, 512), (3, 6, 18, 3), 2),
    "caformer_b36": ((96, 192, 384, 768), (3, 6, 18, 3), 2),
}

def _make(name):
    channels, depths, attn_from = _CFGS[name]

    def entry(**kwargs):
        model = MetaFormer(channels, depths, attn_from, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
