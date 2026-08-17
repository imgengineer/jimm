"""ConvNeXt in flax nnx, NHWC (natural for ConvNeXt). Mirrors timm.models.convnext."""
import jax.numpy as jnp
from flax import nnx

from ..layers import DropPath, global_pool_nhwc, ClassifierMixin
from ..registry import register_model, _cfg

class ConvNeXtBlock(nnx.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init=1e-6, *, rngs):
        self.dwconv = nnx.Conv(dim, dim, (7, 7), feature_group_count=dim, rngs=rngs)
        self.norm = nnx.LayerNorm(dim, rngs=rngs)
        self.pw1 = nnx.Linear(dim, 4 * dim, rngs=rngs)
        self.pw2 = nnx.Linear(4 * dim, dim, rngs=rngs)
        self.gamma = nnx.Param(layer_scale_init * jnp.ones(dim)) if layer_scale_init > 0 else None
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        y = self.dwconv(x)
        y = self.norm(y)
        y = self.pw2(nnx.gelu(self.pw1(y)))
        if self.gamma is not None:
            y = self.gamma[...] * y
        return x + self.drop_path(y)

class ConvNeXt(ClassifierMixin, nnx.Module):
    default_cfg: dict | None = None

    def __init__(self, depths, dims, num_classes=1000, in_chans=3, global_pool="avg",
                 drop_rate=0.0, drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.stem = nnx.Conv(in_chans, dims[0], (4, 4), strides=(4, 4), rngs=rngs)
        self.stem_norm = nnx.LayerNorm(dims[0], rngs=rngs)
        dpr = [drop_path_rate * i / max(sum(depths) - 1, 1) for i in range(sum(depths))]
        stages, k = [], 0
        for i, (n, dim) in enumerate(zip(depths, dims)):
            blocks = []
            if i > 0:
                blocks.append("D")  # downsample between stages
            for _ in range(n):
                blocks.append(ConvNeXtBlock(dim, dpr[k], rngs=rngs))
                k += 1
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.downsamples = nnx.List([
            nnx.Sequential(nnx.LayerNorm(dims[i], rngs=rngs),
                           nnx.Conv(dims[i], dims[i + 1], (2, 2), strides=(2, 2), rngs=rngs))
            for i in range(3)])
        self.num_features = dims[-1]
        self.head_norm = nnx.LayerNorm(dims[-1], rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(dims[-1], num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.stem_norm(self.stem(x))
        for i, stage in enumerate(self.stages):
            if i > 0:
                x = self.downsamples[i - 1](x)
            for blk in stage:
                if blk == "D":
                    continue
                x = blk(x)
        return x

    def forward_head(self, x):
        x = global_pool_nhwc(x, self.global_pool)
        x = self.head_norm(x)
        x = self.head_drop(x)
        return self.fc(x) if self.fc is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

def _convnext(depths, dims, **kwargs):
    model = ConvNeXt(depths, dims, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def convnext_atto(**kwargs):
    return _convnext([2, 2, 6, 2], [40, 80, 160, 320], **kwargs)

@register_model
def convnext_tiny(**kwargs):
    return _convnext([3, 3, 9, 3], [96, 192, 384, 768], **kwargs)

@register_model
def convnext_small(**kwargs):
    return _convnext([3, 3, 27, 3], [96, 192, 384, 768], **kwargs)

@register_model
def convnext_base(**kwargs):
    return _convnext([3, 3, 27, 3], [128, 256, 512, 1024], **kwargs)

@register_model
def convnext_large(**kwargs):
    return _convnext([3, 3, 27, 3], [192, 384, 768, 1536], **kwargs)
