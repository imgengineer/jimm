"""PoolFormer in flax nnx, NHWC. Mirrors timm.models.poolformer."""
import jax.numpy as jnp
from flax import nnx

from ..layers import DropPath, ClassifierMixin
from ..registry import register_model, _cfg

class PoolFormerBlock(nnx.Module):
    def __init__(self, dim, pool_size=3, mlp_ratio=4.0, drop_path=0.0,
                 layer_scale_init=1e-5, *, rngs):
        self.pool_size = pool_size
        self.drop_path = DropPath(drop_path)
        self.norm = nnx.GroupNorm(dim, num_groups=1, rngs=rngs)
        self.mlp_fc1 = nnx.Linear(dim, int(dim * mlp_ratio), rngs=rngs)
        self.mlp_fc2 = nnx.Linear(int(dim * mlp_ratio), dim, rngs=rngs)
        self.scale1 = nnx.Param(layer_scale_init * jnp.ones(dim)) if layer_scale_init > 0 else None
        self.scale2 = nnx.Param(layer_scale_init * jnp.ones(dim)) if layer_scale_init > 0 else None

    def __call__(self, x):
        # token mixer: 3x3 avg pool - identity (no norm before pooling, per reference)
        y = nnx.avg_pool(x, (self.pool_size, self.pool_size), strides=(1, 1), padding="SAME") - x
        if self.scale1 is not None:
            y = self.scale1.value * y
        x = x + self.drop_path(y)
        y = self.mlp_fc2(nnx.gelu(self.mlp_fc1(self.norm(x))))
        if self.scale2 is not None:
            y = self.scale2.value * y
        return x + self.drop_path(y)

class PoolFormer(ClassifierMixin, nnx.Module):
    default_cfg: dict = {}

    def __init__(self, layers, embed_dims, num_classes=1000, in_chans=3, global_pool="avg",
                 drop_rate=0.0, drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = embed_dims[-1]
        dpr = [drop_path_rate * i / max(sum(layers) - 1, 1) for i in range(sum(layers))]
        patches, stages, k = [], [], 0
        for i, (n, dim) in enumerate(zip(layers, embed_dims)):
            prev = in_chans if i == 0 else embed_dims[i - 1]
            patches.append(nnx.Conv(prev, dim, (3, 3) if i else (7, 7),
                                    strides=(2, 2) if i else (4, 4), rngs=rngs))
            blocks = []
            for _ in range(n):
                blocks.append(PoolFormerBlock(dim, drop_path=dpr[k], rngs=rngs))
                k += 1
            stages.append(nnx.List(blocks))
        self.patches = nnx.List(patches)
        self.stages = nnx.List(stages)
        self.head_drop = nnx.Dropout(drop_rate)
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        for patch, stage in zip(self.patches, self.stages):
            x = patch(x)
            for blk in stage:
                x = blk(x)
        return x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

def _poolformer(layers, dims, **kwargs):
    model = PoolFormer(layers, dims, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def poolformer_s12(**kwargs):
    return _poolformer([2, 2, 6, 2], [64, 128, 320, 512], **kwargs)

@register_model
def poolformer_s24(**kwargs):
    return _poolformer([4, 4, 12, 4], [64, 128, 320, 512], **kwargs)

@register_model
def poolformer_s36(**kwargs):
    return _poolformer([6, 6, 18, 6], [64, 128, 320, 512], **kwargs)
