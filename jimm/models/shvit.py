"""SHViT in flax nnx, NHWC. Mirrors timm.models.shvit (single-head vision transformer)."""
import jax.numpy as jnp
from flax import nnx

from ..layers import ConvBNAct, DropPath, ClassifierMixin
from ..registry import register_model, _cfg

class SHViTBlock(nnx.Module):
    """single-head attention over a channel chunk + dw conv on the rest + pointwise MLP."""

    def __init__(self, dim, attn_chs=64, drop_path=0.0, *, rngs):
        self.attn_chs = attn_chs
        self.dw = nnx.Conv(dim - attn_chs, dim - attn_chs, (3, 3),
                           feature_group_count=dim - attn_chs, rngs=rngs)
        self.qkv = nnx.Linear(attn_chs, attn_chs * 3, rngs=rngs)
        self.proj = nnx.Linear(attn_chs, attn_chs, rngs=rngs)
        self.norm = nnx.LayerNorm(dim, rngs=rngs)
        self.fc1 = nnx.Linear(dim, dim * 2, rngs=rngs)
        self.fc2 = nnx.Linear(dim * 2, dim, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        B, H, W, C = x.shape
        xa, xr = jnp.split(x, [self.attn_chs], axis=-1)
        # conv branch
        xr = self.dw(xr)
        # single-head attention branch
        t = xa.reshape(B, H * W, self.attn_chs)
        qkv = self.qkv(t).reshape(B, H * W, 3, self.attn_chs).transpose(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = nnx.softmax(q @ k.transpose(0, 2, 1) * (self.attn_chs ** -0.5), axis=-1)
        xa = self.proj(attn @ v).reshape(B, H, W, -1)
        x = jnp.concatenate([xa, xr], axis=-1)
        return x + self.drop_path(self.fc2(nnx.gelu(self.fc1(self.norm(x)))))

class SHViT(ClassifierMixin, nnx.Module):
    default_cfg: dict = {}

    def __init__(self, channels=(96, 192, 384, 512), depths=(2, 2, 9, 2), num_classes=1000,
                 in_chans=3, global_pool="avg", drop_rate=0.0, drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = channels[-1]
        self.stem = nnx.Conv(in_chans, channels[0], (4, 4), strides=(4, 4), rngs=rngs)
        dpr = [drop_path_rate * i / max(sum(depths) - 1, 1) for i in range(sum(depths))]
        stages, k = [], 0
        for i, (c, d) in enumerate(zip(channels, depths)):
            blocks = [SHViTBlock(c, min(c // 4, 64), dpr[k + j], rngs=rngs) for j in range(d)]
            k += d
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.downsamples = nnx.List([
            nnx.Sequential(nnx.LayerNorm(channels[i], rngs=rngs),
                           nnx.Conv(channels[i], channels[i + 1], (2, 2), strides=(2, 2), rngs=rngs))
            for i in range(3)])
        self.norm = nnx.LayerNorm(channels[-1], rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(channels[-1], num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            if i > 0:
                x = self.downsamples[i - 1](x)
            for blk in stage:
                x = blk(x)
        return self.norm(x)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

_CFGS = {
    "shvit_t1": ((96, 192, 384, 512), (1, 1, 5, 1)),
    "shvit_s1": ((96, 192, 384, 512), (2, 2, 9, 2)),
    "shvit_s2": ((128, 256, 512, 640), (2, 2, 12, 2)),
}

def _make(name):
    channels, depths = _CFGS[name]

    def entry(**kwargs):
        model = SHViT(channels, depths, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
