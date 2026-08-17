"""SwiftFormer in flax nnx, NHWC. Mirrors timm.models.swiftformer (conv encoder + eff additive attention)."""
import jax.numpy as jnp
from flax import nnx

from ..layers import ConvBNAct, DropPath, ClassifierMixin
from ..registry import register_model, _cfg

class ConvEncoder(nnx.Module):
    """dw 3x3 + two 1x1 pointwise, residual."""

    def __init__(self, dim, drop_path=0.0, *, rngs):
        self.dw = ConvBNAct(dim, dim, 3, groups=dim, act="identity", rngs=rngs)
        self.pw1 = ConvBNAct(dim, dim * 2, 1, act="gelu", use_bn=False, rngs=rngs)
        self.pw2 = ConvBNAct(dim * 2, dim, 1, act="identity", use_bn=False, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        return x + self.drop_path(self.pw2(self.pw1(self.dw(x))))

class SwiftAttention(nnx.Module):
    """Additive attention: query->score weights context vector."""

    def __init__(self, dim, *, rngs):
        self.q = nnx.Linear(dim, dim, rngs=rngs)
        self.k = nnx.Linear(dim, dim, rngs=rngs)
        self.v = nnx.Linear(dim, dim, rngs=rngs)
        self.score = nnx.Linear(dim, 1, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(self, x):
        B, N, C = x.shape
        w = nnx.softmax(self.score(self.k(x)).transpose(0, 2, 1), axis=-1)  # (B,1,N)
        ctx = (w @ self.v(x))  # (B,1,C)
        ctx = jnp.broadcast_to(ctx, (B, N, C))
        return self.proj(self.q(x) * ctx)

class SwiftFormerBlock(nnx.Module):
    def __init__(self, dim, drop_path=0.0, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = SwiftAttention(dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.fc1 = nnx.Linear(dim, dim * 2, rngs=rngs)
        self.fc2 = nnx.Linear(dim * 2, dim, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        B, H, W, C = x.shape
        t = x.reshape(B, H * W, C)
        t = t + self.drop_path(self.attn(self.norm1(t)))
        t = t + self.drop_path(self.fc2(nnx.gelu(self.fc1(self.norm2(t)))))
        return t.reshape(B, H, W, C)

class SwiftFormer(ClassifierMixin, nnx.Module):
    default_cfg: dict | None = None

    def __init__(self, channels=(48, 56, 112, 224), depths=(3, 3, 9, 3), swift_from=2,
                 num_classes=1000, in_chans=3, global_pool="avg", drop_rate=0.0,
                 drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = channels[-1]
        self.stem = nnx.List([
            ConvBNAct(in_chans, channels[0] // 2, 3, 2, rngs=rngs),
            ConvBNAct(channels[0] // 2, channels[0], 3, 2, rngs=rngs)])
        dpr = [drop_path_rate * i / max(sum(depths) - 1, 1) for i in range(sum(depths))]
        stages, k = [], 0
        for i, (c, d) in enumerate(zip(channels, depths)):
            blocks = []
            for j in range(d):
                if i >= swift_from:
                    blocks.append(SwiftFormerBlock(c, dpr[k], rngs=rngs))
                else:
                    blocks.append(ConvEncoder(c, dpr[k], rngs=rngs))
                k += 1
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.downsamples = nnx.List([
            ConvBNAct(channels[i], channels[i + 1], 3, 2, rngs=rngs) for i in range(3)])
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
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

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

_CFGS = {
    "swiftformer_xs": ((48, 56, 112, 224), (3, 3, 9, 3)),
    "swiftformer_s": ((48, 64, 168, 224), (3, 3, 9, 3)),
    "swiftformer_l1": ((48, 96, 192, 384), (3, 4, 12, 4)),
}

def _make(name):
    channels, depths = _CFGS[name]

    def entry(**kwargs):
        model = SwiftFormer(channels, depths, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
