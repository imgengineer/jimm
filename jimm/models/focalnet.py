"""FocalNet in flax nnx, NHWC. Mirrors timm.models.focalnet (focal modulation)."""
from flax import nnx

from ..layers import DropPath, ClassifierMixin
from ..registry import register_model, _cfg

class FocalModulation(nnx.Module):
    """Multi-scale dw convs produce gates aggregated into a modulator."""

    def __init__(self, dim, levels=3, *, rngs):
        self.levels = levels
        self.in_proj = nnx.Conv(dim, dim, (1, 1), rngs=rngs)
        self.dw = nnx.List([
            nnx.Conv(dim, dim, (2 * l + 3, 2 * l + 3), feature_group_count=dim, rngs=rngs)
            for l in range(levels)])
        self.gate_proj = nnx.Conv(dim, levels + 1, (1, 1), rngs=rngs)
        self.out_proj = nnx.Conv(dim, dim, (1, 1), rngs=rngs)

    def __call__(self, x):
        h = self.in_proj(x)
        ctx = nnx.gelu(h)
        ctxs = []
        for dw in self.dw:
            ctx = dw(ctx)
            ctx = nnx.gelu(ctx)
            ctxs.append(ctx)
        gates = self.gate_proj(x)  # (B,H,W,levels+1)
        agg = gates[..., :1] * h
        for l, c in enumerate(ctxs):
            agg = agg + gates[..., l + 1:l + 2] * c
        return self.out_proj(agg)

class FocalBlock(nnx.Module):
    def __init__(self, dim, levels=3, mlp_ratio=4.0, drop_path=0.0, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.mod = FocalModulation(dim, levels, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.fc1 = nnx.Linear(dim, int(dim * mlp_ratio), rngs=rngs)
        self.fc2 = nnx.Linear(int(dim * mlp_ratio), dim, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        x = x + self.drop_path(self.mod(self.norm1(x)))
        return x + self.drop_path(self.fc2(nnx.gelu(self.fc1(self.norm2(x)))))

class FocalNet(ClassifierMixin, nnx.Module):
    default_cfg: dict | None = None

    def __init__(self, channels=(96, 192, 384, 768), depths=(2, 2, 6, 2), levels=3,
                 num_classes=1000, in_chans=3, global_pool="avg", drop_rate=0.0,
                 drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = channels[-1]
        self.stem = nnx.Conv(in_chans, channels[0], (4, 4), strides=(4, 4), rngs=rngs)
        self.stem_norm = nnx.LayerNorm(channels[0], rngs=rngs)
        dpr = [drop_path_rate * i / max(sum(depths) - 1, 1) for i in range(sum(depths))]
        stages, k = [], 0
        for i, (c, d) in enumerate(zip(channels, depths)):
            blocks = [FocalBlock(c, levels, 4.0, dpr[k + j], rngs=rngs) for j in range(d)]
            k += d
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.downsamples = nnx.List([
            nnx.Sequential(nnx.LayerNorm(channels[i], rngs=rngs),
                           nnx.Conv(channels[i], channels[i + 1], (2, 2), strides=(2, 2), rngs=rngs))
            for i in range(3)])
        self.head_norm = nnx.LayerNorm(channels[-1], rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(channels[-1], num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.stem_norm(self.stem(x))
        for i, stage in enumerate(self.stages):
            if i > 0:
                x = self.downsamples[i - 1](x)
            for blk in stage:
                x = blk(x)
        return self.head_norm(x)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

_CFGS = {
    "focalnet_tiny_srf": ((96, 192, 384, 768), (2, 2, 6, 2)),
    "focalnet_small_srf": ((96, 192, 384, 768), (2, 2, 18, 2)),
    "focalnet_base_srf": ((128, 256, 512, 1024), (2, 2, 18, 2)),
}

def _make(name):
    channels, depths = _CFGS[name]

    def entry(**kwargs):
        model = FocalNet(channels, depths, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
