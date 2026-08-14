"""Sequencer2D in flax nnx. Mirrors timm.models.sequencer (LSTM-free sequence mixing via Linear)."""
from flax import nnx

from ..layers import DropPath, ClassifierMixin
from ..registry import register_model, _cfg

class SequencerBlock(nnx.Module):
    """channel MLP + spatial mixing via Linear over the H and W token dims."""

    def __init__(self, dim, h, w, mlp_ratio=3.0, drop_path=0.0, *, rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.fc1 = nnx.Linear(dim, int(dim * mlp_ratio), rngs=rngs)
        self.fc2 = nnx.Linear(int(dim * mlp_ratio), dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.h_mix = nnx.Linear(h, h, rngs=rngs)  # mix over H
        self.w_mix = nnx.Linear(w, w, rngs=rngs)  # mix over W
        self.drop_path = DropPath(drop_path)

    def __call__(self, x):
        B, H, W, C = x.shape
        y = self.norm1(x)
        x = x + self.drop_path(self.fc2(nnx.gelu(self.fc1(y))))
        y = self.norm2(x)
        # mix over H: (B, W, C, H) @ Linear(H->H)
        y = self.h_mix(y.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
        # mix over W: (B, H, C, W) @ Linear(W->W)
        y = self.w_mix(y.transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2)
        return x + self.drop_path(y)

class Sequencer2D(ClassifierMixin, nnx.Module):
    default_cfg: dict = {}

    def __init__(self, channels=(192, 384, 768), depths=(7, 7, 7), img_size=224,
                 num_classes=1000, in_chans=3, global_pool="avg", drop_rate=0.0,
                 drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = channels[-1]
        self.stem = nnx.Conv(in_chans, channels[0], (7, 7), strides=(4, 4), rngs=rngs)
        dpr = [drop_path_rate * i / max(sum(depths) - 1, 1) for i in range(sum(depths))]
        stages, k = [], 0
        res = img_size // 4
        for i, (c, d) in enumerate(zip(channels, depths)):
            if i > 0:
                res //= 2
            blocks = [SequencerBlock(c, res, res, 3.0, dpr[k + j], rngs=rngs) for j in range(d)]
            k += d
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.downsamples = nnx.List([
            nnx.Conv(channels[i], channels[i + 1], (2, 2), strides=(2, 2), rngs=rngs)
            for i in range(2)])
        self.norm = nnx.LayerNorm(channels[-1], rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate)
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
    "sequencer2d_s": ((192, 384, 768), (7, 7, 7)),
    "sequencer2d_m": ((256, 512, 1024), (7, 7, 7)),
}

def _make(name):
    channels, depths = _CFGS[name]

    def entry(**kwargs):
        model = Sequencer2D(channels, depths, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
