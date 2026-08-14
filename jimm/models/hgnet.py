"""HGNet in flax nnx, NHWC. Mirrors timm.models.hgnet (hierarchical grouped dw conv blocks)."""
import jax.numpy as jnp
from flax import nnx

from ..layers import ConvBNAct, SqueezeExcite, global_pool_nhwc
from ..registry import register_model, _cfg


class HGBlock(nnx.Module):
    """stack of dw convs with increasing kernel + 1x1 expand/project, residual."""

    def __init__(self, in_chs, mid_chs, out_chs, kernel=3, layers=6, stride=1, se=False, *, rngs):
        self.layers = layers
        convs = [ConvBNAct(in_chs, mid_chs, kernel, stride, act="silu", rngs=rngs)]
        for _ in range(layers - 1):
            convs.append(ConvBNAct(mid_chs, mid_chs, kernel, 1, groups=mid_chs, act="silu", rngs=rngs))
        self.convs = nnx.List(convs)
        self.pw = ConvBNAct(mid_chs, out_chs, 1, act="identity", rngs=rngs)
        self.se = SqueezeExcite(mid_chs, rd_ratio=0.25, rngs=rngs) if se else None
        self.use_sc = stride != 1 or in_chs != out_chs
        if self.use_sc:
            self.sc = ConvBNAct(in_chs, out_chs, 1, stride, act="identity", rngs=rngs)

    def __call__(self, x):
        y = x
        for conv in self.convs:
            y = conv(y)
        if self.se is not None:
            y = self.se(y)
        y = self.pw(y)
        sc = self.sc(x) if self.use_sc else x
        return nnx.relu(y + sc)


_CFGS = {  # (mid, out, layers, stride, se) per stage
    "hgnet_tiny": [(48, 48, 6, 2, 0), (128, 96, 6, 2, 0), (512, 192, 6, 2, 1), (1024, 384, 6, 2, 1)],
    "hgnet_small": [(96, 96, 6, 2, 0), (256, 192, 6, 2, 0), (768, 384, 6, 2, 1), (1536, 768, 6, 2, 1)],
    "hgnet_base": [(160, 192, 7, 2, 0), (352, 256, 7, 2, 0), (1024, 512, 7, 2, 1), (2048, 1024, 7, 2, 1)],
}


class HGNet(nnx.Module):
    default_cfg: dict = {}

    def __init__(self, cfg, num_classes=1000, in_chans=3, global_pool="avg", drop_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.stem = nnx.List([
            ConvBNAct(in_chans, 32, 3, 2, act="silu", rngs=rngs),
            ConvBNAct(32, 32, 3, 1, act="silu", rngs=rngs),
            ConvBNAct(32, 48, 3, 2, act="silu", rngs=rngs)])
        stages, chs = [], 48
        for mid, out, layers, stride, se in cfg:
            stages.append(HGBlock(chs, mid, out, 3, layers, stride, se, rngs=rngs))
            chs = out
        self.stages = nnx.List(stages)
        self.num_features = cfg[-1][1]
        self.head_drop = nnx.Dropout(drop_rate)
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        for layer in self.stem:
            x = layer(x)
        for stage in self.stages:
            x = stage(x)
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


def _make(name):
    cfg = _CFGS[name]

    def entry(**kwargs):
        model = HGNet(cfg, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
