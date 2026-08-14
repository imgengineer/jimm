"""VoVNet v1/v2 in flax nnx, NHWC. Mirrors timm.models.vovnet."""
import jax.numpy as jnp
from flax import nnx

from ..layers import ConvBNAct, SqueezeExcite, global_pool_nhwc
from ..registry import register_model, _cfg


class OSABlock(nnx.Module):
    """One-Shot Aggregation: dense conv chain, concat all, final 1x1."""

    def __init__(self, in_chs, mid_chs, out_chs, layers=5, *, rngs):
        # chain: first conv in_chs -> mid, rest mid -> mid; concat ALL features at the end
        convs = [ConvBNAct(in_chs if i == 0 else mid_chs, mid_chs, 3, rngs=rngs)
                 for i in range(layers)]
        self.convs = nnx.List(convs)
        self.concat_conv = ConvBNAct(in_chs + layers * mid_chs, out_chs, 1, rngs=rngs)

    def __call__(self, x):
        feats = [x]
        for conv in self.convs:
            feats.append(conv(feats[-1]))
        return self.concat_conv(jnp.concatenate(feats, axis=-1))


class VoVNetStage(nnx.Module):
    def __init__(self, in_chs, mid_chs, out_chs, blocks, layers_per_block, ese, *, rngs):
        self.pool = lambda x: nnx.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        blocks_list, chs = [], in_chs
        for _ in range(blocks):
            blocks_list.append(OSABlock(chs, mid_chs, out_chs, layers_per_block, rngs=rngs))
            chs = out_chs
        self.blocks = nnx.List(blocks_list)
        self.ese = SqueezeExcite(out_chs, rngs=rngs) if ese else None

    def __call__(self, x):
        x = self.pool(x)
        for blk in self.blocks:
            x = blk(x)
        if self.ese is not None:
            x = self.ese(x)
        return x


# (mid, out, blocks, layers_per_block) per stage, 4 stages
_CFGS = {
    "vovnet39a":  ([(128, 256, 1, 5), (160, 512, 2, 5), (192, 768, 3, 5), (224, 1024, 2, 5)], False),
    "vovnet57a":  ([(192, 384, 1, 5), (224, 640, 2, 5), (256, 896, 3, 5), (320, 1152, 2, 5)], False),
    "ese_vovnet19b": ([(64, 128, 1, 3), (80, 256, 1, 3), (96, 512, 2, 3), (112, 768, 2, 3)], True),
    "ese_vovnet39b": ([(128, 256, 1, 5), (160, 512, 2, 5), (192, 768, 3, 5), (224, 1024, 2, 5)], True),
}


class VoVNet(nnx.Module):
    default_cfg: dict = {}

    def __init__(self, stages_cfg, ese, num_classes=1000, in_chans=3, global_pool="avg",
                 drop_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.stem = nnx.List([
            ConvBNAct(in_chans, 64, 3, 2, rngs=rngs),
            ConvBNAct(64, 64, 3, 1, rngs=rngs),
            ConvBNAct(64, 128, 3, 1, rngs=rngs)])
        self.stages = nnx.List([
            VoVNetStage(128 if i == 0 else stages_cfg[i - 1][1], mid, out, blocks, lpb, ese, rngs=rngs)
            for i, (mid, out, blocks, lpb) in enumerate(stages_cfg)])
        self.num_features = stages_cfg[-1][1]
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
    stages_cfg, ese = _CFGS[name]

    def entry(**kwargs):
        model = VoVNet(stages_cfg, ese, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
