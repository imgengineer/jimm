"""RepViT in flax nnx, NHWC. Mirrors timm.models.repvit (reparam-style dw blocks + SE)."""
from flax import nnx

from ..layers import DropPath, SqueezeExcite, ClassifierMixin
from ..registry import register_model, _cfg

class RepViTBlock(nnx.Module):
    """token mixer: dw 3x3 (+1x1 fusion at train merged); channel MLP with SE."""

    def __init__(self, dim, mlp_ratio=2.0, use_se=False, drop_path=0.0, *, rngs):
        self.dw = nnx.Conv(dim, dim, (3, 3), use_bias=False, feature_group_count=dim, rngs=rngs)
        self.bn_dw = nnx.BatchNorm(dim, rngs=rngs)
        hidden = int(dim * mlp_ratio)
        self.se = SqueezeExcite(hidden, rd_ratio=0.25, rngs=rngs) if use_se else None  # SE on expanded features
        self.pw1 = nnx.Conv(dim, hidden, (1, 1), rngs=rngs)
        self.bn1 = nnx.BatchNorm(hidden, rngs=rngs)
        self.pw2 = nnx.Conv(hidden, dim, (1, 1), rngs=rngs)
        self.bn2 = nnx.BatchNorm(dim, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        x = x + self.bn_dw(self.dw(x))
        y = self.bn1(self.pw1(x))
        if self.se is not None:
            y = self.se(y)
        y = self.bn2(self.pw2(y))
        return x + self.drop_path(y)

class RepViT(ClassifierMixin, nnx.Module):
    default_cfg: dict | None = None

    def __init__(self, channels=(48, 96, 192, 384), depths=(2, 2, 14, 2), se_from=2,
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
        stages, chs, k = [], channels[0], 0
        for i, (c, d) in enumerate(zip(channels, depths)):
            blocks = []
            for j in range(d):
                stride_down = j == 0 and i > 0
                blocks.append(RepViTBlock(chs if j == 0 else c, 2.0, i >= se_from,
                                          dpr[k], rngs=rngs) if not stride_down else
                              RepViTDown(chs, c, rngs=rngs))
                chs = c
                k += 1
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(channels[-1], num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        for layer in self.stem:
            x = layer(x)
        for stage in self.stages:
            for blk in stage:
                x = blk(x)
        return x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

class RepViTDown(nnx.Module):
    """stride-2 transition: dw 3x3 s2 + 1x1 to out, plus residual pool+conv."""

    def __init__(self, in_chs, out_chs, *, rngs):
        self.dw = nnx.Conv(in_chs, in_chs, (3, 3), strides=(2, 2), use_bias=False,
                           feature_group_count=in_chs, rngs=rngs)
        self.bn_dw = nnx.BatchNorm(in_chs, rngs=rngs)
        self.pw = nnx.Conv(in_chs, out_chs, (1, 1), rngs=rngs)
        self.bn1 = nnx.BatchNorm(out_chs, rngs=rngs)
        self.res_pw = nnx.Conv(in_chs, out_chs, (1, 1), strides=(2, 2), use_bias=False, rngs=rngs)
        self.res_bn = nnx.BatchNorm(out_chs, rngs=rngs)

    def __call__(self, x):
        y = self.bn1(self.pw(self.bn_dw(self.dw(x))))
        return y + self.res_bn(self.res_pw(x))

_CFGS = {
    "repvit_m0_9": ((48, 96, 192, 384), (2, 2, 14, 2), 2),
    "repvit_m1_1": ((56, 112, 224, 448), (2, 2, 14, 2), 2),
    "repvit_m1_5": ((64, 128, 256, 512), (2, 2, 14, 2), 2),
}

def _make(name):
    channels, depths, se_from = _CFGS[name]

    def entry(**kwargs):
        model = RepViT(channels, depths, se_from, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
