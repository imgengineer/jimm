"""CPUBone in flax nnx, NHWC. Mirrors timm.models.cpubone."""
from flax import nnx

from ..layers import ConvBNAct, SqueezeExcite, ClassifierMixin
from ..registry import register_model, _cfg

class FusedMBConv(nnx.Module):
    """Fused 3x3 conv -> 1x1 project."""

    def __init__(self, in_chs, out_chs, stride=1, expand=4, *, rngs):
        mid = in_chs * expand
        self.conv1 = ConvBNAct(in_chs, mid, 3, stride, act="silu", rngs=rngs)
        self.conv2 = ConvBNAct(mid, out_chs, 1, act="identity", rngs=rngs)
        self.use_sc = stride == 1 and in_chs == out_chs

    def __call__(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.use_sc else y

class CPUMBConv(nnx.Module):
    """1x1 expand -> 3x3 dw -> SE -> 1x1 project."""

    def __init__(self, in_chs, out_chs, stride=1, expand=4, *, rngs):
        mid = in_chs * expand
        self.conv1 = ConvBNAct(in_chs, mid, 1, act="silu", rngs=rngs)
        self.dw = ConvBNAct(mid, mid, 3, stride, groups=mid, act="silu", rngs=rngs)
        self.se = SqueezeExcite(mid, rd_ratio=0.25, rngs=rngs)
        self.pw = ConvBNAct(mid, out_chs, 1, act="identity", rngs=rngs)
        self.use_sc = stride == 1 and in_chs == out_chs

    def __call__(self, x):
        y = self.pw(self.se(self.dw(self.conv1(x))))
        return x + y if self.use_sc else y

class CPUBone(ClassifierMixin, nnx.Module):
    default_cfg: dict = {}

    def __init__(self, width_list=(16, 32, 64, 128, 256), depth_list=(0, 1, 1, 3, 4),
                 head_widths=(1280, 1536), num_classes=1000, in_chans=3, global_pool="avg",
                 drop_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = head_widths[0]

        # Stem
        self.stem = ConvBNAct(in_chans, width_list[0], 3, 2, act="silu", rngs=rngs)
        
        stages, chs = [], width_list[0]
        for i, (w, d) in enumerate(zip(width_list[1:], depth_list[1:])):
            blocks = []
            for j in range(d):
                stride = 2 if j == 0 else 1
                if i < 2:
                    blocks.append(FusedMBConv(chs, w, stride=stride, expand=4, rngs=rngs))
                else:
                    blocks.append(CPUMBConv(chs, w, stride=stride, expand=4, rngs=rngs))
                chs = w
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)

        self.head_conv = ConvBNAct(chs, head_widths[0], 1, act="silu", rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(head_widths[0], num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.stem(x)
        for stage in self.stages:
            for blk in stage:
                x = blk(x)
        return self.head_conv(x)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

_CFGS = {
    "cpubone_nano": dict(width_list=[12, 24, 48, 96, 192], depth_list=[0, 1, 1, 1, 2], head_widths=(768, 1024)),
    "cpubone_t0": dict(width_list=[12, 24, 48, 96, 192], depth_list=[0, 1, 1, 2, 3], head_widths=(768, 1024)),
    "cpubone_s0": dict(width_list=[14, 28, 56, 112, 224], depth_list=[0, 1, 1, 2, 3], head_widths=(1024, 1280)),
    "cpubone_b0_bfrobust": dict(width_list=[16, 32, 64, 128, 256], depth_list=[0, 1, 1, 3, 4], head_widths=(1280, 1536)),
    "cpubone_b1_bfrobust": dict(width_list=[20, 40, 80, 160, 320], depth_list=[0, 1, 1, 6, 6], head_widths=(2304, 2560)),
    "cpubone_b1_dwnorm": dict(width_list=[20, 40, 80, 160, 320], depth_list=[0, 1, 1, 6, 6], head_widths=(2304, 2560)),
    "cpubone_b2_bfrobust": dict(width_list=[24, 48, 96, 192, 384], depth_list=[0, 1, 1, 6, 6], head_widths=(2304, 2560)),
    "cpubone_b2pt5_dwnorm": dict(width_list=[24, 48, 96, 192, 384], depth_list=[0, 1, 1, 6, 6], head_widths=(2304, 2560)),
    "cpubone_b3": dict(width_list=[32, 64, 128, 256, 512], depth_list=[1, 2, 3, 6, 6], head_widths=(2560, 2816)),
}

def _make(name):
    cfg = _CFGS[name]

    def entry(**kwargs):
        model = CPUBone(**dict(cfg, **kwargs))
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
