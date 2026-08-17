"""SENet-154 in flax nnx, NHWC. Mirrors timm.models.senet (deep stem + SE bottlenecks, groups=32)."""
from flax import nnx

from ..layers import ConvBNAct, SqueezeExcite, ClassifierMixin
from ..registry import register_model, _cfg

class SENetBottleneck(nnx.Module):
    expansion = 2  # senet154 uses expansion 2 with groups

    def __init__(self, in_chs, chs, stride=1, groups=32, *, rngs):
        out_chs = chs * self.expansion
        self.conv1 = ConvBNAct(in_chs, chs, 1, rngs=rngs)
        self.conv2 = ConvBNAct(chs, chs, 3, stride, groups=groups, rngs=rngs)
        self.conv3 = ConvBNAct(chs, out_chs, 1, act="identity", rngs=rngs)
        self.se = SqueezeExcite(out_chs, rd_ratio=0.0625, rngs=rngs)
        self.short_conv = ConvBNAct(in_chs, out_chs, 1, stride, act="identity", rngs=rngs) \
            if (stride != 1 or in_chs != out_chs) else None

    def __call__(self, x):
        y = self.conv3(self.conv2(self.conv1(x)))
        y = self.se(y)
        sc = x if self.short_conv is None else self.short_conv(x)
        return nnx.relu(y + sc)

class SENet154(ClassifierMixin, nnx.Module):
    default_cfg: dict | None = None

    def __init__(self, layers=(3, 8, 36, 3), num_classes=1000, in_chans=3,
                 global_pool="avg", drop_rate=0.2, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.stem = nnx.List([
            ConvBNAct(in_chans, 64, 3, 2, rngs=rngs),
            ConvBNAct(64, 64, 3, 1, rngs=rngs),
            ConvBNAct(64, 128, 3, 1, rngs=rngs)])
        # senet154 stage widths (bottleneck chs): 64,128,256,512 with expansion 2
        chs, stages = 128, []
        for i, (n, stride) in enumerate(zip(layers, [1, 2, 2, 2])):
            width = 64 * 2**i
            blocks = []
            for j in range(n):
                blocks.append(SENetBottleneck(chs, width, stride if j == 0 else 1, rngs=rngs))
                chs = width * SENetBottleneck.expansion
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.num_features = 512 * SENetBottleneck.expansion  # final stage out: 512*2 = 1024
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        for layer in self.stem:
            x = layer(x)
        x = nnx.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        for stage in self.stages:
            for blk in stage:
                x = blk(x)
        return x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

@register_model
def senet154(**kwargs):
    model = SENet154(**kwargs)
    model.default_cfg = _cfg(input_size=(3, 224, 224))
    return model
