"""ResNeSt (Split-Attention) in flax nnx, NHWC. Mirrors timm.models.resnest."""
import jax.numpy as jnp
from flax import nnx

from ..layers import ConvBNAct, DropPath, ClassifierMixin
from ..registry import register_model, _cfg

class SplitAttnConv(nnx.Module):
    """3x3 conv split into radix branches with cardinal groups, fused by attention."""

    def __init__(self, chs, stride=1, radix=2, cardinality=1, *, rngs):
        self.radix = radix
        self.conv = nnx.Conv(chs, chs * radix, (3, 3), strides=(stride, stride),
                             use_bias=False, feature_group_count=cardinality * radix, rngs=rngs)
        self.bn = nnx.BatchNorm(chs * radix, rngs=rngs)
        self.fc1 = nnx.Linear(chs, max(chs // 2, 32), rngs=rngs)
        self.fc2 = nnx.Linear(max(chs // 2, 32), chs * radix, rngs=rngs)

    def __call__(self, x):
        B, H, W, _ = x.shape
        u = nnx.relu(self.bn(self.conv(x)))
        s = jnp.mean(u.reshape(B, H, W, self.radix, -1).sum(axis=3), axis=(1, 2))  # (B, C)
        a = self.fc2(nnx.relu(self.fc1(s))).reshape(B, self.radix, -1)
        a = nnx.softmax(a, axis=1)  # (B, radix, C)
        u = u.reshape(B, H, W, self.radix, -1)
        return (u * a[:, None, None, :, :]).sum(axis=3)

class ResNeStBottleneck(nnx.Module):
    expansion = 4

    def __init__(self, in_chs, chs, stride=1, drop_path_rate=0.0, *, rngs):
        out_chs = chs * self.expansion
        self.conv1 = nnx.Conv(in_chs, chs, (1, 1), use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(chs, rngs=rngs)
        self.avd = stride > 1
        self.splattn = SplitAttnConv(chs, stride if not self.avd else 1, rngs=rngs)
        self.conv3 = nnx.Conv(chs, out_chs, (1, 1), use_bias=False, rngs=rngs)
        self.bn3 = nnx.BatchNorm(out_chs, rngs=rngs)
        self._sc_stride = stride
        self.short_conv = ConvBNAct(in_chs, out_chs, kernel=1, stride=1, act="identity", rngs=rngs) \
            if (stride != 1 or in_chs != out_chs) else None
        self.drop_path = DropPath(drop_path_rate, rngs=rngs)

    def __call__(self, x):
        y = nnx.relu(self.bn1(self.conv1(x)))
        if self.avd:
            y = nnx.avg_pool(y, (3, 3), strides=(2, 2), padding="SAME")
        y = self.splattn(y)
        y = self.bn3(self.conv3(y))
        if self.short_conv is not None:
            sc = x
            if self._sc_stride > 1:
                sc = nnx.avg_pool(sc, (3, 3), strides=(2, 2), padding="SAME")
            sc = self.short_conv(sc)
        else:
            sc = x
        return nnx.relu(y + self.drop_path(sc))

class ResNeSt(ClassifierMixin, nnx.Module):
    default_cfg: dict = {}

    def __init__(self, layers, num_classes=1000, in_chans=3, global_pool="avg",
                 drop_rate=0.0, drop_path_rate=0.0, deep_stem=True, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = 512 * ResNeStBottleneck.expansion
        if deep_stem:
            self.stem = nnx.List([
                ConvBNAct(in_chans, 32, 3, 2, rngs=rngs),
                ConvBNAct(32, 32, 3, 1, rngs=rngs),
                ConvBNAct(32, 64, 3, 1, rngs=rngs)])
        else:
            self.stem = nnx.List([ConvBNAct(in_chans, 64, 7, 2, rngs=rngs)])
        dpr = [drop_path_rate * i / max(sum(layers) - 1, 1) for i in range(sum(layers))]
        chs, stages, k = 64, [], 0
        for i, (n, stride) in enumerate(zip(layers, [1, 2, 2, 2])):
            width = 64 * 2**i
            blocks = []
            for j in range(n):
                blocks.append(ResNeStBottleneck(chs, width, stride if j == 0 else 1, dpr[k], rngs=rngs))
                chs = width * ResNeStBottleneck.expansion
                k += 1
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        for s in self.stem:
            x = s(x)
        x = nnx.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        for stage in self.stages:
            for blk in stage:
                x = blk(x)
        return x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

def _resnest(layers, deep_stem=True, **kwargs):
    model = ResNeSt(layers, deep_stem=deep_stem, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def resnest14d(**kwargs):
    return _resnest([1, 1, 1, 1], **kwargs)

@register_model
def resnest50d(**kwargs):
    return _resnest([3, 4, 6, 3], **kwargs)

@register_model
def resnest101e(**kwargs):
    return _resnest([3, 4, 23, 3], **kwargs)
