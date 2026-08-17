"""SKNet (Selective Kernel) in flax nnx, NHWC. Mirrors timm.models.sknet."""
import jax.numpy as jnp
from flax import nnx

from ..layers import DropPath, ClassifierMixin
from ..registry import register_model, _cfg
from .resnet import Downsample

class SKConv(nnx.Module):
    """Two kernel branches (3x3, 3x3 dilation 2) fused by channel attention."""

    def __init__(self, chs, stride=1, groups=32, rd_ratio=0.0625, rd_divisor=16, *, rngs):
        self.conv1 = nnx.Conv(chs, chs, (3, 3), strides=(stride, stride), use_bias=False,
                              feature_group_count=groups, rngs=rngs)
        self.bn1 = nnx.BatchNorm(chs, rngs=rngs)
        self.conv2 = nnx.Conv(chs, chs, (3, 3), strides=(stride, stride), use_bias=False,
                              feature_group_count=groups, kernel_dilation=(2, 2), rngs=rngs)
        self.bn2 = nnx.BatchNorm(chs, rngs=rngs)
        rd = max(int(chs * rd_ratio), rd_divisor)
        self.fc = nnx.Linear(chs, rd, rngs=rngs)
        self.fc2 = nnx.Linear(rd, chs * 2, rngs=rngs)

    def __call__(self, x):
        u1 = nnx.relu(self.bn1(self.conv1(x)))
        u2 = nnx.relu(self.bn2(self.conv2(x)))
        s = jnp.mean(u1 + u2, axis=(1, 2))
        z = self.fc2(nnx.relu(self.fc(s))).reshape(x.shape[0], 2, -1)
        a = nnx.softmax(z, axis=1)  # (B, 2, C)
        return a[:, 0, None, None, :] * u1 + a[:, 1, None, None, :] * u2

class SKBottleneck(nnx.Module):
    expansion = 4

    def __init__(self, in_chs, chs, stride=1, drop_path_rate=0.0, *, rngs):
        out_chs = chs * self.expansion
        mid = chs * 2  # SKNet bottleneck width = chs * 2
        self.conv1 = nnx.Conv(in_chs, mid, (1, 1), use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(mid, rngs=rngs)
        self.sk = SKConv(mid, stride, rngs=rngs)
        self.conv3 = nnx.Conv(mid, out_chs, (1, 1), use_bias=False, rngs=rngs)
        self.bn3 = nnx.BatchNorm(out_chs, rngs=rngs)
        self.shortcut = Downsample(in_chs, out_chs, stride, rngs=rngs) if (stride != 1 or in_chs != out_chs) else None
        self.drop_path = DropPath(drop_path_rate, rngs=rngs)

    def __call__(self, x):
        y = nnx.relu(self.bn1(self.conv1(x)))
        y = self.sk(y)
        y = self.bn3(self.conv3(y))
        sc = x if self.shortcut is None else self.shortcut(x)
        return nnx.relu(y + self.drop_path(sc))

class SKNet(ClassifierMixin, nnx.Module):
    default_cfg: dict | None = None

    def __init__(self, layers, num_classes=1000, in_chans=3, global_pool="avg",
                 drop_rate=0.0, drop_path_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = 512 * SKBottleneck.expansion
        self.conv1 = nnx.Conv(in_chans, 64, (7, 7), strides=(2, 2), padding=[(3, 3), (3, 3)],
                              use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(64, rngs=rngs)
        dpr = [drop_path_rate * i / max(sum(layers) - 1, 1) for i in range(sum(layers))]
        chs, stages, k = 64, [], 0
        for i, (n, stride) in enumerate(zip(layers, [1, 2, 2, 2])):
            width = 64 * 2**i
            blocks = []
            for j in range(n):
                blocks.append(SKBottleneck(chs, width, stride if j == 0 else 1, dpr[k], rngs=rngs))
                chs = width * SKBottleneck.expansion
                k += 1
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = nnx.max_pool(nnx.relu(self.bn1(self.conv1(x))), (3, 3), strides=(2, 2), padding="SAME")
        for stage in self.stages:
            for blk in stage:
                x = blk(x)
        return x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

def _sknet(layers, **kwargs):
    model = SKNet(layers, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def skresnet50(**kwargs):
    return _sknet([3, 4, 6, 3], **kwargs)

@register_model
def skresnet101(**kwargs):
    return _sknet([3, 4, 23, 3], **kwargs)
