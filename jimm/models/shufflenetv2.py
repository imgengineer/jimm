"""ShuffleNetV2 in flax nnx, NHWC. Mirrors timm.models.shufflenetv2."""
import jax.numpy as jnp
from flax import nnx

from ..layers import global_pool_nhwc, ClassifierMixin
from ..registry import register_model, _cfg

def channel_shuffle(x, groups):
    b, h, w, c = x.shape
    x = x.reshape(b, h, w, groups, c // groups).transpose(0, 1, 2, 4, 3)
    return x.reshape(b, h, w, c)

class LeftBranch(nnx.Module):
    """stride-2 left branch: 3x3 dw -> 1x1."""

    def __init__(self, in_chs, mid, *, rngs):
        self.dw = nnx.Conv(in_chs, in_chs, (3, 3), strides=(2, 2), use_bias=False,
                           feature_group_count=in_chs, rngs=rngs)
        self.bn_dw = nnx.BatchNorm(in_chs, rngs=rngs)
        self.conv = nnx.Conv(in_chs, mid, (1, 1), use_bias=False, rngs=rngs)
        self.bn = nnx.BatchNorm(mid, rngs=rngs)

    def __call__(self, x):
        return self.bn(self.conv(self.bn_dw(self.dw(x))))

class ShuffleUnit(nnx.Module):
    def __init__(self, in_chs, out_chs, stride, *, rngs):
        self.stride = stride
        mid = out_chs // 2
        # right branch: 1x1 -> 3x3 dw -> 1x1 ; left branch (stride 2): 3x3 dw -> 1x1
        self.conv1 = nnx.Conv(in_chs if stride == 2 else mid, mid, (1, 1), use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(mid, rngs=rngs)
        self.dw = nnx.Conv(mid, mid, (3, 3), strides=(stride, stride), use_bias=False,
                           feature_group_count=mid, rngs=rngs)
        self.bn_dw = nnx.BatchNorm(mid, rngs=rngs)
        self.conv2 = nnx.Conv(mid, mid, (1, 1), use_bias=False, rngs=rngs)
        self.bn2 = nnx.BatchNorm(mid, rngs=rngs)
        self.left = LeftBranch(in_chs, mid, rngs=rngs) if stride == 2 else None

    def __call__(self, x):
        if self.left is None:
            x1, x2 = jnp.split(x, 2, axis=-1)
        else:
            x2 = x
            x1 = self.left(x)
        y = nnx.relu(self.bn1(self.conv1(x2)))
        y = self.bn_dw(self.dw(y))
        y = nnx.relu(self.bn2(self.conv2(y)))
        return channel_shuffle(jnp.concatenate([x1, y], axis=-1), 2)

class ShuffleNetV2(ClassifierMixin, nnx.Module):
    default_cfg: dict = {}

    def __init__(self, out_chs, repeats, num_classes=1000, in_chans=3, global_pool="avg", *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.conv1 = nnx.Conv(in_chans, 24, (3, 3), strides=(2, 2), use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(24, rngs=rngs)
        stages, chs = [], 24
        for out, n in zip(out_chs, repeats):
            blocks = []
            for j in range(n):
                blocks.append(ShuffleUnit(chs, out, 2 if j == 0 else 1, rngs=rngs))
                chs = out
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.conv5 = nnx.Conv(chs, 1024, (1, 1), use_bias=False, rngs=rngs)
        self.bn5 = nnx.BatchNorm(1024, rngs=rngs)
        self.num_features = 1024
        self.fc = nnx.Linear(1024, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = nnx.max_pool(nnx.relu(self.bn1(self.conv1(x))), (3, 3), strides=(2, 2), padding="SAME")
        for stage in self.stages:
            for blk in stage:
                x = blk(x)
        return nnx.relu(self.bn5(self.conv5(x)))

    def forward_head(self, x):
        x = global_pool_nhwc(x, self.global_pool)
        return self.fc(x) if self.fc is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

def _shufflenet(out_chs, **kwargs):
    model = ShuffleNetV2(out_chs, [4, 8, 4], **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def shufflenetv2_x0_5(**kwargs):
    return _shufflenet([48, 96, 192], **kwargs)

@register_model
def shufflenetv2_x1_0(**kwargs):
    return _shufflenet([116, 232, 464], **kwargs)

@register_model
def shufflenetv2_x1_5(**kwargs):
    return _shufflenet([176, 352, 704], **kwargs)

@register_model
def shufflenetv2_x2_0(**kwargs):
    return _shufflenet([244, 488, 976], **kwargs)
