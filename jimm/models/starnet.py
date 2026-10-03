"""StarNet in flax nnx, NHWC. Mirrors timm.models.starnet ("Rewrite the Stars").

Blocks: depthwise 7x7 conv + BN, two 1x1 expansions multiplied element-wise
("star" operation, ReLU6 on one branch), a 1x1 projection + BN, and a second
depthwise 7x7 conv. Each stage starts with a strided 3x3 conv + BN; the
features end with BatchNorm before pooling.
"""

from flax import nnx

from ..layers import BatchNorm, ClassifierMixin, DropPath
from ..registry import _cfg, register_model

_init = nnx.initializers.truncated_normal(0.02)


class ConvBN(nnx.Module):
    """timm ConvBN: a biased convolution, optionally followed by BatchNorm."""

    def __init__(self, in_chs, out_chs, kernel=1, stride=1, groups=1, with_bn=True, *, rngs):
        pad = kernel // 2
        self.conv = nnx.Conv(
            in_chs,
            out_chs,
            (kernel, kernel),
            strides=(stride, stride),
            padding=((pad, pad), (pad, pad)),
            feature_group_count=groups,
            kernel_init=_init,
            rngs=rngs,
        )
        self.bn = BatchNorm(out_chs, epsilon=1e-5, rngs=rngs) if with_bn else None

    def __call__(self, x):
        x = self.conv(x)
        return x if self.bn is None else self.bn(x)


class Block(nnx.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.0, *, rngs):
        self.dwconv = ConvBN(dim, dim, 7, groups=dim, rngs=rngs)
        self.f1 = ConvBN(dim, mlp_ratio * dim, with_bn=False, rngs=rngs)
        self.f2 = ConvBN(dim, mlp_ratio * dim, with_bn=False, rngs=rngs)
        self.g = ConvBN(mlp_ratio * dim, dim, rngs=rngs)
        self.dwconv2 = ConvBN(dim, dim, 7, groups=dim, with_bn=False, rngs=rngs)
        self.drop_path = DropPath(drop_path, rngs=rngs)

    def __call__(self, x):
        y = self.dwconv(x)
        y = nnx.relu6(self.f1(y)) * self.f2(y)
        return x + self.drop_path(self.dwconv2(self.g(y)))


class StarNet(ClassifierMixin, nnx.Module):
    _classifier_attr = "head"

    def __init__(
        self,
        base_dim=32,
        depths=(3, 3, 12, 5),
        mlp_ratio=4,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.stem = ConvBN(in_chans, 32, 3, 2, rngs=rngs)
        total = sum(depths)
        dpr = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]
        stages, prev, k = [], 32, 0
        for i, depth in enumerate(depths):
            dim = base_dim * 2**i
            layers = [ConvBN(prev, dim, 3, 2, rngs=rngs)]
            layers += [Block(dim, mlp_ratio, dpr[k + j], rngs=rngs) for j in range(depth)]
            stages.append(nnx.List(layers))
            prev, k = dim, k + depth
        self.stages = nnx.List(stages)
        self.num_features = prev
        self.norm = BatchNorm(prev, epsilon=1e-5, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.head = (
            nnx.Linear(prev, num_classes, kernel_init=_init, rngs=rngs) if num_classes > 0 else None
        )

    def forward_features(self, x):
        x = nnx.relu6(self.stem(x))
        for stage in self.stages:
            for layer in stage:
                x = layer(x)
        return self.norm(x)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


_CFGS = {  # base_dim, depths, mlp_ratio
    "starnet_s050": (16, (1, 1, 3, 1), 3),
    "starnet_s100": (20, (1, 2, 4, 1), 4),
    "starnet_s150": (24, (1, 2, 4, 2), 3),
    "starnet_s1": (24, (2, 2, 8, 3), 4),
    "starnet_s2": (32, (1, 2, 6, 2), 4),
    "starnet_s3": (32, (2, 2, 8, 4), 4),
    "starnet_s4": (32, (3, 3, 12, 5), 4),
}


def _make(name):
    base_dim, depths, mlp_ratio = _CFGS[name]

    def entry(**kwargs):
        model = StarNet(base_dim, depths, mlp_ratio, **kwargs)
        model.default_cfg = _cfg(interpolation="bicubic")
        return model

    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
