"""NFNet in flax nnx, NHWC. Mirrors timm.models.nfnet (normalizer-free, ScWS convs)."""
import jax
import jax.numpy as jnp
from flax import nnx

from ..layers import SqueezeExcite, global_pool_nhwc
from ..registry import register_model, _cfg


class ScaledStdConv(nnx.Module):
    """Weight-standardized conv (NFNet): w' = (w - mean) / sqrt(var * fan_in), NHWC."""

    def __init__(self, in_chs, out_chs, kernel=3, stride=1, groups=1, *, rngs):
        self.stride, self.groups = stride, groups
        self.kernel = nnx.Param(nnx.initializers.lecun_normal()(
            rngs.params(), (kernel, kernel, in_chs // groups, out_chs)))
        self.bias = nnx.Param(jnp.zeros(out_chs))

    def __call__(self, x):
        w = self.kernel.value
        mean = jnp.mean(w, axis=(0, 1, 2), keepdims=True)
        var = jnp.var(w, axis=(0, 1, 2), keepdims=True)
        fan_in = w.shape[0] * w.shape[1] * w.shape[2]
        w = (w - mean) * jax.lax.rsqrt(var * fan_in + 1e-4)
        return jax.lax.conv_general_dilated(
            x, w, (self.stride, self.stride), "SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=self.groups) + self.bias.value


class NFBlock(nnx.Module):
    """NFNet bottleneck block: residual scaled by beta, activation gamma scaled."""

    def __init__(self, in_chs, out_chs, stride, expansion=2, se_ratio=0.5,
                 alpha=0.2, beta=1.0, *, rngs):
        mid = out_chs * expansion // 2  # nfnet bottleneck
        out = out_chs * expansion
        self.alpha, self.beta = alpha, beta
        self.conv1 = ScaledStdConv(in_chs, mid, 1, rngs=rngs)
        self.conv2 = ScaledStdConv(mid, mid, 3, stride, groups=mid, rngs=rngs)
        self.conv3 = ScaledStdConv(mid, out, 1, rngs=rngs)
        self.se = SqueezeExcite(mid, rd_ratio=se_ratio, rngs=rngs)
        self.do_pool = stride == 2
        self.short_conv = nnx.Conv(in_chs, out, (1, 1), use_bias=False, rngs=rngs) \
            if in_chs != out else None

    def __call__(self, x):
        y = nnx.gelu(self.conv1(x))
        y = nnx.gelu(self.conv2(y))
        y = self.se(y)
        y = nnx.gelu(self.conv3(y)) * self.alpha
        if self.do_pool:
            x = nnx.avg_pool(x, (2, 2), strides=(2, 2), padding="SAME")
        if self.short_conv is not None:
            x = self.short_conv(x)
        return (y + x) * self.beta


class NFNet(nnx.Module):
    default_cfg: dict = {}

    def __init__(self, channels=(256, 512, 1536, 1536), depths=(1, 2, 6, 3), alpha=0.2,
                 num_classes=1000, in_chans=3, global_pool="avg", drop_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.stem = nnx.List([ScaledStdConv(in_chans, 32, 3, 2, rngs=rngs),
                              ScaledStdConv(32, 64, 3, 1, rngs=rngs),
                              ScaledStdConv(64, 128, 3, 2, rngs=rngs),
                              ScaledStdConv(128, channels[0] // 2, 3, 1, rngs=rngs)])
        # beta schedule: 1.0 for first block of net, then residual-preserving
        stages, chs = [], channels[0] // 2
        for i, (c, d) in enumerate(zip(channels, depths)):
            blocks = []
            for j in range(d):
                beta = 1.0 if (i == 0 and j == 0) else 1.0  # simplified: beta=1
                blocks.append(NFBlock(chs, c, 2 if j == 0 and i > 0 else 1, alpha=alpha,
                                      beta=beta, rngs=rngs))
                chs = c * 2
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.num_features = channels[-1] * 2
        self.head_drop = nnx.Dropout(drop_rate)
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        for layer in self.stem:
            x = layer(x)
        for stage in self.stages:
            for blk in stage:
                x = blk(x)
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


_CFGS = {  # channels, depths, alpha
    "nfnet_f0": ((256, 512, 1536, 1536), (1, 2, 6, 3), 0.2),
    "nfnet_f1": ((256, 512, 1536, 1536), (2, 4, 12, 6), 0.2),
    "nfnet_f2": ((256, 512, 1536, 1536), (3, 6, 18, 9), 0.2),
    "nfnet_f3": ((256, 512, 1536, 1536), (4, 8, 24, 12), 0.2),
}


def _make(name):
    channels, depths, alpha = _CFGS[name]

    def entry(**kwargs):
        model = NFNet(channels, depths, alpha, **kwargs)
        model.default_cfg = _cfg()
        return model
    entry.__name__ = name
    return entry


for _name in _CFGS:
    register_model(_make(_name))
