"""DPN (Dual Path Network) in flax nnx, NHWC. Mirrors timm.models.dpn exactly.

Block: c1x1_a (in->r) -> c3x3_b (r->r, groups) -> c1x1_c (r->bw+inc).
Shortcut path: 1x1 proj (in->bw+2*inc) split into residual (bw) + dense-seed (2*inc);
output = residual (bw ch) concat dense (2*inc + inc per block, accumulating).
"""
import jax.numpy as jnp
from flax import nnx

from ..layers import ConvBNAct, global_pool_nhwc, ClassifierMixin
from ..registry import register_model, _cfg

class DualPathBlock(nnx.Module):
    def __init__(self, in_chs, r, bw, inc, groups, key_stride, has_proj, b, *, rngs):
        self.bw, self.inc = bw, inc
        self.has_proj = has_proj
        self.c1x1_w = ConvBNAct(in_chs, bw + 2 * inc, 1, key_stride, rngs=rngs) if has_proj else None
        self.c1x1_a = ConvBNAct(in_chs, r, 1, rngs=rngs)
        self.c3x3_b = ConvBNAct(r, r, 3, key_stride, groups=groups, rngs=rngs)
        if b:
            self.c1 = nnx.Conv(r, bw, (1, 1), use_bias=False, rngs=rngs)
            self.c2 = nnx.Conv(r, inc, (1, 1), use_bias=False, rngs=rngs)
            self.c_bn = nnx.BatchNorm(r, rngs=rngs)
            self.c1x1_c = None
        else:
            self.c1x1_c = ConvBNAct(r, bw + inc, 1, act="identity", rngs=rngs)
            self.c1 = self.c2 = self.c_bn = None

    def __call__(self, x):
        if self.c1x1_w is not None:
            x_s = self.c1x1_w(x)
            x_s1, x_s2 = x_s[..., :self.bw], x_s[..., self.bw:]
        else:
            x_s1, x_s2 = x[..., :self.bw], x[..., self.bw:]
        y = self.c3x3_b(self.c1x1_a(x))
        if self.c1x1_c is not None:
            y = self.c1x1_c(y)
            out1, out2 = y[..., :self.bw], y[..., self.bw:]
        else:
            assert self.c_bn is not None and self.c1 is not None and self.c2 is not None
            y = nnx.relu(self.c_bn(y))
            out1, out2 = self.c1(y), self.c2(y)
        resid = x_s1 + out1
        dense = jnp.concatenate([x_s2, out2], axis=-1)
        return jnp.concatenate([resid, dense], axis=-1)

class DPN(ClassifierMixin, nnx.Module):
    default_cfg: dict = {}

    def __init__(self, k_sec, inc_sec, k_r, groups, small=False, num_init_features=64,
                 b=False, num_classes=1000, in_chans=3, global_pool="avg", drop_rate=0.0, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        bw_factor = 1 if small else 4
        self.conv1_1 = ConvBNAct(in_chans, num_init_features, 3 if small else 7, 2, rngs=rngs)
        stages, in_chs = [], num_init_features
        for i, k in enumerate(k_sec):
            bw = 64 * bw_factor * 2**i
            inc = inc_sec[i]
            r = (k_r * bw) // (64 * bw_factor)
            blocks = [DualPathBlock(in_chs, r, bw, inc, groups, 1 if i == 0 else 2, True, b, rngs=rngs)]
            in_chs = bw + 3 * inc
            for _ in range(2, k + 1):
                blocks.append(DualPathBlock(in_chs, r, bw, inc, groups, 1, False, b, rngs=rngs))
                in_chs += inc
            stages.append(nnx.List(blocks))
        self.stages = nnx.List(stages)
        self.num_features = in_chs
        self.head_norm = nnx.BatchNorm(self.num_features, rngs=rngs)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.conv1_1(x)
        x = nnx.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        for stage in self.stages:
            for blk in stage:
                x = blk(x)
        return x

    def forward_head(self, x):
        x = global_pool_nhwc(x, self.global_pool)
        x = nnx.elu(self.head_norm(x))
        x = self.head_drop(x)
        return self.fc(x) if self.fc is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))

_CFGS = {  # k_sec, inc_sec, k_r, groups, small, num_init_features, b
    "dpn68": ((3, 4, 12, 3), (16, 32, 32, 64), 128, 32, True, 10, False),
    "dpn68b": ((3, 4, 12, 3), (16, 32, 32, 64), 128, 32, True, 10, True),
    "dpn92": ((3, 4, 20, 3), (16, 32, 24, 128), 96, 32, False, 64, False),
    "dpn98": ((3, 6, 20, 3), (16, 32, 32, 128), 160, 40, False, 96, False),
    "dpn107": ((4, 8, 20, 3), (20, 64, 64, 128), 200, 50, False, 128, False),
    "dpn131": ((4, 8, 28, 3), (16, 32, 32, 128), 160, 40, False, 128, False),
}

def _make(name):
    k_sec, inc_sec, k_r, groups, small, nif, b = _CFGS[name]

    def entry(**kwargs):
        model = DPN(k_sec, inc_sec, k_r, groups, small, nif, b, **kwargs)
        model.default_cfg = _cfg(input_size=(3, 224, 224))
        return model
    entry.__name__ = name
    return entry

for _name in _CFGS:
    register_model(_make(_name))
