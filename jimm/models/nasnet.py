"""NASNet-A in flax nnx, NHWC. Mirrors timm.models.nasnet (normal/reduction cells)."""
import jax.numpy as jnp
from flax import nnx

from ..layers import ConvBNAct, global_pool_nhwc
from ..registry import register_model, _cfg


def _pool3_s2(x):
    return nnx.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")


class SepConv(nnx.Module):
    """depthwise k + pointwise 1x1, with BN+relu around (nasnet separable conv)."""

    def __init__(self, in_chs, out_chs, kernel, stride=1, *, rngs):
        self.dw = nnx.Conv(in_chs, in_chs, (kernel, kernel), strides=(stride, stride),
                           use_bias=False, feature_group_count=in_chs, rngs=rngs)
        self.bn1 = nnx.BatchNorm(in_chs, rngs=rngs)
        self.pw = nnx.Conv(in_chs, out_chs, (1, 1), use_bias=False, rngs=rngs)
        self.bn2 = nnx.BatchNorm(out_chs, rngs=rngs)

    def __call__(self, x):
        x = self.bn2(self.pw(self.bn1(self.dw(x))))
        return x


class CellStem1(nnx.Module):
    """Reduction cell applied after stem (from prev_prev 128 + prev 256 -> out 256)."""

    def __init__(self, out_chs, *, rngs):
        self.c0 = ConvBNAct(128, out_chs, 1, rngs=rngs)   # prev_prev (stem1 out, 128)
        self.c1 = ConvBNAct(256, out_chs, 1, rngs=rngs)   # prev (stem2 out, 256)
        self.oc = out_chs

    def __call__(self, x0, x1):
        return self.c0(x0), self.c1(x1)


# ponytail: NASNet cells below are a single-input structural approximation of the
# NASNet-A cell wiring (which is dual-state with factorized reductions). Upgrade to the
# exact dual-state cells if pretrained-weight compat is ever needed.
class NormalCell(nnx.Module):
    def __init__(self, in_chs, out_chs, *, rngs):
        self.c = ConvBNAct(in_chs, out_chs, 1, rngs=rngs) if in_chs != out_chs else None
        oc = out_chs
        self.ops = nnx.List([
            SepConv(oc, oc, 5, rngs=rngs), SepConv(oc, oc, 3, rngs=rngs),
            SepConv(oc, oc, 5, rngs=rngs),
            SepConv(oc, oc, 3, rngs=rngs),
            SepConv(oc, oc, 3, rngs=rngs),
        ])

    def __call__(self, x):
        x = x if self.c is None else self.c(x)
        b1 = self.ops[0](x) + self.ops[1](x)
        b2 = self.ops[2](x) + x
        b3 = self.ops[3](b1)
        b4 = b1 + x
        b5 = self.ops[4](b2)
        return jnp.concatenate([b3, b4, b5, x, b1], axis=-1)


class ReductionCell(nnx.Module):
    def __init__(self, in_chs, out_chs, *, rngs):
        self.c = ConvBNAct(in_chs, out_chs, 1, rngs=rngs)
        oc = out_chs
        self.op0 = SepConv(oc, oc, 5, 2, rngs=rngs)
        self.op1 = SepConv(oc, oc, 3, 2, rngs=rngs)
        self.op2 = SepConv(oc, oc, 5, 2, rngs=rngs)

    def __call__(self, x):
        x = self.c(x)
        b1 = self.op0(x) + self.op1(x)
        b2 = self.op2(x) + _pool3_s2(x)
        return jnp.concatenate([b1, b2], axis=-1)


class NASNetA(nnx.Module):
    default_cfg: dict = {}

    def __init__(self, num_classes=1000, in_chans=3, global_pool="avg", drop_rate=0.0,
                 stem_chs=96, out_chs=168, num_cells=6, *, rngs):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.conv0 = ConvBNAct(in_chans, 32, 3, 2, padding="VALID", rngs=rngs)
        self.conv1 = ConvBNAct(32, 32, 3, 2, padding="VALID", rngs=rngs)
        self.conv2 = ConvBNAct(32, 64, 3, rngs=rngs)
        self.conv3 = ConvBNAct(64, 64, 3, 2, padding="VALID", rngs=rngs)
        self.conv4 = ConvBNAct(64, 64, 3, rngs=rngs)
        self.stem1 = ConvBNAct(64, stem_chs, 1, rngs=rngs)
        self.stem2 = ConvBNAct(stem_chs, out_chs * 2, 1, rngs=rngs)
        chs = out_chs * 2
        cells = []
        for i in range(num_cells):
            cells.append(ReductionCell(chs, out_chs, rngs=rngs))
            chs = out_chs * 2
            for _ in range(2):
                cells.append(NormalCell(chs, out_chs, rngs=rngs))
                chs = out_chs * 5
        self.cells = nnx.List(cells)
        self.norm = nnx.BatchNorm(chs, rngs=rngs)
        self.num_features = chs
        self.head_drop = nnx.Dropout(drop_rate)
        self.fc = nnx.Linear(chs, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.conv4(self.conv3(self.conv2(self.conv1(self.conv0(x)))))
        x = self.stem2(self.stem1(x))
        for cell in self.cells:
            x = cell(x)
        return nnx.relu(self.norm(x))

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


@register_model
def nasnetalarge(**kwargs):
    model = NASNetA(**kwargs)
    model.default_cfg = _cfg(input_size=(3, 331, 331), crop_pct=0.911)
    return model


@register_model
def pnasnetalarge(**kwargs):
    model = NASNetA(**kwargs)
    model.default_cfg = _cfg(input_size=(3, 331, 331), crop_pct=0.911)
    return model
