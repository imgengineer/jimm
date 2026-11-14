"""PNASNet-5-Large in flax nnx, NHWC. Mirrors timm.models.pnasnet.

The cell topology follows Cadene's implementation of the Progressive Neural
Architecture Search network, with TensorFlow ``same`` padding as in timm's
``pnasnet5large``.
"""

import jax.numpy as jnp
from flax import nnx

from ..layers import BatchNorm, ClassifierMixin
from ..registry import _cfg, register_model


def _padding(pad_type):
    return "SAME" if pad_type == "same" else "VALID"


class MaxPool(nnx.Module):
    def __init__(self, stride, pad_type):
        self.stride = stride
        self.pad_type = pad_type

    def __call__(self, x):
        return nnx.max_pool(
            x, (3, 3), strides=(self.stride, self.stride), padding=_padding(self.pad_type)
        )


class SeparableConv2d(nnx.Module):
    def __init__(self, in_chs, out_chs, kernel, stride, pad_type, *, rngs):
        self.depthwise_conv2d = nnx.Conv(
            in_chs,
            in_chs,
            (kernel, kernel),
            strides=(stride, stride),
            padding=_padding(pad_type),
            feature_group_count=in_chs,
            use_bias=False,
            rngs=rngs,
        )
        self.pointwise_conv2d = nnx.Conv(in_chs, out_chs, (1, 1), use_bias=False, rngs=rngs)

    def __call__(self, x):
        return self.pointwise_conv2d(self.depthwise_conv2d(x))


class BranchSeparables(nnx.Module):
    def __init__(self, in_chs, out_chs, kernel, stride=1, stem_cell=False, pad_type="", *, rngs):
        mid = out_chs if stem_cell else in_chs
        self.separable_1 = SeparableConv2d(in_chs, mid, kernel, stride, pad_type, rngs=rngs)
        self.bn_sep_1 = BatchNorm(mid, epsilon=1e-3, rngs=rngs)
        self.separable_2 = SeparableConv2d(mid, out_chs, kernel, 1, pad_type, rngs=rngs)
        self.bn_sep_2 = BatchNorm(out_chs, epsilon=1e-3, rngs=rngs)

    def __call__(self, x):
        x = self.bn_sep_1(self.separable_1(nnx.relu(x)))
        return self.bn_sep_2(self.separable_2(nnx.relu(x)))


class ActConvBn(nnx.Module):
    def __init__(self, in_chs, out_chs, kernel, stride=1, pad_type="", *, rngs):
        self.conv = nnx.Conv(
            in_chs,
            out_chs,
            (kernel, kernel),
            strides=(stride, stride),
            padding=_padding(pad_type),
            use_bias=False,
            rngs=rngs,
        )
        self.bn = BatchNorm(out_chs, epsilon=1e-3, rngs=rngs)

    def __call__(self, x):
        return self.bn(self.conv(nnx.relu(x)))


class PoolConvBn(nnx.Module):
    def __init__(self, in_chs, out_chs, stride, pad_type, *, rngs):
        self.pool_stride = stride
        self.pad_type = pad_type
        self.conv = nnx.Conv(in_chs, out_chs, (1, 1), use_bias=False, rngs=rngs)
        self.bn = BatchNorm(out_chs, epsilon=1e-3, rngs=rngs)

    def __call__(self, x):
        x = nnx.max_pool(
            x, (3, 3), strides=(self.pool_stride, self.pool_stride), padding=_padding(self.pad_type)
        )
        return self.bn(self.conv(x))


class FactorizedReduction(nnx.Module):
    """Factorized spatial reduction: even pixels and shifted odd pixels, each
    projected by a 1x1 conv and concatenated."""

    def __init__(self, in_chs, out_chs, *, rngs):
        self.path1_conv = nnx.Conv(in_chs, out_chs // 2, (1, 1), use_bias=False, rngs=rngs)
        self.path2_conv = nnx.Conv(in_chs, out_chs // 2, (1, 1), use_bias=False, rngs=rngs)
        self.final_path_bn = BatchNorm(out_chs, epsilon=1e-3, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(x)
        x_path1 = self.path1_conv(x[:, ::2, ::2])
        # ZeroPad2d((-1, 1, -1, 1)) drops the first row/column and appends a zero
        # one, so this samples the odd pixels and keeps ceil(size / 2) outputs.
        shifted = x[:, 1:, 1:]
        pad_h, pad_w = x.shape[1] % 2, x.shape[2] % 2
        shifted = jnp.pad(shifted, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)))
        x_path2 = self.path2_conv(shifted[:, ::2, ::2])
        return self.final_path_bn(jnp.concatenate([x_path1, x_path2], axis=-1))


class CellBase(nnx.Module):
    def cell_forward(self, x_left, x_right):
        u0 = self.comb_iter_0_left(x_left) + self.comb_iter_0_right(x_left)
        u1 = self.comb_iter_1_left(x_right) + self.comb_iter_1_right(x_right)
        u2 = self.comb_iter_2_left(x_right) + self.comb_iter_2_right(x_right)
        u3 = self.comb_iter_3_left(u2) + self.comb_iter_3_right(x_right)
        u4_right = x_right if self.comb_iter_4_right is None else self.comb_iter_4_right(x_right)
        u4 = self.comb_iter_4_left(x_left) + u4_right
        return jnp.concatenate([u0, u1, u2, u3, u4], axis=-1)


class CellStem0(CellBase):
    def __init__(self, in_left, out_left, in_right, out_right, pad_type="", *, rngs):
        k = dict(pad_type=pad_type, rngs=rngs)
        self.conv_1x1 = ActConvBn(in_right, out_right, 1, **k)
        self.comb_iter_0_left = BranchSeparables(
            in_left, out_left, 5, stride=2, stem_cell=True, **k
        )
        self.comb_iter_0_right = PoolConvBn(in_left, out_left, 2, pad_type, rngs=rngs)
        self.comb_iter_1_left = BranchSeparables(out_right, out_right, 7, stride=2, **k)
        self.comb_iter_1_right = MaxPool(2, pad_type)
        self.comb_iter_2_left = BranchSeparables(out_right, out_right, 5, stride=2, **k)
        self.comb_iter_2_right = BranchSeparables(out_right, out_right, 3, stride=2, **k)
        self.comb_iter_3_left = BranchSeparables(out_right, out_right, 3, **k)
        self.comb_iter_3_right = MaxPool(2, pad_type)
        self.comb_iter_4_left = BranchSeparables(
            in_right, out_right, 3, stride=2, stem_cell=True, **k
        )
        self.comb_iter_4_right = ActConvBn(out_right, out_right, 1, stride=2, **k)

    def __call__(self, x_left):
        x_right = self.conv_1x1(x_left)
        return self.cell_forward(x_left, x_right)


class Cell(CellBase):
    def __init__(
        self,
        in_left,
        out_left,
        in_right,
        out_right,
        pad_type="",
        is_reduction=False,
        match_prev_layer_dims=False,
        *,
        rngs,
    ):
        stride = 2 if is_reduction else 1
        k = dict(pad_type=pad_type, rngs=rngs)
        self.conv_prev_1x1 = (
            FactorizedReduction(in_left, out_left, rngs=rngs)
            if match_prev_layer_dims
            else ActConvBn(in_left, out_left, 1, **k)
        )
        self.conv_1x1 = ActConvBn(in_right, out_right, 1, **k)
        self.comb_iter_0_left = BranchSeparables(out_left, out_left, 5, stride=stride, **k)
        self.comb_iter_0_right = MaxPool(stride, pad_type)
        self.comb_iter_1_left = BranchSeparables(out_right, out_right, 7, stride=stride, **k)
        self.comb_iter_1_right = MaxPool(stride, pad_type)
        self.comb_iter_2_left = BranchSeparables(out_right, out_right, 5, stride=stride, **k)
        self.comb_iter_2_right = BranchSeparables(out_right, out_right, 3, stride=stride, **k)
        self.comb_iter_3_left = BranchSeparables(out_right, out_right, 3, **k)
        self.comb_iter_3_right = MaxPool(stride, pad_type)
        self.comb_iter_4_left = BranchSeparables(out_left, out_left, 3, stride=stride, **k)
        self.comb_iter_4_right = (
            ActConvBn(out_right, out_right, 1, stride=stride, **k) if is_reduction else None
        )

    def __call__(self, x_left, x_right):
        x_left = self.conv_prev_1x1(x_left)
        x_right = self.conv_1x1(x_right)
        return self.cell_forward(x_left, x_right)


class PNASNet5Large(ClassifierMixin, nnx.Module):
    def __init__(
        self,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        pad_type="",
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = 4320
        self.conv_0 = nnx.Conv(
            in_chans, 96, (3, 3), strides=(2, 2), padding="VALID", use_bias=False, rngs=rngs
        )
        self.bn_0 = BatchNorm(96, epsilon=1e-3, rngs=rngs)
        k = dict(pad_type=pad_type, rngs=rngs)
        self.cell_stem_0 = CellStem0(96, 54, 96, 54, **k)
        self.cell_stem_1 = Cell(
            96, 108, 270, 108, is_reduction=True, match_prev_layer_dims=True, **k
        )
        self.cell_0 = Cell(270, 216, 540, 216, match_prev_layer_dims=True, **k)
        self.cell_1 = Cell(540, 216, 1080, 216, **k)
        self.cell_2 = Cell(1080, 216, 1080, 216, **k)
        self.cell_3 = Cell(1080, 216, 1080, 216, **k)
        self.cell_4 = Cell(1080, 432, 1080, 432, is_reduction=True, **k)
        self.cell_5 = Cell(1080, 432, 2160, 432, match_prev_layer_dims=True, **k)
        self.cell_6 = Cell(2160, 432, 2160, 432, **k)
        self.cell_7 = Cell(2160, 432, 2160, 432, **k)
        self.cell_8 = Cell(2160, 864, 2160, 864, is_reduction=True, **k)
        self.cell_9 = Cell(2160, 864, 4320, 864, match_prev_layer_dims=True, **k)
        self.cell_10 = Cell(4320, 864, 4320, 864, **k)
        self.cell_11 = Cell(4320, 864, 4320, 864, **k)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.fc = nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def forward_features(self, x):
        x_conv_0 = self.bn_0(self.conv_0(x))
        x_stem_0 = self.cell_stem_0(x_conv_0)
        x_stem_1 = self.cell_stem_1(x_conv_0, x_stem_0)
        x_cell_0 = self.cell_0(x_stem_0, x_stem_1)
        x_cell_1 = self.cell_1(x_stem_1, x_cell_0)
        x_cell_2 = self.cell_2(x_cell_0, x_cell_1)
        x_cell_3 = self.cell_3(x_cell_1, x_cell_2)
        x_cell_4 = self.cell_4(x_cell_2, x_cell_3)
        x_cell_5 = self.cell_5(x_cell_3, x_cell_4)
        x_cell_6 = self.cell_6(x_cell_4, x_cell_5)
        x_cell_7 = self.cell_7(x_cell_5, x_cell_6)
        x_cell_8 = self.cell_8(x_cell_6, x_cell_7)
        x_cell_9 = self.cell_9(x_cell_7, x_cell_8)
        x_cell_10 = self.cell_10(x_cell_8, x_cell_9)
        x_cell_11 = self.cell_11(x_cell_9, x_cell_10)
        return nnx.relu(x_cell_11)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


def _pnasnet5large(**kwargs):
    model = PNASNet5Large(pad_type="same", **kwargs)
    model.default_cfg = _cfg(
        input_size=(3, 331, 331),
        crop_pct=0.911,
        interpolation="bicubic",
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )
    return model


@register_model
def pnasnet5large(**kwargs):
    return _pnasnet5large(**kwargs)
