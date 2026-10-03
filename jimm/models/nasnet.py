"""NASNet-A Large in flax nnx, NHWC. Mirrors timm.models.nasnet.

A VALID 3x3 stride-2 conv stem feeds two stem reduction cells, then three
groups of six cells (a first cell that adapts the previous-previous input
through two offset strided 1x1 paths, then five normal cells) separated by
reduction cells. Cells combine ReLU-separable-conv branches (each applied
twice with BatchNorm), 3x3 average and max pools and identities pairwise and
concatenate the results. As in timm, convolutions and pools use TensorFlow
'SAME' padding; stride-2 average pools count the zero padding while stride-1
average pools exclude it.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from ..layers import BatchNorm, ClassifierMixin, global_pool_nhwc
from ..registry import _cfg, register_model


def _bn(chs, *, rngs):
    return BatchNorm(chs, epsilon=1e-3, momentum=0.9, rngs=rngs)


def _conv(in_chs, out_chs, kernel=1, stride=1, groups=1, padding="SAME", *, rngs):
    return nnx.Conv(
        in_chs,
        out_chs,
        (kernel, kernel),
        strides=(stride, stride),
        padding=padding,
        feature_group_count=groups,
        use_bias=False,
        rngs=rngs,
    )


def _max_pool(x, stride):
    window = (1, 3, 3, 1)
    return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, window, (1, stride, stride, 1), "SAME")


def _avg_pool(x, stride):
    window, strides = (1, 3, 3, 1), (1, stride, stride, 1)
    total = jax.lax.reduce_window(x, 0.0, jax.lax.add, window, strides, "SAME")
    if stride > 1:  # AvgPool2dSame pads zeros explicitly, so they count
        return total / 9.0
    ones = jnp.ones((1, *x.shape[1:3], 1), x.dtype)
    return total / jax.lax.reduce_window(ones, 0.0, jax.lax.add, window, strides, "SAME")


class ConvBn(nnx.Module):
    def __init__(self, in_chs, out_chs, *, rngs):
        self.conv = _conv(in_chs, out_chs, 3, 2, padding="VALID", rngs=rngs)
        self.bn = _bn(out_chs, rngs=rngs)

    def __call__(self, x):
        return self.bn(self.conv(x))


class ActConvBn(nnx.Module):
    def __init__(self, in_chs, out_chs, *, rngs):
        self.conv = _conv(in_chs, out_chs, rngs=rngs)
        self.bn = _bn(out_chs, rngs=rngs)

    def __call__(self, x):
        return self.bn(self.conv(nnx.relu(x)))


class SeparableConv2d(nnx.Module):
    def __init__(self, in_chs, out_chs, kernel, stride, *, rngs):
        self.depthwise_conv2d = _conv(in_chs, in_chs, kernel, stride, in_chs, rngs=rngs)
        self.pointwise_conv2d = _conv(in_chs, out_chs, rngs=rngs)

    def __call__(self, x):
        return self.pointwise_conv2d(self.depthwise_conv2d(x))


class BranchSeparables(nnx.Module):
    def __init__(self, in_chs, out_chs, kernel, stride=1, stem_cell=False, *, rngs):
        mid = out_chs if stem_cell else in_chs
        self.separable_1 = SeparableConv2d(in_chs, mid, kernel, stride, rngs=rngs)
        self.bn_sep_1 = _bn(mid, rngs=rngs)
        self.separable_2 = SeparableConv2d(mid, out_chs, kernel, 1, rngs=rngs)
        self.bn_sep_2 = _bn(out_chs, rngs=rngs)

    def __call__(self, x):
        x = self.bn_sep_1(self.separable_1(nnx.relu(x)))
        return self.bn_sep_2(self.separable_2(nnx.relu(x)))


class _Path(nnx.Module):
    """Stride-2 subsampling (optionally offset by one pixel) and a 1x1 conv."""

    def __init__(self, in_chs, out_chs, shift, *, rngs):
        self.shift = shift
        self.conv = _conv(in_chs, out_chs, rngs=rngs)

    def __call__(self, x):
        if self.shift:  # ZeroPad2d((-1, 1, -1, 1)): drop the first row/col, zero-pad the end
            x = jnp.pad(x[:, 1:, 1:], ((0, 0), (0, 1), (0, 1), (0, 0)))
        return self.conv(x[:, ::2, ::2])


class _AdjustPaths(nnx.Module):
    def __init__(self, in_chs, out_chs, *, rngs):
        self.path_1 = _Path(in_chs, out_chs, False, rngs=rngs)
        self.path_2 = _Path(in_chs, out_chs, True, rngs=rngs)
        self.final_path_bn = _bn(2 * out_chs, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(x)
        return self.final_path_bn(jnp.concatenate([self.path_1(x), self.path_2(x)], axis=-1))


class _ReductionBranches(nnx.Module):
    """The shared five-combination reduction pattern of the stem and reduction cells."""

    def _init_branches(self, left_chs, right_chs, chs, stem_cell, *, rngs):
        self.comb_iter_0_left = BranchSeparables(left_chs, chs, 5, 2, rngs=rngs)
        self.comb_iter_0_right = BranchSeparables(right_chs, chs, 7, 2, stem_cell, rngs=rngs)
        self.comb_iter_1_right = BranchSeparables(right_chs, chs, 7, 2, stem_cell, rngs=rngs)
        self.comb_iter_2_right = BranchSeparables(right_chs, chs, 5, 2, stem_cell, rngs=rngs)
        self.comb_iter_4_left = BranchSeparables(chs, chs, 3, 1, rngs=rngs)

    def _combine(self, left, right):
        c0 = self.comb_iter_0_left(left) + self.comb_iter_0_right(right)
        c1 = _max_pool(left, 2) + self.comb_iter_1_right(right)
        c2 = _avg_pool(left, 2) + self.comb_iter_2_right(right)
        c3 = _avg_pool(c0, 1) + c1
        c4 = self.comb_iter_4_left(c0) + _max_pool(left, 2)
        return jnp.concatenate([c1, c2, c3, c4], axis=-1)


class CellStem0(_ReductionBranches):
    def __init__(self, stem_size, num_channels, *, rngs):
        self.conv_1x1 = ActConvBn(stem_size, num_channels, rngs=rngs)
        self._init_branches(num_channels, stem_size, num_channels, True, rngs=rngs)

    def __call__(self, x):
        return self._combine(self.conv_1x1(x), x)


class CellStem1(_ReductionBranches):
    def __init__(self, stem_size, num_channels, *, rngs):
        self.conv_1x1 = ActConvBn(2 * num_channels, num_channels, rngs=rngs)
        self.paths = _AdjustPaths(stem_size, num_channels // 2, rngs=rngs)
        self._init_branches(num_channels, num_channels, num_channels, False, rngs=rngs)

    def __call__(self, x_conv0, x_stem_0):
        return self._combine(self.conv_1x1(x_stem_0), self.paths(x_conv0))


class ReductionCell(_ReductionBranches):
    def __init__(self, in_left, out_left, in_right, out_right, *, rngs):
        self.conv_prev_1x1 = ActConvBn(in_left, out_left, rngs=rngs)
        self.conv_1x1 = ActConvBn(in_right, out_right, rngs=rngs)
        self._init_branches(out_right, out_right, out_right, False, rngs=rngs)

    def __call__(self, x, x_prev):
        return self._combine(self.conv_1x1(x), self.conv_prev_1x1(x_prev))


class NormalCell(nnx.Module):
    def __init__(self, in_left, out_left, in_right, out_right, first=False, *, rngs):
        if first:  # FirstCell: the previous-previous input is at twice the resolution
            self.paths = _AdjustPaths(in_left, out_left, rngs=rngs)
            self.conv_prev_1x1 = None
            out_left *= 2
        else:
            self.paths = None
            self.conv_prev_1x1 = ActConvBn(in_left, out_left, rngs=rngs)
        self.conv_1x1 = ActConvBn(in_right, out_right, rngs=rngs)
        self.comb_iter_0_left = BranchSeparables(out_right, out_right, 5, rngs=rngs)
        self.comb_iter_0_right = BranchSeparables(out_left, out_left, 3, rngs=rngs)
        self.comb_iter_1_left = BranchSeparables(out_left, out_left, 5, rngs=rngs)
        self.comb_iter_1_right = BranchSeparables(out_left, out_left, 3, rngs=rngs)
        self.comb_iter_4_left = BranchSeparables(out_right, out_right, 3, rngs=rngs)

    def __call__(self, x, x_prev):
        left = self.paths(x_prev) if self.paths is not None else self.conv_prev_1x1(x_prev)
        right = self.conv_1x1(x)
        c0 = self.comb_iter_0_left(right) + self.comb_iter_0_right(left)
        c1 = self.comb_iter_1_left(left) + self.comb_iter_1_right(left)
        c2 = _avg_pool(right, 1) + left
        c3 = 2 * _avg_pool(left, 1)
        c4 = self.comb_iter_4_left(right) + right
        return jnp.concatenate([left, c0, c1, c2, c3, c4], axis=-1)


class NASNetALarge(ClassifierMixin, nnx.Module):
    _classifier_attr = "last_linear"

    def __init__(
        self,
        stem_size=96,
        channel_multiplier=2,
        num_features=4032,
        num_classes=1000,
        in_chans=3,
        global_pool="avg",
        drop_rate=0.0,
        *,
        rngs,
    ):
        self.num_classes, self.global_pool = num_classes, global_pool
        self.num_features = num_features
        c = num_features // 24
        self.conv0 = ConvBn(in_chans, stem_size, rngs=rngs)
        self.cell_stem_0 = CellStem0(stem_size, c // channel_multiplier**2, rngs=rngs)
        self.cell_stem_1 = CellStem1(stem_size, c // channel_multiplier, rngs=rngs)
        cells = []
        for g, m in enumerate((1, 2, 4)):
            prev_left = (c, 6 * c, 12 * c)[g]
            first_right = (2 * c, 8 * c, 16 * c)[g]
            cells.append(NormalCell(prev_left, m * c // 2, first_right, m * c, True, rngs=rngs))
            cells.append(NormalCell(first_right, m * c, 6 * m * c, m * c, rngs=rngs))
            cells += [NormalCell(6 * m * c, m * c, 6 * m * c, m * c, rngs=rngs) for _ in range(4)]
            if g < 2:
                cells.append(ReductionCell(6 * m * c, 2 * m * c, 6 * m * c, 2 * m * c, rngs=rngs))
        self.cells = nnx.List(cells)
        self.head_drop = nnx.Dropout(drop_rate, rngs=rngs)
        self.last_linear = self._make_head(num_classes, rngs)

    def _make_head(self, num_classes, rngs):
        return nnx.Linear(self.num_features, num_classes, rngs=rngs) if num_classes > 0 else None

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.global_pool = global_pool if global_pool is not None else self.global_pool
        self.last_linear = self._make_head(num_classes, nnx.Rngs(0))

    def forward_features(self, x):
        x_conv0 = self.conv0(x)
        x_stem_0 = self.cell_stem_0(x_conv0)
        prev, x = x_stem_0, self.cell_stem_1(x_conv0, x_stem_0)
        for cell in self.cells:
            out = cell(x, prev)
            # The cell after a reduction reads the reduction's own previous input.
            prev, x = (prev, out) if isinstance(cell, ReductionCell) else (x, out)
        return nnx.relu(x)

    def forward_head(self, x):
        x = self.head_drop(global_pool_nhwc(x, self.global_pool))
        return self.last_linear(x) if self.last_linear is not None else x

    def __call__(self, x):
        return self.forward_head(self.forward_features(x))


@register_model
def nasnetalarge(**kwargs):
    model = NASNetALarge(**kwargs)
    model.default_cfg = _cfg(
        input_size=(3, 331, 331),
        crop_pct=0.911,
        interpolation="bicubic",
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )
    return model
