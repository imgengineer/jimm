"""NHWC convolution helpers for the timm 1.0.30 mobile architectures."""

import jax
import jax.numpy as jnp
from flax import nnx

from ..layers import BatchNorm


class ConvNormAct(nnx.Module):
    """Use symmetric PyTorch padding, including for stride-two convolutions."""

    def __init__(
        self,
        in_chs,
        out_chs,
        kernel=1,
        stride=1,
        groups=1,
        norm=True,
        act=None,
        use_bias=False,
        bn_weight_init=1.0,
        ndim=2,
        *,
        rngs,
    ):
        self.conv = nnx.Conv(
            in_chs,
            out_chs,
            (kernel,) * ndim,
            strides=(stride,) * ndim,
            padding=((kernel // 2, kernel // 2),) * ndim,
            feature_group_count=groups,
            use_bias=use_bias,
            rngs=rngs,
        )
        self.norm = (
            BatchNorm(
                out_chs,
                epsilon=1e-5,
                momentum=0.9,
                scale_init=nnx.initializers.constant(bn_weight_init),
                rngs=rngs,
            )
            if norm
            else None
        )
        self.act = act

    def __call__(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        return self.act(x) if self.act is not None else x


class ConvTranspose(nnx.Module):
    """Grouped transposed convolution, with HWIO weights and PyTorch padding."""

    def __init__(self, in_chs, out_chs, kernel, stride, padding, groups=1, *, rngs):
        self.kernel = nnx.Param(
            nnx.initializers.lecun_normal()(
                rngs.params(), (kernel, kernel, in_chs // groups, out_chs)
            )
        )
        self.bias = nnx.Param(jnp.zeros(out_chs))
        self.stride = stride
        self.padding = kernel - 1 - padding
        self.groups = groups

    def __call__(self, x):
        x = jax.lax.conv_general_dilated(
            x,
            jnp.flip(self.kernel[...].astype(x.dtype), axis=(0, 1)),
            window_strides=(1, 1),
            padding=((self.padding, self.padding),) * 2,
            lhs_dilation=(self.stride, self.stride),
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=self.groups,
        )
        return x + self.bias[...].astype(x.dtype)
