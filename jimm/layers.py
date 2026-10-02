"""Shared reusable neural network layers and mixins for JAX/Flax NNX models.

Layout Convention:
  - All convolution and spatial feature maps follow the NHWC convention:
    (Batch, Height, Width, Channels).
  - Dense token sequences follow BNC: (Batch, Num_Tokens, Channels).
  - DropPath and BatchNorm modes automatically switch via `model.train()` and `model.eval()`.
"""

from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

__all__ = [
    "BatchNorm",
    "DropPath",
    "drop_path",
    "create_act_layer",
    "conv_general_dilated",
    "use_fast_grouped_conv_grads",
    "PatchEmbed",
    "Mlp",
    "SqueezeExcite",
    "make_divisible",
    "ConvBNAct",
    "ClassifierMixin",
    "global_pool_nhwc",
    "gelu",
    "hswish",
    "relu6",
]


def gelu(x: jax.Array) -> jax.Array:
    """Exact erf-based GELU, matching timm/PyTorch ``nn.GELU()``.

    ``nnx.gelu`` defaults to the tanh approximation, which breaks numerical
    parity when loading timm/PyTorch checkpoints.
    """
    return nnx.gelu(x, approximate=False)


class BatchNorm(nnx.BatchNorm):
    """Batch normalization with timm/PyTorch running-statistics momentum.

    PyTorch's ``momentum=0.1`` update corresponds to Flax ``momentum=0.9``.
    Flax's default of 0.99 averages over about 100 steps, so evaluation-mode
    statistics lag the trained weights on short epochs and small datasets.
    """

    def __init__(self, num_features: int, *, momentum: float = 0.9, **kwargs):
        super().__init__(num_features, momentum=momentum, **kwargs)


def drop_path(
    x: jax.Array,
    rate: float = 0.0,
    deterministic: bool = False,
    *,
    rng: jax.Array | None = None,
) -> jax.Array:
    """Drop paths (Stochastic Depth) per sample (mirrors timm.layers.drop_path)."""
    if rate == 0.0 or deterministic:
        return x
    keep_prob = 1.0 - rate
    if rng is None:
        rng = jax.random.PRNGKey(0)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + jax.random.uniform(rng, shape, dtype=x.dtype)
    binary_tensor = jnp.floor(random_tensor)
    return (x / keep_prob) * binary_tensor


class DropPath(nnx.Module):
    """Stochastic depth with per-sample keep mask (dimension-agnostic: NHWC or BNC).

    Deterministic behavior is toggled automatically by `model.train()` and `model.eval()`.
    """

    def __init__(self, rate: float = 0.0, *, rngs: nnx.Rngs):
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"rate must be between 0 and 1, got {rate}")
        self.rate = rate
        self.deterministic = False
        # Flatten non-batch axes so Dropout's broadcast_dims=(1,)
        # produces one mask per sample for both 4D feature maps and 3D token sequences.
        self.drop = nnx.Dropout(rate, broadcast_dims=(1,), rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.rate == 0.0 or self.deterministic:
            return x
        shape = x.shape
        return self.drop(x.reshape((shape[0], -1)), deterministic=self.deterministic).reshape(shape)


class PatchEmbed(nnx.Module):
    """Convolutional patch embedding layer converting images (B, H, W, C) to (B, H', W', embed_dim)."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        *,
        rngs: nnx.Rngs,
    ):
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nnx.Conv(
            in_chans,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.proj(x)


class Mlp(nnx.Module):
    """Multi-Layer Perceptron (Feed-Forward Network) block with GELU and dropout."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        drop: float = 0.0,
        *,
        kernel_init=None,
        out_kernel_init=None,
        bias_init=None,
        rngs: nnx.Rngs,
    ):
        """Optional initializers override Flax's defaults (``out_kernel_init`` for fc2)."""
        hidden_dim = hidden_dim or dim
        init = {}
        if kernel_init is not None:
            init["kernel_init"] = kernel_init
        if bias_init is not None:
            init["bias_init"] = bias_init
        out_init = init if out_kernel_init is None else {**init, "kernel_init": out_kernel_init}
        self.fc1 = nnx.Linear(dim, hidden_dim, **init, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, dim, **out_init, rngs=rngs)
        self.drop = nnx.Dropout(drop, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.drop(self.fc2(self.drop(gelu(self.fc1(x)))))


# Native Flax NNX activation aliases
hswish = nnx.hard_swish
relu6 = nnx.relu6

_ACTS = {
    "relu": nnx.relu,
    "relu6": nnx.relu6,
    "hswish": nnx.hard_swish,
    "silu": nnx.silu,
    "gelu": gelu,
    "sigmoid": nnx.sigmoid,
    "identity": None,
}


def create_act_layer(act: str = "relu"):
    """Retrieve activation function by name (mirrors timm.layers.create_act_layer)."""
    act_lower = act.lower()
    if act_lower not in _ACTS:
        raise ValueError(f"unsupported activation: {act!r}. Available: {list(_ACTS)}")
    return _ACTS[act_lower]


_NHWC = jax.lax.ConvDimensionNumbers((0, 3, 1, 2), (3, 2, 0, 1), (0, 3, 1, 2))


def _depthwise(x, kernel, padding, precision, preferred_element_type):
    return jax.lax.conv_general_dilated(
        x,
        kernel,
        (1, 1),
        padding,
        dimension_numbers=_NHWC,
        feature_group_count=x.shape[-1],
        precision=precision,
        preferred_element_type=preferred_element_type,
    )


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def _full_map_depthwise(x, kernel, padding, precision, preferred_element_type):
    return _depthwise(x, kernel, padding, precision, preferred_element_type)


def _full_map_depthwise_fwd(x, kernel, padding, precision, preferred_element_type):
    return _depthwise(x, kernel, padding, precision, preferred_element_type), (x, kernel)


def _full_map_depthwise_bwd(padding, precision, preferred_element_type, residuals, grad):
    x, kernel = residuals
    _, input_vjp = jax.vjp(
        lambda x: _depthwise(x, kernel, padding, precision, preferred_element_type), x
    )
    (dx,) = input_vjp(grad)
    kh, kw = kernel.shape[:2]
    xp = jnp.pad(x, ((0, 0), *padding, (0, 0)))
    (hp, wp), (ho, wo) = xp.shape[1:3], grad.shape[1:3]
    # Products of every padded-input and output position, summed over the batch.
    acc = jnp.promote_types(jnp.result_type(xp, grad), jnp.float32)
    pairs = jnp.einsum("bpqc,bhwc->cpqhw", xp, grad, preferred_element_type=acc)
    # Kernel tap (i, j) pairs output (h, w) with padded input (h + i, w + j).
    rows = jnp.arange(hp)[None, :, None] == jnp.arange(kh)[:, None, None] + jnp.arange(ho)
    cols = jnp.arange(wp)[None, :, None] == jnp.arange(kw)[:, None, None] + jnp.arange(wo)
    dk = jnp.einsum("cpqhw,iph,jqw->ijc", pairs, rows.astype(pairs.dtype), cols.astype(pairs.dtype))
    return dx, dk[:, :, None, :].astype(kernel.dtype)


_full_map_depthwise.defvjp(_full_map_depthwise_fwd, _full_map_depthwise_bwd)


def _phase_dilated_conv(x, kernel, stride, dilation, padding, groups, precision, element_type):
    """Dilated convolution as an undilated one over the input's dilation phases."""
    d = dilation
    conv = partial(
        jax.lax.conv_general_dilated,
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=_NHWC,
        feature_group_count=groups,
        precision=precision,
        preferred_element_type=element_type,
    )
    xp = jnp.pad(x, ((0, 0), *padding, (0, 0)))
    if stride == d:  # only phase (0, 0) reaches the strided outputs
        return conv(xp[:, ::d, ::d], kernel)
    b, hp, wp, c = xp.shape
    ho, wo = hp - d * (kernel.shape[0] - 1), wp - d * (kernel.shape[1] - 1)
    m, n = -(-hp // d), -(-wp // d)
    xp = jnp.pad(xp, ((0, 0), (0, m * d - hp), (0, n * d - wp), (0, 0)))
    phases = xp.reshape(b, m, d, n, d, c).transpose(0, 2, 4, 1, 3, 5).reshape(b * d * d, m, n, c)
    y = conv(phases, kernel)
    mo, no = y.shape[1:3]
    y = y.reshape(b, d, d, mo, no, -1).transpose(0, 3, 1, 4, 2, 5)
    return y.reshape(b, d * mo, d * no, -1)[:, :ho, :wo]


def conv_general_dilated(
    lhs,
    rhs,
    window_strides,
    padding,
    lhs_dilation=None,
    rhs_dilation=None,
    dimension_numbers=None,
    feature_group_count=1,
    batch_group_count=1,
    precision=None,
    preferred_element_type=None,
    **kwargs,
):
    """Drop-in ``jax.lax.conv_general_dilated`` avoiding slow grouped-conv kernel gradients.

    XLA lowers some grouped kernel gradients to a cuDNN grouped convolution with the
    batch folded into the channels, which launches one kernel per group: 10-70x
    slower than the forward pass at batch 64. Two NHWC cases are rerouted:

    - depthwise kernels as large as their input map (a 7x7 kernel on the 7x7 last
      stage of ConvNeXt at 224 px) compute the kernel gradient as a batched matmul;
    - dilated grouped convolutions with stride 1 or equal to the dilation (SK-ResNeXt)
      run as undilated convolutions over the input's dilation phases.

    All other calls go straight to ``lax``.
    """
    strides = tuple(window_strides)
    dilation = tuple(rhs_dilation or (1, 1))
    if (
        lhs.ndim == 4
        and feature_group_count > 1
        and batch_group_count == 1
        and tuple(lhs_dilation or (1, 1)) == (1, 1)
        and kwargs.get("out_sharding") is None
        and jax.lax.conv_dimension_numbers(lhs.shape, rhs.shape, dimension_numbers) == _NHWC
    ):
        window = tuple((k - 1) * d + 1 for k, d in zip(rhs.shape[:2], dilation))
        if isinstance(padding, str):
            padding = jax.lax.padtype_to_pads(lhs.shape[1:3], window, strides, padding)
        padding = tuple(tuple(p) for p in padding)
        depthwise = feature_group_count == lhs.shape[3] == rhs.shape[3] and rhs.shape[2] == 1
        if depthwise and strides == dilation == (1, 1) and lhs.shape[1:3] == rhs.shape[:2]:
            return _full_map_depthwise(lhs, rhs, padding, precision, preferred_element_type)
        d = dilation[0]
        if d > 1 and dilation == (d, d) and strides in ((1, 1), (d, d)):
            return _phase_dilated_conv(
                lhs,
                rhs,
                strides[0],
                d,
                padding,
                feature_group_count,
                precision,
                preferred_element_type,
            )
    return jax.lax.conv_general_dilated(
        lhs,
        rhs,
        window_strides,
        padding,
        lhs_dilation,
        rhs_dilation,
        dimension_numbers,
        feature_group_count,
        batch_group_count,
        precision,
        preferred_element_type,
        **kwargs,
    )


def use_fast_grouped_conv_grads(model: nnx.Module) -> nnx.Module:
    """Route a model's grouped ``nnx.Conv`` layers through :func:`conv_general_dilated`.

    ``jimm.create_model`` applies this to every model it builds.
    """
    for _, node in nnx.graph.iter_graph(model):
        if (
            isinstance(node, nnx.Conv)
            and node.feature_group_count > 1
            and node.conv_general_dilated is jax.lax.conv_general_dilated
        ):
            node.conv_general_dilated = conv_general_dilated
    return model


class ConvBNAct(nnx.Module):
    """Universal CNN block: Convolution -> optional BatchNorm -> optional Activation.

    Operates natively on NHWC tensors with HWIO conv kernel layouts.
    """

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        kernel: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        groups: int = 1,
        act: str = "relu",
        use_bn: bool = True,
        dilation: int = 1,
        padding: str = "SAME",
        *,
        rngs: nnx.Rngs,
    ):
        k = (kernel, kernel) if isinstance(kernel, int) else tuple(kernel)
        s = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.conv = nnx.Conv(
            in_chs,
            out_chs,
            k,
            strides=s,
            padding=padding,
            use_bias=not use_bn,
            feature_group_count=groups,
            kernel_dilation=(dilation, dilation),
            rngs=rngs,
        )
        self.bn = BatchNorm(out_chs, rngs=rngs) if use_bn else None
        self.act = _ACTS[act]

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x if self.act is None else self.act(x)


def make_divisible(value: float, divisor: int = 8, min_value: int | None = None) -> int:
    """Round channels to a multiple of ``divisor`` as timm does, staying within 10%."""
    min_value = min_value or divisor
    rounded = max(min_value, int(value + divisor / 2) // divisor * divisor)
    return rounded + divisor if rounded < 0.9 * value else rounded


class SqueezeExcite(nnx.Module):
    """Squeeze-and-Excitation (SE) channel-attention block on NHWC feature maps."""

    def __init__(
        self,
        chs: int,
        rd_ratio: float = 0.25,
        *,
        rd_channels: int | None = None,
        act=nnx.relu,
        gate=nnx.sigmoid,
        rngs: nnx.Rngs,
    ):
        if rd_channels is None:
            try:
                rd_channels = max(int(chs * rd_ratio), 1)
            except (TypeError, ValueError):
                rd_channels = 1
        self.fc1 = nnx.Linear(chs, rd_channels, rngs=rngs)
        self.fc2 = nnx.Linear(rd_channels, chs, rngs=rngs)
        self.act = act
        self.gate = gate

    def __call__(self, x: jax.Array) -> jax.Array:
        s = jnp.mean(x, axis=(1, 2), keepdims=True)
        s = self.gate(self.fc2(self.act(self.fc1(s))))
        return x * s


def global_pool_nhwc(x: jax.Array, pool_type: str = "avg") -> jax.Array:
    """Global spatial pooling over H, W dimensions: (B, H, W, C) -> (B, C).

    Args:
        x: Input tensor of shape (Batch, Height, Width, Channels).
        pool_type: 'avg' for average pooling, 'max' for max pooling, '' for pass-through.
    """
    if pool_type == "avg":
        return jnp.mean(x, axis=(1, 2))
    if pool_type == "max":
        return jnp.max(x, axis=(1, 2))
    if pool_type == "":
        return x
    raise ValueError(f"unsupported pool {pool_type!r}")


class ClassifierMixin:
    """Shared timm-style classification head boilerplate for vision architectures.

    Subclasses configure:
      _classifier_attr: Attribute name of the classifier Linear layer ('fc' for CNNs, 'head' for ViTs).
      _default_global_pool: Default spatial pooling mode ('avg' for CNNs, '' for token models).
    """

    _classifier_attr: str = "fc"
    _default_global_pool: str = "avg"
    default_cfg: dict | None = None

    num_features: int
    head_drop: nnx.Dropout
    num_classes: int
    global_pool: str

    def get_classifier(self) -> nnx.Linear | None:
        """Return the classifier linear layer or None if num_classes=0."""
        return getattr(self, self._classifier_attr)

    def reset_classifier(self, num_classes: int, global_pool: str | None = None) -> None:
        """Reset the classifier head with a new number of classes."""
        if global_pool is None:
            global_pool = self._default_global_pool
        self.num_classes = num_classes
        self.global_pool = global_pool
        if num_classes > 0 and getattr(self, self._classifier_attr) is None:
            raise RuntimeError("cannot re-add classifier to a num_classes=0 model")
        setattr(
            self,
            self._classifier_attr,
            nnx.Linear(self.num_features, num_classes, rngs=nnx.Rngs(0))
            if num_classes > 0
            else None,
        )

    def forward_head(self, x: jax.Array) -> jax.Array:
        """Apply spatial pooling, dropout, and classifier linear projection."""
        x = global_pool_nhwc(x, self.global_pool)
        x = self.head_drop(x)
        fc = getattr(self, self._classifier_attr)
        return fc(x) if fc is not None else x
