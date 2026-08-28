"""Shared reusable neural network layers and mixins for JAX/Flax NNX models.

Layout Convention:
  - All convolution and spatial feature maps follow the NHWC convention:
    (Batch, Height, Width, Channels).
  - Dense token sequences follow BNC: (Batch, Num_Tokens, Channels).
  - DropPath and BatchNorm modes automatically switch via `model.train()` and `model.eval()`.
"""

import jax
import jax.numpy as jnp
from flax import nnx

__all__ = [
    "DropPath",
    "PatchEmbed",
    "Mlp",
    "SqueezeExcite",
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
        rngs: nnx.Rngs,
    ):
        hidden_dim = hidden_dim or dim
        self.fc1 = nnx.Linear(dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, dim, rngs=rngs)
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


class ConvBNAct(nnx.Module):
    """Universal CNN block: Convolution -> optional BatchNorm -> optional Activation.

    Operates natively on NHWC tensors with HWIO conv kernel layouts.
    """

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        kernel: int | tuple[int, int] = 3,
        stride: int = 1,
        groups: int = 1,
        act: str = "relu",
        use_bn: bool = True,
        dilation: int = 1,
        padding: str = "SAME",
        *,
        rngs: nnx.Rngs,
    ):
        k = (kernel, kernel) if isinstance(kernel, int) else tuple(kernel)
        self.conv = nnx.Conv(
            in_chs,
            out_chs,
            k,
            strides=(stride, stride),
            padding=padding,
            use_bias=not use_bn,
            feature_group_count=groups,
            kernel_dilation=(dilation, dilation),
            rngs=rngs,
        )
        self.bn = nnx.BatchNorm(out_chs, rngs=rngs) if use_bn else None
        self.act = _ACTS[act]

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x if self.act is None else self.act(x)


class SqueezeExcite(nnx.Module):
    """Squeeze-and-Excitation (SE) channel-attention block on NHWC feature maps."""

    def __init__(self, chs: int, rd_ratio: float = 0.25, *, rngs: nnx.Rngs):
        rd = max(int(chs * rd_ratio), 1)
        self.fc1 = nnx.Linear(chs, rd, rngs=rngs)
        self.fc2 = nnx.Linear(rd, chs, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        s = jnp.mean(x, axis=(1, 2), keepdims=True)
        s = nnx.sigmoid(self.fc2(nnx.relu(self.fc1(s))))
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
