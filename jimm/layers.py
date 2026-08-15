"""Shared layers, NHWC convention (mirrors timm.models.layers)."""
import jax
import jax.numpy as jnp
from flax import nnx

__all__ = ["DropPath", "PatchEmbed", "Mlp", "SqueezeExcite", "ConvBNAct", "ClassifierMixin",
           "global_pool_nhwc", "hswish", "relu6"]


class DropPath(nnx.Module):
    """Stochastic depth with a per-sample keep mask (dimension-agnostic: works for
    NHWC conv maps and BNC token sequences). deterministic is toggled by model.train()/eval()."""

    def __init__(self, rate: float = 0.0, *, rngs):
        self.rate = rate
        self.rngs = rngs["dropout"].fork() if rate > 0 else None
        self.deterministic = False

    def __call__(self, x):
        if self.rngs is None or self.deterministic:
            return x
        keep = 1.0 - self.rate
        mask = jax.random.bernoulli(self.rngs(), keep, (x.shape[0],))
        return x * mask.reshape((x.shape[0],) + (1,) * (x.ndim - 1)) / keep


class PatchEmbed(nnx.Module):
    """Conv patch embedding, returns (B, H', W', embed_dim)."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, *, rngs):
        self.img_size, self.patch_size = img_size, patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nnx.Conv(in_chans, embed_dim, kernel_size=(patch_size, patch_size),
                             strides=(patch_size, patch_size), rngs=rngs)

    def __call__(self, x):
        return self.proj(x)


class Mlp(nnx.Module):
    def __init__(self, dim, hidden_dim=None, drop=0.0, *, rngs):
        hidden_dim = hidden_dim or dim
        self.fc1 = nnx.Linear(dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, dim, rngs=rngs)
        self.drop = nnx.Dropout(drop, rngs=rngs)

    def __call__(self, x):
        return self.drop(self.fc2(self.drop(nnx.gelu(self.fc1(x)))))


# Native Flax NNX activation aliases
hswish = nnx.hard_swish
relu6 = nnx.relu6

_ACTS = {
    "relu": nnx.relu,
    "relu6": nnx.relu6,
    "hswish": nnx.hard_swish,
    "silu": nnx.silu,
    "gelu": nnx.gelu,
    "sigmoid": nnx.sigmoid,
    "identity": None,
}


class ConvBNAct(nnx.Module):
    """Conv -> optional BN -> optional activation, the universal CNN building block."""

    def __init__(self, in_chs, out_chs, kernel: int | tuple[int, int] = 3, stride=1, groups=1,
                 act="relu", use_bn=True, dilation=1, padding="SAME", *, rngs):
        k = (kernel, kernel) if isinstance(kernel, int) else tuple(kernel)
        self.conv = nnx.Conv(in_chs, out_chs, k, strides=(stride, stride),
                             padding=padding, use_bias=not use_bn,
                             feature_group_count=groups, kernel_dilation=(dilation, dilation), rngs=rngs)
        self.bn = nnx.BatchNorm(out_chs, rngs=rngs) if use_bn else None
        self.act = _ACTS[act]

    def __call__(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x if self.act is None else self.act(x)


class SqueezeExcite(nnx.Module):
    """SE block on NHWC features."""

    def __init__(self, chs, rd_ratio=0.25, *, rngs):
        rd = max(int(chs * rd_ratio), 1)
        self.fc1 = nnx.Linear(chs, rd, rngs=rngs)
        self.fc2 = nnx.Linear(rd, chs, rngs=rngs)

    def __call__(self, x):
        s = jnp.mean(x, axis=(1, 2), keepdims=True)
        s = nnx.sigmoid(self.fc2(nnx.relu(self.fc1(s))))
        return x * s


def global_pool_nhwc(x, pool_type="avg"):
    """(B,H,W,C) -> (B,C). pool_type: 'avg' | 'max' | '' (flatten handled by caller)."""
    if pool_type == "avg":
        return jnp.mean(x, axis=(1, 2))
    if pool_type == "max":
        return jnp.max(x, axis=(1, 2))
    raise ValueError(f"unsupported pool {pool_type!r}")


class ClassifierMixin:
    """Shared timm-style classifier boilerplate (was duplicated across ~90 model files).

    Subclasses set:
      _classifier_attr: name of the classifier Linear ('fc' for conv models, 'head' for ViT-family)
      _default_global_pool: reset_classifier default ('avg' conv, '' token models, 'token' VOLO)
    Token-based models override forward_head with their own pooling logic.
    """
    _classifier_attr = "fc"
    _default_global_pool = "avg"
    # subclass contract (declared so type checkers accept attribute access)
    num_features: int
    head_drop: nnx.Dropout
    num_classes: int
    global_pool: str

    def get_classifier(self):
        return getattr(self, self._classifier_attr)

    def reset_classifier(self, num_classes, global_pool=None):
        if global_pool is None:
            global_pool = self._default_global_pool
        self.num_classes, self.global_pool = num_classes, global_pool
        if num_classes > 0 and getattr(self, self._classifier_attr) is None:
            raise RuntimeError("cannot re-add classifier to a num_classes=0 model")
        setattr(self, self._classifier_attr,
                nnx.Linear(self.num_features, num_classes, rngs=nnx.Rngs(0)) if num_classes > 0 else None)

    def forward_head(self, x):
        x = global_pool_nhwc(x, self.global_pool)
        x = self.head_drop(x)
        fc = getattr(self, self._classifier_attr)
        return fc(x) if fc is not None else x
