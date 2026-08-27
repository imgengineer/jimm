"""jimm: JAX Image Models — A high-performance timm-style library on JAX/Flax NNX.

Architecture & Conventions:
  - Tensor Layout: NHWC (Batch, Height, Width, Channels) throughout, optimized for
    JAX, XLA, NVIDIA Tensor Cores, and TPUs.
  - Model API: Pure Flax NNX object-oriented modules with functional JAX transformations
    (`nnx.jit`, `nnx.grad`, `nnx.vmap`, `nnx.split`, `nnx.merge`).
  - Weights Layout: Conv kernels stored in (H, W, In, Out), Linear weights in (In, Out).
  - Registry: 100% compatible coverage of timm entrypoints and architectures.
"""
import logging

import jax

# Enable NVIDIA JAX-Toolbox recommended O1 optimization level by default.
try:
    jax.config.update("jax_optimization_level", "O1")
except (AttributeError, ValueError, KeyError):
    logging.getLogger(__name__).debug(
        "jax_optimization_level=O1 is unavailable; using JAX defaults",
        exc_info=True,
    )

from . import augment, checkpoint, data, features, layers, models, registry, train, weights  # noqa: F401
from .checkpoint import (
    CheckpointManager,
    load_checkpoint,
    save_checkpoint,
    wait_for_checkpoints,
)
from .data import ImageFolder, Loader, MixupCutmix, create_dataset, create_loader
from .features import FeatureExtractor, FeatureInfo, create_feature_extractor
from .layers import (
    ClassifierMixin,
    ConvBNAct,
    DropPath,
    Mlp,
    PatchEmbed,
    SqueezeExcite,
    global_pool_nhwc,
    hswish,
    relu6,
)
from .registry import (
    create_model,
    get_default_cfg,
    is_model,
    list_models,
    list_modules,
    model_entrypoint,
    register_model,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Registry & Creation
    "create_model",
    "list_models",
    "list_modules",
    "register_model",
    "model_entrypoint",
    "get_default_cfg",
    "is_model",
    # Layers & Mixins
    "DropPath",
    "PatchEmbed",
    "Mlp",
    "SqueezeExcite",
    "ConvBNAct",
    "ClassifierMixin",
    "global_pool_nhwc",
    "hswish",
    "relu6",
    # Feature Extraction
    "FeatureExtractor",
    "FeatureInfo",
    "create_feature_extractor",
    # Checkpointing
    "save_checkpoint",
    "load_checkpoint",
    "wait_for_checkpoints",
    "CheckpointManager",
    # Data & Loaders
    "create_loader",
    "create_dataset",
    "ImageFolder",
    "Loader",
    "MixupCutmix",
    # Submodules
    "models",
    "layers",
    "registry",
    "data",
    "augment",
    "checkpoint",
    "features",
    "weights",
    "train",
]
