"""jimm: JAX Image Models — A high-performance timm-style library on JAX/Flax NNX.

Architecture & Conventions:
  - Tensor Layout: NHWC (Batch, Height, Width, Channels) throughout, optimized for
    JAX, XLA, NVIDIA Tensor Cores, and TPUs.
  - Model API: Pure Flax NNX object-oriented modules with functional JAX transformations
    (`nnx.jit`, `nnx.grad`, `nnx.vmap`, `nnx.split`, `nnx.merge`).
  - Weights Layout: Conv kernels stored in (H, W, In, Out), Linear weights in (In, Out).
  - Registry: Explicitly supported architectures with timm-style entrypoints.
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

from . import (  # noqa: F401
    augment,
    checkpoint,
    data,
    features,
    layers,
    loss,
    models,
    optim,
    registry,
    train,
    weights,
)
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
    create_act_layer,
    drop_path,
    global_pool_nhwc,
    hswish,
    relu6,
)
from .loss import (
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
    cross_entropy,
)
from .optim import (
    create_optimizer,
    make_optimizer,
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
    "drop_path",
    "create_act_layer",
    "PatchEmbed",
    "Mlp",
    "SqueezeExcite",
    "ConvBNAct",
    "ClassifierMixin",
    "global_pool_nhwc",
    "hswish",
    "relu6",
    # Loss functions
    "cross_entropy",
    "LabelSmoothingCrossEntropy",
    "SoftTargetCrossEntropy",
    # Optimizers
    "create_optimizer",
    "make_optimizer",
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
    "loss",
    "optim",
]
