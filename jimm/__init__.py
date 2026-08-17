"""jimm: JAX Image Models. timm-style API on JAX / flax nnx.

NHWC convention throughout (flax native), unlike timm's NCHW.
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

from .registry import (create_model, list_models, list_modules, register_model,
                       model_entrypoint, get_default_cfg, is_model)
from . import models  # noqa: F401
from . import data, checkpoint, features, weights  # noqa: F401

__version__ = "0.1.0"
