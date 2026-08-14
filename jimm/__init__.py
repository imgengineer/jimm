"""jimm: JAX Image Models. timm-style API on JAX / flax nnx.

NHWC convention throughout (flax native), unlike timm's NCHW.
"""
from .registry import (create_model, list_models, list_modules, register_model,
                       model_entrypoint, get_default_cfg, is_model)
from . import models  # noqa: F401
from . import data, checkpoint  # noqa: F401

__version__ = "0.1.0"
