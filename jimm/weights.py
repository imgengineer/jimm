"""Weights conversion and pretrained checkpoint loading for jimm models."""
from typing import Any
import os
import numpy as np
import jax.numpy as jnp
from flax import nnx


def _convert_key(k: str) -> list[str]:
    """Map PyTorch state_dict parameter key to jimm Flax NNX path parts."""
    # ResNet / ConvNeXt stage mapping
    k = k.replace("layer1.", "stages.0.")
    k = k.replace("layer2.", "stages.1.")
    k = k.replace("layer3.", "stages.2.")
    k = k.replace("layer4.", "stages.3.")
    # Shortcuts / downsamples
    k = k.replace("downsample.0.", "shortcut.conv.")
    k = k.replace("downsample.1.", "shortcut.bn.")
    k = k.replace("downsample.", "shortcut.")
    # Batch norm running stats
    k = k.replace("running_mean", "mean")
    k = k.replace("running_var", "var")
    # ViT / Transformer layers
    k = k.replace("transformer.layers.", "blocks.")
    k = k.replace("layers.", "blocks.")
    # Parameter names
    parts = k.split(".")
    # Map final weight -> kernel / scale
    if parts[-1] == "weight":
        if any(w in k for w in ["conv", "fc", "head", "qkv", "proj", "stem", "mlp"]):
            parts[-1] = "kernel"
        elif any(w in k for w in ["bn", "norm"]):
            parts[-1] = "scale"
    return parts


def _convert_tensor(k: str, v: np.ndarray | Any) -> np.ndarray:
    """Convert tensor layout from PyTorch (OIHW / OI) to JAX (HWIO / IO)."""
    a = np.asarray(v)
    # 4D Conv: (O, I, H, W) -> (H, W, I, O)
    if a.ndim == 4:
        return a.transpose(2, 3, 1, 0)
    # 2D Linear: (O, I) -> (I, O)
    if a.ndim == 2 and ("kernel" in k or "fc" in k or "head" in k or "proj" in k or "mlp" in k):
        return a.T
    return a


def load_state_dict(model: nnx.Module, state_dict: dict[str, Any], strict: bool = False) -> tuple[list[str], list[str]]:
    """Load a state dictionary (PyTorch or SafeTensors format) into a Flax NNX model.

    Returns:
        (loaded_keys, missing_keys)
    """
    loaded = []
    missing = []
    for k, v in state_dict.items():
        parts = _convert_key(k)
        converted_v = _convert_tensor(k, v)
        # Navigate to destination in model
        curr = model
        failed = False
        for p in parts[:-1]:
            if p.isdigit():
                try:
                    idx = int(p)
                except ValueError:
                    failed = True
                    break
                if isinstance(curr, (list, nnx.List)) and idx < len(curr):
                    curr = curr[idx]
                else:
                    failed = True
                    break
            else:
                if hasattr(curr, p):
                    curr = getattr(curr, p)
                else:
                    failed = True
                    break
        if failed:
            missing.append(k)
            continue
        attr = parts[-1]
        if hasattr(curr, attr):
            node = getattr(curr, attr)
            if isinstance(node, nnx.Variable):
                val = converted_v
                # Check shape match
                if hasattr(node, "shape") and node.shape == val.shape:
                    node.set_value(jnp.asarray(val))
                    loaded.append(k)
                elif not hasattr(node, "shape"):
                    node.set_value(jnp.asarray(val))
                    loaded.append(k)
                else:
                    missing.append(k)
            else:
                setattr(curr, attr, jnp.asarray(converted_v))
                loaded.append(k)
        else:
            missing.append(k)
    return loaded, missing


def load_pretrained(model: nnx.Module, checkpoint_path: str) -> tuple[list[str], list[str]]:
    """Load pretrained weights from a local file (.npz, .safetensors) into model."""
    if os.path.exists(checkpoint_path):
        if checkpoint_path.endswith(".npz"):
            with np.load(checkpoint_path) as data:
                state_dict = {k: data[k] for k in data.files}
            return load_state_dict(model, state_dict)
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
