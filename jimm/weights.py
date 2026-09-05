"""Weights conversion and pretrained checkpoint loading for jimm models.

Converts standard PyTorch/timm weight formats into JAX/Flax NNX native layout:
  - Conv2D weights: PyTorch (Out, In, H, W) -> JAX (H, W, In, Out)
  - Linear weights: PyTorch (Out, In) -> JAX (In, Out)
  - Parameter paths: PyTorch hierarchical keys -> Flax NNX attribute trees
"""

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from flax import nnx

__all__ = ["load_state_dict", "load_pretrained"]


def _convert_key(k: str) -> list[str]:
    """Map a PyTorch state_dict parameter key to Flax NNX path parts."""
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
    # 4D Conv: PyTorch (O, I, H, W) -> JAX (H, W, I, O)
    if a.ndim == 4:
        return a.transpose(2, 3, 1, 0)
    # 2D Linear: PyTorch (O, I) -> JAX (I, O)
    if a.ndim == 2 and _convert_key(k)[-1] == "kernel":
        return a.T
    return a


def load_state_dict(
    model: nnx.Module,
    state_dict: dict[str, Any],
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Load a dictionary of PyTorch-style parameter arrays into an NNX model.

    Args:
        model: Live Flax NNX model instance.
        state_dict: Parameter dictionary mapping key strings to arrays.
        strict: If True, raises when encountering missing or unmatched parameters.

    Returns:
        Tuple of (loaded_keys, missing_keys).
    """
    loaded: list[str] = []
    missing: list[str] = []
    updates: list[tuple[str, nnx.Variable, Any]] = []

    for k, v in state_dict.items():
        parts = _convert_key(k)
        converted_v = _convert_tensor(k, v)

        # Traverse to target attribute in model graph
        curr: Any = model
        failed = False
        for p in parts[:-1]:
            if p.isdigit():
                idx = int(p)
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
                target = node.get_value()
                value = jnp.asarray(converted_v, dtype=getattr(target, "dtype", None))
                if not hasattr(target, "shape") or target.shape == value.shape:
                    updates.append((k, node, value))
                else:
                    missing.append(k)
            else:
                missing.append(k)
        else:
            missing.append(k)

    if strict:
        supplied = {id(node) for _, node, _ in updates}
        absent = [
            ".".join(map(str, path))
            for path, node in nnx.graph.iter_graph(model)
            if isinstance(node, (nnx.Param, nnx.BatchStat)) and id(node) not in supplied
        ]
        if missing or absent:
            raise RuntimeError(
                f"Failed to load weights strictly: unmatched keys: {missing[:10]}; "
                f"missing model parameters/statistics: {absent[:10]}"
            )

    for key, node, value in updates:
        node.set_value(value)
        loaded.append(key)

    return loaded, missing


def load_pretrained(
    model: nnx.Module,
    checkpoint_path: str,
) -> tuple[list[str], list[str]]:
    """Load pretrained model weights from a local `.npz` archive.

    Args:
        model: Target Flax NNX model.
        checkpoint_path: Path to `.npz` weight file.

    Returns:
        Tuple of (loaded_keys, missing_keys).
    """
    checkpoint = Path(checkpoint_path).expanduser()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not checkpoint.is_file() or checkpoint.suffix.lower() != ".npz":
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path} (expected .npz)")
    with np.load(checkpoint) as data:
        state_dict = {k: data[k] for k in data.files}
    return load_state_dict(model, state_dict)
