"""Orbax-checkpoint serialization and restoration for Flax NNX models & Optax optimizers.

Features:
  - Non-blocking asynchronous background checkpoint writes (MaxText/MaxDiffusion pattern).
  - Lazy initialization of Orbax Checkpointer (avoids spawning CUDA contexts in Grain workers).
  - Pure Python dictionary serialization compatible with `flax.nnx.to_pure_dict`.
  - Automatic recursive integer key casting for `nnx.List` and `nnx.Sequential` submodules.
"""
import errno
import os
from typing import Any

from flax import nnx
import orbax.checkpoint as ocp

__all__ = ["save_checkpoint", "load_checkpoint", "wait_for_checkpoints"]

_checkpointer: ocp.StandardCheckpointer | None = None


def _get_checkpointer() -> ocp.StandardCheckpointer:
    """Lazily instantiate the Orbax checkpointer singleton on first use.

    Avoids importing/initializing CUDA contexts during multiprocessing worker startup.
    """
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = ocp.StandardCheckpointer()
    return _checkpointer


def _item(
    model: nnx.Module,
    optimizer: nnx.Optimizer | None,
    epoch: int,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    """Serialize model and optimizer states into pure dictionaries."""
    item: dict[str, Any] = {
        "epoch": epoch,
        "model": nnx.to_pure_dict(nnx.state(model)),
    }
    if optimizer is not None:
        item["optimizer"] = nnx.to_pure_dict(nnx.state(optimizer))
    if extra:
        item["extra"] = extra
    return item


def save_checkpoint(
    path: str,
    model: nnx.Module,
    optimizer: nnx.Optimizer | None = None,
    epoch: int = 0,
    extra: dict[str, Any] | None = None,
    wait: bool = True,
) -> str:
    """Save model (and optional optimizer) state to disk using Orbax.

    Args:
        path: Target checkpoint directory path.
        model: Flax NNX model instance.
        optimizer: Optional Flax NNX optimizer instance.
        epoch: Current training epoch index.
        extra: Optional arbitrary metadata dictionary (metrics, hyperparameters).
        wait: If True, blocks until write completes; if False, writes asynchronously.

    Returns:
        Absolute path to the saved checkpoint directory.
    """
    path = os.path.abspath(path)
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        if not os.path.isdir(path):
            raise
        if exc.errno != errno.EEXIST:
            raise
    _get_checkpointer().save(path, _item(model, optimizer, epoch, extra), force=True)
    if wait:
        _get_checkpointer().wait_until_finished()
    return path


def wait_for_checkpoints() -> None:
    """Block until all pending asynchronous checkpoint writes complete."""
    if _checkpointer is not None:
        _checkpointer.wait_until_finished()


def _fix_int_keys(d: Any) -> Any:
    """Recursively convert stringified integer keys ('0', '1') to integers for nnx.List/nnx.Sequential."""
    if isinstance(d, dict):
        new_d: dict[Any, Any] = {}
        for k, v in d.items():
            key = k
            if isinstance(k, str) and (k.isdigit() or (k.startswith("-") and k[1:].isdigit())):
                try:
                    key = int(k)
                except ValueError:
                    key = k
            new_d[key] = _fix_int_keys(v)
        return new_d
    return d


def load_checkpoint(
    path: str,
    model: nnx.Module,
    optimizer: nnx.Optimizer | None = None,
) -> int:
    """Restore saved parameters into live Flax NNX model and optimizer instances.

    Args:
        path: Path to checkpoint directory.
        model: Live Flax NNX model to receive restored weights.
        optimizer: Optional live Flax NNX optimizer to receive restored states.

    Returns:
        Restored epoch integer index.
    """
    path = os.path.abspath(path)
    restored = _get_checkpointer().restore(path)
    nnx.update(model, _fix_int_keys(restored["model"]))
    if optimizer is not None and "optimizer" in restored:
        nnx.update(optimizer, _fix_int_keys(restored["optimizer"]))
    return restored.get("epoch", 0)
