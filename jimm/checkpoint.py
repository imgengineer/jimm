"""Orbax-checkpoint serialization and restoration for Flax NNX models & Optax optimizers.

Features:
  - Non-blocking asynchronous background checkpoint writes (MaxText/MaxDiffusion pattern).
  - Lazy initialization of Orbax Checkpointer (avoids spawning CUDA contexts in Grain workers).
  - Pure Python dictionary serialization compatible with `flax.nnx.to_pure_dict`.
  - Automatic recursive integer key casting for `nnx.List` and `nnx.Sequential` submodules.
  - `CheckpointManager`: step-numbered checkpoints with retention (`max_to_keep`)
    and best-checkpoint tracking.
"""
import os
from typing import Any, Callable

import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "wait_for_checkpoints",
    "CheckpointManager",
]

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
    os.makedirs(path, exist_ok=True)
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
            key = (int(k) if isinstance(k, str)
                   and (k.isdigit() or (k.startswith("-") and k[1:].isdigit())) else k)
            new_d[key] = _fix_int_keys(v)
        return new_d
    return d


def _validate_tree(expected: Any, restored: Any, path: str) -> Any:
    """Check a restored tree against the live state tree, returning casted leaves.

    Without this, a shape-mismatched checkpoint is silently accepted by
    ``nnx.update`` and only fails much later with a cryptic XLA error.
    """
    if isinstance(expected, dict):
        if not isinstance(restored, dict):
            raise ValueError(
                f"{path}: checkpoint has {type(restored).__name__}, expected a dict subtree")
        missing = sorted(set(expected) - set(restored), key=repr)
        unexpected = sorted(set(restored) - set(expected), key=repr)
        if missing or unexpected:
            raise ValueError(
                f"{path}: checkpoint tree does not match model state "
                f"(missing keys: {missing}, unexpected keys: {unexpected})")
        return {key: _validate_tree(expected[key], restored[key], f"{path}.{key}")
                for key in expected}
    if hasattr(expected, "shape"):
        value = jnp.asarray(restored)
        if tuple(value.shape) != tuple(expected.shape):
            raise ValueError(
                f"{path}: shape mismatch (checkpoint {tuple(value.shape)} "
                f"vs model {tuple(expected.shape)})")
        if value.dtype != expected.dtype:
            value = value.astype(expected.dtype)
        return value
    return restored


def _apply_restored(restored: Any, model: nnx.Module, optimizer: nnx.Optimizer | None,
                    source: str) -> int:
    """Validate a restored item against live state and load it in place."""
    if not isinstance(restored, dict) or "model" not in restored:
        raise ValueError(f"{source} is not a jimm checkpoint (no 'model' entry)")
    model_state = _validate_tree(
        nnx.to_pure_dict(nnx.state(model)),
        _fix_int_keys(restored["model"]),
        "model",
    )
    optimizer_state = None
    if optimizer is not None:
        if "optimizer" not in restored:
            raise ValueError(f"{source} has no optimizer state")
        optimizer_state = _validate_tree(
            nnx.to_pure_dict(nnx.state(optimizer)),
            _fix_int_keys(restored["optimizer"]),
            "optimizer",
        )
    try:
        epoch = int(restored.get("epoch", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} has an invalid epoch") from exc

    # Do not mutate either object until every requested state has validated.
    nnx.update(model, model_state)
    if optimizer_state is not None:
        nnx.update(optimizer, optimizer_state)
    return epoch


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

    Raises:
        ValueError: If the checkpoint is missing model state or its tree
            structure / shapes do not match the live model.
    """
    path = os.path.abspath(path)
    restored = _get_checkpointer().restore(path)
    return _apply_restored(restored, model, optimizer, path)


class CheckpointManager:
    """Step-numbered checkpoints with retention and best-checkpoint tracking.

    Unlike :func:`save_checkpoint` (one explicit directory per call), checkpoints
    live under numbered step directories (``<directory>/<step>``) managed by
    Orbax, which enables:

      - ``max_to_keep``: automatically delete the oldest checkpoints, keeping
        only the N most recent ones.
      - ``best_fn``/``best_mode``: additionally retain the checkpoint with the
        best metric (e.g. lowest validation loss) even after it would have been
        rotated out by ``max_to_keep``.

    Restores go through this manager as well; ``load_checkpoint`` cannot read
    manager step directories (their metadata lives at the manager root).
    """

    def __init__(
        self,
        directory: str,
        max_to_keep: int | None = None,
        best_fn: Callable[[Any], float] | None = None,
        best_mode: str = "max",
    ):
        if max_to_keep is not None and max_to_keep <= 0:
            raise ValueError("max_to_keep must be positive or None")
        if best_mode not in ("min", "max"):
            raise ValueError("best_mode must be 'min' or 'max'")
        self.directory = os.path.abspath(directory)
        policy = ocp.checkpoint_managers.LatestN(max_to_keep)
        if best_fn is not None:
            policy = ocp.checkpoint_managers.AnyPreservationPolicy([
                policy,
                ocp.checkpoint_managers.BestN(
                    get_metric_fn=best_fn,
                    reverse=best_mode == "min",
                    n=1,
                    keep_checkpoints_without_metrics=False,
                ),
            ])
        options = ocp.CheckpointManagerOptions(
            preservation_policy=policy)
        self._manager = ocp.CheckpointManager(self.directory, options=options)

    def save(
        self,
        step: int,
        model: nnx.Module,
        optimizer: nnx.Optimizer | None = None,
        extra: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        force: bool = True,
    ) -> bool:
        """Save model/optimizer state as step ``step`` (asynchronous).

        Args:
            metrics: Optional metric values (e.g. ``{"val_acc": 0.74}``) used by
                ``best_fn`` for best-checkpoint retention.
            force: If True (the default), overwrite an existing checkpoint for
                the same step; Orbax itself refuses to re-save a step.
        """
        step = int(step)
        if force and step in self._manager.all_steps():
            self._manager.delete(step)
        return self._manager.save(
            step,
            args=ocp.args.StandardSave(_item(model, optimizer, step, extra)),
            metrics=metrics,
        )

    def restore(self, step: int, model: nnx.Module,
                optimizer: nnx.Optimizer | None = None) -> int:
        """Restore step ``step`` into live model/optimizer; returns its epoch."""
        restored = self._manager.restore(int(step), args=ocp.args.StandardRestore())
        return _apply_restored(restored, model, optimizer, f"{self.directory}/{step}")

    def restore_latest(
        self, model: nnx.Module, optimizer: nnx.Optimizer | None = None,
    ) -> tuple[int | None, int | None]:
        """Restore the newest checkpoint; returns ``(step, epoch)``, both None if empty."""
        step = self.latest_step()
        if step is None:
            return None, None
        return step, self.restore(step, model, optimizer)

    def latest_step(self) -> int | None:
        return self._manager.latest_step()

    def all_steps(self) -> list[int]:
        return self._manager.all_steps()

    def wait_until_finished(self) -> None:
        self._manager.wait_until_finished()

    def close(self) -> None:
        """Wait for pending saves, surface any save errors, and release resources."""
        self._manager.check_for_errors()
        self._manager.close()

    def __enter__(self) -> "CheckpointManager":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
