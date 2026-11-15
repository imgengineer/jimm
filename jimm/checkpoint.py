"""Orbax-checkpoint serialization and restoration for Flax NNX models & Optax optimizers.

Features:
  - Background checkpoint writes (MaxText/MaxDiffusion pattern): the array write
    blocks, then Orbax finalizes the checkpoint asynchronously.
  - Lazy initialization of Orbax Checkpointer (avoids spawning CUDA contexts in Grain workers).
  - Pure Python dictionary serialization compatible with `flax.nnx.to_pure_dict`.
  - Automatic recursive integer key casting for `nnx.List` and `nnx.Sequential` submodules.
  - `CheckpointManager`: step-numbered checkpoints with retention (`max_to_keep`)
    and best-checkpoint tracking.
"""

import os
import threading
import time
import warnings
from typing import Any, Callable

import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "wait_for_checkpoints",
    "CheckpointManager",
]

_checkpointer: ocp.StandardCheckpointer | None = None

# `CheckpointManager` step directories look like a single-directory checkpoint but
# keep the metadata of each item in a `default/` subdirectory, so the step directory
# only has the manager-level `_CHECKPOINT_METADATA` file (no root `_METADATA`).
_MANAGER_METADATA_FILE = "_CHECKPOINT_METADATA"
_STEP_METADATA_FILE = "_METADATA"


def _manager_step(path: str) -> tuple[str, int] | None:
    """Return ``(manager directory, step)`` if ``path`` is a manager step directory."""
    normalized = os.path.normpath(path)
    step = os.path.basename(normalized)
    if not step.isdigit() or not os.path.isdir(normalized):
        return None
    if os.path.exists(os.path.join(normalized, _STEP_METADATA_FILE)):
        return None
    if not os.path.exists(os.path.join(normalized, _MANAGER_METADATA_FILE)):
        return None
    return os.path.dirname(normalized), int(step)


def _is_manager_directory(path: str) -> bool:
    """Whether ``path`` is a ``CheckpointManager`` root (numbered step directories)."""
    if not os.path.isdir(path) or os.path.exists(os.path.join(path, _STEP_METADATA_FILE)):
        return False
    if os.path.exists(os.path.join(path, _MANAGER_METADATA_FILE)):
        return True
    names = os.listdir(path)
    return bool(names) and all(
        name.isdigit() and os.path.isdir(os.path.join(path, name)) for name in names
    )


# `CheckpointManagerOptions.enable_background_delete=True` makes checkpoint removals
# asynchronous: `delete()` and `close()` only hand the step numbers/stop signal to
# Orbax's daemon deleter thread and return immediately. Callers therefore still see
# the directory on disk (and can race a concurrent removal) unless jimm waits, which
# is what these helpers do.
_DELETION_POLL_INTERVAL = 0.01
_DELETION_WAIT_TIMEOUT = 300.0


def _background_deleter(manager: ocp.CheckpointManager) -> Any:
    """Return the manager's checkpoint deleter, if this Orbax version exposes one."""
    return getattr(manager, "_checkpoint_deleter", None)


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
            key = (
                int(k)
                if isinstance(k, str) and (k.isdigit() or (k.startswith("-") and k[1:].isdigit()))
                else k
            )
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
                f"{path}: checkpoint has {type(restored).__name__}, expected a dict subtree"
            )
        missing = sorted(set(expected) - set(restored), key=repr)
        unexpected = sorted(set(restored) - set(expected), key=repr)
        if missing or unexpected:
            raise ValueError(
                f"{path}: checkpoint tree does not match model state "
                f"(missing keys: {missing}, unexpected keys: {unexpected})"
            )
        return {
            key: _validate_tree(expected[key], restored[key], f"{path}.{key}") for key in expected
        }
    if hasattr(expected, "shape"):
        value = jnp.asarray(restored)
        if tuple(value.shape) != tuple(expected.shape):
            raise ValueError(
                f"{path}: shape mismatch (checkpoint {tuple(value.shape)} "
                f"vs model {tuple(expected.shape)})"
            )
        if value.dtype != expected.dtype:
            value = value.astype(expected.dtype)
        return value
    return restored


def _apply_restored(
    restored: Any, model: nnx.Module, optimizer: nnx.Optimizer | None, source: str
) -> int:
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

    Directory checkpoints may be written either by :func:`save_checkpoint` or as a
    step directory of a :class:`CheckpointManager` run; both layouts are accepted.

    Raises:
        ValueError: If the checkpoint is missing model state or its tree
            structure / shapes do not match the live model, or if ``path`` is a
            ``CheckpointManager`` root rather than a single checkpoint.
    """
    path = os.path.abspath(path)
    selection = _manager_step(path)
    if selection is not None:
        manager_directory, step = selection
        with CheckpointManager(manager_directory) as manager:
            return manager.restore(step, model, optimizer)
    if _is_manager_directory(path):
        raise ValueError(
            f"{path} is a CheckpointManager directory, not a checkpoint; pass one of its"
            f" step directories, e.g. {os.path.join(path, '<step>')}, or restore the latest"
            f" step with CheckpointManager({path}).restore_latest(model, optimizer)"
        )
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

    Restores go through this manager as well; ``load_checkpoint`` also accepts a
    step directory written here, which is useful for evaluation scripts that take a
    single checkpoint path.
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
            policy = ocp.checkpoint_managers.AnyPreservationPolicy(
                [
                    policy,
                    ocp.checkpoint_managers.BestN(
                        get_metric_fn=best_fn,
                        reverse=best_mode == "min",
                        n=1,
                        keep_checkpoints_without_metrics=False,
                    ),
                ]
            )
        options = ocp.CheckpointManagerOptions(
            preservation_policy=policy,
            enable_background_delete=True,
        )
        self._manager = ocp.CheckpointManager(self.directory, options=options)
        self._closed = False

    def save(
        self,
        step: int,
        model: nnx.Module,
        optimizer: nnx.Optimizer | None = None,
        extra: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        force: bool = True,
    ) -> bool:
        """Save model/optimizer state as step ``step``.

        All JAX processes must call this with the same step and metrics.

        Only the commit/finalize stage is asynchronous: Orbax writes the arrays on the
        calling thread and then finalizes in the background. Successive saves are
        therefore serialized by Orbax (each ``save`` waits for the previous one before
        writing), so there is no growing queue of pending writes to throttle here; use
        :meth:`wait_until_finished` (or :meth:`close`) to observe completion and errors.

        Args:
            metrics: Optional metric values (e.g. ``{"val_acc": 0.74}``) used by
                ``best_fn`` for best-checkpoint retention.
            force: If True (the default), overwrite an existing checkpoint for
                the same step; Orbax itself refuses to re-save a step.
        """
        step = int(step)
        if force and step in self._manager.all_steps():
            self._manager.delete(step)
            self._wait_for_step_deletion(step)
        return self._manager.save(
            step,
            args=ocp.args.StandardSave(_item(model, optimizer, step, extra)),
            metrics=metrics,
        )

    def _wait_for_step_deletion(self, step: int) -> None:
        """Block until a queued background deletion of ``step`` reaches the filesystem.

        With background deletion enabled, ``delete`` only queues the removal, so
        re-saving the same step right away races the new write against a pending
        ``rmtree`` of ``<directory>/<step>``: the concurrent removal can fail the save
        outright, and one that lands late deletes the checkpoint that was just written.
        Waiting for the directory to disappear first keeps overwrites ordered; only the
        (rare) ``force`` path pays the wait.
        """
        path = os.path.join(self.directory, str(step))
        deleter = _background_deleter(self._manager)
        thread = getattr(deleter, "_delete_thread", None)
        if thread is None:
            # Deletion is not backgrounded (or Orbax changed its private API), so
            # there is no queued removal to wait for.
            if os.path.exists(path):
                warnings.warn(
                    f"cannot confirm the background deletion of checkpoint step {step} in"
                    f" {self.directory}; overwriting it may race a pending deletion",
                    RuntimeWarning,
                    stacklevel=3,
                )
            return
        deadline = time.monotonic() + _DELETION_WAIT_TIMEOUT
        while os.path.exists(path):
            if not thread.is_alive():
                # The deleter died (or never started): nothing will remove the path.
                return
            if time.monotonic() > deadline:
                warnings.warn(
                    f"timed out waiting for the background deletion of checkpoint step"
                    f" {step} in {self.directory}; overwriting it may race the pending"
                    f" deletion",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return
            time.sleep(_DELETION_POLL_INTERVAL)

    def restore(self, step: int, model: nnx.Module, optimizer: nnx.Optimizer | None = None) -> int:
        """Restore step ``step`` into live model/optimizer; returns its epoch."""
        restored = self._manager.restore(int(step), args=ocp.args.StandardRestore())
        return _apply_restored(restored, model, optimizer, f"{self.directory}/{step}")

    def restore_latest(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer | None = None,
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

    def metrics(self, step: int) -> dict[str, Any] | None:
        """Return the metrics recorded with ``step`` (e.g. ``{"val_acc": ...}``)."""
        return self._manager.metrics(int(step))

    def wait_until_finished(self) -> None:
        self._manager.wait_until_finished()

    def close(self) -> None:
        """Wait for pending saves and deletions, surface errors, and release resources."""
        if self._closed:
            return
        self._closed = True
        self._manager.check_for_errors()
        self._manager.close()
        self._wait_for_background_deletions()

    def _wait_for_background_deletions(self) -> None:
        """Block until Orbax's deleter thread has drained its queue and exited.

        ``CheckpointManager.close()`` only queues the shutdown sentinel on the daemon
        deleter thread instead of joining it, so it can return while rotated-out
        checkpoints are still being removed from disk. Anything that touches the output
        directory right afterwards -- deleting a run directory in a test's teardown,
        launching another job -- then races the deleter and fails with
        ``FileNotFoundError``. Deletion failures are likewise only stored on the
        deleter, where they are lost with the daemon thread, so report them here.
        """
        deleter = _background_deleter(self._manager)
        thread = getattr(deleter, "_delete_thread", None)
        if thread is not None and thread is not threading.current_thread() and thread.is_alive():
            thread.join(timeout=_DELETION_WAIT_TIMEOUT)
            if thread.is_alive():
                warnings.warn(
                    f"background checkpoint deletion is still running after"
                    f" {_DELETION_WAIT_TIMEOUT:.0f}s in {self.directory}",
                    RuntimeWarning,
                    stacklevel=3,
                )
        error = getattr(deleter, "_exception", None)
        if error is not None:
            warnings.warn(
                f"background checkpoint deletion failed: {error!r}",
                RuntimeWarning,
                stacklevel=3,
            )

    def __enter__(self) -> "CheckpointManager":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
