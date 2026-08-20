"""orbax-checkpoint save/restore for nnx models (+ optional optimizer state)."""
import errno
import os

import orbax.checkpoint as ocp
from flax import nnx

__all__ = ["save_checkpoint", "load_checkpoint", "wait_for_checkpoints"]

_checkpointer = None  # module-level singleton: avoids GC before async write finishes


def _get_checkpointer():
    """Lazily create the checkpointer on first use.

    Instantiating ocp.StandardCheckpointer() initializes the JAX backend and
    allocates a CUDA context (~500 MiB per process). Doing that at import time
    forces every Grain data-loader worker (which re-imports jimm under spawn)
    to grab GPU memory it never uses.
    """
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = ocp.StandardCheckpointer()
    return _checkpointer


def _item(model, optimizer, epoch, extra):
    item = {"epoch": epoch, "model": nnx.state(model).to_pure_dict()}
    if optimizer is not None:
        item["optimizer"] = nnx.state(optimizer).to_pure_dict()
    if extra:
        item["extra"] = extra
    return item


def save_checkpoint(path, model, optimizer=None, epoch=0, extra=None, wait=True):
    """Save model (and optimizer) state to `path`; optionally return immediately."""
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


def wait_for_checkpoints():
    """Block until all asynchronous checkpoint writes finish and surface errors."""
    if _checkpointer is not None:
        _checkpointer.wait_until_finished()


def _fix_int_keys(d):
    """Recursively converts stringified integer keys ('0', '1') into int keys for nnx.List/nnx.Sequential."""
    if isinstance(d, dict):
        new_d = {}
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


def load_checkpoint(path, model, optimizer=None):
    """Restore state saved by save_checkpoint into live model/optimizer. Returns epoch."""
    path = os.path.abspath(path)
    restored = _get_checkpointer().restore(path)
    nnx.update(model, _fix_int_keys(restored["model"]))
    if optimizer is not None and "optimizer" in restored:
        nnx.update(optimizer, _fix_int_keys(restored["optimizer"]))
    return restored.get("epoch", 0)
