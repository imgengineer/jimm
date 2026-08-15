"""orbax-checkpoint save/restore for nnx models (+ optional optimizer state)."""
import os

import orbax.checkpoint as ocp
from flax import nnx

__all__ = ["save_checkpoint", "load_checkpoint"]

_checkpointer = ocp.StandardCheckpointer()  # module-level: avoids GC before async write finishes


def _item(model, optimizer, epoch, extra):
    item = {"epoch": epoch, "model": nnx.state(model).to_pure_dict()}
    if optimizer is not None:
        item["optimizer"] = nnx.state(optimizer).to_pure_dict()
    if extra:
        item["extra"] = extra
    return item


def save_checkpoint(path, model, optimizer=None, epoch=0, extra=None):
    """Save model (and optimizer) state to `path`. Returns path."""
    os.makedirs(path, exist_ok=True)
    _checkpointer.save(path, _item(model, optimizer, epoch, extra), force=True)
    _checkpointer.wait_until_finished()
    return path


def _fix_int_keys(d):
    """Recursively converts stringified integer keys ('0', '1') into int keys for nnx.List/nnx.Sequential."""
    if isinstance(d, dict):
        new_d = {}
        for k, v in d.items():
            if isinstance(k, str) and (k.isdigit() or (k.startswith("-") and k[1:].isdigit())):
                try:
                    k = int(k)
                except ValueError:
                    pass
            new_d[k] = _fix_int_keys(v)
        return new_d
    return d


def load_checkpoint(path, model, optimizer=None):
    """Restore state saved by save_checkpoint into live model/optimizer. Returns epoch."""
    restored = _checkpointer.restore(path)
    nnx.update(model, _fix_int_keys(restored["model"]))
    if optimizer is not None and "optimizer" in restored:
        nnx.update(optimizer, _fix_int_keys(restored["optimizer"]))
    return restored.get("epoch", 0)
