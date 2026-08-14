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


def load_checkpoint(path, model, optimizer=None):
    """Restore state saved by save_checkpoint into live model/optimizer. Returns epoch."""
    item = _item(model, optimizer, 0, None)  # live structure as restore target
    item.pop("extra", None)
    restored = _checkpointer.restore(path, item)
    nnx.replace_by_pure_dict(nnx.state(model), restored["model"])
    if optimizer is not None and "optimizer" in restored:
        nnx.replace_by_pure_dict(nnx.state(optimizer), restored["optimizer"])
    return restored["epoch"]
