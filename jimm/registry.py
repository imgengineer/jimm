"""Model registry, mirrors timm.models.registry."""
import fnmatch
from collections import defaultdict

__all__ = ["register_model", "create_model", "list_models", "list_modules", "model_entrypoint", "get_default_cfg", "is_model"]

_model_entrypoints = {}
_model_to_module = {}
_model_default_cfgs = {}
_module_to_models = defaultdict(set)


def register_model(fn):
    _model_entrypoints[fn.__name__] = fn
    _model_to_module[fn.__name__] = fn.__module__.split(".")[-1]
    _module_to_models[fn.__module__.split(".")[-1]].add(fn.__name__)
    if hasattr(fn, "default_cfg"):
        _model_default_cfgs[fn.__name__] = fn.default_cfg
    return fn


def model_entrypoint(name):
    return _model_entrypoints[name]


def is_model(name):
    return name in _model_entrypoints


def get_default_cfg(name):
    return _model_default_cfgs.get(name, {})


def _cfg(**kwargs):
    d = dict(input_size=(3, 224, 224), crop_pct=0.875, interpolation="bilinear",
             mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), num_classes=1000)
    d.update(kwargs)
    return d


def list_models(filter="", module="", pretrained=False, exclude_filters=""):
    names = sorted(_model_entrypoints)
    if module:
        names = [n for n in names if _model_to_module[n] == module]
    if filter:
        pats = filter if isinstance(filter, (list, tuple)) else [filter]
        names = [n for n in names if any(fnmatch.fnmatch(n, p) for p in pats)]
    if exclude_filters:
        pats = exclude_filters if isinstance(exclude_filters, (list, tuple)) else [exclude_filters]
        names = [n for n in names if not any(fnmatch.fnmatch(n, p) for p in pats)]
    if pretrained:
        names = [n for n in names if _model_default_cfgs.get(n, {}).get("url")]
    return names


def list_modules():
    return sorted(_module_to_models)


def create_model(name, pretrained=False, rngs=None, **kwargs):
    """Mirror of timm.create_model. Extra JAX arg: rngs (flax.nnx.Rngs)."""
    if pretrained:
        raise NotImplementedError(
            "pretrained torch weight porting is not implemented; train from scratch or "
            "restore a jimm orbax checkpoint with jimm.checkpoint.load_checkpoint.")
    if not is_model(name):
        raise ValueError(f"Unknown model {name!r}. Available: {list_models()}")
    from flax import nnx
    model = _model_entrypoints[name](rngs=rngs or nnx.Rngs(0), **kwargs)
    if not getattr(model, "default_cfg", None):  # factory-set cfg (e.g. 299/256 input) wins
        model.default_cfg = get_default_cfg(name)
    return model
