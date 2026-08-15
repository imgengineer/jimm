"""Model registry, mirrors timm.models.registry."""
import fnmatch
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

__all__ = ["register_model", "create_model", "list_models", "list_modules", "model_entrypoint", "get_default_cfg", "is_model"]

_model_entrypoints = {}
_model_to_module = {}
_model_default_cfgs = {}
_module_to_models = defaultdict(set)


def register_model(fn=None, *, default_cfg=None):
    if fn is None:
        return lambda f: register_model(f, default_cfg=default_cfg)
    name = fn.__name__
    _model_entrypoints[name] = fn
    _model_to_module[name] = fn.__module__.split(".")[-1]
    _module_to_models[fn.__module__.split(".")[-1]].add(name)
    if default_cfg is not None:
        _model_default_cfgs[name] = default_cfg
    elif hasattr(fn, "default_cfg"):
        _model_default_cfgs[name] = fn.default_cfg
    return fn


def model_entrypoint(name):
    return _model_entrypoints[name]


def is_model(name):
    return name in _model_entrypoints


def get_default_cfg(name: str) -> dict:
    """Return the default configuration dictionary for a given model name."""
    if name in _model_default_cfgs:
        return dict(_model_default_cfgs[name])
    if name in _model_entrypoints:
        fn = _model_entrypoints[name]
        if hasattr(fn, "default_cfg"):
            _model_default_cfgs[name] = fn.default_cfg
            return dict(fn.default_cfg)
    return {}


def _cfg(**kwargs):
    d = dict(input_size=(3, 224, 224), crop_pct=0.875, interpolation="bilinear",
             mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), num_classes=1000)
    d.update(kwargs)
    return d


def list_models(filter: str | Sequence[str] = "", module: str = "", pretrained: bool = False,
                exclude_filters: str | Sequence[str] = "") -> list[str]:
    names = sorted(_model_entrypoints)
    if module:
        names = [n for n in names if _model_to_module[n] == module]
    if filter:
        pats = [filter] if isinstance(filter, str) else list(filter)
        names = [n for n in names if any(fnmatch.fnmatch(n, p) for p in pats)]
    if exclude_filters:
        pats = [exclude_filters] if isinstance(exclude_filters, str) else list(exclude_filters)
        names = [n for n in names if not any(fnmatch.fnmatch(n, p) for p in pats)]
    if pretrained:
        names = [n for n in names if _model_default_cfgs.get(n, {}).get("url")]
    return names


def list_modules():
    return sorted(_module_to_models)


def create_model(name: str, pretrained: bool | str | dict[str, Any] = False, features_only: bool = False,
                 out_indices: Sequence[int] | None = None, rngs=None, **kwargs):
    """Mirror of timm.create_model. Extra JAX arg: rngs (flax.nnx.Rngs)."""
    if not is_model(name):
        raise ValueError(f"Unknown model {name!r}. Available: {list_models()}")
    from flax import nnx
    if features_only:
        kwargs.setdefault("num_classes", 0)
    model = _model_entrypoints[name](rngs=rngs or nnx.Rngs(0), **kwargs)
    if not getattr(model, "default_cfg", None):
        model.default_cfg = get_default_cfg(name) or _cfg()
    else:
        _model_default_cfgs[name] = model.default_cfg

    if pretrained:
        import importlib
        weights = importlib.import_module("jimm.weights")
        if isinstance(pretrained, str):
            weights.load_pretrained(model, pretrained)
        elif isinstance(pretrained, dict):
            weights.load_state_dict(model, pretrained)
        elif isinstance(pretrained, bool) and pretrained:
            cfg = get_default_cfg(name)
            url = cfg.get("url") if isinstance(cfg, dict) else None
            if url:
                weights.load_pretrained(model, url)
            else:
                raise NotImplementedError(
                    f"No default pretrained weight URL for {name!r}; train from scratch, "
                    "restore an orbax checkpoint with jimm.checkpoint.load_checkpoint, "
                    "or pass pretrained='path/to/weights.npz'.")

    if features_only:
        import importlib
        features = importlib.import_module("jimm.features")
        return features.create_feature_extractor(model, out_indices=out_indices)
    return model
