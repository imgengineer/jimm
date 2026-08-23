"""Model registry and creation factory, mirroring timm.models.registry.

Provides dynamic model registration, fuzzy query/filtering across 1,344 architectures,
and instantiation with Flax NNX lifecycle & RNG management.
"""
from collections import defaultdict
from collections.abc import Callable, Sequence
import fnmatch
from typing import Any

from flax import nnx

__all__ = [
    "register_model",
    "create_model",
    "list_models",
    "list_modules",
    "model_entrypoint",
    "get_default_cfg",
    "is_model",
]

_model_entrypoints: dict[str, Callable[..., nnx.Module]] = {}
_model_to_module: dict[str, str] = {}
_model_default_cfgs: dict[str, dict[str, Any]] = {}
_module_to_models: dict[str, set[str]] = defaultdict(set)


def register_model(fn: Callable[..., nnx.Module] | None = None, *, default_cfg: dict[str, Any] | None = None):
    """Decorator to register a model architecture entrypoint into the jimm global registry.

    Args:
        fn: Entrypoint constructor function returning an nnx.Module instance.
        default_cfg: Optional default model configuration dictionary.
    """
    if fn is None:
        return lambda f: register_model(f, default_cfg=default_cfg)
    name = fn.__name__
    module = fn.__module__.split(".")[-1]
    previous_module = _model_to_module.get(name)
    if previous_module is not None and previous_module != module:
        _module_to_models[previous_module].discard(name)
    _model_entrypoints[name] = fn
    _model_to_module[name] = module
    _module_to_models[module].add(name)
    if default_cfg is not None:
        _model_default_cfgs[name] = default_cfg
    elif hasattr(fn, "default_cfg"):
        _model_default_cfgs[name] = getattr(fn, "default_cfg")
    return fn


def model_entrypoint(name: str) -> Callable[..., nnx.Module]:
    """Retrieve the constructor function for a registered model name."""
    if name not in _model_entrypoints:
        raise ValueError(f"Unknown model {name!r}. Available: {list_models()}")
    return _model_entrypoints[name]


def is_model(name: str) -> bool:
    """Check whether a model architecture name is registered."""
    return name in _model_entrypoints


def get_default_cfg(name: str) -> dict[str, Any]:
    """Return the default configuration dictionary (input size, normalization, etc.) for a model."""
    if name in _model_default_cfgs:
        return dict(_model_default_cfgs[name])
    if name in _model_entrypoints:
        fn = _model_entrypoints[name]
        if hasattr(fn, "default_cfg"):
            _model_default_cfgs[name] = getattr(fn, "default_cfg")
            return dict(_model_default_cfgs[name])
    return {}


def _cfg(**kwargs: Any) -> dict[str, Any]:
    """Helper to build a standard default_cfg dictionary with timm-aligned defaults."""
    d = dict(
        input_size=(3, 224, 224),
        crop_pct=0.875,
        interpolation="bilinear",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        num_classes=1000,
    )
    d.update(kwargs)
    return d


def list_models(
    filter: str | Sequence[str] = "",
    module: str = "",
    pretrained: bool = False,
    exclude_filters: str | Sequence[str] = "",
) -> list[str]:
    """List registered model architecture names matching search patterns.

    Args:
        filter: Wildcard glob pattern (e.g. 'resnet*', ['*convnext*', '*eva*']).
        module: Filter to models defined in a specific module family (e.g. 'resnet').
        pretrained: If True, only return models with registered pretrained weights.
        exclude_filters: Glob pattern to exclude from the result list.

    Returns:
        Sorted list of matching model architecture names.
    """
    names = sorted(_model_entrypoints)
    if module:
        names = [n for n in names if _model_to_module.get(n) == module]
    if filter:
        pats = [filter] if isinstance(filter, str) else list(filter)
        names = [n for n in names if any(fnmatch.fnmatch(n, p) for p in pats)]
    if exclude_filters:
        pats = [exclude_filters] if isinstance(exclude_filters, str) else list(exclude_filters)
        names = [n for n in names if not any(fnmatch.fnmatch(n, p) for p in pats)]
    if pretrained:
        names = [n for n in names if _model_default_cfgs.get(n, {}).get("url")]
    return names


def list_modules() -> list[str]:
    """List all registered architecture module family names (e.g. 'resnet', 'convnext', 'vit')."""
    return sorted(module for module, names in _module_to_models.items() if names)


def create_model(
    name: str,
    pretrained: bool | str | dict[str, Any] = False,
    features_only: bool = False,
    out_indices: Sequence[int] | None = None,
    rngs: nnx.Rngs | None = None,
    **kwargs: Any,
) -> nnx.Module:
    """Instantiate a vision model architecture by name (mirrors `timm.create_model`).

    Args:
        name: Registered model architecture entrypoint (e.g. 'resnet50', 'convnext_tiny').
        pretrained: Load pretrained weights:
          - False: Random initialization.
          - True: Load weights from official default URL if available.
          - str: Path to local `.npz` / `.safetensors` weight file or custom URL.
          - dict: State dictionary of converted array weights.
        features_only: If True, returns a `FeatureExtractor` wrapper returning multi-scale feature maps.
        out_indices: Specific intermediate stage indices to extract when `features_only=True`.
        rngs: Flax NNX PRNG streams container (`nnx.Rngs(0)` by default).
        **kwargs: Additional architecture constructor arguments (num_classes, drop_rate, drop_path_rate, etc.).

    Returns:
        Flax NNX Module instance (or FeatureExtractor wrapper).
    """
    if not is_model(name):
        raise ValueError(f"Unknown model {name!r}. Available: {list_models()}")

    if features_only:
        kwargs.setdefault("num_classes", 0)

    model = _model_entrypoints[name](rngs=rngs or nnx.Rngs(0), **kwargs)

    # Attach or cache default configuration
    if not getattr(model, "default_cfg", None):
        model.default_cfg = get_default_cfg(name) or _cfg()
    else:
        _model_default_cfgs[name] = model.default_cfg

    # Load pretrained weights if requested
    if pretrained:
        from . import weights
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
                    "or pass pretrained='path/to/weights.npz'."
                )

    # Wrap as FeatureExtractor if features_only is requested
    if features_only:
        from . import features
        return features.create_feature_extractor(model, out_indices=out_indices)

    return model
