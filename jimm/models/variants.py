"""Explicit model variants supported by the local architecture implementations.

Only register configurations that match their named architecture. Unsupported
families, stems, attention mechanisms, and training variants must not fall back
to a generic ResNet or ViT. Add new entries with architecture-level tests.
"""

from ..registry import _cfg, is_model, model_entrypoint, register_model
from . import swin_transformer


def _make(name, ctor, args, fixed, input_size):
    def entry(**kwargs):
        model = ctor(*args, **fixed, **kwargs)
        model.default_cfg = _cfg(input_size=input_size)
        return model

    entry.__name__ = name
    entry.__module__ = ctor.__module__
    return entry


_SPECS = [
    (
        ("swin_base_patch4_window12_384",),
        swin_transformer.SwinTransformer,
        (),
        {
            "img_size": 384,
            "embed_dim": 128,
            "depths": (2, 2, 18, 2),
            "num_heads": (4, 8, 16, 32),
            "window_size": 12,
        },
        (3, 384, 384),
    ),
    (
        ("swin_large_patch4_window12_384",),
        swin_transformer.SwinTransformer,
        (),
        {
            "img_size": 384,
            "embed_dim": 192,
            "depths": (2, 2, 18, 2),
            "num_heads": (6, 12, 24, 48),
            "window_size": 12,
        },
        (3, 384, 384),
    ),
    (
        ("swin_large_patch4_window7_224",),
        swin_transformer.SwinTransformer,
        (),
        {"img_size": 224, "embed_dim": 192, "depths": (2, 2, 18, 2), "num_heads": (6, 12, 24, 48)},
        (3, 224, 224),
    ),
]

for _names, _ctor, _args, _fixed, _input_size in _SPECS:
    for _name in _names:
        globals()[_name] = (
            model_entrypoint(_name)
            if is_model(_name)
            else register_model(_make(_name, _ctor, _args, _fixed, _input_size))
        )
