"""timm.models.efficientnet names, built by :mod:`jimm.models._efficientnet` from recorded configs."""

from ._efficientnet import register_module

register_module("efficientnet")


from ..registry import _cfg, model_entrypoint, register_model  # noqa: E402
from ._pruned import PrunedEfficientNet  # noqa: E402


def _pruned_efficientnet(variant, input_size, crop_pct, **kwargs):
    model = PrunedEfficientNet(variant, **kwargs)
    model.default_cfg = _cfg(
        input_size=input_size,
        crop_pct=crop_pct,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )
    return model


@register_model
def efficientnet_b1_pruned(**kwargs):
    return _pruned_efficientnet("efficientnet_b1_pruned", (3, 240, 240), 0.882, **kwargs)


@register_model
def efficientnet_b2_pruned(**kwargs):
    return _pruned_efficientnet("efficientnet_b2_pruned", (3, 260, 260), 0.890, **kwargs)


@register_model
def efficientnet_b3_pruned(**kwargs):
    return _pruned_efficientnet("efficientnet_b3_pruned", (3, 300, 300), 0.904, **kwargs)


@register_model
def efficientnet_es_pruned(**kwargs):
    model = model_entrypoint("efficientnet_es")(**kwargs)
    model.default_cfg = _cfg(interpolation="bicubic")
    return model


@register_model
def efficientnet_el_pruned(**kwargs):
    model = model_entrypoint("efficientnet_el")(**kwargs)
    model.default_cfg = _cfg(input_size=(3, 300, 300), crop_pct=0.904, interpolation="bicubic")
    return model
