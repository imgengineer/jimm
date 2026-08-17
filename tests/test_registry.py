"""Unit tests for jimm.registry."""
import pytest
from flax import nnx

import jimm
from jimm.registry import (
    create_model,
    get_default_cfg,
    is_model,
    list_models,
    list_modules,
    model_entrypoint,
    register_model,
)


def test_list_models_and_modules():
    models = list_models()
    assert len(models) >= 1309
    assert "resnet50" in models
    assert "convnext_tiny" in models
    assert "vit_base_patch16_224" in models

    modules = list_modules()
    assert modules
    assert all(list_models(module=module) for module in modules)
    assert "resnet" in modules
    assert "swin_transformer" in modules

    # Wildcard and pattern filters
    resnet_models = list_models(filter="resnet*")
    assert all("resnet" in m for m in resnet_models)
    assert len(resnet_models) >= 8

    # List filter
    multi_filter = list_models(filter=["resnet18", "convnext_tiny"])
    assert set(multi_filter) == {"resnet18", "convnext_tiny"}

    # Module filter
    vgg_models = list_models(module="vgg")
    assert {"vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"}.issubset(set(vgg_models))

    # Exclude filters
    no_resnet = list_models(filter="resnet*", exclude_filters="*152*")
    assert "resnet152" not in no_resnet
    assert "resnet50" in no_resnet

    no_resnet_list = list_models(filter="resnet*", exclude_filters=["*152*", "*101*"])
    assert "resnet101" not in no_resnet_list
    assert "resnet152" not in no_resnet_list

    # Pretrained filter
    pretrained = list_models(pretrained=True)
    assert isinstance(pretrained, list)


def test_model_entrypoint_and_is_model():
    assert is_model("resnet18")
    assert not is_model("non_existent_model_xyz")

    fn = model_entrypoint("resnet18")
    assert callable(fn)
    assert fn.__name__ == "resnet18"


def test_create_model():
    m = create_model("resnet18", num_classes=10)
    assert isinstance(m, nnx.Module)
    assert getattr(m, "num_classes") == 10
    assert "input_size" in getattr(m, "default_cfg")

    # create with explicit rngs
    m2 = create_model("resnet18", num_classes=5, rngs=nnx.Rngs(42))
    assert getattr(m2, "num_classes") == 5

    # unknown model error
    with pytest.raises(ValueError, match="Unknown model"):
        create_model("unknown_architecture_123")

    # pretrained=True raises NotImplementedError when no URL registered
    with pytest.raises(NotImplementedError, match="No default pretrained weight URL"):
        create_model("resnet18", pretrained=True)


def test_custom_register_model():
    @register_model
    def dummy_custom_model(num_classes=10, *, rngs=None, **kwargs):
        class DummyModel(nnx.Module):
            def __init__(self, num_classes):
                self.num_classes = num_classes
                self.default_cfg = {"input_size": (3, 64, 64)}
        return DummyModel(num_classes)

    assert is_model("dummy_custom_model")
    m = create_model("dummy_custom_model", num_classes=3)
    assert m.num_classes == 3
    assert get_default_cfg("dummy_custom_model") == {"input_size": (3, 64, 64)}
