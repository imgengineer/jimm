"""Tests for PyTorch state dict conversion and weight loading into jimm models."""
import numpy as np
import pytest
from flax import nnx

from jimm.weights import load_pretrained, load_state_dict, _convert_key, _convert_tensor
from jimm.registry import create_model


def test_convert_key():
    assert _convert_key("layer1.0.conv1.weight") == ["stages", "0", "0", "conv1", "kernel"]
    assert _convert_key("layer2.0.downsample.0.weight") == ["stages", "1", "0", "shortcut", "conv", "kernel"]
    assert _convert_key("layer2.0.downsample.1.weight") == ["stages", "1", "0", "shortcut", "bn", "scale"]
    assert _convert_key("bn1.running_mean") == ["bn1", "mean"]
    assert _convert_key("bn1.running_var") == ["bn1", "var"]
    assert _convert_key("fc.weight") == ["fc", "kernel"]


def test_convert_tensor():
    conv_w = np.ones((64, 3, 7, 7), dtype=np.float32)
    conv_k = _convert_tensor("conv1.weight", conv_w)
    assert conv_k.shape == (7, 7, 3, 64)

    fc_w = np.ones((1000, 512), dtype=np.float32)
    fc_k = _convert_tensor("fc.weight", fc_w)
    assert fc_k.shape == (512, 1000)
    assert _convert_tensor("attn.qkv.weight", np.ones((6, 2))).shape == (2, 6)


def test_load_state_dict_resnet():
    m = create_model("resnet18", num_classes=1000, rngs=nnx.Rngs(0))
    state_dict = {
        "conv1.weight": np.ones((64, 3, 7, 7), dtype=np.float32) * 2.5,
        "bn1.weight": np.ones((64,), dtype=np.float32) * 1.5,
        "bn1.bias": np.ones((64,), dtype=np.float32) * 0.5,
        "fc.weight": np.ones((1000, 512), dtype=np.float32) * 0.1,
        "fc.bias": np.ones((1000,), dtype=np.float32) * 0.05,
    }
    loaded, missing = load_state_dict(m, state_dict)
    assert "conv1.weight" in loaded
    assert "bn1.weight" in loaded
    assert "fc.weight" in loaded
    assert float(m.conv1.kernel[...][0, 0, 0, 0]) == 2.5
    assert float(m.bn1.scale[...][0]) == 1.5
    assert float(m.bn1.bias[...][0]) == 0.5


def test_load_state_dict_strict_is_atomic_and_preserves_model_structure():
    m = create_model("resnet18", num_classes=1000, rngs=nnx.Rngs(0))
    before = np.asarray(m.bn1.scale[...]).copy()
    state_dict = {
        "bn1.weight": np.full((64,), 1.75, dtype=np.float16),
        "num_classes": np.array(7),
        "missing.weight": np.array([1.0]),
    }

    with pytest.raises(RuntimeError, match="strictly"):
        load_state_dict(m, state_dict, strict=True)
    np.testing.assert_array_equal(np.asarray(m.bn1.scale[...]), before)
    assert m.num_classes == 1000

    loaded, missing = load_state_dict(m, state_dict)
    assert loaded == ["bn1.weight"]
    assert missing == ["num_classes", "missing.weight"]
    assert m.bn1.scale[...].dtype == before.dtype
    np.testing.assert_allclose(np.asarray(m.bn1.scale[...]), 1.75)


def test_load_pretrained_npz_and_reject_unsupported_files(tmp_path):
    checkpoint = tmp_path / "weights.npz"
    np.savez(checkpoint, **{
        "conv1.weight": np.full((64, 3, 7, 7), 3.25, dtype=np.float32),
    })
    m = create_model(
        "resnet18", pretrained=str(checkpoint), num_classes=1000,
        rngs=nnx.Rngs(0))
    assert np.isclose(float(m.conv1.kernel[...][0, 0, 0, 0]), 3.25)

    unsupported = tmp_path / "weights.safetensors"
    unsupported.write_bytes(b"")
    with pytest.raises(ValueError, match="Unsupported checkpoint format"):
        load_pretrained(m, str(unsupported))
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_pretrained(m, str(tmp_path / "missing.npz"))


def test_create_model_pretrained_dict():
    state_dict = {
        "conv1.weight": np.ones((64, 3, 7, 7), dtype=np.float32) * 4.2,
    }
    m = create_model("resnet18", pretrained=state_dict, num_classes=1000, rngs=nnx.Rngs(0))
    assert np.isclose(float(m.conv1.kernel[...][0, 0, 0, 0]), 4.2, atol=1e-5)


def test_create_model_pretrained_unregistered_url_raises():
    with pytest.raises(NotImplementedError, match="No default pretrained weights"):
        create_model("resnet18", pretrained=True)
