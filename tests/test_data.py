"""Unit tests for jimm.data."""
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

import cv2  # pyright: ignore[reportMissingImports]
import grain.python as grain
import jimm.augment as augment_module
import jimm.data as data_module
import numpy as np
import pytest

from jimm.data import (
    ImageFolder,
    MixupCutmix,
    _DecodeTransform,
    build_auto_augment,
    center_crop_or_pad,
    color_jitter,
    create_dataset,
    create_loader,
    gaussian_blur,
    random_erasing,
    random_crop_or_pad,
    random_flip_left_right,
    random_flip_up_down,
    random_grayscale,
    random_resized_crop,
)


@pytest.fixture
def temp_dataset():
    root = tempfile.mkdtemp()
    for split in ["train", "val"]:
        for cls in ["cat", "dog", "bird"]:
            os.makedirs(f"{root}/{split}/{cls}", exist_ok=True)
            for i in range(8):
                img = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
                cv2.imwrite(
                    f"{root}/{split}/{cls}/img_{i}.png",
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                )
    yield root
    shutil.rmtree(root, ignore_errors=True)


def test_image_folder(temp_dataset):
    ds = ImageFolder(f"{temp_dataset}/train")
    assert len(ds) == 24  # 3 classes * 8 images
    assert set(ds.class_to_idx.keys()) == {"cat", "dog", "bird"}

    sample = ds[0]
    assert "image" in sample
    assert "label" in sample
    assert isinstance(sample["image"], bytes)
    assert isinstance(sample["label"], int)

    # Invalid root with no class subdirs raises ValueError
    empty_dir = tempfile.mkdtemp()
    try:
        with pytest.raises(ValueError, match="no class subdirectories"):
            ImageFolder(empty_dir)
    finally:
        shutil.rmtree(empty_dir, ignore_errors=True)


def test_augmentations(monkeypatch):
    img = np.arange(48 * 48 * 3, dtype=np.uint8).reshape(48, 48, 3)
    cropped = random_resized_crop(img, size=16, scale=(1.0, 1.0), ratio=(1.0, 1.0))
    assert cropped.shape[:2] == (16, 16)

    # Invalid crop ranges use the resize fallback instead of producing an invalid crop.
    fallback = random_resized_crop(img, size=16, scale=(2.0, 2.0), ratio=(1.0, 1.0))
    assert fallback.shape[:2] == (16, 16)

    monkeypatch.setattr(np.random, "rand", lambda: 0.0)
    jittered = color_jitter(img, brightness=0.2, contrast=0.2, saturation=0.2)
    assert jittered.shape == img.shape

    array = np.ones((16, 16, 3), dtype=np.float32)
    erased = random_erasing(array, prob=1.0, sl=0.25, sh=0.25, r1=1.0)
    assert erased.shape == array.shape
    assert np.any(erased == 0.0)


def test_opencv_augmentations():
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    assert random_flip_left_right(img, prob=1.0).shape == img.shape
    assert random_flip_up_down(img, prob=1.0).shape == img.shape
    assert random_grayscale(img, prob=1.0).shape == img.shape
    assert gaussian_blur(img, prob=1.0, sigma=(0.5, 0.5)).shape == img.shape
    assert center_crop_or_pad(img, 16).shape[:2] == (16, 16)
    assert center_crop_or_pad(img[:8, :8], 16).shape[:2] == (16, 16)
    assert random_crop_or_pad(img[:8, :8], 16).shape[:2] == (16, 16)

    for config in ("v0", "original", "3a", "rand-m2-n1", "augmix-m2-w2-d1", "trivialaugment"):
        transform = build_auto_augment(config)
        assert transform is not None
        assert transform(img).shape == img.shape


def test_timm_augmentation_api():
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    assert len(data_module.auto_augment_policy("v0")) == 25
    assert len(data_module.auto_augment_policy("originalr")) == 25
    assert len(data_module.rand_augment_ops(transforms=["Invert"])) == 1
    assert data_module.rand_augment_choices("3a") == [
        "SolarizeIncreasing", "Desaturate", "GaussianBlur"
    ]
    assert data_module.str_to_interp_mode("bilinear") == cv2.INTER_LINEAR
    assert data_module.interp_mode_to_str(cv2.INTER_NEAREST) == "nearest"

    for config in (
        "rand-m9-n3-p1-mstd0.5-mmax12-inc1-t3aw",
        "augmix-m5-w4-d2-a0.5-b1-mstd0.5",
    ):
        transform = build_auto_augment(config)
        assert transform is not None
        assert transform(img).shape == img.shape


def test_augmentation_edge_cases():
    img = np.arange(32 * 40 * 3, dtype=np.uint8).reshape(32, 40, 3)

    for interpolation in (
        "nearest", None, ("bilinear", "bicubic"), cv2.INTER_AREA,
    ):
        assert data_module.resolve_interpolation(cast(Any, interpolation)) is not None
    with pytest.raises(ValueError):
        data_module.resolve_interpolation("unknown")
    with pytest.raises(ValueError):
        data_module.interp_mode_to_str(-1)
    with pytest.raises(ValueError):
        augment_module._size((1, 2, 3))
    with pytest.raises(ValueError):
        augment_module._as_int("bad")
    with pytest.raises(ValueError):
        augment_module._as_float("bad")
    with pytest.raises(ValueError):
        augment_module._rgb(np.zeros((2, 2), dtype=np.uint8)[..., None])
    assert augment_module._rgb(np.zeros((2, 2), dtype=np.uint8)).shape == (2, 2, 3)
    assert augment_module._rgb(np.zeros((2, 2, 4), dtype=np.uint8)).shape == (2, 2, 3)

    assert data_module.resize_keep_ratio(img, size=16).ndim == 3
    assert augment_module.random_resized_crop(
        np.zeros((8, 64, 3), dtype=np.uint8), size=16, scale=(2.0, 2.0),
    ).shape == (16, 16, 3)
    assert augment_module.random_resized_crop(
        np.zeros((64, 8, 3), dtype=np.uint8), size=16, scale=(2.0, 2.0),
    ).shape == (16, 16, 3)
    assert augment_module.random_crop_or_pad(
        img, cast(Any, (16, 20))).shape == (16, 20, 3)
    assert data_module.color_jitter(
        img, brightness=cast(Any, (0.1, 0.2)),
        contrast=cast(Any, (0.8, 1.2)), saturation=cast(Any, (0.8, 1.2)),
        hue=cast(Any, (-0.1, 0.1)), random_order=False,
    ).shape == img.shape
    assert augment_module._range(None, "test") == (0.0, 0.0)
    assert augment_module._hue_range(None) == (0.0, 0.0)
    with pytest.raises(ValueError):
        augment_module._range((1.0,), "test")
    with pytest.raises(ValueError):
        augment_module._hue_range((1.0,))
    assert data_module.color_jitter(img, prob=0.0).shape == img.shape
    assert data_module.random_flip_left_right(img, prob=0.0).shape == img.shape
    assert data_module.random_flip_up_down(img, prob=0.0).shape == img.shape
    assert data_module.random_grayscale(img, prob=0.0).shape == img.shape
    assert data_module.gaussian_blur(img, prob=1.0, sigma=(1.0, 1.0)).shape == img.shape

    for name in (
        "AutoContrast", "Equalize", "Invert", "Solarize", "SolarizeIncreasing",
        "SolarizeAdd", "Color", "ColorIncreasing", "Contrast", "ContrastIncreasing",
        "Brightness", "BrightnessIncreasing", "Sharpness", "SharpnessIncreasing",
        "Desaturate", "GaussianBlur", "GaussianBlurRand", "Rotate",
        "Posterize", "PosterizeOriginal", "PosterizeIncreasing", "ShearX", "ShearY",
        "TranslateX", "TranslateY", "TranslateXRel", "TranslateYRel",
    ):
        result = augment_module.AugmentOp(
            name, prob=1.0, magnitude=5,
            hparams={"translate_const": 8, "translate_pct": 0.2},
        )(img)
        assert result.shape == img.shape
    assert augment_module.AugmentOp("Invert", prob=0.0)(img) is img
    assert augment_module.AugmentOp(
        "Invert", prob=1.0, hparams={"magnitude_std": float("inf")})(img).shape == img.shape
    assert augment_module.AugmentOp(
        "Invert", prob=1.0, hparams={"magnitude_std": 1.0})(img).shape == img.shape
    assert "AugmentOp" in repr(augment_module.AugmentOp("Invert"))
    assert augment_module.AutoAugment([[('Invert', 1.0, 1.0)]])(img).shape == img.shape
    with pytest.raises(ValueError):
        augment_module.AugmentOp("missing", prob=1.0)(img)

    with pytest.raises(ValueError):
        augment_module.auto_augment_policy("missing")
    assert augment_module.auto_augment_transform("v0-mstd0.5")(img).shape == img.shape
    for policy in ("v0r", "original", "originalr", "3a"):
        assert data_module.auto_augment_policy(policy)
    with pytest.raises(ValueError):
        augment_module.auto_augment_transform("v0-unknown1")
    with pytest.raises(ValueError):
        augment_module.rand_augment_transform("rand-unknown1")
    with pytest.raises(ValueError):
        augment_module.augment_and_mix_transform("augmix-unknown1")
    assert build_auto_augment("none") is None
    assert data_module.rand_augment_choices("weights")
    assert data_module.rand_augment_choices("3aw")
    assert data_module.rand_augment_ops(transforms={"Invert": 1})
    assert data_module.augmix_ops(transforms={"Invert": 1})
    assert repr(augment_module.RandAugment([], 0))
    assert repr(augment_module.AugMixAugment([], blended=True))
    assert augment_module.AugMixAugment(
        data_module.augmix_ops(transforms=["Invert"]), width=1, depth=1)(img).shape == img.shape
    assert augment_module.RandAugment(
        data_module.rand_augment_ops(transforms=["Invert"]), 1, [1.0])(img).shape == img.shape
    assert augment_module.RandAugment([], 0)(img) is img


def test_mixup_cutmix(monkeypatch):
    images = np.zeros((2, 8, 8, 3), dtype=np.float32)
    images[1] = 1.0
    labels = np.array([0, 1], dtype=np.int32)
    monkeypatch.setattr(np.random, "beta", lambda *_: 0.5)
    monkeypatch.setattr(np.random, "permutation", lambda _: np.array([1, 0]))

    random_values = iter([0.0, 0.0])
    monkeypatch.setattr(np.random, "rand", lambda: next(random_values))
    mixed, mixed_labels = MixupCutmix(
        mixup_alpha=1.0, cutmix_alpha=0.0, num_classes=2)(images, labels)
    assert mixed.shape == images.shape
    assert mixed_labels.shape == (2, 2)
    assert np.allclose(mixed_labels.sum(axis=1), 1.0)
    assert np.any((mixed > 0.0) & (mixed < 1.0))

    random_values = iter([0.0, 1.0])
    monkeypatch.setattr(np.random, "rand", lambda: next(random_values))
    cutmixed, cutmix_labels = MixupCutmix(
        mixup_alpha=1.0, cutmix_alpha=1.0, num_classes=2)(images, labels)
    assert cutmixed.shape == images.shape
    assert cutmix_labels.shape == (2, 2)

    random_values = iter([1.0])
    monkeypatch.setattr(np.random, "rand", lambda: next(random_values))
    unchanged, unchanged_labels = MixupCutmix(prob=0.0, num_classes=2)(images, labels)
    assert unchanged is images
    assert unchanged_labels is labels


def test_mixup_modes_and_edges():
    images = np.zeros((4, 8, 8, 3), dtype=np.float32)
    labels = np.arange(4, dtype=np.int32)
    with pytest.raises(ValueError, match="mode"):
        MixupCutmix(mode="invalid")

    for mode in ("pair", "elem"):
        mixed, targets = MixupCutmix(
            mixup_alpha=1.0, cutmix_alpha=0.0, mode=mode,
            label_smoothing=0.1, num_classes=4,
        )(images, labels)
        assert mixed.shape == images.shape
        assert targets.shape == (4, 4)

    for mode in ("batch", "elem"):
        mixed, targets = MixupCutmix(
            mixup_alpha=0.0, cutmix_alpha=1.0, mode=mode,
            num_classes=4, cutmix_minmax=(0.25, 0.5),
        )(images, labels)
        assert mixed.shape == images.shape
        assert targets.shape == (4, 4)


def test_data_error_paths(temp_dataset, monkeypatch):
    root = Path(temp_dataset) / "train"
    img = np.full((16, 16, 3), 255, dtype=np.uint8)

    assert data_module._is_within(root, root / "cat")
    assert not data_module._is_within(root, root.parent)

    def fail(*_args):
        raise ValueError("synthetic failure")

    with monkeypatch.context() as mp:
        mp.setattr(augment_module.math, "sqrt", fail)
        assert data_module.random_resized_crop(img, size=8).shape[:2] == (8, 8)
        erased = data_module.random_erasing(np.ones((8, 8, 3), dtype=np.float32), prob=1.0)
        assert np.array_equal(erased, np.ones((8, 8, 3), dtype=np.float32))

    monkeypatch.setattr(np.random, "rand", lambda: 0.0)
    assert data_module.color_jitter(img).shape == img.shape

    with pytest.raises(ValueError, match="no class subdirectories"):
        ImageFolder(Path(temp_dataset) / "missing")
    assert _DecodeTransform(img_size=8, crop_pct=0).resize == 256

    original_iterdir = Path.iterdir
    with monkeypatch.context() as mp:
        mp.setattr(Path, "iterdir", lambda path: (
            (_ for _ in ()).throw(OSError("synthetic failure"))
            if path.name == "cat" else original_iterdir(path)
        ))
        assert len(ImageFolder(root)) == 16

    with monkeypatch.context() as mp:
        mp.setattr(data_module, "_decode_image", fail)
        mp.setattr(data_module, "_read_image", fail)
        with pytest.raises(ValueError, match="unable to cache image"):
            ImageFolder(root, in_memory=True)

    no_resize = ImageFolder(root, in_memory=True, img_size=0)
    assert no_resize[0]["image"].shape == (48, 48, 3)
    with pytest.raises(ValueError, match="worker_buffer_size"):
        data_module.create_loader(root, batch_size=1, worker_buffer_size=0)

    ds = ImageFolder(root)
    with monkeypatch.context() as mp:
        mp.setattr(Path, "open", lambda *_args, **_kwargs: (
            (_ for _ in ()).throw(OSError("synthetic failure"))
        ))
        with pytest.raises(OSError, match="unable to read image"):
            ds[0]

    images = np.zeros((2, 8, 8, 3), dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int32)
    monkeypatch.setattr(np.random, "beta", lambda *_: 0.5)
    monkeypatch.setattr(np.random, "permutation", lambda _: np.array([1, 0]))

    random_values = iter([0.0, 0.0])
    monkeypatch.setattr(np.random, "rand", lambda: next(random_values))
    assert MixupCutmix(
        mixup_alpha=1.0, cutmix_alpha=0.0, num_classes=2)(images, labels)[0].shape == images.shape

    random_values = iter([0.0, 1.0])
    monkeypatch.setattr(np.random, "rand", lambda: next(random_values))
    assert MixupCutmix(
        mixup_alpha=1.0, cutmix_alpha=1.0, num_classes=2)(images, labels)[0].shape == images.shape

    random_values = iter([0.0])
    monkeypatch.setattr(np.random, "rand", lambda: next(random_values))
    unchanged, unchanged_labels = MixupCutmix(
        mixup_alpha=0.0, cutmix_alpha=0.0, num_classes=2)(images, labels)
    assert unchanged is images
    assert unchanged_labels is labels


def test_decode_transform():
    with pytest.raises(ValueError, match="unable to decode"):
        data_module._decode_image(b"not an image")

    # 1. Create a test sample
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(
        ".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    assert ok
    sample = {"image": encoded.tobytes(), "label": 2}

    # 2. Eval transform (center crop)
    t_eval = _DecodeTransform(img_size=32, is_training=False)
    out_eval = t_eval.map(sample)
    assert list(out_eval["image"].shape) == [32, 32, 3]
    assert out_eval["image"].dtype == np.float32
    assert out_eval["label"] == 2

    # Grain's per-record RNG makes multi-worker augmentation reproducible.
    rng_transform = _DecodeTransform(img_size=32, is_training=True, re_prob=0.0)
    out_rng_a = rng_transform.random_map(sample, np.random.default_rng(123))
    out_rng_b = rng_transform.random_map(sample, np.random.default_rng(123))
    assert np.array_equal(out_rng_a["image"], out_rng_b["image"])

    # 3. Train transform (random crop & flip)
    t_train = _DecodeTransform(img_size=32, is_training=True)
    out_train = t_train.map(sample)
    assert list(out_train["image"].shape) == [32, 32, 3]
    assert out_train["image"].dtype == np.float32

    # 4. Array inputs and disabled optional augmentations.
    t_plain = _DecodeTransform(
        img_size=32, is_training=True, hflip=0.0,
        color_jitter_prob=0.0, re_prob=0.0,
    )
    t_rkrc = _DecodeTransform(img_size=32, is_training=True, train_crop_mode="rkrc")
    t_rkrr = _DecodeTransform(img_size=32, is_training=True, train_crop_mode="rkrr")
    assert t_rkrc.map(sample)["image"].shape == (32, 32, 3)
    assert t_rkrr.map(sample)["image"].shape == (32, 32, 3)
    t_aug = _DecodeTransform(
        img_size=32, is_training=True, auto_augment="3a", color_jitter=cast(Any, None),
        re_prob=0.0,
    )
    assert t_aug.map(sample)["image"].shape == (32, 32, 3)
    t_force = _DecodeTransform(
        img_size=32, is_training=True, auto_augment="3a",
        force_color_jitter=True,
        color_jitter=cast(Any, (0.1, 0.1, 0.1, 0.1)), re_prob=0.0,
    )
    assert t_force.map(sample)["image"].shape == (32, 32, 3)
    with pytest.raises(ValueError, match="color_jitter"):
        _DecodeTransform(
            img_size=32, is_training=True,
            color_jitter=cast(Any, (0.1, 0.1)),
        ).map(sample)
    with pytest.raises(ValueError, match="unknown train_crop_mode"):
        _DecodeTransform(img_size=32, is_training=True, train_crop_mode="bad").map(sample)

    out_array = t_plain.map({"image": np.asarray(img), "label": 1})
    out_array2 = t_plain.map({"image": img, "label": 1})
    gray = np.zeros((64, 64), dtype=np.uint8)
    rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    assert t_plain.map({"image": gray, "label": 1})["image"].shape == (32, 32, 3)
    assert t_plain.map({"image": rgba, "label": 1})["image"].shape == (32, 32, 3)
    assert out_array["image"].shape == out_array2["image"].shape == (32, 32, 3)


def test_create_dataset_and_loader(temp_dataset):
    # 1. create_dataset
    ds, tf = create_dataset(f"{temp_dataset}/train", img_size=32, is_training=True)
    assert len(ds) == 24
    assert isinstance(tf, _DecodeTransform)

    # 2. in-memory source and training loader (infinite stream)
    memory_ds = ImageFolder(f"{temp_dataset}/train", in_memory=True, img_size=16)
    assert len(memory_ds) == 24
    assert memory_ds[0]["image"].shape == (16, 16, 3)

    train_loader = create_loader(
        f"{temp_dataset}/train",
        batch_size=4,
        img_size=32,
        is_training=True,
        num_workers=0,
        seed=42,
    )
    assert len(train_loader) == 6  # 24 // 4 = 6 batches per epoch
    it = iter(train_loader)
    b1 = next(it)
    b2 = next(it)
    assert list(b1["image"].shape) == [4, 32, 32, 3]
    assert list(b1["label"].shape) == [4]
    assert list(b2["image"].shape) == [4, 32, 32, 3]

    memory_loader = create_loader(
        f"{temp_dataset}/train", batch_size=4, img_size=32,
        is_training=True, num_workers=0, in_memory=True,
    )
    assert next(iter(memory_loader))["image"].shape == (4, 32, 32, 3)

    # 3. create_loader (eval mode: finite single epoch)
    val_loader = create_loader(
        f"{temp_dataset}/val",
        batch_size=5,
        img_size=32,
        is_training=False,
        num_workers=0,
    )
    val_batches = list(val_loader)
    # 24 samples / batch_size 5 = 4 full + 1 remainder (drop_remainder=False) = 5 batches
    assert len(val_batches) == len(val_loader) == 5
    assert sum(len(b["label"]) for b in val_batches) == 24

    # 4. Multi-host shard options
    shard_opt = grain.ShardOptions(shard_index=1, shard_count=2, drop_remainder=True)
    sharded_loader = create_loader(
        f"{temp_dataset}/train",
        batch_size=4,
        img_size=32,
        is_training=True,
        num_workers=0,
        shard_options=shard_opt,
    )
    assert len(sharded_loader) == 3  # (24 // 2) // 4 = 3 batches on this shard
    train_loader.close()
    memory_loader.close()
    val_loader.close()
    sharded_loader.close()


def test_multiworker_loader_parses_grain_flags(temp_dataset):
    loader = create_loader(
        f"{temp_dataset}/train",
        batch_size=2,
        img_size=16,
        is_training=True,
        num_workers=1,
        worker_buffer_size=1,
        seed=0,
    )
    try:
        assert next(iter(loader))["image"].shape == (2, 16, 16, 3)
    finally:
        loader.close()
