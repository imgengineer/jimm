"""Unit tests for jimm.data."""
import io
import os
import shutil
import tempfile
from pathlib import Path

import grain.python as grain
import jimm.data as data_module
import numpy as np
import pytest
from PIL import Image

from jimm.data import (
    ImageFolder,
    MixupCutmix,
    _DecodeTransform,
    color_jitter,
    create_dataset,
    create_loader,
    random_erasing,
    random_resized_crop,
)


@pytest.fixture
def temp_dataset():
    root = tempfile.mkdtemp()
    for split in ["train", "val"]:
        for cls in ["cat", "dog", "bird"]:
            os.makedirs(f"{root}/{split}/{cls}", exist_ok=True)
            for i in range(8):
                img = Image.fromarray(np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8))
                img.save(f"{root}/{split}/{cls}/img_{i}.png")
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
    img = Image.fromarray(np.arange(48 * 48 * 3, dtype=np.uint8).reshape(48, 48, 3))
    cropped = random_resized_crop(img, size=16, scale=(1.0, 1.0), ratio=(1.0, 1.0))
    assert cropped.size == (16, 16)

    # Invalid crop ranges use the resize fallback instead of producing an invalid crop.
    fallback = random_resized_crop(img, size=16, scale=(2.0, 2.0), ratio=(1.0, 1.0))
    assert fallback.size == (16, 16)

    monkeypatch.setattr(np.random, "rand", lambda: 0.0)
    jittered = color_jitter(img, brightness=0.2, contrast=0.2, saturation=0.2)
    assert jittered.size == img.size

    array = np.ones((16, 16, 3), dtype=np.float32)
    erased = random_erasing(array, prob=1.0, sl=0.25, sh=0.25, r1=1.0)
    assert erased.shape == array.shape
    assert np.any(erased == 0.0)


def test_mixup_cutmix(monkeypatch):
    images = np.zeros((2, 8, 8, 3), dtype=np.float32)
    images[1] = 1.0
    labels = np.array([0, 1], dtype=np.int32)
    monkeypatch.setattr(np.random, "beta", lambda *_: 0.5)
    monkeypatch.setattr(np.random, "permutation", lambda _: np.array([1, 0]))

    random_values = iter([0.0, 0.0])
    monkeypatch.setattr(np.random, "rand", lambda: next(random_values))
    mixed, mixed_labels = MixupCutmix(mixup_alpha=1.0, cutmix_alpha=0.0)(images, labels)
    assert mixed.shape == images.shape
    assert mixed_labels is labels
    assert np.any((mixed > 0.0) & (mixed < 1.0))

    random_values = iter([0.0, 1.0])
    monkeypatch.setattr(np.random, "rand", lambda: next(random_values))
    cutmixed, cutmix_labels = MixupCutmix(mixup_alpha=1.0, cutmix_alpha=1.0)(images, labels)
    assert cutmixed.shape == images.shape
    assert cutmix_labels is labels

    random_values = iter([1.0])
    monkeypatch.setattr(np.random, "rand", lambda: next(random_values))
    unchanged, unchanged_labels = MixupCutmix(prob=0.0)(images, labels)
    assert unchanged is images
    assert unchanged_labels is labels


def test_data_error_paths(temp_dataset, monkeypatch):
    root = Path(temp_dataset) / "train"
    img = Image.new("RGB", (16, 16), color="white")

    assert data_module._is_within(root, root / "cat")
    assert not data_module._is_within(root, root.parent)

    def fail(*_args):
        raise ValueError("synthetic failure")

    with monkeypatch.context() as mp:
        mp.setattr(data_module.math, "sqrt", fail)
        assert data_module.random_resized_crop(img, size=8).size == (8, 8)
        erased = data_module.random_erasing(np.ones((8, 8, 3), dtype=np.float32), prob=1.0)
        assert np.array_equal(erased, np.ones((8, 8, 3), dtype=np.float32))

    monkeypatch.setattr(np.random, "rand", lambda: 0.0)
    monkeypatch.setattr(np.random, "uniform", fail)
    assert data_module.color_jitter(img).size == img.size

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
        assert len(ImageFolder(root, in_memory=True)) == 0

    no_resize = ImageFolder(root, in_memory=True, img_size=0)
    assert no_resize[0]["image"].shape == (48, 48, 3)

    ds = ImageFolder(root)
    with monkeypatch.context() as mp:
        mp.setattr(Path, "open", lambda *_args, **_kwargs: (
            (_ for _ in ()).throw(OSError("synthetic failure"))
        ))
        assert ds[0]["image"] == b""

    images = np.zeros((2, 8, 8, 3), dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int32)
    monkeypatch.setattr(np.random, "beta", fail)
    monkeypatch.setattr(np.random, "permutation", lambda _: np.array([1, 0]))

    random_values = iter([0.0, 0.0])
    monkeypatch.setattr(np.random, "rand", lambda: next(random_values))
    assert MixupCutmix(mixup_alpha=1.0, cutmix_alpha=0.0)(images, labels)[0].shape == images.shape

    random_values = iter([0.0, 1.0])
    monkeypatch.setattr(np.random, "rand", lambda: next(random_values))
    assert MixupCutmix(mixup_alpha=1.0, cutmix_alpha=1.0)(images, labels)[0].shape == images.shape

    with monkeypatch.context() as mp:
        mp.setattr("builtins.round", fail)
        random_values = iter([0.0, 1.0])
        mp.setattr(np.random, "rand", lambda: next(random_values))
        assert MixupCutmix(mixup_alpha=1.0, cutmix_alpha=1.0)(images, labels)[0].shape == images.shape

    random_values = iter([0.0])
    monkeypatch.setattr(np.random, "rand", lambda: next(random_values))
    unchanged, unchanged_labels = MixupCutmix(mixup_alpha=0.0, cutmix_alpha=0.0)(images, labels)
    assert unchanged is images
    assert unchanged_labels is labels


def test_decode_transform():
    # 1. Create a test sample
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    sample = {"image": buf.getvalue(), "label": 2}

    # 2. Eval transform (center crop)
    t_eval = _DecodeTransform(img_size=32, is_training=False)
    out_eval = t_eval.map(sample)
    assert list(out_eval["image"].shape) == [32, 32, 3]
    assert out_eval["image"].dtype == np.float32
    assert out_eval["label"] == 2

    # 3. Train transform (random crop & flip)
    t_train = _DecodeTransform(img_size=32, is_training=True)
    out_train = t_train.map(sample)
    assert list(out_train["image"].shape) == [32, 32, 3]
    assert out_train["image"].dtype == np.float32

    # 4. Array/PIL inputs and disabled optional augmentations.
    t_plain = _DecodeTransform(
        img_size=32, is_training=True, hflip=0.0,
        color_jitter_prob=0.0, re_prob=0.0,
    )
    out_array = t_plain.map({"image": np.asarray(img), "label": 1})
    out_pil = t_plain.map({"image": img, "label": 1})
    assert out_array["image"].shape == out_pil["image"].shape == (32, 32, 3)


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
