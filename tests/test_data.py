"""Unit tests for jimm.data."""
import io
import os
import shutil
import tempfile

import grain.python as grain
import numpy as np
import pytest
from PIL import Image

from jimm.data import ImageFolder, _DecodeTransform, create_dataset, create_loader


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


def test_create_dataset_and_loader(temp_dataset):
    # 1. create_dataset
    ds, tf = create_dataset(f"{temp_dataset}/train", img_size=32, is_training=True)
    assert len(ds) == 24
    assert isinstance(tf, _DecodeTransform)

    # 2. create_loader (training mode: infinite stream)
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
