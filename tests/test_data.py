"""Unit tests for jimm.data."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

import cv2  # pyright: ignore[reportMissingImports]
import grain.python as grain
import numpy as np
import pytest

import jimm.augment as augment_module
import jimm.data as data_module
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
    random_crop_or_pad,
    random_erasing,
    random_flip_left_right,
    random_flip_up_down,
    random_grayscale,
    random_resized_crop,
)


@pytest.mark.parametrize("interpolation", ["nearest", "bilinear", "bicubic", "random"])
def test_eval_transform_uses_requested_interpolation(interpolation):
    image = np.random.default_rng(0).integers(0, 256, (9, 9, 3), dtype=np.uint8)
    transform = _DecodeTransform(
        img_size=4,
        crop_pct=1.0,
        is_training=False,
        interpolation=interpolation,
        mean=(0, 0, 0),
        std=(1, 1, 1),
    )
    mode = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "random": cv2.INTER_LINEAR,
    }[interpolation]
    expected = cv2.resize(image, (4, 4), interpolation=mode).astype(np.float32) / 255
    sample = {"image": image, "label": 0}
    np.testing.assert_allclose(transform.map(sample)["image"], expected, atol=1e-7)
    np.testing.assert_array_equal(transform.map(sample)["image"], transform.map(sample)["image"])


def test_augmix_accepts_negative_depth_from_cli_help():
    policy = build_auto_augment("augmix-m3-w3-d-1")
    assert policy.depth == -1
    image = np.full((16, 16, 3), 127, dtype=np.uint8)
    assert policy(image, rng=np.random.default_rng(0)).shape == image.shape
    for invalid in ("augmix-m3-d", "augmix-m3-d-", "augmix-m3-d-invalid"):
        with pytest.raises(ValueError):
            build_auto_augment(invalid)


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


def test_image_folder_sorts_files_and_preserves_symlinks(tmp_path):
    root = tmp_path / "dataset"
    for name in ("z", "a"):
        (root / name).mkdir(parents=True)
    (root / "z" / "last.JPG").write_bytes(b"last")
    (root / "a" / "..PNG").write_bytes(b"dots")
    (root / "a" / ".png").write_bytes(b"ignored")
    (root / "a" / "first.PNG").write_bytes(b"first")
    (root / "a" / "ignore.txt").write_bytes(b"ignored")
    (root / "a" / "directory.jpg").mkdir()

    external = tmp_path / "external"
    external.mkdir()
    (external / "image.webp").write_bytes(b"external")
    (root / "m").symlink_to(external, target_is_directory=True)
    (root / "a" / "linked.webp").symlink_to(external / "image.webp")
    (root / "a" / "missing.png").symlink_to(external / "missing.png")
    (root / "a" / "loop.jpg").symlink_to(root / "a" / "loop.jpg")

    source = ImageFolder(root)
    assert source.class_to_idx == {"a": 0, "m": 1, "z": 2}
    assert source.samples == [
        (root / "a" / "..PNG", 0),
        (root / "a" / "first.PNG", 0),
        (root / "a" / "linked.webp", 0),
        (root / "m" / "image.webp", 1),
        (root / "z" / "last.JPG", 2),
    ]
    assert [source[index]["image"] for index in range(len(source))] == [
        b"dots",
        b"first",
        b"external",
        b"external",
        b"last",
    ]


def test_in_memory_cache_preserves_source_resolution(temp_dataset):
    # Caching must not change the source image before stochastic augmentation.
    source, transform = create_dataset(f"{temp_dataset}/train", in_memory=True, img_size=32)
    assert source._cache is not None
    shapes = {tuple(shape) for _, _, shape, _ in source._cache.records}
    assert shapes == {(48, 48, 3)}
    sample = transform.map(source[0])
    assert sample["image"].shape == (32, 32, 3)


def test_in_memory_cache_invalidates_when_labels_change(temp_dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, "_IMAGE_CACHE_ROOT", tmp_path / "cache")
    root = Path(temp_dataset) / "train"

    first = ImageFolder(root, in_memory=True)
    assert first[0]["label"] == 0  # bird

    # An empty class changes the sorted class indices without changing image
    # paths or file metadata, so labels must participate in the cache key.
    (root / "aardvark").mkdir()
    second = ImageFolder(root, in_memory=True)
    assert second.class_to_idx["bird"] == 1
    assert second[0]["label"] == 1
    assert first._cache.data_path != second._cache.data_path


def test_create_loader_drop_remainder(temp_dataset):
    loader = create_loader(
        f"{temp_dataset}/val",
        batch_size=5,
        img_size=32,
        is_training=False,
        num_workers=0,
        drop_remainder=True,
    )
    try:
        batches = list(loader)
        assert len(batches) == len(loader) == 4  # 24 // 5, tail dropped
        assert sum(len(b["label"]) for b in batches) == 20
    finally:
        loader.close()

    default_loader = create_loader(
        f"{temp_dataset}/val", batch_size=5, img_size=32, is_training=False, num_workers=0
    )
    try:
        assert len(list(default_loader)) == 5  # default keeps the tail batch
    finally:
        default_loader.close()

    with pytest.raises(ValueError, match="drop_remainder"):
        create_loader(
            f"{temp_dataset}/val",
            batch_size=5,
            img_size=32,
            is_training=False,
            num_workers=0,
            drop_remainder="yes",
        )


def test_no_aug_keeps_training_stream_and_uses_eval_transforms(temp_dataset):
    options = dict(batch_size=4, img_size=32, num_workers=0, shuffle=False)
    training = create_loader(f"{temp_dataset}/train", is_training=True, no_aug=True, **options)
    evaluation = create_loader(f"{temp_dataset}/train", is_training=False, **options)
    try:
        train_stream = iter(training)
        expected_batches = list(evaluation)
        for _ in range(2):
            for expected in expected_batches:
                actual = next(train_stream)
                np.testing.assert_array_equal(actual["label"], expected["label"])
                np.testing.assert_allclose(actual["image"], expected["image"])
    finally:
        training.close()
        evaluation.close()


@pytest.mark.parametrize(
    "batch_size,pad_remainder,drop_remainder,shard_index,shard_count,shuffle",
    [
        (5, False, False, 0, 1, False),
        (16, True, False, 0, 1, False),
        (16, True, False, 1, 3, False),
        (32, False, True, 0, 1, False),
        (3, False, False, 0, 5, False),
        (3, False, False, 4, 5, False),
        (3, False, False, 1, 5, True),
    ],
)
def test_multiworker_eval_batches_match_single_worker(
    temp_dataset, batch_size, pad_remainder, drop_remainder, shard_index, shard_count, shuffle
):
    options = dict(
        batch_size=batch_size,
        img_size=16,
        pad_remainder=pad_remainder,
        drop_remainder=drop_remainder,
        shuffle=shuffle,
        shard_options=grain.ShardOptions(
            shard_index=shard_index, shard_count=shard_count, drop_remainder=False
        ),
    )
    reference = create_loader(f"{temp_dataset}/val", num_workers=0, **options)
    parallel = create_loader(f"{temp_dataset}/val", num_workers=4, **options)
    try:
        expected = list(reference)
        actual = list(parallel)
        assert len(actual) == len(parallel) == len(expected)
        for actual_batch, expected_batch in zip(actual, expected):
            for key in expected_batch:
                np.testing.assert_allclose(actual_batch[key], expected_batch[key])
    finally:
        reference.close()
        parallel.close()


def test_create_loader_pads_eval_without_losing_records(temp_dataset):
    loader = create_loader(
        f"{temp_dataset}/val",
        batch_size=5,
        img_size=32,
        is_training=False,
        num_workers=0,
        pad_remainder=True,
    )
    try:
        batches = list(loader)
        assert len(batches) == len(loader) == 5
        assert all(batch["image"].shape[0] == 5 for batch in batches)
        assert sum(int(np.asarray(batch["valid"]).sum()) for batch in batches) == 24
    finally:
        loader.close()

    shard_loaders = [
        create_loader(
            f"{temp_dataset}/val",
            batch_size=5,
            img_size=32,
            is_training=False,
            num_workers=0,
            pad_remainder=True,
            shard_options=grain.ShardOptions(
                shard_index=shard_index, shard_count=2, drop_remainder=False
            ),
        )
        for shard_index in range(2)
    ]
    try:
        sharded_batches = [list(shard_loader) for shard_loader in shard_loaders]
        assert [len(batches) for batches in sharded_batches] == [3, 3]
        assert all(batch["image"].shape[0] == 5 for batches in sharded_batches for batch in batches)
        assert (
            sum(
                int(np.asarray(batch["valid"]).sum())
                for batches in sharded_batches
                for batch in batches
            )
            == 24
        )
    finally:
        for shard_loader in shard_loaders:
            shard_loader.close()


def test_sharded_loader_keeps_record_remainder(temp_dataset):
    root = Path(temp_dataset) / "val"
    image = np.full((48, 48, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(root / "cat" / "extra.png"), image)

    loaders = [
        create_loader(
            root,
            batch_size=5,
            img_size=32,
            is_training=False,
            num_workers=0,
            drop_remainder=False,
            shard_options=grain.ShardOptions(
                shard_index=shard_index, shard_count=2, drop_remainder=False
            ),
        )
        for shard_index in range(2)
    ]
    try:
        batches = [list(loader) for loader in loaders]
        counts = [sum(len(batch["label"]) for batch in shard) for shard in batches]
        assert counts == [13, 12]
        assert sum(counts) == 25
        assert [len(loader) for loader in loaders] == [3, 3]
    finally:
        for loader in loaders:
            loader.close()


def test_loader_start_step_restores_training_stream(temp_dataset):
    kwargs = dict(
        batch_size=4,
        img_size=32,
        is_training=True,
        num_workers=0,
        seed=17,
        shard_options=grain.ShardOptions(shard_index=1, shard_count=2, drop_remainder=True),
    )
    continuous = create_loader(f"{temp_dataset}/train", **kwargs)
    resumed = create_loader(f"{temp_dataset}/train", **kwargs)
    continuous_iter = iter(continuous)
    try:
        for _ in range(len(continuous)):
            next(continuous_iter)
        expected = next(continuous_iter)

        resumed.set_start_step(len(resumed))
        actual = next(iter(resumed))
        assert np.array_equal(actual["label"], expected["label"])
        assert np.array_equal(actual["image"], expected["image"])
    finally:
        continuous_iter.close()
        continuous.close()
        resumed.close()


def test_augmentations(monkeypatch):
    img = np.arange(48 * 48 * 3, dtype=np.uint8).reshape(48, 48, 3)
    original = img.copy()
    img.setflags(write=False)
    cropped = random_resized_crop(img, size=16, scale=(1.0, 1.0), ratio=(1.0, 1.0))
    assert cropped.shape[:2] == (16, 16)

    # Invalid crop ranges use the resize fallback instead of producing an invalid crop.
    fallback = random_resized_crop(img, size=16, scale=(2.0, 2.0), ratio=(1.0, 1.0))
    assert fallback.shape[:2] == (16, 16)

    monkeypatch.setattr(np.random, "rand", lambda: 0.0)
    jittered = color_jitter(img, brightness=0.2, contrast=0.2, saturation=0.2)
    assert jittered.shape == img.shape
    np.testing.assert_array_equal(img, original)

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
        "SolarizeIncreasing",
        "Desaturate",
        "GaussianBlur",
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
        "nearest",
        None,
        ("bilinear", "bicubic"),
        cv2.INTER_AREA,
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
        np.zeros((8, 64, 3), dtype=np.uint8),
        size=16,
        scale=(2.0, 2.0),
    ).shape == (16, 16, 3)
    assert augment_module.random_resized_crop(
        np.zeros((64, 8, 3), dtype=np.uint8),
        size=16,
        scale=(2.0, 2.0),
    ).shape == (16, 16, 3)
    assert augment_module.random_crop_or_pad(img, cast(Any, (16, 20))).shape == (16, 20, 3)
    assert (
        data_module.color_jitter(
            img,
            brightness=cast(Any, (0.1, 0.2)),
            contrast=cast(Any, (0.8, 1.2)),
            saturation=cast(Any, (0.8, 1.2)),
            hue=cast(Any, (-0.1, 0.1)),
            random_order=False,
        ).shape
        == img.shape
    )
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
        "AutoContrast",
        "Equalize",
        "Invert",
        "Solarize",
        "SolarizeIncreasing",
        "SolarizeAdd",
        "Color",
        "ColorIncreasing",
        "Contrast",
        "ContrastIncreasing",
        "Brightness",
        "BrightnessIncreasing",
        "Sharpness",
        "SharpnessIncreasing",
        "Desaturate",
        "GaussianBlur",
        "GaussianBlurRand",
        "Rotate",
        "Posterize",
        "PosterizeOriginal",
        "PosterizeIncreasing",
        "ShearX",
        "ShearY",
        "TranslateX",
        "TranslateY",
        "TranslateXRel",
        "TranslateYRel",
    ):
        result = augment_module.AugmentOp(
            name,
            prob=1.0,
            magnitude=5,
            hparams={"translate_const": 8, "translate_pct": 0.2},
        )(img)
        assert result.shape == img.shape
    assert augment_module.AugmentOp("Invert", prob=0.0)(img) is img
    assert (
        augment_module.AugmentOp("Invert", prob=1.0, hparams={"magnitude_std": float("inf")})(
            img
        ).shape
        == img.shape
    )
    assert (
        augment_module.AugmentOp("Invert", prob=1.0, hparams={"magnitude_std": 1.0})(img).shape
        == img.shape
    )
    assert "AugmentOp" in repr(augment_module.AugmentOp("Invert"))
    assert augment_module.AutoAugment([[("Invert", 1.0, 1.0)]])(img).shape == img.shape
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
    with pytest.raises(ValueError, match="weights"):
        data_module.rand_augment_ops(transforms={"Invert": 0})
    assert build_auto_augment("none") is None
    assert data_module.rand_augment_choices("weights")
    assert data_module.rand_augment_choices("3aw")
    assert data_module.rand_augment_ops(transforms={"Invert": 1})
    assert data_module.augmix_ops(transforms={"Invert": 1})
    assert repr(augment_module.RandAugment([], 0))
    assert repr(augment_module.AugMixAugment([], blended=True))
    assert (
        augment_module.AugMixAugment(
            data_module.augmix_ops(transforms=["Invert"]), width=1, depth=1
        )(img).shape
        == img.shape
    )
    assert (
        augment_module.RandAugment(data_module.rand_augment_ops(transforms=["Invert"]), 1, [1.0])(
            img
        ).shape
        == img.shape
    )
    assert augment_module.RandAugment([], 0)(img) is img


class _FixedRng:
    """Deterministic draws: ``rand`` returns ``value`` and ranges return their lower bound."""

    def __init__(self, value=0.75):
        self.value = value

    def random(self, size=None):
        return self.value

    def uniform(self, low=0.0, high=1.0, size=None):
        return low

    def integers(self, low, high=None, size=None):
        return low if high is not None else 0

    def shuffle(self, values):
        return None


def _ramp_image():
    return np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, axis=-1)


def test_color_jitter_uses_torchvision_blend_factors():
    img = np.random.default_rng(0).integers(0, 256, (24, 20, 3), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Brightness scales by a factor from [0.6, 1.4] instead of adding a shift.
    darker = color_jitter(img, brightness=0.4, contrast=0, saturation=0, rng=_FixedRng())
    np.testing.assert_array_equal(darker, np.clip(np.rint(img * 0.6), 0, 255))

    mean = int(gray.mean() + 0.5)
    flatter = color_jitter(img, brightness=0, contrast=0.5, saturation=0, rng=_FixedRng())
    expected = np.clip(np.rint(mean + (img.astype(np.int64) - mean) * 0.5), 0, 255)
    np.testing.assert_array_equal(flatter, expected)

    # Saturation factor 0 is the grayscale image in every channel.
    desaturated = color_jitter(img, brightness=0, contrast=0, saturation=1.0, rng=_FixedRng())
    np.testing.assert_array_equal(desaturated, np.repeat(gray[..., None], 3, axis=-1))

    # Identity ranges are skipped, including scalar zero brightness.
    assert np.array_equal(color_jitter(img, 0, 0, 0, 0, rng=_FixedRng()), img)
    assert np.array_equal(color_jitter(img, (1.0, 1.0), None, None, None), img)

    hue = color_jitter(img, 0, 0, 0, hue=0.5, rng=np.random.default_rng(1))
    assert hue.shape == img.shape and hue.dtype == np.uint8


def test_default_color_jitter_preserves_image_content():
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:64, 0:64]
    img = np.stack([x * 3 + 30, y * 2 + 40, (x + y) + 50], axis=-1).astype(np.uint8)
    outputs = [color_jitter(img, 0.4, 0.4, 0.4, rng=rng) for _ in range(200)]
    # The former additive brightness range of +/-1.4 * 255 blanked many images.
    assert min(float(output.std()) for output in outputs) > 5.0
    saturated = np.mean([np.mean((output == 0) | (output == 255)) for output in outputs])
    assert saturated < 0.05


@pytest.mark.parametrize(
    "name,magnitude,expected",
    [
        ("Solarize", 9, lambda v: np.where(v < 230, v, 255 - v)),
        ("SolarizeIncreasing", 9, lambda v: np.where(v < 26, v, 255 - v)),
        ("SolarizeAdd", 5, lambda v: np.where(v < 128, np.minimum(v + 55, 255), v)),
        ("Posterize", 9, lambda v: v & 0xE0),
        ("PosterizeIncreasing", 9, lambda v: v & 0x80),
        ("PosterizeOriginal", 9, lambda v: v & 0xFE),
        # timm: (level / 10) * 1.8 + 0.1, and 1 +/- (level / 10) * 0.9 when increasing.
        ("Brightness", 9, lambda v: np.clip(np.rint(v * ((9 / 10) * 1.8 + 0.1)), 0, 255)),
        ("BrightnessIncreasing", 9, lambda v: np.clip(np.rint(v * (1.0 + (9 / 10) * 0.9)), 0, 255)),
        ("Invert", 0, lambda v: 255 - v),
        ("AutoContrast", 0, lambda v: v),
    ],
)
def test_auto_augment_ops_follow_timm_level_mapping(name, magnitude, expected):
    img = _ramp_image()
    values = img.astype(np.int64)
    result = augment_module.AugmentOp(name, prob=1.0, magnitude=magnitude)(img, rng=_FixedRng())
    np.testing.assert_array_equal(result, expected(values))


def test_auto_augment_photometric_and_geometric_ops():
    img = np.random.default_rng(0).integers(50, 151, (12, 10, 3), dtype=np.uint8)
    hparams = {"translate_const": 4, "img_mean": (1, 2, 3), "interpolation": "bilinear"}

    stretched = augment_module.AugmentOp("AutoContrast", prob=1.0)(img)
    assert stretched.reshape(-1, 3).min(0).tolist() == [0, 0, 0]
    assert stretched.reshape(-1, 3).max(0).min() >= 254  # PIL truncates the scaled maximum

    large = np.random.default_rng(1).integers(50, 151, (64, 64, 3), dtype=np.uint8)
    equalized = augment_module.AugmentOp("Equalize", prob=1.0)(large)
    for channel in range(3):
        order = np.argsort(large[..., channel], axis=None, kind="stable")
        assert np.all(np.diff(equalized[..., channel].ravel()[order].astype(int)) >= 0)
    assert equalized.max() > large.max() and equalized.min() < large.min()

    gray = augment_module.AugmentOp("Desaturate", prob=1.0, magnitude=10)(img)
    assert np.array_equal(gray[..., 0], gray[..., 1]) and np.array_equal(gray[..., 1], gray[..., 2])

    # Geometric ops use PIL's inverse affine mapping and fill with the mean color.
    for magnitude in (0, 10):
        shifted = augment_module.AugmentOp(
            "TranslateX", prob=1.0, magnitude=magnitude, hparams=hparams
        )(img, rng=_FixedRng())
        pixels = magnitude * 4 // 10
        np.testing.assert_array_equal(shifted[:, : 10 - pixels], img[:, pixels:])
        assert (shifted[:, 10 - pixels :] == (1, 2, 3)).all()
    for name in ("Rotate", "ShearX", "ShearY"):
        same = augment_module.AugmentOp(name, prob=1.0, magnitude=0, hparams=hparams)(img)
        np.testing.assert_array_equal(same, img)
    rotated = augment_module.AugmentOp("Rotate", prob=1.0, magnitude=10, hparams=hparams)(img)
    assert (rotated[0, 0] == (1, 2, 3)).all()


def test_random_erasing_follows_timm_box_sampling():
    array = np.ones((32, 32, 3), dtype=np.float32)
    for mode in ("const", "rand", "pixel", "mean"):
        erased = random_erasing(
            array, prob=1.0, sl=0.25, sh=0.25, mode=mode, rng=np.random.default_rng(3)
        )
        changed = np.any(erased != array, axis=-1)
        rows, cols = np.flatnonzero(changed.any(1)), np.flatnonzero(changed.any(0))
        if mode == "mean":
            assert np.array_equal(erased, array)  # the mean of a constant image
            continue
        box = erased[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
        assert changed.sum() == box.shape[0] * box.shape[1]
        assert abs(box.shape[0] * box.shape[1] - 256) <= 40
        if mode == "const":
            assert not box.any()
        elif mode == "rand":
            assert np.ptp(box.reshape(-1, 3), axis=0).max() == 0
        else:
            assert np.ptp(box) > 0
    # Multiple boxes share the sampled area budget.
    erased = random_erasing(array, prob=1.0, sl=0.3, sh=0.3, count=3, rng=np.random.default_rng(0))
    assert 0 < np.mean(erased[..., 0] == 0) <= 0.32
    assert np.array_equal(random_erasing(array, prob=0.0), array)


def test_eval_transform_resizes_shorter_edge_and_center_crops():
    image = np.random.default_rng(0).integers(0, 256, (40, 80, 3), dtype=np.uint8)
    transform = _DecodeTransform(
        img_size=16,
        crop_pct=0.5,
        is_training=False,
        interpolation="bilinear",
        mean=(0, 0, 0),
        std=(1, 1, 1),
    )
    # floor(16 / 0.5) = 32 for the shorter edge keeps the 2:1 aspect ratio.
    resized = cv2.resize(image, (64, 32), interpolation=cv2.INTER_LINEAR)
    expected = resized[8:24, 24:40].astype(np.float32) / 255
    np.testing.assert_allclose(transform.map({"image": image, "label": 0})["image"], expected)
    tall = transform.map({"image": image.transpose(1, 0, 2).copy(), "label": 0})["image"]
    assert tall.shape == (16, 16, 3)


def test_decode_transform_uint8_output_skips_normalization_and_erasing():
    image = np.random.default_rng(0).integers(0, 256, (40, 48, 3), dtype=np.uint8)
    sample = {"image": image, "label": 1}
    for training in (False, True):
        options = dict(img_size=16, is_training=training, re_prob=1.0, color_jitter=None)
        raw = _DecodeTransform(normalize=False, **options)
        normalized = _DecodeTransform(**options, mean=(0, 0, 0), std=(1, 1, 1))
        out = raw.random_map(sample, np.random.default_rng(5))
        assert out["image"].dtype == np.uint8 and out["image"].shape == (16, 16, 3)
        reference = normalized.random_map(sample, np.random.default_rng(5))["image"]
        if not training:
            np.testing.assert_allclose(out["image"] / np.float32(255), reference, atol=1e-7)
        else:
            assert not np.array_equal(out["image"] / np.float32(255), reference)


def test_auto_augment_hparams_match_timm_transform_factory():
    transform = _DecodeTransform(
        img_size=200, is_training=True, auto_augment="rand-m9-n2", interpolation="bicubic"
    )
    op = transform.auto_augment.ops[0]
    assert op.hparams["translate_const"] == 90
    assert op.hparams["img_mean"] == (124, 116, 104)
    assert op.hparams["interpolation"] == "bicubic"
    augmix = _DecodeTransform(img_size=32, is_training=True, auto_augment="augmix-m3-w3")
    assert augmix.auto_augment.ops[0].hparams["translate_pct"] == 0.3
    assert not _DecodeTransform(img_size=32, auto_augment="rand-m9").force_color_jitter
    assert _DecodeTransform(img_size=32, auto_augment="3a").force_color_jitter


def test_mixup_cutmix(monkeypatch):
    images = np.zeros((2, 8, 8, 3), dtype=np.float32)
    images[1] = 1.0
    images.setflags(write=False)
    labels = np.array([0, 1], dtype=np.int32)
    monkeypatch.setattr(np.random, "beta", lambda *_: 0.5)
    monkeypatch.setattr(np.random, "permutation", lambda _: np.array([1, 0]))

    random_values = iter([0.0, 0.0])
    monkeypatch.setattr(np.random, "rand", lambda: next(random_values))
    mixed, mixed_labels = MixupCutmix(mixup_alpha=1.0, cutmix_alpha=0.0, num_classes=2)(
        images, labels
    )
    assert mixed.shape == images.shape
    assert mixed_labels.shape == (2, 2)
    np.testing.assert_array_equal(mixed, np.full_like(images, 0.5))
    np.testing.assert_array_equal(mixed_labels, np.full((2, 2), 0.5))
    assert not np.shares_memory(mixed, images)

    random_values = iter([0.0, 1.0])
    monkeypatch.setattr(np.random, "rand", lambda: next(random_values))
    cutmixed, cutmix_labels = MixupCutmix(mixup_alpha=1.0, cutmix_alpha=1.0, num_classes=2)(
        images, labels
    )
    assert cutmixed.shape == images.shape
    assert cutmix_labels.shape == (2, 2)
    assert not np.shares_memory(cutmixed, images)

    random_values = iter([1.0])
    monkeypatch.setattr(np.random, "rand", lambda: next(random_values))
    unchanged, unchanged_labels = MixupCutmix(prob=0.0, num_classes=2)(images, labels)
    assert unchanged is images
    assert unchanged_labels.shape == (2, 2)


def test_mixup_modes_and_edges():
    images = np.zeros((4, 8, 8, 3), dtype=np.float32)
    labels = np.arange(4, dtype=np.int32)
    with pytest.raises(ValueError, match="mode"):
        MixupCutmix(mode="invalid")
    for kwargs in (
        {"mixup_alpha": -1},
        {"cutmix_alpha": float("nan")},
        {"prob": 2},
        {"switch_prob": -1},
        {"label_smoothing": 1.1},
        {"num_classes": 0},
        {"cutmix_minmax": (0.8, 0.2)},
        {"cutmix_minmax": (0.2,)},
    ):
        with pytest.raises(ValueError):
            MixupCutmix(**kwargs)

    mix = MixupCutmix(num_classes=4)
    with pytest.raises(ValueError, match="images"):
        mix(images[..., :2], labels)
    with pytest.raises(ValueError, match="labels"):
        mix(images, labels[:2])
    with pytest.raises(ValueError, match="soft labels"):
        mix(images, np.zeros((4, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="integer class ids"):
        mix(images, labels.astype(np.float32))
    with pytest.raises(ValueError, match="between 0 and 3"):
        mix(images, np.array([0, 1, 2, -1]))
    with pytest.raises(ValueError, match="finite"):
        mix(images, np.full((4, 4), np.nan))

    for mode in ("pair", "elem"):
        mixed, targets = MixupCutmix(
            mixup_alpha=1.0,
            cutmix_alpha=0.0,
            mode=mode,
            label_smoothing=0.1,
            num_classes=4,
        )(images, labels)
        assert mixed.shape == images.shape
        assert targets.shape == (4, 4)

    for mode in ("batch", "elem"):
        mixed, targets = MixupCutmix(
            mixup_alpha=0.0,
            cutmix_alpha=1.0,
            mode=mode,
            num_classes=4,
            cutmix_minmax=(0.25, 0.5),
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

    with pytest.raises(ValueError, match="unable to scan dataset root"):
        ImageFolder(Path(temp_dataset) / "missing")
    for crop_pct in (0, float("nan"), "bad"):
        with pytest.raises(ValueError, match="crop_pct"):
            _DecodeTransform(img_size=8, crop_pct=cast(Any, crop_pct))
    for kwargs in ({"hflip": 2}, {"re_prob": float("nan")}, {"vflip": "bad"}):
        with pytest.raises(ValueError, match="between 0 and 1"):
            _DecodeTransform(img_size=8, **kwargs)
    with pytest.raises(ValueError, match="std values must be positive"):
        _DecodeTransform(img_size=8, std=(1, 0, 1))

    original_scandir = data_module.os.scandir
    with monkeypatch.context() as mp:
        mp.setattr(
            data_module.os,
            "scandir",
            lambda path: (
                (_ for _ in ()).throw(OSError("synthetic failure"))
                if Path(path).name == "cat"
                else original_scandir(path)
            ),
        )
        with pytest.raises(OSError, match="unable to scan class directory"):
            ImageFolder(root)

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
        mp.setattr(
            Path,
            "open",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("synthetic failure")),
        )
        with pytest.raises(OSError, match="unable to read image"):
            ds[0]

    images = np.zeros((2, 8, 8, 3), dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int32)
    monkeypatch.setattr(np.random, "beta", lambda *_: 0.5)
    monkeypatch.setattr(np.random, "permutation", lambda _: np.array([1, 0]))

    random_values = iter([0.0, 0.0])
    monkeypatch.setattr(np.random, "rand", lambda: next(random_values))
    assert (
        MixupCutmix(mixup_alpha=1.0, cutmix_alpha=0.0, num_classes=2)(images, labels)[0].shape
        == images.shape
    )

    random_values = iter([0.0, 1.0])
    monkeypatch.setattr(np.random, "rand", lambda: next(random_values))
    assert (
        MixupCutmix(mixup_alpha=1.0, cutmix_alpha=1.0, num_classes=2)(images, labels)[0].shape
        == images.shape
    )

    random_values = iter([0.0])
    monkeypatch.setattr(np.random, "rand", lambda: next(random_values))
    unchanged, unchanged_labels = MixupCutmix(mixup_alpha=0.0, cutmix_alpha=0.0, num_classes=2)(
        images, labels
    )
    assert unchanged is images
    assert unchanged_labels.shape == (2, 2)


def test_decode_transform():
    with pytest.raises(ValueError, match="unable to decode"):
        data_module._decode_image(b"not an image")

    # 1. Create a test sample
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
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
        img_size=32,
        is_training=True,
        hflip=0.0,
        color_jitter_prob=0.0,
        re_prob=0.0,
    )
    t_rkrc = _DecodeTransform(img_size=32, is_training=True, train_crop_mode="rkrc")
    t_rkrr = _DecodeTransform(img_size=32, is_training=True, train_crop_mode="rkrr")
    assert t_rkrc.map(sample)["image"].shape == (32, 32, 3)
    assert t_rkrr.map(sample)["image"].shape == (32, 32, 3)
    t_aug = _DecodeTransform(
        img_size=32,
        is_training=True,
        auto_augment="3a",
        color_jitter=cast(Any, None),
        re_prob=0.0,
    )
    assert t_aug.map(sample)["image"].shape == (32, 32, 3)
    t_force = _DecodeTransform(
        img_size=32,
        is_training=True,
        auto_augment="3a",
        force_color_jitter=True,
        color_jitter=cast(Any, (0.1, 0.1, 0.1, 0.1)),
        re_prob=0.0,
    )
    assert t_force.map(sample)["image"].shape == (32, 32, 3)
    with pytest.raises(ValueError, match="color_jitter"):
        _DecodeTransform(
            img_size=32,
            is_training=True,
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
    assert memory_ds[0]["image"].shape == (48, 48, 3)

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
        f"{temp_dataset}/train",
        batch_size=4,
        img_size=32,
        is_training=True,
        num_workers=0,
        in_memory=True,
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


def test_decode_truncated_jpeg():
    # Encode a dummy JPEG image and strip the trailing \xff\xd9 EOI marker
    img = np.ones((32, 32, 3), dtype=np.uint8) * 128
    _, encoded = cv2.imencode(".jpg", img)
    raw = encoded.tobytes()
    assert raw.endswith(b"\xff\xd9")
    truncated_raw = raw[:-2]
    # Verify _decode_image successfully recovers the truncated image
    decoded = data_module._decode_image(truncated_raw)
    assert decoded.shape == (32, 32, 3)
    assert isinstance(decoded, np.ndarray)
