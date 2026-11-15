"""Smoke tests for the timm-style top-level training scripts."""

import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from flax import nnx

import jimm
from jimm.checkpoint import CheckpointManager, load_checkpoint, save_checkpoint

REPO = Path(__file__).resolve().parents[1]
MODEL = "test_mambaout"


def _env():
    return {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(REPO)}


def _run(script, *args, check=True):
    return subprocess.run(
        [sys.executable, str(REPO / script), *map(str, args)],
        cwd=REPO,
        env=_env(),
        text=True,
        capture_output=True,
        check=check,
    )


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("dataset") / "validation"
    rng = np.random.default_rng(0)
    for class_name in ("class_a", "class_b"):
        (root / class_name).mkdir(parents=True)
        for index in range(2):
            image = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
            cv2.imwrite(str(root / class_name / f"{index}.png"), image)
    return root.parent


def test_validate_script(dataset, tmp_path):
    results = tmp_path / "val.csv"
    output = _run(
        "validate.py",
        "--data-dir",
        dataset,
        "--model",
        MODEL,
        "--num-classes",
        2,
        "--img-size",
        64,
        "-b",
        2,
        "--results-file",
        results,
    ).stdout
    assert '"samples": 4' in output
    assert "--result" in output
    assert results.exists()


def test_validate_compilation_cache(dataset, tmp_path):
    cache = tmp_path / "jax-cache"
    _run(
        "validate.py",
        "--data-dir",
        dataset,
        "--model",
        MODEL,
        "--num-classes",
        2,
        "--img-size",
        64,
        "-b",
        2,
        "--compilation-cache",
        cache,
    )
    assert any(cache.rglob("*"))


def test_inference_script(dataset, tmp_path):
    _run(
        "inference.py",
        "--data-dir",
        dataset,
        "--model",
        MODEL,
        "--num-classes",
        2,
        "--img-size",
        64,
        "-b",
        2,
        "--topk",
        2,
        "--results-dir",
        tmp_path,
        "--results-file",
        "preds",
        "--results-format",
        "csv",
        "json",
        "--no-console-results",
    )
    rows = (tmp_path / "preds.csv").read_text().strip().splitlines()
    assert rows[0].startswith("filename,pred_0,label_0,prob_0")
    assert len(rows) == 5
    assert (tmp_path / "preds.json").exists()


def test_benchmark_script():
    output = _run(
        "benchmark.py",
        "--model",
        MODEL,
        "--num-classes",
        2,
        "--img-size",
        64,
        "-b",
        2,
        "--bench",
        "infer",
        "--num-warm-iter",
        0,
        "--num-bench-iter",
        1,
    ).stdout
    assert "infer_samples_per_sec" in output


def test_checkpoint_scripts(tmp_path):
    standalone = tmp_path / "standalone"
    model = jimm.create_model(MODEL, num_classes=2, rngs=nnx.Rngs(0))
    save_checkpoint(str(standalone), model, epoch=1)

    manager_dir = tmp_path / "run"
    manager = CheckpointManager(str(manager_dir), best_fn=lambda metrics: metrics["val_acc"])
    for step, accuracy in enumerate((0.5, 0.7)):
        manager.save(step, model, metrics={"val_acc": accuracy})
    manager.close()

    clean = tmp_path / "clean"
    _run(
        "clean_checkpoint.py",
        "--checkpoint",
        standalone,
        "--model",
        MODEL,
        "--num-classes",
        2,
        "--output",
        clean,
    )
    restored = jimm.create_model(MODEL, num_classes=2, rngs=nnx.Rngs(1))
    assert load_checkpoint(str(clean), restored) == 1

    averaged = tmp_path / "averaged"
    output = _run(
        "avg_checkpoints.py",
        "--input",
        manager_dir,
        "--model",
        MODEL,
        "--num-classes",
        2,
        "--output",
        averaged,
        "-n",
        1,
    ).stdout
    assert '"steps": [\n        1\n    ]' in output
    restored = jimm.create_model(MODEL, num_classes=2, rngs=nnx.Rngs(2))
    load_checkpoint(str(averaged), restored)


def test_bulk_runner(dataset, tmp_path):
    results = tmp_path / "bulk.csv"
    _run(
        "bulk_runner.py",
        "--model-list",
        MODEL,
        "--results-file",
        results,
        "validate.py",
        "--data-dir",
        dataset,
        "--num-classes",
        2,
        "--img-size",
        64,
        "-b",
        2,
    )
    assert results.exists()


def test_distributed_script_syntax():
    subprocess.run(["bash", "-n", str(REPO / "distributed_train.sh")], check=True)
