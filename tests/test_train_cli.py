"""timm-style training arguments, configuration, and checkpoint integration."""

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from flax import nnx

import jimm.train as train_module
from jimm.checkpoint import CheckpointManager
from jimm.optim import make_optimizer
from jimm.train import _parse_args, main


def test_cli_aliases_and_defaults():
    args = _parse_args(["dataset"])
    assert args.data_dir == "dataset"
    assert args.workers == 4 and args.amp is True
    assert args.validation_batch_size == args.batch_size == 128
    assert args.opt == "adamw" and args.sched == "cosine"

    args = _parse_args(
        [
            "dataset",
            "-b",
            "16",
            "-vb",
            "8",
            "-j",
            "2",
            "--aa",
            "rand-m9-n2",
            "--mixup",
            "0.8",
            "--cutmix",
            "1",
            "--checkpoint-hist",
            "3",
            "--gp",
            "avg",
            "--no-amp",
            "--resume",
        ]
    )
    assert (args.batch_size, args.validation_batch_size, args.workers) == (16, 8, 2)
    assert args.auto_augment == "rand-m9-n2"
    assert (args.mixup_alpha, args.cutmix_alpha, args.max_to_keep) == (0.8, 1.0, 3)
    assert args.global_pool == "avg" and args.amp is False and args.resume is True

    legacy = _parse_args(
        [
            "--data-dir",
            "dataset",
            "--auto-augment",
            "rand-m9-n2",
            "--mixup-alpha",
            "0.8",
            "--cutmix-alpha",
            "1",
            "--max-to-keep",
            "3",
        ]
    )
    for name in ("auto_augment", "mixup_alpha", "cutmix_alpha", "max_to_keep"):
        assert getattr(args, name) == getattr(legacy, name)


def test_config_precedence_and_model_kwargs(tmp_path):
    config = tmp_path / "train.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "data": "dataset",
                "batch_size": 16,
                "workers": 4,
                "amp": False,
                "aa": "rand-m9-n2",
                "mixup": 0.8,
                "cutmix": 1,
                "checkpoint-hist": 2,
                "lr": "5e-4",
                "input_size": [3, 32, 32],
                "model_kwargs": {"depth": 2, "embed_dim": 64},
            }
        )
    )
    args = _parse_args(
        [
            "-c",
            str(config),
            "-b",
            "8",
            "--amp",
            "--model-kwargs",
            "depth=3",
            "qk_norm=True",
            "global_pooling=token",
            "ratios=[1,2]",
        ]
    )
    assert args.data_dir == "dataset" and args.img_size == 32
    assert args.batch_size == args.validation_batch_size == 8
    assert args.workers == 4 and args.amp is True
    assert args.lr == 5e-4 and args.auto_augment == "rand-m9-n2"
    assert (args.mixup_alpha, args.cutmix_alpha, args.max_to_keep) == (0.8, 1.0, 2)
    assert args.model_kwargs == {
        "depth": 3,
        "embed_dim": 64,
        "qk_norm": True,
        "global_pooling": "token",
        "ratios": [1, 2],
    }


@pytest.mark.parametrize(
    "settings",
    [
        [],
        False,
        {"unknown": 1},
        {"workers": 1.5},
        {"workers": True},
        {"amp": "false"},
        {"no_amp": True},
        {"opt_eps": None},
        {"opt": "sgd"},
        {"ratio": [0.5]},
        {"model_kwargs": ["depth=2"]},
        {"mixup": 0.2, "mixup_alpha": 0.4},
    ],
)
def test_config_rejects_invalid_values(tmp_path, settings):
    config = tmp_path / "invalid.yaml"
    config.write_text(yaml.safe_dump(settings))
    with pytest.raises(SystemExit, match="2"):
        _parse_args(["dataset", "--config", str(config)])


def test_config_rejects_missing_and_malformed_files(tmp_path):
    config = tmp_path / "invalid.yaml"
    with pytest.raises(SystemExit, match="2"):
        _parse_args(["dataset", "-c", str(config)])
    config.write_text("model: [resnet18\n")
    with pytest.raises(SystemExit, match="2"):
        _parse_args(["dataset", "-c", str(config)])


@pytest.mark.parametrize(
    "options",
    [
        ["--input-size", "3", "32", "48"],
        ["--validation-batch-size", "0"],
        ["--opt-eps", "0"],
        ["--opt-betas", "0.9", "1"],
        ["--warmup-epochs", "-1"],
        ["--min-lr", "0.1"],
        ["--cutmix-minmax", "0.8", "0.2"],
        ["--model-kwargs", "depth"],
        ["--model-kwargs", "num_classes=2"],
    ],
)
def test_cli_rejects_invalid_new_options(options):
    with pytest.raises(SystemExit, match="2"):
        _parse_args(["dataset", *options])


def test_config_training_and_external_resume(tmp_path, monkeypatch, capsys):
    class Model(nnx.Module):
        def __init__(self, width, num_classes, drop_rate, rngs):
            self.hidden = nnx.Linear(3, width, rngs=rngs)
            self.drop = nnx.Dropout(drop_rate, rngs=rngs)
            self.fc = nnx.Linear(width, num_classes, rngs=rngs)

        def __call__(self, x):
            return self.fc(self.drop(nnx.relu(self.hidden(x.mean(axis=(1, 2))))))

    models, optimizer_options, mixups = [], [], []

    def create_model(name, **kwargs):
        assert name == "tiny"
        assert kwargs.pop("pretrained") is False
        assert kwargs.pop("global_pool") == "avg"
        assert kwargs.pop("drop_path_rate") == 0.1
        model = Model(**kwargs)
        models.append(model)
        return model

    def create_optimizer(*args, **kwargs):
        optimizer_options.append(kwargs)
        return make_optimizer(*args, **kwargs)

    def create_train_step(*args, **kwargs):
        mixups.append(kwargs["mixup"])
        return original_train_step(*args, **kwargs)

    original_train_step = train_module.make_cached_train_step
    monkeypatch.setattr(train_module, "create_model", create_model)
    monkeypatch.setattr(train_module, "make_optimizer", create_optimizer)
    monkeypatch.setattr(train_module, "make_cached_train_step", create_train_step)
    for split in ("fit", "holdout"):
        for category in ("cat", "dog"):
            directory = tmp_path / "data" / split / category
            directory.mkdir(parents=True)
            for index in range(4):
                image = np.full((32, 32, 3), 40 + 30 * index, dtype=np.uint8)
                assert cv2.imwrite(str(directory / f"{index}.png"), image)

    config = tmp_path / "train.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "data_dir": str(tmp_path / "data"),
                "model": "tiny",
                "num_classes": 2,
                "input_size": [3, 32, 32],
                "batch_size": 4,
                "validation_batch_size": 3,
                "train_split": "fit",
                "val_split": "holdout",
                "workers": 0,
                "epochs": 1,
                "steps_per_epoch": 1,
                "seed": 23,
                "amp": False,
                "output": str(tmp_path / "output"),
                "experiment": "first",
                "gp": "avg",
                "drop": 0.2,
                "drop_path": 0.1,
                "model_kwargs": {"width": 4},
                "opt_eps": 0.1,
                "opt_betas": [0.8, 0.9],
                "warmup_epochs": 0,
                "warmup_lr": 0.0,
                "min_lr": 1e-5,
                "cutmix_minmax": [0.3, 0.7],
                "mixup_switch_prob": 0.25,
            }
        )
    )
    main(["-c", str(config)])
    first = tmp_path / "output" / "first"
    assert (first / "0").is_dir()
    assert "val loss" in capsys.readouterr().out
    assert optimizer_options[0]["eps"] == 0.1
    assert optimizer_options[0]["betas"] == [0.8, 0.9]
    assert optimizer_options[0]["warmup_epochs"] == 0
    assert optimizer_options[0]["min_lr"] == 1e-5
    assert mixups[0].cutmix_alpha == 1.0
    assert mixups[0].cutmix_minmax == (0.3, 0.7)
    assert mixups[0].switch_prob == 0.25
    saved_args = first / "args.yaml"
    resolved = _parse_args(["-c", str(saved_args)])
    assert resolved.data_dir == str(tmp_path / "data")
    assert resolved.experiment == "first" and resolved.seed == 23
    assert resolved.global_pool == "avg" and resolved.validation_batch_size == 3
    first_weights = jax.tree.map(jnp.array, nnx.state(models[0], nnx.Param))

    main(["-c", str(saved_args), "--epochs", "2", "--experiment", "second", "--resume", str(first)])
    assert "[Resume] Restored checkpoint at epoch 0; resuming at epoch 1" in capsys.readouterr().out
    second = tmp_path / "output" / "second"
    assert (second / "1").is_dir() and not (second / "0").exists()
    optimizer = make_optimizer(models[1], 5e-4, 0.05, 2, 1, eps=0.1, betas=(0.8, 0.9))
    with CheckpointManager(str(second)) as manager:
        assert manager.restore_latest(models[1], optimizer) == (1, 1)
    assert int(optimizer.step[...]) == 2
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(
            jax.tree.leaves(first_weights), jax.tree.leaves(nnx.state(models[1], nnx.Param))
        )
    )
