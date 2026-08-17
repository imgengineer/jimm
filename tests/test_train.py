"""Unit tests for jimm.train."""
import os
import shutil
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from PIL import Image

from jimm.registry import create_model
from jimm.train import (
    cross_entropy,
    eval_step,
    fsdp_shard_model,
    init_distributed,
    main,
    make_cached_eval_step,
    make_cached_train_step,
    make_optimizer,
    train_step,
)


@pytest.fixture
def temp_dataset():
    root = tempfile.mkdtemp()
    for split in ["train", "val"]:
        for cls in ["cat", "dog"]:
            os.makedirs(f"{root}/{split}/{cls}", exist_ok=True)
            for i in range(4):
                img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
                img.save(f"{root}/{split}/{cls}/img_{i}.png")
    yield root
    shutil.rmtree(root, ignore_errors=True)


def test_cross_entropy():
    logits = jnp.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]], dtype=jnp.float32)
    labels = jnp.array([0, 1], dtype=jnp.int32)

    loss_plain = cross_entropy(logits, labels, smoothing=0.0)
    assert float(loss_plain) > 0.0

    loss_smooth = cross_entropy(logits, labels, smoothing=0.1)
    assert float(loss_smooth) > 0.0
    assert float(loss_smooth) != float(loss_plain)


def test_make_optimizer():
    m = create_model("resnet18", num_classes=5, rngs=nnx.Rngs(0))
    opt1 = make_optimizer(m, lr=1e-3, weight_decay=0.01, epochs=1, steps_per_epoch=10, clip_grad=0.0)
    assert isinstance(opt1, nnx.Optimizer)

    opt2 = make_optimizer(m, lr=1e-3, weight_decay=0.01, epochs=1, steps_per_epoch=10, clip_grad=1.0)
    assert isinstance(opt2, nnx.Optimizer)


def test_train_and_eval_step():
    m = create_model("resnet18", num_classes=5, rngs=nnx.Rngs(0))
    m.train()
    opt = make_optimizer(m, lr=1e-3, weight_decay=0.01, epochs=1, steps_per_epoch=10)

    images = jnp.ones((2, 224, 224, 3), dtype=jnp.float32)
    labels = jnp.array([0, 1], dtype=jnp.int32)

    loss, acc = train_step(m, opt, images, labels, smoothing=0.1)
    assert float(loss) > 0.0
    assert 0.0 <= float(acc) <= 1.0

    m.eval()
    v_loss, v_acc = eval_step(m, images, labels)
    assert float(v_loss) > 0.0
    assert 0.0 <= float(v_acc) <= 1.0

    # AMP path and cached wrappers use the same train/eval semantics.
    m.train()
    amp_loss, amp_acc = train_step(m, opt, images, labels, amp=True)
    assert float(amp_loss) > 0.0
    assert 0.0 <= float(amp_acc) <= 1.0

    cached_train = make_cached_train_step(m, opt, amp=True)
    cached_loss, cached_acc = cached_train(images, labels, 0.1)
    assert float(cached_loss) > 0.0
    assert 0.0 <= float(cached_acc) <= 1.0

    plain_cached_train = make_cached_train_step(m, opt)
    plain_loss, plain_acc = plain_cached_train(images, labels, 0.0)
    assert float(plain_loss) > 0.0
    assert 0.0 <= float(plain_acc) <= 1.0

    m.eval()
    cached_eval = make_cached_eval_step(m, amp=True)
    cached_v_loss, cached_v_acc = cached_eval(images, labels)
    assert float(cached_v_loss) > 0.0
    assert 0.0 <= float(cached_v_acc) <= 1.0

    plain_cached_eval = make_cached_eval_step(m)
    plain_v_loss, plain_v_acc = plain_cached_eval(images, labels)
    assert float(plain_v_loss) > 0.0
    assert 0.0 <= float(plain_v_acc) <= 1.0


def test_fsdp_shard_model():
    mesh = jax.sharding.Mesh(jax.devices(), ("data",))
    m = create_model("resnet18", num_classes=5, rngs=nnx.Rngs(0))
    opt = make_optimizer(m, lr=1e-3, weight_decay=0.01, epochs=1, steps_per_epoch=10)

    fsdp_shard_model(m, mesh)
    fsdp_shard_model(opt, mesh)

    # Check model variables have NamedSharding
    for _, node in nnx.graph.iter_graph(m):
        if isinstance(node, nnx.Variable):
            val = node.get_value()
            if isinstance(val, jax.Array):
                assert hasattr(val, "sharding")


def test_init_distributed(monkeypatch):
    calls = []
    monkeypatch.setattr(
        jax.distributed,
        "initialize",
        lambda **kwargs: calls.append(kwargs),
    )
    init_distributed("127.0.0.1:12345", num_processes=2, process_id=1)
    assert calls == [{
        "coordinator_address": "127.0.0.1:12345",
        "num_processes": 2,
        "process_id": 1,
    }]

    def fail_initialize():
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(jax.distributed, "initialize", fail_initialize)
    monkeypatch.setattr(jax, "process_index", lambda: 0)
    monkeypatch.setenv("JAX_COORDINATOR_ADDRESS", "127.0.0.1:12345")
    init_distributed()


def test_main_training_cli(temp_dataset, monkeypatch):
    out_dir = tempfile.mkdtemp()
    try:
        # Run 1 quick epoch CLI on synthetic dataset
        main([
            "--model", "resnet18",
            "--data-dir", temp_dataset,
            "--epochs", "1",
            "--batch-size", "4",
            "--img-size", "32",
            "--num-classes", "2",
            "--workers", "0",
            "--output", out_dir,
        ])
        # Check checkpoint exists
        assert os.path.exists(f"{out_dir}/resnet18/epoch_0")

        # Run with --fsdp
        main([
            "--model", "resnet18",
            "--data-dir", temp_dataset,
            "--epochs", "1",
            "--batch-size", "4",
            "--img-size", "32",
            "--num-classes", "2",
            "--workers", "0",
            "--fsdp",
            "--output", out_dir,
        ])

        # Test batch size divisibility error by mocking local_devices
        monkeypatch.setattr(jax, "local_devices", lambda: [None, None])
        with pytest.raises(ValueError, match="divisible by local device count"):
            main([
                "--model", "resnet18",
                "--data-dir", temp_dataset,
                "--batch-size", "3",  # 3 is not divisible by 2
                "--output", out_dir,
            ])
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
