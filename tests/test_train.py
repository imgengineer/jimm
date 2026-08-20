"""Unit tests for jimm.train."""
import os
import shutil
import tempfile

import cv2  # pyright: ignore[reportMissingImports]
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jimm.augment import MixupCutmix
from jimm.registry import create_model
from jimm.train import (
    _mixup_cutmix_jax,
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
                img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                cv2.imwrite(
                    f"{root}/{split}/{cls}/img_{i}.png",
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                )
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

    soft_labels = nnx.one_hot(labels, 3).astype(jnp.float32)
    loss_soft = cross_entropy(logits, soft_labels)
    assert float(loss_soft) > 0.0

    with pytest.raises(AssertionError):
        cross_entropy(logits, jnp.array([0], dtype=jnp.int32))


def test_make_optimizer():
    m = create_model("resnet18", num_classes=5, rngs=nnx.Rngs(0))
    opt1 = make_optimizer(m, lr=1e-3, weight_decay=0.01, epochs=1, steps_per_epoch=10, clip_grad=0.0)
    assert isinstance(opt1, nnx.Optimizer)

    opt2 = make_optimizer(m, lr=1e-3, weight_decay=0.01, epochs=1, steps_per_epoch=10, clip_grad=1.0)
    assert isinstance(opt2, nnx.Optimizer)


def test_make_optimizer_weight_decay_excludes_1d_params():
    """timm-style grouping: ndim<=1 params (bias/norm) get no weight decay."""

    @nnx.jit
    def zero_grad_step(model, optimizer):
        grads = nnx.grad(lambda m: jnp.zeros((), jnp.float32))(model)
        optimizer.update(model, grads)

    def run(weight_decay):
        m = create_model("resnet18", num_classes=5, rngs=nnx.Rngs(0))
        init = jax.tree.map(lambda p: jnp.array(p), nnx.state(m, nnx.Param).to_pure_dict())
        opt = make_optimizer(m, lr=1e-3, weight_decay=weight_decay,
                             epochs=1, steps_per_epoch=10)
        # step 0 is a no-op (warmup lr=0); step 1 applies lr=peak
        zero_grad_step(m, opt)
        zero_grad_step(m, opt)
        return init, nnx.state(m, nnx.Param).to_pure_dict()

    init, after_no_wd = run(0.0)
    _, after_wd = run(0.1)

    # zero grads + wd=0 -> nothing moves
    for path, leaf in jax.tree_util.tree_flatten_with_path(after_no_wd)[0]:
        ref = {tuple(p): l for p, l in jax.tree_util.tree_flatten_with_path(init)[0]}[tuple(path)]
        assert float(jnp.abs(jnp.asarray(leaf) - jnp.asarray(ref)).max()) == 0.0

    flat_init = {tuple(p): l for p, l in jax.tree_util.tree_flatten_with_path(init)[0]}
    saw_kernel = False
    for path, leaf in jax.tree_util.tree_flatten_with_path(after_wd)[0]:
        ref = flat_init[tuple(path)]
        diff = float(jnp.abs(jnp.asarray(leaf) - jnp.asarray(ref)).max())
        if leaf.ndim >= 2:
            saw_kernel = saw_kernel or diff > 0
        else:
            assert diff == 0.0, f"1-D param at {path} received weight decay"
    assert saw_kernel, "no kernel received weight decay"


def test_train_and_eval_step():
    m = create_model("resnet18", num_classes=5, rngs=nnx.Rngs(0))
    m.train()
    opt = make_optimizer(m, lr=1e-3, weight_decay=0.01, epochs=1, steps_per_epoch=10)

    images = jnp.ones((2, 224, 224, 3), dtype=jnp.float32)
    labels = jnp.array([0, 1], dtype=jnp.int32)

    loss, acc = train_step(m, opt, images, labels, smoothing=0.1)
    assert float(loss) > 0.0
    assert 0.0 <= float(acc) <= 1.0

    soft_labels = nnx.one_hot(labels, 5)
    soft_loss, soft_acc = train_step(m, opt, images, soft_labels)
    assert float(soft_loss) > 0.0
    assert 0.0 <= float(soft_acc) <= 1.0

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

    mixup = MixupCutmix(mixup_alpha=0.8, cutmix_alpha=1.0, num_classes=5)
    mixed_train = make_cached_train_step(m, opt, mixup=mixup)
    mixed_loss, mixed_acc = mixed_train(
        images, labels, 0.1, rng=jax.random.PRNGKey(0))
    assert float(mixed_loss) > 0.0
    assert 0.0 <= float(mixed_acc) <= 1.0

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


def test_jax_mixup_cutmix_modes():
    images = jnp.ones((4, 8, 8, 3), dtype=jnp.float32)
    labels = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    for mode in ("batch", "pair", "elem"):
        for cutmix in (False, True):
            config = MixupCutmix(
                mixup_alpha=0.8 if not cutmix else 0.0,
                cutmix_alpha=1.0 if cutmix else 0.0,
                prob=1.0,
                mode=mode,
                num_classes=4,
            )
            mixed_images, mixed_labels = _mixup_cutmix_jax(
                images, labels, jax.random.PRNGKey(0), config)
            assert mixed_images.shape == images.shape
            assert mixed_labels.shape == (4, 4)
            np.testing.assert_allclose(np.asarray(mixed_labels).sum(axis=1), 1.0)


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
