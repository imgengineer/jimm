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
    StepMetrics,
    _mean_metrics,
    _mixup_cutmix_jax,
    _validate_batch,
    cross_entropy,
    eval_step,
    fsdp_shard_model,
    init_distributed,
    main,
    make_cached_eval_step,
    make_cached_train_step,
    make_optimizer,
    prefetch_to_device,
    train_step,
    train_step_with_metrics,
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

    with pytest.raises(ValueError, match="labels shape"):
        cross_entropy(logits, jnp.array([0], dtype=jnp.int32))
    with pytest.raises(ValueError, match="logits must be 2-D"):
        cross_entropy(logits[None], labels)
    with pytest.raises(ValueError, match="labels shape"):
        cross_entropy(logits, jnp.ones((2, 2)))
    with pytest.raises(ValueError, match="integer class ids"):
        cross_entropy(logits, labels.astype(jnp.float32))
    with pytest.raises(ValueError, match="floating-point targets"):
        cross_entropy(logits, nnx.one_hot(labels, 3).astype(jnp.int32))
    with pytest.raises(ValueError, match="smoothing"):
        cross_entropy(logits, labels, smoothing=1.1)


def test_batch_validation_and_metric_aggregation():
    images = jnp.ones((2, 8, 8, 3))
    labels = jnp.array([0, 1])
    _validate_batch(images, labels)
    with pytest.raises(ValueError, match="NHWC"):
        _validate_batch(images[..., :1], labels)
    with pytest.raises(ValueError, match="batch size"):
        _validate_batch(images, labels[:1])

    assert _mean_metrics([1.0, 3.0], [0.2, 0.6]) == pytest.approx((2.0, 0.4))
    assert _mean_metrics([1.0, 3.0], [0.2, 0.6], [3, 1]) == pytest.approx((1.5, 0.3))
    with pytest.raises(ValueError, match="non-empty"):
        _mean_metrics([], [])
    with pytest.raises(ValueError, match="one positive"):
        _mean_metrics([1.0], [0.5], [0])
    with pytest.raises(FloatingPointError, match="non-finite"):
        _mean_metrics([jnp.nan], [0.5])


def test_make_optimizer():
    m = create_model("resnet18", num_classes=5, rngs=nnx.Rngs(0))
    opt1 = make_optimizer(m, lr=1e-3, weight_decay=0.01, epochs=1, steps_per_epoch=10, clip_grad=0.0)
    assert isinstance(opt1, nnx.Optimizer)

    opt2 = make_optimizer(m, lr=1e-3, weight_decay=0.01, epochs=1, steps_per_epoch=10, clip_grad=1.0)
    assert isinstance(opt2, nnx.Optimizer)

    one_step = make_optimizer(
        m, lr=1e-3, weight_decay=0.01, epochs=1, steps_per_epoch=1)
    assert isinstance(one_step, nnx.Optimizer)

    for kwargs in (
        {"lr": float("nan")},
        {"warmup_ratio": -1},
        {"min_lr_ratio": 2},
    ):
        options = dict(
            lr=1e-3, weight_decay=0.01, epochs=1, steps_per_epoch=10)
        options.update(kwargs)
        with pytest.raises(ValueError):
            make_optimizer(m, **options)


def test_make_optimizer_weight_decay_excludes_1d_params():
    """timm-style grouping: ndim<=1 params (bias/norm) get no weight decay."""

    @nnx.jit
    def zero_grad_step(model, optimizer):
        grads = nnx.grad(lambda m: jnp.zeros((), jnp.float32))(model)
        optimizer.update(model, grads)

    def run(weight_decay):
        m = create_model("resnet18", num_classes=5, rngs=nnx.Rngs(0))
        init = jax.tree.map(lambda p: jnp.array(p), nnx.to_pure_dict(nnx.state(m, nnx.Param)))
        opt = make_optimizer(m, lr=1e-3, weight_decay=weight_decay,
                             epochs=1, steps_per_epoch=10)
        # step 0 is a no-op (warmup lr=0); step 1 applies lr=peak
        zero_grad_step(m, opt)
        zero_grad_step(m, opt)
        return init, nnx.to_pure_dict(nnx.state(m, nnx.Param))

    init, after_no_wd = run(0.0)
    _, after_wd = run(0.1)

    # zero grads + wd=0 -> nothing moves
    for path, leaf in jax.tree.flatten_with_path(after_no_wd)[0]:
        ref = {tuple(p): l for p, l in jax.tree.flatten_with_path(init)[0]}[tuple(path)]
        assert float(jnp.abs(jnp.asarray(leaf) - jnp.asarray(ref)).max()) == 0.0

    flat_init = {tuple(p): l for p, l in jax.tree.flatten_with_path(init)[0]}
    saw_kernel = False
    for path, leaf in jax.tree.flatten_with_path(after_wd)[0]:
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


def test_eval_step_ignores_padded_examples():
    class LogitModel(nnx.Module):
        def __call__(self, images):
            return images[:, 0, 0, :]

    logits = jnp.array([
        [4.0, 1.0, 0.0],
        [0.0, 4.0, 1.0],
        [10.0, 0.0, 0.0],
    ])
    images = logits[:, None, None, :]
    labels = jnp.array([0, 1, 2], dtype=jnp.int32)
    valid = jnp.array([True, True, False])

    loss, accuracy = eval_step(LogitModel(), images, labels, valid=valid)
    expected_loss = cross_entropy(logits[:2], labels[:2])
    assert float(loss) == pytest.approx(float(expected_loss))
    assert float(accuracy) == pytest.approx(1.0)


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


def test_jax_mixup_identity_applies_label_smoothing():
    images = jnp.ones((4, 8, 8, 3), dtype=jnp.float32)
    labels = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    config = MixupCutmix(
        mixup_alpha=0.8, cutmix_alpha=0.0, prob=0.5,
        mode="batch", label_smoothing=0.1, num_classes=4)
    # PRNGKey(0) draws uniform ~0.948 >= prob, so this batch takes the
    # identity (skip-mixing) branch.
    mixed_images, mixed_labels = _mixup_cutmix_jax(
        images, labels, jax.random.PRNGKey(0), config)
    # Batches that skip mixing must still receive smoothed soft targets.
    expected = np.eye(4) * 0.9 + 0.1 / 4
    np.testing.assert_allclose(np.asarray(mixed_labels), expected)
    np.testing.assert_allclose(np.asarray(mixed_images), images)


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
        # Keep a validation tail so the CLI exercises padding and metric masking.
        os.remove(f"{temp_dataset}/val/cat/img_3.png")
        profile_calls = []
        monkeypatch.setattr(
            jax.profiler, "start_trace",
            lambda path: profile_calls.append(("start", path)))
        monkeypatch.setattr(
            jax.profiler, "stop_trace",
            lambda: profile_calls.append(("stop", None)))
        # Run 1 quick epoch CLI on synthetic dataset
        main([
            "--model", "resnet18",
            "--data-dir", temp_dataset,
            "--epochs", "1",
            "--batch-size", "4",
            "--img-size", "32",
            "--num-classes", "2",
            "--workers", "0",
            "--steps-per-epoch", "1",
            "--profile-step", "0",
            "--profile-dir", f"{out_dir}/profiles",
            "--output", out_dir,
        ])
        assert profile_calls == [
            ("start", f"{out_dir}/profiles"),
            ("stop", None),
        ]
        # Check checkpoint exists (step-numbered manager layout)
        assert os.path.exists(f"{out_dir}/resnet18/0")

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


@pytest.mark.parametrize("options", [
    ["--prefetch", "0"],
    ["--log-interval", "0"],
    ["--steps-per-epoch", "0"],
    ["--max-to-keep", "0"],
    ["--workers", "-1"],
    ["--profile-step", "0"],
    ["--profile-dir", "profiles"],
    ["--profile-step", "-1", "--profile-dir", "profiles"],
    ["--lr", "nan"],
    ["--mixup-alpha", "-1"],
    ["--smoothing", "1.1"],
    ["--drop-path", "1"],
    ["--dist-num-processes", "2"],
    ["--dist-coordinator-address", "localhost:1", "--dist-num-processes", "2",
     "--dist-process-id", "2"],
])
def test_main_rejects_invalid_cli(options):
    with pytest.raises(SystemExit, match="2"):
        main(["--data-dir", "unused", *options])


def test_main_training_cli_resume(temp_dataset, capsys):
    out_dir = tempfile.mkdtemp()
    try:
        common = [
            "--model", "resnet18",
            "--data-dir", temp_dataset,
            "--batch-size", "4",
            "--img-size", "32",
            "--num-classes", "2",
            "--workers", "0",
            "--output", out_dir,
        ]
        main(common + ["--epochs", "2", "--max-to-keep", "1"])
        assert os.path.exists(f"{out_dir}/resnet18/1")

        # Resume continues from the latest checkpoint instead of restarting.
        main(common + ["--epochs", "3", "--resume"])
        assert "[Resume] Restored checkpoint at epoch 1" in capsys.readouterr().out
        assert os.path.exists(f"{out_dir}/resnet18/2")
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def test_prefetch_to_device():
    mesh = jax.sharding.Mesh(jax.devices(), ("data",))
    P = jax.sharding.PartitionSpec
    data_sharding = jax.sharding.NamedSharding(mesh, P("data", None, None, None))
    label_sharding = jax.sharding.NamedSharding(mesh, P("data",))

    dummy_batches = [
        {"image": np.ones((2, 16, 16, 3), dtype=np.float32), "label": np.array([0, 1], dtype=np.int32)},
        {"image": np.ones((2, 16, 16, 3), dtype=np.float32) * 2, "label": np.array([1, 0], dtype=np.int32)},
    ]

    stream = prefetch_to_device(iter(dummy_batches), data_sharding, label_sharding, prefetch_size=2)
    items = list(stream)
    assert len(items) == 2
    for images, labels in items:
        assert isinstance(images, jax.Array)
        assert isinstance(labels, jax.Array)
        assert images.shape == (2, 16, 16, 3)
        assert labels.shape == (2,)

    masked_batches = [dict(dummy_batches[0], valid=np.array([True, False]))]
    masked_items = list(prefetch_to_device(
        iter(masked_batches), data_sharding, label_sharding,
        prefetch_size=1, mask_sharding=label_sharding))
    assert len(masked_items) == 1
    _, _, valid = masked_items[0]
    assert valid.shape == (2,)
    assert valid.dtype == jnp.bool_

    with pytest.raises(ValueError, match="prefetch_size"):
        list(prefetch_to_device(
            iter(dummy_batches), data_sharding, label_sharding, prefetch_size=0))


def test_train_step_with_metrics():
    m = create_model("resnet18", num_classes=5, rngs=nnx.Rngs(0))
    m.train()
    opt = make_optimizer(m, lr=1e-3, weight_decay=0.01, epochs=1, steps_per_epoch=10)

    images = jnp.ones((2, 224, 224, 3), dtype=jnp.float32)
    labels = jnp.array([0, 1], dtype=jnp.int32)

    metrics = train_step_with_metrics(m, opt, images, labels, smoothing=0.1)
    assert isinstance(metrics, StepMetrics)
    assert float(metrics.loss) > 0.0
    assert 0.0 <= float(metrics.accuracy) <= 1.0
    assert float(metrics.grad_norm) >= 0.0
