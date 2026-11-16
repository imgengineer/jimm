"""Unit tests for jimm.checkpoint."""

import os
import shutil
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from jimm.checkpoint import (
    CheckpointManager,
    load_checkpoint,
    save_checkpoint,
    wait_for_checkpoints,
)
from jimm.layers import Mlp
from jimm.registry import create_model


def test_checkpoint_save_and_load_roundtrip():
    root = tempfile.mkdtemp()
    try:
        # 1. Instantiate model and optimizer
        m = create_model("resnet18", num_classes=5, rngs=nnx.Rngs(0))
        m.eval()
        opt = nnx.Optimizer(m, optax.adamw(1e-3), wrt=nnx.Param)

        x = jnp.ones((2, 224, 224, 3), dtype=jnp.float32)
        out_before = np.asarray(m(x))

        # 2. Save with optimizer, epoch, and extra metadata
        ckpt_dir = f"{root}/ckpt_epoch_5"
        saved_path = save_checkpoint(
            ckpt_dir,
            m,
            optimizer=opt,
            epoch=5,
            extra={"val_acc": 0.95, "best_epoch": 5},
            wait=False,
        )
        assert saved_path == ckpt_dir

        # 3. Corrupt weights to zeros
        def _zero(a):
            if isinstance(a, jax.Array) and not jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
                return jnp.zeros_like(a)
            return a

        nnx.update(m, jax.tree.map(_zero, nnx.to_pure_dict(nnx.state(m))))
        out_corrupt = np.asarray(m(x))
        assert not np.allclose(out_corrupt, out_before, rtol=1e-3, atol=1e-3)

        # 4. Restore model + optimizer
        restored_epoch = load_checkpoint(ckpt_dir, m, optimizer=opt)
        assert restored_epoch == 5

        # Verify output is restored exactly
        out_after = np.asarray(m(x))
        np.testing.assert_allclose(out_after, out_before, rtol=1e-5, atol=1e-5)

        # 5. Restore into a fresh model without optimizer
        m_fresh = create_model("resnet18", num_classes=5, rngs=nnx.Rngs(42))
        m_fresh.eval()
        epoch2 = load_checkpoint(ckpt_dir, m_fresh)
        assert epoch2 == 5
        out_fresh = np.asarray(m_fresh(x))
        np.testing.assert_allclose(out_fresh, out_before, rtol=1e-5, atol=1e-5)
        wait_for_checkpoints()

    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_load_checkpoint_shape_mismatch_raises():
    root = tempfile.mkdtemp()
    try:
        m = create_model("resnet18", num_classes=5, rngs=nnx.Rngs(0))
        ckpt = save_checkpoint(f"{root}/ck", m, epoch=1)
        # Different head shape must fail at restore time with a clear error,
        # not later inside XLA with a cryptic dot_general message.
        wrong = create_model("resnet18", num_classes=7, rngs=nnx.Rngs(0))
        with pytest.raises(ValueError, match="shape mismatch"):
            load_checkpoint(ckpt, wrong)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_load_checkpoint_reads_checkpoint_manager_steps():
    root = tempfile.mkdtemp()
    try:
        source, source_opt = _trained_mlp(0)
        with CheckpointManager(f"{root}/run") as mgr:
            mgr.save(2, source, source_opt)
            mgr.wait_until_finished()

        # Manager step directories keep their metadata at the manager root, so
        # load_checkpoint() used to fail on them with a bare FileNotFoundError even
        # though the evaluation scripts accept any checkpoint directory.
        target, target_opt = _trained_mlp(99)
        assert load_checkpoint(f"{root}/run/2", target, target_opt) == 2

        x = jnp.ones((2, 8))
        np.testing.assert_allclose(np.asarray(target(x)), np.asarray(source(x)), rtol=1e-6)
        np.testing.assert_allclose(
            np.asarray(nnx.to_pure_dict(nnx.state(target_opt))["step"]),
            np.asarray(nnx.to_pure_dict(nnx.state(source_opt))["step"]),
        )

        # The manager root itself names no checkpoint, so ask for a step explicitly.
        with pytest.raises(ValueError, match="step directories"):
            load_checkpoint(f"{root}/run", Mlp(8, rngs=nnx.Rngs(0)))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_checkpoint_validation_is_atomic_and_requires_optimizer_state():
    root = tempfile.mkdtemp()
    try:
        source, source_opt = _trained_mlp(0)
        with_opt = save_checkpoint(f"{root}/with_opt", source, source_opt)
        model_only = save_checkpoint(f"{root}/model_only", source)

        target = Mlp(8, rngs=nnx.Rngs(99))
        target_opt = nnx.Optimizer(target, optax.sgd(1e-3), wrt=nnx.Param)
        x = jnp.ones((2, 8))
        before = np.asarray(target(x))

        with pytest.raises(ValueError, match="optimizer"):
            load_checkpoint(with_opt, target, target_opt)
        np.testing.assert_array_equal(np.asarray(target(x)), before)

        with pytest.raises(ValueError, match="no optimizer state"):
            load_checkpoint(model_only, target, target_opt)
        np.testing.assert_array_equal(np.asarray(target(x)), before)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _trained_mlp(seed):
    """Small model/optimizer pair with a nonzero optimizer step count."""
    m = Mlp(8, rngs=nnx.Rngs(seed))
    o = nnx.Optimizer(m, optax.adamw(1e-3), wrt=nnx.Param)
    x = jnp.ones((2, 8))
    for _ in range(3):
        _loss, grads = nnx.value_and_grad(lambda model: jnp.mean(model(x) ** 2))(m)
        o.update(m, grads)
    return m, o


def test_checkpoint_manager_retention_and_restore():
    root = tempfile.mkdtemp()
    try:
        mgr = CheckpointManager(
            f"{root}/run", max_to_keep=2, best_fn=lambda m: m["val_acc"], best_mode="max"
        )
        saved = {}
        # val_acc peaks at step 1; keep it in addition to the two latest.
        for step, acc in enumerate([0.1, 0.9, 0.5, 0.7]):
            m, o = _trained_mlp(step)
            mgr.save(step, m, o, metrics={"val_acc": acc})
            saved[step] = (m, o)
        mgr.wait_until_finished()
        assert mgr.all_steps() == [1, 2, 3]
        assert mgr.latest_step() == 3

        fresh_m = Mlp(8, rngs=nnx.Rngs(99))
        fresh_o = nnx.Optimizer(fresh_m, optax.adamw(1e-3), wrt=nnx.Param)
        step, epoch = mgr.restore_latest(fresh_m, fresh_o)
        assert (step, epoch) == (3, 3)

        x = jnp.ones((2, 8))
        np.testing.assert_allclose(np.asarray(fresh_m(x)), np.asarray(saved[3][0](x)), rtol=1e-6)
        # Optimizer state (including the step counter) is restored too.
        np.testing.assert_allclose(
            np.asarray(nnx.to_pure_dict(nnx.state(fresh_o))["step"]),
            np.asarray(nnx.to_pure_dict(nnx.state(saved[3][1]))["step"]),
        )

        # Manager restores validate structure/shapes like load_checkpoint.
        wrong = Mlp(16, rngs=nnx.Rngs(0))
        with pytest.raises(ValueError, match="shape mismatch"):
            mgr.restore(3, wrong)
        mgr.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_checkpoint_manager_empty():
    root = tempfile.mkdtemp()
    try:
        with pytest.raises(ValueError, match="max_to_keep"):
            CheckpointManager(f"{root}/invalid_count", max_to_keep=0)
        with pytest.raises(ValueError, match="best_mode"):
            CheckpointManager(f"{root}/invalid_mode", best_mode="highest")
        mgr = CheckpointManager(f"{root}/run")
        assert mgr.latest_step() is None
        assert mgr.all_steps() == []
        assert mgr.restore_latest(Mlp(8, rngs=nnx.Rngs(0))) == (None, None)
        mgr.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_checkpoint_manager_context_manager():
    root = tempfile.mkdtemp()
    try:
        with CheckpointManager(f"{root}/mgr") as mgr:
            m = Mlp(8, rngs=nnx.Rngs(0))
            mgr.save(0, m)
            mgr.wait_until_finished()
            assert mgr.latest_step() == 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _slow_background_deletions(mgr, delay=0.1):
    """Slow down Orbax's background deleter so queued removals stay observable."""
    deleter = mgr._manager._checkpoint_deleter
    original = deleter._standard_deleter.delete

    def slow_delete(step):
        time.sleep(delay)
        original(step)

    deleter._standard_deleter.delete = slow_delete


def _deleter_thread(mgr):
    return mgr._manager._checkpoint_deleter._delete_thread


def test_checkpoint_manager_overwrite_does_not_race_background_deletion():
    """Re-saving a step must not race the queued removal of that same step.

    With background deletion the removal of an overwritten step is only queued, so a
    re-save used to run concurrently with it: the two removals of ``<dir>/<step>`` could
    fail the save, or the queued removal could land afterwards and discard the fresh
    checkpoint while Orbax still listed the step.
    """
    root = tempfile.mkdtemp()
    try:
        mgr = CheckpointManager(f"{root}/run")
        model, optimizer = _trained_mlp(0)
        mgr.save(0, model, optimizer)
        mgr.wait_until_finished()
        _slow_background_deletions(mgr)

        mgr.save(0, model, optimizer)  # force-overwrite of the existing step 0
        mgr.wait_until_finished()
        assert mgr.all_steps() == [0]

        fresh, fresh_opt = _trained_mlp(99)
        assert mgr.restore(0, fresh, fresh_opt) == 0
        x = jnp.ones((2, 8))
        np.testing.assert_allclose(np.asarray(fresh(x)), np.asarray(model(x)), rtol=1e-6)

        mgr.close()  # drains the deletion queue, so the step directory must have survived
        assert os.path.isdir(f"{root}/run/0")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_checkpoint_manager_close_waits_for_background_deletions():
    """close() must not return while rotated-out checkpoints are still being deleted."""
    root = tempfile.mkdtemp()
    try:
        mgr = CheckpointManager(f"{root}/run", max_to_keep=1)
        _slow_background_deletions(mgr)
        model = _trained_mlp(0)[0]
        for step in range(3):
            mgr.save(step, model)
            mgr.wait_until_finished()
        thread = _deleter_thread(mgr)

        mgr.close()

        # Every deletion queued before close() has reached the filesystem...
        assert [s for s in range(3) if os.path.isdir(f"{root}/run/{s}")] == [2]
        # ...and the deleter is gone, so nothing can mutate the directory afterwards.
        assert not thread.is_alive()
        shutil.rmtree(root)  # a live deleter would race this teardown
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_checkpoint_invalid_tree_and_corrupt_data():
    from jimm.checkpoint import _apply_restored, _fix_int_keys, _validate_tree

    assert _fix_int_keys({"0": {"-1": 2, "head": 3}, "layer_1": 4, "--1": 5}) == {
        0: {-1: 2, "head": 3},
        "layer_1": 4,
        "--1": 5,
    }

    with pytest.raises(ValueError, match="expected a dict subtree"):
        _validate_tree({"a": 1}, 123, "root")

    with pytest.raises(ValueError, match="missing keys"):
        _validate_tree({"a": 1, "b": 2}, {"a": 1}, "root")

    with pytest.raises(ValueError, match="unexpected keys"):
        _validate_tree({"a": 1}, {"a": 1, "extra": 2}, "root")

    m = Mlp(8, rngs=nnx.Rngs(0))
    with pytest.raises(ValueError, match="not a jimm checkpoint"):
        _apply_restored("not_a_dict", m, None, "source")

    with pytest.raises(ValueError, match="invalid epoch"):
        _apply_restored(
            {"model": nnx.to_pure_dict(nnx.state(m)), "epoch": "bad_epoch"}, m, None, "source"
        )
