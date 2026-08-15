"""Unit tests for jimm.checkpoint."""
import shutil
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from jimm.checkpoint import load_checkpoint, save_checkpoint
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
        )
        assert saved_path == ckpt_dir

        # 3. Corrupt weights to zeros
        def _zero(a):
            if isinstance(a, jax.Array) and not jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
                return jnp.zeros_like(a)
            return a

        nnx.update(m, jax.tree.map(_zero, nnx.state(m).to_pure_dict()))
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

    finally:
        shutil.rmtree(root, ignore_errors=True)
