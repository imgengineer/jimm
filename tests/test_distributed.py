"""Exercise the training CLI with two real JAX processes on CPU."""

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("fsdp", [False, True])
def test_distributed_checkpoint_and_resume(tmp_path, fsdp):
    import cv2
    import numpy as np

    for split in ("train", "val"):
        for label in ("a", "b"):
            directory = tmp_path / "data" / split / label
            directory.mkdir(parents=True)
            for index in range(3):
                cv2.imwrite(
                    str(directory / f"{index}.png"),
                    np.full((8, 8, 3), 40 * index, np.uint8),
                )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{sock.getsockname()[1]}"
    env = os.environ.copy()
    for key in list(env):
        if key.lower().endswith("_proxy") or key in ("JAX_COORDINATOR_ADDRESS", "SLURM_JOB_ID"):
            env.pop(key)
    env.update(
        JAX_PLATFORMS="cpu",
        XLA_FLAGS="--xla_force_host_platform_device_count=1",
        OMP_NUM_THREADS="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[1]),
    )
    processes = []
    try:
        for rank in range(2):
            processes.append(
                subprocess.Popen(
                    [sys.executable, __file__, str(rank), address, str(tmp_path), str(int(fsdp))],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )
        for process in processes:
            output, _ = process.communicate(timeout=45)
            assert process.returncode == 0, output
            assert "RESUME_OK" in output
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()


def _run_worker():
    rank, address, root, fsdp = sys.argv[1:]
    rank = int(rank)
    if hasattr(os, "sched_getaffinity"):
        cpus = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, cpus[rank * 2 : rank * 2 + 2] or cpus[:1])

    import jax
    from flax import nnx

    import jimm
    from jimm.checkpoint import CheckpointManager
    from jimm.train import main

    jax.distributed.initialize(address, num_processes=2, process_id=rank)

    class Tiny(nnx.Module):
        def __init__(self, rngs):
            self.hidden = nnx.Linear(3, 8, rngs=rngs)
            self.fc = nnx.Linear(8, 2, rngs=rngs)

        def __call__(self, x):
            return self.fc(nnx.relu(self.hidden(x.mean(axis=(1, 2)))))

    @jimm.register_model
    def distributed_tiny(rngs, **kwargs):
        return Tiny(rngs)

    common = [
        "--model",
        "distributed_tiny",
        "--data-dir",
        f"{root}/data",
        "--output",
        f"{root}/output",
        "--batch-size",
        "2",
        "--img-size",
        "8",
        "--num-classes",
        "2",
        "--workers",
        "0",
        "--steps-per-epoch",
        "1",
        "--max-to-keep",
        "1",
        "--no-amp",
    ]
    if fsdp == "1":
        common.append("--fsdp")
    try:
        main(common + ["--epochs", "1"])
        main(common + ["--epochs", "2", "--resume"])
        with CheckpointManager(f"{root}/output/distributed_tiny") as manager:
            assert manager.latest_step() == 1
        print("RESUME_OK", flush=True)
    finally:
        jax.distributed.shutdown()


if __name__ == "__main__":
    _run_worker()
