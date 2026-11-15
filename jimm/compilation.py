"""JAX persistent compilation cache configuration."""

from pathlib import Path

import jax


def configure_compilation_cache(path: str) -> None:
    """Enable JAX's persistent compilation cache; must run before the first compile."""
    if not path:
        return
    cache_dir = Path(path).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    # Cache every compiled program, including short ones; training and the
    # evaluation scripts have few jitted functions, so extra writes are cheap.
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
