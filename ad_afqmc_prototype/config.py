from __future__ import annotations

import os
import platform
import socket
import sys
import warnings
from dataclasses import dataclass
from typing import Optional


def is_jupyter_notebook() -> bool:
    try:
        from IPython.core.getipython import get_ipython

        ip = get_ipython()
        return ip is not None and "IPKernelApp" in ip.config
    except Exception:
        return False


@dataclass
class AfqmcConfig:
    """
    Global configuration.

    use_gpu:
      - None  : auto (prefer GPU if available, else CPU)
      - True  : force GPU (error if unavailable)
      - False : force CPU
    """

    use_gpu: Optional[bool] = None
    single_precision: bool = False
    quiet: bool = True  # suppress prints


afqmc_config = AfqmcConfig()

_configured_once = False


def configure_once(
    *,
    use_gpu: Optional[bool] = None,
    single_precision: Optional[bool] = None,
    quiet: Optional[bool] = None,
) -> None:
    """
    Configure JAX once, subsequent calls do nothing.
    Use GPU if available by default.
    """
    global _configured_once
    if _configured_once:
        return

    if use_gpu is not None:
        afqmc_config.use_gpu = use_gpu
    if single_precision is not None:
        afqmc_config.single_precision = single_precision
    if quiet is not None:
        afqmc_config.quiet = quiet

    setup_jax(
        use_gpu=afqmc_config.use_gpu,
        single_precision=afqmc_config.single_precision,
        quiet=afqmc_config.quiet,
    )
    _configured_once = True


def _detect_gpu() -> bool:
    """Detect GPU hardware without importing JAX (NVIDIA or AMD)."""
    import shutil
    import subprocess

    # NVIDIA: check nvidia-smi
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r.returncode == 0 and "GPU" in r.stdout:
                return True
        except Exception:
            pass

    # AMD ROCm: /dev/kfd is the kernel fusion driver
    if os.path.exists("/dev/kfd"):
        return True

    return False


def setup_jax(*, use_gpu: Optional[bool], single_precision: bool, quiet: bool) -> None:
    """
    Configure JAX runtime.
    """
    jax_already_imported = "jax" in sys.modules

    # resolve auto-detection before touching JAX
    if use_gpu is None:
        use_gpu = _detect_gpu()
    afqmc_config.use_gpu = use_gpu

    # env vars only take effect if JAX hasn't been imported yet
    if jax_already_imported and use_gpu:
        warnings.warn(
            "JAX was imported before AFQMC configuration; "
            "GPU memory settings (XLA_PYTHON_CLIENT_PREALLOCATE, "
            "XLA_PYTHON_CLIENT_ALLOCATOR) may not take effect. "
            "For full control, call config.configure_once(...) before importing jax.",
            stacklevel=3,
        )
    if not jax_already_imported:
        if not single_precision:
            os.environ.setdefault("JAX_ENABLE_X64", "1")
        if use_gpu:
            os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
            os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
            os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
        else:
            os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

    from jax import config as jax_config

    # these two work even after import
    jax_config.update("jax_threefry_partitionable", False)
    if not single_precision:
        jax_config.update("jax_enable_x64", True)

    # platform_name only works before backend init
    if not jax_already_imported:
        if use_gpu:
            jax_config.update("jax_platform_name", "gpu")
        else:
            jax_config.update("jax_platform_name", "cpu")

    # verify GPU actually initialized
    if use_gpu:
        import jax

        platforms = {d.platform for d in jax.devices()}
        if "gpu" not in platforms:
            raise RuntimeError(
                "GPU was detected/requested, but JAX did not initialize a GPU backend. "
                "Ensure jaxlib with CUDA or ROCm support is installed."
            )

    if not quiet and use_gpu:
        _print_host_info()


def _print_host_info() -> None:
    hostname = socket.gethostname()
    uname_info = platform.uname()
    print(f"# Hostname: {hostname}")
    print("# Using GPU (Policy A).")
    print(f"# System: {uname_info.system}")
    print(f"# Node Name: {uname_info.node}")
    print(f"# Release: {uname_info.release}")
    print(f"# Version: {uname_info.version}")
    print(f"# Machine: {uname_info.machine}")
    print(f"# Processor: {uname_info.processor}")
