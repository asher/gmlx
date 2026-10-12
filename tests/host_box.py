"""The machine the unit tests see.

Unit tests price models against one fixed test box, a 128 GB Mac, so they
give the same result on a CI runner of about 7.5 GB as on a large Mac.
GMLX_TEST_HOST_RAM_GB makes this process report a Mac of that much memory
underneath, as the CI runner is. GMLX_TEST_REAL_HOST=1 turns the fixed box
off, so a run sees that host, or the real one.
docs/internals/testing.md has the rules this serves.
"""
from __future__ import annotations

import os

HOST_RAM_ENV = "GMLX_TEST_HOST_RAM_GB"
REAL_HOST_ENV = "GMLX_TEST_REAL_HOST"

GB = 1 << 30

# An idle M5 Max with 128 GB, as Metal and the kernel report it.
TEST_BOX = {
    "device_name": "Apple M5 Max",
    "architecture": "applegpu_g17s",
    "memory_size": 137438953472,
    "max_recommended_working_set_size": 120259084288,
    "max_buffer_length": 86586540032,
}
TEST_BOX_VM = {
    "free": 80 * GB, "inactive": 15 * GB, "purgeable": 1 * GB,
    "speculative": 2 * GB, "filebacked": 20 * GB, "wired": 6 * GB,
    "compressor": 0,
}


def small_host_bytes() -> int | None:
    raw = os.environ.get(HOST_RAM_ENV, "").strip()
    if not raw:
        return None
    return int(float(raw) * 1e9)


def small_host(mem: int) -> dict:
    """A CI runner of ``mem`` bytes: a virtual GPU with no Apple chip name."""
    ws = mem * 2 // 3
    return {"device_name": "Apple Paravirtual device",
            "architecture": "air64_v27", "memory_size": mem,
            "max_recommended_working_set_size": ws, "max_buffer_length": ws}


def small_host_vm(mem: int) -> dict:
    """A busy runner of ``mem`` bytes: little the kernel can hand back."""
    return {"free": mem // 32, "inactive": mem // 8, "purgeable": 0,
            "speculative": mem // 64, "filebacked": mem // 16,
            "wired": mem // 4, "compressor": mem // 16}


def install_host(box: dict, vm: dict) -> None:
    """Report ``box`` and the kernel counters ``vm`` from the lowest reads:
    MLX's ``device_info``, ``os.sysconf("SC_PHYS_PAGES")`` and
    ``gmlx.serve.kernel_vm.snapshot``. A later call wraps an earlier one."""
    import mlx.core as mx

    from gmlx.serve import kernel_vm

    # A test turns the counters off with kernel_vm._disabled. The reported
    # counters are consistent, so the sysctl cross-check passes; run
    # against the real sysctl it would turn them off for the session.
    kernel_vm.snapshot = lambda: (
        None if kernel_vm._disabled is not None else dict(vm))
    kernel_vm.selfcheck = lambda: kernel_vm._disabled
    real_info, real_sysconf = mx.device_info, os.sysconf
    mem = int(box["memory_size"])

    def device_info(*args, **kwargs):
        info = dict(real_info(*args, **kwargs))
        if "memory_size" in info:
            info.update(box)
        return info

    def sysconf(name):
        if name in ("SC_PHYS_PAGES", os.sysconf_names.get("SC_PHYS_PAGES")):
            return mem // real_sysconf("SC_PAGE_SIZE")
        return real_sysconf(name)

    # gmlx.device.metal_info asks a stock device_info whether Metal is
    # there; this one stands in for the stock one.
    device_info.__module__ = real_info.__module__
    mx.device_info = device_info
    os.sysconf = sysconf


def install() -> None:
    mem = small_host_bytes()
    if mem is not None:
        install_host(small_host(mem), small_host_vm(mem))
    if os.environ.get(REAL_HOST_ENV, "") != "1":
        install_host(TEST_BOX, TEST_BOX_VM)
