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

# An M5 Max with 128 GB, as Metal reports it.
TEST_BOX = {
    "device_name": "Apple M5 Max",
    "architecture": "applegpu_g17s",
    "memory_size": 137438953472,
    "max_recommended_working_set_size": 120259084288,
    "max_buffer_length": 86586540032,
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


def install_host(box: dict) -> None:
    """Report ``box`` from the lowest reads: MLX's ``device_info`` and
    ``os.sysconf("SC_PHYS_PAGES")``. A later call wraps an earlier one."""
    import mlx.core as mx

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
        install_host(small_host(mem))
    if os.environ.get(REAL_HOST_ENV, "") != "1":
        install_host(TEST_BOX)
