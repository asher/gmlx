"""The machine the unit tests see.

Every test runs on the CI runner by default: a Mac VM of 7.5 GB with a
virtual GPU, the smallest machine the suite meets. A local run therefore
prices models as CI does, whatever Mac it runs on. A test that needs more
memory declares the Mac it assumes with ``@pytest.mark.host_gb(N)`` and
derives its expected numbers from that machine (``mac(N)``), never from a
literal. When kernels run on the real GPU (no KQUANT_FORCE_CPU), only the
memory is simulated: the chip name and architecture stay the real ones, so
kernel routing matches the hardware that runs it.
GMLX_TEST_HOST_RAM_GB changes the default size.
GMLX_TEST_REAL_HOST=1 turns the simulation off, so a run sees the real host.
docs/internals/testing.md has the rules this serves.
"""
from __future__ import annotations

import os

HOST_RAM_ENV = "GMLX_TEST_HOST_RAM_GB"
REAL_HOST_ENV = "GMLX_TEST_REAL_HOST"
RUNNER_GB = 7.5

GB = 1 << 30

# Metal's recommended working set and buffer cap as fractions of RAM, as an
# Apple silicon Mac of 64 GB or more reports them.
_WS_FRACTION = 0.875
_BUFFER_FRACTION = 0.63

# What a GPU run keeps from the real device.
_CHIP_KEYS = ("device_name", "architecture")

_real: dict = {}
_current: dict = {}


def runner(gb: float = RUNNER_GB) -> tuple[dict, dict]:
    """A CI runner of ``gb`` GB (decimal): a virtual GPU, no Apple chip
    name, little the kernel can hand back."""
    mem = int(gb * 1e9)
    ws = mem * 2 // 3
    box = {"device_name": "Apple Paravirtual device",
           "architecture": "air64_v27", "memory_size": mem,
           "max_recommended_working_set_size": ws, "max_buffer_length": ws}
    vm = {"free": mem // 32, "inactive": mem // 8, "purgeable": 0,
          "speculative": mem // 64, "filebacked": mem // 16,
          "wired": mem // 4, "compressor": mem // 16}
    return box, vm


def mac(gb: int) -> tuple[dict, dict]:
    """An idle Apple silicon Mac of ``gb`` GiB."""
    mem = gb * GB
    box = {"device_name": "Apple M5 Max", "architecture": "applegpu_g17s",
           "memory_size": mem,
           "max_recommended_working_set_size": int(mem * _WS_FRACTION),
           "max_buffer_length": int(mem * _BUFFER_FRACTION)}
    vm = {"free": mem * 5 // 8, "inactive": mem // 8, "purgeable": mem // 128,
          "speculative": mem // 64, "filebacked": mem // 6, "wired": mem // 20,
          "compressor": 0}
    return box, vm


def default() -> tuple[dict, dict]:
    raw = os.environ.get(HOST_RAM_ENV, "").strip()
    return runner(float(raw) if raw else RUNNER_GB)


def real_device_info() -> dict:
    """What the real device reports, under any simulated machine."""
    import mlx.core as mx

    return dict(_real.get("device_info", mx.device_info)())


def use(box: dict, vm: dict) -> None:
    """Make ``box`` and ``vm`` the machine every read reports from now on."""
    _current["box"], _current["vm"] = dict(box), dict(vm)


def install() -> None:
    """Report the current machine from the lowest reads: MLX's
    ``device_info``, ``os.sysconf("SC_PHYS_PAGES")``,
    ``gmlx.serve.kernel_vm.snapshot`` and the free-RAM reads under it.
    Starts on ``default()``."""
    if os.environ.get(REAL_HOST_ENV, "") == "1" or _real:
        return
    import mlx.core as mx

    from gmlx.serve import kernel_vm

    use(*default())
    real_info, real_sysconf = mx.device_info, os.sysconf
    _real["device_info"] = real_info

    # A test turns the counters off with kernel_vm._disabled. The reported
    # counters are consistent, so the sysctl cross-check passes; run
    # against the real sysctl it would turn them off for the session.
    kernel_vm.snapshot = lambda: (
        None if kernel_vm._disabled is not None else dict(_current["vm"]))
    kernel_vm.selfcheck = lambda: kernel_vm._disabled

    def device_info(*args, **kwargs):
        info = dict(real_info(*args, **kwargs))
        if "memory_size" in info:
            box = _current["box"]
            if not os.environ.get("KQUANT_FORCE_CPU"):
                box = {k: v for k, v in box.items() if k not in _CHIP_KEYS}
            info.update(box)
        return info

    def sysconf(name):
        if name in ("SC_PHYS_PAGES", os.sysconf_names.get("SC_PHYS_PAGES")):
            mem = int(_current["box"]["memory_size"])
            return mem // real_sysconf("SC_PAGE_SIZE")
        return real_sysconf(name)

    # gmlx.device.metal_info asks a stock device_info whether Metal is
    # there; this one stands in for the stock one.
    device_info.__module__ = real_info.__module__
    mx.device_info = device_info
    os.sysconf = sysconf
    _simulate_free_ram()


def free_ram_bytes() -> int:
    """free + inactive of the current machine, what every free-RAM read
    reports under the simulation."""
    vm = _current["vm"]
    return int(vm["free"] + vm["inactive"])


def real(name: str):
    """The real read ``install`` replaced: ``apc_free_ram`` (the copied
    mlx-vlm apc's stock read) or ``vm_stat_free_ram`` (gmlx's
    /usr/bin/vm_stat read). Without the simulation, the bound one."""
    if name in _real:
        return _real[name]
    if name == "apc_free_ram":
        from gmlx._vendor.mlx_vlm import apc

        return apc._free_ram_bytes
    from gmlx.cache import apc_manager

    return apc_manager._vm_stat_free_ram_bytes


def _simulate_free_ram() -> None:
    """The free-RAM reads below the mach counters report the machine too:
    the copied apc's stock read (psutil, else vm_stat on PATH), which the
    APC disk-restore gate and the memory headroom call until the APC
    manager build rebinds it, and gmlx's /usr/bin/vm_stat read, which that
    build binds when the mach counters are off. Otherwise a test reads the
    simulated machine or the real one by which read happens to be bound."""
    from gmlx._vendor.mlx_vlm import apc
    from gmlx.cache import apc_manager

    _real["apc_free_ram"] = apc._free_ram_bytes
    _real["vm_stat_free_ram"] = apc_manager._vm_stat_free_ram_bytes
    apc._free_ram_bytes = free_ram_bytes
    apc_manager._vm_stat_free_ram_bytes = free_ram_bytes
