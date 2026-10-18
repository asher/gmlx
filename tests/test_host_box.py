"""The simulated host reports through every free-RAM read, and a test on
model files from this Mac sees this Mac (host_box.py)."""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("GMLX_TEST_REAL_HOST") == "1", reason="the real host")


def _reads():
    from gmlx._vendor.mlx_vlm import apc
    from gmlx.cache import apc_manager
    from gmlx.serve import kernel_vm

    return (apc._free_ram_bytes(), apc_manager._vm_stat_free_ram_bytes(),
            kernel_vm.available_bytes())


def _near(got, want) -> bool:
    # Free RAM moves between two reads on a busy Mac.
    return abs(got - want) <= max(2 << 30, want // 10)


def _metal() -> dict | None:
    # The Metal device, as gmlx reads it; under KQUANT_FORCE_CPU the
    # default device is the CPU, which reports no memory.
    from gmlx.device import metal_info

    return metal_info()


# Session scoped like gguf_dir and asked for before it, so it reads the
# device as the test sets up, before the gate fixture does, in any order.
@pytest.fixture(scope="session")
def device_before_the_gate():
    return _metal()


# A test on GGUFs from this Mac never runs in CI and prices its models for
# this Mac, from its setup on.
def test_a_test_on_local_ggufs_sees_this_mac_from_its_setup(
        device_before_the_gate, gguf_dir):
    import host_box

    real = host_box.real_device_info()
    assert device_before_the_gate == real
    assert _metal() == real
    pages = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    assert pages == real["memory_size"]


def test_a_test_on_local_ggufs_reads_the_free_ram_of_this_mac(gguf_dir):
    import host_box

    apc_read, vm_stat_read, available = _reads()
    real = host_box.real("snapshot")()
    assert _near(available, real["free"] + real["inactive"])
    assert _near(vm_stat_read, host_box.real("vm_stat_free_ram")())
    assert _near(apc_read, host_box.real("apc_free_ram")())


def test_the_ci_runner_reports_its_free_ram_through_the_stock_read():
    """The copied apc's stock read (psutil, else vm_stat) would see the
    real Mac; under the simulation it reports the runner, under the 2 GB
    floor of the exact disk-restore gate."""
    import host_box

    assert set(_reads()) == {host_box.free_ram_bytes()}
    if not os.environ.get(host_box.HOST_RAM_ENV):
        assert host_box.free_ram_bytes() < 2 << 30


@pytest.mark.host_gb(64)
def test_a_declared_mac_reports_its_free_ram_through_every_read():
    import host_box

    vm = host_box.mac(64)[1]
    assert set(_reads()) == {vm["free"] + vm["inactive"]}
