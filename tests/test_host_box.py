"""The simulated host reports through every free-RAM read (host_box.py)."""
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
