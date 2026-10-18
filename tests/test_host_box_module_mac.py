"""A module that declares a Mac for its unit rows: a row on local GGUFs
still sees this Mac, from its setup or from the fetch of a gate in its body
(host_box.py)."""
from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.skipif(os.environ.get("GMLX_TEST_REAL_HOST") == "1",
                       reason="the real host"),
    pytest.mark.host_gb(16),
]


def _metal() -> dict | None:
    # The Metal device, as gmlx reads it; under KQUANT_FORCE_CPU the
    # default device is the CPU, which reports no memory.
    from gmlx.device import metal_info

    return metal_info()


def _metal_memory() -> int:
    from gmlx.device import metal_value

    return metal_value("memory_size")


def _ram(sysconf=None):
    sysconf = sysconf or os.sysconf
    return sysconf("SC_PHYS_PAGES") * sysconf("SC_PAGE_SIZE")


def _real_ram():
    import host_box

    return _ram(host_box.real("sysconf"))


def _reads():
    from gmlx._vendor.mlx_vlm import apc
    from gmlx.serve import kernel_vm

    return apc._free_ram_bytes(), kernel_vm.available_bytes()


def test_a_row_on_local_ggufs_sees_this_mac(gguf_dir):
    import host_box

    assert _metal() == host_box.real_device_info()
    assert _ram() == _real_ram()


def test_a_unit_row_sees_the_mac_of_its_module():
    import host_box

    box, vm = host_box.mac(16)
    assert _metal_memory() == _ram() == box["memory_size"]
    assert set(_reads()) == {vm["free"] + vm["inactive"]}


def test_a_gate_fetched_in_the_body_switches_the_row_to_this_mac(request):
    """As a row that picks its model folder as it runs does, whether or
    not an earlier row set the folder up."""
    import host_box

    box = host_box.mac(16)[0]
    assert _metal_memory() == _ram() == box["memory_size"]
    request.getfixturevalue("gguf_dir")
    assert _metal() == host_box.real_device_info()
    assert _ram() == _real_ram()
