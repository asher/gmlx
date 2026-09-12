"""Shared test helpers importable from any test directory."""

import os


def _real_apple_gpu() -> bool:
    """True on a real Apple GPU; False under KQUANT_FORCE_CPU, in a VM
    (Paravirtual device), or when mlx cannot report a device. Import-time
    callable so module-level pytestmark can use it (not a fixture)."""
    if os.environ.get("KQUANT_FORCE_CPU"):
        return False
    try:
        import mlx.core as mx

        name = str(mx.device_info().get("device_name", ""))
    except Exception:
        return False
    return "Apple" in name and "Paravirtual" not in name
