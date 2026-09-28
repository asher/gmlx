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


def pin_apc_arms(monkeypatch):
    """Pin every APC binding the pooling and kvarn installers rebind, and
    give the capability registry a private copy, so an install inside one
    test does not outlive it. Returns the mlx_vlm.apc module."""
    import pytest

    apc = pytest.importorskip("mlx_vlm.apc")
    adapters = pytest.importorskip("mlx_vlm.apc_adapters")
    coordinator = pytest.importorskip("mlx_vlm.apc_coordinator")
    for name in ("clone_cache_entry", "merge_cache_entries",
                 "build_prefix_cache_plan"):
        monkeypatch.setattr(adapters, name, getattr(adapters, name))
    adapters.register_default_capabilities()
    monkeypatch.setattr(adapters, "_CAPABILITY", dict(adapters._CAPABILITY))
    monkeypatch.setattr(coordinator, "build_prefix_cache_plan",
                        coordinator.build_prefix_cache_plan)
    monkeypatch.setattr(apc, "_read_safetensors_tensor",
                        apc._read_safetensors_tensor)
    monkeypatch.setattr(apc, "model_apc_mode", apc.model_apc_mode)
    for name in ("_snapshot_exact_cache_entry", "_load_exact_cache_entry"):
        monkeypatch.setattr(apc.DiskBlockStore, name,
                            getattr(apc.DiskBlockStore, name))
    return apc


def bind_stock_checkpoint_methods(batch):
    """Bind the stock PromptProcessingBatch checkpoint predicates to a fake
    batch, so the stock checkpoint store runs against it unchanged. A fake
    without an APC coordinator reads its checkpoint mode from _apc_mode.
    Returns the batch."""
    import types

    from mlx_vlm.generate.ar import PromptProcessingBatch

    for name in ("_apc_uses_checkpoints", "_apc_pending_checkpoint"):
        setattr(batch, name,
                types.MethodType(getattr(PromptProcessingBatch, name), batch))
    return batch
