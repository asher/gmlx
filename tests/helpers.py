"""Shared test helpers importable from any test directory."""

import os


def _real_apple_gpu() -> bool:
    """True on a real Apple GPU; False under KQUANT_FORCE_CPU, in a VM
    (Paravirtual device), or when mlx cannot report a device. Import-time
    callable so module-level pytestmark can use it (not a fixture)."""
    if os.environ.get("KQUANT_FORCE_CPU"):
        return False
    try:
        import host_box

        name = str(host_box.real_device_info().get("device_name", ""))
    except Exception:
        return False
    return "Apple" in name and "Paravirtual" not in name


def pin_apc_arms(monkeypatch):
    """Pin every APC binding the pooling and kvarn installers rebind, and
    give the capability registry a private copy, so an install inside one
    test does not outlive it. Returns the mlx_vlm.apc module."""
    import pytest

    apc = pytest.importorskip("gmlx._vendor.mlx_vlm.apc")
    adapters = pytest.importorskip("gmlx._vendor.mlx_vlm.apc_adapters")
    coordinator = pytest.importorskip("gmlx._vendor.mlx_vlm.apc_coordinator")
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

    from gmlx._vendor.mlx_vlm.generate.ar import PromptProcessingBatch

    for name in ("_apc_uses_checkpoints", "_apc_pending_checkpoint"):
        setattr(batch, name,
                types.MethodType(getattr(PromptProcessingBatch, name), batch))
    return batch


def mint_type43_gguf(path, tensors, *, pad_last=True, cut=0, tail=0,
                     arch="deepseek41") -> bytes:
    """A GGUF whose tensors all carry GGML type id 43, which gguf-py cannot
    write. ``tensors`` is ``[(name, dims, fmt, data)]`` in file order, with
    ``fmt`` a key of ``TYPE_43_FORMATS`` and ``data`` the tensor's bytes or
    None for zeros. ``cut`` drops bytes off the end of the file and ``tail``
    appends zeros to it."""
    import struct

    from gmlx.load.headerscan import TYPE_43_FORMATS

    align = 32

    def s(b: bytes) -> bytes:
        return struct.pack("<Q", len(b)) + b

    kv = s(b"general.architecture") + struct.pack("<I", 8) + s(arch.encode())
    infos, off, blobs = b"", 0, []
    for name, dims, fmt, data in tensors:
        n = 1
        for d in dims:
            n *= d
        block, tsize = TYPE_43_FORMATS[fmt]
        size = n // block * tsize
        blob = bytes(size) if data is None else bytes(data)
        assert len(blob) == size, (name, len(blob), size)
        infos += (s(name.encode()) + struct.pack("<I", len(dims))
                  + struct.pack(f"<{len(dims)}Q", *dims)
                  + struct.pack("<IQ", 43, off))
        blobs.append(blob)
        off = (off + size + align - 1) // align * align
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1) + kv + infos
    body = bytearray(head + bytes(-len(head) % align))
    for i, blob in enumerate(blobs):
        body += blob
        if pad_last or i + 1 < len(blobs):
            body += bytes(-len(body) % align)
    out = bytes(body[:len(body) - cut]) + bytes(tail)
    with open(path, "wb") as f:
        f.write(out)
    return out
