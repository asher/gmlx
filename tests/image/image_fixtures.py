"""Helpers for the image tests: header fixtures as views, official configs,
and small synthetic GGUFs written with gguf-py."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "image"
Z_HEADER = "z_image_turbo_q4_k_m.header.json"
Q_HEADER = "qwen_image21_q4_k_m.header.json"


def load_json(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def official_config(name: str) -> dict:
    return {k: v for k, v in load_json(name).items() if not k.startswith("_")}


def fixture_view(name: str, *, drop=(), add=None, kv=None, rename=None):
    """A HeaderView over a header fixture. ``drop`` removes names that start
    with any given string, ``add`` maps name -> (shape, codec), ``rename``
    maps old -> new, ``kv`` replaces the KVs."""
    from gmlx.image.fingerprint import HeaderView, Tensor

    d = load_json(name)
    tensors = {}
    for n, shape, codec in d["tensors"]:
        if any(n.startswith(p) for p in drop):
            continue
        n = (rename or {}).get(n, n)
        tensors[n] = Tensor(tuple(shape), codec)
    for n, (shape, codec) in (add or {}).items():
        tensors[n] = Tensor(tuple(shape), codec)
    return HeaderView(kv=dict(d["kv"] if kv is None else kv), tensors=tensors)


def write_gguf(path, tensors: dict, arch: str | None = None) -> str:
    """Write ``tensors`` (name -> (array, codec)) to a GGUF. Arrays are in
    numpy order. BF16 takes a uint16 array of raw bits. ``arch=None``
    writes no general.architecture key."""
    import gguf

    w = gguf.GGUFWriter(str(path), arch or "none")
    if arch is None:
        w.kv_data[0].pop("general.architecture", None)
    for name, (arr, codec) in tensors.items():
        raw = getattr(gguf.GGMLQuantizationType, codec)
        if codec == "BF16":
            w.add_tensor(name, np.ascontiguousarray(arr, dtype=np.uint16),
                         raw_shape=arr.shape, raw_dtype=raw)
        else:
            w.add_tensor(name, np.ascontiguousarray(arr), raw_dtype=raw)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


def bf16_bits(x: np.ndarray) -> np.ndarray:
    """Truncated bf16 bits of float32 values that bf16 represents exactly."""
    return (np.asarray(x, np.float32).view(np.uint32) >> 16).astype(np.uint16)
