"""Helpers for the image tests: header fixtures as views, official configs,
and small synthetic GGUFs written with gguf-py."""
from __future__ import annotations

import functools
import json
import os
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


@functools.lru_cache(maxsize=None)
def _image_dits(root: str, family: str) -> tuple:
    from gmlx.image import fingerprint as fp
    from gmlx.image.registry import is_image_arch
    from gmlx.load.headerscan import scan_gguf

    found = []
    for path in sorted(Path(root).rglob("*.gguf")):
        try:
            arch = scan_gguf(str(path), include_tensors=False).kv.get(
                "general.architecture")
            if arch is not None and not is_image_arch(arch):
                continue
            got = fp.identify(fp.view_from_scans([scan_gguf(str(path))]))
        except Exception:
            continue
        if got.family.name == family:
            found.append(str(path))
    return tuple(sorted(found, key=lambda p: (os.path.getsize(p), p)))


def image_dits(root, family: str) -> list[str]:
    """The DiT GGUFs of ``family`` under ``root``, smallest first, found
    by fingerprint as the loader finds them. A DiT file may carry no
    general.architecture key: the Qwen-Image 2.1 files of
    unsloth/Qwen-Image-2.1-GGUF have no KV at all, so an index by that
    key does not list them. A file under a text arch is not read."""
    return list(_image_dits(str(root), family))


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


# A tiny Z-Image config: two heads of 32, one refiner block each, two layers.
TINY_Z_CONFIG = {
    "all_f_patch_size": [1], "all_patch_size": [2], "axes_dims": [8, 12, 12],
    "axes_lens": [64, 32, 32], "cap_feat_dim": 16, "dim": 64,
    "in_channels": 4, "n_heads": 2, "n_kv_heads": 2, "n_layers": 2,
    "n_refiner_layers": 1, "norm_eps": 1e-5, "qk_norm": True,
    "rope_theta": 256.0, "t_scale": 1000.0, "head_dim": 32, "ffn_dim": 170,
    "adaln_dim": 32, "t_freq_dim": 16, "t_mid_dim": 24,
    "final_norm_eps": 1e-6, "seq_multiple": 32,
}


def tiny_z_model(seed: int = 0):
    """A tiny Z-Image DiT with random fp32 weights and fp32 activations,
    and its parameters as float64 numpy arrays by name."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from gmlx.image.families.z_image.model import ZImageDiT

    m = ZImageDiT(TINY_Z_CONFIG)
    rng = np.random.default_rng(seed)
    params = []
    for k, v in tree_flatten(m.parameters()):
        a = rng.standard_normal(v.shape) * 0.2
        if k.endswith("norm1.weight"):
            a += 1.0
        params.append((k, mx.array(a.astype(np.float32))))
    m.load_weights(params)
    m.act_dtype = mx.float32
    m.eval()
    return m, {k: np.array(v, np.float64) for k, v in params}
