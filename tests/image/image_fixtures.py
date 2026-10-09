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
        arch = scan_gguf(str(path), include_tensors=False).kv.get(
            "general.architecture")
        if arch is not None and not is_image_arch(arch):
            continue
        try:
            got = fp.identify(fp.view_from_scans([scan_gguf(str(path))]))
        except fp.ImageModelError:
            # Not an image DiT that gmlx takes. Any other failure of the
            # scan or of the fingerprint is a fault, and is raised: a
            # lookup that hid it would turn it into a skip of every row
            # on the real file.
            continue
        if got.family.name == family:
            found.append(str(path))
    return tuple(sorted(found, key=lambda p: (os.path.getsize(p), p)))


def image_dits(root, family: str) -> list[str]:
    """The DiT GGUFs of ``family`` under ``root``, smallest first, found
    by fingerprint as the loader finds them. A DiT file may carry no
    general.architecture key: the Qwen-Image 2.1 files of
    unsloth/Qwen-Image-2.1-GGUF have no KV at all, so an index by that
    key does not list them. A file under a text arch is not read, and a
    file the fingerprint refuses is left out. A file that cannot be
    scanned, and a fingerprint that fails any other way, raise."""
    return list(_image_dits(str(root), family))


def _fallback_type(codec: str):
    """A stand-in for a GGML type that gguf-py has no member for (Q2_0),
    with its id and block geometry from headerscan's QUANT_TYPE_FALLBACK,
    entered in gguf-py's size table. The caller removes the entry."""
    import gguf

    from gmlx.load.headerscan import QUANT_TYPE_FALLBACK

    (type_id, geometry), = [(k, v[1]) for k, v in QUANT_TYPE_FALLBACK.items()
                            if v[0] == codec]

    class _Type:
        name, value = codec, type_id

        def __int__(self):
            return type_id

        def __index__(self):
            return type_id

    raw = _Type()
    gguf.GGML_QUANT_SIZES[raw] = geometry
    return raw


def write_gguf(path, tensors: dict, arch: str | None = None) -> str:
    """Write ``tensors`` (name -> (array, codec)) to a GGUF. Arrays are in
    numpy order. BF16 takes a uint16 array of raw bits. A codec that
    gguf-py does not name takes a uint8 array of wire bytes, one row of
    blocks per matrix row. ``arch=None`` writes no general.architecture
    key."""
    import gguf

    w = gguf.GGUFWriter(str(path), arch or "none")
    if arch is None:
        w.kv_data[0].pop("general.architecture", None)
    minted = []
    try:
        for name, (arr, codec) in tensors.items():
            raw = getattr(gguf.GGMLQuantizationType, codec, None)
            if raw is None:
                raw = _fallback_type(codec)
                minted.append(raw)
            if codec == "BF16":
                w.add_tensor(name, np.ascontiguousarray(arr, dtype=np.uint16),
                             raw_shape=arr.shape, raw_dtype=raw)
            else:
                w.add_tensor(name, np.ascontiguousarray(arr), raw_dtype=raw)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
    finally:
        for raw in minted:
            del gguf.GGML_QUANT_SIZES[raw]
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


# A small Z-Image config that both kernel routes of gmlx.image.ops take:
# head dim 64 and a 256-wide residual, so a 64x64 latent gives the main
# layers 1024 image rows.
KQ_Z_CONFIG = dict(TINY_Z_CONFIG, dim=256, n_heads=4, n_kv_heads=4,
                   head_dim=64, ffn_dim=64, axes_dims=[16, 24, 24])
Z_GROUPS = ("noise_refiner", "context_refiner", "layers")


def kq_leaf(codec: str, out: int, inp: int, seed: int = 0, bias=False):
    """A KQuantLinear of random weights in ``codec``."""
    import mlx.core as mx
    import mlx_kquant as kq
    from mlx_kquant.nn import KQuantLinear

    rng = np.random.default_rng(seed)
    w = mx.array((rng.standard_normal((out, inp)) / 16).astype(np.float32))
    m = KQuantLinear(inp, out, bias, codec)
    m.weight, m.scales = kq.quantize(w, codec)
    mx.eval(m.weight, m.scales)
    return m


def kq_z_model(dtype, groups=Z_GROUPS):
    """A small Z-Image DiT of KQ_Z_CONFIG with random weights and
    activations in ``dtype``. The w1 and w3 of each block of ``groups``
    are q4_k leaves, and every other leaf is float."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from gmlx.image.families.z_image.model import ZImageDiT

    m = ZImageDiT(KQ_Z_CONFIG)
    rng = np.random.default_rng(0)
    params = []
    for k, v in tree_flatten(m.parameters()):
        a = rng.standard_normal(v.shape) * 0.1
        if k.endswith("norm1.weight"):
            a += 1.0
        params.append((k, mx.array(a.astype(np.float32)).astype(dtype)))
    m.load_weights(params)
    i = 0
    for group in Z_GROUPS:
        for blk in getattr(m, group):
            if group in groups:
                blk.feed_forward.w1 = kq_leaf("q4_k", 64, 256, 10 + i)
                blk.feed_forward.w3 = kq_leaf("q4_k", 64, 256, 20 + i)
            i += 1
    m.act_dtype = dtype
    m.eval()
    return m


def qk_rounded_once(q, k, q_w, k_w, cos, sin, eps):
    """A stand-in for mlx-kquant's ``qk_norm_rope`` on the stock ops, with
    the kernel's arithmetic: the norm and the rotation in fp32 and one
    rounding at the write."""
    import mlx.core as mx

    from gmlx.image.ops import attention

    f = mx.float32
    out = attention.qk_norm_rope(q.astype(f), k.astype(f), q_w.astype(f),
                                 k_w.astype(f), cos, sin, eps)
    return tuple(o.astype(q.dtype) for o in out)


def glu_rounded_once(x, w_gate, w_up, codec):
    """A stand-in for mlx-kquant's ``quantized_matmul_glu``: both products
    on the dequantized weights and their swiglu in fp32, rounded once."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx_kquant as kq

    f, s = mx.float32, mx.zeros((1,), mx.uint8)
    xf = x.astype(f)
    g = xf @ kq.dequantize(w_gate, s, codec, f).T
    u = xf @ kq.dequantize(w_up, s, codec, f).T
    return (nn.silu(g) * u).astype(x.dtype)


def lora_rounded_once(x, w, codec, lora_a, lora_b, *, strength=None,
                      bands=None):
    """A stand-in for mlx-kquant's ``quantized_matmul_lora``: the base
    product on the dequantized weight and the low-rank term in fp32,
    rounded once. ``lora_a`` is (in, rank) and ``lora_b`` (rank, out).
    With ``bands``, rows of (first column, end column, first rank, ranks),
    a column of a band reads that block of ranks alone and a column of no
    band reads none."""
    import mlx.core as mx
    import mlx_kquant as kq

    f, s = mx.float32, mx.zeros((1,), mx.uint8)
    xf = x.astype(f)
    z = xf @ lora_a.astype(f)
    if strength is not None:
        z = z * strength.astype(f)
    b = lora_b.astype(f)
    if bands is not None:
        keep = np.zeros(b.shape, np.float32)
        for c0, c1, z0, r in bands:
            keep[z0:z0 + r, c0:c1] = 1.0
        b = b * mx.array(keep)
    return (xf @ kq.dequantize(w, s, codec, f).T + z @ b).astype(x.dtype)
