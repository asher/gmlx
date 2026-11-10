"""Helpers for the video tests: header fixtures as views, the official
configs, and the video DiT GGUFs under a directory."""
from __future__ import annotations

import functools
import json
import os
import time
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "video"
H3_HEADER = "h3_fl2va_q4_k.header.json"
H3_CONFIG = "h3_transformer_config.json"
H3_IDENTITY = "h3_identity.json"
H3_IDENTITY_MOLBAL = "h3_identity_molbal.json"
H3_IDENTITY_MOLBAL_TURBO = "h3_identity_molbal_turbo.json"
H3_IDENTITY_FASTH3 = "h3_identity_fasth3.json"
FASTH3_FV_GEOMETRY = "fasth3_fv_geometry.npz"


# Clips at each pacing regime of the row budget (api.EVAL_ROWS 32768):
# (size, frames, text rows, packed rows, points between evaluations).
ROW_REGIMES = [
    ((1344, 768), 362, 25, 109087, 1),
    ((1344, 768), 124, 188, 37898, 1),
    ((1344, 768), 124, 1, 37711, 1),
    ((768, 768), 124, 188, 21914, 2),
    ((672, 384), 124, 188, 9926, 4),
    ((512, 512), 22, 188, 2054, 16),
]
ROW_REGIME_IDS = ["1344x768x362", "1344x768x124", "1344x768x124-short-caption",
                  "768x768x124", "672x384x124", "512x512x22"]


# The size of the stock Qwen3-VL-32B GGUF's vocabulary as the stub lays it
# out: past MiniMax's seven ids.
N_VOCAB = 151676


def tokenizer_kv() -> dict:
    """GGUF tokenizer metadata laid out as the stock Qwen3-VL-32B file's:
    byte-level BPE with the one merge "H e", and filler entries at the
    seven ids MiniMax's tokenizer adds."""
    from tokenizers import pre_tokenizers

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tokens = alphabet + ["He"]
    tokens += [f"[PAD{i}]" for i in range(len(tokens), N_VOCAB)]
    tokens[151643] = "<|endoftext|>"
    types = [1] * (len(alphabet) + 1) + [5] * (N_VOCAB - len(alphabet) - 1)
    types[151643] = 3
    return {"general.architecture": "qwen3vl",
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.pre": "qwen2",
            "tokenizer.ggml.tokens": tokens,
            "tokenizer.ggml.merges": ["H e"],
            "tokenizer.ggml.token_type": types,
            "tokenizer.ggml.eos_token_id": 151643,
            "tokenizer.ggml.padding_token_id": 151643}


@functools.cache
def gguf_tokenizer():
    """The tokenizer gmlx builds from ``tokenizer_kv``, built quietly as
    the header tokenizer is."""
    from gmlx.load import loadlog
    from gmlx.load.tokenizer import load_tokenizer_from_gguf

    with loadlog.quiet():
        return load_tokenizer_from_gguf(tokenizer_kv(), "qwen3vl")


def load_json(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def official_config(name: str) -> dict:
    return {k: v for k, v in load_json(name).items() if not k.startswith("_")}


def fixture_view(name: str, *, drop=(), add=None, kv=None, prefix=""):
    """A HeaderView over a header fixture. ``drop`` removes names that start
    with any given string, ``add`` maps name -> (shape, codec), ``kv``
    replaces the KVs, ``prefix`` goes before every name."""
    from gmlx.image.fingerprint import HeaderView, Tensor

    d = load_json(name)
    tensors = {}
    for n, shape, codec in d["tensors"]:
        if any(n.startswith(p) for p in drop):
            continue
        tensors[prefix + n] = Tensor(tuple(shape), codec)
    for n, (shape, codec) in (add or {}).items():
        tensors[prefix + n] = Tensor(tuple(shape), codec)
    return HeaderView(kv=dict(d["kv"] if kv is None else kv), tensors=tensors)


def fasth3_view(*, gates: bool = True, arch: str | None = "wan"):
    """The FastH3 v2 file's names and shapes as a view: the base header
    with a VSA gate (5376, 7168) Q4_K in each of its 50 blocks, and its
    arch label."""
    add = {f"blocks.{i}.attn.to_gate_compress.weight": ((5376, 7168), "Q4_K")
           for i in range(50)} if gates else {}
    kv = {} if arch is None else {"general.architecture": arch}
    return fixture_view(H3_HEADER, add=add, kv=kv)


def fasth3_hashes() -> tuple:
    """The FastH3 file's identity and qkv layout hashes."""
    row = load_json(H3_IDENTITY_FASTH3)["files"][
        "fastvideo_fasth3_8step_v2_pruned-Q4_K_M.gguf"]
    return row["variant_hash"], row["layout_hash"]


def fasth3_gguf(path) -> str:
    """A GGUF with the FastH3 file's header (``fasth3_view``) and no
    tensor data."""
    import gguf
    import numpy as np

    w = gguf.GGUFWriter(str(path), arch="wan")
    for name, t in fasth3_view().tensors.items():
        codec = gguf.GGMLQuantizationType[t.codec]
        block, size = gguf.GGML_QUANT_SIZES[codec]
        shape = tuple(reversed(t.shape))
        w.add_tensor_info(name, shape, np.float32,
                          int(np.prod(shape)) // block * size,
                          raw_dtype=codec)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    w.close()
    return str(path)


def vsa_kernel(monkeypatch, nax: bool = True) -> None:
    """An mlx-kquant with ``vsa_attention``, on a GPU with NAX units or
    without, so a FastH3 run does not depend on the machine's kq."""
    import mlx_kquant as kq

    monkeypatch.setattr(kq, "vsa_attention", getattr(
        kq, "vsa_attention", lambda *a, **k: None), raising=False)
    monkeypatch.setattr(kq, "nax_available", lambda: nax)


def qk8_kernel(monkeypatch, nax: bool = True) -> None:
    """An mlx-kquant with ``sdpa_qk8``, on a GPU with NAX units or
    without, so a run with int8 attention does not depend on the
    machine's kq."""
    import mlx_kquant as kq

    for name in ("sdpa_qk8_inputs", "sdpa_qk8"):
        monkeypatch.setattr(kq, name, getattr(kq, name, lambda *a, **k: None),
                            raising=False)
    monkeypatch.setattr(kq, "nax_available", lambda: nax)


def int8_kernel(monkeypatch, nax: bool = True) -> None:
    """An mlx-kquant with the int8 attention and int8 linear ops, on a GPU
    with NAX units or without, so a run at any int8 setting does not
    depend on the machine's kq."""
    import mlx_kquant as kq

    qk8_kernel(monkeypatch, nax)
    for name in ("int8_quantize_rows", "int8_matmul"):
        monkeypatch.setattr(kq, name, getattr(kq, name, lambda *a, **k: None),
                            raising=False)


@functools.lru_cache(maxsize=None)
def _video_dits(root: str, family: str) -> tuple:
    from gmlx.image.fingerprint import view_from_scans
    from gmlx.load.headerscan import scan_gguf
    from gmlx.video import fingerprint as vfp

    found = []
    for path in sorted(Path(root).rglob("*.gguf")):
        if not path.is_file():
            continue
        arch = scan_gguf(str(path), include_tensors=False).kv.get(
            "general.architecture")
        if arch not in (None, "minimax_h3"):
            continue
        try:
            got = vfp.identify(view_from_scans([scan_gguf(str(path))]))
        except vfp.VideoModelError:
            continue
        if got.family.name == family:
            found.append(str(path))
    return tuple(sorted(found, key=lambda p: (os.path.getsize(p), p)))


def video_dits(root, family: str) -> list[str]:
    """The DiT GGUFs of ``family`` under ``root``, smallest first, found
    by fingerprint as the loader finds them. A file the video rules
    refuse is left out; any other failure of the scan raises."""
    return list(_video_dits(str(root), family))


def wait_for(path: str, limit: float = 5.0) -> bool:
    """Whether ``path`` exists within ``limit`` seconds: a stand-in
    ffmpeg opens its output a few ms after it starts."""
    t = time.monotonic()
    while not os.path.exists(path) and time.monotonic() - t < limit:
        time.sleep(0.01)
    return os.path.exists(path)
