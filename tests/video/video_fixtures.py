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
