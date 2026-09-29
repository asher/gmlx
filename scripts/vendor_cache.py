#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Write the mlx-vlm 0.7.3 KV cache classes and SDPA helpers into gmlx/cache.

SRC is the ``mlx_vlm`` package directory of mlx-vlm 0.7.3, for example the
installed site-packages copy or an unzipped wheel. ``kv_cache.py`` is all
of ``models/cache.py``; ``attention.py`` is ``models/base.py`` from
``create_attention_mask`` through ``scaled_dot_product_attention``. Each
edit is an exact, counted string replacement, so a source of another
shape fails here instead of producing a different copy.

``--check`` compares against the files in the tree, prints the drift
baseline as JSON (the version, the sha256 of each source file and of the
extracted lines, and every counted edit) and exits 1 on a difference.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

VERSION = "0.7.3"
BASE_FIRST, BASE_LAST = 263, 430

HEADER = """# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
# Portions copyright (c) 2025 Prince Canuma (mlx-vlm, MIT)
# Portions copyright (c) 2023-2024 Apple Inc. (mlx-lm, MIT)
# Vendored from mlx-vlm {version}, mlx_vlm/models/{src}.
"""

# ASCII for tests/test_ascii_hygiene.py.
CACHE_EDITS = [
    ("stays\n    unquantized \u2014 it is sensitive",
     "stays\n    unquantized, since it is sensitive"),
    ('"""Batch pad lifecycle \u2014 same contract as',
     '"""Batch pad lifecycle, the same contract as'),
    ("(B, H, S, \u2026) \u2014 same as BatchKVCache.",
     "(B, H, S, ...), as in BatchKVCache."),
]

# TurboQuant stays in mlx-vlm: its classes are looked up at call time, and
# none can exist while mlx_vlm.turboquant is not loaded.
BASE_EDITS = [
    ("layout is ``(B, 1, \u2026, 1, L, K)``.",
     "layout is ``(B, 1, ..., 1, L, K)``."),
    ("    if not isinstance(cache, BatchTurboQuantKVCache):\n",
     '    tq = sys.modules.get("mlx_vlm.turboquant")\n'
     "    if tq is None or not isinstance(cache, tq.BatchTurboQuantKVCache):\n"),
    ("    if isinstance(cache, (TurboQuantKVCache, BatchTurboQuantKVCache)):\n",
     '    tq = sys.modules.get("mlx_vlm.turboquant")\n'
     "    if tq is not None and isinstance(\n"
     "        cache, (tq.TurboQuantKVCache, tq.BatchTurboQuantKVCache)\n"
     "    ):\n"),
]

BASE_IMPORTS = """import sys
from typing import Optional

import mlx.core as mx
from mlx.utils import tree_map

from .kv_cache import create_causal_mask


"""

CACHE_DOC = """mlx-vlm {version} KV cache classes, owned by gmlx.

All of ``mlx_vlm/models/cache.py``: every cache class, ``_BaseCache``,
``CacheMemory`` and the module helpers. It imports nothing beyond mlx
and the standard library. Edits to the upstream file: comments are ASCII.
"""

BASE_DOC = """mlx-vlm {version} mask and SDPA helpers, owned by gmlx.

``create_attention_mask``, ``create_ssm_mask``,
``align_attention_mask_to_scores``, ``quantized_scaled_dot_product_attention``,
``_turboquant_attention_applies`` and ``scaled_dot_product_attention`` from
``mlx_vlm/models/base.py``. Edits to the upstream lines: the imports are
written here and ``create_causal_mask`` comes from ``kv_cache``; the
TurboQuant cache classes stay in mlx-vlm and are looked up in
``sys.modules`` at call time; comments are ASCII.
"""


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sub(text: str, old: str, new: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"expected 1 x {old!r}, found {n}")
    return text.replace(old, new)


def sub_all(text: str, pairs) -> str:
    for old, new in pairs:
        text = sub(text, old, new)
    return text


def base_extract(src: str) -> str:
    lines = src.splitlines(keepends=True)
    first, last = BASE_FIRST - 1, BASE_LAST
    if not lines[first].startswith("def create_attention_mask("):
        raise SystemExit(f"base.py:{BASE_FIRST} is not create_attention_mask")
    if not lines[last - 1].startswith("    )") or lines[last].strip():
        raise SystemExit(f"base.py:{BASE_LAST} does not end the SDPA helper")
    if not lines[last + 2].startswith("def expand2square("):
        raise SystemExit("base.py: expand2square no longer follows the SDPA")
    return "".join(lines[first:last])


def source_version(src: Path) -> str:
    m = re.search(r'__version__ = "([^"]+)"', (src / "version.py").read_text())
    if m is None:
        raise SystemExit(f"{src}/version.py names no __version__")
    return m.group(1)


def build(src: Path):
    """Return (files, baseline) for the package directory ``src``."""
    version = source_version(src)
    if version != VERSION:
        raise SystemExit(f"{src} is mlx-vlm {version}, expected {VERSION}")
    cache_src = (src / "models/cache.py").read_text()
    base_src = (src / "models/base.py").read_text()
    extract = base_extract(base_src)

    cache = sub_all(cache_src, CACHE_EDITS)
    attention = BASE_IMPORTS + sub_all(extract, BASE_EDITS)

    files = {}
    for name, rel, doc, body, extra in (
        ("kv_cache.py", "cache.py", CACHE_DOC, cache, ""),
        ("attention.py", f"base.py lines {BASE_FIRST} to {BASE_LAST}",
         BASE_DOC, attention,
         f"# Extract sha256: {sha256(extract)}\n"),
    ):
        head = HEADER.format(version=version, src=rel)
        src_file = "cache.py" if name == "kv_cache.py" else "base.py"
        text = (src / "models" / src_file).read_text()
        head += f"# Source sha256: {sha256(text)}\n" + extra
        files[name] = head + '"""' + doc.format(version=version) + '"""\n\n' + body
    for name, text in files.items():
        if not text.isascii():
            raise SystemExit(f"{name}: non-ASCII left after the edits")

    baseline = {
        "package": "mlx-vlm",
        "version": version,
        "sources": {
            "models/cache.py": {
                "sha256": sha256(cache_src),
                "lines": cache_src.count("\n"),
            },
            "models/base.py": {
                "sha256": sha256(base_src),
                "lines": base_src.count("\n"),
                "extract": {"first": BASE_FIRST, "last": BASE_LAST,
                            "sha256": sha256(extract)},
            },
        },
        "outputs": {
            "gmlx/cache/kv_cache.py": {
                "source": "models/cache.py",
                "edits": [{"old": o, "new": n, "count": 1}
                          for o, n in CACHE_EDITS],
            },
            "gmlx/cache/attention.py": {
                "source": f"models/base.py:{BASE_FIRST}-{BASE_LAST}",
                "imports": BASE_IMPORTS,
                "edits": [{"old": o, "new": n, "count": 1}
                          for o, n in BASE_EDITS],
            },
        },
    }
    return files, baseline


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path,
                    help=f"mlx_vlm package directory of mlx-vlm {VERSION}")
    ap.add_argument("--dst", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "gmlx/cache",
                    help="output directory (default: gmlx/cache)")
    ap.add_argument("--check", action="store_true",
                    help="compare with the files in --dst and print the "
                         "drift baseline instead of writing")
    args = ap.parse_args()
    files, baseline = build(args.src)
    stale = []
    for name, text in files.items():
        path = args.dst / name
        out = baseline["outputs"][f"gmlx/cache/{name}"]
        out["sha256"] = sha256(text)
        if args.check:
            out["match"] = path.exists() and path.read_text() == text
            if not out["match"]:
                stale.append(name)
        else:
            path.write_text(text)
            print(f"wrote {path}")
    if args.check:
        print(json.dumps(baseline, indent=1, sort_keys=True))
    if stale:
        print("differs from the generated copy: " + ", ".join(stale),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
