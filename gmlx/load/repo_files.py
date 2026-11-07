"""A GGUF repo's model files, grouped by quant from a file listing.

The refresh script records each catalog quant from these groups, and a
pull finds the same quant in the listing at main, so both read a repo
the same way."""

from __future__ import annotations

import os
import re

from gmlx.load.discovery import quant_tag
from gmlx.load.preflight import strip_shard_suffix

# A hyphen-separated token of a file's quant suffix: UD, MTP, a codec
# (Q4_K_M, IQ3_XXS, PTQ1_0, PQ2_0, Q8_0) or a full-precision type.
_QTOKEN = re.compile(r"^(UD|MTP|[A-Z]*Q\d[A-Z0-9_]*|BF16|F16|F32|NVFP4)$")
_COMPANION = re.compile(r"(^mmproj|^mtp-|dflash|dspark|lilicorr|imatrix)", re.I)


def is_companion(path: str) -> bool:
    return bool(_COMPANION.search(os.path.basename(path))) or path.startswith("MTP/")


def quant_of(basename: str) -> str:
    """The whole quant suffix of a model file name: the trailing run of
    quant tokens (``UD-Q4_K_XL``, ``PQ2_0-MTP-Q8_0``), or "" when the name
    ends in none."""
    stem = basename[:-len(".gguf")] if basename.endswith(".gguf") else basename
    tokens = stem.split("-")
    run: list = []
    while tokens and _QTOKEN.match(tokens[-1]):
        run.insert(0, tokens.pop())
    if not run or not quant_tag(basename):
        return ""
    return "-".join(run)


def model_files(files: dict) -> dict:
    """The repo's model files grouped by split set: ``{first shard path:
    (quant, ud, [shard paths], bytes)}``. Companions and full-precision
    files are left out."""
    groups: dict = {}
    for path, size in files.items():
        if not path.endswith(".gguf") or is_companion(path):
            continue
        groups.setdefault(strip_shard_suffix(path), []).append((path, size))
    out = {}
    for stem, shards in groups.items():
        shards.sort()
        quant = quant_of(os.path.basename(stem))
        if not quant or quant in ("BF16", "F16", "F32"):
            continue
        out[shards[0][0]] = (quant, quant.startswith("UD-"), [p for p, _ in shards],
                             sum(s for _, s in shards))
    return out


def file_for_quant(groups: dict, quant: str) -> str | None:
    """The split set whose quant suffix is ``quant``, ``UD-`` prefix exact."""
    for first, (q, _ud, _shards, _n) in groups.items():
        if q.upper() == quant.upper():
            return first
    return None


def find_recorded(groups: dict, path: str, quant: str) -> str | None:
    """The split set a recorded first-shard ``path`` of ``quant`` names in a
    listing: the same path; else the same name split into another number of
    shards; else the only set of that quant. ``None`` when the quant is gone
    or more than one set could be it."""
    if path in groups:
        return path
    stem = strip_shard_suffix(path)
    same = [first for first in groups if strip_shard_suffix(first) == stem]
    if len(same) == 1:
        return same[0]
    of_quant = [first for first, (q, *_rest) in groups.items() if q.upper() == quant.upper()]
    return of_quant[0] if len(of_quant) == 1 else None
