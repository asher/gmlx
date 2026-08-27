"""GGUF preflight: shard discovery, codec classification (unsupported-codec
hard-fail), and the architecture gate - everything that must pass *before*
``kq.load_gguf`` runs.

``kq.load_gguf`` raises a cryptic ``unsupported type <N>`` for any tensor whose
codec has no kernel (e.g. the ternary TQ family). Preflight reads the tensor-type
enum from the GGUF header via gguf-py first, so the refusal names the codec (e.g.
``TQ1_0``) and lists what *is* supported - and the arch gate fires before any
tensor bytes are touched.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass

from .arch_table import ArchEntry
from .arch_table import gate as _gate_arch

# The 21 codecs the kq.* kernels implement (GGML type names).
SUPPORTED_QUANT_TYPES = frozenset({
    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K",
    "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0",
    "IQ4_NL", "IQ4_XS", "IQ3_S", "IQ3_XXS",
    "IQ2_XXS", "IQ2_XS", "IQ2_S",
    "IQ1_S", "IQ1_M",
    "MXFP4", "NVFP4",
})
# Micro-scaling float codecs that additionally have MLX-native packed kernels.
# Served like every other codec above (zero-copy ggml wire bytes through the
# kq kernels), or de-interleaved into MLX's packed layout at load
# (gguf/native_fp.py) and dispatched through mx.quantized_matmul /
# mx.gather_qmm with mode="mxfp4"/"nvfp4"; GMLX_NATIVE_FP picks the
# layout (see loader).
NATIVE_FP_TYPES = frozenset({"MXFP4", "NVFP4"})
# Non-quantized tensor types loaded natively; step-7 cast policy handles dtype.
NATIVE_TYPES = frozenset({
    "F32", "F16", "BF16", "F64", "I8", "I16", "I32", "I64",
})

# Split-GGUF shard suffix (`<name>-00001-of-NNNNN.gguf`) - the one definition;
# discovery/manage/launch import it (and the helpers below) rather than
# recompiling their own.
SPLIT_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


def shard_names(filename: str) -> list[str]:
    """Expand a split-GGUF name to every shard name (pure string work, no
    filesystem lookup); non-split -> ``[filename]``. Raises ``ValueError`` on a
    zero shard count."""
    m = SPLIT_SHARD_RE.search(filename)
    if not m:
        return [filename]
    total = int(m.group(2))
    if total < 1:
        raise ValueError(
            f"malformed split-GGUF name (zero shard count): {filename}")
    prefix = filename[:m.start()]
    return [f"{prefix}-{i:05d}-of-{total:05d}.gguf" for i in range(1, total + 1)]


def is_first_shard(filename: str) -> bool:
    """A non-split name, or the first shard of a split set (the canonical entry)."""
    m = SPLIT_SHARD_RE.search(filename)
    return not m or int(m.group(1)) == 1


def strip_shard_suffix(name: str) -> str:
    """Collapse a ``-NNNNN-of-MMMMM.gguf`` suffix back to ``.gguf``."""
    return SPLIT_SHARD_RE.sub(".gguf", name)


def find_split_shards(gguf_path: str) -> list[str]:
    """Detect a split GGUF and return all shard paths in order.

    Split GGUFs follow ``<name>-00001-of-NNNNN.gguf``. The first shard holds the
    full metadata block; tensor data spans all shards (each shard holds a disjoint
    subset). Returns ``[gguf_path]`` for a non-split file.

    A split set is either whole or broken: an absent shard means missing tensors,
    and the load merges shards then ``load_weights(strict=False)`` - which would
    silently leave the gaps at random init. So a missing shard is a **hard error**
    here (named, with the count) rather than a quiet partial load.
    """
    gguf_path = os.path.expanduser(gguf_path)
    m = SPLIT_SHARD_RE.search(gguf_path)
    if not m:
        return [gguf_path]
    total = int(m.group(2))
    prefix = gguf_path[:m.start()]
    shards = [prefix + f"-{i:05d}-of-{total:05d}.gguf" for i in range(1, total + 1)]
    missing = [p for p in shards if not os.path.isfile(p)]
    if missing:
        names = ", ".join(os.path.basename(p) for p in missing[:3])
        more = "" if len(missing) <= 3 else f", +{len(missing) - 3} more"
        raise FileNotFoundError(
            f"incomplete split GGUF: {len(missing)}/{total} shard(s) missing "
            f"({names}{more}). Re-pull or complete the download before loading.")
    return shards


class UnsupportedCodecError(Exception):
    """A GGUF containing a tensor codec with no kquant kernel (e.g. ternary TQ)."""

    def __init__(self, arch: str, unsupported: dict[str, int]):
        self.arch = arch
        self.unsupported = dict(unsupported)
        listing = ", ".join(
            f"{k}x{unsupported[k]}" for k in sorted(unsupported))
        n = sum(unsupported.values())
        super().__init__(
            f"GGUF (arch={arch!r}) uses codecs this build can't load - "
            f"unsupported tensor types: {listing} ({n} tensors). "
            "Supported codecs: q2_k-q6_k, q4_0/q4_1/q5_0/q5_1/q8_0, and the "
            "iq1/iq2/iq3/iq4 families. Re-quantize to a supported codec "
            "(e.g. Q4_K_M / Q5_K_M / Q6_K).")


@dataclass(frozen=True)
class Preflight:
    arch: str
    entry: ArchEntry
    shards: list[str]
    codec_histogram: dict[str, int]   # GGML type name -> tensor count
    n_tensors: int
    n_params: int = 0                 # logical elements across all tensors


def preflight(gguf_path: str, *, arch: str | None = None,
              hf_source: str | None = None) -> Preflight:
    """Validate a GGUF before loading: discover shards, refuse IQ/unsupported
    codecs (naming them), refuse truncated files, and gate on the architecture.
    Reads only the GGUF header - no tensor data - so it is cheap on multi-GB
    files.

    Raises ``UnsupportedCodecError``, ``arch_table.UnsupportedArchError``, or
    ``ValueError`` for a truncated/incomplete download.
    """
    from .headerscan import scan_gguf

    shards = find_split_shards(gguf_path)
    scan0 = scan_gguf(shards[0])
    detected = arch or scan0.kv.get("general.architecture")
    if not detected:
        raise ValueError("GGUF missing 'general.architecture' KV field - "
                         "can't detect arch")

    hist: dict[str, int] = {}
    unsupported: dict[str, int] = {}
    n_params = 0
    for i, shard in enumerate(shards):
        scan = scan0 if i == 0 else scan_gguf(shard)
        if scan.truncated:
            raise ValueError(
                f"{shard}: truncated GGUF (tensor table expects "
                f"{scan.expected_size} bytes, file has {scan.size}) - "
                "incomplete download?")
        for t in scan.tensors:
            tname = t.type_name
            hist[tname] = hist.get(tname, 0) + 1
            n_params += math.prod(t.shape)
            if (tname not in SUPPORTED_QUANT_TYPES
                    and tname not in NATIVE_TYPES
                    and tname not in NATIVE_FP_TYPES):
                unsupported[tname] = unsupported.get(tname, 0) + 1

    if unsupported:
        raise UnsupportedCodecError(detected, unsupported)

    entry = _gate_arch(detected, hf_source=hf_source)
    return Preflight(arch=detected, entry=entry, shards=shards,
                     codec_histogram=hist, n_tensors=sum(hist.values()),
                     n_params=n_params)
