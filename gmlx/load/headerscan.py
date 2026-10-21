"""Fast GGUF metadata scan.

mmap + skip-parse of the KV block and tensor infos: every GGUF value is
length-prefixed, so large arrays (tokenizer vocab/merges) are seeked past
instead of materialized. Multi-GB files scan in milliseconds where gguf-py's
``GGUFReader`` spends seconds building per-element views. Tensor data is
never touched; the tensor table also yields the expected file size, which
catches truncated downloads without reading a single weight.

KVs come back as a plain dict, the shape the ``gguf_meta`` readers and
``discovery._classify_meta`` already accept.
"""

from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass

_MAGIC = b"GGUF"
_DEFAULT_ALIGNMENT = 32

# GGML type ids newer than the installed gguf-py: id -> (name, (block, type_size)).
# Id 43 has two owners; resolve it with type_43_format, not from this table.
QUANT_TYPE_FALLBACK = {
    42: ("Q2_0", (64, 18)),       # llama.cpp GGML_TYPE_Q2_0
    43: ("STQ1_0", (256, 42)),    # llama.cpp PR #22836
    # PrismML/llama.cpp (branch prism, no upstream PR): PQ2_0 is commit
    # 8bbb28b76, PTQ1_0 is e19819227. Private ids; they may move.
    142: ("PQ2_0", (128, 34)),
    143: ("PTQ1_0", (128, 28)),
}

# Two unmerged llama.cpp formats use type id 43: STQ1_0 (PR #22836, ternary
# weights) and MXFP8 (PR #30817, gather-only lookup tables, one E8M0 scale
# byte and 32 E4M3 bytes per block). Their block sizes differ 6.3x, so the
# bytes a tensor spans in the file tell them apart.
TYPE_43 = 43
TYPE_43_FORMATS = {"STQ1_0": (256, 42), "MXFP8": (32, 33)}
# Bytes past an STQ1_0 tensor at the end of a file that still read as
# STQ1_0; a longer span short of MXFP8 is a cut MXFP8 table.
TYPE_43_END_SLACK = 1 << 20

# Block geometry by type name, for the types gguf-py does not know.
FALLBACK_SIZES_BY_NAME = {
    **{name: geom for name, geom in QUANT_TYPE_FALLBACK.values()},
    **TYPE_43_FORMATS,
}


def type_43_format(n_elems: int, span: int | None, alignment: int, *,
                   at_end: bool = False) -> str:
    """The format of a type-43 tensor of ``n_elems`` values whose data spans
    ``span`` bytes: up to the next tensor, or to the end of the file when
    ``at_end``. A span that is unknown (a header prefix with no file size)
    reads as STQ1_0, the older owner; the local scan decides again.

    The format is the largest that fits in the span, as mlx-kquant reads
    it. At the end of the file a span short of MXFP8 but more than
    TYPE_43_END_SLACK past STQ1_0 is a cut MXFP8 table, which the size
    check flags."""
    if span is None:
        return "STQ1_0"
    sizes = {name: n_elems // block * tsize
             for name, (block, tsize) in TYPE_43_FORMATS.items()}
    can = {name for name, (block, _) in TYPE_43_FORMATS.items()
           if n_elems % block == 0}
    stq, mx = sizes["STQ1_0"], sizes["MXFP8"]
    slack = max(alignment, TYPE_43_END_SLACK)
    if "MXFP8" in can and (span >= mx or (
            at_end and ("STQ1_0" not in can or span >= stq + slack))):
        return "MXFP8"
    if "STQ1_0" in can and (span >= stq or at_end):
        return "STQ1_0"
    raise ValueError(
        f"type-43 tensor of {n_elems} values spans {span} bytes, which is "
        f"neither STQ1_0 ({stq} bytes) nor MXFP8 ({mx} bytes)")


def type_43_spans(entries, data_end: int | None) -> dict[int, tuple[int | None, bool]]:
    """``{index: (span, at_end)}`` for the type-43 entries of a tensor table.
    ``entries`` is ``[(ggml_type, data_offset), ...]`` in file order and
    ``data_end`` the data section's length, or None when unknown."""
    order = sorted(range(len(entries)), key=lambda i: entries[i][1])
    out: dict[int, tuple[int | None, bool]] = {}
    for pos, i in enumerate(order):
        if entries[i][0] != TYPE_43:
            continue
        if pos + 1 < len(order):
            out[i] = (entries[order[pos + 1]][1] - entries[i][1], False)
        else:
            out[i] = (None if not data_end else data_end - entries[i][1], True)
    return out

# GGUF metadata value types -> struct code (fixed-size scalars only).
_SCALAR = {
    0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
    6: "f", 7: "?", 10: "Q", 11: "q", 12: "d",
}
_STRING = 8
_ARRAY = 9


@dataclass(frozen=True)
class TensorMeta:
    name: str
    shape: tuple[int, ...]
    type_name: str
    offset: int          # relative to the data section
    nbytes: int


@dataclass(frozen=True)
class HeaderScan:
    path: str
    version: int
    kv: dict
    skipped: dict        # key -> element count, for arrays over the limit
    tensors: list[TensorMeta]
    n_tensors: int
    data_offset: int
    expected_size: int   # data_offset + end of last tensor; 0 without tensors
    size: int            # actual file size
    truncated: bool


def _read_str(buf, off: int) -> tuple[str, int]:
    (n,) = struct.unpack_from("<Q", buf, off)
    off += 8
    return bytes(buf[off:off + n]).decode("utf-8", errors="replace"), off + n


def _skip_str(buf, off: int) -> int:
    (n,) = struct.unpack_from("<Q", buf, off)
    return off + 8 + n


def _read_value(buf, off: int, vtype: int, limit: int):
    """Return (value, new_offset). Arrays longer than ``limit`` return their
    element count as an ``int`` sentinel wrapped in ``_Skipped``."""
    code = _SCALAR.get(vtype)
    if code is not None:
        (v,) = struct.unpack_from("<" + code, buf, off)
        return v, off + struct.calcsize(code)
    if vtype == _STRING:
        return _read_str(buf, off)
    if vtype == _ARRAY:
        etype, count = struct.unpack_from("<IQ", buf, off)
        off += 12
        ecode = _SCALAR.get(etype)
        if ecode is not None:
            esize = struct.calcsize(ecode)
            if count > limit:
                return _Skipped(count), off + count * esize
            vals = list(struct.unpack_from(f"<{count}{ecode}", buf, off))
            return vals, off + count * esize
        if etype == _STRING:
            if count > limit:
                for _ in range(count):
                    off = _skip_str(buf, off)
                return _Skipped(count), off
            vals = []
            for _ in range(count):
                s, off = _read_str(buf, off)
                vals.append(s)
            return vals, off
        # array of arrays: recurse (rare; no skip accounting per element)
        vals = []
        for _ in range(count):
            v, off = _read_value(buf, off, etype, limit)
            vals.append(v)
        return vals, off
    raise ValueError(f"unknown GGUF value type {vtype}")


class _Skipped(int):
    """Element count of an array that was seeked past, not read."""


def scan_gguf(path: str, *, include_tensors: bool = True,
              array_limit: int = 2048) -> HeaderScan:
    """Parse a GGUF's KV block (and optionally tensor infos) without touching
    tensor data. Raises ``ValueError`` on a malformed header."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        buf = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        try:
            return _scan(buf, path, size, include_tensors, array_limit)
        finally:
            buf.close()


def scan_bytes(buf: bytes, path: str, size: int, *,
               array_limit: int = 2048) -> HeaderScan:
    """Parse a header prefix already in memory (a remote range read).
    ``size`` is the whole file's size when known, else 0. Raises
    ``ValueError`` on a malformed header and ``struct.error`` or
    ``IndexError`` when the prefix ends before the tensor table does."""
    return _scan(buf, path, size, True, array_limit)


def _scan(buf, path, size, include_tensors, limit) -> HeaderScan:
    if buf[:4] != _MAGIC:
        raise ValueError(f"{path}: not a GGUF (bad magic)")
    version, n_tensors, n_kv = struct.unpack_from("<IQQ", buf, 4)
    if version < 2:
        raise ValueError(f"{path}: GGUF v{version} not supported")
    off = 24

    kv: dict = {}
    skipped: dict = {}
    for _ in range(n_kv):
        key, off = _read_str(buf, off)
        (vtype,) = struct.unpack_from("<I", buf, off)
        v, off = _read_value(buf, off + 4, vtype, limit)
        if isinstance(v, _Skipped):
            skipped[key] = int(v)
        else:
            kv[key] = v

    tensors: list[TensorMeta] = []
    expected = 0
    alignment = int(kv.get("general.alignment") or _DEFAULT_ALIGNMENT)
    if include_tensors:
        from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
        raw = []
        for _ in range(n_tensors):
            name, off = _read_str(buf, off)
            (n_dims,) = struct.unpack_from("<I", buf, off)
            off += 4
            dims = struct.unpack_from(f"<{n_dims}Q", buf, off)
            off += 8 * n_dims
            ttype, toff = struct.unpack_from("<IQ", buf, off)
            off += 12
            raw.append((name, dims, ttype, toff))
        data_offset = (off + alignment - 1) // alignment * alignment
        spans = type_43_spans([(r[2], r[3]) for r in raw],
                              size - data_offset if size else None)
        for i, (name, dims, ttype, toff) in enumerate(raw):
            n_elems = 1
            for d in dims:
                n_elems *= d
            if ttype == TYPE_43:
                span, at_end = spans[i]
                try:
                    tname = type_43_format(n_elems, span, alignment,
                                           at_end=at_end)
                except ValueError as e:
                    raise ValueError(f"{path}: {name}: {e}") from None
                block, tsize = TYPE_43_FORMATS[tname]
            else:
                try:
                    qt = GGMLQuantizationType(ttype)
                    block, tsize = GGML_QUANT_SIZES[qt]
                    tname = qt.name
                except (ValueError, KeyError):
                    if ttype not in QUANT_TYPE_FALLBACK:
                        raise ValueError(
                            f"{path}: unknown ggml tensor type {ttype}")
                    tname, (block, tsize) = QUANT_TYPE_FALLBACK[ttype]
            nbytes = n_elems // block * tsize
            tensors.append(TensorMeta(name, tuple(int(d) for d in dims),
                                      tname, toff, nbytes))
        if tensors:
            expected = data_offset + max(t.offset + t.nbytes for t in tensors)
    else:
        data_offset = 0
        n_tensors = int(n_tensors)

    return HeaderScan(path=path, version=version, kv=kv, skipped=skipped,
                      tensors=tensors, n_tensors=int(n_tensors),
                      data_offset=data_offset, expected_size=expected,
                      size=size, truncated=bool(expected and size < expected))
