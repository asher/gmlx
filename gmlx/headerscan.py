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
        for _ in range(n_tensors):
            name, off = _read_str(buf, off)
            (n_dims,) = struct.unpack_from("<I", buf, off)
            off += 4
            dims = struct.unpack_from(f"<{n_dims}Q", buf, off)
            off += 8 * n_dims
            ttype, toff = struct.unpack_from("<IQ", buf, off)
            off += 12
            try:
                qt = GGMLQuantizationType(ttype)
                block, tsize = GGML_QUANT_SIZES[qt]
                tname = qt.name
            except (ValueError, KeyError):
                raise ValueError(f"{path}: unknown ggml tensor type {ttype}")
            n_elems = 1
            for d in dims:
                n_elems *= d
            nbytes = n_elems // block * tsize
            tensors.append(TensorMeta(name, tuple(int(d) for d in dims),
                                      tname, toff, nbytes))
        data_offset = (off + alignment - 1) // alignment * alignment
        if tensors:
            expected = data_offset + max(t.offset + t.nbytes for t in tensors)
    else:
        data_offset = 0
        n_tensors = int(n_tensors)

    return HeaderScan(path=path, version=version, kv=kv, skipped=skipped,
                      tensors=tensors, n_tensors=int(n_tensors),
                      data_offset=data_offset, expected_size=expected,
                      size=size, truncated=bool(expected and size < expected))
