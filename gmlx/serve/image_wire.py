"""The wire format of the image step contract.

A body is a safetensors container: an 8 byte little-endian header length,
a JSON header that names each tensor's dtype, shape and byte range, and
the raw bytes. The request fields travel as a JSON string under
``__metadata__["gmlx"]``. Both sides write the header with no spaces, the
tensors in the order given, and pad it with spaces to a multiple of 8
bytes, so one set of tensors and fields gives one body.

numpy has no bfloat16, so a BF16 tensor reads as uint16 and is viewed as
bfloat16 when it becomes an MLX array.

A tensor's hash is the sha256 of its dtype name, its shape as JSON and its
raw bytes, and names a conditioning or an adapter matrix the server keeps.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from typing import Any

CONTENT_TYPE = "application/vnd.gmlx.tensors+safetensors"
CONTRACT_HEADER = "X-Gmlx-Steps-Contract"
CONTRACT = (1, 0)
PREFIX = "/internal/image-steps/v1"
# The dtypes the contract carries, with numpy's name and the item size.
DTYPES = {"F32": ("<f4", 4), "F16": ("<f2", 2), "BF16": ("<u2", 2),
          "U8": ("u1", 1), "BOOL": ("?", 1)}
_SHORT = {"F32": "f32", "F16": "f16", "BF16": "bf16", "U8": "u8",
          "BOOL": "bool"}
MAX_HEADER_BYTES = 1 << 24


class WireError(ValueError):
    """A body or header the contract refuses, with its HTTP status."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class WireTensor:
    """One tensor of a body: its dtype name, its shape and a view of its
    raw bytes."""
    dtype: str
    shape: tuple
    raw: memoryview

    @property
    def nbytes(self) -> int:
        return len(self.raw)

    def hash(self) -> str:
        return tensor_hash(self.dtype, self.shape, self.raw)

    def numpy(self):
        """The tensor as a numpy array, BF16 as its uint16 bits."""
        import numpy as np

        np_dt, _ = DTYPES[self.dtype]
        return np.frombuffer(self.raw, dtype=np_dt).reshape(self.shape)

    def mx(self):
        """The tensor as an MLX array of its own dtype."""
        import mlx.core as mx

        a = mx.array(self.numpy())
        return a.view(mx.bfloat16) if self.dtype == "BF16" else a


def tensor_hash(dtype: str, shape, raw) -> str:
    h = hashlib.sha256()
    h.update(dtype.encode())
    h.update(json.dumps([int(d) for d in shape]).encode())
    h.update(raw)
    return h.hexdigest()


def short_dtype(dtype: str) -> str:
    """The contract's short name of a dtype: f32, f16, bf16, u8, bool."""
    return _SHORT[dtype]


def read(body, *, max_bytes: int | None = None) -> tuple[dict, dict]:
    """``(tensors, fields)`` of a body: each tensor as a :class:`WireTensor`
    by name, in header order, and the request fields of its metadata.
    Raises :class:`WireError` for a body that is not a well formed
    container of the contract's dtypes, and 413 past ``max_bytes``."""
    view = memoryview(body)
    if max_bytes is not None and len(view) > max_bytes:
        raise WireError(f"the body is {len(view)} bytes, over the limit of "
                        f"{max_bytes}", 413)
    if len(view) < 8:
        raise WireError("the body is shorter than a safetensors header")
    (n,) = struct.unpack("<Q", view[:8])
    if n > MAX_HEADER_BYTES or 8 + n > len(view):
        raise WireError("the safetensors header length is out of range")
    try:
        header = json.loads(bytes(view[8:8 + n]))
    except ValueError:
        raise WireError("the safetensors header is not JSON") from None
    if not isinstance(header, dict):
        raise WireError("the safetensors header is not a JSON object")
    meta = header.pop("__metadata__", None) or {}
    if not isinstance(meta, dict) or not all(
            isinstance(v, str) for v in meta.values()):
        raise WireError("__metadata__ must map names to strings")
    data = view[8 + n:]
    tensors: dict[str, WireTensor] = {}
    spans = []
    for name, spec in header.items():
        if not isinstance(spec, dict) or set(spec) != {
                "dtype", "shape", "data_offsets"}:
            raise WireError(f"tensor {name!r}: needs dtype, shape and "
                            "data_offsets")
        dt = spec["dtype"]
        if dt not in DTYPES:
            raise WireError(f"tensor {name!r}: dtype {dt!r} is not one of "
                            f"{', '.join(DTYPES)}")
        shape = spec["shape"]
        offs = spec["data_offsets"]
        if (not isinstance(shape, list) or not all(
                isinstance(d, int) and not isinstance(d, bool) and d >= 0
                for d in shape)
                or not isinstance(offs, list) or len(offs) != 2
                or not all(isinstance(o, int) and not isinstance(o, bool)
                           for o in offs)):
            raise WireError(f"tensor {name!r}: bad shape or data_offsets")
        start, end = offs
        count = 1
        for d in shape:
            count *= d
        if not 0 <= start <= end <= len(data) or \
                end - start != count * DTYPES[dt][1]:
            raise WireError(f"tensor {name!r}: data_offsets do not hold "
                            f"{count} {dt} values in the body")
        spans.append((start, end, name))
        tensors[name] = WireTensor(dt, tuple(shape), data[start:end])
    spans.sort()
    for (_s0, e0, a), (s1, _e1, b) in zip(spans, spans[1:]):
        if s1 < e0:
            raise WireError(f"tensors {a!r} and {b!r} overlap")
    fields: dict = {}
    if "gmlx" in meta:
        try:
            fields = json.loads(meta["gmlx"])
        except ValueError:
            raise WireError("__metadata__.gmlx is not JSON") from None
        if not isinstance(fields, dict):
            raise WireError("__metadata__.gmlx is not a JSON object")
    return tensors, fields


def _raw(a) -> tuple[str, list, bytes]:
    """dtype name, shape and bytes of a numpy or MLX array."""
    import numpy as np

    mod = type(a).__module__
    if mod.startswith("mlx"):
        import mlx.core as mx

        if a.dtype == mx.bfloat16:
            return "BF16", list(a.shape), np.array(a.view(mx.uint16)).tobytes()
        a = np.array(a)
    a = np.asarray(a)
    dt = {("f", 4): "F32", ("f", 2): "F16", ("u", 1): "U8",
          ("b", 1): "BOOL"}.get((a.dtype.kind, a.dtype.itemsize))
    if dt is None:
        raise WireError(f"dtype {a.dtype} is not one the contract carries")
    return dt, list(a.shape), np.ascontiguousarray(
        a, dtype=DTYPES[dt][0]).tobytes()


def write(tensors: dict, fields: dict | None = None) -> bytes:
    """A body of ``tensors`` (numpy or MLX arrays by name, BF16 as an MLX
    bfloat16 array or a ``(\"BF16\", shape, bytes)`` triple) and the request
    ``fields``."""
    header: dict[str, Any] = {}
    chunks = []
    off = 0
    for name, a in tensors.items():
        if isinstance(a, tuple):
            dt, shape, raw = a[0], list(a[1]), bytes(a[2])
        else:
            dt, shape, raw = _raw(a)
        header[name] = {"dtype": dt, "shape": shape,
                        "data_offsets": [off, off + len(raw)]}
        chunks.append(raw)
        off += len(raw)
    if fields is not None:
        header["__metadata__"] = {"gmlx": json.dumps(
            fields, sort_keys=True, separators=(",", ":"))}
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    return struct.pack("<Q", len(hj)) + hj + b"".join(chunks)


def parse_version(value: str | None) -> tuple[int, int] | None:
    """``(major, minor)`` of a contract header, or None when absent."""
    if value is None:
        return None
    major, _, minor = value.strip().partition(".")
    try:
        return int(major), int(minor or 0)
    except ValueError:
        raise WireError(f"{CONTRACT_HEADER} must be MAJOR.MINOR, not "
                        f"{value!r}") from None


def check_version(value: str | None) -> None:
    """Refuse a request whose contract major is not the server's, naming
    the side to upgrade. A request without the header is refused too."""
    got = parse_version(value)
    if got is None:
        raise WireError(f"the request has no {CONTRACT_HEADER} header; send "
                        f"{CONTRACT[0]}.{CONTRACT[1]}")
    if got[0] != CONTRACT[0]:
        side = "the gmlx server" if got[0] > CONTRACT[0] else "the client"
        raise WireError(
            f"step contract {got[0]}.{got[1]} is not served; this server "
            f"speaks {CONTRACT[0]}.{CONTRACT[1]}. Upgrade {side}.", 400)
