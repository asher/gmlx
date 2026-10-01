"""The lossless VAE packer: safetensors in, a VAE GGUF out.

Tensor bytes are copied unchanged under their official names in torch
layout, so other GGUF readers can use the file. The default dtype is the
source dtype, which is the lossless case. A downcast is an explicit option
and records ``gmlx.vae.lossless = false``. The packer writes to a partial
file, reads it back, compares every tensor's sha256 and the metadata with
what it meant to write, and only then moves it into place.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np

from ..fingerprint import canonical_bf16
from .reader import (VAES, VAEError, check_names, identify, read_header,
                     safetensors_header)

PACKER_VERSION = "1"

_NP = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16}


@dataclass(frozen=True)
class PackReport:
    path: str
    family: str
    tensors: int
    payload_bytes: int
    source_sha256: str
    dtype: str                      # "source", or the GGUF dtype written
    lossless: bool
    metadata: dict
    digests: dict = field(repr=False)   # name -> sha256 of payload bytes


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(16 << 20):
            h.update(chunk)
    return h.hexdigest()


def _family(header, family: str | None) -> str:
    names = [family] if family else sorted(VAES)
    errors = []
    for name in names:
        try:
            cfg = identify(header, name)
        except VAEError as e:
            errors.append(str(e))
            continue
        row = VAES[name]
        params = dict(_tree_params(row.build(cfg)))
        check_names(header, params, row)
        return name
    raise VAEError(errors[0] if len(errors) == 1 else
                   f"{header.path} matches no image family's VAE: "
                   + "; ".join(errors))


def _tree_params(module):
    from mlx.utils import tree_flatten

    return tree_flatten(module.parameters())


def _convert(raw: np.ndarray, src: str, dst: str) -> tuple[np.ndarray, bool]:
    """``raw`` (the source bytes as uint8) in dtype ``dst``, and whether
    the conversion is exact."""
    if dst == src:
        return raw.view(_NP[src]), True
    if src == "BF16":
        f32 = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    else:
        f32 = raw.view(_NP[src]).astype(np.float32)
    if dst == "F32":
        return f32, True
    if dst == "BF16":
        return np.frombuffer(canonical_bf16(f32.tobytes(), "F32"),
                             np.uint16), False
    out = f32.astype(np.float16)
    if not np.array_equal(np.isfinite(out), np.isfinite(f32)):
        raise VAEError("a value overflows float16; keep the source dtype")
    return out, False


def pack_vae(src, dst, *, source_repo: str, source_revision: str,
             config_json: str, family: str | None = None,
             dtype: str | None = None, overwrite: bool = False) -> PackReport:
    """Pack the safetensors VAE ``src`` into the GGUF ``dst``.

    ``family`` defaults to the one family whose VAE the file fills.
    ``dtype`` is None for the source dtype, or "F32", "F16" or "BF16".
    """
    src, dst = str(src), str(dst)
    if not os.path.isfile(src):
        raise VAEError(f"{src} is not a file")
    with open(src, "rb") as f:
        if f.read(4) == b"GGUF":
            raise VAEError(f"{src} is already a GGUF")
    if os.path.exists(dst) and not overwrite:
        raise VAEError(f"{dst} exists")
    if dtype is not None and dtype not in _NP:
        raise VAEError(f"pack dtype {dtype!r} is not F32, F16 or BF16")
    try:
        json.loads(config_json)
    except ValueError as e:
        raise VAEError(f"config_json is not JSON: {e}") from None

    vh = read_header(src)
    fam = _family(vh, family)
    header, data_start = safetensors_header(src)
    header.pop("__metadata__", None)
    order = sorted(header, key=lambda n: header[n]["data_offsets"][0])

    meta = {
        "general.architecture": "autoencoder",
        "general.type": "vae",
        "gmlx.vae.family": fam,
        "gmlx.vae.source_repo": source_repo,
        "gmlx.vae.source_revision": source_revision,
        "gmlx.vae.source_file": os.path.basename(src),
        "gmlx.vae.source_sha256": _sha256_file(src),
        "gmlx.vae.layout": "torch-oihw",
        "gmlx.vae.config_json": config_json,
        "gmlx.packer.version": PACKER_VERSION,
    }
    mm = np.memmap(src, dtype=np.uint8, mode="r")
    tensors, digests = [], {}
    lossless = True
    for name in order:
        info = header[name]
        a, b = info["data_offsets"]
        raw = mm[data_start + a:data_start + b]
        shape = tuple(int(d) for d in info["shape"]) or (1,)
        out_dt = dtype or info["dtype"]
        data, exact = _convert(raw, info["dtype"], out_dt)
        lossless = lossless and exact
        data = data.reshape(shape)
        digests[name] = hashlib.sha256(data.tobytes()).hexdigest()
        tensors.append((name, data, shape, out_dt))
    if not lossless:
        meta["gmlx.vae.lossless"] = False

    tmp = dst + ".partial"
    try:
        _write(tmp, meta, tensors)
        _verify(tmp, meta, tensors, digests)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return PackReport(
        path=dst, family=fam, tensors=len(tensors),
        payload_bytes=sum(t[1].nbytes for t in tensors),
        source_sha256=meta["gmlx.vae.source_sha256"],
        dtype=dtype or "source", lossless=lossless, metadata=meta,
        digests=digests)


def _write(path: str, meta: dict, tensors: list) -> None:
    import gguf

    w = gguf.GGUFWriter(path, meta["general.architecture"])
    w.add_type(meta["general.type"])
    for key, value in meta.items():
        if key.startswith("general."):
            continue
        if isinstance(value, bool):
            w.add_bool(key, value)
        else:
            w.add_string(key, value)
    for name, data, shape, dt in tensors:
        w.add_tensor(name, data, raw_shape=shape,
                     raw_dtype=getattr(gguf.GGMLQuantizationType, dt))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def _field(reader, key):
    f = reader.fields.get(key)
    return None if f is None else f.contents()


def _verify(path: str, meta: dict, tensors: list, digests: dict) -> None:
    import gguf

    r = gguf.GGUFReader(path)
    for key, value in meta.items():
        got = _field(r, key)
        if got != value:
            raise VAEError(f"packed {key} reads back as {got!r}, "
                           f"not {value!r}")
    got = {t.name: t for t in r.tensors}
    if set(got) != set(digests):
        raise VAEError("the packed tensor names differ from the source")
    for name, _, shape, dt in tensors:
        t = got[name]
        if (t.tensor_type.name != dt or
                tuple(int(d) for d in reversed(t.shape)) != shape):
            raise VAEError(f"packed {name} reads back as {t.tensor_type.name} "
                           f"{tuple(reversed(t.shape))}")
        if hashlib.sha256(t.data.tobytes()).hexdigest() != digests[name]:
            raise VAEError(f"packed {name} differs from the source bytes")
