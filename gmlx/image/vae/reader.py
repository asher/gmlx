"""The VAE reader: GGUF first, safetensors as the fallback.

A VAE is identified by tensor names and shapes, never by an arch string.
The prefixes ``vae.`` and ``first_stage_model.`` are stripped, and the
FLUX.1 autoencoder is also read in its original layout. The whole header is
checked before any tensor bytes are read. Loading is strict both ways:
every module parameter is filled, and every file tensor is consumed or on
the family's expected-unused list.

Both formats hold torch layout. Conv kernels (O, I, kH, kW) become
(O, kH, kW, I), and every weight is cast to the decode dtype as an owned,
contiguous array.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, replace
from typing import Callable

import mlx.core as mx
from mlx.utils import tree_flatten

from ..fingerprint import ImageModelError
from . import flux1_ae

STRIP_PREFIXES = ("vae.", "first_stage_model.")
FLOAT_CODECS = ("F32", "F16", "BF16")
LAYOUT = "torch-oihw"


class VAEError(ImageModelError):
    """A VAE file or request that gmlx refuses, with the reason."""


@dataclass(frozen=True)
class VaeRow:
    """The VAE an image family decodes with."""
    title: str
    config_from_shapes: Callable    # torch shapes -> config, or None
    build: Callable                 # config -> nn.Module
    latent_channels: int
    latent_space: Callable          # family constants -> config fields
    expected_unused: tuple = ()     # name prefixes left unread


def _flux1_space(c: dict) -> dict:
    return {"scaling_factor": c["vae_scaling_factor"],
            "shift_factor": c["vae_shift_factor"]}


VAES = {
    "z_image": VaeRow("the FLUX.1 autoencoder", flux1_ae.config_from_shapes,
                      flux1_ae.Flux1AE, 16, _flux1_space),
}


@dataclass(frozen=True)
class VaeHeader:
    """A VAE file's tensors before any bytes are read. Names are diffusers
    names, shapes torch order."""
    path: str
    format: str                     # "gguf" or "safetensors"
    layout: str                     # "diffusers" or "original"
    names: dict                     # diffusers name -> file name
    shapes: dict                    # diffusers name -> torch shape
    dtypes: dict                    # diffusers name -> "F32", "F16", "BF16"
    metadata: dict


def is_gguf(path: str) -> bool:
    with open(path, "rb") as f:
        return f.read(4) == b"GGUF"


def safetensors_header(path: str) -> tuple[dict, int]:
    """The parsed JSON header of a safetensors file and the offset its data
    starts at."""
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) != 8:
            raise VAEError(f"{path} is neither a GGUF nor a safetensors file")
        (n,) = struct.unpack("<Q", head)
        try:
            header = json.loads(f.read(n))
        except (ValueError, MemoryError, OverflowError):
            raise VAEError(f"{path} is neither a GGUF nor a safetensors "
                           "file") from None
    if not isinstance(header, dict):
        raise VAEError(f"{path} is neither a GGUF nor a safetensors file")
    return header, 8 + n


def _raw_header(path: str):
    if is_gguf(path):
        from gmlx.load.headerscan import scan_gguf

        scan = scan_gguf(path)
        shapes = {t.name: tuple(int(d) for d in reversed(t.shape))
                  for t in scan.tensors}
        dtypes = {t.name: t.type_name for t in scan.tensors}
        return "gguf", shapes, dtypes, dict(scan.kv)
    header, _ = safetensors_header(path)
    meta = header.pop("__metadata__", None) or {}
    shapes = {n: tuple(int(d) for d in v["shape"]) for n, v in header.items()}
    dtypes = {n: v["dtype"] for n, v in header.items()}
    return "safetensors", shapes, dtypes, dict(meta)


def _strip(names):
    for p in STRIP_PREFIXES:
        if any(n.startswith(p) for n in names):
            return {(n[len(p):] if n.startswith(p) else n): n for n in names}
    return {n: n for n in names}


def read_header(path: str) -> VaeHeader:
    """Names, shapes and dtypes under diffusers names, with no bytes read."""
    fmt, shapes, dtypes, meta = _raw_header(str(path))
    if fmt == "gguf":
        layout = meta.get("gmlx.vae.layout", LAYOUT)
        if layout != LAYOUT:
            raise VAEError(f"{path} has VAE layout {layout!r}; gmlx reads "
                           f"{LAYOUT!r}")
    names = _strip(shapes)
    layout = "diffusers"
    if flux1_ae.is_original_layout(names):
        layout = "original"
        levels = flux1_ae.original_levels(names)
        names = {flux1_ae.original_name(n, levels): f
                 for n, f in names.items()}
    out_shapes = {}
    for n, f in names.items():
        s = shapes[f]
        if layout == "original" and ".attentions.0.to_" in n and len(s) == 4:
            s = s[:2]
        out_shapes[n] = s
    return VaeHeader(str(path), fmt, layout, names, out_shapes,
                     {n: dtypes[f] for n, f in names.items()}, meta)


def vae_row(family: str) -> VaeRow:
    try:
        return VAES[family]
    except KeyError:
        known = ", ".join(sorted(VAES))
        raise VAEError(f"no VAE is known for image family {family!r}; "
                       f"known: {known}") from None


def identify(header: VaeHeader, family: str):
    """The VAE config for ``family``, or a VAEError that says what the file
    holds instead."""
    row = vae_row(family)
    cfg = row.config_from_shapes(header.shapes)
    if cfg is None:
        first = ", ".join(list(header.shapes)[:4])
        raise VAEError(f"{header.path} is not a VAE gmlx image reads: "
                       f"{len(header.shapes)} tensors, first {first}")
    if cfg.latent_channels != row.latent_channels:
        raise VAEError(f"this VAE has {cfg.latent_channels} latent channels, "
                       f"and {family} decodes with {row.title} at "
                       f"{row.latent_channels}")
    bad = sorted({d for d in header.dtypes.values()
                  if d not in FLOAT_CODECS})
    if bad:
        raise VAEError(f"VAE tensors must be F32, F16 or BF16, and this file "
                       f"holds {', '.join(bad)}")
    from ..registry import get_family

    space = row.latent_space(get_family(family).constants.values)
    return replace(cfg, **space)


def check_names(header: VaeHeader, params: dict, row: VaeRow) -> None:
    missing = sorted(set(params) - set(header.shapes))
    extra = sorted(n for n in set(header.shapes) - set(params)
                   if not any(n.startswith(u) for u in row.expected_unused))
    wrong = sorted(n for n in set(params) & set(header.shapes)
                   if _mlx_shape(header.shapes[n]) != params[n].shape)
    parts = []
    if missing:
        parts.append(f"{len(missing)} missing ({', '.join(missing[:8])})")
    if extra:
        parts.append(f"{len(extra)} unexpected ({', '.join(extra[:8])})")
    if wrong:
        parts.append(f"{len(wrong)} with another shape "
                     f"({', '.join(wrong[:8])})")
    if parts:
        raise VAEError(f"{header.path} does not fill {row.title}: "
                       + "; ".join(parts))


def _mlx_shape(shape: tuple) -> tuple:
    return (shape[0], *shape[2:], shape[1]) if len(shape) == 4 else shape


def resolve_dtype(dtype) -> mx.Dtype:
    """The decode dtype: bf16 or fp32, never fp16. None picks bf16 on chips
    with native bf16 and fp32 on M1 and M2 chips."""
    if dtype is None:
        from gmlx.load.dtypes import has_native_bfloat16

        return mx.bfloat16 if has_native_bfloat16() else mx.float32
    names = {"bf16": mx.bfloat16, "bfloat16": mx.bfloat16,
             "fp32": mx.float32, "float32": mx.float32,
             "fp16": mx.float16, "float16": mx.float16}
    dt = names.get(dtype, dtype) if isinstance(dtype, str) else dtype
    if dt == mx.float16:
        raise VAEError("the VAE never runs in float16; use bfloat16 or "
                       "float32")
    if dt not in (mx.bfloat16, mx.float32):
        raise VAEError(f"VAE dtype {dtype!r} is not bfloat16 or float32")
    return dt


def _file_arrays(header: VaeHeader) -> dict:
    if header.format == "gguf":
        import mlx_kquant as kq

        arrays, codecs, _, _ = kq.load_gguf(header.path, zero_copy=False)
        if codecs:
            raise VAEError(f"VAE tensors must be F32, F16 or BF16, and "
                           f"{header.path} holds {sorted(set(codecs.values()))}")
        return arrays
    return mx.load(header.path)


def load_vae(path, *, family: str, dtype=None):
    """The VAE module for ``family`` from a GGUF or safetensors file, with
    weights in the decode dtype. The module carries ``family``, ``source``
    and ``source_format``."""
    dt = resolve_dtype(dtype)
    row = vae_row(family)
    header = read_header(str(path))
    cfg = identify(header, family)
    vae = row.build(cfg)
    params = dict(tree_flatten(vae.parameters()))
    check_names(header, params, row)

    arrays = _file_arrays(header)
    weights = []
    for name in params:
        a = arrays.pop(header.names[name])
        if tuple(a.shape) != header.shapes[name]:
            a = a.reshape(header.shapes[name])
        if a.ndim == 4:
            a = a.transpose(0, 2, 3, 1)
        weights.append((name, mx.contiguous(a.astype(dt))))
    del arrays
    mx.eval([w for _, w in weights])
    vae.load_weights(weights, strict=True)
    vae.family = family
    vae.source = header.path
    vae.source_format = header.format
    return vae
