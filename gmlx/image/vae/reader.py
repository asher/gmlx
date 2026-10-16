"""The VAE reader: GGUF first, safetensors as the fallback.

A VAE is identified by tensor names and shapes, never by an arch string.
The prefixes ``vae.`` and ``first_stage_model.`` are stripped, and the
FLUX.1, FLUX.2 and Qwen-Image autoencoders are also read in their original
layouts. The whole header is
checked before any tensor bytes are read. Loading is strict both ways:
every module parameter is filled, and every file tensor is consumed or is
one that the family's config leaves unread, at the shape the config gives
it. A file that
holds another family's VAE, or a relative of one with other channel counts
or video kernels, is refused with what it holds. A GGUF that carries the
VAE's config text is refused when a field gmlx takes from the family's
constants differs there.

Both formats hold torch layout. Conv kernels (O, I, kH, kW) become
(O, kH, kW, I), and every weight is cast to the decode dtype as an owned,
contiguous array, except those the family keeps in float32. A tensor
that stays unread may hold integers, and a GGUF holds a scalar at shape
(1,). A kernel of rank 5, (O, I, T, kH, kW), is a causal
video conv, which pads its frames in front: one frame reads its last time
slice alone, and that slice is the kernel. A GGUF holds such a kernel
flattened to rank 4, with its shape under ``comfy.gguf.orig_shape``.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, replace
from typing import Callable

import mlx.core as mx
from mlx.utils import tree_flatten

from ..fingerprint import ImageModelError
from . import flux1_ae, flux2_ae, qwen_image21_vae, qwen_image_vae
from .conv import conv_path, set_conv_path

STRIP_PREFIXES = ("vae.", "first_stage_model.")
FLOAT_CODECS = ("F32", "F16", "BF16")
LAYOUT = "torch-oihw"
CONFIG_KEY = "gmlx.vae.config_json"
ORIG_SHAPE_KEY = "comfy.gguf.orig_shape."


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
    expected_unused: Callable | None = None     # config -> the tensors a
                                                # file may hold that stay
                                                # unread, name -> torch shape
    image_channels: int | None = None   # in and out, when the family fixes
                                        # them
    relative: Callable | None = None    # torch shapes -> what a file with
                                        # this tree holds instead, or None
    config_disagreements: Callable | None = None    # vae/config.json dict
                                        # -> fields that differ from the
                                        # VAE's constants
    fp32_keep: tuple = ()               # parameters kept in float32


def _flux1_space(c: dict) -> dict:
    return {"scaling_factor": c["vae_scaling_factor"],
            "shift_factor": c["vae_shift_factor"]}


def _own_space(c: dict) -> dict:
    return {}


VAES = {
    "z_image": VaeRow("the FLUX.1 autoencoder", flux1_ae.config_from_shapes,
                      flux1_ae.Flux1AE, 16, _flux1_space),
    # The latent mean and deviation come with the VAE's own constants.
    "qwen_image21": VaeRow(
        "the Qwen-Image 2.1 autoencoder",
        qwen_image21_vae.config_from_shapes, qwen_image21_vae.QwenImage21VAE,
        qwen_image21_vae.LATENT_CHANNELS, _own_space,
        qwen_image21_vae.expected_unused,
        image_channels=qwen_image21_vae.IMAGE_CHANNELS,
        relative=qwen_image21_vae.relative,
        config_disagreements=qwen_image21_vae.config_disagreements),
    # The latent mean and deviation come with the VAE's own constants.
    "krea2": VaeRow(
        "the Qwen-Image autoencoder",
        qwen_image_vae.config_from_shapes, qwen_image_vae.QwenImageVAE,
        qwen_image_vae.LATENT_CHANNELS, _own_space,
        qwen_image_vae.expected_unused,
        image_channels=qwen_image_vae.IMAGE_CHANNELS,
        config_disagreements=qwen_image_vae.config_disagreements),
    # The latent BatchNorm statistics come with the VAE and stay float32.
    "flux3_klein": VaeRow(
        "the FLUX.2 autoencoder", flux2_ae.config_from_shapes,
        flux2_ae.Flux2AE, 32, _own_space, flux2_ae.expected_unused,
        config_disagreements=flux2_ae.config_disagreements,
        fp32_keep=flux2_ae.BN_STATS),
}


@dataclass(frozen=True)
class VaeHeader:
    """A VAE file's tensors before any bytes are read. Names are diffusers
    names, shapes torch order."""
    path: str
    format: str                     # "gguf" or "safetensors"
    layout: str                     # "diffusers", "original" (FLUX.1
                                    # and FLUX.2) or "wan" (Qwen-Image)
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
    import os

    if is_gguf(path):
        from gmlx.load.headerscan import scan_gguf

        try:
            scan = scan_gguf(path)
        except (struct.error, IndexError) as e:
            raise VAEError(
                f"{path}: truncated GGUF (the file ends inside its header, "
                f"at {os.path.getsize(path)} bytes)") from e
        if scan.truncated:
            raise VAEError(
                f"{path}: truncated GGUF (tensor table expects "
                f"{scan.expected_size} bytes, file has {scan.size})")
        shapes = {t.name: tuple(int(d) for d in reversed(t.shape))
                  for t in scan.tensors}
        dtypes = {t.name: t.type_name for t in scan.tensors}
        return ("gguf", restore_shapes(path, shapes, scan.kv), dtypes,
                dict(scan.kv))
    header, start = safetensors_header(path)
    meta = header.pop("__metadata__", None) or {}
    try:
        end = max((int(v["data_offsets"][1]) for v in header.values()),
                  default=0)
        shapes = {n: tuple(int(d) for d in v["shape"])
                  for n, v in header.items()}
        dtypes = {n: v["dtype"] for n, v in header.items()}
    except (KeyError, IndexError, TypeError, ValueError):
        raise VAEError(f"{path} is neither a GGUF nor a safetensors "
                       "file") from None
    size = os.path.getsize(path)
    if start + end > size:
        raise VAEError(f"{path}: truncated safetensors (its header expects "
                       f"{start + end} bytes, file has {size})")
    return "safetensors", shapes, dtypes, dict(meta)


def restore_shapes(path: str, shapes: dict, kv: dict) -> dict:
    """``shapes`` with the shape each ``comfy.gguf.orig_shape`` key stores
    for a tensor the GGUF holds flattened."""
    out = dict(shapes)
    for key, value in kv.items():
        name = key[len(ORIG_SHAPE_KEY):]
        if not key.startswith(ORIG_SHAPE_KEY) or name not in out:
            continue
        shape = tuple(int(v) for v in value)
        if math.prod(shape) != math.prod(out[name]):
            raise VAEError(f"{path}: {name} holds {math.prod(out[name])} "
                           f"values, and its stored shape {shape} has "
                           f"{math.prod(shape)}")
        out[name] = shape
    return out


def strip_names(names) -> dict:
    """Stripped name to file name, after the first VAE prefix any name
    carries."""
    for p in STRIP_PREFIXES:
        if any(n.startswith(p) for n in names):
            return {(n[len(p):] if n.startswith(p) else n): n for n in names}
    return {n: n for n in names}


def diffusers_names(names: dict) -> tuple[str, dict]:
    """The layout of stripped names (name -> file name) and the names
    under diffusers names: "original" for the FLUX.1 and FLUX.2
    autoencoders' own layout, "wan" for the Qwen-Image autoencoder's, else
    "diffusers"."""
    if flux1_ae.is_original_layout(names):
        levels = flux1_ae.original_levels(names)
        return "original", {flux1_ae.original_name(n, levels): f
                            for n, f in names.items()}
    if qwen_image_vae.is_original_layout(names):
        return "wan", qwen_image_vae.diffusers_names(names)
    return "diffusers", names


def read_header(path: str) -> VaeHeader:
    """Names, shapes and dtypes under diffusers names, with no bytes read."""
    fmt, shapes, dtypes, meta = _raw_header(str(path))
    if fmt == "gguf":
        layout = meta.get("gmlx.vae.layout", LAYOUT)
        if layout != LAYOUT:
            raise VAEError(f"{path} has VAE layout {layout!r}; gmlx reads "
                           f"{LAYOUT!r}")
    layout, names = diffusers_names(strip_names(shapes))
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
        raise VAEError(_holds(header, family, row))
    if cfg.latent_channels != row.latent_channels:
        raise VAEError(f"this VAE has {cfg.latent_channels} latent channels, "
                       f"and {family} decodes with {row.title} at "
                       f"{row.latent_channels}")
    if row.image_channels is not None and (
            cfg.in_channels, cfg.out_channels) != (row.image_channels,) * 2:
        raise VAEError(f"this VAE has {cfg.in_channels} image channels in "
                       f"and {cfg.out_channels} out, and {family} decodes "
                       f"with {row.title} at {row.image_channels}")
    text = header.metadata.get(CONFIG_KEY)
    if header.format == "gguf" and text is not None:
        check_config(header.path, text, family)
    unread = row.expected_unused(cfg) if row.expected_unused else {}
    bad = sorted({d for n, d in header.dtypes.items()
                  if d not in FLOAT_CODECS and n not in unread})
    if bad:
        raise VAEError(f"VAE tensors must be F32, F16 or BF16, and this file "
                       f"holds {', '.join(bad)}")
    from ..registry import get_family

    space = row.latent_space(get_family(family).constants.values)
    return replace(cfg, **space)


def check_config(path: str, text: str, family: str) -> None:
    """Refuse the vae/config.json text of ``path`` when a field that gmlx
    takes from the constants of ``family``'s VAE differs in it."""
    row = vae_row(family)
    if row.config_disagreements is None:
        return
    try:
        config = json.loads(text)
    except ValueError:
        config = None
    if not isinstance(config, dict):
        raise VAEError(f"{path}: its VAE config is not a JSON object")
    bad = row.config_disagreements(config)
    if bad:
        raise VAEError(f"{path}: its VAE config differs from {row.title} in "
                       f"{', '.join(bad)}")


def _holds(header: VaeHeader, family: str, row: VaeRow) -> str:
    """Why ``header`` is not ``family``'s VAE, naming what it holds when
    that is another family's VAE or a relative of one."""
    for name, other in VAES.items():
        if name != family and other.config_from_shapes(
                header.shapes) is not None:
            return (f"{header.path} holds {other.title}, the {name} VAE, "
                    f"and {family} decodes with {row.title}")
    for other in VAES.values():
        what = other.relative and other.relative(header.shapes)
        if what:
            return (f"{header.path} holds {what}, and {family} decodes "
                    f"with {row.title}")
    first = ", ".join(list(header.shapes)[:4])
    return (f"{header.path} is not a VAE gmlx image reads: "
            f"{len(header.shapes)} tensors, first {first}")


def check_names(header: VaeHeader, params: dict, row: VaeRow, cfg) -> None:
    unread = row.expected_unused(cfg) if row.expected_unused else {}
    missing = sorted(set(params) - set(header.shapes))
    extra = sorted(set(header.shapes) - set(params) - set(unread))
    wrong = sorted(n for n in set(params) & set(header.shapes)
                   if _mlx_shape(header.shapes[n]) != params[n].shape)
    # A tensor that stays unread is still held to its shape, so a file of
    # another model does not pass on the names alone.
    wrong += sorted(n for n, shape in unread.items() if n in header.shapes
                    and not _same_shape(header, n, shape))
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


def _same_shape(header: VaeHeader, name: str, shape: tuple) -> bool:
    """Whether the file's ``name`` has torch shape ``shape``. A GGUF holds
    a scalar at shape (1,)."""
    got = tuple(header.shapes[name])
    return got == tuple(shape) or (header.format == "gguf" and shape == ()
                                   and got == (1,))


def _mlx_shape(shape: tuple) -> tuple:
    if len(shape) == 5:
        return (shape[0], *shape[3:], shape[1])
    return (shape[0], *shape[2:], shape[1]) if len(shape) == 4 else shape


def frame_kernel(a: mx.array) -> mx.array:
    """The kernel a causal video conv (O, I, T, kH, kW) runs on one frame:
    its last time slice."""
    return a[:, :, -1]


def resolve_dtype(dtype) -> mx.Dtype:
    """The decode dtype: bf16 or fp32, never fp16. None picks bf16 on chips
    with native bf16 and fp32 on M1 and M2 chips."""
    if dtype is None:
        from ..dtypes import vae_dtype

        return vae_dtype()
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


def _file_arrays(header: VaeHeader, skip=()) -> dict:
    """The file's arrays by file name. A GGUF loads none for the file names
    in ``skip``, so an unread tensor of a type MLX does not load stays
    unread."""
    if header.format == "gguf":
        import mlx_kquant as kq

        arrays, codecs, _, _ = kq.load_gguf(header.path, zero_copy=False,
                                            skip=list(skip))
        if codecs:
            raise VAEError(f"VAE tensors must be F32, F16 or BF16, and "
                           f"{header.path} holds {sorted(set(codecs.values()))}")
        return arrays
    return mx.load(header.path)


def load_vae(path, *, family: str, dtype=None, conv: str | None = None):
    """The VAE module for ``family`` from a GGUF or safetensors file, with
    weights in the decode dtype. ``conv`` picks the conv path, "kq",
    "exact" or "winograd" (``conv.CONV`` when None; see ``conv``). The module carries
    ``family``, ``source``, ``source_format`` and ``conv``."""
    dt = resolve_dtype(dtype)
    conv = conv_path(conv)
    row = vae_row(family)
    header = read_header(str(path))
    cfg = identify(header, family)
    vae = row.build(cfg)
    params = dict(tree_flatten(vae.parameters()))
    check_names(header, params, row, cfg)

    unread = row.expected_unused(cfg) if row.expected_unused else {}
    arrays = _file_arrays(header, [header.names[n] for n in unread
                                   if n in header.names])
    keep = set(row.fp32_keep)
    weights = []
    for name in params:
        a = arrays.pop(header.names[name])
        if tuple(a.shape) != header.shapes[name]:
            a = a.reshape(header.shapes[name])
        if a.ndim == 5:
            a = frame_kernel(a)
        if a.ndim == 4:
            a = a.transpose(0, 2, 3, 1)
        weights.append((name, mx.contiguous(
            a.astype(mx.float32 if name in keep else dt))))
    del arrays
    mx.eval([w for _, w in weights])
    vae.load_weights(weights, strict=True)
    vae.family = family
    vae.source = header.path
    vae.source_format = header.format
    vae.conv = set_conv_path(vae, conv)
    return vae
