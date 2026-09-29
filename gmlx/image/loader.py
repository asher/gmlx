"""Load an image DiT GGUF into its family's module tree.

Module attribute names equal the GGUF tensor names after the family's one
prefix strip, so no remap table exists. Quantized matrices become
``KQuantLinear`` leaves over zero-copy wire bytes. The shared text loader
installs them and casts float tensors, with its text-model patches off.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .fingerprint import ImageModelError, Identified, identify, view_from_scans

_ACT_DTYPES = (mx.bfloat16, mx.float16)
_ORIG_SHAPE_KEY = "comfy.gguf.orig_shape."


@dataclass(frozen=True)
class DitInfo:
    identified: Identified
    shards: tuple
    act_dtype: mx.Dtype
    n_kquant: int                   # KQuantLinear leaves after install
    n_owned_casts: int              # tensors load_dit cast before the loop
    file_bytes: int


def _names_or_none(names: list[str]) -> str:
    shown = ", ".join(names[:6])
    more = f" (+{len(names) - 6} more)" if len(names) > 6 else ""
    return (shown + more) if names else "none"


def check_names(model: nn.Module, names, title: str) -> None:
    """Raise unless the model's parameter names equal the tensor names."""
    params = {k for k, _ in tree_flatten(model.parameters())}
    tensors = set(names)
    if params == tensors:
        return
    missing = sorted(params - tensors)
    extra = sorted(tensors - params)
    raise ImageModelError(
        f"the {title} module tree and the file disagree. Parameters without "
        f"a tensor ({len(missing)}): {_names_or_none(missing)}. Tensors "
        f"without a parameter ({len(extra)}): {_names_or_none(extra)}.")


def owned_casts(weights: dict, codecs: dict, fp32_keep: tuple,
                act: mx.Dtype) -> set[str]:
    """Casts the shared loop does not make, in place. BF16 tensors under a
    keep substring go to fp32. Under fp16 every other BF16 tensor goes to
    fp16 through fp32, since a cast between two 2-byte types can donate the
    buffer into the read-only mapping. Returns the names cast."""
    cast: set[str] = set()
    for k in list(weights):
        v = weights[k]
        if k in codecs or v.dtype != mx.bfloat16:
            continue
        if any(s in k for s in fp32_keep):
            weights[k] = v.astype(mx.float32)
        elif act == mx.float16:
            weights[k] = v.astype(mx.float32).astype(mx.float16)
        else:
            continue
        cast.add(k)
    return cast


def load_dit(path: str, *, family: str | None = None,
             variant: str | None = None, act_dtype: mx.Dtype | None = None,
             log=print) -> tuple[nn.Module, DitInfo]:
    """Load a DiT GGUF. Raises ``ImageModelError`` for a file gmlx image
    refuses and ``UnsupportedCodecError`` for a codec it cannot run."""
    from mlx_kquant.nn import KQuantLinear

    from gmlx.load.headerscan import scan_gguf
    from gmlx.load.loader import (
        _active_now, _install_and_load, collect_after_load,
        weights_source_key)
    from gmlx.load.preflight import (
        NATIVE_TYPES, SUPPORTED_QUANT_TYPES, UnsupportedCodecError,
        find_split_shards)
    from gmlx.load.wire import _RemapDict, load_gguf_wire_bytes

    act = mx.bfloat16 if act_dtype is None else act_dtype
    if act not in _ACT_DTYPES:
        raise ValueError(f"act_dtype must be bfloat16 or float16, not {act}")

    shards = find_split_shards(path)
    scans = [scan_gguf(s) for s in shards]
    for s in scans:
        if s.truncated:
            raise ImageModelError(
                f"{s.path}: truncated GGUF (tensor table expects "
                f"{s.expected_size} bytes, file has {s.size})")
    view = view_from_scans(scans)
    bad = Counter(t.codec for t in view.tensors.values()
                  if t.codec not in SUPPORTED_QUANT_TYPES
                  and t.codec not in NATIVE_TYPES)
    if bad:
        raise UnsupportedCodecError(view.arch or "none", dict(bad))
    reshaped = sorted(k for s in scans for k in s.kv
                      if k.startswith(_ORIG_SHAPE_KEY))
    if reshaped:
        raise ImageModelError(
            f"{len(reshaped)} tensors were reshaped by the converter "
            f"({reshaped[0][len(_ORIG_SHAPE_KEY):]} first). gmlx image "
            "loads DiT files whose tensors keep their original shapes")

    ident = identify(view, family=family, variant=variant)
    fam = ident.family
    if fam.build is None:
        raise ImageModelError(
            f"this is a {fam.title} GGUF, which this gmlx cannot run yet")
    log(f"[image] {fam.title} ({ident.variant}), layout {ident.layout}, "
        f"{len(view.tensors)} tensors")

    before = _active_now()
    arrays, meta, _, _, _ = load_gguf_wire_bytes(
        path, zero_copy=True, shards=shards, expect_quant=False,
        arch=view.arch)
    p = ident.prefix

    def strip(k: str) -> str:
        return k[len(p):] if p and k.startswith(p) else k

    stripped = _RemapDict()
    for k, v in arrays.items():
        stripped[strip(k)] = v
    weights = dict(stripped)
    codecs = {strip(k): c for k, c in meta.items()}
    del arrays, stripped

    no_alias = owned_casts(weights, codecs, fam.fp32_keep, act)
    model = fam.build(ident.config)
    model.act_dtype = act
    check_names(model, ident.view.tensors, fam.title)
    _install_and_load(
        model, weights, codecs, log=log, sanitize=False, no_alias=no_alias,
        fp32_keep=fam.fp32_keep, source_key=weights_source_key(*shards),
        active_before=before, act_dtype=act, text_patches=False)
    n_kq = sum(1 for _, m in model.named_modules()
               if isinstance(m, KQuantLinear))
    collect_after_load()
    info = DitInfo(identified=ident, shards=tuple(shards), act_dtype=act,
                   n_kquant=n_kq, n_owned_casts=len(no_alias),
                   file_bytes=sum(s.size for s in scans))
    return model, info
