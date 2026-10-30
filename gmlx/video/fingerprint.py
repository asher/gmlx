"""Identify a video DiT GGUF from its header.

The video families' rules run before the text-arch check, since a video
DiT can carry a text arch label (some H3 converters write ``qwen3vl`` or
``wan``); a label that contradicts the tensors is refused by the family's
name. The variant comes from the image stack's identity hash of small
float tensors. A known file is also named by a hash of rows of q, k and v
in its fused qkv tensor (``layout_identity``), which fixes the row order
that the header cannot show.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

from gmlx.image.fingerprint import (HeaderView, ImageModelError, Refusal,
                                    _text_arch, describe_unknown, strip_prefix,
                                    weight_identity)

from .registry import VideoFamily, VideoVariantRow, families, get_family


class VideoModelError(ValueError):
    """A video GGUF that gmlx refuses, with the reason."""


@dataclass(frozen=True)
class VideoIdentified:
    family: VideoFamily
    config: dict
    prefix: str
    layout: str                     # the qkv row order, or "unverified"
    variant: str                    # "unknown" when no identity row matches
    identity_hash: str | None
    layout_hash: str | None
    view: HeaderView


def layout_reads(shape: tuple, nbytes: int, rows: int) -> tuple | None:
    """The (offset from the tensor's start, length) of the bytes the
    layout identity hashes: ``rows`` whole rows at the start of each third
    of a fused qkv matrix of GGUF ``shape``, the first rows of q, k and v
    head 0 in [q; k; v] order. A file interleaved per head holds other
    heads' rows there. None if the rows are not whole blocks of bytes."""
    n = shape[-1]
    if not rows or n % 3 or nbytes % n or rows > n // 3:
        return None
    row = nbytes // n
    return tuple((s * row, rows * row) for s in (0, n // 3, 2 * n // 3))


def layout_digest(chunks) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c)
    return h.hexdigest()


def layout_identity(view: HeaderView, fam: VideoFamily) -> str | None:
    """sha256 of the bytes of ``layout_reads`` of ``fam.layout_tensor``,
    or None without file bytes."""
    t = view.tensors.get(fam.layout_tensor)
    if t is None or t.path is None:
        return None
    reads = layout_reads(t.shape, t.nbytes, fam.layout_rows)
    if reads is None:
        return None
    fd = os.open(t.path, os.O_RDONLY)
    try:
        chunks = [os.pread(fd, n, t.offset + off) for off, n in reads]
    finally:
        os.close(fd)
    if any(len(c) != n for c, (_, n) in zip(chunks, reads)):
        return None
    return layout_digest(chunks)


def video_family_of(view: HeaderView) -> VideoFamily | None:
    """The video family whose tensors an unstripped view has, or None."""
    for fam in families():
        if fam.fingerprint(strip_prefix(view, fam.strip_prefixes)) is not None:
            return fam
    return None


def _arch_list(fam: VideoFamily) -> str:
    return " or ".join(sorted(repr(a) if a else "none"
                              for a in fam.arch_strings))


def _variant(fam: VideoFamily, row: VideoVariantRow | None,
             lrow: VideoVariantRow | None, given: str | None) -> str:
    """The variant of a file whose identity row is ``row`` and layout row
    ``lrow``. The layout row names a file outright; the identity tensors
    name only a base, which a merged variant of it shares, so a variant
    given for a file with no layout row may be that base or a merge of
    it."""
    named = lrow.name if lrow else row.name if row else "unknown"
    if given is None:
        return named
    names = fam.variant_names()
    if given not in names:
        raise VideoModelError(f"variant {given!r} is not a {fam.title} "
                              f"variant; known: {', '.join(names)}")
    if lrow is not None and given != lrow.name:
        raise VideoModelError(f"variant={given!r} contradicts the weights, "
                              f"which match {fam.title} {lrow.name}")
    if lrow is None and row is not None and given != row.name:
        g = fam.variant_row(given)
        if g is None or g.base != row.name:
            raise VideoModelError(f"variant={given!r} contradicts the "
                                  f"weights, which match {fam.title} "
                                  f"{row.name}")
    return given


def _refuse_image(view: HeaderView) -> None:
    """Refuse an image DiT by name, as gmlx image refuses a video one."""
    from gmlx.image.fingerprint import identify as image_identify
    from gmlx.image.registry import families as image_families

    for fam in image_families():
        try:
            v = strip_prefix(view, fam.strip_prefixes)
        except ImageModelError:
            continue
        if fam.fingerprint(v) is None:
            continue
        try:
            got = image_identify(view)
        except ImageModelError as e:
            raise VideoModelError(
                f"this is an image GGUF gmlx does not load: {e}") from None
        raise VideoModelError(f"this is a {got.family.title} image GGUF, "
                              "run it with gmlx image")


def identify(view: HeaderView, *, family: str | None = None,
             variant: str | None = None) -> VideoIdentified:
    """Classify an unstripped header view, as ``variant`` when given and
    the weights allow it. Raises ``VideoModelError``."""
    arch = view.arch
    for fam in (get_family(family),) if family else families():
        try:
            v = strip_prefix(view, fam.strip_prefixes)
        except ImageModelError as e:
            raise VideoModelError(str(e)) from e
        res = fam.fingerprint(v)
        if res is None:
            continue
        if isinstance(res, Refusal):
            raise VideoModelError(res.reason)
        if arch not in fam.arch_strings:
            raise VideoModelError(
                f"arch {arch!r} contradicts the tensor layout, which is "
                f"{fam.title} (arch {_arch_list(fam)})")
        ident = weight_identity(v, fam.identity_tensors)
        if ident in fam.refused:
            raise VideoModelError(fam.refused[ident])
        row: VideoVariantRow | None = fam.variants.get(ident)
        lhash = layout_identity(v, fam)
        lrow = fam.layouts.get(lhash)
        return VideoIdentified(
            family=fam, config=fam.synth_config(v), prefix=v.prefix,
            layout=lrow.layout if lrow else "unverified",
            variant=_variant(fam, row, lrow, variant), identity_hash=ident,
            layout_hash=lhash, view=v)
    if family:
        raise VideoModelError(
            f"not a {get_family(family).title} file. "
            f"{describe_unknown(view, 'video')}")
    _refuse_image(view)
    if _text_arch(arch):
        raise VideoModelError(
            f"arch {arch!r} is a text model, use gmlx.load_model")
    raise VideoModelError(describe_unknown(view, "video"))
