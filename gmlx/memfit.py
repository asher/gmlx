"""Will this model fit this Mac? - shared size-vs-RAM estimates.

A model costs roughly its GGUF file size in memory (the weights are run
straight from the wire bytes), plus the KV cache, which grows with context.
This module turns "file size vs machine RAM" into a coarse three-way verdict
that `validate`, `pull`, and `doctor` all share, so the first place a new
user meets a model file already says whether it can run here.

The thresholds are deliberately coarse (macOS caps GPU-wirable memory at a
machine-dependent majority share of RAM, and the KV cache varies per family):

* ``fits``  - size <= 65% of RAM: weights plus an everyday KV cache fit.
* ``tight`` - size <= 85% of RAM: it loads, but leaves little room for the
  KV cache (or the rest of the machine); a quantized KV cache or a smaller
  quant is advisable.
* ``over``  - size > 85% of RAM: past what the GPU path can hold. MoE models
  can still run by streaming experts from disk (docs/streaming.md); dense
  models need a smaller quant.

Verdicts are advisory - nothing here refuses anything.
"""
from __future__ import annotations

import os

FIT_FITS = "fits"
FIT_TIGHT = "tight"
FIT_OVER = "over"

_TIGHT_SHARE = 0.65   # above this share of RAM, KV headroom gets scarce
_OVER_SHARE = 0.85    # above this, the GPU path can't hold weights + cache


def total_ram_bytes() -> int | None:
    """The machine's physical memory, or ``None`` when undetectable.
    Module-level seam: monkeypatched in tests."""
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def classify_fit(size_bytes: int, ram_bytes: int | None = None) -> str | None:
    """``fits`` / ``tight`` / ``over`` for a model of ``size_bytes``, or ``None``
    when the size or RAM is unknown."""
    if ram_bytes is None:
        ram_bytes = total_ram_bytes()
    if not ram_bytes or not size_bytes or size_bytes <= 0:
        return None
    share = size_bytes / ram_bytes
    if share <= _TIGHT_SHARE:
        return FIT_FITS
    if share <= _OVER_SHARE:
        return FIT_TIGHT
    return FIT_OVER


def fit_label(fit: str | None) -> str:
    """The short listing-column form of a verdict (empty for unknown)."""
    return {FIT_FITS: "fits", FIT_TIGHT: "tight",
            FIT_OVER: "over RAM"}.get(fit, "")


def fit_sentence(size_bytes: int, ram_bytes: int | None = None) -> str | None:
    """A one-line human verdict for a single model, or ``None`` when unknown.
    Used by ``validate`` (report line) and ``pull`` (post-verdict note)."""
    from .lifecycle import human_gb

    if ram_bytes is None:
        ram_bytes = total_ram_bytes()
    fit = classify_fit(size_bytes, ram_bytes)
    if fit is None:
        return None
    ram = human_gb(ram_bytes, 0)
    if fit == FIT_FITS:
        return f"fits this Mac's {ram} RAM (with room for the KV cache)"
    if fit == FIT_TIGHT:
        return (f"tight on this Mac's {ram} RAM - little room for the KV "
                "cache; consider --kv-bits 8 or a smaller quant")
    return (f"exceeds this Mac's {ram} RAM - a MoE model can still run "
            "with --stream-experts (docs/streaming.md); otherwise pick a "
            "smaller quant")
