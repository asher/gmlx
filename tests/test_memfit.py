#!/usr/bin/env python3
"""Size-vs-RAM verdicts (``gmlx.memfit``) - the shared classification behind
validate/pull's size line and doctor's memory row. CPU-only."""

from __future__ import annotations

from gmlx import memfit

_GB = 1 << 30


def test_classify_fit_thresholds():
    ram = 64 * _GB
    assert memfit.classify_fit(30 * _GB, ram) == memfit.FIT_FITS
    assert memfit.classify_fit(int(0.65 * ram), ram) == memfit.FIT_FITS
    assert memfit.classify_fit(int(0.70 * ram), ram) == memfit.FIT_TIGHT
    assert memfit.classify_fit(int(0.85 * ram), ram) == memfit.FIT_TIGHT
    assert memfit.classify_fit(int(0.90 * ram), ram) == memfit.FIT_OVER
    assert memfit.classify_fit(200 * _GB, ram) == memfit.FIT_OVER


def test_classify_fit_unknowns_return_none(monkeypatch):
    assert memfit.classify_fit(0, 64 * _GB) is None
    assert memfit.classify_fit(-1, 64 * _GB) is None
    assert memfit.classify_fit(10 * _GB, 0) is None
    # ram=None falls back to the machine probe; an unreadable probe -> None.
    monkeypatch.setattr(memfit, "total_ram_bytes", lambda: None)
    assert memfit.classify_fit(10 * _GB, None) is None


def test_fit_label():
    assert memfit.fit_label(memfit.FIT_FITS) == "fits"
    assert memfit.fit_label(memfit.FIT_TIGHT) == "tight"
    assert memfit.fit_label(memfit.FIT_OVER) == "over RAM"
    assert memfit.fit_label(None) == ""


def test_fit_sentence_wording():
    ram = 64 * _GB
    assert "fits this Mac's 64 GB RAM" in memfit.fit_sentence(30 * _GB, ram)
    tight = memfit.fit_sentence(50 * _GB, ram)
    assert "tight" in tight and "--kv-bits" in tight
    over = memfit.fit_sentence(100 * _GB, ram)
    assert "exceeds" in over and "--stream-experts" in over


def test_fit_sentence_no_ram_is_none(monkeypatch):
    monkeypatch.setattr(memfit, "total_ram_bytes", lambda: None)
    assert memfit.fit_sentence(30 * _GB) is None


def test_total_ram_bytes_positive_on_this_machine():
    ram = memfit.total_ram_bytes()
    assert ram is None or ram > 0
