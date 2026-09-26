"""A model's own fast-disk setting beats GMLX_DECODE_FAST_DISK."""
from __future__ import annotations

import types

from gmlx.stream.decode_feeder import DecodeFeeder


def test_model_setting_beats_the_env(monkeypatch):
    monkeypatch.setenv("GMLX_DECODE_FAST_DISK", "off")
    fake = types.SimpleNamespace(_probe_read_rate=lambda: 0.0)
    assert DecodeFeeder._resolve_fast_disk(fake, "on") is True
    assert DecodeFeeder._resolve_fast_disk(fake, None) is False
    monkeypatch.setenv("GMLX_DECODE_FAST_DISK", "on")
    assert DecodeFeeder._resolve_fast_disk(fake, "off") is False
