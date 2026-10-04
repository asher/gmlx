"""gmlx/container/notices.py: lines that print once, or once in an interval,
for their key."""

from __future__ import annotations

import json

from gmlx.container import notices
from gmlx.container.notices import Once


def test_a_once_line_prints_once_for_its_key():
    first = [Once("a", "k1"), "plain", Once("b", "k2")]
    assert notices.due(first, now=100) == ["a", "plain", "b"]
    assert notices.due(first, now=200) == ["plain"]
    assert notices.due([Once("a again", "k1"), Once("c", "k3")], now=300) == ["c"]


def test_an_interval_line_repeats_after_its_interval():
    line = Once("old image", "age:x", every=notices.DAY)
    assert notices.due([line], now=0) == ["old image"]
    assert notices.due([line], now=notices.DAY - 1) == []
    assert notices.due([line], now=notices.DAY) == ["old image"]


def test_without_record_nothing_is_recorded():
    assert notices.due([Once("a", "k")], record=False) == ["a"]
    assert notices.due([Once("a", "k")]) == ["a"]


def test_lines_passed_without_record_are_recorded_later():
    shown = notices.due([Once("a", "k"), "plain", Once("b", "k2", every=10)], record=False,
                        now=100)
    assert notices.due([Once("a", "k")], now=100) == ["a"]
    notices.record(shown, now=100)
    assert notices.due([Once("a", "k"), Once("b", "k2", every=10)], now=105) == []
    assert notices.due([Once("b", "k2", every=10)], now=110) == ["b"]


def test_the_record_keeps_the_newest_keys(monkeypatch):
    monkeypatch.setattr(notices, "NOTICES_MAX", 3)
    for i in range(5):
        notices.due([Once(str(i), f"k{i}")], now=i)
    assert sorted(json.loads(notices._path().read_text())) == ["k2", "k3", "k4"]


def test_a_record_that_cannot_be_written_prints_every_line(monkeypatch):
    def fail(*_a, **_kw):
        raise PermissionError("read-only")
    monkeypatch.setattr(notices, "write_record", fail)
    assert notices.due([Once("a", "k")]) == ["a"]
    assert notices.due([Once("a", "k")]) == ["a"]
