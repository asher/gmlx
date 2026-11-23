"""The MLX cache clear a prefill chunk leaves for the GPU's next work."""
from __future__ import annotations

import mlx.core as mx
import pytest

import gmlx.serve.capacity as capacity
from gmlx.gen import deferred_clear


@pytest.fixture
def calls(monkeypatch):
    seen = []
    monkeypatch.setattr(mx, "clear_cache", lambda: seen.append("clear"))
    monkeypatch.setattr(capacity, "note_own_before_clear",
                        lambda: seen.append("book"))
    monkeypatch.setattr(deferred_clear, "_pending", False)
    return seen


def test_a_chunk_end_leaves_the_clear_for_one_later_run(calls, monkeypatch):
    monkeypatch.setattr(deferred_clear, "_ENABLED", True)
    deferred_clear.run()
    assert calls == []
    deferred_clear.at_chunk_end()
    assert calls == [] and deferred_clear.pending()
    deferred_clear.run()
    assert calls == ["book", "clear"] and not deferred_clear.pending()
    deferred_clear.run()
    assert calls == ["book", "clear"]


def test_a_clear_by_other_code_settles_the_one_left_waiting(calls, monkeypatch):
    monkeypatch.setattr(deferred_clear, "_ENABLED", True)
    deferred_clear.at_chunk_end()
    deferred_clear.settled()
    deferred_clear.run()
    assert calls == []
    deferred_clear.at_chunk_end()
    deferred_clear.clear_now()
    deferred_clear.run()
    assert calls == ["book", "clear"]


def test_the_lever_off_clears_at_the_chunk_end(calls, monkeypatch):
    monkeypatch.setattr(deferred_clear, "_ENABLED", None)
    monkeypatch.setenv("GMLX_SPEC_CLEAR_AFTER_SUBMIT", "0")
    deferred_clear.at_chunk_end()
    assert calls == ["book", "clear"] and not deferred_clear.pending()
