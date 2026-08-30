#!/usr/bin/env python3
"""The live-streaming-install registry: a released install gives its wired
bytes back before the next one sizes itself, and one still held is charged
against the next."""

from __future__ import annotations

import gc

import pytest

from gmlx.stream import installs


@pytest.fixture(autouse=True)
def _clean_registry():
    installs._LIVE.clear()
    yield
    installs._LIVE.clear()


class _Holder:
    """Stands in for a model: weak-referenceable, carries the helpers."""

    def __init__(self):
        self._kq_decode_feeder = None
        self._kq_weights_pin = None


class _Helper:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_record_accumulates_per_model():
    # The loader records twice: the weight pin, then the arena it sizes
    # afterwards. Both are the same install and must sum, not compete.
    m = _Holder()
    installs.record(m, 10)
    installs.record(m, 25)
    assert installs.live_wired_bytes() == 35


def test_a_held_install_is_still_charged():
    m = _Holder()
    installs.record(m, 40)
    assert installs.reclaim_dead() == 0
    assert installs.live_wired_bytes() == 40
    assert m is not None  # keep it held for the assertions above


def test_reclaim_dead_collects_a_cycle_refcounting_cannot():
    # The feeder holds the modules and the modules hold the feeder, so a
    # dropped model keeps its arena wired until a collection runs, and its
    # weakref resolves the whole time.
    m = _Holder()
    feeder = _Helper()
    feeder.owner = m           # feeder -> model
    m._kq_decode_feeder = feeder   # model -> feeder
    installs.record(m, 60)

    gc.disable()
    try:
        del m, feeder
        assert installs.live_wired_bytes() == 60, (
            "an uncollected cycle must still read as held; if this fails the "
            "test no longer reproduces the condition reclaim_dead exists for")
        assert installs.reclaim_dead() == 60
    finally:
        gc.enable()
    assert installs.live_wired_bytes() == 0


def test_release_closes_every_helper_and_drops_the_record():
    m = _Holder()
    feeder, pin = _Helper(), _Helper()
    m._kq_decode_feeder, m._kq_weights_pin = feeder, pin
    installs.record(m, 90)

    installs.release(m)

    assert feeder.closed and pin.closed
    assert m._kq_decode_feeder is None and m._kq_weights_pin is None
    assert installs.live_wired_bytes() == 0


def test_release_survives_a_helper_that_raises():
    # One failing close must not leak the others' arenas and fds.
    class _Bad(_Helper):
        def close(self):
            raise RuntimeError("closed twice")

    m = _Holder()
    pin = _Helper()
    m._kq_decode_feeder, m._kq_weights_pin = _Bad(), pin
    installs.release(m)
    assert pin.closed


def test_streaming_owner_descends_the_wrapper_chain():
    # The served object is a text-only vlm adapter that forwards nothing, so
    # reading the helpers off it finds none and the install looks absent.
    inner = _Holder()
    inner._kq_decode_feeder = _Helper()

    class _Wrapper:
        pass

    outer, mid = _Wrapper(), _Wrapper()
    mid._model = inner
    outer.language_model = mid

    assert installs.streaming_owner(outer) is inner


def test_streaming_owner_returns_the_original_when_nothing_streams():
    m = _Holder()
    assert installs.streaming_owner(m) is m
