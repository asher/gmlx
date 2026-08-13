"""Draft-depth resolution across the drafter families.

Two quantities travel on the drafter config: ``block_size`` (the depth drafted
per round when nobody asks for one) and ``native_block_size`` (the deepest block
the drafter can produce). ``--draft-block-size`` moves the first one in either
direction and can never pass the second.

Pure Python -- the stubs carry only the flags the resolver reads.
"""

from __future__ import annotations

import logging

import pytest

from gmlx.drafter_protocol import default_block_size, native_block_size
from gmlx.mtp_load import _drafter_block_depths
from gmlx.spec_helpers import _mtp_next_block_size, _resolve_block_total


class _Cfg:
    def __init__(self, block_size, native=None, runtime=None):
        self.block_size = block_size
        if native is not None:
            self.native_block_size = native
        if runtime is not None:
            self.runtime_block_size = runtime


class _Drafter:
    def __init__(self, cfg, *, cap=True):
        self.config = cfg
        self.cap_at_configured_depth = cap
        self.prefer_requested_block_size = not cap
        self.accept_lens = []


def _muse(**kw):
    """Declares a ceiling: muse-glimmer dflash, dspark."""
    return _Drafter(_Cfg(kw.pop("block_size", 16), native=kw.pop("native", 16), **kw))


def _ds4_mtp(**kw):
    """Caps at its load-time depth but declares no ceiling: ds4 mtp, hy3."""
    return _Drafter(_Cfg(kw.pop("block_size", 3), **kw))


def _qwen(**kw):
    """Honors any requested depth: qwen native head."""
    return _Drafter(_Cfg(kw.pop("block_size", 3), **kw), cap=False)


# --- the two config quantities ---------------------------------------------

class TestAccessors:

    def test_ceiling_is_none_when_undeclared(self):
        assert native_block_size(_Cfg(3)) is None

    def test_ceiling_is_none_when_zero(self):
        assert native_block_size(_Cfg(3, native=0)) is None

    def test_ceiling_reads_the_declared_depth(self):
        assert native_block_size(_Cfg(4, native=16)) == 16

    def test_default_is_the_block_size(self):
        assert default_block_size(_Cfg(4, native=16)) == 4

    def test_runtime_overrides_the_default_upward(self):
        assert default_block_size(_Cfg(4, native=16, runtime=12)) == 12

    def test_runtime_overrides_the_default_downward(self):
        assert default_block_size(_Cfg(16, native=16, runtime=4)) == 4


# --- what a run resolves to -------------------------------------------------

class TestResolveBlockTotal:

    @pytest.mark.parametrize("build", [_muse, _ds4_mtp, _qwen])
    def test_no_request_uses_the_configured_default(self, build):
        drafter = build(block_size=3)
        assert _resolve_block_total(drafter, None) == 3

    @pytest.mark.parametrize("build", [_muse, _ds4_mtp, _qwen])
    def test_no_request_prefers_the_runtime_default(self, build):
        drafter = build(block_size=3, runtime=2)
        assert _resolve_block_total(drafter, None) == 2

    def test_request_below_the_ceiling_is_honored(self):
        assert _resolve_block_total(_muse(block_size=4, native=16), 12) == 12

    def test_request_at_the_ceiling_is_honored(self):
        assert _resolve_block_total(_muse(block_size=4, native=16), 16) == 16

    def test_request_past_the_ceiling_clamps(self):
        assert _resolve_block_total(_muse(block_size=4, native=16), 24) == 16

    def test_request_below_the_default_lowers_it(self):
        assert _resolve_block_total(_muse(block_size=16, native=16), 4) == 4

    def test_undeclared_ceiling_still_caps_at_the_load_time_depth(self):
        assert _resolve_block_total(_ds4_mtp(block_size=3), 8) == 3

    def test_uncapped_family_honors_any_depth(self):
        assert _resolve_block_total(_qwen(block_size=3), 8) == 8

    def test_runtime_default_does_not_cap_an_explicit_request(self):
        drafter = _muse(block_size=16, native=16, runtime=4)
        assert _resolve_block_total(drafter, None) == 4
        assert _resolve_block_total(drafter, 16) == 16


# --- per-round sizing -------------------------------------------------------

class TestNextBlockSize:

    def test_round_honors_a_depth_within_the_ceiling(self):
        assert _mtp_next_block_size(_muse(native=16), 16, 4, 128) == 16

    def test_round_clamps_to_the_ceiling(self):
        assert _mtp_next_block_size(_muse(native=16), 24, 16, 128) == 16

    def test_round_clamps_to_the_remaining_budget(self):
        assert _mtp_next_block_size(_muse(native=16), 16, 16, 5) == 5

    def test_uncapped_family_keeps_the_requested_depth(self):
        assert _mtp_next_block_size(_qwen(), 8, 3, 128) == 8


class TestDepthWarning:

    def test_a_deeper_request_warns_once(self, caplog):
        drafter = _muse(block_size=4, native=16)
        with caplog.at_level(logging.WARNING, logger="gmlx.spec_helpers"):
            assert _resolve_block_total(drafter, 24) == 16
            assert _mtp_next_block_size(drafter, 24, 16, 128) == 16
        warned = [r for r in caplog.records if "deeper than" in r.message]
        assert len(warned) == 1

    def test_a_request_within_the_ceiling_is_silent(self, caplog):
        with caplog.at_level(logging.WARNING, logger="gmlx.spec_helpers"):
            assert _resolve_block_total(_muse(block_size=4, native=16), 16) == 16
        assert not [r for r in caplog.records if "deeper than" in r.message]

    def test_the_configured_default_is_silent(self, caplog):
        drafter = _muse(block_size=16, native=16, runtime=32)
        with caplog.at_level(logging.WARNING, logger="gmlx.spec_helpers"):
            assert _resolve_block_total(drafter, None) == 16
        assert not [r for r in caplog.records if "deeper than" in r.message]


# --- what the loaders stamp -------------------------------------------------

class TestLoaderDepths:

    def test_the_family_default_bounds_the_runtime_depth(self):
        assert _drafter_block_depths(32, 16) == (32, 16)

    def test_the_ceiling_bounds_the_family_default(self):
        assert _drafter_block_depths(8, 16) == (8, 8)

    def test_no_family_default_drafts_the_full_block(self):
        assert _drafter_block_depths(6) == (6, 6)

    def test_the_runtime_depth_keeps_one_draft(self):
        assert _drafter_block_depths(32, 1) == (32, 2)
