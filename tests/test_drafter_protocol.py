"""Unit tests for drafter_protocol: validate_drafter + DrafterAdapter.

Pure Python -- no GPU, no model loading, no mocks. Hand-written stubs
exercise the contract checking and delegation mechanics.
"""

from __future__ import annotations

import pytest

from gmlx.drafter_protocol import (
    DrafterAdapter,
    _check_accepts_left_padding,
    validate_drafter,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _Cfg:
    def __init__(self, block_size=3):
        self.block_size = block_size


class _FullDrafter:
    """Satisfies the full required contract (like QwenMTPDrafter)."""

    def __init__(self):
        self.config = _Cfg()
        self.accept_lens = []
        self._reset_calls = []
        self._bind_calls = []

    def reset(self, target_model, left_padding=None):
        self._reset_calls.append(("reset", target_model, left_padding))
        return ["cache"]

    def set_shared_kv(self, shared_kv, kv_offset, position=None,
                      kv_valid_len=None, left_padding=None):
        pass

    def draft_block(self, last_bonus, hidden, cache, block_size, sampler,
                    token_dtype=None, greedy=False):
        return None

    def bind(self, target_model):
        self._bind_calls.append(target_model)
        return self

    def filter_batch(self, keep):
        return "filtered"


class _NoLeftPaddingDrafter:
    """reset() does NOT accept left_padding (like Gemma4AssistantDraftModel)."""

    def __init__(self):
        self.config = _Cfg()
        self.accept_lens = []
        self._reset_calls = []

    def reset(self, target_model):
        self._reset_calls.append(("reset", target_model))
        return ["cache"]

    def set_shared_kv(self, shared_kv, kv_offset, position=None,
                      kv_valid_len=None, left_padding=None):
        pass

    def draft_block(self, last_bonus, hidden, cache, block_size, sampler,
                    token_dtype=None, greedy=False):
        return None

    def bind(self, target_model):
        return self


class _KwargsResetDrafter(_NoLeftPaddingDrafter):
    """reset() accepts **kwargs (should detect left_padding support)."""

    def reset(self, target_model, **kwargs):
        self._reset_calls.append(("reset", target_model, kwargs))
        return ["cache"]


# ---------------------------------------------------------------------------
# validate_drafter tests
# ---------------------------------------------------------------------------


class TestValidateDrafter:

    def test_full_drafter_passes(self):
        validate_drafter(_FullDrafter())

    def test_missing_config(self):
        d = _FullDrafter()
        del d.config
        with pytest.raises(RuntimeError, match="missing config"):
            validate_drafter(d)

    def test_bad_block_size(self):
        d = _FullDrafter()
        d.config = _Cfg()
        d.config.block_size = "not_a_number"
        with pytest.raises(RuntimeError, match="block_size must be int"):
            validate_drafter(d)

    def test_missing_block_size(self):
        d = _FullDrafter()
        d.config = object()
        with pytest.raises(RuntimeError, match="block_size must be int"):
            validate_drafter(d)

    def test_accept_lens_not_list(self):
        d = _FullDrafter()
        d.accept_lens = ()
        with pytest.raises(RuntimeError, match="accept_lens must be a list"):
            validate_drafter(d)

    def test_missing_accept_lens(self):
        d = _FullDrafter()
        del d.accept_lens
        with pytest.raises(RuntimeError, match="missing accept_lens"):
            validate_drafter(d)

    def test_missing_draft_block(self):
        d = _FullDrafter()
        d.draft_block = None
        with pytest.raises(RuntimeError, match="draft_block"):
            validate_drafter(d)

    def test_missing_set_shared_kv(self):
        d = _FullDrafter()
        d.set_shared_kv = "not_callable"
        with pytest.raises(RuntimeError, match="set_shared_kv"):
            validate_drafter(d)

    def test_missing_bind(self):
        d = _FullDrafter()
        d.bind = 42
        with pytest.raises(RuntimeError, match="bind"):
            validate_drafter(d)

    def test_missing_reset(self):
        d = _FullDrafter()
        d.reset = None
        with pytest.raises(RuntimeError, match="reset"):
            validate_drafter(d)

    def test_reset_without_left_padding_fails(self):
        d = _NoLeftPaddingDrafter()
        with pytest.raises(RuntimeError, match="left_padding"):
            validate_drafter(d)

    def test_reset_without_left_padding_passes_when_wrapped(self):
        d = _NoLeftPaddingDrafter()
        a = DrafterAdapter(d)
        validate_drafter(a)

    def test_kwargs_reset_passes(self):
        validate_drafter(_KwargsResetDrafter())

    def test_multiple_errors_all_reported(self):
        d = _FullDrafter()
        del d.config
        del d.accept_lens
        d.draft_block = None
        with pytest.raises(RuntimeError) as exc_info:
            validate_drafter(d)
        msg = str(exc_info.value)
        assert "missing config" in msg
        assert "missing accept_lens" in msg
        assert "draft_block" in msg

    def test_error_names_adapter(self):
        d = _NoLeftPaddingDrafter()
        del d.config
        a = DrafterAdapter(d)
        with pytest.raises(RuntimeError, match="DrafterAdapter._NoLeftPaddingDrafter"):
            validate_drafter(a)


# ---------------------------------------------------------------------------
# _check_accepts_left_padding tests
# ---------------------------------------------------------------------------


class TestCheckAcceptsLeftPadding:

    def test_explicit_param(self):
        assert _check_accepts_left_padding(_FullDrafter()) is True

    def test_no_param(self):
        assert _check_accepts_left_padding(_NoLeftPaddingDrafter()) is False

    def test_kwargs(self):
        assert _check_accepts_left_padding(_KwargsResetDrafter()) is True


# ---------------------------------------------------------------------------
# DrafterAdapter tests
# ---------------------------------------------------------------------------


class TestDrafterAdapter:

    def test_reset_absorbs_left_padding(self):
        inner = _NoLeftPaddingDrafter()
        a = DrafterAdapter(inner)
        result = a.reset("model", left_padding=[0, 0])
        assert result == ["cache"]
        assert inner._reset_calls == [("reset", "model")]

    def test_reset_forwards_left_padding(self):
        inner = _FullDrafter()
        a = DrafterAdapter(inner)
        result = a.reset("model", left_padding=[1, 2])
        assert result == ["cache"]
        assert inner._reset_calls == [("reset", "model", [1, 2])]

    def test_bind_returns_adapter(self):
        inner = _FullDrafter()
        a = DrafterAdapter(inner)
        result = a.bind("model")
        assert result is a
        assert inner._bind_calls == ["model"]

    def test_delegates_existing_attribute(self):
        inner = _FullDrafter()
        a = DrafterAdapter(inner)
        assert getattr(a, "filter_batch", None) is not None
        assert a.filter_batch("keep") == "filtered"

    def test_missing_attribute_raises(self):
        inner = _NoLeftPaddingDrafter()
        a = DrafterAdapter(inner)
        with pytest.raises(AttributeError):
            a.filter_batch

    def test_missing_attribute_getattr_default(self):
        inner = _NoLeftPaddingDrafter()
        a = DrafterAdapter(inner)
        assert getattr(a, "filter_batch", None) is None

    def test_config_delegates(self):
        inner = _FullDrafter()
        a = DrafterAdapter(inner)
        assert a.config.block_size == 3

    def test_accept_lens_delegates(self):
        inner = _FullDrafter()
        a = DrafterAdapter(inner)
        a.accept_lens.append(2.0)
        assert inner.accept_lens == [2.0]

    def test_supports_greedy_draft_argmax_absent(self):
        inner = _NoLeftPaddingDrafter()
        a = DrafterAdapter(inner)
        assert getattr(a, "supports_greedy_draft_argmax", False) is False

    def test_supports_greedy_draft_argmax_present(self):
        inner = _FullDrafter()
        inner.supports_greedy_draft_argmax = True
        a = DrafterAdapter(inner)
        assert getattr(a, "supports_greedy_draft_argmax", False) is True
