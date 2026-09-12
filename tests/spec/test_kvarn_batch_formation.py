"""Batch formation over a kvarn batch KV stack: the verify block clamps to
the decode kernels' width (KVARN_BATCH_QL) for the generator's life when
the batch is wider than one row, and a packed batch cache without the
ragged rollback contract gates the batch for its life, never re-arming.

Drives _owned_decode_rounds_batch with the armable fakes of the preempt
tests: the drafter records the block size each draft_block call asked for."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from gmlx.spec.speculative import _owned_decode_rounds_batch, _width_cap_logged
import gmlx.spec.speculative as spec

from test_mtp_preempt_resume import _ArmableDrafter, _VerifyEchoLM
from test_mtp_width_cap import _FakeCache


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("GMLX_MTP_WIDTH_CAP", "MLX_VLM_GGUF_SPEC_WIDTH_CAP",
                "GMLX_MTP_PREEMPT", "GMLX_MTP_RESUME"):
        monkeypatch.delenv(var, raising=False)
    spec._width_cap_memo = ("", None)
    _width_cap_logged.clear()
    yield
    spec._width_cap_memo = ("", None)
    _width_cap_logged.clear()


class _KvarnBatchFake(_FakeCache):
    """A BatchKVarNKVCache stand-in: the ragged rollback contract."""

    ragged_trim = True


class _PackedBatchFake(_FakeCache):
    """A packed batch cache without the ragged contract (a stale stack)."""

    kv_quant_scheme = "kvarn"

    def __init__(self, width=1, offset=8):
        super().__init__(width, offset)
        self.left_padding = mx.zeros((width,), dtype=mx.int32)
        self._idx = offset


class _WidthDrafter(_ArmableDrafter):
    """Records the block size each draft asked for; drafts nothing."""

    def draft_block(self, b, hidden, kv, n, sampler, dtype, **kw):
        self.draft_calls.append(int(n))
        return mx.zeros((int(b.shape[0]), 0), dtype=dtype)


def _drive(drafter, prompt_cache, *, B, max_tokens, emitted=None, lm=None):
    lm = lm if lm is not None else _VerifyEchoLM()
    gen = _owned_decode_rounds_batch(
        SimpleNamespace(), drafter, lm, prompt_cache,
        hidden=None,
        b=list(range(1, B + 1)),
        shared_kv=None,
        seed_tokens=None,
        emitted=emitted if emitted is not None else [1] * B,
        max_tokens=max_tokens,
        sampler=None,
        draft_block_size=None,
        stop_check=None,
    )
    return [(toks, meta) for toks, meta in gen], lm


def test_kvarn_batch_clamps_the_block(capsys):
    d = _WidthDrafter(cap=0, block_size=8)
    out, lm = _drive(d, [_KvarnBatchFake(width=2)], B=2, max_tokens=20)
    assert d.draft_calls and max(d.draft_calls) == 4
    assert all(n <= 4 for n in d.draft_calls)
    assert lm.verify_widths and lm.plain_widths == []
    err = capsys.readouterr().err
    assert "clamp: kvarn batch KV verifies at 4 queries; block 8 -> 4" in err
    assert err.count("width-cap clamp") == 1


def test_b1_keeps_the_full_block():
    d = _WidthDrafter(cap=0, block_size=8)
    _drive(d, [_KvarnBatchFake(width=1)], B=1, max_tokens=20)
    assert d.draft_calls and max(d.draft_calls) == 8


def test_fp16_batch_keeps_the_full_block():
    d = _WidthDrafter(cap=0, block_size=8)
    _drive(d, [_FakeCache(width=2)], B=2, max_tokens=20)
    assert d.draft_calls and max(d.draft_calls) == 8


def test_clamp_reaches_a_nested_stack():
    # CacheList members are walked, so a kvarn layer inside one clamps.
    d = _WidthDrafter(cap=0, block_size=8)
    nested = SimpleNamespace(caches=[_FakeCache(width=2), _KvarnBatchFake(width=2)])
    _drive(d, [_FakeCache(width=2), nested], B=2, max_tokens=20)
    assert d.draft_calls and max(d.draft_calls) == 4


def test_clamp_holds_after_the_batch_drains():
    """block_total is computed once: a batch that drains to one row keeps
    drafting at the clamped width until its generator ends."""
    d = _WidthDrafter(cap=0, block_size=8)
    out, _ = _drive(d, [_KvarnBatchFake(width=2)], B=2, max_tokens=24,
                    emitted=[1, 22])
    # the second row retires within a couple of rounds; the survivor runs on
    assert len(out) > 6 and all(len(toks) == 2 for toks, _ in out[:1])
    assert max(d.draft_calls) == 4 and all(n <= 4 for n in d.draft_calls)
    assert len(d.draft_calls) > 4


def test_packed_batch_cache_gates_for_life(capsys):
    d = _ArmableDrafter(cap=0)
    out, lm = _drive(d, [_PackedBatchFake(width=2)], B=2, max_tokens=12,
                     emitted=[1, 10])
    assert d.draft_calls == [] and d.prefill_calls == []
    assert lm.verify_widths == [] and lm.plain_widths
    # rows retired under any cap along the way: no re-arm, no capture round
    assert len(out) > 3
    err = capsys.readouterr().err
    assert "gate: packed batch KV without ragged rollback (_PackedBatchFake)" in err
    # the ragged contract at the same width speculates
    d2 = _ArmableDrafter(cap=0)
    _, lm2 = _drive(d2, [_KvarnBatchFake(width=2)], B=2, max_tokens=6)
    assert d2.draft_calls and lm2.verify_widths


def test_packed_gate_is_not_a_width_gate():
    # A single packed row is the B=1 path's business, not this gate's.
    d = _ArmableDrafter(cap=0)
    _, lm = _drive(d, [_PackedBatchFake(width=1)], B=1, max_tokens=6)
    assert d.draft_calls and lm.verify_widths
