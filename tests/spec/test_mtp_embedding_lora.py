"""MTP rounds with a token embedding adapter in rows mode. The target
embeds through a ``LoRAKQuantEmbedding`` inside each round's publish; the
drafter binds the base table through ``lora_base`` at reset, as the real
binds do, and drafts with nothing published. No draft lookup raises,
every draft lookup is a base row, and every verify forward embeds each row
at its own strength. CPU-only (toy target and drafter)."""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import gmlx.spec.speculative as spec
from gmlx import lora_rows
from gmlx.load.modules import LoRAKQuantEmbedding, lora_base
from gmlx.spec.speculative import _width_cap_logged

from test_mtp_width_cap import _FakeCache
from test_spec_logprobs import V, _ChainDrafter, _TableLM


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for var in ("GMLX_MTP_WIDTH_CAP", "MLX_VLM_GGUF_SPEC_WIDTH_CAP",
                "GMLX_MTP_MIXED", "GMLX_MTP_RESUME", "GMLX_MTP_STOCH"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(spec, "_ROUND_GATE", 0.0)
    monkeypatch.setattr(spec, "_ROUND_GATE_AUTO", False)
    spec._width_cap_memo = ("", None)
    _width_cap_logged.clear()
    lora_rows.configure("rows", 1)
    yield
    lora_rows.configure("static", 1)
    spec._width_cap_memo = ("", None)
    _width_cap_logged.clear()


def _table():
    base = nn.Embedding(V, 8)
    base.weight = mx.array((np.arange(V * 8, dtype=np.float32).reshape(V, 8) % 9) * 0.5)
    a = mx.array((np.arange(V * 2, dtype=np.float32).reshape(V, 2) % 5) - 2.0)
    b = mx.array((np.arange(16, dtype=np.float32).reshape(8, 2) % 3) * 0.25)
    return base, LoRAKQuantEmbedding(base, a, b, 0.5)


class _EmbedLM(_TableLM):
    """Embeds every forward's ids through the target's table and records
    the ids and rows."""

    def __init__(self, embed):
        super().__init__()
        self.embed = embed
        self.seen = []

    def _record(self, x):
        rows = self.embed(x)
        mx.eval(rows)
        self.seen.append((np.array(x), rows))

    def __call__(self, x, cache=None, **kw):
        self._record(x)
        return super().__call__(x, cache=cache, **kw)

    def speculative_verify_hidden(self, x, cache, **kw):
        self._record(x)
        return super().speculative_verify_hidden(x, cache, **kw)


class _EmbedDrafter(_ChainDrafter):
    """Binds the target's base table at reset and looks its draft input up
    in it, with nothing published."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.lookups = []

    def reset(self, model, left_padding=None):
        self.table = lora_base(model.model.embed_tokens)
        return super().reset(model, left_padding)

    def draft_block(self, b, hidden, kv, n, sampler, dtype, **kw):
        ids = mx.array(b).reshape(-1, 1)
        assert lora_rows._local.current is None
        self.lookups.append((np.array(ids), self.table(ids)))
        return super().draft_block(b, hidden, kv, n, sampler, dtype, **kw)


def _check_verify_rows(lm, wrap, strengths):
    assert lm.seen
    for ids, rows in lm.seen:
        assert ids.shape[0] == len(strengths)
        for i, s in enumerate(strengths):
            with lora_rows.tuple_scope((s,), 1):
                want = wrap(mx.array(ids[i:i + 1]))
            assert mx.allclose(rows[i:i + 1], want, rtol=1e-6, atol=1e-6), (i, s)


def _check_draft_rows(drafter, base):
    assert drafter.lookups
    for ids, rows in drafter.lookups:
        assert mx.array_equal(rows, base(mx.array(ids)))


def test_batched_rounds_draft_base_rows_and_verify_each_rows_strength():
    base, wrap = _table()
    strengths = (1.0, 0.0, 1.5)
    model = SimpleNamespace(model=SimpleNamespace(embed_tokens=wrap))
    lm = _EmbedLM(wrap)
    drafter = _EmbedDrafter(cap=0, block_size=3)
    with lora_rows.bind(lora_rows.UidRegistry()):
        for i, s in enumerate(strengths):
            lora_rows.register_uid(f"u{i}", (s,))
        gen = spec._owned_decode_rounds_batch(
            model, drafter, lm, [_FakeCache(width=3)], hidden=mx.zeros((3, 4, 8)),
            b=[3, 8, 13], shared_kv=None, seed_tokens=None, emitted=[1] * 3,
            max_tokens=7, sampler=None, draft_block_size=None,
            uids=["u0", "u1", "u2"])
        out = list(gen)
        _check_verify_rows(lm, wrap, strengths)
    assert out and drafter.draft_calls
    _check_draft_rows(drafter, base)


def test_scalar_rounds_draft_base_rows_and_verify_at_the_rows_strength():
    base, wrap = _table()
    model = SimpleNamespace(model=SimpleNamespace(embed_tokens=wrap), _kq_row_uids=["u"])
    lm = _EmbedLM(wrap)
    drafter = _EmbedDrafter(cap=0, block_size=3)
    with lora_rows.bind(lora_rows.UidRegistry()):
        lora_rows.register_uid("u", (1.5,))
        gen = spec._owned_decode_rounds(
            model, drafter, lm, [_FakeCache(width=1)], hidden=mx.zeros((1, 4, 8)), b=3,
            shared_kv=None, seed_tokens=None, emitted=1, max_tokens=7,
            sampler=None, draft_block_size=None)
        toks = list(gen)
        _check_verify_rows(lm, wrap, (1.5,))
    assert toks and drafter.draft_calls
    _check_draft_rows(drafter, base)
