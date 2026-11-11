"""The width policy: what a batch decodes with at each width.

A drafter that drafts for one row at a time (DFlash2) drafts only while one
request is live: a batch of two or more decodes plain, and the last row left
drafts again. The native head of a dense Qwen target drafts for every row up
to its switch width, and a wider batch decodes plain. GMLX_MTP_MIXED=1 brings
the mixed batch back.

Loop-level tests on the echo target and drafters of test_mtp_mixed_batch.py.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

import gmlx.spec.speculative as spec
from gmlx.spec.mtp_load import _stamp_mtp_width_cap
from gmlx.spec.speculative import _width_cap_logged

from test_mtp_mixed_batch import (VOCAB, _chain, _drive, _RowDrafter,
                                  _streams)
from test_mtp_width_cap import DENSE_PROJ, _fake_model


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("GMLX_MTP_WIDTH_CAP", "MLX_VLM_GGUF_SPEC_WIDTH_CAP",
                "GMLX_MTP_MIXED", "GMLX_MTP_RESUME"):
        monkeypatch.delenv(var, raising=False)
    spec._width_cap_memo = ("", None)
    _width_cap_logged.clear()
    yield
    spec._width_cap_memo = ("", None)
    _width_cap_logged.clear()


def _one_row_drafter():
    """A DFlash2-style drafter: its own context, one row at a time."""
    d = _RowDrafter(cap=1, limit=1, block_size=4)
    d.set_shared_kv = lambda *a, **k: None    # the batched rounds bind it
    return d


def test_a_batch_of_two_decodes_plain_and_the_last_row_drafts_again():
    d = _one_row_drafter()
    out, _, _ = _drive(d, B=2, max_tokens=40,
                       stop_check=lambda r, t: r == 0 and t >= 13)
    streams = _streams(out, 2)
    assert streams[0] == _chain(10, 3)
    assert streams[1] == _chain(20, 39)
    # while both rows are live every round is one plain token a row
    both = [meta for toks, meta in out if toks[0] is not None]
    assert len(both) == 3 and all(m["round_len"] == 1 for m in both)
    # nothing drafts until the batch is down to one row, which then arms
    # from a capture of its last position and drafts its own chain
    names = d.names()
    arm = names.index("prefill")
    assert "draft" not in names[:arm]
    assert [e for e in d.log if e[0] == "prefill"] == [("prefill", 1)]
    drafts = [e[1] for e in d.log[arm:] if e[0] == "draft"]
    assert drafts and all(23 < t < 60 for t in drafts)
    assert "accept" in names[arm:]
    assert any(meta["round_len"] > 1 for _, meta in out)


def test_the_opt_in_still_gives_the_mixed_batch(monkeypatch):
    monkeypatch.setenv("GMLX_MTP_MIXED", "1")
    d = _RowDrafter(cap=1, limit=1, block_size=4)
    out, lm, _ = _drive(d, B=2, max_tokens=13)
    streams = _streams(out, 2)
    assert streams[0] == _chain(10, 12)
    assert streams[1] == _chain(20, 12)
    # the oldest row drafts from the first round, and the other row rides
    # the same verify behind padding drafts
    assert d.names()[:4] == ["reset", "prefill", "open", "draft"]
    assert d.log[3] == ("draft", 10)
    assert lm.inputs[0][1] == [20, 20, 20]


class _HeadDrafter:
    """A native head: it keeps its own state for every row and drafts the
    echo chain for the whole batch."""

    uses_shared_kv = False
    supports_greedy_draft_argmax = True

    def __init__(self, block_size=4):
        self.config = SimpleNamespace(block_size=block_size)
        self.accept_lens = []
        self.draft_lens = []
        self.drafted = []

    def reset(self, model, left_padding=None):
        return []

    def set_shared_kv(self, *a, **kw):
        pass

    def prefill_from_target_hidden(self, toks, hidden, bonus, sampler, dtype,
                                   **kw):
        pass

    def draft_block(self, b, hidden, cache, n, sampler, dtype, **kw):
        self.drafted.append(int(b.shape[0]))
        chain = (b[:, None] + mx.arange(1, n)[None, :]) % VOCAB
        return chain.astype(dtype)

    def accept_verified_tokens_batch(self, vh, drafts, accepted, new,
                                     sampler, dtype, **kw):
        pass

    def filter_batch(self, keep):
        pass


def _native_head():
    """The head as the loader stamps it for a dense Qwen target, and its
    switch width."""
    d = _stamp_mtp_width_cap(_HeadDrafter(), "qwen3_5",
                             target=_fake_model(DENSE_PROJ),
                             log=lambda *a, **k: None)
    return d, d.mtp_width_cap


def test_the_native_head_drafts_at_its_switch_width_and_not_above():
    d, width = _native_head()
    out, lm, _ = _drive(d, B=width, max_tokens=9)
    for r, toks in _streams(out, width).items():
        assert toks == _chain(10 * (r + 1), 8)
    # every round drafts for every row and verifies a full block
    assert d.drafted and set(d.drafted) == {width}
    assert all(len(rows) == width and len(rows[0]) == 4
               for rows in lm.inputs)

    d, width = _native_head()
    out, _, _ = _drive(d, B=width + 1, max_tokens=9)
    for r, toks in _streams(out, width + 1).items():
        assert toks == _chain(10 * (r + 1), 8)
    assert d.drafted == []
    assert all(meta["round_len"] == 1 for _, meta in out)
