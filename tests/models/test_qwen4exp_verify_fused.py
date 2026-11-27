"""GMLX_Q4_VERIFY_FUSED: a verify's gated-delta layers run as recorded kq
graphs. The hiddens and the cache states carry the bits of the plain path
through full accepts and rollbacks; a record is dropped when one of the
layer's arrays is replaced or a LoRA is installed, and the path stays off
for a layer with a LoRA and for a process that swaps weights around its
layer calls."""

import mlx.core as mx
import mlx_kquant as kq
import numpy as np
import pytest

from helpers import _real_apple_gpu
from mlx.utils import tree_flatten

import gmlx.models.qwen4_exp.model as q4

from test_qwen4exp_gdn_verify_v2 import _lm

pytestmark = [
    pytest.mark.skipif(
        not hasattr(kq, "fuse_graph"), reason="kq build without fuse_graph"),
    pytest.mark.skipif(not _real_apple_gpu(), reason="GPU kernels"),
]

# (rows, kept rows): full accepts and rollbacks of each width often enough
# that every input layout is recorded and then replayed.
ROUNDS = ([(4, 4)] * 4 + [(4, 2)] * 4 + [(4, 4), (3, 3), (3, 1), (3, 3)]
          + [(4, 1), (4, 4), (4, 3), (4, 4), (3, 3), (3, 2), (3, 3), (4, 4)])


@pytest.fixture(autouse=True)
def _levers(monkeypatch):
    monkeypatch.setattr(q4, "_VERIFY_BLOCKED", False)
    monkeypatch.setattr(q4, "_VERIFY_FUSED", False)
    monkeypatch.setattr(q4, "_VERIFY_FUSED_QSA", True)
    monkeypatch.setattr(q4, "_VERIFY_FUSED_CAP", True)
    monkeypatch.setattr(q4, "_DECODE_FUSED", False)
    monkeypatch.setattr(q4, "_FUSED_LIMITS", (4096, 256 << 20))
    yield
    kq.env_cache(False)


def _gdn_layers(lm):
    return [layer for layer in lm.model.layers
            if layer.is_linear and "ple" not in layer]


def _scale_a_weight(lm, factor=1.5):
    """Replace one float array of the first gated-delta layer."""
    layer = _gdn_layers(lm)[0]
    name, w = next((k, v) for k, v in tree_flatten(layer.parameters())
                   if v.dtype in (mx.float16, mx.float32) and v.size > 8)
    node = layer
    *path, leaf = name.split(".")
    for part in path:
        node = node[int(part)] if part.isdigit() else node[part]
    node[leaf] = w * factor
    mx.eval(node[leaf])


def _records(lm) -> int:
    return q4.verify_fused_stats(lm)["records"]


def _walk(monkeypatch, fused, *, rounds=ROUNDS, swap_at=None, before=None,
          step_at=()):
    monkeypatch.setattr(q4, "_VERIFY_FUSED", fused)
    kq.env_cache(fused)
    lm, args = _lm(True)
    if before is not None:
        before(lm)
    mx.random.seed(5)
    cache = lm.make_cache()
    mx.eval(lm(mx.random.randint(0, args.vocab_size, (1, 9)),
               cache=cache).logits)
    out = []
    for i, (S, kept) in enumerate(rounds):
        if i == swap_at:
            _scale_a_weight(lm)
        if i in step_at:
            # A plain decode step between rounds takes the pending replay.
            tok = mx.random.randint(0, args.vocab_size, (1, 1))
            out.append(np.array(lm(tok, cache=cache).logits.astype(mx.float32)))
        block = mx.random.randint(0, args.vocab_size, (1, S))
        hid, _, sink = lm.speculative_verify_hidden(block, cache)
        lm.rollback_speculative_cache(cache, sink, kept - 1, S)
        mx.eval(hid, [c.state for c in cache])
        out.append(np.array(hid.astype(mx.float32)))
    out.extend(np.array(a.astype(mx.float32))
               for c in cache for a in c.state if isinstance(a, mx.array))
    return lm, out


def _equal(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.shape == y.shape and np.array_equal(x, y)


def test_fused_rounds_equal_the_plain_ones(monkeypatch):
    _, plain = _walk(monkeypatch, False)
    lm, fused = _walk(monkeypatch, True)
    stats = q4.verify_fused_stats(lm)
    assert stats["records"] > 0 and stats["replays"] + stats["plain_runs"] > 0
    _equal(plain, fused)


def test_a_layout_met_after_a_replayed_full_accept_records(monkeypatch):
    """A replayed full accept leaves the pending conv window and the
    replay their own step arrays, so a layout first recorded after it
    takes a record like any other, and no layer is pinned to the plain
    call."""
    rounds = [(4, 4)] * 4 + [(3, 3), (4, 4)] * 4
    _, plain = _walk(monkeypatch, False, rounds=rounds)
    lm, fused = _walk(monkeypatch, True, rounds=rounds)
    _equal(plain, fused)
    for layer in _gdn_layers(lm):
        assert "plain" not in vars(layer)["_verify_fused"][1].values()


def test_a_rollback_twice_on_one_verify(monkeypatch):
    def run(fused):
        monkeypatch.setattr(q4, "_VERIFY_FUSED", fused)
        kq.env_cache(fused)
        lm, args = _lm(True)
        mx.random.seed(7)
        cache = lm.make_cache()
        mx.eval(lm(mx.random.randint(0, args.vocab_size, (1, 9)),
                   cache=cache).logits)
        out = []
        for _ in range(6):
            block = mx.random.randint(0, args.vocab_size, (1, 4))
            hid, _, sink = lm.speculative_verify_hidden(block, cache)
            # A stop inside the round: the rollback runs again, shorter.
            lm.rollback_speculative_cache(cache, sink, 2, 4)
            lm.rollback_speculative_cache(cache, sink, 0, 3)
            mx.eval(hid, [c.state for c in cache])
            out.append(np.array(hid.astype(mx.float32)))
        return lm, out

    _, plain = run(False)
    lm, fused = run(True)
    assert _records(lm) > 0
    _equal(plain, fused)


def test_decode_steps_between_fused_rounds(monkeypatch):
    steps = (5, 6, 11, 17)
    _, plain = _walk(monkeypatch, False, step_at=steps)
    lm, fused = _walk(monkeypatch, True, step_at=steps)
    assert _records(lm) > 0
    _equal(plain, fused)


def test_a_replaced_weight_is_seen_by_the_record(monkeypatch):
    _, plain = _walk(monkeypatch, False, swap_at=9)
    lm, fused = _walk(monkeypatch, True, swap_at=9)
    assert _records(lm) > 0
    _, unswapped = _walk(monkeypatch, False)
    assert not np.array_equal(plain[len(ROUNDS) - 1], unswapped[len(ROUNDS) - 1])
    _equal(plain, fused)


def test_a_layer_with_a_lora_stays_plain(monkeypatch):
    def stamp(lm):
        layer = _gdn_layers(lm)[0]
        leaf = next(m for m in layer.modules() if m is not layer)
        object.__setattr__(leaf, "_kq_lora", object())

    lm, _ = _walk(monkeypatch, True, before=stamp)
    first, *rest = _gdn_layers(lm)
    assert all(r == "plain" for r in vars(first)["_verify_fused"][1].values())
    assert _records(lm) > 0 or not rest


def test_an_invalidation_drops_the_records(monkeypatch):
    lm, _ = _walk(monkeypatch, True, rounds=ROUNDS[:4])
    assert _records(lm) > 0
    q4.invalidate_verify_compile()
    cache = lm.make_cache()
    mx.eval(lm(mx.array([[1, 2, 3]]), cache=cache).logits)
    hid, _, _ = lm.speculative_verify_hidden(mx.array([[1, 2, 3, 4]]), cache)
    mx.eval(hid)
    # The first verify after it runs plain again, on a fresh table.
    assert _records(lm) == 0


def test_a_weight_swap_turns_the_path_off(monkeypatch):
    monkeypatch.setattr(q4, "_VERIFY_FUSED", True)
    assert q4._verify_fused()
    q4.invalidate_verify_compile(block=True)
    assert not q4._verify_fused()


def test_the_attention_layers_take_records_too(monkeypatch):
    # A served model is in eval mode; a training layer stays plain.
    def serve(lm):
        lm.eval()

    _, plain = _walk(monkeypatch, False, before=serve)
    lm, fused = _walk(monkeypatch, True, before=serve)
    qsa = [layer for layer in lm.model.layers if not layer.is_linear]
    held = [r for layer in qsa
            for r in vars(layer).get("_verify_fused", (0, {}, []))[1].values()]
    assert held and all(isinstance(r, tuple) for r in held)
    _equal(plain, fused)
    monkeypatch.setattr(q4, "_VERIFY_FUSED_QSA", False)
    lm, gdn_only = _walk(monkeypatch, True, before=serve)
    assert not any("_verify_fused" in vars(layer) for layer in
                   lm.model.layers if not layer.is_linear)
    _equal(plain, gdn_only)


def test_an_attention_record_leaves_the_indexer_keys_out(monkeypatch):
    """The plain call leaves the indexer keys lazy and below the sparse
    depth only a cache evaluation reads them, so an attention record that
    returned them would add their projection to every round and step."""
    lm, _ = _decode(monkeypatch, True)
    held = [(key, r) for layer in lm.model.layers
            if not layer.is_linear and "indexer" in layer.self_attn
            for key, r in vars(layer)["_verify_fused"][1].items()
            if isinstance(r, tuple)]
    assert held
    for (shape, dtype), r in held:
        assert len(kq.fused_call(r[0], [mx.zeros(shape, dtype)])) == 6


def test_rollbacks_and_steps_in_eval_mode(monkeypatch):
    def serve(lm):
        lm.eval()

    steps = (5, 6, 11, 17)
    _, plain = _walk(monkeypatch, False, before=serve, step_at=steps)
    lm, fused = _walk(monkeypatch, True, before=serve, step_at=steps)
    assert _records(lm) > 0
    _equal(plain, fused)


# Widths and kept counts as sampled text draws them: no pair of
# consecutive widths repeats early.
MIXED = [(S, max(1, (S * (i % 5 + 1)) // 5))
         for i, S in enumerate([2, 3, 4, 5, 2, 4, 3, 5, 5, 4, 2, 2, 3, 3, 5,
                                3, 2, 5, 2, 4, 4, 5, 3, 4, 2] * 3)]


def _gdn_layouts(lm) -> int:
    return max(len(vars(layer)["_verify_fused"][1])
               for layer in _gdn_layers(lm))


def test_a_record_serves_every_width_of_the_round_before(monkeypatch):
    """With the fixed row capacity a gated-delta layer's layouts are its
    own round's width and the pending flags; with per-width arrays they
    are pairs of widths, and mixed rounds keep meeting new ones."""
    _, plain = _walk(monkeypatch, False, rounds=MIXED)
    lm, fused = _walk(monkeypatch, True, rounds=MIXED)
    _equal(plain, fused)
    capped = _gdn_layouts(lm)
    stats = q4.verify_fused_stats(lm)
    assert stats["replays"] + stats["plain_runs"] > 0
    monkeypatch.setattr(q4, "_VERIFY_FUSED_CAP", False)
    lm, per_width = _walk(monkeypatch, True, rounds=MIXED)
    _equal(plain, per_width)
    # 4 widths after a round, and the first round's after the prefill.
    assert capped <= 5 < _gdn_layouts(lm)


def test_the_record_table_is_bounded(monkeypatch):
    """Over the bound the least recently used records are released; a
    released layout records again at its next use, with the same bits."""
    _, plain = _walk(monkeypatch, False, rounds=MIXED)
    monkeypatch.setattr(q4, "_FUSED_LIMITS", (8, 256 << 20))
    monkeypatch.setattr(q4, "_FUSED_HELD", [0, 0, 0])
    lm, fused = _walk(monkeypatch, True, rounds=MIXED)
    _equal(plain, fused)
    stats = q4.verify_fused_stats(lm)
    assert stats["evicted"] > 0
    assert 0 < stats["records"] <= 8 + 2
    # The byte bound, with every table entry priced at 1 MiB (the small
    # model's graphs keep no program).
    monkeypatch.setattr(q4, "_FUSED_LIMITS", (4096, 4 << 20))
    monkeypatch.setattr(q4, "_FUSED_HELD", [0, 0, 0])
    monkeypatch.setattr(q4, "_FUSED_CHECK_EVERY", 2)
    monkeypatch.setattr(q4, "_fused_bytes", lambda rec: 1 << 20)
    lm, fused = _walk(monkeypatch, True, rounds=MIXED)
    _equal(plain, fused)
    stats = q4.verify_fused_stats(lm)
    assert stats["evicted"] > 0 and 0 < stats["records"] <= 2 * 4 + 2


def _decode(monkeypatch, fused, steps=14, rounds=()):
    """Prefill, then one-token decode steps, with verify rounds (plain
    ones) and their rollbacks among them."""
    monkeypatch.setattr(q4, "_DECODE_FUSED", fused)
    kq.env_cache(fused)
    lm, args = _lm(True)
    lm.eval()
    mx.random.seed(11)
    cache = lm.make_cache()
    mx.eval(lm(mx.random.randint(0, args.vocab_size, (1, 9)),
               cache=cache).logits)
    out = []
    for i in range(steps):
        if i in rounds:
            block = mx.random.randint(0, args.vocab_size, (1, 4))
            hid, _, sink = lm.speculative_verify_hidden(block, cache)
            lm.rollback_speculative_cache(cache, sink, rounds[i] - 1, 4)
            out.append(np.array(hid.astype(mx.float32)))
        tok = mx.random.randint(0, args.vocab_size, (1, 1))
        logits = lm(tok, cache=cache).logits
        mx.eval(logits, [c.state for c in cache])
        out.append(np.array(logits.astype(mx.float32)))
    out.extend(np.array(a.astype(mx.float32))
               for c in cache for a in c.state if isinstance(a, mx.array))
    return lm, out


def test_decode_steps_through_records(monkeypatch):
    """GMLX_Q4_DECODE_FUSED: one-token steps carry the plain bits, alone
    and after a verify round left a pending replay (full accept and
    rollback)."""
    _, plain = _decode(monkeypatch, False)
    lm, fused = _decode(monkeypatch, True)
    assert _records(lm) > 0
    assert any(k[0] == "decode" for layer in _gdn_layers(lm)
               for k in vars(layer)["_verify_fused"][1])
    _equal(plain, fused)
    rounds = {3: 4, 5: 2, 6: 2, 9: 4, 10: 1, 12: 4}
    _, plain = _decode(monkeypatch, False, rounds=rounds)
    lm, fused = _decode(monkeypatch, True, rounds=rounds)
    assert _records(lm) > 0
    _equal(plain, fused)


def test_the_decode_and_verify_records_turn_the_kq_cache_on_once(monkeypatch):
    """Each call that turns kq's lever cache on drops the Metal calls the
    records have learned: with both switches on, the first verify after
    some decode steps leaves the decode records' calls in place."""
    from gmlx.models import kq_levers

    calls = []
    monkeypatch.setattr(kq, "env_cache", calls.append)
    monkeypatch.setattr(kq_levers, "_ON", [False])
    monkeypatch.setattr(q4, "_DECODE_FUSED", None)
    monkeypatch.setattr(q4, "_VERIFY_FUSED", None)
    monkeypatch.setenv("GMLX_Q4_DECODE_FUSED", "1")
    monkeypatch.setenv("GMLX_Q4_VERIFY_FUSED", "1")
    assert q4._decode_fused() and q4._verify_fused()
    assert calls == [True]


@pytest.mark.parametrize("fused", [False, True])
def test_a_verify_that_holds_its_rows_submits_none_and_carries_the_same_bits(
        monkeypatch, fused):
    """hold_verify_rows: a verify built while the rows are held submits
    nothing as it is built, and its hidden is the one of a verify that
    submitted its rows early."""
    def run(hold):
        monkeypatch.setattr(q4, "_VERIFY_FUSED", fused)
        monkeypatch.setattr(q4, "_PLE_EARLY", True)
        kq.env_cache(fused)
        lm, args = _lm(True)
        lm.eval()
        mx.random.seed(13)
        cache = lm.make_cache()
        mx.eval(lm(mx.random.randint(0, args.vocab_size, (1, 9)),
                   cache=cache).logits)
        out, early = [], []
        real = mx.async_eval
        for i in range(6):
            block = mx.random.randint(0, args.vocab_size, (1, 4))
            lm.hold_verify_rows(hold and i % 2 == 0)
            with monkeypatch.context() as m:
                m.setattr(mx, "async_eval",
                          lambda *a, **k: (early.append(i), real(*a, **k))[1])
                hid, _, sink = lm.speculative_verify_hidden(block, cache)
            lm.hold_verify_rows(False)
            lm.rollback_speculative_cache(cache, sink, 2, hid.shape[1])
            mx.eval(hid, [c.state for c in cache])
            out.append(np.array(hid.astype(mx.float32)))
        return out, early

    try:
        plain, early = run(False)
        assert set(early) == set(range(6))
        held, early = run(True)
        assert set(early) == {1, 3, 5}
    finally:
        kq.env_cache(False)
    for a, b in zip(plain, held):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("fused", [False, True])
def test_a_cut_block_takes_the_rows_of_the_whole_one(monkeypatch, fused):
    """reuse_verify_rows: after a verify of the whole block was built and
    dropped, the verify of its leading rows from the same cache state
    carries the bits of one that reads its own table rows, and reads none.
    Without the call, or from another state, the rows are read."""
    import copy

    def run(reuse):
        monkeypatch.setattr(q4, "_VERIFY_FUSED", fused)
        kq.env_cache(fused)
        lm, args = _lm(True)
        lm.eval()
        reads = []
        ple = next(layer.ple for layer in lm.model.layers if "ple" in layer)
        emb = type(ple.embedding)
        real = emb.__call__
        monkeypatch.setattr(emb, "__call__", lambda self, *a, **k: (
            reads.append(1), real(self, *a, **k))[1])
        mx.random.seed(13)
        cache = lm.make_cache()
        mx.eval(lm(mx.random.randint(0, args.vocab_size, (1, 9)),
                   cache=cache).logits)
        out = []
        for i, (S, cut, kept) in enumerate(
                [(4, 2, 2), (4, 3, 1), (4, 4, 4), (3, 1, 1), (4, 2, 2)] * 3):
            block = mx.random.randint(0, args.vocab_size, (1, S))
            snap = [(c, {k: (copy.copy(v) if isinstance(v, list) else v)
                         for k, v in vars(c).items()}) for c in cache]
            lm.speculative_verify_hidden(block, cache)
            n = len(reads)
            if cut < S:
                for c, state in snap:
                    vars(c).clear()
                    vars(c).update(state)
                if reuse:
                    lm.reuse_verify_rows(cache)
                hid, _, sink = lm.speculative_verify_hidden(block[:, :cut], cache)
                assert len(reads) - n == (0 if reuse else 1)
            else:
                hid, _, sink = lm.speculative_verify_hidden(
                    mx.random.randint(0, args.vocab_size, (1, 2)), cache)
                assert len(reads) - n == 1
            lm.rollback_speculative_cache(cache, sink, kept - 1, hid.shape[1])
            mx.eval(hid, [c.state for c in cache])
            out.append(np.array(hid.astype(mx.float32)))
        out.extend(np.array(a.astype(mx.float32))
                   for c in cache for a in c.state if isinstance(a, mx.array))
        return out

    _equal(run(False), run(True))
