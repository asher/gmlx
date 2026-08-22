"""DFlash 2 drafter on a tiny random qwen3_5 target (no GGUF, no weights).

Covers the target capture seam (``DFlashCaptureHooks`` on the owned qwen3_5
language model), the owned drafter's ring and block mask, the selector
lattice, the draw/stash routine shared with DFlash v1, and the engine
contract: whatever the drafter proposes, the verify walk emits the target's
own greedy tokens.

The reference-parity oracle for the conv and selector math lives in
``test_dflash2_reference.py`` (pinned fixtures from the real checkpoints);
the tests here are structural and fast.
"""

import dataclasses

import numpy as np
import mlx.core as mx
import pytest

from mlx_vlm.models.cache import BufferedRotatingKVCache, KVCache

from gmlx import qwen35_owned
from gmlx.dflash_drafter import (
    DFlash2Drafter,
    DFlashConfig,
    DFlashDrafter,
    _scatter,
    _second_choice,
    block_attention_mask,
    draw_rows,
    greedy_walk,
)
from gmlx.drafter_protocol import DraftStash

from test_vlm_mtp_gating import _cfg, _top

CAPTURE = (0, 2)
BLOCK = 4
TOP_K = 4
N_GEN = 16
# With tied embeddings every random draw of the tiny target collapses its
# greedy chain onto one token, which makes the identity claim vacuous; untied,
# this draw yields 15 distinct tokens over N_GEN steps at a 1.9e-2 minimum
# top-2 margin on the verify path.
SEED = 9
GREEDY_TIE_TOL = 1e-3


def _tcfg():
    return dataclasses.replace(_cfg(), tie_word_embeddings=False)


def _target(seed=SEED):
    mx.random.seed(seed)
    lm = qwen35_owned.language_model_class("qwen3_5")(_tcfg(), _top())
    mx.eval(lm.parameters())
    return lm


def _config(cfg, *, n_layers=2, block_size=BLOCK, native_block_size=None,
            window=512, layer_types=None, is_causal=None, top_k=TOP_K):
    return DFlashConfig(
        hidden_size=cfg.hidden_size,
        intermediate_size=64,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        vocab_size=cfg.vocab_size,
        max_position_embeddings=1024,
        rope_theta=10000.0,
        block_size=block_size,
        native_block_size=native_block_size,
        mask_token_id=7,
        target_layer_ids=list(CAPTURE),
        num_target_layers=cfg.num_hidden_layers,
        layer_types=layer_types or ["sliding_attention"] * n_layers,
        sliding_window=window,
        is_causal=is_causal,
        conv_kernel_size=2,
        conv_group_size=16,
        selector_rank=8,
        selector_top_k=top_k,
    )


def _drafter(lm=None, **kw):
    drafter = DFlash2Drafter(_config(_tcfg(), **kw))
    mx.eval(drafter.parameters())
    if lm is not None:
        drafter.reset(lm)
    return drafter


def _packed_width(cfg):
    return cfg.hidden_size * (1 + len(CAPTURE))


def _verify(lm, ids, cache):
    hid, _, gdn = lm.speculative_verify_hidden(ids, cache)
    return hid, gdn


# --- target capture seam ------------------------------------------------------

def test_bare_target_is_unchanged_when_not_armed():
    lm, cfg = _target(), _tcfg()
    hid, gdn = _verify(lm, mx.array([[1, 2, 3, 4]]), lm.make_cache())
    assert hid.shape[-1] == cfg.hidden_size
    assert lm._dflash_capture is None


def test_armed_prefill_packs_without_the_verify_path():
    """Prefill captures must not ride ``capture_layer_ids``: in the owned
    qwen3_5 layers that flag also routes every GEMM of a multi-row call
    through the verify kernels. The sink is independent, so a prefill keeps
    ``gdn_states`` unset while the hidden packs."""
    lm, cfg = _target(), _tcfg()
    lm.set_dflash_capture(CAPTURE)
    out = lm(mx.array([[1, 2, 3, 4, 5]]), cache=lm.make_cache(), return_hidden=True)
    assert out.hidden_states[-1].shape == (1, 5, _packed_width(cfg))
    assert out.gdn_states is None
    assert out.logits.shape == (1, 5, cfg.vocab_size)


def test_capture_order_is_trunk_then_layers_in_order():
    """Both sides on the verify path (``capture_layer_ids`` switches the owned
    layers to it), so the only difference is who collects the captures."""
    lm, cfg = _target(), _tcfg()
    prompt = mx.array([[1, 2, 3, 4]])
    h = cfg.hidden_size
    stock_cache, armed_cache = lm.make_cache(), lm.make_cache()
    lm(prompt, cache=stock_cache)
    lm(prompt, cache=armed_cache)

    blk = mx.array([[9, 8, 7]])
    stock = lm(blk, cache=stock_cache, capture_layer_ids=list(CAPTURE),
               return_hidden=True, skip_logits=True).hidden_states
    lm.set_dflash_capture(CAPTURE)
    packed, _ = _verify(lm, blk, armed_cache)
    mx.eval(packed, *stock)
    assert len(stock) == 3 and packed.shape == (1, 3, h * 3)
    parts = [packed[..., i * h:(i + 1) * h] for i in range(3)]
    assert float(mx.abs(parts[0] - stock[-1]).max().item()) == 0.0
    assert float(mx.abs(parts[1] - stock[0]).max().item()) == 0.0
    assert float(mx.abs(parts[2] - stock[1]).max().item()) == 0.0


def test_verify_returns_the_packed_hidden_with_gdn_states():
    lm, cfg = _target(), _tcfg()
    lm.set_dflash_capture(CAPTURE)
    cache = lm.make_cache()
    _verify(lm, mx.array([[1, 2, 3]]), cache)
    hid, gdn = _verify(lm, mx.array([[4] * BLOCK]), cache)
    assert hid.shape == (1, BLOCK, _packed_width(cfg))
    assert gdn is not None and len(gdn) > 0


def test_trunk_slice_recovers_the_unpacked_hidden():
    lm = _target()
    ids = mx.array([[1, 2, 3, 4]])
    bare, _ = _verify(lm, ids, lm.make_cache())
    lm.set_dflash_capture(CAPTURE)
    packed, _ = _verify(lm, ids, lm.make_cache())
    trunk = lm._dflash_trunk(packed)
    mx.eval(bare, trunk)
    assert trunk.shape == bare.shape
    assert float(mx.abs(trunk - bare).max().item()) == 0.0
    lm.set_dflash_capture(())
    ref = lm.speculative_logits_from_hidden(bare)
    lm.set_dflash_capture(CAPTURE)
    got = lm.speculative_logits_from_hidden(packed)
    mx.eval(ref, got)
    assert float(mx.abs(ref - got).max().item()) == 0.0


def test_trunk_is_materialized_before_the_logit_head(monkeypatch):
    """The real head is a quantized kernel that reads the buffer directly; a
    lazy strided view hands it the packed strides."""
    lm, cfg = _target(), _tcfg()
    lm.set_dflash_capture(CAPTURE)
    calls = []
    real = mx.contiguous
    monkeypatch.setattr(
        mx, "contiguous", lambda x, *a, **k: (calls.append(x.shape), real(x, *a, **k))[1])
    lm.speculative_logits_from_hidden(mx.zeros((1, 3, _packed_width(cfg))))
    assert calls


def test_argmax_hook_matches_the_logits_hook():
    lm = _target()
    lm.set_dflash_capture(CAPTURE)
    packed, _ = _verify(lm, mx.array([[1, 2, 3, 4]]), lm.make_cache())
    logits = lm.speculative_logits_from_hidden(packed)
    am = lm.speculative_argmax_from_hidden(packed)
    mx.eval(logits, am)
    assert am.tolist() == mx.argmax(logits, axis=-1).tolist()


def test_verify_logits_returns_correct_logits_and_the_packed_hidden():
    """Unreachable for this target today (the argmax and logits hooks answer
    first), pinned so a dispatch-order change cannot bring back a stride bug."""
    lm, cfg = _target(), _tcfg()
    ids = mx.array([[1, 2, 3, 4]])
    bare_cache = lm.make_cache()
    bare_hid, _ = _verify(lm, ids, bare_cache)
    ref = mx.argmax(lm.speculative_logits_from_hidden(bare_hid), axis=-1)

    lm.set_dflash_capture(CAPTURE)
    hid, _, gdn, tok = lm.speculative_verify_logits(
        ids, lm.make_cache(), lambda logits: mx.argmax(logits, axis=-1))
    mx.eval(hid, tok, ref)
    assert hid.shape == (1, 4, _packed_width(cfg))
    assert tok.tolist() == ref.tolist()


def test_rollback_trims_every_layer_cache():
    lm = _target()
    lm.set_dflash_capture(CAPTURE)
    cache = lm.make_cache()
    _verify(lm, mx.array([[1, 2, 3, 4, 5, 6]]), cache)
    attn = [c for c in cache if hasattr(c, "offset")]
    assert attn
    before = [c.offset for c in attn]
    _, gdn = _verify(lm, mx.array([[7] * BLOCK]), cache)
    assert all(c.offset == b + BLOCK for c, b in zip(attn, before))
    lm.rollback_speculative_cache(cache, gdn, 1, BLOCK)
    assert all(c.offset == b + 2 for c, b in zip(attn, before))


# --- drafter -----------------------------------------------------------------

def test_drafter_satisfies_the_protocol():
    from gmlx.drafter_protocol import validate_drafter

    lm = _target()
    drafter = _drafter(lm)
    validate_drafter(drafter)
    assert drafter.kind_label == "dflash2"
    assert drafter.uses_shared_kv is False
    assert drafter.requires_owned_engine is True
    assert drafter.supports_q_stash is True
    assert drafter.hidden_capture_limit == 511


def test_dflash2_needs_a_selector_and_a_conv():
    cfg = _tcfg()
    with pytest.raises(ValueError, match="selector_top_k"):
        DFlash2Drafter(_config(cfg, top_k=0))
    base = _config(cfg)
    base.conv_kernel_size = 0
    with pytest.raises(ValueError, match="conv_kernel_size"):
        DFlash2Drafter(base)
    assert not _config(cfg, top_k=0).is_dflash2
    assert isinstance(DFlashDrafter(_config(cfg, top_k=0)), DFlashDrafter)


def test_make_cache_uses_temporal_rings_sized_to_the_window():
    drafter = _drafter(window=8, layer_types=["sliding_attention", "full_attention"])
    ring, full = drafter.make_cache()
    assert type(ring) is BufferedRotatingKVCache
    assert ring.max_size == 7 and ring.keep == 0 and ring.buffer_size == 64
    assert type(full) is KVCache
    with pytest.raises(NotImplementedError):
        drafter.make_cache(left_padding=[0])


def test_captures_take_the_trailing_block_and_reject_a_bare_hidden():
    cfg = _tcfg()
    drafter = _drafter()
    h = cfg.hidden_size
    packed = mx.concatenate(
        [mx.zeros((1, 2, h)), mx.ones((1, 2, h)), mx.full((1, 2, h), 2.0)], axis=-1)
    caps = drafter._captures(packed)
    mx.eval(caps)
    assert caps.shape[-1] == h * len(CAPTURE)
    assert float(caps[0, 0, 0].item()) == 1.0
    assert float(caps[0, 0, h].item()) == 2.0
    with pytest.raises(ValueError, match="packed hidden width"):
        drafter._captures(mx.zeros((1, 2, h)))


def test_greedy_draft_block_shape_and_ceiling():
    lm = _target()
    drafter = _drafter(lm, block_size=2, native_block_size=BLOCK)
    drafts = drafter.draft_block(3, None, None, BLOCK, None, greedy=True)
    mx.eval(drafts)
    assert drafts.shape == (1, BLOCK - 1)
    assert all(0 <= t < _tcfg().vocab_size for t in drafts[0].tolist())
    with pytest.raises(RuntimeError, match="at most"):
        drafter.draft_block(3, None, None, BLOCK + 1, None, greedy=True)
    cold = _drafter()
    with pytest.raises(RuntimeError, match="reset"):
        cold.draft_block(3, None, None, BLOCK, None, greedy=True)


# --- selector lattice ---------------------------------------------------------

def test_lattice_matches_a_numpy_transcription_and_greedy_walk_is_argmax():
    drafter = _drafter()
    sel = drafter.candidate_selector
    L, H, V, k = BLOCK - 1, _tcfg().hidden_size, _tcfg().vocab_size, TOP_K
    mx.random.seed(11)
    hidden = mx.random.normal((L, H))
    logits = mx.random.normal((L, V))
    anchor = mx.array(5)
    cands, first, edges = sel.lattice(hidden, logits, anchor)
    mx.eval(cands, first, edges)

    hp = np.array(hidden) @ np.array(sel.hidden_projection.weight).T
    pred = np.array(sel.predecessor_codebook.weight)
    succ = np.array(sel.successor_codebook.weight)
    lg = np.array(logits)
    c = np.array(cands)
    for p in range(L):
        assert set(c[p].tolist()) == set(np.argsort(lg[p])[-k:].tolist())
    ref_first = lg[0][c[0]] + succ[c[0]] @ (pred[5] * hp[0])
    assert np.allclose(np.array(first), ref_first, atol=1e-4)
    for p in range(1, L):
        ref = lg[p][c[p]][None, :] + (pred[c[p - 1]] * hp[p][None, :]) @ succ[c[p]].T
        assert np.allclose(np.array(edges[p - 1]), ref, atol=1e-4)

    path = greedy_walk(cands, first, edges)
    sel_i = int(np.argmax(ref_first))
    ref_path = [int(c[0][sel_i])]
    for p in range(1, L):
        row = np.array(edges[p - 1])[sel_i]
        sel_i = int(np.argmax(row))
        ref_path.append(int(c[p][sel_i]))
    assert path.tolist() == ref_path


def test_greedy_draft_is_identical_with_and_without_a_stash():
    lm, cfg = _target(), _tcfg()
    drafter = _drafter(lm)
    plain = drafter.draft_block(3, None, None, BLOCK, None, greedy=True)
    stash = DraftStash(pq=[], top2=[])
    logged = drafter.draft_block(3, None, None, BLOCK, None, greedy=True, stash=stash)
    mx.eval(plain, logged)
    assert plain.tolist() == logged.tolist()
    assert len(stash.pq) == BLOCK - 1 and len(stash.top2) == BLOCK - 1
    for row, tok, second in zip(stash.pq, logged[0].tolist(), stash.top2):
        row = np.array(row)
        assert row.shape == (cfg.vocab_size,)
        finite = np.isfinite(row)
        assert finite.sum() == TOP_K
        assert int(np.argmax(row)) == tok
        assert int(second.item()) != tok and finite[int(second.item())]


# --- block mask and ring ------------------------------------------------------

def test_block_mask_cases():
    assert block_attention_mask(5, 3, None, False) is None
    assert block_attention_mask(4, 3, 8, False) is None
    m = np.array(block_attention_mask(20, 4, 8, False))
    assert m.shape == (4, 24)
    for i in range(4):
        ctx = np.nonzero(m[i, :20])[0].tolist()
        assert ctx == list(range(13 + i, 20)), f"row {i} keeps {ctx}"
        assert m[i, 20:].all()
    causal = np.array(block_attention_mask(20, 4, 8, True))
    assert (causal[:, :20] == m[:, :20]).all()
    assert (causal[:, 20:] == np.tri(4, dtype=bool)).all()
    only_causal = np.array(block_attention_mask(3, 4, None, True))
    assert only_causal[:, :3].all()
    assert (only_causal[:, 3:] == np.tri(4, dtype=bool)).all()


def _inject(drafter, n, seed):
    mx.random.seed(seed)
    caps = mx.random.normal((1, n, _tcfg().hidden_size * len(CAPTURE)))
    drafter.append_context(caps)
    return caps


def _draft_hidden(drafter, block):
    h = drafter._draft_hidden(block)
    mx.eval(h)
    return np.array(h)


@pytest.mark.parametrize("n_rows", [20, 200])
def test_ring_is_temporal_and_the_window_trims_the_slack(n_rows):
    """With window 8 the ring keeps 7 rows plus 64 rows of rollback slack; a
    block row i must see exactly the 7-i most recent keys. A rotated ring, or
    a mask that ignores the slack rows, changes the draft hidden. Rows go in
    round-sized inserts, the way the engine commits them."""
    lm = _target()
    block = mx.array([[3, 7, 7, 7]])
    drafter = _drafter(lm, window=8)
    mx.random.seed(5)
    caps = mx.random.normal((1, n_rows, _tcfg().hidden_size * len(CAPTURE)))
    for i in range(0, n_rows, BLOCK):
        drafter.append_context(caps[:, i:i + BLOCK])
    ring = drafter._cache[0]
    assert ring.offset == n_rows
    held = ring.state[0].shape[2]
    assert 7 <= held <= 7 + 64 + BLOCK
    if n_rows > 7 + 64:
        assert held < n_rows, "the ring never compacted"
    want = _draft_hidden(drafter, block)

    drafter.reset(lm)
    drafter.append_context(caps[:, -7:])
    got = _draft_hidden(drafter, block)
    assert np.allclose(want, got, atol=1e-4), np.abs(want - got).max()

    drafter.reset(lm)
    drafter.append_context(caps[:, -6:])
    assert not np.allclose(want, _draft_hidden(drafter, block), atol=1e-4)


def test_sliding_layers_mask_and_full_layers_do_not(monkeypatch):
    lm = _target()
    drafter = _drafter(lm, window=8, layer_types=["sliding_attention", "full_attention"])
    _inject(drafter, 12, seed=1)
    masks = []
    real = mx.fast.scaled_dot_product_attention
    monkeypatch.setattr(
        mx.fast, "scaled_dot_product_attention",
        lambda *a, **k: (masks.append(k.get("mask")), real(*a, **k))[1])
    drafter._draft_hidden(mx.array([[3, 7, 7, 7]]))
    assert len(masks) == 2
    assert masks[0] is not None and masks[0].shape == (4, 16)
    assert masks[1] is None
    assert drafter._cache[1].offset == 12


def test_causal_flag_adds_the_block_triangle(monkeypatch):
    lm = _target()
    masks = []
    real = mx.fast.scaled_dot_product_attention
    monkeypatch.setattr(
        mx.fast, "scaled_dot_product_attention",
        lambda *a, **k: (masks.append(k.get("mask")), real(*a, **k))[1])
    _drafter(lm, window=8, n_layers=1)._draft_hidden(mx.array([[3, 7, 7, 7]]))
    _drafter(lm, window=8, n_layers=1, is_causal=True)._draft_hidden(mx.array([[3, 7, 7, 7]]))
    assert masks[0] is None
    assert (np.array(masks[1]) == np.tri(4, dtype=bool)).all()


# --- draw / stash -------------------------------------------------------------

def _rows(n=3, w=16, v=128, seed=2):
    mx.random.seed(seed)
    rows = mx.random.normal((n, w))
    support = mx.stack([mx.random.permutation(v)[:w] for _ in range(n)])
    mx.eval(rows, support)
    return rows, support


def test_scatter_sets_finite_values_on_a_minus_inf_base():
    rows, support = _rows()
    out = np.array(_scatter(rows, support, float("-inf"), 128))
    assert out.shape == (3, 128)
    assert np.isfinite(out).sum(axis=-1).tolist() == [16, 16, 16]
    for i in range(3):
        assert np.allclose(out[i][np.array(support[i])], np.array(rows[i]))
    assert _scatter(rows, None, 0.0, 128) is rows


def test_second_choice_is_the_runner_up():
    rows, _ = _rows()
    r = np.array(rows)
    assert _second_choice(rows).tolist() == [int(np.argsort(x)[-2]) for x in r]


def test_stochastic_draw_stashes_the_array_it_drew_from(monkeypatch):
    from gmlx.speculative import _STOCH_DRAFT, _pq_probs

    rows, support = _rows()
    seen = []
    real = mx.random.categorical
    monkeypatch.setattr(mx.random, "categorical",
                        lambda x, **k: (seen.append(x), real(x, **k))[1])
    stash = DraftStash(q=[], pq=[], top2=[])
    toks, idx = draw_rows(rows, support, vocab=128, greedy=False, sampler=None, stash=stash)
    mx.eval(toks, idx)
    assert len(seen) == 1 and len(stash.q) == 3
    drawn_from = np.exp(np.array(seen[0]))
    q = np.stack([np.array(x) for x in stash.q])
    assert q.shape == (3, 128)
    for i in range(3):
        assert np.allclose(q[i][np.array(support[i])], drawn_from[i], atol=1e-6)
        assert q[i].sum() == pytest.approx(1.0, abs=1e-5)
        assert q[i][int(toks[i])] > 0
        assert int(support[i][int(idx[i])]) == int(toks[i])
    ref = np.array(_pq_probs(rows, *_STOCH_DRAFT))
    assert np.allclose(drawn_from, ref, atol=1e-6)


def test_stochastic_draw_frequencies_match_the_stashed_q():
    rows, support = _rows(n=1)
    counts = np.zeros(128)
    mx.random.seed(0)
    n = 2000
    q = None
    for _ in range(n):
        stash = DraftStash(q=[])
        toks, _ = draw_rows(rows, support, vocab=128, greedy=False, sampler=None, stash=stash)
        counts[int(toks[0])] += 1
        q = np.array(stash.q[0])
    assert np.abs(counts / n - q).max() < 0.05


def _annotated(temp=1.0, top_p=0.95, top_k=20, min_p=0.0):
    from mlx_lm.sample_utils import make_sampler

    from gmlx.speculative import annotate_sampling_params

    sampler = make_sampler(temp=temp, top_p=top_p, top_k=top_k, min_p=min_p)
    annotate_sampling_params(sampler, temp=temp, top_p=top_p, top_k=top_k, min_p=min_p)
    return sampler


def test_the_cli_sampler_cannot_take_a_compact_row():
    """Why the reconstruction exists. mlx_lm's sampler is compiled, and a
    ValueError raised mid-trace leaves ``mx.random.state`` holding a tracer:
    every later random draw in the process fails to eval until a reseed."""
    rows, _ = _rows()
    with pytest.raises(ValueError, match="top_k"):
        _annotated()(rows)
    mx.random.seed(0)
    mx.eval(mx.random.normal((1,)))


def test_annotated_sampler_is_reconstructed_on_the_compact_row(monkeypatch):
    from gmlx.speculative import _pq_probs

    rows, support = _rows()
    sampler = _annotated()
    seen = []
    real = mx.random.categorical
    monkeypatch.setattr(mx.random, "categorical",
                        lambda x, **k: (seen.append(x), real(x, **k))[1])
    toks, idx = draw_rows(rows, support, vocab=128, greedy=False, sampler=sampler, stash=None)
    mx.eval(toks, idx)
    assert len(seen) == 1
    ref = np.array(_pq_probs(rows, 1.0, 16, 0.95))
    assert np.allclose(np.exp(np.array(seen[0])), ref, atol=1e-6)
    # top-p trims the tail: the reconstructed row is not the plain softmax
    assert (ref == 0).any()
    for i in range(3):
        assert int(support[i][int(idx[i])]) == int(toks[i])


def test_filtered_sampler_is_called_on_the_compact_row():
    rows, support = _rows()
    calls = []

    class _Serve:
        temperature = 1.0

        def _filtered(self, x):
            return x

        def __call__(self, x):
            calls.append(x.shape)
            return mx.argmax(x, axis=-1)

    toks, idx = draw_rows(rows, support, vocab=128, greedy=False, sampler=_Serve(), stash=None)
    mx.eval(toks)
    assert calls == [(3, 16)]
    assert toks.tolist() == [int(support[i][int(np.argmax(np.array(rows[i])))]) for i in range(3)]


def test_opaque_sampler_gets_the_scattered_row():
    rows, support = _rows()
    calls = []

    def sampler(x):
        calls.append(x.shape)
        return mx.argmax(x, axis=-1)

    toks, idx = draw_rows(rows, support, vocab=128, greedy=False, sampler=sampler, stash=None)
    mx.eval(toks, idx)
    assert calls == [(3, 128)]
    for i in range(3):
        assert int(support[i][int(idx[i])]) == int(toks[i])


def test_full_width_rows_pass_through_the_sampler():
    rows, _ = _rows(w=128)
    calls = []

    def sampler(x):
        calls.append(x.shape)
        return mx.argmax(x, axis=-1)

    stash = DraftStash(pq=[], top2=[])
    toks, idx = draw_rows(rows, None, vocab=128, greedy=False, sampler=sampler, stash=stash)
    mx.eval(toks)
    assert calls == [(3, 128)]
    assert toks.tolist() == idx.tolist()
    assert len(stash.pq) == 3 and stash.pq[0].shape == (128,)


def test_logging_never_changes_a_sampled_draft():
    lm = _target()
    drafter = _drafter(lm)
    sampler = _annotated()
    mx.random.seed(9)
    plain = drafter.draft_block(3, None, None, BLOCK, sampler)
    mx.eval(plain)
    mx.random.seed(9)
    stash = DraftStash(pq=[], top2=[])
    logged = drafter.draft_block(3, None, None, BLOCK, sampler, stash=stash)
    mx.eval(logged)
    assert plain.tolist() == logged.tolist()
    assert len(stash.pq) == BLOCK - 1 and len(stash.top2) == BLOCK - 1


def test_stochastic_draft_walks_the_realized_predecessor():
    """Row p is the lattice row of the token drawn at p-1: the stashed q at p
    must be a distribution whose support is position p's candidates, and the
    drawn token must be on it."""
    lm = _target()
    drafter = _drafter(lm)
    mx.random.seed(4)
    stash = DraftStash(q=[], pq=[], top2=[])
    drafts = drafter.draft_block(3, None, None, BLOCK, None, stash=stash)
    mx.eval(drafts)
    assert drafts.shape == (1, BLOCK - 1)
    assert len(stash.q) == BLOCK - 1
    for q, pq, tok in zip(stash.q, stash.pq, drafts[0].tolist()):
        q, pq = np.array(q), np.array(pq)
        assert q.sum() == pytest.approx(1.0, abs=1e-5)
        assert q[tok] > 0
        assert np.isfinite(pq).sum() == TOP_K
        assert set(np.nonzero(q)[0]) <= set(np.nonzero(np.isfinite(pq))[0])


def test_pq_graph_separates_the_sweep_on_a_padded_row():
    from gmlx.speculative import _PQ_SWEEP, _pq_graph

    lm = _target()
    drafter = _drafter(lm)
    stash = DraftStash(pq=[], top2=[])
    drafter.draft_block(3, None, None, BLOCK, None, greedy=True, stash=stash)
    mx.random.seed(6)
    rows = np.stack([np.array(r) for r in stash.pq])
    target = mx.array(np.where(np.isfinite(rows), rows, -6.0))
    target = target + mx.random.normal(target.shape) * 0.5
    stats = np.array(_pq_graph(target, stash.pq))
    assert stats.shape == (1 + len(_PQ_SWEEP), BLOCK - 1)
    assert np.isfinite(stats).all()
    assert (stats[1:] <= 1.0 + 1e-6).all() and (stats[1:] >= 0).all()
    assert len({tuple(np.round(r, 6)) for r in stats[1:]}) > 1


def test_v1_drafter_rides_the_same_draw_routine():
    from test_muse_glimmer_mtp import _build, _build_drafter

    lm, cfg = _build()
    drafter = _build_drafter(cfg)
    mx.eval(drafter.parameters())
    drafter.reset(lm)
    mx.random.seed(1)
    stash = DraftStash(q=[], pq=[], top2=[])
    drafts = drafter.draft_block(3, None, None, BLOCK, None, stash=stash)
    mx.eval(drafts)
    assert drafts.shape == (1, BLOCK - 1)
    assert len(stash.q) == BLOCK - 1 and stash.q[0].shape == (cfg["vocab_size"],)
    assert len(stash.pq) == BLOCK - 1 and len(stash.top2) == BLOCK - 1
    sampler = _annotated()
    mx.random.seed(2)
    plain = drafter.draft_block(3, None, None, BLOCK, sampler)
    mx.eval(plain)
    mx.random.seed(2)
    logged = drafter.draft_block(3, None, None, BLOCK, sampler, stash=DraftStash(pq=[], top2=[]))
    mx.eval(logged)
    assert plain.tolist() == logged.tolist()


# --- engine contract ----------------------------------------------------------

def _greedy_reference(lm, prompt, n):
    """Greedy chain through the verify path. The owned qwen3_5 verify path is
    fp32-accurate but not bit-identical to plain decode, so the reference
    and the walk must share it for the identity claim to be exact."""
    cache = lm.make_cache()
    hid, _ = _verify(lm, prompt, cache)
    logits = lm.speculative_logits_from_hidden(hid)[0, -1]
    ref, margins = [], []
    for _ in range(n):
        top = mx.sort(logits)[-2:]
        margins.append(float((top[1] - top[0]).item()))
        t = int(mx.argmax(logits).item())
        ref.append(t)
        hid, _ = _verify(lm, mx.array([[t]]), cache)
        logits = lm.speculative_logits_from_hidden(hid)[0, -1]
    return ref, margins


def test_verify_walk_is_token_identical_to_greedy():
    """The drafter proposes through its own lattice from random weights, so
    most rounds reject early; the walk must still emit the target's greedy
    chain and keep the drafter's ring in step with the target cache."""
    lm = _target()
    prompt = mx.array([[1, 2, 3, 4, 5]])
    ref, margins = _greedy_reference(lm, prompt, N_GEN)
    assert min(margins) > GREEDY_TIE_TOL, (
        "the pinned draw no longer has an unambiguous greedy chain; choose "
        "another SEED rather than weakening the identity claim")
    assert len(set(ref)) >= 4

    drafter = _drafter(lm)
    lm.set_dflash_capture(CAPTURE)
    cache = lm.make_cache()
    fa = next(c for c in cache if hasattr(c, "offset"))
    hid, gdn = _verify(lm, prompt, cache)
    tok = int(lm.speculative_argmax_from_hidden(hid)[0, -1].item())
    drafter.prefill_from_target_hidden(prompt, hid, tok, None, greedy=True)
    assert drafter._cache[0].offset == prompt.shape[1]
    got, accepts = [tok], []
    while len(got) < N_GEN:
        drafts = drafter.draft_block(tok, None, None, BLOCK, None, greedy=True)
        mx.eval(drafts)
        drafts = drafts[0].tolist()
        hid, gdn = _verify(lm, mx.array([[tok] + drafts]), cache)
        rows = lm.speculative_argmax_from_hidden(hid)[0].tolist()
        accepted = 0
        for d, r in zip(drafts, rows):
            if int(r) != d:
                break
            accepted += 1
        accepts.append(accepted)
        got.extend(drafts[:accepted])
        tok = int(rows[accepted])
        got.append(tok)
        lm.rollback_speculative_cache(cache, gdn, accepted, BLOCK)
        drafter.accept_verified_tokens(hid, mx.array([drafts]), accepted, [], None, greedy=True)
        assert drafter._cache[0].offset == fa.offset
    assert got[:N_GEN] == ref


def _engine_walk(lm, drafter, prompt, n, *, sampler=None):
    """Drive the engine's round loop the way generate_speculative seeds it."""
    from gmlx import speculative as sp

    lm.set_dflash_capture(CAPTURE)
    cache = lm.make_cache()
    out = lm(prompt, cache=cache, return_hidden=True, return_shared_kv=True)
    hidden = out.hidden_states[-1]
    first = out.logits[:, -1, :]
    b = int((mx.argmax(first, axis=-1) if sampler is None else sampler(first)).item())
    sp._buffer_mtp_target_cache(cache, drafter, BLOCK)
    toks = [b]
    toks.extend(sp._owned_decode_rounds(
        lm, drafter, lm, cache, hidden=hidden, b=b,
        shared_kv=out.shared_kv_states, seed_tokens=prompt, emitted=1,
        max_tokens=n, sampler=sampler, draft_block_size=BLOCK))
    return toks


def _engine_reference(lm, prompt, n):
    """Greedy chain the way the engine sees it: plain prefill picks the first
    token, every later token comes off the verify path."""
    cache = lm.make_cache()
    logits = lm(prompt, cache=cache).logits[0, -1]
    ref = []
    for _ in range(n):
        t = int(mx.argmax(logits).item())
        ref.append(t)
        hid, _ = _verify(lm, mx.array([[t]]), cache)
        logits = lm.speculative_logits_from_hidden(hid)[0, -1]
    return ref


def test_engine_rounds_emit_the_greedy_chain():
    lm = _target()
    prompt = mx.array([[1, 2, 3, 4, 5]])
    ref = _engine_reference(lm, prompt, N_GEN)
    assert len(set(ref)) >= 4
    got = _engine_walk(lm, _drafter(lm), prompt, N_GEN)
    assert got[:N_GEN] == ref


def _recording(drafter, monkeypatch):
    """Record, per draft_block call, the stash row counts after the call (the
    engine clears the stash lists once a round is walked)."""
    seen = []
    orig = drafter.draft_block

    def draft_block(*a, **kw):
        out = orig(*a, **kw)
        st = kw.get("stash")
        seen.append(None if st is None else {
            "rows": int(out.shape[1]),
            **{f: (None if getattr(st, f) is None else len(getattr(st, f)))
               for f in ("q", "pq", "top2")}})
        return out

    monkeypatch.setattr(drafter, "draft_block", draft_block)
    return seen


@pytest.mark.parametrize("sampled", [False, True])
def test_plain_rounds_pass_no_stash(monkeypatch, sampled):
    from gmlx import speculative as sp

    for name in ("_PQ_LOG", "_TOP2_LOG", "_STOCH_ACCEPT"):
        monkeypatch.setattr(sp, name, False)
    lm = _target()
    drafter = _drafter(lm)
    seen = _recording(drafter, monkeypatch)
    mx.random.seed(SEED)
    _engine_walk(lm, drafter, mx.array([[1, 2, 3, 4, 5]]), 8,
                 sampler=_annotated() if sampled else None)
    assert seen and all(s is None for s in seen)


def test_v1_rows_keep_exact_match_under_stochastic(monkeypatch, caplog):
    """Independent block rows are not a conditional proposal; the engine
    keeps exact-match for such drafters (measured on Muse v1: 6.7 -> 1.7
    accepted per round when forced)."""
    from gmlx import speculative as sp
    from gmlx.muse_glimmer_dflash import MuseGlimmerDFlashDrafter

    assert MuseGlimmerDFlashDrafter.stochastic_draft is False
    assert DFlash2Drafter.stochastic_draft is True
    for name in ("_PQ_LOG", "_TOP2_LOG"):
        monkeypatch.setattr(sp, name, False)
    monkeypatch.setattr(sp, "_STOCH_ACCEPT", True)
    lm = _target()
    drafter = _drafter(lm)
    monkeypatch.setattr(type(drafter), "stochastic_draft", False)
    seen = _recording(drafter, monkeypatch)
    mx.random.seed(SEED)
    with caplog.at_level("WARNING"):
        _engine_walk(lm, drafter, mx.array([[1, 2, 3, 4, 5]]), 8, sampler=_annotated())
    assert seen and all(s is None for s in seen)
    assert "independent rows" in caplog.text and "opaque" not in caplog.text


@pytest.mark.parametrize("switch", ["_PQ_LOG", "_TOP2_LOG", "_STOCH_ACCEPT"])
def test_instrumented_rounds_hand_the_block_drafter_a_stash(monkeypatch, switch):
    """One stash per round with one entry per drafted row, and the round loop
    walks it without a head-shaped wrapper."""
    from gmlx import speculative as sp

    for name in ("_PQ_LOG", "_TOP2_LOG", "_STOCH_ACCEPT"):
        monkeypatch.setattr(sp, name, name == switch)
    lm = _target()
    drafter = _drafter(lm)
    seen = _recording(drafter, monkeypatch)
    mx.random.seed(SEED)
    toks = _engine_walk(lm, drafter, mx.array([[1, 2, 3, 4, 5]]), 12,
                        sampler=_annotated())
    assert len(toks) >= 12 and all(isinstance(t, int) for t in toks)
    field = {"_PQ_LOG": "pq", "_TOP2_LOG": "top2", "_STOCH_ACCEPT": "q"}[switch]
    assert seen and seen[0]["rows"] == BLOCK - 1
    for s in seen:
        assert s is not None
        assert s[field] == s["rows"]
        assert all(s[o] is None for o in ("q", "pq", "top2") if o != field)
