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
import os

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import pytest

from gmlx._vendor.mlx_vlm.models.cache import BufferedRotatingKVCache, KVCache

import gmlx.models.qwen35.owned as qwen35_owned
import gmlx.spec.dflash_drafter as dd
from gmlx.spec.dflash_drafter import (
    DFlash2Drafter,
    DFlashConfig,
    DFlashDrafter,
    _scatter,
    _second_choice,
    block_attention_mask,
    draw_rows,
    greedy_walk,
)
from gmlx.spec.drafter_protocol import DraftStash

from test_vlm_mtp_gating import _cfg, _top

CAPTURE = (0, 2)
BLOCK = 4

# speculative_verify_logits runs upstream's lm_head verify GEMV, a Metal
# kernel gated on import-time availability, not the active device.
_NEEDS_GPU = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="upstream's lm_head verify GEMV is Metal-only")
TOP_K = 4
N_GEN = 16
# With tied embeddings every random draw of the tiny target collapses its
# greedy chain onto one token, which makes the identity claim vacuous; untied,
# this draw yields 15 distinct tokens over N_GEN steps at a 1.9e-2 minimum
# top-2 margin on the verify path.
SEED = 9
GREEDY_TIE_TOL = 1e-3


def _tcfg():
    # The V heads are tiled, as in the GGUF a speculative target loads from.
    return dataclasses.replace(_cfg(), tie_word_embeddings=False,
                               kv_head_layout="tiled")


def _target(seed=SEED):
    mx.random.seed(seed)
    lm = qwen35_owned.language_model_class("qwen3_5")(_tcfg(), _top())
    mx.eval(lm.parameters())
    return lm


def _config(cfg, *, n_layers=2, block_size=BLOCK, native_block_size=None,
            window=512, layer_types=None, is_causal=None, top_k=TOP_K,
            **extra):
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
        **extra,
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


@_NEEDS_GPU
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
    from gmlx.spec.drafter_protocol import validate_drafter

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
    assert ring.max_size == 7 and ring.keep == 0
    assert ring.buffer_size == dd._RING_SLACK
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


# --- draft-side head ----------------------------------------------------------

def _head_patched(monkeypatch, mode):
    from gmlx.spec.mtp_load import _patch_draft_head_quantized

    if mode is None:
        monkeypatch.delenv("GMLX_DRAFT_HEAD", raising=False)
    else:
        monkeypatch.setenv("GMLX_DRAFT_HEAD", mode)
    lm, drafter = _target(), _drafter()
    _patch_draft_head_quantized(drafter, "lm_head")
    drafter.reset(lm)
    return lm, drafter


@pytest.mark.parametrize("mode", [None, "q4"])
def test_draft_head_is_a_quantized_copy_and_the_target_keeps_its_own(
        monkeypatch, mode):
    """The drafter's logits come from a quantized copy of a float target
    head. The target keeps its own head, so verify is unchanged, and a
    second reset reinstalls the copy after bind restores the target's."""
    lm, drafter = _head_patched(monkeypatch, mode)
    target_head = lm.lm_head
    h = mx.random.normal((1, 3, _tcfg().hidden_size), key=mx.random.key(4))
    ref = target_head(h)
    for _ in range(2):
        assert drafter.lm_head is not target_head
        assert "lm_head" not in drafter
        got = drafter._logits(h)
        rel = (mx.abs(got - ref).max() / mx.abs(ref).max()).item()
        assert rel < (2e-2 if mode is None else 1.5e-1)
        drafter.reset(lm)
    assert lm.lm_head is target_head
    assert target_head.weight.dtype != mx.uint8


def test_draft_head_f16_keeps_the_target_head(monkeypatch):
    lm, drafter = _head_patched(monkeypatch, "f16")
    assert drafter.lm_head is lm.lm_head


# --- selector lattice ---------------------------------------------------------

def test_lattice_matches_a_numpy_transcription_and_greedy_walk_is_argmax():
    # The seed comes before the drafter: its selector weights are random
    # draws too, and the tolerance below is sized for this draw.
    mx.random.seed(11)
    drafter = _drafter()
    sel = drafter.candidate_selector
    L, H, V, k = BLOCK - 1, _tcfg().hidden_size, _tcfg().vocab_size, TOP_K
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
    # Not 1e-4: MLX dispatches the f32 GEMMs here through the tensor cores
    # at TF32 precision where available, so the lattice scores differ from
    # the full-f32 numpy transcription: by 3.7e-4 for this draw, and by
    # 3.3e-4 to 2.2e-3 over 64 seeds.
    ref_first = lg[0][c[0]] + succ[c[0]] @ (pred[5] * hp[0])
    assert np.allclose(np.array(first), ref_first, atol=2e-3)
    for p in range(1, L):
        ref = lg[p][c[p]][None, :] + (pred[c[p - 1]] * hp[p][None, :]) @ succ[c[p]].T
        assert np.allclose(np.array(edges[p - 1]), ref, atol=2e-3)

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


def test_block_mask_context_window_is_the_same_span_for_every_row():
    assert block_attention_mask(8, 4, None, False, context=8) is None
    m = np.broadcast_to(
        np.array(block_attention_mask(20, 4, None, False, context=8)), (4, 24))
    for i in range(4):
        assert np.nonzero(m[i, :20])[0].tolist() == list(range(12, 20))
        assert m[i, 20:].all()
    both = np.array(block_attention_mask(20, 4, 6, False, context=8))
    for i in range(4):
        keep = np.nonzero(both[i, :20])[0].tolist()
        assert keep == list(range(max(12, 15 + i), 20)), f"row {i} keeps {keep}"


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
def test_ring_is_temporal_and_the_window_trims_the_slack(n_rows, monkeypatch):
    """With window 8 the ring keeps 7 rows plus 64 rows of rollback slack; a
    block row i must see exactly the 7-i most recent keys. A rotated ring, or
    a mask that ignores the slack rows, changes the draft hidden. Rows go in
    round-sized inserts, the way the engine commits them."""
    monkeypatch.setattr(dd, "_RING_SLACK", 64)
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
    # Not 1e-4: TF32 tensor-core GEMMs accumulate ~2e-3 across the drafter
    # forward; the trimmed-vs-fresh split this test pins diverges by ~O(1).
    assert np.allclose(want, got, atol=5e-3), np.abs(want - got).max()

    drafter.reset(lm)
    drafter.append_context(caps[:, -6:])
    assert not np.allclose(want, _draft_hidden(drafter, block), atol=1e-4)


@pytest.mark.parametrize("in_place", [True, False])
def test_sliding_layers_mask_and_full_layers_do_not(monkeypatch, in_place):
    lm = _target()
    monkeypatch.setattr(dd, "_RING_IN_PLACE", in_place)
    drafter = _drafter(lm, window=8, layer_types=["sliding_attention", "full_attention"])
    _inject(drafter, 12, seed=1)
    masks, keys = [], []
    real = mx.fast.scaled_dot_product_attention

    def spy(*a, **k):
        masks.append(k.get("mask"))
        keys.append(a[1].shape[2])
        return real(*a, **k)

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", spy)
    drafter._draft_hidden(mx.array([[3, 7, 7, 7]]))
    assert len(masks) == 2
    # The sliding layer keeps only the window - 1 rows it can see. Read in
    # place, a full ring orders its keys so the window is a causal mask.
    assert keys[0] == 11
    if in_place:
        assert isinstance(masks[0], str) and masks[0] == "causal"
    else:
        assert masks[0] is not None and masks[0].shape == (4, 11)
    assert masks[1] is None
    assert [c.offset for c in drafter._cache] == [7, 12]


@pytest.mark.parametrize("kw, n_rows", [
    (dict(window=8), 2),            # open window: no mask
    (dict(window=8), 5),            # the block crosses the window: array mask
    (dict(window=8), 7),            # full ring: causal order
    (dict(window=8), 30),           # full ring with slack rows held
    (dict(window=8), 200),          # after compactions
    (dict(window=8, is_causal=True), 30),
    (dict(window=8, block_size=1, native_block_size=1), 30),
    (dict(window=8, layer_types=["full_attention"] * 2,
          context_window=6), 30),
])
def test_ring_read_in_place_matches_the_ring_copy(monkeypatch, kw, n_rows):
    """The draft over the ring's own buffer equals the draft over a copy of
    the ring, leaves the ring rows as they were, and a later commit and
    draft agree too: the block rows parked after the ring never count."""
    lm = _target()
    width = int(kw.get("block_size", BLOCK))
    block = mx.array([[3] + [7] * (width - 1)])
    mx.random.seed(5)
    caps = mx.random.normal((1, n_rows + BLOCK, _tcfg().hidden_size * len(CAPTURE)))
    got = {}
    for in_place in (False, True):
        monkeypatch.setattr(dd, "_RING_IN_PLACE", in_place)
        mx.random.seed(21)
        drafter = _drafter(lm, **kw)
        for i in range(0, n_rows, BLOCK):
            drafter.append_context(caps[:, i:min(i + BLOCK, n_rows)])
        mx.eval(drafter.draft_eval_state())
        before = [(c._idx, np.array(c.state[0]), np.array(c.state[1]))
                  for c in drafter._cache]
        first = _draft_hidden(drafter, block)
        again = _draft_hidden(drafter, block)
        for c, (idx, k, v) in zip(drafter._cache, before):
            assert c._idx == idx
            assert np.array_equal(np.array(c.state[0]), k)
            assert np.array_equal(np.array(c.state[1]), v)
        drafter.append_context(caps[:, n_rows:])
        got[in_place] = (first, again, _draft_hidden(drafter, block))
    for a, b in zip(got[True], got[False]):
        assert np.allclose(a, b, atol=5e-3), np.abs(a - b).max()
    assert np.array_equal(got[True][0], got[True][1])


@_NEEDS_GPU
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
def test_ring_read_in_place_at_the_served_attention_shape(monkeypatch, dtype):
    """Head dim 128 with 32 heads over 8 KV heads and a full 2047-row ring:
    the shape the attention kernels route by. The in-place view is strided
    and its mask is the causal string."""
    cfg = DFlashConfig(
        hidden_size=256, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=32, num_key_value_heads=8, head_dim=128,
        rms_norm_eps=1e-6, vocab_size=64, max_position_embeddings=8192,
        rope_theta=10000.0, block_size=8, mask_token_id=7,
        target_layer_ids=[0], num_target_layers=2,
        layer_types=["sliding_attention"], sliding_window=2048)
    drafter = DFlashDrafter(cfg)
    drafter.set_dtype(dtype)
    attn = drafter.layers[0].self_attn
    mx.random.seed(3)
    for lin in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
        lin.weight = (mx.random.normal(lin.weight.shape) * 0.05).astype(dtype)
    ctx = mx.random.normal((1, 2047 + 40, 256)).astype(dtype)
    block = mx.random.normal((1, 8, 256)).astype(dtype)
    out = {}
    for in_place in (False, True):
        monkeypatch.setattr(dd, "_RING_IN_PLACE", in_place)
        (cache,) = drafter.make_cache()
        attn.inject(ctx[:, :2047], drafter.rope, cache)
        for i in range(2047, 2087, 8):
            attn.inject(ctx[:, i:i + 8], drafter.rope, cache)
        assert cache._idx == 2087
        out[in_place] = np.array(
            attn.draft(block, drafter.rope, cache).astype(mx.float32))
    tol = 2e-2 if dtype == mx.bfloat16 else 2e-3
    scale = np.abs(out[False]).max()
    assert np.abs(out[True] - out[False]).max() <= tol * scale


def _conv(hidden, kernel_size, dtype, seed):
    mx.random.seed(seed)
    conv = dd.GroupedDynamicConv(hidden, kernel_size, 16)
    conv.base_kernel = mx.random.normal(conv.base_kernel.shape).astype(dtype)
    conv.kernel_projection.weight = (
        mx.random.normal(conv.kernel_projection.weight.shape) * 0.2).astype(dtype)
    return conv


@_NEEDS_GPU
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("rows, kernel_size", [(1, 2), (4, 2), (8, 2), (8, 3)])
def test_conv_kernel_matches_the_op_chain(monkeypatch, dtype, rows, kernel_size):
    """One dispatch a side gives the op chain's values: bit for bit at 16
    bits, where every product and add rounds as the chain's ops do."""
    if dd._conv_kernels() is None:
        pytest.skip("no Metal kernels on this device")
    conv = _conv(64, kernel_size, dtype, seed=rows)
    x = mx.random.normal((1, rows, 64)).astype(dtype)
    y = mx.random.normal((1, rows, 64)).astype(dtype)
    res = mx.random.normal((1, rows, 64)).astype(dtype)

    def run():
        a, proj = conv.prepare(x)
        return [np.array(t.astype(mx.float32)) for t in (
            a, conv.finish(y, proj), conv.finish(y, proj, res))]

    calls = []
    fused = dd.GroupedDynamicConv._fused

    def counted(self, *a):
        out = fused(self, *a)
        calls.append(out is not None)
        return out

    monkeypatch.setattr(dd.GroupedDynamicConv, "_fused", counted)
    got = run()
    assert calls == [True, True, True]
    monkeypatch.setattr(dd, "_conv_kerns", False)
    want = run()
    for a, b in zip(got, want):
        if dtype == mx.float32:
            assert np.allclose(a, b, rtol=1e-5, atol=1e-5)
        else:
            assert np.array_equal(a, b), np.abs(a - b).max()


@_NEEDS_GPU
def test_conv_kernel_leaves_a_mixed_dtype_to_the_op_chain():
    """A projection in another dtype than the rows promotes in the op chain,
    so the kernel declines it."""
    if dd._conv_kernels() is None:
        pytest.skip("no Metal kernels on this device")
    conv = _conv(64, 2, mx.float32, seed=0)
    x = mx.random.normal((1, 4, 64)).astype(mx.bfloat16)
    proj = conv.kernel_projection(x.astype(mx.float32))
    assert conv._fused(x, proj, 0, None) is None
    assert conv._fused(x, proj.astype(mx.bfloat16), 0, None) is not None
    assert conv.finish(x, proj, x).dtype == mx.float32


def _lattice(rows, k, dtype, seed):
    mx.random.seed(seed)
    cands = mx.random.permutation(5000)[: rows * k].reshape(rows, k).astype(mx.int32)
    first = (mx.random.normal((k,)) * 3).astype(dtype)
    edges = (mx.random.normal((rows - 1, k, k)) * 3).astype(dtype)
    return cands, first, edges


@_NEEDS_GPU
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("rows, k, floor", [(7, 16, 0.5), (7, 16, 0.0),
                                            (2, 4, 0.05), (15, 16, 0.01)])
def test_walk_kernel_takes_the_argmax_path(dtype, rows, k, floor):
    """The one-dispatch walk picks the argmax path's columns and tokens,
    gives each pick's log-probability and cuts at the same row."""
    cands, first, edges = _lattice(rows, k, dtype, seed=rows + k)
    walked = dd.selector_walk(cands, first, edges, floor)
    if walked is None:
        pytest.skip("no Metal kernels on this device")
    toks, sel, lp, keep, q = walked
    want = dd.greedy_path(first, edges)
    assert q is None
    assert sel.tolist() == want.tolist()
    assert toks.tolist() == greedy_walk(cands, first, edges).tolist()
    want_lp = dd.path_logprobs(first, edges, want)
    assert np.allclose(np.array(lp), np.array(want_lp), atol=1e-4)
    want_keep = rows if floor <= 0 else int(dd.rows_over_floor(want_lp, floor).item())
    assert int(keep.item()) == want_keep


@_NEEDS_GPU
@pytest.mark.parametrize("proposal", [(0.6, 20, 0.95, 0.0), (1.0, 5, 0.9, 0.0),
                                      (0.8, 0, 1.0, 0.1), (1.3, 3, 0.5, 0.02)])
def test_walk_kernel_draws_from_the_sharpened_proposal(proposal):
    """Each sampled pick comes with the proposal row it was drawn from: the
    _pq_probs row of the realized predecessor's lattice row. The picks
    follow that row's probabilities."""
    from gmlx.spec.speculative import _pq_probs

    rows, k = 7, 16
    cands, first, edges = _lattice(rows, k, mx.bfloat16, seed=4)
    if dd.selector_walk(cands, first, edges, 0.5, proposal) is None:
        pytest.skip("no Metal kernels on this device")
    mx.random.seed(8)
    counts = np.zeros(k)
    n = 1500
    for i in range(n):
        toks, sel, lp, keep, q = dd.selector_walk(cands, first, edges, 0.5, proposal)
        sel_l = sel.tolist()
        counts[sel_l[0]] += 1
        if i % 100:
            continue
        row = first
        for p in range(rows):
            want = np.array(_pq_probs(row[None], *proposal)[0])
            got = np.array(q[p])
            assert np.allclose(got, want, atol=2e-4), (p, np.abs(got - want).max())
            assert got[sel_l[p]] > 0
            assert toks[p].item() == cands[p, sel_l[p]].item()
            r = np.array(row.astype(mx.float32))
            assert abs(float(lp[p].item()) - (r[sel_l[p]] - np.logaddexp.reduce(r))) < 1e-4
            if p + 1 < rows:
                row = edges[p][sel_l[p]]
    want = np.array(_pq_probs(first[None], *proposal)[0])
    sigma = np.sqrt(want * (1 - want) / n)
    assert (np.abs(counts / n - want) <= 4.5 * sigma + 1e-9).all(), (counts / n, want)


@_NEEDS_GPU
@pytest.mark.parametrize("on, scored", [(True, True), (True, False), (False, False)])
def test_sampled_block_stashes_the_rows_it_drew_from(monkeypatch, on, scored):
    """A sampled DFlash 2 block with a q stash: one full-width proposal row
    per draft, each a distribution with mass on its drawn token, through
    the walk kernel (scoring its own lattice rows or not) and through the
    op chain alike."""
    monkeypatch.setattr(dd, "_WALK_KERNEL", on)
    monkeypatch.setattr(dd, "_LATTICE_WALK", scored)
    lm = _target()
    drafter = _drafter(lm, window=8, top_k=4)
    _inject(drafter, 12, seed=1)
    mx.random.seed(3)
    stash = DraftStash(q=[])
    toks = drafter.draft_block(3, None, None, BLOCK, None, greedy=False, stash=stash)
    toks = toks.tolist()[0]
    assert len(stash.q) == len(toks) == BLOCK - 1
    for t, q in zip(toks, stash.q):
        q = np.array(q)
        assert q.shape == (_tcfg().vocab_size,)
        assert abs(q.sum() - 1) < 1e-4 and q[t] > 0
        assert (q > 0).sum() <= 4


@_NEEDS_GPU
def test_a_block_the_logs_record_is_not_cut():
    """The walk that feeds the p/q and top-2 logs has no test of a cut
    draft, so a block it records keeps every draft: a cut on the draft's
    own pick would bias the token at the cut."""
    lm = _target()
    drafter = _drafter(lm, window=8, top_k=4)
    _inject(drafter, 12, seed=1)
    object.__setattr__(drafter, "row_floor", 0.999)
    mx.random.seed(3)
    plain = drafter.draft_block(3, None, None, BLOCK, None, greedy=False,
                                stash=DraftStash(q=[]))
    assert plain.shape[1] < BLOCK - 1
    for logged in (DraftStash(q=[], pq=[]), DraftStash(q=[], top2=[])):
        mx.random.seed(3)
        toks = drafter.draft_block(3, None, None, BLOCK, None, greedy=False,
                                   stash=logged)
        assert toks.shape[1] == BLOCK - 1


def _walk_rows(dtype, rows=7, seed=11):
    mx.random.seed(seed)
    drafter = _drafter(top_k=8)
    H, V = _tcfg().hidden_size, _tcfg().vocab_size
    hidden = mx.random.normal((rows, H)).astype(dtype)
    logits = (mx.random.normal((rows, V)) * 4).astype(dtype)
    return drafter.candidate_selector, hidden, logits, mx.array(5)


def _path_rows(first, edges, sel):
    rows = [first] + [edges[p - 1][int(sel[p - 1].item())] for p in range(1, len(sel))]
    return [np.array(r.astype(mx.float32)) for r in rows]


@_NEEDS_GPU
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("floor", [0.5, 0.0])
def test_lattice_walk_takes_the_argmax_path_of_the_lattice(dtype, floor):
    """The walk that scores its own lattice rows takes the argmax path of
    the lattice the two-step form builds, with its pick log-probabilities
    and row cut."""
    sel, hidden, logits, anchor = _walk_rows(dtype)
    fused = sel.walk(hidden, logits, anchor, None, floor)
    if fused is None:
        pytest.skip("no Metal kernels on this device")
    cands, toks, cols, lp, keep, q = fused
    ref_c, first, edges = sel.lattice(hidden, logits, anchor)
    assert q is None and cands.tolist() == ref_c.tolist()
    assert toks.tolist() == np.take_along_axis(
        np.array(cands), np.array(cols)[:, None], axis=-1)[:, 0].tolist()
    # f32 lattice GEMMs run at TF32 where available, so a pick may differ
    # from the op chain's only within that precision.
    tol = 2e-3 if dtype == mx.float32 else 0.0
    for p, r in enumerate(_path_rows(first, edges, cols)):
        assert r[int(cols[p].item())] >= r.max() - tol
    assert np.allclose(np.array(lp), np.array(dd.path_logprobs(first, edges, cols)),
                       atol=tol + 1e-4)
    want_keep = 7 if floor <= 0 else int(dd.rows_over_floor(lp, floor).item())
    assert int(keep.item()) == want_keep
    if dtype == mx.bfloat16:
        assert cols.tolist() == dd.greedy_path(first, edges).tolist()


@_NEEDS_GPU
def test_lattice_walk_draws_from_the_rows_it_scores():
    """Sampled, each pick comes with the sharpened proposal of the lattice
    row of its realized predecessor, and the first picks follow it."""
    from gmlx.spec.speculative import _pq_probs

    proposal = (0.6, 20, 0.95, 0.0)
    sel, hidden, logits, anchor = _walk_rows(mx.bfloat16)
    if sel.walk(hidden, logits, anchor, None, 0.0, proposal) is None:
        pytest.skip("no Metal kernels on this device")
    _, first, edges = sel.lattice(hidden, logits, anchor)
    mx.random.seed(8)
    n, k = 1500, sel.top_k
    counts = np.zeros(k)
    for i in range(n):
        cands, toks, cols, lp, keep, q = sel.walk(hidden, logits, anchor, None, 0.0, proposal)
        counts[int(cols[0].item())] += 1
        if i % 100:
            continue
        for p, r in enumerate(_path_rows(first, edges, cols)):
            want = np.array(_pq_probs(mx.array(r)[None], *proposal)[0])
            assert np.allclose(np.array(q[p]), want, atol=2e-4), p
            assert np.array(q[p])[int(cols[p].item())] > 0
    want = np.array(_pq_probs(first[None], *proposal)[0])
    sigma = np.sqrt(want * (1 - want) / n)
    assert (np.abs(counts / n - want) <= 4.5 * sigma + 1e-9).all(), (counts / n, want)


@_NEEDS_GPU
def test_greedy_block_and_estimate_match_with_the_lattice_walk(monkeypatch):
    """A greedy block drafts the same tokens, cut and round estimate whether
    the walk scores its own lattice rows or the whole lattice is built."""
    lm = _target()
    drafter = _drafter(lm, window=8, top_k=4)
    _inject(drafter, 12, seed=1)
    object.__setattr__(drafter, "row_floor", 0.05)
    got = {}
    for scored in (True, False):
        monkeypatch.setattr(dd, "_LATTICE_WALK", scored)
        toks = drafter.draft_block(3, None, None, BLOCK, None, greedy=True)
        got[scored] = (toks.tolist(), float(drafter.round_estimate.item()),
                       drafter._round_path[0] is None)
    assert got[True][0] == got[False][0]
    assert abs(got[True][1] - got[False][1]) < 1e-4
    assert got[False][2] is False
    if dd._lattice_walk_kernels() is not None:
        assert got[True][2] is True


def _heads(out_dims, in_dims):
    """One head of every form head_rows slices, with the float weight."""
    mx.random.seed(17)
    w = mx.random.normal((out_dims, in_dims)) * 0.1
    lin = nn.Linear(in_dims, out_dims, bias=False)
    lin.weight = w
    emb = nn.Embedding(out_dims, in_dims)
    emb.weight = w
    forms = {"linear": lin, "tied": emb.as_linear,
             "affine": nn.QuantizedLinear.from_linear(lin, group_size=32, bits=4)}
    try:
        import mlx_kquant as kq
        from mlx_kquant.nn import KQuantEmbedding, KQuantLinear
    except ImportError:
        return forms
    if dd.mx.default_device().type != mx.DeviceType.gpu:
        return forms
    for codec in ("q8_0", "q6_k", "q4_k"):
        wire, scales = kq.quantize(w, codec)
        ql = KQuantLinear(in_dims, out_dims, False, codec)
        ql.weight, ql.scales = wire, scales
        forms[codec] = ql
    qe = KQuantEmbedding(out_dims, in_dims, "q6_k")
    qe.weight, qe.scales = forms["q6_k"].weight, forms["q6_k"].scales
    forms["tied q6_k"] = qe.as_linear
    return forms


@pytest.mark.parametrize("seed", [0, 2])
@pytest.mark.parametrize("lo, hi", [(256, 768), (0, 512), (512, 1024)])
def test_head_rows_are_the_head_without_the_skipped_ids(lo, hi, seed):
    """Every head form gives the whole head's columns outside the range,
    in id order. A K-quant kernel may take another route for the fewer
    rows, so a column can differ in its last bfloat16 bit."""
    mx.random.seed(seed)
    x = mx.random.normal((1, 3, 256))
    for name, head in _heads(1024, 256).items():
        got = dd.head_rows(head, x, lo, hi)
        assert got is not None, name
        full = np.array(head(x).astype(mx.float32))
        keep = np.r_[0:lo, hi:1024]
        assert got.shape == (1, 3, keep.size), name
        assert np.allclose(np.array(got.astype(mx.float32)), full[..., keep],
                           rtol=2 ** -7, atol=1e-3), name


def test_head_rows_leaves_unknown_heads_whole():
    assert dd.head_rows(lambda x: x, mx.zeros((1, 2, 8)), 0, 4) is None

    class Gated(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = mx.zeros((16, 8))
            self.gate = mx.ones((16,))

        def __call__(self, x):
            return (x @ self.weight.T) * self.gate

    assert dd.head_rows(Gated(), mx.zeros((1, 2, 8)), 0, 4) is None


def test_longest_free_run_skips_the_unused_ids():
    occ = np.zeros(64, dtype=bool)
    occ[[0, 1, 2, 40, 63]] = True
    assert dd._longest_free_run(occ, 64 * 128) == (3 * 128, 40 * 128)
    occ[20] = True                      # the run splits: 41..62 is longest
    assert dd._longest_free_run(occ, 64 * 128) == (41 * 128, 63 * 128)
    occ[::4] = True                     # no run of an eighth of the ids
    assert dd._longest_free_run(occ, 64 * 128) is None


def _subset_drafter(lm, vocab_ids):
    drafter = _drafter(lm, window=8)
    _inject(drafter, 12, seed=1)
    dd._open_ring(drafter._cache, int(drafter.config.vocab_size), vocab_ids)
    return drafter


def test_draft_head_skips_ids_the_context_has_not_used(monkeypatch):
    """A context that has used only low ids drafts over a head without the
    unused tail, maps the candidates back to ids, and widens the head
    again once a commit uses an id inside the skipped run."""
    lm = _target()
    vocab = int(_tcfg().vocab_size)
    monkeypatch.setattr(dd, "_HEAD_BLOCK", 8)
    calls = []
    real = dd.head_rows

    def spy(head, x, lo, hi):
        calls.append((lo, hi))
        return real(head, x, lo, hi)

    monkeypatch.setattr(dd, "head_rows", spy)
    drafter = _subset_drafter(lm, list(range(0, 16)))
    skip = drafter._head_skip()
    assert skip == (16, vocab)
    sel, seen = drafter.candidate_selector, []
    real_walk, real_lattice = sel.walk, sel.lattice

    def walk(*a):
        out = real_walk(*a)
        if out is not None:
            seen.append(out[0])
        return out

    def lattice(*a):
        out = real_lattice(*a)
        seen.append(out[0])
        return out

    monkeypatch.setattr(sel, "walk", walk)
    monkeypatch.setattr(sel, "lattice", lattice)
    mx.random.seed(3)
    toks = drafter.draft_block(3, None, None, BLOCK, None, greedy=True)
    assert calls == [skip]
    cands = np.array(seen[-1])
    assert (cands < 16).all()
    # The same block over the whole head picks the best ids below 16.
    full = drafter._logits(drafter._draft_hidden(drafter._block_tokens(3, BLOCK, mx.int32))[:, 1:])[0]
    full = np.array(full.astype(mx.float32))
    for r in range(BLOCK - 1):
        assert set(cands[r].tolist()) == set(np.argsort(-full[r, :16])[:TOP_K].tolist())
    assert all(t < 16 for t in toks.tolist()[0])
    drafter.accept_verified_tokens(
        mx.zeros((1, 2, _packed_width(_tcfg()))), toks, 1, [40, 5], None)
    assert drafter._head_skip() == (48, vocab)


def test_draft_head_is_whole_without_prompt_ids(monkeypatch):
    lm = _target()
    drafter = _drafter(lm, window=8)
    _inject(drafter, 12, seed=1)
    assert drafter._head_skip() is None
    monkeypatch.setattr(dd, "_HEAD_SUBSET", False)
    assert _subset_drafter(lm, [0])._head_skip() is None


def test_lattice_candidates_over_a_wide_vocabulary_are_the_top_k():
    """From 8192 ids up the lattice takes its candidates from the
    hierarchical top-k: the same ids as a full sort."""
    from gmlx.spec.dflash_drafter import CandidateSelector

    mx.random.seed(2)
    sel = CandidateSelector(32, 9000, rank=8, top_k=16)
    hidden = mx.random.normal((5, 32))
    logits = mx.random.normal((5, 9000))
    cands, first, edges = sel.lattice(hidden, logits, mx.array(3))
    want = np.argsort(-np.array(logits), axis=-1)[:, :16]
    assert [sorted(r) for r in np.array(cands).tolist()] == [
        sorted(r) for r in want.tolist()]
    assert first.shape == (16,) and edges.shape == (4, 16, 16)
    unary = np.take_along_axis(np.array(logits), np.array(cands), axis=-1)
    hp = np.array(sel.hidden_projection(hidden))
    succ = np.array(sel.successor_codebook(cands))
    pred0 = np.array(sel.predecessor_codebook(mx.array([3])))[0]
    assert np.allclose(np.array(first), unary[0] + succ[0] @ (pred0 * hp[0]),
                       atol=1e-4)


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


# --- NoPE layers and the context window ---------------------------------------

def _caps(n, seed):
    mx.random.seed(seed)
    return mx.random.normal((1, n, _tcfg().hidden_size * len(CAPTURE)))


def test_rope_layers_needs_one_entry_per_layer():
    with pytest.raises(ValueError, match="entries for"):
        _config(_tcfg(), rope_layers=[True])


@pytest.mark.parametrize("rope_layers,invariant", [
    ([False, False], True), ([True, False], False)])
def test_a_nope_layer_ignores_context_order(rope_layers, invariant):
    """Unmasked non-causal attention without RoPE is a set function of the
    context rows, so reversing them leaves the draft unchanged. A roped
    layer sees the order."""
    lm = _target()
    block = mx.array([[3, 7, 7, 7]])
    caps = _caps(10, seed=3)
    drafter = _drafter(lm, layer_types=["full_attention"] * 2,
                       rope_layers=rope_layers)
    drafter.append_context(caps)
    want = _draft_hidden(drafter, block)
    drafter.reset(lm)
    drafter.append_context(caps[:, ::-1])
    got = _draft_hidden(drafter, block)
    assert np.allclose(want, got, atol=5e-3) == invariant


@pytest.mark.parametrize("n_rows", [30, 200])
def test_context_window_bounds_the_full_layer(n_rows):
    """A full layer with context window 8 sees the last 8 committed rows
    however many went in, through ring compactions."""
    lm = _target()
    block = mx.array([[3, 7, 7, 7]])
    drafter = _drafter(lm, layer_types=["full_attention"] * 2,
                       context_window=8)
    ring = drafter._cache[0]
    assert isinstance(ring, BufferedRotatingKVCache) and ring.max_size == 9
    assert drafter.hidden_capture_limit == 511
    caps = _caps(n_rows, seed=4)
    for i in range(0, n_rows, BLOCK):
        drafter.append_context(caps[:, i:i + BLOCK])
    assert ring.offset == n_rows
    want = _draft_hidden(drafter, block)

    drafter.reset(lm)
    drafter.append_context(caps[:, -8:])
    assert np.allclose(want, _draft_hidden(drafter, block), atol=5e-3)
    drafter.reset(lm)
    drafter.append_context(caps[:, -7:])
    assert not np.allclose(want, _draft_hidden(drafter, block), atol=1e-4)


def test_a_long_seed_projects_only_each_layers_span():
    lm = _target()
    drafter = _drafter(lm, window=8,
                       layer_types=["sliding_attention", "full_attention"],
                       context_window=20)
    assert drafter.hidden_capture_limit == 20
    drafter.append_context(_caps(20, seed=6))
    held = [int(c.state[0].shape[2]) for c in drafter._cache]
    assert held == [7, 20]


@pytest.mark.parametrize("rows", [1, 3, 12])
def test_one_call_context_kv_writes_the_ring_each_layer_would(monkeypatch, rows):
    """A commit that projects every layer's context K/V in one call writes
    the ring each layer projecting its own writes, and a commit longer than
    a ring leaves the projection to the layer."""
    lm = _target()
    rings = {}
    for multi in (True, False):
        monkeypatch.setattr(dd, "_QMM_MULTI", multi)
        mx.random.seed(4)
        drafter = _drafter(lm, window=8)
        drafter.append_context(_caps(rows, seed=6))
        rings[multi] = [np.array(x) for c in drafter._cache for x in c.state]
    assert len(rings[True]) == len(rings[False])
    for a, b in zip(rings[True], rings[False]):
        assert np.array_equal(a, b)


def test_a_commit_started_early_drafts_the_same_block(monkeypatch):
    lm = _target()
    mx.random.seed(2)
    verify_hidden = mx.random.normal((1, 3, _packed_width(_tcfg())))
    out = {}
    for early in (True, False):
        monkeypatch.setattr(dd, "_COMMIT_ASYNC", early)
        mx.random.seed(4)
        drafter = _drafter(lm, window=8)
        _inject(drafter, 12, seed=1)
        drafter.accept_verified_tokens(verify_hidden, None, 1, [40, 5], None)
        out[early] = drafter.draft_block(3, None, None, BLOCK, None, greedy=True).tolist()
    assert out[True] == out[False]


def test_the_kept_hidden_is_the_hidden_of_the_rows_the_ring_holds(monkeypatch):
    """Under GMLX_SPEC_APC_LONGER a request leaves the target hidden of its
    last rows: the prompt capture and each accept's committed rows, less
    the rows a counted accept or an undo took back, cut at the stored
    length and at the capture limit."""
    monkeypatch.setattr(dd, "_KEEP_TAIL", True)
    lm = _target()
    width = _packed_width(_tcfg())
    mx.random.seed(3)
    prompt = mx.random.normal((1, 12, width))
    rounds = [mx.random.normal((1, BLOCK, width)) for _ in range(4)]
    drafter = _drafter(lm, window=8)
    limit = drafter.hidden_capture_limit
    assert limit == 7
    drafter.prefill_from_target_hidden(None, prompt, 0, None)
    tail = drafter._tail
    # A whole accept, a counted accept trimmed by two rows, an accept taken
    # back whole, then one of two rows.
    drafter.accept_verified_tokens(rounds[0], None, 3, [], None)
    mark = drafter.accept_mark(4)
    drafter.accept_verified_tokens(rounds[1], None, 3, [], None)
    drafter.accept_trim(mark, 2)
    mark = drafter.accept_mark(3)
    drafter.accept_verified_tokens(rounds[2], None, 2, [], None)
    drafter.accept_undo(mark)
    drafter.accept_verified_tokens(rounds[3], None, 1, [], None)
    want = mx.concatenate(
        [prompt, rounds[0], rounds[1][:, :2], rounds[3][:, :2]], axis=1)
    end = 12 + 8
    assert tail.round_rows == 8
    got = tail.take(12, end)
    assert mx.array_equal(got, want[:, -limit:]).item()
    # A record stored three tokens short of the request's end: the rows
    # that end there, as many as the log still holds.
    short = tail.take(12, end - 3)
    n = int(short.shape[1])
    assert 0 < n <= limit
    assert mx.array_equal(short, want[:, -3 - n:-3]).item()
    # A cache that holds a row the drafter was never fed: nothing to keep.
    assert tail.take(12, end + 1) is None
    # The next request's log is its own.
    drafter.reset(lm)
    assert drafter._tail is not tail and tail.round_rows == 8

    monkeypatch.setattr(dd, "_KEEP_TAIL", False)
    assert _drafter(lm, window=8)._tail is None


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
    from gmlx.spec.speculative import _STOCH_DRAFT, _pq_probs

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
    from gmlx._vendor.mlx_lm.sample_utils import make_sampler

    from gmlx.spec.speculative import annotate_sampling_params

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
    from gmlx.spec.speculative import _pq_probs

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
    from gmlx.spec.speculative import _PQ_SWEEP, _pq_graph

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
    import gmlx.spec.speculative as sp

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
    import gmlx.spec.speculative as sp

    for name in ("_PQ_LOG", "_TOP2_LOG", "_STOCH_ACCEPT"):
        monkeypatch.setattr(sp, name, False)
    lm = _target()
    drafter = _drafter(lm)
    seen = _recording(drafter, monkeypatch)
    mx.random.seed(SEED)
    _engine_walk(lm, drafter, mx.array([[1, 2, 3, 4, 5]]), 8,
                 sampler=_annotated() if sampled else None)
    assert seen and all(s is None for s in seen)


@pytest.mark.parametrize("explicit", [True, False])
def test_v1_rows_keep_exact_match_under_stochastic(monkeypatch, caplog,
                                                   explicit):
    """Independent block rows are not a conditional proposal; the engine
    keeps exact-match for such drafters (measured on Muse v1: 6.7 -> 1.7
    accepted per round when forced). It warns only when stochastic
    acceptance was asked for, not under the default."""
    import gmlx.spec.speculative as sp
    from gmlx.models.muse_glimmer.dflash import MuseGlimmerDFlashDrafter

    assert MuseGlimmerDFlashDrafter.stochastic_draft is False
    assert DFlash2Drafter.stochastic_draft is True
    for name in ("_PQ_LOG", "_TOP2_LOG"):
        monkeypatch.setattr(sp, name, False)
    monkeypatch.setattr(sp, "_STOCH_ACCEPT", True)
    monkeypatch.setattr(sp, "_STOCH_EXPLICIT", explicit)
    lm = _target()
    drafter = _drafter(lm)
    monkeypatch.setattr(type(drafter), "stochastic_draft", False)
    seen = _recording(drafter, monkeypatch)
    mx.random.seed(SEED)
    with caplog.at_level("WARNING"):
        _engine_walk(lm, drafter, mx.array([[1, 2, 3, 4, 5]]), 8, sampler=_annotated())
    assert seen and all(s is None for s in seen)
    assert ("independent rows" in caplog.text) is explicit
    assert "opaque" not in caplog.text


@pytest.mark.parametrize("switch", ["_PQ_LOG", "_TOP2_LOG", "_STOCH_ACCEPT"])
def test_instrumented_rounds_hand_the_block_drafter_a_stash(monkeypatch, switch):
    """One stash per round with one entry per drafted row, and the round loop
    walks it without a head-shaped wrapper."""
    import gmlx.spec.speculative as sp

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


def test_the_capture_prices_only_the_rows_the_drafter_keeps():
    # Packed hidden (1 + taps), its tap slice (taps) and the fc output (1),
    # 2 bytes each, over the prompt rows up to the capture limit.
    cfg = _tcfg()
    row = cfg.hidden_size * (2 * len(CAPTURE) + 2) * 2
    windowed = DFlash2Drafter(_config(cfg, window=512))
    assert windowed.capture_held_bytes(100) == 100 * row
    assert windowed.capture_held_bytes(10_000) == 511 * row
    full = ["sliding_attention", "full_attention"]
    wide = DFlash2Drafter(_config(cfg, window=512, layer_types=full,
                                  context_window=2000))
    assert wide.capture_held_bytes(10_000) == 2000 * row
    unbounded = DFlash2Drafter(_config(
        cfg, window=None, layer_types=["full_attention"] * 2))
    assert unbounded.capture_held_bytes(10_000) == 0


def test_the_target_prices_the_capture_while_the_drafter_lives():
    import gc
    from types import SimpleNamespace

    from gmlx.gen import prefill_decay as pd

    drafter = DFlash2Drafter(_config(_tcfg(), window=512))
    target = SimpleNamespace()
    pd.note_drafter_held(target, drafter)
    assert pd.drafter_held_bytes(target, 100) == drafter.capture_held_bytes(100)
    wrapped = SimpleNamespace(language_model=target)
    assert pd.drafter_held_bytes(wrapped, 100) > 0
    del drafter
    gc.collect()
    assert pd.drafter_held_bytes(target, 100) == 0
    assert pd.drafter_held_bytes(SimpleNamespace(), 100) == 0
