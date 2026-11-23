"""Early draft: a round commits and submits the next draft before it yields
its tokens. The tokens are the ones the plain order emits, and a consumer
that stops inside a round leaves the target cache and the drafter ring as
the plain order leaves them."""

import mlx.core as mx
import numpy as np
import pytest

import gmlx.spec.speculative as sp

from test_dflash2_drafter import (
    BLOCK,
    CAPTURE,
    N_GEN,
    SEED,
    _annotated,
    _drafter,
    _engine_reference,
    _target,
)

PROMPT = [[1, 2, 3, 4, 5]]


@pytest.fixture(autouse=True)
def _plain_rounds(monkeypatch):
    monkeypatch.setattr(sp, "_ROUND_GATE", 0.0)
    monkeypatch.setattr(sp, "_ROUND_GATE_AUTO", False)
    monkeypatch.setattr(sp, "_ROUND_PROFILE", False)
    monkeypatch.setattr(sp, "_DEFER_CUT", True)
    monkeypatch.setattr(sp, "_ONE_STREAM", True)
    monkeypatch.setattr(sp, "_STOCH_ACCEPT", True)
    for name in ("_PQ_LOG", "_TOP2_LOG"):
        monkeypatch.setattr(sp, name, False)


def _arrays(v, out):
    if isinstance(v, mx.array):
        out.append(np.array(v.astype(mx.float32)))
    elif isinstance(v, (list, tuple)):
        for x in v:
            _arrays(x, out)
    elif isinstance(v, dict):
        for k in sorted(v):
            _arrays(v[k], out)
    elif v is not None and not isinstance(v, (int, float, str, bool)):
        raise TypeError(type(v).__name__)
    return out


@pytest.fixture(autouse=True)
def _plain_round_levers(monkeypatch):
    """The round levers start in their off form: each test turns on the
    ones it compares against the plain order."""
    import gmlx.spec.dflash_drafter as dd

    monkeypatch.setattr(sp, "_ACCEPT_COUNTED", False)
    monkeypatch.setattr(sp, "_VERIFY_AHEAD", 0)
    monkeypatch.setattr(dd, "_COUNTED_SHORT", False)


def _run(monkeypatch, early, *, stop_after=None, sampler=None, n=N_GEN,
         floor=0.0):
    """Tokens, target cache arrays, ring arrays and ring counters of one
    walk; ``stop_after`` closes the generator after that many tokens."""
    monkeypatch.setattr(sp, "_DRAFT_EARLY", early)
    lm = _target()
    monkeypatch.setattr(type(lm), "speculative_rollback_repeatable", True,
                        raising=False)
    drafter = _drafter(lm)
    object.__setattr__(drafter, "row_floor", floor)
    drafts = []
    orig = drafter.draft_block

    def draft_block(*a, **kw):
        drafts.append(len(toks))
        return orig(*a, **kw)

    monkeypatch.setattr(drafter, "draft_block", draft_block)
    prompt = mx.array(PROMPT)
    mx.random.seed(SEED)
    lm.set_dflash_capture(CAPTURE)
    cache = lm.make_cache()
    out = lm(prompt, cache=cache, return_hidden=True, return_shared_kv=True)
    first = out.logits[:, -1, :]
    b = int((mx.argmax(first, axis=-1) if sampler is None
             else sampler(first)).item())
    sp._buffer_mtp_target_cache(cache, drafter, BLOCK)
    toks = [b]
    gen = sp._owned_decode_rounds(
        lm, drafter, lm, cache, hidden=out.hidden_states[-1], b=b,
        shared_kv=out.shared_kv_states, seed_tokens=prompt, emitted=1,
        max_tokens=n, sampler=sampler, draft_block_size=BLOCK)
    kept = None
    for t in gen:
        toks.append(t)
        if stop_after is not None and len(toks) - 1 >= stop_after:
            before = [np.array(x) for x in mx.random.state]
            gen.close()
            kept = all(np.array_equal(x, np.array(y))
                       for x, y in zip(before, mx.random.state))
            break
    ring = drafter._cache
    return {
        "toks": toks,
        "drafts": drafts,
        "target": _arrays([c.state for c in cache], []),
        "ring": _arrays([c.state for c in ring], []),
        "counts": [(int(c.offset), int(c._idx)) for c in ring],
        "occ": None if ring.occ is None else ring.occ.copy(),
        "rng": [np.array(x) for x in mx.random.state],
        "kept": kept,
        "pending": drafter.row_cut_pending,
    }


def _rng_keys(a, b):
    """The RNG key compared between two runs. A stop leaves it where it
    was: a draft drawn ahead keeps its draws, since another request may
    have drawn after it."""
    if a["kept"] is None and b["kept"] is None:
        return ("rng",)
    assert a["kept"] and b["kept"]
    return ()


def _same(a, b):
    assert a["toks"] == b["toks"]
    assert a["counts"] == b["counts"]
    for key in ("target", "ring") + _rng_keys(a, b):
        assert len(a[key]) == len(b[key])
        for x, y in zip(a[key], b[key]):
            assert x.shape == y.shape and np.array_equal(x, y)
    assert (a["occ"] is None) == (b["occ"] is None)
    if a["occ"] is not None:
        assert np.array_equal(a["occ"], b["occ"])
    assert not a["pending"] and not b["pending"]


def test_the_early_order_emits_the_greedy_chain_and_drafts_ahead(monkeypatch):
    lm = _target()
    ref = _engine_reference(lm, mx.array(PROMPT), N_GEN)
    plain = _run(monkeypatch, False, floor=0.5)
    early = _run(monkeypatch, True, floor=0.5)
    assert early["toks"][:N_GEN] == ref == plain["toks"][:N_GEN]
    # Each draft after the first was submitted before the tokens of the
    # round ahead of it went out: fewer tokens were out when it started.
    assert len(early["drafts"]) == len(plain["drafts"])
    assert early["drafts"][0] == plain["drafts"][0]
    assert all(e < p for e, p in zip(early["drafts"][1:], plain["drafts"][1:]))


def test_sampled_rounds_draw_the_same_tokens_early(monkeypatch):
    plain = _run(monkeypatch, False, sampler=_annotated(), floor=0.5)
    early = _run(monkeypatch, True, sampler=_annotated(), floor=0.5)
    assert early["drafts"] != plain["drafts"]
    _same(plain, early)


@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("stop_after", range(1, 10))
def test_a_stop_inside_a_round_leaves_the_plain_order_state(
        monkeypatch, sampled, stop_after):
    kw = {"sampler": _annotated()} if sampled else {}
    plain = _run(monkeypatch, False, stop_after=stop_after, n=64, floor=0.5,
                 **kw)
    kw = {"sampler": _annotated()} if sampled else {}
    early = _run(monkeypatch, True, stop_after=stop_after, n=64, floor=0.5,
                 **kw)
    _same(plain, early)


def test_a_ring_that_would_compact_is_not_marked(monkeypatch):
    lm = _target()
    drafter = _drafter(lm)
    monkeypatch.setattr(sp, "_DRAFT_EARLY", False)
    assert drafter.accept_mark(2) is None        # no ring rows yet
    c = drafter._cache[0]
    keys = mx.zeros((1, 2, 3, 8))
    c.update_and_fetch(keys, keys)
    for other in drafter._cache[1:]:
        other.update_and_fetch(keys, keys)
    mark = drafter.accept_mark(2)
    assert mark is not None
    before = [(int(x.offset), int(x._idx)) for x in drafter._cache]
    for x in drafter._cache:
        x.update_and_fetch(keys[:, :, :2], keys[:, :, :2])
    drafter.accept_undo(mark)
    assert [(int(x.offset), int(x._idx)) for x in drafter._cache] == before
    # One spare row left: two more would grow or compact the ring.
    c._idx = int(c.keys.shape[2]) - 1
    assert drafter.accept_mark(2) is None
