"""Bound replay of a verify (GMLX_SPEC_VERIFY_BOUND): the session's rules.

The tiny model's records run their primitives (its hyper-connection norm
and float MoE are not on kq's direct list), so kq refuses to bind them. The
tests that need records to run outside an eval put a stand-in behind the
session that runs a record at once, the way a bound replay does once its
command buffer has completed: the in-place writes of the whole-layer record
land when the record is called, not when the caller evaluates.
"""

import copy

import mlx.core as mx
import mlx_kquant as kq
import numpy as np
import pytest

import gmlx.models.qwen4_exp.model as q4
from gmlx.models.qwen4_exp import verify_bound as vb
from gmlx.spec import verify_submit

from helpers import _real_apple_gpu
from test_qwen4exp_qsa_whole import PASS, PROMPT, _equal, _lm

pytestmark = pytest.mark.skipif(
    not (hasattr(kq, "bound_call") and _real_apple_gpu()),
    reason="needs a kq build with bound sessions and a real Apple GPU")


@pytest.fixture(autouse=True)
def _levers(monkeypatch):
    monkeypatch.setattr(q4, "_VERIFY_BLOCKED", False)
    monkeypatch.setattr(q4, "_VERIFY_FUSED", True)
    monkeypatch.setattr(q4, "_VERIFY_FUSED_QSA", True)
    monkeypatch.setattr(q4, "_VERIFY_FUSED_CAP", True)
    monkeypatch.setattr(q4, "_FUSED_WHOLE", True)
    monkeypatch.setattr(q4, "_FUSED_LIMITS", (4096, 256 << 20))
    monkeypatch.setenv("GMLX_Q4_KV_TAIL", "64")
    monkeypatch.setattr(vb, "_MODE", 0)
    monkeypatch.setattr(vb, "_kq_mod", [None])
    for k in vb.stats:
        monkeypatch.setitem(vb.stats, k, {} if isinstance(
            vb.stats[k], dict) else 0)
    kq.env_cache(True)
    yield
    kq.env_cache(False)


class _AtOnce:
    """Stands in for the kq module behind a session: a record runs when
    it is called. ``lazy_ok`` False refuses a lazy input the way kq does
    (an array is taken for lazy until this stand-in or a submit of the
    session has seen it evaluated)."""

    def __init__(self):
        self.calls = 0
        self.commits = 0
        self.copies = 0
        self.open = 0
        self.max_open = 0
        self.refuse = set()
        # What kq says of a refused record: still learning, or never.
        self.state = "refused"

    def fused_direct_info(self, handle):
        return {"state": self.state}

    def bound_session(self):
        return 0

    def bound_call(self, sid, handle, ins):
        if handle in self.refuse:
            return 4
        outs = kq.fused_call(handle, ins)
        mx.eval(outs)
        self.calls += 1
        self.open += 1
        self.max_open = max(self.max_open, self.open)
        return outs

    def bound_copy(self, sid, arrays):
        outs = [mx.concatenate([a[:, :1], a[:, 1:]], axis=1) for a in arrays]
        mx.eval(outs)
        self.copies += len(outs)
        self.open += 1
        return outs

    def bound_commit(self, sid):
        self.commits += 1
        self.open = 0
        return True

    fused_call = staticmethod(kq.fused_call)


def _rounds(lm, cache, plan, out, now=True):
    args = lm.args
    for S, kept in plan:
        block = mx.random.randint(0, args.vocab_size, (1, S))
        if now:
            with verify_submit.now():
                hid, _, sink = lm.speculative_verify_hidden(block, cache)
        else:
            hid, _, sink = lm.speculative_verify_hidden(block, cache)
        lm.rollback_speculative_cache(cache, sink, kept - 1, S)
        mx.eval(hid)
        out.append(np.array(hid.astype(mx.float32)))


def _prefilled(seed=5):
    lm, args = _lm(mx.bfloat16)
    mx.random.seed(seed)
    cache = lm.make_cache()
    mx.eval(lm(mx.random.randint(0, args.vocab_size, (1, PROMPT)),
               cache=cache).logits)
    return lm, cache


def _run(monkeypatch, mode, shim=None, plan=None, now=True):
    monkeypatch.setattr(vb, "_MODE", mode)
    monkeypatch.setattr(vb, "_kq_mod", [shim])
    lm, cache = _prefilled()
    out: list = []
    _rounds(lm, cache, plan or PASS * 4, out, now=now)
    out.extend(np.array(a.astype(mx.float32))
               for c in cache for a in c.state if isinstance(a, mx.array))
    return out


def test_records_run_at_once_leave_the_rounds_as_they_were(monkeypatch):
    # Every record of the verify from the first bound layer on runs when
    # it is called; hidden and caches equal the lazy path's, rollbacks in.
    ref = _run(monkeypatch, 0)
    shim = _AtOnce()
    got = _run(monkeypatch, 1, shim)
    _equal(ref, got)
    assert shim.calls > 0 and vb.stats["whole"] > 0
    # One buffer of ours is never left open at the end of a forward.
    assert shim.open == 0
    assert vb.stats["bound_calls"] == shim.calls


def test_the_ple_block_as_a_record_leaves_the_rounds_as_they_were(monkeypatch):
    # The whole form runs the PLE block from its table rows on as one
    # record, in the lazy path and with records run at once.
    monkeypatch.setattr(q4, "_FUSED_WHOLE", False)
    ref = _run(monkeypatch, 0)
    monkeypatch.setattr(q4, "_FUSED_WHOLE", True)
    real, hits = q4._fused_ple_verify, [0]

    def spy(*a):
        r = real(*a)
        hits[0] += r is not None
        return r

    monkeypatch.setattr(q4, "_fused_ple_verify", spy)
    _equal(ref, _run(monkeypatch, 0))
    lazy_hits = hits[0]
    assert lazy_hits > 0
    shim = _AtOnce()
    _equal(ref, _run(monkeypatch, 1, shim))
    assert hits[0] > lazy_hits and shim.calls > 0


def test_the_lever_needs_the_callers_word(monkeypatch):
    # A verify its caller may drop unevaluated (built ahead, gated) runs
    # nothing early.
    shim = _AtOnce()
    ref = _run(monkeypatch, 0)
    got = _run(monkeypatch, 1, shim, now=False)
    _equal(ref, got)
    assert shim.calls == 0 and vb.stats["verifies"] == 0


def test_a_record_kq_cannot_bind_leaves_the_rest_of_the_verify_lazy(monkeypatch):
    # The real kq: the tiny model's records are not direct or have no
    # program, so the first bound call is refused and the verify is the
    # lazy graph of before.
    ref = _run(monkeypatch, 0, plan=PASS * 2)
    got = _run(monkeypatch, 1, None, plan=PASS * 2)
    _equal(ref, got)
    assert vb.stats["verifies"] > 0 and vb.stats["bound_calls"] == 0
    assert vb.stats["whole"] == 0
    assert all(why in (1, 4) for _, why in vb.stats["ended"])


@pytest.mark.parametrize("state", ["refused", "recorded"])
def test_a_refusal_in_the_middle(monkeypatch, state):
    # From the third pass every attention record is refused. A record that
    # never replays direct ends the bound part: the layers before it stay
    # bound, the buffer is committed, the rest is lazy. A record still
    # learning its program runs in mlx's eval and the layers after it stay
    # bound.
    ref = _run(monkeypatch, 0, plan=PASS * 3)
    shim = _AtOnce()
    shim.state = state
    monkeypatch.setattr(vb, "_MODE", 1)
    monkeypatch.setattr(vb, "_kq_mod", [shim])
    lm, cache = _prefilled()
    out: list = []
    _rounds(lm, cache, PASS * 2, out)
    attn = set()
    for layer in lm.model.layers:
        if not layer.is_linear:
            for rec in vars(layer)["_verify_fused"][1].values():
                if isinstance(rec, tuple):
                    attn.add(rec[0])
    shim.refuse |= attn
    seen: list = []
    real, begin = shim.bound_call, vb.begin

    def spy(sid, handle, ins):
        r = real(sid, handle, ins)
        seen[-1].append((handle in attn, type(r) is not int))
        return r

    def spy_begin(B):
        seen.append([])
        return begin(B)

    shim.bound_call = spy
    monkeypatch.setattr(vb, "begin", spy_begin)
    before = shim.commits
    _rounds(lm, cache, PASS, out)
    out.extend(np.array(a.astype(mx.float32))
               for c in cache for a in c.state if isinstance(a, mx.array))
    _equal(ref, out)
    assert shim.commits > before and shim.open == 0
    # Records sent to the session after the first refused one of a verify,
    # and whether each was bound.
    after = [b for v in seen for i, (is_attn, b) in enumerate(v)
             if not is_attn and any(x[0] for x in v[:i])]
    if state == "refused":
        assert any(why == 4 for _, why in vb.stats["ended"])
        assert not after
    else:
        assert not vb.stats["ended"]
        assert vb.stats["lazy_rec"] and after and all(after)


def test_a_copied_cache_keeps_its_rows_under_records_run_at_once(monkeypatch):
    # A prompt cache entry and its deep copy share the tail arrays. The
    # copy is trimmed and continues with records that write in place the
    # moment they are called; the entry must still hold its rows after.
    # Fails when neither the session's copies nor qsa_whole.own are made.
    def run(mode, shim):
        monkeypatch.setattr(vb, "_MODE", mode)
        monkeypatch.setattr(vb, "_kq_mod", [shim])
        lm, entry = _prefilled(seed=7)
        out: list = []
        plan = [(4, 4), (5, 2), (3, 3), (4, 4)]
        _rounds(lm, entry, plan * 3, out)
        clone = copy.deepcopy(entry)
        for c in clone:
            if type(c) is q4.QSAKVCache:
                c.trim(9)
        _rounds(lm, clone, plan * 3, out)
        _rounds(lm, entry, plan * 3, out)
        return out

    monkeypatch.setattr(q4, "_VERIFY_FUSED", False)
    monkeypatch.setattr(q4, "_FUSED_WHOLE", False)
    plain = run(0, None)
    monkeypatch.setattr(q4, "_VERIFY_FUSED", True)
    monkeypatch.setattr(q4, "_FUSED_WHOLE", True)
    shim = _AtOnce()
    got = run(1, shim)
    assert shim.calls > 0 and shim.copies > 0
    _equal(plain, got)


def test_a_handout_before_the_rollback_keeps_its_rows_under_records_run_at_once(
        monkeypatch):
    # A holder that took the tail arrays after a verify appended and
    # before its rollback still reads the rows it took, with the next
    # verify's records writing in place the moment they are called.
    # Fails when neither the session's copies nor qsa_whole.own are made.
    monkeypatch.setattr(vb, "_MODE", 1)
    shim = _AtOnce()
    monkeypatch.setattr(vb, "_kq_mod", [shim])
    lm, cache = _prefilled(seed=11)
    args = lm.args
    held = []
    for i in range(12):
        block = mx.random.randint(0, args.vocab_size, (1, 4))
        with verify_submit.now():
            hid, _, sink = lm.speculative_verify_hidden(block, cache)
        mx.eval(hid, [c._kt for c in cache if type(c) is q4.QSAKVCache])
        if i >= 4:
            for c in cache:
                if type(c) is q4.QSAKVCache:
                    for a in (c._kt, c._vt):
                        held.append((a, c._tl, np.array(
                            a[..., :c._tl, :].astype(mx.float32))))
        lm.rollback_speculative_cache(cache, sink, 0, 4)
    assert held and shim.calls > 0
    for a, tl, want in held:
        assert np.array_equal(
            np.array(a[..., :tl, :].astype(mx.float32)), want)
