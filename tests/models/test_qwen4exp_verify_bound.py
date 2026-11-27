"""Bound replay of a verify (GMLX_SPEC_VERIFY_BOUND): the session's rules.

The tiny model's records run their primitives (its hyper-connection norm
and float MoE are not on kq's direct list), so kq refuses to bind them
unless KQ_FUSED_DIRECT_ANY lets them replay direct (the round loop test).
The other tests that need records to run outside an eval put a stand-in
behind the session that runs a record at once, the way a bound replay does
once its command buffer has completed: the in-place writes of the
whole-layer record land when the record is called, not when the caller
evaluates.
"""

import copy
import threading

import mlx.core as mx
import mlx_kquant as kq
import numpy as np
import pytest

import gmlx.models.qwen4_exp.model as q4
from gmlx.models.qwen4_exp import verify_bound as vb
from gmlx.serve import tick_guard
from gmlx.spec import verify_submit
from gmlx.upstream import gdn_patches

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
    monkeypatch.setattr(gdn_patches, "_host_steps", threading.local())
    monkeypatch.setattr(vb, "_local", threading.local())
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

    def bound_wait(self, sid):
        pass

    def bound_session_close(self, sid):
        pass

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


def _prefilled(seed=5, prep=None):
    lm, args = _lm(mx.bfloat16)
    if prep is not None:
        prep(lm)
    mx.random.seed(seed)
    cache = lm.make_cache()
    mx.eval(lm(mx.random.randint(0, args.vocab_size, (1, PROMPT)),
               cache=cache).logits)
    return lm, cache


def _run(monkeypatch, mode, shim=None, plan=None, now=True, prep=None):
    monkeypatch.setattr(vb, "_MODE", mode)
    monkeypatch.setattr(vb, "_kq_mod", [shim])
    lm, cache = _prefilled(prep=prep)
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


def _record(n):
    """A direct record of y = x @ w * 0.5 + 1 over x [n, n] float32, past
    its learning (kq's list holds its primitives)."""
    rng = np.random.default_rng(3)
    w = mx.array(rng.standard_normal((n, n)).astype(np.float32) / n)

    def f(x):
        return (x @ w) * 0.5 + 1.0

    x = _rand(n, 49)
    handle = kq.fuse_graph([x], [f(x)], direct=True)
    for seed in (50, 51):
        mx.eval(kq.fused_call(handle, [_rand(n, seed)]))
    assert kq.fused_direct_info(handle)["state"] == "replay"
    return handle, f


def _rand(n, seed):
    a = mx.array(np.random.default_rng(seed).standard_normal(
        (n, n)).astype(np.float32))
    mx.eval(a)
    return a


def test_a_stage_between_records_runs_after_our_buffer(monkeypatch):
    # kq's own command buffers, no stand-in. A stage built from primitives
    # between two records reads the first record's output: the session
    # commits its buffer ahead of the stage's mlx submission (mlx's encoder
    # then waits for the session's fence), and the second record is bound
    # on the stage while mlx still runs it (the session's encoder waits for
    # mlx's fence). Fails when the submission does not commit first: the
    # stage reads the first output before it is written.
    n = 1024
    handle, f = _record(n)
    monkeypatch.setattr(vb, "_MODE", 1)
    x = _rand(n, 5)
    want_y = f(x)
    want = f(want_y * 2.0 - 3.0)
    mx.eval(want_y, want)
    try:
        with verify_submit.now():
            s = vb.begin(1)
            assert s is not None
            try:
                y = vb.call(handle, [x])
                out = vb.call(handle, [y[0] * 2.0 - 3.0])
            finally:
                vb.end(s)
        info = kq.bound_session_info(s.sid)
        vb.wait()
        assert np.array_equal(np.array(out[0]), np.array(want))
        assert vb.stats["bound_calls"] == 2 and vb.stats["submits"] == 1
        assert info["fence_waits"] > 0
    finally:
        kq.fuse_release(handle)


def test_a_threads_session_closes_when_the_thread_ends(monkeypatch):
    # kq's own sessions. A server's generation worker is a thread a model
    # load, which makes its records and runs its verifies on its own
    # stream: the verifies open a session, whose pool holds the scratch of
    # the largest verify it ran. When the thread ends kq holds the session
    # no more, or every load and unload leaves one behind.
    import gc

    from gmlx._vendor.mlx_vlm._stream_cleanup import clear_mlx_streams

    monkeypatch.setattr(vb, "_MODE", 1)
    seen = {}

    def worker():
        n = 256
        handle, f = _record(n)
        try:
            x = _rand(n, 8)
            want = f(x)
            mx.eval(want)
            s, out = _verify(handle, x)
            vb.check()
            seen["sid"] = s.sid
            seen["equal"] = np.array_equal(np.array(out[0]), np.array(want))
        finally:
            kq.fuse_release(handle)
            clear_mlx_streams()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert seen.get("equal")
    del t
    gc.collect()
    with pytest.raises(ValueError, match="bad session"):
        kq.bound_session_info(seen["sid"])


def _fail_hook():
    from mlx_kquant import _ext

    return getattr(_ext, "_bound_fail", None)


_OOM = "Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
_TIMEOUT = "Caused GPU Timeout Error (00000002:kIOGPUCommandBufferCallbackErrorTimeout)"


def _verify(handle, x, fail=None, now=True):
    """One bound verify of the record on ``x``; ``fail`` (what, next) marks
    the session failed through kq's test hook ahead of the call."""
    with verify_submit.now():
        s = vb.begin(1)
        assert s is not None
        try:
            if fail is not None:
                _fail_hook()(s.sid, fail[0], next=fail[1])
            return s, vb.call(handle, [x])
        finally:
            vb.end(s)


@pytest.mark.skipif(_fail_hook() is None,
                    reason="needs a kq build with the _bound_fail test hook")
@pytest.mark.parametrize("why", [_OOM, _TIMEOUT])
def test_a_failed_session_buffer_raises_at_the_round_check(monkeypatch, why):
    # kq's own buffers. The verify's buffer fails (the hook marks it as it
    # is committed): the verify returns and an mlx read of its output runs,
    # then the round's check raises mlx's words on this thread, which
    # tick_guard classes as it classes mlx's with the lever off. The
    # session is closed; the next verify opens a new one and runs.
    n = 256
    handle, f = _record(n)
    monkeypatch.setattr(vb, "_MODE", 1)
    x = _rand(n, 6)
    want = f(x)
    mx.eval(want)
    try:
        s, out = _verify(handle, x, (why, True))
        mx.eval(out[0] + 0)
        with pytest.raises(RuntimeError) as e:
            vb.check()
        assert str(e.value) == f"[METAL] Command buffer execution failed: {why}."
        assert tick_guard.is_memory_error(e.value) == (why == _OOM)
        assert tick_guard.is_gpu_fault(e.value) == (why == _TIMEOUT)
        with pytest.raises(ValueError):
            kq.bound_session_info(s.sid)
        vb.check()
        s2, out2 = _verify(handle, x)
        assert s2.sid != s.sid
        vb.check()
        assert np.array_equal(np.array(out2[0]), np.array(want))
    finally:
        kq.fuse_release(handle)


@pytest.mark.skipif(_fail_hook() is None,
                    reason="needs a kq build with the _bound_fail test hook")
def test_a_failure_the_round_did_not_check_raises_at_the_next_verify(
        monkeypatch):
    # A failure kq knows while the verify runs ends it at the next record
    # (raised once, the session closed by then). A round that raised before
    # its check leaves it to the next verify's begin, which raises it and
    # opens no work on the failed session.
    n = 256
    handle, f = _record(n)
    monkeypatch.setattr(vb, "_MODE", 1)
    x = _rand(n, 7)
    want = f(x)
    mx.eval(want)
    try:
        with pytest.raises(RuntimeError, match="execution failed: "):
            _verify(handle, x, (_OOM, False))
        assert getattr(vb._local, "sid", None) is None
        vb.check()
        _verify(handle, x, (_TIMEOUT, True))
        with pytest.raises(RuntimeError, match="Timeout"):
            _verify(handle, x)
        s, out = _verify(handle, x)
        vb.check()
        assert np.array_equal(np.array(out[0]), np.array(want))
    finally:
        kq.fuse_release(handle)


def _lora_every_layer(lm):
    """A LoRA on a linear of every layer, as an adapter install leaves the
    model (its recorded verify graphs dropped)."""
    import mlx.nn as nn

    from gmlx.load.modules import LoRAKQuantLinear

    mx.random.seed(7)
    for layer in lm.model.layers:
        name, m = next((n, m) for n, m in layer.named_modules()
                       if n and type(m) is nn.Linear)
        *path, key = name.split(".")
        parent = layer
        for p in path:
            parent = parent[int(p)] if isinstance(parent, list) else parent[p]
        out_d, in_d = m.weight.shape
        parent[key] = LoRAKQuantLinear(
            m, mx.random.normal((4, in_d)) * 0.01,
            mx.random.normal((out_d, 4)) * 0.01, 0.5)
    q4.invalidate_verify_compile()


@pytest.mark.parametrize("why", ["streamed experts", "an adapter on every layer"])
def test_a_verify_with_no_records_opens_no_session(monkeypatch, why):
    # Streamed experts swap the weights around each layer call, so the
    # recorded verify is off for the process from the first swap
    # (stream/feeder_common.py); a layer with a LoRA runs plain. A session
    # there would bind nothing and submit the verify's input on its own.
    monkeypatch.setattr(q4, "_VERIFY_EPOCH", q4._VERIFY_EPOCH)
    prep = (_lora_every_layer if why == "an adapter on every layer"
            else lambda lm: q4.invalidate_verify_compile(block=True))
    ref = _run(monkeypatch, 0, plan=PASS * 2, prep=prep)
    got = _run(monkeypatch, 1, None, plan=PASS * 2, prep=prep)
    _equal(ref, got)
    assert vb.stats["verifies"] == 0 and vb.stats["submits"] == 0


def test_host_step_counts_stay_on_the_thread_of_a_session(monkeypatch):
    # The records of a bound verify read step counts built on the host. A
    # model on another thread (a server's generation worker, one a model
    # load) builds its counts as before, while the session lives and after.
    # A thread whose session failed builds them as before too.
    from gmlx._vendor.mlx_vlm._stream_cleanup import clear_mlx_streams

    def cached():
        return gdn_patches.replay_steps(3, 2) is gdn_patches.replay_steps(3, 2)

    monkeypatch.setattr(vb, "_MODE", 1)
    seen = {}

    def worker():
        n = 256
        handle, _ = _record(n)
        try:
            _verify(handle, _rand(n, 8))
            vb.check()
            seen["session"] = cached()
            other = threading.Thread(
                target=lambda: seen.setdefault("other", cached()))
            other.start()
            other.join()
            vb._drop()
            seen["dropped"] = cached()
        finally:
            kq.fuse_release(handle)
            clear_mlx_streams()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    later = threading.Thread(target=lambda: seen.setdefault("later", cached()))
    later.start()
    later.join()
    assert seen == {"session": True, "other": False, "dropped": False,
                    "later": False}


def _spec_decode(monkeypatch, mode, n_gen=40, block=4):
    """Greedy speculative decode of the tiny model with its MTP drafter
    through the round loop (stream_speculative, _owned_decode_rounds): the
    tokens, the target's cache states and the bound stats. KQ_FUSED_DIRECT_ANY
    lets the records learn their programs with the primitives kq does not
    list, so with the lever on they bind."""
    from gmlx.models.qwen4_exp.mtp import Qwen4ExpMTPConfig, Qwen4ExpMTPDrafter
    from gmlx.spec.speculative import stream_speculative
    from test_qwen4exp_mtp import _randomize, _Wrap

    monkeypatch.setenv("KQ_FUSED_DIRECT_ANY", "1")
    kq.env_cache(True)
    monkeypatch.setattr(vb, "_MODE", mode)
    monkeypatch.setattr(vb, "_local", threading.local())
    monkeypatch.setattr(q4, "_FUSED_WHOLE", None)
    for k in vb.stats:
        monkeypatch.setitem(vb.stats, k, {} if isinstance(
            vb.stats[k], dict) else 0)
    lm, args = _lm(mx.bfloat16)
    drafter = Qwen4ExpMTPDrafter(Qwen4ExpMTPConfig(text=args, block_size=block))
    _randomize(drafter, seed=3)
    drafter.set_dtype(mx.bfloat16)
    drafter.reset(_Wrap(lm))
    mx.random.seed(11)
    prompt = mx.random.randint(0, args.vocab_size, (1, PROMPT))
    cache = lm.make_cache()
    toks = list(stream_speculative(
        lm, drafter, prompt, prompt_cache=cache, max_tokens=n_gen,
        sampler=None, draft_block_size=block))
    states = [np.array(a.astype(mx.float32)) for c in cache for a in c.state
              if isinstance(a, mx.array)]
    return toks, states, dict(vb.stats), q4._fused_whole()


def test_the_round_loop_binds_records_with_the_same_tokens(monkeypatch):
    # kq's own sessions through the round loop, bound verify on against
    # off: the records bind (no verify ends its bound part early), and the
    # tokens and the caches after the last round are the same.
    off = _spec_decode(monkeypatch, 0)
    on = _spec_decode(monkeypatch, 1)
    assert off[2]["verifies"] == 0 and off[3] is False
    assert on[3] is True
    assert on[2]["verifies"] > 0 and on[2]["bound_calls"] > 0
    assert on[2]["whole"] > 0 and on[2]["ended"] == {}
    assert len(on[0]) == len(off[0]) > 0 and on[0] == off[0]
    _equal(off[1], on[1])
