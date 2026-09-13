#!/usr/bin/env python3
"""Sampling-profile injection, MTP thinking-budget transport, XTC, and the
batch sampler - carved from test_server_patches.py. CPU-only."""
from __future__ import annotations

import importlib
import types

import pytest

pytest.importorskip("mlx_vlm")

import gmlx.serve.patches as sp  # noqa: E402
from gmlx.serve.patches import _common as sp_common  # noqa: E402
from gmlx.serve.patches import sampling as sp_sampling  # noqa: E402
import gmlx.serve.bridge_vlm as serving  # noqa: E402

_APP = importlib.import_module("mlx_vlm.server.app")
_UTILS = importlib.import_module("mlx_vlm.utils")
_PKG = importlib.import_module("mlx_vlm.server")

from test_server_patches import _FakeThinkTok, _spec  # noqa: E402


# 1. sampling injection (pure)
def test_inject_overrides_unset_keeps_client_set():
    args = types.SimpleNamespace(temperature=0.7, top_p=0.95, top_k=5, max_tokens=512)
    request = types.SimpleNamespace(model_fields_set={"top_k"})   # client set top_k
    spec = _spec(temperature=0.2, top_p=0.9, top_k=99, max_tokens=2048)
    sp_sampling._inject_profile_sampling(args, request, spec)
    assert args.temperature == 0.2       # injected (unset)
    assert args.top_p == 0.9             # injected (unset)
    assert args.top_k == 5               # kept (client set)
    assert args.max_tokens == 2048       # injected (unset)


def test_inject_max_tokens_alias_respects_max_output_tokens():
    args = types.SimpleNamespace(max_tokens=512)
    request = types.SimpleNamespace(model_fields_set={"max_output_tokens"})
    sp_sampling._inject_profile_sampling(args, request, _spec(max_tokens=2048))
    assert args.max_tokens == 512        # responses API set it -> not overridden


# 1b. ignore-eos: forced-length decode
def test_install_ignore_eos_suppresses_stop():
    crit = _UTILS.StoppingCriteria([7, 8])
    assert crit(7) is True               # baseline: 7 is an eos id -> stop
    sp.install_ignore_eos()
    assert crit(7) is False              # patched: EOS never stops decode
    assert crit(8) is False
    assert crit(123) is False
    sp.install_ignore_eos()              # idempotent
    assert crit(7) is False


def test_inject_noop_without_spec_or_sampling():
    args = types.SimpleNamespace(temperature=0.7)
    request = types.SimpleNamespace(model_fields_set=set())
    sp_sampling._inject_profile_sampling(args, request, None)
    sp_sampling._inject_profile_sampling(args, request, _spec())          # empty sampling
    assert args.temperature == 0.7


def test_inject_skips_unknown_arg_attr():
    args = types.SimpleNamespace(temperature=0.7)               # no top_p attr
    request = types.SimpleNamespace(model_fields_set=set())
    sp_sampling._inject_profile_sampling(args, request, _spec(top_p=0.5))
    assert not hasattr(args, "top_p")


def test_inject_thinking_budget_from_profile():
    # off by default: GenerationArguments.thinking_budget is None; a profile/model
    # value seeds it when the client didn't ask.
    args = types.SimpleNamespace(thinking_budget=None)
    request = types.SimpleNamespace(model_fields_set=set())
    sp_sampling._inject_profile_sampling(args, request, _spec(thinking_budget=1024))
    assert args.thinking_budget == 1024


def test_inject_thinking_budget_request_wins():
    args = types.SimpleNamespace(thinking_budget=256)           # client sent 256
    request = types.SimpleNamespace(model_fields_set={"thinking_budget"})
    sp_sampling._inject_profile_sampling(args, request, _spec(thinking_budget=1024))
    assert args.thinking_budget == 256                          # not clobbered


def test_inject_thinking_budget_off_by_default():
    args = types.SimpleNamespace(thinking_budget=None)
    request = types.SimpleNamespace(model_fields_set=set())
    sp_sampling._inject_profile_sampling(args, request, _spec(temperature=0.2))  # no budget
    assert args.thinking_budget is None                         # stays off


# 1c. server thinking_budget on MTP models (mtp_thinking)
from gmlx.serve.patches import mtp_thinking as sp_mtp  # noqa: E402


@pytest.fixture
def _mtp_seams():
    """Force the owned-prefill class flag on (install-order precondition) and
    snapshot the three methods mtp_thinking wraps."""
    from gmlx.spec.engine import _FULL_PREFILL_FLAG
    gen = importlib.import_module("mlx_vlm.server.generation")
    ar = importlib.import_module("mlx_vlm.generate.ar")
    cls = gen.ResponseGenerator
    had_flag = getattr(ar.PromptProcessingBatch, _FULL_PREFILL_FLAG, False)
    setattr(ar.PromptProcessingBatch, _FULL_PREFILL_FLAG, True)
    saved = (cls.generate, cls._make_thinking_budget_criteria,
             ar.PromptProcessingBatch.generate)
    yield gen, ar
    (cls.generate, cls._make_thinking_budget_criteria,
     ar.PromptProcessingBatch.generate) = saved
    if not had_flag:
        delattr(ar.PromptProcessingBatch, _FULL_PREFILL_FLAG)


def _mtp_self():
    return types.SimpleNamespace(
        draft_model=object(), draft_kind="mtp",
        tokenizer=_FakeThinkTok(),
        _thinking_token_ids=lambda args: (99, 100))


def _budget_args(**kw):
    base = dict(thinking_budget=6, enable_thinking=False, seed=None,
                temperature=1.0, thinking_start_token=None,
                thinking_end_token=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_mtp_thinking_install_refuses_without_owned_prefill():
    from gmlx.spec.engine import _FULL_PREFILL_FLAG
    gen = importlib.import_module("mlx_vlm.server.generation")
    ar = importlib.import_module("mlx_vlm.generate.ar")
    had_flag = getattr(ar.PromptProcessingBatch, _FULL_PREFILL_FLAG, False)
    if had_flag:
        delattr(ar.PromptProcessingBatch, _FULL_PREFILL_FLAG)
    before = gen.ResponseGenerator.generate
    try:
        sp_mtp.install_mtp_thinking_budget()
        assert gen.ResponseGenerator.generate is before   # refused, unbound
    finally:
        if had_flag:
            setattr(ar.PromptProcessingBatch, _FULL_PREFILL_FLAG, True)


def test_mtp_thinking_defers_budget_only_for_mtp(_mtp_seams):
    gen, _ar = _mtp_seams
    cls = gen.ResponseGenerator
    seen = []

    def stub(self, prompt, images=None, audio=None, args=None, videos=None):
        seen.append(args.thinking_budget if args is not None else None)
        return "gen"

    cls.generate = stub
    sp_mtp.install_mtp_thinking_budget()
    args = _budget_args()
    assert cls.generate(_mtp_self(), "p", args=args) == "gen"
    assert seen[-1] is None                              # moved aside
    assert getattr(args, sp_mtp._DEFERRED_ATTR) == 6
    # Non-MTP drafter: untouched, so the upstream raise still fires there.
    eagle = types.SimpleNamespace(draft_model=object(), draft_kind="eagle")
    args2 = _budget_args()
    cls.generate(eagle, "p", args=args2)
    assert seen[-1] == 6 and not hasattr(args2, sp_mtp._DEFERRED_ATTR)
    # Plain model and args=None: untouched.
    plain = types.SimpleNamespace(draft_model=None, draft_kind=None)
    args3 = _budget_args()
    cls.generate(plain, "p", args=args3)
    assert seen[-1] == 6
    cls.generate(_mtp_self(), "p")                       # args=None tolerated


def test_mtp_thinking_criteria_restores_even_on_early_out(_mtp_seams):
    gen, _ar = _mtp_seams
    cls = gen.ResponseGenerator
    cls._make_thinking_budget_criteria = lambda self, args, input_ids: None
    sp_mtp.install_mtp_thinking_budget()
    make = cls._make_thinking_budget_criteria
    args = _budget_args(thinking_budget=None)
    setattr(args, sp_mtp._DEFERRED_ATTR, 6)
    crit = make(_mtp_self(), args, [1, 2, 3])
    assert args.thinking_budget == 6                     # restored
    assert not hasattr(args, sp_mtp._DEFERRED_ATTR)
    # Delegate returned None: the hook rides a duck-shaped carrier that the
    # plain batch loop can call without raising.
    hook = crit._kq_mtp_hook
    assert hook is not None and hook.budget == 6
    assert crit(5) is None and crit.pop_forced_token_id() is None
    # Prompt ending inside an open think block seeds the hook in-thinking.
    args_open = _budget_args(thinking_budget=None)
    setattr(args_open, sp_mtp._DEFERRED_ATTR, 6)
    assert make(_mtp_self(), args_open, [1, 2, 99])._kq_mtp_hook.in_thinking


def test_mtp_thinking_criteria_restores_on_raise(_mtp_seams):
    gen, _ar = _mtp_seams
    cls = gen.ResponseGenerator

    def boom(self, args, input_ids):
        raise RuntimeError("delegate failed")

    cls._make_thinking_budget_criteria = boom
    sp_mtp.install_mtp_thinking_budget()
    args = _budget_args(thinking_budget=None)
    setattr(args, sp_mtp._DEFERRED_ATTR, 6)
    with pytest.raises(RuntimeError):
        cls._make_thinking_budget_criteria(_mtp_self(), args, [1])
    assert args.thinking_budget == 6                     # not stranded


def test_mtp_thinking_full_chain_with_seed_and_tbfix(_mtp_seams):
    # Runtime chain mtp -> seed -> tbfix: one call restores the deferred
    # budget, stashes the seed, builds the armed criteria, and attaches the
    # rounds hook to it.
    import gmlx.serve.seed_rows as sr
    gen, ar = _mtp_seams
    cls = gen.ResponseGenerator
    saved_insert = (ar.BatchGenerator.insert, ar.GenerationBatch._step,
                    ar.SpeculativeGenerationBatch.next)
    sr._PENDING.clear()
    try:
        sp.install_thinking_budget_fix()
        sr.install_per_request_seed()
        sp_mtp.install_mtp_thinking_budget()
        args = _budget_args(thinking_budget=None, seed=11)
        setattr(args, sp_mtp._DEFERRED_ATTR, 6)
        crit = cls._make_thinking_budget_criteria(_mtp_self(), args, [1, 2])
        assert args.thinking_budget == 6
        assert sr._PENDING == [11]
        assert crit is not None and crit.in_thinking is False   # tbfix armed
        assert crit._kq_mtp_hook is not None and crit._kq_mtp_hook.budget == 6
    finally:
        (ar.BatchGenerator.insert, ar.GenerationBatch._step,
         ar.SpeculativeGenerationBatch.next) = saved_insert
        sr._PENDING.clear()


def test_mtp_thinking_transport_stash_and_batch_drop(_mtp_seams):
    _gen, ar = _mtp_seams
    ar.PromptProcessingBatch.generate = \
        lambda self, sampler, *a, **k: self._out
    sp_mtp.install_mtp_thinking_budget()
    wrapper = ar.PromptProcessingBatch.generate
    hook = object()
    crit = types.SimpleNamespace(_kq_mtp_hook=hook)
    cache_entry = types.SimpleNamespace()
    batch = types.SimpleNamespace(prompt_cache=[cache_entry], uids=["u"])
    me = types.SimpleNamespace(
        draft_model=object(), draft_kind="mtp",
        thinking_budget_criteria=[crit], _out=batch)
    assert wrapper(me, None) is batch
    assert cache_entry._kq_mtp_thinking_hook is hook     # B==1 stash
    # B>1: dropped, nothing stashed.
    c2 = types.SimpleNamespace()
    batch2 = types.SimpleNamespace(prompt_cache=[c2], uids=["u", "v"])
    me2 = types.SimpleNamespace(
        draft_model=object(), draft_kind="mtp",
        thinking_budget_criteria=[crit, crit], _out=batch2)
    wrapper(me2, None)
    assert not hasattr(c2, "_kq_mtp_thinking_hook")
    # Criteria/rows mismatch: dropped, not indexed blindly.
    c3 = types.SimpleNamespace()
    batch3 = types.SimpleNamespace(prompt_cache=[c3], uids=["u"])
    me3 = types.SimpleNamespace(
        draft_model=object(), draft_kind="mtp",
        thinking_budget_criteria=[crit, crit], _out=batch3)
    wrapper(me3, None)
    assert not hasattr(c3, "_kq_mtp_thinking_hook")
    # Non-MTP batch: untouched.
    c4 = types.SimpleNamespace()
    batch4 = types.SimpleNamespace(prompt_cache=[c4], uids=["u"])
    me4 = types.SimpleNamespace(
        draft_model=None, draft_kind=None,
        thinking_budget_criteria=[crit], _out=batch4)
    wrapper(me4, None)
    assert not hasattr(c4, "_kq_mtp_thinking_hook")


def test_mtp_thinking_flags_carry_through_preflight(_mtp_seams):
    # The defer wrap carries earlier flags forward and stamps its own, so a
    # later mem_preflight re-install must see its flag and not double-wrap.
    from gmlx.serve import mem_preflight as mp
    gen, _ar = _mtp_seams
    cls = gen.ResponseGenerator

    def stub(self, prompt, images=None, audio=None, args=None, videos=None):
        return "gen"

    stub.__dict__[mp._INSTALLED_FLAG] = True             # preflight installed
    cls.generate = stub
    sp_mtp.install_mtp_thinking_budget()
    wrapped = cls.generate
    assert wrapped is not stub
    assert getattr(wrapped, mp._INSTALLED_FLAG, False)   # carried forward
    mp.install_memory_preflight()
    assert cls.generate is wrapped                       # no double wrap
    sp_mtp._install_defer(cls)
    assert cls.generate is wrapped                       # own re-install no-ops


# 7. XTC sampling injection
def test_attach_xtc_noop_without_request_or_profile():
    args = types.SimpleNamespace(logits_processors=None)
    request = types.SimpleNamespace(model_fields_set=set())
    sp_sampling._attach_xtc(args, request, None)
    assert args.logits_processors is None


def test_attach_xtc_appends_processor_from_request_extras():
    args = types.SimpleNamespace(logits_processors=None)
    request = types.SimpleNamespace(xtc_probability=1.0, xtc_threshold=0.2)
    sp_sampling._attach_xtc(args, request, None)
    assert args.logits_processors is not None and len(args.logits_processors) == 1
    # functional: prob=1.0 always triggers; threshold 0.2 with probs ~[.6,.3,.1]
    # masks the top token, so argmax moves to the runner-up.
    import math

    import mlx.core as mx
    logits = mx.log(mx.array([[0.6, 0.3, 0.1]]))
    out = args.logits_processors[0](mx.array([0]), logits)
    assert int(mx.argmax(out, axis=-1).item()) == 1
    assert math.isinf(float(out[0, 0].item()))


def test_attach_xtc_profile_fallback_and_request_precedence():
    spec = _spec(xtc_probability=1.0, xtc_threshold=0.3)
    token = serving.set_active_spec(spec)
    try:
        args = types.SimpleNamespace(logits_processors=None)
        sp_sampling._attach_xtc(args, types.SimpleNamespace(), None)
        assert args.logits_processors and len(args.logits_processors) == 1
        # an explicit client 0.0 wins over the profile and disables XTC
        args2 = types.SimpleNamespace(logits_processors=None)
        sp_sampling._attach_xtc(args2, types.SimpleNamespace(xtc_probability=0.0), None)
        assert args2.logits_processors is None
    finally:
        serving.reset_active_spec(token)


def test_attach_xtc_string_zero_disables():
    # extra="allow" preserves raw JSON types: a client's "0" (string) is truthy,
    # but must still disable XTC after coercion - the live bug this pins down.
    for raw in ("0", "0.0", 0, 0.0):
        args = types.SimpleNamespace(logits_processors=None)
        sp_sampling._attach_xtc(args, types.SimpleNamespace(xtc_probability=raw), None)
        assert args.logits_processors is None, f"xtc_probability={raw!r}"


def test_attach_xtc_string_prob_attaches():
    args = types.SimpleNamespace(logits_processors=None)
    request = types.SimpleNamespace(xtc_probability="0.5", xtc_threshold="0.2")
    sp_sampling._attach_xtc(args, request, None)
    assert args.logits_processors is not None and len(args.logits_processors) == 1


def test_attach_xtc_garbage_prob_rejects_400():
    # matches the neighboring coercion behavior (_sampling_float): typed 400,
    # never a 500 out of the handler, and args stay untouched.
    from fastapi import HTTPException
    args = types.SimpleNamespace(logits_processors=None)
    request = types.SimpleNamespace(xtc_probability="lots")
    with pytest.raises(HTTPException) as ei:
        sp_sampling._attach_xtc(args, request, None)
    assert ei.value.status_code == 400
    assert args.logits_processors is None


def test_xtc_special_tokens_dedup_and_defensive():
    class _Tok:
        eos_token_id = 7

        def encode(self, s, add_special_tokens=True):
            return [7]

    assert sp_sampling._xtc_special_tokens(types.SimpleNamespace(tokenizer=_Tok())) == [7]
    assert sp_sampling._xtc_special_tokens(None) == []

    class _IntEosTok(_Tok):
        # regression (live server): TokenizersBackend exposes eos_token_ids as
        # a bare int - iterating it raised "'int' object is not iterable"
        eos_token_ids = 9

    assert sp_sampling._xtc_special_tokens(
        types.SimpleNamespace(tokenizer=_IntEosTok())) == [7, 9]


def test_install_xtc_wraps_and_stacks_with_profile_injection():
    sp.install_gen_args_profile_injection()
    sp.install_xtc_sampling()
    fn = _APP._build_gen_args
    assert getattr(fn, sp_common._PATCH_FLAG, False)      # carried forward
    assert getattr(fn, sp_sampling._XTC_FLAG, False)
    sp.install_xtc_sampling()                      # idempotent
    assert _APP._build_gen_args is fn


# 7a2. top_k / min_p aware batch sampler (the historical dropped-top_k bug class)
def _kept_ids(sampler, probs):
    """Vocab ids surviving the sampler's filter for one row of probs, plus the
    masked [1, k] logits (sorted desc by prob)."""
    import mlx.core as mx
    logits = mx.log(mx.array([probs]))
    masked, part, order = sampler._filtered(logits)
    kept = []
    for j in range(masked.shape[-1]):
        if float(masked[0, j].item()) != float("-inf"):
            kept.append(int(part[0, int(order[0, j].item())].item()))
    return kept, masked


def test_fast_sampler_hierarchical_topk_matches_flat():
    # Large vocabs route _filtered's top-k through the hierarchical id
    # selector; the surviving id SET must equal the flat argpartition's
    # (order within the set is re-sorted downstream either way).
    import mlx.core as mx
    for v, seed in ((201088, 0), (200005, 1), (131072, 2)):
        lp = mx.random.normal((1, v), key=mx.random.key(seed))
        lp = lp.astype(mx.float32)
        mx.eval(lp)
        hier = set(sp_sampling._topk_ids(lp, 20)[0].tolist())
        flat = set(mx.argpartition(-lp, kth=19, axis=-1)[:, :20][0].tolist())
        assert hier == flat


def test_fast_sampler_masking():
    S = sp_sampling._FastPositionedSampler
    probs = [0.4, 0.3, 0.2, 0.1]
    # top_k=2: exactly the two most probable survive
    kept, _ = _kept_ids(S(temperature=1.0, top_k=2), probs)
    assert kept == [0, 1]
    # top_p=0.5: nucleus keeps ids 0,1 (mass-before 0.0 and 0.4 < 0.5)
    kept, _ = _kept_ids(S(temperature=1.0, top_p=0.5), probs)
    assert kept == [0, 1]
    # min_p=0.6: threshold 0.4*0.6=0.24 -> 0.3 stays, 0.2 pruned
    kept, _ = _kept_ids(S(temperature=1.0, min_p=0.6), probs)
    assert kept == [0, 1]
    # llama.cpp order: top_k FIRST, top_p over the k renormalized survivors.
    # [0.36, 0.34, 0.30] @ top_k=2 renorms to [0.514, 0.486]; top_p=0.45 then
    # drops the runner-up (mass-before 0.514 > 0.45). Vocab-order top_p would
    # have kept it (0.36 < 0.45).
    kept, _ = _kept_ids(S(temperature=1.0, top_k=2, top_p=0.45),
                        [0.36, 0.34, 0.30])
    assert kept == [0]
    # the argmax can never be filtered away (_MIN_KEEP)
    kept, _ = _kept_ids(S(temperature=1.0, top_p=1e-9), probs)
    assert kept == [0]
    # temperature scales the surviving logits (applied last)
    import math
    _, masked = _kept_ids(S(temperature=0.5, top_k=2), probs)
    assert math.isclose(float(masked[0, 0].item()), math.log(0.4) / 0.5,
                        rel_tol=1e-5)


def test_fast_sampler_call_shapes_and_determinism():
    import mlx.core as mx
    s = sp_sampling._FastPositionedSampler(temperature=0.7, top_k=1)
    logits = mx.log(mx.array([[0.1, 0.2, 0.6, 0.1]]))
    # top_k=1 leaves a single candidate -> always the argmax id
    assert int(s(logits).item()) == 2
    # a drafter's [B, 1, V] block keeps its leading shape
    assert s(logits[:, None, :]).shape == (1, 1)


def test_fast_sampler_install_lands():
    """Identity check on the REAL upstream class: an mlx-vlm rename of
    ResponseGenerator._make_sampler must fail here, not silently no-op."""
    gen = importlib.import_module("mlx_vlm.server.generation")
    cls = gen.ResponseGenerator
    original = cls._make_sampler
    try:
        sp.install_fast_sampler()
        patched = cls._make_sampler
        assert patched is not original                      # actually swapped
        assert getattr(patched, sp_sampling._FAST_SAMPLER_FLAG, False)
        sp.install_fast_sampler()                           # idempotent
        assert cls._make_sampler is patched
        me = types.SimpleNamespace()
        # greedy keeps the batch engine's argmax fast path
        assert patched(me, types.SimpleNamespace(temperature=0)) is None
        s = patched(me, types.SimpleNamespace(temperature=0.6, top_p=0.9,
                                              top_k=40, min_p=0.05, seed=3))
        assert isinstance(s, sp_sampling._FastPositionedSampler)
        assert (s.top_k, s.min_p, s.seed) == (40, 0.05, 3)
    finally:
        cls._make_sampler = original
