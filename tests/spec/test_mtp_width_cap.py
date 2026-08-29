"""MTP batch-width gate: speculation runs only while the live decode batch is
at or under a per-family cap; wider batches decode plain for the rest of the
generator (a latch -- re-arming a drafter mid-flight would mean re-seeding
every row's hidden/shared-KV, the seam that produced the 2026-07 injection
crashes). The drafter stays loaded throughout.

Drives _owned_decode_rounds_batch directly with fakes. The strict fake drafter
raises on every forward-work method, so "gated" is asserted structurally: if
the gate ever drafts, the test errors rather than silently measuring nothing.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from gmlx.spec.speculative import (
    _mtp_width_cap,
    _owned_decode_rounds_batch,
    _width_cap_logged,
)
import gmlx.spec.speculative as spec

VOCAB = 32


@pytest.fixture(autouse=True)
def _clean_width_cap_env(monkeypatch):
    monkeypatch.delenv("GMLX_MTP_WIDTH_CAP", raising=False)
    monkeypatch.delenv("MLX_VLM_GGUF_SPEC_WIDTH_CAP", raising=False)
    spec._width_cap_memo = ("", None)
    _width_cap_logged.clear()
    yield
    spec._width_cap_memo = ("", None)
    _width_cap_logged.clear()


class _FakeCache:
    """Target KV cache: tracks width and offset, nothing else."""

    def __init__(self, width=1, offset=8):
        self.width = width
        self.offset = offset
        self.extend_calls = 0

    def extend(self, other):
        self.extend_calls += 1
        self.width += getattr(other, "width", 1)

    def filter(self, keep):
        self.width = int(keep.size)


class _FakeLM:
    """Target: emits a deterministic per-row token stream."""

    def __init__(self):
        self.calls = []
        self._rope_deltas = None

    def __call__(self, x, cache=None, **kw):
        self.calls.append(tuple(x.shape))
        B, L = x.shape
        logits = mx.zeros((B, L, VOCAB))
        # row r -> token (r + 3), stable across rounds
        rows = mx.arange(B) + 3
        onehot = (mx.arange(VOCAB)[None, :] == rows[:, None]).astype(mx.float32)
        logits = logits + onehot[:, None, :] * 10.0
        return SimpleNamespace(logits=logits)


class _StrictDrafter:
    """Records the cheap bookkeeping calls; raises on any GPU-side drafter
    work. Gating is asserted structurally: a gate that leaks into drafting
    errors out instead of silently measuring nothing. reset/set_shared_kv are
    recorded rather than fatal so a test can start ungated (they run in the
    speculative seed block) and still prove the trip.
    """

    uses_shared_kv = True

    def __init__(self, cap=0, limit=0, block_size=4):
        self.config = SimpleNamespace(block_size=block_size)
        self.accept_lens = []
        self.draft_lens = []
        self.mtp_width_cap = cap
        self.mtp_width_limit = limit
        self.reset_calls = []
        self.shared_kv_calls = 0
        self.forward_calls = []

    def reset(self, model, left_padding=None):
        self.reset_calls.append(left_padding)
        self.accept_lens = []
        self.draft_lens = []
        return []

    def set_shared_kv(self, *a, **kw):
        self.shared_kv_calls += 1

    def _forbidden(self, name):
        self.forward_calls.append(name)
        raise AssertionError(f"drafter.{name} called while gated")

    def draft_block(self, *a, **kw):
        return self._forbidden("draft_block")

    def inject_rows(self, *a, **kw):
        return self._forbidden("inject_rows")

    def prefill_from_target_hidden(self, *a, **kw):
        return self._forbidden("prefill_from_target_hidden")

    def accept_verified_tokens_batch(self, *a, **kw):
        return self._forbidden("accept_verified_tokens_batch")

    def filter_batch(self, *a, **kw):
        return self._forbidden("filter_batch")


class _EchoLM:
    """Target whose next token is a function of the INPUT token rather than the
    row slot, so every row carries its own distinct stream. That makes a
    mis-sliced double buffer visible: a surviving row would inherit a retired
    row's stream instead of continuing its own."""

    def __init__(self):
        self.widths = []
        self._rope_deltas = None

    def __call__(self, x, cache=None, **kw):
        B, _ = x.shape
        self.widths.append(B)
        nxt = (x[:, -1] + 1) % VOCAB
        onehot = (mx.arange(VOCAB)[None, :] == nxt[:, None]).astype(mx.float32)
        return SimpleNamespace(logits=(onehot * 10.0)[:, None, :])


def _drive(drafter, *, B=3, max_tokens=3, sampler=None, model=None,
           prompt_cache=None, lm=None, rounds=None, stop_check=None,
           eos_token_ids=None):
    model = model if model is not None else SimpleNamespace()
    lm = lm if lm is not None else _FakeLM()
    prompt_cache = prompt_cache if prompt_cache is not None else [
        _FakeCache(width=B)]
    gen = _owned_decode_rounds_batch(
        model, drafter, lm, prompt_cache,
        hidden=mx.zeros((B, 1, 8)),
        b=list(range(1, B + 1)),
        shared_kv={"full": (mx.zeros((B, 2, 4, 4)), mx.zeros((B, 2, 4, 4)))},
        seed_tokens=None,
        emitted=[1] * B,
        max_tokens=max_tokens,
        sampler=sampler,
        draft_block_size=None,
        stop_check=stop_check,
        eos_token_ids=eos_token_ids,
    )
    out = []
    for i, (toks, meta) in enumerate(gen):
        out.append((toks, meta))
        if rounds is not None and len(out) >= rounds:
            gen.close()
            break
    return out, lm, prompt_cache


# -- load-time stamping --------------------------------------------------

def _stamped(model_type, env=None, monkeypatch=None, target=None):
    from gmlx.spec.mtp_load import _stamp_mtp_width_cap

    if env is not None:
        monkeypatch.setenv("MLX_VLM_GGUF_SPEC_WIDTH_CAP", env)
    d = SimpleNamespace()
    _stamp_mtp_width_cap(d, model_type, target=target, log=lambda *a, **k: None)
    return d.mtp_width_cap, d.mtp_width_limit


class _FakeProj:
    """Stands in for a switch/dense projection: MoE variants carry either a
    num_experts attribute (quantized/K-quant) or a stacked 3D weight."""

    def __init__(self, weight=None, num_experts=None):
        self.weight = weight
        if num_experts is not None:
            self.num_experts = num_experts


class _FakeMLP:
    def __init__(self, proj):
        self.gate_proj = proj

    def modules(self):
        return [self]


class _FakeLayer:
    def __init__(self, proj):
        self._mlp = _FakeMLP(proj)

    def modules(self):
        return [self, self._mlp]


def _fake_model(proj, *, vlm=False):
    """A model whose decoder layers hold one projection each. vlm=True nests it
    the way a VLM wrapper does (model.language_model.model.layers)."""
    inner = SimpleNamespace(layers=[_FakeLayer(proj)])
    if vlm:
        return SimpleNamespace(language_model=SimpleNamespace(model=inner))
    return SimpleNamespace(model=inner)


DENSE_PROJ = _FakeProj(weight=mx.zeros((8, 4)))
MOE_PROJ_3D = _FakeProj(weight=mx.zeros((6, 8, 4)))
MOE_PROJ_ATTR = _FakeProj(weight=None, num_experts=6)


# -- structural MoE detection --------------------------------------------

def test_model_is_moe_detects_stacked_expert_weight():
    from gmlx.load.loader import model_is_moe

    assert model_is_moe(_fake_model(MOE_PROJ_3D))
    assert model_is_moe(_fake_model(MOE_PROJ_ATTR))


def test_model_is_moe_false_on_dense():
    """A dense 2D weight must NOT read as MoE. _switch_num_experts would return
    weight.shape[0] here (the output width), which is why it can't be the
    test."""
    from gmlx.load.loader import model_is_moe

    assert not model_is_moe(_fake_model(DENSE_PROJ))


def test_model_is_moe_walks_the_vlm_wrapper():
    from gmlx.load.loader import model_is_moe

    assert model_is_moe(_fake_model(MOE_PROJ_3D, vlm=True))
    assert not model_is_moe(_fake_model(DENSE_PROJ, vlm=True))


def test_model_is_moe_no_layers_is_false():
    from gmlx.load.loader import model_is_moe

    assert not model_is_moe(SimpleNamespace())


# -- the MoE rule overrides the family default ---------------------------

def test_moe_target_caps_at_one_over_any_family_default():
    """Routed experts cap at 1 whatever the family says: the gemma assistant
    (2, earned on the dense 31B) and dense qwen (uncapped) both drop."""
    moe = _fake_model(MOE_PROJ_3D)
    assert _stamped("gemma4_assistant", target=moe) == (1, 0)
    assert _stamped("qwen3_5", target=moe) == (1, 0)
    # an arch with no table row inherits it too, instead of the fallback 2
    assert _stamped("brand_new_moe_arch", target=moe) == (1, 0)


def test_dense_target_keeps_the_family_default():
    dense = _fake_model(DENSE_PROJ)
    assert _stamped("gemma4_assistant", target=dense) == (2, 0)
    assert _stamped("qwen3_5", target=dense) == (0, 0)
    assert _stamped("brand_new_arch", target=dense) == (2, 0)


def test_env_override_still_beats_the_moe_rule(monkeypatch):
    """The MoE cap is a default, not a limit -- a measurement run can ungate
    it. (Hard limits are the only thing env may not cross.)"""
    moe = _fake_model(MOE_PROJ_3D)
    assert _stamped("gemma4_assistant", env="0", monkeypatch=monkeypatch,
                    target=moe) == (0, 0)
    assert _stamped("gemma4_assistant", env="3", monkeypatch=monkeypatch,
                    target=moe) == (3, 0)


def test_moe_rule_does_not_unclamp_a_hard_limit():
    """deepseek4/hy3 are MoE and already B=1-only; the rule must not disturb
    the limit that keeps their drafters from raising."""
    moe = _fake_model(MOE_PROJ_3D)
    assert _stamped("deepseek_v4", target=moe) == (1, 1)
    assert _stamped("hy_v3", target=moe) == (1, 1)


def test_absent_target_falls_back_to_the_family_default():
    """Every in-tree stamp site passes target=, but the parameter is optional
    so a caller that cannot supply one still gets defined behavior."""
    assert _stamped("gemma4_assistant") == (2, 0)


def test_stamp_family_defaults():
    # dense qwen nextn: measured a win through c4, so uncapped
    assert _stamped("qwen3_5") == (0, 0)
    assert _stamped("qwen3_5_text") == (0, 0)
    # MoE nextn loses at B=2 (0.78x aggregate, d14k) -> speculate at B=1 only
    assert _stamped("qwen3_5_moe") == (1, 0)
    assert _stamped("qwen3_5_moe_text") == (1, 0)
    # the gemma assistant wins c1/c2 on the dense 31B, knees at B>=3
    assert _stamped("gemma4_assistant") == (2, 0)
    # B=1-only drafters: cap AND hard limit
    assert _stamped("hy_v3") == (1, 1)
    assert _stamped("deepseek_v4") == (1, 1)


def test_stamp_unknown_arch_is_conservative():
    """An unmapped family gets the safe cap, not uncapped: two of five known
    drafters are B=1-only and two of three measured families needed a cap."""
    assert _stamped("brand_new_arch") == (2, 0)


def test_stamp_env_override(monkeypatch):
    assert _stamped("qwen3_5_moe", env="4", monkeypatch=monkeypatch) == (4, 0)
    assert _stamped("qwen3_5_moe", env="0", monkeypatch=monkeypatch) == (0, 0)
    # empty = the config declared nothing, keep the family default
    assert _stamped("qwen3_5_moe", env="", monkeypatch=monkeypatch) == (1, 0)


def test_stamp_env_cannot_cross_hard_limit(monkeypatch):
    """Crossing a B=1-only drafter's limit is a guaranteed exception, not a
    perf trade, so config/env may not do it."""
    assert _stamped("deepseek_v4", env="4", monkeypatch=monkeypatch) == (1, 1)
    assert _stamped("hy_v3", env="0", monkeypatch=monkeypatch) == (1, 1)


# -- resolver ------------------------------------------------------------

def test_cap_from_drafter_attr():
    assert _mtp_width_cap(_StrictDrafter(cap=2)) == 2
    assert _mtp_width_cap(_StrictDrafter(cap=0)) == 0
    assert _mtp_width_cap(SimpleNamespace()) == 0


def test_env_overrides_attr(monkeypatch):
    monkeypatch.setenv("GMLX_MTP_WIDTH_CAP", "3")
    assert _mtp_width_cap(_StrictDrafter(cap=2)) == 3
    monkeypatch.setenv("GMLX_MTP_WIDTH_CAP", "0")
    assert _mtp_width_cap(_StrictDrafter(cap=2)) == 0


def test_bad_env_falls_back_to_attr(monkeypatch):
    monkeypatch.setenv("GMLX_MTP_WIDTH_CAP", "wide")
    assert _mtp_width_cap(_StrictDrafter(cap=2)) == 2


def test_hard_limit_clamps_uncapped(monkeypatch):
    # 0 means uncapped, so a limit must override it -- a naive min() would
    # keep 0 and reinstate the crash the limit exists to prevent.
    monkeypatch.setenv("GMLX_MTP_WIDTH_CAP", "0")
    assert _mtp_width_cap(_StrictDrafter(cap=1, limit=1)) == 1
    monkeypatch.setenv("GMLX_MTP_WIDTH_CAP", "4")
    assert _mtp_width_cap(_StrictDrafter(cap=1, limit=1)) == 1


def test_no_limit_leaves_cap_alone(monkeypatch):
    # The limit >= 1 guard: without it, "cap exceeds limit 0" would clamp a
    # deliberately capped model back to uncapped.
    monkeypatch.setenv("GMLX_MTP_WIDTH_CAP", "3")
    assert _mtp_width_cap(_StrictDrafter(cap=2, limit=0)) == 3
    assert _mtp_width_cap(_StrictDrafter(cap=2, limit=0)) == 3


def test_memo_does_not_leak_clamp_across_drafters(monkeypatch):
    """The memo covers the raw-string parse only. Memoizing the clamped result
    (as the process-global pacing resolver does) would let a B=1-only model's
    clamp cap an unrelated model on the next lookup."""
    monkeypatch.setenv("GMLX_MTP_WIDTH_CAP", "0")
    assert _mtp_width_cap(_StrictDrafter(cap=1, limit=1)) == 1
    assert _mtp_width_cap(_StrictDrafter(cap=0, limit=0)) == 0


# -- entry gating --------------------------------------------------------

def test_entry_gated_never_drafts():
    d = _StrictDrafter(cap=2)
    out, lm, _ = _drive(d, B=3, max_tokens=3)
    assert d.forward_calls == []
    assert d.shared_kv_calls == 0        # drafter view never armed
    # reset still runs (cheap, and keeps drafter-release reasoning off the
    # critical path) but without a batched left_padding
    assert d.reset_calls == [None]
    # plain-decode cadence: one token per row per round
    assert all(meta == {"round_pos": 0, "round_len": 1} for _, meta in out)
    assert lm.calls and all(shape[1] == 1 for shape in lm.calls)


def test_entry_gated_streams_expected_tokens():
    d = _StrictDrafter(cap=2)
    out, _, _ = _drive(d, B=3, max_tokens=3)
    # _FakeLM maps row r -> token r+3
    assert out[0][0] == [3, 4, 5]


def test_boundary_is_inclusive():
    """B == cap still speculates; only B > cap gates."""
    d = _StrictDrafter(cap=2)
    with pytest.raises(AssertionError, match="draft_block"):
        _drive(d, B=2, max_tokens=3)


def test_b_equals_cap_one_speculates():
    d = _StrictDrafter(cap=1)
    with pytest.raises(AssertionError, match="draft_block"):
        _drive(d, B=1, max_tokens=3)


class _B1OnlyDrafter(_StrictDrafter):
    """reset refuses any left_padding list, even the trivial [0] -- the
    dsv4/muse B=1-only contract."""

    def reset(self, model, left_padding=None):
        if left_padding is not None:
            raise NotImplementedError("B=1 only")
        return super().reset(model, left_padding=left_padding)


def test_b1_only_drafter_arms_bare_at_width_one():
    """A width-1 batch (a preempted scalar rebuilds into the batch loop at
    B=1) must arm a B=1-only drafter with a bare reset, not crash on the
    trivial [0] padding list."""
    d = _B1OnlyDrafter(cap=1, limit=1)
    with pytest.raises(AssertionError, match="draft_block"):
        _drive(d, B=1, max_tokens=3)
    assert d.reset_calls == [None]


def test_env_zero_ungates_a_capped_drafter(monkeypatch):
    monkeypatch.setenv("GMLX_MTP_WIDTH_CAP", "0")
    d = _StrictDrafter(cap=2)
    with pytest.raises(AssertionError, match="draft_block"):
        _drive(d, B=3, max_tokens=3)


def test_accept_stats_untouched_while_gated():
    """0-accept plain rounds must not dilute the reported acceptance rate."""
    d = _StrictDrafter(cap=2)
    _drive(d, B=3, max_tokens=3)
    assert d.accept_lens == []
    assert d.draft_lens == []


def test_gated_honors_sampler():
    seen = {}

    def sampler(logprobs):
        seen["shape"] = logprobs.shape
        # normalized logprobs, not raw logits
        seen["max"] = float(mx.max(logprobs).item())
        return mx.argmax(logprobs, axis=-1)

    d = _StrictDrafter(cap=2)
    out, _, _ = _drive(d, B=3, max_tokens=2, sampler=sampler)
    assert seen["shape"] == (3, VOCAB)
    assert seen["max"] <= 0.0
    assert out[0][0] == [3, 4, 5]


# -- trip on injection ---------------------------------------------------

def _queue_injection(model, lm, *, rows=1, offset=9):
    model._generator_injections = [{
        "uids": ["u"] * rows,
        "prompt_cache": [_FakeCache(width=rows, offset=offset)],
        "hidden": mx.zeros((rows, 1, 8)),
        "prompt_tokens": mx.zeros((rows, 4), dtype=mx.int32),
        "first_tokens": mx.array([7] * rows, dtype=mx.int32),
        "first_tokens_list": [7] * rows,
        "shared_kv_states": {
            "full": (mx.zeros((rows, 2, 4, 4)), mx.zeros((rows, 2, 4, 4)))},
    }]


def test_trip_on_injection_latches_and_keeps_streaming():
    d = _StrictDrafter(cap=3)
    model = SimpleNamespace()
    lm = _FakeLM()
    cache = _FakeCache(width=3)
    # B=3 == cap, so this batch is born SPECULATING (the seed arms the drafter)
    # and only the queued admission pushes it to 4 > 3.
    _queue_injection(model, lm)
    out, _, _ = _drive(d, B=3, max_tokens=4, model=model, lm=lm,
                       prompt_cache=[cache])
    assert d.shared_kv_calls == 1         # armed at entry...
    assert d.forward_calls == []          # ...then never drafted or injected
    assert cache.extend_calls == 1        # target cache still widened
    assert out and all(meta["round_len"] == 1 for _, meta in out)
    # injected row streams alongside the originals
    assert len(out[0][0]) == 4


def test_trip_widens_rope_deltas():
    """The gated plain step is the same lm(x, cache) call verify makes, so
    qwen-VL targets still need their cached mrope deltas at the live width."""
    d = _StrictDrafter(cap=3)
    model = SimpleNamespace()
    lm = _FakeLM()
    lm._rope_deltas = mx.zeros((3, 1), dtype=mx.int32)
    _queue_injection(model, lm)
    _drive(d, B=3, max_tokens=2, model=model, lm=lm,
           prompt_cache=[_FakeCache(width=3)])
    assert lm._rope_deltas.shape[0] == 4


def test_injection_bookkeeping_runs_while_gated():
    d = _StrictDrafter(cap=2)
    model = SimpleNamespace()
    lm = _FakeLM()
    _queue_injection(model, lm, rows=2)
    out, _, _ = _drive(d, B=3, max_tokens=2, model=model, lm=lm,
                       prompt_cache=[_FakeCache(width=3)])
    assert len(out[0][0]) == 5           # 3 original + 2 injected rows
    assert model._generator_injections == []


def test_multi_entry_injection_sums_for_the_trip():
    """Two queued singletons on a width-2 cap-2 batch make width 4: the trip
    predicate must sum the queue, not read the first entry."""
    d = _StrictDrafter(cap=2)
    model = SimpleNamespace()
    lm = _FakeLM()
    _queue_injection(model, lm)
    model._generator_injections.append(dict(model._generator_injections[0]))
    out, _, _ = _drive(d, B=2, max_tokens=2, model=model, lm=lm,
                       prompt_cache=[_FakeCache(width=2)])
    assert d.forward_calls == []
    assert len(out[0][0]) == 4


def test_no_flip_back_when_width_drops():
    """Rows finishing back under the cap must not re-arm the drafter: the
    latch holds for the generator's life."""
    d = _StrictDrafter(cap=2)
    model = SimpleNamespace()
    lm = _FakeLM()
    _queue_injection(model, lm)
    # max_tokens 4 -> rows retire at different times as emitted counts differ
    out, _, _ = _drive(d, B=3, max_tokens=4, model=model, lm=lm,
                       prompt_cache=[_FakeCache(width=3)])
    assert d.forward_calls == []
    assert len(d.reset_calls) == 1


def test_injected_row_budget_outlives_host_scalar():
    """The caller's max_tokens is max() over the rows present at generator
    start; an injected row carries its own budget in the entry and must
    stream to it, not truncate at the host batch's frozen scalar."""
    d = _StrictDrafter(cap=2)
    model = SimpleNamespace()
    lm = _EchoLM()
    _queue_injection(model, lm)
    model._generator_injections[0]["max_tokens"] = [6]
    out, _, _ = _drive(d, B=3, max_tokens=2, model=model, lm=lm,
                       prompt_cache=[_FakeCache(width=3)])
    assert d.forward_calls == []
    # round 1: hosts (b=[1,2,3] -> 2,3,4) finish at the scalar; the injected
    # row (first token 7 -> 8) keeps its own stream for four more rounds
    assert out[0][0] == [2, 3, 4, 8]
    assert [toks[3] for toks, _ in out[1:]] == [9, 10, 11, 12]
    # hosts stay finished (None slots) while the injected row streams
    assert all(toks[:3] == [None] * 3 for toks, _ in out[1:])


def test_round_log_notes_injected_budget(monkeypatch, tmp_path):
    """The session header logs the budget at generator start; a drain that
    adopts rows with their own budgets must leave a corrected trace."""
    log = tmp_path / "rounds.tsv"
    monkeypatch.setenv("GMLX_ROUND_LOG", str(log))
    d = _StrictDrafter(cap=2)
    model = SimpleNamespace()
    lm = _FakeLM()
    _queue_injection(model, lm)
    model._generator_injections[0]["max_tokens"] = [6]
    _drive(d, B=3, max_tokens=2, model=model, lm=lm,
           prompt_cache=[_FakeCache(width=3)])
    text = log.read_text()
    assert "# session kv=8 maxtok=2" in text
    assert "# inject rows=1 maxtok=6" in text


def test_injection_without_budgets_inherits_host_scalar():
    """An entry without per-row budgets (older producer, test fake) keeps the
    pre-existing behavior: the injected row stops at the host scalar."""
    d = _StrictDrafter(cap=2)
    model = SimpleNamespace()
    lm = _FakeLM()
    _queue_injection(model, lm)
    out, _, _ = _drive(d, B=3, max_tokens=3, model=model, lm=lm,
                       prompt_cache=[_FakeCache(width=3)])
    assert len(out) == 2
    assert all(len(toks) == 4 for toks, _ in out)


def test_env_lowered_mid_flight_trips(monkeypatch):
    """Per-round env read: a live server can be re-gated for an A/B."""
    d = _StrictDrafter(cap=0)
    model = SimpleNamespace()
    lm = _FakeLM()
    prompt_cache = [_FakeCache(width=3)]
    gen = _owned_decode_rounds_batch(
        model, d, lm, prompt_cache,
        hidden=mx.zeros((3, 1, 8)), b=[1, 2, 3],
        shared_kv={"full": (mx.zeros((3, 2, 4, 4)), mx.zeros((3, 2, 4, 4)))},
        seed_tokens=None, emitted=[1, 1, 1], max_tokens=6, sampler=None,
        draft_block_size=None)
    # ungated at entry: the strict drafter would raise on the first round, so
    # gate before pulling anything
    monkeypatch.setenv("GMLX_MTP_WIDTH_CAP", "2")
    spec._width_cap_memo = ("", None)
    toks, meta = next(gen)
    gen.close()
    assert meta == {"round_pos": 0, "round_len": 1}
    assert d.forward_calls == []


# -- stress: width transitions in both directions ------------------------

def test_width_transition_stress_both_directions():
    """Generator 1 speculates at B<=cap, trips on admission, keeps decoding as
    rows drain; generator 2 (fresh call, narrow) speculates again -- the
    re-evaluation point is generator start, not width recovery."""
    model = SimpleNamespace()
    lm = _FakeLM()
    d = _StrictDrafter(cap=2)
    _queue_injection(model, lm, rows=2)
    out, _, _ = _drive(d, B=2, max_tokens=5, model=model, lm=lm,
                       prompt_cache=[_FakeCache(width=2)])
    assert d.forward_calls == []
    assert len(out[0][0]) == 4

    # fresh generator, width under the cap -> speculation is offered again
    d2 = _StrictDrafter(cap=2)
    with pytest.raises(AssertionError, match="draft_block"):
        _drive(d2, B=2, max_tokens=3)


def test_all_finished_adoption_stays_gated():
    """When every row finishes in the same round and an injection is already
    queued, the loop drops the finished rows and adopts the newcomers as the
    batch. That branch touches drafter/hidden/shared-KV state a gated round
    never built, so it has to stay behind the gate too -- and the adopted
    batch inherits the latch rather than re-arming."""
    d = _StrictDrafter(cap=1)
    model = SimpleNamespace()
    lm = _FakeLM()
    cache = _FakeCache(width=2)
    gen = _owned_decode_rounds_batch(
        model, d, lm, [cache],
        hidden=mx.zeros((2, 1, 8)), b=[1, 2],
        shared_kv={"full": (mx.zeros((2, 2, 4, 4)), mx.zeros((2, 2, 4, 4)))},
        seed_tokens=None, emitted=[1, 1], max_tokens=2, sampler=None,
        draft_block_size=None)

    # Round 1 takes both rows to the budget. The yield is the generator's only
    # suspension point, so queueing here is exactly the race the branch
    # exists for: the injection lands before the all-finished check runs.
    first, meta = next(gen)
    assert meta == {"round_pos": 0, "round_len": 1}
    _queue_injection(model, lm)

    adopted, meta = next(gen)          # finished rows dropped, newcomer adopted
    assert meta == {"round_pos": 0, "round_len": 1}
    # the retired rows go quiet and only the adopted row emits
    assert adopted[:2] == [None, None] and adopted[2] is not None
    assert d.forward_calls == []       # no re-arm on the adopted batch
    assert d.shared_kv_calls == 0
    assert model._generator_injections == []
    gen.close()


def test_gated_generator_exhausts_rather_than_stranding_rows():
    """mlx-vlm finishes every unfinished row with finish_reason=length when the
    rounds generator exits, so a gated batch must run to the budget itself."""
    d = _StrictDrafter(cap=1)
    out, _, _ = _drive(d, B=2, max_tokens=4)
    # emitted starts at 1, so 3 more rounds reach the budget
    assert len(out) == 3


# -- gated rounds are double-buffered -------------------------------------

def test_gated_stream_survives_a_row_retiring_mid_flight():
    """The gated round dispatches the NEXT forward before reading this round's
    tokens, so when a row finishes the already-dispatched lookahead is one row
    too wide. It gets sliced by the same keep_mx as the cache; if that slice
    were missing or mis-ordered, a surviving row would continue another row's
    stream (or the forward would hit a width mismatch)."""
    lm = _EchoLM()
    out, _, cache = _drive(
        _StrictDrafter(cap=2), B=3, max_tokens=4, lm=lm,
        stop_check=lambda orig, tok: orig == 1)

    emitted = [toks for toks, _ in out]
    # rows start at [1, 2, 3]; each row's stream is its own +1 chain, and row 1
    # is stopped after its first token
    assert emitted == [
        [2, 3, 4],
        [3, None, 5],
        [4, None, 6],
    ]
    assert cache[0].width == 2, "cache should have been filtered to survivors"


def test_gated_rounds_dispatch_one_forward_ahead():
    """Round 1 primes and dispatches a lookahead (two forwards); every later
    round costs exactly one, which is where the overlap comes from. The width
    drop shows the lookahead following the filtered batch."""
    lm = _EchoLM()
    _drive(_StrictDrafter(cap=2), B=3, max_tokens=4, lm=lm,
           stop_check=lambda orig, tok: orig == 1)
    assert lm.widths == [3, 3, 2, 2]


def test_gated_stream_matches_a_single_buffered_reference():
    """Double buffering must not perturb the token stream: with no retirement
    the emitted rows are the plain +1 chains from their seeds."""
    lm = _EchoLM()
    out, _, _ = _drive(_StrictDrafter(cap=2), B=3, max_tokens=4, lm=lm)
    assert [toks for toks, _ in out] == [
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
    ]


# -- LoRA rows on the gated round -----------------------------------------

class _RowsLM(_FakeLM):
    """Target that records the LoRA row vector published for each forward,
    the way the rows-mode wrappers read it (raises when nothing is
    published)."""

    def __init__(self):
        super().__init__()
        self.rows = []

    def __call__(self, x, cache=None, **kw):
        from gmlx import lora_rows
        self.rows.append(lora_rows.row_scales(0).tolist())
        return super().__call__(x, cache=cache, **kw)


def test_gated_round_publishes_lora_rows():
    """Width-capped batches run only the gated (plain) target step; it must
    publish the batch's LoRA row vector like the verify rounds do, else every
    concurrent adapter stream fails in rows mode (ds4 + adapters, 2026-08-26).
    """
    from gmlx import lora_rows

    lora_rows.configure("rows", 1)
    try:
        lora_rows.register_uid("u0", (1.0,))
        lora_rows.register_uid("u2", (1.0,))
        model = SimpleNamespace(_kq_row_uids=["u0", "u1", "u2"])
        lm = _RowsLM()
        out, _, _ = _drive(_StrictDrafter(cap=2), B=3, max_tokens=3,
                           model=model, lm=lm)
        assert out[0][0] == [3, 4, 5]
        assert lm.rows and all(r == [1.0, 0.0, 1.0] for r in lm.rows)
        with pytest.raises(lora_rows.LoraRowsError):
            lora_rows.row_scales(0)   # cleared after every forward
    finally:
        lora_rows.configure("static", 1)
