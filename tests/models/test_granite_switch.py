"""Granite Switch (GGUF arch ``graniteswitch``): the router slot rule, the
switched LoRA delta, substitute embeddings, batching, the KV layer policy
around the router cache, and the template's adapter placement.

References: llama.cpp a37cc4610 ``src/models/granite-switch.cpp`` for the
forward, IBM's granite-switch 43ccd29 ``src/granite_switch/hf/switch/multi.py``
for the slot rule, and the chat template of ibm-granite/granite-switch-4.1-8b at
0d1625d (``tests/fixtures/granite_switch_template.jinja``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx
from mlx.utils import tree_flatten

from gmlx.models.granite_switch.model import (
    Model,
    ModelArgs,
    SlotRoute,
    SwitchLoRA,
    pack_lora,
    SwitchRouterBatchCache,
    SwitchRouterCache,
)

pytestmark = pytest.mark.usefixtures("_cpu")

VOCAB = 64
GUARD, DECIDE = 60, 61          # the two control tokens
SUBS = {GUARD: 5, DECIDE: 9}    # the tokens whose rows they take
SLOT = {GUARD: 1, DECIDE: 2}


@pytest.fixture()
def _cpu():
    # The CPU GatherMM is float32-only; the model computes deltas there at
    # float32, so these tests pin the CPU for exact comparisons.
    with mx.stream(mx.cpu):
        prev = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            yield
        finally:
            mx.set_default_device(prev)


def _args(**over) -> ModelArgs:
    kw = dict(
        model_type="granite_switch", hidden_size=64, num_hidden_layers=3,
        intermediate_size=96, num_attention_heads=4, num_key_value_heads=2,
        rms_norm_eps=1e-6, vocab_size=VOCAB, logits_scaling=4.0,
        attention_multiplier=0.25, embedding_multiplier=2.0,
        residual_multiplier=0.5, num_adapters=2, max_lora_rank=4,
        adapter_token_ids=[GUARD, DECIDE],
        adapter_substitute_ids=[SUBS[GUARD], SUBS[DECIDE]])
    kw.update(over)
    return ModelArgs(**kw)


def _wire_lora(args, rng) -> dict:
    """One layer's adapter tensors as the GGUF lays them out: ``(S, r, in)``
    and ``(S, out, r)`` per target, slot 0 zero."""
    d, ff, r = args.hidden_size, args.intermediate_size, args.max_lora_rank
    kv = args.num_key_value_heads * (d // args.num_attention_heads)
    dims = {"q": (d, d), "k": (d, kv), "v": (d, kv), "o": (d, d),
            "gate": (d, ff), "up": (d, ff), "down": (ff, d)}
    out = {}
    for t, (n_in, n_out) in dims.items():
        for h, shape in (("a", (r, n_in)), ("b", (n_out, r))):
            w = rng.standard_normal((args.num_adapters + 1, *shape)) * 0.2
            w[0] = 0.0
            out[f"{t}_{h}"] = w.astype(np.float32)
    return out


def _model(seed=0, **over) -> Model:
    """Random weights; the adapter stacks go through sanitize from the wire
    layout, as on load."""
    model = Model(_args(**over))
    rng = np.random.default_rng(seed)
    weights = {}
    for k, v in tree_flatten(model.parameters()):
        if ".switch_lora." in k:
            continue
        w = rng.standard_normal(v.shape).astype(np.float32) * 0.2
        if k.endswith("norm.weight"):
            w = 1.0 + w * 0.1
        weights[k] = mx.array(w)
    for i in range(model.args.num_hidden_layers):
        for leaf, w in _wire_lora(model.args, rng).items():
            weights[f"model.layers.{i}.switch_lora.{leaf}"] = mx.array(w)
    model.load_weights(list(model.sanitize(weights).items()))
    mx.eval(model.parameters())
    return model


def _ref_slots(ids) -> list[int]:
    """IBM's MultiSwitch rule: the slot of the latest control token at or
    before each position, or 0 with none."""
    slot, out = 0, []
    for t in ids:
        slot = SLOT.get(t, slot)
        out.append(slot)
    return out


def _np(a):
    return np.array(a.astype(mx.float32))


# Router slots

@pytest.mark.parametrize("ids", [
    [1, 2, 3, 4],                              # no control token
    [1, GUARD, 2, 3],                          # one
    [1, GUARD, 2, DECIDE, 3],                  # multi.py's [T,A,T,B,T]
    [DECIDE, 1, GUARD, 2, GUARD, 3],           # back to the first adapter
    [GUARD, DECIDE, DECIDE, 4],                # a repeat changes nothing
], ids=["none", "one", "switch", "switch-back", "repeat"])
def test_slots_follow_ibms_rule(ids):
    inner = _model().model
    got = inner.slots(mx.array([ids]), None)
    assert got.dtype == mx.uint32
    assert got[0].tolist() == _ref_slots(ids)


def test_slots_match_multi_py_example():
    assert _ref_slots([1, GUARD, 2, DECIDE, 3]) == [0, 1, 1, 2, 2]


def test_slots_carry_across_a_cached_prefix():
    """A control token in an earlier chunk, or in an extracted and merged
    prefix, still sets the slot of later tokens."""
    inner = _model().model
    ids = [1, GUARD, 2, 3, DECIDE, 4, GUARD, 5]
    for cut in range(1, len(ids)):
        router = SwitchRouterCache()
        a = inner.slots(mx.array([ids[:cut]]), router)
        b = inner.slots(mx.array([ids[cut:]]), router)
        assert a[0].tolist() + b[0].tolist() == _ref_slots(ids), cut

    rows = [SwitchRouterCache(), SwitchRouterCache()]
    inner.slots(mx.array([[1, GUARD, 2]]), rows[0])
    inner.slots(mx.array([[3, DECIDE, 5, 6]]), rows[1])
    batch = SwitchRouterCache.merge(rows)
    assert type(batch) is SwitchRouterBatchCache
    assert inner.slots(mx.array([[7], [7]]), batch)[:, 0].tolist() == [1, 2]
    step = inner.slots(mx.array([[DECIDE], [GUARD]]), batch)
    assert step[:, 0].tolist() == [2, 1]
    back = batch.extract(0)
    assert type(back) is SwitchRouterCache
    assert inner.slots(mx.array([[8]]), back)[0].tolist() == [2]


def test_left_padded_rows_match_unpadded_rows():
    """Left padding adds no control lanes, so a padded row's slots and
    logits match the row run alone, on prefill and decode."""
    from gmlx.cache.kv_cache import BatchKVCache

    model = _model(seed=1)
    pad = 3
    short = [1, GUARD, 2, 3]
    long = [4, 5, 6, DECIDE, 7, 8, 9]
    cache = [BatchKVCache([pad, 0]) for _ in model.layers]
    cache.append(SwitchRouterCache().to_batch([pad, 0]))
    out = model(mx.array([[0] * pad + short, long]), cache=cache)
    step = model(mx.array([[DECIDE], [3]]), cache=cache)

    alone = model.make_cache()
    ref = model(mx.array([short]), cache=alone)
    ref_step = model(mx.array([[DECIDE]]), cache=alone)
    np.testing.assert_allclose(_np(out[0, pad:]), _np(ref[0]), atol=1e-4)
    np.testing.assert_allclose(_np(step[0]), _np(ref_step[0]), atol=1e-4)


def test_router_stays_float32_under_float16_activations():
    model = _model(seed=2)
    model.set_dtype(mx.float16)
    cache = model.make_cache()
    model(mx.array([[1, GUARD, 2, DECIDE, 3]]), cache=cache)
    assert cache[0].keys.dtype == mx.float16
    router = cache[-1]
    assert router.keys.dtype == mx.float32 and router.values.dtype == mx.float32
    keys = router.keys[0, 0, : router.offset, 0].tolist()
    values = router.values[0, 0, : router.offset, 0].tolist()
    assert keys == [0.0, 1.0, 0.0, 1.0, 0.0]
    assert values == [0.0, 1.0, 0.0, 2.0, 0.0]


# Forward

@pytest.mark.parametrize("count_live", [False, True], ids=["masked", "counted"])
@pytest.mark.parametrize("shape", [(1, 3), (2, 5), (3, 11)],
                         ids=["per-row", "per-row-batch", "slot-sorted"])
def test_packed_deltas_match_the_wire_math(shape, count_live):
    """Each projection adds B_s (A_s x) of its targets for the token's slot
    s, from the wire stacks, with slot 0 adding nothing."""
    args = _args()
    rng = np.random.default_rng(3)
    wire = _wire_lora(args, rng)
    slots = rng.integers(0, 3, shape).astype(np.uint32)
    slots.flat[0] = 1
    slots.flat[1] = 2
    route = SlotRoute(mx.array(slots), count_live=count_live)
    assert route.uniform is None
    assert (route.order is not None) == (slots.size >= SlotRoute.SORT_MIN_ROWS)
    assert route.n_live == (int((slots > 0).sum()) if count_live else None)
    _check_deltas(args, wire, slots, route, rng)


@pytest.mark.parametrize("slots", [
    [[1] * 20],
    [[0] * 7 + [2] * 13, [0] * 20],
    [[0, 0, 1], [1, 0, 1]],
], ids=["all-live", "with-plain-rows", "few-rows"])
def test_one_slot_chunks_take_the_dense_delta(slots):
    """When every adapted row shares a slot, the delta is two dense matmuls
    on that slot's pair, masked on the plain rows."""
    args = _args()
    rng = np.random.default_rng(9)
    wire = _wire_lora(args, rng)
    slots = np.array(slots, dtype=np.uint32)
    route = SlotRoute(mx.array(slots), count_live=True)
    assert route.uniform == int(slots.max()) and route.order is None
    _check_deltas(args, wire, slots, route, rng)


def _check_deltas(args, wire, slots, route, rng):
    import mlx.nn as nn

    lora = SwitchLoRA(args)
    lora.load_weights(list(pack_lora(
        {k: mx.array(v) for k, v in wire.items()}).items()))

    def oracle(targets, x):
        parts = [np.einsum("bloi,bli->blo", wire[f"{t}_b"][slots],
                           np.einsum("blri,bli->blr", wire[f"{t}_a"][slots], x))
                 for t in targets]
        return np.concatenate(parts, axis=-1)

    for p, targets in (("qkv", ("q", "k", "v")), ("o", ("o",)),
                       ("down", ("down",))):
        n_in = wire[f"{targets[0]}_a"].shape[-1]
        x = rng.standard_normal((*slots.shape, n_in)).astype(np.float32)
        zero = nn.Linear(n_in, oracle(targets, x).shape[-1], bias=False)
        zero.weight = mx.zeros_like(zero.weight)
        got = _np(lora.project(p, zero, mx.array(x), route))
        np.testing.assert_allclose(got, oracle(targets, x), rtol=1e-4,
                                   atol=1e-4, err_msg=p)


def _routes_seen(model, monkeypatch) -> list:
    seen = []
    project = SwitchLoRA.project

    def spy(self, p, base, x, route):
        seen.append(route)
        return project(self, p, base, x, route)

    monkeypatch.setattr(SwitchLoRA, "project", spy)
    return seen


def test_a_wide_plain_chunk_skips_the_adapters(monkeypatch):
    model = _model()
    seen = _routes_seen(model, monkeypatch)
    n = SwitchLoRA.EPILOGUE_MAX_ROWS + 8
    mx.eval(model(mx.array([list(range(10, 10 + n))]), cache=model.make_cache()))
    assert seen and all(r is None for r in seen)


def test_a_wide_adapted_chunk_counts_its_live_rows(monkeypatch):
    model = _model()
    seen = _routes_seen(model, monkeypatch)
    ids = [11] * 30 + [GUARD] + [12] * 9
    want = _np(model(mx.array([ids])))
    mx.eval(model(mx.array([ids]), cache=model.make_cache()))
    assert {(r.n_live, r.uniform) for r in seen} == {(10, 1)}
    monkeypatch.setattr(SlotRoute, "__init__", _uncounted(SlotRoute.__init__))
    np.testing.assert_allclose(_np(model(mx.array([ids]))), want,
                               rtol=1e-5, atol=1e-5)


def _uncounted(init):
    def wrapped(self, slots, count_live=False):
        init(self, slots, count_live=False)
    return wrapped


def test_a_compiled_forward_falls_back_to_masked_rows():
    model = _model()
    ids = mx.array([[11] * 30 + [DECIDE] + [12] * 9])
    want = _np(model(ids))
    got = _np(mx.compile(lambda x: model(x))(ids))
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_no_route_runs_the_base_alone():
    import mlx.nn as nn

    base = nn.Linear(8, 4, bias=False)
    x = mx.ones((1, 2, 8))
    assert mx.array_equal(SwitchLoRA(_args()).project("o", base, x, None),
                          base(x))


def test_control_token_embeds_as_its_substitute():
    inner = _model().model
    ctrl = inner.embed(mx.array([[1, GUARD, DECIDE]]))
    subs = inner.embed(mx.array([[1, SUBS[GUARD], SUBS[DECIDE]]]))
    assert mx.array_equal(ctrl, subs)
    assert mx.array_equal(
        ctrl, inner.embed_tokens(mx.array([[1, 5, 9]])) * 2.0)


def test_the_adapter_starts_at_its_control_token():
    """Positions before the control token run the base weights; the token
    and everything after it run the adapter."""
    model = _model(seed=4)
    ids = [1, 2, GUARD, 3, 4]
    switched = _np(model(mx.array([ids])))
    plain = _np(model(mx.array([[1, 2, SUBS[GUARD], 3, 4]])))
    assert np.array_equal(switched[0, :2], plain[0, :2])
    assert np.abs(switched[0, 2:] - plain[0, 2:]).max() > 1e-2


def test_batched_rows_match_per_row_forwards():
    model = _model(seed=5)
    rows = [[1, GUARD, 2, 3], [4, 5, 6, 7], [DECIDE, 8, GUARD, 9]]
    batched = _np(model(mx.array(rows)))
    for i, r in enumerate(rows):
        np.testing.assert_allclose(batched[i], _np(model(mx.array([r]))[0]),
                                   rtol=0, atol=1e-5)


def test_slot_sorted_prefill_matches_per_row_products(monkeypatch):
    model = _model(seed=8)
    ids = mx.array([[1, 2, GUARD, 3, 4, DECIDE, 5, 6, 7, 8, GUARD, 9, 10,
                     11, 12, 13, 14, 15],
                    [3] * 18])
    sorted_out = _np(model(ids))
    monkeypatch.setattr(SlotRoute, "SORT_MIN_ROWS", 1 << 30)
    np.testing.assert_allclose(sorted_out, _np(model(ids)), atol=1e-5)


def test_cached_decode_matches_the_full_forward():
    model = _model(seed=6)
    ids = [1, 2, DECIDE, 3, 4, 5, 6]
    full = _np(model(mx.array([ids])))
    cache = model.make_cache()
    steps = [_np(model(mx.array([ids[:4]]), cache=cache))[0]]
    for t in ids[4:]:
        steps.append(_np(model(mx.array([[t]]), cache=cache))[0])
    np.testing.assert_allclose(np.concatenate(steps), full[0], atol=1e-4)


def test_tied_head_divides_by_the_logit_scale():
    model = _model(seed=7)
    ids = mx.array([[1, 2, 3]])
    h = model.model(ids)
    want = model.model.embed_tokens.as_linear(h) / 4.0
    assert mx.array_equal(model(ids), want)


def test_remap_covers_every_wire_tensor_onto_real_params():
    """Every tensor llama.cpp's graniteswitch loader creates maps onto a
    parameter, and every parameter is loaded."""
    from gmlx.load.remap import RemapDecision, parse_gguf_name

    args = _args()
    params = {k for k, _ in tree_flatten(Model(args).parameters())}
    names = ["token_embd.weight", "output_norm.weight"]
    per_layer = ["attn_norm.weight", "attn_qkv.weight", "attn_output.weight",
                 "ffn_norm.weight", "ffn_gate.weight", "ffn_up.weight",
                 "ffn_down.weight"]
    for t in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate",
              "ffn_up", "ffn_down"):
        per_layer += [f"{t}.lora_a", f"{t}.lora_b"]
    for i in range(args.num_hidden_layers):
        names += [f"blk.{i}.{t}" for t in per_layer]
    mapped = {}
    for name in names:
        dec = parse_gguf_name("graniteswitch", name)
        assert dec.kind == RemapDecision.KIND_MAP, (name, dec.reason)
        assert dec.transform == "passthrough", name
        mapped[dec.hf_name] = None
    shapes = {**dict(tree_flatten(Model(args).parameters()))}
    wire = _wire_lora(args, np.random.default_rng(0))
    weights = {}
    for k in mapped:
        leaf = k.rpartition(".switch_lora.")[2]
        weights[k] = (mx.array(wire[leaf]) if ".switch_lora." in k
                      else mx.zeros_like(shapes[k]))
    packed = Model(args).sanitize(weights)
    assert set(packed) == params
    for k, v in packed.items():
        assert v.shape == shapes[k].shape, k


def test_prefill_decay_prices_fused_attention_without_the_router(monkeypatch):
    import gmlx.gen.prefill_decay as pd

    monkeypatch.setattr(pd, "cpu_default_device", lambda: False)
    model = Model(_args(hidden_size=512))          # head dim 128
    profile = pd.resolve_score_profile(model, model.make_cache())
    assert profile is not None and profile.fused_min_step == 9


# KV policy with the router cache

def test_kv_policy_keeps_the_last_real_layer_plain():
    from gmlx.cache.kv_cache import kv_policy_flags
    from gmlx.cache.kv_policy import resolve_kv_quant_policy

    stack = _model().make_cache()
    assert kv_policy_flags(stack) == [True, True, False, False]
    policy = resolve_kv_quant_policy(stack, kv_bits=4, kv_group_size=64)
    assert [p.quantize for p in policy.per_layer] == [True, True, False, False]


def test_quantize_stack_converts_only_the_planned_layers():
    from gmlx.cache.kv_cache import KVCache, QuantizedKVCache
    from gmlx.cache.kv_policy import (arm_stack, hold_fp16,
                                      quantize_stack, resolve_kv_quant_policy)

    stack = _model().make_cache()
    policy = resolve_kv_quant_policy(stack, kv_bits=4, kv_group_size=64)
    quantize_stack(stack, policy)
    assert [type(c) for c in stack] == [
        QuantizedKVCache, QuantizedKVCache, KVCache, SwitchRouterCache]

    held = _model().make_cache()
    arm_stack(held, policy)
    assert not hasattr(held[2], "to_quantized")
    assert type(held[3]) is SwitchRouterCache
    router = SwitchRouterCache()
    assert hold_fp16(router) is router


def test_live_batch_build_keeps_the_last_real_layer_plain():
    from gmlx._vendor.mlx_vlm.generate.ar import _make_cache
    from gmlx.cache.kv_cache import BatchKVCache, BatchQuantizedKVCache

    caches = _make_cache(_model(), [0, 2], kv_bits=4, kv_group_size=64)
    assert [type(c) for c in caches] == [
        BatchQuantizedKVCache, BatchQuantizedKVCache, BatchKVCache,
        SwitchRouterBatchCache]


def test_exact_restore_alignment_keeps_the_last_real_layer_plain():
    from gmlx._vendor.mlx_vlm.apc import _align_exact_batch_caches_to_kv_policy
    from gmlx.cache.kv_cache import BatchKVCache, BatchQuantizedKVCache

    model = _model(hidden_size=256)     # head dim 64, one quant group
    caches = [BatchKVCache([0]) for _ in model.layers]
    caches.append(SwitchRouterBatchCache([0]))
    model(mx.array([[1, GUARD, 2]]), cache=caches)
    out = _align_exact_batch_caches_to_kv_policy(
        caches, {"bits": 4, "group_size": 64})
    assert [type(c) for c in out] == [
        BatchQuantizedKVCache, BatchQuantizedKVCache, BatchKVCache,
        SwitchRouterBatchCache]
    assert out[3] is caches[3]


def test_admission_prices_the_router_as_a_side_cache():
    from gmlx.serve.mem_preflight import LayerGeometry, stack_geometry

    model = _model()
    geo = stack_geometry(model, model.make_cache())
    assert len(geo) == 4
    assert geo[3] == LayerGeometry(False, None, 0.0, 8.0)
    assert all(g.attn for g in geo[:3])


def test_apc_takes_the_checkpoint_strategy():
    from gmlx._vendor.mlx_vlm.apc import model_apc_mode

    assert model_apc_mode(_model()) == "exact"


# Decisions through the systemone template

class _CharTokens:
    """One token per character below the role and control ids."""

    SPECIAL = {"<|start_of_role|>": 50, "<|end_of_role|>": 51,
               "<|end_of_text|>": 52, "<|decision|>": DECIDE}

    def render(self, text):
        return text

    def ids(self, text):
        import re

        parts = re.findall(r"<\|[a-z_-]+\|>|.", text, flags=re.S)
        return [self.SPECIAL.get(t, ord(t[0]) % 48) for t in parts]


def test_template_decisions_match_full_forwards():
    """The reader prefills the state turn on the base weights, then reads
    each question's tail through broadcast views of that prefix, router
    lanes included. Each answer must equal the label softmax of one full
    forward over the question's whole prompt."""
    from gmlx.systemone import ar_reader, letters
    from gmlx.systemone.model_template import TemplateFormat

    source = (Path(__file__).parents[1] / "fixtures"
              / "granite_switch_systemone.jinja").read_text()
    labels = [chr(65 + i) for i in range(26)]
    fmt = TemplateFormat(source, {"type": "granite-switch"}, labels, [DECIDE])
    tokens = _CharTokens()
    body = {"state": "a ticket",
            "questions": {"team": {"type": "choice", "instructions": "Who?",
                                   "criteria": {"pay": "Payments", "acct": None,
                                                "other": "Else"}},
                          "ok": {"type": "noul", "instructions": "Fine?"}}}
    schema, state = letters.parse(body), letters.state_text(body)
    ids, split = ar_reader.split_point(tokens, state, fmt)
    assert DECIDE not in ids
    assert tokens.ids(fmt.prefix_text(state))[split] == DECIDE
    model = _model()
    label_ids = [ord(c) % 48 for c in labels]
    reader = ar_reader.LetterReader(model, label_ids)
    out = ar_reader.run_to_end(ar_reader.decide_letters(
        reader, tokens, schema, state, style=fmt))
    assert out["diagnostics"]["readout"] == "template"
    assert out["diagnostics"]["prefix"]["tokens"] == split
    # The head's letter rows serve, with the logit scale applied after them.
    assert out["diagnostics"]["path"] == "rows"
    for q in schema["questions"]:
        opts = fmt.options(q)
        logits = model(mx.array([tokens.ids(fmt.prompt(state, q, opts))]))[0, -1]
        p = letters.softmax([float(logits[i]) for i in label_ids[:len(opts)]])
        a = out["answers"][q["id"]]
        if q["type"] == "noul":
            assert a["noul"] == pytest.approx(p[0], abs=1e-5)
        else:
            assert list(a["probabilities"].values()) == pytest.approx(p, abs=1e-5)


def test_a_logit_scale_the_forward_does_not_apply_reads_full_logits():
    from gmlx.systemone import ar_reader

    model = _model()
    model.args.logits_scaling = 8.0         # the forward still divides by 4
    reader = ar_reader.LetterReader(model, list(range(8)))
    assert reader.check() == "logits" and reader.step is None


def test_decision_prefixes_take_views_and_the_exact_tier():
    import types

    from gmlx.systemone import ar_reader
    from gmlx.systemone.prefixes import ApcPrefixes

    model = _model()
    n = model.args.num_hidden_layers + 1
    assert ar_reader._view_kinds(model.make_cache()) == ["kv"] * n
    manager = types.SimpleNamespace(block_size=16)
    assert ApcPrefixes(manager, 0, model.make_cache).tier == "exact"


# Template placement

_TEMPLATE = (Path(__file__).parents[1] / "fixtures"
             / "granite_switch_template.jinja").read_text()


def _render(messages, **kw) -> str:
    from transformers.utils.chat_template_utils import render_jinja_template

    out, _ = render_jinja_template(conversations=[messages],
                                   chat_template=_TEMPLATE,
                                   add_generation_prompt=True, **kw)
    return out[0]


def test_no_adapter_name_adds_no_control_token():
    text = _render([{"role": "user", "content": "<guardian> hi"}])
    assert "<|guardian-core|>" not in text and "<|decision|>" not in text


def test_guardian_replaces_the_first_char_of_its_invocation_text():
    text = _render([{"role": "user", "content": "<guardian> is this ok?"}],
                   adapter_name="guardian-core")
    assert ("<|start_of_role|>user<|end_of_role|><|guardian-core|>guardian>"
            " is this ok?<|end_of_text|>") in text


def test_decision_marks_the_last_user_message_holding_its_text():
    msgs = [{"role": "user", "content": "### DECISION\nold"},
            {"role": "user", "content": "State:\nx"},
            {"role": "user", "content": "### DECISION\nQ"}]
    text = _render(msgs, adapter_name="decision")
    assert text.count("<|decision|>") == 1
    assert "### DECISION\nold" in text
    assert text.endswith(
        "<|start_of_role|>user<|end_of_role|><|decision|> DECISION\nQ"
        "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>")


@pytest.mark.parametrize("name,token", [("guardian-core", "<|guardian-core|>"),
                                        ("decision", "<|decision|>")])
def test_without_invocation_text_the_token_opens_the_assistant_turn(name, token):
    text = _render([{"role": "user", "content": "hi"}], adapter_name=name)
    assert text.endswith(f"<|end_of_text|>\n{token}assistant<|end_of_role|>")


def test_gguf_tokenizer_places_the_control_ids(gguf_index):
    """On IBM's GGUF, the rendered control tokens encode as their ids, and
    the decision marker reads as the README's ids with ``###`` swapped."""
    from conftest import require_arch
    from gmlx.distill.tokens import tokenizer_from_gguf

    tok = tokenizer_from_gguf(require_arch(gguf_index, "graniteswitch"))
    ids = tok.apply_chat_template(
        [{"role": "user", "content": "State:\nx"},
         {"role": "user", "content": "### DECISION\nQ"}],
        adapter_name="decision", add_generation_prompt=True,
        tokenize=True, return_dict=False)
    i = ids.index(100353)
    assert ids[i:i + 3] == [100353, 43917, 25189]
    assert 100352 not in ids
    ids = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}], adapter_name="guardian-core",
        add_generation_prompt=True, tokenize=True, return_dict=False)
    assert ids[-4:] == [198, 100352, 78191, 100265]


def test_tool_calls_take_the_json_tools_parser():
    """The template's <tool_call> JSON round-trips through the stock
    json_tools parser."""
    from gmlx._vendor.mlx_vlm.tools.parsers.json_tools import parse_tool_call
    from gmlx._vendor.mlx_vlm.tools.registry import _infer_tool_parser

    assert _infer_tool_parser(_TEMPLATE) == "json_tools"
    call = {"name": "get_weather", "arguments": {"city": "Paris"}}
    text = _render([
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"type": "function", "function": call}]},
        {"role": "user", "content": "thanks"}])
    body = text.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0]
    assert parse_tool_call(body) == call
