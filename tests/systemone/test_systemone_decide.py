"""The /v1/systemone decision logic against a scripted fake read engine:
stages, chunks, samples, thoughts, prompt reuse and the diagnostics shape."""

from __future__ import annotations

import math
import random
import zlib

import pytest
from test_systemone_template import WordEnc

from gmlx.systemone.contract import jev_answers
from gmlx.systemone.decide import (
    GROUP_SEED_STRIDE,
    SAMPLE_SEED_STRIDE,
    decide,
    slot_distribution,
)
from gmlx.systemone.reads import (
    Cancelled,
    ReadResult,
    SampleRead,
    SlotRead,
    label_id_union,
)
from gmlx.systemone.schema import jev_schema
from gmlx.systemone.template import TemplateResolver, system_text

THOUGHT = [501, 502, 503]

# Key sets of the vLLM proxy's diagnostics (structured_server.py decide and
# decide_group at ab3de6edf2).
DECISION_KEYS = {"steps", "stages", "skipped", "chunks", "chunk_prompt", "sequential",
                 "conditioning", "thought", "samples", "timing", "prompt_tokens",
                 "questions", "engine"}
GROUP_KEYS = {"steps", "samples", "timing", "thought", "prompt_tokens", "questions",
              "engine"}
QUESTION_KEYS = {"pos", "entropy", "label_mass", "argmax_is_label"}
THOUGHT_KEYS = {"tokens", "closed", "ms", "text"}


class FakeCache:
    def __init__(self, ids):
        self.ids = list(ids)
        self.extends = []

    def extend(self, ids):
        self.extends.append(list(ids))
        self.ids = self.ids + list(ids)


class FakeEngine:
    """Reads answer from ``weights(seed, slot_index, n_labels)``, the
    unnormalized weights of a slot's own labels. Every other id in the read's
    label union gets ``other``, so a positive ``other`` spreads mass over the
    union as a constrained read does."""

    def __init__(self, weights=None, other=0.0, closed=True):
        self.weights = weights or (lambda seed, i, n: [50.0] + [1.0] * (n - 1))
        self.other = other
        self.closed = closed
        self.prefills, self.caches, self.reads, self.thinks = [], [], [], []

    def prefill(self, ids):
        self.prefills.append(list(ids))
        self.caches.append(FakeCache(ids))
        return self.caches[-1]

    def read(self, prompt, req):
        self.reads.append((list(prompt.ids), req))
        allowed = label_id_union(req.slots)
        samples = []
        for seed in req.seeds:
            slots = []
            for i, s in enumerate(req.slots):
                own = dict(zip(s.label_ids, self.weights(seed, i, len(s.label_ids))))
                raw = {a: own.get(a, self.other) for a in allowed}
                raw = {a: v for a, v in raw.items() if v > 0}
                z = sum(raw.values())
                top = {a: math.log(v / z) for a, v in raw.items()}
                slots.append(SlotRead(max(top, key=top.get), top))
            samples.append(SampleRead(seed, (), tuple(slots)))
        return ReadResult(tuple(samples), len(prompt.ids), {})

    def think(self, prompt_ids, budget, *, stop_id, canvas_width):
        self.thinks.append({"prompt": list(prompt_ids), "budget": budget,
                            "stop_id": stop_id, "canvas_width": canvas_width})
        return list(THOUGHT), {"closed": self.closed, "ms": 2.0}


class Chat:
    """``chat_ids(system_text, thinking)``: distinct ids per system text."""

    def __init__(self):
        self.calls = []

    def __call__(self, sys_text, thinking):
        self.calls.append((sys_text, thinking))
        return [2, 105, 900 + zlib.crc32(sys_text.encode()) % 1000,
                98 if thinking else 97, 106, 105]


def _run(body, *, engine=None, enc=None, seed=42, should_stop=None, constrained=True):
    schema = jev_schema({"state": "s", **body})
    enc = enc or WordEnc()
    engine = engine or FakeEngine()
    resolver = TemplateResolver(enc, 64)
    chat = Chat()
    out, rows = decide(schema, "s", engine=engine, resolver=resolver, chat_ids=chat,
                       seed=seed, constrained=constrained, canvas_len=64,
                       decode=enc.decode, should_stop=should_stop)
    return {"out": out, "rows": rows, "engine": engine, "enc": enc, "chat": chat,
            "resolver": resolver, "schema": schema}


def _qs(*ids, **per):
    return {"questions": {q: {"type": "noul", **per.get(q, {})} for q in ids}}


def _entropy(ps):
    return -sum(p * math.log(p) for p in ps if p > 0)


# slot_distribution


def test_slot_distribution_renormalizes_labels_and_takes_entropy_over_the_union():
    top = {10: math.log(0.4), 11: math.log(0.1), 20: math.log(0.3), 21: math.log(0.2)}
    d = slot_distribution(top, [10, 11])
    assert d["probs"] == pytest.approx([0.8, 0.2])
    assert d["label_mass"] == pytest.approx(0.5)
    assert d["entropy"] == pytest.approx(_entropy([0.4, 0.1, 0.3, 0.2]))
    assert d["argmax_is_label"] is True
    assert slot_distribution(top, [11, 21])["argmax_is_label"] is False


def test_slot_distribution_floors_a_label_missing_from_the_returned_set():
    top = {10: -0.1, 30: -3.0}
    d = slot_distribution(top, [10, 11])
    floor = -3.0 - 5.0
    want = [math.exp(-0.1), math.exp(floor)]
    assert d["probs"] == pytest.approx([w / sum(want) for w in want])


# one group


def test_a_single_group_reads_once_on_the_chat_prompt():
    r = _run(_qs("q1", "q2"))
    eng, enc, res, out = r["engine"], r["enc"], r["resolver"], r["out"]
    sys_text = system_text(r["schema"])
    assert r["chat"].calls == [(sys_text, False)]
    assert eng.prefills == [Chat()(sys_text, False)]
    assert eng.thinks == []
    (prompt, req), = eng.reads
    assert list(req.template) == res.scaffold + enc("q1: yes\nq2: yes")
    assert req.seeds == (42,)
    assert req.width == res.canvas_width(req.template) == 16
    assert (req.steps, req.pinned, req.constrained) == (1, False, True)
    assert out["answers"]["q1"]["label"] == "yes"
    assert out["answers"]["q1"]["confidence"] == pytest.approx(50 / 51)
    assert out["answers"]["q1"]["noul"] == pytest.approx(50 / 51)
    assert r["rows"] == len(req.template) + 1
    d = out["diagnostics"]
    assert d["prompt_tokens"] == len(prompt) == 6
    assert (d["stages"], d["chunks"], d["skipped"]) == ([["q1", "q2"]], [["q1", "q2"]], {})
    assert (d["conditioning"], d["chunk_prompt"], d["thought"]) == (None, "own", None)
    assert d["engine"] == "gmlx" and d["timing"]["reads"] == 1
    assert d["questions"]["q2"]["pos"] == req.slots[1].pos


def test_diagnostics_key_sets_match_the_proxy():
    out = _run(_qs("q1"))["out"]
    d = out["diagnostics"]
    assert set(d) == DECISION_KEYS
    assert set(d["samples"]) == {"n", "tops", "policy"}
    assert set(d["samples"]["policy"]) == {"mode", "max", "threshold", "extended",
                                           "first_read_entropy"}
    assert set(d["timing"]) == {"total_ms", "reads"}
    assert set(d["questions"]["q1"]) == QUESTION_KEYS
    assert set(out["answers"]["q1"]) == {"type", "label", "confidence",
                                         "probabilities", "noul"}


def test_multi_part_diagnostics_list_each_group():
    r = _run({**_qs("q1", "q2", "q3"), "chunk_rows": 8, "think": 16, "samples": 2})
    d = r["out"]["diagnostics"]
    assert set(d) == DECISION_KEYS
    assert set(d["samples"]) == {"n", "tops", "policy"}
    assert d["samples"]["n"] == [2, 2, 2]
    assert all(set(p) == {"mode", "n", "extended", "first_read_entropy"}
               for p in d["samples"]["policy"])
    assert all(set(t) == THOUGHT_KEYS for t in d["thought"])
    assert set(r["out"]["answers"]["q1"]) == {"type", "label", "confidence",
                                              "probabilities", "noul", "stderr",
                                              "agreement"}


def test_step_and_constrained_settings_reach_the_read():
    r = _run({**_qs("q1"), "steps": 3}, constrained=False)
    (_, req), = r["engine"].reads
    assert (req.steps, req.pinned, req.constrained) == (3, True, False)


# samples


def test_auto_samples_stop_after_one_confident_read():
    r = _run(_qs("q1"))
    pol = r["out"]["diagnostics"]["samples"]["policy"]
    assert [req.seeds for _, req in r["engine"].reads] == [(42,)]
    assert pol["extended"] is False
    assert pol["first_read_entropy"]["q1"] == pytest.approx(_entropy([50 / 51, 1 / 51]))


def test_auto_samples_extend_from_seed_plus_one():
    r = _run(_qs("q1"), engine=FakeEngine(weights=lambda s, i, n: [1.0] * n))
    seeds = [req.seeds for _, req in r["engine"].reads]
    assert seeds == [(42,), (43, 43 + SAMPLE_SEED_STRIDE, 43 + 2 * SAMPLE_SEED_STRIDE)]
    d = r["out"]["diagnostics"]
    assert d["samples"]["n"] == 4 and d["timing"]["reads"] == 4
    assert d["samples"]["policy"]["extended"] is True
    assert r["engine"].prefills == [r["engine"].prefills[0]]


def test_auto_max_one_never_extends():
    r = _run({**_qs("q1"), "auto_max": 1},
             engine=FakeEngine(weights=lambda s, i, n: [1.0] * n))
    assert len(r["engine"].reads) == 1
    assert r["out"]["diagnostics"]["samples"]["policy"]["extended"] is False


def test_auto_threshold_is_compared_to_the_largest_first_entropy():
    body = {**_qs("q1", "q2"), "auto_threshold": 0.5}
    weights = (lambda s, i, n: [50.0, 1.0] if i == 0 else [3.0, 1.0])
    r = _run(body, engine=FakeEngine(weights=weights))
    # q2's entropy is H(0.75, 0.25) = 0.56, over the threshold.
    assert r["out"]["diagnostics"]["samples"]["policy"]["extended"] is True


def test_fixed_samples_give_mean_stderr_and_agreement():
    odd = 42 + 3 * SAMPLE_SEED_STRIDE
    weights = (lambda s, i, n: [1.0, 9.0] if s == odd else [9.0, 1.0])
    r = _run({**_qs("q1"), "samples": 4}, engine=FakeEngine(weights=weights))
    (_, req), = r["engine"].reads
    assert req.seeds == tuple(42 + k * SAMPLE_SEED_STRIDE for k in range(4))
    a = r["out"]["answers"]["q1"]
    assert a["label"] == "yes"
    assert a["noul"] == pytest.approx(0.7) and a["confidence"] == pytest.approx(0.7)
    assert a["probabilities"] == pytest.approx({"yes": 0.7, "no": 0.3})
    assert a["stderr"] == pytest.approx(0.2)
    assert a["agreement"] == pytest.approx(0.75)
    s = r["out"]["diagnostics"]["samples"]
    assert s["policy"] == {"mode": "fixed", "n": 4, "extended": None,
                           "first_read_entropy": None}
    h = _entropy([0.9, 0.1])
    assert s["tops"] == [{"q1": ["yes", pytest.approx(0.9), pytest.approx(h)]}] * 3 + [
        {"q1": ["no", pytest.approx(0.9), pytest.approx(h)]}]


def test_entropy_covers_the_label_union_of_the_read():
    # Two yes/no questions share their label ids, so the second is a choice.
    body = {"questions": {"q1": {"type": "noul"},
                          "q2": {"type": "choice", "criteria": {"x": "", "y": ""}}},
            "samples": 1}
    r = _run(body, engine=FakeEngine(other=1.0))
    (_, req), = r["engine"].reads
    assert len(label_id_union(req.slots)) == 4
    ps = [50 / 53, 1 / 53, 1 / 53, 1 / 53]
    q = r["out"]["diagnostics"]["questions"]["q1"]
    assert q["entropy"] == [pytest.approx(_entropy(ps))]
    assert q["label_mass"] == pytest.approx(51 / 53)
    assert r["out"]["answers"]["q1"]["confidence"] == pytest.approx(50 / 51)


def test_scores_and_choices_aggregate_and_map_to_jev_shapes():
    body = {"questions": {
        "sev": {"type": "score", "criteria": ["low", "mid", "high"]},
        "team": {"type": "choice", "criteria": {"billing": "b", "tech": "t"}},
    }, "samples": 1}
    weights = (lambda s, i, n: [1.0, 2.0, 7.0] if i == 0 else [1.0, 3.0])
    r = _run(body, engine=FakeEngine(weights=weights))
    sev, team = r["out"]["answers"]["sev"], r["out"]["answers"]["team"]
    assert sev["level"] == "high" and sev["label"] == "3"
    assert sev["score"] == pytest.approx(1 * 0.1 + 2 * 0.2 + 3 * 0.7)
    assert team["choice"] == "tech" and team["label"] == "B"
    jev = jev_answers(r["schema"], r["out"])
    assert jev["sev"]["score"] == pytest.approx(0 * 0.1 + 1 * 0.2 + 2 * 0.7)
    assert jev["team"] == {"type": "choice", "choice": "tech",
                           "probabilities": pytest.approx({"billing": 0.25, "tech": 0.75}),
                           "confidence": pytest.approx(0.75)}


# stages and conditioning


def test_stages_follow_dependencies_and_keep_declaration_order():
    r = _run(_qs("q1", "q2", "q3", q2={"depends_on": ["q1"]}))
    d = r["out"]["diagnostics"]
    assert d["stages"] == [["q1", "q3"], ["q2"]]
    assert d["chunks"] == [["q1", "q3"], ["q2"]]
    assert (d["conditioning"], d["chunk_prompt"]) == ("prefill", "full")
    assert [len(req.slots) for _, req in r["engine"].reads] == [2, 1]
    seeds = [req.seeds[0] for _, req in r["engine"].reads]
    assert seeds == [42, 42 + GROUP_SEED_STRIDE]


def test_the_first_chained_stage_without_a_thought_reads_on_the_scaffold():
    r = _run(_qs("q1", "q2", q2={"depends_on": ["q1"]}))
    eng, enc, res = r["engine"], r["enc"], r["resolver"]
    sys_full = system_text(r["schema"])
    chat_prompt = Chat()(sys_full, False)
    (p1, req1), (p2, req2) = eng.reads
    assert p1 == chat_prompt
    assert list(req1.template) == res.scaffold + enc("q1: yes")
    assert p2 == chat_prompt + res.scaffold + enc("q1: yes")
    assert list(req2.template) == enc("\nq2: yes")
    assert eng.prefills == [chat_prompt]
    assert eng.caches[0].extends == [res.scaffold + enc("q1: yes")]
    assert all(c == (sys_full, False) for c in r["chat"].calls)


def test_later_stages_extend_the_prompt_with_only_the_new_tokens():
    body = _qs("a", "b", "c", b={"depends_on": ["a"]}, c={"depends_on": ["b"]})
    r = _run(body)
    eng, enc, res = r["engine"], r["enc"], r["resolver"]
    assert len(eng.prefills) == 1
    assert eng.caches[0].extends == [res.scaffold + enc("a: yes"), enc("\nb: yes")]
    assert eng.reads[2][0] == eng.prefills[0] + res.scaffold + enc("a: yes\nb: yes")


def test_a_retokenized_join_forces_a_fresh_prefill():
    body = _qs("a", "b", "c", b={"depends_on": ["a"]}, c={"depends_on": ["b"]})
    enc = WordEnc(merges=[" yes\n"])
    r = _run(body, enc=enc)
    eng, res = r["engine"], r["resolver"]
    target = eng.prefills[0] + res.scaffold + enc("a: yes\nb: yes")
    assert enc("a: yes\nb: yes")[:3] != enc("a: yes")
    assert eng.prefills[1:] == [target]
    assert eng.caches[0].extends == [res.scaffold + enc("a: yes")]
    assert eng.reads[2][0] == target


def test_a_chained_decision_writes_its_thought_once():
    body = {**_qs("a", "b", "c", b={"depends_on": ["a"]}, c={"depends_on": ["b"]}),
            "think": 32}
    r = _run(body)
    eng, enc, res = r["engine"], r["enc"], r["resolver"]
    sys_full = system_text(r["schema"])
    (t,) = eng.thinks
    prompt = Chat()(sys_full, True) + res.thought_open
    assert t == {"prompt": prompt, "budget": 32, "stop_id": 101, "canvas_width": 64}
    base = prompt + THOUGHT + res.thought_close
    assert eng.prefills == [base]
    assert list(eng.reads[0][1].template) == enc("a: yes")
    assert eng.reads[2][0] == base + enc("a: yes\nb: yes")
    assert r["chat"].calls == [(sys_full, True)]
    d = r["out"]["diagnostics"]
    assert d["thought"] == {"tokens": 3, "closed": True, "ms": 2.0,
                            "text": enc.decode(THOUGHT)}
    templates = [req.template for _, req in eng.reads]
    assert r["rows"] == sum(len(t) + 1 for t in templates) + 3


def test_a_chunked_single_stage_writes_one_thought_per_chunk():
    r = _run({**_qs("q1", "q2", "q3"), "chunk_rows": 8, "think": 16})
    eng, res = r["engine"], r["resolver"]
    assert len(eng.thinks) == 3
    thinking = [s for s, t in r["chat"].calls if t]
    assert len(thinking) == 3
    for i, text in enumerate(thinking, 1):
        assert f"Question q{i}:" in text
        assert all(f"Question q{j}:" not in text for j in (1, 2, 3) if j != i)
        assert eng.thinks[i - 1]["prompt"] == Chat()(text, True) + res.thought_open
    assert [req.seeds for _, req in eng.reads] == [
        (42,), (42 + GROUP_SEED_STRIDE,), (42 + 2 * GROUP_SEED_STRIDE,)]
    d = r["out"]["diagnostics"]
    assert d["chunks"] == [["q1"], ["q2"], ["q3"]]
    assert (d["conditioning"], d["chunk_prompt"]) == (None, "own")
    assert len(d["thought"]) == 3
    assert r["rows"] == sum(len(req.template) + 1 + 3 for _, req in eng.reads)


def test_shared_chunks_use_the_full_chunked_system_text():
    r = _run({**_qs("q1", "q2", "q3"), "chunk_rows": 8, "chunk_prompt": "shared"})
    want = system_text(r["schema"], chunked=True)
    assert [c for c in r["chat"].calls] == [(want, False)] * 3
    assert r["out"]["diagnostics"]["chunk_prompt"] == "shared"


def test_an_alone_question_gets_its_own_group():
    r = _run(_qs("q1", "q2", "q3", "q4", q2={"alone": True}))
    assert r["out"]["diagnostics"]["chunks"] == [["q1"], ["q2"], ["q3", "q4"]]


def test_sequential_chunks_condition_on_earlier_chunks():
    r = _run({**_qs("q1", "q2"), "chunk_rows": 8, "sequential": True})
    eng, enc, res = r["engine"], r["enc"], r["resolver"]
    d = r["out"]["diagnostics"]
    assert (d["conditioning"], d["chunk_prompt"], d["sequential"]) == (
        "prefill", "full", True)
    assert d["stages"] == [["q1", "q2"]] and d["chunks"] == [["q1"], ["q2"]]
    (p1, _), (p2, req2) = eng.reads
    assert p2 == p1 + res.scaffold + enc("q1: yes")
    assert list(req2.template) == enc("\nq2: yes")


# ask and ask_if


def test_a_failed_ask_if_skips_the_question_with_a_null_answer():
    r = _run(_qs("q1", "q2", q2={"ask_if": {"q1": ["no"]}}))
    out = r["out"]
    assert out["answers"]["q2"] is None
    assert out["diagnostics"]["skipped"] == {
        "q2": {"because": "q1", "was": "yes", "wanted": ["no"]}}
    assert out["diagnostics"]["stages"] == [["q1"]]
    assert len(r["engine"].reads) == 1
    # A skip lists the samples per part even with one part.
    assert out["diagnostics"]["samples"]["n"] == [1]
    assert jev_answers(r["schema"], out)["q2"] is None


def test_a_met_ask_if_asks_the_question():
    body = {"questions": {
        "team": {"type": "choice", "criteria": {"billing": "b", "tech": "t"}},
        "refund": {"type": "noul", "ask_if": {"team": ["billing"]}},
    }}
    r = _run(body)
    assert r["out"]["answers"]["refund"]["label"] == "yes"
    assert r["out"]["diagnostics"]["stages"] == [["team"], ["refund"]]


def test_ask_limits_the_questions_answered():
    r = _run({**_qs("q1", "q2", q2={"depends_on": ["q1"]}), "ask": ["q1"]})
    assert list(r["out"]["answers"]) == ["q1"]
    assert list(jev_answers(r["schema"], r["out"])) == ["q1"]
    assert r["out"]["diagnostics"]["stages"] == [["q1"]]
    # One stage, so the decision is not chained.
    assert r["out"]["diagnostics"]["chunk_prompt"] == "own"


# think: "auto"


class ThoughtEngine(FakeEngine):
    """Unsure without a thought in the prompt, sure with one."""

    def read(self, prompt, req):
        sure = THOUGHT[0] in prompt.ids
        self.weights = (lambda seed, i, n: [50.0] + [1.0] * (n - 1)) if sure \
            else (lambda seed, i, n: [1.0, 1.2] + [1.0] * (n - 2))
        return super().read(prompt, req)


THINK_AUTO_KEYS = {"threshold", "budget", "thought", "unsure", "confidence", "reads",
                   "total_ms"}


def test_think_auto_keeps_a_sure_first_pass():
    r = _run({**_qs("q1"), "think": "auto"})
    d = r["out"]["diagnostics"]
    auto = d["think_auto"]
    assert r["engine"].thinks == []
    assert set(auto) == THINK_AUTO_KEYS
    assert (auto["threshold"], auto["budget"], auto["thought"], auto["unsure"]) \
        == (0.8, 64, False, [])
    assert auto["confidence"] == {"q1": r["out"]["answers"]["q1"]["confidence"]}
    assert auto["reads"] == d["timing"]["reads"]
    assert set(d) == DECISION_KEYS | {"think_auto"}


def test_think_auto_runs_again_with_a_thought_when_an_answer_is_unsure():
    eng = ThoughtEngine()
    r = _run({**_qs("q1", "q2"), "think": "auto", "think_budget": 48}, engine=eng)
    out, d = r["out"], r["out"]["diagnostics"]
    auto = d["think_auto"]
    assert [t["budget"] for t in eng.thinks] == [48]
    assert (auto["budget"], auto["thought"], auto["unsure"]) == (48, True, ["q1", "q2"])
    assert all(c < 0.8 for c in auto["confidence"].values())
    assert out["answers"]["q1"]["label"] == "yes"
    assert out["answers"]["q1"]["confidence"] > 0.9
    assert d["thought"]["tokens"] == len(THOUGHT)
    # Four unsure reads, then one sure read on the thought. Timing covers
    # both runs, and think_auto holds the first.
    assert auto["reads"] == 4
    assert d["timing"]["reads"] == 5
    # The rows are the returned run's, as for the same request with think 48.
    second = _run({**_qs("q1", "q2"), "think": 48}, engine=ThoughtEngine())
    assert r["rows"] == second["rows"]


def test_think_auto_lists_unsure_questions_in_question_order():
    r = _run({**_qs("zeta", "alpha"), "think": "auto"}, engine=ThoughtEngine())
    assert r["out"]["diagnostics"]["think_auto"]["unsure"] == ["zeta", "alpha"]


def test_think_auto_threshold_decides_what_is_unsure():
    r = _run({**_qs("q1"), "think": "auto", "think_threshold": 0.5},
             engine=ThoughtEngine())
    assert r["out"]["diagnostics"]["think_auto"]["thought"] is False
    assert r["engine"].thinks == []


def test_think_auto_ignores_skipped_questions():
    body = {**_qs("q1", "q2", q2={"ask_if": {"q1": ["no"]}}), "think": "auto"}
    r = _run(body)
    assert r["out"]["answers"]["q2"] is None
    assert r["out"]["diagnostics"]["think_auto"]["unsure"] == []


# determinism and cancellation


def _seeded(seed, i, n):
    rng = random.Random(seed * 31 + i)
    return [rng.uniform(0.5, 5.0) for _ in range(n)]


def test_a_seed_replays_the_same_answers():
    body = {**_qs("q1", "q2", q2={"depends_on": ["q1"]}), "samples": 3}
    a = _run(body, engine=FakeEngine(weights=_seeded), seed=7)["out"]
    b = _run(body, engine=FakeEngine(weights=_seeded), seed=7)["out"]
    c = _run(body, engine=FakeEngine(weights=_seeded), seed=8)["out"]
    assert a["answers"] == b["answers"]
    assert a["diagnostics"]["samples"]["tops"] == b["diagnostics"]["samples"]["tops"]
    assert a["answers"] != c["answers"]


def test_should_stop_cancels_before_any_read():
    eng = FakeEngine()
    with pytest.raises(Cancelled):
        _run(_qs("q1"), engine=eng, should_stop=lambda: True)
    assert eng.prefills == [] and eng.reads == []
