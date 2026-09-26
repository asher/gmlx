"""The letter readout matches OpenJev's helper on the same letter scores.

The helper runs with the settings of its serving guide and a fake client
that scores each option letter by a hash of the prompt. ``letters`` runs its
decision generator on the same scores. Both must send the same prompts and
return the same answers, up to the helper's rounding to 4 decimals."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

from gmlx.systemone import letters
from gmlx.systemone.contract import jev_answers
from gmlx.systemone.schema import Limits, SchemaError

REF = Path(__file__).with_name("_openjev_helper_ref.py")
SETTINGS = {"READOUT_T": "0.85", "READOUT_NOUL_T": "1.829074",
            "READOUT_NOUL_BIAS": "0", "READOUT_TARGETED": "1",
            "READOUT_INSTR_STYLE": "pyrepr"}
LETTER_BASE = 1000


def score(text: str, i: int) -> float:
    """A log-probability-like score for letter ``i`` in prompt ``text``."""
    h = hashlib.sha256(f"{text}\0{i}".encode()).digest()
    return -8.0 * int.from_bytes(h[:4], "big") / 2**32


@pytest.fixture
def helper(monkeypatch):
    for k, v in SETTINGS.items():
        monkeypatch.setenv(k, v)
    openai = types.ModuleType("openai")
    openai.OpenAI = lambda base_url, api_key: types.SimpleNamespace(base_url=base_url)
    monkeypatch.setitem(sys.modules, "openai", openai)
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
    spec = importlib.util.spec_from_file_location("_openjev_helper_ref", REF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._ids.update({L: LETTER_BASE + i for i, L in enumerate(mod.LETTERS)})
    sent = []

    def create(messages, extra_body, **kw):
        content = messages[0]["content"]
        sent.append(content)
        top = [types.SimpleNamespace(token=f"token_id:{t}",
                                     logprob=score(content, t - LETTER_BASE))
               for t in extra_body["logprob_token_ids"]]
        choice = types.SimpleNamespace(logprobs=types.SimpleNamespace(
            content=[types.SimpleNamespace(top_logprobs=top)]))
        return types.SimpleNamespace(choices=[choice], usage=types.SimpleNamespace(
            prompt_tokens=len(content)))

    mod.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    mod.sent = sent
    return mod


def _helper_body(helper, body, orderings=1):
    helper.PERMS = orderings
    state = helper.with_image(body["state"])
    answers, tokens = {}, 0
    for qid, q in body["questions"].items():
        a, t = helper.ANSWER[q["type"]](state, q)
        answers[qid], tokens = a, tokens + t
    return answers, tokens


def _gmlx_body(body):
    schema = letters.parse(body, Limits(max_samples=8))
    read = letters.decision(schema, letters.state_text(body))
    texts = []
    try:
        batch = next(read)
        while True:
            texts += [p.text for p in batch]
            batch = read.send([[score(p.text, i) for i in range(p.letters)]
                               for p in batch])
    except StopIteration as stop:
        out = stop.value
    return jev_answers(schema, out), texts, out


def _close(got, want):
    if isinstance(want, dict):
        assert set(got) == set(want)
        for k in want:
            _close(got[k], want[k])
    elif isinstance(want, float):
        assert got == pytest.approx(want, abs=6e-5)
    else:
        assert got == want


TICKET = "Customer message: I was charged twice for my order last week and nobody has replied."
SERVE_EXAMPLE = {
    "route": {"type": "choice", "instructions": "Which team should handle this message?",
              "criteria": {"billing": "charges, refunds, invoices",
                           "shipping": "delivery and tracking",
                           "technical": "bugs and login problems"}},
    "angry": {"type": "noul", "instructions": "Is the customer angry?"},
    "urgency": {"type": "score", "instructions": "How urgent is this message?",
                "criteria": ["can wait", "should be handled today",
                             "needs an immediate reply"]},
}


def _options(n):
    return {f"opt{i}": (f"option number {i}" if i % 3 else None) for i in range(n)}


CASES = [
    ("serve_example", {"state": TICKET, "questions": SERVE_EXAMPLE}, 1),
    ("dict_state_non_ascii", {
        "state": {"page": "Caf\u00e9 m\u00fcnchen", "city": "\u6771\u4eac",
                  "items": [1, 2]},
        "questions": {
            "open": {"type": "noul", "instructions": {"goal": "is it open?"},
                     "criteria": {"true": "open now", "false": {"why": "closed"}}},
            "pick": {"type": "choice", "instructions": ["pick", "one"],
                     "criteria": {"a": {"k": "\u00e9"}, "b": None}},
        }}, 1),
    ("list_state", {"state": ["a", "b"], "questions": {"q": SERVE_EXAMPLE["angry"]}}, 1),
    ("samples_3", {"state": TICKET, "questions": SERVE_EXAMPLE}, 3),
    ("letters_52", {"state": TICKET, "questions": {"q": {
        "type": "choice", "instructions": "Which one?", "criteria": _options(52)}}}, 1),
    ("composed_60", {"state": TICKET, "questions": {"q": {
        "type": "choice", "instructions": "Which one?", "criteria": _options(60)}}}, 1),
    ("composed_120_samples_2", {"state": TICKET, "questions": {"q": {
        "type": "choice", "instructions": "Which one?", "criteria": _options(120)}}}, 2),
    ("score_long", {"state": TICKET, "questions": {"q": {
        "type": "score", "instructions": "How bad?",
        "criteria": [f"level {i}" for i in range(12)]}}}, 1),
]


@pytest.mark.parametrize("name,body,orderings", CASES, ids=[c[0] for c in CASES])
def test_answers_match_the_helper(helper, name, body, orderings):
    body = dict(body, samples=orderings)
    want, want_tokens = _helper_body(helper, body, orderings)
    got, texts, _ = _gmlx_body(body)
    assert sorted(texts) == sorted(helper.sent)
    assert sum(len(t) for t in texts) == want_tokens
    for qid in want:
        _close(got[qid], want[qid])


def test_a_dict_state_keeps_its_characters():
    text = letters.state_text({"state": {"city": "\u6771\u4eac"}})
    assert text == '{"city": "\u6771\u4eac"}'


def test_ask_if_skips_a_question_on_the_earlier_answer():
    body = {"state": TICKET, "questions": {
        "angry": SERVE_EXAMPLE["angry"],
        "calm_down": {"type": "noul", "instructions": "Offer a coupon?",
                      "ask_if": {"angry": ["yes"]}},
    }}
    _, texts, out = _gmlx_body(body)
    first = out["answers"]["angry"]["label"]
    if first == "yes":
        assert out["stages"] == [["angry"], ["calm_down"]] and len(texts) == 2
    else:
        assert out["skipped"] == {"calm_down": {"because": "angry", "was": "no",
                                                "wanted": ["yes"]}}
        assert out["answers"]["calm_down"] is None and len(texts) == 1


def test_ask_answers_only_the_asked_questions():
    body = {"state": TICKET, "questions": SERVE_EXAMPLE, "ask": ["route"]}
    _, texts, out = _gmlx_body(body)
    assert list(out["answers"]) == ["route"] and len(texts) == 1


@pytest.mark.parametrize("questions,message", [
    ({"q": {"type": "noul"}}, "instructions is required"),
    ({"q": {"type": "noul", "instructions": "x", "criteria": ["a"]}}, "noul criteria"),
    ({"q": {"type": "maybe", "instructions": "x"}}, "unknown type"),
    ({"q": {"type": "score", "instructions": "x", "criteria": ["only"]}},
     "at least two"),
    ({}, "non-empty map"),
])
def test_bad_questions_are_schema_errors(questions, message):
    with pytest.raises(SchemaError, match=message):
        letters.parse({"state": "s", "questions": questions})


def test_many_options_parse_without_a_cap():
    schema = letters.parse({"state": "s", "questions": {"q": {
        "type": "choice", "instructions": "x", "criteria": _options(200)}}})
    assert len(schema["questions"][0]["options"]) == 200


@pytest.mark.parametrize("samples,orderings", [("auto", 1), (1, 1), (4, 4), (99, 8)])
def test_samples_count_the_orderings(samples, orderings):
    body = {"state": "s", "questions": {"q": SERVE_EXAMPLE["angry"]}, "samples": samples}
    assert letters.parse(body, Limits(max_samples=8))["orderings"] == orderings


@pytest.mark.parametrize("samples", [0, True, "four", 1.5])
def test_bad_samples_are_schema_errors(samples):
    body = {"state": "s", "questions": {"q": SERVE_EXAMPLE["angry"]}, "samples": samples}
    with pytest.raises(SchemaError, match="samples"):
        letters.parse(body)


def test_ignored_fields_name_the_unused_settings():
    body = {"state": "s", "steps": 2, "think": "auto", "seed": 3, "samples": 2,
            "questions": {"a": {"type": "noul", "instructions": "x", "alone": True},
                          "b": {"type": "noul", "instructions": "y",
                                "depends_on": ["a"]}}}
    assert letters.ignored(body) == {"steps", "think", "seed", "alone", "depends_on"}
    schema = letters.parse(body)
    assert [q["depends_on"] for q in schema["questions"]] == [[], []]


def test_image_states_are_recognized():
    assert letters.has_image({"screenshot": "data:image/png;base64,AAAA"})
    assert letters.has_image({"image": "x" * 2001})
    assert not letters.has_image({"image": "a caption"})
    assert not letters.has_image("data:image/png;base64,AAAA")


def test_a_non_finite_score_fails_the_read():
    read = letters.decision(letters.parse(
        {"state": "s", "questions": {"q": SERVE_EXAMPLE["angry"]}}), "s")
    next(read)
    with pytest.raises(letters.ReadoutIncomplete):
        read.send([[0.0, float("nan")]])
