"""Decisions through a model's own ``systemone`` template, against
llama.cpp's decision server (``tools/server/server-decision.cpp`` at
a37cc4610) and the decision adapter's README (ibm-granite/granitelib-core-r1.0
at 84df8cfc, ``decision/README.md``).

The template fixture is ``tokenizer.chat_template.systemone`` of
ibm-granite/granite-switch-4.1-8b-GGUF at 9581e330."""

from __future__ import annotations

import math
import re
import types
from pathlib import Path

import pytest

from gmlx.load.tokenizer import _decision_meta, _named_templates
from gmlx.systemone import ar_reader, letters, model_template
from gmlx.systemone.model_template import TemplateFormat, TemplateTokens

FIXTURES = Path(__file__).parents[1] / "fixtures"
SYSTEMONE = (FIXTURES / "granite_switch_systemone.jinja").read_text()
DEFAULT = (FIXTURES / "granite_switch_template.jinja").read_text()
LABELS = [chr(65 + i) for i in range(26)]

STATE = ("My payouts have failed three times. The bank says everything is fine. "
         "Can someone fix this?")
QUESTION = "Which team should handle this ticket?"
OPTIONS = {"payments": "Payout failures and payment processing",
           "account": "Login and account access",
           "other": "Something else"}


def _fmt(meta=None, labels=LABELS, cut=()):
    return TemplateFormat(SYSTEMONE, meta or {"type": "granite-switch"}, labels, cut)


def _question(body, qid):
    return next(q for q in letters.parse(body)["questions"] if q["id"] == qid)


def _body(**questions):
    return {"state": STATE, "questions": questions}


def test_the_readme_ticket_renders_as_the_default_template_with_the_adapter():
    from transformers.utils.chat_template_utils import _compile_jinja_template

    opts = [f"{label}: {desc}" for label, desc in zip(LABELS, OPTIONS.values())]
    tail = ["Question:", QUESTION, "", "Options:", *opts, "",
            "Reply with exactly one option label."]
    msgs = [{"role": "user", "content": "State:\n" + STATE},
            {"role": "user", "content": "### DECISION\n" + "\n".join(tail)}]
    want = _compile_jinja_template(DEFAULT).render(
        messages=msgs, add_generation_prompt=True, adapter_name="decision",
        bos_token="", eos_token="<|end_of_text|>")
    q = _question(_body(team={"type": "choice", "instructions": QUESTION,
                              "criteria": OPTIONS}), "team")
    fmt = _fmt()
    assert fmt.prompt(STATE, q, fmt.options(q)) == want


def test_a_choice_without_a_description_shows_its_name():
    q = _question(_body(t={"type": "choice", "instructions": "?",
                           "criteria": {"payments": None, "other": ""}}), "t")
    fmt = _fmt()
    text = fmt.prompt(STATE, q, fmt.options(q))
    assert "A: payments\nB: other\n" in text


def test_score_adds_its_sentence_once():
    q = _question(_body(s={"type": "score", "instructions": "How urgent?",
                           "criteria": ["low", "mid", "high"]}), "s")
    fmt = _fmt()
    text = fmt.prompt(STATE, q, fmt.options(q))
    assert text.count("Rate along the ordered levels") == 1
    assert "A: low\nB: mid\nC: high\n" in text
    assert letters.LetterFormat().prompt(STATE, q, q["options"]).count(
        "Rate along the ordered levels") == 1


def test_noul_lists_true_first_with_the_template_defaults():
    q = _question(_body(n={"type": "noul", "instructions": "Is it urgent?"}), "n")
    fmt = _fmt()
    assert [o["key"] for o in fmt.options(q)] == ["true", "false"]
    text = fmt.prompt(STATE, q, fmt.options(q))
    assert "A: The statement is true.\nB: The statement is false.\n" in text
    q = _question(_body(n={"type": "noul", "instructions": "?",
                           "criteria": {"false": "calm", "true": "urgent"}}), "n")
    assert "A: urgent\nB: calm\n" in fmt.prompt(STATE, q, fmt.options(q))


def _run(read, scores_for):
    """Drive a letters read, scoring each pass with ``scores_for(pass)``."""
    try:
        batch = next(read)
        while True:
            batch = read.send([scores_for(p) for p in batch])
    except StopIteration as stop:
        return stop.value


def test_answers_are_a_softmax_at_temperature_one_by_default():
    body = _body(team={"type": "choice", "instructions": QUESTION, "criteria": OPTIONS},
                 n={"type": "noul", "instructions": "Is it urgent?"})
    schema = letters.parse(body)
    z = [2.0, 0.5, -1.0]
    out = _run(letters.decision(schema, STATE, _fmt()), lambda p: z[:p.letters])
    team, n = out["answers"]["team"], out["answers"]["n"]
    want = letters.softmax(z)
    assert team["choice"] == "payments"
    assert list(team["probabilities"].values()) == pytest.approx(want)
    assert n["noul"] == pytest.approx(letters.softmax(z[:2])[0])
    assert n["label"] == "yes"


def test_noul_reads_one_order_when_samples_ask_for_more():
    # The template labels true A whatever the order, so a reversed order
    # would read true as false.
    body = {**_body(n={"type": "noul", "instructions": "Is it urgent?"},
                    team={"type": "choice", "instructions": QUESTION,
                          "criteria": OPTIONS}), "samples": 4}
    schema = letters.parse(body)
    seen = []

    def scores(p):
        seen.append(p.text)
        return [2.0, 0.0, -1.0][:p.letters]

    out = _run(letters.decision(schema, STATE, _fmt()), scores)
    assert out["answers"]["n"]["noul"] == pytest.approx(letters.softmax([2.0, 0.0])[0])
    noul = [t for t in seen if "Is it urgent?" in t]
    assert len(set(noul)) == 1
    assert len(set(t for t in seen if QUESTION in t)) > 1
    texts = letters.pass_texts(schema, STATE, _fmt())
    assert sum("Is it urgent?" in t for t in texts) == 1


def test_the_gguf_temperature_and_score_bias_apply():
    meta = {"type": "granite-switch", "temperature": {"choice.3_5": 2.0, "score": 0.5},
            "score_bias": {3: [1.0, 0.0, -1.0]}}
    fmt = _fmt(meta)
    choice = _question(_body(c={"type": "choice", "instructions": "?",
                                "criteria": OPTIONS}), "c")
    score = _question(_body(s={"type": "score", "instructions": "?",
                               "criteria": ["a", "b", "c"]}), "s")
    z = [1.0, 2.0, 3.0]
    assert fmt.probs(z, choice, 3) == pytest.approx(letters.softmax([v / 2 for v in z]))
    assert fmt.probs(z, score, 3) == pytest.approx(
        letters.softmax([(v + b) / 0.5 for v, b in zip(z, [1.0, 0.0, -1.0])]))
    assert fmt.probs(z[:2], choice, 2) == pytest.approx(letters.softmax(z[:2]))


class _Wrapper:
    """A tokenizer whose single letters and a few pairs are one token."""

    PAIRS = {"AA", "AB", "ZZ"}

    def encode(self, text, add_special_tokens=False):
        if len(text) == 1 or text in self.PAIRS:
            return [sum(ord(c) * 100 ** i for i, c in enumerate(text))]
        return [ord(c) for c in text]


def test_labels_are_letters_then_single_token_pairs():
    tokens = TemplateTokens(_Wrapper())
    labels = tokens.label_texts()
    assert labels == LABELS + ["AA", "AB", "ZZ"]
    assert len(set(tokens.letter_ids(labels))) == len(labels)


def test_past_the_labels_options_read_in_chunks():
    fmt = _fmt(labels=LABELS[:4])
    crit = {f"o{i}": None for i in range(10)}
    schema = letters.parse(_body(c={"type": "choice", "instructions": "?",
                                    "criteria": crit}))
    seen = []

    def scores(p):
        seen.append(p.letters)
        return [0.0] * p.letters

    out = _run(letters.decision(schema, STATE, fmt), scores)
    assert max(seen) <= 4
    assert sum(out["answers"]["c"]["probabilities"].values()) == pytest.approx(1.0)


class _CharTokens:
    """One token per character; the role and control tokens are single ids."""

    SPECIAL = {"<|start_of_role|>": 1, "<|end_of_role|>": 2, "<|end_of_text|>": 3,
               "<|decision|>": 4}

    def ids(self, text):
        parts = re.findall(r"<\|[a-z_-]+\|>|.", text, flags=re.S)
        return [self.SPECIAL.get(p, ord(p[0]) + 10) for p in parts]


def test_the_split_ends_before_the_control_token():
    fmt = _fmt(cut=[4])
    tokens = _CharTokens()
    ids, n = fmt.split(tokens, STATE)
    full = tokens.ids(fmt.prefix_text(STATE))
    assert full[n] == 4 and ids == full[:n]
    header = tokens.ids("<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>")
    assert ids[-len(header):] == header
    assert ar_reader.split_point(tokens, STATE, fmt) == (ids, n)


def test_without_a_control_token_the_split_is_the_common_start():
    fmt = _fmt()
    tokens = _CharTokens()
    ids, n = fmt.split(tokens, STATE)
    question = tokens.ids("<|decision|> DECISION\nQuestion:\n")
    assert ids[-len(question):] == question


def test_the_tokenizer_reads_named_templates_and_decision_metadata():
    meta = {"tokenizer.chat_template": "default",
            "tokenizer.chat_template.systemone": "sys1",
            "tokenizer.chat_templates": ["systemone"],
            "graniteswitch.decision.type": "granite-switch",
            "graniteswitch.decision.temperature.choice": 0.8,
            "graniteswitch.decision.temperature.score.3_5": 0.0,
            "graniteswitch.decision.score_bias.3.2": 0.25,
            "graniteswitch.decision.score_bias.3.4": 1.0,
            "graniteswitch.decision.score_bias.x": 1.0}
    assert _named_templates(meta) == {"systemone": "sys1"}
    got = _decision_meta(meta, "graniteswitch")
    assert got == {"type": "granite-switch", "temperature": {"choice": 0.8},
                   "score_bias": {3: [0.0, 0.0, 0.25]}}
    assert _decision_meta({}, "graniteswitch") is None


def _processor(meta, templates):
    tok = types.SimpleNamespace(gguf_decision=meta, gguf_named_templates=templates)
    return types.SimpleNamespace(_wrapper=types.SimpleNamespace(tokenizer=tok))


def test_only_known_decision_types_take_the_template_readout():
    meta = {"type": "granite-switch", "temperature": {}, "score_bias": {}}
    assert model_template.find(_processor(meta, {"systemone": "t"})) == ("t", meta)
    assert model_template.find(_processor({**meta, "type": "openjev"},
                                          {"systemone": "t"})) is None
    assert model_template.find(_processor(meta, {})) is None
    assert model_template.find(_processor(None, {"systemone": "t"})) is None


def test_letter_answers_keep_their_noul_recalibration():
    schema = letters.parse(_body(n={"type": "noul", "instructions": "?"}))
    out = _run(letters.decision(schema, STATE, letters.PLAIN), lambda p: [1.0, 0.0])
    py = letters.softmax([1.0 / letters.T, 0.0])[0]
    z = math.log(py / (1 - py)) / letters.NOUL_T + letters.NOUL_BIAS
    assert out["answers"]["n"]["noul"] == pytest.approx(1 / (1 + math.exp(-z)))
