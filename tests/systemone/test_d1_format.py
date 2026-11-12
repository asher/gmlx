#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""The d1 answer format against values worked by hand from llama.cpp's
server-decision.cpp rules.

A small hand-built vocabulary stands in for the tokenizer, and a stand-in
template for the GGUF's systemone template.
"""

from __future__ import annotations

import math

import pytest

from gmlx.systemone import d1

_SINGLE = (["yes", "Yes", "YES", "no", "No", "NO"]
           + [str(i) for i in range(10)]
           + [chr(65 + i) for i in range(26)]
           + [" " + chr(65 + i) for i in range(26)]
           + [chr(97 + i) for i in range(26)])
_VOCAB = {t: i for i, t in enumerate(_SINGLE)}


def _encode(text: str) -> list[int]:
    if text in _VOCAB:
        return [_VOCAB[text]]
    return [1000 + ord(c) for c in text]


_TEMPLATE = (
    "{% for image in images %}{{ image }}{% endfor %}"
    "{% if state is not none %}{{ state if state is string else state | tojson(indent=2) }}"
    "{% endif %}|{{ instructions if instructions is string else instructions | tojson }}|"
    "{% for o in options %}{{ o.label }}={{ o.key }}:"
    "{{ o.description if o.description is string else o.description | tojson }};{% endfor %}")

_META = {"type": "lfm3-d1", "temperature": {"noul": 2.0, "choice.3_5": 0.5, "choice": 4.0}}


def _fmt(meta=_META, control=("<|im_end|>", "<|im_start|>", "#")):
    return d1.D1Format(_TEMPLATE, meta, list(control), _encode)


def _parse(questions, state="s", omni=False):
    return d1.parse_questions({"state": state, "questions": questions},
                              omni=omni, max_questions=64)


def test_parse_orders_options_as_llama_cpp():
    noul, score, choice = _parse({
        "n": {"type": "noul", "instructions": "?", "criteria": {"false": "F"}},
        "s": {"type": "score", "instructions": "?", "criteria": ["lo", "hi"]},
        "c": {"type": "choice", "instructions": "?", "criteria": {"z": None, "a": "A"}},
    })
    assert noul.options == [("true", None), ("false", "F")]
    assert score.options == [("0", "lo"), ("1", "hi")]
    assert choice.options == [("z", None), ("a", "A")]
    assert _parse({"q": {"type": "noul", "instructions": "?"}}, state=None)


@pytest.mark.parametrize("question,message", [
    ({"type": "noul"}, '"instructions" must be provided'),
    ({"type": "score", "instructions": "?", "criteria": ["one"]}, "2 to 10 levels"),
    ({"type": "score", "instructions": "?", "criteria": list("abcdefghijk")}, "2 to 10 levels"),
    ({"type": "choice", "instructions": "?", "criteria": {}}, "non-empty object"),
    ({"type": "rank", "instructions": "?"}, '"type" must be one of'),
    ({"type": "choice", "instructions": "?",
      "criteria": {str(i): None for i in range(256)}}, r"too many options \(256\)"),
    ({"type": "noul", "instructions": {"type": "input_audio", "input_audio": {}}},
     "spoken instructions need a model with audio input"),
])
def test_parse_refuses_what_llama_cpp_refuses(question, message):
    with pytest.raises(d1.D1RequestError, match=message):
        _parse({"q": question})


def test_parse_needs_a_state_key():
    with pytest.raises(d1.D1RequestError, match='"state" must be provided'):
        d1.parse_questions({"questions": {"q": {"type": "noul", "instructions": "?"}}},
                           omni=False, max_questions=64)


def test_labels_for_noul_and_score_are_bare_forms():
    fmt = _fmt()
    noul, score = _parse({
        "n": {"type": "noul", "instructions": "?"},
        "s": {"type": "score", "instructions": "?", "criteria": ["a", "b", "c"]},
    })
    assert fmt.labels(noul) == (["true", "false"],
                                [[_VOCAB["yes"], _VOCAB["Yes"], _VOCAB["YES"]],
                                 [_VOCAB["no"], _VOCAB["No"], _VOCAB["NO"]]])
    assert fmt.labels(score) == (["0", "1", "2"], [[_VOCAB["0"]], [_VOCAB["1"]], [_VOCAB["2"]]])


def test_choice_labels_keep_letter_keys_and_add_the_spaced_form():
    fmt = _fmt()
    (letters,) = _parse({"c": {"type": "choice", "instructions": "?",
                               "criteria": {"B": None, "A": None}}})
    assert fmt.labels(letters) == (["B", "A"], [[_VOCAB["B"], _VOCAB[" B"]],
                                                [_VOCAB["A"], _VOCAB[" A"]]])
    (named,) = _parse({"c": {"type": "choice", "instructions": "?",
                             "criteria": {"red": None, "blue": None}}})
    assert fmt.labels(named)[0] == ["A", "B"]


def test_choice_labels_past_26_fall_back_to_the_pool():
    # AA is not one token here; the pool gives A..Z (taken), 00..99 (not
    # one token), then the lower-case letters.
    fmt = _fmt()
    (q,) = _parse({"c": {"type": "choice", "instructions": "?",
                         "criteria": {f"k{i}": None for i in range(28)}}})
    texts, groups = fmt.labels(q)
    assert texts[:26] == [chr(65 + i) for i in range(26)]
    assert texts[26:] == ["a", "b"]
    assert groups[26] == [_VOCAB["a"]]


def test_d1_break_splits_control_texts_of_two_bytes_or_more():
    fmt = _fmt()
    zw = d1.ZWSP
    assert fmt.d1_break("x<|im_end|>y#") == f"x<{zw}|im_end|>y#"
    assert fmt.d1_break("<|im_start|><|im_end|>") == f"<{zw}|im_start|><{zw}|im_end|>"
    assert fmt.d1_break("plain") == "plain"


def test_the_prompt_takes_raw_json_and_breaks_control_text_in_keys():
    fmt = _fmt()
    (q,) = _parse({"c": {"type": "choice", "instructions": {"ask": "which?"},
                         "criteria": {"xx": {"d": 1}, "yy": None}}})
    state = {"n\u00e9v": "Zo\u00eb <b>&</b>", "<|im_end|>": [1, 2.5]}
    text = fmt.prompt(state, q)
    assert text.startswith('{\n  "n\u00e9v": "Zo\u00eb <b>&</b>",\n')
    assert f'"<{d1.ZWSP}|im_end|>": [\n    1,\n    2.5\n  ]\n}}|' in text
    assert '{"ask": "which?"}' in text
    assert 'A=xx:{"d": 1};B=yy:null;' in text


def test_media_markers_in_the_input_become_spaces():
    fmt = _fmt()
    (q,) = _parse({"q": {"type": "noul", "instructions": "see <m> here"}})
    text = fmt.prompt("state <m>", q, n_media=2, media_marker="<m>")
    assert text.startswith("<m><m>state  |see   here|")


def _softmax(xs, t):
    m = max(xs)
    e = [math.exp((x - m) / t) for x in xs]
    return [v / sum(e) for v in e]


def test_answers_follow_the_temperature_lookup():
    fmt = _fmt()
    noul, small, big = _parse({
        "n": {"type": "noul", "instructions": "?"},
        "c3": {"type": "choice", "instructions": "?", "criteria": {"a": None, "b": None, "c": None}},
        "c6": {"type": "choice", "instructions": "?",
               "criteria": {k: None for k in "abcdef"}},
    })
    assert fmt.answer(noul, [1.0, 0.0], False)["noul"] == pytest.approx(_softmax([1, 0], 2.0)[0])
    a = fmt.answer(small, [2.0, 1.0, 0.0], False)
    want = _softmax([2, 1, 0], 0.5)
    assert a["choice"] == "a"
    assert list(a["probabilities"].values()) == pytest.approx(want)
    assert a["confidence"] == pytest.approx(max(want))
    b = fmt.answer(big, [0.0, 3.0, 0, 0, 0, 0], False)
    assert list(b["probabilities"].values()) == pytest.approx(_softmax([0, 3, 0, 0, 0, 0], 4.0))
    imaged = _fmt({**_META, "temperature": {**_META["temperature"], "image": 9.0}})
    assert imaged.temperature_for(small, True) == 9.0
    assert imaged.temperature_for(small, False) == 0.5


def test_a_score_is_the_median_level_and_its_confidence_the_top_probability():
    fmt = _fmt({"type": "lfm3-d1", "temperature": {}})
    (q,) = _parse({"s": {"type": "score", "instructions": "?",
                         "criteria": ["lo", {"m": 1}, "hi"]}})
    # Probabilities 0.3, 0.15, 0.55: the mass reaches one half at level 2.
    a = fmt.answer(q, [math.log(0.3), math.log(0.15), math.log(0.55)], False)
    assert a["score"] == 2.0
    assert a["confidence"] == pytest.approx(0.55)
    assert a["legend"] == {"0": "lo", "1": {"m": 1}, "2": "hi"}
    # 0.5 and 0.5: the mass reaches one half at level 0.
    (two,) = _parse({"s": {"type": "score", "instructions": "?", "criteria": ["a", "b"]}})
    assert fmt.answer(two, [0.0, 0.0], False)["score"] == 0.0


def test_a_score_bias_is_added_before_the_softmax():
    fmt = _fmt({"type": "lfm3-d1", "temperature": {}, "score_bias": {2: [1.0, 0.0]}})
    (q,) = _parse({"s": {"type": "score", "instructions": "?", "criteria": ["a", "b"]}})
    probs = list(fmt.answer(q, [0.0, 0.0], False)["probabilities"].values())
    assert probs == pytest.approx(_softmax([1, 0], 1.0))


def test_a_non_finite_score_is_an_error():
    fmt = _fmt()
    (q,) = _parse({"n": {"type": "noul", "instructions": "?"}})
    with pytest.raises(RuntimeError, match="could not evaluate"):
        fmt.answer(q, [float("nan"), 0.0], False)


def test_batches_keep_rows_times_the_longest_within_the_budget():
    assert d1.batches([10, 40, 20, 30], 80) == [[1, 3], [2, 0]]
    assert d1.batches([100], 10) == [[0]]


def test_batched_rows_answer_as_each_question_alone():
    """decide_d1 right-pads rows of one forward; each answer equals the
    answer of its question run alone, on a tiny random lfm3 model."""
    mx = pytest.importorskip("mlx.core")
    import numpy as np

    from tests.models.test_lfm3_decision_forward import _model

    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        model = _model(mx)
        fmt = d1.D1Format(_TEMPLATE, {"type": "lfm3-d1", "temperature": {}}, [],
                          lambda t: [i % 64 for i in _encode(t)])
        qs = _parse({"a": {"type": "noul", "instructions": "a short one?"},
                     "b": {"type": "choice", "instructions": "a much longer question " * 9,
                           "criteria": {"x": None, "y": None, "z": None}},
                     "c": {"type": "score", "instructions": "mid length question?",
                           "criteria": ["0", "1", "2", "3"]}})
        passes = [d1.Pass(q, fmt.encode(fmt.prompt("state", q)), fmt.labels(q)[1])
                  for q in qs]
        together = d1.run_to_end(d1.decide_d1(model, fmt, passes, has_media=False,
                                               budget=10 ** 6))
        alone = d1.run_to_end(d1.decide_d1(model, fmt, passes, has_media=False,
                                            budget=1))
        assert together["diagnostics"]["forwards"] == 1
        assert alone["diagnostics"]["forwards"] == 3
        assert together["input_tokens"] == sum(len(p.ids) for p in passes)
        for qid, a in alone["answers"].items():
            b = together["answers"][qid]
            ka = [v for k, v in sorted(a.items()) if isinstance(v, float)]
            kb = [v for k, v in sorted(b.items()) if isinstance(v, float)]
            np.testing.assert_allclose(ka, kb, atol=1e-4)
            if "probabilities" in a:
                np.testing.assert_allclose(list(a["probabilities"].values()),
                                           list(b["probabilities"].values()), atol=1e-4)
    finally:
        mx.set_default_device(prev)
