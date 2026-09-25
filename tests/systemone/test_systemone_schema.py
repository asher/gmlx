"""Schema parsing and the Jev body mapping for /v1/systemone: every
SchemaError branch, the label conventions and the answer format switch."""

from __future__ import annotations

import pytest

from gmlx.systemone.contract import jev_answer, jev_answers, jev_state, log_labels, usage
from gmlx.systemone.schema import (
    DEFAULT_SEED,
    JEV_EXTENSIONS,
    Limits,
    SchemaError,
    jev_schema,
    parse_schema,
    parse_seed,
    schedule,
)


def _noul(qid, **extra):
    return {"id": qid, "type": "noul", **extra}


def _schema(*qs, **extra):
    return {"questions": list(qs), **extra}


def _raises(value, match, limits=Limits()):
    with pytest.raises(SchemaError, match=match):
        parse_schema(value, limits)


# parse_schema: question list and ids


@pytest.mark.parametrize("value", [None, [], {}, {"questions": {}}, {"questions": []}])
def test_a_missing_or_empty_question_list_is_refused(value):
    _raises(value, "non-empty questions array")


def test_the_question_count_is_capped():
    qs = [_noul(f"q{i}") for i in range(5)]
    _raises(_schema(*qs), "at most 4 questions", Limits(max_questions=4))
    assert len(parse_schema(_schema(*qs[:4]), Limits(max_questions=4))["questions"]) == 4


@pytest.mark.parametrize("qid", ["", "   ", "a:b", "a\nb"])
def test_a_bad_question_id_is_refused(qid):
    _raises(_schema(_noul(qid)), "must be non-empty")


def test_a_duplicate_id_is_refused():
    _raises(_schema(_noul("q"), _noul("q")), "duplicate question id")


def test_an_id_is_stripped():
    assert parse_schema(_schema(_noul("  q1 ")))["questions"][0]["id"] == "q1"


def test_an_unknown_type_is_refused():
    _raises(_schema({"id": "q", "type": "rank"}), "unknown type 'rank'")


# parse_schema: alternatives and labels


@pytest.mark.parametrize("kind", ["noul", "bool", "boolean"])
def test_boolean_aliases_become_noul(kind):
    q = parse_schema(_schema({"id": "q", "type": kind,
                              "criteria": {"true": "T", "false": "F"}}))["questions"][0]
    assert q["type"] == "noul"
    assert q["labels"] == ["yes", "no"]
    assert q["choices"] == [("yes", "T"), ("no", "F")]


def test_choice_labels_are_letters_and_options_take_two_shapes():
    q = parse_schema(_schema({"id": "q", "type": "choice", "options": [
        {"name": "red", "description": "warm"}, "blue", {"name": "green"}]}))["questions"][0]
    assert q["labels"] == ["A", "B", "C"]
    assert q["choices"] == [("red", "warm"), ("blue", None), ("green", None)]


def test_score_labels_are_digits_up_to_nine_levels():
    q = parse_schema(_schema({"id": "q", "type": "score",
                              "levels": list(range(9))}))["questions"][0]
    assert q["labels"] == [str(i) for i in range(1, 10)]
    assert q["choices"][0] == ("0", None)


def test_score_labels_are_letters_from_ten_levels():
    q = parse_schema(_schema({"id": "q", "type": "score",
                              "levels": list(range(10))}))["questions"][0]
    assert q["labels"] == list("ABCDEFGHIJ")


@pytest.mark.parametrize("q", [
    {"id": "q", "type": "choice", "options": ["only"]},
    {"id": "q", "type": "choice"},
    {"id": "q", "type": "score", "levels": [1]},
])
def test_fewer_than_two_alternatives_is_refused(q):
    _raises(_schema(q), "at least two alternatives")


def test_more_than_26_alternatives_is_refused():
    opts = [f"o{i}" for i in range(27)]
    _raises(_schema({"id": "q", "type": "choice", "options": opts}), "at most 26")
    q = parse_schema(_schema({"id": "q", "type": "choice", "options": opts[:26]}))
    assert q["questions"][0]["labels"][-1] == "Z"


# parse_schema: dependencies


@pytest.mark.parametrize("deps", ["q1", [1], {"q1": 1}])
def test_depends_on_must_be_a_list_of_ids(deps):
    _raises(_schema(_noul("q1"), _noul("q2", depends_on=deps)), "depends_on must be")


@pytest.mark.parametrize("ask_if", [["q1"], {"q1": "yes"}, {"q1": []}])
def test_ask_if_must_map_ids_to_non_empty_lists(ask_if):
    _raises(_schema(_noul("q1"), _noul("q2", ask_if=ask_if)), "ask_if must map")


def test_depends_on_an_unknown_or_own_id_is_refused():
    _raises(_schema(_noul("q1", depends_on=["zz"])), "depends on unknown question 'zz'")
    _raises(_schema(_noul("q1", depends_on=["q1"])), "depends on unknown question 'q1'")


def test_ask_if_values_must_be_answer_names():
    _raises(_schema(_noul("q1"), _noul("q2", ask_if={"q1": ["maybe"]})),
            "ask_if values for 'q1' must be among")
    c = {"id": "c", "type": "choice", "options": ["red", "blue"]}
    q = parse_schema(_schema(c, _noul("q2", ask_if={"c": ["blue"]})))["questions"][1]
    assert q["ask_if"] == {"c": ["blue"]}


def test_ask_if_keys_join_depends_on_without_repeats():
    q = parse_schema(_schema(_noul("a"), _noul("b"),
                             _noul("c", depends_on=["b", "a"], ask_if={"a": ["yes"]})))
    assert q["questions"][2]["depends_on"] == ["b", "a"]


def test_a_dependency_cycle_is_refused():
    _raises(_schema(_noul("a", depends_on=["b"]), _noul("b", depends_on=["a"])),
            "dependency cycle among a, b")


def test_schedule_orders_stages_and_keeps_declaration_order():
    qs = parse_schema(_schema(_noul("a"), _noul("b", depends_on=["c"]), _noul("c"),
                              _noul("d", depends_on=["a"])))["questions"]
    assert [[q["id"] for q in lv] for lv in schedule(qs)] == [["a", "c"], ["b", "d"]]


def test_schedule_ignores_a_dependency_outside_the_list():
    # decide schedules an ask subset whose closure is already checked.
    qs = [{"id": "b", "depends_on": ["a"]}]
    assert [[q["id"] for q in lv] for lv in schedule(qs)] == [["b"]]


# parse_schema: sampling, steps and the other extensions


def test_samples_default_to_auto():
    p = parse_schema(_schema(_noul("q")))["policy"]
    assert p == {"mode": "auto", "max": 4, "threshold": 0.1}


def test_auto_max_is_clamped_and_converted():
    assert parse_schema(_schema(_noul("q"), auto_max="7"))["policy"]["max"] == 7
    assert parse_schema(_schema(_noul("q"), auto_max=0))["policy"]["max"] == 1
    lim = Limits(max_samples=5)
    assert parse_schema(_schema(_noul("q"), auto_max=99), lim)["policy"]["max"] == 5


@pytest.mark.parametrize("value", ["four", None, [4], float("inf")])
def test_a_bad_auto_max_is_a_schema_error(value):
    _raises(_schema(_noul("q"), auto_max=value), "auto_max must be an integer")


@pytest.mark.parametrize("value", ["low", None, [0.1]])
def test_a_bad_auto_threshold_is_a_schema_error(value):
    _raises(_schema(_noul("q"), auto_threshold=value), "auto_threshold must be a number")


def test_fixed_samples_are_clamped():
    assert parse_schema(_schema(_noul("q"), samples=3))["policy"] == {"mode": "fixed", "n": 3}
    lim = Limits(max_samples=2)
    assert parse_schema(_schema(_noul("q"), samples=9), lim)["policy"]["n"] == 2


@pytest.mark.parametrize("value", [0, -1, 2.0, "3", "many", None])
def test_bad_samples_are_refused(value):
    _raises(_schema(_noul("q"), samples=value), "samples must be")


def test_steps_are_clamped_to_one_through_eight():
    assert parse_schema(_schema(_noul("q")))["steps"] == 1
    assert parse_schema(_schema(_noul("q"), steps=0))["steps"] == 1
    assert parse_schema(_schema(_noul("q"), steps="3"))["steps"] == 3
    assert parse_schema(_schema(_noul("q"), steps=20))["steps"] == 8


@pytest.mark.parametrize("value", ["two", None, {}, float("nan"), float("inf")])
def test_bad_steps_are_a_schema_error(value):
    _raises(_schema(_noul("q"), steps=value), "steps must be an integer")


@pytest.mark.parametrize("ask", ["q1", [], ["zz"], [1], [["q1"]]])
def test_ask_must_list_known_ids(ask):
    _raises(_schema(_noul("q1"), ask=ask), "ask must list question ids")


def test_ask_must_include_what_its_questions_depend_on():
    _raises(_schema(_noul("q1"), _noul("q2", depends_on=["q1"]), ask=["q2"]),
            "ask names 'q2' but not everything it depends on")
    s = parse_schema(_schema(_noul("q1"), _noul("q2", depends_on=["q1"]), ask=["q1", "q2"]))
    assert s["ask"] == ["q1", "q2"]


@pytest.mark.parametrize("value", [7, 8.0, "16", True])
def test_bad_chunk_rows_are_refused(value):
    _raises(_schema(_noul("q"), chunk_rows=value), "chunk_rows must be")


def test_chunk_rows_and_prompt_defaults():
    s = parse_schema(_schema(_noul("q")))
    assert (s["chunk_rows"], s["chunk_prompt"], s["sequential"]) == (None, "own", False)
    s = parse_schema(_schema(_noul("q"), chunk_rows=8, chunk_prompt="shared", sequential=1))
    assert (s["chunk_rows"], s["chunk_prompt"], s["sequential"]) == (8, "shared", True)


def test_a_bad_chunk_prompt_is_refused():
    _raises(_schema(_noul("q"), chunk_prompt="full"), "chunk_prompt must be")


@pytest.mark.parametrize("value", [True, -1, 4097, 1.5, "64"])
def test_a_bad_think_budget_is_refused(value):
    _raises(_schema(_noul("q"), think=value), "think must be a thought budget")


def test_think_accepts_the_range_ends():
    assert parse_schema(_schema(_noul("q"), think=0))["think"] == 0
    assert parse_schema(_schema(_noul("q"), think=4096))["think"] == 4096


def test_the_format_switches_to_indexed_past_ten_questions():
    ten = [_noul(f"q{i}") for i in range(10)]
    assert parse_schema(_schema(*ten))["format"] == "lines"
    assert parse_schema(_schema(*ten, _noul("q10")))["format"] == "indexed"


def test_parsed_schema_keys():
    s = parse_schema(_schema(_noul("q"), instructions="Be strict."))
    assert set(s) == {"questions", "instructions", "policy", "steps", "think", "ask",
                      "chunk_rows", "chunk_prompt", "sequential", "format"}
    assert s["instructions"] == "Be strict."
    assert set(s["questions"][0]) == {"id", "type", "instructions", "choices", "labels",
                                      "depends_on", "ask_if", "alone"}


# parse_seed


def test_the_seed_defaults_to_42():
    assert parse_seed({}) == DEFAULT_SEED == 42


@pytest.mark.parametrize("value,want", [(7, 7), ("11", 11), (3.9, 3)])
def test_the_seed_converts_like_int(value, want):
    assert parse_seed({"seed": value}) == want


@pytest.mark.parametrize("value", ["abc", None, [1], {}, float("inf")])
def test_a_non_integer_seed_is_a_schema_error(value):
    with pytest.raises(SchemaError, match="seed: must be an integer"):
        parse_seed({"seed": value})


# jev_schema


@pytest.mark.parametrize("qs", [None, [], {}, "q"])
def test_jev_questions_must_be_a_non_empty_map(qs):
    with pytest.raises(SchemaError, match="non-empty map"):
        jev_schema({"questions": qs})


def test_a_jev_question_must_be_an_object():
    with pytest.raises(SchemaError, match="must be an object"):
        jev_schema({"questions": {"q": "yes or no"}})


@pytest.mark.parametrize("q,match", [
    ({"type": "noul", "criteria": "yes"}, "noul criteria must be an object"),
    ({"type": "choice"}, "choice criteria must map"),
    ({"type": "choice", "criteria": {}}, "choice criteria must map"),
    ({"type": "choice", "criteria": ["a", "b"]}, "choice criteria must map"),
    ({"type": "score", "criteria": {"a": 1}}, "score criteria must be an ordered list"),
    ({"type": "rank", "criteria": []}, "unknown type 'rank'"),
])
def test_bad_jev_criteria_are_refused(q, match):
    with pytest.raises(SchemaError, match=match):
        jev_schema({"questions": {"q": q}})


def test_jev_questions_map_to_the_schema():
    s = jev_schema({
        "questions": {
            "urgent": {"type": "noul", "instructions": "Is it urgent?",
                       "criteria": {"true": "needs action today", "false": "can wait"}},
            "team": {"type": "choice", "instructions": {"hint": "route"},
                     "criteria": {"billing": "money", "tech": "bugs"},
                     "depends_on": ["urgent"]},
            "sev": {"type": "score", "criteria": ["low", "mid", "high"],
                    "ask_if": {"urgent": ["yes"]}, "alone": True},
            "fyi": {"type": "noul"},
        },
        "samples": 2, "steps": 3, "think": 16, "seed": 5, "images": [],
        "instructions": "Triage.", "unknown_key": 1,
    })
    by = {q["id"]: q for q in s["questions"]}
    assert list(by) == ["urgent", "team", "sev", "fyi"]
    assert by["urgent"]["choices"] == [("yes", "needs action today"), ("no", "can wait")]
    assert by["team"]["instructions"] == '{"hint": "route"}'
    assert by["team"]["choices"] == [("billing", "money"), ("tech", "bugs")]
    assert by["team"]["depends_on"] == ["urgent"]
    assert by["sev"]["choices"] == [("low", None), ("mid", None), ("high", None)]
    assert by["sev"]["alone"] is True and by["sev"]["depends_on"] == ["urgent"]
    assert by["fyi"]["choices"] == [("yes", None), ("no", None)]
    assert s["policy"] == {"mode": "fixed", "n": 2}
    assert (s["steps"], s["think"], s["instructions"]) == (3, 16, "Triage.")


def test_jev_extensions_list():
    assert JEV_EXTENSIONS == ("instructions", "samples", "auto_max", "auto_threshold",
                              "steps", "think", "ask", "chunk_rows", "chunk_prompt",
                              "sequential")


def test_jev_schema_applies_the_limits():
    body = {"questions": {f"q{i}": {"type": "noul"} for i in range(3)}}
    with pytest.raises(SchemaError, match="at most 2 questions"):
        jev_schema(body, Limits(max_questions=2))


# contract


def test_jev_state_passes_text_and_serializes_the_rest():
    assert jev_state({"state": "ticket text"}) == "ticket text"
    assert jev_state({"state": {"a": [1, 2]}}) == '{"a": [1, 2]}'
    assert jev_state({"state": ""}) == ""
    with pytest.raises(SchemaError, match="state: required"):
        jev_state({})


def test_jev_answer_shapes():
    noul = {"id": "n", "type": "noul", "choices": [("yes", None), ("no", None)]}
    choice = {"id": "c", "type": "choice", "choices": [("red", None), ("blue", None)]}
    score = {"id": "s", "type": "score",
             "choices": [("low", None), ("mid", None), ("high", None)]}
    assert jev_answer(noul, {"noul": 0.8, "label": "yes"}) == {"type": "noul", "noul": 0.8}
    assert jev_answer(choice, {"choice": "blue", "confidence": 0.6,
                               "probabilities": {"red": 0.4, "blue": 0.6}}) == {
        "type": "choice", "choice": "blue", "confidence": 0.6,
        "probabilities": {"red": 0.4, "blue": 0.6}}
    got = jev_answer(score, {"confidence": 0.5, "score": 2.3,
                             "probabilities": {"low": 0.2, "mid": 0.3, "high": 0.5}})
    # The Jev score is the 0-indexed expectation, not the 1-indexed internal one.
    assert got["score"] == pytest.approx(0 * 0.2 + 1 * 0.3 + 2 * 0.5)
    assert got["legend"] == {"0": "low", "1": "mid", "2": "high"}
    assert got["probabilities"] == {"0": 0.2, "1": 0.3, "2": 0.5}
    assert got["confidence"] == 0.5 and got["type"] == "score"
    assert jev_answer(noul, None) is None


def test_jev_answers_follow_the_schema_order():
    s = jev_schema({"questions": {"b": {"type": "noul"}, "a": {"type": "noul"}}})
    body = {"answers": {"a": None, "b": {"noul": 0.1, "label": "no"}}}
    assert list(jev_answers(s, body)) == ["b", "a"]
    assert jev_answers(s, body)["a"] is None


def test_usage_and_log_labels():
    assert usage(12, 5) == {"input_tokens": 12, "output_tokens": 5}
    assert log_labels({"a": {"label": "yes"}, "b": None}) == "a=yes b=skipped"
