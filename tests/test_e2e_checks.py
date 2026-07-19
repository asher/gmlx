#!/usr/bin/env python3
"""Unit tests for the e2e floor detectors (``tests/e2e/checks.py``). Pure CPU - no
model, no server. Guards the repetition / mojibake / schema logic so a tuning change
can't silently make the e2e harness blind to a regression it's meant to catch."""
from __future__ import annotations

import pathlib
import sys

import pytest

# checks.py lives in the (intentionally non-pytest-collected) e2e harness dir.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "e2e"))
checks = pytest.importorskip("checks")


# transport / schema
def _chat_body(text, finish="stop", pt=10, ct=5):
    return {"choices": [{"message": {"role": "assistant", "content": text},
                         "finish_reason": finish}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct}}


def test_http_ok_rejects_non200():
    assert not checks.check_http_ok(500, {"e": 1})
    assert not checks.check_http_ok(200, "oops")
    assert checks.check_http_ok(200, {"choices": []})


def test_extract_chat_text_string_and_parts():
    assert checks.extract_chat_text(_chat_body("hi")) == "hi"
    parts = {"choices": [{"message": {"content": [{"text": "a"}, {"text": "b"}]},
                          "finish_reason": "stop"}]}
    assert checks.extract_chat_text(parts) == "ab"
    assert checks.extract_chat_text({"choices": []}) is None


def test_chat_schema_needs_content_and_finish():
    assert checks.check_chat_schema(_chat_body("hi"))
    assert not checks.check_chat_schema({"choices": [{"message": {"content": "x"}}]})
    assert not checks.check_chat_schema({"choices": []})


def test_finish_reason_and_usage():
    assert checks.check_finish_reason(_chat_body("x", finish="stop"))
    assert not checks.check_finish_reason(_chat_body("x", finish="error"))
    assert checks.check_usage(_chat_body("x", pt=3, ct=2))
    assert not checks.check_usage(_chat_body("x", pt=3, ct=0))


# content sanity
def test_nonempty():
    assert not checks.check_nonempty("   ")
    assert checks.check_nonempty("ok")
    assert not checks.check_nonempty("hi", min_chars=10)


def test_mojibake():
    assert checks.check_no_mojibake("clean text")
    assert not checks.check_no_mojibake("br�ken")


def test_nan_tokens():
    assert checks.check_no_nan_tokens("the information is useful")   # 'inf' inside word
    assert not checks.check_no_nan_tokens("nan nan nan nan")


# repetition / degeneration (the crux)
def test_repetition_passes_clean_prose():
    text = ("Photosynthesis converts light energy into chemical energy. Plants "
            "absorb sunlight through chlorophyll, split water, and fix carbon "
            "dioxide into sugars during the Calvin cycle, releasing oxygen.")
    assert checks.detect_repetition(text)


def test_repetition_passes_enumeration():
    # counting must NOT be flagged (every token distinct -> high diversity)
    assert checks.detect_repetition(" ".join(str(i) for i in range(1, 60)))


def test_repetition_passes_short_refrain():
    # a chorus repeated a couple of times is fine
    text = ("la la la and the song goes on and on with a happy little tune "
            "la la la and the song goes on once more into the bright morning")
    assert checks.detect_repetition(text)


def test_repetition_flags_ngram_loop():
    text = "I cannot do that. " * 8
    assert not checks.detect_repetition(text)


def test_repetition_flags_char_loop():
    assert not checks.detect_repetition("ha" * 40)


def test_repetition_flags_low_diversity():
    text = ("yes yes yes yes no yes yes yes yes yes yes no yes yes yes yes yes "
            "yes yes yes no yes yes yes yes yes yes yes yes no yes yes")
    assert not checks.detect_repetition(text)


def test_repetition_flags_single_word_domination():
    text = ("buy " * 20) + "now please consider the offer today friend"
    assert not checks.detect_repetition(text)


# anchored helpers
def test_contains_any_all_ci():
    assert checks.check_contains("The capital is Paris.", "paris")
    assert not checks.check_contains("The capital is Paris.", "london")
    assert checks.check_contains("a b c", ["a", "c"], mode="all")
    assert not checks.check_contains("a b", ["a", "c"], mode="all")


def test_uppercase_fraction():
    assert checks.fraction_uppercase_letters("HELLO WORLD") == 1.0
    assert checks.fraction_uppercase_letters("hello") == 0.0
    assert abs(checks.fraction_uppercase_letters("Hi") - 0.5) < 1e-9


# the floor bundle
def test_floor_bundle_pass_and_fail():
    good = checks.floor_text_checks(200, _chat_body(
        "Paris is the capital of France, a major European cultural center."))
    assert good.ok, good.failures

    looped = checks.floor_text_checks(200, _chat_body("spam spam " * 30))
    assert not looped.ok
    assert any(f.name == "repetition" for f in looped.failures)

    broken = checks.floor_text_checks(500, {"error": "boom"})
    assert not broken.ok
    assert broken.failures[0].name == "http_ok"
