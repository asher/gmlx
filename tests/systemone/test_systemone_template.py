"""Answer templates for /v1/systemone: system text, answer lines, slot
resolution against a word-level fake tokenizer, canvas widths and the
template LRU."""

from __future__ import annotations

import re

import pytest

from gmlx.systemone.reads import (
    LABEL_ID_CAP,
    PAD,
    SEED_VOCAB,
    TURN_CLOSE,
    Slot,
    build_canvas,
    label_id_union,
    pinned_positions,
)
from gmlx.systemone.schema import SchemaError, parse_schema
from gmlx.systemone.template import (
    FORMATS,
    SCAFFOLD_TEXT,
    TemplateResolver,
    answer_text,
    system_text,
)

_TOKEN = re.compile(r"<\|channel>|<channel\|>| ?[A-Za-z0-9_]+|\n| |[^\sA-Za-z0-9_]")


class WordEnc:
    """A word with its leading space is one token, so "yes" and " yes" get
    different ids. The thought tags are single tokens, as in the GGUF."""

    def __init__(self, merges=()):
        self.vocab = {"<|channel>": 100, "<channel|>": 101, "\n": 107, "thought": 45518}
        self.pattern = re.compile(
            "|".join([re.escape(m) for m in merges] + [_TOKEN.pattern]))
        self.calls = 0

    def id(self, piece: str) -> int:
        if piece not in self.vocab:
            self.vocab[piece] = 1000 + len(self.vocab)
        return self.vocab[piece]

    def __call__(self, text: str) -> list[int]:
        self.calls += 1
        return [self.id(t) for t in self.pattern.findall(text)]

    def decode(self, ids) -> str:
        inv = {v: k for k, v in self.vocab.items()}
        return "".join(inv.get(i, "?") for i in ids)


def _schema(*qs, **extra):
    return parse_schema({"questions": list(qs), **extra})


def _noul(qid, **extra):
    return {"id": qid, "type": "noul", **extra}


# system_text and answer_text


def test_system_text_lists_questions_and_the_reply_format():
    s = _schema(
        {"id": "urgent", "type": "noul", "instructions": " Is it urgent? ",
         "criteria": {"true": "act today", "false": None}},
        {"id": "team", "type": "choice", "instructions": "Route it.",
         "options": [{"name": "billing", "description": " money "}, "tech"]},
        instructions="Triage the ticket.")
    text = system_text(s)
    assert "\nTriage the ticket.\n" in text
    assert "\nQuestion urgent: Is it urgent?\n  yes: act today\n  no\n" in text
    assert "\nQuestion team: Route it.\n  A: billing (money)\n  B: tech\n" in text
    assert text.endswith(FORMATS["lines"][2])
    assert system_text(s, chunked=True).endswith(
        "answer every line that is present.")


def test_answer_text_in_both_formats():
    s = _schema(_noul("a"), {"id": "b", "type": "score", "levels": [1, 2, 3]})
    qs = s["questions"]
    assert answer_text(qs, [0, 2]) == "a: yes\nb: 3"
    assert answer_text(qs, [1, 0], "indexed") == "ano b1"


def test_the_indexed_format_changes_the_reply_instruction():
    s = _schema(*[_noul(f"q{i}") for i in range(11)])
    assert s["format"] == "indexed"
    assert system_text(s).endswith(FORMATS["indexed"][2])


# TemplateResolver


def test_the_resolver_owns_the_thought_ids():
    enc = WordEnc()
    r = TemplateResolver(enc, 64)
    assert r.thought_open == [100, 45518, 107]
    assert r.thought_close == [101]
    assert r.scaffold == enc(SCAFFOLD_TEXT) == r.thought_open + r.thought_close


def test_a_tokenizer_that_merges_the_tags_is_refused():
    with pytest.raises(RuntimeError, match="does not split the thought tags"):
        TemplateResolver(WordEnc(merges=["\n<channel|>"]), 64)


def test_a_multi_token_close_tag_is_refused():
    def enc(text):
        out = []
        for piece in _TOKEN.findall(text):
            out += [7, 8] if piece == "<channel|>" else [hash(piece) % 50 + 200]
        return out

    with pytest.raises(RuntimeError, match="close tag is not a single token"):
        TemplateResolver(enc, 64)


@pytest.mark.parametrize("n,want", [(0, 16), (15, 16), (16, 32), (31, 32), (47, 48),
                                    (63, 64), (64, 64), (200, 64)])
def test_canvas_width_rounds_up_to_the_step_and_caps_at_the_canvas(n, want):
    r = TemplateResolver(WordEnc(), 64)
    assert r.canvas_width([1] * n) == want


def test_rows_for_counts_scaffold_answers_and_the_turn_close():
    enc = WordEnc()
    r = TemplateResolver(enc, 64)
    qs = _schema(_noul("a"), _noul("b"))["questions"]
    # scaffold 4 + "a: yes\nb: yes" 7 + turn close 1
    assert r.rows_for(qs, "lines") == 12
    assert r.rows_for(qs[:1], "lines") == 8


def test_slots_use_the_space_prefixed_label_ids():
    enc = WordEnc()
    r = TemplateResolver(enc, 64)
    s = _schema(_noul("a"), {"id": "b", "type": "choice", "options": ["x", "y", "z"]})
    template, slots = r.resolve(s["questions"], r.scaffold, "", "lines")
    assert list(template) == r.scaffold + enc("a: yes\nb: A")
    assert slots == (Slot(pos=6, label_ids=(enc.id(" yes"), enc.id(" no"))),
                     Slot(pos=10, label_ids=(enc.id(" A"), enc.id(" B"), enc.id(" C"))))
    assert enc.id("yes") not in slots[0].label_ids
    assert enc.id(" yes") != enc.id("yes")


def test_a_lead_shifts_the_slots_and_an_empty_head_drops_the_scaffold():
    enc = WordEnc()
    r = TemplateResolver(enc, 64)
    qs = _schema(_noul("a"))["questions"]
    template, slots = r.resolve(qs, [], "\n", "lines")
    assert list(template) == enc("\na: yes")
    assert slots[0].pos == 3


def test_indexed_slots_resolve():
    enc = WordEnc()
    r = TemplateResolver(enc, 64)
    qs = _schema(*[_noul(f"q{i}") for i in range(11)])["questions"]
    template, slots = r.resolve(qs, [], "", "indexed")
    # This fake joins an id and its label into one token ("q0yes").
    assert len(template) == 11
    assert [s.pos for s in slots] == list(range(11))
    assert slots[0].label_ids == (enc.id("q0yes"), enc.id("q0no"))


def test_a_template_longer_than_the_canvas_is_refused():
    r = TemplateResolver(WordEnc(), 16)
    qs = _schema(_noul("a"), _noul("b"), _noul("c"))["questions"]
    # scaffold 4 + three answer lines 11 + turn close 1 fills 16 rows.
    r.resolve(qs, r.scaffold, "", "lines")
    qs = _schema(_noul("a"), _noul("b"), _noul("c"), _noul("d"))["questions"]
    with pytest.raises(SchemaError, match="the canvas holds 15"):
        r.resolve(qs, r.scaffold, "", "lines")


def test_a_label_that_spans_two_tokens_is_refused():
    base = WordEnc()

    def enc(text):
        return base(text.replace(" no", " n o"))

    r = TemplateResolver(enc, 64)
    with pytest.raises(SchemaError, match="label 'no' is not a single token"):
        r.resolve(_schema(_noul("a"))["questions"], [], "", "lines")


def test_labels_that_do_not_share_a_slot_are_refused():
    base = WordEnc()

    def enc(text):
        # " B" also rewrites the token after it, so two positions change.
        return base(text.replace(" B\n", " B\nX"))

    r = TemplateResolver(enc, 64)
    qs = _schema({"id": "a", "type": "choice", "options": ["x", "y"]},
                 _noul("b"))["questions"]
    with pytest.raises(SchemaError, match="labels do not share one template slot"):
        r.resolve(qs, [], "", "lines")


def test_two_labels_with_one_id_are_refused():
    base = WordEnc()

    def enc(text):
        return base(text.replace(" C", " B"))

    r = TemplateResolver(enc, 64)
    qs = _schema({"id": "a", "type": "choice", "options": ["x", "y", "z"]})["questions"]
    with pytest.raises(SchemaError, match="two labels tokenize to the same id"):
        r.resolve(qs, [], "", "lines")


def test_template_for_caches_by_head_lead_format_and_labels():
    enc = WordEnc()
    r = TemplateResolver(enc, 64, cache_size=2)
    s = _schema(_noul("a"))
    first = r.template_for(s, r.scaffold, "")
    calls = enc.calls
    assert r.template_for(s, list(r.scaffold), "") is first
    assert enc.calls == calls
    assert r.template_for(s, [], "\n") is not first


def test_template_for_evicts_the_least_recently_used_entry():
    enc = WordEnc()
    r = TemplateResolver(enc, 64, cache_size=2)
    s = _schema(_noul("a"))
    one = r.template_for(s, [], "")
    two = r.template_for(s, [], "\n")
    assert r.template_for(s, [], "") is one      # "" is now the newest
    r.template_for(s, [], " ")                   # evicts "\n"
    assert len(r._cache) == 2
    calls = enc.calls
    assert r.template_for(s, [], "") is one
    assert enc.calls == calls
    again = r.template_for(s, [], "\n")
    assert again == two and again is not two
    assert enc.calls > calls


# reads helpers


def test_build_canvas_seeds_slots_in_order():
    import random

    slots = (Slot(1, (5, 6)), Slot(3, (7, 8)))
    canvas = build_canvas((10, 11, 12, 13, 14), slots, 16, seed=9, vocab=1000)
    # Drawn below the proxy's vocabulary constant, then wrapped into this one.
    rng = random.Random(9)
    first, second = rng.randrange(SEED_VOCAB) % 1000, rng.randrange(SEED_VOCAB) % 1000
    assert canvas == [10, first, 12, second, 14, TURN_CLOSE] + [PAD] * 10
    full = build_canvas((10, 11, 12, 13, 14), slots, 16, seed=9, vocab=SEED_VOCAB)
    rng = random.Random(9)
    assert (full[1], full[3]) == (rng.randrange(SEED_VOCAB), rng.randrange(SEED_VOCAB))
    swapped = build_canvas((10, 11, 12, 13, 14), slots[::-1], 16, seed=9, vocab=1000)
    assert (swapped[3], swapped[1]) == (first, second)
    as_dicts = [{"pos": 1, "label_ids": [5, 6]}, {"pos": 3, "label_ids": [7, 8]}]
    assert build_canvas((10, 11, 12, 13, 14), as_dicts, 16, 9, 1000) == canvas


def test_label_id_union_sorts_dedups_and_caps():
    assert label_id_union([Slot(0, (9, 3)), {"pos": 1, "label_ids": [3, 5]}]) == [3, 5, 9]
    big = [Slot(i, (2 * i, 2 * i + 1)) for i in range(100)]
    assert label_id_union(big) == list(range(LABEL_ID_CAP))


def test_pinned_positions_hold_everything_but_the_slots_past_one_step():
    slots = (Slot(2, (1, 2)), Slot(5, (3, 4)))
    assert pinned_positions(slots, 8, 1) == []
    assert pinned_positions(slots, 8, 2) == [0, 1, 3, 4, 6, 7]
