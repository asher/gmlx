"""The rows a prompt's bytes bound, before a tokenizer loads, against the
rows of each family's own tokenizer, built from the header of its encoder
GGUF: ASCII, CJK, emoji, characters whose NFC form holds more bytes, the
empty prompt and a prompt past the token cap.
Skipped unless KQUANT_TEST_GGUF_DIR holds a GGUF that passes a family's
encoder checks. CPU."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

FAMILIES = ("z_image", "krea2", "qwen_image21", "flux3_klein")
PROMPTS = {
    "ascii": "a red bicycle leaning on a brick wall at dusk",
    "cjk": "黄昏时分，一辆红色自"
           "行车靠在砖墙上",
    "emoji": "a bike \U0001F6B2 by a wall \U0001F9F1 at dusk \U0001F306 "
             "\U0001F44D\U0001F3FD",
    # A Hebrew presentation form of three bytes, whose NFC form is three
    # code points of six bytes and five tokens: under the token cap of
    # every family, more tokens than the bytes as given.
    "nfc": "שּׁ" * 96,
    "empty": "",
    "past the cap": "a red bicycle leaning on a brick wall " * 800,
}


@pytest.fixture(scope="module", params=FAMILIES)
def family(request, gguf_index):
    """The spec of a family's encoder and the tokenizer of its GGUF."""
    from gmlx.distill.tokens import tokenizer_from_gguf
    from gmlx.image import text_encoder as te

    spec = te.encoder_spec(request.param)
    for p in gguf_index.get(spec.arch, []):
        try:
            te.check_encoder_file(te._header_view(p), spec)
        except te.EncoderMismatchError:
            continue
        return spec, tokenizer_from_gguf(p)
    pytest.skip(f"no {spec.name} encoder GGUF of {request.param} under "
                "KQUANT_TEST_GGUF_DIR")


def _rows(spec, tokenizer, prompt: str) -> int:
    """Rows of the features of ``prompt``: its token ids at the cap, less
    the dropped ones, and its pad rows."""
    from gmlx.image import text_encoder as te

    n = len(te.token_ids(spec, tokenizer, prompt))
    return n - spec.drop + te.pad_rows(spec, n)


@pytest.mark.parametrize("kind", PROMPTS)
def test_the_bytes_of_a_prompt_bound_its_rows(family, kind):
    from gmlx.image import text_encoder as te

    spec, tok = family
    prompt = PROMPTS[kind]
    assert te.rows_at_most(spec, prompt) >= _rows(spec, tok, prompt)


def test_a_prompt_past_the_cap_is_priced_at_the_cap(family):
    from gmlx.image import text_encoder as te

    spec, tok = family
    prompt = PROMPTS["past the cap"]
    rows = _rows(spec, tok, prompt)
    if spec.max_tokens is None:
        assert rows > 4096 and te.rows_at_most(spec, prompt) > rows
    else:
        assert te.rows_at_most(spec, prompt) == rows == (
            spec.max_tokens - spec.drop)


def test_the_nfc_form_holds_more_rows_than_the_bytes_given(family):
    """The tokenizer reads the NFC form: the bytes of the text as given
    bound fewer rows than the prompt holds."""
    from gmlx.image import text_encoder as te

    spec, tok = family
    prompt = PROMPTS["nfc"]
    n = len(te.render(spec, prompt).encode("utf-8"))
    given = n - spec.drop + te.pad_rows(spec, n)
    assert given < _rows(spec, tok, prompt) <= te.rows_at_most(spec, prompt)
