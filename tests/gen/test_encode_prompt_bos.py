#!/usr/bin/env python3
"""The single-BOS rule every generate entry point encodes through
(gmlx.gen.generation.encode_prompt)."""

from __future__ import annotations

import pytest

from gmlx.gen.generation import encode_prompt

BOS = "<\uff5cbegin\u2581of\u2581sentence\uff5c>"


class _Tok:
    """A GGUF tokenizer with add_bos_token true: encode() prepends BOS
    unless the caller opts out."""

    bos_token = BOS

    def encode(self, text, add_special_tokens=True):
        ids = [ord(c) for c in text.replace(BOS, "\x00")]
        return ([0] + ids) if add_special_tokens else ids


def test_a_bos_leading_prompt_is_not_doubled():
    ids = encode_prompt(_Tok(), BOS + "hi")
    assert ids.count(0) == 1 and ids[0] == 0


def test_a_raw_prompt_still_gets_one():
    # llama.cpp's raw-completion convention, which add_bos_token exists for.
    assert encode_prompt(_Tok(), "hi") == [0, ord("h"), ord("i")]


def test_a_tokenizer_without_a_bos_string_opts_in():
    tok = _Tok()
    tok.bos_token = None
    assert encode_prompt(tok, "hi")[0] == 0


@pytest.mark.parametrize("flag", [True, False])
def test_the_deepseek41_template_stays_single_bos(flag):
    """Both conversions ship the same template; only the ds4 one sets
    add_bos_token."""
    class _T(_Tok):
        def encode(self, text, add_special_tokens=True):
            ids = [ord(c) for c in text.replace(BOS, "\x00")]
            return ([0] + ids) if (add_special_tokens and flag) else ids

    assert encode_prompt(_T(), BOS + "<think>").count(0) == 1
