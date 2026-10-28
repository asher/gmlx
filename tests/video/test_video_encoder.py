"""The H3 encoder spec's token ids on a GGUF tokenizer laid out as the
stock Qwen3-VL-32B file's: filler entries at the seven ids MiniMax's
tokenizer adds. CPU-only, no weights."""
from __future__ import annotations

import dataclasses

import pytest

from gmlx.image import text_encoder as te
from gmlx.video.families.h3 import family as hf

SPEC = hf.FAMILY.encoder()
N_VOCAB = 151676


@pytest.fixture(scope="module")
def tok():
    from tokenizers import pre_tokenizers

    from gmlx.load.tokenizer import load_tokenizer_from_gguf

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tokens = alphabet + ["He"]
    tokens += [f"[PAD{i}]" for i in range(len(tokens), N_VOCAB)]
    tokens[151643] = "<|endoftext|>"
    types = [1] * (len(alphabet) + 1) + [5] * (N_VOCAB - len(alphabet) - 1)
    types[151643] = 3
    return load_tokenizer_from_gguf({
        "general.architecture": "qwen3vl",
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen2",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.merges": ["H e"],
        "tokenizer.ggml.token_type": types,
        "tokenizer.ggml.eos_token_id": 151643,
        "tokenizer.ggml.padding_token_id": 151643,
    }, "qwen3vl")


def _enc(tok, text):
    return tok.encode(text, add_special_tokens=False)


def test_the_seven_tokens_take_their_fixed_ids_between_the_pieces(tok):
    prompt = ("He says <d>[English] Hello.</d> then sings "
              "<|lyrics_start|>la la<|lyrics_end|><|cutoff|>")
    assert te.token_ids(SPEC, tok, prompt) == (
        _enc(tok, "He says ") + [151669] + _enc(tok, "[English] Hello.")
        + [151670] + _enc(tok, " then sings ") + [151672]
        + _enc(tok, "la la") + [151673, 151671])
    for text, i in hf.ADDED_TOKENS:
        assert tok.convert_ids_to_tokens(i) == f"[PAD{i}]"
        assert i not in _enc(tok, text)
        assert te.token_ids(SPEC, tok, text) == [i]


def test_a_prompt_without_the_tokens_is_the_tokenizers_own(tok):
    text = "He walks in the rain, umbrella up"
    assert te.token_ids(SPEC, tok, text) == _enc(tok, text)


def test_an_empty_prompt_is_the_pad_token(tok):
    assert te.token_ids(SPEC, tok, "") == [151643]
    no_empty = dataclasses.replace(SPEC, empty_ids=())
    with pytest.raises(ValueError, match="has no tokens"):
        te.token_ids(no_empty, tok, "")


def test_the_row_bound_holds_every_prompt_and_the_empty_one(tok):
    prompts = ["", "He", "He says <d>[English] Hello.</d>",
               "<|cutoff|>", "cafe\u0301 au lait"]
    for p in prompts:
        assert te.rows_at_most(SPEC, p) >= len(te.token_ids(SPEC, tok, p))
    assert te.rows_at_most(SPEC, "") == 1
