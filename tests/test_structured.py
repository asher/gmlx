#!/usr/bin/env python3
"""Structured outputs (`response_format: json_schema`) regression guard.

The feature itself is mlx-vlm's: its server wraps `_build_gen_args` (where the
grammar is compiled) in `try/except -> HTTP 400`, and `build_json_schema_logits_
processor` constrains decoding via llguidance (a hard dep of mlx-vlm). gmlx
adds no code on the happy path. The one gmlx-specific risk is that our
*synthesized* GGUF tokenizer must remain acceptable to `llguidance.hf.from_
tokenizer` across mlx-vlm/llguidance bumps - so this freezes that contract:

  synthesized tokenizer -> from_tokenizer accepts it -> a JSON schema compiles
  -> the grammar actually masks the first token.

CPU only, no model/download - the tokenizer is the minimal byte-level BPE used by
test_tokenizer.py, and the masking math runs on the CPU device.
"""
from __future__ import annotations

import pytest

from tokenizers import pre_tokenizers  # noqa: E402

from gmlx.tokenizer import load_tokenizer_from_gguf  # noqa: E402

# A self-contained minimal ByteLevel BPE (256 byte-alphabet tokens + specials +
# a couple of merges), identical in spirit to test_tokenizer.py - no download.
_SPECIALS = ["<s>", "</s>", "<pad>"]
_ALPHABET = sorted(pre_tokenizers.ByteLevel.alphabet())
_MERGED = ["He", "wo"]
_MERGES = ["H e", "w o"]

SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "year": {"type": "integer"}},
    "required": ["title", "year"],
    "additionalProperties": False,
}


def _synth_tokenizer():
    toks = _SPECIALS + _ALPHABET + _MERGED
    token_type = [3, 3, 3] + [1] * (len(toks) - 3)
    meta = {
        "general.architecture": "qwen2",
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen2",
        "tokenizer.ggml.tokens": toks,
        "tokenizer.ggml.merges": _MERGES,
        "tokenizer.ggml.token_type": token_type,
        "tokenizer.ggml.bos_token_id": 0,
        "tokenizer.ggml.eos_token_id": 1,
        "tokenizer.ggml.padding_token_id": 2,
    }
    return load_tokenizer_from_gguf(meta, "qwen2")


def test_llguidance_accepts_synthesized_tokenizer():
    """The adapter llguidance.hf.from_tokenizer must accept our GGUF-synthesized
    tokenizer - the single gmlx-specific dependency for json_schema."""
    llgh = pytest.importorskip("llguidance.hf")
    llg_tok = llgh.from_tokenizer(_synth_tokenizer())
    assert llg_tok.vocab_size > 0


def test_json_schema_processor_builds_against_synth_tokenizer():
    """A JSON schema compiles into a logits processor against our tokenizer (no
    raise) - the path mlx-vlm's server takes for response_format=json_schema."""
    pytest.importorskip("llguidance")
    from mlx_vlm.structured import (
        LLGuidanceLogitsProcessor,
        build_json_schema_logits_processor,
    )
    proc = build_json_schema_logits_processor(_synth_tokenizer(), SCHEMA)
    assert isinstance(proc, LLGuidanceLogitsProcessor)


def test_grammar_masks_first_token():
    """The compiled grammar actually constrains: its first-token bitmask is a
    strict, non-empty subset of the vocab (a JSON object must open with `{`).

    Uses llguidance's CPU numpy bitmask path - the same matcher the processor
    drives, minus the GPU-only `llguidance.mlx.apply_token_bitmask`, so it stays
    in the CPU suite."""
    import llguidance.numpy
    import numpy as np
    from llguidance import LLMatcher

    from mlx_vlm.structured import build_json_schema_logits_processor

    proc = build_json_schema_logits_processor(_synth_tokenizer(), SCHEMA)
    vocab = proc.llg_tokenizer.vocab_size
    matcher = LLMatcher(proc.llg_tokenizer, proc.grammar)
    assert not matcher.get_error()
    bitmask = llguidance.numpy.allocate_token_bitmask(1, vocab)
    llguidance.numpy.fill_next_token_bitmask(matcher, bitmask, 0)
    n_allowed = int(np.unpackbits(bitmask[0].view(np.uint8)).sum())
    assert 0 < n_allowed < vocab          # grammar forbids most first tokens
