#!/usr/bin/env python3
"""Structured outputs (`response_format: json_schema`) regression guard.

The feature itself is mlx-vlm's: its server wraps `_build_gen_args` (where the
grammar is compiled) in `try/except -> HTTP 400`, and `build_json_schema_logits_
processor` constrains decoding via llguidance (a hard dep of mlx-vlm). gmlx
edits that path. One edit caches an llguidance tokenizer only as long as its
tokenizer lives, so a reloaded model at a freed id gets its own vocab; the last
test here guards it. The other gmlx-specific risk is that our *synthesized* GGUF tokenizer must
remain acceptable to `llguidance.hf.from_tokenizer` across mlx-vlm/llguidance
bumps - so this freezes that contract:

  synthesized tokenizer -> from_tokenizer accepts it -> a JSON schema compiles
  -> the grammar actually masks the first token.

CPU only, no model/download - the tokenizer is the minimal byte-level BPE used by
test_tokenizer.py, and the masking math runs on the CPU device.
"""
from __future__ import annotations

import pytest

from tokenizers import pre_tokenizers  # noqa: E402

from gmlx.load.tokenizer import load_tokenizer_from_gguf  # noqa: E402

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
    from gmlx._vendor.mlx_vlm.structured import (
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

    from gmlx._vendor.mlx_vlm.structured import build_json_schema_logits_processor

    proc = build_json_schema_logits_processor(_synth_tokenizer(), SCHEMA)
    vocab = proc.llg_tokenizer.vocab_size
    matcher = LLMatcher(proc.llg_tokenizer, proc.grammar)
    assert not matcher.get_error()
    bitmask = llguidance.numpy.allocate_token_bitmask(1, vocab)
    llguidance.numpy.fill_next_token_bitmask(matcher, bitmask, 0)
    n_allowed = int(np.unpackbits(bitmask[0].view(np.uint8)).sum())
    assert 0 < n_allowed < vocab          # grammar forbids most first tokens


class _Tok:
    pass


def test_tokenizer_at_a_used_id_gets_its_own_llg_tokenizer(monkeypatch):
    """A tokenizer at the id of a freed tokenizer gets its own llguidance
    tokenizer, not the freed one's (a reloaded model's grammar vocab). Every
    tokenizer gets the same id here, so the collision does not depend on the
    allocator."""
    import gc

    import llguidance.hf

    from gmlx._vendor.mlx_vlm import structured

    monkeypatch.setattr(structured, "_llg_tokenizer_cache", {})
    keyed = []

    def one_id(obj):
        keyed.append(id(obj))
        return 1

    monkeypatch.setattr(structured, "id", one_id, raising=False)
    built = []

    def from_tokenizer(tok):
        built.append(tok)
        return object()

    monkeypatch.setattr(llguidance.hf, "from_tokenizer", from_tokenizer)

    def llg(tok):
        return structured.build_json_schema_logits_processor(
            tok, SCHEMA).llg_tokenizer

    old = _Tok()
    old_llg = llg(old)
    assert keyed == [id(old)]             # the key goes through the stand-in
    assert llg(old) is old_llg
    new = _Tok()
    new_llg = llg(new)
    assert new_llg is not old_llg
    assert built == [old, new]
    del old, built[:]
    gc.collect()
    assert llg(new) is new_llg
    assert built == []
    del new
    gc.collect()
    assert structured._llg_tokenizer_cache == {}


_REPLY = '{"title": "Hello world", "year": 1999}'


def _mask(proc):
    import llguidance.numpy

    vocab = proc.llg_tokenizer.vocab_size
    bm = llguidance.numpy.allocate_token_bitmask(1, vocab)
    llguidance.numpy.fill_next_token_bitmask(proc.ll_matchers[0], bm, 0)
    return bm.copy()


@pytest.mark.parametrize("tail_only", [False, True], ids=["full", "apc-tail"])
def test_a_resumed_grammar_masks_as_the_live_one_at_every_cut(
        monkeypatch, tail_only):
    """A requeued row replays its prompt plus the tokens it delivered. Live,
    the matcher consumed every delivered token; the replay's first call
    consumes the last one, so the resumed processor holds the rest. The cut
    after '{"title":' made the matcher refuse ':' when the replay fed it a
    second time."""
    pytest.importorskip("llguidance")
    import mlx.core as mx
    import numpy as np

    from gmlx._vendor.mlx_vlm import structured

    # Only the matcher state matters; the Metal mask kernel is not run.
    monkeypatch.setattr(structured, "_apply_llguidance_mask",
                        lambda logits, mask: logits)
    tok = _synth_tokenizer()
    prompt = tok.encode("Give JSON", add_special_tokens=False)
    reply = tok.encode(_REPLY, add_special_tokens=False)
    cuts = list(range(1, len(reply)))
    assert tok.encode('{"title":', add_special_tokens=False) == reply[:9]
    for cut in cuts:
        committed = reply[:cut]
        live = structured.build_json_schema_logits_processor(tok, SCHEMA)
        logits = mx.zeros((1, live.llg_tokenizer.vocab_size))
        live(mx.array(prompt), logits)
        for t in committed:
            live.process_last_token(t, logits)
        replay = live.resumed(committed)
        ctx = committed[-1:] if tail_only else prompt + committed
        replay(mx.array(ctx), logits)
        assert np.array_equal(_mask(replay), _mask(live)), cut


def test_a_resumed_grammar_with_nothing_delivered_starts_afresh():
    pytest.importorskip("llguidance")
    from gmlx._vendor.mlx_vlm.structured import (
        build_json_schema_logits_processor,
    )

    proc = build_json_schema_logits_processor(_synth_tokenizer(), SCHEMA)
    fresh = proc.resumed([])
    assert fresh is not proc and fresh.is_first_token
