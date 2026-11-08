"""Synthetic GGUF tokenizers for the think-marker tests. No model needed.

The vocab is the byte-level alphabet, plus a row's single-token strings as
control tokens and a row's BPE merges, built by the loader's own
``load_tokenizer_from_gguf``. A merge lets a test reproduce a real vocab
that encodes a marker with other pieces inside a prompt than alone (the
Gemma vocab encodes ``<think></think>`` with the piece ``></``).
"""

from __future__ import annotations

from tokenizers import pre_tokenizers

from gmlx.load.tokenizer import finish_gguf_tokenizer, load_tokenizer_from_gguf

ALPHABET = sorted(pre_tokenizers.ByteLevel.alphabet())
BASE = ("<s>", "</s>", "<pad>")
# The loader builds a byte-level BPE only for a vocab with merges.
_BASE_MERGES = ("H e",)


def tokenizer(template, *, singles=(), merges=(), model_type=None):
    """A raw HF fast tokenizer as the GGUF loader builds it. ``merges``
    are GGUF merge lines ("a b"); each result joins the vocab."""
    merges = list(dict.fromkeys(_BASE_MERGES + tuple(merges)))
    specials = list(BASE) + [s for s in dict.fromkeys(singles) if s not in BASE]
    merged = [t for t in dict.fromkeys(m.replace(" ", "") for m in merges)
              if t not in specials and t not in ALPHABET]
    toks = specials + ALPHABET + merged
    meta = {
        "general.architecture": "qwen2",
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen2",
        "tokenizer.ggml.tokens": toks,
        "tokenizer.ggml.merges": merges,
        "tokenizer.ggml.token_type": [3] * len(specials)
        + [1] * (len(ALPHABET) + len(merged)),
        "tokenizer.ggml.bos_token_id": 0,
        "tokenizer.ggml.eos_token_id": 1,
        "tokenizer.ggml.padding_token_id": 2,
    }
    if template is not None:
        meta["tokenizer.chat_template"] = template
    raw = load_tokenizer_from_gguf(meta, "qwen2")
    finish_gguf_tokenizer(raw, model_type)
    return raw


def text_processor(raw):
    """The processor that text and MTP serve build: the raw backend as
    ``.tokenizer``, other names forwarded to mlx-lm's wrapper."""
    from gmlx._vendor.mlx_lm.tokenizer_utils import TokenizerWrapper
    from gmlx.serve.bridge_vlm import _make_text_processor

    return _make_text_processor(TokenizerWrapper(
        raw, eos_token_ids=getattr(raw, "_gguf_eos_token_ids", None)))


def ids(tok, text) -> list[int]:
    return [int(t) for t in tok.encode(text, add_special_tokens=False)]
