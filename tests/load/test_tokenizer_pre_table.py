"""The byte-level pre-tokenizer table: each `tokenizer.ggml.pre` name gets
llama.cpp's split regexes and flags.

`data/llama_cpp_pre.json` is a snapshot of llama.cpp's table, written by
`scripts/llama_pre_snapshot.py`. `data/pre_pieces.json` holds probe texts and
the pieces each pattern set gives them under the `regex` module, checked
against llama-tokenize ids before they were frozen. CPU-only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from tokenizers import Regex, pre_tokenizers

from gmlx.load import tokenizer as tk
from gmlx.load.tokenizer import (
    GPT2_PATTERN,
    LLAMA3_PATTERN,
    O200K_PATTERN,
    PRE_TOKENIZERS,
    QWEN2_PATTERN,
    QWEN35_PATTERN,
    TEKKEN_NUM3_PATTERN,
    TEKKEN_PATTERN,
    load_tokenizer_from_gguf,
    pre_tokenizer,
)

_DATA = Path(__file__).parent / "data"
_SNAPSHOT = json.loads((_DATA / "llama_cpp_pre.json").read_text())
_PIECES = json.loads((_DATA / "pre_pieces.json").read_text())

# Names llama.cpp knows that the table leaves to the fallback and warning.
UNLISTED_PRES = {
    "default": "llama.cpp's own fallback regex set, no gmlx spelling",
    "deepseek-coder": "regex set with no gmlx spelling",
    "deepseek-llm": "regex set with no gmlx spelling",
    "falcon": "regex set with no gmlx spelling",
    "bloom": "architecture gmlx cannot load",
    "poro-chat": "regex set with no gmlx spelling",
    "gpt3-finnish": "regex set with no gmlx spelling",
    "viking": "regex set with no gmlx spelling",
    "chameleon": "regex set with no gmlx spelling",
    "afmoe": "regex set with no gmlx spelling",
    "youtu": "regex set with no gmlx spelling",
    "superbpe": "regex set with no gmlx spelling",
    "spark2_5": "regex set with no gmlx spelling",
    "minicpm5": "regex set with no gmlx spelling",
    "laguna": "line-split regex set with no gmlx spelling",
    "jais-2": "regex set with no gmlx spelling",
    "tiny_aya": "regex set with no gmlx spelling",
    "cohere2moe": "regex set with no gmlx spelling",
    "bailingmoe": "regex set with no gmlx spelling",
    "bailingmoe2": "regex set with no gmlx spelling",
    "llada-moe": "regex set with no gmlx spelling",
    "seed-coder": "regex set with no gmlx spelling",
    "ufakzeka": "regex set with no gmlx spelling",
    "exaone-moe": "regex set with no gmlx spelling",
    "granite-embed-multi-97m": "embedding model gmlx cannot load",
    "nemotron-3.5-embed": "embedding model gmlx cannot load",
    "gemma4": "SPM-style vocab, not byte-level",
    "granite-embed-multi-311m": "SPM-style vocab, not byte-level",
    "sarvam-moe": "SPM-style vocab, not byte-level",
    "domyn-small": "SPM-style vocab, not byte-level",
    "mmbert": "SPM-style vocab, not byte-level",
    "soprano": "raw UTF-8 vocab, not byte-level",
    "whitespace": "raw UTF-8 vocab, not byte-level",
}
# Table names llama.cpp does not list, with the name whose entry they share.
ALIASES = {"qwen3": "qwen2", "qwen35moe": "qwen35", "hyv4": "hy_v4"}
# Entries whose regex is not llama.cpp's spelling of the same split.
OWN_SPELLING = {
    "kimi-k2": "llama.cpp runs a hand-coded splitter behind a stub regex",
}

_GROUPS = [
    ("qwen2 qwen3 deepseek-r1-qwen f2llmv2 kormo megrez grok-2 hunyuan "
     "solar-open stablelm2", (QWEN2_PATTERN,), False, None),
    ("qwen35 qwen35moe", (QWEN35_PATTERN,), False, None),
    ("llama-bpe llama3 llama-v3 falcon3 falcon-h1 midm-2.0 jina-v5-nano lfm2",
     (LLAMA3_PATTERN,), True, True),
    ("dbrx smaug-bpe chatglm-bpe", (LLAMA3_PATTERN,), False, None),
    ("glm4 glm5", (LLAMA3_PATTERN,), True, None),
    ("gpt-2 olmo mpt jais trillion granite-docling phi-2 exaone4 gigachat "
     "a.x-4.0 mellum modern-bert roberta-bpe jina-de jina-es jina-v1-en "
     "jina-v2-code jina-v2-de jina-v2-es", (GPT2_PATTERN,), False, None),
    ("smollm starcoder codeshell command-r exaone refact minerva-7b mellum2",
     (r"\p{N}", GPT2_PATTERN), False, None),
    ("tekken pixtral", (TEKKEN_PATTERN,), True, True),
    ("tekken-num3", (TEKKEN_NUM3_PATTERN,), True, True),
    ("gpt-4o llama4 kanana2 talkie minimax-m2", (O200K_PATTERN,), False, None),
]


@pytest.mark.parametrize("names,patterns,ignore_merges,add_bos", _GROUPS,
                         ids=[g[0].split()[0] for g in _GROUPS])
def test_each_name_selects_its_patterns_and_flags(names, patterns,
                                                  ignore_merges, add_bos):
    for name in names.split():
        entry = pre_tokenizer(name)
        assert entry.patterns == patterns, name
        assert (entry.ignore_merges, entry.add_bos) == (ignore_merges,
                                                         add_bos), name


@pytest.mark.parametrize("name", ["some-future-pre", ""])
def test_an_unknown_name_falls_back_to_the_qwen2_split(name):
    entry = pre_tokenizer(name)
    assert entry.patterns == (QWEN2_PATTERN,)
    assert (entry.ignore_merges, entry.add_bos, entry.nfc) == (False, None,
                                                               False)


def test_every_llama_cpp_name_is_listed_or_unlisted_with_a_reason():
    names = set(_SNAPSHOT["names"])
    assert not set(UNLISTED_PRES) & set(PRE_TOKENIZERS)
    assert names - set(UNLISTED_PRES) - set(PRE_TOKENIZERS) == set()
    assert set(UNLISTED_PRES) <= names
    assert set(PRE_TOKENIZERS) - names == set(ALIASES)
    for alias, name in ALIASES.items():
        assert PRE_TOKENIZERS[alias] == PRE_TOKENIZERS[name]


def _canonical(pattern: str) -> str:
    """One spelling for llama.cpp's and gmlx's forms of the same regex."""
    pattern = pattern.replace("\r", r"\r").replace("\n", r"\n")
    pattern = pattern.replace(
        "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])",
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)")
    pattern = pattern.replace(r"((?=[\p{L}\p{M}])([^a-z]))",
                              r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]")
    pattern = pattern.replace(r"((?=[\p{L}\p{M}])([^A-Z]))",
                              r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]")
    return re.sub(r"\\x\{([0-9A-Fa-f]{4})\}", r"\\u\1", pattern)


@pytest.mark.parametrize("name", sorted(_SNAPSHOT["names"]))
def test_each_entry_matches_the_llama_cpp_snapshot(name):
    if name not in PRE_TOKENIZERS:
        return
    snap = _SNAPSHOT["names"][name]
    entry = PRE_TOKENIZERS[name]
    assert entry.ignore_merges == snap["ignore_merges"]
    assert entry.add_bos == (True if snap["add_bos"] else None)
    if name in OWN_SPELLING:
        return
    lcpp = [_canonical(p) for p in _SNAPSHOT["regex"][snap["pre_type"]]]
    assert [_canonical(p) for p in entry.patterns] == lcpp


def _split(patterns, text):
    seq = pre_tokenizers.Sequence(
        [pre_tokenizers.Split(Regex(p), behavior="isolated")
         for p in patterns])
    return [p for p, _ in seq.pre_tokenize_str(text)]


@pytest.mark.parametrize("name", sorted(_PIECES["sets"]))
def test_the_tokenizers_engine_splits_as_the_frozen_pieces(name):
    frozen = _PIECES["sets"][name]
    patterns = pre_tokenizer(frozen["pre"]).patterns
    for probe, text in _PIECES["probes"].items():
        assert _split(patterns, text) == frozen["pieces"][probe], probe


_SPECIALS = ["<s>", "</s>", "<pad>"]
_ALPHABET = sorted(pre_tokenizers.ByteLevel.alphabet())


def _meta(pre: str, **extra) -> dict:
    tokens = _SPECIALS + _ALPHABET + ["he", "ll", "lo", "hello"]
    meta = {
        "general.architecture": "llama",
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": pre,
        "tokenizer.ggml.tokens": tokens,
        # No merge builds "hello" from "he" and "ll".
        "tokenizer.ggml.merges": ["h e", "l l", "l o"],
        "tokenizer.ggml.token_type": [3, 3, 3] + [1] * (len(tokens) - 3),
        "tokenizer.ggml.bos_token_id": 0,
        "tokenizer.ggml.eos_token_id": 1,
    }
    meta.update({f"tokenizer.ggml.{k}": v for k, v in extra.items()})
    return meta


@pytest.mark.parametrize("pre,key,first_is_bos", [
    ("llama-bpe", None, True),
    ("pixtral", None, True),
    ("llama-bpe", False, False),
    ("qwen2", None, False),
    ("qwen2", True, True),
])
def test_a_missing_add_bos_key_takes_the_entry_default(pre, key,
                                                       first_is_bos):
    extra = {} if key is None else {"add_bos_token": key}
    tok = load_tokenizer_from_gguf(_meta(pre, **extra), "llama")
    assert (tok.encode("hello")[0] == 0) is first_is_bos
    assert tok._gguf_add_bos_token is first_is_bos


@pytest.mark.parametrize("pre,ids", [("llama-bpe", 1), ("glm4", 1),
                                     ("tekken", 1), ("qwen2", 3),
                                     ("gpt-2", 3)])
def test_ignore_merges_emits_a_whole_vocab_word_as_one_id(pre, ids):
    tok = load_tokenizer_from_gguf(_meta(pre), "llama")
    out = tok.encode("hello", add_special_tokens=False)
    assert len(out) == ids
    assert tok.decode(out) == "hello"


# Names whose reference tokenizer.json declares an NFC normalizer.
NFC_PRES = {"qwen2", "qwen3", "qwen35", "qwen35moe", "deepseek-r1-qwen",
            "k2-horizon"}


@pytest.mark.parametrize("pre", sorted(PRE_TOKENIZERS) + ["", "unknown"])
def test_nfc_runs_only_where_the_entry_sets_it(pre):
    assert pre_tokenizer(pre).nfc is (pre in NFC_PRES)
    built = tk._build_bytelevel_bpe(["a", "b"], [], pre_id=pre)
    assert (built.normalizer is not None) is (pre in NFC_PRES)


@pytest.mark.parametrize("pre,nfc", [("llama-bpe", False), ("qwen35", True)])
def test_a_decomposed_accent_keeps_its_ids_without_nfc(pre, nfc):
    tok = load_tokenizer_from_gguf(_meta(pre), "llama")
    decomposed = tok.encode("e\u0301", add_special_tokens=False)
    composed = tok.encode("\u00e9", add_special_tokens=False)
    assert (decomposed == composed) is nfc
    assert tok.decode(decomposed) == ("\u00e9" if nfc else "e\u0301")


@pytest.mark.parametrize("pre", ["some-future-pre", ""])
def test_an_unknown_or_empty_name_warns_once(pre, capsys):
    load_tokenizer_from_gguf(_meta(pre), "llama")
    err = capsys.readouterr().err
    assert err.count("unknown pre-tokenizer") == 1
    assert err.count(repr(pre)) == 1


@pytest.mark.parametrize("pre", sorted(PRE_TOKENIZERS))
def test_a_listed_name_does_not_warn(pre, capsys):
    load_tokenizer_from_gguf(_meta(pre), "llama")
    assert "pre-tokenizer" not in capsys.readouterr().err
