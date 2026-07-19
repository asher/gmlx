#!/usr/bin/env python3
"""Tokenizer synthesis: round-trip + dual-mode (GGUFReader vs decoded-dict).

``load_tokenizer_from_gguf`` reads ~10 GGUF KV fields through helpers that have
two code paths (a gguf-py reader vs the decoded dict from ``kq.load_gguf``). The
regression risk is those paths drifting, so the core test builds a tokenizer
both ways from the *same* values and asserts identical encode/decode. The vocab
is a self-contained minimal ByteLevel BPE (the 256 byte-alphabet tokens + a few
merges) so it round-trips any ASCII text with no model download - CI-able.
"""

from __future__ import annotations

import numpy as np

from tokenizers import pre_tokenizers  # noqa: E402

from gmlx.tokenizer import load_tokenizer_from_gguf  # noqa: E402

# Special tokens occupy ids 0..2; the byte alphabet follows.
_SPECIALS = ["<s>", "</s>", "<pad>"]
_ALPHABET = sorted(pre_tokenizers.ByteLevel.alphabet())
# A couple of merged tokens (+ their merges) so _classify sees has_merges=True
# and routes to the bytelevel builder.
_MERGED = ["He", "wo"]
_MERGES = ["H e", "w o"]


def _tokens():
    return _SPECIALS + _ALPHABET + _MERGED


def _bytelevel_meta() -> dict:
    toks = _tokens()
    token_type = [3, 3, 3] + [1] * (len(toks) - 3)  # specials are control type
    return {
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


SAMPLES = ["Hello, world!", "def f(x): return x*2", "two  spaces"]


def test_bytelevel_roundtrip_and_specials():
    tok = load_tokenizer_from_gguf(_bytelevel_meta(), "qwen2")
    assert tok.bos_token == "<s>" and tok.eos_token == "</s>"
    assert tok.pad_token == "<pad>"
    for s in SAMPLES:
        ids = tok.encode(s, add_special_tokens=False)
        assert tok.decode(ids, skip_special_tokens=False) == s


# llama.cpp stores CONTROL token text raw while merges stay in byte-level
# space. MiniMax-M3's unk is the raw U+FFFD bytes AND a merge product (its
# byte-level spelling "ï¿½"), which broke BPE init with "out of vocabulary"
# until the builder started dropping merges whose parts/product aren't vocab
# entries (llama.cpp can never resolve those merges to an id either). The
# fixture reproduces that shape: a raw "�" control token whose byte-level
# spelling only exists as a merge product.
def test_bytelevel_raw_control_token_merge_dropped():
    toks = _tokens() + ["ï¿", "�"]
    token_type = [3, 3, 3] + [1] * (len(toks) - 4) + [3]  # raw unk is control
    unk_id = len(toks) - 1
    meta = {
        "general.architecture": "qwen2",
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen2",
        "tokenizer.ggml.tokens": toks,
        "tokenizer.ggml.merges": _MERGES + ["ï ¿", "ï¿ ½"],
        "tokenizer.ggml.token_type": token_type,
        "tokenizer.ggml.bos_token_id": 0,
        "tokenizer.ggml.eos_token_id": 1,
        "tokenizer.ggml.unknown_token_id": unk_id,
    }
    tok = load_tokenizer_from_gguf(meta, "qwen2")  # BPE init must not raise
    # The raw control token matches atomically (AddedToken), at its VOCAB id -
    # a fresh added-token id here would be out of the model's embedding range.
    ids = tok.encode("�", add_special_tokens=False)
    assert ids == [unk_id]
    assert tok.decode(ids, skip_special_tokens=False) == "�"
    # The resolvable merge still applies; ordinary text is unaffected.
    ids = tok.encode("Hello", add_special_tokens=False)
    assert all(0 <= i < len(toks) for i in ids)
    assert tok.decode(ids) == "Hello"


# Digit grouping is the one pre-tokenizer clause that differs across byte-level
# families. "10" + a "1 0" merge only collapses to a single id when the
# pretokenizer keeps "10" in one chunk; Llama-3 (\p{N}{1,3}) does, Qwen (\p{N},
# every digit isolated) does not. Add a "10" token + its merge to the fixture.
def _digit_meta(pre: str) -> dict:
    toks = _tokens() + ["10"]
    token_type = [3, 3, 3] + [1] * (len(toks) - 3)
    return {
        "general.architecture": "llama" if "llama" in pre else "qwen2",
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": pre,
        "tokenizer.ggml.tokens": toks,
        "tokenizer.ggml.merges": _MERGES + ["1 0"],
        "tokenizer.ggml.token_type": token_type,
        "tokenizer.ggml.bos_token_id": 0,
        "tokenizer.ggml.eos_token_id": 1,
        "tokenizer.ggml.padding_token_id": 2,
    }


def test_llama3_groups_multidigit_numbers():
    # Regression guard: Llama-3's \p{N}{1,3} keeps "10" one pretoken so the
    # "1 0" merge fires -> a single id (matches llama.cpp/HF). The bug routed
    # llama-bpe through Qwen's single-digit clause and over-split it.
    tok = load_tokenizer_from_gguf(_digit_meta("llama-bpe"), "llama")
    ten = tok.convert_tokens_to_ids("10")
    ids = tok.encode("10", add_special_tokens=False)
    assert ids == [ten]
    assert tok.decode(ids, skip_special_tokens=False) == "10"


def test_qwen_isolates_each_digit():
    # Qwen's \p{N} keeps each digit its own pretoken, so "1 0" never merges
    # -> two ids. Qwen's deliberate arithmetic tokenization, preserved.
    tok = load_tokenizer_from_gguf(_digit_meta("qwen2"), "qwen2")
    ten = tok.convert_tokens_to_ids("10")
    ids = tok.encode("10", add_special_tokens=False)
    assert ids != [ten] and len(ids) == 2
    assert tok.decode(ids, skip_special_tokens=False) == "10"


def test_digit_clause_selected_by_pre():
    # The selector picks the family-correct digit clause from the GGUF pre hint;
    # unlisted pres keep the historical single-digit default (no regression).
    from gmlx.tokenizer import _bytelevel_split_patterns
    for pre in ("llama-bpe", "llama3"):
        assert r"\p{N}{1,3}" in _bytelevel_split_patterns(pre)[0]
    assert r"\p{N}+" in _bytelevel_split_patterns("gpt-2")[0]
    for pre in ("qwen2", "qwen3", "qwen35", "some-future-pre"):
        assert r"\p{N}{1,3}" not in _bytelevel_split_patterns(pre)[0]
    # MiniMax (M2 + M3 GGUFs both ship pre='minimax-m2'): llama.cpp's
    # PRE_TYPE_MINIMAX_M2 regex is the o200k pattern verbatim - full-pattern
    # override, not a digit-clause swap (16k parity fails 15492-vs-16384
    # prompt tokens on the digit grouping alone without it).
    from gmlx.tokenizer import O200K_PATTERN
    assert _bytelevel_split_patterns("minimax-m2") == (O200K_PATTERN,)


def test_deepseek3_pre_is_multi_regex_sequence():
    # llama.cpp's DEEPSEEK3_LLM case applies THREE regexes sequentially (the
    # same case covers pre='deepseek-v3', 'joyai-llm' [DeepSeek V4 Flash], and
    # 'hunyuan-dense'). The selector must return the tuple, and the builder
    # must chain one Split per pattern (mirroring sequential splitting).
    from gmlx.tokenizer import _DEEPSEEK3_PATTERNS, _bytelevel_split_patterns
    for pre in ("deepseek-v3", "joyai-llm", "hunyuan-dense"):
        assert _bytelevel_split_patterns(pre) == _DEEPSEEK3_PATTERNS
    assert len(_DEEPSEEK3_PATTERNS) == 3
    # Verified against ds4 --dump-tokens on the real V4 Flash GGUF (exact-id
    # parity on digit/CJK/contraction/punctuation corpora, 2026-07).
    tok = load_tokenizer_from_gguf(_digit_meta("joyai-llm"), "deepseek4")
    # \p{N}{1,3} keeps "10" one pretoken -> the "1 0" merge fires (the old
    # fallback sent deepseek-v3 through the single-digit clause: 2 ids).
    ten = tok.convert_tokens_to_ids("10")
    assert tok.encode("10", add_special_tokens=False) == [ten]
    # Sequential splitting: the CJK clause isolates kanji runs from ASCII, and
    # the punctuation clause splits "..." off words - each piece round-trips.
    for text in ("abc日本語10", "Hello... 999"):
        ids = tok.encode(text, add_special_tokens=False)
        assert tok.decode(ids, skip_special_tokens=False) == text


def test_add_bos_token_prepends_on_raw_path_only():
    # add_bos_token=True -> encode() prepends BOS with the default
    # add_special_tokens=True (raw-completion parity with llama.cpp), but
    # add_special_tokens=False (the chat-path encode convention) does not, so a
    # template that already carries a literal BOS never doubles up.
    meta = _bytelevel_meta()
    meta["tokenizer.ggml.add_bos_token"] = True
    tok = load_tokenizer_from_gguf(meta, "qwen2")
    bos = tok.bos_token_id
    assert getattr(tok, "_gguf_add_bos_token", False) is True
    assert tok.encode("Hello", add_special_tokens=True)[0] == bos
    assert tok.encode("Hello", add_special_tokens=False)[0] != bos
    # a leading literal BOS + add_special_tokens=False stays single-BOS
    doubled = tok.encode(tok.bos_token + "Hello", add_special_tokens=False)
    assert sum(1 for t in doubled if t == bos) == 1


def test_no_add_bos_when_flag_absent():
    # The fixture omits tokenizer.ggml.add_bos_token -> default off; encode()
    # leaves BOS to the caller / chat template (matches qwen-style tokenizers).
    tok = load_tokenizer_from_gguf(_bytelevel_meta(), "qwen2")
    assert getattr(tok, "_gguf_add_bos_token", None) is False
    assert tok.encode("Hello", add_special_tokens=True)[0] != tok.bos_token_id


def test_chat_template_override_is_honored():
    tmpl = "{% for m in messages %}<<{{m['role']}}>>{{m['content']}}{% endfor %}"
    tok = load_tokenizer_from_gguf(_bytelevel_meta(), "qwen2",
                                   chat_template_override=tmpl)
    assert tok.chat_template == tmpl
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}], tokenize=False)
    assert "<<user>>hi" in rendered


def test_render_once_then_encode_matches_tokenized_template():
    # chat renders the template once (tokenize=False) and encodes the string;
    # this must be token-identical to apply_chat_template(tokenize=True) on
    # the mlx-lm TokenizerWrapper chat actually holds (render + encode with
    # add_special_tokens=False internally).
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    tmpl = ("{% for m in messages %}<s>{{ m['role'] }}: {{ m['content'] }}"
            "</s>{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}")
    tok = TokenizerWrapper(load_tokenizer_from_gguf(
        _bytelevel_meta(), "qwen2", chat_template_override=tmpl))
    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "Hello, world!"},
        {"role": "assistant", "content": "wo"},
        {"role": "user", "content": "two  spaces"},
    ]
    direct = tok.apply_chat_template(messages, add_generation_prompt=True)
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    assert tok.encode(text, add_special_tokens=False) == direct


def test_reader_and_dict_paths_agree(tmp_path):
    """A tokenizer built from a GGUFReader must encode/decode identically to one
    built from the decoded-dict metadata - the dual-mode helper contract."""
    from gguf import GGUFWriter, GGUFReader

    meta = _bytelevel_meta()
    meta["tokenizer.ggml.add_bos_token"] = True
    p = tmp_path / "tok.gguf"
    w = GGUFWriter(str(p), "qwen2")  # sets general.architecture
    w.add_bool("tokenizer.ggml.add_bos_token", True)  # a real GGUF BOOL field
    w.add_string("tokenizer.ggml.model", meta["tokenizer.ggml.model"])
    w.add_string("tokenizer.ggml.pre", meta["tokenizer.ggml.pre"])
    w.add_array("tokenizer.ggml.tokens", meta["tokenizer.ggml.tokens"])
    w.add_array("tokenizer.ggml.merges", meta["tokenizer.ggml.merges"])
    w.add_array("tokenizer.ggml.token_type", meta["tokenizer.ggml.token_type"])
    w.add_uint32("tokenizer.ggml.bos_token_id", 0)
    w.add_uint32("tokenizer.ggml.eos_token_id", 1)
    w.add_uint32("tokenizer.ggml.padding_token_id", 2)
    # GGUFWriter needs at least one tensor to produce a valid file.
    w.add_tensor("token_embd.weight",
                 np.zeros((len(meta["tokenizer.ggml.tokens"]), 8), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    from_dict = load_tokenizer_from_gguf(meta, "qwen2")
    from_reader = load_tokenizer_from_gguf(GGUFReader(str(p), "r"), "qwen2")

    assert from_dict.get_vocab() == from_reader.get_vocab()
    assert (from_dict.bos_token_id, from_dict.eos_token_id) == \
        (from_reader.bos_token_id, from_reader.eos_token_id)
    # add_bos_token arrived as a real GGUF BOOL: the reader-path tokenizer
    # prepends BOS on the raw path, same observable as the dict-path tests.
    assert from_reader.encode("Hello", add_special_tokens=True)[0] == \
        from_reader.bos_token_id
    assert from_dict.encode("Hello", add_special_tokens=True)[0] == \
        from_dict.bos_token_id
    for s in SAMPLES:
        a = from_dict.encode(s, add_special_tokens=False)
        b = from_reader.encode(s, add_special_tokens=False)
        assert a == b, s
        assert from_dict.decode(a) == from_reader.decode(b) == s


# --chat-template resolution: inline string, file path, and the two
# silent-garbage paths (mistyped file path, malformed Jinja) that must raise.
def test_resolve_chat_template_inline_and_file(tmp_path):
    from gmlx.loader import _resolve_chat_template

    inline = "{{ messages[0]['content'] }}"
    assert _resolve_chat_template(None) is None
    assert _resolve_chat_template(inline) == inline
    p = tmp_path / "tmpl.jinja"
    p.write_text(inline)
    assert _resolve_chat_template(str(p)) == inline


def test_resolve_chat_template_path_typo_raises():
    import pytest

    from gmlx.loader import _resolve_chat_template

    with pytest.raises(ValueError, match="not found"):
        _resolve_chat_template("/no/such/template.jinja")


def test_resolve_chat_template_bad_jinja_raises():
    import pytest

    from gmlx.loader import _resolve_chat_template

    with pytest.raises(ValueError, match="not valid Jinja"):
        _resolve_chat_template("{% if %}")


# End-of-generation stop set: llama.cpp folds eot/eom/FIM-rep/sep/pad token ids
# (declared in GGUF metadata) into its EOG set on top of eos_token_id. The
# turn-end token is frequently NOT the eos - GLM declares <|user|> as
# eot_token_id - so a loader that stops only on eos runs past the model's own
# turn boundary into degenerate looping output.
def test_metadata_stop_ids_reads_eot_eom_fim():
    from gmlx.tokenizer import _metadata_stop_ids

    meta = {
        "tokenizer.ggml.eot_token_id": 154827,   # GLM <|user|>
        "tokenizer.ggml.eom_token_id": 200,
        "tokenizer.ggml.fim_rep_token_id": 7,
        "tokenizer.ggml.fim_sep_token_id": 8,
        "tokenizer.ggml.fim_pad_token_id": 9,
    }
    assert _metadata_stop_ids(meta) == [154827, 200, 7, 8, 9]
    assert _metadata_stop_ids({}) == []                 # nothing declared

    # With a vocab size, out-of-range ids (-1 sentinel, uint32 wraparound) drop,
    # matching _special_id - they must not alias a real vocab row downstream.
    meta_oob = {
        "tokenizer.ggml.eot_token_id": 5,
        "tokenizer.ggml.eom_token_id": -1,
        "tokenizer.ggml.fim_rep_token_id": 999999,
    }
    assert _metadata_stop_ids(meta_oob, 10) == [5]


def test_eot_token_id_folded_into_stop_set():
    # GLM regression: the turn-end token lives in eot_token_id, and nothing
    # trails the assistant turn in a no-generation-prompt render, so the
    # chat-template heuristic alone misses it. The metadata read must catch it.
    meta = _bytelevel_meta()                            # eos=1
    meta["tokenizer.ggml.eot_token_id"] = 2            # a distinct declared stop
    tok = load_tokenizer_from_gguf(meta, "qwen2")
    assert tok._gguf_eos_token_ids[0] == 1             # primary eos stays first
    assert 2 in tok._gguf_eos_token_ids                # eot folded in


def test_eom_and_fim_stop_ids_folded():
    meta = _bytelevel_meta()                            # eos=1
    meta["tokenizer.ggml.eom_token_id"] = 0
    meta["tokenizer.ggml.fim_rep_token_id"] = 2
    tok = load_tokenizer_from_gguf(meta, "qwen2")
    for tid in (0, 1, 2):
        assert tid in tok._gguf_eos_token_ids


def test_stop_set_dedups_when_eot_equals_eos():
    meta = _bytelevel_meta()                            # eos=1
    meta["tokenizer.ggml.eot_token_id"] = 1            # same id as eos
    tok = load_tokenizer_from_gguf(meta, "qwen2")
    assert tok._gguf_eos_token_ids.count(1) == 1       # no duplicate


# add_eos_token: a post-processor appends EOS on the raw path (parity with
# llama.cpp), bypassed on the chat path (add_special_tokens=False) so a template
# carrying its own EOS never doubles up - symmetric with add_bos_token.
def test_add_eos_token_appends_on_raw_path_only():
    meta = _bytelevel_meta()
    meta["tokenizer.ggml.add_eos_token"] = True
    tok = load_tokenizer_from_gguf(meta, "qwen2")
    eos = tok.eos_token_id
    assert getattr(tok, "_gguf_add_eos_token", False) is True
    assert tok.encode("Hello", add_special_tokens=True)[-1] == eos
    assert tok.encode("Hello", add_special_tokens=False)[-1] != eos


def test_add_bos_and_eos_together_wrap_raw_path():
    meta = _bytelevel_meta()
    meta["tokenizer.ggml.add_bos_token"] = True
    meta["tokenizer.ggml.add_eos_token"] = True
    tok = load_tokenizer_from_gguf(meta, "qwen2")
    ids = tok.encode("Hello", add_special_tokens=True)
    assert ids[0] == tok.bos_token_id and ids[-1] == tok.eos_token_id
    inner = tok.encode("Hello", add_special_tokens=False)  # chat-path convention
    assert inner[0] != tok.bos_token_id and inner[-1] != tok.eos_token_id


# suppress_tokens: folded into logit_bias as a strong negative so they're never
# sampled. The stash must survive mlx-lm's TokenizerWrapper proxy (the real
# footgun - _-prefixed attrs don't proxy), so exercise it through the wrapper.
def test_suppress_tokens_merge_through_tokenizer_wrapper():
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from gmlx.tokenizer import merge_suppressed_tokens

    meta = _bytelevel_meta()
    meta["tokenizer.ggml.suppress_tokens"] = [5, 6]
    raw = load_tokenizer_from_gguf(meta, "qwen2")
    assert raw.gguf_suppress_tokens == [5, 6]

    wrapped = TokenizerWrapper(raw)                     # proxies non-_ attrs
    merged = merge_suppressed_tokens({3: 2.0}, wrapped)
    assert merged[3] == 2.0                             # user bias preserved
    assert merged[5] < -1e6 and merged[6] < -1e6        # suppressed


def test_suppress_tokens_absent_is_noop():
    from gmlx.tokenizer import merge_suppressed_tokens

    tok = load_tokenizer_from_gguf(_bytelevel_meta(), "qwen2")
    assert tok.gguf_suppress_tokens == []
    assert merge_suppressed_tokens(None, tok) is None
    assert merge_suppressed_tokens({3: 1.0}, tok) == {3: 1.0}


# Out-of-range special-token ids: the -1 "unset" sentinel and HF's
# pad == vocab_size convention must read as "no such token", never index the
# vocab (a -1 silently made the LAST vocab entry BOS).
def test_out_of_range_special_ids_treated_as_unset():
    meta = _bytelevel_meta()
    meta["tokenizer.ggml.bos_token_id"] = -1
    meta["tokenizer.ggml.padding_token_id"] = len(_tokens())
    tok = load_tokenizer_from_gguf(meta, "qwen2")
    assert tok.bos_token is None
    assert tok.pad_token is None
    assert tok.eos_token == "</s>"          # in-range ids unaffected


def test_token_type_length_mismatch_tolerated():
    meta = _bytelevel_meta()
    meta["tokenizer.ggml.token_type"] = meta["tokenizer.ggml.token_type"][:-5]
    tok = load_tokenizer_from_gguf(meta, "qwen2")   # used to IndexError
    ids = tok.encode("Hello", add_special_tokens=False)
    assert tok.decode(ids, skip_special_tokens=False) == "Hello"


def test_out_of_range_suppress_tokens_ignored():
    from gmlx.tokenizer import merge_suppressed_tokens
    tok = load_tokenizer_from_gguf(_bytelevel_meta(), "qwen2")
    tok.gguf_suppress_tokens = [3, -1, len(tok) + 5]
    merged = merge_suppressed_tokens(None, tok)
    assert 3 in merged                      # in-range kept
    assert -1 not in merged                 # -1 must not bias the LAST token
    assert len(tok) + 5 not in merged
