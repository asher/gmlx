"""Build a `PreTrainedTokenizerFast` from GGUF tokenizer metadata.

`load_tokenizer_from_gguf(meta, arch)` returns a fully functional fast
tokenizer (BPE) wrapped in `PreTrainedTokenizerFast`, with chat template,
special tokens, and `add_bos_token` honored - the equivalent of running
`AutoTokenizer.from_pretrained(<hf_source>)` but sourced entirely from
GGUF KV metadata.

Two construction paths, both BPE (verified empirically against the HF
tokenizer.json for each family - the design doc's Unigram-for-gemma4
prescription was wrong; gemma-4 also exports BPE with byte_fallback):

  - SPM-style BPE      (gemma4, gemma3; Llama-2/Mistral/Vicuna):
      Replace " " -> ▁ in the normalizer; pre-tokenize by splitting on " "
      (merged_with_previous); decoder reverses ▁ -> " " and applies byte
      fallback. byte_fallback=True in the BPE model. Two sub-cases:
        * explicit merges (gemma) - used as-is.
        * mergeless scored vocab (classic SentencePiece, e.g. Llama-2) -
          merges are reconstructed from `tokenizer.ggml.scores` and a dummy
          ▁ prefix is prepended (sentencepiece add_dummy_prefix), matching
          HF's LlamaTokenizerFast bit-for-bit.

  - ByteLevel BPE      (qwen35, qwen3, llama3, gpt2):
      NFC normalizer; pre-tokenize via GPT-4-style regex split + ByteLevel;
      ByteLevel decoder + post-processor. byte_fallback=False.

No transformers monkey-patching, no vendored converters. Built directly
from `tokenizers` primitives.
"""

from __future__ import annotations


from tokenizers import (
    AddedToken,
    Regex,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
)
from transformers import PreTrainedTokenizerFast

from . import loadlog

# The dual-mode GGUF KV readers (decoded KV dict or gguf-py GGUFReader) live
# in gguf_meta; aliased to keep the vocab-builder call sites short.
from .gguf_meta import (
    read_bool as _read_bool,
    read_float_array as _read_float_array,
    read_int as _read_int,
    read_int_array as _read_int_array,
    read_str_array as _read_str_array,
    read_string as _read_string,
)


# GPT-4-style word-split pattern used by Qwen2/3/3.5, Llama3, and GPT-2 BPE
# pre-tokenizers. Matches contractions, letters, digits, punctuation runs, and
# whitespace separately so byte-level encoding can isolate words. The one clause
# that differs across these families is how runs of digits are grouped, so it's
# a parameter:
#   Qwen2/3   \p{N}       every digit isolated - Qwen's deliberate arithmetic split
#   Llama-3   \p{N}{1,3}  groups of up to 3 digits (cl100k / GPT-4)
#   GPT-2     \p{N}+      the whole digit run
# Sending Llama-3 / GPT-2 through Qwen's single-digit clause over-tokenizes
# numbers ("10" -> "1","0" instead of "10"), degrading math/code/date quality
# (llama.cpp/HF give "10" a single id). The digit clause is selected from the
# GGUF `tokenizer.ggml.pre` hint by :func:`_bytelevel_split_patterns`.
def _bytelevel_pattern(digit_clause: str) -> str:
    return (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
        r"[^\r\n\p{L}\p{N}]?\p{L}+|"
        f"{digit_clause}|"
        r" ?[^\s\p{L}\p{N}]+[\r\n]*|"
        r"\s*[\r\n]+|"
        r"\s+(?!\S)|"
        r"\s+"
    )


# `tokenizer.ggml.pre` hint -> digit-grouping clause. Unlisted byte-level pres
# fall back to Qwen's single-digit clause (the historical default - no behavior
# change for them; only the known multi-digit families are upgraded).
_DIGIT_CLAUSE_BY_PRE = {
    "qwen2": r"\p{N}", "qwen3": r"\p{N}",
    "qwen35": r"\p{N}", "qwen35moe": r"\p{N}",
    "llama-bpe": r"\p{N}{1,3}", "llama3": r"\p{N}{1,3}",
    "gpt-2": r"\p{N}+",
}
_DEFAULT_DIGIT_CLAUSE = r"\p{N}"

# o200k (GPT-4o) word-split pattern, verbatim from the upstream
# tokenizer.json. Structurally different from the GPT-4 family above:
# contraction suffixes attach to the word clauses ("don't" is one
# pre-token, not "don"+"'t"), letter runs split on case transitions, and
# digits group up to 3. A digit-clause swap on the generic pattern can't
# express this, so it's a full-pattern override.
O200K_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    r"\p{N}{1,3}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n/]*|"
    r"\s*[\r\n]+|"
    r"\s+(?!\S)|"
    r"\s+"
)

# DeepSeek-V3 family word-split: llama.cpp applies these three regexes
# sequentially (each further splits the previous pass's pieces), verbatim
# from PRE_TYPE_DEEPSEEK3_LLM (llama-vocab.cpp:318-325; the same case also
# handles JOYAI_LLM and HUNYUAN_DENSE). A tuple here becomes a chain of
# Split pre-tokenizers in _build_bytelevel_bpe. The second regex is the CJK
# run clause (kanji U+4E00-9FA5 + hiragana U+3040-309F + katakana
# U+30A0-30FF), spelled with \u escapes to keep this file ASCII.
_DEEPSEEK3_PATTERNS = (
    r"\p{N}{1,3}",
    "[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+",
    r"""[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+"""
    r"""|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+"""
    r"""| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
)

# Pres whose split regex is not a digit-clause variant of the generic
# pattern (llama.cpp PRE_TYPE_GPT4O members, plus MiniMax: llama.cpp's
# PRE_TYPE_MINIMAX_M2 regex is the o200k pattern verbatim - MiniMax-M2 and
# MiniMax-M3 GGUFs both ship pre='minimax-m2'). Values are a single pattern
# or a tuple of patterns applied in sequence (llama.cpp multi-regex pres).
_PATTERN_BY_PRE = {
    "gpt-4o": O200K_PATTERN, "llama4": O200K_PATTERN,
    "kanana2": O200K_PATTERN, "talkie": O200K_PATTERN,
    "minimax-m2": O200K_PATTERN,
    # DeepSeek-V3/R1 conversions ship pre='deepseek-v3'; DeepSeek V4 Flash
    # ships pre='joyai-llm'; llama.cpp maps both (and hunyuan-dense) to the
    # same 3-regex set. Before this entry, deepseek-v3 silently fell back to
    # the single-digit clause (over-splitting numbers and CJK).
    "deepseek-v3": _DEEPSEEK3_PATTERNS,
    "joyai-llm": _DEEPSEEK3_PATTERNS,
    "hunyuan-dense": _DEEPSEEK3_PATTERNS,
}


def _bytelevel_split_patterns(pre_id: str) -> tuple[str, ...]:
    """The byte-level word-split regex(es) for a GGUF `tokenizer.ggml.pre`
    hint, with the family-correct digit-grouping clause (or a full-pattern
    override for families the generic pattern can't express). Multi-regex
    pres (deepseek-v3/joyai-llm) return several patterns to be applied in
    sequence, mirroring llama.cpp's sequential regex splitting."""
    if pre_id in _PATTERN_BY_PRE:
        pat = _PATTERN_BY_PRE[pre_id]
        return pat if isinstance(pat, tuple) else (pat,)
    return (_bytelevel_pattern(
        _DIGIT_CLAUSE_BY_PRE.get(pre_id, _DEFAULT_DIGIT_CLAUSE)),)

# pre-tokenizer hint string -> algorithm bucket. Hints come from the
# `tokenizer.ggml.pre` GGUF KV (set by llama.cpp's
# convert_hf_to_gguf.py based on the source tokenizer's metadata).
_BYTELEVEL_PRES = {"qwen2", "qwen3", "qwen35", "qwen35moe", "llama-bpe", "llama3"}


# Public entry point

# gemma-4 multimodal marker roles. The boundary + soft-token placeholder tokens
# live in the GGUF vocab as CONTROL (type-3) entries, but nothing in the GGUF
# labels their *role* - that is an arch convention, ported from llama.cpp's mtmd
# (tools/mtmd/mtmd.cpp gemma4v: img_beg "<|image>", img_end "<image|>") and the
# matching HF processor. mlx-vlm's Gemma4Processor reads these off the tokenizer
# via getattr, so attaching them here keeps the VLM processor fully GGUF-derived
# (no HF download). Maps tokenizer attribute -> literal token string.
_GEMMA4_MM_TOKEN_ROLES = {
    "boi_token":   "<|image>",
    "image_token": "<|image|>",
    "eoi_token":   "<image|>",
    "boa_token":   "<|audio>",
    "audio_token": "<|audio|>",
    "eoa_token":   "<audio|>",
}


def _attach_vlm_token_attrs(fast, tokens: list[str]) -> None:
    """Attach gemma-4 multimodal marker attributes to a GGUF-built tokenizer.

    No-op unless the gemma-4 image-marker triple (<|image>, <|image|>, <image|>)
    is present in the vocab, so text-only / non-gemma-4 tokenizers are untouched
    and the text path stays byte-identical (these attributes don't affect
    encode/decode). Sets the marker strings + the soft-token *ids* the model's
    masked_scatter keys on (image_token_id / audio_token_id)."""
    if not all(t in tokens for t in ("<|image>", "<|image|>", "<image|>")):
        return
    id_of = {t: i for i, t in enumerate(tokens)}
    for attr, tstr in _GEMMA4_MM_TOKEN_ROLES.items():
        if tstr in id_of:
            setattr(fast, attr, tstr)
    fast.image_token_id = id_of.get("<|image|>")
    if "<|audio|>" in id_of:
        fast.audio_token_id = id_of["<|audio|>"]


def load_tokenizer_from_gguf(
    meta, arch: str, *, chat_template_override: str | None = None,
) -> PreTrainedTokenizerFast:
    """Build a HF fast tokenizer from GGUF tokenizer metadata.

    ``chat_template_override`` (inline Jinja string) replaces whatever chat
    template the GGUF metadata carries. It is applied to the fast tokenizer
    *before* turn-end-EOS inference, so multi-EOS detection runs against the
    override (bolting it on afterwards would mis-detect EOS ids). Use it for
    GGUFs with broken/missing templates, the Mistral/Tekken tokenizer gotchas,
    or to force thinking vs non-thinking templates.
    """
    tokens = _read_str_array(meta, "tokenizer.ggml.tokens")
    if tokens is None:
        raise ValueError("GGUF missing tokenizer.ggml.tokens")
    raw_merges = _read_str_array(meta, "tokenizer.ggml.merges")
    scores = _read_float_array(meta, "tokenizer.ggml.scores")
    token_types = _read_int_array(meta, "tokenizer.ggml.token_type")
    if token_types is not None and len(token_types) != len(tokens):
        loadlog.verbose_print(
            f"[tokenizer] token_type length {len(token_types)} != vocab "
            f"{len(tokens)}; ignoring the overhang")

    model_id = _read_string(meta, "tokenizer.ggml.model") or ""
    pre_id = _read_string(meta, "tokenizer.ggml.pre") or ""

    def _special_id(key: str) -> int | None:
        # Out-of-range values (the -1 "unset" sentinel, uint32 wraparound, or
        # HF's pad == vocab_size convention) mean "no such special token";
        # indexing them would silently alias an arbitrary vocab entry.
        tid = _read_int(meta, f"tokenizer.ggml.{key}")
        if tid is None or 0 <= tid < len(tokens):
            return tid
        loadlog.verbose_print(
            f"[tokenizer] ignoring out-of-range {key}={tid} "
            f"(vocab {len(tokens)})")
        return None

    bos_id = _special_id("bos_token_id")
    eos_id = _special_id("eos_token_id")
    pad_id = _special_id("padding_token_id")
    unk_id = _special_id("unknown_token_id")
    # add_bos_token mirrors llama.cpp's raw-prompt convention (auto-prepend
    # BOS), which the real AutoTokenizer for SPM instruct models - gemma,
    # Llama-2 - also does. Honored via a BOS post-processor below, not the
    # PreTrainedTokenizerFast.add_bos_token flag (vestigial for fast
    # tokenizers; behavior comes from the post-processor). The chat path stays
    # single-BOS because the only encode sites (generate/serve) pass
    # add_special_tokens=False whenever the prompt already starts with the BOS
    # string the chat template injected.
    add_bos = _read_bool(meta, "tokenizer.ggml.add_bos_token")
    add_eos = _read_bool(meta, "tokenizer.ggml.add_eos_token")

    chat_template = _read_string(meta, "tokenizer.chat_template")
    if chat_template_override is not None:
        chat_template = chat_template_override

    style = _classify(model_id, pre_id,
                      has_merges=bool(raw_merges), has_scores=bool(scores))
    if style == "bytelevel":
        tok = _build_bytelevel_bpe(tokens, raw_merges, pre_id=pre_id)
    elif style == "spm":
        unk_str = tokens[unk_id] if unk_id is not None else None
        if raw_merges:
            # Explicit merges (gemma4/gemma3): byte-level pre-tokenizing SPM BPE.
            tok = _build_spm_bpe(tokens, _parse_merges(raw_merges),
                                 unk_str=unk_str)
        else:
            # Mergeless scored vocab (classic SentencePiece): reconstruct merges
            # from scores. Whether to prepend the dummy ▁ prefix is the
            # sentencepiece add_dummy_prefix flag - `tokenizer.ggml.add_space_prefix`
            # when present (gemma sets it False), else its sentencepiece default
            # True (Llama-2/Mistral/Vicuna leave it unset).
            if scores is None:
                raise ValueError(
                    "GGUF SPM tokenizer has neither merges nor scores")
            asp = _read_bool(meta, "tokenizer.ggml.add_space_prefix")
            add_prefix = True if asp is None else asp
            tok = _build_spm_bpe(tokens, _derive_spm_merges(tokens, scores),
                                 unk_str=unk_str, add_prefix_space=add_prefix)
    else:
        raise NotImplementedError(
            f"cannot synthesize a tokenizer from this GGUF's metadata "
            f"(arch={arch!r} model={model_id!r} pre={pre_id!r}) - pass "
            f"--hf-source <the checkpoint's HF repo id> to load the "
            f"tokenizer from Hugging Face instead")

    # Special tokens: registered as AddedToken so the fast tokenizer treats
    # them as atomic during encode and emits them verbatim during decode.
    bos_str = tokens[bos_id] if bos_id is not None else None
    eos_str = tokens[eos_id] if eos_id is not None else None
    pad_str = tokens[pad_id] if pad_id is not None else None

    special_tokens: list[str] = []
    seen: set[str] = set()
    for t in (bos_str, eos_str, pad_str):
        if t is not None and t not in seen:
            special_tokens.append(t)
            seen.add(t)
    # GGUF type=3 = control tokens. Add them as specials so they round-trip
    # through encode/decode without splitting.
    user_defined: list[str] = []
    if token_types is not None:
        for tstr, ttype in zip(tokens, token_types):
            if ttype == 3 and tstr not in seen:
                special_tokens.append(tstr)
                seen.add(tstr)
            # GGUF type=4 = USER_DEFINED (e.g. <think>, </think>, <tool_call>).
            # These are already in the BPE vocab, but unless registered as added
            # tokens the pre-tokenizer lets BPE split the chat template's literal
            # "<think>" into ["<th","ink",">"] - a malformed thinking/tool-call
            # marker that drives the model into degenerate output. Register them
            # as atomic but non-special added tokens (matching HF's special=false
            # for these), so they encode to their single vocab id yet stay
            # visible on decode.
            elif ttype == 4 and tstr not in seen:
                user_defined.append(tstr)
                seen.add(tstr)
    if special_tokens:
        tok.add_special_tokens([AddedToken(s, normalized=False, special=True)
                                for s in special_tokens])
    if user_defined:
        tok.add_tokens([AddedToken(s, normalized=False, special=False)
                        for s in user_defined])

    # Honor add_bos_token / add_eos_token: a post-processor wraps the sequence
    # with BOS/EOS when encode() is called with the default
    # add_special_tokens=True (raw-completion parity with llama.cpp). Composed
    # onto any existing post-processor (ByteLevel) rather than replacing it.
    # encode(..., add_special_tokens=False) - used by every internal chat-path
    # encode site - bypasses it, so a chat template that already carries a
    # literal BOS/EOS never doubles up.
    wrap_bos = add_bos and bos_str is not None and bos_id is not None
    wrap_eos = add_eos and eos_str is not None and eos_id is not None
    if wrap_bos or wrap_eos:
        specials = []
        if wrap_bos:
            specials.append((bos_str, bos_id))
        if wrap_eos:
            specials.append((eos_str, eos_id))
        head = [bos_str] if wrap_bos else []
        tail = [eos_str] if wrap_eos else []
        wrap_proc = processors.TemplateProcessing(
            single=" ".join(head + ["$A"] + tail),
            pair=" ".join(head + ["$A"] + tail + ["$B"] + tail),
            special_tokens=specials,
        )
        existing = tok.post_processor
        tok.post_processor = (
            processors.Sequence([existing, wrap_proc])
            if existing is not None else wrap_proc)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=bos_str,
        eos_token=eos_str,
        pad_token=pad_str,
        unk_token=tokens[unk_id] if unk_id is not None else None,
        chat_template=chat_template,
    )

    _self_test_roundtrip(fast)

    fast._gguf_add_bos_token = bool(add_bos)
    fast._gguf_add_eos_token = bool(add_eos)
    # Non-underscore name so it proxies through mlx-lm's TokenizerWrapper to the
    # generate sites (the wrapper routes _-prefixed attrs back to itself).
    fast.gguf_suppress_tokens = _read_int_array(
        meta, "tokenizer.ggml.suppress_tokens") or []
    _attach_vlm_token_attrs(fast, tokens)

    template_eos = _infer_turn_end_eos(fast, eos_id, tokens, token_types)
    meta_eos = _metadata_stop_ids(meta, len(tokens))
    all_eos = _dedup_ids(
        ([eos_id] if eos_id is not None else []) + template_eos + meta_eos)
    fast._gguf_eos_token_ids = all_eos

    extra_eos = [tid for tid in all_eos if tid != eos_id]
    eos_str = f"eos={eos_id}"
    if extra_eos:
        extra_strs = [f"{tid}={tokens[tid]!r}" if 0 <= tid < len(tokens)
                      else str(tid) for tid in extra_eos]
        eos_str += f" extra_eos=[{', '.join(extra_strs)}]"

    loadlog.verbose_print(
        f"[tokenizer] built from GGUF: vocab={len(tokens)} "
        f"style={style} model={model_id!r} pre={pre_id!r} "
        f"bos={bos_id}{'+auto' if add_bos else ''} {eos_str} "
        f"chat_template={'yes' if chat_template else 'no'}"
        + (f" ({len(chat_template)} chars)" if chat_template else ""))
    return fast


# Style classification

def _classify(model_id: str, pre_id: str, *,
              has_merges: bool, has_scores: bool) -> str:
    """Pick a construction style from GGUF tokenizer.ggml.{model,pre}.

    GGUF's `model` field is unreliable across families (gemma4 advertises
    "gemma4" but the on-disk HF tokenizer is BPE; "gpt2" is a generic BPE
    marker shared by many byte-level BPEs). Combine with `pre` and the presence
    of merges/scores to pick the right path.
    """
    # ByteLevel BPE conversions always ship a merge list.
    if has_merges and (pre_id in _BYTELEVEL_PRES or model_id == "gpt2"):
        return "bytelevel"
    if has_merges:
        # SPM-style BPE with explicit merges (gemma4/gemma3), or any other
        # merge-bearing BPE we treat as SPM by default - ByteLevel mis-decoding
        # is loud (mojibake), so this default is safer than guessing ByteLevel.
        return "spm"
    if has_scores:
        # Mergeless but scored = classic SentencePiece (Llama-2/Mistral/Vicuna);
        # merges are reconstructed from scores at build time.
        return "spm"
    # Mergeless and scoreless -> Unigram, which we have no target for.
    return "unsupported"


# ByteLevel BPE (qwen35, qwen3, llama3, gpt2)

def _parse_merges(raw_merges: list[str]) -> list[tuple[str, str]]:
    """Split each merge string at its first ASCII space.

    GGUF stores merges as "<token1> <token2>" (a single literal space
    separator). HF tokenizer expects a list of (token1, token2) tuples.
    Tokens themselves never contain literal ASCII space - gemma4 uses ▁,
    qwen35 uses Ġ - so first-space split is unambiguous.
    """
    out: list[tuple[str, str]] = []
    for m in raw_merges:
        sep = m.find(" ")
        if sep < 0:
            raise ValueError(f"malformed merge (no space): {m!r}")
        out.append((m[:sep], m[sep + 1:]))
    return out


def _build_bytelevel_bpe(tokens: list[str], raw_merges: list[str],
                         pre_id: str = "") -> Tokenizer:
    vocab = {tok: i for i, tok in enumerate(tokens)}
    # llama.cpp stores CONTROL/special token text raw (decoded), while the BPE
    # merges stay in byte-level space, so a merge whose product is a control
    # token references a spelling the vocab doesn't carry - MiniMax-M3's unk
    # is the raw U+FFFD bytes, whose byte-level spelling "ï¿½" appears in 5
    # merges. HF BPE hard-fails init on such merges ("out of vocabulary");
    # llama.cpp keeps them in bpe_ranks but the merged text can never resolve
    # to a token id, so dropping them reproduces its effective behavior.
    # (Aliasing the byte-level spelling to the same id is not an option: the
    # duplicate-id vocab key doesn't survive the tokenizer's JSON round-trip
    # inside PreTrainedTokenizerFast's deepcopy.)
    parsed = _parse_merges(raw_merges)
    merges = [(a, b) for a, b in parsed
              if a in vocab and b in vocab and (a + b) in vocab]
    if len(merges) != len(parsed):
        loadlog.verbose_print(
            f"[tokenizer] dropped {len(parsed) - len(merges)} merges whose "
            "parts/product are not vocab entries (control-token spellings)")
    tok = Tokenizer(models.BPE(
        vocab=vocab, merges=merges,
        byte_fallback=False, fuse_unk=False))
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(p), behavior="isolated", invert=False)
        for p in _bytelevel_split_patterns(pre_id)
    ] + [
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    return tok


# SPM-style BPE (gemma4, gemma3 - explicit merges; llama-2/mistral - derived)

def _derive_spm_merges(tokens: list[str],
                       scores: list[float]) -> list[tuple[str, str]]:
    """Reconstruct the BPE merge list from a scored SentencePiece vocab.

    Classic SPM GGUFs (Llama-2, Mistral-7B, Vicuna, LLaVA text tower) store
    `tokenizer.ggml.{tokens,scores}` but *no* merges. This is the same
    extraction HF's ``LlamaConverter`` / ``SentencePieceExtractor`` runs over
    the sentencepiece proto: for every vocab piece, enumerate the splits whose
    two halves are both in the vocab, order each piece's candidate splits by the
    two halves' ids, then order all merges by the merged piece's score (highest
    first = earliest merge). O(vocab x avg_piece_len), not O(vocab^2).
    """
    vocab = {t: i for i, t in enumerate(tokens)}
    merges: list[tuple[float, str, str]] = []
    for piece, score in zip(tokens, scores):
        if len(piece) < 2:
            continue
        local: list[tuple[int, int, str, str]] = []
        for j in range(1, len(piece)):
            left, right = piece[:j], piece[j:]
            li, ri = vocab.get(left), vocab.get(right)
            if li is not None and ri is not None:
                local.append((li, ri, left, right))
        local.sort(key=lambda x: (x[0], x[1]))
        for li, ri, left, right in local:
            merges.append((score, left, right))
    merges.sort(key=lambda x: x[0], reverse=True)
    return [(left, right) for _score, left, right in merges]


def _build_spm_bpe(tokens: list[str], merges: list[tuple[str, str]],
                   *, unk_str: str | None,
                   add_prefix_space: bool = False) -> Tokenizer:
    """Build an SPM-style BPE tokenizer.

    ``add_prefix_space`` mirrors sentencepiece's ``add_dummy_prefix`` (the
    Llama-2/Mistral default): a leading ``▁`` is prepended so the first word
    tokenizes identically to a mid-sentence one (``Hello`` -> ``▁Hello``), and
    the decoder strips that one leading space back off. gemma's GGUF merges are
    passed with it ``False`` (its validated behavior is unchanged).
    """
    vocab = {tok: i for i, tok in enumerate(tokens)}
    tok = Tokenizer(models.BPE(
        vocab=vocab, merges=merges,
        unk_token=unk_str,
        byte_fallback=True,
        fuse_unk=True,
        ignore_merges=False))
    if add_prefix_space:
        # Exact LlamaTokenizerFast (legacy) recipe: Prepend ▁ + Replace, no
        # pre-tokenizer, and a trailing Strip in the decoder for the dummy
        # prefix.
        tok.normalizer = normalizers.Sequence([
            normalizers.Prepend("▁"),
            normalizers.Replace(" ", "▁"),
        ])
        tok.pre_tokenizer = None
        tok.decoder = decoders.Sequence([
            decoders.Replace("▁", " "),
            decoders.ByteFallback(),
            decoders.Fuse(),
            decoders.Strip(content=" ", left=1, right=0),
        ])
    else:
        tok.normalizer = normalizers.Replace(" ", "▁")
        tok.pre_tokenizer = pre_tokenizers.Split(
            " ", behavior="merged_with_previous", invert=False)
        tok.decoder = decoders.Sequence([
            decoders.Replace("▁", " "),
            decoders.ByteFallback(),
            decoders.Fuse(),
        ])
    return tok


# Turn-end EOS inference

def _infer_turn_end_eos(
    fast: PreTrainedTokenizerFast,
    eos_id: int | None,
    tokens: list[str],
    token_types: list[int] | None,
) -> list[int]:
    """Find additional EOS token IDs by detecting the turn-end marker from
    the chat template.

    Renders a test conversation with a sentinel in the assistant slot, then
    checks which control token (GGUF type=3) appears immediately after the
    sentinel.  Returns IDs that differ from the primary eos_id.
    """
    if not fast.chat_template:
        return []
    try:
        rendered = fast.apply_chat_template(
            [{"role": "user", "content": "hi"},
             {"role": "assistant", "content": "GGUF_SENTINEL"}],
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        return []

    pos = rendered.rfind("GGUF_SENTINEL")
    if pos < 0:
        return []
    suffix = rendered[pos + len("GGUF_SENTINEL"):]
    if not suffix:
        return []

    extra: list[int] = []
    if token_types is not None:
        for tid, (tstr, ttype) in enumerate(zip(tokens, token_types)):
            if ttype == 3 and tid != eos_id and tstr and suffix.startswith(tstr):
                extra.append(tid)
                break
    return extra


# GGUF metadata keys for end-of-generation tokens that llama.cpp folds into its
# stop set (see llama-vocab.cpp): end-of-turn, end-of-message, and the FIM
# rep/sep/pad markers. The turn-end token is frequently not the eos - GLM
# declares <|user|> as eot_token_id - so a loader that stops only on
# eos_token_id (and the chat-template heuristic above, which misses it because
# nothing trails the assistant turn in a no-generation-prompt render) runs past
# the model's own turn boundary into degenerate, looping output.
_EOG_METADATA_KEYS = (
    "tokenizer.ggml.eot_token_id",
    "tokenizer.ggml.eom_token_id",
    "tokenizer.ggml.fim_rep_token_id",
    "tokenizer.ggml.fim_sep_token_id",
    "tokenizer.ggml.fim_pad_token_id",
)


def _metadata_stop_ids(meta, n_vocab: int | None = None) -> list[int]:
    """End-of-generation token ids declared in GGUF metadata (eot/eom/FIM),
    mirroring llama.cpp's EOG set. Order-preserving; may repeat the primary eos
    or itself - the caller dedups. When ``n_vocab`` is given, out-of-range ids
    (the -1 "unset" sentinel, uint32 wraparound) are dropped, matching
    ``_special_id`` - an out-of-range stop id would otherwise alias a vocab row
    in the XTC special-token mask (generation.py) or the log line below."""
    out: list[int] = []
    for key in _EOG_METADATA_KEYS:
        tid = _read_int(meta, key)
        if tid is None:
            continue
        if n_vocab is not None and not (0 <= tid < n_vocab):
            loadlog.verbose_print(
                f"[tokenizer] ignoring out-of-range {key}={tid} "
                f"(vocab {n_vocab})")
            continue
        out.append(tid)
    return out


def _dedup_ids(ids: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for tid in ids:
        if tid not in seen:
            out.append(tid)
            seen.add(tid)
    return out


# Tokens the model must never emit (tokenizer.ggml.suppress_tokens). Gemma-4
# declares its multimodal placeholder ids (<image|>/<audio|>) here so the
# text-only checkpoint can't surface them - llama.cpp applies the same mask
# inside the gemma4 graph. We fold them into the sampler's logit_bias as a
# strongly-negative additive bias: model-agnostic (only models that declare the
# key are affected) and needs no model-forward seam.
_SUPPRESS_BIAS = -1e9


def merge_suppressed_tokens(logit_bias: dict | None, tokenizer) -> dict | None:
    """Fold a model's GGUF-declared suppress_tokens into a logit_bias dict so
    they're never sampled. No-op when the model declares none."""
    suppress = getattr(tokenizer, "gguf_suppress_tokens", None)
    if not suppress:
        return logit_bias
    # Same out-of-range rule as the special-token ids: a -1 sentinel would
    # silently suppress the last vocab token; >= vocab would index out of range.
    try:
        vocab = len(tokenizer)
    except TypeError:   # TokenizerWrapper proxies attrs but not __len__
        vocab = getattr(tokenizer, "vocab_size", None)
    merged = dict(logit_bias or {})
    for tid in suppress:
        if 0 <= int(tid) and (vocab is None or int(tid) < vocab):
            merged.setdefault(int(tid), _SUPPRESS_BIAS)
        else:
            loadlog.verbose_print(
                f"[tokenizer] ignoring out-of-range suppress token {tid} "
                f"(vocab {vocab})")
    return merged


# Round-trip self-test

def _self_test_roundtrip(fast: PreTrainedTokenizerFast) -> None:
    """Catch gross misconfiguration (wrong normalizer/decoder pairing).

    Fast-fail before the caller burns hours on a model run with a broken
    tokenizer. Doesn't catch all encode-divergence-from-HF cases; the
    caller should also do a fixture-set comparison vs HF when one is
    available.
    """
    samples = [
        "Hello, world!",
        "def f(x): return x*2",
    ]
    for s in samples:
        ids = fast.encode(s, add_special_tokens=False)
        back = fast.decode(ids, skip_special_tokens=False)
        if back != s:
            raise RuntimeError(
                f"tokenizer round-trip failed: {s!r} -> {ids[:10]} -> {back!r}")
