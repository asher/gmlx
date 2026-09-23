"""Tokenizer facts the cache and the view need: BOS handling, encoding with
byte end offsets, and the identity test on two vocab maps. Every function
takes a gmlx tokenizer wrapper."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gmlx.load.tokenizer import backend, hf_inner, special_ids, vocab_map_hash

# ---------------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------------

def tokenizer_from_gguf(path: str):
    """Build the HF fast tokenizer from a GGUF header only (no tensor
    reads) through gmlx's synthesizer. Returns a PreTrainedTokenizerFast
    with gmlx's _gguf_* attributes set."""
    import gguf

    from gmlx.load.tokenizer import bundled_chat_template_for_arch, load_tokenizer_from_gguf
    reader = gguf.GGUFReader(path)
    f = reader.fields["general.architecture"]
    arch = bytes(f.parts[f.data[0]]).decode()
    # the same template the model loader installs, so align and train
    # see one student identity
    fast = load_tokenizer_from_gguf(reader, arch, chat_template_override=bundled_chat_template_for_arch(arch))
    llamacpp_bos_default(fast, reader)
    return fast


def llamacpp_bos_default(tokenizer, gguf) -> None:
    """Apply llama.cpp's add_bos default to a tokenizer synthesized from a
    GGUF that carries no tokenizer.ggml.add_bos_token key. gguf is a path
    or an open GGUFReader. Every distill entry point that takes a GGUF
    tokenizer from gmlx passes through here."""
    import gguf as gguf_py
    reader = gguf_py.GGUFReader(gguf) if isinstance(gguf, (str, Path)) else gguf
    if "tokenizer.ggml.add_bos_token" in reader.fields:
        return
    pre_f = reader.fields.get("tokenizer.ggml.pre")
    pre = bytes(pre_f.parts[pre_f.data[0]]).decode() if pre_f is not None else ""
    if pre in _LLAMA3_PRE_ADD_BOS:
        _prepend_bos(hf_inner(tokenizer))


# llama.cpp (llama-vocab.cpp) defaults add_bos to true for these
# pre-tokenizers when the GGUF carries no tokenizer.ggml.add_bos_token key;
# gmlx's synthesizer reads the key only, so a Llama-3 GGUF without it would
# feed the student BOS-less rows that llama.cpp would prefix.
_LLAMA3_PRE_ADD_BOS = frozenset({"llama3", "llama-v3", "llama-bpe", "falcon3", "falcon-h1",
                                 "pixtral", "midm-2.0", "lfm2", "jina-v5-nano"})


def _prepend_bos(fast) -> None:
    """Compose a BOS-prefixing post-processor onto a synthesized fast
    tokenizer that has none, and mark it as adding BOS."""
    from tokenizers import processors
    bos_id = fast.bos_token_id
    bos_str = fast.bos_token
    if bos_id is None or bos_str is None:
        return
    probe = fast.encode("x", add_special_tokens=True)
    if probe and probe[0] == bos_id:
        fast._gguf_add_bos_token = True
        return
    tok = fast.backend_tokenizer
    wrap = processors.TemplateProcessing(single=f"{bos_str} $A", pair=f"{bos_str} $A $B",
                                         special_tokens=[(bos_str, bos_id)])
    existing = tok.post_processor
    tok.post_processor = processors.Sequence([existing, wrap]) if existing is not None else wrap
    fast._gguf_add_bos_token = True


def tokenizer_from_dir(path: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path)


def load_tokenizer(path: str):
    """GGUF file -> header-only synth; directory -> AutoTokenizer."""
    p = Path(path)
    if p.is_file() and p.suffix == ".gguf":
        return tokenizer_from_gguf(str(p))
    if p.is_dir():
        ggufs = sorted(p.glob("*.gguf"))
        if ggufs and not (p / "tokenizer.json").exists():
            return tokenizer_from_gguf(str(ggufs[0]))
        return tokenizer_from_dir(str(p))
    raise FileNotFoundError(path)


def logits_width_from_gguf(path: str) -> int:
    """output.weight ne[1], or token_embd.weight ne[1] for tied heads. Header
    only."""
    import gguf
    reader = gguf.GGUFReader(path)
    by_name = {t.name: t for t in reader.tensors}
    for name in ("output.weight", "token_embd.weight"):
        t = by_name.get(name)
        if t is not None:
            # gguf-py reports ne in GGUF order (ne[0] = fastest); the vocab
            # width is the last dimension.
            return int(t.shape[-1])
    raise ValueError(f"{path}: no output.weight or token_embd.weight")


def logits_width_from_mlx_dir(path: str) -> int:
    """lm_head.weight or model.embed_tokens.weight rows from safetensors
    headers, no tensor reads."""
    import struct
    for st in sorted(Path(path).glob("*.safetensors")):
        with open(st, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        for name in ("lm_head.weight", "model.embed_tokens.weight",
                     "language_model.model.embed_tokens.weight"):
            if name in header:
                return int(header[name]["shape"][0])
    raise ValueError(f"{path}: no lm_head or embed_tokens in safetensors")


def bos_id(tokenizer) -> int | None:
    inner = hf_inner(tokenizer)
    return inner.bos_token_id


def adds_bos(tokenizer) -> bool:
    inner = hf_inner(tokenizer)
    v = getattr(inner, "_gguf_add_bos_token", None)
    if v is not None:
        return bool(v)
    ids = inner.encode("x", add_special_tokens=True)
    return bool(ids) and inner.bos_token_id is not None and ids[0] == inner.bos_token_id


def _pad_style(tok: str | None) -> bool:
    return tok is None or tok.startswith("[PAD") or tok.startswith("<pad") \
        or tok.startswith("<unused") or tok.startswith("<|pad")


def identity_pair(teacher_tok, student_tok) -> tuple[bool, str]:
    """Equal vocab maps, or the shorter a prefix of the longer with only
    pad-style surplus ids. Returns (identity, reason)."""
    ti, si = hf_inner(teacher_tok), hf_inner(student_tok)
    nt, ns = len(ti), len(si)
    n = min(nt, ns)
    if vocab_map_hash(ti) == vocab_map_hash(si) and nt == ns:
        return True, "equal vocab maps"
    tt = ti.convert_ids_to_tokens(list(range(n)))
    st = si.convert_ids_to_tokens(list(range(n)))
    spec = set(ti.all_special_ids) | set(si.all_special_ids)
    for i in range(n):
        if i in spec:
            continue
        if tt[i] != st[i]:
            return False, f"vocab maps differ at id {i}: {tt[i]!r} vs {st[i]!r}"
    longer = ti if nt > ns else si
    surplus = longer.convert_ids_to_tokens(list(range(n, max(nt, ns))))
    bad = [n + i for i, t in enumerate(surplus) if not _pad_style(t)]
    if bad:
        return False, f"surplus ids are not pad-style, first {bad[0]}"
    return True, f"prefix maps, {max(nt, ns) - n} pad-style surplus ids"


def _special_text_bytes(inner, tid: int) -> bytes | None:
    """The text a special id renders as, cached on the tokenizer object
    itself (a module-level map keyed by id() would outlive the tokenizer
    and could answer for another one at the same address)."""
    cache = getattr(inner, "_gmlx_special_text", None)
    if cache is None:
        cache = {}
        try:
            inner._gmlx_special_text = cache
        except AttributeError:
            pass
    tid = int(tid)
    if tid not in cache:
        try:
            t = inner.convert_ids_to_tokens(tid)
        except Exception:
            t = None
        cache[tid] = t.encode("utf-8") if isinstance(t, str) and t else None
    return cache[tid]


def segment_markers(tokenizer) -> set[int]:
    """The ids after which the tokenizer opens a new segment: every special
    id (BOS, EOS, the control tokens a GGUF flags special, which
    transformers keeps out of all_special_ids) and every added token, a
    plain one such as <think> included, since the pre-tokenizer splits at
    added tokens. A dummy prefix follows any of them."""
    inner = hf_inner(tokenizer)
    return set(special_ids(tokenizer)) | {int(t) for t in getattr(inner, "added_tokens_decoder", {})}


def zero_width_indices(ids, ends, markers: set[int]) -> list[int]:
    """The token indices that span no bytes after a segment marker: the
    dummy prefix of a Llama-2 style tokenizer after a mid-row marker."""
    ids, ends = np.asarray(ids), np.asarray(ends, dtype=np.int64)
    return [i for i in range(1, len(ids)) if ends[i] == ends[i - 1] and ends[i] > 0 and int(ids[i - 1]) in markers]


def encode_with_byte_ends(tokenizer, text_bytes: bytes,
                          tb: list[bytes | None],
                          add_special_tokens: bool = True) -> tuple[np.ndarray, np.ndarray, bool]:
    """Token ids and per-token end byte offsets (BOS = 0) for NFC bytes.

    Offsets come from token byte lengths; when their sum does not equal the
    text length (a byte-level split inside a multibyte char, a dummy
    prefix), the row falls back to Encoding.offsets through a char-to-byte
    prefix table and is flagged. Returns (ids int32, end_bytes uint32,
    flagged)."""
    text = text_bytes.decode("utf-8")
    inner = hf_inner(tokenizer)
    b = backend(tokenizer)
    enc = b.encode(text, add_special_tokens=add_special_tokens)
    ids = np.asarray(enc.ids, dtype=np.int32)
    special = set(special_ids(tokenizer))
    markers = segment_markers(tokenizer)
    ends = np.zeros(len(ids), dtype=np.uint32)
    pos = 0
    ok = True
    seg_start = True
    for i, tid in enumerate(ids):
        tid = int(tid)
        bb = tb[tid] if tid < len(tb) else None
        if bb is None:
            # a special rendered as text (a chat frame) or a control token
            # outside all_special_ids (gemma-4's <|turn>, Qwen's
            # <|im_start|>) spans its literal bytes; a special the backend
            # added (BOS) spans none; anything else is unknown
            sb = _special_text_bytes(inner, tid)
            if sb and text_bytes[pos:pos + len(sb)] == sb:
                pos += len(sb)
            elif tid not in special:
                ok = False
                break
            ends[i] = pos
            seg_start = True
            continue
        if seg_start and bb.startswith(b" ") and text_bytes[pos:pos + 1] != b" ":
            # a dummy prefix (Llama-2, Mistral SPM): the tokenizer prepends
            # a space the text does not hold, zero width here
            bb = bb[1:]
        # a literal added token (<think>) also opens a segment
        seg_start = tid in markers
        pos += len(bb)
        ends[i] = pos
    if ok and pos == len(text_bytes):
        return ids, ends, False
    return ids, offsets_to_byte_ends(enc.offsets, text, ids, tb), True


def offsets_to_byte_ends(offsets, text: str, ids, tb: list[bytes | None]) -> np.ndarray:
    """End byte offsets from the backend's char offsets: each end through a
    char-to-byte prefix table, and the pieces of one char span (the byte
    fallback of a multibyte character) spread over the span one token's
    bytes at a time, so ends stay strictly increasing."""
    char_to_byte = np.zeros(len(text) + 1, dtype=np.int64)
    acc = 0
    for ci, ch in enumerate(text):
        char_to_byte[ci] = acc
        acc += len(ch.encode("utf-8"))
    char_to_byte[len(text)] = acc
    ends = np.zeros(len(ids), dtype=np.uint32)
    i = 0
    n = len(ids)
    while i < n:
        s, e = offsets[i]
        j = i + 1
        while j < n and tuple(offsets[j]) == (s, e):
            j += 1
        span_end = int(char_to_byte[e]) if e <= len(text) else acc
        if j - i == 1:
            ends[i] = span_end
        else:
            at = int(char_to_byte[s]) if s <= len(text) else acc
            for k in range(i, j - 1):
                tid = int(ids[k])
                bb = tb[tid] if tid < len(tb) else None
                at = min(at + max(len(bb) if bb is not None else 1, 1), span_end)
                ends[k] = at
            ends[j - 1] = span_end
        i = j
    # Specials at the start carry end 0; monotone repair for any offset the
    # backend reports as (0, 0) mid-row.
    for i in range(1, len(ends)):
        if ends[i] < ends[i - 1]:
            ends[i] = ends[i - 1]
    return ends


