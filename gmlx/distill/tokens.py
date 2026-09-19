"""Tokenizer facts the cache and the view need: BOS handling, encoding with
byte end offsets, and the identity test on two vocab maps. Every function
takes a gmlx tokenizer wrapper."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gmlx.load.tokenizer import backend, hf_inner, vocab_map_hash

# ---------------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------------

def tokenizer_from_gguf(path: str):
    """Build the HF fast tokenizer from a GGUF header only (no tensor
    reads) through gmlx's synthesizer. Returns a PreTrainedTokenizerFast
    with gmlx's _gguf_* attributes set."""
    import gguf

    from gmlx.load.tokenizer import load_tokenizer_from_gguf
    reader = gguf.GGUFReader(path)
    f = reader.fields["general.architecture"]
    arch = bytes(f.parts[f.data[0]]).decode()
    fast = load_tokenizer_from_gguf(reader, arch)
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


_SPECIAL_TEXT: dict[tuple[int, int], bytes | None] = {}


def _special_text_bytes(inner, tid: int) -> bytes | None:
    key = (id(inner), int(tid))
    if key not in _SPECIAL_TEXT:
        try:
            t = inner.convert_ids_to_tokens(int(tid))
        except Exception:
            t = None
        _SPECIAL_TEXT[key] = t.encode("utf-8") if isinstance(t, str) and t else None
    return _SPECIAL_TEXT[key]


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
    special = set(inner.all_special_ids)
    ends = np.zeros(len(ids), dtype=np.uint32)
    pos = 0
    ok = True
    for i, tid in enumerate(ids):
        if tid in special:
            # a special rendered as text (a chat frame) spans its own bytes;
            # one the backend added (BOS) spans none
            sb = _special_text_bytes(inner, tid)
            if sb and text_bytes[pos:pos + len(sb)] == sb:
                pos += len(sb)
            ends[i] = pos
            continue
        bb = tb[tid] if tid < len(tb) else None
        if bb is None:
            # a control token outside all_special_ids (gemma-4's <|turn>,
            # Qwen's <|im_start|>) rendered as text spans its literal bytes
            sb = _special_text_bytes(inner, tid)
            if sb and text_bytes[pos:pos + len(sb)] == sb:
                pos += len(sb)
                ends[i] = pos
                continue
            ok = False
            break
        pos += len(bb)
        ends[i] = pos
    if ok and pos == len(text_bytes):
        return ids, ends, False
    # Fallback: char offsets -> byte offsets.
    char_to_byte = np.zeros(len(text) + 1, dtype=np.int64)
    acc = 0
    for ci, ch in enumerate(text):
        char_to_byte[ci] = acc
        acc += len(ch.encode("utf-8"))
    char_to_byte[len(text)] = acc
    ends = np.zeros(len(ids), dtype=np.uint32)
    for i, (s, e) in enumerate(enc.offsets):
        ends[i] = char_to_byte[e] if e <= len(text) else acc
    # Specials at the start carry end 0; monotone repair for any offset the
    # backend reports as (0, 0) mid-row.
    for i in range(1, len(ends)):
        if ends[i] < ends[i - 1]:
            ends[i] = ends[i - 1]
    return ids, ends, True


