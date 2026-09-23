"""Corpus readers for the teacher pass: jsonl, a directory of text files or a
Hugging Face dataset id, NFC normalized and cut into windows, plus the
generator sidecar a synthetic corpus carries."""
from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------

def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def iter_corpus(spec: str, text_key: str = "text", limit: int | None = None,
                hf_split: str = "train", prefix: str | None = None) -> Iterator[tuple[str, str]]:
    """(doc_id, text) from a jsonl file, a directory of text files, or an
    HF dataset id (streamed). Text is NFC-normalized here, once. ``prefix``
    names a jsonl file inside a directory corpus in its ids by relative
    path, so two files with one basename never share an id."""
    p = Path(spec)
    n = 0
    if p.is_file():
        with open(p, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                yield f"{prefix or p.name}:{i}", nfc(obj[text_key])
                n += 1
                if limit and n >= limit:
                    return
        return
    if p.is_dir():
        for f in sorted(p.rglob("*")):
            if f.is_file() and f.suffix in (".txt", ".md", ".py", ".json", ".jsonl"):
                if f.suffix == ".jsonl":
                    for did, t in iter_corpus(str(f), text_key, prefix=str(f.relative_to(p))):
                        yield did, t
                        n += 1
                        if limit and n >= limit:
                            return
                    continue
                yield str(f.relative_to(p)), nfc(f.read_text(encoding="utf-8", errors="replace"))
                n += 1
                if limit and n >= limit:
                    return
        return
    import datasets  # function-local, as gen/benchmarks.py:48
    name, _, config = spec.partition("@")
    ds = datasets.load_dataset(name, config or None, split=hf_split, streaming=True)
    for i, row in enumerate(ds):
        yield f"{spec}:{i}", nfc(row[text_key])
        n += 1
        if limit and n >= limit:
            return


def generator_sidecar(corpus: str) -> tuple[dict | None, str]:
    """The generator sidecar beside a synthetic corpus file (<corpus>.gen.json)
    as the manifest's generator block, with its fingerprint (sha256 of
    the block, 12 hex chars); (None, "") for a corpus without one."""
    p = Path(corpus).expanduser()
    side = p.with_suffix(p.suffix + ".gen.json") if p.is_file() else None
    if side is None or not side.exists():
        return None, ""
    block = json.loads(side.read_text(encoding="utf-8"))
    fp = hashlib.sha256(json.dumps(block, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
    return block, fp


def norm_messages(msgs: list[dict]) -> list[dict]:
    """Copies of the messages with content and reasoning_content
    NFC-normalized; other fields (tool_calls, name) pass through."""
    out = []
    for m in msgs:
        m = dict(m)
        for k in ("content", "reasoning_content"):
            if isinstance(m.get(k), str):
                m[k] = nfc(m[k])
        out.append(m)
    return out


def iter_conversations(spec: str, key: str = "messages", student_key: str | None = "student_messages",
                       limit: int | None = None, hf_split: str = "train"
                       ) -> Iterator[tuple[str, list[dict], list[dict] | None]]:
    """(doc_id, messages, student_messages) from a jsonl file or an HF
    dataset id whose rows carry a list of {role, content} dicts under key
    and, optionally, a second list under student_key (the student's own
    view of the conversation; None when the row has none). Contents are
    NFC-normalized (norm_messages)."""
    def one(row, doc):
        st = row.get(student_key) if student_key else None
        return doc, norm_messages(row[key]), (norm_messages(st) if st else None)
    p = Path(spec)
    n = 0
    if p.is_file():
        with open(p, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                yield one(json.loads(line), f"{p.name}:{i}")
                n += 1
                if limit and n >= limit:
                    return
        return
    import datasets  # function-local, as gen/benchmarks.py:48
    name, _, config = spec.partition("@")
    ds = datasets.load_dataset(name, config or None, split=hf_split, streaming=True)
    for i, row in enumerate(ds):
        yield one(row, f"{spec}:{i}")
        n += 1
        if limit and n >= limit:
            return


def per_turn_rows(msgs: list[dict]) -> list[list[dict]]:
    """One conversation per assistant turn with content: the messages up to
    and including that turn, so every target turn is the final turn of
    its row and the template renders it the way inference does."""
    out = []
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant" and (m.get("content") or "").strip():
            out.append(list(msgs[:i + 1]))
    return out


def same_reply(a: list[dict], b: list[dict]) -> bool:
    """Whether two message lists end on the same assistant message (role,
    content and reasoning_content equal)."""
    if not a or not b:
        return False
    x, y = a[-1], b[-1]
    return (x.get("role") == y.get("role") == "assistant" and (x.get("content") or "") == (y.get("content") or "")
            and (x.get("reasoning_content") or "") == (y.get("reasoning_content") or ""))


