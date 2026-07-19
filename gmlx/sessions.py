"""Chat session persistence: schema-v1 JSON files under the XDG data dir,
plus a markdown transcript export. Pure file logic - no model, no terminal -
so ``chat.py`` owns all REPL wiring and this module stays test-friendly.

Schema v1 (one JSON object per file)::

    version: 1
    created / updated: ISO-8601 local timestamps
    model: {path, arch, mmproj, draft_gguf}
    system_prompt / sampling / reasoning / render / theme / colorblind /
    thinking_budget: session settings at save time
    messages: [{role, content (raw, markers intact), ts, images, audios,
                canceled, think_open, stats}]
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

SCHEMA_VERSION = 1


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def sessions_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(base) / "gmlx" / "chats"


def default_session_name(model_path: str) -> str:
    """``<model-slug>-YYYYmmdd-HHMMSS`` - unique enough per chat session."""
    try:
        from .discovery import derive_id

        slug = derive_id(os.path.basename(model_path))[0]
    except Exception:
        slug = os.path.splitext(os.path.basename(model_path))[0]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", slug).strip("-.").lower() or "chat"
    return f"{slug}-{time.strftime('%Y%m%d-%H%M%S')}"


def session_path(name: str) -> Path:
    return sessions_dir() / f"{name}.json"


def save_session(doc: dict, name: str) -> Path:
    """Atomic write (tmp + rename); stamps version/created/updated."""
    path = session_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = dict(doc)
    doc["version"] = SCHEMA_VERSION
    doc.setdefault("created", _now())
    doc["updated"] = _now()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return path


def load_session(ref: str) -> tuple[dict, Path]:
    """Load by name (in :func:`sessions_dir`) or by explicit ``.json`` path."""
    p = Path(os.path.expanduser(ref))
    if not (p.suffix == ".json" and p.is_file()):
        p = session_path(ref)
    doc = json.loads(p.read_text())
    if not isinstance(doc, dict):
        # ValueError keeps this in the (OSError, ValueError) class every
        # caller already handles; a bare AttributeError would kill the chat UI.
        raise ValueError(f"session file {p} is not a JSON object")
    v = doc.get("version")
    if v != SCHEMA_VERSION:
        raise ValueError(f"unsupported session version {v!r} in {p}")
    return doc, p


def list_sessions() -> list[dict]:
    """Newest-first summaries: {name, path, updated, model, turns}."""
    out = []
    d = sessions_dir()
    if not d.is_dir():
        return out
    for p in d.glob("*.json"):
        try:
            doc = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(doc, dict) or doc.get("version") != SCHEMA_VERSION:
            continue   # stray non-session .json must not DoS the session list
        msgs = doc.get("messages") or []
        model = doc.get("model")
        model = model if isinstance(model, dict) else {}
        out.append(
            {
                "name": p.stem,
                "path": p,
                "updated": doc.get("updated", ""),
                "model": os.path.basename(model.get("path", "")),
                "turns": sum(1 for m in msgs
                             if isinstance(m, dict) and m.get("role") == "assistant"),
            }
        )
    out.sort(key=lambda s: s["updated"], reverse=True)
    return out


def latest_for_model(model_path: str) -> str | None:
    """Name of the most recently updated session for ``model_path``, if any.
    ``model_path`` may also be a served model id (chat --assistant), which is
    recorded verbatim rather than abspathed."""
    targets = {model_path, os.path.abspath(model_path)}
    for s in list_sessions():
        try:
            doc, _ = load_session(s["name"])
        except (OSError, ValueError):
            continue
        model = doc.get("model")
        if isinstance(model, dict) and model.get("path") in targets:
            return s["name"]
    return None


def split_thinking(text: str, think_open: bool = False) -> tuple[str, str]:
    """``(reasoning, answer)`` portions of a raw reply (markers dropped)."""
    from .reasoning import ReasoningFilter

    rf = ReasoningFilter(start_in_thinking=think_open)
    spans = rf.feed(text) + rf.flush()
    reason = "".join(t for t, m in spans if m == "reason").strip()
    answer = "".join(t for t, m in spans if m == "answer").strip()
    return reason, answer


def export_markdown(doc: dict, path: str) -> Path:
    """Render the session as a markdown transcript; thinking spans become
    collapsed ``<details>`` blocks."""
    model = (doc.get("model") or {}).get("path", "")
    lines = [f"# Chat - {os.path.basename(model) or 'session'}", ""]
    meta = [m for m in (doc.get("created"), doc.get("updated")) if m]
    if meta or model:
        bits = [" -> ".join(meta)] if meta else []
        if model:
            bits.append(f"model: `{model}`")
        lines += ["*" + " · ".join(bits) + "*", ""]
    sp = doc.get("system_prompt")
    if sp:
        lines += ["## System", "", sp, ""]
    for m in doc.get("messages") or []:
        role = m.get("role")
        if role == "system":
            continue
        ts = f" ({m['ts']})" if m.get("ts") else ""
        if role == "user":
            lines += [f"## User{ts}", ""]
            for kind in ("images", "audios"):
                for f in m.get(kind) or []:
                    lines.append(f"*[{kind[:-1]}: `{f}`]*")
            lines += [m.get("content", ""), ""]
            continue
        canceled = " *(canceled)*" if m.get("canceled") else ""
        lines += [f"## Assistant{ts}{canceled}", ""]
        reason, answer = split_thinking(
            m.get("content", ""), m.get("think_open", False)
        )
        if reason:
            lines += [
                "<details><summary>thinking</summary>",
                "",
                reason,
                "",
                "</details>",
                "",
            ]
        lines += [answer, ""]
    out = Path(os.path.expanduser(path))
    out.write_text("\n".join(lines).rstrip("\n") + "\n")
    return out
