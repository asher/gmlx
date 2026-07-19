"""Over-generation probe (run/chat, experimental).

Force a fixed window of tokens past the point a turn would have ended, to
capture what the model emits right after its own stop token (e.g. unprompted
self-critique). The window reuses the same KV cache.

Two modes share a two-phase structure. Phase 1 generates normally and stops at
the natural end-of-generation (EOG) token (the "seam"). Phase 2 continues from
phase 1's cache:

  free continuation -- re-feed the seam token with the EOG set neutralized and
  force a fixed number of tokens.

  injected critique -- append a real follow-up turn and let the model answer it.

Both phase-2 forms can use their own sampler, so the continuation can decode
tighter than the main turn. Phase 2 only prefills the bridge tokens; phase 1's
context stays in the cache.
"""
from __future__ import annotations

import contextlib
import json
import os


@contextlib.contextmanager
def suppressed_eos(tokenizer):
    """Temporarily clear the tokenizer's EOG set so stream_generate will not
    stop on it, then restore it on exit. Yields the captured real EOG ids.
    TokenizerWrapper.__setattr__ routes ``eos_token_ids`` to its internal set.
    """
    real = set(getattr(tokenizer, "eos_token_ids", None) or [])
    tokenizer.eos_token_ids = set()
    try:
        yield real
    finally:
        tokenizer.eos_token_ids = real


def assistant_open_tokens(tokenizer, template_kwargs=None):
    """The token ids the chat template appends for ``add_generation_prompt``
    (the assistant-open marker, e.g. ``<|assistant|>\\n``). Computed by diffing
    a one-message render with and without the generation prompt, so it works
    across templates without hard-coding role tokens. Returns the trailing
    tokens after the longest common prefix."""
    tk = template_kwargs or {}
    probe = [{"role": "user", "content": "x"}]
    no_gen = list(
        tokenizer.apply_chat_template(probe, add_generation_prompt=False, **tk)
    )
    with_gen = list(
        tokenizer.apply_chat_template(probe, add_generation_prompt=True, **tk)
    )
    n = 0
    while n < len(no_gen) and n < len(with_gen) and no_gen[n] == with_gen[n]:
        n += 1
    return with_gen[n:]


def build_critique_bridge(tokenizer, seam_token_id, content, template_kwargs=None):
    """Token ids that turn the model's finished answer into a critique turn,
    appended to phase 1's cache. Reuses the model's own seam token to open the
    follow-up turn, then the critique text, then the assistant-open suffix. No
    full-conversation re-render, so the conversation-start scaffold (BOS, etc.)
    is not duplicated against the cache."""
    body = list(tokenizer.encode("\n" + content, add_special_tokens=False))
    opener = [int(seam_token_id)] if seam_token_id is not None else []
    return opener + body + assistant_open_tokens(tokenizer, template_kwargs)


def collect_interim_eos(tokens, eos_ids):
    """EOG tokens the model re-emits inside the forced window (it tried to end
    the turn again). Returned as ``{"index", "token_id"}`` records."""
    eos = {int(t) for t in (eos_ids or [])}
    return [
        {"index": i, "token_id": int(t)}
        for i, t in enumerate(tokens)
        if int(t) in eos
    ]


def seam_marker(seam):
    """Inline divider printed in verbose mode where the model's turn ended."""
    tid = seam["token_id"]
    txt = (seam.get("text") or "").strip()
    label = f"id={tid}" + (f" {txt!r}" if txt else "")
    return f"\n\n===== over-generation: forced past stop {label} =====\n"


def append_log(path, record):
    """Append one record as a JSON line (collection sink for analysis)."""
    path = os.path.abspath(os.path.expanduser(path))
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
