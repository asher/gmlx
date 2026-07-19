"""In-process LLM-as-judge for response coherence.

The floor checks (``checks.py``) catch mechanical degeneration deterministically;
the judge rules on *semantic* coherence - is the answer fluent, on-topic, and free
of subtle looping/drift that a regex can't catch. It runs as a decoupled final
phase against stored transcripts, so it never shares the GPU with a server
subprocess: every server is already torn down before the judge model loads.

Heavy imports (gmlx -> mlx) are deferred to :meth:`load`, so ``--dry-run`` and
the prompt/scenario construction stay CPU-only and importable anywhere.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

_RUBRIC = """\
You are a strict QA grader for an automated LLM-server test. You are given a USER \
REQUEST and the model's RESPONSE. Judge ONLY the quality of the response.

Do NOT show your reasoning. Do NOT think out loud. Output ONLY a single JSON object \
on one line and nothing before or after it, with exactly these keys:
  "coherent": true or false   (fluent, grammatical, and a sensible reply to the request)
  "repetition": true or false (contains pathological repetition or looping)
  "on_topic": true or false   (actually addresses the request)
  "score": an integer 1 to 5  (1 = garbage, 5 = excellent)
  "reason": a short string (one sentence)

USER REQUEST:
{request}

MODEL RESPONSE:
{response}

Respond with ONLY the JSON object now:"""


@dataclass
class Verdict:
    ok: bool                       # passes the judge gate
    coherent: Optional[bool]
    repetition: Optional[bool]
    on_topic: Optional[bool]
    score: Optional[int]
    reason: str
    parsed: bool                   # did we get usable JSON back
    raw: str = ""

    def to_dict(self) -> dict:
        return {"ok": self.ok, "coherent": self.coherent,
                "repetition": self.repetition, "on_topic": self.on_topic,
                "score": self.score, "reason": self.reason,
                "parsed": self.parsed, "raw": self.raw[:400]}


class Judge:
    def __init__(self, model_path: str, *, hf_source: Optional[str] = None,
                 head_chars: int = 1600, tail_chars: int = 1200,
                 min_score: int = 3, max_new_tokens: int = 600):
        self.model_path = model_path
        self.hf_source = hf_source
        self.head_chars = head_chars
        self.tail_chars = tail_chars
        self.min_score = min_score
        self.max_new_tokens = max_new_tokens
        self._model = self._tok = None

    def load(self) -> None:
        if self._model is not None:
            return
        from gmlx import load_model            # deferred heavy import
        self._model, _cfg, self._tok = load_model(
            self.model_path, hf_source=self.hf_source, verbose=False)

    def _clip(self, text: str) -> str:
        """Keep the head and tail (looping shows at the tail) within a budget."""
        if len(text) <= self.head_chars + self.tail_chars:
            return text
        return (text[:self.head_chars]
                + "\n...[middle elided]...\n"
                + text[-self.tail_chars:])

    def _format_no_think(self, rubric: str) -> tuple:
        """Pre-apply the chat template with thinking turned OFF, so a reasoning model
        answers directly instead of narrating. ``enable_thinking=False`` (Qwen-style)
        and ``thinking=False`` are passed as template vars - harmless and unused on a
        template that doesn't define them. Returns ``(prompt, already_templated)``;
        falls back to the plain template, then the raw rubric, if the call errors.
        """
        tok = self._tok
        if getattr(tok, "chat_template", None) is None:
            return rubric, False
        msgs = [{"role": "user", "content": rubric}]
        for extra in ({"enable_thinking": False, "thinking": False},
                      {"enable_thinking": False}, {}):
            try:
                text = tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, **extra)
                return text, True
            except Exception:                  # noqa: BLE001 - template ignores/rejects
                continue
        return rubric, False

    def score(self, request: str, response: str) -> Verdict:
        from gmlx import generate
        self.load()
        rubric = _RUBRIC.format(request=request[:1200],
                                response=self._clip(response or "(empty)"))
        prompt, templated = self._format_no_think(rubric)
        out = generate(self._model, self._tok, prompt,
                       max_tokens=self.max_new_tokens,
                       apply_chat_template=not templated, verbose=False)
        return self._parse(out)

    def _parse(self, out: str) -> Verdict:
        obj = _extract_json(out)
        if obj is None:
            # No JSON object (e.g. a thinking model that narrated its verdict but
            # never emitted the final JSON). Salvage the verdict from the prose.
            obj = _parse_prose(out)
        if obj is None:
            # Truly nothing usable - a judge miss, not a model failure. Treat as a
            # soft pass (parsed=False) so judge flakiness can't fail an otherwise-
            # clean response; surfaced in the report for inspection.
            return Verdict(ok=True, coherent=None, repetition=None, on_topic=None,
                           score=None, reason="judge produced no parseable verdict",
                           parsed=False, raw=out)
        coherent = _as_bool(obj.get("coherent"))
        repetition = _as_bool(obj.get("repetition"))
        on_topic = _as_bool(obj.get("on_topic"))
        score = _as_int(obj.get("score"))
        # Hard fail when the judge is clear: incoherent, repeating, or low score.
        bad = (coherent is False) or (repetition is True) \
            or (score is not None and score < self.min_score)
        return Verdict(ok=not bad, coherent=coherent, repetition=repetition,
                       on_topic=on_topic, score=score,
                       reason=str(obj.get("reason", ""))[:300], parsed=True, raw=out)


_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)

# prose-fallback patterns: a thinking model often narrates the verdict inline as
# `"coherent": Yes` / `repetition: No` / `score: 4` even when it never emits JSON.
_BOOL_WORD = r"(yes|no|true|false)"
_COHERENT_RE = re.compile(rf'coherent"?\s*[:=]?\s*\*{{0,2}}\s*{_BOOL_WORD}', re.IGNORECASE)
_REPET_RE = re.compile(rf'repetition"?\s*[:=]?\s*\*{{0,2}}\s*{_BOOL_WORD}', re.IGNORECASE)
_ONTOPIC_RE = re.compile(rf'on[\s_-]?topic"?\s*[:=]?\s*\*{{0,2}}\s*{_BOOL_WORD}', re.IGNORECASE)
_SCORE_RE = re.compile(r'score"?\s*[:=]?\s*\*{0,2}\s*([1-5])\b', re.IGNORECASE)
_SCORE_FRAC_RE = re.compile(r'\b([1-5])\s*/\s*5\b')


def _word_bool(m) -> Optional[bool]:
    if not m:
        return None
    return m.group(1).strip().lower() in ("yes", "true")


def _parse_prose(text: str) -> Optional[dict]:
    """Best-effort verdict extraction from free-form (thinking) judge output. Returns
    None if not even a coherent/repetition/score signal is present."""
    if not text:
        return None
    coherent = _word_bool(_COHERENT_RE.search(text))
    repetition = _word_bool(_REPET_RE.search(text))
    on_topic = _word_bool(_ONTOPIC_RE.search(text))
    sm = _SCORE_RE.search(text) or _SCORE_FRAC_RE.search(text)
    score = int(sm.group(1)) if sm else None
    if coherent is None and repetition is None and score is None:
        return None
    return {"coherent": coherent, "repetition": repetition, "on_topic": on_topic,
            "score": score, "reason": "(salvaged from judge prose)"}


def _extract_json(text: str) -> Optional[dict]:
    """Pull the first balanced-ish JSON object out of the judge output."""
    if not text:
        return None
    # Prefer the last {...} block (the model may restate the rubric first).
    blocks = _JSON_RE.findall(text)
    for blk in reversed(blocks):
        try:
            obj = json.loads(blk)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1")
    return None


def _as_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
