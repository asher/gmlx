# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Decisions with Liquid's d1 models: noul, choice and score questions,
each answered in one bidirectional forward over its own prompt.

Ported from llama.cpp's ``tools/server/server-decision.cpp`` (MIT) at
e8df20d23, function by function:

- ``parse_questions``: the request's questions and their options;
- ``d1_break``: a zero-width space after the first character of any
  control-token text the input spells, so user text stays text;
- ``d1_labels``: the label token groups read for each option;
- ``render``: the GGUF's ``systemone`` template, given raw JSON values;
- ``get_temperature`` and ``format_answer``: a softmax at the GGUF's
  temperature; a score is the median level and the confidence is the
  largest probability.

The template text is read from the GGUF at run time.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Optional, Sequence

NAME = "systemone"
TYPES = ("lfm3-d1", "lfm3-d1-omni")
MAX_OPTIONS = 255
NESTING_LIMIT = 128
ZWSP = "\u200b"


class D1RequestError(ValueError):
    """A request a d1 model cannot answer: a 400."""


@dataclass
class Question:
    id: str
    type: str
    instructions: Any
    options: list = field(default_factory=list)   # [(key, description)]


def _check_depth(value, depth: int = 0) -> None:
    if depth >= NESTING_LIMIT:
        raise D1RequestError(
            f"JSON is nested more than {NESTING_LIMIT} levels deep")
    if isinstance(value, dict):
        for v in value.values():
            _check_depth(v, depth + 1)
    elif isinstance(value, list):
        for v in value:
            _check_depth(v, depth + 1)


def is_spoken(instructions) -> bool:
    return isinstance(instructions, dict) and instructions.get("type") == "input_audio"


def parse_questions(body: dict, *, omni: bool, max_questions: int) -> list[Question]:
    """The questions of a request, in its order. A null state is allowed:
    the media can be the whole state."""
    if "state" not in body:
        raise D1RequestError('"state" must be provided')
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise D1RequestError('"questions" must be a non-empty object')
    if len(questions) > max_questions:
        raise D1RequestError(
            f'"questions" has {len(questions)} questions; the limit is {max_questions}')
    _check_depth(body.get("state"))
    out = []
    for qid, q in questions.items():
        def err(msg, qid=qid):
            return D1RequestError(f"questions.{qid}: {msg}")

        if not isinstance(q, dict):
            raise err("must be an object")
        instructions = q.get("instructions")
        if instructions is None:
            raise err('"instructions" must be provided')
        if is_spoken(instructions) and not omni:
            raise err("spoken instructions need a model with audio input")
        _check_depth(instructions)
        kind = q.get("type")
        kind = kind if isinstance(kind, str) else ""
        criteria = q.get("criteria")
        question = Question(qid, kind, instructions)
        if kind == "choice":
            if not isinstance(criteria, dict) or not criteria:
                raise err('"criteria" must be a non-empty object')
            question.options = list(criteria.items())
        elif kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise err('"criteria" must be an array of 2 to 10 levels')
            question.options = [(str(i), d) for i, d in enumerate(criteria)]
        elif kind == "noul":
            if criteria is not None and not isinstance(criteria, dict):
                raise err('"criteria" must be an object')
            crit = criteria or {}
            # The true option comes first.
            question.options = [("true", crit.get("true")),
                                ("false", crit.get("false"))]
        else:
            raise err('"type" must be one of: choice, score, noul')
        if len(question.options) > MAX_OPTIONS:
            raise err(f"too many options ({len(question.options)}), this model "
                      f"supports at most {MAX_OPTIONS}")
        _check_depth(criteria)
        out.append(question)
    return out


def _bucket(n: int) -> str:
    return "2" if n <= 2 else "3_5" if n <= 5 else "6_10" if n <= 10 else "11"


def _choice_pool() -> list[str]:
    pool = [chr(65 + i) for i in range(26)]
    pool += [f"{i:02d}" for i in range(100)]
    pool += [chr(97 + i) for i in range(26)]
    pool += [f"#{i}" for i in range(200)]
    pool += [chr(65 + a) + chr(65 + b) for a in range(26) for b in range(26)]
    return pool


_POOL = _choice_pool()


class D1Format:
    """The prompts and answers of one d1 model."""

    name = "d1"

    def __init__(self, source: str, meta: dict, control_texts: Sequence[str],
                 encode: Callable[[str], list[int]]):
        from transformers.utils.chat_template_utils import _compile_jinja_template

        # The transformers tojson filter writes what llama.cpp's does:
        # json.dumps with ensure_ascii off and no HTML escaping.
        self.template = _compile_jinja_template(source)
        self.type = meta.get("type")
        self.omni = self.type == "lfm3-d1-omni"
        self.temperature = dict(meta.get("temperature") or {})
        self.score_bias = {int(k): list(v)
                           for k, v in (meta.get("score_bias") or {}).items()}
        self.encode = encode
        spellings = sorted({t for t in control_texts if len(t.encode()) >= 2},
                           key=lambda t: len(t.encode()), reverse=True)
        self._spell = (re.compile("|".join(re.escape(t) for t in spellings))
                       if spellings else None)
        self._single: dict[str, Optional[int]] = {}

    # Input text

    def d1_break(self, text: str) -> str:
        """A zero-width space after the first character of each control-token
        text, again until none is left (a break can uncover another)."""
        if self._spell is None:
            return text
        while True:
            out = self._spell.sub(lambda m: m.group(0)[0] + ZWSP + m.group(0)[1:], text)
            if out == text:
                return out
            text = out

    def _escape(self, value) -> Any:
        if isinstance(value, str):
            return self.d1_break(value)
        if isinstance(value, list):
            return [self._escape(v) for v in value]
        if isinstance(value, dict):
            return {self.d1_break(k): self._escape(v) for k, v in value.items()}
        return value

    # Labels

    def _one_token(self, text: str) -> Optional[int]:
        if text not in self._single:
            ids = self.encode(text)
            self._single[text] = ids[0] if len(ids) == 1 else None
        return self._single[text]

    def _group(self, forms) -> list[int]:
        out: list[int] = []
        for form in forms:
            tok = self._one_token(form)
            if tok is not None and tok not in out:
                out.append(tok)
        return out

    def labels(self, q: Question) -> tuple[list[str], list[list[int]]]:
        """The label text shown for each option, and the token ids whose
        largest logit is that option's score."""
        texts: list[str] = []
        groups: list[list[int]] = []
        if q.type != "choice":
            for key, _ in q.options:
                if q.type == "score":
                    group = self._group([key])
                elif key == "true":
                    group = self._group(["yes", "Yes", "YES"])
                else:
                    group = self._group(["no", "No", "NO"])
                if not group:
                    raise RuntimeError(f"decision label is not a single token: {key}")
                texts.append(key)
                groups.append(group)
            return texts, groups

        n = len(q.options)
        letters = all(len(k) == 1 and k.isascii() and k.isalpha() for k, _ in q.options)
        codes = []
        for i, (key, _) in enumerate(q.options):
            if letters:
                codes.append(key)
            elif i < 26:
                codes.append(chr(65 + i))
            else:
                codes.append(chr(65 + (i - 26) // 26) + chr(65 + (i - 26) % 26))
        used: set[int] = set()

        def take(code: str) -> bool:
            ids = self.encode(code)
            if len(ids) != 1 or ids[0] in used:
                return False
            used.add(ids[0])
            group = [ids[0]] + [t for t in self._group([" " + code]) if t != ids[0]]
            texts.append(code)
            groups.append(group)
            return True

        for code in codes:
            if not take(code) and not any(take(c) for c in _POOL):
                raise D1RequestError(f"no single-token label left for {n} options")
        return texts, groups

    # Prompt

    def prompt(self, state, q: Question, *, n_media: int = 0,
               media_marker: str = "") -> str:
        """The question's prompt. With media, any media-marker text in the
        input becomes a space, and the template puts one marker per item;
        a spoken question is heard where its text would be."""
        texts, _ = self.labels(q)
        inp: dict[str, Any] = {
            "id": q.id,
            "type": q.type,
            "instructions": q.instructions,
            "state": state,
            "options": [{"key": k, "description": d, "label": texts[i]}
                        for i, (k, d) in enumerate(q.options)],
        }
        spoken = is_spoken(q.instructions)
        if spoken:
            inp["instructions"] = ""
        inp = self._escape(inp)
        images: list[str] = []
        if n_media > 0 or spoken:
            inp = _replace_text(inp, media_marker, " ")
            images = [media_marker] * n_media
        inp["images"] = images
        if spoken:
            inp["instructions"] = media_marker
        return self.template.render(**inp)

    # Answer

    def temperature_for(self, q: Question, has_media: bool) -> float:
        t = 1.0
        for key in (f"{q.type}.{_bucket(len(q.options))}", q.type):
            if key in self.temperature:
                t = self.temperature[key]
                break
        if has_media and "image" in self.temperature:
            t = self.temperature["image"]
        return t

    def answer(self, q: Question, scores: Sequence[float], has_media: bool) -> dict:
        """llama.cpp's answer for one question from its option scores."""
        n = len(q.options)
        if len(scores) != n:
            raise RuntimeError("decision result does not match the number of options")
        if any(math.isnan(s) or s == math.inf for s in scores):
            raise RuntimeError("the model could not evaluate the decision")
        logits = [float(s) for s in scores]
        if q.type == "score" and n in self.score_bias:
            logits = [v + b for v, b in zip(logits, self.score_bias[n])]
        t = self.temperature_for(q, has_media)
        top = max(logits)
        exps = [math.exp((v - top) / t) for v in logits]
        total = sum(exps)
        probs = [e / total for e in exps]
        if q.type == "noul":
            return {"type": "noul", "noul": probs[0]}
        probabilities = {k: p for (k, _), p in zip(q.options, probs)}
        confidence = max(probs)
        if q.type == "choice":
            best = probs.index(max(probs))
            return {"type": "choice", "choice": q.options[best][0],
                    "probabilities": probabilities, "confidence": confidence}
        mass, level = 0.0, n - 1
        for i, p in enumerate(probs):
            if mass >= 0.5:
                break
            mass += p
            level = i
        return {"type": "score", "score": float(level),
                "legend": {k: d for k, d in q.options},
                "probabilities": probabilities, "confidence": confidence}


def _replace_text(value, search: str, replace: str) -> Any:
    if not search:
        return value
    if isinstance(value, str):
        return value.replace(search, replace)
    if isinstance(value, list):
        return [_replace_text(v, search, replace) for v in value]
    if isinstance(value, dict):
        return {k: _replace_text(v, search, replace) for k, v in value.items()}
    return value


def _tokenizer(processor):
    wrapper = getattr(processor, "_wrapper", processor)
    return getattr(wrapper, "tokenizer", wrapper)


def setup(processor, *, lock=None) -> Optional[D1Format]:
    """The d1 format for a model, or None when its GGUF names no d1
    decision type. ``lock`` guards the tokenizer."""
    import contextlib

    from .ar_reader import tokenizer_of

    tok = _tokenizer(processor)
    meta = getattr(tok, "gguf_decision", None)
    if not meta or meta.get("type") not in TYPES:
        return None
    source = (getattr(tok, "gguf_named_templates", None) or {}).get(NAME)
    if not source:
        raise RuntimeError('decision model has no "systemone" template')
    wrapper = tokenizer_of(processor)
    guard = lock if lock is not None else contextlib.nullcontext()

    def encode(text: str) -> list[int]:
        # Special-token text in the prompt is parsed, as llama.cpp does.
        with guard:
            return list(wrapper.encode(text, add_special_tokens=False))

    return D1Format(source, meta, getattr(tok, "gguf_control_texts", None) or [],
                    encode)


def text_of(model):
    """The lfm3 text model under a server or VLM wrapper."""
    from .ar_reader import text_model

    m = text_model(model)
    return getattr(m, "language_model", None) or m


# The decision


@dataclass
class Pass:
    """One question's prompt, ready to run."""
    question: Question
    ids: list
    groups: list
    embeddings: Any = None       # [len(ids), hidden] when the prompt has media


def batches(lengths: Sequence[int], budget: int) -> list[list[int]]:
    """Question indexes in batches of rows, longest first, so that rows x
    the longest row stays within ``budget`` tokens."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    out: list[list[int]] = []
    for i in order:
        if out and (len(out[-1]) + 1) * lengths[out[-1][0]] <= budget:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def decide_d1(text, fmt: D1Format, passes: list[Pass], *, has_media: bool,
              budget: int, should_stop: Optional[Callable[[], bool]] = None,
              ) -> Generator[None, None, dict]:
    """Answer every pass, a batch of rows per forward. A generator that
    yields after each block of each forward; returns the answers and
    diagnostics."""
    import time

    import mlx.core as mx
    import numpy as np

    from gmlx.models.lfm3.decision import decision_forward

    started = time.time()
    answers: dict[str, dict] = {}
    lengths = [len(p.ids) for p in passes]
    groups_run = batches(lengths, budget)
    for rows in groups_run:
        if should_stop is not None and should_stop():
            from .reads import Cancelled

            raise Cancelled("decision cancelled")
        n = max(lengths[i] for i in rows)
        if any(passes[i].embeddings is not None for i in rows):
            parts = []
            for i in rows:
                e = passes[i].embeddings
                if e is None:
                    e = text.model.embed_tokens(mx.array(passes[i].ids))
                parts.append(mx.pad(e, [(0, n - e.shape[0]), (0, 0)]))
            inputs = mx.stack(parts)
        else:
            inputs = mx.array([passes[i].ids + [0] * (n - lengths[i]) for i in rows])
        logits = yield from decision_forward(text, inputs, [lengths[i] for i in rows])
        del inputs
        logits = np.asarray(logits.astype(mx.float32))
        for r, i in enumerate(rows):
            p = passes[i]
            scores = [float(logits[r, g].max()) for g in p.groups]
            answers[p.question.id] = fmt.answer(p.question, scores, has_media)
    ordered = {p.question.id: answers[p.question.id] for p in passes}
    return {
        "answers": ordered,
        "input_tokens": sum(lengths),
        "diagnostics": {
            "engine": "gmlx",
            "readout": fmt.name,
            "mode": "per-question",
            "forwards": len(groups_run),
            "timing": {"total_ms": (time.time() - started) * 1e3},
        },
    }


def run_to_end(step: Generator[None, None, dict]) -> dict:
    """Drive :func:`decide_d1` to its return value."""
    while True:
        try:
            next(step)
        except StopIteration as stop:
            return stop.value


def response(result: dict, model: str) -> dict:
    """The response body, in llama-server's shape plus diagnostics."""
    return {
        "model": model,
        "answers": result["answers"],
        "usage": {"input_tokens": int(result["input_tokens"]), "output_tokens": 0},
        "diagnostics": result["diagnostics"],
    }


def body_media(body: dict) -> bool:
    """Whether a request carries media: top-level ``images`` or ``files``,
    media parts in a message-shaped state, or a spoken question."""
    if body.get("images") or body.get("files"):
        return True
    state = body.get("state")
    messages = state.get("messages") if isinstance(state, dict) else state
    if isinstance(messages, list):
        for msg in messages:
            content = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(content, list) and any(
                    isinstance(p, dict) and p.get("type") in
                    ("image_url", "input_audio", "video_url") for p in content):
                return True
    questions = body.get("questions")
    if isinstance(questions, dict):
        return any(isinstance(q, dict) and is_spoken(q.get("instructions"))
                   for q in questions.values())
    return False
