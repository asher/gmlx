"""Decisions through a model's own ``systemone`` chat template.

A GGUF can ship a decision prompt as the named chat template
``tokenizer.chat_template.systemone`` and name its protocol in
``<arch>.decision.type``. For the types listed in ``TYPES`` the reader
renders that template once per question and reads the option labels at
the last position, as llama.cpp's decision server does
(``tools/server/server-decision.cpp`` at a37cc4610):

- labels are A..Z, then the two-letter codes AA..ZZ that are one token;
- a noul question lists true before false, and its answer is the
  probability of the true label, with no recalibration;
- probabilities are a softmax at the GGUF's temperature for the question
  type and option count (1.0 when it names none), after its score biases.

Granite Switch is the first such model: its template opens the question
turn with the ``decision`` adapter's control token."""

from __future__ import annotations

import json
from typing import Any

from . import letters

NAME = "systemone"
# decision.type values whose template this readout drives.
TYPES = ("granite-switch",)
MAX_LABELS = 255


def _tokenizer(processor):
    wrapper = getattr(processor, "_wrapper", processor)
    return getattr(wrapper, "tokenizer", wrapper)


def find(processor) -> tuple[str, dict] | None:
    """``(template source, decision metadata)`` when the model ships a
    systemone template for a decision type this readout drives."""
    tok = _tokenizer(processor)
    meta = getattr(tok, "gguf_decision", None)
    source = (getattr(tok, "gguf_named_templates", None) or {}).get(NAME)
    if not source or not meta or meta.get("type") not in TYPES:
        return None
    return source, meta


_PROBE = {"id": "q", "type": "choice",
          "raw": {"instructions": "?", "criteria": {"a": None}}}


def _bucket(n: int) -> str:
    return "2" if n <= 2 else "3_5" if n <= 5 else "6_10" if n <= 10 else "11"


def _raw(v):
    """A description as the template takes it: the request's value, or None
    for an empty one."""
    return None if v is None or v == "" else v


class TemplateFormat:
    """The pass prompts and answers of the systemone template readout, in
    the role ``letters.LetterFormat`` plays for the letter readout."""

    name = "template"

    def __init__(self, source: str, meta: dict, label_texts: list[str],
                 cut_ids=()):
        from transformers.utils.chat_template_utils import _compile_jinja_template

        self.template = _compile_jinja_template(source)
        self.temperature = dict(meta.get("temperature") or {})
        self.score_bias = {int(k): list(v)
                           for k, v in (meta.get("score_bias") or {}).items()}
        self.labels = list(label_texts)
        self.max_options = len(self.labels)
        self.cut_ids = frozenset(int(i) for i in cut_ids)

    def options(self, q) -> list[dict]:
        crit = q["raw"].get("criteria")
        if q["type"] == "choice":
            return [{"key": k, "description": _raw(v) if _raw(v) is not None else k}
                    for k, v in crit.items()]
        if q["type"] == "score":
            return [{"key": str(i), "description": _raw(v)}
                    for i, v in enumerate(crit)]
        crit = crit or {}
        return [{"key": "true", "description": _raw(crit.get("true"))},
                {"key": "false", "description": _raw(crit.get("false"))}]

    def prompt(self, state: str, q, options) -> str:
        instructions = q["raw"].get("instructions")
        return self.template.render(
            state=state, instructions=instructions, type=q["type"], id=q["id"],
            options=[{**o, "label": self.labels[i]} for i, o in enumerate(options)],
            images=[])

    def size(self, option) -> int:
        d = option["description"]
        return len(option["key"]) + len(d if isinstance(d, str) else json.dumps(d))

    def probs(self, scores: list[float], q, n: int) -> list[float]:
        letters.check_finite(scores)
        z = list(scores)
        if q["type"] == "score" and len(z) in self.score_bias:
            z = [v + b for v, b in zip(z, self.score_bias[len(z)])]
        t = 1.0
        for key in (f"{q['type']}.{_bucket(n)}", q["type"]):
            if key in self.temperature:
                t = self.temperature[key]
                break
        return letters.softmax([v / t for v in z])

    def noul(self, p: list[float]) -> float:
        return p[0]

    def prefix_text(self, state: str) -> str:
        """A one-option pass on ``state``: at least as long as its prefix."""
        return self.prompt(state, _PROBE, self.options(_PROBE))

    def split(self, tokens, state: str) -> tuple[list[int], int]:
        """The ids before the first adapter control token of a pass prompt:
        the state turn, which runs on the base weights. Without such a token,
        the common start of two passes that differ in their question."""
        q = _PROBE
        one = tokens.ids(self.prefix_text(state))
        for i, t in enumerate(one):
            if t in self.cut_ids:
                return one[:i], i
        q2 = {**q, "raw": {"instructions": "!", "criteria": {"b": None}}}
        two = tokens.ids(self.prompt(state, q2, self.options(q2)))
        n = 0
        for a, b in zip(one, two):
            if a != b:
                break
            n += 1
        return one[:n], n


class TemplateTokens:
    """Pass prompts that are already rendered: ids are the prompt text
    tokenized with its special tokens and nothing added, as llama.cpp
    tokenizes a decision prompt."""

    def __init__(self, processor, *, lock=None):
        import contextlib

        self.wrapper = getattr(processor, "_wrapper", processor)
        self.lock = lock if lock is not None else contextlib.nullcontext()
        self._ids: dict[str, list[int]] = {}

    def render(self, text: str) -> str:
        return text

    def ids(self, text: str) -> list[int]:
        got = self._ids.get(text)
        if got is None:
            with self.lock:
                got = self._ids[text] = list(
                    self.wrapper.encode(text, add_special_tokens=False))
        return got

    def label_texts(self) -> list[str]:
        """A..Z, then AA..ZZ, the codes that are one token, at most 255."""
        codes = [chr(65 + i) for i in range(26)]
        codes += [a + b for a in codes[:26] for b in codes[:26]]
        out, seen = [], set()
        with self.lock:
            for c in codes:
                ids = self.wrapper.encode(c, add_special_tokens=False)
                if len(ids) == 1 and ids[0] not in seen and len(out) < MAX_LABELS:
                    out.append(c)
                    seen.add(ids[0])
        if len(out) < 2:
            raise ValueError("the tokenizer gives fewer than two option labels "
                             "their own token")
        return out

    def letter_ids(self, labels: list[str] | None = None) -> list[int]:
        labels = labels if labels is not None else self.label_texts()
        with self.lock:
            return [self.wrapper.encode(c, add_special_tokens=False)[0] for c in labels]


def setup(processor, model, *, lock=None) -> tuple[TemplateFormat, TemplateTokens] | None:
    """The format and tokens of the template readout for this model, or
    None when it ships no template this readout drives."""
    got = find(processor)
    if got is None:
        return None
    source, meta = got
    tokens = TemplateTokens(processor, lock=lock)
    cut = _control_ids(model)
    return TemplateFormat(source, meta, tokens.label_texts(), cut), tokens


def _control_ids(model) -> list[int]:
    from .ar_reader import text_model

    m: Any = text_model(model)
    args = getattr(m, "args", None) or getattr(getattr(m, "language_model", None), "args", None)
    return list(getattr(args, "adapter_token_ids", None) or [])
