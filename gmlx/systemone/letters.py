# SPDX-License-Identifier: Apache-2.0
# Ported from openjev/openjev helper/shim.py at revision 0b6bb6e (the text
# lane of the targeted readout) and modified for gmlx (Apache-2.0; see
# licenses/openjev-LICENSE).
"""The letter readout: OpenJev's decision protocol for autoregressive models.

Each question becomes one user prompt that lists its options under letters.
The answer is the model's score for each of those letters at the first
output position, divided by a temperature and normalized over the letters.
A question with more than 52 options reads each chunk of at most 52, then
the chunk winners, and composes one distribution over all options.

This module is pure Python: the request fields, the prompt text of every
pass, and the answers. Each question is a generator that yields the passes
it needs and receives their candidate scores, so a reader can batch the
passes of many questions into one forward. ``ar_reader`` runs the passes."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Any, Generator

from .decide import answer_name
from .schema import Limits, SchemaError, jev_schema, schedule

LETTERS = [chr(65 + i) for i in range(26)] + [chr(97 + i) for i in range(26)]
# The settings OpenJev's measurements used (serve/SERVE.md).
T = 0.85
NOUL_T = 1.829074
NOUL_BIAS = 0.0
NOUL_TRUE = "The statement is true."
NOUL_FALSE = "The statement is false."
SCORE_SUFFIX = " Rate along the ordered levels below (lowest first)."
ANSWER_LINE = "Answer with the letter of the best option only."

# Request fields the letter readout does not use, top level and per question.
IGNORED = ("instructions", "auto_max", "auto_threshold", "steps", "think",
           "think_threshold", "think_budget", "chunk_rows", "chunk_prompt",
           "sequential", "seed")
IGNORED_PER_QUESTION = ("depends_on", "alone")


class ReadoutIncomplete(RuntimeError):
    """A candidate score came back missing or not finite."""


@dataclass(frozen=True)
class Pass:
    """One prompt to read: the user message and how many letters it lists."""
    text: str
    letters: int


Read = Generator[list[Pass], list[list[float]], Any]


def has_image(state) -> bool:
    """True when the helper would send part of ``state`` as an image."""
    if not isinstance(state, dict):
        return False
    for key in ("screenshot", "image"):
        v = state.get(key)
        if isinstance(v, str) and (v.startswith("data:image") or len(v) > 2000):
            return True
    return False


def state_text(body) -> str:
    """The state as prompt text: an object as JSON with its characters kept,
    anything else as its string."""
    state = body.get("state")
    if state is None:
        raise SchemaError("state: required")
    if isinstance(state, dict):
        return json.dumps(state, ensure_ascii=False)
    return str(state)


def prefix_text(state: str) -> str:
    """The user message up to the state's end, the part every pass shares."""
    return f"State:\n{state}"


def prompt_text(state: str, instructions: str, options) -> str:
    lines = "\n".join(f"[{LETTERS[i]}] {k}: {d}" for i, (k, d) in enumerate(options))
    return (prefix_text(state)
            + f"\n\nQuestion: {instructions}\nOptions:\n{lines}\n\n{ANSWER_LINE}")


def _instructions(q) -> str:
    i = q.get("instructions")
    return i if isinstance(i, str) else str(i)


def _desc(v) -> str:
    if v is None:
        return ""
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def parse(body, limits: Limits = Limits()) -> dict:
    """The letter schema of a Jev request body. The structural rules are the
    Jev ones, without a cap on options. Each question needs ``instructions``.
    ``samples`` is the number of option orders to average."""
    raw = body.get("questions")
    trimmed = {}
    if isinstance(raw, dict):
        trimmed = {
            qid: ({k: v for k, v in q.items() if k not in IGNORED_PER_QUESTION}
                  if isinstance(q, dict) else q)
            for qid, q in raw.items()}
    shared = {"questions": trimmed if isinstance(raw, dict) else raw}
    if "ask" in body:
        shared["ask"] = body["ask"]
    checked = jev_schema(shared, limits, max_alternatives=None)
    questions = []
    for q in checked["questions"]:
        src = raw[q["id"]]
        if src.get("instructions") is None:
            raise SchemaError(f"question {q['id']!r}: instructions is required")
        instructions = _instructions(src)
        crit = src.get("criteria")
        if q["type"] == "choice":
            options = [(k, _desc(v)) for k, v in crit.items()]
        elif q["type"] == "score":
            options = [(str(i), _desc(level)) for i, level in enumerate(crit)]
            instructions += SCORE_SUFFIX
        else:
            crit = crit or {}
            options = [("yes", _desc(crit.get("true")) or NOUL_TRUE),
                       ("no", _desc(crit.get("false")) or NOUL_FALSE)]
        questions.append({**q, "instructions": instructions, "options": options})
    samples = body.get("samples", "auto")
    if samples == "auto":
        orderings = 1
    elif isinstance(samples, int) and not isinstance(samples, bool) and samples >= 1:
        orderings = min(samples, limits.max_samples)
    else:
        raise SchemaError('schema: samples must be a positive count or "auto"')
    return {"questions": questions, "ask": checked["ask"], "orderings": orderings}


def ignored(body) -> set:
    """The request fields this readout ignores, per-question ones by name."""
    out = set(body) & set(IGNORED)
    for q in (body.get("questions") or {}).values():
        if isinstance(q, dict):
            out |= set(q) & set(IGNORED_PER_QUESTION)
    return out


def _softmax(z: list[float]) -> list[float]:
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def _probs(scores: list[float]) -> list[float]:
    """Letter scores to probabilities: divided by ``T``, then normalized.
    Log-probabilities and raw logits give the same result."""
    if any(not math.isfinite(v) for v in scores):
        bad = sum(not math.isfinite(v) for v in scores)
        raise ReadoutIncomplete(f"{bad} of {len(scores)} candidate scores not finite")
    return _softmax([v / T for v in scores])


def _readout(state: str, instructions: str, options, orderings: int) -> Read:
    """Probabilities aligned with ``options``, averaged over ``orderings``
    letterings. One ordering keeps the given order."""
    if orderings <= 1:
        (scores,) = yield [Pass(prompt_text(state, instructions, options), len(options))]
        return _probs(scores)
    orders = []
    for j in range(orderings):
        order = list(range(len(options)))
        random.Random(j).shuffle(order)
        orders.append(order)
    got = yield [Pass(prompt_text(state, instructions, [options[i] for i in order]),
                      len(options)) for order in orders]
    acc = [0.0] * len(options)
    for order, scores in zip(orders, got):
        p = _probs(scores)
        for pos, i in enumerate(order):
            acc[i] += p[pos] / orderings
    return acc


def gather(reads: list[Read]) -> Read:
    """Run reads side by side: each round yields every pending pass of every
    read, and the result lists each read's value."""
    values: list[Any] = [None] * len(reads)
    pending: dict[int, list[Pass]] = {}
    for i, r in enumerate(reads):
        try:
            pending[i] = next(r)
        except StopIteration as e:
            values[i] = e.value
    while pending:
        order = list(pending)
        got = yield [p for i in order for p in pending[i]]
        pos, nxt = 0, {}
        for i in order:
            n = len(pending[i])
            part, pos = got[pos:pos + n], pos + n
            try:
                nxt[i] = reads[i].send(part)
            except StopIteration as e:
                values[i] = e.value
        pending = nxt
    return values


def distribution(state: str, instructions: str, options, orderings: int) -> Read:
    """The option distribution. Above 52 options each near-equal chunk of at
    most 52 is read, then the chunk winners, and the two compose as
    p(i) ~ p_final(chunk of i) * p_chunk(i) / p_chunk(winner of that chunk)."""
    if len(options) <= 52:
        return (yield from _readout(state, instructions, options, orderings))
    k = -(-len(options) // 52)
    size = -(-len(options) // k)
    chunks = [options[i:i + size] for i in range(0, len(options), size)]
    parts = yield from gather(
        [_readout(state, instructions, c, orderings) for c in chunks])
    wins = [max(range(len(p)), key=p.__getitem__) for p in parts]
    final = yield from _readout(
        state, instructions, [c[w] for c, w in zip(chunks, wins)], orderings)
    raw = [final[c] * pi / p[w] for c, (p, w) in enumerate(zip(parts, wins))
           for pi in p]
    s = sum(raw)
    return [v / s for v in raw]


def answer(state: str, q, orderings: int) -> Read:
    """One question's internal answer: the chosen label, the probabilities by
    option name and the chosen probability."""
    p = yield from distribution(state, q["instructions"], q["options"], orderings)
    names = [c[0] for c in q["choices"]]
    if q["type"] == "noul":
        py = min(max(p[0], 1e-4), 1 - 1e-4)
        z = math.log(py / (1 - py)) / NOUL_T + NOUL_BIAS
        yes = 1 / (1 + math.exp(-z))
        return {"type": "noul", "label": "yes" if yes >= 0.5 else "no",
                "noul": yes, "probabilities": {"yes": yes, "no": 1 - yes},
                "chosen_probability": max(yes, 1 - yes)}
    top = max(range(len(p)), key=p.__getitem__)
    a = {"type": q["type"], "label": names[top],
         "probabilities": dict(zip(names, p)), "chosen_probability": p[top]}
    a["choice" if q["type"] == "choice" else "level"] = names[top]
    return a


def decision(schema, state: str) -> Read:
    """The whole request: the questions in stages by their ``ask_if``
    dependencies, every question of a stage read side by side. Returns the
    answers, the stages and the skipped questions."""
    qs = [q for q in schema["questions"]
          if not schema.get("ask") or q["id"] in schema["ask"]]
    by_id = {q["id"]: q for q in qs}
    answered: dict[str, Any] = {}
    stages, skipped = [], {}
    for level in schedule(qs):
        asked = []
        for q in level:
            failed = next(((dep, vals) for dep, vals in q["ask_if"].items()
                           if answer_name(by_id[dep], answered.get(dep)) not in vals),
                          None)
            if failed:
                answered[q["id"]] = None
                skipped[q["id"]] = {
                    "because": failed[0],
                    "was": answer_name(by_id[failed[0]], answered.get(failed[0])),
                    "wanted": failed[1],
                }
                continue
            asked.append(q)
        if not asked:
            continue
        stages.append([q["id"] for q in asked])
        got = yield from gather([answer(state, q, schema["orderings"]) for q in asked])
        answered.update({q["id"]: a for q, a in zip(asked, got)})
    return {"answers": answered, "stages": stages, "skipped": skipped}
