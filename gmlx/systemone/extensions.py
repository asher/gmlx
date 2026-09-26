"""gmlx's additions to the Jev decision API: ``think: "auto"`` and the
server's request defaults.

``think: "auto"`` runs a decision without a thought, and runs it again with
one when an answer is unsure. ``think_threshold`` and ``think_budget`` set
the confidence floor and the thought budget. ``server.systemone`` supplies
``think`` and both settings for a request that omits them."""

from __future__ import annotations

import functools
from typing import Callable, Optional

from .decide import ReadEngine, decide_once
from .schema import Limits, SchemaError, _float, _int, jev_schema
from .template import TemplateResolver

# Request fields gmlx adds to the vLLM set: the settings of ``think: "auto"``.
GMLX_EXTENSIONS = ("think_threshold", "think_budget")

THINK_THRESHOLD = 0.8
THINK_BUDGET = 64


def request_schema(body, limits: Limits = Limits(),
                   defaults: dict | None = None) -> dict:
    """The schema of a request body as ``jev_schema`` parses it, with the
    server's ``defaults`` for the think fields the body omits. With
    ``think: "auto"`` the schema starts without a thought and carries the
    auto settings as ``think_auto``."""
    def value(key, default=None):
        return body.get(key, (defaults or {}).get(key, default))

    think = value("think", 0)
    auto = None
    if think == "auto":
        auto = {
            "threshold": _float(value("think_threshold", THINK_THRESHOLD),
                                "think_threshold"),
            "budget": _int(value("think_budget", THINK_BUDGET), "think_budget"),
        }
        if not 0 < auto["threshold"] <= 1:
            raise SchemaError("schema: think_threshold must be above 0 and at most 1")
        if not 1 <= auto["budget"] <= 4096:
            raise SchemaError("schema: think_budget must be 1 to 4096 tokens")
        think = 0
    schema = jev_schema(dict(body, think=think), limits)
    schema["think_auto"] = auto
    return schema


def ignored_fields(body, schema) -> set:
    """Request fields that parse but change nothing: the think settings when
    ``think`` is not ``"auto"``."""
    if schema.get("think_auto") is not None:
        return set()
    return set(body) & set(GMLX_EXTENSIONS)


def decide(
    schema,
    state_text: str,
    *,
    engine: ReadEngine,
    resolver: TemplateResolver,
    chat_ids: Callable[[str, bool], list[int]],
    seed: int,
    constrained: bool,
    canvas_len: int,
    decode: Callable[[list[int]], str],
    should_stop: Optional[Callable[[], bool]] = None,
):
    """One decision over a text state. ``chat_ids(system_text, thinking)``
    renders the chat prompt for ``state_text`` as token ids. Returns the
    body (answers and diagnostics) and the completion-token count.

    Without ``think: "auto"`` this is ``decide_once``. With it, the
    decision runs without a thought first. When an answer's confidence is
    below the threshold, it runs again with a thought of the auto budget,
    and that run gives the answers, the samples, the question diagnostics
    and the completion-token count. ``timing`` covers both runs, and
    ``think_auto`` holds the first run's confidences, reads and time."""
    auto = schema.get("think_auto")
    run = functools.partial(
        decide_once, engine=engine, resolver=resolver, chat_ids=chat_ids,
        seed=seed, constrained=constrained, canvas_len=canvas_len,
        decode=decode, should_stop=should_stop)
    if not auto:
        return run(schema)
    first, first_rows = run(schema)
    confidence = {qid: a["confidence"] for qid, a in first["answers"].items()
                  if a is not None}
    unsure = [q["id"] for q in schema["questions"]
              if confidence.get(q["id"], 1.0) < auto["threshold"]]
    first_timing = first["diagnostics"]["timing"]
    info = {"threshold": auto["threshold"], "budget": auto["budget"],
            "thought": bool(unsure), "unsure": unsure, "confidence": confidence,
            "reads": first_timing["reads"], "total_ms": first_timing["total_ms"]}
    if not unsure:
        first["diagnostics"]["think_auto"] = info
        return first, first_rows
    body, rows = run(dict(schema, think=auto["budget"]))
    timing = body["diagnostics"]["timing"]
    timing["total_ms"] += first_timing["total_ms"]
    timing["reads"] += first_timing["reads"]
    body["diagnostics"]["think_auto"] = info
    return body, rows
