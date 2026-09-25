# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from vLLM examples/features/structured_diffusion/structured_server.py
# at ab3de6edf2 and modified for gmlx (Apache-2.0; see
# licenses/vllm-LICENSE).
"""Question schema parsing and the Jev request-body mapping.

The rules match the vLLM proxy. A value that cannot be converted to a
number raises ``SchemaError``, which the route answers with 422."""

from __future__ import annotations

import json
from dataclasses import dataclass

JEV_EXTENSIONS = (
    "instructions",
    "samples",
    "auto_max",
    "auto_threshold",
    "steps",
    "think",
    "ask",
    "chunk_rows",
    "chunk_prompt",
    "sequential",
)

# Request fields gmlx adds to the vLLM set: the settings of ``think: "auto"``.
GMLX_EXTENSIONS = ("think_threshold", "think_budget")

THINK_THRESHOLD = 0.8
THINK_BUDGET = 64

DEFAULT_SEED = 42


class SchemaError(ValueError):
    pass


@dataclass(frozen=True)
class Limits:
    max_questions: int = 64
    max_samples: int = 32


def _int(value, what: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        raise SchemaError(f"schema: {what} must be an integer") from None


def _float(value, what: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise SchemaError(f"schema: {what} must be a number") from None


def parse_seed(body: dict) -> int:
    """The request ``seed`` as an int, 42 when absent."""
    value = body.get("seed", DEFAULT_SEED)
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        raise SchemaError("seed: must be an integer") from None


def parse_schema(value, limits: Limits = Limits()) -> dict:
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("questions"), list)
        or not value["questions"]
    ):
        raise SchemaError("schema: needs a non-empty questions array")
    if len(value["questions"]) > limits.max_questions:
        raise SchemaError(f"schema: at most {limits.max_questions} questions")
    qs = []
    seen: set = set()
    for q in value["questions"]:
        qid = str(q.get("id", "")).strip()
        if not qid or ":" in qid or "\n" in qid:
            raise SchemaError(
                f"question id {qid!r} must be non-empty, no ':' or newline"
            )
        if qid in seen:
            raise SchemaError(f"duplicate question id {qid!r}")
        seen.add(qid)
        kind = q.get("type")
        if kind in ("noul", "bool", "boolean"):
            kind = "noul"
            crit = q.get("criteria") or {}
            choices = [("yes", crit.get("true")), ("no", crit.get("false"))]
            labels = ["yes", "no"]
        elif kind == "choice":
            opts = q.get("options") or []
            choices = [
                (o["name"], o.get("description"))
                if isinstance(o, dict)
                else (str(o), None)
                for o in opts
            ]
            labels = [chr(ord("A") + i) for i in range(len(choices))]
        elif kind == "score":
            choices = [(str(level), None) for level in (q.get("levels") or [])]
            labels = (
                [str(i + 1) for i in range(len(choices))]
                if len(choices) <= 9
                else [chr(ord("A") + i) for i in range(len(choices))]
            )
        else:
            raise SchemaError(f"question {qid!r}: unknown type {kind!r}")
        if len(choices) < 2:
            raise SchemaError(f"question {qid!r}: needs at least two alternatives")
        if len(choices) > 26:
            raise SchemaError(f"question {qid!r}: at most 26 alternatives")
        deps = q.get("depends_on") or []
        ask_if = q.get("ask_if") or {}
        if not isinstance(deps, list) or not all(isinstance(d, str) for d in deps):
            raise SchemaError(
                f"question {qid!r}: depends_on must be a list of question ids"
            )
        if not isinstance(ask_if, dict) or not all(
            isinstance(v, list) and v for v in ask_if.values()
        ):
            raise SchemaError(
                f"question {qid!r}: ask_if must map a question id to a "
                "non-empty list of its answers"
            )
        qs.append(
            {
                "id": qid,
                "type": kind,
                "instructions": str(q.get("instructions", "")),
                "choices": choices,
                "labels": labels,
                "depends_on": list(dict.fromkeys(list(deps) + list(ask_if))),
                "ask_if": ask_if,
                "alone": bool(q.get("alone", False)),
            }
        )
    by_id = {q["id"]: q for q in qs}
    for q in qs:
        for dep in q["depends_on"]:
            if dep not in by_id or dep == q["id"]:
                raise SchemaError(
                    f"question {q['id']!r}: depends on unknown question {dep!r}"
                )
        for dep, vals in q["ask_if"].items():
            names = [c[0] for c in by_id[dep]["choices"]]
            if any(v not in names for v in vals):
                raise SchemaError(
                    f"question {q['id']!r}: ask_if values for {dep!r} "
                    f"must be among {names}"
                )
    schedule(qs)  # refuses a cycle
    samples = value.get("samples", "auto")
    if samples == "auto":
        policy = {
            "mode": "auto",
            "max": max(
                1, min(_int(value.get("auto_max", 4), "auto_max"), limits.max_samples)
            ),
            "threshold": _float(value.get("auto_threshold", 0.1), "auto_threshold"),
        }
    elif isinstance(samples, int) and samples >= 1:
        policy = {"mode": "fixed", "n": min(samples, limits.max_samples)}
    else:
        raise SchemaError('schema: samples must be a positive count or "auto"')
    ask = value.get("ask")
    if ask is not None:
        if (
            not isinstance(ask, list)
            or not ask
            or any(not isinstance(a, str) or a not in seen for a in ask)
        ):
            raise SchemaError("schema: ask must list question ids from this schema")
        for q in qs:
            if q["id"] in ask and any(d not in ask for d in q["depends_on"]):
                raise SchemaError(
                    f"schema: ask names {q['id']!r} but not everything it depends on"
                )
    chunk_rows = value.get("chunk_rows")
    if chunk_rows is not None and (not isinstance(chunk_rows, int) or chunk_rows < 8):
        raise SchemaError("schema: chunk_rows must be an integer of at least 8")
    chunk_prompt = value.get("chunk_prompt", "own")
    if chunk_prompt not in ("shared", "own"):
        raise SchemaError('schema: chunk_prompt must be "shared" or "own"')
    sequential = bool(value.get("sequential", False))
    think = value.get("think", 0)
    think_auto = None
    if think == "auto":
        think_auto = {
            "threshold": _float(value.get("think_threshold", THINK_THRESHOLD),
                                "think_threshold"),
            "budget": _int(value.get("think_budget", THINK_BUDGET), "think_budget"),
        }
        if not 0 < think_auto["threshold"] <= 1:
            raise SchemaError("schema: think_threshold must be above 0 and at most 1")
        if not 1 <= think_auto["budget"] <= 4096:
            raise SchemaError("schema: think_budget must be 1 to 4096 tokens")
        think = 0
    elif isinstance(think, bool) or not isinstance(think, int) or not 0 <= think <= 4096:
        raise SchemaError(
            'schema: think must be a thought budget in tokens, 0 to 4096, or "auto"')
    return {
        "questions": qs,
        "instructions": value.get("instructions"),
        "policy": policy,
        "steps": max(1, min(_int(value.get("steps", 1), "steps"), 8)),
        "think": think,
        "think_auto": think_auto,
        "ask": ask,
        "chunk_rows": chunk_rows,
        "chunk_prompt": chunk_prompt,
        "sequential": sequential,
        "format": "lines" if len(qs) <= 10 else "indexed",
    }


def ignored_fields(body, schema) -> set:
    """Request fields that parse but change nothing: the think settings when
    ``think`` is not ``"auto"``."""
    if schema["think_auto"] is not None:
        return set()
    return set(body) & set(GMLX_EXTENSIONS)


def schedule(qs):
    """Questions in stages: a question's stage comes after the stages of
    everything it depends on. Declaration order is kept within a stage."""
    ids = {q["id"] for q in qs}
    pending = list(qs)
    done: set = set()
    levels = []
    while pending:
        level = [
            q
            for q in pending
            if all(d in done or d not in ids for d in q["depends_on"])
        ]
        if not level:
            raise SchemaError(
                "schema: dependency cycle among " + ", ".join(q["id"] for q in pending)
            )
        levels.append(level)
        done |= {q["id"] for q in level}
        pending = [q for q in pending if q["id"] not in done]
    return levels


def jev_schema(body, limits: Limits = Limits(), defaults: dict | None = None) -> dict:
    """The schema from a Jev request body. ``defaults`` holds extension
    values, such as the server's ``think``, for fields the body omits."""
    qs = body.get("questions")
    if not isinstance(qs, dict) or not qs:
        raise SchemaError("questions: needs a non-empty map of id -> question")
    out = []
    for qid, q in qs.items():
        if not isinstance(q, dict):
            raise SchemaError(f"question {qid!r}: must be an object")
        kind, crit, ins = q.get("type"), q.get("criteria"), q.get("instructions", "")
        item = {
            "id": qid,
            "type": kind,
            "instructions": ins if isinstance(ins, str) else json.dumps(ins),
        }
        if kind == "noul":
            if crit is not None and not isinstance(crit, dict):
                raise SchemaError(
                    f"question {qid!r}: noul criteria must be an object with "
                    "true and false"
                )
            item["criteria"] = crit
        elif kind == "choice":
            if not isinstance(crit, dict) or not crit:
                raise SchemaError(
                    f"question {qid!r}: choice criteria must map option names "
                    "to descriptions"
                )
            item["options"] = [
                {"name": str(n), "description": d} for n, d in crit.items()
            ]
        elif kind == "score":
            if not isinstance(crit, list):
                raise SchemaError(
                    f"question {qid!r}: score criteria must be an ordered list "
                    "of levels"
                )
            item["levels"] = crit
        else:
            raise SchemaError(f"question {qid!r}: unknown type {kind!r}")
        for key in ("depends_on", "ask_if", "alone"):
            if key in q:
                item[key] = q[key]
        out.append(item)
    schema = {k: body[k] for k in JEV_EXTENSIONS + GMLX_EXTENSIONS if k in body}
    for k, v in (defaults or {}).items():
        schema.setdefault(k, v)
    schema["questions"] = out
    return parse_schema(schema, limits)
