# SPDX-License-Identifier: Apache-2.0
# Ported from vLLM examples/features/structured_diffusion/structured_server.py
# at ab3de6edf2 (Apache-2.0; see licenses/vllm-LICENSE).
"""The Jev decision contract: the state text and the answer shapes."""

from __future__ import annotations

import json

from .schema import SchemaError


def jev_state(body) -> str:
    """The user message: the state as given when it is a string, else its
    JSON."""
    state = body.get("state")
    if state is None:
        raise SchemaError("state: required")
    return state if isinstance(state, str) else json.dumps(state)


def jev_answer(q, a):
    if a is None:
        return None
    if q["type"] == "noul":
        return {"type": "noul", "noul": a["noul"]}
    if q["type"] == "choice":
        return {
            "type": "choice",
            "choice": a["choice"],
            "probabilities": a["probabilities"],
            "confidence": a["confidence"],
        }
    names = [c[0] for c in q["choices"]]
    probs = {str(i): a["probabilities"][n] for i, n in enumerate(names)}
    return {
        "type": "score",
        "score": sum(i * p for i, p in enumerate(probs.values())),
        "legend": {str(i): n for i, n in enumerate(names)},
        "probabilities": probs,
        "confidence": a["confidence"],
    }


def jev_answers(schema, body) -> dict:
    return {
        q["id"]: jev_answer(q, body["answers"][q["id"]])
        for q in schema["questions"]
    }


def usage(input_tokens: int, output_tokens: int) -> dict:
    return {"input_tokens": int(input_tokens), "output_tokens": int(output_tokens)}


def log_labels(answers) -> str:
    """``id=label`` pairs for the log line, ``skipped`` for a null answer."""
    return " ".join(
        f"{k}={v['label'] if v else 'skipped'}" for k, v in answers.items()
    )
