#!/usr/bin/env python3
"""Ternary Bonsai 2 27B (PTQ1_0 and PQ2_0) against recorded PrismML fork
references.

The fork's Metal build is the only decoder of these codecs outside gmlx,
so its greedy runs are checked in under ``tests/fixtures/bonsai_oracle``
with the fork's top-20 logprobs at every step; ``reference.json`` carries
the recipe. Each run teacher-forces the fork's tokens through gmlx on both
routes, one prefill of prompt plus tokens and a prompt prefill followed
by one-token decode steps, and gates the logprob delta on the fork's
top-20 entries and the argmax agreement. The greedy text itself is not
asserted: a near-tie step flips on activation rounding alone.

``integration``; skips unless the run's file is under
``KQUANT_TEST_GGUF_DIR``.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from test_long_context import _load

ARCH = "qwen35"
_ORACLE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "bonsai_oracle"

# Observed bf16 deltas on the fork's top-20 are 0.14-0.33 nats, inside
# gmlx's own prefill-versus-decode spread of 0.19-0.38; float16 halves
# both. The argmax matched at every step; two flips allow for near ties.
MAX_TOPK_DELTA = 0.75
MIN_TOP1 = 46

pytestmark = pytest.mark.integration


def _reference() -> dict:
    return json.loads((_ORACLE_DIR / "reference.json").read_text())


def _runs():
    return {r["name"]: r for r in _reference()["runs"]}


_MODELS: dict[str, tuple] = {}


@pytest.fixture(scope="module")
def bonsai_model(gguf_index):
    """One load per Bonsai file for the whole module, released at teardown."""

    def get(filename):
        if filename not in _MODELS:
            paths = [p for p in gguf_index.get(ARCH, []) if Path(p).name == filename]
            if not paths:
                pytest.skip(f"{filename} not under KQUANT_TEST_GGUF_DIR")
            _MODELS[filename] = _load(paths[0])
        return _MODELS[filename]

    yield get
    from gmlx.stream.installs import release

    for model, _, _ in _MODELS.values():
        release(model)
    _MODELS.clear()


def _logprobs(logits):
    return mx.log(mx.softmax(logits.astype(mx.float32), axis=-1))


@pytest.mark.parametrize("name", sorted(_runs()))
def test_logprobs_match_the_fork(bonsai_model, name):
    from mlx_lm.models.cache import make_prompt_cache

    run = _runs()[name]
    model, _, tok = bonsai_model(run["file"])
    text = (_ORACLE_DIR / run["prompt"]).read_text()
    ids = tok.encode(text, add_special_tokens=False)
    assert ids == run["prompt_ids"], f"{name}: tokenization differs from the fork"

    tokens = run["tokens"]
    full = ids + tokens
    lp_pre = _logprobs(model(mx.array(full)[None])[0])
    cache = make_prompt_cache(model)
    rows = [model(mx.array(ids)[None], cache=cache)[0, -1]]
    for t in tokens[:-1]:
        rows.append(model(mx.array([[t]]), cache=cache)[0, -1])
    lp_dec = _logprobs(mx.stack(rows))
    mx.eval(lp_pre, lp_dec)
    pre = np.array(lp_pre)[len(ids) - 1 : len(full) - 1]
    dec = np.array(lp_dec)

    worst = 0.0
    top1 = {"prefill": 0, "decode": 0}
    for i, (tok_id, top) in enumerate(zip(tokens, run["top"])):
        top_ids = [t for t, _ in top]
        top_lp = np.array([lp for _, lp in top])
        for route, lp in (("prefill", pre), ("decode", dec)):
            worst = max(worst, float(np.abs(lp[i, top_ids] - top_lp).max()))
            top1[route] += int(lp[i].argmax()) == tok_id
    n = len(tokens)
    assert worst < MAX_TOPK_DELTA, f"{name}: max top-20 logprob delta {worst:.3f} nats"
    assert min(top1.values()) >= MIN_TOP1, f"{name}: argmax agreement {top1} of {n}"
