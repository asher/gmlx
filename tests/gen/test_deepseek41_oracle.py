#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash against a recorded llama.cpp greedy reference.

The oracle is CPU-only and prefills the 246 GB Q2_K file at 2.3 tok/s, so
its answers are checked in under ``tests/fixtures/deepseek41_oracle``
rather than regenerated; ``reference.json`` carries the recipe.

At Q2_K the greedy trajectory is not a usable instrument: two gmlx runs
that differ only in prefill chunk size fork from each other at the same
token where gmlx and the oracle fork. The answer is stable across every
run, so that is what these gates assert. The needle run also needs the
compressed pool and the indexer's top-k, which the 128-token window
cannot reach.

``integration``; skips without a deepseek41 GGUF under
``KQUANT_TEST_GGUF_DIR``.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import pytest

from test_long_context import _load, _require

ARCH = "deepseek41"
_ORACLE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "deepseek41_oracle"

pytestmark = pytest.mark.integration


def _reference() -> dict:
    return json.loads((_ORACLE_DIR / "reference.json").read_text())


def _runs():
    return {r["name"]: r for r in _reference()["runs"]}


@pytest.fixture(scope="module")
def ds41_model(gguf_index):
    """One V4.1 load for the whole file, released at teardown: a streaming
    load mlocks a weight pin and a 75 GB arena, and dropping the model
    gives neither back until the feeder cycle is collected."""
    path = _require(gguf_index, ARCH)
    model, config, tok = _load(path)
    yield path, model, config, tok
    from gmlx.stream.installs import release

    release(model)


@pytest.mark.parametrize("name", ["arith", "needle_2k"])
def test_recorded_prompt_tokenizes_to_the_oracle_length(ds41_model, name):
    # The prompt files are gmlx's own rendered output, so a length drift
    # means the bundled template changed under the recorded answers.
    _, _, _, tok = ds41_model
    run = _runs()[name]
    text = (_ORACLE_DIR / run["prompt"]).read_text()
    ids = tok.encode(text, add_special_tokens=False)
    assert len(ids) == run["prompt_tokens"]
    assert ids.count(0) == 1 and ids[0] == 0      # exactly one BOS, at the front


def _greedy(model, tok, ids, n):
    from mlx_lm.generate import stream_generate

    out = [int(r.token) for r in
           stream_generate(model, tok, mx.array(ids), max_tokens=n)]
    return tok.decode(out)


@pytest.mark.parametrize("name", ["arith", "needle_2k"])
def test_answer_matches_the_oracle(ds41_model, name):
    _, model, _, tok = ds41_model
    run = _runs()[name]
    text = (_ORACLE_DIR / run["prompt"]).read_text()
    ids = tok.encode(text, add_special_tokens=False)

    got = _greedy(model, tok, ids, run["n_predict"])
    answer = got.split("</think>")[-1]
    assert run["answer"] in answer, (
        f"{name}: oracle answers {run['answer']!r}\n  got: {got[-200:]!r}")
