"""Id prompts and echo scores against llama-server, on real weights.

tests/fixtures/llama_completion_ids.json holds, per GGUF, llama-server's
ids for a text prompt, its greedy reply to those ids with the logprob of
each reply token, and the logprob of each reply token read from a prefill
of the ids before it (``scripts/llama_fixtures.py completion-ids``). Each
row serves the GGUF in a child process and checks that:

- ``/tokenize`` gives llama-server's prompt ids;
- ``/v1/completions`` on the ids gives the same greedy ids, each logprob
  within the decode bound of llama-server's;
- one echo request on the prompt and reply ids scores each reply token
  within the scoring bound of llama-server's prefill read.

``integration``; finds each GGUF by name and size under
``KQUANT_TEST_GGUF_DIR`` and skips without it.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")

# A device import, so the GPU gate takes this file.
import mlx.core as mx  # noqa: E402,F401

import llama_fixture  # noqa: E402

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
FIXTURE = llama_fixture.load("llama_completion_ids.json")
# The decode bound of tests/serve/test_prefill_logp.py, and MARGIN times it
# for a scored token, as in tests/serve/test_score_gguf.py.
LOGP_TOL = 1e-3
LOGP_SPREAD = 0.25
MARGIN = 2.0
ARCH = {"SmolLM2-135M-Instruct-Q8_0.gguf": "llama",
        "Qwen3.5-9B-Q4_K_M.gguf": "qwen35"}

_CHILD = r"""
import importlib, json, sys
spec = json.loads(sys.argv[1])
from fastapi.testclient import TestClient
import gmlx.serve.bridge_vlm as serving
from gmlx.config import build_config
from gmlx.serve.patches import install_server_patches
from gmlx.serve.residency import install_gguf_residency_pool

cfg = build_config({
    "server": {"family_defaults": False},
    "models": {"m": {"path": spec["gguf"], "speculative": False}}})
serving.register_resolved_models(cfg)
serving.install_gguf_server_bridge()
install_gguf_residency_pool()
install_server_patches(cfg, reload_fn=lambda *a, **k: None)
app = importlib.import_module("gmlx._vendor.mlx_vlm.server.app").app

def ok(r):
    assert r.status_code == 200, r.text
    return r.json()

def tid(name):
    return int(name.split(":", 1)[1])

out = {}
with TestClient(app, base_url="http://127.0.0.1") as client:
    out["tokenized"] = ok(client.post("/tokenize", json={
        "model": "m", "prompt": spec["prompt"]}))["tokens"]
    body = ok(client.post("/v1/completions", json={
        "model": "m", "prompt": spec["prompt_ids"], "max_tokens": len(spec["ids"]),
        "temperature": 0, "logprobs": 5, "return_tokens_as_token_ids": True}))
    lp = body["choices"][0]["logprobs"]
    out["ids"] = [tid(t) for t in lp["tokens"]]
    out["logprobs"] = lp["token_logprobs"]
    ids = spec["prompt_ids"] + spec["ids"]
    body = ok(client.post("/v1/completions", json={
        "model": "m", "prompt": ids, "echo": True, "logprobs": 1,
        "max_tokens": 0, "return_tokens_as_token_ids": True}))
    lp = body["choices"][0]["logprobs"]
    assert [tid(t) for t in lp["tokens"]] == ids
    out["echo"] = lp["token_logprobs"][len(spec["prompt_ids"]):]
print("RESULT " + json.dumps(out))
"""


def _bound(logprob: float) -> float:
    return LOGP_TOL + LOGP_SPREAD * (1 - math.exp(logprob))


@pytest.fixture(scope="module", params=FIXTURE["rows"],
                ids=[r["run"].split("/")[1] for r in FIXTURE["rows"]])
def run(request, gguf_index):
    row = request.param
    path = llama_fixture.find_file(FIXTURE, row["gguf"],
                                   gguf_index.get(ARCH[row["gguf"]], []))
    spec = {"gguf": path, "prompt": row["prompt"],
            "prompt_ids": row["prompt_ids"], "ids": row["ids"]}
    r = subprocess.run([sys.executable, "-c", _CHILD, json.dumps(spec)],
                       capture_output=True, text=True, cwd=REPO,
                       env=dict(os.environ))
    assert r.returncode == 0, r.stderr[-4000:]
    line = next(x for x in reversed(r.stdout.splitlines())
                if x.startswith("RESULT "))
    return row, json.loads(line[len("RESULT "):])


def test_tokenize_gives_the_llama_server_prompt_ids(run):
    row, out = run
    assert out["tokenized"] == row["prompt_ids"]


def test_an_id_prompt_generates_the_llama_server_reply(run):
    row, out = run
    assert out["ids"] == row["ids"]
    gaps = [abs(a - b) for a, b in zip(out["logprobs"], row["logprobs"])]
    print("worst generated gap:", max(gaps), "of its bound:",
          max(g / _bound(w) for g, w in zip(gaps, row["logprobs"])))
    over = [(k, g) for k, (g, want) in enumerate(zip(gaps, row["logprobs"]))
            if g > _bound(want)]
    assert not over, over


def test_echo_scores_each_reply_token_as_llama_server_prefills_it(run):
    row, out = run
    gaps = [abs(a - b) for a, b in zip(out["echo"], row["prefill_logprobs"])]
    print("worst echo gap:", max(gaps), "of its bound:",
          max(g / (MARGIN * _bound(w)) for g, w in zip(gaps, row["prefill_logprobs"])))
    over = [(k, g) for k, (g, want) in enumerate(zip(gaps, row["prefill_logprobs"]))
            if g > MARGIN * _bound(want)]
    assert len(out["echo"]) == len(row["prefill_logprobs"])
    assert not over, over
