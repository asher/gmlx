"""Scoring a reply at its own tokens, where its text tokenizes differently.

SmolLM2-135M-Instruct's greedy reply to PROMPT writes a blank line as two
single newline tokens before the next item. As text, the same reply
tokenizes the two newlines as one token. Scoring the generated ids gives
the next token's logprob close to the one chat generated it with. Scoring
the text, as an assistant prefill does, reads other ids and gets a value
far off, or does not rank the token in its top 20 at all.

``integration``; finds the GGUF by name under ``KQUANT_TEST_GGUF_DIR`` and
skips without it.
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

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
GGUF = "SmolLM2-135M-Instruct-Q8_0.gguf"
# Pinned by a search over a few benign prompts (logs/g11-ws-explore2).
PROMPT = "Write a numbered list of three tips for sleep."
# As tests/serve/test_score_gguf.py.
LOGP_TOL = 1e-3
LOGP_SPREAD = 0.25
MARGIN = 2.0

_CHILD = r"""
import importlib, json, sys
spec = json.loads(sys.argv[1])
from fastapi.testclient import TestClient
import gmlx.serve.bridge_vlm as serving
from gmlx.config import build_config
from gmlx.serve.patches import install_server_patches
from gmlx.serve.residency import install_gguf_residency_pool

cfg = build_config({
    "server": {"family_defaults": False, "cache": {"enabled": False}},
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

msgs = [{"role": "user", "content": spec["prompt"]}]
out = {}
with TestClient(app, base_url="http://127.0.0.1") as c:
    r = ok(c.post("/v1/chat/completions", json={
        "model": "m", "messages": msgs, "temperature": 0, "max_tokens": 32,
        "return_token_ids": True, "logprobs": True}))
    P, R = r["prompt_token_ids"], r["choices"][0]["token_ids"]
    out["P"], out["R"] = P, R
    out["generated"] = [e["logprob"] for e in r["choices"][0]["logprobs"]["content"]]
    out["pieces"] = [ok(c.post("/detokenize", json={"model": "m", "tokens": [t]}))["prompt"]
                     for t in R]
    k = next((i for i in range(2, len(R))
              if out["pieces"][i - 2:i] == ["\n", "\n"]
              and out["pieces"][i][:1].strip()), None)
    out["k"] = k
    if k is not None:
        scored = ok(c.post("/v1/completions", json={
            "model": "m", "prompt": P + R, "echo": True, "logprobs": 0,
            "max_tokens": 0}))
        out["by_ids"] = scored["choices"][0]["logprobs"]["token_logprobs"]
        text = ok(c.post("/detokenize", json={"model": "m", "tokens": R[:k]}))["prompt"]
        cont = msgs + [{"role": "assistant", "content": text}]
        out["text_ids"] = ok(c.post("/tokenize", json={
            "model": "m", "messages": cont, "continue_final_message": True,
            "add_generation_prompt": False}))["tokens"]
        pre = ok(c.post("/v1/chat/completions", json={
            "model": "m", "messages": cont, "temperature": 0, "max_tokens": 1,
            "logprobs": True, "top_logprobs": 20, "return_tokens_as_token_ids": True,
            "continue_final_message": True, "add_generation_prompt": False}))
        e = pre["choices"][0]["logprobs"]["content"][0]
        tops = {tid(t["token"]): t["logprob"] for t in e["top_logprobs"]}
        tops.setdefault(tid(e["token"]), e["logprob"])
        out["by_text"] = tops.get(R[k])
print("RESULT " + json.dumps(out))
"""


@pytest.fixture(scope="module")
def run(gguf_dir):
    hits = sorted(Path(gguf_dir).rglob(GGUF))
    if not hits:
        pytest.skip(f"no {GGUF} under KQUANT_TEST_GGUF_DIR")
    spec = {"gguf": str(hits[0]), "prompt": PROMPT}
    r = subprocess.run([sys.executable, "-c", _CHILD, json.dumps(spec)],
                       capture_output=True, text=True, cwd=REPO,
                       env=dict(os.environ))
    assert r.returncode == 0, r.stderr[-4000:]
    line = next(x for x in reversed(r.stdout.splitlines())
                if x.startswith("RESULT "))
    return json.loads(line[len("RESULT "):])


def _bound(logprob):
    return MARGIN * (LOGP_TOL + LOGP_SPREAD * (1 - math.exp(logprob)))


def test_the_reply_has_a_newline_run_before_a_letter(run):
    assert run["k"] is not None, run["pieces"]
    assert run["k"] < 32


def test_scoring_the_ids_matches_the_generated_logprobs(run):
    n = len(run["P"])
    for i, want in enumerate(run["generated"]):
        got = run["by_ids"][n + i]
        assert abs(got - want) <= _bound(want), (i, got, want)


def test_scoring_the_text_reads_other_ids_and_misses(run):
    k, P, R = run["k"], run["P"], run["R"]
    assert run["text_ids"] != P + R[:k]
    want = run["generated"][k]
    got = run["by_text"]
    print(f"k={k}: generated {want:.4f}, by ids {run['by_ids'][len(P) + k]:.4f}, "
          f"by text {got}")
    assert got is None or abs(got - want) > 10 * _bound(want)
