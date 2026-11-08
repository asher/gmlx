"""Scoring with an adapter strength set per request. The rank-1 Qwen3.5-9B
adapter of the kld fixture loads at strength 0, and a score that asks for
``lora`` scale 0.5 equals, within LORA_TOL, the score of a server that
loads the adapter at a default strength of 0.5. Both differ from the bare
model's score.

``integration``; finds the GGUF and the adapter by name and size and skips
without them.
"""

from __future__ import annotations

import json
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
ADAPTER_FIXTURE = llama_fixture.load("kld/llama_perplexity_adapter.json")
ADAPTER = ADAPTER_FIXTURE["rows"][0]["adapter"]
GGUF = "Qwen3.5-9B-Q4_K_M.gguf"
PROMPT = ("The quick brown fox jumps over the lazy dog. A journey of a "
          "thousand miles begins with a single step.")
# A request scale and a default strength publish the same scale to the
# rows; the bound leaves room for a different kernel order.
LORA_TOL = 0.01

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
    "models": {"m": {"path": spec["gguf"], "speculative": False,
                     "adapters": [{"path": spec["adapter"],
                                   "strength": spec["strength"]}]}}})
serving.register_resolved_models(cfg)
serving.install_gguf_server_bridge()
install_gguf_residency_pool()
install_server_patches(cfg, reload_fn=lambda *a, **k: None)
app = importlib.import_module("gmlx._vendor.mlx_vlm.server.app").app

def score(c, ids, extra):
    r = c.post("/v1/completions", json={
        "model": "m", "prompt": ids, "echo": True, "logprobs": 0,
        "max_tokens": 0, **extra})
    assert r.status_code == 200, r.text
    return r.json()["choices"][0]["logprobs"]["token_logprobs"]

out = {}
with TestClient(app, base_url="http://127.0.0.1") as c:
    ids = c.post("/tokenize", json={"model": "m", "prompt": spec["prompt"]}).json()["tokens"]
    for name, extra in spec["requests"].items():
        out[name] = score(c, ids, extra)
print("RESULT " + json.dumps(out))
"""


def _child(spec):
    r = subprocess.run([sys.executable, "-c", _CHILD, json.dumps(spec)],
                       capture_output=True, text=True, cwd=REPO,
                       env=dict(os.environ))
    assert r.returncode == 0, r.stderr[-4000:]
    line = next(x for x in reversed(r.stdout.splitlines())
                if x.startswith("RESULT "))
    return json.loads(line[len("RESULT "):])


def test_a_request_scale_scores_as_a_default_strength(gguf_index):
    path = next((p for p in gguf_index.get("qwen35", []) if Path(p).name == GGUF), None)
    if path is None:
        pytest.skip(f"no {GGUF} under KQUANT_TEST_GGUF_DIR")
    adapter = llama_fixture.find_file(
        ADAPTER_FIXTURE, ADAPTER, [llama_fixture.FIXTURES / "kld" / ADAPTER])
    base = {"gguf": path, "adapter": adapter, "prompt": PROMPT}
    asked = _child({**base, "strength": 0, "requests": {
        "bare": {}, "half": {"lora": [{"id": 0, "scale": 0.5}]}}})
    default = _child({**base, "strength": 0.5, "requests": {"half": {}}})
    half, static, bare = asked["half"][1:], default["half"][1:], asked["bare"][1:]
    gap = max(abs(a - b) for a, b in zip(half, static))
    moved = max(abs(a - b) for a, b in zip(half, bare))
    print(f"request scale vs default strength: max gap {gap:.4g}; "
          f"adapter moved the scores by up to {moved:.4g}")
    assert gap <= LORA_TOL
    assert moved > LORA_TOL
