"""Assistant prefill on real weights: a prompt that continues the first k
generated tokens tokenizes to the generation prompt plus those k ids, and
the prefill scores the next generated token close to the logprob the model
gave it while it generated them. One Qwen row repeats with an adapter
loaded at strength 0 and requested at strength 1 through ``lora``, so the
adapter's delta runs in the decode and in the prefill.

``integration``; finds each GGUF of tests/fixtures/llama_continuation.json
by name and size under ``KQUANT_TEST_GGUF_DIR`` and skips without it.
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
FIXTURE = llama_fixture.load("llama_continuation.json")
# The rank-1 Qwen3.5-9B adapter that the gmlx kld parity test scores.
ADAPTER_FIXTURE = llama_fixture.load("kld/llama_perplexity_adapter.json")
ADAPTER = ADAPTER_FIXTURE["rows"][0]["adapter"]
# The prefilled prompt is scored by a prefill, the generated tokens by
# decode steps, and the two kernels round differently. The error of a
# logprob scales with the probability left to the other tokens, so the
# bound is LOGP_TOL + LOGP_SPREAD * (1 - p). Measured on GPU and CPU, the
# worst case was the Qwen3.5-9B adapter row on CPU at k=2: diff 0.107 at
# logprob -0.72, 0.21 per unit of (1 - p). Next was gemma-4-12B think-on
# on GPU: diff 0.115 at logprob -1.10, 0.17 per unit.
LOGP_TOL = 1e-3
LOGP_SPREAD = 0.25
TOP = 5
STEPS = 6
HISTORY = [{"role": "system", "content": "Be brief."},
           {"role": "user", "content": "Name three rivers in Europe."}]

# Rows where llama-server's continued prompt is the template's own
# generation prompt plus the prefill (tests/fixtures/llama_continuation.json).
# Gemma 4 with thinking on writes the prefill's own opening first, so its
# prefill starts after that text.
# (id, GGUF, architecture, thinking, part the prefill fills, opening,
# adapter)
ROWS = [
    ("qwen35_9b-think-off-content", "Qwen3.5-9B-Q4_K_M.gguf", "qwen35",
     False, "content", "", None),
    ("qwen35_9b-think-on-reasoning", "Qwen3.5-9B-Q4_K_M.gguf", "qwen35",
     True, "reasoning_content", "", None),
    ("qwen35_9b-think-off-content-adapter", "Qwen3.5-9B-Q4_K_M.gguf", "qwen35",
     False, "content", "", ADAPTER),
    ("gemma4_12b-think-off-content", "gemma-4-12B-it-Q6_K.gguf", "gemma4",
     False, "content", "", None),
    ("gemma4_12b-think-on-reasoning", "gemma-4-12B-it-Q6_K.gguf", "gemma4",
     True, "reasoning_content", "<|channel>thought\n", None),
]

# Each row serves its GGUF in a child process: the loader bridge and the
# residency pool are process-wide installs that no fixture undoes, and the
# model's memory goes with the child.
_CHILD = r"""
import importlib, json, os, sys
import mlx.core as mx
if os.environ.get("KQUANT_FORCE_CPU"):
    mx.set_default_device(mx.cpu)
spec = json.loads(sys.argv[1])
from fastapi.testclient import TestClient
import gmlx.serve.bridge_vlm as serving
from gmlx.config import build_config
from gmlx.serve.patches import install_server_patches
from gmlx.serve.residency import install_gguf_residency_pool

entry = {"path": spec["gguf"], "speculative": False}
if spec["adapter"]:
    entry["adapters"] = [{"path": spec["adapter"], "strength": 0}]
cfg = build_config({
    "server": {"family_defaults": False, "cache": {"enabled": False}},
    "models": {"m": entry}})
serving.register_resolved_models(cfg)
serving.install_gguf_server_bridge()
install_gguf_residency_pool()
install_server_patches(cfg, reload_fn=lambda *a, **k: None)
app = importlib.import_module("gmlx._vendor.mlx_vlm.server.app").app
gen = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
prompts = []
real = gen.ResponseGenerator._preprocess_request

def spy(self, prompt, *a, **k):
    out = real(self, prompt, *a, **k)
    ids = out["input_ids"] if isinstance(out, dict) else out
    prompts.append([int(t) for t in ids.reshape(-1).tolist()])
    return out

gen.ResponseGenerator._preprocess_request = spy
base = {"model": "m", "temperature": 0, "logprobs": True,
        "return_tokens_as_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": spec["thinking"]}}
plain = dict(base)
if spec["adapter"]:
    base["lora"] = [{"id": 0, "scale": 1.0}]

def tid(name):
    return int(name.split(":", 1)[1])

def entries(r):
    assert r.status_code == 200, r.text
    return [{"id": tid(e["token"]), "logprob": e["logprob"],
             "bytes": e["bytes"],
             "top": [[tid(t["token"]), t["logprob"]] for t in e["top_logprobs"]]}
            for e in r.json()["choices"][0]["logprobs"]["content"]]

def text_of(es):
    return bytes(b for e in es for b in e["bytes"]).decode()

out = {}
with TestClient(app, base_url="http://127.0.0.1") as client:
    if spec["adapter"]:
        r = client.post("/v1/chat/completions", json={
            **plain, "messages": spec["history"], "max_tokens": 1,
            "top_logprobs": spec["top"]})
        out["plain"] = entries(r)
    r = client.post("/v1/chat/completions", json={
        **base, "messages": spec["history"], "max_tokens": spec["steps"] + 9})
    out["generated"] = gen_entries = entries(r)
    out["prompt"] = prompts[-1]
    out["prefilled"] = []
    skip = next(n for n in range(len(gen_entries) + 1)
                if len(text_of(gen_entries[:n]).encode()) >= len(spec["opening"]))
    out["skip"] = skip
    for k in range(skip + 1, skip + spec["steps"] + 1):
        text = text_of(gen_entries[skip:k])
        final = {"role": "assistant", "content": ""}
        final[spec["part"]] = text
        r = client.post("/v1/chat/completions", json={
            **base, "max_tokens": 1, "top_logprobs": spec["top"],
            "continue_final_message": True,
            "add_generation_prompt": False,
            "messages": [*spec["history"], final]})
        out["prefilled"].append({"k": k, "text": text, "prompt": prompts[-1],
                                 "entries": entries(r)})
print("RESULT " + json.dumps(out))
"""


@pytest.fixture(scope="module", params=ROWS, ids=[r[0] for r in ROWS])
def run(request, gguf_index):
    """One row served in its child: the generated entries, and each
    prefilled request's prompt ids and next-token entry."""
    rid, gguf, arch, thinking, part, opening, adapter = request.param
    path = llama_fixture.find_file(FIXTURE, gguf, gguf_index.get(arch, []))
    if adapter is not None:
        adapter = llama_fixture.find_file(
            ADAPTER_FIXTURE, adapter, [llama_fixture.FIXTURES / "kld" / adapter])
    spec = {"gguf": path, "thinking": thinking, "part": part,
            "history": HISTORY, "steps": STEPS, "opening": opening,
            "top": TOP, "adapter": adapter}
    r = subprocess.run([sys.executable, "-c", _CHILD, json.dumps(spec)],
                       capture_output=True, text=True, cwd=REPO,
                       env={**os.environ, "TOP_LOGPROBS_K": str(TOP)})
    assert r.returncode == 0, r.stderr[-4000:]
    line = next(x for x in reversed(r.stdout.splitlines())
                if x.startswith("RESULT "))
    out = json.loads(line[len("RESULT "):])
    generated = out["generated"]
    skip = out["skip"]
    assert bytes(b for e in generated[:skip] for b in e["bytes"]).decode() \
        == opening
    assert len(generated) > skip + STEPS, generated
    if adapter is not None:
        # The row runs the adapter: its first token's logprobs moved off
        # the bare model's.
        (bare,) = out["plain"]
        was = dict(map(tuple, bare["top"])).get(generated[0]["id"])
        assert was is None or abs(was - generated[0]["logprob"]) > 10 * LOGP_TOL, \
            (bare, generated[0])
    return rid, out


def test_a_prefill_tokenizes_to_the_tokens_it_continues(run):
    """The claim of the prefill: the continued prompt is the generation
    prompt and the k generated tokens, as llama-server renders it."""
    rid, out = run
    ids = [e["id"] for e in out["generated"]]
    for step in out["prefilled"]:
        k = step["k"]
        assert step["prompt"] == out["prompt"] + ids[:k], (rid, k, step["text"])


def test_a_prefill_scores_the_next_token_as_decode_did(run):
    """The prefill's logprob of the next generated token, read from its top
    entries, so a near tie that picks another token still compares."""
    rid, out = run
    generated = out["generated"]
    over = []
    for step in out["prefilled"]:
        k = step["k"]
        want = generated[k]
        (nxt,) = step["entries"]
        top = dict(map(tuple, nxt["top"]))
        assert want["id"] in top, (rid, k)
        got = top[want["id"]]
        diff = abs(got - want["logprob"])
        bound = LOGP_TOL + LOGP_SPREAD * (1 - math.exp(want["logprob"]))
        print(f"{rid} k={k} prefilled {got:.6f} generated "
              f"{want['logprob']:.6f} diff {diff:.2e} bound {bound:.2e}")
        if diff > bound:
            over.append((k, diff, bound))
    assert not over, (rid, over)
