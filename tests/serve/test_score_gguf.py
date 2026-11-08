"""Scoring on real weights, through the served routes. Each row serves one
GGUF in a child process and:

- generates a greedy reply on chat with ``return_token_ids`` and
  ``logprobs``, and checks that ``/tokenize`` gives the same prompt ids;
- scores the prompt and reply ids in one ``/v1/completions`` request, and
  checks each reply token's score against the logprob chat gave it;
- scores again and gets the same entries bit for bit, also after a chat
  request with the same prompt fills the prompt cache, and a chat request
  after the score still hits the cache;
- generates from an id prompt and from the same text, and gets the same
  greedy ids;
- with ``interleave``, scores a long prompt while a chat row decodes: the
  chat ids equal its lone run and the entries equal the idle score.

``integration``; finds each GGUF by name under ``KQUANT_TEST_GGUF_DIR``
and skips without it.
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
# A scored reply token is a prefill value, its generated logprob a decode
# value, and the two round differently in bf16. The bound is the decode
# bound of tests/serve/test_prefill_logp.py widened by MARGIN, set from the
# worst gap measured per row (the logs/g11-score-margin runs).
LOGP_TOL = 1e-3
LOGP_SPREAD = 0.25
MARGIN = 2.0
STEPS = 24
# The system message is long enough that the prompt cache stores blocks of
# the prompt.
SYSTEM = "Be brief. " + " ".join(
    f"Rule {i}: answer in plain words and keep to the question." for i in range(80))
HISTORY = [{"role": "system", "content": SYSTEM},
           {"role": "user", "content": "Name three rivers in Europe."}]
TEXT = "The capital of France is"

# (id, GGUF file name, environment, interleave, speculative). The MTP row
# serves its model with its MTP head and the prompt cache on, a
# checkpoint-tier hybrid.
ROWS = [
    ("smollm2", "SmolLM2-135M-Instruct-Q8_0.gguf", {}, True, False),
    ("smollm2-kv8", "SmolLM2-135M-Instruct-Q8_0.gguf", {"KV_BITS": "8"}, False, False),
    ("qwen35_9b", "Qwen3.5-9B-Q4_K_M.gguf", {}, False, False),
    ("qwen35_9b-kv8", "Qwen3.5-9B-Q4_K_M.gguf", {"KV_BITS": "8"}, False, False),
    ("qwen36_27b-mtp",
     "Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-Q6_K.gguf", {},
     False, True),
]

_CHILD = r"""
import importlib, json, sys, threading
import mlx.core as mx
spec = json.loads(sys.argv[1])
from fastapi.testclient import TestClient
import gmlx.serve.bridge_vlm as serving
from gmlx.config import build_config
from gmlx.gen import score_reader
from gmlx.serve.patches import install_server_patches
from gmlx.serve.residency import install_gguf_residency_pool

cfg = build_config({
    "server": {"family_defaults": False, "cache": {"enabled": True}},
    "models": {"m": {"path": spec["gguf"], "speculative": spec["speculative"]}}})
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

def chat(client, max_tokens, **kw):
    return ok(client.post("/v1/chat/completions", json={
        "model": "m", "messages": spec["history"], "temperature": 0,
        "max_tokens": max_tokens, "logprobs": True,
        "return_token_ids": True, "return_tokens_as_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False}, **kw}))

def score(client, ids):
    body = ok(client.post("/v1/completions", json={
        "model": "m", "prompt": ids, "echo": True, "logprobs": 1,
        "max_tokens": 0, "return_tokens_as_token_ids": True}))
    lp = body["choices"][0]["logprobs"]
    assert [tid(t) for t in lp["tokens"]] == ids
    return lp["token_logprobs"], lp["top_logprobs"]

out = {}
with TestClient(app, base_url="http://127.0.0.1") as client:
    first = chat(client, spec["steps"])
    P = first["prompt_token_ids"]
    R = first["choices"][0]["token_ids"]
    out["P"], out["R"] = P, R
    out["generated"] = [e["logprob"] for e in first["choices"][0]["logprobs"]["content"]]
    out["generated_ids"] = [tid(e["token"]) for e in first["choices"][0]["logprobs"]["content"]]
    tok = ok(client.post("/tokenize", json={
        "model": "m", "messages": spec["history"],
        "chat_template_kwargs": {"enable_thinking": False}}))
    out["tokenized"] = tok["tokens"]
    out["max_model_len"] = tok["max_model_len"]
    ok(client.post("/v1/cache/reset", json={}))
    out["cold"] = score(client, P + R)
    out["repeat"] = score(client, P + R)
    filled = chat(client, spec["steps"])
    out["filled_cached"] = filled["usage"]["prompt_tokens_details"]["cached_tokens"]
    out["warm"] = score(client, P + R)
    after = chat(client, 1)
    out["after_cached"] = after["usage"]["prompt_tokens_details"]["cached_tokens"]
    # Greedy from an id prompt and from its text.
    ids = ok(client.post("/tokenize", json={"model": "m", "prompt": spec["text"]}))["tokens"]
    gen = {}
    for name, prompt in (("text", spec["text"]), ("ids", ids)):
        body = ok(client.post("/v1/completions", json={
            "model": "m", "prompt": prompt, "max_tokens": 8, "temperature": 0,
            "logprobs": 0, "return_tokens_as_token_ids": True}))
        gen[name] = [tid(t) for t in body["choices"][0]["logprobs"]["tokens"]]
    out["greedy"] = gen
    rg = importlib.import_module("gmlx._vendor.mlx_vlm.server.runtime").runtime.response_generator
    reader = score_reader.reader_for(getattr(rg, "_rg", rg))
    out["path"] = {"split": reader.split is not None, "refusal": reader.refusal,
                   "dtype": str(reader.dtype), "width": reader.width,
                   "reduce": reader.reduce}
    if spec["interleave"]:
        long_ids = (P + R) * 2
        score_reader.FORWARD_TOKENS = 16
        # Count chat steps, and read the count at each scoring step.
        gen_mod = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
        steps = {"chat": 0, "seen": []}
        real_step = gen_mod.ResponseGenerator._step

        def counting_step(self, *a, **k):
            steps["chat"] += 1
            return real_step(self, *a, **k)

        gen_mod.ResponseGenerator._step = counting_step
        real_score = score_reader.ScoreReader.score

        def watched(self, *a, **k):
            run = real_score(self, *a, **k)
            try:
                while True:
                    next(run)
                    steps["seen"].append(steps["chat"])
                    yield
            except StopIteration as e:
                return e.value

        score_reader.ScoreReader.score = watched
        out["idle_long"] = score(client, long_ids)
        lone = chat(client, 48)
        out["lone"] = lone["choices"][0]["token_ids"]
        started = threading.Event()
        got = {}

        def run_chat():
            started.set()
            got["chat"] = chat(client, 48)["choices"][0]["token_ids"]

        t = threading.Thread(target=run_chat)
        t.start()
        started.wait()
        got["score"] = score(client, long_ids)
        t.join()
        out["busy_chat"] = got["chat"]
        out["busy_long"] = got["score"]
        out["seen"] = steps["seen"]
print("RESULT " + json.dumps(out))
"""


def _find(gguf_index, name):
    for paths in gguf_index.values():
        for p in paths:
            if Path(p).name == name:
                return p
    pytest.skip(f"no {name} under KQUANT_TEST_GGUF_DIR")


@pytest.fixture(scope="module", params=ROWS, ids=[r[0] for r in ROWS])
def run(request, gguf_index):
    rid, name, env, interleave, speculative = request.param
    spec = {"gguf": _find(gguf_index, name), "history": HISTORY,
            "steps": STEPS, "text": TEXT, "interleave": interleave,
            "speculative": speculative}
    r = subprocess.run([sys.executable, "-c", _CHILD, json.dumps(spec)],
                       capture_output=True, text=True, cwd=REPO,
                       env={**os.environ, **env})
    assert r.returncode == 0, r.stderr[-4000:]
    line = next(x for x in reversed(r.stdout.splitlines())
                if x.startswith("RESULT "))
    out = json.loads(line[len("RESULT "):])
    print(f"{rid}: path {out['path']}, P {len(out['P'])} R {len(out['R'])}")
    return rid, out


def test_tokenize_gives_the_prompt_chat_read(run):
    rid, out = run
    assert out["tokenized"] == out["P"], rid
    assert out["max_model_len"], rid
    assert out["generated_ids"] == out["R"], rid


def test_a_reply_scores_close_to_its_generated_logprobs(run):
    rid, out = run
    values, _top = out["cold"]
    n = len(out["P"])
    assert values[0] is None and None not in values[1:]
    over, worst = [], 0.0
    for i, want in enumerate(out["generated"]):
        got = values[n + i]
        diff = abs(got - want)
        worst = max(worst, diff)
        bound = MARGIN * (LOGP_TOL + LOGP_SPREAD * (1 - math.exp(want)))
        if diff > bound:
            over.append((i, got, want, bound))
    print(f"{rid}: worst reply gap {worst:.4g} over {len(out['generated'])} tokens")
    assert not over, (rid, over)


def test_a_score_repeats_bit_for_bit_and_reads_no_cache(run):
    rid, out = run
    assert out["repeat"] == out["cold"], rid
    assert out["warm"] == out["cold"], rid
    # The cold scores stored nothing for chat to hit, and a chat request
    # after the warm score still hits the prefix chat stored.
    print(f"{rid}: cached tokens {out['filled_cached']} then {out['after_cached']}")
    assert out["filled_cached"] == 0, rid
    assert out["after_cached"] > 0, rid


def test_an_id_prompt_generates_as_its_text(run):
    rid, out = run
    assert out["greedy"]["ids"] == out["greedy"]["text"], rid


def test_a_score_beside_a_chat_row_changes_neither(run):
    rid, out = run
    if "busy_long" not in out:
        pytest.skip("this row does not interleave")
    assert out["busy_chat"] == out["lone"], rid
    assert out["busy_long"] == out["idle_long"], rid
    # The last score ran while the chat row decoded: chat steps ran between
    # its forwards.
    busy = out["seen"][-len(out["seen"]) // 2:]
    print(f"{rid}: chat steps seen across the busy score: {busy[0]} to {busy[-1]}")
    assert busy[-1] > busy[0], rid
