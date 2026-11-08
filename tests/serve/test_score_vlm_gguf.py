"""Scoring beside an image chat row on a model served with its mmproj.

MiniCPM-V-4.6 (arch ``qwen35``) keeps rope positions and deltas for its
image rows on the language model. A scoring job runs between that row's
decode steps, on a text prompt of several forwards. Its entries equal the
model run chunked at the same widths from a text row's rope state, bit for
bit, and equal its idle run; the image row's ids equal its lone run.

``integration``; finds the GGUF and its mmproj by name under
``KQUANT_TEST_GGUF_DIR`` and skips without them.
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

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
GGUF = "MiniCPM-V-4_6-Q4_K_M.gguf"
MMPROJ = "mmproj-model-f16.gguf"
# The forward width, small so the text prompt takes several forwards.
WIDTH = 16
TEXT = ("The quick brown fox jumps over the lazy dog. A journey of a "
        "thousand miles begins with a single step. To be or not to be, "
        "that is the question. All that glitters is not gold. ") * 2

_CHILD = r"""
import base64, importlib, io, json, sys, threading
import numpy as np
import mlx.core as mx
from PIL import Image
spec = json.loads(sys.argv[1])
from fastapi.testclient import TestClient
import gmlx.serve.bridge_vlm as serving
from gmlx.cache.kv_cache import make_prompt_cache
from gmlx.config import build_config
from gmlx.gen import score_reader
from gmlx.serve.engine_jobs import seed_text_rope
from gmlx.serve.patches import install_server_patches
from gmlx.serve.residency import install_gguf_residency_pool

cfg = build_config({
    "server": {"family_defaults": False, "cache": {"enabled": False}},
    "models": {"m": {"path": spec["gguf"], "mmproj": spec["mmproj"],
                     "speculative": False}}})
serving.register_resolved_models(cfg)
serving.install_gguf_server_bridge()
install_gguf_residency_pool()
install_server_patches(cfg, reload_fn=lambda *a, **k: None)
app = importlib.import_module("gmlx._vendor.mlx_vlm.server.app").app
runtime = importlib.import_module("gmlx._vendor.mlx_vlm.server.runtime").runtime
score_reader.FORWARD_TOKENS = spec["width"]

x = np.linspace(0, 255, 224, dtype=np.uint8)
pixels = np.stack([np.tile(x, (224, 1)), np.tile(x[:, None], (1, 224)),
                   np.full((224, 224), 96, np.uint8)], axis=-1)
buf = io.BytesIO()
Image.fromarray(pixels).save(buf, format="PNG")
url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def ok(r):
    assert r.status_code == 200, r.text
    return r.json()

def image_chat(client):
    body = ok(client.post("/v1/chat/completions", json={
        "model": "m", "temperature": 0, "max_tokens": 48,
        "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": "Describe this image in detail."}]}]}))
    return body["choices"][0]["token_ids"]

def score(client, ids):
    body = ok(client.post("/v1/completions", json={
        "model": "m", "prompt": ids, "echo": True, "logprobs": 3,
        "max_tokens": 0, "return_tokens_as_token_ids": True}))
    lp = body["choices"][0]["logprobs"]
    return lp["token_logprobs"], lp["top_logprobs"]

out = {}
with TestClient(app, base_url="http://127.0.0.1") as client:
    ids = ok(client.post("/tokenize", json={"model": "m", "prompt": spec["text"]}))["tokens"]
    out["n"] = len(ids)
    out["lone"] = image_chat(client)
    out["idle"] = score(client, ids)
    steps = {"chat": 0, "seen": []}
    gen_mod = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
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
    started = threading.Event()
    got = {}

    def run_chat():
        started.set()
        got["chat"] = image_chat(client)

    t = threading.Thread(target=run_chat)
    t.start()
    started.wait()
    got["score"] = score(client, ids)
    t.join()
    out["busy_chat"], out["busy"], out["seen"] = got["chat"], got["score"], steps["seen"]

    # The reference: the language model run chunked at the reader's widths,
    # from a text row's rope state.
    rg = runtime.response_generator
    rg = getattr(rg, "_rg", rg)
    reader = score_reader.reader_for(rg)
    out["path"] = {"split": reader.split is not None, "refusal": reader.refusal,
                   "width": reader.width}
    model = rg.model
    lm = model.language_model
    seed_text_rope(model)
    cache = make_prompt_cache(lm)
    rows = []
    for d, w in reader.plan(len(ids) - 1):
        o = lm(mx.array([ids[d:d + w]]), cache=cache)
        rows.append(getattr(o, "logits", o)[0])
    z = mx.concatenate(rows, axis=0).astype(mx.float32)
    lp = z - mx.logsumexp(z, axis=-1, keepdims=True)
    out["reference"] = [float(v) for v in
                        mx.take_along_axis(lp, mx.array(ids[1:])[:, None], axis=-1)[:, 0].tolist()]
    out["forwards"] = len(reader.plan(len(ids) - 1))
print("RESULT " + json.dumps(out))
"""


def _find(gguf_dir, name):
    hits = sorted(Path(gguf_dir).rglob(name))
    if not hits:
        pytest.skip(f"no {name} under KQUANT_TEST_GGUF_DIR")
    return str(hits[0])


@pytest.fixture(scope="module")
def run(gguf_dir):
    path = _find(gguf_dir, GGUF)
    mmproj = str(Path(path).with_name(MMPROJ))
    if not Path(mmproj).exists():
        pytest.skip(f"no {MMPROJ} beside {GGUF}")
    spec = {"gguf": path, "mmproj": mmproj, "text": TEXT, "width": WIDTH}
    r = subprocess.run([sys.executable, "-c", _CHILD, json.dumps(spec)],
                       capture_output=True, text=True, cwd=REPO,
                       env=dict(os.environ))
    assert r.returncode == 0, r.stderr[-4000:]
    line = next(x for x in reversed(r.stdout.splitlines())
                if x.startswith("RESULT "))
    out = json.loads(line[len("RESULT "):])
    print(f"path {out['path']}, {out['n']} ids in {out['forwards']} forwards")
    return out


def test_a_score_beside_an_image_row_reads_text_positions(run):
    # The ids-against-embeddings probe passed: the job reads the text as
    # chat reads it.
    assert run["path"]["refusal"] is None
    busy_values, _ = run["busy"]
    idle_values, _ = run["idle"]
    assert run["forwards"] >= 3
    assert run["busy"] == run["idle"]
    assert busy_values[0] is None
    assert busy_values[1:] == run["reference"]
    assert idle_values[1:] == run["reference"]
    busy = run["seen"][-run["forwards"]:]
    print(f"image chat steps seen across the busy score: {busy[0]} to {busy[-1]}")
    assert busy[-1] > busy[0]


def test_the_image_row_decodes_as_alone(run):
    assert run["busy_chat"] == run["lone"]
