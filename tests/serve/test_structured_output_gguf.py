"""``response_format`` on real weights through ``gmlx serve``: a JSON schema
reply parses and holds only the schema's key, with thinking off, on its
template default and on with a thinking budget, and with family defaults
on.

``integration``; finds each GGUF under ``KQUANT_TEST_GGUF_DIR`` by name
and skips without it. Each model serves in a child process. Run it alone
on an idle machine: it loads Gemma 4 12B, Qwen3.5-9B and gpt-oss-20b.
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
FIXTURE = llama_fixture.load("llama_continuation.json")
PROMPT = "What is 3 + 4? Reply as JSON."
SCHEMA = {"type": "object", "properties": {"answer": {"type": "integer"}},
          "required": ["answer"], "additionalProperties": False}

_OFF = {"chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 64}
_DEFAULT = {"max_tokens": 256}
_BUDGET = {"chat_template_kwargs": {"enable_thinking": True},
           "thinking_budget": 64, "max_tokens": 256}
_TOP_OFF = {"enable_thinking": False, "max_tokens": 64}

# Request id -> (body extras, streamed, has a thinking budget).
REQUESTS = {
    "off": (_OFF, False, False),
    "default": (_DEFAULT, False, False),
    "default-stream": (_DEFAULT, True, False),
    "budget": (_BUDGET, False, True),
    "top-off": (_TOP_OFF, False, False),
    "default-64": ({"max_tokens": 64}, False, False),
    "default-budget": ({"thinking_budget": 128, "max_tokens": 512}, False,
                       True),
}
# Model -> (architecture, GGUF, the fixture names it, its requests).
MODELS = {
    "gemma4_12b": ("gemma4", "gemma-4-12B-it-Q6_K.gguf", True,
                   ("off", "default", "default-stream", "budget")),
    "qwen35_9b": ("qwen35", "Qwen3.5-9B-Q4_K_M.gguf", True,
                  ("off", "default", "default-stream", "budget")),
    "smollm2": ("llama", "SmolLM2-135M-Instruct-Q8_0.gguf", True,
                ("default-64", "top-off")),
    "qwen3_0.6b": ("qwen3", "Qwen3-0.6B-Q8_0.gguf", False,
                   ("default-budget", "top-off")),
    "gptoss_20b": ("gpt-oss", "gpt-oss-20b-MXFP4.gguf", False, ("default",)),
}

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

cfg = build_config({
    "server": {"cache": {"enabled": False}},
    "models": {"m": {"path": spec["gguf"], "speculative": False}}})
serving.register_resolved_models(cfg)
serving.install_gguf_server_bridge()
install_gguf_residency_pool()
install_server_patches(cfg, reload_fn=lambda *a, **k: None)
app = importlib.import_module("gmlx._vendor.mlx_vlm.server.app").app
fmt = {"type": "json_schema",
       "json_schema": {"name": "answer", "schema": spec["schema"]}}
out = {}
with TestClient(app, base_url="http://127.0.0.1") as client:
    for rid, (extra, stream) in spec["requests"].items():
        body = {"model": "m", "temperature": 0, "response_format": fmt,
                "messages": [{"role": "user", "content": spec["prompt"]}],
                **extra}
        if not stream:
            r = client.post("/v1/chat/completions", json=body)
            got = {"status": r.status_code}
            if r.status_code == 200:
                choice = r.json()["choices"][0]
                got.update(finish=choice["finish_reason"],
                           content=choice["message"].get("content"),
                           reasoning=choice["message"].get("reasoning_content")
                           or choice["message"].get("reasoning"))
            else:
                got["error"] = r.text
            out[rid] = got
            continue
        content, reasoning, finish = [], [], None
        with client.stream("POST", "/v1/chat/completions",
                           json={**body, "stream": True}) as r:
            status = r.status_code
            for line in r.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[len("data: "):])
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    content.append(delta.get("content") or "")
                    reasoning.append(delta.get("reasoning_content")
                                     or delta.get("reasoning") or "")
                    finish = choice.get("finish_reason") or finish
        out[rid] = {"status": status, "finish": finish,
                    "content": "".join(content),
                    "reasoning": "".join(reasoning)}
print("RESULT " + json.dumps(out))
"""


def _path(gguf_index, arch, name, in_fixture):
    if in_fixture:
        return llama_fixture.find_file(FIXTURE, name,
                                       gguf_index.get(arch, []))
    path = next((p for p in gguf_index.get(arch, [])
                 if Path(p).name == name), None)
    if path is None:
        pytest.skip(f"{name} is not under KQUANT_TEST_GGUF_DIR")
    return path


@pytest.fixture(scope="module", params=list(MODELS))
def served(request, gguf_index):
    """One model's replies, served in its child."""
    arch, name, in_fixture, rids = MODELS[request.param]
    spec = {"gguf": _path(gguf_index, arch, name, in_fixture),
            "schema": SCHEMA, "prompt": PROMPT,
            "requests": {rid: REQUESTS[rid][:2] for rid in rids}}
    r = subprocess.run([sys.executable, "-c", _CHILD, json.dumps(spec)],
                       capture_output=True, text=True, cwd=REPO,
                       env=dict(os.environ))
    assert r.returncode == 0, r.stderr[-4000:]
    line = next(x for x in reversed(r.stdout.splitlines())
                if x.startswith("RESULT "))
    return request.param, json.loads(line[len("RESULT "):])


def test_each_reply_is_the_schema(served):
    model, replies = served
    bad = {}
    for rid, got in replies.items():
        try:
            assert got["status"] == 200, got
            assert got["finish"] == "stop", got
            value = json.loads(got["content"])
            assert list(value) == ["answer"], got
            assert type(value["answer"]) is int, got
            if REQUESTS[rid][2]:
                assert (got["reasoning"] or "").strip(), got
        except (AssertionError, TypeError, ValueError) as e:
            bad[rid] = f"{type(e).__name__}: {e}"
    assert not bad, (model, bad)
