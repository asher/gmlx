"""A systemone decision on a kept prefix, on MiniCPM-V-4.6 served with its
mmproj and its letter scores read from the full logits.

The first decision on a state reads its prefix cold and keeps it. An image
chat request then runs, which leaves an image row's rope positions and
deltas on the language model. The second decision on the same state reads
the kept prefix and only its tails. A second server reads the same
decisions on the model's default letter path, and both full-logit
decisions answer as its first one within TOL.

On this model, full-logit reads through the outer class answer the same
too, so this file checks the route and the kept prefix on real weights.
tests/serve/test_job_rope_qwen35.py checks the positions.

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
# A cold prefix and a kept one run the same forwards over the same ids, so
# the answers agree to rounding; reading at wrong positions moves them far
# more. Set from the measured run (logs/g11-systemone-prefix).
TOL = 0.02

BODY = {
    "model": "m",
    "state": {"ticket": "I was charged twice for my March invoice. "
                        "Please refund the duplicate charge."},
    "questions": {
        "urgent": {"type": "noul",
                   "instructions": "Does the customer need a reply within the hour?"},
        "team": {"type": "choice",
                 "instructions": "Which team should handle the ticket?",
                 "criteria": {"billing": "invoices, charges, refunds",
                              "infra": "outages, errors, latency",
                              "product": "features, UI bugs"}},
        "refund": {"type": "noul",
                   "instructions": "Does the customer ask for a refund?"},
    },
}

_CHILD = r"""
import base64, importlib, io, json, sys
import numpy as np
from PIL import Image
spec = json.loads(sys.argv[1])
from fastapi.testclient import TestClient
import gmlx.serve.bridge_vlm as serving
from gmlx.config import build_config
from gmlx.serve.patches import install_server_patches
from gmlx.serve.residency import install_gguf_residency_pool
from gmlx.systemone import ar_reader

# With force_logits, the letter scores come from the full logits.
real_check = ar_reader.LetterReader.check

def full_logits(self, *a, **k):
    real_check(self, *a, **k)
    self.path, self.rows, self.step = "logits", None, None
    return self.path

if spec["force_logits"]:
    ar_reader.LetterReader.check = full_logits

cfg = build_config({
    "server": {"family_defaults": False, "cache": {"enabled": True}},
    "models": {"m": {"path": spec["gguf"], "mmproj": spec["mmproj"],
                     "speculative": False}}})
serving.register_resolved_models(cfg)
serving.install_gguf_server_bridge()
install_gguf_residency_pool()
install_server_patches(cfg, reload_fn=lambda *a, **k: None)
app = importlib.import_module("gmlx._vendor.mlx_vlm.server.app").app

x = np.linspace(0, 255, 224, dtype=np.uint8)
pixels = np.stack([np.tile(x, (224, 1)), np.tile(x[:, None], (1, 224)),
                   np.full((224, 224), 96, np.uint8)], axis=-1)
buf = io.BytesIO()
Image.fromarray(pixels).save(buf, format="PNG")
url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def ok(r):
    assert r.status_code == 200, r.text
    return r.json()

out = {}
with TestClient(app, base_url="http://127.0.0.1") as c:
    out["cold"] = ok(c.post("/v1/systemone", json=spec["body"]))
    ok(c.post("/v1/chat/completions", json={
        "model": "m", "temperature": 0, "max_tokens": 16,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": "Describe this image."}]}]}))
    out["warm"] = ok(c.post("/v1/systemone", json=spec["body"]))
print("RESULT " + json.dumps(out))
"""


def _probabilities(answer: dict) -> dict:
    if answer["type"] == "noul":
        return {"yes": answer["noul"]}
    return dict(answer["probabilities"])


@pytest.fixture(scope="module")
def run(gguf_dir):
    hits = sorted(Path(gguf_dir).rglob(GGUF))
    if not hits:
        pytest.skip(f"no {GGUF} under KQUANT_TEST_GGUF_DIR")
    mmproj = hits[0].with_name(MMPROJ)
    if not mmproj.exists():
        pytest.skip(f"no {MMPROJ} beside {GGUF}")
    out = {}
    for force in (False, True):
        spec = {"gguf": str(hits[0]), "mmproj": str(mmproj), "body": BODY,
                "force_logits": force}
        r = subprocess.run([sys.executable, "-c", _CHILD, json.dumps(spec)],
                           capture_output=True, text=True, cwd=REPO,
                           env=dict(os.environ))
        assert r.returncode == 0, r.stderr[-4000:]
        line = next(x for x in reversed(r.stdout.splitlines())
                    if x.startswith("RESULT "))
        out["logits" if force else "default"] = json.loads(line[len("RESULT "):])
    return out


def _gap(x: dict, y: dict) -> float:
    worst = 0.0
    for q, ax in x["answers"].items():
        a, b = _probabilities(ax), _probabilities(y["answers"][q])
        assert a.keys() == b.keys(), q
        worst = max([worst] + [abs(a[k] - b[k]) for k in a])
    return worst


def test_the_second_decision_reads_the_kept_prefix_on_full_logits(run):
    for name in ("cold", "warm"):
        print(name, json.dumps(run["logits"][name]["diagnostics"])[:400])
    for name in ("cold", "warm"):
        assert run["logits"][name]["diagnostics"]["path"] == "logits"
    assert run["logits"]["warm"]["diagnostics"]["prefix"]["reused"]
    print("default path:", run["default"]["cold"]["diagnostics"]["path"])


def test_full_logit_decisions_answer_as_the_default_path(run):
    """The default path reads the prefix and the tails through the trunk,
    which numbers each chunk from its cache offset. The full-logit reads
    match it, cold and on the kept prefix after an image row ran."""
    ref = run["default"]["cold"]
    gaps = {name: _gap(ref, run["logits"][name]) for name in ("cold", "warm")}
    gaps["default-warm"] = _gap(ref, run["default"]["warm"])
    print("worst answer gaps against the default path's cold read:", gaps)
    assert max(gaps.values()) <= TOL
