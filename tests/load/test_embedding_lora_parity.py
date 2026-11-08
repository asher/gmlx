"""A token embedding adapter on real weights against llama-server: the
stories15M MoE model under the checked-in adapter (a token_embd pair and
three attention pairs) gives llama-server's greedy ids at strengths 0, 1
and 1.5, and moves each top-k logprob by the amount llama-server's run
moved it, within the fixture's tolerance. Without the embedding pair the
shift misses that tolerance.

``integration``; finds stories15M_MOE-F16.gguf by name and size under
``KQUANT_TEST_GGUF_DIR`` and skips without it. Runs in float32 with exact
products: on a device whose float32 products run at TF32 the test reruns
itself in a child with ``MLX_ENABLE_TF32=0``.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

# A device import, so the GPU gate takes this file.
import mlx.core as mx  # noqa: E402

import llama_fixture  # noqa: E402

pytestmark = pytest.mark.integration

FIXTURE = llama_fixture.load("lora/stories15m_moe_embd.json")
ADAPTER = llama_fixture.FIXTURES / "lora" / "stories15m_moe_embd.gguf"
GGUF = "stories15M_MOE-F16.gguf"


def _exact() -> bool:
    from gmlx.image.float32 import f32_products_exact

    return f32_products_exact()


def _rerun_without_tf32(request) -> None:
    """Rerun this test in a child process with TF32 off, where it gates."""
    env = dict(os.environ, MLX_ENABLE_TF32="0")
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         request.node.nodeid], cwd=request.config.rootpath, env=env,
        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]


@pytest.fixture(scope="module")
def model(gguf_index):
    from gmlx import lora_rows
    from gmlx.load.adapter import apply_gguf_adapter
    from gmlx.load.loader import load_model

    path = llama_fixture.find_file(FIXTURE, GGUF, gguf_index.get("mixtral", []))
    llama_fixture.check_checked_in(FIXTURE, ADAPTER)
    mp = pytest.MonkeyPatch()
    mp.setenv("GMLX_ACTIVATION_DTYPE", "float32")
    try:
        m, config, _tok = load_model(path)
        apply_gguf_adapter(getattr(m, "language_model", m), config, str(ADAPTER),
                           base_arch="llama")
        lora_rows.configure("static", 1)
        yield m
    finally:
        mp.undo()
        lora_rows.configure("static", 1)


def _logprobs(model, ids, strength):
    from gmlx import lora_rows

    lora_rows.set_rows([strength])
    try:
        logits = model(mx.array([ids]))
        mx.eval(logits)
    finally:
        lora_rows.clear_rows()
    x = np.array(logits[0].astype(mx.float32), dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def _worst_shift(model, row) -> tuple[float, list]:
    """The largest gap between gmlx's and llama-server's shift of a top-k
    logprob, and gmlx's greedy ids, over the row's positions."""
    p = len(row["prompt_ids"])
    seq = row["prompt_ids"] + row["ids"]
    n = len(row["ids"])
    lp_s = _logprobs(model, seq, row["strength"])[p - 1:p - 1 + n]
    lp_0 = _logprobs(model, seq, 0.0)[p - 1:p - 1 + n]
    worst = 0.0
    for k in range(n):
        top, base = dict(map(tuple, row["top"][k])), dict(map(tuple, row["base_top"][k]))
        for i in set(top) & set(base):
            worst = max(worst, abs((lp_s[k][i] - lp_0[k][i]) - (top[i] - base[i])))
    return worst, [int(np.argmax(r)) for r in lp_s]


def test_the_adapter_moves_logprobs_as_llama_server_does(model, request):
    if not _exact():
        _rerun_without_tf32(request)
        return
    tol = FIXTURE["tolerances"]["shift"]
    for row in FIXTURE["rows"]:
        worst, ids = _worst_shift(model, row)
        assert ids == row["ids"], row["strength"]
        assert worst <= tol, (row["strength"], worst, tol)


def test_without_the_embedding_pair_the_shift_misses(model, request):
    if not _exact():
        _rerun_without_tf32(request)
        return
    from gmlx.load.modules import LoRAKQuantEmbedding

    emb = model.model.embed_tokens
    assert isinstance(emb, LoRAKQuantEmbedding)
    row = next(r for r in FIXTURE["rows"] if r["strength"] == 1.5)
    scale, emb.scale = emb.scale, 0.0
    try:
        worst, _ids = _worst_shift(model, row)
    finally:
        emb.scale = scale
    assert worst > 2 * FIXTURE["tolerances"]["shift"]
