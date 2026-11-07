"""Serve-engine logprobs on real weights: a lone row through mlx-vlm's
batch engine, with a profile-style repetition penalty active, reports the
f32 log_softmax of a plain forward over the same tokens.

``integration``; skips unless ``KQUANT_TEST_GGUF_DIR`` holds a GGUF of one
of the arches below.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

import mlx.core as mx

pytest.importorskip("gmlx._vendor.mlx_vlm")

pytestmark = pytest.mark.integration

ARCHES = ["qwen3", "llama", "smollm3", "falcon-h1"]
N_TOKENS = 16
K = 20
# The logits are bf16, so a one-step rounding difference on a large logit
# moves a logprob by one bf16 step at that logit's size (0.125 between 16 and
# 32), and the normalizer moves with it. The GPU measured 0.063 at most on
# qwen3 and falcon-h1, and 0.133 on SmolLM2, whose logits reach 16. The CPU
# matches exactly.
LSE_TOL = 0.02


def _tol(logits) -> float:
    """One bf16 step at the row's largest logit, plus the normalizer's
    share."""
    top = float(np.abs(logits).max())
    return 2.0 ** (np.floor(np.log2(top)) - 7) + LSE_TOL
PROMPT = "The quick brown fox jumps over the lazy dog. The quick brown fox"


def _require(gguf_index, arch):
    paths = gguf_index.get(arch)
    if not paths:
        pytest.skip(f"no {arch!r} GGUF under KQUANT_TEST_GGUF_DIR")
    return paths[0]


@pytest.mark.parametrize("arch", ARCHES)
def test_a_lone_row_reports_the_plain_forwards_logprobs(gguf_index, arch,
                                                        monkeypatch):
    from gmlx import load_model
    from gmlx._vendor.mlx_lm.models.cache import make_prompt_cache
    from gmlx._vendor.mlx_vlm.sample_utils import make_logits_processors
    from gmlx._vendor.mlx_vlm.utils import StoppingCriteria
    from gmlx.models.vlm_text_only import TextOnlyModel
    from gmlx.serve.row_sampling import RowSettings

    ar = importlib.import_module("gmlx._vendor.mlx_vlm.generate.ar")
    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    raw, _config, tok = load_model(_require(gguf_index, arch), verbose=False)
    model = TextOnlyModel(raw)
    lm = model.language_model
    ids = list(tok.encode(PROMPT))

    proc = SimpleNamespace(tokenizer=SimpleNamespace(
        stopping_criteria=StoppingCriteria([], None)))
    # The engine's own stream was made at import, maybe before the test
    # device was set. Both runs use one device, so the bound is per backend.
    gen = ar.BatchGenerator(lm, proc, prefill_step_size=None,
                            max_tokens=N_TOKENS, logprobs_cap=K,
                            stream=mx.default_stream(mx.default_device()))
    (uid,) = gen.insert(
        [ids], N_TOKENS,
        logits_processors=[make_logits_processors(None, 1.05)],
        prompt_kwargs=[{"inputs_embeds": lm.input_embeds(mx.array([ids]))}],
        row_settings=[RowSettings(logprobs=True, top_k=K)])
    resps = []
    while gen.has_work:
        _p, rs = gen.next()
        resps += [r for r in rs if r.uid == uid]
    gen.close()
    assert len(resps) == N_TOKENS

    # The reference runs in the engine's order, the prompt then one token
    # per forward, with no processors.
    toks = [int(r.token) for r in resps]
    cache = make_prompt_cache(raw)
    rows, tols = [], []
    for part in [ids] + [[t] for t in toks[:-1]]:
        out = lm(mx.array([part]), cache=cache)
        last = getattr(out, "logits", out)[0, -1].astype(mx.float32)
        rows.append(np.array(last - mx.logsumexp(last)))
        tols.append(_tol(np.array(last)))
    for i, r in enumerate(resps):
        row, tol = rows[i], tols[i]
        assert abs(r.token_logprob - row[r.token]) <= tol, (arch, i)
        top_ids = [t for t, _ in r.top_logprobs]
        vals = np.array([v for _, v in r.top_logprobs])
        assert len(top_ids) == K
        np.testing.assert_allclose(vals, np.sort(row)[::-1][:K], atol=tol)
        np.testing.assert_allclose(vals, row[top_ids], atol=tol)
        assert (np.diff(vals) <= 0).all()
