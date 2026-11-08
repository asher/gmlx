"""The score reader on real weights. At the default width a short prompt
takes one forward, and its entries equal a plain forward's f32
log_softmax bit for bit. At a width forced small the prompt takes several
forwards, and its entries equal the same model run chunked at those widths
bit for bit. Both runs pass the ids-against-embeddings probe and take the
full path.

``integration``; skips unless ``KQUANT_TEST_GGUF_DIR`` holds a GGUF of one
of the arches below.
"""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

pytest.importorskip("gmlx._vendor.mlx_vlm")

pytestmark = pytest.mark.integration

ARCHES = ["qwen3", "llama", "smollm3", "falcon-h1", "gemma4", "qwen35"]
K = 5
# The forward width of the chunked run, small so the prompt takes several.
WIDTH = 8
PROMPT = ("The quick brown fox jumps over the lazy dog. A journey of a "
          "thousand miles begins with a single step. To be or not to be, "
          "that is the question. All that glitters is not gold.")


def _require(gguf_index, arch):
    paths = gguf_index.get(arch)
    if not paths:
        pytest.skip(f"no {arch!r} GGUF under KQUANT_TEST_GGUF_DIR")
    return paths[0]


def _drain(steps):
    try:
        while True:
            next(steps)
    except StopIteration as e:
        return e.value


def _log_softmax(lm, chunks):
    from gmlx.cache.kv_cache import make_prompt_cache

    cache = make_prompt_cache(lm)
    rows = []
    for chunk in chunks:
        out = lm(mx.array([chunk]), cache=cache)
        rows.append(getattr(out, "logits", out)[0])
    x = mx.concatenate(rows, axis=0).astype(mx.float32)
    return np.array(x - mx.logsumexp(x, axis=-1, keepdims=True))


def _check(entries, ids, ref):
    assert len(entries) == len(ids) - 1
    for i, (value, top_ids, top_vals) in enumerate(entries):
        assert value == ref[i, ids[i + 1]], i
        assert top_ids is not None and top_vals is not None
        assert list(top_vals) == [float(v) for v in ref[i, top_ids]], i


@pytest.mark.parametrize("arch", ARCHES)
def test_scores_equal_the_models_own_forward(gguf_index, arch, monkeypatch):
    from gmlx.gen import score_reader as sr
    from gmlx.serve.bridge_vlm import load_serveable_model

    path = _require(gguf_index, arch)
    # The model as the server loads it.
    model, processor, _config = load_serveable_model(path)
    lm = model.language_model
    ids = list(processor.tokenizer.encode(PROMPT))
    assert len(ids) > 4 * WIDTH

    reader = sr.ScoreReader(model, streamed=False)
    _drain(reader.build())
    print(f"{arch} {path}: refusal={reader.refusal} dtype={reader.dtype} "
          f"width={reader.width} reduce={reader.reduce}")
    assert reader.refusal is None, (arch, reader.refusal)
    assert reader.split is None

    entries = _drain(reader.score(ids, K))
    assert reader.forwards == 1
    _check(entries, ids, _log_softmax(lm, [ids[:-1]]))
    assert _drain(reader.score(ids, K)) == entries

    monkeypatch.setattr(sr, "FORWARD_TOKENS", WIDTH)
    plan = reader.plan(len(ids) - 1)
    assert len(plan) >= 4
    chunked = _drain(reader.score(ids, K))
    _check(chunked, ids, _log_softmax(lm, [ids[d:d + w] for d, w in plan]))


def _tol(logits) -> float:
    """One bf16 step at the row's largest logit, plus the normalizer's
    share, as tests/gen/test_serve_logprobs_gguf.py bounds it."""
    top = float(np.abs(logits).max())
    return 2.0 ** (np.floor(np.log2(top)) - 7) + 0.02


def test_a_streamed_moe_model_scores_on_the_path_its_check_picks(gguf_index):
    """With its experts streamed, the reader takes the trunk and head split
    only when the split reproduces the model's own logits; either way the
    entries are within one bf16 step of a plain forward."""
    from gmlx.cache.kv_cache import make_prompt_cache
    from gmlx.gen import score_reader as sr
    from gmlx.serve.bridge_vlm import load_serveable_model

    path = _require(gguf_index, "qwen35moe")
    model, processor, _config = load_serveable_model(
        path, stream="experts", feeder_prefill=False, feeder_decode=False)
    ids = list(processor.tokenizer.encode(PROMPT))
    reader = sr.ScoreReader(model)
    assert reader.streamed
    _drain(reader.build())
    print(f"qwen35moe {path}: split={reader.split is not None} "
          f"refusal={reader.refusal} width={reader.width} reduce={reader.reduce}")
    assert reader.refusal is None
    entries = _drain(reader.score(ids, K))

    lm = model.language_model
    out = lm(mx.array([ids[:-1]]), cache=make_prompt_cache(lm))
    z = getattr(out, "logits", out)[0].astype(mx.float32)
    lp, raw = np.array(z - mx.logsumexp(z, axis=-1, keepdims=True)), np.array(z)
    worst = 0.0
    for i, (value, _top_ids, _top_vals) in enumerate(entries):
        diff = abs(value - lp[i, ids[i + 1]])
        worst = max(worst, diff)
        assert diff <= _tol(raw[i]), i
    print(f"qwen35moe: worst gap to a plain forward {worst:.4g}")
