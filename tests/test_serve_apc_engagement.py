#!/usr/bin/env python3
"""Serve-path APC engagement: the manager must actually be consulted.

mlx-vlm 0.6.4 silently disengaged APC for served mlx-lm-arch models (found
live on gemma-4: ``apc_enabled: true``, every counter zero). ``ar.py``'s
``BatchGenerator`` drops the ``apc_manager`` whenever ``model_apc_mode(model)``
resolves ``None``, and that probe isinstance-gates the model's own
``make_cache()`` entries against mlx-vlm's (since 0.6.4, vendored) cache
classes -- mlx_lm-origin caches fail the gate. ``model_apc_mode``'s source was
byte-identical across the break, so seam fingerprints can't catch this class
of behavioral-composition regression; the unit half lives in
``test_apc_pooling::test_apc_engages_for_mlx_lm_origin_model``, and this file
attests the full composition: a real GGUF through ``load_serveable_model``
(which applies the runtime-origin ``make_cache`` rebind), driven through the
stock ``BatchGenerator`` with a live ``APCManager``, must move the counters --
stores after the first request, prefix hits + matched tokens on a second
request sharing the prefix.

``integration`` + ``slow``; needs a small dense GGUF (qwen3-0.6b) under
``KQUANT_TEST_GGUF_DIR``.
"""
from __future__ import annotations

import pytest

pytest.importorskip("mlx_vlm")

import mlx.core as mx  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.slow]

GREEDY = lambda x: mx.argmax(x, axis=-1)  # noqa: E731
N_DECODE = 8
PREFIX_TOKENS = 200  # >> one APC block (16), so the shared prefix stores/hits


@pytest.fixture(scope="module")
def serveable(gguf_index):
    """(model, tokenizer) for a small dense arch via the real serve loader."""
    from gmlx import server_bridge_vlm as serving

    paths = gguf_index.get("qwen3")
    if not paths:
        pytest.skip(f"no 'qwen3' GGUF under KQUANT_TEST_GGUF_DIR "
                    f"(have: {sorted(gguf_index)})")
    path = paths[0]
    model, processor, _config = serving.load_serveable_model(path)
    return model, processor


def _build_ids(tokenizer, suffix):
    parts, i = [], 0
    while len(tokenizer.encode("".join(parts))) < PREFIX_TOKENS:
        parts.append(f"Section {i}. The clockmaker adjusted the escapement. ")
        i += 1
    prefix_ids = tokenizer.encode("".join(parts))[:PREFIX_TOKENS]
    return list(prefix_ids) + list(tokenizer.encode(suffix))


def _drive(model, processor, ids, manager):
    """One request through the stock engine, exactly as serve builds it."""
    import importlib

    ar = importlib.import_module("mlx_vlm.generate.ar")
    input_ids = mx.array([ids], dtype=mx.int32)
    emb = model.get_input_embeddings(input_ids=input_ids)
    gen = ar.BatchGenerator(
        model, processor, sampler=GREEDY, max_tokens=N_DECODE,
        apc_manager=manager,
    )
    assert gen.apc_manager is manager, (
        "BatchGenerator dropped the apc_manager at construction: "
        f"model_apc_mode resolved {gen.apc_mode!r} -- the 0.6.4 "
        "cache-origin disengagement (make_cache rebind missing?)")
    uids = gen.insert([ids], [N_DECODE],
                      prompt_kwargs=[{"inputs_embeds": emb.inputs_embeds}])
    toks = []
    while gen.has_work:
        _prompt_responses, gen_responses = gen.next()
        for r in gen_responses:
            if r.uid in uids and r.finish_reason is None:
                toks.append(int(r.token))
    gen.close()
    return toks


def test_serve_apc_stores_then_hits(serveable):
    from mlx_vlm.apc import APCManager

    model, processor = serveable
    tokenizer = processor.tokenizer
    manager = APCManager(num_blocks=512, block_size=16)

    toks = _drive(model, processor,
                  _build_ids(tokenizer, " It began to rain."), manager)
    assert toks, "no tokens generated on the first request"
    s = manager.stats
    assert s.stores + s.exact_stores > 0, (
        "first request retired without storing anything into the APC "
        f"(stats: {s}) -- harvest never ran")

    toks = _drive(model, processor,
                  _build_ids(tokenizer, " The sun came out."), manager)
    assert toks, "no tokens generated on the second request"
    s = manager.stats
    assert s.hits + s.exact_hits > 0 and s.matched_tokens > 0, (
        "second request shared a stored prefix but the APC served nothing "
        f"(stats: {s}) -- lookup path disengaged")
