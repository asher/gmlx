#!/usr/bin/env python3
"""Serve-path APC engagement, per cache-shape family: the tier that claims
a family must actually move ITS OWN counters.

History: mlx-vlm 0.6.4 silently disengaged APC for served mlx-lm-arch
models (found live on gemma-4), and the 2026-08 audit found the ckpt tier
had never engaged on any hybrid/SWA arch -- both invisible because the
only engagement test ran one dense model and asserted counter
disjunctions. This gate runs one model per family through the real serve
composition (``load_serveable_model`` + the gmlx engine installs + stock
``BatchGenerator``) and asserts tier-specific counters: block families
move ``stores``/``hits``, exact families ``exact_stores``/``exact_hits``,
ckpt families the gmlx ``ckpt_stores``/``ckpt_hits``. No disjunctions --
a family asserting the wrong tier's counters is exactly how this bug
class hides.

``integration`` + ``slow``; needs GGUFs under ``KQUANT_TEST_GGUF_DIR``
(families with no GGUF skip loudly). Multi-GB rows (gpt-oss 12 GB,
qwen3.6-27B 15 GB) are opt-in: set ``GMLX_TEST_BIG_GGUFS=1`` (part of the
pre-release checklist; CI cannot run any of this file).
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("mlx_vlm")

import mlx.core as mx  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.slow]

GREEDY = lambda x: mx.argmax(x, axis=-1)  # noqa: E731
N_DECODE = 8

# (family id, general.architecture key, tier, prompt tokens, big?)
# Arch keys verified against the GGUF headers on disk (conftest's index
# derives them from general.architecture, not directory names). ckpt rows
# use identical-resend reuse (the N-1 replay record; prompts must clear
# GMLX_APC_CKPT_REPLAY_MIN=1024), block/exact rows use shared-prefix
# reuse (their tiers serve partial prefixes).
FAMILIES = [
    ("dense-block", "qwen3", "block", 200, False),
    ("swa-moe-ckpt", "gpt-oss", "ckpt", 1200, True),
    ("gdn-ckpt", "qwen35", "ckpt", 1200, True),
    ("cachelist-exact", "falcon-h1", "exact", 200, False),
]


@pytest.fixture(scope="module", params=FAMILIES, ids=[f[0] for f in FAMILIES])
def family(request, gguf_index):
    fam, arch, tier, prompt_tokens, big = request.param
    if big and os.environ.get("GMLX_TEST_BIG_GGUFS") != "1":
        pytest.skip(f"{fam}: multi-GB GGUF row; set GMLX_TEST_BIG_GGUFS=1 "
                    "(pre-release checklist)")
    paths = gguf_index.get(arch)
    if not paths:
        pytest.skip(f"no {arch!r} GGUF under KQUANT_TEST_GGUF_DIR "
                    f"(have: {sorted(gguf_index)})")
    import gmlx.serve.bridge_vlm as serving
    import gmlx.spec.engine as spec_engine

    spec_engine.install_full_prompt_mtp_prefill()   # the serve installs
    model, processor, _config = serving.load_serveable_model(paths[0])
    return fam, tier, prompt_tokens, model, processor


def _build_ids(tokenizer, n_tokens, suffix):
    parts, i = [], 0
    while len(tokenizer.encode("".join(parts))) < n_tokens:
        parts.append(f"Section {i}. The clockmaker adjusted the escapement. ")
        i += 1
    prefix_ids = tokenizer.encode("".join(parts))[:n_tokens]
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


def test_family_tier_engages(family):
    from mlx_vlm.apc import model_apc_mode

    from gmlx.cache.apc_manager import GmlxAPCManager
    from gmlx.cache.snapshot import ckpt_supported

    fam, tier, prompt_tokens, model, processor = family
    tokenizer = processor.tokenizer
    lm = model.language_model if hasattr(model, "language_model") else model

    # Routing: the family must land on the tier this row claims.
    mode = model_apc_mode(lm)
    assert mode == {"block": "block"}.get(tier, "exact"), (
        f"{fam}: model_apc_mode resolved {mode!r}")
    assert ckpt_supported(lm.make_cache()) == (tier == "ckpt"), (
        f"{fam}: ckpt eligibility flipped")

    manager = GmlxAPCManager(num_blocks=512, block_size=16)
    ids = _build_ids(tokenizer, prompt_tokens, " It began to rain.")
    toks = _drive(model, processor, ids, manager)
    assert toks, f"{fam}: no tokens generated on the first request"
    s1 = manager.stats_snapshot()

    if tier == "ckpt":
        assert s1["ckpt_stores"] > 0, (
            f"{fam}: ckpt tier armed but stored nothing (stats: {s1})")
        ids2 = ids                          # identical resend: N-1 replay
    else:
        key = "stores" if tier == "block" else "exact_stores"
        assert s1[key] > 0, (
            f"{fam}: first request stored nothing via {key} (stats: {s1})")
        ids2 = _build_ids(tokenizer, prompt_tokens, " The sun came out.")

    toks = _drive(model, processor, ids2, manager)
    assert toks, f"{fam}: no tokens generated on the second request"
    s2 = manager.stats_snapshot()
    hit_key = {"block": "lookups_hit", "exact": "exact_hits",
               "ckpt": "ckpt_hits"}[tier]
    assert s2[hit_key] > 0, (
        f"{fam}: second request shared a stored prefix but {hit_key} "
        f"never moved (stats: {s2})")
    assert s2["matched_tokens"] > 0, (
        f"{fam}: {hit_key} moved but no tokens were served from cache "
        f"(stats: {s2})")
    if tier == "ckpt":
        assert s2["ckpt_matched_tokens"] >= len(ids) - 1, (
            f"{fam}: identical resend adopted "
            f"{s2['ckpt_matched_tokens']} < N-1={len(ids) - 1}")
