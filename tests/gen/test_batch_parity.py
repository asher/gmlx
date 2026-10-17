#!/usr/bin/env python3
"""Batched-decode correctness: continuous batching must be numerically faithful.

Batched decode here runs mlx-lm's ``BatchGenerator`` over a ``BatchKVCache``,
driving the same K-quant leaf modules (``KQuantLinear`` /
``KQuantSwitchLinear`` ``gather_qmm`` / absorbed-MLA ``KQuantMultiLinear``)
that single-stream decode uses. (The gmlx server runs mlx-vlm's batch engine,
not this one; this harness is the reference batched path for the kernels both
share.) This test gates that batched decode is correct for every loadable
arch family.

The gate is NOT "token-for-token identical to single-stream". Batched and
unbatched GEMM use different reduction orders, so at an exact logit tie greedy
argmax can break the tie either way - a benign floating-point artifact, not a
masking/position/quant error. Asserting bit-identicality would falsely fail a
correct engine. The meaningful gate is three properties:

  1. ``test_batch_b1_token_exact`` - a single sequence run through the batched
     engine (batch size 1, no batch-shape effect) IS token-for-token identical
     to single-stream ``generate_step``. This proves the batched code path and
     the K-quant kernels are numerically faithful.

  2. ``test_batch_deterministic`` - a uniform batch (the same prompt repeated)
     produces identical output on every row. This proves no race / uninitialized
     state in the batched path.

  3. ``test_batch_divergence_only_at_ties`` - in a ragged (varying-length) batch,
     wherever a sequence's first divergence from single-stream occurs, the
     single-stream logprob margin between the two competing tokens is ~0 (an
     exact tie). A real attention/masking/position bug instead picks a
     confidently-wrong token (margin >> 1).
     ``test_batch_divergence_only_at_ties_native_fp`` checks the same with the
     MXFP4 or NVFP4 layout pinned to wire and to packed.

``integration`` + ``slow``; skips unless ``KQUANT_TEST_GGUF_DIR`` points at real
GGUFs (see ``conftest``). Select one arch with ``-k qwen2``.
"""

from __future__ import annotations

import pytest

import mlx.core as mx  # noqa: E402

# Arch families to sweep; each auto-skips when no matching GGUF is present. The
# set spans the K-quant module types: dense (KQuantLinear), MoE (gather_qmm),
# absorbed-MLA (KQuantMultiLinear), and MXFP4. Dense+SWA coverage comes from
# gemma3 / gemma4 (both use scaled_dot_product_attention and batch correctly).
#
# gemma2 is deliberately EXCLUDED: mlx-lm's gemma2 is the lone arch that hand-rolls
# attention to apply the +-50 attn-logit softcap, then does a manual `scores + mask`
# (gemma2.py) instead of sdpa - and that path mishandles BatchGenerator's ragged
# left-padded mask, so ragged batched decode diverges confidently. It is an upstream
# mlx-lm defect, not a gmlx/kq one: single-stream matches llama.cpp token-for-
# token, batch-size-1 == single-stream, and the sdpa-based gemma3/gemma4 batch cleanly.
CANDIDATE_ARCHES = [
    "qwen2", "qwen3", "llama", "gemma3", "phi3", "glm4",
    "qwen2moe", "qwen3moe", "mixtral", "glm4moe", "gemma4",
    "deepseek2", "nemotron_h_moe", "gpt-oss", "seed_oss", "smollm3", "granite",
    "ernie4_5-moe", "minimax-m2", "minimax-m3", "hunyuan-moe", "granitehybrid",
    "falcon-h1",
    "qwen3next",
    "kimi-k3",
    "muse-spark",
]

# Deliberately varying lengths -> a ragged batch -> exercises the left-padding /
# mask machinery that a position or masking bug would corrupt.
PROMPTS = [
    "Hi.",
    "The capital of France is",
    "In a quiet village nestled between two green hills, there lived an old "
    "clockmaker who",
    "Explain, step by step and in careful detail, how a modern lithium-ion "
    "battery stores and releases energy, beginning from the chemistry of the "
    "anode and cathode and ending with",
]

N_TOKENS = 32

# A benign tie-flip is a reduction-order swap of a top-2 near-tie: batched argmax
# takes single-stream's immediate runner-up. The margin is measured in logprobs,
# and since log_softmax only subtracts a shared constant the logprob gap EQUALS the
# logit-space gap - so a benign flip's margin is bounded by one bf16 logit ULP. That
# ULP scales with logit magnitude (0.0/0.0625/0.125 measured on qwen3/phi3/gemma4,
# but ~0.5-1.0 nat once an uncapped model's top logits run large), so a tight bound
# would flake on a genuine tie. A real attention/masking/position bug instead
# promotes a far-rank token at *many* nats: the excluded gemma2 batched path (see
# CANDIDATE_ARCHES) showed rank-16-to-137 tokens at 2.3-7.0 nats - the divergence
# magnitude this bound is calibrated against. 1.3 clears any plausible bf16-ULP tie
# yet stays far below that real-bug regime (the failure prints the rank to confirm).
TIE_EPS = 1.3

GREEDY = lambda x: mx.argmax(x, axis=-1)

pytestmark = pytest.mark.integration


# helpers
def _require(gguf_index, arch):
    paths = gguf_index.get(arch)
    if not paths:
        pytest.skip(f"no {arch!r} GGUF under KQUANT_TEST_GGUF_DIR "
                    f"(have: {sorted(gguf_index)})")
    return paths[0]


def _encode(tok, text):
    return list(tok.encode(text))


def _single_stream(model, ids, n):
    """Greedy single-stream decode; return (token_ids, per-step logprob vectors)."""
    from gmlx._vendor.mlx_lm.generate import generate_step
    from gmlx._vendor.mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    toks, logprobs = [], []
    for tok, lp in generate_step(mx.array(ids), model, max_tokens=n,
                                 sampler=GREEDY, prompt_cache=cache):
        toks.append(int(tok))
        logprobs.append(lp)
    return toks, logprobs


def _batched(model, ids_list, n):
    """Greedy batched decode; return per-input token-id lists (input order)."""
    from gmlx._vendor.mlx_lm.generate import BatchGenerator

    gen = BatchGenerator(model, sampler=GREEDY)  # greedy, no stop tokens
    uids = gen.insert(ids_list, [n] * len(ids_list))
    results = {u: [] for u in uids}
    with gen.stats():
        while responses := gen.next_generated():
            for r in responses:
                if r.finish_reason != "stop":
                    results[r.uid].append(int(r.token))
    gen.close()
    return [results[u] for u in uids]


def _first_divergence(a, b):
    for k in range(min(len(a), len(b))):
        if a[k] != b[k]:
            return k
    return None


def _native_fp_codecs(path):
    """The MXFP4 and NVFP4 codecs of a GGUF's tensors, every shard."""
    from gmlx.load.headerscan import scan_gguf
    from gmlx.load.native_fp import NATIVE_FP_CODECS
    from gmlx.load.preflight import find_split_shards

    return {t.type_name.lower() for p in find_split_shards(path)
            for t in scan_gguf(p).tensors} & NATIVE_FP_CODECS


def _native_fp_layouts(model):
    """The layouts the native-fp tensors loaded in: "packed" (MLX's packed
    stacks) or "wire" (the file bytes behind the kq kernels)."""
    from gmlx.load.modules import NativeFPSwitchLinear
    from gmlx.load.native_fp import NATIVE_FP_CODECS

    seen = set()
    for _, m in model.named_modules():
        if isinstance(m, NativeFPSwitchLinear):
            seen.add("packed")
        elif getattr(m, "kquant_type", None) in NATIVE_FP_CODECS:
            seen.add("wire")
    return seen


def _assert_ties_only(model, tok, arch):
    ids_list = [_encode(tok, p) for p in PROMPTS]
    refs = [_single_stream(model, ids, N_TOKENS) for ids in ids_list]
    batched = _batched(model, ids_list, N_TOKENS)

    for i, bat in enumerate(batched):
        ref_toks, ref_lps = refs[i]
        if bat == ref_toks:
            continue
        k = _first_divergence(ref_toks, bat)
        assert k is not None
        # At the first divergence the prefix is identical, so the single-stream
        # logprob vector at step k is the shared distribution both paths sampled.
        lp = ref_lps[k]
        margin = float(mx.max(lp)) - float(lp[bat[k]])
        # The batched token's rank in single-stream's own ordering (0 = the top
        # token) corroborates tie vs bug: a reduction-order flip can only swap the
        # top-2 (rank 1), while a masking/position bug promotes a far-rank token.
        rank = int((lp > lp[bat[k]]).sum())
        assert margin <= TIE_EPS, (
            f"{arch}: prompt {i} diverged at tok {k} on a NON-tie "
            f"(margin {margin:.4f} nats > {TIE_EPS}; batched tok is single-stream "
            f"rank {rank}); ref tok {ref_toks[k]} vs batched tok {bat[k]} - "
            f"indicates a batched-decode bug, not a floating-point tie-break")


# tests
@pytest.mark.parametrize("arch", CANDIDATE_ARCHES)
def test_batch_b1_token_exact(gguf_index, arch):
    """Batch size 1 must reproduce single-stream exactly, every prompt."""
    from gmlx import load_model

    model, _config, tok = load_model(_require(gguf_index, arch), verbose=False)
    for i, text in enumerate(PROMPTS):
        ids = _encode(tok, text)
        ref, _ = _single_stream(model, ids, N_TOKENS)
        b1 = _batched(model, [ids], N_TOKENS)[0]
        assert b1 == ref, (
            f"{arch}: batch-size-1 decode diverged from single-stream on "
            f"prompt {i} (first diff at tok {_first_divergence(ref, b1)})")


@pytest.mark.parametrize("arch", CANDIDATE_ARCHES)
def test_batch_deterministic(gguf_index, arch):
    """A uniform batch (same prompt x4) must yield identical rows."""
    from gmlx import load_model

    model, _config, tok = load_model(_require(gguf_index, arch), verbose=False)
    ids = _encode(tok, PROMPTS[1])
    rows = _batched(model, [ids] * 4, N_TOKENS)
    assert all(r == rows[0] for r in rows), (
        f"{arch}: uniform batch produced non-identical rows (nondeterministic)")


@pytest.mark.parametrize("arch", CANDIDATE_ARCHES)
def test_batch_divergence_only_at_ties(gguf_index, arch):
    """In a ragged batch, any first divergence from single-stream must sit on a
    logit tie (benign fp tie-break), never a confidently-wrong token (a bug)."""
    from gmlx import load_model

    model, _config, tok = load_model(_require(gguf_index, arch), verbose=False)
    _assert_ties_only(model, tok, arch)


@pytest.mark.parametrize("layout", ["wire", "packed"])
@pytest.mark.parametrize("arch", CANDIDATE_ARCHES)
def test_batch_divergence_only_at_ties_native_fp(gguf_index, arch, layout,
                                                 monkeypatch):
    """The ragged row with the MXFP4 or NVFP4 layout pinned. Auto picks wire
    only on a Mac the model does not fit, so a host runs one layout and
    this row runs both."""
    import mlx_kquant as kq

    from gmlx import load_model

    path = _require(gguf_index, arch)
    codecs = _native_fp_codecs(path)
    if not codecs:
        pytest.skip(f"{arch}: no MXFP4 or NVFP4 tensors, so no layout choice")
    if layout == "wire" and not codecs <= set(kq.codecs()):
        pytest.skip(f"this mlx-kquant build lacks {sorted(codecs)} wire kernels")
    monkeypatch.setenv("GMLX_NATIVE_FP", layout)
    model, _config, tok = load_model(path, verbose=False)
    assert layout in _native_fp_layouts(model)
    _assert_ties_only(model, tok, arch)
