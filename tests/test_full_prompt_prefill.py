#!/usr/bin/env python3
"""MTP serve-path integration tests: prefill, APC, batching, injection.

Exercises the owned MTP engine through the same ``BatchGenerator`` ->
``PromptProcessingBatch`` -> ``SpeculativeGenerationBatch`` pipeline the
server uses.

``integration`` + ``slow``; skips unless ``KQUANT_TEST_MTP_GGUF`` points at a
GGUF with MTP support. For assistant-model drafters (gemma-4), also set
``KQUANT_TEST_MTP_DRAFT_GGUF`` to the companion drafter GGUF.
"""
from __future__ import annotations

import contextlib
import logging
import os
import re

import pytest

import mlx.core as mx

pytestmark = [pytest.mark.integration, pytest.mark.slow]

GREEDY = lambda x: mx.argmax(x, axis=-1)

N_DECODE = 16

_SEED = (
    "The history of science is a long and winding story, full of false "
    "starts, lucky accidents, and the slow accumulation of careful "
    "measurement. From the earliest observations of the night sky to the "
    "development of quantum mechanics, each generation has built upon the "
    "work of its predecessors, often in unexpected ways. "
)

_SEED_B = (
    "Mathematics provides the language in which the laws of physics are "
    "expressed, from the differential equations of classical mechanics to "
    "the abstract algebras of particle physics. Each new mathematical "
    "framework has opened doors that were previously invisible. "
)

_SEED_C = (
    "The ocean covers more than seventy percent of the Earth's surface and "
    "contains an estimated ninety-seven percent of all water. Its currents "
    "drive weather patterns, regulate temperature, and sustain ecosystems "
    "that remain largely unexplored by modern science. "
)

_SEED_D = (
    "The development of writing systems transformed human civilization by "
    "enabling the transmission of knowledge across generations without "
    "relying on oral tradition alone. From cuneiform tablets to digital "
    "text, each medium has shaped how ideas spread and evolve. "
)

_SEED_E = (
    "Volcanic eruptions have shaped the Earth's landscape for billions of "
    "years, building mountain ranges, creating fertile soils, and sometimes "
    "triggering mass extinctions that redirected the course of evolution "
    "in ways that geologists are still working to understand. "
)


@pytest.fixture(scope="session")
def mtp_gguf():
    path = os.environ.get("KQUANT_TEST_MTP_GGUF")
    if not path:
        pytest.skip("set KQUANT_TEST_MTP_GGUF to a GGUF with MTP support")
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        pytest.skip(f"KQUANT_TEST_MTP_GGUF={path!r} not found")
    return path


@pytest.fixture(scope="session")
def mtp_draft_gguf():
    path = os.environ.get("KQUANT_TEST_MTP_DRAFT_GGUF")
    if not path:
        return None
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        pytest.skip(f"KQUANT_TEST_MTP_DRAFT_GGUF={path!r} not found")
    return path


@pytest.fixture(scope="session")
def mtp_model(mtp_gguf, mtp_draft_gguf):
    from gmlx.mtp_load import load_mtp_model
    return load_mtp_model(
        mtp_gguf, draft_gguf_path=mtp_draft_gguf, verbose=False
    )


def _build_prompt(tokenizer, n_tokens, seed=_SEED):
    """Build a prompt of at least ``n_tokens`` tokens from repeated seed text."""
    parts = []
    i = 1
    while len(tokenizer.encode("".join(parts))) < n_tokens + 64:
        parts.append(f"Section {i}. {seed}")
        i += 1
    text = "".join(parts)
    ids = tokenizer.encode(text)[:n_tokens]
    return ids


def _ref_first_token(model, ids):
    """Un-chunked greedy first token from the bare language model."""
    from mlx_lm.models.cache import make_prompt_cache
    lm = model.language_model
    cache = make_prompt_cache(lm)
    input_ids = mx.array([ids], dtype=mx.int32)
    out = lm(input_ids, cache=cache)
    logits = out.logits if hasattr(out, "logits") else out
    return int(mx.argmax(logits[:, -1, :], axis=-1).item())


def _make_processor(tokenizer):
    from gmlx.server_bridge_vlm import _make_text_processor
    return _make_text_processor(tokenizer)


def _embed_prompts(model, ids_list):
    prompt_kwargs_list = []
    for ids in ids_list:
        input_ids = mx.array([ids], dtype=mx.int32)
        emb_out = model.get_input_embeddings(input_ids=input_ids)
        prompt_kwargs_list.append({"inputs_embeds": emb_out.inputs_embeds})
    return prompt_kwargs_list


def _install_patches():
    from gmlx.spec_engine import (
        install_full_prompt_mtp_prefill,
        install_owned_spec_engine,
        install_continuous_batch_admission,
    )
    install_full_prompt_mtp_prefill()
    install_owned_spec_engine()
    install_continuous_batch_admission()


# The real MTP load patches process-wide seams in mlx_vlm's qwen3_5 module
# (verify fold, batched-verify SDPA, bf16 verify linear). Later files in the
# same process (test_qwen35_verify_fold) install against the pristine
# functions, so snapshot them and restore at module teardown. The snapshot
# runs at import (collection) time: the session-scoped mtp_model fixture
# instantiates before module-scoped fixtures, so a setup_module hook would
# capture the already-patched seams.
def _snapshot_q35_seams():
    # AttributeError guard: a renamed upstream seam must not break collection
    # of this file (the tests themselves skip without the env var).
    try:
        from mlx_vlm.models.qwen3_5 import language as q35
        from gmlx import gdn_patches, qwen35_verify_fold
        return (
            q35._target_verify_left_padded_attention,
            q35.scaled_dot_product_attention,
            q35._target_verify_linear,
            gdn_patches._BATCHED_VERIFY_SDPA_PATCHED,
            gdn_patches._BF16_VERIFY_LINEAR_PATCHED,
            qwen35_verify_fold._installed,
        )
    except (ImportError, AttributeError):
        return None


_q35_seams = _snapshot_q35_seams()


def teardown_module(module):
    if _q35_seams is None:
        return
    from mlx_vlm.models.qwen3_5 import language as q35
    from gmlx import gdn_patches, qwen35_verify_fold
    (q35._target_verify_left_padded_attention,
     q35.scaled_dot_product_attention,
     q35._target_verify_linear) = _q35_seams[:3]
    gdn_patches._BATCHED_VERIFY_SDPA_PATCHED = _q35_seams[3]
    gdn_patches._BF16_VERIFY_LINEAR_PATCHED = _q35_seams[4]
    qwen35_verify_fold._installed = _q35_seams[5]


def _run_mtp(model, drafter, tokenizer, ids_list, n, apc_manager=None):
    """Run MTP decode through BatchGenerator. Single insert, drain to completion.

    ``apc_manager`` goes through the real construction seam: the stash
    wrapper captures it onto the model (or clears a stale stash when None),
    exactly as the server's generation worker does.
    """
    from mlx_vlm.generate.ar import BatchGenerator

    _install_patches()

    processor = _make_processor(tokenizer)
    prompt_kwargs_list = _embed_prompts(model, ids_list)
    gen = BatchGenerator(
        model,
        processor,
        sampler=GREEDY,
        draft_model=drafter,
        draft_kind="mtp",
        greedy_sampling=True,
        max_tokens=n,
        prefill_step_size=2048,
        apc_manager=apc_manager,
    )
    uids = gen.insert(
        ids_list, [n] * len(ids_list),
        prompt_kwargs=prompt_kwargs_list,
    )
    results = {u: [] for u in uids}
    while gen.has_work:
        _prompt_responses, gen_responses = gen.next()
        for r in gen_responses:
            if r.finish_reason is None:
                results[r.uid].append(int(r.token))
    gen.close()
    return [results[u] for u in uids]


# ---------------------------------------------------------------------------
# B=1 prefill correctness
# ---------------------------------------------------------------------------

def test_long_prompt_first_token_parity(mtp_model):
    """A >2048-token prompt (chunked prefill) produces the same first token
    as an un-chunked forward pass through the bare language model."""
    model, drafter, config, tokenizer = mtp_model
    ids = _build_prompt(tokenizer, 3000)
    assert len(ids) >= 2048, f"prompt too short: {len(ids)} tokens"

    ref = _ref_first_token(model, ids)
    toks = _run_mtp(model, drafter, tokenizer, [ids], N_DECODE)[0]
    assert len(toks) > 0, "no tokens generated"
    assert toks[0] == ref, (
        f"first token diverged: mtp={toks[0]} ref={ref} -- "
        f"hidden corruption in chunked prefill"
    )


def test_short_prompt_first_token_parity(mtp_model):
    """A short prompt (<2048, no chunking) produces the same first token
    as the bare language model."""
    model, drafter, config, tokenizer = mtp_model
    ids = _build_prompt(tokenizer, 100)

    ref = _ref_first_token(model, ids)
    toks = _run_mtp(model, drafter, tokenizer, [ids], N_DECODE)[0]
    assert len(toks) > 0, "no tokens generated"
    assert toks[0] == ref, (
        f"first token diverged: mtp={toks[0]} ref={ref}"
    )


def test_multi_length_first_token_parity(mtp_model):
    """Prompts at chunk boundaries all produce the correct first token.
    Uses distinct seed texts to prevent APC cross-contamination."""
    model, drafter, config, tokenizer = mtp_model

    cases = [
        (200, _SEED),
        (2048, _SEED_B),
        (3000, _SEED_C),
    ]

    # Compute all refs before any _run_mtp calls so earlier MTP runs
    # cannot contaminate later ref computations via model state.
    refs = {}
    for length, seed in cases:
        ids = _build_prompt(tokenizer, length, seed=seed)
        refs[length] = (ids, _ref_first_token(model, ids))

    for length, seed in cases:
        # Clear APC before each case so entries from the previous
        # iteration cannot produce a false hit.
        if hasattr(model, "_spec_prefix_cache"):
            del model._spec_prefix_cache
        ids, ref = refs[length]
        toks = _run_mtp(model, drafter, tokenizer, [ids], N_DECODE)[0]
        assert len(toks) > 0, f"length={length}: no tokens generated"
        assert toks[0] == ref, (
            f"length={length}: first token diverged: mtp={toks[0]} ref={ref}"
        )


def test_prefill_step_env_override(monkeypatch):
    """Stock nulls prefill_step_size for speculative batches; the MTP
    re-enable must resolve None from the serve-side PREFILL_STEP_SIZE env
    override rather than pin the stock default. Exercised directly on the
    patched prompt_step (no model needed): the resolution runs before the
    needs_processing gate."""
    import types

    from mlx_vlm.generate.ar import PromptProcessingBatch

    from gmlx import spec_engine

    spec_engine.install_full_prompt_mtp_prefill()
    monkeypatch.setattr(spec_engine, "_mtp_prefill_init", lambda s: None)

    def fake_batch():
        return types.SimpleNamespace(
            draft_kind="mtp",
            prefill_step_size=None,
            needs_processing=lambda: False,
        )

    monkeypatch.setenv("PREFILL_STEP_SIZE", "97")
    fake = fake_batch()
    assert PromptProcessingBatch.prompt_step(fake) == 0
    assert fake.prefill_step_size == 97

    monkeypatch.delenv("PREFILL_STEP_SIZE")
    fake = fake_batch()
    PromptProcessingBatch.prompt_step(fake)
    assert fake.prefill_step_size == 2048


def test_prefill_step_env_override_e2e(mtp_model, monkeypatch):
    """Serve-shaped construction (prefill_step_size=None, no APC): the init
    re-enable restores chunking at the env-provided step, and first-token
    parity holds across the chunked prefill."""
    from mlx_vlm.generate.ar import BatchGenerator, PromptProcessingBatch

    model, drafter, config, tokenizer = mtp_model
    _install_patches()
    monkeypatch.setenv("PREFILL_STEP_SIZE", "97")
    if hasattr(model, "_spec_prefix_cache"):
        del model._spec_prefix_cache

    orig_step = PromptProcessingBatch.prompt_step
    steps = set()

    def spy(self):
        n = orig_step(self)
        steps.add(self.prefill_step_size)
        return n

    monkeypatch.setattr(PromptProcessingBatch, "prompt_step", spy)

    ids = _build_prompt(tokenizer, 300)
    ref = _ref_first_token(model, ids)
    processor = _make_processor(tokenizer)
    prompt_kwargs_list = _embed_prompts(model, [ids])
    gen = BatchGenerator(
        model,
        processor,
        sampler=GREEDY,
        draft_model=drafter,
        draft_kind="mtp",
        greedy_sampling=True,
        max_tokens=N_DECODE,
        prefill_step_size=None,
    )
    uids = gen.insert([ids], [N_DECODE], prompt_kwargs=prompt_kwargs_list)
    toks = []
    while gen.has_work:
        _prompt_responses, gen_responses = gen.next()
        for r in gen_responses:
            if r.finish_reason is None and r.uid == uids[0]:
                toks.append(int(r.token))
    gen.close()
    assert steps == {97}, f"env step not used in chunked prefill: {steps}"
    assert len(toks) > 0, "no tokens generated"
    assert toks[0] == ref, f"first token diverged: mtp={toks[0]} ref={ref}"


def test_long_context_token_parity(mtp_model):
    """8192 and 16384 token prompts: chunked prefill produces the same greedy
    first token as the unchunked reference. At these lengths, batch-GEMM
    numerics in the logit head cause a bounded logit diff (~1-1.5) between
    unchunked (all positions in one matmul) and chunked (2048 per matmul), but
    the argmax must be stable."""
    model, drafter, config, tokenizer = mtp_model

    cases = [
        (8192, _SEED_D),
        (16384, _SEED_E),
    ]

    refs = {}
    for length, seed in cases:
        ids = _build_prompt(tokenizer, length, seed=seed)
        refs[length] = (ids, _ref_first_token(model, ids))

    for length, seed in cases:
        if hasattr(model, "_spec_prefix_cache"):
            del model._spec_prefix_cache
        ids, ref = refs[length]
        toks = _run_mtp(model, drafter, tokenizer, [ids], N_DECODE)[0]
        assert len(toks) > 0, f"length={length}: no tokens generated"
        assert toks[0] == ref, (
            f"length={length}: first token diverged: mtp={toks[0]} ref={ref}"
        )


# ---------------------------------------------------------------------------
# APC (automatic prefix caching)
# ---------------------------------------------------------------------------

def test_apc_warm_start(mtp_model):
    """Two sequential requests sharing a long prefix: the second gets a
    SpecPrefixCache hit and decodes normally.

    First-token equality against an un-chunked reference is NOT asserted:
    with windowed sliding-KV trim, chunked and un-chunked prefill take
    different fp accumulation paths, and near-tied top logits can flip.
    Numerical correctness of the roundtrip itself is pinned by
    test_apc_restore_transparency below.
    """
    model, drafter, config, tokenizer = mtp_model
    vocab = int(config.get("vocab_size", 0)) or 256000

    if hasattr(model, "_spec_prefix_cache"):
        del model._spec_prefix_cache

    prefix_ids = _build_prompt(tokenizer, 3000)
    toks_cold = _run_mtp(model, drafter, tokenizer, [prefix_ids], N_DECODE)[0]
    assert len(toks_cold) > 0, "cold start: no tokens"
    assert all(0 <= t < vocab for t in toks_cold), "cold start: token out of range"

    spec_cache = getattr(model, "_spec_prefix_cache", None)
    assert spec_cache is not None, "SpecPrefixCache not created"
    assert len(spec_cache) == 1, f"expected 1 entry, got {len(spec_cache)}"

    # Turn-2 suffixes end mid-phrase on a strong collocation: a prompt that
    # ends a full sentence puts EOS in a near-tie with the continuation, and
    # chunked-vs-restored fp differences flip it (the row then finishes with
    # zero collected tokens).
    suffix_text = (
        " Now consider the implications of general relativity on quantum "
        "field theory in curved spacetime, framing the Unruh effect and "
        "Hawking radiation in the language of the second law of"
    )
    suffix_ids = tokenizer.encode(suffix_text)
    turn2_ids = prefix_ids + suffix_ids

    hit = spec_cache.lookup(mx.array([turn2_ids], dtype=mx.int32))
    assert hit is not None, "turn-2 prompt did not hit the prefix cache"
    assert hit[0] == len(prefix_ids), (
        f"hit prefix_len {hit[0]} != stored prefix {len(prefix_ids)}"
    )

    toks_warm = _run_mtp(model, drafter, tokenizer, [turn2_ids], N_DECODE)[0]
    assert len(toks_warm) > 0, "warm start: no tokens"
    assert all(0 <= t < vocab for t in toks_warm), "warm start: token out of range"


def test_apc_restore_transparency(mtp_model):
    """store -> restore is numerically transparent: suffix logits after a
    restore are bitwise-equal to running the suffix on the live cache the
    snapshot was taken from (identical chunk boundaries, so any difference
    is introduced by the snapshot/restore path itself)."""
    from mlx_lm.models.cache import make_prompt_cache

    from gmlx.prefix_cache import SpecPrefixCache

    model, drafter, config, tokenizer = mtp_model
    lm = model.language_model

    prefix_ids = _build_prompt(tokenizer, 3000)
    suffix_ids = tokenizer.encode(
        " Now consider the implications of general relativity on quantum "
        "field theory in curved spacetime, particularly the Unruh effect "
        "and Hawking radiation as thermodynamic phenomena. "
    )
    turn2_ids = prefix_ids + suffix_ids

    def _chunked(ids, cache, step=2048):
        logits = None
        for i in range(0, len(ids), step):
            chunk = mx.array([ids[i:i + step]], dtype=mx.int32)
            out = lm(chunk, cache=cache)
            logits = out.logits if hasattr(out, "logits") else out
            mx.eval([c.state for c in cache] + [logits])
        return logits

    # One chunked prefill; snapshot it, then keep using the live cache.
    cache_live = make_prompt_cache(lm)
    _chunked(prefix_ids, cache_live)
    spec = SpecPrefixCache()
    hidden = mx.zeros((1, 1, 8))
    spec.store(mx.array([prefix_ids], dtype=mx.int32), cache_live, hidden)

    # Live path: suffix on the original cache (also checks store() did not
    # mutate the source cache).
    live = _chunked(suffix_ids, cache_live)[0, -1, :].astype(mx.float32)
    del cache_live
    mx.clear_cache()

    # Restored path: same suffix, same chunking, snapshot-restored cache.
    hit = spec.lookup(mx.array([turn2_ids], dtype=mx.int32))
    assert hit is not None, "lookup missed the stored prefix"
    prefix_len, entry = hit
    assert prefix_len == len(prefix_ids)
    cache_restored = make_prompt_cache(lm)
    spec.restore(entry, cache_restored)
    restored = _chunked(
        turn2_ids[prefix_len:], cache_restored)[0, -1, :].astype(mx.float32)

    diff = float(mx.abs(live - restored).max())
    live_tok = int(mx.argmax(live).item())
    restored_tok = int(mx.argmax(restored).item())
    assert restored_tok == live_tok, (
        f"restored argmax {restored_tok} != live argmax {live_tok} "
        f"(max |logit diff| {diff:.6f})"
    )
    assert diff < 1e-3, f"restore not transparent: max |logit diff| {diff}"


# ---------------------------------------------------------------------------
# L1 APC: the shared APCManager (exact / block / disk) below L0
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _capture_spec_log():
    """Collect gmlx INFO log messages (spec_engine + engine.speculative)."""
    log = logging.getLogger("gmlx")
    messages = []
    handler = logging.Handler()
    handler.emit = lambda record: messages.append(record.getMessage())
    log.addHandler(handler)
    old_level = log.level
    log.setLevel(logging.INFO)
    try:
        yield messages
    finally:
        log.removeHandler(handler)
        log.setLevel(old_level)


def _clear_l0(model):
    if hasattr(model, "_spec_prefix_cache"):
        del model._spec_prefix_cache


def _l1_turn2_ids(tokenizer, prefix_ids):
    # Ends mid-phrase on a strong collocation (see test_apc_warm_start): a
    # sentence-final suffix near-ties EOS as the first token and fp noise
    # between the cold and warm paths flips it.
    suffix_text = (
        " Now consider how error-correcting codes protect deep-space "
        "telemetry against burst noise, with the effective throughput "
        "expressed in bits per"
    )
    return prefix_ids + tokenizer.encode(suffix_text)


def _cold_mtp_first_token(model, drafter, tokenizer, ids):
    """Cold first token through the same speculative pipeline.

    An un-chunked bare-LM forward is not a safe reference for the warm-cache
    asserts: chunked-vs-unchunked prefill numerics can flip near-tied logits
    (e.g. newline variants) and false-fail the parity check. Only a cold run
    through the same path isolates the cache restore.
    """
    _clear_l0(model)
    try:
        return _run_mtp(model, drafter, tokenizer, [ids], N_DECODE)[0][0]
    finally:
        _clear_l0(model)


def test_l1_exact_warm_start(mtp_model):
    """With L0 emptied between turns, the second request must be served by
    the shared APCManager (L1) with first-token parity against a cold
    reference. Exercises the stash seam, the lookup ladder, and the
    suffix-only drafter seeding (correctness only; acceptance is a
    separate measurement)."""
    from mlx_vlm.apc import APCManager

    model, drafter, config, tokenizer = mtp_model
    _install_patches()
    _clear_l0(model)
    manager = APCManager(num_blocks=2048, block_size=16)

    prefix_ids = _build_prompt(tokenizer, 3000, seed=_SEED_C)
    turn2_ids = _l1_turn2_ids(tokenizer, prefix_ids)
    ref = _cold_mtp_first_token(model, drafter, tokenizer, turn2_ids)

    try:
        toks_cold = _run_mtp(
            model, drafter, tokenizer, [prefix_ids], N_DECODE,
            apc_manager=manager)[0]
        assert len(toks_cold) > 0, "turn 1: no tokens"
        stats = manager.stats_snapshot()
        assert stats["exact_stores"] >= 1 or stats["stores"] >= 1, (
            f"turn 1 stored nothing in L1: {stats}"
        )

        _clear_l0(model)  # force turn 2 onto L1
        with _capture_spec_log() as messages:
            toks_warm = _run_mtp(
                model, drafter, tokenizer, [turn2_ids], N_DECODE,
                apc_manager=manager)[0]
        assert any("APC L1 hit" in m for m in messages), (
            "L1 lookup did not fire; log: " + "; ".join(messages[-5:])
        )
        assert not any(
            m.startswith("APC hit:") for m in messages
        ), "L0 unexpectedly served the warm turn"
        assert len(toks_warm) > 0, "turn 2: no tokens"
        assert toks_warm[0] == ref, (
            f"L1 warm first token {toks_warm[0]} != cold ref {ref} -- "
            f"restored KV misaligned"
        )
        stats = manager.stats_snapshot()
        assert stats["matched_tokens"] > 0, f"no matched tokens: {stats}"
    finally:
        model._kq_apc_manager = None
        manager.close()


def test_l1_short_prompt_store_and_hit(mtp_model):
    """A short prompt (no chunked prefill, so prompt_step never runs) must
    still store to L1 via the generate-path init, and a follow-up prompt
    sharing it as a prefix must hit."""
    from mlx_vlm.apc import APCManager

    model, drafter, config, tokenizer = mtp_model
    _install_patches()
    _clear_l0(model)
    manager = APCManager(num_blocks=2048, block_size=16)

    prefix_ids = _build_prompt(tokenizer, 200, seed=_SEED_D)
    turn2_ids = _l1_turn2_ids(tokenizer, prefix_ids)
    ref = _cold_mtp_first_token(model, drafter, tokenizer, turn2_ids)

    try:
        _run_mtp(model, drafter, tokenizer, [prefix_ids], N_DECODE,
                 apc_manager=manager)
        stats = manager.stats_snapshot()
        assert stats["exact_stores"] >= 1 or stats["stores"] >= 1, (
            f"short-prompt prefill stored nothing in L1: {stats}"
        )

        _clear_l0(model)
        with _capture_spec_log() as messages:
            toks_warm = _run_mtp(
                model, drafter, tokenizer, [turn2_ids], N_DECODE,
                apc_manager=manager)[0]
        assert any("APC L1 hit" in m for m in messages), (
            "L1 lookup did not fire on the short-prefix turn; log: "
            + "; ".join(messages[-5:])
        )
        assert toks_warm[0] == ref, (
            f"L1 warm first token {toks_warm[0]} != cold ref {ref}"
        )
    finally:
        model._kq_apc_manager = None
        manager.close()


def test_l1_sidecar_warm_start(mtp_model):
    """A warm L1 turn must restore the drafter-KV sidecar stored by the cold
    turn: turn 1 logs the sidecar store, turn 2 logs the sidecar hit and
    keeps first-token parity. Third run disables the sidecar via the kill
    switch and must fall back to a plain L1 hit (acceptance-parity numbers
    are the D-run's job; this certifies plumbing + correctness)."""
    import gmlx.speculative as _spec
    import gmlx.spec_engine as _eng
    from mlx_vlm.apc import APCManager

    model, drafter, config, tokenizer = mtp_model
    if not getattr(drafter, "supports_kv_sidecar", False):
        pytest.skip("drafter has no KV sidecar support (assistant drafter)")
    _install_patches()
    _clear_l0(model)
    manager = APCManager(num_blocks=2048, block_size=16)
    # Each turn stores checkpoint + post-prefill + retirement entries; the
    # default 2-slot LRU would evict the prefix entry the third run needs.
    manager._exact_cache_max = 8

    prefix_ids = _build_prompt(tokenizer, 3000, seed=_SEED_C)
    turn2_ids = _l1_turn2_ids(tokenizer, prefix_ids)
    ref = _cold_mtp_first_token(model, drafter, tokenizer, turn2_ids)

    try:
        with _capture_spec_log() as messages:
            _run_mtp(model, drafter, tokenizer, [prefix_ids], N_DECODE,
                     apc_manager=manager)
        assert any("APC sidecar store" in m for m in messages), (
            "cold turn stored no drafter sidecar; log: "
            + "; ".join(messages[-5:])
        )

        _clear_l0(model)
        with _capture_spec_log() as messages:
            toks_warm = _run_mtp(
                model, drafter, tokenizer, [turn2_ids], N_DECODE,
                apc_manager=manager)[0]
        assert any("APC L1 hit" in m for m in messages)
        assert any("APC sidecar hit" in m for m in messages), (
            "sidecar lookup did not fire on the warm turn; log: "
            + "; ".join(messages[-5:])
        )
        assert toks_warm[0] == ref, (
            f"sidecar warm first token {toks_warm[0]} != cold ref {ref} -- "
            f"restored drafter KV corrupted the round"
        )

        _clear_l0(model)
        old_spec, old_eng = _spec._SIDECAR_DISABLED, \
            _eng._SPEC_APC_SIDECAR_DISABLED
        _spec._SIDECAR_DISABLED = True
        _eng._SPEC_APC_SIDECAR_DISABLED = True
        try:
            with _capture_spec_log() as messages:
                toks_off = _run_mtp(
                    model, drafter, tokenizer, [turn2_ids], N_DECODE,
                    apc_manager=manager)[0]
        finally:
            _spec._SIDECAR_DISABLED = old_spec
            _eng._SPEC_APC_SIDECAR_DISABLED = old_eng
        assert any("APC L1 hit" in m for m in messages)
        assert not any("APC sidecar hit" in m for m in messages), (
            "kill switch did not disable the sidecar lookup"
        )
        assert toks_off[0] == ref
    finally:
        model._kq_apc_manager = None
        manager.close()


def _is_hybrid(model):
    from mlx_lm.models.cache import ArraysCache
    lm = getattr(model, "language_model", None) or model
    try:
        return any(isinstance(c, ArraysCache) for c in lm.make_cache())
    except Exception:
        return False


def test_ckpt_warm_start_and_fallback(mtp_model):
    """On a gated-delta hybrid the warm L1 turn must be served by the
    checkpoint tier (attn-KV blocks + recurrent-state sidecar, tier=ckpt in
    the hit log) with first-token parity. With the kill switch thrown the
    tier is invisible both ways: a re-run of the cold turn stores stock
    exact entries again, and the following warm turn hits tier=exact with
    the same parity."""
    import gmlx.spec_engine as _eng
    from mlx_vlm.apc import APCManager

    model, drafter, config, tokenizer = mtp_model
    if not _is_hybrid(model):
        pytest.skip("not a hybrid model; checkpoint tier not applicable")
    _install_patches()
    _clear_l0(model)
    manager = APCManager(num_blocks=2048, block_size=16)
    manager._exact_cache_max = 8

    prefix_ids = _build_prompt(tokenizer, 3000, seed=_SEED_D)
    turn2_ids = _l1_turn2_ids(tokenizer, prefix_ids)
    ref = _cold_mtp_first_token(model, drafter, tokenizer, turn2_ids)

    try:
        with _capture_spec_log() as messages:
            _run_mtp(model, drafter, tokenizer, [prefix_ids], N_DECODE,
                     apc_manager=manager)
        assert any("APC ckpt store" in m for m in messages), (
            "cold turn stored no checkpoint; log: "
            + "; ".join(messages[-8:])
        )

        _clear_l0(model)
        with _capture_spec_log() as messages:
            toks_warm = _run_mtp(
                model, drafter, tokenizer, [turn2_ids], N_DECODE,
                apc_manager=manager)[0]
        assert any("tier=ckpt" in m for m in messages), (
            "warm turn not served by the checkpoint tier; log: "
            + "; ".join(messages[-8:])
        )
        assert toks_warm[0] == ref, (
            f"ckpt warm first token {toks_warm[0]} != cold ref {ref} -- "
            f"reassembled KV/state misaligned"
        )

        old = _eng._SPEC_APC_CKPT_DISABLED
        _eng._SPEC_APC_CKPT_DISABLED = True
        try:
            _clear_l0(model)
            with _capture_spec_log() as messages:
                _run_mtp(model, drafter, tokenizer, [prefix_ids], N_DECODE,
                         apc_manager=manager)
            assert not any("tier=ckpt" in m for m in messages), (
                "kill switch did not stop ckpt lookups"
            )
            _clear_l0(model)
            with _capture_spec_log() as messages:
                toks_exact = _run_mtp(
                    model, drafter, tokenizer, [turn2_ids], N_DECODE,
                    apc_manager=manager)[0]
            assert any("tier=exact" in m for m in messages), (
                "exact-tier fallback did not serve; log: "
                + "; ".join(messages[-8:])
            )
            assert toks_exact[0] == ref, (
                f"exact fallback first token {toks_exact[0]} != ref {ref}"
            )
        finally:
            _eng._SPEC_APC_CKPT_DISABLED = old
    finally:
        model._kq_apc_manager = None
        manager.close()


def test_l1_sidecar_retirement_multiturn(mtp_model):
    """Chat-shaped turn 2 (turn-1 prompt + reply + new text) must hit the
    RETIREMENT entry and find the drafter sidecar stored at retirement.
    The other sidecar test extends the prompt only, so it lands on the
    post-prefill keys and never exercises the retirement store point."""
    from mlx_vlm.apc import APCManager

    model, drafter, config, tokenizer = mtp_model
    if not getattr(drafter, "supports_kv_sidecar", False):
        pytest.skip("drafter has no KV sidecar support (assistant drafter)")
    _install_patches()
    _clear_l0(model)
    manager = APCManager(num_blocks=2048, block_size=16)
    manager._exact_cache_max = 8

    prefix_ids = _build_prompt(tokenizer, 3000, seed=_SEED_B)
    try:
        with _capture_spec_log() as messages:
            reply = _run_mtp(model, drafter, tokenizer, [prefix_ids],
                             N_DECODE, apc_manager=manager)[0]
        retire_msgs = [m for m in messages if "APC retire store:" in m]
        assert retire_msgs, (
            "turn 1 did not retire; log: " + "; ".join(messages[-8:])
        )
        stored_len = int(
            re.search(r"tokens=(\d+)", retire_msgs[-1]).group(1))
        if _is_hybrid(model):
            assert any("APC ckpt store" in m for m in messages), (
                "hybrid retirement did not go through the checkpoint tier; "
                "log: " + "; ".join(messages[-8:])
            )
        assert any("APC sidecar store (retire)" in m for m in messages), (
            "retirement stored no drafter sidecar; log: "
            + "; ".join(messages[-8:])
        )

        seq = list(prefix_ids) + [int(t) for t in reply]
        assert stored_len <= len(seq)
        turn2_ids = seq[:stored_len] + tokenizer.encode(
            " Now relate this history to the total cost of")
        ref = _cold_mtp_first_token(model, drafter, tokenizer, turn2_ids)

        _clear_l0(model)
        with _capture_spec_log() as messages:
            toks_warm = _run_mtp(
                model, drafter, tokenizer, [turn2_ids], N_DECODE,
                apc_manager=manager)[0]
        hits = [m for m in messages if "APC L1 hit" in m]
        assert hits, "turn 2 missed L1"
        hit_prefix = int(re.search(r"prefix=(\d+)", hits[-1]).group(1))
        assert hit_prefix == stored_len, (
            f"turn 2 hit prefix {hit_prefix}, expected the retirement "
            f"entry at {stored_len}"
        )
        assert any("APC sidecar hit" in m for m in messages), (
            "no sidecar at the retirement prefix; log: "
            + "; ".join(messages[-8:])
        )
        assert toks_warm[0] == ref, (
            f"retirement-warm first token {toks_warm[0]} != cold ref {ref}"
        )
    finally:
        model._kq_apc_manager = None
        manager.close()


def test_l1_disk_restart(mtp_model, tmp_path):
    """L1 entries persist on the SSD tier across a manager 'restart': a
    fresh manager over the same disk directory (empty memory tiers) must
    serve the shared prefix with first-token parity."""
    from mlx_vlm.apc import APCManager, DiskBlockStore

    model, drafter, config, tokenizer = mtp_model
    _install_patches()
    _clear_l0(model)

    prefix_ids = _build_prompt(tokenizer, 3000, seed=_SEED_E)
    turn2_ids = _l1_turn2_ids(tokenizer, prefix_ids)
    ref = _cold_mtp_first_token(model, drafter, tokenizer, turn2_ids)

    manager1 = APCManager(
        num_blocks=2048, block_size=16,
        disk=DiskBlockStore(tmp_path / "apc", namespace="test"))
    try:
        _run_mtp(model, drafter, tokenizer, [prefix_ids], N_DECODE,
                 apc_manager=manager1)
        stats = manager1.stats_snapshot()
        assert stats["disk_writes"] >= 1, f"nothing written to disk: {stats}"
    finally:
        model._kq_apc_manager = None
        manager1.close()  # joins the disk writer; flushes pending shards

    manager2 = APCManager(
        num_blocks=2048, block_size=16,
        disk=DiskBlockStore(tmp_path / "apc", namespace="test"))
    try:
        _clear_l0(model)
        with _capture_spec_log() as messages:
            toks_warm = _run_mtp(
                model, drafter, tokenizer, [turn2_ids], N_DECODE,
                apc_manager=manager2)[0]
        assert any("APC L1 hit" in m for m in messages), (
            "L1 disk tier did not serve after restart; log: "
            + "; ".join(messages[-5:])
        )
        if _is_hybrid(model):
            assert any("tier=ckpt" in m for m in messages), (
                "hybrid restart hit did not come from the checkpoint tier; "
                "log: " + "; ".join(messages[-5:])
            )
        assert toks_warm[0] == ref, (
            f"disk-warm first token {toks_warm[0]} != cold ref {ref}"
        )
        stats = manager2.stats_snapshot()
        assert stats["disk_hits"] >= 1 or stats["exact_hits"] >= 1, (
            f"restart lookup did not read the disk tier: {stats}"
        )
    finally:
        model._kq_apc_manager = None
        manager2.close()


def test_l1_kill_switch(mtp_model, monkeypatch):
    """GMLX_SPEC_APC=0 must disable the L1 lookup even with a manager
    stashed. (The flag is read at import; patch the module constant.)"""
    from mlx_vlm.apc import APCManager
    import gmlx.spec_engine as spec_engine

    model, drafter, config, tokenizer = mtp_model
    _install_patches()
    _clear_l0(model)
    manager = APCManager(num_blocks=2048, block_size=16)

    prefix_ids = _build_prompt(tokenizer, 3000, seed=_SEED_B)
    monkeypatch.setattr(spec_engine, "_SPEC_APC_DISABLED", True)
    try:
        with _capture_spec_log() as messages:
            _run_mtp(model, drafter, tokenizer, [prefix_ids], N_DECODE,
                     apc_manager=manager)
        assert not any("APC" in m for m in messages), (
            "APC activity despite kill switch: " + "; ".join(messages[-5:])
        )
        stats = manager.stats_snapshot()
        assert stats["exact_stores"] == 0 and stats["stores"] == 0, (
            f"kill switch did not stop L1 stores: {stats}"
        )
    finally:
        model._kq_apc_manager = None
        manager.close()


# ---------------------------------------------------------------------------
# B>1 batched decode (owned_server_rounds_batch)
# ---------------------------------------------------------------------------

def _drafter_supports_batch(drafter):
    """The owned batch round needs drafter.reset(model, left_padding=...)."""
    import inspect
    reset = getattr(drafter, "reset", None)
    if reset is None:
        return False
    sig = inspect.signature(reset)
    return "left_padding" in sig.parameters


def test_batch_b2_first_token_parity(mtp_model):
    """Each prompt in a B=2 batch produces the same first token as the same
    prompt run alone at B=1."""
    model, drafter, config, tokenizer = mtp_model
    if not _drafter_supports_batch(drafter):
        pytest.skip("drafter does not support batched reset (assistant-model)")

    ids_a = _build_prompt(tokenizer, 500, seed=_SEED)
    ids_b = _build_prompt(tokenizer, 1200, seed=_SEED_B)

    ref_a = _ref_first_token(model, ids_a)
    ref_b = _ref_first_token(model, ids_b)

    batch_results = _run_mtp(model, drafter, tokenizer, [ids_a, ids_b], N_DECODE)
    assert len(batch_results[0]) > 0, "prompt A: no tokens"
    assert len(batch_results[1]) > 0, "prompt B: no tokens"
    assert batch_results[0][0] == ref_a, (
        f"prompt A: batch={batch_results[0][0]} ref={ref_a}"
    )
    assert batch_results[1][0] == ref_b, (
        f"prompt B: batch={batch_results[1][0]} ref={ref_b}"
    )


def test_batch_b3_mixed_lengths(mtp_model):
    """Three prompts at extreme length ratios (100, 2048, 3000). Each must
    produce the correct first token. Exercises left-padding alignment."""
    model, drafter, config, tokenizer = mtp_model
    if not _drafter_supports_batch(drafter):
        pytest.skip("drafter does not support batched reset (assistant-model)")

    ids_a = _build_prompt(tokenizer, 100, seed=_SEED)
    ids_b = _build_prompt(tokenizer, 2048, seed=_SEED_B)
    ids_c = _build_prompt(tokenizer, 3000, seed=_SEED_C)

    ref_a = _ref_first_token(model, ids_a)
    ref_b = _ref_first_token(model, ids_b)
    ref_c = _ref_first_token(model, ids_c)

    results = _run_mtp(model, drafter, tokenizer, [ids_a, ids_b, ids_c], N_DECODE)
    for i, (toks, ref) in enumerate(zip(results, [ref_a, ref_b, ref_c])):
        assert len(toks) > 0, f"prompt {i}: no tokens"
        assert toks[0] == ref, (
            f"prompt {i}: batch={toks[0]} ref={ref}"
        )


# ---------------------------------------------------------------------------
# Continuous-batch injection (insert mid-decode)
# ---------------------------------------------------------------------------

def test_continuous_batch_injection(mtp_model):
    """A second request injected mid-decode of the first. The injected
    request's first token must match its solo B=1 reference. The first
    request's first token (emitted before injection) must also match."""
    from mlx_vlm.generate.ar import BatchGenerator

    model, drafter, config, tokenizer = mtp_model
    if not _drafter_supports_batch(drafter):
        pytest.skip("drafter does not support batched reset (assistant-model)")

    _install_patches()

    ids_first = _build_prompt(tokenizer, 500, seed=_SEED)
    ids_second = _build_prompt(tokenizer, 300, seed=_SEED_B)

    ref_first = _ref_first_token(model, ids_first)
    ref_second = _ref_first_token(model, ids_second)

    processor = _make_processor(tokenizer)
    gen = BatchGenerator(
        model,
        processor,
        sampler=GREEDY,
        draft_model=drafter,
        draft_kind="mtp",
        greedy_sampling=True,
        max_tokens=N_DECODE,
        prefill_step_size=2048,
    )

    kwargs_first = _embed_prompts(model, [ids_first])
    uids_first = gen.insert([ids_first], [N_DECODE], prompt_kwargs=kwargs_first)

    results = {uids_first[0]: []}
    injected = False
    uids_second = None

    while gen.has_work:
        _prompt_responses, gen_responses = gen.next()
        for r in gen_responses:
            if r.uid not in results:
                results[r.uid] = []
            if r.finish_reason is None:
                results[r.uid].append(int(r.token))

        if not injected and len(results[uids_first[0]]) >= 3:
            kwargs_second = _embed_prompts(model, [ids_second])
            uids_second = gen.insert(
                [ids_second], [N_DECODE], prompt_kwargs=kwargs_second
            )
            results[uids_second[0]] = []
            injected = True

    gen.close()

    assert injected, "injection never fired (first request finished too fast)"
    toks_first = results[uids_first[0]]
    toks_second = results[uids_second[0]]
    assert len(toks_first) > 0, "first request: no tokens"
    assert len(toks_second) > 0, "injected request: no tokens"
    assert toks_first[0] == ref_first, (
        f"first request first token {toks_first[0]} != ref {ref_first} -- "
        f"corrupted before injection"
    )
    assert toks_second[0] == ref_second, (
        f"injected request first token {toks_second[0]} != ref {ref_second} -- "
        f"injection corrupted output"
    )


def _drain_generator(gen, results, inject_at):
    """Run gen to completion; fire each (trigger_fn, insert_fn) in inject_at
    once its trigger first returns True. Returns injected uid lists."""
    injected_uids = [None] * len(inject_at)
    fired = [False] * len(inject_at)
    while gen.has_work:
        _prompt_responses, gen_responses = gen.next()
        for r in gen_responses:
            if r.uid not in results:
                results[r.uid] = []
            if r.finish_reason is None:
                results[r.uid].append(int(r.token))
        for i, (trigger, insert) in enumerate(inject_at):
            if not fired[i] and trigger(results):
                injected_uids[i] = insert()
                results[injected_uids[i][0]] = []
                fired[i] = True
    gen.close()
    assert all(fired), "injection trigger never fired"
    return injected_uids


def test_scalar_injection_defers_and_completes(mtp_model):
    """A request injected while a B=1 (scalar rounds) request is decoding
    must produce its full, correct output. Previously the injection was
    merged into the scalar batch: the scalar generator never drains
    model._generator_injections, so the injected stream was truncated after
    one token, its continuation re-dispatched from the finished row's cache,
    and the stale entry poisoned the next batch's first-round drain."""
    from mlx_vlm.generate.ar import BatchGenerator

    model, drafter, config, tokenizer = mtp_model
    if not _drafter_supports_batch(drafter):
        pytest.skip("drafter does not support batched reset (assistant-model)")

    _install_patches()

    ids_a = _build_prompt(tokenizer, 500, seed=_SEED)
    ids_c = _build_prompt(tokenizer, 300, seed=_SEED_B)
    ref_c = _run_mtp(model, drafter, tokenizer, [ids_c], 6)[0]

    processor = _make_processor(tokenizer)
    gen = BatchGenerator(
        model, processor, sampler=GREEDY, draft_model=drafter,
        draft_kind="mtp", greedy_sampling=True, max_tokens=N_DECODE,
        prefill_step_size=2048)
    uids_a = gen.insert([ids_a], [N_DECODE],
                        prompt_kwargs=_embed_prompts(model, [ids_a]))
    results = {uids_a[0]: []}
    (uids_c,) = _drain_generator(gen, results, [(
        lambda res: len(res[uids_a[0]]) >= 3,
        lambda: gen.insert([ids_c], [N_DECODE],
                           prompt_kwargs=_embed_prompts(model, [ids_c])),
    )])

    assert not getattr(model, "_generator_injections", None), (
        "stale injection entry left on the model"
    )
    got = results[uids_c[0]]
    n = min(len(got), len(ref_c))
    assert n >= 5 and got[:n] == ref_c[:n], (
        f"injected request content diverged/truncated: {got[:6]} vs solo {ref_c}"
    )


def test_injection_when_all_rows_finish_same_round(mtp_model):
    """An injection queued while the BATCH generator's active rows all
    finish must still be adopted. Previously the all(finished) break exited
    without draining model._generator_injections: the injected request
    stalled and the stale entry was adopted by an unrelated later batch."""
    from mlx_vlm.generate.ar import BatchGenerator

    model, drafter, config, tokenizer = mtp_model
    if not _drafter_supports_batch(drafter):
        pytest.skip("drafter does not support batched reset (assistant-model)")

    _install_patches()

    ids_a = _build_prompt(tokenizer, 500, seed=_SEED)
    ids_b = _build_prompt(tokenizer, 1200, seed=_SEED_C)
    ids_c = _build_prompt(tokenizer, 300, seed=_SEED_B)
    ref_c = _run_mtp(model, drafter, tokenizer, [ids_c], 6)[0]

    processor = _make_processor(tokenizer)
    gen = BatchGenerator(
        model, processor, sampler=GREEDY, draft_model=drafter,
        draft_kind="mtp", greedy_sampling=True, max_tokens=N_DECODE,
        prefill_step_size=2048)
    # Tight budgets so the batch's final round comes fast; C lands either
    # mid-final-round (exit-path drain) or at a round top (loop drain) --
    # both must adopt it with correct content.
    kwargs = _embed_prompts(model, [ids_a, ids_b])
    uids_ab = gen.insert([ids_a, ids_b], [2, 3], prompt_kwargs=kwargs)
    results = {uid: [] for uid in uids_ab}
    (uids_c,) = _drain_generator(gen, results, [(
        lambda res: len(res[uids_ab[0]]) >= 1,
        lambda: gen.insert([ids_c], [N_DECODE],
                           prompt_kwargs=_embed_prompts(model, [ids_c])),
    )])

    assert not getattr(model, "_generator_injections", None), (
        "stale injection entry left on the model"
    )
    got = results[uids_c[0]]
    n = min(len(got), len(ref_c))
    assert n >= 5 and got[:n] == ref_c[:n], (
        f"injected request content diverged/truncated: {got[:6]} vs solo {ref_c}"
    )


# ---------------------------------------------------------------------------
# APC + continuous-batch injection (the production-valuable interaction)
# ---------------------------------------------------------------------------

def test_apc_hit_on_injected_request(mtp_model):
    """Request A is mid-decode; request B shares A's long cached prefix,
    gets injected, hits APC (skips prefix re-prefill), and joins the B>1
    batch with the correct first token.

    This is the production-valuable APC interaction: multi-turn chat where
    successive requests share system_prompt + conversation_history, and a
    new request arrives while a prior request is still decoding.

    Validates the only real correctness surface: an APC-restored single-row
    KVCache (with its restored offset) lands at the correct per-row
    offset/left-padding when BatchKVCache.extend injects it into the live
    batch. First-token parity on the injected, APC-hit row proves that
    alignment is correct.

    Also asserts that PromptProcessingBatch prefills single-request (B=1)
    -- if mlx-vlm ever coalesces prefills into B>1, the b==1 guard in
    spec_engine silently disables APC and this test fails loudly via the
    APC-hit assertion rather than producing a silent degradation.
    """
    import logging
    from mlx_vlm.generate.ar import BatchGenerator

    model, drafter, config, tokenizer = mtp_model
    if not _drafter_supports_batch(drafter):
        pytest.skip("drafter does not support batched reset (assistant-model)")

    _install_patches()

    # Clear any prior APC state so we control the cache lifecycle.
    if hasattr(model, "_spec_prefix_cache"):
        del model._spec_prefix_cache

    # Build a long shared prefix (request A's full prompt).
    prefix_ids = _build_prompt(tokenizer, 3000, seed=_SEED)

    # Request B = same prefix + a distinct suffix.
    # Mid-phrase ending (see test_apc_warm_start): sentence-final suffixes
    # near-tie EOS at the first token.
    suffix_text = (
        " Now consider the thermodynamic implications of Landauer's "
        "principle on reversible computation, with the minimum erasure "
        "cost at room temperature expressed in joules per"
    )
    suffix_ids = tokenizer.encode(suffix_text)
    ids_second = prefix_ids + suffix_ids

    # Solo B=1 references for both requests.
    ref_first = _ref_first_token(model, prefix_ids)
    ref_second = _ref_first_token(model, ids_second)

    processor = _make_processor(tokenizer)
    gen = BatchGenerator(
        model,
        processor,
        sampler=GREEDY,
        draft_model=drafter,
        draft_kind="mtp",
        greedy_sampling=True,
        max_tokens=N_DECODE,
        prefill_step_size=2048,
    )

    # Insert request A -- its prefill stores the APC entry.
    kwargs_first = _embed_prompts(model, [prefix_ids])
    uids_first = gen.insert(
        [prefix_ids], [N_DECODE], prompt_kwargs=kwargs_first
    )

    results = {uids_first[0]: []}
    injected = False
    uids_second = None
    apc_hit_seen = False

    # Capture APC log to verify the hit actually fired.
    apc_log = logging.getLogger("gmlx.spec_engine")
    log_messages = []
    handler = logging.Handler()
    handler.emit = lambda record: log_messages.append(record.getMessage())
    apc_log.addHandler(handler)
    old_level = apc_log.level
    apc_log.setLevel(logging.INFO)

    try:
        while gen.has_work:
            _prompt_responses, gen_responses = gen.next()
            for r in gen_responses:
                if r.uid not in results:
                    results[r.uid] = []
                if r.finish_reason is None:
                    results[r.uid].append(int(r.token))

            # After request A has emitted a few tokens (proving it is
            # mid-decode), inject request B which shares A's prefix.
            if not injected and len(results[uids_first[0]]) >= 3:
                kwargs_second = _embed_prompts(model, [ids_second])
                uids_second = gen.insert(
                    [ids_second], [N_DECODE], prompt_kwargs=kwargs_second
                )
                results[uids_second[0]] = []
                injected = True
    finally:
        apc_log.removeHandler(handler)
        apc_log.setLevel(old_level)

    gen.close()

    assert injected, "injection never fired (first request finished too fast)"

    # Verify APC actually fired on request B's prefill.
    apc_hit_seen = any("APC hit" in m for m in log_messages)
    assert apc_hit_seen, (
        "APC hit did not fire on the injected request -- either "
        "PromptProcessingBatch is no longer single-request (B=1), or the "
        "prefix was not cached after request A's prefill. "
        "Log messages: " + "; ".join(log_messages[-5:])
    )

    toks_first = results[uids_first[0]]
    toks_second = results[uids_second[0]]
    assert len(toks_first) > 0, "first request: no tokens"
    assert len(toks_second) > 0, "injected request: no tokens"

    assert toks_first[0] == ref_first, (
        f"request A first token {toks_first[0]} != ref {ref_first} -- "
        f"corrupted before injection"
    )
    assert toks_second[0] == ref_second, (
        f"injected APC-hit request B first token {toks_second[0]} != "
        f"ref {ref_second} -- APC-restored cache offset misaligned "
        f"after BatchKVCache.extend injection"
    )
