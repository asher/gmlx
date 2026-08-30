#!/usr/bin/env python3
"""HY4 long-context gates that need no llama.cpp reference.

The shared sweep in ``test_long_context.py`` covers HY4 too, but its parity
test needs an oracle and HY4's oracle is expensive and shallow: patch 0002
implements STQ1_0 for CPU and CUDA only, so llama.cpp has to run ``-dev none
--no-op-offload`` and prefills the 229 GB file at ~2.25 tok/s. That puts a 16k
reference out of reach and makes even a 4k one a half-hour run.

So the DSA key-selection chain - the part a shallow oracle cannot reach - is
validated against the model itself instead, three ways:

  * Below ``index_topk`` cached keys the selection covers every key, so the
    sparse path and ``GMLX_HY4_SPARSE_DISABLE=1`` must produce IDENTICAL
    tokens. This isolates a gather/mask defect from real sparsity.
  * Above the boundary the two paths legitimately differ, but both must still
    retrieve a fact planted ~100 tokens into the prompt. A corrupted selection
    chain (a shared layer that lost the preceding layer's pick, an off-by-one
    in the top-k gather) drops the needle while staying fluent.
  * The tiled online-softmax prefill and the direct path must agree token for
    token on the same prompt.

All three are ``integration`` + ``slow`` and skip without a HY4 GGUF under
``KQUANT_TEST_GGUF_DIR``. On a 128 GB box the model runs on the expert-
streaming tier, so each of these is minutes, not seconds, and the whole file
shares one load (see ``hy4_model``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import mlx.core as mx  # noqa: E402

from tests.gen.test_long_context import (  # noqa: E402
    _NEEDLE,
    _build_prompt,
    _load,
    _require,
)

ARCH = "hyv4"

# Recorded llama.cpp greedy references. Checked in rather than regenerated
# because reproducing them needs a patched llama.cpp worktree that no CI has,
# and the cross-boundary run alone costs 22 minutes of CPU-only prefill.
_ORACLE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "hy_v4_oracle"

# Comfortably below the model's index_topk of 2048, so the indexer returns no
# selection at all and sparse-vs-dense must be bit-identical.
BELOW_BOUNDARY = 1024

# Above it, so the selection is live for most of the prompt. Kept modest: the
# needle sits ~100 tokens in, which is what makes retrieval discriminative,
# not the total depth.
ABOVE_BOUNDARY = int(os.environ.get("GMLX_HY4_LONGCTX_TOKENS", "4096"))

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _greedy_text(model, tok, ids, n):
    from mlx_lm.generate import stream_generate

    out = [int(r.token) for r in
           stream_generate(model, tok, mx.array(ids), max_tokens=n)]
    return out, tok.decode(out)


@pytest.fixture(scope="module")
def hy4_model(gguf_index):
    """One HY4 load for the whole file, released at teardown.

    Every gate here is a property of one model, so loading per test buys
    nothing and costs a great deal. On the streaming tier a load mlocks an
    every-token weight pin and a decode arena, and a dropped model does not
    give them back at the drop: the decode feeder holds every MoE module
    and every MoE module holds the feeder, so the tree waits for a
    generational collection. Two wired installs at once is how a 128 GB box
    earns a watchdog panic - wired pages are the one kind memory pressure
    and jetsam cannot take back, so the machine deadlocks rather than
    killing anything.

    Sharing the load is safe because the flags these tests move are read
    per call: ``GMLX_HY4_SPARSE_DISABLE`` in ``_sparse_disabled`` and the
    tiling thresholds through ``monkeypatch``, which undoes itself.
    """
    path = _require(gguf_index, ARCH)
    model, config, tok = _load(path)
    yield path, model, config, tok
    from gmlx.stream.installs import release

    release(model)


def test_sparse_and_dense_agree_below_the_selection_boundary(
        hy4_model, monkeypatch):
    # The identity case: with fewer cached keys than index_topk the indexer
    # returns None, so the two paths run the same math and any difference is
    # a defect in the selection plumbing rather than in sparsity itself.
    _, model, _, tok = hy4_model
    _, ids = _build_prompt(tok, BELOW_BOUNDARY)

    sparse_ids, sparse_text = _greedy_text(model, tok, ids, 24)

    monkeypatch.setenv("GMLX_HY4_SPARSE_DISABLE", "1")
    dense_ids, dense_text = _greedy_text(model, tok, ids, 24)

    assert sparse_ids == dense_ids, (
        "below index_topk the selection is the identity, so the paths must "
        f"agree token for token\n  sparse: {sparse_text[:80]!r}\n"
        f"  dense : {dense_text[:80]!r}")


def _needle_depth(config) -> int:
    return min(ABOVE_BOUNDARY,
               int(config.get("max_position_embeddings") or 8192) - 32)


def test_needle_recall_above_the_selection_boundary(hy4_model):
    # The oracle-free retrieval gate. The planted code sits ~100 tokens into a
    # prompt several thousand long, so answering the cloze requires the key
    # selection to keep a mid-context key alive across every shared layer.
    _, model, config, tok = hy4_model
    depth = _needle_depth(config)
    assert depth > 2048, (
        f"depth {depth} does not cross index_topk; this gate needs a prompt "
        f"above the DSA boundary to mean anything")
    _, ids = _build_prompt(tok, depth)

    _, text = _greedy_text(model, tok, ids, 32)
    assert _NEEDLE in text, (
        f"needle {_NEEDLE!r} not retrieved from depth {depth}: mid-context "
        f"attention or the shared top-k chain is corrupt\n  got: {text[:120]!r}")


def test_forced_dense_also_retrieves_the_needle(hy4_model, monkeypatch):
    # The control for the test above. If the dense path also fails, the fault
    # is not in the selection - it is in the attention underneath it, and the
    # sparse result carries no signal about the indexer.
    _, model, config, tok = hy4_model
    monkeypatch.setenv("GMLX_HY4_SPARSE_DISABLE", "1")
    depth = _needle_depth(config)
    _, ids = _build_prompt(tok, depth)

    _, text = _greedy_text(model, tok, ids, 32)
    assert _NEEDLE in text, (
        f"forced-dense attention lost the needle at depth {depth}: the defect "
        f"is below the indexer\n  got: {text[:120]!r}")


def _oracle_runs():
    spec = json.loads((_ORACLE_DIR / "reference.json").read_text())
    return [pytest.param(r, id=r["name"]) for r in spec["runs"]]


@pytest.mark.parametrize("run", _oracle_runs())
def test_recorded_oracle_parity(run, hy4_model):
    # Greedy text parity against the checked-in llama.cpp continuation. The
    # cross_boundary case is the gate the live sweep cannot reach: its
    # reference crosses the DSA top_k, so a broken shared-selection chain
    # shows up here and nowhere else without a 22-minute reference run.
    _, model, _, tok = hy4_model

    prompt = (_ORACLE_DIR / run["prompt"]).read_text()
    ids = tok.encode(prompt, add_special_tokens=False)
    if tok.bos_token_id is not None and len(ids) + 1 == run["prompt_tokens"]:
        ids = [tok.bos_token_id] + ids
    assert len(ids) == run["prompt_tokens"], (
        f"tokenized to {len(ids)} tokens, the reference saw "
        f"{run['prompt_tokens']}: pretokenizer drift, not an attention bug")

    _, text = _greedy_text(model, tok, ids, run["n_predict"])
    ours, theirs = text.strip(), run["continuation"].strip()
    common = os.path.commonprefix([ours, theirs])
    # Same rule as the live sweep: a shared prefix carrying the needle is
    # agreement, because past the cloze answer the distribution flattens and
    # two correct engines legally tie-flip.
    assert _NEEDLE in common or len(common) >= min(32, len(theirs)), (
        f"{run['name']}: diverged from the recorded reference\n"
        f"  ours  : {ours[:100]!r}\n  oracle: {theirs[:100]!r}")


def test_tiled_prefill_agrees_with_the_direct_path(hy4_model, monkeypatch):
    # The tiled online softmax is exact by construction; this checks the
    # construction on the real weights. Force the tiling on for a prompt the
    # thresholds would otherwise send down the direct path.
    import gmlx.models.hy_v4.model as hy_v4_model

    _, model, _, tok = hy4_model
    _, ids = _build_prompt(tok, BELOW_BOUNDARY)

    direct_ids, direct_text = _greedy_text(model, tok, ids, 16)

    monkeypatch.setattr(hy_v4_model, "_STREAM_MIN_KEYS", 256)
    monkeypatch.setattr(hy_v4_model, "_STREAM_Q", 128)
    tiled_ids, tiled_text = _greedy_text(model, tok, ids, 16)

    assert direct_ids == tiled_ids, (
        "the tiled prefill is an exact rewrite of the direct path\n"
        f"  direct: {direct_text[:80]!r}\n  tiled : {tiled_text[:80]!r}")
