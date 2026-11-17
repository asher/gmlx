"""Serve-engine batch correctness on real weights: rows of different lengths
run together through mlx-vlm's batch engine, built as serve builds it, give
each row the tokens it gets alone, except where the row alone saw a near tie.

``tests/gen/test_batch_parity.py`` checks the same rule on mlx-lm's batch
engine, which shares the model code and kernels but not the served engine's
padding, prefill grouping or prompt cache. Here each row also runs alone
through the served engine, so the comparison covers that engine and nothing
else. On a checkpoint-tier model a group of short-lived rows (a client cap
of a few tokens) prefills together, and the same rule holds for it.

``integration``; skips unless ``KQUANT_TEST_GGUF_DIR`` holds a GGUF of one
of the arches below. Each arch loads its smallest GGUF there. Short rows
group only on layouts with recurrent state; on gemma 4 the test checks they
prefill alone.
"""

from __future__ import annotations

import importlib
import os

import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")

import mlx.core as mx  # noqa: E402

pytestmark = pytest.mark.integration

# A dense block-tier control, then the checkpoint tier: gated-delta
# recurrent layers (qwen35) and sliding-window attention (gemma4).
ARCHES = ["qwen3", "qwen35", "gemma4"]

PROMPTS = [
    "Hi.",
    "The capital of France is",
    "In a quiet village nestled between two green hills, there lived an old "
    "clockmaker who",
    "Explain, step by step and in careful detail, how a modern lithium-ion "
    "battery stores and releases energy, beginning from the chemistry of the "
    "anode and cathode and ending with",
]

N_TOKENS = 16
N_SHORT = 4
K = 20
# As in test_batch_parity: a benign flip swaps a near tie within about one
# bf16 logit step at the logit's size; a padding, mask or position bug
# promotes a far-rank token by many nats.
TIE_EPS = 1.3
GREEDY = lambda x: mx.argmax(x, axis=-1)  # noqa: E731


@pytest.fixture(scope="module", params=ARCHES)
def served(request, gguf_index):
    arch = request.param
    paths = gguf_index.get(arch)
    if not paths:
        pytest.skip(f"no {arch!r} GGUF under KQUANT_TEST_GGUF_DIR "
                    f"(have: {sorted(gguf_index)})")
    import gmlx.serve.bridge_vlm as serving
    import gmlx.spec.mtp_prefill as mtp_prefill

    mtp_prefill.install_full_prompt_mtp_prefill()   # the serve installs
    path = min(paths, key=os.path.getsize)
    model, processor, _config = serving.load_serveable_model(path)
    return arch, model, processor


class _Widths:
    """The row count of every prompt batch the engine builds."""

    def __init__(self, monkeypatch, ar):
        self.sizes = []
        init = ar.PromptProcessingBatch.__init__
        sizes = self.sizes

        def spy(batch, *a, **k):
            init(batch, *a, **k)
            sizes.append(len(batch.uids))

        monkeypatch.setattr(ar.PromptProcessingBatch, "__init__", spy)


def _run(model, processor, id_lists, n, short=False):
    """The rows through one served engine with its own empty prompt cache:
    per row, the greedy tokens and each step's top-K logprobs."""
    from gmlx._vendor.mlx_vlm import apc as _apc
    from gmlx.cache.apc_manager import GmlxAPCManager
    from gmlx.serve.row_sampling import RowSettings

    ar = importlib.import_module("gmlx._vendor.mlx_vlm.generate.ar")
    lm = model.language_model if hasattr(model, "language_model") else model
    gen = ar.BatchGenerator(
        model, processor, sampler=GREEDY, top_logprobs_k=K,
        completion_batch_size=8,
        apc_manager=GmlxAPCManager(num_blocks=512, block_size=16))
    salt = _apc.semantic_extra_hash(
        tenant=None, image_hash=0, media={"audio": None, "video": None},
        model=lm, processor=processor)
    kwargs = [{"inputs_embeds": model.get_input_embeddings(
                   input_ids=mx.array([ids], dtype=mx.int32)).inputs_embeds,
               "_apc_semantic_hash": salt} for ids in id_lists]
    rows = [RowSettings(logprobs=True, top_k=K, short_lived=short)
            for _ in id_lists]
    try:
        uids = gen.insert(id_lists, [n] * len(id_lists),
                          prompt_kwargs=kwargs, row_settings=rows)
        out = {u: [] for u in uids}
        while gen.has_work:
            _p, responses = gen.next()
            for r in responses:
                if r.uid in out and r.finish_reason != "stop":
                    out[r.uid].append(
                        (int(r.token), dict(r.top_logprobs or ())))
        ckpt = bool(getattr(gen, "_kq_ckpt_tier", False))
        groups = int(getattr(gen, "_kq_short_group_width", 1) or 1) > 1
    finally:
        gen.close()
    return [out[u] for u in uids], (ckpt, groups)


def _assert_ties_only(arch, label, together, alone):
    for i, (bat, ref) in enumerate(zip(together, alone)):
        bat_toks = [t for t, _ in bat]
        ref_toks = [t for t, _ in ref]
        assert len(bat_toks) == len(ref_toks), (
            f"{arch} {label}: prompt {i} gave {len(bat_toks)} tokens "
            f"together and {len(ref_toks)} alone")
        k = next((j for j, (a, b) in enumerate(zip(ref_toks, bat_toks))
                  if a != b), None)
        if k is None:
            continue
        # Up to k the rows saw the same tokens, so the row alone's logprobs
        # at k are the distribution both runs chose from.
        lps = ref[k][1]
        top = max(lps.values())
        got = lps.get(bat_toks[k])
        rank = sorted(lps.values(), reverse=True).index(got) if got is not None else None
        margin = top - got if got is not None else float("inf")
        assert margin <= TIE_EPS, (
            f"{arch} {label}: prompt {i} diverged at token {k} on a non-tie "
            f"(margin {margin:.4f} nats > {TIE_EPS}; the batched token is "
            f"{'outside the top ' + str(K) if rank is None else 'rank ' + str(rank)} "
            f"of the row alone); alone {ref_toks[k]}, together {bat_toks[k]}")


def test_rows_together_diverge_from_alone_only_at_ties(served, monkeypatch):
    arch, model, processor = served
    ar = importlib.import_module("gmlx._vendor.mlx_vlm.generate.ar")
    widths = _Widths(monkeypatch, ar)
    ids = [list(processor.tokenizer.encode(p)) for p in PROMPTS]
    assert len({len(x) for x in ids}) == len(ids), "prompts must differ in length"
    together, _tier = _run(model, processor, ids, N_TOKENS)
    alone = [_run(model, processor, [x], N_TOKENS)[0][0] for x in ids]
    assert widths.sizes[-len(ids):] == [1] * len(ids)
    _assert_ties_only(arch, "rows together", together, alone)


def test_a_short_lived_group_diverges_from_alone_only_at_ties(served,
                                                              monkeypatch):
    arch, model, processor = served
    ar = importlib.import_module("gmlx._vendor.mlx_vlm.generate.ar")
    widths = _Widths(monkeypatch, ar)
    ids = [list(processor.tokenizer.encode(p)) for p in PROMPTS]
    grouped, (ckpt, groups) = _run(model, processor, ids, N_SHORT,
                                   short=True)
    if not ckpt:
        pytest.skip(f"{arch} is not on the checkpoint tier; its short rows "
                    "admit as before")
    if not groups:
        assert widths.sizes == [1] * len(ids), (
            f"{arch}: short rows on a layout without recurrent state "
            f"prefilled in batches of {widths.sizes}")
        return
    assert widths.sizes[0] == len(ids), (
        f"{arch}: short-lived rows prefilled in batches of {widths.sizes}")
    widths.sizes.clear()
    alone = [_run(model, processor, [x], N_SHORT, short=True)[0][0]
             for x in ids]
    assert widths.sizes == [1] * len(ids)
    _assert_ties_only(arch, "short-lived group", grouped, alone)
