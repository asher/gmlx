"""Chunked prefill of gemma-4 image prompts under bidirectional image
attention.

Every prefill loop chunks an image or video prompt with its boundaries at
block edges: the stock serve prompt step (one row, and a left-padded
batch whose rows' blocks overlap), the MTP target's prompt step, and the
fixed-step loop behind run and chat. A block longer than the step
prefills whole in one chunk. Each chunk's sliding mask carries the block
overlay on its own columns, so the logits match transformers run on the
whole prompt, and the sliding caches stay within the window plus one
chunk. A left-padded row keeps its first real key when a one-column chunk
past a full window comes before a block's chunk. Runs on the CPU stream.
"""

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_vlm.generate import ar, dispatch
from mlx_vlm.models.gemma4 import language as g4

import gmlx.load.vlm as gvlm
import gmlx.spec.engine as engine
import gmlx.spec.mtp_prefill as mtp_prefill
from gmlx.gen import media_spans

from test_gemma4_image_masks import (
    _N,
    _TOL,
    _embeds,
    _models,
    _reference,
    _text,
    _types,
)

# A window narrower than both blocks, and a step shorter than the first.
_W, _STEP = 8, 8
_BLOCKS = [(6, 20), (24, 34)]


class _Drafter:
    """No teacher-force or seed hooks: the MTP step keeps the last chunk."""


@pytest.fixture
def chunked(monkeypatch):
    monkeypatch.setenv("PREFILL_STEP_SIZE", str(_STEP))
    monkeypatch.setenv("GMLX_PREFILL_MIN_TAIL", "0")
    monkeypatch.setenv("GMLX_CB_PHASE", "0")
    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    monkeypatch.setattr(engine, "_SPEC_APC_DISABLED", True)
    monkeypatch.setattr(mtp_prefill, "_SPEC_APC_DISABLED", True)
    # Serve's order: the span wrapper goes outside the MTP body.
    mtp_prefill.install_full_prompt_mtp_prefill()
    assert media_spans.install_span_aware_prompt_step()


def _serve(m, rows, embeds, types, mtp, on_chunk=None):
    """The widths of the chunks a served batch runs, then the last logits
    of one forward over the remaining columns, as generate does."""
    kw = dict(model=m, uids=list(range(len(rows))), input_ids=rows,
              max_tokens=[4] * len(rows), inputs_embeds=embeds,
              prompt_kwargs={"mm_token_type_ids": types})
    if mtp:
        # Serve-shaped: stock nulls the step for a drafter batch.
        kw.update(prefill_step_size=None, draft_model=_Drafter(),
                  draft_kind="mtp")
    else:
        kw.update(prefill_step_size=_STEP)
    batch = ar.PromptProcessingBatch(**kw)
    chunks = []
    while batch.needs_processing():
        n = batch.prompt_step()
        if n <= 0:
            break
        chunks.append(n)
        if on_chunk is not None:
            on_chunk(batch)
    out = m(None, inputs_embeds=batch._inputs_embeds, cache=batch.prompt_cache,
            mm_token_type_ids=batch._prompt_kwargs["mm_token_type_ids"])
    return chunks, out.logits[:, -1, :].astype(mx.float32)


def _cuts(chunks, blocks):
    """Chunk boundaries that fall strictly inside a block."""
    bounds = np.cumsum(chunks).tolist()
    return [p for p in bounds for a, b in blocks if a < p < b]


def _one_pass(m, embeds, types):
    out = m(None, inputs_embeds=embeds, cache=m.make_cache(),
            mm_token_type_ids=types)
    return out.logits[:, -1, :].astype(mx.float32)


@pytest.mark.parametrize("kind", [1, 2], ids=["image", "video"])
@pytest.mark.parametrize("path", ["plain", "mtp"])
@pytest.mark.parametrize("model_type", ["gemma4", "gemma4_unified"])
def test_chunked_prefill_matches_transformers(model_type, path, kind,
                                              chunked):
    hf_mod, tc, hf, lm, target = _models(model_type, _W)
    m = lm if path == "plain" else target
    embeds = _embeds()
    types = _types(_BLOCKS) * kind
    t = mx.array(types.astype(np.int32))
    with mx.stream(mx.cpu):
        chunks, got = _serve(m, [list(range(1, _N + 1))], mx.array(embeds),
                             t, path == "mtp")
        one = _one_pass(m, mx.array(embeds), t)
        got, one = np.array(got[0]), np.array(one[0])
    assert chunks == [6, 14, 4, 10, 5]
    assert _cuts(chunks, _BLOCKS) == []
    assert np.abs(got - one).max() <= 1e-4
    ref = _reference(hf_mod, tc, hf, embeds, types)[-1]
    assert np.abs(got - ref).max() <= _TOL


@pytest.mark.parametrize("path", ["plain", "mtp"])
def test_a_long_image_prompt_chunks_with_bounded_sliding_caches(path,
                                                                chunked):
    n, block = 400, (100, 107)
    hf_mod, tc, hf, lm, target = _models("gemma4", _W)
    m = lm if path == "plain" else target
    sliding = [i for i, lyr in enumerate(m.model.layers)
               if lyr.layer_type == "sliding_attention"]
    held = []

    def on_chunk(batch):
        held.append(max(int(batch.prompt_cache[i].keys.shape[2])
                        for i in sliding))

    embeds, types = _embeds(n), _types([block], n)
    with mx.stream(mx.cpu):
        chunks, got = _serve(m, [list(range(1, n + 1))], mx.array(embeds),
                             mx.array(types.astype(np.int32)),
                             path == "mtp", on_chunk)
        got = np.array(got[0])
    assert len(chunks) > 1 and sum(chunks) == n - 1
    assert _cuts(chunks, [block]) == []
    assert max(held) <= _W + _STEP
    ref = _reference(hf_mod, tc, hf, embeds, types)[-1]
    assert np.abs(got - ref).max() <= _TOL


@pytest.mark.parametrize("path", ["plain", "mtp"])
@pytest.mark.parametrize("model_type", ["gemma4", "gemma4_unified"])
def test_a_left_padded_image_batch_chunks_and_matches_each_row(
        model_type, path, chunked):
    """Two image rows, the shorter one left-padded so that its block
    overlaps both blocks of the longer row. The chunks keep the merged
    span whole, and each row matches transformers run on that row alone."""
    hf_mod, tc, hf, lm, target = _models(model_type, _W)
    m = lm if path == "plain" else target
    short, short_block = 30, (4, 16)
    pad = _N - short
    long_e, short_e = _embeds(), _embeds(short, seed=8)
    long_t, short_t = _types(_BLOCKS), _types([short_block], short)
    embeds = np.concatenate([long_e, np.concatenate(
        [np.zeros((1, pad, long_e.shape[2]), np.float32), short_e], 1)], 0)
    types = np.concatenate([long_t, np.concatenate(
        [np.zeros((1, pad), np.int64), short_t], 1)], 0).astype(np.int32)
    rows = [list(range(1, _N + 1)), list(range(1, short + 1))]
    with mx.stream(mx.cpu):
        chunks, got = _serve(m, rows, mx.array(embeds), mx.array(types),
                             path == "mtp")
        got = np.array(got)
    assert chunks == [6, 28, 5]
    shifted = (short_block[0] + pad, short_block[1] + pad)
    assert _cuts(chunks, _BLOCKS + [shifted]) == []
    for row, (e, t) in enumerate(((long_e, long_t), (short_e, short_t))):
        ref = _reference(hf_mod, tc, hf, e, t)[-1]
        assert np.abs(got[row] - ref).max() <= _TOL, row


@pytest.mark.parametrize("path", ["plain", "mtp"])
@pytest.mark.parametrize("model_type", ["gemma4", "gemma4_unified"])
def test_a_block_one_column_past_a_chunk_edge_keeps_the_padded_row(
        model_type, path, chunked):
    """The longer row's block starts one column past a chunk edge after
    the sliding window has filled, so a one-column chunk runs and the
    block's chunk follows it. The left-padded row's first real key is
    still inside the window, and each row matches transformers run on
    that row alone."""
    hf_mod, tc, hf, lm, target = _models(model_type, _W)
    m = lm if path == "plain" else target
    long_block, short, short_block = (17, 30), 25, (3, 8)
    pad = _N - short
    long_e, short_e = _embeds(), _embeds(short, seed=8)
    long_t, short_t = _types([long_block]), _types([short_block], short)
    embeds = np.concatenate([long_e, np.concatenate(
        [np.zeros((1, pad, long_e.shape[2]), np.float32), short_e], 1)], 0)
    types = np.concatenate([long_t, np.concatenate(
        [np.zeros((1, pad), np.int64), short_t], 1)], 0).astype(np.int32)
    rows = [list(range(1, _N + 1)), list(range(1, short + 1))]
    with mx.stream(mx.cpu):
        chunks, got = _serve(m, rows, mx.array(embeds), mx.array(types),
                             path == "mtp")
        got = np.array(got)
    assert chunks == ([8, 8, 1, 13, 8, 1] if path == "plain"
                      else [16, 1, 13, 8, 1])
    for row, (e, t) in enumerate(((long_e, long_t), (short_e, short_t))):
        ref = _reference(hf_mod, tc, hf, e, t)[-1]
        assert np.abs(got[row] - ref).max() <= _TOL, row


def _container(model_type):
    """The container on the weights of ``_models``' language model."""
    _hf_mod, _tc, _hf, lm, _target = _models(model_type, _W)
    text_type = ("gemma4_unified_text" if model_type == "gemma4_unified"
                 else "gemma4_text")
    model, _ = gvlm.build_vlm_model({
        "model_type": model_type, "vision_config": {},
        "text_config": {**_text(_W), "model_type": text_type,
                        "rope_parameters": _tc.rope_parameters,
                        "use_double_wide_mlp": False}})
    model.language_model.load_weights(list(tree_flatten(lm.parameters())))
    mx.eval(model.parameters())
    return model


@pytest.mark.parametrize("model_type", ["gemma4", "gemma4_unified"])
def test_run_prefill_chunks_at_block_edges(model_type, monkeypatch):
    """mlx-vlm's generate_step, which run and chat drive, cuts its chunks
    at the block edges and gives the first token's log-probabilities that
    one pass gives."""
    assert media_spans.install_block_aware_generate_step()
    monkeypatch.setattr(ar, "generation_stream", mx.cpu)
    model = _container(model_type)
    # No pixels: the token types alone mark the blocks.
    types = _types(_BLOCKS)
    ids = mx.array([list(range(1, _N + 1))])
    widths = []
    build = g4.Gemma4TextModel._make_masks

    def spy(self, h, cache, mm_token_type_ids=None):
        widths.append(int(h.shape[1]))
        return build(self, h, cache, mm_token_type_ids)

    monkeypatch.setattr(g4.Gemma4TextModel, "_make_masks", spy)

    def first(step):
        """The first token's log-probabilities and the prefill forwards'
        widths (the loop runs the next decode forward before it yields)."""
        widths.clear()
        gen = dispatch.generate_step(
            ids, model, None, None, max_tokens=1, prefill_step_size=step,
            mm_token_type_ids=mx.array(types.astype(np.int32)))
        _tok, logprobs = next(gen)
        gen.close()
        prefill = []
        while sum(prefill) < _N:
            prefill.append(widths[len(prefill)])
        return np.array(logprobs.astype(mx.float32)), prefill

    with mx.stream(mx.cpu):
        chunked, chunk_widths = first(_STEP)
        whole, whole_widths = first(None)
    assert whole_widths == [_N]
    assert chunk_widths == [6, 14, 4, 14, 1, 1]
    assert _cuts(chunk_widths, _BLOCKS) == []
    assert np.abs(chunked - whole).max() <= 1e-4
