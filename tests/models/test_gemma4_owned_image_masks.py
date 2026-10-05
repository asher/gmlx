"""The owned Gemma 4 text model masks image blocks as transformers does.

transformers builds the masks under bidirectional image attention with
``create_masks_for_vision_model``: full-attention layers stay causal, and
sliding layers take AND(window, OR(causal, block)). llama.cpp's gemma4
graph does the same. mlx-vlm 0.7.3 ORs each block into both layer types
and past the sliding window.

The owned model builds the causal masks, then rebuilds the sliding mask
of a chunk that holds image or video tokens with the block overlay on the
chunk's own columns inside the window. Each chunk after a cached prefix
gets the rows of the one-pass masks, and the model gives transformers'
logits for gemma4 and gemma4_unified text configs on shared weights. Only
the transformers comparison needs torch. The upstream half runs in a child
on the files of tests/pristine.py (gemma4_probes.py). Runs on the CPU
stream, whose float32 matmuls match torch's closely.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from gmlx.models.gemma4 import language as owned

import gemma4_image_case as case
import gemma4_probes as probes

DRIVER = Path(__file__).with_name("gemma4_probes.py")


@pytest.fixture(scope="module")
def stock() -> dict:
    r = subprocess.run([sys.executable, str(DRIVER), "--cpu"],
                       capture_output=True, text=True, env=dict(os.environ),
                       timeout=600)
    assert r.returncode == 0, r.stderr[-4000:]
    return json.loads(r.stdout)


@pytest.mark.parametrize("name", ["text", "wide", "narrow"])
@pytest.mark.parametrize("model_type", ["gemma4", "gemma4_unified"])
def test_logits_match_transformers(model_type, name):
    window, blocks = case.CASES[name]
    hf_mod, tc, hf, lm = case.hf_models(model_type, window)
    e, t = case.embeds(), case.types(blocks)
    ref = case.reference(hf_mod, tc, hf, e, t)
    got = case.one_pass_logits(lm, e, t)
    assert np.abs(got - ref).max() <= case.TOL


# Each image block in one chunk, and the text chunks after each block.
@pytest.mark.parametrize("start,end", [(6, 20), (20, 24), (24, 34), (34, 40)])
@pytest.mark.parametrize("whole_types", [False, True],
                         ids=["chunk_types", "prompt_types"])
@pytest.mark.parametrize("model_type", ["gemma4", "gemma4_unified"])
def test_a_chunk_gets_the_one_pass_rows(model_type, start, end, whole_types):
    """A chunk after a cached prefix, with its edges at block edges, gets
    the rows of the one-pass masks for its queries, over the keys the
    caches hold (the sliding cache keeps the last window - 1 prefix keys).
    Full-attention layers stay causal. Token types for the whole prompt
    are sliced at the cache offset."""
    window, blocks = case.CASES["narrow"]
    lm = case.owned_lm(model_type, window)
    tm = lm.model
    types_np = case.types(blocks)
    types = mx.array(types_np.astype(np.int32))
    embeds = mx.array(case.embeds())
    sliding = case.one_pass_sliding(types_np, window)
    kept = min(window - 1, start)
    with mx.stream(mx.cpu):
        cache = lm.make_cache()
        mx.eval(lm(None, inputs_embeds=embeds[:, :start], cache=cache,
                   mm_token_type_ids=types[:, :start]).logits)
        cache = cache + [None] * (len(tm.layers) - len(cache))
        passed = types if whole_types else types[:, start:end]
        masks = tm._make_masks(embeds[:, start:end], cache, passed)
    for layer, mask in zip(tm.layers, masks):
        if layer.layer_type == "full_attention":
            assert isinstance(mask, str) and mask == "causal"
            continue
        want = sliding[start:end, start - kept:end]
        got = np.array(mx.broadcast_to(mask, (1, 1) + want.shape))[0, 0]
        assert (got == want).all()


def test_the_sliding_layers_share_one_mask():
    """One mask object per layer type: the batched route's pad relay keys
    on that identity."""
    window, blocks = case.CASES["narrow"]
    tm = case.owned_lm("gemma4", window).model
    masks = tm._make_masks(mx.zeros((1, case.N, case.HIDDEN)),
                           [None] * len(tm.layers),
                           mx.array(case.types(blocks).astype(np.int32)))
    for kind in ("sliding_attention", "full_attention"):
        ids = {id(m) for layer, m in zip(tm.layers, masks)
               if layer.layer_type == kind}
        assert len(ids) == 1, kind


def test_owned_masks_keep_the_block_to_sliding_layers_in_the_window():
    assert probes.image_mask_spread(owned) == {
        "full_attention": {"causal": True},
        "sliding_attention": {"past_window": False, "block_ahead": True},
    }


def test_stock_masks_put_the_block_in_every_layer_past_the_window(stock):
    assert stock["image_mask_spread"] == {
        "full_attention": {"causal": False},
        "sliding_attention": {"past_window": True, "block_ahead": True},
    }
