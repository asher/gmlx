"""The owned Gemma 4 models chunk every prompt, images included.

Under bidirectional image attention, stock declines chunked prefill for a
prompt with an image or video and no audio, since its overlay needs the
whole prompt in one pass: the language model's policy says so, and the
gemma4_unified container says so from the token types or the ids, with
two to five host syncs a request. gmlx chunks every prompt: the prefill
loops keep each image or video block inside one chunk, and the owned
``_make_masks`` puts the block on the chunk's sliding mask. The owned
policies keep only the drafter rule and take no host sync.

The stock halves run in a child that applies no loader install
(gemma4_probes.py). The served tests run the owned language model through
the stock serve prompt step and gmlx's MTP prompt step, on the CPU
stream.
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
from gmlx.models.gemma4 import unified as owned_uni

import gemma4_probes as probes
from test_gemma4_image_audio import _served, serve_env  # noqa: F401
from test_gemma4_owned_image_masks import (
    _CASES,
    _N,
    _TOL,
    _embeds,
    _models,
    _reference,
    _types,
)

DRIVER = Path(__file__).with_name("gemma4_probes.py")


@pytest.fixture(scope="module")
def stock() -> dict:
    r = subprocess.run([sys.executable, str(DRIVER), "--cpu"],
                       capture_output=True, text=True, env=dict(os.environ),
                       timeout=600)
    assert r.returncode == 0, r.stderr[-4000:]
    return json.loads(r.stdout)


def test_the_policy_table():
    lm = probes.tiny(owned)
    draft = object()
    assert lm.chunked_prefill_policy(draft_model=None, prefill_kwargs={}) is True
    assert lm.chunked_prefill_policy(
        draft_model=draft, draft_kind="mtp", prefill_kwargs={}) is False
    assert lm.chunked_prefill_policy(
        draft_model=draft, draft_kind="mtp",
        prefill_kwargs={"return_hidden": True, "return_shared_kv": True},
    ) is True
    assert lm.chunked_prefill_policy(
        draft_model=draft, draft_kind="dflash",
        prefill_kwargs={"capture_layer_ids": [1]}) is False


_CHUNKS = {"plain": True, "mtp": True, "syncs": 0}


def test_owned_language_model_chunks_every_prompt():
    assert probes.lm_policy(owned) == {
        name: _CHUNKS for name in probes.POLICY_PROMPTS}


def test_stock_language_model_declines_an_image_without_audio(stock):
    got = stock["lm_policy"]
    assert got["image"] == {"plain": False, "mtp": False, "syncs": 4}
    for name in ("image_audio", "audio", "text"):
        assert got[name] == {"plain": True, "mtp": True, "syncs": 4}, name


def test_owned_unified_container_never_declines():
    assert probes.unified_policy(owned_uni) == {
        name: {"by_types": True, "by_ids": True, "syncs": 0}
        for name in probes.POLICY_PROMPTS}


def test_stock_unified_container_declines_an_image_without_audio(stock):
    """Two syncs from the token types, and two or three from the ids."""
    got = stock["unified_policy"]
    assert got["image"] == {"by_types": False, "by_ids": False, "syncs": 4}
    assert got["image_audio"] == {"by_types": True, "by_ids": True,
                                  "syncs": 4}
    for name in ("audio", "text"):
        assert got[name] == {"by_types": True, "by_ids": True,
                             "syncs": 5}, name


def test_owned_unified_container_keeps_no_chunking_state():
    """Embedding a request leaves no chunking answer on the container or
    on its language model."""
    model = probes.unified(owned_uni)
    ids, types = probes.policy_prompt(True, False, model.config.image_token_id,
                                      model.config.audio_token_id)
    model.get_input_embeddings(input_ids=mx.where(types == 1, 0, ids),
                               mm_token_type_ids=types)
    for obj in (model, model.language_model):
        assert not hasattr(obj, "no_chunked_prefill")


@pytest.mark.parametrize("path", ["plain", "mtp"])
@pytest.mark.parametrize("model_type", ["gemma4", "gemma4_unified"])
def test_a_served_left_padded_batch_matches_transformers(model_type, path,
                                                          serve_env):  # noqa: F811
    """An image row beside a shorter text row, left-padded, served in
    chunks with their boundaries at the image edges: each row matches
    transformers run on that row alone."""
    window, blocks = _CASES["narrow"]
    hf_mod, tc, hf, lm = _models(model_type, window)
    short = 24
    image_e, text_e = _embeds(), _embeds(short, seed=8)
    image_t, text_t = _types(blocks), _types([], short)
    embeds = np.concatenate(
        [image_e, np.concatenate(
            [np.zeros((1, _N - short, image_e.shape[-1]), np.float32),
             text_e], 1)], 0)
    types = np.concatenate([image_t, _types([], _N)], 0).astype(np.int32)
    rows = [list(range(1, _N + 1)), list(range(1, short + 1))]
    with mx.stream(mx.cpu):
        chunks, got = _served(lm, rows, mx.array(embeds), mx.array(types),
                              path == "mtp")
        got = np.array(got)
    assert chunks == [6, 16, 16, 1]
    for row, (e, t) in enumerate(((image_e, image_t), (text_e, text_t))):
        ref = _reference(hf_mod, tc, hf, e, t)[-1]
        assert np.abs(got[row] - ref).max() <= _TOL, row
