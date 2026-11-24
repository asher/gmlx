"""Hy4 vision container on a tiny random config: the NeoX-order rope at
explicit positions, an image split across prefill chunks, and the walk on
layer 0's cache. CPU-only."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from gmlx.models.hy_v4.model import rope_at
from gmlx.models.hy_v4.vlm_model import Model as VLModel
from gmlx.models.hy_v4.vlm_model import ModelConfig
from gmlx.models.hy_v4_flash import positions as rp
from gmlx.models.hy_v4_flash.vision import VisionConfig

from test_hy_v4_model import _args

VOCAB = 32
PAD = 30            # the image pad token of the tiny vocab


@pytest.fixture(autouse=True)
def _on_cpu():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


def _container(index_topk=4):
    # rope 4 dims = 2 pairs: pair 0 reads t, pair 1 reads h.
    tc = _args(rope_traditional=False, rope_sections=[1, 1],
               index_topk=index_topk)
    vc = VisionConfig(depth=1, hidden_size=16, num_heads=2,
                      intermediate_size=24, patch_size=2, in_channels=3,
                      spatial_merge_size=2, merge_hidden_size=20,
                      out_hidden_size=tc.hidden_size, position_grid=8)
    model = VLModel(ModelConfig(text_config=tc, vision_config=vc,
                                vocab_size=VOCAB, image_token_id=PAD))
    mx.random.seed(7)
    model.load_weights(
        [(k, mx.random.normal(v.shape, scale=0.5).astype(v.dtype))
         for k, v in tree_flatten(model.parameters())])
    mx.eval(model.parameters())
    return model


def _prompt(rows, cols, lead=5, tail=9, seed=0):
    rng = np.random.default_rng(seed)
    return ([int(t) for t in rng.integers(3, PAD, lead)]
            + [VOCAB + k for k in rp.image_slot_kinds(rows, cols)]
            + [int(t) for t in rng.integers(3, PAD, tail)])


@pytest.mark.parametrize("traditional", [False, True])
def test_rope_at_consecutive_positions_is_the_plain_rope(traditional):
    d, base = 64, 1e7
    x = mx.random.normal((1, 2, 5, d))
    plain = nn.RoPE(d, traditional=traditional, base=base)(x, offset=11)
    flat = mx.arange(11, 16)
    same = mx.stack([flat, flat, flat], axis=1)
    for at in (flat, same):
        got = rope_at(x, at, base, (12, 10, 10), traditional)
        assert mx.allclose(got, plain, atol=1e-4).item()


def test_image_rows_take_the_scaled_table_of_the_text_rows():
    """A YaRN run option swaps the text rope for a scaled table; rows at
    explicit positions must rotate on the same table and magnitude."""
    from gmlx.load import rope_scaling as rs
    from gmlx.load.rope_swap import scaled_rope

    d, base = 64, 1e7
    sc = rs.RopeScaling(type="yarn", freq_scale=0.25, original_context=4096,
                        ext_factor=1.0, attn_factor=1.0, freq_base=base)
    rope = scaled_rope(d, base, sc)
    assert rope.mscale != 1.0
    x = mx.random.normal((1, 2, 5, d))
    plain = rope(x, offset=11)
    flat = mx.arange(11, 16)
    same = mx.stack([flat, flat, flat], axis=1)
    for at in (flat, same):
        got = rope_at(x, at, base, (12, 10, 10), False, rope)
        assert mx.allclose(got, plain, atol=1e-4).item()


def test_neox_rope_rotates_each_pair_on_its_axis():
    d, base = 8, 1e4
    rng = np.random.default_rng(3)
    x = rng.standard_normal((1, 1, 3, d)).astype(np.float32)
    pos = rng.integers(0, 50, (3, 3))
    got = np.array(rope_at(mx.array(x), mx.array(pos), base, (2, 1, 1),
                           False))
    want = np.zeros_like(x)
    half = d // 2
    for i in range(half):
        axis = 1 if i % 3 == 1 and i < 3 else 2 if i % 3 == 2 and i < 3 else 0
        ang = pos[:, axis] * base ** (-2 * i / d)
        c, s = np.cos(ang), np.sin(ang)
        x1, x2 = x[..., i], x[..., i + half]
        want[..., i] = x1 * c - x2 * s
        want[..., i + half] = x1 * s + x2 * c
    assert np.allclose(got, want, atol=1e-4)


def _run(model, ids, emb, splits, drop_host=False):
    """Prefill pieces with their embeddings, then decode steps on ids."""
    lm = model.language_model
    cache = model.make_cache()
    outs, a = [], 0
    for b, embedded in splits:
        piece = mx.array([ids[a:b]])
        out = lm(piece, cache=cache,
                 inputs_embeds=emb[:, a:b] if embedded else None).logits
        mx.eval(out, [c.state for c in cache])
        if drop_host:
            slot = cache[0].caches[-1]
            if hasattr(slot, rp._HOST):
                delattr(slot, rp._HOST)
        outs.append(out)
        a = b
    return mx.concatenate(outs, axis=1)


@pytest.mark.parametrize("mode", ["sparse", "dense", "no_indexer"])
def test_an_image_split_across_chunks_matches_one_pass(mode, monkeypatch):
    if mode == "dense":
        monkeypatch.setenv("GMLX_HY4_SPARSE_DISABLE", "1")
    model = _container(index_topk=0 if mode == "no_indexer" else 4)
    rows, cols = 3, 4                               # 15 slots
    ids = _prompt(rows, cols, lead=5, tail=9)
    n_prompt = len(ids) - 6
    mx.random.seed(1)
    pv = mx.random.normal((2 * rows * 2 * cols, 3 * 2 * 2))
    emb = model.get_input_embeddings(
        mx.array([ids]), pv,
        image_patch_grid=[[2 * rows, 2 * cols]]).inputs_embeds
    mx.eval(emb)
    steps = [(i, False) for i in range(n_prompt + 1, len(ids) + 1)]

    one = _run(model, ids, emb, [(n_prompt, True)] + steps)
    cuts = [(3, True), (7, True), (10, True), (11, True), (20, True),
            (22, True), (n_prompt, True)]
    split = _run(model, ids, emb, cuts + steps)
    assert mx.allclose(one, split, atol=1e-4, rtol=1e-4).item()
    restored = _run(model, ids, emb, cuts + steps, drop_host=True)
    assert mx.allclose(one, restored, atol=1e-4, rtol=1e-4).item()


def test_image_positions_reach_the_layers():
    """The same embeddings run as text put every row at its slot. The rows
    before the image agree and the rows from the image on do not."""
    model = _container()
    lm = model.language_model
    ids = _prompt(3, 4, lead=5, tail=9)
    emb = mx.random.normal((1, len(ids), lm.config.hidden_size))
    ours = np.array(lm(mx.array([ids]), cache=model.make_cache(),
                       inputs_embeds=emb).logits)
    text = [PAD if t >= VOCAB else t for t in ids]
    flat = np.array(lm(mx.array([text]), cache=model.make_cache(),
                       inputs_embeds=emb).logits)
    assert np.allclose(ours[0, :5], flat[0, :5], atol=1e-5)
    assert np.abs(ours[0, 5:] - flat[0, 5:]).max() > 1e-3


def test_the_indexer_ropes_image_rows_at_their_slots(monkeypatch):
    from gmlx.models.hy_v4.model import HyV4Indexer

    seen = []
    real = HyV4Indexer.__call__

    def spy(self, *a, pos=None, **kw):
        seen.append(pos)
        return real(self, *a, pos=pos, **kw)

    monkeypatch.setattr(HyV4Indexer, "__call__", spy)
    model = _container()
    lm = model.language_model
    ids = _prompt(3, 4, lead=5, tail=9)
    emb = mx.random.normal((1, len(ids), lm.config.hidden_size))
    mx.eval(lm(mx.array([ids]), cache=model.make_cache(),
               inputs_embeds=emb).logits)
    assert seen and all(p is not None and p.rows is not None for p in seen)


def test_a_text_cache_refuses_image_slots():
    from gmlx.models.hy_v4.model import Model as TextModel

    model = _container()
    text_cache = TextModel.make_cache(model.language_model)
    ids = mx.array([_prompt(2, 2)])
    emb = mx.random.normal((1, ids.shape[1], 16))
    with pytest.raises(ValueError, match="vision model's cache"):
        model.language_model(ids, cache=text_cache, inputs_embeds=emb)


def test_container_sanitize_drops_layers_past_the_trunk():
    model = _container()
    n = model.config.text_config.num_hidden_layers
    w = {"vision_tower.image_newline": mx.zeros((16,)),
         "language_model.model.layers.0.self_attn.sinks": mx.zeros((2,)),
         f"language_model.model.layers.{n}.self_attn.sinks": mx.zeros((2,))}
    assert sorted(model.sanitize(w)) == [
        "language_model.model.layers.0.self_attn.sinks",
        "vision_tower.image_newline"]


def test_the_serve_engine_sees_the_image_slot_ids():
    """The engine reads media ids off the language model's config; the
    prefix cache must keep reusable prefixes clear of image slots."""
    from gmlx.cache.apc_media import install_media_token_ids

    install_media_token_ids()
    from gmlx._vendor.mlx_vlm.apc import multimodal_token_ids_from_config

    model = _container()
    ids = multimodal_token_ids_from_config(model.language_model.config)
    assert {VOCAB + k for k in range(rp.N_SLOT_KINDS)} <= set(ids)
