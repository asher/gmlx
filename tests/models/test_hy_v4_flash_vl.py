"""Hy4-flash vision container on a tiny random config: the (t, h, w) walk
over an image's slots, the 3-axis rope, the embedding splice, and an image
split across prefill chunks. CPU-only."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from gmlx.cache.kv_cache import ArraysCache
from gmlx.models.hy_v4_flash import positions as rp
from gmlx.models.hy_v4_flash.model import Engram, EngramHash, rope_at
from gmlx.models.hy_v4_flash.vision import VisionConfig
from gmlx.models.hy_v4_flash.vlm_model import Model as VLModel
from gmlx.models.hy_v4_flash.vlm_model import ModelConfig

from test_hy_v4_flash_model import VOCAB, _args

PAD = 90            # the image pad token of the tiny vocab
KEEP = 3            # index_kpool - 1
T = rp.TEXT


@pytest.fixture(autouse=True)
def _on_cpu():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


def _container():
    tc = _args()
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
    """Text, one image of rows x cols merged cells, text."""
    rng = np.random.default_rng(seed)
    return ([int(t) for t in rng.integers(3, PAD, lead)]
            + [VOCAB + k for k in rp.image_slot_kinds(rows, cols)]
            + [int(t) for t in rng.integers(3, PAD, tail)])


def test_media_ids_are_the_slot_kinds():
    model = _container()
    assert model.config.media_token_ids == [VOCAB, VOCAB + 1, VOCAB + 2]
    assert model.config.image_token_index == PAD


def test_the_walk_is_the_reference_s_example():
    # A 2 x 2 merged grid from position 0, as the reference documents it.
    kinds = rp.image_slot_kinds(2, 2)
    assert kinds == [rp.PATCH, rp.PATCH, rp.NEWLINE,
                     rp.PATCH, rp.PATCH, rp.LAST]
    pos = rp.plan(None, kinds + [T], 0, 7, KEEP)
    t, h, w = np.array(pos.rows).T.tolist()
    assert t == [0, 0, 0, 0, 0, 0, 3]
    assert h == [0, 0, 0, 1, 1, 1, 3]
    assert w == [0, 1, 2, 0, 1, 2, 3]
    assert pos.start is None
    assert rp.image_advance(2, 2) == 3


def test_text_after_an_image_sits_before_its_slot():
    slot = ArraysCache(2)
    # Five text tokens, then an image of 2 rows of 3 cells and a newline.
    first = rp.plan(slot, [T] * 5 + rp.image_slot_kinds(2, 3), 0, 13, KEEP)
    rows = np.array(first.rows)
    assert rows[:5].tolist() == [[p] * 3 for p in range(5)]
    assert rows[5:].tolist() == [[5, 5 + r, 5 + c]
                                 for r in range(2) for c in range(4)]
    # The image took 8 slots and moved the position on by max(2, 4).
    nxt = rp.plan(slot, None, 13, 1, KEEP)
    assert nxt.start == 9 and nxt.rows is None
    # A pooled key sits at its group's first slot: slot 12 is the image's
    # last, slot 8 its first of the second row.
    assert np.array(nxt.slots(12, 4, 1)).tolist() == [[5, 6, 8]]
    assert np.array(rp.plan(slot, None, 14, 3, KEEP).slots(14, 1, 3)).tolist() \
        == [10, 11, 12]


def test_the_walk_resumes_inside_an_image_and_from_a_restored_cache():
    kinds = [T] * 2 + rp.image_slot_kinds(3, 2) + [T] * 4
    whole = np.array(rp.plan(None, kinds, 0, len(kinds), KEEP).rows)
    for cut in (1, 3, 6, 10, 11, 12):
        slot = ArraysCache(2)
        a = rp.plan(slot, kinds[:cut], 0, cut, KEEP)
        # A snapshot restore keeps the array and drops the host copy.
        delattr(slot, rp._HOST)
        b = rp.plan(slot, kinds[cut:], cut, len(kinds) - cut, KEEP)
        got = [np.array(p.rows) if p.rows is not None else
               np.array([[p.start + i] * 3 for i in range(n)])
               for p, n in ((a, cut), (b, len(kinds) - cut))]
        assert np.array_equal(np.concatenate(got), whole), cut


def test_text_rows_inside_an_image_are_refused():
    slot = ArraysCache(2)
    rp.plan(slot, [T, rp.PATCH, rp.PATCH], 0, 3, KEEP)
    with pytest.raises(ValueError, match="inside an image"):
        rp.plan(slot, None, 3, 1, KEEP)


def test_three_axis_rope_is_the_reference_rule():
    d, base = 64, 1e7
    rng = np.random.default_rng(3)
    x = rng.standard_normal((1, 2, 6, d)).astype(np.float32)
    pos = rng.integers(0, 4000, (6, 3))
    got = np.array(rope_at(mx.array(x), mx.array(pos), base))

    want = np.zeros_like(x)
    for i in range(d // 2):
        axis = 1 if i % 3 == 1 and i < 30 else 2 if i % 3 == 2 and i < 30 else 0
        ang = pos[:, axis] * base ** (-2 * i / d)
        c, s = np.cos(ang), np.sin(ang)
        x1, x2 = x[..., 2 * i], x[..., 2 * i + 1]
        want[..., 2 * i] = x1 * c - x2 * s
        want[..., 2 * i + 1] = x1 * s + x2 * c
    assert np.allclose(got, want, atol=2e-3)
    # 12 pairs read t, 10 h and 10 w.
    assert np.bincount(np.array(rp.pair_axes(32))).tolist() == [12, 10, 10]

    # A row with t = h = w is the plain rope at that position.
    same = mx.array([[7, 7, 7], [8, 8, 8], [9, 9, 9]])
    xs = mx.array(x[:, :, :3])
    plain = nn.RoPE(d, traditional=True, base=base)(xs, offset=7)
    assert mx.allclose(rope_at(xs, same, base), plain, atol=1e-4).item()


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
            slot = lm.model._history_slot(cache)
            if hasattr(slot, rp._HOST):
                delattr(slot, rp._HOST)
        outs.append(out)
        a = b
    return mx.concatenate(outs, axis=1)


@pytest.mark.parametrize("sparse", [True, False])
def test_an_image_split_across_chunks_matches_one_pass(sparse, monkeypatch):
    if not sparse:
        monkeypatch.setenv("GMLX_HY4_SPARSE_DISABLE", "1")
    model = _container()
    rows, cols = 3, 4                               # 15 slots
    ids = _prompt(rows, cols, lead=5, tail=9)
    n_prompt = len(ids) - 6                         # the last 6 are decoded
    mx.random.seed(1)
    pv = mx.random.normal((2 * rows * 2 * cols, 3 * 2 * 2))
    emb = model.get_input_embeddings(
        mx.array([ids]), pv,
        image_patch_grid=[[2 * rows, 2 * cols]]).inputs_embeds
    mx.eval(emb)
    steps = [(i, False) for i in range(n_prompt + 1, len(ids) + 1)]

    one = _run(model, ids, emb, [(n_prompt, True)] + steps)
    # Cuts before the image, inside a row, on a row's newline, on the
    # image's last slot and after it. Pools of 4 slots straddle each.
    cuts = [(3, True), (7, True), (10, True), (11, True), (20, True),
            (22, True), (n_prompt, True)]
    split = _run(model, ids, emb, cuts + steps)
    assert mx.allclose(one, split, atol=1e-4, rtol=1e-4).item()
    restored = _run(model, ids, emb, cuts + steps, drop_host=True)
    assert mx.allclose(one, restored, atol=1e-4, rtol=1e-4).item()


def test_the_slot_positions_reach_the_layers(monkeypatch):
    """GMLX_HY4_IMAGE_POS_1D=1 puts every row at its slot, as llama.cpp
    runs this model. The text before the image is the same either way and
    everything from the image on is not."""
    model = _container()
    ids = _prompt(3, 4, lead=5, tail=9)
    emb = mx.random.normal((1, len(ids), model.config.text_config.hidden_size))
    piece = [(len(ids), True)]
    ours = np.array(_run(model, ids, emb, piece))
    monkeypatch.setenv("GMLX_HY4_IMAGE_POS_1D", "1")
    flat = np.array(_run(model, ids, emb, piece))
    assert np.allclose(ours[0, :5], flat[0, :5], atol=1e-5)
    assert np.abs(ours[0, 20:] - flat[0, 20:]).max() > 1e-2


def test_embeddings_splice_the_tower_rows_onto_the_slots():
    model = _container()
    rows, cols = 2, 3
    ids = _prompt(rows, cols)
    mx.random.seed(2)
    pv = mx.random.normal((2 * rows * 2 * cols, 12))
    grid = [[2 * rows, 2 * cols]]
    emb = model.get_input_embeddings(
        mx.array([ids]), pv, image_patch_grid=grid).inputs_embeds
    assert emb.shape == (1, len(ids), model.config.text_config.hidden_size)

    slots = [i for i, t in enumerate(ids) if t >= VOCAB]
    assert len(slots) == rows * (cols + 1)
    want = model.vision_tower(pv, 2 * rows, 2 * cols)
    assert np.allclose(np.array(emb[0, slots]), np.array(want), atol=1e-5)
    # Each row's last slot is the newline.
    newline = np.array(model.vision_tower.image_newline)
    for i in slots[cols::cols + 1]:
        assert np.allclose(np.array(emb[0, i]), newline, atol=1e-6)

    text = [i for i, t in enumerate(ids) if t < VOCAB]
    ref = model.language_model.model.embed_tokens(
        mx.array([ids[i] for i in text]))
    assert np.allclose(np.array(emb[0, text]), np.array(ref), atol=1e-6)


def test_row_count_and_missing_pixels_are_errors():
    model = _container()
    ids = mx.array([_prompt(2, 3)])
    with pytest.raises(ValueError, match="no pixel_values"):
        model.get_input_embeddings(ids, None)
    with pytest.raises(ValueError, match="image slots"):
        model.get_input_embeddings(
            ids, mx.zeros((4 * 4, 12)), image_patch_grid=[[4, 4]])


def test_an_image_slot_hashes_as_the_pad_token_and_takes_no_write(
        monkeypatch):
    model = _container()
    ids = _prompt(1, 2, lead=3, tail=2)             # 3 text, 3 slots, 2 text
    seen = {}
    hash_call, write_call = EngramHash.__call__, Engram.__call__

    def hashed(self, input_ids, history):
        seen["ids"] = np.array(input_ids).reshape(-1).tolist()
        return hash_call(self, input_ids, history)

    def written(self, h, rows, live):
        seen["live"] = np.array(live).reshape(-1).tolist()
        return write_call(self, h, rows, live)

    monkeypatch.setattr(EngramHash, "__call__", hashed)
    monkeypatch.setattr(Engram, "__call__", written)
    emb = mx.random.normal((1, len(ids), model.config.text_config.hidden_size))
    model.language_model(mx.array([ids]), cache=model.make_cache(),
                         inputs_embeds=emb)
    assert seen["ids"] == ids[:3] + [PAD] * 3 + ids[6:]
    assert seen["live"] == [True] * 3 + [False] * 3 + [True] * 2


def test_the_cache_holds_the_positions_beside_the_token_history():
    model = _container()
    cache = model.make_cache()
    slot = model.language_model.model._history_slot(cache)
    assert len(slot.cache) == 2 and not rp.active(slot)
    ids = _prompt(1, 2, lead=3, tail=2)
    emb = mx.random.normal((1, len(ids), model.config.text_config.hidden_size))
    model.language_model(mx.array([ids]), cache=cache, inputs_embeds=emb)
    assert rp.active(slot)
    # A text model's cache has no place for them.
    from gmlx.models.hy_v4_flash.model import Model as TextModel

    text_cache = TextModel(_args()).make_cache()
    with pytest.raises(ValueError, match="vision model's make_cache"):
        model.language_model(mx.array([ids]), cache=text_cache,
                             inputs_embeds=emb)


# The draft block beside the vision tower.

def _drafter(model):
    from gmlx.models.hy_v4_flash.mtp import (
        HyV4FlashMTPConfig,
        HyV4FlashMTPDrafter,
    )

    drafter = HyV4FlashMTPDrafter(
        HyV4FlashMTPConfig(text_config=model.config.text_config))
    mx.random.seed(3)
    drafter.load_weights(
        [(k, mx.random.normal(v.shape, scale=0.05).astype(v.dtype))
         for k, v in tree_flatten(drafter.parameters())])
    mx.eval(drafter.parameters())
    return drafter


def test_the_container_s_language_model_is_an_mtp_target():
    from gmlx.load.mtp_target import _MTP_TARGET_HOOKS_BY_TYPE, _spec_hook_key

    model = _container()
    lm = model.language_model
    for hook in _MTP_TARGET_HOOKS_BY_TYPE[_spec_hook_key("hy_v4_flash_vl")]:
        assert callable(getattr(lm, hook, None)), hook
    ids = mx.array([[5, 9, 2, 40]])
    out = lm(ids, cache=model.make_cache(), return_hidden=True,
             skip_logits=True)
    assert out.logits is None
    hidden = out.hidden_states[-1]
    assert hidden.shape == (1, 4, lm.args.hidden_size)
    plain = lm(ids, cache=model.make_cache()).logits
    assert mx.allclose(lm.speculative_logits_from_hidden(hidden), plain,
                       atol=1e-5).item()


def test_text_only_greedy_mtp_on_the_container_gives_the_plain_tokens():
    from gmlx.spec.speculative import stream_speculative

    model = _container()
    lm = model.language_model
    mx.random.seed(11)
    prompt = mx.random.randint(0, PAD, (1, 41)).astype(mx.int32)

    cache = model.make_cache()
    logits = lm(prompt, cache=cache).logits
    ref = []
    for _ in range(16):
        ref.append(int(mx.argmax(logits[0, -1]).item()))
        logits = lm(mx.array([[ref[-1]]], dtype=mx.int32), cache=cache).logits

    drafter = _drafter(model)
    drafter.reset(model)
    spec = list(stream_speculative(
        model, drafter, prompt, prompt_cache=model.make_cache(),
        max_tokens=16, sampler=None, draft_block_size=2, prefill_chunk=8))
    assert spec == ref
    assert any(a == 0 for a in drafter.accept_lens)


@pytest.mark.parametrize("accepted", [0, 2])
def test_a_rejected_draft_after_an_image_leaves_the_walk_of_the_kept_rows(
        accepted):
    """A served image request on an MTP build: the verify forward moves
    the position walk on by the whole block, and a rollback takes it back
    to the rows that stay."""
    model = _container()
    lm = model.language_model
    ids = _prompt(3, 4, lead=5, tail=2)
    emb = mx.random.normal((1, len(ids), lm.args.hidden_size))
    mx.random.seed(1)
    block = mx.random.randint(0, PAD, (1, 4))
    nxt = mx.random.randint(0, PAD, (1, 2))
    kept = accepted + 1

    cache_a = model.make_cache()
    lm(mx.array([ids]), cache=cache_a, inputs_embeds=emb)
    _, _, sink = lm.speculative_verify_hidden(block, cache_a)
    lm.rollback_speculative_cache(cache_a, sink, accepted, 4)
    logits_a = lm(nxt, cache=cache_a).logits

    cache_b = model.make_cache()
    lm(mx.array([ids]), cache=cache_b, inputs_embeds=emb)
    lm(block[:, :kept], cache=cache_b)
    logits_b = lm(nxt, cache=cache_b).logits

    assert float(mx.abs(logits_a - logits_b).max()) < 2e-3
    slot_a = lm.model._history_slot(cache_a)
    slot_b = lm.model._history_slot(cache_b)
    assert mx.array_equal(slot_a[1], slot_b[1]).item()


def test_the_draft_block_reads_an_image_slot_as_the_pad_token():
    model = _container()
    drafter = _drafter(model)
    drafter.reset(model)
    hidden = mx.random.normal((1, 5, model.config.text_config.hidden_size))
    states = []
    for ids in ([7, VOCAB, VOCAB + 1, VOCAB + 2, 9], [7, PAD, PAD, PAD, 9]):
        cache = drafter.make_cache()
        drafter.seed_chunk(mx.array([ids]), hidden, cache)
        states.append(cache[0][0].state[0])
    assert mx.array_equal(states[0], states[1]).item()
