"""DeepSeek-V4.1-Flash-Vision container: block geometry, the embedding
splice onto the sentinel ids, and the image mask the text tower derives
from them. CPU-only, tiny synthetic weights."""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from gmlx.models.deepseek_v4.vision import VisionConfig
from gmlx.models.deepseek_v41 import image_block as ib
from gmlx.models.deepseek_v41.vlm_model import Model as VLModel
from gmlx.models.deepseek_v41.vlm_model import ModelConfig

from test_deepseek_v41_model import _args, _randomized


def _container(**over):
    over.setdefault("vision_router_bias", True)
    tc = _args(**over)
    vc = VisionConfig(depth=1, hidden_size=16, num_heads=2,
                      intermediate_size=16, patch_size=2, in_channels=3,
                      out_hidden_size=tc.hidden_size, downsample_ratio=3,
                      max_wh_ratio=None, max_n_token=1024)
    model = VLModel(ModelConfig(text_config=tc, vision_config=vc,
                                vocab_size=tc.vocab_size))
    mx.eval(model.parameters())
    return _randomized(model), tc


def _prompt(vocab, n_h, n_w, lead=5, tail=4, seed=0):
    """Text, one image block, text. Returns the id list."""
    rng = np.random.default_rng(seed)
    types, _ = ib.build_image_block(n_h, n_w)
    return ([int(t) for t in rng.integers(3, vocab, lead)]
            + [vocab + int(t) for t in types]
            + [int(t) for t in rng.integers(3, vocab, tail)])


def test_media_ids_cover_every_block_type():
    model, tc = _container()
    vocab = tc.vocab_size
    assert model.config.media_token_ids == [vocab + t for t in range(4)]
    assert model.language_model.config.media_token_ids == \
        model.config.media_token_ids


def test_embeddings_splice_rows_sentinels_and_text():
    model, tc = _container()
    vocab = tc.vocab_size
    n_vit, n_h, n_w = 6, 2, 2            # 6x6 patches -> 2x2 aligner grid
    ids = _prompt(vocab, n_h, n_w)
    pv = mx.random.normal((n_vit * n_vit, 3 * 2 * 2))
    meta = [[n_vit, n_vit, n_h, n_w]]
    emb = model.get_input_embeddings(
        mx.array([ids]), pv, image_meta=meta).inputs_embeds
    mx.eval(emb)
    assert emb.shape == (1, len(ids), tc.hidden_size)
    assert np.isfinite(np.array(emb)).all()

    for t, vec in ((ib.IMAGE_START, model.image_start),
                   (ib.IMAGE_NEW_LINE, model.image_newline),
                   (ib.IMAGE_END, model.image_end)):
        pos = [i for i, x in enumerate(ids) if x == vocab + t]
        assert pos
        assert np.allclose(np.array(emb[0, pos]), np.array(vec)[None], atol=1e-6)

    # aligner rows land on the IMAGE slots in reading order
    rows = model.vision_tower(pv, n_vit, n_vit)
    img_pos = [i for i, x in enumerate(ids) if x == vocab + ib.IMAGE]
    assert len(img_pos) == n_h * n_w
    assert np.allclose(np.array(emb[0, img_pos]), np.array(rows), atol=1e-5)

    txt = [i for i, x in enumerate(ids) if x < vocab]
    ref = model.language_model.model.embed_tokens(
        mx.array([ids[i] for i in txt]))
    assert np.allclose(np.array(emb[0, txt]), np.array(ref), atol=1e-6)


def test_row_count_and_missing_pixels_are_errors():
    model, tc = _container()
    vocab = tc.vocab_size
    n_vit, n_h, n_w = 6, 2, 2
    ids = _prompt(vocab, n_h, n_w)
    pv = mx.random.normal((n_vit * n_vit, 3 * 2 * 2))
    meta = [[n_vit, n_vit, n_h, n_w]]
    with pytest.raises(ValueError, match="IMAGE slots"):
        model.get_input_embeddings(
            mx.array([ids + [vocab + ib.IMAGE]]), pv, image_meta=meta)
    with pytest.raises(ValueError, match="pixel_values"):
        model.get_input_embeddings(mx.array([ids]), None)
    with pytest.raises(ValueError, match="image_meta"):
        model.get_input_embeddings(mx.array([ids]), pv)
    plain = model.get_input_embeddings(
        mx.array([ids[:4]]), None).inputs_embeds
    assert plain.shape == (1, 4, tc.hidden_size)


def test_forward_runs_and_the_text_tower_sees_the_image_mask(monkeypatch):
    model, tc = _container()
    vocab = tc.vocab_size
    n_vit, n_h, n_w = 6, 2, 2
    ids = _prompt(vocab, n_h, n_w)
    pv = mx.random.normal((n_vit * n_vit, 3 * 2 * 2))
    meta = [[n_vit, n_vit, n_h, n_w]]

    seen = {}
    inner = model.language_model.model
    orig = type(inner).__call__

    def spy(self, inputs, cache=None, **kw):
        seen["mask"] = kw.get("image_mask")
        seen["ids"] = inputs
        return orig(self, inputs, cache, **kw)

    monkeypatch.setattr(type(inner), "__call__", spy)
    out = model(mx.array([ids]), pv, cache=model.make_cache(),
                image_meta=meta)
    logits = getattr(out, "logits", out)
    mx.eval(logits)
    assert logits.shape == (1, len(ids), vocab)
    assert np.isfinite(np.array(logits)).all()

    # every block position is an image token, and only those
    mask = np.array(seen["mask"])[0]
    want = np.array([x >= vocab for x in ids])
    assert np.array_equal(mask, want)
    assert mask.sum() == ib.num_image_tokens(n_h, n_w)
    assert int(np.array(seen["ids"]).max()) >= vocab   # clamped inside


def test_sentinel_ids_are_clamped_before_any_gather():
    model, tc = _container()
    vocab = tc.vocab_size
    n_vit, n_h, n_w = 6, 2, 2
    ids = _prompt(vocab, n_h, n_w)
    pv = mx.random.normal((n_vit * n_vit, 3 * 2 * 2))
    raw = mx.array([ids])
    emb = model.get_input_embeddings(
        raw, pv, image_meta=[[n_vit, n_vit, n_h, n_w]]).inputs_embeds
    mask = raw >= vocab
    inner = model.language_model.model
    a = inner(raw, model.make_cache(), input_embeddings=emb, image_mask=mask)
    clamped = mx.where(mask, mx.zeros_like(raw), raw)
    b = inner(clamped, model.make_cache(), input_embeddings=emb,
              image_mask=mask)
    mx.eval(a, b)
    assert np.allclose(np.array(a), np.array(b), atol=1e-6)


def test_decode_tokens_take_the_plain_text_path(monkeypatch):
    model, tc = _container()
    seen = {}
    inner = model.language_model.model
    orig = type(inner).__call__

    def spy(self, inputs, cache=None, **kw):
        seen["mask"] = kw.get("image_mask")
        return orig(self, inputs, cache, **kw)

    monkeypatch.setattr(type(inner), "__call__", spy)
    cache = model.make_cache()
    out = model(mx.array([[5, 6, 7]]), None, cache=cache)
    mx.eval(out.logits)
    assert seen["mask"] is None              # a text chunk carries no block
    out = model(mx.array([[8]]), None, cache=cache)
    mx.eval(out.logits)
    assert seen["mask"] is None              # a single decode id never can
