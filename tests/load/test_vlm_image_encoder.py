"""A served chat model as the text encoder of an image model. A chat
model whose entry has its mmproj is served as a VLM, and an image request
borrows it through _borrow_for and api._borrowed, as it borrows a
text-only one. The tiny random Qwen3-VL pair of test_qwen3vl and a tiny
Qwen3.5 VLM, no files."""

from __future__ import annotations

import dataclasses
import importlib
import threading
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

pytest.importorskip("gmlx._vendor.mlx_vlm")

from gmlx.image import text_encoder as te  # noqa: E402
from gmlx.serve.patches import images  # noqa: E402
from test_qwen3vl import (  # noqa: E402
    _MM_META, _TOKENS, _mm_arrays, _processor, _text_model, _tiny_vlm)


def _tiny(family, text, layers=2):
    """The family's encoder spec at the size of a tiny model, with the
    digests of ``text``'s norms as the identity, pad rows of a token in
    the tiny vocab, and no F32 norms, which need the GGUF."""
    spec = te.encoder_spec(family)
    taps = (layers,) if len(spec.tap_layers) == 1 else (1, layers)
    inner = te._inner(text)
    identity = tuple(
        te.IdentityTensor(name, path, te._bf16_digest(w))
        for name, path, w in (
            ("blk.1.attn_norm.weight", "model.layers.1.input_layernorm.weight",
             inner.layers[1].input_layernorm.weight),
            ("output_norm.weight", "model.norm.weight", inner.norm.weight)))
    pads = dict(pad_id=0, pad_multiple=8) if spec.pad_multiple else {}
    return dataclasses.replace(spec, hidden=64, layers=layers, tap_layers=taps,
                               identity=identity, vision_identity=(),
                               fp32_norms=False, **pads)


def _tower_identity(vlm, sha256=None):
    """A vision identity row of the tiny tower's first norm, with its own
    digest or ``sha256``."""
    w = vlm.vision_tower.blocks[0].norm1.weight
    return (te.IdentityTensor("v.blk.0.ln1.weight",
                              "vision_tower.blocks.0.norm1.weight",
                              sha256 or te._bf16_digest(w)),)


class _Tokenizer:
    """One id inside the tiny vocab for each character."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 28 for c in text]


def _borrow(monkeypatch, model, spec, fn):
    """``fn(encoder)`` run as serve runs an image request whose encoder is
    a resident chat model, with ``model`` as the model its engine serves."""
    import gmlx.serve.engine_jobs as engine_jobs

    app_mod = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")
    runtime_mod = importlib.import_module(
        "gmlx._vendor.mlx_vlm.server.runtime")
    rg = SimpleNamespace(model=model, tokenizer=_Tokenizer(),
                         processor=_processor("qwen3_vl"))
    monkeypatch.setattr(app_mod, "get_cached_model", lambda *a, **k: None)
    monkeypatch.setattr(runtime_mod, "runtime",
                        SimpleNamespace(response_generator=rg))

    def run_on_engine(engine_rg, job, **kw):
        steps = job(engine_rg, lambda: False)
        try:
            next(steps)
        except StopIteration as done:
            return done.value
        raise AssertionError("the encode took more than one step")

    monkeypatch.setattr(engine_jobs, "run_on_engine", run_on_engine)
    borrow, _hold = images._borrow_for(SimpleNamespace(encoder_model="chat"))
    pipe = SimpleNamespace(family=SimpleNamespace(encoder=lambda: spec),
                           encoder=None)
    return borrow(pipe, lambda: fn(pipe.encoder), threading.Event(), None)


def _encodes_as(monkeypatch, served, text, spec, prompts=("a red fox",)):
    """Borrow ``served`` for ``spec`` and check that it encodes each of
    ``prompts`` as ``text``, the text class of its weights, does."""
    model, owned, got = _borrow(
        monkeypatch, served, spec,
        lambda enc: (enc.model, enc.owned, [enc.encode(p) for p in prompts]))
    assert not owned
    alone = te.EncoderHandle(text, _Tokenizer(), spec, owned=True)
    for p, tap in zip(prompts, got):
        want = alone.encode(p)
        assert tap.pads == want.pads
        assert tap.features.shape == want.features.shape
        assert float(mx.abs(tap.features - want.features).max()) < 1e-4
    return model, got


@pytest.mark.parametrize("family", ["qwen_image21", "krea2", "flux3_klein"])
def test_serve_borrows_a_vlm_chat_model_as_the_encoder(monkeypatch, family):
    """Qwen-Image 2.1 conditions on Qwen3-VL-8B with its tower, and Krea 2
    and FLUX.3 klein on Qwen3-VL-4B read as text, FLUX.3 klein with pad
    rows. The encoder is the resident VLM, and it encodes a prompt as the
    text tower of its weights does on its own."""
    vlm = _tiny_vlm()
    text = _text_model(vlm)
    spec = _tiny(family, text)
    ropes = [b.self_attn.rotary_emb for b in vlm.language_model.model.layers]
    model, got = _encodes_as(monkeypatch, vlm, text, spec,
                             ("a red fox", "abc", "a fox in the snow"))
    assert model is vlm
    assert bool(spec.pad_multiple) == all(t.pads for t in got)
    assert all(a is b for a, b in zip(
        ropes, [b.self_attn.rotary_emb
                for b in vlm.language_model.model.layers]))


def test_a_pad_row_of_a_vlm_turns_at_the_pad_position_on_every_axis():
    """FLUX.3 klein places its pad row at rotary position 1. Through the
    three axes of a VLM's rotary embedding, the pad row turns as a token
    at position 1 on each axis does, and the real tokens as before."""
    vlm = _tiny_vlm()
    attn = vlm.language_model.model.layers[0].self_attn
    emb, n = attn.rotary_emb, 5
    mx.random.seed(4)
    q = mx.random.normal((1, attn.n_heads, n + 1, attn.head_dim))
    k = mx.random.normal((1, attn.n_kv_heads, n + 1, attn.head_dim))

    def axes(*pos):
        return mx.tile(mx.array(pos)[None, None], (3, 1, 1))

    with te.pad_rope(vlm, n, 1):
        got = attn.rotary_emb.apply_rotary(q, k, axes(*range(n + 1)))
    want = emb.apply_rotary(q, k, axes(*range(n), 1))
    assert attn.rotary_emb is emb
    assert all(bool(mx.array_equal(g, w)) for g, w in zip(got, want))


@pytest.mark.parametrize("family", ["flux2_klein", "qwen_image21"])
def test_serve_borrows_a_text_chat_model_as_the_encoder(monkeypatch, family):
    """FLUX.2 klein on a text-only Qwen3 chat model, and Qwen-Image 2.1 on
    a Qwen3-VL one served without its mmproj, which serve holds in a
    VLM-shaped wrapper: the encoder is the text model it wraps. With no
    tower, Qwen-Image's tower rows are not read, and a generation
    encodes."""
    vlm = _tiny_vlm()
    text = _text_model(vlm)
    spec = _tiny(family, text)
    if family == "qwen_image21":
        spec = dataclasses.replace(spec, vision_identity=_tower_identity(vlm))
    wrapper = SimpleNamespace(_is_text_model=True,
                              language_model=SimpleNamespace(_model=text))
    model, _got = _encodes_as(monkeypatch, wrapper, text, spec)
    assert model is text


def _tiny_qwen35_vlm():
    """A tiny Qwen3.5 VLM, the Qwen3-VL tower remapped and the text half
    random, the text class with its language model's weights, and the
    layer count."""
    import gmlx.load.vlm as gvlm
    from gmlx.load.config_synth import synthesize_config
    from gmlx.load.loader import build_model
    from test_qwen35_rope_scaling import _SHAPES, _meta

    meta = _meta("qwen35")
    meta["tokenizer.ggml.tokens"] = list(_TOKENS)
    vlm, _ = gvlm.build_vlm_model(
        gvlm.synthesize_vlm_config("qwen3_5", meta, _SHAPES, _MM_META))
    # The Qwen3.5 tower has no deepstack mergers, so it skips theirs.
    vision, _skipped, _ = gvlm.remap_vision_arrays(
        _mm_arrays(), "qwen3_5", mm_meta=_MM_META)
    mx.random.seed(3)
    weights = dict(vision)
    for k, v in tree_flatten(vlm.language_model.parameters()):
        weights[f"language_model.{k}"] = mx.random.normal(v.shape) * 0.05
    vlm.load_weights(list(weights.items()), strict=True)
    text, _ = build_model(synthesize_config(meta, tensor_shapes=_SHAPES))
    text.load_weights([(f"language_model.{k}", v) for k, v in tree_flatten(
        vlm.language_model.parameters())], strict=True)
    mx.eval(vlm.parameters(), text.parameters())
    return vlm, text, meta["qwen35.block_count"]


def test_serve_borrows_a_qwen35_vlm_chat_model_as_the_moss_image_encoder(
        monkeypatch):
    """MOSS-Image reads Qwen3.5-9B after its final norm. A Qwen3.5 chat
    model served with its mmproj encodes as its text class does."""
    vlm, text, layers = _tiny_qwen35_vlm()
    model, _got = _encodes_as(monkeypatch, vlm, text,
                              _tiny("moss_image", text, layers))
    assert model is vlm


def _edit_reference(enc):
    """The grid and the tower output shape and deepstack count of a
    Qwen-Image 2.1 edit reference, read through ``enc``."""
    from gmlx.image.conditioning import Reference
    from gmlx.image.families.qwen_image21 import edit

    pixels, grid = edit._pixels(
        enc, Reference(np.full((256, 256, 4), 200, np.uint8)))
    main, deep = enc.model.vision_tower(pixels, mx.array([grid]))
    mx.eval(main, *deep)
    return grid, main.shape, len(deep)


def test_a_borrowed_vlm_encoder_reads_an_edit_reference(monkeypatch):
    """A Qwen-Image 2.1 edit runs its reference image through the image
    processor and the vision tower of the served chat model, whose norms
    are the spec's."""
    vlm = _tiny_vlm()
    spec = dataclasses.replace(_tiny("qwen_image21", _text_model(vlm)),
                               vision_identity=_tower_identity(vlm))
    grid, shape, deep = _borrow(monkeypatch, vlm, spec, _edit_reference)
    assert grid == [1, 16, 16]
    assert shape == (64, 64) and deep == 2


def test_a_borrowed_vlm_encoder_with_another_tower_is_refused(monkeypatch):
    """A chat entry whose mmproj is another projector of the same width:
    the text weights are the spec's and the tower's norms are not, and
    the borrow says so before an edit reads a reference with it."""
    vlm = _tiny_vlm()
    spec = dataclasses.replace(_tiny("qwen_image21", _text_model(vlm)),
                               vision_identity=_tower_identity(vlm, "0" * 64),
                               vision_identity_source="the reference mmproj")
    with pytest.raises(te.EncoderMismatchError,
                       match="v.blk.0.ln1.weight differ from the reference"):
        _borrow(monkeypatch, vlm, spec, _edit_reference)
