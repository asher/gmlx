"""The MOSS-Image text encoder: the template against the release's chat
template, the spec, and the read after the final norm on a tiny Qwen3.5
model, whose recurrent layers the tap cannot run. The real-file rows
need the unsloth Qwen3.5-9B GGUFs under KQUANT_TEST_GGUF_DIR."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from image_fixtures import FIXTURES, load_json

from gmlx.image import text_encoder as te

_TEMPLATE = "moss_image_chat_template.jinja"
_PROMPTS = ["a red bicycle", "", " ", "{} {0} {{x}}", "line one\nline two",
            "caf\u00e9 \u732b", "  a cat\n\n", "\ta\u3000"]
_FILES = ("Qwen3.5-9B-Q8_0.gguf", "Qwen3.5-9B-Q4_K_M.gguf")


def _render_official(prompt: str, **kw) -> str:
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    tmpl = env.from_string((FIXTURES / _TEMPLATE).read_text())
    return tmpl.render(messages=[{"role": "user", "content": prompt}],
                       add_generation_prompt=True, **kw)


@pytest.mark.parametrize("prompt", _PROMPTS)
def test_the_template_is_the_official_render_with_thinking_on(prompt):
    """The release renders one user turn with the generation prompt and
    enable_thinking True, and tokenizes it with no special tokens."""
    spec = te.encoder_spec("moss_image")
    assert te.render(spec, prompt) == _render_official(
        prompt, enable_thinking=True)
    assert te.render(spec, prompt) != _render_official(
        prompt, enable_thinking=False)


def test_the_template_fixture_is_pinned():
    row = load_json("manifest.json")[_TEMPLATE]
    data = (FIXTURES / _TEMPLATE).read_bytes()
    assert hashlib.sha256(data).hexdigest() == row["sha256"]
    assert (row["repo"], row["file"]) == ("OpenMOSS-Team/MOSS-Image-SFT",
                                         "tokenizer/chat_template.jinja")
    from gmlx.image.registry import get_family

    assert row["revision"] == get_family("moss_image").constants.revision


def test_the_read_and_the_trim_are_in_the_template_id():
    import dataclasses

    spec = te.encoder_spec("moss_image")
    ids = {spec.template_id,
           dataclasses.replace(spec, final_norm=False,
                               tap_layers=(32,)).template_id,
           dataclasses.replace(spec, trim=False).template_id}
    assert len(ids) == 3


@pytest.mark.parametrize("prompt", ["  a red bicycle\n", "\ta red bicycle ",
                                    "a red bicycle\u3000"])
def test_surrounding_whitespace_encodes_as_the_stripped_prompt(prompt):
    """The release's template trims the prompt, so gmlx's ids for a prompt
    with surrounding whitespace are the ids of the stripped prompt."""
    import dataclasses

    spec = te.encoder_spec("moss_image")
    tok = _CharTokenizer()
    want = te.token_ids(spec, tok, "a red bicycle")
    assert te.token_ids(spec, tok, prompt) == want
    untrimmed = dataclasses.replace(spec, trim=False)
    assert te.token_ids(untrimmed, tok, prompt) != want


def test_a_final_norm_spec_takes_no_tap_options():
    spec = te.encoder_spec("moss_image")
    import dataclasses

    with pytest.raises(ValueError, match="after layer 32 alone"):
        dataclasses.replace(spec, tap_layers=(31,))
    for kw in ({"drop": 1}, {"stack": True}, {"pad_multiple": 8},
               {"fp32_qkv": True}, {"activation": "float32"}):
        with pytest.raises(ValueError, match="takes no"):
            dataclasses.replace(spec, **kw)


# The read on a tiny Qwen3.5 model.

_NEEDS_GPU = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="the qwen3_5 GDN forward is Metal-only")


class _CharTokenizer:
    """One id per character, below the tiny vocabulary."""

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return [ord(c) % 120 for c in text]


_TINY = dict(model_type="qwen3_5", hidden_size=64, intermediate_size=128,
             linear_num_value_heads=4, linear_num_key_heads=2,
             linear_key_head_dim=32, linear_value_head_dim=32,
             linear_conv_kernel_dim=4, num_hidden_layers=4,
             num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=128,
             num_key_value_heads=2, max_position_embeddings=2048,
             tie_word_embeddings=True, head_dim=32,
             rope_parameters={"type": "default", "mrope_section": [2, 1, 1],
                              "rope_theta": 100000,
                              "partial_rotary_factor": 0.25},
             full_attention_interval=4, kv_head_layout="tiled")


def _built(m):
    mx.eval(m.parameters())
    m.eval()
    return m


def _tiny_text():
    """The class the text loader builds for a qwen35 GGUF."""
    from gmlx._vendor.mlx_lm.models.qwen3_5 import Model, ModelArgs

    mx.random.seed(5)
    return _built(Model(ModelArgs.from_dict(dict(_TINY))))


def _tiny_multimodal():
    """The language model of a Qwen3.5 load with its vision tower, which a
    resident chat model may be."""
    from gmlx._vendor.mlx_vlm.models.qwen3_5.config import TextConfig
    from gmlx.models.qwen35.owned import OwnedQwen3_5LanguageModel

    top = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2),
                          image_token_id=124, video_token_id=125,
                          vision_start_token_id=126)
    mx.random.seed(5)
    return _built(OwnedQwen3_5LanguageModel(TextConfig(**_TINY), top))


_BUILDS = pytest.mark.parametrize("build", [_tiny_text, _tiny_multimodal],
                                  ids=["text", "multimodal"])


def _spec():
    import dataclasses

    return dataclasses.replace(te.encoder_spec("moss_image"), hidden=64,
                               layers=4, tap_layers=(4,), identity=())


def _logits(m, ids):
    """The chat path's prefill of ``ids`` on a fresh cache."""
    out = m(mx.array([ids]), cache=m.make_cache())
    return getattr(out, "logits", out)


@_NEEDS_GPU
@_BUILDS
def test_the_read_is_the_chat_prefill_before_its_output_head(build):
    """The features, through the tied output head, give the logits of the
    chat path's prefill: the same masks, positions and recurrent state."""
    m = build()
    ids = te.token_ids(_spec(), _CharTokenizer(), "a red bicycle")
    got = te.final_hidden(m, ids)
    assert got.shape == (1, len(ids), 64)
    head = te._inner(m).embed_tokens.as_linear
    assert mx.allclose(head(got), _logits(m, ids), atol=1e-4,
                       rtol=1e-4).item()


@_NEEDS_GPU
@_BUILDS
def test_an_encoding_is_the_final_norm_of_the_last_layer(build, monkeypatch):
    m = build()
    tok = _CharTokenizer()
    spec = _spec()
    enc = te.EncoderHandle(m, tok, spec, owned=False).encode("a red bicycle")
    ids = list(enc.ids)
    assert ids == te.token_ids(spec, tok, "a red bicycle")
    assert enc.features.shape == (len(ids), 64) and enc.pads == 0
    assert enc.template_id == spec.template_id
    assert mx.array_equal(enc.features, te.final_hidden(m, ids)[0]).item()
    # The last layer's output before the norm is not the read.
    monkeypatch.setattr(te._inner(m), "norm", lambda h: h)
    before = te.final_hidden(m, ids)[0]
    assert not mx.allclose(before, enc.features, atol=1e-2).item()


@_NEEDS_GPU
def test_a_read_leaves_a_resident_models_positions_alone():
    """A borrowed multimodal chat model keeps the positions of its last
    call for its next one, and an encoding through it does not touch
    them."""
    m = _tiny_multimodal()
    held = mx.arange(7)[None, None]
    m._position_ids, m._rope_deltas = held, mx.zeros((1, 1))
    deltas = m._rope_deltas
    te.EncoderHandle(m, _CharTokenizer(), _spec(), owned=False).encode("ab")
    assert m._position_ids is held and m._rope_deltas is deltas


# The real files.

@pytest.mark.integration
@pytest.mark.parametrize("name", _FILES)
def test_the_unsloth_file_passes_the_header_and_identity_checks(
        name, gguf_index):
    spec = te.encoder_spec("moss_image")
    found = [p for p in gguf_index.get(spec.arch, []) if Path(p).name == name]
    if not found:
        pytest.skip(f"no {name} under KQUANT_TEST_GGUF_DIR")
    view = te._header_view(found[0])
    te.check_encoder_file(view, spec)
    assert te.file_identity(view, spec) == {
        r.gguf_name: r.sha256 for r in spec.identity}


@_NEEDS_GPU
@_BUILDS
def test_the_identity_reads_the_loaded_norms(build):
    """The identity of a loaded encoder, which a borrowed chat model is
    checked by, reads its norms under either class."""
    import dataclasses
    import hashlib

    import numpy as np

    m = build()
    row = te.IdentityTensor("output_norm.weight", "model.norm.weight", "")
    spec = dataclasses.replace(_spec(), identity=(row,))
    bits = np.array(te._inner(m).norm.weight.astype(mx.bfloat16).view(
        mx.uint16))
    assert te.model_identity(m, spec) == {
        "output_norm.weight": hashlib.sha256(bits.tobytes()).hexdigest()}


# F32 norms.

def test_the_f32_norms_are_moss_images_alone():
    """Only MOSS-Image encodes on the F32 norm weights; every other
    family's encoder, and a chat load, runs on the norms as loaded."""
    from gmlx.image.registry import families

    got = {}
    for fam in families():
        try:
            got[fam.name] = te.encoder_spec(fam).fp32_norms
        except te.ImageModelError:
            continue
    assert {k for k, v in got.items() if v} == {"moss_image"}
    assert len(got) > 1
    import dataclasses

    spec = te.encoder_spec("moss_image")
    assert spec.template_id != dataclasses.replace(
        spec, fp32_norms=False).template_id


def _f32_of(m):
    """F32 weights for every norm of the tiny bf16 model, each off its
    bf16 value by less than half a bf16 step, so it rounds to it."""
    inner = te._inner(m)
    out = {}
    for i, b in enumerate(inner.layers):
        for where in ("input_layernorm", "post_attention_layernorm",
                      "self_attn.q_norm", "self_attn.k_norm",
                      "linear_attn.norm"):
            mod = te._module(b, where)
            if mod is not None:
                w = mod.weight.astype(mx.float32)
                out[f"layers.{i}.{where}"] = w * (1 + 2 ** -11)
    out["norm"] = inner.norm.weight.astype(mx.float32) * (1 + 2 ** -11)
    for where, w in out.items():
        held = te._module(inner, where).weight
        assert mx.array_equal(w.astype(mx.bfloat16), held).item(), where
    return out


def _norm_state(m):
    """Every norm module and its weight, by identity."""
    inner = te._inner(m)
    mods = [mod for _, mod in inner.named_modules()
            if "norm" in type(mod).__name__.lower()]
    return [(id(mod), id(mod.weight)) for mod in mods]


@_NEEDS_GPU
@_BUILDS
def test_an_encode_on_f32_norms_restores_a_resident_models_norms(build):
    """A borrowed chat model encodes on the F32 norms and gets its own back
    after the encode, and after one that raises: its prefill logits are
    bit for bit what they were."""
    m = build()
    m.set_dtype(mx.bfloat16)
    norms = _f32_of(m)
    tok = _CharTokenizer()
    spec = _spec()
    ids = te.token_ids(spec, tok, "a red bicycle")
    before, state = _logits(m, ids), _norm_state(m)
    plain = te.EncoderHandle(m, tok, spec, owned=False).encode("a red bicycle")
    h = te.EncoderHandle(m, tok, spec, owned=False)
    h.norms = norms
    seen = []
    call = te._Float32RMSNorm.__call__

    def spy(self, x):
        y = call(self, x)
        seen.append((x.dtype, y.dtype))
        return y

    te._Float32RMSNorm.__call__ = spy
    try:
        f32 = h.encode("a red bicycle")
    finally:
        te._Float32RMSNorm.__call__ = call
    assert seen and set(seen) == {(mx.bfloat16, mx.bfloat16)}
    assert f32.features.dtype == plain.features.dtype == mx.bfloat16
    assert not mx.array_equal(f32.features, plain.features).item()
    assert _norm_state(m) == state
    assert mx.array_equal(_logits(m, ids), before).item()

    import gmlx.image.text_encoder as mod

    real = mod.final_hidden

    def boom(model, ids):
        assert _norm_state(model) != state
        raise RuntimeError("boom")

    mod.final_hidden = boom
    try:
        with pytest.raises(RuntimeError, match="boom"):
            h.encode("a red bicycle")
    finally:
        mod.final_hidden = real
    assert _norm_state(m) == state
    assert mx.array_equal(_logits(m, ids), before).item()



@_BUILDS
def test_interleaved_f32_norm_contexts_restore_the_models_own(build):
    """Two contexts on one model, entered A, B and left A, B, as two
    encodes on one shared model would be, leave its own norms; F32
    norms stay in place until the last one leaves."""
    m = build()
    m.set_dtype(mx.bfloat16)
    norms = _f32_of(m)
    state = _norm_state(m)
    inner = te._inner(m)
    a, b = te.float32_norms(m, norms), te.float32_norms(m, norms)
    a.__enter__()
    b.__enter__()
    a.__exit__(None, None, None)
    assert isinstance(inner.layers[0].input_layernorm, te._Float32RMSNorm)
    b.__exit__(None, None, None)
    assert _norm_state(m) == state
    assert not isinstance(inner.layers[0].input_layernorm,
                          te._Float32RMSNorm)
    with te.float32_norms(m, norms):
        with te.float32_norms(m, norms):
            pass
        assert _norm_state(m) != state
    assert _norm_state(m) == state
    other = {k: v * 1 for k, v in norms.items()}
    with te.float32_norms(m, norms):
        with pytest.raises(RuntimeError, match="other F32 norms"):
            with te.float32_norms(m, other):
                pass
    assert _norm_state(m) == state

@_NEEDS_GPU
def test_f32_norms_that_are_not_the_models_are_refused(tmp_path, monkeypatch):
    m = _tiny_text()
    m.set_dtype(mx.bfloat16)
    norms = _f32_of(m)
    rows = tuple((k, np.array(v)) for k, v in norms.items())
    monkeypatch.setattr(te, "_gguf_norms", lambda *a: rows)
    p = tmp_path / "enc.gguf"
    p.write_bytes(b"x")
    got = te.read_norms(str(p), m, _spec())
    assert set(got) == set(norms)
    bad = ((rows[0][0], rows[0][1] * 1.5),) + rows[1:]
    monkeypatch.setattr(te, "_gguf_norms", lambda *a: bad)
    with pytest.raises(te.EncoderMismatchError, match="is not the loaded"):
        te.read_norms(str(p), m, _spec())


@pytest.mark.integration
@pytest.mark.parametrize("name", _FILES)
def test_the_encoder_reads_105_f32_norms_and_a_chat_load_keeps_bf16(
        name, gguf_index):
    """The encoder of a real file reads the 81 offset norms and the 24
    gated norms at F32. The loaded model, and a chat load of the same
    file, hold every norm in bfloat16."""
    from gmlx.load.loader import load_model

    spec = te.encoder_spec("moss_image")
    found = [p for p in gguf_index.get(spec.arch, []) if Path(p).name == name]
    if not found:
        pytest.skip(f"no {name} under KQUANT_TEST_GGUF_DIR")
    h = te.load_encoder(found[0], spec, log=lambda *_: None)
    try:
        roles = {}
        for where, w in h.norms.items():
            assert w.dtype == mx.float32, where
            roles.setdefault(where.split(".", 2)[-1], 0)
            roles[where.split(".", 2)[-1]] += 1
        assert roles == {"input_layernorm": 32, "post_attention_layernorm": 32,
                         "self_attn.q_norm": 8, "self_attn.k_norm": 8,
                         "linear_attn.norm": 24, "norm": 1}
        chat = load_model(found[0])[0]
        for where in h.norms:
            a = te._module(te._inner(h.model), where).weight
            b = te._module(te._inner(chat), where).weight
            assert a.dtype == b.dtype == mx.bfloat16, where
            assert mx.array_equal(a, b).item(), where
        del chat
    finally:
        h.release()
