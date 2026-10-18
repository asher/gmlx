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


def _tiny_qwen35():
    from gmlx._vendor.mlx_vlm.models.qwen3_5.config import TextConfig
    from gmlx.models.qwen35.owned import OwnedQwen3_5LanguageModel

    cfg = TextConfig(
        model_type="qwen3_5", hidden_size=64, intermediate_size=128,
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
    top = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2),
                          image_token_id=124, video_token_id=125,
                          vision_start_token_id=126)
    mx.random.seed(5)
    m = OwnedQwen3_5LanguageModel(cfg, top)
    mx.eval(m.parameters())
    m.eval()
    return m


def _spec():
    import dataclasses

    return dataclasses.replace(te.encoder_spec("moss_image"), hidden=64,
                               layers=4, tap_layers=(4,), identity=())


@_NEEDS_GPU
def test_the_read_equals_the_chat_prefill_of_the_stack():
    """The stack's call with no cache gives what the chat path's prefill
    gives with a fresh cache: the same masks, positions and float32
    recurrent state."""
    m = _tiny_qwen35()
    ids = te.token_ids(_spec(), _CharTokenizer(), "a red bicycle")
    got = te.final_hidden(m, ids)
    x = mx.array([ids])
    want = m.model(x, cache=m.make_cache())
    assert got.shape == (1, len(ids), 64)
    assert mx.allclose(got, want, atol=1e-5, rtol=1e-5).item()


@_NEEDS_GPU
def test_an_encoding_is_the_final_norm_of_the_last_layer():
    m = _tiny_qwen35()
    tok = _CharTokenizer()
    spec = _spec()
    enc = te.EncoderHandle(m, tok, spec, owned=False).encode("a red bicycle")
    ids = list(enc.ids)
    assert ids == te.token_ids(spec, tok, "a red bicycle")
    assert enc.features.shape == (len(ids), 64) and enc.pads == 0
    assert enc.template_id == spec.template_id
    want = te.final_hidden(m, ids)[0]
    assert mx.array_equal(enc.features, want).item()
    sink: list = []
    rows = m.model(mx.array([ids]), cache=m.make_cache(),
                   capture_layer_ids=[3], hidden_sink=sink)
    assert mx.allclose(rows[0], want, atol=1e-5, rtol=1e-5).item()
    # The last layer's output before the norm is not the read.
    assert not mx.allclose(sink[0][0], want, atol=1e-2).item()


@_NEEDS_GPU
def test_a_read_leaves_a_resident_models_positions_alone():
    """A borrowed chat model keeps the positions of its last call for its
    next one, and an encoding through it does not touch them."""
    m = _tiny_qwen35()
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
