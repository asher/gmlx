"""The text encoder tap: the Z-Image template against the official chat
template, the token cap, the tap walk against the forward of the copied
mlx-lm Qwen3 class, the encoder identity checks and release. CPU tier, on
a tiny random Qwen3."""
from __future__ import annotations

import dataclasses
import hashlib
import weakref

import numpy as np
import pytest

import mlx.core as mx

from image_fixtures import FIXTURES, load_json

from gmlx.image import registry
from gmlx.image import text_encoder as te

_TEMPLATE = "z_image_turbo_chat_template.jinja"
_PROMPTS = ["a red bicycle", "", " ", "{} {0} {{x}}", "line one\nline two",
            "caf\u00e9 \u732b"]


def _render_official(prompt: str, **kw) -> str:
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    tmpl = env.from_string((FIXTURES / _TEMPLATE).read_text())
    return tmpl.render(messages=[{"role": "user", "content": prompt}],
                       add_generation_prompt=True, **kw)


@pytest.mark.parametrize("prompt", _PROMPTS)
def test_z_image_template_equals_the_official_render_with_thinking_on(prompt):
    assert te.render(te.Z_IMAGE, prompt) == _render_official(
        prompt, enable_thinking=True)


def test_thinking_off_renders_differently():
    off = _render_official("a", enable_thinking=False)
    assert off != te.render(te.Z_IMAGE, "a")
    assert off.endswith("<think>\n\n</think>\n\n")


def test_template_fixture_is_pinned():
    row = load_json("manifest.json")[_TEMPLATE]
    assert row["repo"] == "Tongyi-MAI/Z-Image-Turbo"
    assert len(row["revision"]) == 40


def test_family_lookup():
    assert te.encoder_spec("z_image") is te.Z_IMAGE
    assert te.Z_IMAGE.tap_layers == (35,) and te.Z_IMAGE.layers == 36
    assert te.Z_IMAGE.max_tokens == 512
    with pytest.raises(te.ImageModelError, match="no text encoder"):
        te.encoder_spec("no_such_family")
    fam = registry.get_family("qwen_image21")
    assert te.encoder_spec("qwen_image21") == te.encoder_spec(fam)
    assert te.encoder_spec(fam).name == "qwen3-vl-8b"
    with pytest.raises(te.ImageModelError, match="no text encoder"):
        te.encoder_spec(dataclasses.replace(fam, name="other", encoder=None))


def test_the_qwen_image21_spec_is_the_row_of_the_reference_pipeline():
    """The template, the dropped rows and the empty prompt against what
    the oracle recorded of the pipeline's own call on each prompt."""
    from gmlx.image.families.qwen_image21 import family as qf

    spec = te.encoder_spec("qwen_image21")
    m = load_json("oracle/check2_qwen_image21.manifest.json")
    ins = m["inputs"]
    assert (spec.arch, spec.model_type) == ("qwen3vl", "qwen3")
    assert (spec.hidden, spec.layers, spec.tap_layers) == (4096, 36, (36,))
    assert spec.max_tokens is None
    assert spec.drop == ins["drop_index"] == 14
    assert "all 36 layers run" in ins["tap"]
    assert "before the final norm" in ins["tap"]
    for key, prompt in ins["prompts"].items():
        assert te.render(spec, prompt) == ins["rendered"][key], key
        assert ins["lengths"][key] - ins["rows"][key] == spec.drop
    assert ins["prompts"]["empty"] == "" and spec.empty == " "
    row = qf.PROMPT
    assert spec.template == row.values["template_t2i"]
    assert spec.template.startswith(row.values["system_turn"])
    assert spec.template.count("{}") == 1 and "think" not in spec.template
    assert row.revision in m["oracle"]["packages"]["diffusers"]["source"]
    assert m["files"]["encoder"]["sha256"] in spec.identity_source
    assert len(spec.identity) == 3
    # The Z-Image row keeps its key: no dropped rows, an empty prompt as
    # it stands.
    z = te.Z_IMAGE
    assert (z.drop, z.empty) == (0, "")
    assert z.template_id == hashlib.sha256(z.template.encode()).hexdigest()
    assert spec.template_id != hashlib.sha256(
        spec.template.encode()).hexdigest()


class _CharTokenizer:
    """One id per character, no special tokens."""

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return [ord(c) % 1000 for c in text]


def test_token_cap_applies_to_the_rendered_text():
    tok = _CharTokenizer()
    short = te.token_ids(te.Z_IMAGE, tok, "hi")
    assert short == [ord(c) % 1000 for c in te.render(te.Z_IMAGE, "hi")]
    long = te.token_ids(te.Z_IMAGE, tok, "x" * 2000)
    assert len(long) == 512
    assert long == [ord(c) % 1000
                    for c in te.render(te.Z_IMAGE, "x" * 2000)][:512]


def _tiny_qwen3(layers: int = 4, seed: int = 0, tied: bool = True):
    from mlx.utils import tree_flatten
    from gmlx._vendor.mlx_lm.models import qwen3

    args = qwen3.ModelArgs(
        model_type="qwen3", hidden_size=64, num_hidden_layers=layers,
        intermediate_size=96, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=128, num_key_value_heads=2, max_position_embeddings=512,
        rope_theta=1e6, head_dim=16, tie_word_embeddings=tied)
    mx.random.seed(seed)
    m = qwen3.Model(args)
    rng = np.random.default_rng(seed)
    params = [(k, mx.array((rng.standard_normal(v.shape) * 0.3
                            + (1.0 if "norm" in k else 0.0)).astype(np.float32)))
              for k, v in tree_flatten(m.parameters())]
    m.load_weights(params)
    m.eval()
    config = {"model_type": "qwen3", "hidden_size": 64,
              "num_hidden_layers": layers}
    return m, config


def _spec(layers=4, tap=(3,), identity=(), **kw):
    return te.EncoderSpec(
        name="tiny", arch="qwen3", model_type="qwen3", hidden=64,
        layers=layers, tap_layers=tap, template="<u>{}</u>", max_tokens=16,
        identity=identity, identity_source="test", **kw)


def test_tap_at_the_last_layer_is_the_stock_forward_before_its_norm():
    m, _ = _tiny_qwen3()
    ids = [3, 17, 42, 5, 99, 1, 64]
    (h,) = te.tap(m, ids, layers=(4,))
    stock = m.model(mx.array([ids]))
    normed = mx.fast.rms_norm(h, m.model.norm.weight, 1e-6)
    assert h.shape == (1, 7, 64)
    assert mx.allclose(normed, stock, atol=1e-5, rtol=1e-5).item()


def test_tap_stops_after_the_last_tap_layer_and_orders_results():
    m, _ = _tiny_qwen3()

    class _Raise:
        def __call__(self, *a):
            raise AssertionError("ran past the tap")

    ids = [3, 17, 42]
    h1, h3 = te.tap(m, ids, layers=(1, 3))
    full3 = te.tap(m, ids, layers=(3,))[0]
    m.model.layers[3] = _Raise()
    assert mx.array_equal(te.tap(m, ids, layers=(3,))[0], full3).item()
    x = m.model.embed_tokens(mx.array([ids]))
    ref1 = m.model.layers[0](x, "causal", None)
    assert mx.array_equal(h1, ref1).item()
    assert mx.array_equal(h3, full3).item()
    with pytest.raises(ValueError, match="outside"):
        te.tap(m, ids, layers=(0,))


def test_tap_of_one_token_runs_without_a_mask():
    m, _ = _tiny_qwen3()
    (h,) = te.tap(m, [7], layers=(2,))
    x = m.model.embed_tokens(mx.array([[7]]))
    for blk in m.model.layers[:2]:
        x = blk(x, None, None)
    assert mx.array_equal(h, x).item()


def _identity_rows(m):
    rows = []
    for gguf, path in (("blk.0.attn_norm.weight",
                        "model.layers.0.input_layernorm.weight"),
                       ("output_norm.weight", "model.norm.weight")):
        w = te._model_param(m, path)
        rows.append(te.IdentityTensor(gguf, path, te._bf16_digest(w)))
    return tuple(rows)


def test_check_encoder_refuses_shape_and_identity_mismatches():
    m, cfg = _tiny_qwen3()
    spec = _spec(identity=_identity_rows(m))
    te.check_encoder(m, cfg, spec)
    with pytest.raises(te.EncoderMismatchError,
                       match="hidden size 64, expected 128"):
        te.check_encoder(m, cfg, dataclasses.replace(spec, hidden=128))
    with pytest.raises(te.EncoderMismatchError, match="model type 'llama'"):
        te.check_encoder(m, {**cfg, "model_type": "llama"}, spec)
    other, _ = _tiny_qwen3(seed=1)
    with pytest.raises(te.EncoderMismatchError,
                       match="blk.0.attn_norm.weight, output_norm.weight"):
        te.check_encoder(other, cfg, spec)
    te.check_encoder(other, cfg, spec, override=True)


def test_check_encoder_file_reads_the_header_and_norm_bytes(tmp_path):
    rng = np.random.default_rng(0)
    norm0 = rng.standard_normal(64).astype(np.float32)
    outn = rng.standard_normal(64).astype(np.float32)
    path = tmp_path / "enc.gguf"
    import gguf

    w = gguf.GGUFWriter(str(path), "qwen3")
    w.add_uint32("qwen3.embedding_length", 64)
    w.add_uint32("qwen3.block_count", 4)
    w.add_tensor("blk.0.attn_norm.weight", norm0)
    w.add_tensor("output_norm.weight", outn)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    rows = tuple(
        te.IdentityTensor(g, p, te._bf16_digest(mx.array(a)))
        for g, p, a in (("blk.0.attn_norm.weight", "x", norm0),
                        ("output_norm.weight", "y", outn)))
    view = te._header_view(str(path))
    te.check_encoder_file(view, _spec(identity=rows))
    bad = (rows[0], te.IdentityTensor("output_norm.weight", "y", "0" * 64))
    with pytest.raises(te.EncoderMismatchError, match="output_norm.weight"):
        te.check_encoder_file(view, _spec(identity=bad))
    te.check_encoder_file(view, _spec(identity=bad), override=True)
    with pytest.raises(te.EncoderMismatchError, match="4 layers, expected 5"):
        te.check_encoder_file(view, _spec(layers=5, identity=rows))


def test_owned_handle_releases_and_a_borrowed_one_does_not():
    m, cfg = _tiny_qwen3()
    spec = _spec()
    owned = te.EncoderHandle(m, _CharTokenizer(), spec, owned=True)
    ref = weakref.ref(m)
    del m
    got = owned.encode("ab")
    assert got.ids == tuple(ord(c) % 1000 for c in "<u>ab</u>")
    assert len(got.hidden) == 1 and got.hidden[0].shape[-1] == 64
    assert got.template_id == spec.template_id
    assert owned.release() is True
    assert owned.released and owned.model is None
    assert ref() is None
    assert owned.release() is False
    with pytest.raises(RuntimeError, match="released"):
        owned.encode("ab")

    m2, cfg2 = _tiny_qwen3()
    borrowed = te.borrow_encoder(m2, _CharTokenizer(), cfg2, spec)
    assert borrowed.release() is False
    assert borrowed.model is m2


def test_dropped_rows_run_and_stay_out_of_the_features():
    """The features are the rows after the dropped ones, of a forward
    that ran them all: they are not the encoding of the later tokens
    alone."""
    m, _ = _tiny_qwen3()
    tok = _CharTokenizer()
    plain, spec = _spec(), _spec(drop=3)
    whole = te.EncoderHandle(m, tok, plain, owned=False).encode("ab")
    got = te.EncoderHandle(m, tok, spec, owned=False).encode("ab")
    assert got.ids == whole.ids and got.drop == 3 and whole.drop == 0
    assert mx.array_equal(got.hidden[0], whole.hidden[0]).item()
    assert got.features.shape == (len(got.ids) - 3, 64)
    assert mx.array_equal(got.features, whole.hidden[0][0, 3:]).item()
    assert mx.array_equal(whole.features, whole.hidden[0][0]).item()
    (alone,) = te.tap(m, list(got.ids[3:]), layers=spec.tap_layers)
    assert not mx.allclose(alone[0], got.features, atol=1e-3).item()
    # The dropped count is part of what a conditioning depends on.
    assert spec.template_id != plain.template_id
    assert _spec(drop=4).template_id != spec.template_id


def test_a_prompt_must_leave_a_row_after_the_dropped_ones():
    tok = _CharTokenizer()
    n = len(te.render(_spec(), ""))
    assert len(te.token_ids(_spec(drop=n), tok, "a")) == n + 1
    with pytest.raises(ValueError, match=f"first {n} are dropped"):
        te.token_ids(_spec(drop=n), tok, "")
    with pytest.raises(ValueError, match="no tokens"):
        te.token_ids(dataclasses.replace(_spec(), max_tokens=0), tok, "a")


def test_an_empty_prompt_is_rendered_as_the_spec_says():
    spec = _spec(empty=" ")
    assert te.render(spec, "") == "<u> </u>"
    assert te.render(spec, "a") == "<u>a</u>"
    assert te.render(_spec(), "") == "<u></u>"
    assert te.render(te.Z_IMAGE, "") == te.Z_IMAGE.template.replace("{}", "")


def test_an_owned_encoder_holds_no_output_matrix(monkeypatch):
    """``load_encoder`` drops the output matrix of a model that has its
    own, and the tap of the model is what it was. A model with tied
    embeddings has none, and a borrowed model keeps its own."""
    from gmlx.load import loader

    m, cfg = _tiny_qwen3(tied=False)
    ids = [3, 17, 42, 5]
    assert "lm_head" in m
    (before,) = te.tap(m, ids, layers=(4,))
    mx.eval(before)
    monkeypatch.setattr(te, "_header_view", lambda path: None)
    monkeypatch.setattr(te, "check_encoder_file", lambda *a, **k: None)
    monkeypatch.setattr(loader, "load_model",
                        lambda path: (m, cfg, _CharTokenizer()))
    h = te.load_encoder("enc.gguf", _spec(), log=lambda *_: None)
    assert h.owned and h.model is m and "lm_head" not in m
    assert mx.array_equal(te.tap(m, ids, layers=(4,))[0], before).item()
    assert te.drop_output_head(m) == 0

    tied, _ = _tiny_qwen3()
    assert "lm_head" not in tied and te.drop_output_head(tied) == 0
    assert te.drop_output_head(_tiny_qwen3(tied=False)[0]) == 128 * 64 * 4

    m2, cfg2 = _tiny_qwen3(tied=False)
    te.borrow_encoder(m2, _CharTokenizer(), cfg2, _spec())
    assert "lm_head" in m2


def test_a_released_encoder_leaves_the_free_memory_reading(monkeypatch):
    """The mapped weights an owned load registers for the free-memory
    reading go when the encoder is released, and the output matrix goes
    when it is dropped. A load inside the window of another owner leaves
    them to that owner."""
    from gmlx.gen import prefill_decay as pd
    from gmlx.load import loader

    key = ("enc.gguf",)
    head = 128 * 64 * 4

    def load(path):
        m, cfg = _tiny_qwen3(tied=False)
        # The tiny model owns its arrays, so the registration stands for
        # mapped ones of the same size, the output matrix among them.
        pd.note_untracked_weights(10 * head, key=key)
        object.__setattr__(m, "_kq_weights_key", key)
        return m, cfg, _CharTokenizer()

    monkeypatch.setattr(te, "_header_view", lambda path: None)
    monkeypatch.setattr(te, "check_encoder_file", lambda *a, **k: None)
    monkeypatch.setattr(loader, "load_model", load)
    monkeypatch.setattr(te.mx, "get_active_memory", lambda: 0)
    assert pd.untracked_weight_bytes_for(key) == 0

    h = te.load_encoder("enc.gguf", _spec(), log=lambda *_: None)
    assert pd.untracked_weights_owner() is None
    assert h.weights_owner is not None
    assert pd.untracked_weight_bytes_for(key) == 9 * head
    assert h.release() is True
    assert pd.untracked_weight_bytes_for(key) == 0
    assert h.weights_owner is None and h.release() is False

    # A second load of the file counts again, and so does its release.
    h = te.load_encoder("enc.gguf", _spec(), log=lambda *_: None)
    assert pd.untracked_weight_bytes_for(key) == 9 * head
    h.release()
    assert pd.untracked_weight_bytes_for(key) == 0

    pool = object()
    pd.set_untracked_weights_owner(pool)
    try:
        h = te.load_encoder("enc.gguf", _spec(), log=lambda *_: None)
        assert pd.untracked_weights_owner() is pool
    finally:
        pd.set_untracked_weights_owner(None)
    assert h.weights_owner is None
    h.release()
    assert pd.untracked_weight_bytes_for(key) == 9 * head
    pd.forget_untracked_weights(pool)
    assert pd.untracked_weight_bytes_for(key) == 0


def test_a_tap_runs_in_the_dtype_its_spec_names():
    """The layers of a tap with an activation dtype run in it by promotion
    over the weights as loaded, and the conditioning key says so."""
    from mlx.utils import tree_map

    m, cfg = _tiny_qwen3()
    ids = [3, 17, 42, 5, 9]
    (want,) = te.tap(m, ids, layers=(4,))
    assert want.dtype == mx.float32
    m.update(tree_map(lambda v: v.astype(mx.bfloat16), m.parameters()))
    (low,) = te.tap(m, ids, layers=(4,))
    (high,) = te.tap(m, ids, layers=(4,), activation="float32")
    assert low.dtype == mx.bfloat16 and high.dtype == mx.float32

    def gap(x):
        return mx.sqrt(((x.astype(mx.float32) - want) ** 2).sum()).item()

    assert gap(high) < gap(low)
    spec = _spec(layers=4, tap=(4,), activation="float32")
    h = te.borrow_encoder(m, _CharTokenizer(), cfg, spec)
    assert h.dtype_name == "float32"
    got = h.encode("ab")
    assert got.hidden[0].dtype == mx.float32
    plain = te.borrow_encoder(m, _CharTokenizer(), cfg,
                              _spec(layers=4, tap=(4,)))
    assert plain.dtype_name == "bfloat16"
    assert plain.encode("ab").hidden[0].dtype == mx.bfloat16
    assert te.encoder_spec("qwen_image21").activation == "float32"
    assert te.Z_IMAGE.activation is None
