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
import mlx.nn as nn

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
    edit = load_json("oracle/check2_qwen_image21_edit.manifest.json")
    assert edit["files"]["mmproj"]["sha256"] in spec.vision_identity_source
    assert len(spec.vision_identity) == 3
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


def _spec(layers=4, tap=(3,), identity=(), max_tokens=16, **kw):
    return te.EncoderSpec(
        name="tiny", arch="qwen3", model_type="qwen3", hidden=64,
        layers=layers, tap_layers=tap, template="<u>{}</u>",
        max_tokens=max_tokens,
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


def _mmproj(path, arch="clip", width=64):
    import gguf

    rng = np.random.default_rng(1)
    norms = {n: rng.standard_normal(16).astype(np.float32)
             for n in ("v.blk.0.ln1.weight", "v.post_ln.weight")}
    w = gguf.GGUFWriter(str(path), arch)
    w.add_bool("clip.has_vision_encoder", True)
    w.add_uint32("clip.vision.projection_dim", width)
    for n, a in norms.items():
        w.add_tensor(n, a)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return tuple(te.IdentityTensor(n, "x", te._bf16_digest(mx.array(a)))
                 for n, a in norms.items())


def test_an_mmproj_is_checked_by_its_header_and_norm_bytes(tmp_path):
    """The mmproj is refused by name before any weights load: another
    architecture, a projector to another width, or norm bytes that differ
    from the spec's rows unless the override accepts them."""
    rows = _mmproj(tmp_path / "mm.gguf")
    view = te._header_view(str(tmp_path / "mm.gguf"))
    spec = _spec(vision_type="qwen3_vl", vision_identity=rows,
                 vision_identity_source="the tiny mmproj")
    te.check_mmproj_file(view, spec)
    bad = (rows[0], te.IdentityTensor("v.post_ln.weight", "x", "0" * 64))
    other = dataclasses.replace(spec, vision_identity=bad)
    with pytest.raises(te.EncoderMismatchError,
                       match=r"v\.post_ln\.weight differ from the tiny"):
        te.check_mmproj_file(view, other)
    te.check_mmproj_file(view, other, override=True)
    with pytest.raises(te.EncoderMismatchError,
                       match="projects to 64, and the encoder's hidden size "
                             "is 32"):
        te.check_mmproj_file(view, dataclasses.replace(spec, hidden=32),
                             override=True)
    _mmproj(tmp_path / "llm.gguf", arch="qwen3")
    with pytest.raises(te.EncoderMismatchError,
                       match="architecture is 'qwen3', expected a clip"):
        te.check_mmproj_file(te._header_view(str(tmp_path / "llm.gguf")),
                             spec, override=True)


def test_an_mmproj_is_checked_before_the_tower_loads(monkeypatch):
    seen = []
    monkeypatch.setattr(te, "_header_view", lambda path: path)
    monkeypatch.setattr(te, "check_encoder_file", lambda *a, **k: None)

    def refuse(view, spec, *, override=False):
        seen.append((view, override))
        raise te.EncoderMismatchError("not the tower")

    monkeypatch.setattr(te, "check_mmproj_file", refuse)
    monkeypatch.setattr(te, "_load_owned", lambda *a: seen.append("load"))
    spec = _spec(vision_type="qwen3_vl")
    with pytest.raises(te.EncoderMismatchError, match="not the tower"):
        te.load_encoder("enc.gguf", spec, mmproj="mm.gguf", override=True,
                        log=lambda *_: None)
    assert seen == [("mm.gguf", True)]


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
    # The release returns the freed buffers MLX keeps, so the DiT that
    # loads next does not stand beside them.
    freed = mx.ones((1 << 20,))
    mx.eval(freed)
    del freed
    assert mx.get_cache_memory() > 0
    assert owned.release() is True
    assert mx.get_cache_memory() == 0
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


def test_a_kept_tail_survives_the_token_cap():
    """The cap cuts the prompt and keeps the rows after it, as Krea 2's
    pipeline does: the first ``max_tokens - keep_tail`` ids, then the last
    ``keep_tail``."""
    tok = _CharTokenizer()
    spec = _spec(keep_tail=5)
    full = [ord(c) % 1000 for c in te.render(spec, "x" * 40)]
    got = te.token_ids(spec, tok, "x" * 40)
    assert len(got) == 16
    assert got == full[:11] + full[-5:]
    assert got[-5:] == [ord(c) for c in "x</u>"]
    # Under the cap the ids are the rendered text.
    assert te.token_ids(spec, tok, "ab") == [
        ord(c) % 1000 for c in te.render(spec, "ab")]
    assert te.token_ids(_spec(), tok, "x" * 40) == full[:16]


def test_stacked_features_hold_every_tap_side_by_side():
    m, _ = _tiny_qwen3()
    tok = _CharTokenizer()
    one = te.EncoderHandle(m, tok, _spec(tap=(1, 3), drop=2),
                           owned=False).encode("ab")
    spec = _spec(tap=(1, 3), drop=2, stack=True)
    got = te.EncoderHandle(m, tok, spec, owned=False).encode("ab")
    assert got.stack and not one.stack
    n = len(got.ids) - 2
    assert got.features.shape == (n, 2 * 64)
    assert mx.array_equal(got.features[:, :64], got.hidden[0][0, 2:]).item()
    assert mx.array_equal(got.features[:, 64:], got.hidden[1][0, 2:]).item()
    assert mx.array_equal(one.features, got.features[:, :64]).item()
    # Stacking is part of what a conditioning depends on.
    assert spec.template_id != _spec(tap=(1, 3), drop=2).template_id


_FLUX3_TEMPLATE = "flux3_klein_chat_template.jinja"


@pytest.mark.parametrize("prompt", _PROMPTS)
def test_the_flux3_template_is_the_official_render_of_one_user_turn(prompt):
    """FLUX.3 klein's reference renders the Qwen3-VL chat template of its
    text encoder on one user turn with the generation prompt."""
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    tmpl = env.from_string((FIXTURES / _FLUX3_TEMPLATE).read_text())
    want = tmpl.render(messages=[{"role": "user", "content": prompt}],
                       add_generation_prompt=True)
    assert te.render(te.encoder_spec("flux3_klein"), prompt) == want
    row = load_json("manifest.json")[_FLUX3_TEMPLATE]
    assert row["repo"] == "black-forest-labs/flux-3-image-klein"
    assert len(row["revision"]) == 40


def test_the_flux3_spec_pads_as_the_reference_does():
    from gmlx.image.families.flux3_klein import family as ff

    spec = te.encoder_spec("flux3_klein")
    assert (spec.arch, spec.model_type, spec.hidden, spec.layers) == (
        "qwen3vl", "qwen3", 2560, 36)
    assert spec.tap_layers == (4, 8, 12, 16, 20, 24, 28, 32) and spec.stack
    assert (spec.pad_multiple, spec.max_tokens) == (80, 4096)
    assert (spec.pad_id, spec.pad_position) == (151643, 1)
    assert spec.drop == 0 and spec.activation == "float32"
    assert spec.identity_source == ff.ENCODER_IDENTITY_SOURCE
    # The same encoder weights as Krea 2's, in its own template.
    krea = te.encoder_spec("krea2")
    assert spec.identity == krea.identity
    assert spec.template_id != krea.template_id


_FLUX2_TEMPLATE = "flux2_klein_chat_template.jinja"


@pytest.mark.parametrize("prompt", _PROMPTS)
def test_the_flux2_template_is_the_official_render_with_thinking_off(prompt):
    """FLUX.2 klein's reference renders the Qwen3 chat template of the
    official tokenizer on one user turn with the generation prompt and
    enable_thinking False. That template is Z-Image's, byte for byte."""
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    text = (FIXTURES / _FLUX2_TEMPLATE).read_text()
    assert text == (FIXTURES / _TEMPLATE).read_text()
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    want = env.from_string(text).render(
        messages=[{"role": "user", "content": prompt}],
        add_generation_prompt=True, enable_thinking=False)
    assert te.render(te.encoder_spec("flux2_klein"), prompt) == want
    row = load_json("manifest.json")[_FLUX2_TEMPLATE]
    assert row["repo"] == "black-forest-labs/FLUX.2-klein-4B"
    assert len(row["revision"]) == 40


def test_the_flux2_spec_pads_as_the_reference_does():
    from gmlx.image.families.flux2_klein import family as ff

    spec = te.encoder_spec("flux2_klein")
    assert (spec.arch, spec.model_type, spec.hidden, spec.layers) == (
        "qwen3", "qwen3", 2560, 36)
    assert spec.tap_layers == (9, 18, 27) and spec.stack
    # Every prompt fills 512 rows, its own and the pads.
    assert spec.pad_multiple == spec.max_tokens == 512
    assert (spec.pad_id, spec.pad_position) == (151643, None)
    assert spec.drop == 0 and spec.activation == "float32"
    assert spec.identity_source == ff.ENCODER_IDENTITY_SOURCE
    assert [te.pad_rows(spec, n) for n in (1, 33, 511, 512)] == [
        511, 479, 1, 0]
    assert te.rows_at_most(spec, "a cat") == 512
    # The same encoder weights as Z-Image's, in another template.
    assert spec.identity == te.Z_IMAGE.identity
    assert spec.template_id != te.Z_IMAGE.template_id


def test_pad_rows_reach_the_next_multiple_under_the_cap():
    spec = _spec(pad_multiple=8, pad_id=0)
    assert [te.pad_rows(spec, n) for n in (1, 3, 8, 9, 15, 16)] == [
        7, 5, 0, 7, 1, 0]
    capped = dataclasses.replace(spec, max_tokens=12)
    assert [te.pad_rows(capped, n) for n in (3, 8, 9, 12)] == [5, 0, 3, 0]
    assert te.pad_rows(_spec(), 3) == 0
    with pytest.raises(ValueError, match="pad id"):
        te.pad_rows(_spec(pad_multiple=8), 3)
    with pytest.raises(ValueError, match="no dropped rows"):
        te.pad_rows(_spec(pad_multiple=8, pad_id=0, drop=2), 3)
    # The pad rule is part of what a conditioning depends on.
    ids = {s.template_id for s in (
        _spec(), spec, dataclasses.replace(spec, pad_multiple=4),
        dataclasses.replace(spec, pad_id=1),
        dataclasses.replace(spec, pad_position=0),
        dataclasses.replace(spec, pad_position=None))}
    assert len(ids) == 6
    # A spec with no pads keeps the key it had before pads existed, and a
    # spec with pads at one position the key it had before pads at their
    # own indices existed.
    assert _spec().template_id == dataclasses.replace(
        _spec(), pad_id=0, pad_position=3).template_id
    flux3 = te.encoder_spec("flux3_klein")
    assert flux3.template_id == hashlib.sha256(
        (flux3.template + "\x00stack\x00pad=80,151643,1").encode()).hexdigest()


@pytest.mark.parametrize("family", ["z_image", "krea2", "qwen_image21",
                                    "flux3_klein"])
def test_the_rows_of_a_prompt_are_bounded_by_its_utf8_bytes(family):
    """Byte-level BPE gives each token at least one byte, so the bound of
    a prompt is its UTF-8 bytes, not its code points: a prompt of 2, 3 and
    4 byte characters is bounded as an ASCII prompt of its bytes. A
    character whose NFC form holds more bytes is bounded at those."""
    spec = te.encoder_spec(family)
    for ch, size in (("\u00e9", 2), ("\u732b", 3), ("\U0001F6B2", 4),
                     ("\ufb2c", 6)):
        prompt = "a cat " + ch * 12
        assert te.rows_at_most(spec, prompt) == te.rows_at_most(
            spec, "a cat " + "x" * (12 * size)), (family, ch)


def _np_forward(m, seq, positions, allowed, layers):
    """The tiny Qwen3's decoder layers in float64 numpy, with one rotary
    position per row and a boolean mask of the keys each row reads."""
    from mlx.utils import tree_flatten

    p = {k: np.array(v, np.float64) for k, v in tree_flatten(m.parameters())}
    nh, nkv, hd = 4, 2, 16
    inv = 1e6 ** (-np.arange(0, hd, 2) / hd)
    ang = np.asarray(positions, np.float64)[:, None] * inv[None]
    cos, sin = np.cos(ang), np.sin(ang)

    def rms(x, w):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6) * w

    def rope(t):
        a, b = t[..., :hd // 2], t[..., hd // 2:]
        return np.concatenate([a * cos - b * sin, a * sin + b * cos], -1)

    x = p["model.embed_tokens.weight"][np.asarray(seq)]
    length = len(seq)
    out = {}
    for i in range(max(layers)):
        w = {k[len(f"model.layers.{i}."):]: v for k, v in p.items()
             if k.startswith(f"model.layers.{i}.")}
        h = rms(x, w["input_layernorm.weight"])

        def heads(name, n):
            return (h @ w[f"self_attn.{name}_proj.weight"].T).reshape(
                length, n, hd).transpose(1, 0, 2)

        q = rope(rms(heads("q", nh), w["self_attn.q_norm.weight"]))
        k = rope(rms(heads("k", nkv), w["self_attn.k_norm.weight"]))
        v = heads("v", nkv)
        k, v = np.repeat(k, nh // nkv, 0), np.repeat(v, nh // nkv, 0)
        s = np.where(allowed, q @ k.transpose(0, 2, 1) * hd ** -0.5, -np.inf)
        a = np.exp(s - s.max(-1, keepdims=True))
        a /= a.sum(-1, keepdims=True)
        o = (a @ v).transpose(1, 0, 2).reshape(length, nh * hd)
        x = x + o @ w["self_attn.o_proj.weight"].T
        h = rms(x, w["post_attention_layernorm.weight"])
        g = h @ w["mlp.gate_proj.weight"].T
        x = x + (g / (1 + np.exp(-g)) * (h @ w["mlp.up_proj.weight"].T)
                 ) @ w["mlp.down_proj.weight"].T
        if i + 1 in layers:
            out[i + 1] = x
    return [out[k] for k in layers]


def _close(got, want, tol=2e-5):
    g = np.array(got[0], np.float64)
    return np.linalg.norm(g - want) / np.linalg.norm(want) < tol


def _cpu_tap(*a, **kw):
    """A tap on the CPU, whose float32 products are not rounded to
    TF32 as the GPU's are, so a float64 reference reads it at 2e-5."""
    with mx.stream(mx.cpu):
        return te.tap(*a, **kw)


def test_pad_rows_are_the_reference_forward_of_the_padded_prompt():
    """The reference runs every pad row: token pad_id, at the pad position,
    reading the real keys alone under the causal mask. Each tap of the
    padded prompt equals that forward, row for row."""
    m, _ = _tiny_qwen3()
    ids, pad, pad_id = [3, 17, 42, 5, 99, 1, 64], 5, 9
    n = len(ids)
    seq = ids + [pad_id] * pad
    allowed = np.tril(np.ones((n + pad, n + pad), bool))
    allowed[:, n:] = False
    layers = (1, 4)
    plain = _np_forward(m, ids, range(n), np.tril(np.ones((n, n), bool)),
                        layers)
    for got, want in zip(_cpu_tap(m, ids, layers=layers), plain):
        assert _close(got, want)
    for position in (1, 0, 5):
        want = _np_forward(m, seq, list(range(n)) + [position] * pad,
                           allowed, layers)
        got = _cpu_tap(m, ids, layers=layers, pad=pad, pad_id=pad_id,
                     pad_position=position)
        for g, w in zip(got, want):
            assert g.shape == (1, n + pad, 64)
            assert _close(g, w), position
            assert np.allclose(w[n:], w[n], rtol=1e-12, atol=0)
    # Pads that read themselves, or each other as causal tokens, are
    # another forward.
    for wrong in (np.tril(np.ones((n + pad, n + pad), bool)),
                  allowed | np.eye(n + pad, dtype=bool)):
        (w,) = _np_forward(m, seq, list(range(n)) + [1] * pad, wrong, (4,))
        (g,) = _cpu_tap(m, ids, layers=(4,), pad=pad, pad_id=pad_id)
        assert not _close(g, w, tol=1e-2)


def test_pad_rows_at_their_own_indices_are_the_reference_forward(
        monkeypatch):
    """With no pad position each pad row runs at its own index, reading
    the real keys alone, as FLUX.2 klein's reference runs them. The pad
    rows then differ, and pads at one position or read as causal tokens
    are another forward."""
    m, _ = _tiny_qwen3()
    ids, pad, pad_id = [3, 17, 42, 5, 99, 1, 64], 5, 9
    n = len(ids)
    seq = ids + [pad_id] * pad
    allowed = np.tril(np.ones((n + pad, n + pad), bool))
    allowed[:, n:] = False
    layers = (1, 4)
    want = _np_forward(m, seq, range(n + pad), allowed, layers)
    got = _cpu_tap(m, ids, layers=layers, pad=pad, pad_id=pad_id,
                   pad_position=None)
    for g, w in zip(got, want):
        assert g.shape == (1, n + pad, 64)
        assert _close(g, w)
        spread = np.linalg.norm(w[n + 1:] - w[n], axis=1) / np.linalg.norm(
            w[n])
        assert spread.min() > 1e-3
    for wrong in (_np_forward(m, seq, list(range(n)) + [n] * pad, allowed,
                              (4,)),
                  _np_forward(m, seq, range(n + pad),
                              np.tril(np.ones((n + pad, n + pad), bool)),
                              (4,))):
        assert not _close(got[1], wrong[0], tol=1e-2)
    # The real rows are the plain tap's, and one token runs with its pads.
    (plain,) = _cpu_tap(m, ids, layers=(4,))
    real = np.array(got[1][0, :n], np.float64)
    ref = np.array(plain[0], np.float64)
    assert np.linalg.norm(real - ref) / np.linalg.norm(ref) < 1e-5
    (one,) = _cpu_tap(m, [7], layers=(2,), pad=2, pad_id=9,
                      pad_position=None)
    want = _np_forward(m, [7, 9, 9], [0, 1, 2],
                       np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], bool),
                       (2,))[0]
    assert _close(one, want)
    with pytest.raises(ValueError, match="pad id"):
        _cpu_tap(m, ids, layers=(3,), pad=2, pad_position=None)
    # No rope is swapped for pads at their own indices.
    def swapped(*_a, **_k):
        raise AssertionError("pad_rope ran")

    monkeypatch.setattr(te, "pad_rope", swapped)
    te.tap(m, [3, 17], layers=(1,), pad=2, pad_id=9, pad_position=None)
    with pytest.raises(AssertionError, match="pad_rope ran"):
        te.tap(m, [3, 17], layers=(1,), pad=2, pad_id=9)


def test_the_real_rows_of_a_padded_tap_are_the_tap_of_the_prompt():
    m, _ = _tiny_qwen3()
    ids = [3, 17, 42, 5]
    (plain,) = te.tap(m, ids, layers=(3,))
    (padded,) = te.tap(m, ids, layers=(3,), pad=3, pad_id=9)
    # The CPU's float32 GEMM rounds 5 rows apart from 4: the real rows read
    # 5.4e-7 of relative L2 from the plain tap there, and match on the GPU.
    # Real rows that read the pad row read 1.1 away.
    got = np.array(padded[0, :4], np.float64)
    want = np.array(plain[0], np.float64)
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < 1e-5
    # One token and its pads: the pad row reads the token alone.
    (one,) = _cpu_tap(m, [7], layers=(2,), pad=2, pad_id=9)
    want = _np_forward(m, [7, 9, 9], [0, 1, 1],
                       np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], bool),
                       (2,))[0]
    assert _close(one, want)
    with pytest.raises(ValueError, match="pad id"):
        _cpu_tap(m, ids, layers=(3,), pad=2)


def test_a_padded_tap_restores_the_ropes_and_refuses_a_block_without_one():
    m, _ = _tiny_qwen3()
    ropes = [b.self_attn.rope for b in m.model.layers]

    def restored():
        return all(b.self_attn.rope is r
                   for b, r in zip(m.model.layers, ropes))

    te.tap(m, [3, 17], layers=(4,), pad=2, pad_id=9)
    assert restored()

    class _Boom(Exception):
        pass

    with pytest.raises(_Boom):
        with te.pad_rope(m, 2, 1):
            assert all(isinstance(b.self_attn.rope, te._PadRope)
                       for b in m.model.layers)
            raise _Boom
    assert restored()

    class _NoAttention:
        def __call__(self, x, mask, cache):
            return x

    m.model.layers[2] = _NoAttention()
    with pytest.raises(TypeError, match="_NoAttention has no self_attn"):
        te.tap(m, [3, 17], layers=(4,), pad=2, pad_id=9)
    assert all(m.model.layers[i].self_attn.rope is ropes[i] for i in (0, 1, 3))
    # A tap without pads swaps nothing, so it runs.
    te.tap(m, [3, 17], layers=(4,))


class _Bf16Product(nn.Module):
    """A product that returns bfloat16 whatever its input, as a Q8_0
    KQuantLinear does on a float32 input."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def __call__(self, x):
        return self.inner(x).astype(mx.bfloat16)


class _Float32Product(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def __call__(self, x):
        return self.inner(x).astype(mx.float32)


def _bf16_products(m):
    """The tiny model as a quantized file loads: bf16 weights, so bf16
    norm scales, and products that return bf16."""
    m.set_dtype(mx.bfloat16)
    for b in m.model.layers:
        for o, names in ((b.self_attn, ("q_proj", "k_proj", "v_proj",
                                        "o_proj")),
                         (b.mlp, ("gate_proj", "up_proj", "down_proj"))):
            for name in names:
                setattr(o, name, _Bf16Product(getattr(o, name)))


def test_fp32_qkv_runs_the_qk_norm_rope_and_attention_in_float32(
        monkeypatch):
    from gmlx._vendor.mlx_lm.models import qwen3

    m, _ = _tiny_qwen3()
    _bf16_products(m)
    seen = []
    sdpa = qwen3.scaled_dot_product_attention

    def spy(q, k, v, **kw):
        seen.append((q.dtype, k.dtype, v.dtype))
        return sdpa(q, k, v, **kw)

    monkeypatch.setattr(qwen3, "scaled_dot_product_attention", spy)
    products = [[getattr(b.self_attn, n) for n in te._QKV]
                for b in m.model.layers]
    ids = [3, 17, 42, 5]
    kw = dict(layers=(2, 4), activation="float32", pad=3, pad_id=9,
              pad_position=None)
    plain = te.tap(m, ids, **kw)
    assert len(seen) == 4 and set(seen) == {(mx.bfloat16,) * 3}
    seen.clear()
    cast = te.tap(m, ids, fp32_qkv=True, **kw)
    assert len(seen) == 4 and set(seen) == {(mx.float32,) * 3}
    assert all(getattr(b.self_attn, n) is p
               for b, ps in zip(m.model.layers, products)
               for n, p in zip(te._QKV, ps))
    for a, b in zip(plain, cast):
        assert a.dtype == b.dtype == mx.float32
        assert not mx.array_equal(a, b).item()
    # The cast tap is the tap of a model whose q, k and v products return
    # float32 after their own rounding, and the other products are as
    # they were.
    for b in m.model.layers:
        for n in te._QKV:
            setattr(b.self_attn, n, _Float32Product(getattr(b.self_attn, n)))
    for a, b in zip(cast, te.tap(m, ids, **kw)):
        assert mx.array_equal(a, b).item()
    # The products are restored after a tap that raises, and a block with
    # no products to cast swaps none.
    m2, _ = _tiny_qwen3()
    before = [[getattr(b.self_attn, n) for n in te._QKV]
              for b in m2.model.layers]

    def same():
        return all(getattr(b.self_attn, n) is p
                   for b, ps in zip(m2.model.layers, before)
                   for n, p in zip(te._QKV, ps))

    with pytest.raises(RuntimeError, match="boom"):
        with te.float32_qkv(m2):
            assert all(isinstance(b.self_attn.q_proj, te._Float32Out)
                       for b in m2.model.layers)
            raise RuntimeError("boom")
    assert same()

    class _NoAttention:
        def __call__(self, x, mask, cache):
            return x

    keep = m2.model.layers[2]
    m2.model.layers[2] = _NoAttention()
    with pytest.raises(TypeError, match="_NoAttention has no self_attn"):
        te.tap(m2, ids, layers=(4,), fp32_qkv=True)
    m2.model.layers[2] = keep
    assert same()


def test_an_encoding_casts_qkv_only_when_its_spec_says():
    m, _ = _tiny_qwen3()
    _bf16_products(m)
    tok = _CharTokenizer()
    spec = _spec(tap=(1, 3), stack=True, pad_multiple=8, pad_id=9,
                 max_tokens=24, pad_position=None)
    got = {}
    for flag in (False, True):
        s = dataclasses.replace(spec, fp32_qkv=flag)
        enc = te.EncoderHandle(m, tok, s, owned=False).encode("ab")
        want = te.tap(m, list(enc.ids), layers=(1, 3), pad=enc.pads,
                      pad_id=9, pad_position=None, fp32_qkv=flag)
        assert mx.array_equal(enc.features,
                              mx.concatenate([w[0] for w in want], -1)).item()
        assert enc.template_id == spec.template_id
        got[flag] = enc.features
    assert not mx.array_equal(got[False], got[True]).item()


def test_only_the_flux2_specs_cast_qkv():
    specs = {}
    for fam in registry.families():
        try:
            specs[fam.name] = te.encoder_spec(fam)
        except te.ImageModelError:
            continue
    assert {n for n, s in specs.items() if s.fp32_qkv} == {"flux2_klein",
                                                           "flux2_klein_9b"}
    assert len(specs) > 2


def test_an_encoding_keeps_its_pad_rows():
    m, _ = _tiny_qwen3()
    tok = _CharTokenizer()
    spec = _spec(tap=(1, 3), stack=True, pad_multiple=8, pad_id=9,
                 max_tokens=24)
    got = te.EncoderHandle(m, tok, spec, owned=False).encode("ab")
    n = len(got.ids)
    assert got.ids == tuple(ord(c) for c in te.render(spec, "ab"))
    assert got.pads == te.pad_rows(spec, n) == 16 - n
    assert got.features.shape == (16, 2 * 64)
    want = te.tap(m, list(got.ids), layers=(1, 3), pad=got.pads, pad_id=9)
    assert mx.array_equal(got.features,
                          mx.concatenate([w[0] for w in want], -1)).item()
    assert got.template_id == spec.template_id
    plain = te.EncoderHandle(m, tok, _spec(tap=(1, 3), stack=True,
                                           max_tokens=24),
                             owned=False).encode("ab")
    assert plain.pads == 0 and plain.features.shape == (n, 2 * 64)
    # Pads at their own indices reach the tap with the spec's rule.
    each = dataclasses.replace(spec, pad_position=None)
    got = te.EncoderHandle(m, tok, each, owned=False).encode("ab")
    want = te.tap(m, list(got.ids), layers=(1, 3), pad=got.pads, pad_id=9,
                  pad_position=None)
    assert got.pads == 16 - n and got.template_id == each.template_id
    assert mx.array_equal(got.features,
                          mx.concatenate([w[0] for w in want], -1)).item()
