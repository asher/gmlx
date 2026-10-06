"""The LoRA reader: suffix and prefix detection per key, refusals by
name, and the keys that match nothing."""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from gmlx.image.lora import reader
from gmlx.image.lora.reader import (AdapterError, AdapterFamilyError,
                                    LoraKeyMap, UnmatchedKeysError,
                                    UnsupportedAdapterError)
from lora_fixtures import DIALECTS, dialect_keys, make_pair, write_keys

KEYMAP = LoraKeyMap(
    family="toy", title="Toy",
    targets={
        "layers.0.attention.to_q": ("layers.0.attention.qkv", (0, 0, 8)),
        "layers.0.attention.to_k": ("layers.0.attention.qkv", (0, 8, 8)),
        "layers.0.attention.qkv": ("layers.0.attention.qkv", None),
        "layers.0.feed_forward.w1": ("layers.0.feed_forward.w1", None),
        "all_x_embedder.2-1": ("x_embedder", None),
    },
    diffusers_only=frozenset({"layers.0.attention.to_q",
                              "layers.0.attention.to_k",
                              "all_x_embedder.2-1"}),
    text_names=("toy_te",),
    text_targets=("model.layers.0.self_attn.q_proj",
                  "model.layers.1.mlp.up_proj"))

SUFFIX_PAIRS = (
    (".lora_down.weight", ".lora_up.weight"),
    (".lora_A.weight", ".lora_B.weight"),
    (".lora_A.default.weight", ".lora_B.default.weight"),
    ("_lora.down.weight", "_lora.up.weight"),
    (".lora.down.weight", ".lora.up.weight"),
    (".lora_linear_layer.down.weight", ".lora_linear_layer.up.weight"),
    (".lora_A", ".lora_B"),
)


def _pair(seed=0, out=8, inp=6, rank=2, alpha=None):
    return make_pair(np.random.default_rng(seed), out, inp, rank, alpha)


@pytest.mark.parametrize("down_s,up_s", SUFFIX_PAIRS)
@pytest.mark.parametrize("alpha", (None, 4.0))
def test_every_suffix_spelling_reads_as_the_same_roles(down_s, up_s, alpha):
    down, up, _ = _pair()
    keys = {"some.module" + down_s: mx.array(down),
            "some.module" + up_s: mx.array(up)}
    if alpha is not None:
        keys["some.module.alpha"] = mx.array(alpha)
    raw = reader.parse_pairs(keys, "x.safetensors")
    assert list(raw.pairs) == ["some.module"]
    pair = raw.pairs["some.module"]
    assert np.array_equal(np.array(pair.down), down)
    assert np.array_equal(np.array(pair.up), up)
    assert pair.alpha == alpha and pair.rank == 2


@pytest.mark.parametrize("stem,module,band", (
    ("model.diffusion_model.layers.0.attention.qkv",
     "layers.0.attention.qkv", None),
    ("diffusion_model.layers.0.attention.to_k", "layers.0.attention.qkv",
     (0, 8, 8)),
    ("transformer.layers.0.attention.to_q", "layers.0.attention.qkv",
     (0, 0, 8)),
    ("base_model.model.layers.0.feed_forward.w1", "layers.0.feed_forward.w1",
     None),
    ("base_model.model.transformer.layers.0.feed_forward.w1",
     "layers.0.feed_forward.w1", None),
    ("unet.layers.0.feed_forward.w1", "layers.0.feed_forward.w1", None),
    ("layers.0.feed_forward.w1", "layers.0.feed_forward.w1", None),
    ("lora_unet_layers_0_feed_forward_w1", "layers.0.feed_forward.w1", None),
    ("lycoris_layers_0_attention_to_q", "layers.0.attention.qkv", (0, 0, 8)),
    ("lora_unet_all_x_embedder_2-1", "x_embedder", None),
    ("transformer.all_x_embedder.2-1", "x_embedder", None),
))
def test_every_prefix_resolves_to_its_module_and_band(stem, module, band):
    down, up, _ = _pair()
    raw = reader.parse_pairs({stem + ".lora_A.weight": mx.array(down),
                              stem + ".lora_B.weight": mx.array(up)}, "f")
    res = reader.resolve(raw, KEYMAP)
    assert [(t.module, t.band) for t in res.dit] == [(module, band)]
    assert res.text == () and res.unmatched == ()


@pytest.mark.parametrize("dialect", sorted(DIALECTS))
def test_each_dialect_file_reads_and_resolves(tmp_path, dialect):
    rng = np.random.default_rng(1)
    alpha = 3.0 if DIALECTS[dialect][4] else None
    adapter = {"layers.0.attention.to_q": make_pair(rng, 8, 6, 2, alpha),
               "layers.0.attention.to_k": make_pair(rng, 8, 6, 4, alpha),
               "layers.0.feed_forward.w1": make_pair(rng, 5, 6, 2, alpha)}
    path = write_keys(tmp_path / "a.safetensors",
                      dialect_keys(dialect, adapter))
    raw = reader.read_lora(path)
    assert len(raw.pairs) == 3
    res = reader.resolve(raw, KEYMAP)
    got = {(t.module, t.band): t.pair for t in res.dit}
    assert set(got) == {("layers.0.attention.qkv", (0, 0, 8)),
                        ("layers.0.attention.qkv", (0, 8, 8)),
                        ("layers.0.feed_forward.w1", None)}
    k = got[("layers.0.attention.qkv", (0, 8, 8))]
    assert k.rank == 4 and k.alpha == alpha
    assert np.array_equal(np.array(k.up),
                          adapter["layers.0.attention.to_k"][1])
    # Underscore keys that spell diffusers names are the ones ComfyUI drops.
    warned = any("ComfyUI" in w for w in res.warnings)
    assert warned == (dialect == "kohya")


@pytest.mark.parametrize("prefix,warned", (
    ("lora_unet_", True), ("lora_transformer_", True), ("lycoris_", False)))
def test_the_comfyui_warning_is_for_the_prefixes_comfyui_drops(prefix,
                                                                warned):
    # ComfyUI maps the diffusers names under lycoris_ and under no other
    # underscore prefix.
    down, up, _ = _pair()
    keys = {}
    for name in ("layers_0_attention_to_q", "layers_0_attention_to_k",
                 "layers_0_feed_forward_w1"):
        keys[f"{prefix}{name}.lora_down.weight"] = mx.array(down)
        keys[f"{prefix}{name}.lora_up.weight"] = mx.array(up)
    res = reader.resolve(reader.parse_pairs(keys, "f.safetensors"), KEYMAP)
    assert len(res.dit) == 3 and res.unmatched == ()
    notes = [w for w in res.warnings if "ComfyUI" in w]
    assert len(notes) == (1 if warned else 0)
    if warned:
        assert "2 underscore keys" in notes[0]


@pytest.mark.parametrize("part,kind", (
    ("hada_w1_a", "LoHa"), ("hada_w2_b", "LoHa"),
    ("lokr_w1", "LoKr"), ("lokr_w2", "LoKr"), ("lokr_w2_a", "LoKr"),
    ("dora_scale", "DoRA"), ("lora_magnitude_vector", "DoRA"),
    ("lora_magnitude_vector.default", "DoRA"),
    ("lora_magnitude_vector.default.weight", "DoRA"),
    ("diff", "full-weight diff"), ("diff_b", "full-weight diff"),
    ("set_weight", "full-weight diff"),
    ("lora_mid.weight", "Tucker LoCon"),
))
def test_other_adapter_types_are_refused_by_name(tmp_path, monkeypatch,
                                                 part, kind):
    down, up, _ = _pair()
    stem = "lora_unet_layers_0_feed_forward_w1"
    keys = {stem + ".lora_down.weight": down, stem + ".lora_up.weight": up,
            f"{stem}.{part}": np.ones((2, 2), np.float32)}
    path = write_keys(tmp_path / "a.safetensors", keys)
    # The refusal comes from the header, before any tensor is read.
    monkeypatch.setattr(mx, "load", lambda *a, **k: pytest.fail("loaded"))
    with pytest.raises(UnsupportedAdapterError, match=kind) as e:
        reader.read_lora(path)
    assert e.value.kind == kind and f"{stem}.{part}" in str(e.value)
    with pytest.raises(UnsupportedAdapterError, match=kind):
        reader.parse_pairs({k: mx.array(v) for k, v in keys.items()})


def test_keys_with_no_known_suffix_raise_and_list_the_first_few():
    down, up, _ = _pair()
    keys = {"a.lora_A.weight": mx.array(down),
            "a.lora_B.weight": mx.array(up)}
    keys.update({f"a.mystery_{i}": mx.array(down) for i in range(6)})
    with pytest.raises(UnmatchedKeysError) as e:
        reader.parse_pairs(keys, "f.safetensors")
    msg = str(e.value)
    assert "6 of 8 keys" in msg and "a.mystery_0" in msg
    assert "a.mystery_3" in msg and "a.mystery_4" not in msg
    assert "and 2 more" in msg and len(e.value.keys) == 6
    raw = reader.parse_pairs(keys, "f.safetensors", allow_unmatched=True)
    assert list(raw.pairs) == ["a"] and len(raw.stray) == 6


def test_targets_that_name_no_module_raise_and_list_the_first_few():
    down, up, _ = _pair()
    keys = {}
    stems = ["transformer.layers.0.feed_forward.w1"] + [
        f"transformer.layers.{i}.feed_forward.w9" for i in range(5)]
    for s in stems:
        keys[s + ".lora_A.weight"] = mx.array(down)
        keys[s + ".lora_B.weight"] = mx.array(up)
    raw = reader.parse_pairs(keys, "f.safetensors")
    with pytest.raises(UnmatchedKeysError) as e:
        reader.resolve(raw, KEYMAP)
    msg = str(e.value)
    assert "5 of 6 targets name no Toy module" in msg
    assert stems[1] in msg and stems[4] in msg and stems[5] not in msg
    assert e.value.keys == tuple(stems[1:])
    res = reader.resolve(raw, KEYMAP, allow_unmatched=True)
    assert len(res.dit) == 1 and res.unmatched == tuple(stems[1:])
    assert any("skipped" in w for w in res.warnings)


def test_an_adapter_for_another_family_is_refused():
    down, up, _ = _pair()
    keys = {}
    for s in ("transformer.transformer_blocks.0.attn.to_q",
              "lora_unet_double_blocks_0_img_attn_qkv"):
        keys[s + ".lora_down.weight"] = mx.array(down)
        keys[s + ".lora_up.weight"] = mx.array(up)
    raw = reader.parse_pairs(keys, "flux.safetensors")
    for allow in (False, True):
        with pytest.raises(AdapterFamilyError, match="not a Toy adapter"):
            reader.resolve(raw, KEYMAP, allow_unmatched=allow)


def test_broken_pairs_are_refused():
    down, up, _ = _pair()
    d, u = mx.array(down), mx.array(up)
    for keys, match in (
            ({"a.lora_A.weight": d}, "no up tensor"),
            ({"a.lora_B.weight": u}, "no down tensor"),
            ({"a.alpha": mx.array(1.0)}, "an alpha and no pair"),
            ({"a.lora_A.weight": d, "a.lora_B.weight": u[:, :1]},
             "ranks differ"),
            ({"a.lora_A.weight": d, "a.lora_down.weight": d,
              "a.lora_B.weight": u}, "two down tensors"),
            ({"a.lora_A.weight": d, "a.lora_B.weight": u,
              "a.alpha": mx.array([1.0, 2.0])}, "not a scalar"),
            ({"a.lora_A.weight": mx.zeros((2, 6, 3, 3)),
              "a.lora_B.weight": u}, "linear layers only")):
        with pytest.raises(AdapterError, match=match):
            reader.parse_pairs(keys, "f")
    with pytest.raises(AdapterError, match="holds no LoRA pair"):
        reader.parse_pairs({}, "f")


def test_a_one_by_one_kernel_pair_reads_as_a_matrix():
    down, up, _ = _pair()
    raw = reader.parse_pairs({
        "a.lora_down.weight": mx.array(down.reshape(2, 6, 1, 1)),
        "a.lora_up.weight": mx.array(up.reshape(8, 2, 1, 1))}, "f")
    assert raw.pairs["a"].down.shape == (2, 6)
    assert raw.pairs["a"].up.shape == (8, 2)


@pytest.mark.parametrize("stem,path", (
    ("text_encoders.toy_te.transformer.model.layers.0.self_attn.q_proj",
     "model.layers.0.self_attn.q_proj"),
    ("text_encoders.toy_te.model.layers.1.mlp.up_proj",
     "model.layers.1.mlp.up_proj"),
    ("text_encoder.layers.0.self_attn.q_proj",
     "model.layers.0.self_attn.q_proj"),
    ("text_encoder.model.layers.0.self_attn.q_proj",
     "model.layers.0.self_attn.q_proj"),
    ("lora_te_layers_1_mlp_up_proj", "model.layers.1.mlp.up_proj"),
    ("lora_te1_model_layers_0_self_attn_q_proj",
     "model.layers.0.self_attn.q_proj"),
))
def test_text_encoder_keys_resolve_to_the_text_half(stem, path):
    down, up, _ = _pair()
    raw = reader.parse_pairs({stem + ".lora_down.weight": mx.array(down),
                              stem + ".lora_up.weight": mx.array(up)}, "f")
    res = reader.resolve(raw, KEYMAP)
    assert res.dit == () and [t.module for t in res.text] == [path]
    bad = stem.replace("q_proj", "z_proj").replace("up_proj", "z_proj")
    raw = reader.parse_pairs({bad + ".lora_down.weight": mx.array(down),
                              bad + ".lora_up.weight": mx.array(up)}, "f")
    with pytest.raises(AdapterFamilyError):
        reader.resolve(raw, KEYMAP)


def test_two_keys_on_one_target_are_refused():
    down, up, _ = _pair()
    keys = {}
    for s in ("transformer.layers.0.feed_forward.w1",
              "lora_unet_layers_0_feed_forward_w1"):
        keys[s + ".lora_down.weight"] = mx.array(down)
        keys[s + ".lora_up.weight"] = mx.array(up)
    with pytest.raises(AdapterError, match="both target"):
        reader.resolve(reader.parse_pairs(keys, "f"), KEYMAP)


def test_a_file_that_is_not_safetensors_is_refused(tmp_path):
    p = tmp_path / "a.gguf"
    p.write_bytes(b"GGUF" + b"\0" * 64)
    with pytest.raises(AdapterError, match="not a safetensors file"):
        reader.read_lora(str(p))
    with pytest.raises(AdapterError, match="No such file"):
        reader.read_lora(str(tmp_path / "missing.safetensors"))


def test_the_header_vet_places_stems_on_the_family_and_the_dit(tmp_path):
    """With a key map the vet refuses, from the header alone, what
    ``resolve`` and the load refuse: a file for another family, a stem
    that names no module, and a module the DiT file does not hold."""
    down, up, _ = _pair()
    held = {"layers.0.attention.qkv", "layers.0.feed_forward.w1",
            "x_embedder"}

    def write(name, stems):
        keys = {}
        for stem in stems:
            keys[stem + ".lora_A.weight"] = down
            keys[stem + ".lora_B.weight"] = up
        return write_keys(tmp_path / name, keys)

    good = write("good.safetensors", [
        "transformer.layers.0.attention.to_q",
        "lora_unet_layers_0_feed_forward_w1",
        "text_encoders.toy_te.transformer.model.layers.0.self_attn.q_proj"])
    assert reader.vet_header(good) == 3
    assert reader.vet_header(good, keymap=KEYMAP, modules=held) == 3
    flux = write("flux.safetensors", [
        "transformer.transformer_blocks.0.attn.to_q",
        "lora_unet_double_blocks_0_img_attn_qkv"])
    assert reader.vet_header(flux) == 2
    for allow in (False, True):
        with pytest.raises(AdapterFamilyError, match="not a Toy adapter"):
            reader.vet_header(flux, keymap=KEYMAP, allow_unmatched=allow)
    extra = write("extra.safetensors", [
        "transformer.layers.0.feed_forward.w1",
        "transformer.layers.0.feed_forward.w9"])
    with pytest.raises(UnmatchedKeysError) as e:
        reader.vet_header(extra, keymap=KEYMAP)
    assert "1 of 2 targets name no Toy module" in str(e.value)
    assert e.value.keys == ("transformer.layers.0.feed_forward.w9",)
    assert reader.vet_header(extra, keymap=KEYMAP, allow_unmatched=True) == 2
    # The map has the target and the DiT file holds no weight for it.
    with pytest.raises(AdapterError, match="layers.0.feed_forward.w1, "
                                           "which this DiT does not have"):
        reader.vet_header(good, keymap=KEYMAP,
                          modules=held - {"layers.0.feed_forward.w1"})


W1 = "transformer.layers.0.feed_forward.w1"
TO_Q = "transformer.layers.0.attention.to_q"
TEXT_Q = "text_encoder.layers.0.self_attn.q_proj"


def _good(stem: str) -> dict:
    down, up, _ = _pair()
    return {stem + ".lora_down.weight": down, stem + ".lora_up.weight": up}


def test_the_header_vet_places_text_stems_on_the_family_encoder(tmp_path):
    """A text stem counts only when it names a layer of this family's
    text encoder. A text-only file for another encoder is a file for
    another family, and beside a DiT stem its text stems are unmatched,
    as ``resolve`` finds after the read."""
    clip = ["lora_te1_text_model_encoder_layers_0_self_attn_q_proj",
            "text_encoders.clip_l.transformer.text_model.encoder.layers.0"
            ".mlp.fc1"]
    other = write_keys(tmp_path / "clip.safetensors",
                       {k: v for s in clip for k, v in _good(s).items()})
    mixed = write_keys(tmp_path / "mixed.safetensors", {
        **_good(W1), **_good(clip[0])})
    own = write_keys(tmp_path / "own.safetensors", _good(TEXT_Q))
    for path, kind, text in (
            (other, AdapterFamilyError, "clip.safetensors is not a Toy "
                                        "adapter: none of its 2 targets"),
            (mixed, UnmatchedKeysError, "1 of 2 targets name no Toy "
                                        f"module: {clip[0]}")):
        for allow in (False, True):
            if allow and kind is UnmatchedKeysError:
                assert reader.vet_header(path, keymap=KEYMAP,
                                         allow_unmatched=True) == 2
                continue
            with pytest.raises(kind) as want:
                reader.resolve(reader.read_lora(path), KEYMAP,
                               allow_unmatched=allow)
            assert text in str(want.value)
            with pytest.raises(kind) as got:
                reader.vet_header(path, keymap=KEYMAP, allow_unmatched=allow)
            assert type(got.value) is kind
            assert str(got.value) == str(want.value)
    assert reader.vet_header(own, keymap=KEYMAP) == 1
    assert reader.vet_header(own, keymap=KEYMAP, modules=set()) == 1


@pytest.mark.parametrize("keys, text", [
    ({**_good(W1), TO_Q + ".lora_down.weight": _pair()[0]},
     f"{TO_Q} has no up tensor"),
    ({**_good(W1), TO_Q + ".lora_up.weight": _pair()[1]},
     f"{TO_Q} has no down tensor"),
    ({**_good(W1), TO_Q + ".alpha": np.float32(4.0)},
     f"{TO_Q} has an alpha and no pair"),
    ({TO_Q + ".alpha": np.float32(4.0)}, f"{TO_Q} has an alpha and no pair"),
    ({**_good(W1), W1 + ".lora_A.weight": _pair()[0]},
     f"{W1} has two down tensors"),
], ids=["no up", "no down", "alpha beside a pair", "alpha alone",
        "two down"])
def test_the_header_vet_refuses_a_broken_pair_as_the_read_does(tmp_path,
                                                               keys, text):
    """The names in the header show a stem with half a pair, with an alpha
    alone or with two tensors of one role. The vet refuses each with the
    error the read gives, with or without a key map."""
    path = write_keys(tmp_path / "broken.safetensors", keys)
    with pytest.raises(AdapterError) as want:
        reader.read_lora(path)
    assert text in str(want.value)
    for kw in ({}, {"allow_unmatched": True}, {"keymap": KEYMAP},
               {"keymap": KEYMAP, "modules": {"layers.0.feed_forward.w1"}}):
        with pytest.raises(AdapterError) as got:
            reader.vet_header(path, **kw)
        assert type(got.value) is type(want.value)
        assert str(got.value) == str(want.value)


@pytest.mark.parametrize("stems, text", [
    ((W1, "lora_unet_layers_0_feed_forward_w1"),
     "both target layers.0.feed_forward.w1"),
    ((TO_Q, "lora_unet_layers_0_attention_to_q", W1),
     "both target layers.0.attention.qkv band (0, 0, 8)"),
    ((W1, TEXT_Q, "lora_te_layers_0_self_attn_q_proj"),
     "both target model.layers.0.self_attn.q_proj"),
], ids=["whole weight", "band", "text"])
def test_the_header_vet_refuses_two_stems_on_one_module(tmp_path, stems,
                                                        text):
    """Two stems that resolve to one module and band are refused by the
    vet with the error ``resolve`` gives, in the stems' file order."""
    keys = {}
    for stem in stems:
        keys.update(_good(stem))
    path = write_keys(tmp_path / "twice.safetensors", keys)
    with pytest.raises(AdapterError) as want:
        reader.resolve(reader.read_lora(path), KEYMAP)
    assert text in str(want.value)
    # Without a key map the vet cannot place a stem.
    assert reader.vet_header(path) == len(stems)
    for kw in ({}, {"allow_unmatched": True},
               {"modules": {"layers.0.feed_forward.w1"}}):
        with pytest.raises(AdapterError) as got:
            reader.vet_header(path, keymap=KEYMAP, **kw)
        assert type(got.value) is type(want.value)
        assert str(got.value) == str(want.value)
    # The band targets of one module are not one target.
    apart = {}
    for stem in (TO_Q, "transformer.layers.0.attention.to_k",
                 "transformer.layers.0.attention.qkv"):
        apart.update(_good(stem))
    path = write_keys(tmp_path / "apart.safetensors", apart)
    assert reader.vet_header(path, keymap=KEYMAP) == 3
