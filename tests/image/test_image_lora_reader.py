"""The LoRA reader: suffix and prefix detection per key, refusals by
name, and the keys that match nothing."""
from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
import pytest

from gmlx.image.lora import reader
from gmlx.image.lora.reader import (AdapterError, AdapterFamilyError,
                                    LoraKeyMap, UnmatchedKeysError,
                                    UnsupportedAdapterError)
from lora_fixtures import (DIALECTS, dialect_keys, header, make_pair,
                           write_keys, write_ordered)

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


def test_the_header_vet_walks_the_stems_in_file_order(tmp_path):
    """A writer may list the names in any order. The vet walks the stems
    in the header's order, as the read does, so both name the same stems
    in the same places."""
    under = "lora_unet_layers_0_feed_forward_w1"
    stems = [W1, under]
    assert stems != sorted(stems)
    keys = {}
    for stem in stems:
        keys.update(_good(stem))
    path = write_ordered(tmp_path / "twice.safetensors", keys)
    assert list(reader.read_lora(path).pairs) == stems
    with pytest.raises(AdapterError) as want:
        reader.resolve(reader.read_lora(path), KEYMAP)
    assert f"{under} and {W1} both target" in str(want.value)
    with pytest.raises(AdapterError) as got:
        reader.vet_header(path, keymap=KEYMAP)
    assert str(got.value) == str(want.value)
    # The stems that name no module are listed in that order too.
    odd = ["transformer.layers.0.feed_forward.w9",
           "lora_unet_layers_0_feed_forward_w8"]
    keys = {}
    for stem in odd + [W1]:
        keys.update(_good(stem))
    path = write_ordered(tmp_path / "odd.safetensors", keys)
    with pytest.raises(UnmatchedKeysError) as want:
        reader.resolve(reader.read_lora(path), KEYMAP)
    with pytest.raises(UnmatchedKeysError) as got:
        reader.vet_header(path, keymap=KEYMAP)
    assert got.value.keys == want.value.keys == tuple(odd)
    assert str(got.value) == str(want.value)


# The LoRA config in a file's header, as diffusers' save_lora_weights
# writes it for a file without alpha tensors.

Q, K, W1 = ("layers.0.attention.to_q", "layers.0.attention.to_k",
            "layers.0.feed_forward.w1")
CONFIG = {
    "r": 2, "lora_alpha": 1.5, "target_modules": ["to_q", "to_k", "w1"],
    "rank_pattern": {}, "alpha_pattern": {}, "exclude_modules": None,
    "lora_dropout": 0.1, "bias": "none", "init_lora_weights": "gaussian",
    "use_dora": False, "use_rslora": False, "lora_bias": False,
    "fan_in_fan_out": False, "inference_mode": True, "peft_type": "LORA",
    "megatron_core": "megatron.core", "loftq_config": {},
    "qalora_group_size": 16, "runtime_config": {"ephemeral": False},
    "modules_to_save": None, "layers_to_transform": None,
}


def _saved(ranks: dict, alphas: dict | None = None,
           prefix: str = "transformer.") -> dict:
    """Keys of one pair per target at the rank given, spelled as
    save_lora_weights spells them, with an alpha tensor where given."""
    keys = {}
    for i, (target, rank) in enumerate(ranks.items()):
        out, inp = (8, 6) if target != W1 else (12, 6)
        down, up, _ = _pair(i, out, inp, rank)
        keys[f"{prefix}{target}.lora_A.weight"] = mx.array(down)
        keys[f"{prefix}{target}.lora_B.weight"] = mx.array(up)
    for target, alpha in (alphas or {}).items():
        keys[f"{prefix}{target}.alpha"] = mx.array(alpha)
    return keys


def _resolved(keys: dict, fields: dict | None, component="transformer"):
    meta = None if fields is None else header(fields, component)
    return reader.resolve(reader.parse_pairs(keys, "f", metadata=meta),
                          KEYMAP)


def _alphas(res) -> dict:
    return {t.pair.stem.removeprefix("transformer."): t.pair.alpha
            for t in res.dit}


def test_a_header_config_gives_each_pair_its_alpha():
    keys = _saved({Q: 2, K: 3, W1: 2})
    assert _alphas(_resolved(keys, None)) == {Q: None, K: None, W1: None}
    assert _resolved(keys, None).warnings == ()
    # The most common rank is r, another rank is in rank_pattern, and
    # alpha_pattern gives a module an alpha of its own. A pattern matches
    # the end of a module name after a dot, and "^" holds it to the whole
    # name.
    fields = {**CONFIG, "rank_pattern": {"^" + K: 3},
              "alpha_pattern": {"w1": 4.0, "attention.to_k": 6.0}}
    res = _resolved(keys, fields)
    assert _alphas(res) == {Q: 1.5, K: 6.0, W1: 4.0}
    assert [(t.module, t.band) for t in res.dit] == [
        ("layers.0.attention.qkv", (0, 0, 8)),
        ("layers.0.attention.qkv", (0, 8, 8)),
        ("layers.0.feed_forward.w1", None)]
    # ComfyUI reads no header, so the three pairs run at another scale
    # there, and the warning says so.
    assert len(res.warnings) == 1
    assert "3 pairs take an alpha from the adapter header" in res.warnings[0]
    assert "ComfyUI does not read the header" in res.warnings[0]
    # An alpha equal to the rank is the scale 1 that every reader gives.
    plain = _resolved(keys, {**CONFIG, "lora_alpha": 2,
                             "rank_pattern": {K: 3},
                             "alpha_pattern": {K: 3}})
    assert _alphas(plain) == {Q: 2.0, K: 3.0, W1: 2.0}
    assert plain.warnings == ()


@pytest.mark.parametrize("patterns, want", [
    # The first key in header order that matches counts. The writer sorts
    # the keys, so that order is the sorted one.
    ({"to_q": 7.0, "attention.to_q": 9.0}, 9.0),
    ({"to_q": 7.0, "zz": 9.0}, 7.0),
    # "^" anchors a key to the start of the name.
    ({"^attention.to_q": 7.0}, 1.5),
    ({"^" + Q: 7.0}, 7.0),
    # A key matches whole name parts only.
    ({"o_q": 7.0}, 1.5),
    ({"0.attention.to_q": 8.0}, 8.0),
    # A key is a regular expression.
    ({r"layers\.\d+\.attention\.to_(q|k)": 7.0}, 7.0),
    ({r"to_[kv]": 7.0}, 1.5),
])
def test_a_header_pattern_matches_a_module_as_peft_matches_it(patterns,
                                                              want):
    res = _resolved(_saved({Q: 2}), {**CONFIG, "alpha_pattern": patterns})
    assert _alphas(res) == {Q: want}


REFUSED = [
    ("use_dora", True, "DoRA"),
    ("use_rslora", True, "rank-stabilized LoRA"),
    ("lora_bias", True, "LoRA with a bias"),
    ("bias", "all", "LoRA with trained base biases"),
    ("bias", "lora_only", "LoRA with trained base biases"),
    ("fan_in_fan_out", True, "LoRA on a transposed weight"),
    ("use_qalora", True, "QALoRA"),
    ("alora_invocation_tokens", [5, 6], "activated LoRA"),
    ("arrow_config", {"top_k": 3}, "Arrow routing"),
    ("layer_replication", [[0, 2], [1, 3]], "LoRA on replicated layers"),
    ("layers_to_transform", [0], "LoRA on chosen layers"),
    ("layers_pattern", "layers", "LoRA on chosen layers"),
    ("modules_to_save", ["x_embedder"], "LoRA with replaced modules"),
    ("trainable_token_indices", [3], "LoRA with trained token rows"),
    ("target_parameters", ["weight"], "LoRA on bare parameters"),
    ("a_field_of_a_later_writer", True, "adapter header"),
]


@pytest.mark.parametrize("name, value, kind", REFUSED,
                         ids=[f"{n}-{i}" for i, (n, _, _) in
                              enumerate(REFUSED)])
def test_a_header_field_that_changes_the_delta_is_refused_by_name(
        tmp_path, name, value, kind):
    keys = _saved({Q: 2})
    fields = {**CONFIG, name: value}
    said = f"the adapter header sets transformer.{name}"
    with pytest.raises(UnsupportedAdapterError, match=said) as e:
        reader.header_configs(header(fields), "f")
    assert e.value.kind == kind
    assert ("a field gmlx image does not read" if kind == "adapter header"
            else f"which makes this {kind}") in str(e.value)
    with pytest.raises(UnsupportedAdapterError, match=said):
        _resolved(keys, fields)
    # The header alone refuses the file, before a tensor or a model loads.
    path = write_keys(tmp_path / "f.safetensors", keys, header(fields))
    with pytest.raises(UnsupportedAdapterError, match=said):
        reader.vet_header(path)
    with pytest.raises(UnsupportedAdapterError, match=said):
        reader.vet_header(path, keymap=KEYMAP)
    # Left off, the field changes nothing.
    for off in (None, False, [], {}, "", 0,
                *(["none"] if name == "bias" else [])):
        assert _alphas(_resolved(keys, {**CONFIG, name: off})) == {Q: 1.5}


def test_a_header_config_is_read_whole_or_refused():
    keys = _saved({Q: 2})
    assert reader.header_configs(None) == {}
    assert reader.header_configs({"format": "pt"}) == {}
    assert reader.header_configs({reader.HEADER_ENTRY: ""}) == {}
    cfg, = reader.header_configs(header(CONFIG)).values()
    assert (cfg.rank, cfg.alpha, cfg.targets, cfg.excluded) == (
        2, 1.5, ("to_q", "to_k", "w1"), None)
    for entry, match in (
            ("{", "is not JSON"),
            ("[1]", "is not a JSON object"),
            ('{"r": 2}', "names no component and field"),
            ('{".r": 2}', "names no component and field")):
        with pytest.raises(AdapterError, match=match):
            reader.header_configs({reader.HEADER_ENTRY: entry}, "f")
    for change, match in (
            ({"r": None}, "transformer.r as None"),
            ({"r": 0}, "transformer.r as 0"),
            ({"r": 2.5}, "transformer.r as 2.5"),
            ({"r": True}, "transformer.r as True"),
            ({"lora_alpha": None}, "transformer.lora_alpha as None"),
            ({"lora_alpha": "8"}, "transformer.lora_alpha as '8'"),
            ({"lora_alpha": float("nan")}, "not a finite number"),
            ({"lora_alpha": float("inf")}, "not a finite number"),
            ({"target_modules": None}, "gives no transformer.target_modules"),
            ({"target_modules": []}, "gives no transformer.target_modules"),
            ({"target_modules": 3}, "neither a regular expression nor"),
            ({"target_modules": "("}, "is not a regular expression"),
            ({"exclude_modules": ["a", 1]}, "neither a regular expression"),
            ({"rank_pattern": ["to_q"]}, "rank_pattern is not a mapping"),
            ({"rank_pattern": {"to_q": 0}}, "gives to_q the value 0"),
            ({"rank_pattern": {"to_q": 1.5}}, "gives to_q the value 1.5"),
            ({"alpha_pattern": {"to_q": "x"}}, "gives to_q the value 'x'"),
            ({"alpha_pattern": {"(": 1.0}}, "not a regular expression")):
        with pytest.raises(AdapterError, match=match) as e:
            _resolved(keys, {**CONFIG, **change})
        assert not isinstance(e.value, UnsupportedAdapterError), change
    with pytest.raises(AdapterError, match="transformer.r as None"):
        _resolved(keys, {k: v for k, v in CONFIG.items() if k != "r"})


def test_a_pair_that_disagrees_with_the_header_is_refused():
    fields = {**CONFIG, "alpha_pattern": {"to_k": 0.1}}
    # An alpha tensor equal to the header's alpha is the same pair.
    same = _resolved(_saved({Q: 2, K: 2}, {Q: 1.5, K: 0.1}), fields)
    assert _alphas(same) == {Q: 1.5, K: float(np.float32(0.1))}
    assert same.warnings == ()
    # Equal is equal after rounding to the tensor's dtype.
    for dtype in (mx.float16, mx.bfloat16):
        keys = _saved({K: 2}, {K: mx.array(0.1, dtype)})
        alpha, = _alphas(_resolved(keys, fields)).values()
        assert alpha != float(np.float32(0.1)) and abs(alpha - 0.1) < 1e-3
    # Another alpha is refused, and both are named.
    with pytest.raises(AdapterError,
                       match=f"transformer.{Q} has an alpha tensor of 3, "
                             f"and the adapter header gives {Q} the alpha "
                             "1.5"):
        _resolved(_saved({Q: 2}, {Q: 3.0}), fields)
    # A pair of another rank than the header's does not load in the
    # reference.
    with pytest.raises(AdapterError,
                       match=f"transformer.{K} has rank 3, and the adapter "
                             f"header gives {K} the rank 2"):
        _resolved(_saved({Q: 2, K: 3}), fields)
    assert _alphas(_resolved(_saved({Q: 2, K: 3}), {
        **fields, "rank_pattern": {"to_k": 3}})) == {Q: 1.5, K: 0.1}


@pytest.mark.parametrize("change", [
    {"target_modules": ["to_q", "to_k"]},
    {"target_modules": r".*\.attention\.to_[qk]"},
    {"exclude_modules": ["w1"]},
    {"exclude_modules": r".*feed_forward.*"},
])
def test_a_pair_the_header_leaves_out_is_refused(tmp_path, change):
    """The reference loader gives such a module no adapter and drops the
    pair, so a file that has one is refused."""
    keys = _saved({Q: 2, K: 2, W1: 2})
    said = (f"transformer.{W1} is a pair on {W1}, which the "
            "transformer.target_modules and exclude_modules of the adapter "
            "header leave out")
    with pytest.raises(AdapterError, match=said):
        _resolved(keys, {**CONFIG, **change})
    path = write_keys(tmp_path / "f.safetensors", keys,
                      header({**CONFIG, **change}))
    assert reader.vet_header(path) == 3
    with pytest.raises(AdapterError, match=said):
        reader.vet_header(path, keymap=KEYMAP)
    del keys[f"transformer.{W1}.lora_A.weight"]
    del keys[f"transformer.{W1}.lora_B.weight"]
    assert _alphas(_resolved(keys, {**CONFIG, **change})) == {Q: 1.5,
                                                              K: 1.5}


def test_a_header_config_covers_its_own_component(tmp_path):
    te = "text_encoder.model.layers.0.self_attn.q_proj"
    down, up, _ = _pair(5, 8, 6, 4)
    text = {te + ".lora_A.weight": mx.array(down),
            te + ".lora_B.weight": mx.array(up)}
    keys = {**_saved({Q: 2}), **text}
    te_config = {"r": 4, "lora_alpha": 2.0, "target_modules": ["q_proj"]}
    both = header(CONFIG)
    both[reader.HEADER_ENTRY] = json.dumps({
        **json.loads(both[reader.HEADER_ENTRY]),
        **json.loads(header(te_config, "text_encoder")[reader.HEADER_ENTRY])})
    res = reader.resolve(reader.parse_pairs(keys, "f", metadata=both),
                         KEYMAP)
    assert _alphas(res) == {Q: 1.5}
    assert [(t.module, t.pair.alpha) for t in res.text] == [
        ("model.layers.0.self_attn.q_proj", 2.0)]
    # A header with a config for another component alone leaves the
    # reference with nothing to build this one's config from.
    for comp, fields, other, stem in (
            ("transformer", CONFIG, "text_encoder", te),
            ("text_encoder", te_config, "transformer", "transformer." + Q)):
        with pytest.raises(AdapterError,
                           match=f"has a LoRA config for {comp} and none "
                                 f"for {other}, which {stem} belongs to"):
            _resolved(keys, fields, comp)


def test_values_that_are_not_finite_are_refused_at_the_read(tmp_path):
    down, up, _ = _pair()
    d, u = mx.array(down), mx.array(up)
    for alpha in (float("inf"), float("-inf"), float("nan")):
        for dtype in (mx.float32, mx.float16):
            with pytest.raises(AdapterError,
                               match="a.alpha is .*, which is not finite"):
                reader.parse_pairs({"a.lora_A.weight": d,
                                    "a.lora_B.weight": u,
                                    "a.alpha": mx.array(alpha, dtype)}, "f")
    for bad in (float("inf"), float("nan")):
        for dtype in (mx.float32, mx.float16, mx.bfloat16):
            for which in ("lora_A", "lora_B"):
                arr = np.array(down if which == "lora_A" else up)
                arr[-1, -1] = bad
                keys = {"a.lora_A.weight": d, "a.lora_B.weight": u,
                        "b.lora_A.weight": d, "b.lora_B.weight": u,
                        f"b.{which}.weight": mx.array(arr).astype(dtype)}
                with pytest.raises(
                        AdapterError,
                        match=f"f: b.{which}.weight holds a value that is "
                              "not finite"):
                    reader.parse_pairs(keys, "f")
    # A file is refused the same way, and the largest finite values pass.
    arr = np.array(up)
    arr[0, 0] = np.nan
    path = write_keys(tmp_path / "nan.safetensors", {
        f"transformer.{Q}.lora_A.weight": down,
        f"transformer.{Q}.lora_B.weight": arr})
    with pytest.raises(AdapterError, match="holds a value that is not "
                                           "finite"):
        reader.read_lora(path)
    arr[0, 0] = np.finfo(np.float32).max
    raw = reader.parse_pairs({"a.lora_A.weight": d,
                              "a.lora_B.weight": mx.array(arr),
                              "a.alpha": mx.array(3.0e38)}, "f")
    assert raw.pairs["a"].alpha == float(np.float32(3.0e38))
