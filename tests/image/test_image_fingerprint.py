"""Image GGUF fingerprints: family rules, relatives refused by name, config
synthesis against the official configs, and the variant identity hash.
CPU-only. Header fixtures hold names, shapes and codecs, no weights."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from image_fixtures import (
    FIXTURES, Q_HEADER, Z_HEADER, bf16_bits, fixture_view, load_json,
    official_config, write_gguf)

from gmlx.image import fingerprint as fp
from gmlx.image.registry import VariantRow, families, get_family, is_image_arch

_QB0 = "model.diffusion_model.transformer_blocks.0."


def _refused(view, **kw) -> str:
    with pytest.raises(fp.ImageModelError) as ei:
        fp.identify(view, **kw)
    return str(ei.value)


def test_families_sort_by_order_and_name():
    assert [f.name for f in families()] == ["z_image", "qwen_image21"]
    assert get_family("z_image").title == "Z-Image"
    with pytest.raises(ValueError, match="known: z_image, qwen_image21"):
        get_family("flux")
    assert is_image_arch("lumina2") and is_image_arch("qwen_image")
    assert not is_image_arch("llama") and not is_image_arch(None)


def test_z_image_header_matches_and_config_equals_the_official_one():
    got = fp.identify(fixture_view(Z_HEADER))
    assert got.family.name == "z_image"
    assert (got.prefix, got.layout) == ("", "default")
    # A fixture view has no file bytes, so no identity hash.
    assert (got.variant, got.identity_hash) == ("unknown", None)
    official = official_config("z_image_turbo_transformer_config.json")
    assert {k: got.config[k] for k in official} == official
    assert got.config["ffn_dim"] == 10240
    assert got.config["adaln_dim"] == 256
    assert got.config["head_dim"] == 128


def test_qwen_image21_header_matches_and_config_equals_the_official_one():
    got = fp.identify(fixture_view(Q_HEADER))
    assert got.family.name == "qwen_image21"
    assert (got.prefix, got.layout) == ("model.diffusion_model.", "fused")
    assert got.variant == "default"
    assert "img_in.weight" in got.view.tensors
    official = official_config("qwen_image21_transformer_config.json")
    assert {k: got.config[k] for k in official} == official


def test_qwen_image21_unfused_mlp_is_its_own_layout():
    base = fixture_view(Q_HEADER)
    tensors = {}
    for n, t in base.tensors.items():
        if n.endswith("img_mlp.gate_up.weight"):
            stem = n[: -len("gate_up.weight")]
            tensors[stem + "gate_layer.weight"] = fp.Tensor((4096, 12288), t.codec)
            tensors[stem + "proj.weight"] = fp.Tensor((4096, 12288), t.codec)
        else:
            tensors[n] = t
    got = fp.identify(fp.HeaderView(kv={}, tensors=tensors))
    assert got.layout == "unfused"
    assert got.config["mlp_ratio"] == 3


def test_the_constants_rows_name_the_fixture_revisions():
    manifest = load_json("manifest.json")
    for fam, cfg in (("z_image", "z_image_turbo_transformer_config.json"),
                     ("qwen_image21", "qwen_image21_transformer_config.json")):
        row = get_family(fam).constants
        assert (row.repo, row.revision) == (
            manifest[cfg]["repo"], manifest[cfg]["revision"])


# transformer/config.json of Qwen/Qwen-Image-2.1 at the manifest's revision.
QWEN_CONFIG_SHA256 = (
    "56ae3281c4e6c2d1aa3658252d187488071815fd79bef15808bb0205fc1a2241")


def test_the_qwen_config_fixture_is_the_official_file_in_bytes():
    data = (FIXTURES / "qwen_image21_transformer_config.json").read_bytes()
    assert hashlib.sha256(data).hexdigest() == QWEN_CONFIG_SHA256


def test_header_fixtures_are_small_and_pinned():
    manifest = load_json("manifest.json")
    for name in (Z_HEADER, Q_HEADER):
        d = load_json(name)
        assert len(d["tensors"]) == d["n_tensors"]
        assert len(manifest[name]["revision"]) == 40
        assert len(manifest[name]["sha256"]) == 64
    assert load_json(Z_HEADER)["n_tensors"] == 453
    assert load_json(Q_HEADER)["n_tensors"] == 265


@pytest.mark.parametrize("key,name", [
    ("clip_text_pooled_proj.0.weight", "NewBie"),
    ("siglip_embedder.0.weight", "Z-Image Omni or Edit"),
    ("dec_net.cond_embed.weight", "Z-Image pixel space"),
])
def test_z_image_relatives_are_refused_by_name(key, name):
    msg = _refused(fixture_view(Z_HEADER, add={key: ((8, 8), "F32")}))
    assert msg == (f"this is a {name} GGUF, a Z-Image relative that gmlx "
                   "image does not support")


def test_lumina_image_2_is_refused_by_its_width():
    view = fixture_view(Z_HEADER, add={
        "cap_embedder.1.weight": ((2304, 2304), "BF16")})
    assert "Lumina-Image 2.0" in _refused(view)


def test_a_file_without_pad_tokens_is_ming_image():
    view = fixture_view(Z_HEADER, drop=("x_pad_token", "cap_pad_token"))
    assert "Ming-Image" in _refused(view)


def test_a_z_image_shape_mismatch_names_the_tensor_and_the_count():
    view = fixture_view(Z_HEADER, drop=("layers.29.",), add={
        "x_embedder.weight": ((64, 4096), "BF16")})
    msg = _refused(view)
    assert msg.startswith("a Z-Image relative that gmlx image does not support")
    assert "x_embedder.weight is (64, 4096), Z-Image has (64, 3840)" in msg
    assert "layers has 29 blocks, Z-Image has 30" in msg


def test_qwen_relatives_are_refused_by_name():
    lens = fixture_view(Q_HEADER, add={
        _QB0 + "attn.norm_added_q.weight": ((128,), "F32"),
        _QB0 + "img_mlp.w1.weight": ((4096, 12288), "Q4_K")})
    assert "a Lens GGUF" in _refused(lens)
    mage = fixture_view(Q_HEADER, add={
        "model.diffusion_model.txt_norm.weight": ((2560,), "F32"),
        "model.diffusion_model.proj_out.weight": ((4096, 128), "Q8_0")})
    assert "a Mage-Flow GGUF" in _refused(mage)
    q20 = {"model.diffusion_model.txt_norm.weight": ((3584,), "F32"),
           _QB0 + "attn.add_q_proj.weight": ((3072, 3072), "Q4_K")}
    assert "a Qwen-Image 20B GGUF" in _refused(fixture_view(Q_HEADER, add=q20))
    edit = dict(q20, **{"model.diffusion_model.__index_timestep_zero__":
                        ((1,), "F32")})
    assert "Qwen-Image-Edit 2511" in _refused(fixture_view(Q_HEADER, add=edit))
    layered = dict(q20, **{
        "model.diffusion_model.time_text_embed.addition_t_embedding.weight":
            ((4096, 2), "F32")})
    assert "Qwen-Image-Layered" in _refused(fixture_view(Q_HEADER, add=layered))


def test_a_qwen_file_with_biases_is_refused_with_the_reason():
    view = fixture_view(Q_HEADER, add={
        "model.diffusion_model.img_in.bias": ((4096,), "F32")})
    msg = _refused(view)
    assert msg.startswith("a Qwen-Image relative")
    assert "it has bias tensors" in msg


def test_a_text_arch_is_sent_to_load_model():
    view = fixture_view(Z_HEADER, kv={"general.architecture": "qwen3"})
    assert _refused(view) == "arch 'qwen3' is a text model, use gmlx.load_model"


def test_an_arch_that_contradicts_the_layout_names_both():
    view = fixture_view(Z_HEADER, kv={"general.architecture": "qwen_image"})
    assert _refused(view) == ("arch 'qwen_image' contradicts the tensor "
                              "layout, which is Z-Image (arch 'lumina2' or "
                              "none)")


def test_other_diffusion_families_are_refused_by_name():
    view = fp.HeaderView(kv={"general.architecture": "flux"}, tensors={
        "double_blocks.0.img_attn.proj.weight": fp.Tensor((3072, 3072), "Q4_K"),
        "img_in.weight": fp.Tensor((64, 3072), "BF16")})
    assert _refused(view) == ("this is a FLUX GGUF, which gmlx image does "
                              "not support")
    wan = fp.HeaderView(kv={}, tensors={
        "model.diffusion_model." + n: fp.Tensor((8, 8), "F32") for n in (
            "blocks.0.self_attn.norm_q.weight", "text_embedding.2.weight",
            "head.modulation")})
    assert "a Wan GGUF" in _refused(wan)


# The keys ComfyUI-GGUF's convert.py bans, which mark diffusers layouts.
_BANNED = ("transformer_blocks.0.attn.norm_added_k.weight",
           "transformer_blocks.0.attn.add_q_proj.weight",
           "joint_transformer_blocks.3.ff_context.out_projection.weight")


def test_no_other_family_row_uses_a_banned_key():
    keys = {k for _, sets in fp._OTHER_FAMILIES for ks in sets for k in ks}
    assert not keys & set(_BANNED)


def test_a_diffusers_flux_file_is_not_named_sd3():
    b0 = "transformer_blocks.0.attn."
    view = fp.HeaderView(kv={}, tensors={
        "x_embedder.weight": fp.Tensor((64, 3072), "BF16"),
        "context_embedder.weight": fp.Tensor((4096, 3072), "Q4_K"),
        b0 + "add_q_proj.weight": fp.Tensor((3072, 3072), "Q4_K"),
        b0 + "norm_added_k.weight": fp.Tensor((128,), "F32"),
        "single_transformer_blocks.0.attn.to_q.weight":
            fp.Tensor((3072, 3072), "Q4_K")})
    msg = _refused(view)
    assert "Stable Diffusion 3" not in msg
    assert msg.startswith("unrecognized image GGUF: arch=none, 5 tensors")


def test_the_reference_sd3_and_auraflow_layouts_keep_their_names():
    sd3 = fp.HeaderView(kv={}, tensors={
        "joint_blocks.0.x_block.attn.qkv.weight": fp.Tensor((8, 24), "F32")})
    assert "a Stable Diffusion 3 GGUF" in _refused(sd3)
    aura = fp.HeaderView(kv={}, tensors={
        "double_layers.3.modX.1.weight": fp.Tensor((8, 8), "F32")})
    assert "this is an AuraFlow GGUF" in _refused(aura)


def test_an_unknown_file_is_described():
    tensors = {f"blocks.{i}.attn.qkv.weight": fp.Tensor((5120, 15360), "Q4_K")
               for i in range(3)}
    tensors.update({f"blocks.{i}.norm.weight": fp.Tensor((5120,), "F32")
                    for i in range(3)})
    tensors["head.weight"] = fp.Tensor((5120, 64), "F32")
    msg = _refused(fp.HeaderView(kv={}, tensors=tensors))
    assert msg == (
        "unrecognized image GGUF: arch=none, 7 tensors, codecs F32 x4 "
        "Q4_K x3, groups blocks x3 head x1, shapes blocks.0.attn.qkv.weight "
        "(5120, 15360), blocks.1.attn.qkv.weight (5120, 15360), "
        "blocks.2.attn.qkv.weight (5120, 15360), head.weight (5120, 64)")


def test_a_forced_family_that_does_not_match_is_refused():
    msg = _refused(fixture_view(Z_HEADER), family="qwen_image21")
    assert msg.startswith("not a Qwen-Image 2.1 file. unrecognized image GGUF")


def test_prefix_strip_collision_is_refused():
    view = fp.HeaderView(kv={}, tensors={
        "x_embedder.weight": fp.Tensor((64, 3840), "BF16"),
        "model.diffusion_model.x_embedder.weight": fp.Tensor((64, 3840), "BF16")})
    with pytest.raises(fp.ImageModelError, match="prefix strip collision"):
        fp.strip_prefix(view, ("model.diffusion_model.",))


def test_canonical_bf16_rounds_to_nearest_even():
    vals = np.array([1.0, -2.5, 3.140625, 1e-3], np.float32)
    exact = bf16_bits(vals[:3]).tobytes()
    assert fp.canonical_bf16(vals[:3].tobytes(), "F32") == exact
    assert fp.canonical_bf16(exact, "BF16") == exact
    assert fp.canonical_bf16(vals[:3].astype(np.float16).tobytes(),
                             "F16") == exact
    # 1 + 2^-8 is a tie between bf16 1.0 and 1 + 2^-7: even keeps 1.0.
    # 1 + 3 * 2^-8 ties upward to 1 + 2^-6.
    ties = np.array([1 + 2 ** -8, 1 + 3 * 2 ** -8], np.float32)
    got = np.frombuffer(fp.canonical_bf16(ties.tobytes(), "F32"), np.uint16)
    assert got.tolist() == [0x3F80, 0x3F82]
    with pytest.raises(fp.ImageModelError, match="stored as Q8_0"):
        fp.canonical_bf16(b"\0" * 34, "Q8_0")


def _identity_file(tmp_path, codec: str, seed: int = 0) -> fp.HeaderView:
    from gmlx.load.headerscan import scan_gguf

    rng = np.random.default_rng(seed)
    vals = {n: bf16_bits(rng.standard_normal(shape).astype(np.float32))
            for n, shape in (("x_pad_token", (1, 3840)),
                             ("cap_pad_token", (1, 3840)),
                             ("final_layer.linear.bias", (64,)))}
    tensors = {}
    for n, bits in vals.items():
        f32 = (bits.astype(np.uint32) << 16).view(np.float32)
        tensors[n] = (bits, "BF16") if codec == "BF16" else (f32, "F32")
    p = write_gguf(tmp_path / f"id-{codec}-{seed}.gguf", tensors, "lumina2")
    return fp.view_from_scans([scan_gguf(p)])


def test_identity_hash_is_the_same_for_f32_and_bf16_storage(tmp_path):
    fam = get_family("z_image")
    h32 = fp.weight_identity(_identity_file(tmp_path, "F32"),
                             fam.identity_tensors)
    h16 = fp.weight_identity(_identity_file(tmp_path, "BF16"),
                             fam.identity_tensors)
    assert h32 is not None and h32 == h16
    other = fp.weight_identity(_identity_file(tmp_path, "F32", seed=1),
                               fam.identity_tensors)
    assert other != h32


def test_identity_table_lookup_and_variant_override(tmp_path, monkeypatch):
    fam = get_family("z_image")
    view = _identity_file(tmp_path, "F32")
    assert fp._variant(fam, view, None)["variant"] == "unknown"
    # An unknown hash takes an explicit variant.
    assert fp._variant(fam, view, "base")["variant"] == "base"
    h = fp.weight_identity(view, fam.identity_tensors)
    monkeypatch.setitem(fam.variants, h, VariantRow("turbo", "r", "v", ()))
    assert fp._variant(fam, view, None) == {"variant": "turbo",
                                            "identity_hash": h}
    with pytest.raises(fp.ImageModelError, match="contradicts the weights"):
        fp._variant(fam, view, "base")
    with pytest.raises(fp.ImageModelError, match="not a Z-Image variant"):
        fp._variant(fam, view, "edit")


def test_the_table_holds_turbo_and_base():
    fam = get_family("z_image")
    assert fam.variant_names() == ("base", "turbo")
    assert fam.identity_tensors == (
        "x_pad_token", "cap_pad_token", "final_layer.linear.bias")
