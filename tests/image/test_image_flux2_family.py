"""The FLUX.2 klein registry row: the 4B files and the 9B told apart and
refused, the config against the official transformer/config.json of both
4B checkpoints (check 1), the variant from the identity hash, relatives
refused with the reason, and the line between FLUX.2 klein and the other
FLUX models. CPU-only.
Header fixtures hold names, shapes and codecs, no weights. The rows on
the real files read their headers and the identity tensor, and skip
without them."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from image_fixtures import (FIXTURES, Q_HEADER, Z_HEADER, bf16_bits,
                            fixture_view, load_json, official_config,
                            write_gguf)

from gmlx.image import fingerprint as fp
from gmlx.image.families.flux2_klein import family as f2
from gmlx.image.registry import get_family

KLEIN_4B = "flux2_klein_4b_q4_k_m.header.json"
KLEIN_9B = "flux2_klein_9b_q4_k_m.header.json"
CONFIG_4B = "flux2_klein_4b_transformer_config.json"
CONFIG_BASE_4B = "flux2_klein_base_4b_transformer_config.json"
FLUX3 = "flux3_klein_q4_k_m.header.json"
KREA2 = "krea2_turbo_q4_k_m.header.json"
DISTILLED_HASH = ("591c63d49ea6d7cb55c1d349febb8736613013aef145462659f19816"
                  "ed3639a9")
BASE_HASH = "2a60e24acf60f890172dfdf3a887d7f9dc51675c515a566126bb24ddab3d2fce"


def _refused(view, **kw) -> str:
    with pytest.raises(fp.ImageModelError) as ei:
        fp.identify(view, **kw)
    return str(ei.value)


@pytest.mark.parametrize("config", [CONFIG_4B, CONFIG_BASE_4B])
@pytest.mark.parametrize("arch", ["flux", None])
def test_the_4b_file_gives_the_official_config_of_both_checkpoints(config,
                                                                    arch):
    """Check 1: the distilled and Base 4B files share the header, and the
    config synthesized from it is each official transformer config."""
    view = fixture_view(KLEIN_4B, kv={"general.architecture": arch} if arch
                        else {})
    got = fp.identify(view)
    assert got.family.name == "flux2_klein"
    assert (got.prefix, got.layout) == ("", "default")
    # A fixture view has no file bytes, so no identity hash.
    assert (got.variant, got.identity_hash) == ("unknown", None)
    assert got.config == official_config(config)


NOT_YET = ("FLUX.2 klein 9B is not supported yet. gmlx image runs FLUX.2 "
           "klein 4B and base 4B.")


def test_the_9b_file_is_flux2_klein_9b_and_refused_by_name():
    """The 9B widths and depths of Klein9BParams, told apart from the 4B
    and refused by name, pointing to the 4B. The official 9B config is
    gated, so check 1 at 9B waits for it."""
    view = fixture_view(KLEIN_9B)
    assert f2.size_of(view) == "9B" and f2.RUN_SIZES == ("4B",)
    assert f2.fingerprint(view) == fp.Refusal(NOT_YET)
    assert _refused(view) == NOT_YET
    assert _refused(view, family="flux2_klein") == NOT_YET
    want = dict(official_config(CONFIG_4B), joint_attention_dim=12288,
                num_attention_heads=32, num_layers=8, num_single_layers=24)
    assert f2.synth_config(view) == want


def test_the_config_fixtures_are_the_official_files_in_bytes():
    manifest = load_json("manifest.json")
    for name in (CONFIG_4B, CONFIG_BASE_4B):
        data = (FIXTURES / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == manifest[name]["sha256"]
        assert len(data) == manifest[name]["size"]
        assert manifest[name]["file"] == "transformer/config.json"
    row = get_family("flux2_klein").constants
    assert (row.repo, row.revision) == (manifest[CONFIG_4B]["repo"],
                                        manifest[CONFIG_4B]["revision"])
    # The checkpoints differ in the path they were saved from alone.
    a, b = load_json(CONFIG_4B), load_json(CONFIG_BASE_4B)
    assert a["_name_or_path"] != b["_name_or_path"]
    assert official_config(CONFIG_4B) == official_config(CONFIG_BASE_4B)


def test_the_header_fixtures_are_pinned():
    manifest = load_json("manifest.json")
    for name, n in ((KLEIN_4B, 149), (KLEIN_9B, 201)):
        d = load_json(name)
        assert d["n_tensors"] == len(d["tensors"]) == n
        assert d["kv"] == {"general.architecture": "flux"}
        assert len(manifest[name]["revision"]) == 40
        assert len(manifest[name]["sha256"]) == 64


@pytest.mark.parametrize("name,size", [(KLEIN_4B, "4B"), (KLEIN_9B, "9B")])
def test_the_header_holds_exactly_the_expected_tensors(name, size):
    view = fixture_view(name)
    assert {n: t.shape for n, t in view.tensors.items()} == f2.expected(size)


def test_the_identity_table_names_both_4b_variants_with_their_sampling():
    fam = get_family("flux2_klein")
    assert fam.identity_tensors == ("img_in.weight",)
    assert fam.variant_names() == ("base", "distilled")
    rows = {v.name: v for v in fam.variants.values()}
    assert fam.variants[DISTILLED_HASH] is rows["distilled"]
    assert fam.variants[BASE_HASH] is rows["base"]
    assert "flux-2-klein-4b-Q4_K_M.gguf" in rows["distilled"].files
    assert "flux-2-klein-base-4b-Q4_K_M.gguf" in rows["base"].files
    assert rows["distilled"].repo == "unsloth/FLUX.2-klein-4B-GGUF"
    assert rows["base"].repo == "unsloth/FLUX.2-klein-base-4B-GGUF"
    distilled, base = fam.sampling["distilled"], fam.sampling["base"]
    assert (distilled.steps, distilled.cfg) == (4, 1.0)
    assert (base.steps, base.cfg) == (50, 4.0)
    assert distilled.revision == base.revision == (
        "50fe5162777813d869182b139e83b10743caef15")
    # A file whose identity matches no row samples without defaults.
    assert fam.unknown_variant is None and not fam.needs_variant


def test_every_published_quant_keeps_the_identity_tensor_bf16():
    """img_in.weight is BF16 in every GGUF of both unsloth 4B repos, the
    F16 file included, so every file of a checkpoint gives the hash of
    its variant row. The range reads of the fixtures hold the codec and
    the hash of each file, and the F16 header stores every other matrix
    F16 and still matches the 4B config."""
    from collections import Counter

    table = load_json("flux2_klein_img_in_identity.json")["repos"]
    want = {"unsloth/FLUX.2-klein-4B-GGUF": ("distilled", DISTILLED_HASH),
            "unsloth/FLUX.2-klein-base-4B-GGUF": ("base", BASE_HASH)}
    assert set(table) == set(want)
    fam = get_family("flux2_klein")
    for repo, row in table.items():
        variant, digest = want[repo]
        files = row["files"]
        assert len(files) == 15, repo
        assert {f.rsplit("-", 1)[1] for f in files} >= {
            "F16.gguf", "BF16.gguf", "Q4_K_M.gguf", "Q8_0.gguf"}
        assert {v["img_in.weight"] for v in files.values()} == {"BF16"}
        assert {v["sha256"] for v in files.values()} == {digest}
        assert fam.variants[digest].name == variant
    view = fixture_view("flux2_klein_4b_f16.header.json")
    assert view.tensors["img_in.weight"].codec == "BF16"
    assert Counter(t.codec for t in view.tensors.values()) == {
        "F32": 60, "F16": 80, "BF16": 9}
    got = fp.identify(view)
    assert got.family.name == "flux2_klein"
    assert got.config == official_config(CONFIG_4B)


@pytest.mark.parametrize("digest,variant", [(DISTILLED_HASH, "distilled"),
                                            (BASE_HASH, "base"),
                                            ("0" * 64, "unknown")])
def test_the_identity_names_the_variant(monkeypatch, digest, variant):
    monkeypatch.setattr(fp, "weight_identity", lambda view, names: digest)
    got = fp.identify(fixture_view(KLEIN_4B))
    assert (got.variant, got.identity_hash) == (variant, digest)


def test_a_variant_given_by_name_must_match_the_weights(monkeypatch):
    monkeypatch.setattr(fp, "weight_identity",
                        lambda view, names: DISTILLED_HASH)
    view = fixture_view(KLEIN_4B)
    assert fp.identify(view, variant="distilled").variant == "distilled"
    assert _refused(view, variant="base") == (
        "variant='base' contradicts the weights, which match FLUX.2 klein "
        "distilled")


def _identity_file(tmp_path, codec: str, seed: int):
    from gmlx.load.headerscan import scan_gguf

    rng = np.random.default_rng(seed)
    bits = bf16_bits(rng.standard_normal((16, 8)).astype(np.float32))
    stored = ((bits, "BF16") if codec == "BF16" else
              ((bits.astype(np.uint32) << 16).view(np.float32), codec))
    p = write_gguf(tmp_path / f"id-{codec}-{seed}.gguf",
                   {"img_in.weight": stored}, "flux")
    return fp.view_from_scans([scan_gguf(p)])


def test_the_identity_is_the_same_for_every_storage(tmp_path):
    names = get_family("flux2_klein").identity_tensors
    hashes = {fp.weight_identity(_identity_file(tmp_path, c, 0), names)
              for c in ("F32", "BF16")}
    assert len(hashes) == 1 and None not in hashes
    assert fp.weight_identity(_identity_file(tmp_path, "F32", 1),
                              names) not in hashes


@pytest.mark.parametrize("name,change,reason", [
    (KLEIN_4B, {"drop": ("single_blocks.19.",)},
     "single_blocks has 19 blocks, FLUX.2 klein 4B has 20"),
    (KLEIN_4B, {"add": {"single_blocks.20.linear2.weight":
                        ((12288, 3072), "Q4_K")}},
     "single_blocks has 21 blocks, FLUX.2 klein 4B has 20"),
    (KLEIN_9B, {"drop": ("double_blocks.7.",)},
     "double_blocks has 7 blocks, FLUX.2 klein 9B has 8"),
    (KLEIN_4B, {"drop": ("double_blocks.2.txt_mlp.0.",)},
     "double_blocks.2.txt_mlp.0.weight is missing"),
    (KLEIN_4B, {"add": {"single_blocks.0.linear1.weight":
                        ((3072, 21504), "Q4_K")}},
     "single_blocks.0.linear1.weight is (3072, 21504), FLUX.2 klein 4B has "
     "(3072, 27648)"),
    (KLEIN_4B, {"add": {"double_blocks.0.img_attn.qkv.bias":
                        ((9216,), "F32")}},
     "it has bias tensors"),
    (KLEIN_4B, {"add": {"guidance_in.in_layer.weight":
                        ((256, 3072), "BF16")}},
     "guidance_in.in_layer.weight is not a FLUX.2 klein tensor"),
], ids=["single", "single extra", "9b double", "missing", "shape", "bias",
        "guidance"])
def test_a_relative_is_refused_with_the_reason(name, change, reason):
    msg = _refused(fixture_view(name, **change))
    assert "a FLUX.2 klein relative that gmlx image does not support" in msg
    assert reason in msg


def test_a_long_refusal_counts_what_it_does_not_show():
    msg = _refused(fixture_view(KLEIN_4B, drop=("single_blocks.0.",
                                                "single_blocks.1.")))
    assert "single_blocks has 18 blocks" in msg
    assert msg.endswith("more)")


@pytest.mark.parametrize("prefix", ["model.diffusion_model.",
                                    "diffusion_model."])
def test_a_prefixed_file_loads_under_its_prefix(prefix):
    view = fixture_view(KLEIN_4B)
    view = fp.HeaderView(kv=view.kv, tensors={
        prefix + n: t for n, t in view.tensors.items()})
    got = fp.identify(view)
    assert (got.family.name, got.prefix) == ("flux2_klein", prefix)


def test_a_label_of_another_layout_is_refused():
    view = fixture_view(KLEIN_4B, kv={"general.architecture": "lumina2"})
    assert _refused(view) == (
        "arch 'lumina2' contradicts the tensor layout, which is FLUX.2 klein "
        "(arch 'flux' or none)")


def _bfl(txt_in: tuple, img_in: tuple, *extra: str) -> fp.HeaderView:
    """A file in the FLUX block names of BFL's flux and flux2 code."""
    tensors = {f2.FLUX_DOUBLE: (txt_in[1:] * 2, "Q4_K"),
               f2.TXT_IN: (txt_in, "Q4_K"),
               f2.IMG_IN: (img_in, "BF16")}
    tensors.update({n: ((txt_in[1], 9216), "BF16") for n in extra})
    return fp.HeaderView(kv={"general.architecture": "flux"}, tensors={
        n: fp.Tensor(s, c) for n, (s, c) in tensors.items()})


_MOD2 = "double_stream_modulation_img.lin.weight"


@pytest.mark.parametrize("view,name", [
    (_bfl((4096, 3072), (64, 3072), "double_blocks.0.img_mod.lin.weight"),
     "FLUX.1"),
    (_bfl((15360, 6144), (128, 6144), _MOD2), "FLUX.2 dev"),
    (_bfl((5120, 5120), (128, 5120), _MOD2), "FLUX.2"),
], ids=["flux1", "flux2 dev", "flux2 other"])
def test_the_other_flux_models_are_refused_by_name(view, name):
    assert f2.flux_model(view) == name
    want = (f"this is a {name} GGUF, which gmlx image does not support; "
            "FLUX.2 klein and FLUX.3 klein are the FLUX models it runs")
    assert f2.fingerprint(view) == fp.Refusal(want)
    assert _refused(view) == want


def test_another_model_in_the_flux1_names_keeps_the_generic_refusal():
    """LongCat-Image as ComfyUI stores it: the FLUX.1 block names with the
    text width of its own encoder."""
    view = _bfl((3584, 3072), (64, 3072), "double_blocks.0.img_mod.lin.weight")
    assert f2.flux_model(view) is None and f2.fingerprint(view) is None
    assert _refused(view) == ("this is a FLUX GGUF, which gmlx image does "
                              "not support")


@pytest.mark.parametrize("name", [KLEIN_4B, KLEIN_9B])
def test_no_family_before_it_takes_a_flux2_file(name):
    view = fixture_view(name)
    for other in ("z_image", "krea2", "qwen_image21"):
        fam = get_family(other)
        assert fam.order < f2.FAMILY.order
        assert fam.fingerprint(fp.strip_prefix(view, fam.strip_prefixes)) \
            is None, other
    assert f2.FAMILY.order < get_family("flux3_klein").order


@pytest.mark.parametrize("name", [Z_HEADER, Q_HEADER, KREA2, FLUX3])
def test_flux2_takes_no_file_of_another_family(name):
    view = fp.strip_prefix(fixture_view(name), f2.FAMILY.strip_prefixes)
    assert f2.fingerprint(view) is None


def test_a_flux3_file_given_as_flux2_is_refused():
    assert _refused(fixture_view(FLUX3), family="flux2_klein").startswith(
        "not a FLUX.2 klein file. unrecognized image GGUF")


def _klein_files(root) -> list:
    files = sorted(p for p in root.rglob("flux-2-klein-*.gguf")
                   if p.is_file())
    if not files:
        pytest.skip("no FLUX.2 klein GGUF under KQUANT_TEST_GGUF_DIR")
    return files


def test_each_real_file_has_the_header_of_its_size(gguf_dir):
    """Distilled and Base files of a size hold the same names, shapes and
    codecs, those of the size's fixture. Headers only."""
    from gmlx.load.headerscan import scan_gguf

    for p in _klein_files(gguf_dir):
        fixture = KLEIN_9B if "-9b-" in p.name else KLEIN_4B
        want = [list(t) for t in load_json(fixture)["tensors"]]
        scan = scan_gguf(str(p))
        got = [[t.name, list(t.shape), t.type_name] for t in scan.tensors]
        assert got == want, p.name


def test_each_real_4b_file_is_named_by_its_identity(gguf_dir):
    """The identity of a 4B file names the variant of its repo. The 9B
    rows wait for the 9B configs, so a 9B file's tensors are not read."""
    from gmlx.load.headerscan import scan_gguf

    fam = get_family("flux2_klein")
    rows = {f: v for v in fam.variants.values() for f in v.files}
    files = [p for p in _klein_files(gguf_dir) if p.name in rows]
    if not files:
        pytest.skip("no FLUX.2 klein 4B GGUF under KQUANT_TEST_GGUF_DIR")
    for p in files:
        got = fp.identify(fp.view_from_scans([scan_gguf(str(p))]))
        assert got.family.name == "flux2_klein"
        assert got.variant == rows[p.name].name, p.name
        assert fam.variants[got.identity_hash] is rows[p.name]
