"""The MOSS-Image registry row: the realrebelai files, the config against
the official transformer/config.json (check 1), the SFT and Base identity
of every published quant, relatives refused with the reason, and the line
between MOSS-Image and every other family in both directions. CPU-only.
Header fixtures hold names, shapes and codecs, no weights."""
from __future__ import annotations

import hashlib

import pytest

from image_fixtures import (FIXTURES, Q_HEADER, Z_HEADER, fixture_view,
                            load_json, official_config)

from gmlx.image import fingerprint as fp
from gmlx.image.families.moss_image import family as mf
from gmlx.image.registry import get_family, is_image_arch

SFT = "moss_image_sft_q4_k_m.header.json"
CONFIG = "moss_image_transformer_config.json"
IDENTITY = "moss_image_identity.json"
KREA2 = "krea2_turbo_q4_k_m.header.json"
OTHERS = {Z_HEADER: "Z-Image", Q_HEADER: "Qwen-Image 2.1",
          KREA2: "Krea 2",
          "flux3_klein_q4_k_m.header.json": "FLUX.3 klein",
          "flux2_klein_4b_q4_k_m.header.json": "FLUX.2 klein",
          "flux2_klein_9b_q4_k_m.header.json": "FLUX.2 klein 9B"}


def _refused(view, **kw) -> str:
    with pytest.raises(fp.ImageModelError) as ei:
        fp.identify(view, **kw)
    return str(ei.value)


def test_the_file_is_moss_image_with_the_official_config():
    view = fixture_view(SFT)
    assert view.arch == "moss_image" and is_image_arch("moss_image")
    got = fp.identify(view)
    assert got.family.name == "moss_image"
    assert (got.prefix, got.layout) == ("", "default")
    official = official_config(CONFIG)
    assert {k: got.config[k] for k in official} == official
    assert (got.config["head_dim"], got.config["num_bands"],
            got.config["num_text_blocks"]) == (128, 16, 2)


def test_the_fixtures_are_pinned():
    manifest = load_json("manifest.json")
    data = (FIXTURES / CONFIG).read_bytes()
    assert hashlib.sha256(data).hexdigest() == manifest[CONFIG]["sha256"]
    assert len(data) == manifest[CONFIG]["size"]
    row = get_family("moss_image").constants
    assert (row.repo, row.revision) == (manifest[CONFIG]["repo"],
                                        manifest[CONFIG]["revision"])
    d = load_json(SFT)
    assert d["n_tensors"] == len(d["tensors"]) == 463
    assert manifest[SFT]["revision"] == load_json(IDENTITY)["revision"]


def test_every_published_quant_is_named_by_its_identity():
    """The hash of each file of the repository, read by range, names the
    variant of its file name, and the variant row lists the file."""
    fam = get_family("moss_image")
    assert fam.identity_tensors == mf.IDENTITY_TENSORS
    files = load_json(IDENTITY)["files"]
    assert len(files) == 20
    for name, row in files.items():
        want = "sft" if name.startswith("MOSS-Image-SFT-") else "base"
        assert row["identity_codecs"] == ["F16"] * 3, name
        variant = fam.variants[row["sha256"]]
        assert variant.name == want and name in variant.files, name
        assert variant.repo == load_json(IDENTITY)["repo"]
    assert fam.variant_names() == ("base", "sft")


@pytest.mark.parametrize("change,reason", [
    ({"drop": ("layers.41.",)}, "layers has 41 blocks, MOSS-Image has 42"),
    ({"drop": ("text_embedder.blocks.1.",)},
     "text_embedder.blocks has 1 blocks, MOSS-Image has 2"),
    ({"drop": ("layers.7.q_norm.",)}, "layers.7.q_norm.weight is missing"),
    ({"add": {"layers.0.proj.ffn_up_gate.weight": ((3072, 12288), "Q4_K")}},
     "layers.0.proj.ffn_up_gate.weight is (3072, 12288), MOSS-Image has "
     "(3072, 16384)"),
    ({"add": {"text_embedder.proj.weight": ((2560, 3072), "F16")}},
     "text_embedder.proj.weight is (2560, 3072), MOSS-Image has "
     "(4096, 3072)"),
    ({"add": {"layers.0.proj.q.bias": ((3072,), "F16")}},
     "layers.0.proj.q.bias is not a MOSS-Image tensor"),
    ({"add": {"layers.0.moe.gate.weight": ((3072, 64), "F16")}},
     "layers.0.moe.gate.weight is not a MOSS-Image tensor"),
], ids=["depth", "refiners", "missing", "ffn width", "text width", "bias",
        "extra"])
def test_a_relative_is_refused_with_the_reason(change, reason):
    msg = _refused(fixture_view(SFT, **change))
    assert "MOSS-Image relative" in msg and reason in msg


def test_a_long_refusal_counts_what_it_does_not_show():
    msg = _refused(fixture_view(SFT, drop=("layers.40.", "layers.41.")))
    assert "layers has 40 blocks" in msg


@pytest.mark.parametrize("prefix", ["model.diffusion_model.",
                                    "diffusion_model."])
def test_a_prefixed_file_loads_under_its_prefix(prefix):
    view = fixture_view(SFT)
    view = fp.HeaderView(kv=view.kv, tensors={
        prefix + n: t for n, t in view.tensors.items()})
    got = fp.identify(view)
    assert (got.family.name, got.prefix) == ("moss_image", prefix)


def test_a_file_without_a_label_loads_and_another_label_is_refused():
    assert fp.identify(fixture_view(SFT, kv={})).family.name == "moss_image"
    assert _refused(fixture_view(SFT, kv={"general.architecture": "flux"})
                    ) == ("arch 'flux' contradicts the tensor layout, which "
                          "is MOSS-Image (arch 'moss_image' or none)")


@pytest.mark.parametrize("name", list(OTHERS))
def test_another_family_given_as_moss_image_is_refused_by_name(name):
    view = fixture_view(name)
    assert fp.identify(view).family.title == OTHERS[name]
    assert _refused(view, family="moss_image") == (
        f"this is a {OTHERS[name]} GGUF, not a MOSS-Image one")


def test_no_other_family_takes_a_moss_image_file():
    view = fixture_view(SFT)
    for fam in (f for f in fp._families(None) if f.name != "moss_image"):
        assert fam.fingerprint(fp.strip_prefix(view, fam.strip_prefixes)) \
            is None, fam.name


@pytest.mark.parametrize("name", list(OTHERS))
def test_a_moss_image_file_given_as_another_family_is_refused(name):
    view = fixture_view(SFT)
    other = fp.identify(fixture_view(name)).family
    assert _refused(view, family=other.name).startswith(
        f"not a {other.title} file. unrecognized image GGUF: "
        "arch=moss_image, 463 tensors")


def test_an_error_in_another_familys_rule_is_not_hidden():
    """A Krea 2 file whose flattened projector carries a stored shape that
    is not a list of integers raises the same error under
    family="moss_image" as it does with no family given."""
    proj = "txtfusion.projector.weight"
    kv = {**load_json(KREA2)["kv"],
          f"comfy.gguf.orig_shape.{proj}": ["1", "twelve"]}
    view = fixture_view(KREA2, add={proj: ((12,), "F32")}, kv=kv)
    with pytest.raises(ValueError) as plain:
        fp.identify(view)
    with pytest.raises(ValueError) as given:
        fp.identify(view, family="moss_image")
    assert not isinstance(given.value, fp.ImageModelError)
    assert str(given.value) == str(plain.value)


def test_a_file_no_family_takes_keeps_the_generic_refusal():
    view = fp.HeaderView(kv={}, tensors={
        "blocks.0.attn.qkv.weight": fp.Tensor((64, 192), "F16")})
    assert mf.fingerprint(view) is None
    assert _refused(view, family="moss_image").startswith(
        "not a MOSS-Image file. unrecognized image GGUF")
