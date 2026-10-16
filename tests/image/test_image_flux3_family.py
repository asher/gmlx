"""The FLUX.3 klein registry row: both publishers' files, Base, the config
against the official config.json and transformer/config.json (check 1),
relatives refused with the reason, the FLUX.1 and FLUX.2 dev files that
share the flux label refused by name, and the FLUX.2 klein files left to
their own row. CPU-only. Header fixtures hold
names, shapes and codecs, no weights."""
from __future__ import annotations

import hashlib

import pytest

from image_fixtures import (FIXTURES, Q_HEADER, Z_HEADER, fixture_view,
                            load_json, official_config)

from gmlx.image import fingerprint as fp
from gmlx.image.families.flux3_klein import family as ff
from gmlx.image.registry import get_family, is_image_arch

DISTILLED = "flux3_klein_q4_k_m.header.json"
ABIRAY = "flux3_klein_abiray_q4_k_m.header.json"
BASE = "flux3_klein_base_q4_k_m.header.json"
FLUX2_4B = "flux2_klein_4b_q4_k_m.header.json"
FLUX2_9B = "flux2_klein_9b_q4_k_m.header.json"
CONFIG = "flux3_klein_config.json"
TRANSFORMER = "flux3_klein_transformer_config.json"
KREA2 = "krea2_turbo_q4_k_m.header.json"


def _refused(view, **kw) -> str:
    with pytest.raises(fp.ImageModelError) as ei:
        fp.identify(view, **kw)
    return str(ei.value)


@pytest.mark.parametrize("name", [DISTILLED, ABIRAY])
def test_both_publishers_are_distilled_with_the_official_config(name):
    view = fixture_view(name)
    assert view.arch == "flux"
    got = fp.identify(view)
    assert got.family.name == "flux3_klein"
    assert (got.prefix, got.layout) == ("", "default")
    assert (got.variant, got.identity_hash) == ("distilled", None)
    official = official_config(TRANSFORMER)
    assert {k: got.config[k] for k in official} == official
    assert got.config["dit"] == load_json(CONFIG)["dit"]


def test_the_publishers_store_the_small_tensors_differently():
    """The fingerprint reads names and shapes, so it takes both."""
    rebel, abiray = fixture_view(DISTILLED), fixture_view(ABIRAY)
    for n in ("emb_in.image.weight", "single_stream_modulations.txt.lin.weight"):
        assert rebel.tensors[n].codec == "BF16"
    assert abiray.tensors["emb_in.image.weight"].codec == "F16"
    assert abiray.tensors["single_stream_modulations.txt.lin.weight"].codec \
        == "Q4_K"


def test_base_is_named_by_its_tensors():
    view = fixture_view(BASE)
    assert not any(view.has(n) for n in ff.REF_TIME)
    got = fp.identify(view)
    assert (got.family.name, got.variant) == ("flux3_klein", "base")
    official = dict(official_config(TRANSFORMER), flow_map=False)
    assert {k: got.config[k] for k in official} == official
    assert got.config["dit"] == dict(load_json(CONFIG)["dit"], flow_map=False)


def test_the_distilled_file_less_ref_time_in_is_base():
    got = fp.identify(fixture_view(DISTILLED, drop=("ref_time_in.",)))
    assert got.variant == "base" and not got.config["flow_map"]


@pytest.mark.parametrize("name,named,other", [(DISTILLED, "distilled", "base"),
                                              (BASE, "base", "distilled")])
def test_a_variant_given_by_name_must_match_the_tensors(name, named, other):
    assert fp.identify(fixture_view(name), variant=named).variant == named
    assert _refused(fixture_view(name), variant=other) == (
        f"variant={other!r} contradicts the tensors, which are FLUX.3 klein "
        f"{named}")


def test_the_config_fixtures_are_the_official_files_in_bytes():
    manifest = load_json("manifest.json")
    for name in (CONFIG, TRANSFORMER):
        data = (FIXTURES / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == manifest[name]["sha256"]
        assert len(data) == manifest[name]["size"]
    row = get_family("flux3_klein").constants
    assert (row.repo, row.revision) == (manifest[CONFIG]["repo"],
                                        manifest[CONFIG]["revision"])


def test_the_header_fixtures_are_pinned():
    manifest = load_json("manifest.json")
    for name, n in ((DISTILLED, 359), (ABIRAY, 359), (BASE, 357)):
        d = load_json(name)
        assert d["n_tensors"] == len(d["tensors"]) == n
        assert len(manifest[name]["revision"]) == 40
        assert len(manifest[name]["sha256"]) == 64


def test_the_encoder_constants_follow_the_reference_code():
    """The taps and the pad multiple equal config.json. Its max_tokens is
    8192, but the reference encoder truncates at 4096, which gmlx keeps."""
    enc = load_json(CONFIG)["text_encoder"]
    assert list(ff.PROMPT.values["tap_layers"]) == enc["layers"]
    assert ff.PROMPT.values["pad_multiple"] == enc["pad_multiple"]
    assert (enc["max_tokens"], ff.PROMPT.values["max_tokens"]) == (8192, 4096)
    assert ff.PROMPT.revision.startswith("45e57c9")


@pytest.mark.parametrize("name,title", [(FLUX2_4B, "FLUX.2 klein 4B"),
                                        (FLUX2_9B, "FLUX.2 klein 9B")])
@pytest.mark.parametrize("arch", ["flux", None])
def test_flux2_klein_given_as_flux3_is_refused_by_name(name, title, arch):
    view = fixture_view(name, kv={"general.architecture": arch} if arch
                        else {})
    if title.endswith("9B"):
        assert _refused(view) == (
            "FLUX.2 klein 9B is not supported yet. gmlx image runs FLUX.2 "
            "klein 4B and base 4B.")
    else:
        assert fp.identify(view).family.name == "flux2_klein"
    assert _refused(view, family="flux3_klein") == (
        f"this is a {title} GGUF, not a FLUX.3 klein one")


def _bfl(txt_in: tuple, img_in: tuple, *extra: str) -> fp.HeaderView:
    """A file in the FLUX block names of BFL's flux and flux2 code."""
    tensors = {"double_blocks.0.img_attn.proj.weight": (txt_in[1:] * 2, "Q4_K"),
               "txt_in.weight": (txt_in, "Q4_K"),
               "img_in.weight": (img_in, "BF16")}
    tensors.update({n: ((txt_in[1], 9216), "BF16") for n in extra})
    return fp.HeaderView(kv={"general.architecture": "flux"}, tensors={
        n: fp.Tensor(s, c) for n, (s, c) in tensors.items()})


@pytest.mark.parametrize("view,name", [
    (_bfl((4096, 3072), (64, 3072), "double_blocks.0.img_mod.lin.weight"),
     "FLUX.1"),
    (_bfl((15360, 6144), (128, 6144), "double_stream_modulation_img.lin.weight"),
     "FLUX.2 dev"),
    (_bfl((5120, 5120), (128, 5120), "double_stream_modulation_img.lin.weight"),
     "FLUX.2"),
], ids=["flux1", "flux2 dev", "flux2 other"])
def test_the_older_flux_models_are_refused_by_name(view, name):
    want = (f"this is a {name} GGUF, which gmlx image does not support; "
            "FLUX.2 klein and FLUX.3 klein are the FLUX models it runs")
    assert _refused(view) == want
    assert _refused(view, family="flux3_klein") == want


def test_another_model_in_the_flux1_names_keeps_the_generic_refusal():
    """LongCat-Image as ComfyUI stores it: the FLUX.1 block names with the
    text width of its own encoder."""
    view = _bfl((3584, 3072), (64, 3072), "double_blocks.0.img_mod.lin.weight")
    assert ff.fingerprint(view) is None
    assert _refused(view) == ("this is a FLUX GGUF, which gmlx image does "
                              "not support")


@pytest.mark.parametrize("change,reason", [
    ({"drop": ("single_blocks.27.",)},
     "single_blocks has 27 blocks, FLUX.3 klein has 28"),
    ({"drop": ("content_mode_blocks.image_cond.",)},
     "content_mode_blocks.image_cond has 0 blocks, FLUX.3 klein has 5"),
    ({"add": {"txt_mode_blocks.5.q_proj.weight": ((3072, 3072), "Q4_K")}},
     "txt_mode_blocks has 6 blocks, FLUX.3 klein has 5"),
    ({"drop": ("ref_time_in.out_layer.",)},
     "it has one of ref_time_in.in_layer and ref_time_in.out_layer"),
    ({"drop": ("txt_mode_blocks.2.mlp_in.",)},
     "txt_mode_blocks.2.mlp_in.weight is missing"),
    ({"add": {ff.TXT_IN: ((7680, 3072), "BF16")}},
     "txt_in.weight is (7680, 3072), FLUX.3 klein has (20480, 3072)"),
    ({"add": {"content_mode_blocks.image.0.q_proj.bias": ((3072,), "F32")}},
     "it has bias tensors"),
    ({"add": {"final_layer.text.linear.weight": ((3072, 20480), "BF16")}},
     "final_layer.text.linear.weight is not a FLUX.3 klein tensor"),
], ids=["single", "no image_cond", "text depth", "half ref_time_in",
        "missing", "text width", "bias", "extra"])
def test_a_relative_is_refused_with_the_reason(change, reason):
    msg = _refused(fixture_view(DISTILLED, **change))
    assert "FLUX.3 klein relative" in msg and reason in msg


def test_a_long_refusal_counts_what_it_does_not_show():
    msg = _refused(fixture_view(DISTILLED, drop=("single_blocks.0.",
                                                 "single_blocks.1.")))
    assert "single_blocks has 26 blocks" in msg
    assert msg.endswith("more)")


@pytest.mark.parametrize("prefix", ["model.diffusion_model.",
                                    "diffusion_model."])
def test_a_prefixed_file_loads_under_its_prefix(prefix):
    view = fixture_view(DISTILLED)
    view = fp.HeaderView(kv=view.kv, tensors={
        prefix + n: t for n, t in view.tensors.items()})
    got = fp.identify(view)
    assert (got.family.name, got.prefix) == ("flux3_klein", prefix)


@pytest.mark.parametrize("arch", ["flux", None])
def test_flux_and_no_label_load_as_flux3_klein(arch):
    kv = {"general.architecture": arch} if arch else {}
    assert fp.identify(fixture_view(DISTILLED, kv=kv)).family.name == \
        "flux3_klein"


def test_a_label_of_another_layout_is_refused():
    view = fixture_view(DISTILLED, kv={"general.architecture": "lumina2"})
    assert _refused(view) == (
        "arch 'lumina2' contradicts the tensor layout, which is FLUX.3 klein "
        "(arch 'flux' or none)")
    assert not is_image_arch("flux")


@pytest.mark.parametrize("name", [DISTILLED, BASE])
def test_no_other_family_takes_a_flux3_file(name):
    view = fixture_view(name)
    for other in ("z_image", "krea2", "qwen_image21", "flux2_klein"):
        fam = get_family(other)
        assert fam.fingerprint(fp.strip_prefix(view, fam.strip_prefixes)) \
            is None, other


@pytest.mark.parametrize("name", [Z_HEADER, Q_HEADER, KREA2])
def test_flux3_takes_no_file_of_another_family(name):
    view = fp.strip_prefix(fixture_view(name), ff.FAMILY.strip_prefixes)
    assert ff.fingerprint(view) is None


def _scan(name, arch):
    """A shard scan of a header fixture, or of a view, as the header-only
    read gives it."""
    from types import SimpleNamespace

    view = fixture_view(name) if isinstance(name, str) else name
    return SimpleNamespace(
        path="x.gguf", data_offset=0, kv={"general.architecture": arch},
        tensors=[SimpleNamespace(name=n, shape=t.shape, type_name=t.codec,
                                 offset=0, nbytes=0)
                 for n, t in view.tensors.items()])


def test_pull_names_each_flux_file_by_its_tensors_and_refuses_flux1():
    from gmlx.commands import manage
    from gmlx.load import remote

    report = remote.HeaderReport("flux", {"Q4_K": 3}, {}, 3)
    assert manage._image_role(report, None) == (None, None)
    assert manage._image_role(report, [_scan(DISTILLED, "flux")]) == (
        "dit", "flux3_klein")
    assert manage._image_role(report, [_scan(FLUX2_4B, "flux")]) == (
        "dit", "flux2_klein")
    scan = _scan(_bfl((4096, 3072), (64, 3072),
                      "double_blocks.0.img_mod.lin.weight"), "flux")
    assert manage._image_role(report, [scan]) == (None, None)
    v = manage._verdict(remote.parse_ref("hf:o/r/m.gguf"), report,
                        size_bytes=1 << 30, scans=[scan])
    assert v["image"] is None and not v["usable"]
    assert v["arch_error"] == (
        "an image GGUF gmlx does not load: this is a FLUX.1 GGUF, which gmlx "
        "image does not support; FLUX.2 klein and FLUX.3 klein are the FLUX "
        "models it runs")
