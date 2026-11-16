"""The MiniMax-H3-Image registry row: the config against the official
transformer/config.json (check 1), the sampling row against the release's
pdd_config.json, relatives refused with the reason, the line to the H3
video file, and the family found by identify. The tests call its rule
directly. CPU-only. Header fixtures hold names, shapes and codecs, no
weights."""
from __future__ import annotations

import hashlib

import pytest
from image_fixtures import FIXTURES, fixture_view, load_json, official_config

from gmlx.image.families.h3_image import family as hi

HEADER = "h3_image_pruned_ud_q4_k_xl.header.json"
CONFIG = "h3_image_transformer_config.json"
PDD = "h3_image_pdd_config.json"


def _view(**kw):
    return fixture_view(HEADER, **kw)


def _refused(view) -> str:
    got = hi.fingerprint(view)
    assert isinstance(got, hi.Refusal), got
    return got.reason


def test_the_fixtures_are_pinned():
    manifest = load_json("manifest.json")
    for name in (CONFIG, PDD):
        data = (FIXTURES / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == manifest[name]["sha256"]
        assert len(data) == manifest[name]["size"]
        row = hi.FAMILY.constants
        assert (row.repo, row.revision) == (manifest[name]["repo"],
                                            manifest[name]["revision"])
    d = load_json(HEADER)
    assert d["n_tensors"] == len(d["tensors"]) == 528
    variant, = hi.FAMILY.variants.values()
    assert manifest[HEADER]["revision"] == variant.revision
    assert manifest[HEADER]["file"] in variant.files


def test_check1_the_config_is_the_official_one():
    view = _view()
    assert view.arch == "hyvid"
    assert hi.fingerprint(view) == hi.Match()
    got = hi.synth_config(view)
    official = official_config(CONFIG)
    assert {k: got[k] for k in official} == official
    assert set(got) - set(official) == {"adaln_grid", "adaln_basis",
                                        "pdd_steps", "pdd_sigmas"}
    assert (got["adaln_grid"], got["adaln_basis"]) == (1025, 8)
    assert got["pdd_steps"] == load_json(PDD)["num_inference_steps"] == 6
    assert got["pdd_sigmas"] == load_json(PDD)["sigmas"]


def test_the_sampling_row_is_the_release_grid():
    """Six steps at guidance 1 on pdd_config.json's sigmas without the
    final zero, the same for a file of no known identity."""
    pdd = load_json(PDD)
    fam = hi.FAMILY
    row = fam.sampling_row("pruned")
    assert row.sigmas == tuple(pdd["sigmas"][:-1])
    assert (row.steps, row.cfg) == (pdd["num_inference_steps"], 1.0)
    assert (row.repo, row.revision) == (fam.constants.repo,
                                        fam.constants.revision)
    assert fam.sampling_row("unknown") is row
    assert fam.constants.values["pdd_sigmas"] == pdd["sigmas"]


def test_a_request_without_a_size_takes_the_release_size():
    """The row gives every H3-Image file, of any identity, 2048x2048 when
    a request names no size; the other families keep 1024x1024."""
    from gmlx.image import api
    from gmlx.image.registry import families

    for variant in ("pruned", "unknown"):
        assert api.sampler_defaults("h3_image", variant).size == (2048, 2048)
    for fam in families():
        if fam is not hi.FAMILY:
            assert {r.size for r in fam.sampling.values()} <= {(1024, 1024)}


@pytest.mark.parametrize("variant", ["pruned", "unknown"])
def test_steps_other_than_six_and_guidance_other_than_one_are_refused(
        variant):
    """The release runs its six PDD steps at guidance 1, and a file of no
    known identity samples as it does."""
    from gmlx.image import api

    fam = hi.FAMILY
    api.check_steps(fam, variant, 6)
    api.check_guidance(fam, variant, 1.0)
    with pytest.raises(ValueError, match=f"MiniMax-H3-Image {variant} runs "
                       "6 steps, the schedule it was distilled to, not 4"):
        api.check_steps(fam, variant, 4)
    with pytest.raises(ValueError) as e:
        api.check_guidance(fam, variant, 2.0)
    assert str(e.value) == (f"the MiniMax-H3-Image {variant} DiT samples "
                            "without guidance: drop guidance 2")


def test_the_label_rule_takes_hyvid_or_none():
    assert hi.FAMILY.arch_strings == frozenset({None})
    assert hi.FAMILY.shared_arch_strings == frozenset({"hyvid"})


def test_identify_finds_the_family_on_the_release_header():
    """The family is discovered: identify names the release header
    MiniMax-H3-Image under its shared label, the weights it cannot read
    give no variant hash, and the family's synthesized config is the one
    its rule gives."""
    from gmlx.image import fingerprint as ifp
    from gmlx.image.registry import families, get_family

    assert get_family("h3_image") is hi.FAMILY
    assert families()[-1] is hi.FAMILY
    got = ifp.identify(_view())
    assert got.family is hi.FAMILY and got.identity_hash is None
    assert got.config == hi.FAMILY.synth_config(_view())
    with pytest.raises(ifp.ImageModelError, match="contradicts the tensor "
                       "layout, which is MiniMax-H3-Image"):
        ifp.identify(fixture_view(HEADER, kv={
            "general.architecture": "lumina2"}))


def test_an_h3_video_file_is_not_taken():
    """A file with the audio stream is H3 video's, whatever else it has."""
    add = {"audio_patch_proj.weight": ((32, 5376), "F32"),
           "audio_patch_proj.bias": ((5376,), "F32")}
    assert hi.fingerprint(_view(add=add)) is None
    assert hi.fingerprint(_view(drop=("condition_proj.",))) is None


@pytest.mark.parametrize("change, reason", [
    ({"drop": ("blocks.49.",)}, "blocks has 49 blocks"),
    ({"drop": ("token_refiner.blocks.1.",)},
     "token_refiner.blocks has 1 blocks"),
    ({"drop": ("rope.inv_freq",)}, "rope.inv_freq is missing"),
    ({"add": {"final_layer.video_out.weight": ((5376, 96), "F32"),
              "final_layer.video_out.bias": ((96,), "F32")}},
     "final_layer.video_out.weight is (5376, 96)"),
    ({"add": {"blocks.0.attn.to_gate_compress.weight":
              ((5376, 7168), "Q4_K")}},
     "blocks.0.attn.to_gate_compress.weight is not a"),
])
def test_a_relative_is_refused_with_the_reason(change, reason):
    msg = _refused(_view(**change))
    assert msg.startswith(f"a {hi.TITLE} relative") and reason in msg, msg


def test_the_full_time_embedder_form_is_refused():
    msg = _refused(_view(add={
        "time_embedder.proj_in.weight": ((256, 5376), "F32")}))
    assert "full time embedder" in msg


def test_a_scaled_int8_file_is_refused_by_name():
    msg = _refused(_view(add={
        "blocks.0.mlp.fc1.weight_scale": ((28672,), "F32")}))
    assert "weight_scale" in msg


def test_a_head_bank_of_another_count_has_no_config():
    """synth_config holds the head count to the grid's steps, past the
    fingerprint's table."""
    view = _view(add={"final_layer.video_out.weight": ((5376, 480), "F32")})
    with pytest.raises(hi.ImageModelError, match="one head for each"):
        hi.synth_config(view)
