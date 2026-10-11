"""The Krea 2 registry row: both GGUF layouts, the config against the
reference defaults (check 1), relatives refused with the reason, the
variant identity, and the line between Krea 2 and Qwen-Image 2.1, whose
arch string one Krea 2 converter writes. CPU-only. Header fixtures hold
names, shapes and codecs, no weights."""
from __future__ import annotations

import numpy as np
import pytest

from image_fixtures import (Q_HEADER, Z_HEADER, bf16_bits, fixture_view,
                            load_json, official_config, write_gguf)

from gmlx.image import fingerprint as fp
from gmlx.image.families.krea2 import family as kf
from gmlx.image.registry import get_family

TURBO = "krea2_turbo_q4_k_m.header.json"
# The header of a Krea 2 Raw file, the one fixture of the krea2 arch
# layout; its shapes are those of Turbo.
RAW = "krea2_raw_q4_k_m.header.json"
DEFAULTS = "krea2_transformer_config.json"


def _refused(view, **kw) -> str:
    with pytest.raises(fp.ImageModelError) as ei:
        fp.identify(view, **kw)
    return str(ei.value)


@pytest.mark.parametrize("name,arch", [(TURBO, "qwen_image"), (RAW, "krea2")])
def test_both_converters_are_krea2_with_the_reference_config(name, arch):
    view = fixture_view(name)
    assert view.arch == arch
    got = fp.identify(view)
    assert got.family.name == "krea2"
    assert (got.prefix, got.layout) == ("", "last_mlp")
    official = official_config(DEFAULTS)
    assert {k: got.config[k] for k in official} == official
    assert got.config["out_channels"] == 64 and got.config["last_mlp"]


def test_the_raw_file_stores_the_projector_flattened():
    view = fixture_view(RAW)
    assert view.shape(kf.PROJECTOR) == (12,)
    assert kf.projector_shape(view) == (1, 12)
    assert kf.projector_shape(fixture_view(TURBO)) is None
    assert fixture_view(TURBO).shape(kf.PROJECTOR) == (12, 1)


def test_a_file_without_the_final_mlp_is_the_plain_layout():
    got = fp.identify(fixture_view(RAW, drop=kf.LAST_MLP))
    assert got.layout == "plain" and not got.config["last_mlp"]


def test_the_header_fixtures_are_pinned():
    manifest = load_json("manifest.json")
    for name, n in ((TURBO, 432), (RAW, 432)):
        d = load_json(name)
        assert d["n_tensors"] == len(d["tensors"]) == n
        assert len(manifest[name]["revision"]) == 40
        assert len(manifest[name]["sha256"]) == 64
    row = get_family("krea2").constants
    assert (row.repo, row.revision) == (manifest[DEFAULTS]["repo"],
                                        manifest[DEFAULTS]["revision"])


@pytest.mark.parametrize("change,reason", [
    ({"drop": ("blocks.27.",)}, "blocks has 27 blocks, Krea 2 has 28"),
    ({"drop": ("last.up.",)}, "it has one of last.up and last.down"),
    ({"drop": ("txtfusion.refiner_blocks.1.",)},
     "txtfusion.refiner_blocks has 1 blocks, Krea 2 has 2"),
    ({"add": {"blocks.0.attn.wk.weight": ((6144, 6144), "Q4_K")}},
     "blocks.0.attn.wk.weight is (6144, 6144), Krea 2 has (6144, 1536)"),
    ({"kv": {"general.architecture": "krea2",
             "comfy.gguf.orig_shape.txtfusion.projector.weight": [12, 1]}},
     "its stored shape is (12, 1), not (1, 12)"),
    ({"add": {kf.PROJECTOR: ((24, 1), "F32")}},
     "txtfusion.projector.weight is (24, 1), Krea 2 has (12, 1)"),
], ids=["blocks", "half the final mlp", "refiner", "kv width", "orig shape",
        "projector"])
def test_a_relative_is_refused_with_the_reason(change, reason):
    base = RAW if "kv" in change else TURBO
    msg = _refused(fixture_view(base, **change))
    assert "Krea 2 relative" in msg and reason in msg


@pytest.mark.parametrize("name", [TURBO, RAW])
def test_a_krea2_file_is_never_qwen_image21(name):
    view = fixture_view(name)
    assert get_family("qwen_image21").fingerprint(view) is None
    assert get_family("z_image").fingerprint(view) is None
    msg = _refused(view, family="qwen_image21")
    assert "not a Qwen-Image 2.1 file" in msg


def test_a_qwen_image21_file_is_never_krea2():
    view = fp.strip_prefix(fixture_view(Q_HEADER), ("model.diffusion_model.",))
    assert kf.fingerprint(view) is None
    assert fp.identify(fixture_view(Q_HEADER)).family.name == "qwen_image21"
    assert "not a Krea 2 file" in _refused(fixture_view(Q_HEADER),
                                           family="krea2")


def test_a_krea2_file_under_a_qwen_name_keeps_krea2():
    """The arch string of a Qwen-Image file is no evidence: the tensor
    names decide."""
    view = fixture_view(TURBO, kv={"general.architecture": "qwen_image"})
    assert fp.identify(view).family.name == "krea2"


@pytest.mark.parametrize("name", [TURBO, RAW])
@pytest.mark.parametrize("arch", ["krea-2", "flux", "pig"])
def test_the_labels_other_converters_write_load_as_krea2(name, arch):
    """krea-2 names Krea 2. flux and pig are labels that converters write
    for other models too, and the tensors decide."""
    kv = dict(load_json(name)["kv"], **{"general.architecture": arch})
    got = fp.identify(fixture_view(name, kv=kv))
    assert (got.family.name, got.layout) == ("krea2", "last_mlp")


def test_a_label_of_another_layout_names_every_krea2_label():
    view = fixture_view(TURBO, kv={"general.architecture": "lumina2"})
    assert _refused(view) == (
        "arch 'lumina2' contradicts the tensor layout, which is Krea 2 "
        "(arch 'flux' or 'krea-2' or 'krea2' or 'pig' or 'qwen_image' or "
        "none)")


def test_manage_names_a_krea2_file_under_a_shared_label_by_its_tensors():
    """The header-only check of `gmlx pull` names a family by its tensors.
    Without them, a shared label names none, and krea-2 names Krea 2."""
    from gmlx.commands import manage
    from gmlx.load.remote import HeaderReport

    for arch, want in (("pig", (None, None)), ("flux", (None, None)),
                       ("krea-2", ("dit", "krea2")),
                       ("qwen_image21", ("dit", "qwen_image21"))):
        assert manage._image_role(HeaderReport(arch), None) == want, arch
    assert manage._image_role(HeaderReport("pig"), [_scan(TURBO, "pig")]) \
        == ("dit", "krea2")


def _scan(name, arch, **change):
    """A shard scan of a header fixture as the header-only read gives it."""
    from types import SimpleNamespace

    return SimpleNamespace(
        path="x.gguf", data_offset=0, kv={"general.architecture": arch},
        tensors=[SimpleNamespace(name=n, shape=t.shape, type_name=t.codec,
                                 offset=0, nbytes=0)
                 for n, t in fixture_view(name, **change).tensors.items()])


_QB0 = "model.diffusion_model.transformer_blocks.0."
_REFUSED = {
    "Z-Image": (Z_HEADER, {}),
    "Qwen-Image 2.1": (Q_HEADER, {}),
    "Ming-Image": (Z_HEADER, {"drop": ("x_pad_token", "cap_pad_token")}),
    "Lens": (Q_HEADER, {"add": {
        _QB0 + "attn.norm_added_q.weight": ((128,), "F32"),
        _QB0 + "img_mlp.w1.weight": ((4096, 12288), "Q4_K")}}),
}


@pytest.mark.parametrize("arch", ["flux", "pig"])
@pytest.mark.parametrize("model", sorted(_REFUSED))
def test_pull_refuses_a_file_under_a_shared_label_that_gmlx_image_refuses(
        model, arch):
    """`gmlx pull` and `check` name a family only where gmlx image loads
    the file, and word the refusal as `gmlx list` does."""
    from gmlx.commands import manage
    from gmlx.load import remote

    name, change = _REFUSED[model]
    scan = _scan(name, arch, **change)
    with pytest.raises(fp.ImageModelError):
        fp.identify(fp.HeaderView(kv=scan.kv, tensors=fixture_view(
            name, **change).tensors))
    report = remote.HeaderReport(arch, {"Q4_K": 3}, {}, 3)
    assert manage._image_role(report, [scan]) == (None, None)
    v = manage._verdict(remote.parse_ref("hf:o/r/m.gguf"), report,
                        size_bytes=1 << 30, scans=[scan])
    assert v["image"] is None and not v["usable"]
    assert v["arch_error"] == "an image GGUF gmlx does not load"


def _identity_file(tmp_path, codec: str, seed: int, flat: bool):
    from gmlx.load.headerscan import scan_gguf

    rng = np.random.default_rng(seed)
    proj = bf16_bits(rng.standard_normal((1, 12)).astype(np.float32))
    bias = bf16_bits(rng.standard_normal(6144).astype(np.float32))
    if flat:
        proj = proj.reshape(12)

    def stored(bits):
        f32 = (bits.astype(np.uint32) << 16).view(np.float32)
        return (bits, "BF16") if codec == "BF16" else (f32, codec)

    p = write_gguf(tmp_path / f"id-{codec}-{seed}-{flat}.gguf",
                   {kf.PROJECTOR: stored(proj), "first.bias": stored(bias)},
                   "krea2")
    return fp.view_from_scans([scan_gguf(p)])


def test_the_identity_is_the_same_for_every_storage(tmp_path):
    names = get_family("krea2").identity_tensors
    assert names == (kf.PROJECTOR, "first.bias")
    hashes = {fp.weight_identity(_identity_file(tmp_path, c, 0, flat), names)
              for c in ("F32", "BF16") for flat in (False, True)}
    assert len(hashes) == 1 and None not in hashes
    other = fp.weight_identity(_identity_file(tmp_path, "F32", 1, False),
                               names)
    assert other not in hashes


def test_the_identity_table_holds_turbo_with_its_sampling():
    fam = get_family("krea2")
    assert fam.variant_names() == ("turbo",)
    rows = {v.name: v for v in fam.variants.values()}
    assert "Krea-2-Turbo-Q4_K_M.gguf" in rows["turbo"].files
    assert list(fam.sampling) == ["turbo"]
    turbo = fam.sampling["turbo"]
    assert (turbo.steps, turbo.cfg) == (8, 1.0)
    assert dict(turbo.quality)["high"] == 8
    # A file whose identity matches no row samples without defaults.
    assert fam.unknown_variant is None


def test_the_encoder_taps_twelve_layers_on_the_reference_template():
    spec = get_family("krea2").encoder()
    assert spec.tap_layers == tuple(range(2, 36, 3))
    assert spec.drop == 34 and spec.keep_tail == 5 and spec.stack
    assert spec.max_tokens == 34 + 512
    assert spec.template.count("{}") == 1
    assert spec.template.endswith("<|im_end|>\n<|im_start|>assistant\n")


RAW_HASH = "284b54edd003f36239b1d30e62b64fba898dc0598d4434eb7985639fcaa6112c"


@pytest.mark.parametrize("variant", [None, "turbo"])
def test_a_raw_file_is_refused_and_named_against_turbo(monkeypatch,
                                                       variant):
    monkeypatch.setattr(fp, "weight_identity", lambda view, names: RAW_HASH)
    msg = _refused(fixture_view(RAW), variant=variant)
    assert "Krea 2 Raw" in msg and "Krea 2 Turbo" in msg


def test_a_file_of_unknown_weights_runs_only_under_a_variant_by_name():
    """A Krea 2 file whose identity matches no row samples on no default
    schedule: the load and the request refuse it until a variant is
    given, and the refusal names both settings."""
    fam = get_family("krea2")
    got = fp.identify(fixture_view(TURBO))
    assert got.variant == "unknown"
    why = fam.variant_needed(got.variant)
    assert "`variant: turbo` on its image_models entry" in why
    assert "--variant turbo" in why
    assert fam.variant_needed("turbo") is None
    assert fp.identify(fixture_view(TURBO), variant="turbo").variant == \
        "turbo"
    assert "not a Krea 2 variant" in _refused(fixture_view(TURBO),
                                              variant="raw")
    # Z-Image samples an unknown file on steps and guidance given.
    assert get_family("z_image").variant_needed("unknown") is None


def test_the_load_refuses_a_file_of_unknown_weights(monkeypatch):
    from types import SimpleNamespace

    import gmlx.load.preflight as pre
    from gmlx.image import loader

    view = fixture_view(TURBO)
    monkeypatch.setattr(pre, "find_split_shards", lambda path: [path])
    monkeypatch.setattr(loader, "_scan_shard", lambda s: SimpleNamespace(
        truncated=False, kv={}, skipped={}))
    monkeypatch.setattr(loader, "view_from_scans", lambda scans: view)
    with pytest.raises(fp.ImageModelError, match="--variant turbo"):
        loader.load_dit("/k.gguf", identified=fp.identify(view),
                        log=lambda *a: None)


@pytest.mark.parametrize("model,arch", [("Ming-Image", "lumina2"),
                                        ("Ming-Image", None),
                                        ("Lens", "qwen_image")])
def test_pull_names_no_family_for_a_relative_its_family_refuses(model, arch):
    """A relative that a family refuses by name names no family in `gmlx
    pull`, under that family's own label too."""
    from gmlx.commands import manage
    from gmlx.load import remote

    name, change = _REFUSED[model]
    report = remote.HeaderReport(arch, {"Q4_K": 3}, {}, 3)
    assert manage._image_role(report, [_scan(name, arch, **change)]) == (
        None, None)


@pytest.mark.parametrize("arch", ["flux", "pig"])
def test_image_pull_reuses_a_krea2_file_on_disk_under_a_shared_label(
        monkeypatch, arch):
    """The scan `gmlx image pull` reuses files from reads the tensors of a
    file labeled flux or pig, and takes a Krea 2 one as its DiT."""
    from gmlx.image import catalog

    kv = dict(load_json(TURBO)["kv"], **{"general.architecture": arch})
    monkeypatch.setattr(catalog, "_kv", lambda path: kv)
    monkeypatch.setattr(catalog, "_view",
                        lambda path: fixture_view(TURBO, kv=kv))
    (got,) = catalog.classify("krea-2-turbo.gguf")
    assert (got.role, got.family) == ("dit", "krea2")
