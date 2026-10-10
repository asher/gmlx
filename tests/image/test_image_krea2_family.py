"""The Krea 2 registry row: both GGUF layouts, the config against the
reference defaults (check 1), relatives refused with the reason, the
variant identity, and the line between Krea 2 and Qwen-Image 2.1, whose
arch string one Krea 2 converter writes. CPU-only. Header fixtures hold
names, shapes and codecs, no weights."""
from __future__ import annotations

import numpy as np
import pytest

from image_fixtures import (Q_HEADER, bf16_bits, fixture_view, load_json,
                            official_config, write_gguf)

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
