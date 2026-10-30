"""The MiniMax-H3 registry row: the config against the official
transformer/config.json (check 1), the identity and qkv layout of every
published quant, the Ref2VA rungs and relatives refused with the reason,
the module tree against the file's names and shapes, and the line between
gmlx video and gmlx image in both directions. CPU-only. Header fixtures
hold names, shapes and codecs, no weights."""
from __future__ import annotations

import hashlib

import pytest
from video_fixtures import (FIXTURES, H3_CONFIG, H3_HEADER, H3_IDENTITY,
                            H3_IDENTITY_MOLBAL, H3_IDENTITY_MOLBAL_TURBO,
                            fixture_view, load_json, official_config)

from gmlx.image import fingerprint as ifp
from gmlx.video import fingerprint as vfp
from gmlx.video.families.h3 import family as hf
from gmlx.video.registry import get_family


def _refused(view, **kw) -> str:
    with pytest.raises(vfp.VideoModelError) as ei:
        vfp.identify(view, **kw)
    return str(ei.value)


def test_the_fixtures_are_pinned():
    manifest = load_json("manifest.json")
    assert set(manifest) == {p.name for p in FIXTURES.iterdir()
                             if p.is_file() and p.name != "manifest.json"}
    data = (FIXTURES / H3_CONFIG).read_bytes()
    assert hashlib.sha256(data).hexdigest() == manifest[H3_CONFIG]["sha256"]
    row = get_family("h3").constants
    assert (row.repo, row.revision) == (manifest[H3_CONFIG]["repo"],
                                        manifest[H3_CONFIG]["revision"])
    d = load_json(H3_HEADER)
    assert d["n_tensors"] == len(d["tensors"]) == 532
    ident = load_json(H3_IDENTITY)
    assert manifest[H3_HEADER]["revision"] == ident["revision"]
    assert ident["files"][manifest[H3_HEADER]["file"]]["sha256"] == \
        manifest[H3_HEADER]["sha256"]
    for name in (H3_IDENTITY, H3_IDENTITY_MOLBAL, H3_IDENTITY_MOLBAL_TURBO):
        assert (manifest[name]["repo"], manifest[name]["revision"]) == (
            load_json(name)["repo"], load_json(name)["revision"])


def test_check1_the_config_is_the_official_one():
    got = vfp.identify(fixture_view(H3_HEADER))
    official = official_config(H3_CONFIG)
    assert {k: got.config[k] for k in official} == official
    assert set(got.config) - set(official) == {"adaln_grid", "adaln_basis"}
    assert (got.config["adaln_grid"], got.config["adaln_basis"]) == (1025, 8)


def test_the_module_tree_is_the_file():
    """Every parameter of the built model is a tensor of the file, in the
    file's shape, and the reverse."""
    from mlx.utils import tree_flatten

    view = fixture_view(H3_HEADER)
    model = hf.FAMILY.build(hf.synth_config(view))
    params = {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())}
    want = {n: tuple(reversed(t.shape)) for n, t in view.tensors.items()}
    assert params == want


def test_every_published_quant_is_named_by_its_identity():
    """The variant hash of each file of unsloth's and molbal's repositories,
    read by range, names FL2VA or refuses Ref2VA, and each FL2VA layout
    hash names its file and repository with the [q; k; v] layout. molbal's
    merged Turbo files keep the FL2VA identity tensors. molbal's scaled
    int8 files have no layout row: gmlx refuses them by their weight_scale
    tensors."""
    fam = get_family("h3")
    named = set()
    for fixture in (H3_IDENTITY, H3_IDENTITY_MOLBAL,
                    H3_IDENTITY_MOLBAL_TURBO):
        ident = load_json(fixture)
        assert tuple(ident["identity_tensors"]) == fam.identity_tensors
        assert (ident["layout_tensor"], ident["layout_rows"]) == (
            fam.layout_tensor, fam.layout_rows)
        for name, row in ident["files"].items():
            if "_ref2va_" in name:
                assert "Ref2VA" in fam.refused[row["variant_hash"]], name
                assert row["layout_hash"] not in fam.layouts, name
                continue
            assert fam.variants[row["variant_hash"]].name == "fl2va", name
            if row["layout_codec"] == "I8":
                assert row["layout_hash"] not in fam.layouts, name
                continue
            lay = fam.layouts[row["layout_hash"]]
            assert lay.layout == "qkv" and name in lay.files, name
            assert (lay.repo, lay.revision) == (ident["repo"],
                                                ident["revision"]), name
            named.add(name)
    assert named == {f for r in fam.layouts.values() for f in r.files}
    unsloth = {n for n in load_json(H3_IDENTITY)["files"] if "_fl2va_" in n}
    assert unsloth == set(next(iter(fam.variants.values())).files)


def test_a_header_alone_runs_unverified():
    got = vfp.identify(fixture_view(H3_HEADER))
    assert got.family.name == "h3"
    assert (got.variant, got.layout, got.prefix) == ("unknown", "unverified",
                                                     "")


def _qkv_view(tmp_path, rows):
    """A view of a fused qkv whose GGUF rows are ``rows`` (bytes each)."""
    path = tmp_path / "qkv"
    raw = b"".join(rows)
    path.write_bytes(b"x" * 100 + raw)
    t = ifp.Tensor((len(rows[0]) // 2, len(rows)), "F16", str(path), 100,
                   len(raw))
    return ifp.HeaderView(kv={}, tensors={hf.LAYOUT_TENSOR: t})


def test_the_layout_hash_tells_a_per_head_interleave_from_qkv(tmp_path):
    """Rows of q, k and v head 0 in [q; k; v] order are what the hash
    reads, so a file interleaved per head, with the same names, shapes and
    head-0 q rows, hashes differently."""
    heads, hd = 3, 8
    row = {(p, h, i): bytes([p, h, i]) * 64
           for p in range(3) for h in range(heads) for i in range(hd)}
    qkv = [row[p, h, i] for p in range(3) for h in range(heads)
           for i in range(hd)]
    per_head = [row[p, h, i] for h in range(heads) for p in range(3)
                for i in range(hd)]
    assert qkv[:hd] == per_head[:hd]
    n = hf.LAYOUT_ROWS
    want = hashlib.sha256(b"".join(
        qkv[s + i] for s in (0, heads * hd, 2 * heads * hd)
        for i in range(n))).hexdigest()
    assert vfp.layout_identity(_qkv_view(tmp_path, qkv), hf.FAMILY) == want
    assert vfp.layout_identity(_qkv_view(tmp_path, per_head),
                               hf.FAMILY) != want


def test_a_ref2va_file_is_refused_by_name(monkeypatch):
    ref = next(r["variant_hash"] for n, r in
               load_json(H3_IDENTITY)["files"].items() if "_ref2va_" in n)
    monkeypatch.setattr(vfp, "weight_identity", lambda view, names: ref)
    msg = _refused(fixture_view(H3_HEADER))
    assert "Ref2VA checkpoint" in msg and "follow-up" in msg



def _named(monkeypatch, ident, layout):
    monkeypatch.setattr(vfp, "weight_identity", lambda view, names: ident)
    monkeypatch.setattr(vfp, "layout_identity", lambda view, fam: layout)
    return fixture_view(H3_HEADER)


def test_a_merged_turbo_file_is_its_own_variant(monkeypatch):
    """molbal's merged Turbo files share FL2VA's identity tensors, so the
    layout row names them, with their own sampling defaults."""
    files = load_json(H3_IDENTITY_MOLBAL_TURBO)["files"]
    row = files["minimax_h3_fl2v_turbo_4step_v1.0_768p_Q4_0.gguf"]
    view = _named(monkeypatch, row["variant_hash"], row["layout_hash"])
    got = vfp.identify(view)
    assert (got.variant, got.layout) == (hf.TURBO4, "qkv")
    lay = hf.FAMILY.variant_row(hf.TURBO4)
    assert (lay.base, lay.merged) == ("fl2va", "lightx2v's Turbo 4-step "
                                      "v1.0 768p LoRA")
    s = hf.FAMILY.sampling[hf.TURBO4]
    assert (s.steps, s.shifts) == (4, {"video": 6.0, "audio": 3.0})
    assert hf.FAMILY.sampling["fl2va"].shifts == {"video": 12.0,
                                                  "audio": 3.0}
    assert set(hf.FAMILY.sampling) == set(hf.FAMILY.variant_names())
    assert vfp.identify(view, variant=hf.TURBO4).variant == hf.TURBO4
    assert _refused(view, variant="fl2va") == (
        "variant='fl2va' contradicts the weights, which match MiniMax-H3 "
        "fl2va-turbo4")


def test_a_given_variant_must_fit_the_weights(monkeypatch):
    """With no layout row, the identity tensors name only the base, so
    the base or a merge of it may be given; a name gmlx does not know is
    refused with the known ones."""
    base = load_json(H3_IDENTITY)["files"][
        "minimax_h3_fl2va_pruned-Q4_K.gguf"]["variant_hash"]
    view = _named(monkeypatch, base, None)
    assert vfp.identify(view).variant == "fl2va"
    assert vfp.identify(view, variant=hf.TURBO4).variant == hf.TURBO4
    assert _refused(view, variant="turbo") == (
        "variant 'turbo' is not a MiniMax-H3 variant; known: fl2va, "
        "fl2va-turbo4")
    unsloth = next(h for h, r in hf.LAYOUTS.items() if r.name == "fl2va")
    view = _named(monkeypatch, base, unsloth)
    assert _refused(view, variant=hf.TURBO4) == (
        "variant='fl2va-turbo4' contradicts the weights, which match "
        "MiniMax-H3 fl2va")

@pytest.mark.parametrize("change,reason", [
    ({"drop": ("blocks.49.",)}, "blocks has 49 blocks, MiniMax-H3 has 50"),
    ({"drop": ("token_refiner.blocks.1.",)},
     "token_refiner.blocks has 1 blocks, MiniMax-H3 has 2"),
    ({"drop": ("blocks.7.attn.q_norm.",)},
     "blocks.7.attn.q_norm.weight is missing"),
    ({"add": {"blocks.0.mlp.fc1.weight": ((5376, 14336), "Q4_K")}},
     "blocks.0.mlp.fc1.weight is (5376, 14336), MiniMax-H3 has "
     "(5376, 28672)"),
    ({"add": {"blocks.0.attn.to_gate_compress.weight": ((5376, 7168),
                                                        "Q4_K")}},
     "blocks.0.attn.to_gate_compress.weight is not a MiniMax-H3 tensor"),
], ids=["depth", "refiners", "missing", "ffn width", "sparse gate"])
def test_a_relative_is_refused_with_the_reason(change, reason):
    msg = _refused(fixture_view(H3_HEADER, **change))
    assert "MiniMax-H3 relative" in msg and reason in msg


def test_the_full_time_embedder_form_is_refused():
    msg = _refused(fixture_view(H3_HEADER, add={
        "time_embedder.proj_in.weight": ((256, 5376), "F32")}))
    assert "full time embedder" in msg and "curve-table" in msg


@pytest.mark.parametrize("prefix", ["model.diffusion_model.",
                                    "diffusion_model."])
def test_a_prefixed_file_loads_under_its_prefix(prefix):
    got = vfp.identify(fixture_view(H3_HEADER, prefix=prefix))
    assert (got.family.name, got.prefix) == ("h3", prefix)


def test_a_name_left_twice_by_the_prefix_is_refused():
    view = fixture_view(H3_HEADER, prefix="diffusion_model.")
    view.tensors["rope.inv_freq"] = view.tensors[
        "diffusion_model.rope.inv_freq"]
    assert "prefix strip collision" in _refused(view)


def test_a_scaled_int8_file_is_refused_by_name():
    """molbal's Q8_CR and U16G files: int8 linears with a weight_scale
    tensor beside each."""
    msg = _refused(fixture_view(H3_HEADER, add={
        "blocks.0.attn.qkv_proj.weight_scale": ((21504,), "F32")}))
    assert msg == ("a MiniMax-H3 GGUF with int8 linears and separate "
                   "weight_scale tensors (1), which gmlx video does not "
                   "read: use a K-quant or Q8_0 file")


@pytest.mark.parametrize("arch", ["minimax_h3", None])
def test_its_own_label_or_none_loads(arch):
    kv = {} if arch is None else {"general.architecture": arch}
    assert vfp.identify(fixture_view(H3_HEADER, kv=kv)).family.name == "h3"


@pytest.mark.parametrize("arch", ["wan", "qwen3vl"])
def test_another_label_is_refused_as_h3(arch):
    """A text arch label on H3 tensors gets the H3 message, not "a text
    model", since the video rules run before the text-arch check."""
    msg = _refused(fixture_view(H3_HEADER,
                                kv={"general.architecture": arch}))
    assert msg == (f"arch {arch!r} contradicts the tensor layout, which is "
                   f"MiniMax-H3 (arch 'minimax_h3' or none)")


def test_a_text_model_and_an_unknown_file_are_refused():
    llama = ifp.HeaderView(kv={"general.architecture": "llama"}, tensors={
        "token_embd.weight": ifp.Tensor((4096, 32000), "Q4_K")})
    assert _refused(llama) == ("arch 'llama' is a text model, use "
                               "gmlx.load_model")
    other = ifp.HeaderView(kv={}, tensors={
        "net.0.weight": ifp.Tensor((64, 64), "F16")})
    assert _refused(other).startswith("unrecognized video GGUF: arch=none")


@pytest.mark.parametrize("arch", [None, "minimax_h3", "qwen3vl", "wan"])
def test_gmlx_image_refuses_an_h3_file_by_name(arch):
    """Also under a text arch label, which the image rules would otherwise
    answer with "a text model"."""
    kv = {} if arch is None else {"general.architecture": arch}
    with pytest.raises(ifp.ImageModelError) as ei:
        ifp.identify(fixture_view(H3_HEADER, kv=kv))
    assert str(ei.value) == ("this is a MiniMax-H3 video GGUF, run it with "
                             "gmlx video")


def test_gmlx_video_takes_no_image_file():
    import image_fixtures

    for name in (image_fixtures.Z_HEADER, image_fixtures.Q_HEADER):
        view = image_fixtures.fixture_view(name)
        assert vfp.video_family_of(view) is None
        title = ifp.identify(view).family.title
        assert _refused(view) == (f"this is a {title} image GGUF, run it "
                                  "with gmlx image")


def test_an_image_file_gmlx_image_refuses_is_named_as_it_names_it(
        monkeypatch):
    import image_fixtures

    # FLUX.1, which the FLUX.2 klein row refuses by name.
    flux1 = ifp.HeaderView(kv={}, tensors={
        n: ifp.Tensor(sh, c) for n, sh, c in (
            ("double_blocks.0.img_attn.proj.weight", (3072, 3072), "Q4_K"),
            ("double_blocks.0.img_mod.lin.weight", (3072, 18432), "Q4_K"),
            ("single_blocks.0.linear1.weight", (3072, 21504), "Q4_K"),
            ("txt_in.weight", (4096, 3072), "Q4_K"),
            ("img_in.weight", (64, 3072), "BF16"))})
    # Krea 2 Raw, whose tensors match and whose weights gmlx image refuses.
    monkeypatch.setattr(ifp, "weight_identity", lambda view, names: (
        "284b54edd003f36239b1d30e62b64fba898dc0598d4434eb7985639fcaa6112c"))
    raw = image_fixtures.fixture_view("krea2_raw_q4_k_m.header.json")
    for view in (flux1, raw):
        with pytest.raises(ifp.ImageModelError) as ei:
            ifp.identify(view)
        assert _refused(view) == ("this is an image GGUF gmlx does not "
                                  f"load: {ei.value}")
