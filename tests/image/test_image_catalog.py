"""The image catalog: recipes, memory classes, and the image files on disk
found by their weights, then paired by discovery."""
from __future__ import annotations

import hashlib
from pathlib import Path

import gguf
import numpy as np
import pytest

from gmlx.config import DiscoverSpec
from gmlx.image import catalog
from gmlx.image import text_encoder as te
from gmlx.load import discovery


def test_every_recipe_has_its_words_and_dit_sizes():
    from gmlx.image.catalog_facts import DIT_BYTES
    from helpers import banned_words

    assert set(DIT_BYTES) == set(catalog.RECIPES)
    for r in catalog.RECIPES.values():
        assert r.use and len(r.use) <= 31, r.name   # the GOOD FOR column at 80 columns
        assert r.blurb.endswith('.') and r.speed.endswith('.'), r.name
        assert len(f'  {r.blurb} Apache 2.0') <= 76 and len(f'  {r.speed}') <= 76, r.name
        assert set(DIT_BYTES[r.name]) == set(r.dit_quants), r.name
        assert all(DIT_BYTES[r.name].values()), r.name
        assert banned_words('\n'.join([r.use, r.blurb, r.speed])) == [], r.name


def test_download_bytes_sums_the_dit_the_encoder_and_the_vae():
    r = catalog.recipe('z-image-turbo')
    ram = 16 << 30
    dq, _eq = catalog.default_quants(r, ram)
    assert catalog.dit_bytes(r, dq) > 0 and catalog.dit_bytes(r, 'nope') == 0
    assert catalog.download_bytes(r, ram) == catalog.dit_bytes(r, dq) + catalog.companion_bytes(r, ram)
    assert catalog.download_bytes(r, ram, 'Q8_0') == (catalog.dit_bytes(r, 'Q8_0')
                                                     + catalog.companion_bytes(r, ram))
    krea = catalog.recipe('krea-2-turbo')
    assert catalog.image_size(krea, ram) == '512x512'
    assert catalog.image_size(krea, 24 << 30) == '1024x1024'
    assert catalog.image_size(r, ram) == catalog.IMAGE_SIZE == '1024x1024'


def test_a_recipe_resolves_by_name_or_alias():
    assert catalog.recipe("z-image-turbo").family == "z_image"
    assert catalog.recipe("Qwen-Image-2.1").name == "qwen-image"
    assert catalog.recipe("qwen-image-2.1-turbo").variant == "turbo"
    assert catalog.recipe("hf:unsloth/x") is None
    assert catalog.recipe_for("qwen_image21", "turbo").name == \
        "qwen-image-turbo"
    assert catalog.recipe_for("qwen_image21", "unknown").name == "qwen-image"
    assert catalog.recipe("krea-2-raw") is None
    assert catalog.recipe("krea-2") is None
    assert catalog.recipe_for("krea2", "turbo").name == "krea-2-turbo"
    assert catalog.recipe_for("flux3_klein", "distilled").name == "flux-3-klein"
    assert catalog.recipe_for("flux3_klein", "base").name == "flux-3-klein-base"
    assert catalog.recipe("FLUX-3-Klein").variant == "distilled"
    assert catalog.recipe_for("flux2_klein", "distilled").name == "flux-2-klein"
    assert catalog.recipe_for("flux2_klein", "base").name == "flux-2-klein-base"
    assert catalog.recipe_for("flux2_klein_9b", "distilled").name == \
        "flux-2-klein-9b"
    assert catalog.recipe_for("flux2_klein_9b", "base").name == \
        "flux-2-klein-base-9b"
    assert catalog.recipe_for("flux2_klein_9b", "unknown").name == \
        "flux-2-klein-9b"
    assert catalog.recipe_for("flux2_klein_9b_kv", "distilled").name == \
        "flux-2-klein-9b-kv"


@pytest.mark.parametrize("gb, cls", [(8, 16), (16, 16), (18, 16), (24, 24),
                                     (36, 32), (48, 32), (64, 64), (128, 64),
                                     (None, 16)])
def test_a_mac_takes_the_largest_memory_class_it_holds(gb, cls):
    assert catalog.memory_class(None if gb is None else gb << 30) == cls


def test_every_recipe_has_a_default_for_every_class_among_its_quants():
    for r in catalog.RECIPES.values():
        for c in catalog.MEMORY_CLASSES:
            dit, enc = catalog.DEFAULT_QUANTS[r.name][c]
            assert dit in r.dit_quants and enc in r.encoder_quants


def _write(path, arch, kv=(), tensors=(), gtype=None):
    w = gguf.GGUFWriter(str(path), arch)
    if gtype:
        w.add_type(gtype)
    for k, v in kv:
        if isinstance(v, int):
            w.add_uint32(k, v)
        else:
            w.add_string(k, v)
    for name, arr in tensors:
        w.add_tensor(name, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


_NORM = np.random.default_rng(0).standard_normal(64).astype(np.float32)


def _encoder(path, norm=_NORM):
    return _write(path, "qwen3", kv=(("qwen3.embedding_length", 64),
                                     ("qwen3.block_count", 4)),
                  tensors=(("blk.0.attn_norm.weight", norm),
                           ("blk.0.attn_q.weight",
                            np.zeros((64, 64), np.float32))))


def _vae(path, family="z_image", sha="ab" * 32):
    return _write(path, "autoencoder", gtype="vae",
                  kv=(("gmlx.vae.family", family),
                      ("gmlx.vae.source_sha256", sha)),
                  tensors=(("decoder.conv_in.weight",
                            np.zeros((4, 4), np.float32)),))


def _dit(path):
    return _write(path, "lumina2",
                  tensors=(("x_embedder.weight", np.zeros((8, 8), np.float32)),))


@pytest.fixture
def tiny_spec(monkeypatch):
    """Z-Image conditions on a 64-wide, 4-layer qwen3 whose identity is
    ``_NORM``, and its DiT fingerprint is any lumina2 file."""
    import mlx.core as mx

    row = te.IdentityTensor("blk.0.attn_norm.weight", "x",
                            te._bf16_digest(mx.array(_NORM)))
    spec = te.EncoderSpec(name="tiny", arch="qwen3", model_type="qwen3",
                          hidden=64, layers=4, tap_layers=(3,),
                          template="{}", identity=(row,))
    monkeypatch.setitem(te.ENCODERS, "z_image", spec)

    def dit_identity(path):
        with open(path, "rb") as f:
            head = f.read(4096)
        return ("z_image", "turbo") if b"lumina2" in head else None

    monkeypatch.setattr(catalog, "dit_identity", dit_identity)
    return spec


def test_files_are_found_by_their_weights_not_their_names(tmp_path,
                                                          tiny_spec):
    enc = _encoder(tmp_path / "renamed-anything.gguf")
    other = _encoder(tmp_path / "Qwen3-4B-Q4_K_M.gguf",
                     norm=_NORM + 1.0)                # a fine-tune
    vae = _vae(tmp_path / "v.gguf")
    dit = _dit(tmp_path / "d.gguf")
    got = {(f.role, f.path) for f in catalog.scan([tmp_path])}
    assert got == {("encoder", enc), ("vae", vae), ("dit", dit)}
    assert other not in {p for _r, p in got}
    assert catalog.vae_identity(vae) == ("z_image", "ab" * 32)


def test_discovery_classifies_image_files_by_their_header(tmp_path):
    vae = _vae(tmp_path / "v.gguf")
    dit = _dit(tmp_path / "d.gguf")
    assert discovery.classify_gguf(vae).kind == "image-vae"
    assert discovery.classify_gguf(dit).kind == "image-dit"
    assert discovery.classify_gguf(_encoder(tmp_path / "e.gguf")).kind \
        == "model"


def test_a_flux_labeled_file_of_another_model_is_skipped_as_an_image_file(
        tmp_path, capsys):
    """Krea 2 files may carry the flux label, so discovery reads the
    tensors of a flux file, and skips one whose tensors are another
    model's."""
    flux = _write(tmp_path / "flux1-dev-Q4_K_M.gguf", "flux", tensors=(
        ("double_blocks.0.img_attn.proj.weight", np.zeros((8, 8), np.float32)),))
    assert discovery.classify_gguf(flux).kind == "image-dit"
    assert catalog.classify(flux) == []
    chat, images = _scan(tmp_path)
    assert (chat, images) == ([], [])
    assert (f"skip (an image GGUF gmlx does not load): {flux}"
            in capsys.readouterr().err)


def _scan(root, **kw):
    images: list = []
    chat = discovery.scan_dirs([DiscoverSpec(dir=str(root), recursive=True)],
                               [str(root)], images=images, **kw)
    return chat, images


def test_the_klein_recipes_share_bfls_bf16_flux2_autoencoder():
    """FLUX.2 klein at both sizes and FLUX.3 klein pull one VAE: BFL's BF16
    file of the FLUX.2 autoencoder from the 4B repository at a pinned
    revision, with its config, packed in its own dtype under an
    Apache-2.0 license line."""
    names = ("flux-2-klein", "flux-2-klein-base", "flux-2-klein-9b",
             "flux-2-klein-base-9b", "flux-2-klein-9b-kv", "flux-3-klein",
             "flux-3-klein-base")
    rows = {(r.vae_repo, r.vae_revision, r.vae_file, r.vae_config,
             r.vae_sha256, r.vae_pack, r.vae_license, r.vae_bytes)
            for r in map(catalog.recipe, names)}
    assert rows == {("black-forest-labs/FLUX.2-klein-4B",
                     "e7b7dc27f91deacad38e78976d1f2b499d76a294",
                     "vae/diffusion_pytorch_model.safetensors",
                     "vae/config.json",
                     "ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04",
                     "flux2-vae-BF16.gguf", "apache-2.0", catalog.recipe(
                         "flux-2-klein").vae_bytes)}
    for name in names[:2]:
        r = catalog.recipe(name)
        assert (r.family, r.license, r.encoder_license) == (
            "flux2_klein", "apache-2.0", "apache-2.0")
        assert (r.encoder_repo, r.encoder_name("Q8_0")) == (
            "unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q8_0.gguf")
        assert r.encoder_bytes == catalog.recipe(
            "z-image-turbo").encoder_bytes
    assert catalog.recipe("flux-2-klein").dit_name("Q8_0") == \
        "flux-2-klein-4b-Q8_0.gguf"
    assert catalog.recipe("flux-2-klein-base").dit_name("Q4_K_M") == \
        "flux-2-klein-base-4b-Q4_K_M.gguf"


NON_COMMERCIAL = ("non-commercial use only, with content filters or a review "
                  "of each image for unlawful or infringing content before it "
                  "is shared, and a disclosure that it is made with AI where "
                  "the law requires one")


@pytest.mark.parametrize("name, family, variant, repo, license", [
    ("flux-2-klein-9b", "flux2_klein_9b", "distilled",
     "unsloth/FLUX.2-klein-9B-GGUF", "FLUX.2-klein-9B/blob/main/LICENSE.md"),
    ("flux-2-klein-base-9b", "flux2_klein_9b", "base",
     "unsloth/FLUX.2-klein-base-9B-GGUF",
     "FLUX.2-klein-base-9B/blob/main/LICENSE.md"),
    ("flux-2-klein-9b-kv", "flux2_klein_9b_kv", "distilled",
     "QuantStack/FLUX.2-Klein-9B-KV-GGUF",
     "FLUX.2-klein-9b-kv/blob/main/LICENSE")])
def test_the_9b_recipes_carry_the_flux_non_commercial_license(
        name, family, variant, repo, license):
    """The 9B checkpoints are under the FLUX Non-Commercial License, and
    the recipes say so, as Qwen-Image 2.1's do. They condition on the
    Qwen3-8B Q8_0 GGUF, Apache-2.0, and pull the DiT of the row of their
    variant."""
    from gmlx.image.registry import get_family

    r = catalog.recipe(name)
    assert (r.family, r.variant, r.dit_repo) == (family, variant, repo)
    assert r.license == "flux-non-commercial-license"
    assert r.license_note == NON_COMMERCIAL
    assert r.license_url == f"https://huggingface.co/black-forest-labs/{license}"
    assert (r.encoder_repo, r.encoder_name("Q8_0"), r.encoder_quants) == (
        "unsloth/Qwen3-8B-GGUF", "Qwen3-8B-Q8_0.gguf", ("Q8_0",))
    assert (r.encoder_license, r.encoder_bytes) == ("apache-2.0",
                                                    8_709_519_168)
    rows = {v.name: v for v in get_family(family).variants.values()}
    assert rows[variant].repo == r.dit_repo
    for q in r.dit_quants:
        assert r.dit_name(q) in rows[variant].files


KLEIN = ("flux3_klein", "flux2_klein", "flux2_klein_9b", "flux2_klein_9b_kv")


@pytest.mark.parametrize("label", KLEIN)
def test_a_pack_of_the_flux2_autoencoder_serves_every_klein_family(
        tmp_path, monkeypatch, label):
    """The scan offers a pack of the FLUX.2 autoencoder to FLUX.2 klein at
    both sizes and FLUX.3 klein whichever family it names, and a pull of
    any takes it from disk. A VAE of another autoencoder serves its
    family alone."""
    from gmlx.commands import image_pull

    others = tuple(f for f in KLEIN if f != label)
    assert catalog.vae_families(label) == (label, *others)
    for fam in ("z_image", "qwen_image21", "krea2", "nonesuch"):
        assert catalog.vae_families(fam) == (fam,)
    sha = catalog.recipe("flux-2-klein").vae_sha256
    vae = _vae(tmp_path / "flux2-vae-BF16.gguf", family=label, sha=sha)
    (tmp_path / "krea").mkdir()
    _vae(tmp_path / "krea" / "qwen-image-vae-BF16.gguf", family="krea2")
    got = {(f.role, f.family, f.path) for f in catalog.scan([tmp_path])}
    assert {g for g in got if g[2] == vae} == {("vae", f, vae) for f in KLEIN}
    monkeypatch.setattr(image_pull, "_repo_sizes", lambda *a: {})
    for name in ("flux-2-klein", "flux-2-klein-base", "flux-2-klein-9b",
                 "flux-2-klein-base-9b", "flux-2-klein-9b-kv", "flux-3-klein",
                 "flux-3-klein-base"):
        plan = image_pull.plan_pull(
            catalog.recipe(name), dit_quant="Q8_0", encoder_quant="Q8_0",
            encoder_quant_given=False, only=image_pull.PARTS,
            roots=[str(tmp_path)], base_dir=str(tmp_path), nest=True)
        part = plan.part("vae")
        assert (part.path, part.download) == (vae, False), name


def test_discovery_pairs_a_dit_with_its_encoder_and_vae(tmp_path, tiny_spec,
                                                        capsys):
    (tmp_path / "enc").mkdir()
    (tmp_path / "dit").mkdir()
    enc = _encoder(tmp_path / "enc" / "Qwen3-4B-Q4_K_M.gguf")
    dit = _dit(tmp_path / "dit" / "z-image-turbo-Q4_K_M.gguf")
    far = _vae(tmp_path / "enc" / "far-vae.gguf")
    near = _vae(tmp_path / "dit" / "near-vae.gguf")
    chat, images = _scan(tmp_path)
    assert [m.path for m in chat] == [enc]
    (im,) = images
    assert im.path == dit and im.vae == near and im.vae != far
    # The encoder is a chat model of this scan, so the entry shares it.
    assert im.encoder == chat[0].id
    assert im.id.startswith("z-image-turbo")
    assert "image model" in capsys.readouterr().err


@pytest.mark.parametrize("fam, label", [("flux2_klein", "flux3_klein"),
                                        ("flux3_klein", "flux2_klein"),
                                        ("flux2_klein_9b", "flux3_klein"),
                                        ("flux2_klein_9b", "flux2_klein"),
                                        ("flux3_klein", "flux2_klein_9b")])
def test_discovery_pairs_a_klein_dit_with_the_other_klein_familys_vae(
        tmp_path, tiny_spec, monkeypatch, fam, label):
    """A pack of the FLUX.2 autoencoder named for one klein family pairs
    with a DiT of the other, and a VAE of another autoencoder beside the
    DiT does not."""
    # The klein registry rows give their encoder specs.
    spec_of = te.encoder_spec
    monkeypatch.setattr(te, "encoder_spec",
                        lambda f: tiny_spec if f == fam else spec_of(f))

    def dit_identity(path):
        with open(path, "rb") as f:
            head = f.read(4096)
        return (fam, "distilled") if b"lumina2" in head else None

    monkeypatch.setattr(catalog, "dit_identity", dit_identity)
    (tmp_path / "dit").mkdir()
    (tmp_path / "vae").mkdir()
    _encoder(tmp_path / "Qwen3-4B-Q8_0.gguf")
    _dit(tmp_path / "dit" / "klein-Q4_K_M.gguf")
    _vae(tmp_path / "dit" / "qwen-image-vae-BF16.gguf", family="krea2")
    vae = _vae(tmp_path / "vae" / "flux2-vae-BF16.gguf", family=label)
    _chat, images = _scan(tmp_path)
    (im,) = images
    assert im.vae == vae


def test_a_configured_chat_model_is_shared_as_the_encoder(tmp_path,
                                                          tiny_spec):
    lib = tmp_path / "lib"
    lib.mkdir()
    enc = _encoder(tmp_path / "Qwen3-4B-Q4_K_M.gguf")
    _dit(lib / "z-image-turbo-Q4_K_M.gguf")
    _vae(lib / "v.gguf")
    _chat, images = _scan(lib, encoder_ids={enc: "qwen3-4b"})
    assert images[0].encoder == "qwen3-4b"


@pytest.mark.parametrize("towers, shared", [(set(), False),
                                            ({"vl"}, True)])
def test_a_chat_model_is_shared_only_with_the_tower_its_family_wants(
        tmp_path, tiny_spec, monkeypatch, towers, shared):
    monkeypatch.setattr(catalog, "wants_tower", lambda fam: True)
    lib = tmp_path / "lib"
    lib.mkdir()
    enc = _encoder(tmp_path / "Qwen3-4B-Q4_K_M.gguf")
    _dit(lib / "z-image-turbo-Q4_K_M.gguf")
    _vae(lib / "v.gguf")
    _chat, images = _scan(lib, encoder_ids={enc: "vl"}, tower_ids=towers)
    assert images[0].encoder == ("vl" if shared else enc)


def test_an_incomplete_set_names_the_pull_that_fills_it(tmp_path, tiny_spec,
                                                        capsys):
    _dit(tmp_path / "z-image-turbo-Q4_K_M.gguf")
    _chat, images = _scan(tmp_path)
    assert images == []
    err = capsys.readouterr().err
    assert "without its encoder and vae" in err
    assert "gmlx pull z-image-turbo --quant Q4_K_M --only encoder,vae" in err


def test_merge_adds_the_paired_image_models(tmp_path, tiny_spec):
    from gmlx.config import ServerCfg, ServerDefaults

    _encoder(tmp_path / "Qwen3-4B-Q4_K_M.gguf")
    _dit(tmp_path / "z-image-turbo-Q4_K_M.gguf")
    _vae(tmp_path / "v.gguf")
    cfg = ServerCfg(host="127.0.0.1", port=8080, model_dirs=[str(tmp_path)],
                    defaults=ServerDefaults(), models={},
                    discover=[DiscoverSpec(dir=str(tmp_path))])
    discovery.merge_discovered(cfg)
    (iid,) = cfg.image_models
    assert cfg.image_models[iid].encoder in cfg.models


def test_a_chat_folder_reads_no_image_tables(tmp_path):
    import subprocess
    import sys

    _encoder(tmp_path / "Qwen3-4B-Q4_K_M.gguf")
    code = ("import sys\n"
            "from gmlx.config import DiscoverSpec\n"
            "from gmlx.load import discovery\n"
            f"discovery.scan_dirs([DiscoverSpec(dir={str(tmp_path)!r})], [],"
            " images=[])\n"
            "bad = sorted(m for m in sys.modules if m.startswith('gmlx.image'))\n"
            "assert not bad, bad\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]


def test_sha256_file_reads_the_whole_file(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"x" * (3 << 20))
    assert catalog.sha256_file(str(p)) == hashlib.sha256(
        b"x" * (3 << 20)).hexdigest()


def test_every_class_pulls_the_q8_encoder_and_the_largest_the_reference_dit():
    for r in catalog.RECIPES.values():
        # No 16 GB budget holds the 9.5 GB Q8_0 file of Qwen3.5-9B.
        small = {16} if r.family == "moss_image" else set()
        assert {catalog.DEFAULT_QUANTS[r.name][c][1]
                for c in catalog.MEMORY_CLASSES if c not in small} == {"Q8_0"}
        assert {catalog.DEFAULT_QUANTS[r.name][c][1]
                for c in small} <= {"Q4_K_M"}
        assert catalog.DEFAULT_QUANTS[r.name][64][0] == r.dit_quants[-1]


@pytest.mark.integration
def test_a_krea_2_pull_takes_the_parts_on_disk_by_their_weights(gguf_dir,
                                                               tmp_path):
    from image_dir import image_dir

    from gmlx.commands import image_pull

    r = catalog.RECIPES["krea-2-turbo"]
    roots = [str(gguf_dir), str(image_dir())]
    found = catalog.scan(roots, families=("krea2",))
    dits = [f.path for f in found if f.role == "dit" and f.variant == "turbo"]
    vaes = [f for f in found if f.role == "vae"]
    if not dits or not vaes or not any(
            p.name.startswith("Qwen3-VL-4B-Instruct-")
            for d in roots for p in Path(d).rglob("*.gguf")):
        pytest.skip("a Krea 2 Turbo DiT, its encoder and its packed VAE are "
                    "not all under KQUANT_TEST_GGUF_DIR and GMLX_TEST_IMAGE_DIR")
    plan = image_pull.plan_pull(
        r, dit_quant=image_pull._quant_of(dits[0]), encoder_quant="Q8_0",
        encoder_quant_given=False, only=image_pull.PARTS, roots=roots,
        base_dir=str(tmp_path), nest=True)
    assert [(p.role, p.download) for p in plan.parts] == [
        ("dit", False), ("encoder", False), ("vae", False)]
    assert Path(plan.part("encoder").path).name.startswith(
        "Qwen3-VL-4B-Instruct-")
    assert catalog.vae_identity(plan.part("vae").path) == ("krea2",
                                                           r.vae_sha256)


@pytest.mark.integration
@pytest.mark.parametrize("name", ["flux-3-klein", "flux-3-klein-base"])
def test_a_flux3_pull_takes_the_krea_2_encoder_file_by_its_weights(
        gguf_dir, tmp_path, name):
    from image_dir import image_dir

    from gmlx.commands import image_pull

    r = catalog.RECIPES[name]
    roots = [str(gguf_dir), str(image_dir())]
    found = catalog.scan(roots, families=("flux3_klein", "krea2"))
    dits = [f.path for f in found if f.role == "dit"
            and f.family == "flux3_klein" and f.variant == r.variant]
    vaes = [f for f in found if f.role == "vae" and f.family == "flux3_klein"
            and f.sha256 == r.vae_sha256]
    encs = {f.family: f.path for f in found if f.role == "encoder"}
    if not dits or not vaes or "flux3_klein" not in encs:
        pytest.skip(f"a {r.title} DiT, the Qwen3-VL-4B encoder and the packed "
                    "BF16 FLUX.2 VAE are not all under KQUANT_TEST_GGUF_DIR "
                    "and GMLX_TEST_IMAGE_DIR")
    # One encoder file serves both families.
    assert encs.get("krea2") == encs["flux3_klein"]
    plan = image_pull.plan_pull(
        r, dit_quant=image_pull._quant_of(dits[0]), encoder_quant="Q8_0",
        encoder_quant_given=False, only=image_pull.PARTS, roots=roots,
        base_dir=str(tmp_path), nest=True)
    assert [(p.role, p.download) for p in plan.parts] == [
        ("dit", False), ("encoder", False), ("vae", False)]
    assert plan.part("encoder").path == encs["flux3_klein"]
    label, sha = catalog.vae_identity(plan.part("vae").path)
    assert sha == r.vae_sha256 and "flux3_klein" in catalog.vae_families(label)
    assert catalog.dit_identity(plan.part("dit").path) == ("flux3_klein",
                                                           r.variant)


@pytest.mark.integration
@pytest.mark.parametrize("name", ["flux-2-klein", "flux-2-klein-base"])
def test_a_flux2_pull_takes_the_z_image_encoder_and_the_klein_vae_on_disk(
        gguf_dir, tmp_path, name):
    """The Qwen3-4B file of Z-Image and the FLUX.2 VAE pack, whichever
    klein family it names, serve the pull with nothing downloaded."""
    from image_dir import image_dir

    from gmlx.commands import image_pull

    r = catalog.RECIPES[name]
    roots = [str(gguf_dir), str(image_dir())]
    found = catalog.scan(roots, families=("flux2_klein", "z_image"))
    dits = [f.path for f in found if f.role == "dit"
            and f.family == "flux2_klein" and f.variant == r.variant]
    vaes = [f for f in found if f.role == "vae" and f.family == "flux2_klein"
            and f.sha256 == r.vae_sha256]
    encs = {f.family: f.path for f in found if f.role == "encoder"
            and Path(f.path).name == "Qwen3-4B-Q8_0.gguf"}
    if not dits or not vaes or "flux2_klein" not in encs:
        pytest.skip(f"a {r.title} DiT, the Qwen3-4B Q8_0 encoder and the "
                    "packed BF16 FLUX.2 VAE are not all under "
                    "KQUANT_TEST_GGUF_DIR and GMLX_TEST_IMAGE_DIR")
    assert encs.get("z_image") == encs["flux2_klein"]
    plan = image_pull.plan_pull(
        r, dit_quant=image_pull._quant_of(dits[0]), encoder_quant="Q8_0",
        encoder_quant_given=False, only=image_pull.PARTS, roots=roots,
        base_dir=str(tmp_path), nest=True)
    assert [(p.role, p.download) for p in plan.parts] == [
        ("dit", False), ("encoder", False), ("vae", False)]
    assert plan.part("encoder").path == encs["flux2_klein"]
    label, sha = catalog.vae_identity(plan.part("vae").path)
    assert sha == r.vae_sha256 and "flux2_klein" in catalog.vae_families(label)
    assert catalog.dit_identity(plan.part("dit").path) == ("flux2_klein",
                                                           r.variant)


def test_a_32_gb_mac_pulls_qwen_image_at_q8_0_since_its_f16_load_does_not_fit():
    assert catalog.default_quants(catalog.RECIPES["qwen-image"],
                                  32 << 30) == ("Q8_0", "Q8_0")


def test_each_class_default_encoder_has_its_own_size():
    """A recipe whose class defaults take more than one encoder quant has
    each file in ENCODER_BYTES, so the fit of a DiT prices the one its
    class pulls."""
    for r in catalog.RECIPES.values():
        names = {r.encoder_name(catalog.DEFAULT_QUANTS[r.name][c][1])
                 for c in catalog.MEMORY_CLASSES}
        if len(names) > 1:
            assert names <= set(catalog.ENCODER_BYTES), r.name
    m = catalog.RECIPES["moss-image"]
    assert catalog.companion_bytes(m, 16 << 30) == 5_680_522_464 + m.vae_bytes
    assert catalog.companion_bytes(m, 64 << 30) == m.encoder_bytes + m.vae_bytes
