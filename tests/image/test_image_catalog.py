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


def test_a_recipe_resolves_by_name_or_alias():
    assert catalog.recipe("z-image-turbo").family == "z_image"
    assert catalog.recipe("Qwen-Image-2.1").name == "qwen-image"
    assert catalog.recipe("qwen-image-2.1-turbo").variant == "turbo"
    assert catalog.recipe("hf:unsloth/x") is None
    assert catalog.recipe_for("qwen_image21", "turbo").name == \
        "qwen-image-turbo"
    assert catalog.recipe_for("qwen_image21", "unknown").name == "qwen-image"
    assert catalog.recipe("krea-2-raw").variant == "raw"
    assert catalog.recipe_for("krea2", "raw").name == "krea-2"
    assert catalog.recipe_for("krea2", "turbo").name == "krea-2-turbo"


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


def _scan(root, **kw):
    images: list = []
    chat = discovery.scan_dirs([DiscoverSpec(dir=str(root), recursive=True)],
                               [str(root)], images=images, **kw)
    return chat, images


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
        assert {catalog.DEFAULT_QUANTS[r.name][c][1]
                for c in catalog.MEMORY_CLASSES} == {"Q8_0"}
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


def test_a_32_gb_mac_pulls_qwen_image_at_q8_0_since_its_f16_load_does_not_fit():
    assert catalog.default_quants(catalog.RECIPES["qwen-image"],
                                  32 << 30) == ("Q8_0", "Q8_0")
