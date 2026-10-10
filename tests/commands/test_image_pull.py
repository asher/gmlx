"""``gmlx pull <image model>``: the plan, reuse by identity, the summary,
the VAE pack and its checks, and the image_models entry."""
from __future__ import annotations

import json
import os

import pytest

import gmlx.commands.manage as manage
import gmlx.load.remote as remote
from gmlx.image import catalog

Z = catalog.RECIPES["z-image-turbo"]
Q = catalog.RECIPES["qwen-image"]


@pytest.fixture
def lib(tmp_path, monkeypatch):
    """A config whose model_dirs is <tmp>/lib, no network, 1 TB free, a
    32 GB Mac, and no running server."""
    root = tmp_path / "lib"
    root.mkdir()
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"server:\n  model_dirs:\n    - {root}\n")
    monkeypatch.setattr(remote, "hf_list_dir", lambda *a, **k: [])
    monkeypatch.setattr(manage, "_disk_free", lambda p: 1 << 40)
    monkeypatch.setattr("gmlx.load.memfit.total_ram_bytes", lambda: 32 << 30)
    monkeypatch.setattr("gmlx.serve.server._reload_running",
                        lambda path, skip: None)
    monkeypatch.setattr(catalog, "encoder_matches", lambda p, f, kv=None: (
        os.path.basename(p).startswith(("Qwen3-4B", "Qwen3-VL"))))
    return root, cfg


class Disk:
    """What catalog.scan sees: files by role, rebuilt from the folder."""

    def __init__(self, monkeypatch, root, sha=Z.vae_sha256):
        self.root = root
        self.sha = sha
        monkeypatch.setattr(catalog, "scan", self.scan)

    def scan(self, roots, *, recursive=True, families=None):
        out = []
        for r in roots:
            for d, _s, files in os.walk(r):
                for f in sorted(files):
                    p = os.path.join(d, f)
                    if not f.endswith(".gguf"):
                        continue
                    if f.startswith("z-image-turbo"):
                        out.append(catalog.ImageFile(p, "dit", "z_image",
                                                     "turbo"))
                    elif f.startswith("Qwen3-4B"):
                        out.append(catalog.ImageFile(p, "encoder", "z_image"))
                    elif "vae" in f:
                        out.append(catalog.ImageFile(p, "vae", "z_image",
                                                     sha256=self.sha))
        return out


def _downloads(monkeypatch, sizes=None):
    got = []

    def fake(repo, filename, revision, dest_dir):
        dest = os.path.join(dest_dir, filename)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(b"{}" if filename.endswith(".json") else b"x" * 10)
        got.append((repo, filename, revision))
        return dest

    monkeypatch.setattr(manage, "_hf_download", fake)
    return got


def _packs(monkeypatch, *, sha=Z.vae_sha256, reads_back=True):
    import gmlx.image.vae as vae_pkg

    packed = []

    def pack(src, dst, **kw):
        with open(dst, "wb") as f:
            f.write(b"g" * 20)
        packed.append((src, dst, kw))

    monkeypatch.setattr(vae_pkg, "pack_vae", pack)
    monkeypatch.setattr(catalog, "sha256_file", lambda p: sha)
    monkeypatch.setattr(catalog, "vae_identity", lambda p: (
        ("z_image", Z.vae_sha256) if reads_back else ("z_image", "0" * 64)))
    return packed


def _cfg(cfg):
    from gmlx.config import load_config

    return load_config(cfg)


def test_a_bare_pull_fetches_all_three_packs_the_vae_and_registers(
        lib, monkeypatch, capsys):
    root, cfg = lib
    Disk(monkeypatch, root)
    got = _downloads(monkeypatch)
    packed = _packs(monkeypatch)
    assert manage.cmd_pull(["z-image-turbo", "--config", str(cfg)]) == 0
    dq, eq = catalog.DEFAULT_QUANTS["z-image-turbo"][32]
    assert (Z.dit_repo, Z.dit_name(dq), "main") in got
    assert (Z.encoder_repo, Z.encoder_name(eq), "main") in got
    assert (Z.vae_repo, Z.vae_file, Z.vae_revision) in got
    (src, dst, kw) = packed[0]
    assert kw["source_revision"] == Z.vae_revision and kw["family"] == "z_image"
    # The source is gone after the pack, and the packed GGUF stays.
    assert not os.path.exists(src) and os.path.exists(dst)
    c = _cfg(cfg)
    m = c.image_models["z-image-turbo"]
    assert m.path == f"unsloth__Z-Image-Turbo-GGUF/{Z.dit_name(dq)}"
    assert m.vae == f"Tongyi-MAI__Z-Image-Turbo/{Z.vae_pack}"
    assert m.encoder.endswith(Z.encoder_name(eq))
    assert c.defaults.image_model == "z-image-turbo"
    out = capsys.readouterr().out
    assert "license: Z-Image-Turbo apache-2.0" in out
    assert "the default for a 32 GB Mac" in out


def test_keep_source_keeps_the_safetensors(lib, monkeypatch):
    root, cfg = lib
    Disk(monkeypatch, root)
    _downloads(monkeypatch)
    packed = _packs(monkeypatch)
    assert manage.cmd_pull(["z-image-turbo", "--config", str(cfg),
                            "--keep-source"]) == 0
    assert os.path.exists(packed[0][0])


def test_a_vae_source_with_another_digest_is_refused(lib, monkeypatch,
                                                     capsys):
    root, cfg = lib
    Disk(monkeypatch, root)
    _downloads(monkeypatch)
    packed = _packs(monkeypatch, sha="0" * 64)
    assert manage.cmd_pull(["z-image-turbo", "--config", str(cfg)]) == 1
    assert packed == []
    assert "has sha256" in capsys.readouterr().err
    assert _cfg(cfg).image_models == {}


def test_a_packed_vae_that_reads_back_wrong_is_refused(lib, monkeypatch,
                                                       capsys):
    root, cfg = lib
    Disk(monkeypatch, root)
    _downloads(monkeypatch)
    _packs(monkeypatch, reads_back=False)
    assert manage.cmd_pull(["z-image-turbo", "--config", str(cfg)]) == 1
    assert "does not read back" in capsys.readouterr().err


def test_parts_on_disk_are_used_where_they_are(lib, monkeypatch, capsys):
    root, cfg = lib
    other = root / "elsewhere"
    other.mkdir()
    (other / "Qwen3-4B-Q8_0.gguf").write_bytes(b"e")
    (other / "packed-vae.gguf").write_bytes(b"v")
    Disk(monkeypatch, root)
    got = _downloads(monkeypatch)
    _packs(monkeypatch)
    assert manage.cmd_pull(["z-image-turbo", "--config", str(cfg)]) == 0
    # Only the DiT downloads: the encoder matches at another quant, and the
    # VAE by its source digest, whatever their names.
    assert [g[0] for g in got] == [Z.dit_repo]
    m = _cfg(cfg).image_models["z-image-turbo"]
    assert m.encoder == "elsewhere/Qwen3-4B-Q8_0.gguf"
    assert m.vae == "elsewhere/packed-vae.gguf"
    assert "on disk" in capsys.readouterr().out


def test_an_encoder_quant_that_is_asked_for_is_not_replaced(lib, monkeypatch,
                                                            capsys):
    root, cfg = lib
    (root / "Qwen3-4B-Q8_0.gguf").write_bytes(b"e")
    Disk(monkeypatch, root)
    got = _downloads(monkeypatch)
    _packs(monkeypatch)
    assert manage.cmd_pull(["z-image-turbo", "--config", str(cfg),
                            "--encoder-quant", "q4_k_m"]) == 0
    assert (Z.encoder_repo, "Qwen3-4B-Q4_K_M.gguf", "main") in got
    # A quant asked for is not the memory class default.
    head = capsys.readouterr().out.splitlines()[0]
    assert head.endswith("encoder Q4_K_M") and "default" not in head


def test_a_configured_chat_model_is_written_as_the_encoder_id(lib,
                                                              monkeypatch):
    root, cfg = lib
    (root / "Qwen3-4B-Q4_K_M.gguf").write_bytes(b"e")
    cfg.write_text(cfg.read_text()
                   + "models:\n  qwen3-4b:\n    path: Qwen3-4B-Q4_K_M.gguf\n")
    Disk(monkeypatch, root)
    _downloads(monkeypatch)
    _packs(monkeypatch)
    assert manage.cmd_pull(["z-image-turbo", "--config", str(cfg)]) == 0
    assert _cfg(cfg).image_models["z-image-turbo"].encoder == "qwen3-4b"


def test_the_summary_gives_sizes_and_free_disk_before_the_download(
        lib, monkeypatch, capsys):
    root, cfg = lib
    Disk(monkeypatch, root)
    dq, eq = catalog.DEFAULT_QUANTS["z-image-turbo"][32]
    listing = {Z.dit_name(dq): 5 << 30, Z.encoder_name(eq): 2 << 30,
               Z.vae_file: 160 << 20, Z.vae_config: 1000}
    monkeypatch.setattr(remote, "hf_list_dir", lambda repo, path, rev="main",
                        **k: [(n, "file", s) for n, s in listing.items()])
    monkeypatch.setattr(manage, "_disk_free", lambda p: 3 << 30)
    got = _downloads(monkeypatch)
    assert manage.cmd_pull(["z-image-turbo", "--config", str(cfg)]) == 1
    assert got == []
    out, err = capsys.readouterr()
    assert "download 7.2 GB, 3.0 GB free at" in out
    # An image model holds no KV cache, so the fit line names none.
    assert "the three parts: 7.2 GB, which fit this Mac's 32 GB RAM" in out
    assert "KV" not in out
    assert "not enough disk space" in err


def test_only_fetches_the_named_parts_and_registers_once_whole(lib,
                                                               monkeypatch,
                                                               capsys):
    root, cfg = lib
    Disk(monkeypatch, root)
    got = _downloads(monkeypatch)
    _packs(monkeypatch)
    assert manage.cmd_pull(["z-image-turbo", "--config", str(cfg),
                            "--only", "vae"]) == 0
    assert [g[0] for g in got] == [Z.vae_repo, Z.vae_repo]
    assert _cfg(cfg).image_models == {}
    out, err = capsys.readouterr()
    assert "not registered: dit, encoder" in err
    # A part left out has no file yet, so the plan names none for it.
    assert "skipped (--only)" in out and "unsloth__Z-Image-Turbo-GGUF" not in out
    assert manage.cmd_pull(["z-image-turbo", "--config", str(cfg),
                            "--only", "dit,encoder"]) == 0
    assert "z-image-turbo" in _cfg(cfg).image_models


def test_a_second_pull_of_the_same_dit_adds_no_entry(lib, monkeypatch,
                                                     capsys):
    root, cfg = lib
    Disk(monkeypatch, root)
    _downloads(monkeypatch)
    _packs(monkeypatch)
    assert manage.cmd_pull(["z-image-turbo", "--config", str(cfg)]) == 0
    assert manage.cmd_pull(["z-image-turbo", "--config", str(cfg)]) == 0
    assert list(_cfg(cfg).image_models) == ["z-image-turbo"]
    assert "already configured as z-image-turbo" in capsys.readouterr().out


@pytest.mark.parametrize("argv, msg", [
    (["--quant", "Q2_K"], "--quant Q2_K is not a quant of z-image-turbo"),
    (["--encoder-quant", "F16"], "is not a quant"),
    (["--only", "dit,tower"], "--only takes"),
])
def test_bad_image_flags_are_usage_errors(lib, monkeypatch, capsys, argv,
                                          msg):
    _root, cfg = lib
    assert manage.cmd_pull(["z-image-turbo", "--config", str(cfg)] + argv) \
        == 2
    assert msg in capsys.readouterr().err


def test_image_flags_need_an_image_model_name(capsys):
    assert manage.cmd_pull(["hf:org/repo/x.gguf", "--quant", "Q4_K_M"]) == 2
    assert "go with an image model name" in capsys.readouterr().err


def test_json_prints_the_plan(lib, monkeypatch, capsys):
    root, cfg = lib
    Disk(monkeypatch, root)
    _downloads(monkeypatch)
    _packs(monkeypatch)
    assert manage.cmd_pull(["z-image-turbo", "--config", str(cfg),
                            "--json", "--no-register"]) == 0
    out = capsys.readouterr().out
    plan = json.loads(out[:out.index("\n}\n") + 2])
    assert plan["memory_class"] == 32 and plan["license"] == "apache-2.0"
    assert [p["role"] for p in plan["parts"]] == ["dit", "encoder", "vae"]
    assert _cfg(cfg).image_models == {}


def test_the_qwen_recipe_takes_the_vision_tower_beside_its_encoder(
        lib, monkeypatch, capsys):
    root, cfg = lib
    monkeypatch.setattr(catalog, "scan", lambda *a, **k: [])
    got = _downloads(monkeypatch)
    monkeypatch.setattr(catalog, "sha256_file", lambda p: Q.vae_sha256)
    monkeypatch.setattr(catalog, "vae_identity",
                        lambda p: ("qwen_image21", Q.vae_sha256))
    import gmlx.image.vae as vae_pkg
    monkeypatch.setattr(vae_pkg, "pack_vae", lambda src, dst, **kw: open(
        dst, "wb").write(b"g"))
    assert manage.cmd_pull(["qwen-image", "--config", str(cfg)]) == 0
    assert (Q.encoder_repo, Q.mmproj_file, "main") in got
    m = _cfg(cfg).image_models["qwen-image"]
    assert m.encoder_mmproj.endswith(Q.mmproj_file)
    assert m.vae.endswith(Q.vae_pack)
    # The research license allows no commercial use, and the plan says so.
    assert ("license: Qwen-Image 2.1 qwen-research, non-commercial use only "
            "(https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE)"
            in capsys.readouterr().out)


# validate and pull of a single image GGUF
def _report(arch, gtype=None):
    return remote.HeaderReport(arch, {"Q4_K": 3}, {}, 3, gtype)


def test_an_image_dit_skips_the_chat_arch_gate_and_fits_with_its_parts():
    ref = remote.parse_ref("hf:unsloth/Z-Image-Turbo-GGUF/z-image-turbo-Q4_K_M.gguf")
    v = manage._verdict(ref, _report("lumina2"), size_bytes=5 << 30)
    assert v["image"] == "dit" and v["image_family"] == "z_image"
    assert v["combined_bytes"] == (5 << 30) + Z.encoder_bytes + Z.vae_bytes
    assert v["usable"] and v["arch_error"] is None


def test_an_image_vae_is_a_companion():
    v = manage._verdict(remote.parse_ref("hf:o/r/v.gguf"),
                        _report("autoencoder", "vae"), size_bytes=1 << 20)
    assert v["image"] == "vae" and v["fit"] is None and v["usable"]


def test_a_chat_gguf_is_no_image():
    v = manage._verdict(remote.parse_ref("hf:o/r/m.gguf"), _report("llama"),
                        size_bytes=1 << 30)
    assert v["image"] is None and v["loadable"]


# list, ps, rm and doctor over image_models entries
_IMG_CFG = """
server:
  model_dirs:
    - <LIB>
  defaults:
    image_model: zit
models:
  qwen3-4b:
    path: enc.gguf
  vl:
    path: vl.gguf
    mmproj: tower.gguf
image_models:
  zit:
    path: zit.gguf
    encoder: qwen3-4b
    vae: zvae.gguf
  qi:
    path: qi.gguf
    encoder: vl
    vae: qvae.gguf
  qt:
    path: qt.gguf
    encoder: own.gguf
    encoder_mmproj: tower.gguf
    vae: qvae.gguf
"""

_FILES = ("enc.gguf", "vl.gguf", "tower.gguf", "zit.gguf", "zvae.gguf",
          "qi.gguf", "qt.gguf", "qvae.gguf", "own.gguf")


@pytest.fixture
def images(tmp_path, monkeypatch):
    lib = tmp_path / "lib"
    lib.mkdir()
    for f in _FILES:
        (lib / f).write_bytes(b"x" * 100)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(_IMG_CFG.replace("<LIB>", str(lib)))
    fams = {"zit.gguf": ("z_image", "turbo"), "qi.gguf": ("qwen_image21", "base"),
            "qt.gguf": ("qwen_image21", "turbo")}
    monkeypatch.setattr(catalog, "dit_identity",
                        lambda p: fams.get(os.path.basename(p)))
    monkeypatch.setattr("gmlx.serve.server._reload_running",
                        lambda path, skip: None)
    return lib, cfg


def test_list_shows_the_kind_and_the_image_entries(images, capsys):
    _lib, cfg = images
    assert manage.cmd_list(["--config", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "KIND" in out
    z = next(line for line in out.splitlines() if "zit" in line)
    assert "image" in z and "Z-Image" in z and "encoder qwen3-4b" in z
    assert "* zit" in out and "default image model: zit" in out
    q = next(line for line in out.splitlines() if " qi " in line)
    assert "edit" in q


@pytest.mark.parametrize("flags, ids", [
    (["--image"], {"zit", "qi", "qt"}),
    (["--family", "qwen-image-2.1"], {"qi", "qt"}),
    (["--family", "z_image"], {"zit"}),
    (["--edit"], {"qi", "qt"}),
    (["--shared-encoder"], {"zit", "qi"}),
    (["--edit", "--shared-encoder"], {"qi"}),
])
def test_list_filters_the_image_models(images, capsys, flags, ids):
    _lib, cfg = images
    assert manage.cmd_list(["--config", str(cfg), "--json"] + flags) == 0
    rows = json.loads(capsys.readouterr().out)["models"]
    assert {r["id"] for r in rows} == ids


def test_list_sizes_an_image_model_by_the_files_it_loads(images, capsys):
    _lib, cfg = images
    manage.cmd_list(["--config", str(cfg), "--json"])
    rows = {r["id"]: r for r in json.loads(capsys.readouterr().out)["models"]}
    assert rows["zit"]["size_bytes"] == 200          # the encoder is shared
    assert rows["qt"]["size_bytes"] == 400           # dit, vae, tower, own


def test_rm_of_an_image_model_keeps_what_others_use(images, capsys):
    lib, cfg = images
    assert manage.cmd_rm(["qt", "--config", str(cfg), "--yes"]) == 0
    # The DiT and its own encoder go; the VAE and tower stay for qi and vl.
    assert not (lib / "qt.gguf").exists() and not (lib / "own.gguf").exists()
    assert (lib / "qvae.gguf").exists() and (lib / "tower.gguf").exists()
    assert "vae kept" in capsys.readouterr().out
    assert manage.cmd_rm(["zit", "--config", str(cfg), "--yes"]) == 0
    assert (lib / "enc.gguf").exists() and not (lib / "zvae.gguf").exists()
    c = _cfg(cfg)
    assert set(c.image_models) == {"qi"} and c.defaults.image_model is None
    assert "qwen3-4b" in c.models


def test_rm_never_removes_a_chat_model_an_image_model_shares(images, capsys):
    lib, cfg = images
    assert manage.cmd_rm(["qwen3-4b", "--config", str(cfg), "--yes"]) == 2
    assert "text encoder of image model zit" in capsys.readouterr().err
    assert (lib / "enc.gguf").exists() and "qwen3-4b" in _cfg(cfg).models


def test_rm_of_a_chat_model_keeps_a_gguf_an_image_model_names(images):
    lib, cfg = images
    cfg.write_text(cfg.read_text().replace(
        "models:\n  qwen3-4b:", "models:\n  own-chat:\n    path: own.gguf\n"
        "  qwen3-4b:", 1))
    assert manage.cmd_rm(["own-chat", "--config", str(cfg), "--yes"]) == 0
    assert (lib / "own.gguf").exists()
    assert "own-chat" not in _cfg(cfg).models


def test_ps_image_shows_only_image_models(monkeypatch, capsys):
    import io

    body = {"server": {"resident_models": [
        {"ids": ["chat"], "kind": "chat", "model_path": "/m/c.gguf"},
        {"ids": ["zit"], "kind": "image", "model_path": "/m/z.gguf"}]}}

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(manage.urllib.request, "urlopen",
                        lambda req, timeout=None: _Resp(
                            json.dumps(body).encode()))
    assert manage.cmd_ps(["--url", "http://127.0.0.1:9", "--image",
                          "--json"]) == 0
    assert [e["ids"] for e in json.loads(capsys.readouterr().out)] == [["zit"]]


def test_doctor_checks_every_file_and_the_identity_of_each_part(
        images, monkeypatch):
    import gmlx.commands.doctor as doctor
    from gmlx.config import load_config

    lib, cfg = images
    monkeypatch.setattr(catalog, "vae_identity", lambda p: (
        {"zvae.gguf": "z_image"}.get(os.path.basename(p), "qwen_image21"),
        None))
    monkeypatch.setattr(catalog, "encoder_matches", lambda p, f, kv=None:
                        os.path.basename(p) != "own.gguf")
    monkeypatch.setattr(catalog, "mmproj_matches", lambda p, f: True)
    c = load_config(cfg)
    assert doctor.check_models(c)["status"] == "PASS"
    row = doctor.check_image(c)
    assert row["status"] == "FAIL"
    assert "qt: the text encoder is not the one qwen_image21" in row["detail"]
    os.remove(lib / "qvae.gguf")
    m = doctor.check_models(c)
    assert m["status"] == "FAIL" and "qi: missing vae" in m["detail"]


def test_doctor_memory_counts_an_image_model_with_its_parts(images,
                                                            monkeypatch):
    import gmlx.commands.doctor as doctor
    from gmlx.config import load_config

    _lib, cfg = images
    monkeypatch.setattr("gmlx.load.memfit.total_ram_bytes", lambda: 1 << 30)
    monkeypatch.setattr("gmlx.load.memfit.classify_fit",
                        lambda size, ram: "over" if size > 350 else "fits")
    row = doctor.check_memory(load_config(cfg))
    # zit is its DiT, VAE and shared encoder (300 B); qt adds its own
    # encoder and tower to its DiT and VAE (400 B).
    assert row["status"] == "WARN" and "zit" not in row["detail"]
    assert "qt (0 KB with its encoder and VAE)" in row["detail"]
