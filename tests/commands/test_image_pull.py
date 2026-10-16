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


def test_a_file_in_the_way_of_a_download_is_refused_by_name(lib, monkeypatch,
                                                          capsys):
    root, cfg = lib
    dq, _eq = catalog.DEFAULT_QUANTS["z-image-turbo"][32]
    d = root / "unsloth__Z-Image-Turbo-GGUF"
    d.mkdir()
    junk = d / Z.dit_name(dq)
    junk.write_bytes(b"not a dit")
    disk = Disk(monkeypatch, root)
    scan = disk.scan
    monkeypatch.setattr(catalog, "scan", lambda roots, **k: [
        f for f in scan(roots) if f.path != str(junk)])
    got = _downloads(monkeypatch)
    _packs(monkeypatch)
    assert manage.cmd_pull(["z-image-turbo", "--config", str(cfg)]) == 1
    assert got == [] and junk.read_bytes() == b"not a dit"
    assert (f"error: {junk} is not the Z-Image-Turbo DiT, and a download "
            "would land on it" in capsys.readouterr().err)
    assert _cfg(cfg).image_models == {}


def test_a_partial_download_resumes(lib, monkeypatch):
    root, cfg = lib
    dq, _eq = catalog.DEFAULT_QUANTS["z-image-turbo"][32]
    d = root / "unsloth__Z-Image-Turbo-GGUF"
    d.mkdir()
    (d / (Z.dit_name(dq) + ".part")).write_bytes(b"x" * 4)
    Disk(monkeypatch, root)
    got = _downloads(monkeypatch)
    _packs(monkeypatch)
    assert manage.cmd_pull(["z-image-turbo", "--config", str(cfg)]) == 0
    assert (Z.dit_repo, Z.dit_name(dq), "main") in got


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


def test_a_packed_vae_of_another_source_is_not_reused(lib, monkeypatch):
    root, cfg = lib
    other = root / "elsewhere"
    other.mkdir()
    (other / "packed-vae.gguf").write_bytes(b"v")
    Disk(monkeypatch, root, sha="0" * 64)
    got = _downloads(monkeypatch)
    _packs(monkeypatch)
    assert manage.cmd_pull(["z-image-turbo", "--config", str(cfg)]) == 0
    assert (Z.vae_repo, Z.vae_file, Z.vae_revision) in got
    assert _cfg(cfg).image_models["z-image-turbo"].vae != \
        "elsewhere/packed-vae.gguf"


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


def test_a_default_pull_takes_the_class_encoder_quant_over_another_on_disk(
        lib, monkeypatch, capsys):
    root, cfg = lib
    (root / "Qwen3-4B-Q4_K_M.gguf").write_bytes(b"e")
    (root / "Qwen3-4B-Q8_0.gguf").write_bytes(b"e")
    Disk(monkeypatch, root)
    _downloads(monkeypatch)
    _packs(monkeypatch)
    assert manage.cmd_pull(["z-image-turbo", "--config", str(cfg)]) == 0
    assert _cfg(cfg).image_models["z-image-turbo"].encoder == \
        "Qwen3-4B-Q8_0.gguf"


def test_an_encoder_of_another_quant_is_used_and_named_in_the_plan(
        lib, monkeypatch, capsys):
    root, cfg = lib
    (root / "Qwen3-4B-Q4_K_M.gguf").write_bytes(b"e")
    Disk(monkeypatch, root)
    got = _downloads(monkeypatch)
    _packs(monkeypatch)
    assert manage.cmd_pull(["z-image-turbo", "--config", str(cfg)]) == 0
    assert not any(f.startswith("Qwen3-4B") for _r, f, _v in got)
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


def test_the_plan_reaches_a_pipe_before_the_first_download(lib, monkeypatch):
    import sys

    root, cfg = lib
    Disk(monkeypatch, root)
    _packs(monkeypatch)
    calls = []

    class Stream:
        def __init__(self, name):
            self.name = name
            self.text = []

        def write(self, text):
            self.text.append(text)
            return len(text)

        def flush(self):
            if any("license:" in t for t in self.text):
                calls.append(f"{self.name} plan")

    out = Stream("stdout")
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", Stream("stderr"))
    got = _downloads(monkeypatch)

    def fetch(*a):
        calls.append("download")
        return got_fetch(*a)

    got_fetch = manage._hf_download
    monkeypatch.setattr(manage, "_hf_download", fetch)
    assert manage.cmd_pull(["z-image-turbo", "--config", str(cfg)]) == 0
    assert got and calls[0] == "stdout plan"


def _bf16(monkeypatch):
    from types import SimpleNamespace

    from gmlx.image import dtypes
    monkeypatch.setattr(dtypes, "image_activation", lambda fam, **k:
                        SimpleNamespace(dit="bfloat16", vae="bfloat16"))


def test_the_fit_line_counts_the_cast_of_a_16_bit_dit(lib, monkeypatch,
                                                      capsys):
    root, cfg = lib
    Disk(monkeypatch, root)
    _bf16(monkeypatch)
    listing = {Q.dit_name("F16"): 10 << 30, Q.encoder_name("Q8_0"): 2 << 30,
               Q.mmproj_file: 1 << 30, Q.vae_file: 160 << 20,
               Q.vae_config: 1000}
    monkeypatch.setattr(remote, "hf_list_dir", lambda repo, path, rev="main",
                        **k: [(n, "file", s) for n, s in listing.items()])
    monkeypatch.setattr(manage, "_disk_free", lambda p: 1 << 30)
    _downloads(monkeypatch)
    assert manage.cmd_pull(["qwen-image", "--quant", "F16", "--config",
                            str(cfg)]) == 1
    out = capsys.readouterr().out
    # 13.2 GB alone fit a 32 GB Mac; with the F16 tensors the load casts
    # to bfloat16 beside them they are tight.
    assert "the three parts: 13.2 GB, 23.2 GB while the DiT loads, which " \
        "are tight on this Mac's 32 GB RAM" in out


def test_the_fit_line_says_when_the_parts_fit_in_stages(lib, monkeypatch,
                                                        capsys):
    root, cfg = lib
    Disk(monkeypatch, root)
    monkeypatch.setattr("gmlx.load.memfit.total_ram_bytes", lambda: 16 << 30)
    listing = {Q.dit_name("Q4_K_M"): 3900 << 20,
               Q.encoder_name("Q8_0"): 8100 << 20,
               Q.mmproj_file: 1100 << 20, Q.vae_file: 1300 << 20,
               Q.vae_config: 1000}
    monkeypatch.setattr(remote, "hf_list_dir", lambda repo, path, rev="main",
                        **k: [(n, "file", s) for n, s in listing.items()])
    monkeypatch.setattr(manage, "_disk_free", lambda p: 1 << 30)
    _downloads(monkeypatch)
    assert manage.cmd_pull(["qwen-image", "--config", str(cfg)]) == 1
    out = capsys.readouterr().out
    assert "which are more than this Mac's 16 GB RAM" in out
    assert ("in stages, the encoder alone and then the DiT and the VAE, they "
            "need 9.0 GB at most, which fit this Mac's 16 GB RAM") in out


def test_the_disk_check_counts_the_packed_vae_beside_its_source(
        lib, monkeypatch, capsys):
    root, cfg = lib
    Disk(monkeypatch, root)
    dq, eq = catalog.DEFAULT_QUANTS["z-image-turbo"][32]
    listing = {Z.dit_name(dq): 5 << 30, Z.encoder_name(eq): 2 << 30,
               Z.vae_file: 160 << 20, Z.vae_config: 1000}
    monkeypatch.setattr(remote, "hf_list_dir", lambda repo, path, rev="main",
                        **k: [(n, "file", s) for n, s in listing.items()])
    # Room for the downloads, but not for the VAE packed beside its source.
    monkeypatch.setattr(manage, "_disk_free",
                        lambda p: sum(listing.values()) + Z.vae_bytes - 1)
    got = _downloads(monkeypatch)
    assert manage.cmd_pull(["z-image-turbo", "--config", str(cfg)]) == 1
    assert got == [] and "not enough disk space" in capsys.readouterr().err


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


def test_the_discovery_hint_completes_the_set_of_the_dit_on_disk(
        lib, monkeypatch, capsys):
    root, cfg = lib
    (root / "z-image-turbo-Q4_K_M.gguf").write_bytes(b"d")
    Disk(monkeypatch, root)
    _downloads(monkeypatch)
    _packs(monkeypatch)
    assert manage.cmd_pull(["z-image-turbo", "--config", str(cfg), "--quant",
                            "Q4_K_M", "--only", "encoder,vae"]) == 0
    m = _cfg(cfg).image_models["z-image-turbo"]
    assert m.path == "z-image-turbo-Q4_K_M.gguf"


def test_a_renamed_dit_of_the_published_size_is_used(lib, monkeypatch):
    root, cfg = lib
    dq, _eq = catalog.DEFAULT_QUANTS["z-image-turbo"][32]
    (root / "z-image-turbo.gguf").write_bytes(b"d" * 7)
    monkeypatch.setattr(remote, "hf_list_dir", lambda repo, path, rev="main",
                        **k: [(Z.dit_name(dq), "file", 7)])
    Disk(monkeypatch, root)
    got = _downloads(monkeypatch)
    _packs(monkeypatch)
    assert manage.cmd_pull(["z-image-turbo", "--config", str(cfg)]) == 0
    assert (Z.dit_repo, Z.dit_name(dq), "main") not in got
    assert _cfg(cfg).image_models["z-image-turbo"].path == "z-image-turbo.gguf"


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


def test_krea_2_packs_a_vae_without_config(lib, monkeypatch, capsys):
    root, cfg = lib
    K = catalog.RECIPES["krea-2-turbo"]
    monkeypatch.setattr(catalog, "scan", lambda *a, **k: [])
    got = _downloads(monkeypatch)
    monkeypatch.setattr(catalog, "sha256_file", lambda p: K.vae_sha256)
    monkeypatch.setattr(catalog, "vae_identity",
                        lambda p: ("krea2", K.vae_sha256))
    import gmlx.image.vae as vae_pkg
    packed = []

    def pack(src, dst, **kw):
        with open(dst, "wb") as f:
            f.write(b"g")
        packed.append((src, kw))

    monkeypatch.setattr(vae_pkg, "pack_vae", pack)
    assert manage.cmd_pull(["krea-2-turbo", "--config", str(cfg)]) == 0
    dq, eq = catalog.DEFAULT_QUANTS["krea-2-turbo"][32]
    assert (K.dit_repo, f"Krea-2-Turbo-{dq}.gguf", "main") in got
    # The VAE repo holds no config.json, so only the safetensors comes.
    assert [g for g in got if g[0] == K.vae_repo] == [
        (K.vae_repo, K.vae_file, K.vae_revision)]
    ((src, kw),) = packed
    assert kw["config_json"] == "{}" and kw["family"] == "krea2"
    assert not os.path.exists(src)
    m = _cfg(cfg).image_models["krea-2-turbo"]
    assert m.path == f"Abiray__Krea-2-Turbo-GGUF/Krea-2-Turbo-{dq}.gguf"
    assert (root / m.path).is_file()
    assert m.encoder.endswith(K.encoder_name(eq)) and m.encoder_mmproj is None
    assert m.vae == f"Comfy-Org__Krea-2/{K.vae_pack}"
    assert ("license: Krea 2 Turbo krea-2-community, free commercial use only "
            "under $1M annual revenue, affiliates counted, and a deployer "
            "must filter content" in capsys.readouterr().out)


def _krea_pull(lib, monkeypatch, ram_gb, *args):
    root, cfg = lib
    K = catalog.RECIPES["krea-2-turbo"]
    monkeypatch.setattr("gmlx.load.memfit.total_ram_bytes",
                        lambda: ram_gb << 30)
    monkeypatch.setattr(catalog, "scan", lambda *a, **k: [])
    _downloads(monkeypatch)
    monkeypatch.setattr(catalog, "sha256_file", lambda p: K.vae_sha256)
    monkeypatch.setattr(catalog, "vae_identity",
                        lambda p: ("krea2", K.vae_sha256))
    import gmlx.image.vae as vae_pkg
    monkeypatch.setattr(vae_pkg, "pack_vae", lambda src, dst, **kw: open(
        dst, "wb").write(b"g"))
    assert manage.cmd_pull(["krea-2-turbo", "--config", str(cfg),
                            *args]) == 0
    return _cfg(cfg).image_models.get("krea-2-turbo")


@pytest.mark.parametrize("ram_gb, args, size", [
    (16, (), "512x512"),
    (8, (), "512x512"),
    (16, ("--encoder-quant", "q4_k_m"), "512x512"),
    (16, ("--quant", "q4_k_m"), None),
    (24, (), None),
    (24, ("--quant", "q3_k_m"), None),
    (32, (), None),
])
def test_a_krea_2_pull_writes_the_size_the_16_gb_default_makes(
        lib, monkeypatch, capsys, ram_gb, args, size):
    m = _krea_pull(lib, monkeypatch, ram_gb, *args)
    assert m.defaults.get("size") == size
    out = capsys.readouterr().out
    assert ("the entry gets defaults.size 512x512, the largest image this "
            "DiT makes on a 16 GB Mac" in out) == (size is not None)


def test_the_json_plan_names_the_size_the_entry_gets(lib, monkeypatch,
                                                     capsys):
    assert _krea_pull(lib, monkeypatch, 16, "--json", "--no-register") is None
    out = capsys.readouterr().out
    plan = json.loads(out[:out.index("\n}\n") + 2])
    assert plan["memory_class"] == 16 and plan["default_size"] == "512x512"


F = catalog.RECIPES["flux-3-klein"]
FB = catalog.RECIPES["flux-3-klein-base"]
K = catalog.RECIPES["krea-2-turbo"]


def _shared_scan(roots, *, recursive=True, families=None):
    """What catalog.scan sees of Krea 2 and FLUX.3 files: the Qwen3-VL-4B
    file is the encoder of both families."""
    out = []
    for r in roots:
        for d, _s, files in os.walk(r):
            for f in sorted(files):
                p = os.path.join(d, f)
                if f.startswith("flux3_image_klein_sd-"):
                    out.append(catalog.ImageFile(p, "dit", "flux3_klein",
                                                 "distilled"))
                elif f.startswith("flux3_image_klein-"):
                    out.append(catalog.ImageFile(p, "dit", "flux3_klein",
                                                 "base"))
                elif f.startswith("Krea-2-Turbo-"):
                    out.append(catalog.ImageFile(p, "dit", "krea2", "turbo"))
                elif f.startswith("Qwen3-VL-4B-Instruct-"):
                    out += [catalog.ImageFile(p, "encoder", fam)
                            for fam in ("krea2", "flux3_klein")]
                elif f == F.vae_pack:
                    out.append(catalog.ImageFile(p, "vae", "flux3_klein",
                                                 sha256=F.vae_sha256))
                elif f == K.vae_pack:
                    out.append(catalog.ImageFile(p, "vae", "krea2",
                                                 sha256=K.vae_sha256))
    return [f for f in out if families is None or f.family in families]


def _shared_pull(lib, monkeypatch, name, ram_gb=32):
    root, cfg = lib
    r = catalog.recipe(name)
    monkeypatch.setattr("gmlx.load.memfit.total_ram_bytes",
                        lambda: ram_gb << 30)
    monkeypatch.setattr(catalog, "scan", _shared_scan)
    got = _downloads(monkeypatch)
    monkeypatch.setattr(catalog, "sha256_file", lambda p: r.vae_sha256)
    monkeypatch.setattr(catalog, "vae_identity",
                        lambda p: (r.family, r.vae_sha256))
    import gmlx.image.vae as vae_pkg
    packed = []

    def pack(src, dst, **kw):
        with open(dst, "wb") as f:
            f.write(b"g")
        packed.append((src, kw))

    monkeypatch.setattr(vae_pkg, "pack_vae", pack)
    assert manage.cmd_pull([name, "--config", str(cfg)]) == 0
    return got, packed, _cfg(cfg).image_models.get(name)


@pytest.mark.parametrize("r, ram_gb, quant, folder", [
    (F, 32, "Q8_0", "DISTILLED/flux3_image_klein_sd-"),
    (F, 16, "Q4_K_M", "DISTILLED/flux3_image_klein_sd-"),
    (FB, 24, "Q8_0", "BASE/flux3_image_klein-"),
    (FB, 16, "Q4_K_M", "BASE/flux3_image_klein-"),
])
def test_a_flux3_pull_takes_its_variant_folder_and_packs_the_comfy_vae(
        lib, monkeypatch, capsys, r, ram_gb, quant, folder):
    root, cfg = lib
    got, packed, m = _shared_pull(lib, monkeypatch, r.name, ram_gb)
    assert catalog.DEFAULT_QUANTS[r.name][catalog.memory_class(
        ram_gb << 30)][0] == quant
    assert (r.dit_repo, f"{folder}{quant}.gguf", "main") in got
    assert (r.encoder_repo, "Qwen3-VL-4B-Instruct-Q8_0.gguf", "main") in got
    # The FLUX.2 VAE folder holds no config.json, so only the safetensors
    # comes, and its folders go with it after the pack.
    assert [g for g in got if g[0] == r.vae_repo] == [
        (r.vae_repo, r.vae_file, r.vae_revision)]
    ((src, kw),) = packed
    assert kw["config_json"] == "{}" and kw["family"] == "flux3_klein"
    assert kw["source_revision"] == r.vae_revision
    assert not (root / "Comfy-Org__flux2-dev" / "split_files").exists()
    assert m.path == f"realrebelai__FLUX-3-Image-Klein_GGUFs/{folder}{quant}.gguf"
    assert (root / m.path).is_file()
    assert m.vae == "Comfy-Org__flux2-dev/flux2-vae-F32.gguf"
    assert m.encoder_mmproj is None and not m.defaults.get("size")
    assert (f"license: {r.title} flux-kommunity, non-commercial use, and "
            "commercial use of the images only under $5M annual revenue, "
            "affiliates counted, with content filters or a review of each "
            "image for unlawful or infringing content before it is shared, "
            "and a disclosure that it is made with AI where the law requires "
            "one (https://huggingface.co/black-forest-labs/flux-3-image-klein/"
            "blob/main/LICENSE.md); encoder apache-2.0"
            in capsys.readouterr().out)


def test_a_flux3_pull_after_krea_2_shares_its_encoder_file(lib, monkeypatch):
    got, _, krea = _shared_pull(lib, monkeypatch, "krea-2-turbo")
    got, _, flux = _shared_pull(lib, monkeypatch, "flux-3-klein")
    # Only the DiT and the VAE download: the encoder Krea 2's pull fetched
    # is FLUX.3's too, by its weights.
    assert [g[0] for g in got] == [F.dit_repo, F.vae_repo]
    assert flux.encoder == krea.encoder
    assert flux.encoder.endswith("Qwen3-VL-4B-Instruct-Q8_0.gguf")
    got, _, base = _shared_pull(lib, monkeypatch, "flux-3-klein-base")
    assert [g[0] for g in got] == [FB.dit_repo]
    assert base.encoder == krea.encoder and base.vae == flux.vae


def test_a_flux3_pull_takes_no_encoder_quant_but_q8_0(lib, capsys):
    root, cfg = lib
    assert manage.cmd_pull(["flux-3-klein", "--encoder-quant", "Q4_K_M",
                            "--config", str(cfg)]) == 2
    assert ("--encoder-quant Q4_K_M is not a quant of flux-3-klein; choose "
            "one of Q8_0" in capsys.readouterr().err)


def _qwen_pull(lib, monkeypatch, chat_mmproj):
    root, cfg = lib
    (root / "Qwen3-VL-8B-Instruct-Q8_0.gguf").write_bytes(b"e")
    files = [root / "Qwen3-VL-8B-Instruct-Q8_0.gguf"]
    extra = ""
    if chat_mmproj:
        (root / "chat-tower.gguf").write_bytes(b"t")
        extra = "    mmproj: chat-tower.gguf\n"
    cfg.write_text(cfg.read_text() + "models:\n  vl:\n    path: "
                   "Qwen3-VL-8B-Instruct-Q8_0.gguf\n" + extra)
    monkeypatch.setattr(catalog, "scan", lambda roots, **k: [
        catalog.ImageFile(str(f), "encoder", "qwen_image21") for f in files])
    got = _downloads(monkeypatch)
    monkeypatch.setattr(catalog, "sha256_file", lambda p: Q.vae_sha256)
    monkeypatch.setattr(catalog, "vae_identity",
                        lambda p: ("qwen_image21", Q.vae_sha256))
    import gmlx.image.vae as vae_pkg
    monkeypatch.setattr(vae_pkg, "pack_vae", lambda src, dst, **kw: open(
        dst, "wb").write(b"g"))
    assert manage.cmd_pull(["qwen-image", "--config", str(cfg)]) == 0
    return got, _cfg(cfg)


def test_a_shared_chat_encoder_brings_its_own_tower(lib, monkeypatch):
    got, c = _qwen_pull(lib, monkeypatch, chat_mmproj=True)
    assert (Q.encoder_repo, Q.mmproj_file, "main") not in got
    m = c.image_models["qwen-image"]
    assert m.encoder == "vl" and m.encoder_mmproj is None


def test_a_chat_encoder_without_a_tower_is_named_by_file_with_the_tower(
        lib, monkeypatch):
    from gmlx.config import resolve_image_model

    got, c = _qwen_pull(lib, monkeypatch, chat_mmproj=False)
    assert (Q.encoder_repo, Q.mmproj_file, "main") in got
    m = c.image_models["qwen-image"]
    assert m.encoder.endswith("Qwen3-VL-8B-Instruct-Q8_0.gguf")
    assert m.encoder_mmproj.endswith(Q.mmproj_file)
    assert resolve_image_model("qwen-image", c).encoder_mmproj is not None


# validate and pull of a single image GGUF
def _report(arch, gtype=None):
    return remote.HeaderReport(arch, {"Q4_K": 3}, {}, 3, gtype)


def test_an_image_dit_skips_the_chat_arch_gate_and_fits_with_its_parts():
    ref = remote.parse_ref("hf:unsloth/Z-Image-Turbo-GGUF/z-image-turbo-Q4_K_M.gguf")
    v = manage._verdict(ref, _report("lumina2"), size_bytes=5 << 30)
    assert v["image"] == "dit" and v["image_family"] == "z_image"
    assert v["combined_bytes"] == (5 << 30) + Z.encoder_bytes + Z.vae_bytes
    assert v["usable"] and v["arch_error"] is None


@pytest.mark.parametrize("gen, chip", [(13, "M1"), (14, "M2")])
def test_a_dit_this_chip_does_not_run_is_refused_by_name(monkeypatch, capsys,
                                                         gen, chip):
    import gmlx.load.dtypes as load_dtypes

    monkeypatch.setattr(load_dtypes, "gpu_arch_gen", lambda: gen)
    monkeypatch.delenv("GMLX_IMAGE_ALLOW_FP16", raising=False)
    ref = remote.parse_ref("hf:unsloth/Z-Image-Turbo-GGUF/z-image-turbo-Q4_K_M.gguf")
    v = manage._verdict(ref, _report("lumina2"), size_bytes=5 << 30)
    assert not v["usable"]
    assert v["chip_error"].startswith("Z-Image needs an M3 or newer, and "
                                      f"this Mac has an {chip} chip. Set "
                                      "GMLX_IMAGE_ALLOW_FP16=1")
    manage._print_report(v)
    assert f"not loadable: {v['chip_error']}" in capsys.readouterr().out
    monkeypatch.setenv("GMLX_IMAGE_ALLOW_FP16", "1")
    assert manage._verdict(ref, _report("lumina2"), size_bytes=5 << 30)["usable"]


def test_a_dit_pull_this_chip_does_not_run_says_so_alone(tmp_path,
                                                        monkeypatch, capsys):
    import gmlx.load.dtypes as load_dtypes

    monkeypatch.setattr(load_dtypes, "gpu_arch_gen", lambda: 14)
    monkeypatch.delenv("GMLX_IMAGE_ALLOW_FP16", raising=False)
    monkeypatch.setattr(manage, "_resolve_to_file", lambda ref: ref)
    monkeypatch.setattr(manage, "_build_report", lambda ref, max_mb: (
        _report("lumina2"), 1, None))
    monkeypatch.setattr(manage, "_ref_size_bytes", lambda ref, rep: 5 << 30)
    monkeypatch.setattr(manage, "_download_ref",
                        lambda ref, dest: pytest.fail("must not download"))
    rc = manage.cmd_pull(["hf:unsloth/Z-Image-Turbo-GGUF/"
                          "z-image-turbo-Q4_K_M.gguf", "--to", str(tmp_path)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "error: Z-Image needs an M3 or newer, and this Mac has an M2 chip" \
        in err
    assert err.rstrip().endswith(". Pass --force to download it anyway")
    assert "unloadable" not in err and "quant" not in err


def test_a_16_bit_dit_fits_with_the_cast_its_load_holds(monkeypatch):
    from types import SimpleNamespace

    _bf16(monkeypatch)
    monkeypatch.setattr(manage, "_image_role",
                        lambda report, scans: ("dit", "qwen_image21"))
    t = SimpleNamespace(name="w", shape=(4, 4), type_name="F16", offset=0,
                        nbytes=3 << 30)
    scans = [SimpleNamespace(path="/d.gguf", data_offset=0, tensors=[t],
                             kv={})]
    ref = remote.parse_ref("hf:unsloth/Qwen-Image-2.1-GGUF/qwen-image-2.1-F16.gguf")
    v = manage._verdict(ref, _report("qwen_image"), size_bytes=4 << 30,
                        scans=scans)
    assert v["combined_bytes"] == (7 << 30) + Q.encoder_bytes + Q.vae_bytes


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


def test_list_edit_needs_the_vision_tower(images, capsys):
    _lib, cfg = images
    cfg.write_text(cfg.read_text().replace("    encoder_mmproj: tower.gguf\n",
                                           ""))
    assert manage.cmd_list(["--config", str(cfg), "--json", "--edit"]) == 0
    rows = json.loads(capsys.readouterr().out)["models"]
    assert {r["id"] for r in rows} == {"qi"}


def test_list_edit_takes_flux3_klein_on_its_encoder_alone(images, capsys,
                                                          monkeypatch):
    """FLUX.3 klein's encoder reads the prompt alone, so its entry edits
    without a tower, as /v1/models says."""
    lib, cfg = images
    (lib / "f3.gguf").write_bytes(b"x" * 100)
    cfg.write_text(cfg.read_text() + "  f3:\n    path: f3.gguf\n"
                   "    encoder: vl.gguf\n    vae: qvae.gguf\n")
    fams = {"f3.gguf": ("flux3_klein", "distilled"),
            "qi.gguf": ("qwen_image21", "base"),
            "qt.gguf": ("qwen_image21", "turbo")}
    monkeypatch.setattr(catalog, "dit_identity",
                        lambda p: fams.get(os.path.basename(p)))
    assert manage.cmd_list(["--config", str(cfg), "--json", "--edit"]) == 0
    rows = json.loads(capsys.readouterr().out)["models"]
    assert {r["id"] for r in rows} == {"qi", "qt", "f3"}


@pytest.mark.parametrize("spelling, family, variant, edit", [
    ("flux-3-klein", "flux3_klein", "distilled", True),
    ("FLUX.3 klein", "flux3_klein", "distilled", True),
    ("flux3_klein", "flux3_klein", "distilled", True),
    ("flux-3-klein-base", "flux3_klein", "base", True),
    ("qwen-image-2.1", "qwen_image21", "base", True),
    ("qwen-image-2.1-turbo", "qwen_image21", "turbo", True),
    ("z-image-turbo", "z_image", "turbo", False),
    ("Z-Image", "z_image", "turbo", False),
])
@pytest.mark.parametrize("identifies", [True, False],
                         ids=["header", "no-header"])
def test_list_reads_the_family_of_an_entry_as_the_server_does(
        images, capsys, monkeypatch, spelling, family, variant, edit,
        identifies):
    """A configured family is matched on letters and digits, as the
    server matches it, so every spelling the server takes lists the same
    family and edit flag as /v1/models."""
    from gmlx.image.registry import get_family

    lib, cfg = images
    (lib / "new.gguf").write_bytes(b"x" * 100)
    cfg.write_text(cfg.read_text() + f'  new:\n    path: new.gguf\n'
                   f'    family: "{spelling}"\n    encoder: vl\n'
                   "    vae: qvae.gguf\n")
    fams = {"new.gguf": (family, variant)} if identifies else {}
    monkeypatch.setattr(catalog, "dit_identity",
                        lambda p: fams.get(os.path.basename(p)))
    assert manage.cmd_list(["--config", str(cfg), "--json", "--edit"]) == 0
    rows = json.loads(capsys.readouterr().out)["models"]
    assert ("new" in {r["id"] for r in rows}) == edit
    assert manage.cmd_list(["--config", str(cfg), "--json"]) == 0
    row = next(r for r in json.loads(capsys.readouterr().out)["models"]
               if r["id"] == "new")
    assert (row["family"], row["family_title"], row["edit"]) == (
        family, get_family(family).title, edit)


def test_list_keeps_a_family_no_image_family_names(images, capsys):
    lib, cfg = images
    (lib / "new.gguf").write_bytes(b"x" * 100)
    cfg.write_text(cfg.read_text() + '  new:\n    path: new.gguf\n'
                   '    family: "flux-4"\n    encoder: vl\n'
                   "    vae: qvae.gguf\n")
    assert manage.cmd_list(["--config", str(cfg), "--json"]) == 0
    row = next(r for r in json.loads(capsys.readouterr().out)["models"]
               if r["id"] == "new")
    assert (row["family"], row["family_title"], row["edit"]) == (
        "flux-4", None, False)


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


def _discovered_image(monkeypatch, lib, **kw):
    import gmlx.load.discovery as discovery
    from gmlx.config import ImageModelCfg

    def disc(cfg, *, progress=False, images=None):
        if images is not None:
            images.append(ImageModelCfg(id="qt8", path=str(lib / "qt8.gguf"),
                                        **kw))
        return []
    monkeypatch.setattr(discovery, "discovered_models", disc)


def test_rm_keeps_the_encoder_and_vae_a_discovered_image_model_uses(
        images, monkeypatch, capsys):
    lib, cfg = images
    (lib / "qt8.gguf").write_bytes(b"x" * 100)
    # qi no longer uses qvae, so only the discovered DiT beside qt does.
    cfg.write_text(cfg.read_text().replace(
        "  qi:\n    path: qi.gguf\n    encoder: vl\n    vae: qvae.gguf\n",
        "") + f"discover:\n  - dir: {lib}\n")
    _discovered_image(monkeypatch, lib, encoder=str(lib / "own.gguf"),
                      vae=str(lib / "qvae.gguf"))
    assert manage.cmd_rm(["qt", "--config", str(cfg), "--yes"]) == 0
    assert not (lib / "qt.gguf").exists()
    assert (lib / "own.gguf").exists() and (lib / "qvae.gguf").exists()
    out = capsys.readouterr().out
    assert "encoder kept" in out and "vae kept" in out


def test_rm_never_removes_a_chat_model_a_discovered_image_model_shares(
        images, monkeypatch, capsys):
    lib, cfg = images
    (lib / "qt8.gguf").write_bytes(b"x" * 100)
    cfg.write_text(f"server:\n  model_dirs:\n    - {lib}\nmodels:\n  vl:\n"
                   f"    path: vl.gguf\n    mmproj: tower.gguf\n"
                   f"discover:\n  - dir: {lib}\n")
    _discovered_image(monkeypatch, lib, encoder="vl",
                      vae=str(lib / "qvae.gguf"))
    assert manage.cmd_rm(["vl", "--config", str(cfg), "--yes"]) == 2
    assert "text encoder of image model qt8" in capsys.readouterr().err
    assert (lib / "vl.gguf").exists()


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


def _link(lib, how: str) -> str:
    """own.gguf as another entry names it: through a symlink to the file,
    or through a symlink to its folder."""
    if how == "file":
        (lib / "own-link.gguf").symlink_to(lib / "own.gguf")
        return "own-link.gguf"
    (lib.parent / "lib-link").symlink_to(lib)
    return str(lib.parent / "lib-link" / "own.gguf")


@pytest.mark.parametrize("how", ["file", "folder"])
def test_rm_keeps_an_encoder_another_image_model_reaches_by_a_link(
        images, capsys, how):
    lib, cfg = images
    (lib / "qt2.gguf").write_bytes(b"x" * 100)
    cfg.write_text(cfg.read_text() + f"  qt2:\n    path: qt2.gguf\n"
                   f"    encoder: {_link(lib, how)}\n    vae: qvae.gguf\n")
    assert manage.cmd_rm(["qt", "--config", str(cfg), "--yes"]) == 0
    assert not (lib / "qt.gguf").exists() and (lib / "own.gguf").exists()
    assert "encoder kept: another model uses it" in capsys.readouterr().out


@pytest.mark.parametrize("how", ["file", "folder"])
def test_rm_of_a_chat_model_keeps_a_gguf_an_image_model_reaches_by_a_link(
        images, how):
    lib, cfg = images
    cfg.write_text(cfg.read_text().replace(
        "    encoder: own.gguf\n", f"    encoder: {_link(lib, how)}\n").replace(
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


def test_doctor_flags_a_vae_of_another_family(images, monkeypatch):
    import gmlx.commands.doctor as doctor
    from gmlx.config import load_config

    _lib, cfg = images
    monkeypatch.setattr(catalog, "vae_identity",
                        lambda p: ("qwen_image21", None))
    monkeypatch.setattr(catalog, "encoder_matches", lambda p, f, kv=None: True)
    monkeypatch.setattr(catalog, "mmproj_matches", lambda p, f: True)
    row = doctor.check_image(load_config(cfg))
    assert row["status"] == "FAIL"
    assert "zit: the VAE is not the z_image VAE" in row["detail"]
    assert "qi:" not in row["detail"]


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


def test_doctor_memory_counts_the_shared_encoder_and_the_dit_cast(
        images, monkeypatch):
    import gmlx.commands.doctor as doctor
    import gmlx.commands.image_preflight as pf
    from gmlx.config import load_config

    _lib, cfg = images
    monkeypatch.setattr("gmlx.load.memfit.total_ram_bytes", lambda: 1 << 30)
    monkeypatch.setattr("gmlx.load.memfit.classify_fit",
                        lambda size, ram: "over" if size > 250 else "fits")
    # zit is 300 B with its shared encoder, 200 B without it.
    assert "zit" in doctor.check_memory(load_config(cfg))["detail"]
    monkeypatch.setattr("gmlx.load.memfit.classify_fit",
                        lambda size, ram: "over" if size > 350 else "fits")
    monkeypatch.setattr(pf, "dit_cast_bytes", lambda path, family=None: (
        100 if path.endswith("zit.gguf") else 0))
    assert "zit" in doctor.check_memory(load_config(cfg))["detail"]


def test_every_image_model_name_in_the_pull_help_resolves(capsys):
    import re

    with pytest.raises(SystemExit):
        manage.cmd_pull(["--help"])
    # argparse wraps the help at hyphens.
    text = re.sub(r"-\s+", "-", " ".join(capsys.readouterr().out.split()))
    lists = re.findall(r"An image model name \(([^)]*)\)", text)
    assert lists
    for listed in lists:
        names = {n.strip() for n in listed.split(",")}
        assert names == set(catalog.RECIPES)


@pytest.mark.parametrize("name", sorted(catalog.RECIPES))
@pytest.mark.parametrize("gen, chip", [(13, "M1"), (14, "M2")])
def test_a_chip_without_bfloat16_refuses_the_pull_before_any_download(
        lib, monkeypatch, capsys, name, gen, chip):
    import gmlx.load.dtypes as load_dtypes

    root, cfg = lib
    Disk(monkeypatch, root)
    got = _downloads(monkeypatch)
    _packs(monkeypatch)
    monkeypatch.setattr(load_dtypes, "gpu_arch_gen", lambda: gen)
    monkeypatch.delenv("GMLX_IMAGE_ALLOW_FP16", raising=False)
    assert manage.cmd_pull([name, "--config", str(cfg)]) == 1
    assert got == []
    err = capsys.readouterr().err
    title = catalog.RECIPES[name].title
    assert f"{title} needs an M3 or newer, and this Mac has an {chip} chip" \
        in err
    assert err.rstrip().endswith(". Pass --force to download it anyway")
    assert ("GMLX_IMAGE_ALLOW_FP16=1" in err) == (name == "z-image-turbo")
    assert manage.cmd_pull([name, "--config", str(cfg), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["error"].startswith(
        f"{title} needs an M3 or newer")
    manage.cmd_pull([name, "--config", str(cfg), "--force"])
    assert got


@pytest.mark.parametrize("name, goes", [("z-image-turbo", True),
                                        ("qwen-image", False)])
def test_the_float16_opt_in_lets_only_z_image_pull_on_an_m1(
        lib, monkeypatch, capsys, name, goes):
    import gmlx.load.dtypes as load_dtypes

    root, cfg = lib
    Disk(monkeypatch, root)
    got = _downloads(monkeypatch)
    _packs(monkeypatch)
    monkeypatch.setattr(load_dtypes, "gpu_arch_gen", lambda: 13)
    monkeypatch.setenv("GMLX_IMAGE_ALLOW_FP16", "1")
    manage.cmd_pull([name, "--config", str(cfg)])
    err = capsys.readouterr().err
    assert ("needs an M3 or newer" not in err) == goes
    assert bool(got) == goes
