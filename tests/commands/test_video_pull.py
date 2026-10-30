"""``gmlx video pull``: the plan, files found on disk by their weights, the
download with its sha256 check, the summary, and the video_models entry;
and the files ``gmlx rm`` keeps for a video entry.
The recipe's pins are replaced by small files, the folder scan by a
classifier of names and the downloads by stand-ins. A pull of the real
recipe from the model folder runs when its files are there."""
from __future__ import annotations

import dataclasses
import hashlib
import os

import pytest

import gmlx.commands.manage as manage
from gmlx.commands import video_pull
from gmlx.video import catalog, files

REAL = catalog.RECIPES["h3"]


def _pin(pin: catalog.Pin) -> catalog.Pin:
    """``pin`` for a file whose bytes are its own path in the repo."""
    data = pin.file.encode()
    return dataclasses.replace(pin, sha256=hashlib.sha256(data).hexdigest(),
                               size=len(data))


SMALL = dataclasses.replace(REAL, **{
    role: _pin(p) for role, p in REAL.pins().items()})


@pytest.fixture(autouse=True)
def _mac(monkeypatch):
    """An M3 with room for every recipe. The chip and memory tests set
    their own."""
    from gmlx.load import dtypes
    from gmlx.serve import capacity

    monkeypatch.setattr(dtypes, "_arch_gen", 15)
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: 100 << 30)
    for var in ("GMLX_OVERCOMMIT", "GMLX_GOV_MARGIN", "GMLX_GOV_RESERVE_GB"):
        monkeypatch.delenv(var, raising=False)


def _classify(path: str):
    name = os.path.basename(path)
    for role, key in (("dit", "minimax_h3_fl2va"), ("encoder", "Qwen3-VL-32B"),
                      ("mmproj", "mmproj"), ("vae", "video_vae"),
                      ("audio_vae", "audio_vae"), ("tiny", "taeh3")):
        if key in name:
            if role != "dit":
                return files.VideoFile(path, role, "h3")
            # Markers in a DiT's name stand for what its weights say.
            return files.VideoFile(
                path, role, "h3",
                "fl2va-turbo4" if "turbo4" in name else "fl2va",
                "unverified" if "unverified" in name else "qkv")
    return None


@pytest.fixture
def lib(tmp_path, monkeypatch):
    """A config whose model_dirs is <tmp>/lib, the small recipe, a name
    classifier for the scan, and stand-in downloads that record each
    file."""
    root = tmp_path / "lib"
    root.mkdir()
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"server:\n  model_dirs:\n    - {root}\n")
    monkeypatch.setitem(catalog.RECIPES, "h3", SMALL)
    # As the real scan, hidden folders and partial files are skipped.
    monkeypatch.setattr(files, "scan", lambda roots, fam: [
        f for r in roots for d, _, ns in os.walk(r) for n in sorted(ns)
        if "/." not in d and not n.endswith(".part")
        and (f := _classify(os.path.join(d, n))) is not None])
    got: list = []

    def hf(repo, filename, revision, dest_dir):
        dest = os.path.join(dest_dir, filename)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(filename.encode())
        got.append(filename)
        return dest

    def http(url, dest):
        with open(dest, "wb") as f:
            f.write(SMALL.tiny.file.encode())
        got.append(url)

    monkeypatch.setattr(manage, "_hf_download", hf)
    monkeypatch.setattr(video_pull, "_http_download", http)
    return root, cfg, got


def _pull(cfg, *argv):
    return video_pull.cmd_pull(["h3", "--config", str(cfg), *argv])


def test_a_pull_downloads_every_file_and_adds_the_entry(lib, capsys):
    from gmlx.config import load_config, resolve_video_model

    root, cfg, got = lib
    assert _pull(cfg, "--yes") == 0
    assert len(got) == 7
    out = capsys.readouterr().out
    assert "license: minimax-h3-community-license-agreement" in out
    assert ("video and audio VAEs: minimax-h3-community-license-agreement"
            in out)
    assert f"registered h3 in {cfg}" in out
    rm = resolve_video_model("h3", load_config(cfg))
    assert rm.path == str(root / "unsloth__MiniMax-H3-GGUF" / REAL.dit.file)
    assert rm.adapters == ((str(root / "lightx2v__Minimax-h3-Turbo"
                                / REAL.lora.file), 1.0),)
    assert rm.defaults == {"steps": 8, "shift": 6.0, "size": "1344x768",
                           "seconds": 5.0}
    # Again: every file is found and nothing downloads.
    got.clear()
    assert _pull(cfg, "--yes") == 0
    assert got == []
    assert f"h3 is already configured as h3 in {cfg}" in (
        capsys.readouterr().out)


def test_files_on_disk_are_found_by_their_weights(lib, capsys):
    root, cfg, got = lib
    (root / "mine").mkdir()
    for role, pin in SMALL.pins().items():
        # Other names: the scan finds them by role, the LoRA by its sha256.
        name = "renamed.safetensors" if role == "lora" else (
            f"my-{os.path.basename(pin.file)}")
        (root / "mine" / name).write_bytes(pin.file.encode())
    assert _pull(cfg, "--dry-run") == 0
    out = capsys.readouterr().out
    assert "download" not in out and out.count("found: ") == 7
    assert str(root / "mine" / "renamed.safetensors") in out
    assert got == []


def test_a_wrong_download_is_refused_and_not_kept(lib, monkeypatch, capsys):
    root, cfg, _ = lib

    def corrupt(repo, filename, revision, dest_dir):
        dest = os.path.join(dest_dir, filename)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(b"corrupt")
        return dest

    monkeypatch.setattr(manage, "_hf_download", corrupt)
    assert _pull(cfg, "--yes") == 1
    assert "the download has sha256" in capsys.readouterr().err
    # The folders the pull made go too.
    assert os.listdir(root) == []


def test_a_foreign_file_in_the_way_blocks_the_pull(lib, capsys):
    root, cfg, got = lib
    dest = root / "lightx2v__Minimax-h3-Turbo" / REAL.lora.file
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"something else")
    assert _pull(cfg, "--yes") == 1
    assert "is there and is not the MiniMax-H3 Turbo 8-step LoRA" in (
        capsys.readouterr().err)
    assert got == []
    import json

    assert _pull(cfg, "--json") == 1
    assert json.loads(capsys.readouterr().out)["blocked"] == [str(dest)]


def test_no_terminal_and_no_yes_downloads_nothing(lib, monkeypatch, capsys):
    _, cfg, got = lib
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert _pull(cfg) == 1
    assert "pass --yes" in capsys.readouterr().err and got == []


@pytest.mark.integration
def test_the_model_folder_holds_every_file_of_the_recipe(tmp_path):
    """The real recipe against the folder that holds the lab's files:
    every pin is found there and nothing downloads."""
    d = os.environ.get("KQUANT_TEST_GGUF_DIR")
    if not d or not os.path.exists(os.path.join(d, "unsloth-h3",
                                                REAL.dit.file)):
        pytest.skip("KQUANT_TEST_GGUF_DIR holds no MiniMax-H3 files")
    for name in ("h3", "h3-fast", "h3-base"):
        plan = video_pull.plan_pull(catalog.RECIPES[name], roots=[d],
                                    base_dir=str(tmp_path))
        missing = [p.role for p in plan.parts if not p.found]
        assert missing == [] and plan.blocked == [], name


def _dit_dest(root):
    return root / "unsloth__MiniMax-H3-GGUF" / REAL.dit.file


def test_a_dit_of_other_weights_or_an_unverified_layout_is_not_taken(
        lib, capsys):
    root, cfg, _ = lib
    for name in ("minimax_h3_fl2va-turbo4.gguf",
                 "minimax_h3_fl2va-unverified.gguf"):
        (root / name).write_bytes(b"x")
    assert _pull(cfg, "--dry-run") == 0
    out = capsys.readouterr().out
    assert f"download 0.0 GB: {_dit_dest(root)}" in out


def test_the_pinned_name_is_taken_first(lib, capsys):
    root, cfg, _ = lib
    for d in ("a", "b"):
        (root / d).mkdir()
    (root / "a" / "my-minimax_h3_fl2va.gguf").write_bytes(b"x")
    (root / "b" / REAL.dit.file).write_bytes(b"x")
    assert _pull(cfg, "--dry-run") == 0
    assert f"found: {root / 'b' / REAL.dit.file}" in capsys.readouterr().out


def test_a_lora_of_the_same_size_and_other_bytes_is_not_taken(lib, capsys):
    root, cfg, _ = lib
    other = bytearray(SMALL.lora.file.encode())
    other[0] ^= 1
    (root / SMALL.lora.name).write_bytes(bytes(other))
    assert video_pull.plan_pull(SMALL, roots=[str(root)],
                                base_dir=str(root)).part("lora").found is False


def test_model_dirs_with_a_variable_download_under_its_value(
        lib, monkeypatch, tmp_path, capsys):
    root, cfg, _ = lib
    monkeypatch.setenv("GMLX_TEST_LIB", str(root))
    cfg.write_text("server:\n  model_dirs:\n    - $GMLX_TEST_LIB\n")
    monkeypatch.chdir(tmp_path)
    assert _pull(cfg, "--yes") == 0
    assert _dit_dest(root).exists()
    assert not (tmp_path / "$GMLX_TEST_LIB").exists()
    assert f"registered h3 in {cfg}" in capsys.readouterr().out


def test_an_id_another_block_takes_gets_a_number(lib, capsys):
    from gmlx.config import load_config

    root, cfg, _ = lib
    cfg.write_text(f"server:\n  model_dirs:\n    - {root}\n"
                   "models:\n  h3:\n    path: chat.gguf\n")
    assert _pull(cfg, "--yes") == 0
    assert f"registered h3-2 in {cfg}" in capsys.readouterr().out
    assert set(load_config(cfg).video_models) == {"h3-2"}
    # Again: the entry is there under its number.
    assert _pull(cfg, "--yes") == 0
    assert "h3 is already configured as h3-2" in capsys.readouterr().out
    assert set(load_config(cfg).video_models) == {"h3-2"}


def test_each_recipe_on_one_dit_gets_its_own_entry(lib, monkeypatch, capsys):
    from gmlx.config import load_config

    root, cfg, _ = lib
    assert _pull(cfg, "--yes") == 0
    for name in ("h3-fast", "h3-base"):
        r = catalog.RECIPES[name]
        monkeypatch.setitem(catalog.RECIPES, name, dataclasses.replace(
            r, **{role: _pin(p) for role, p in r.pins().items()}))
        assert video_pull.cmd_pull([name, "--config", str(cfg),
                                    "--yes"]) == 0
        assert f"registered {name} in {cfg}" in capsys.readouterr().out
    vm = load_config(cfg).video_models
    assert vm["h3-base"].path == vm["h3-fast"].path == vm["h3"].path
    assert vm["h3-base"].adapters == ()
    assert vm["h3-fast"].adapters != vm["h3"].adapters


def test_a_model_folder_behind_a_link_is_registered_once(lib, tmp_path,
                                                         capsys):
    from gmlx.config import load_config

    root, cfg, _ = lib
    (tmp_path / "link").symlink_to(root)
    # An entry whose path does not resolve must not stop the lookup.
    cfg.write_text(f"server:\n  model_dirs:\n    - {tmp_path / 'link'}\n"
                   "video_models:\n  other:\n    path: 'hf:a b/c/x.gguf'\n"
                   "    encoder: e.gguf\n    vae: v.safetensors\n"
                   "    audio_vae: a.safetensors\n")
    assert _pull(cfg, "--yes") == 0
    assert _pull(cfg, "--yes") == 0
    assert "h3 is already configured as h3" in capsys.readouterr().out
    assert set(load_config(cfg).video_models) == {"other", "h3"}


def test_a_broken_transfer_resumes_on_the_next_pull(lib, monkeypatch,
                                                    capsys):
    root, cfg, got = lib
    calls = []
    real = manage._hf_download

    def flaky(repo, filename, revision, dest_dir):
        calls.append(filename)
        dest = os.path.join(dest_dir, filename)
        if filename == REAL.dit.file and not os.path.exists(dest + ".part"):
            with open(dest + ".part", "wb") as f:
                f.write(filename.encode()[:10])
            raise OSError("connection reset")
        if os.path.exists(dest + ".part"):
            # The transfer goes on from the bytes already there.
            assert os.path.getsize(dest + ".part") == 10
            os.remove(dest + ".part")
        return real(repo, filename, revision, dest_dir)

    monkeypatch.setattr(manage, "_hf_download", flaky)
    assert _pull(cfg, "--yes") == 1
    assert "connection reset" in capsys.readouterr().err
    plan = video_pull.plan_pull(SMALL, roots=[str(root)], base_dir=str(root))
    assert plan.part("dit").need == SMALL.dit.size - 10
    assert _pull(cfg, "--yes") == 0
    assert _dit_dest(root).read_bytes() == REAL.dit.file.encode()
    assert not any(n.startswith(".gmlx-pull-")
                   for _, ds, _ in os.walk(root) for n in ds)


def test_too_little_free_disk_is_refused_unless_forced(lib, monkeypatch,
                                                       capsys):
    _, cfg, got = lib
    monkeypatch.setattr(manage, "_disk_free", lambda path: 10)
    assert _pull(cfg, "--yes") == 1
    assert "not enough disk space" in capsys.readouterr().err and got == []
    assert _pull(cfg, "--yes", "--force") == 0 and len(got) == 7


def test_an_m2_is_refused_unless_forced(lib, monkeypatch, capsys):
    from gmlx.load import dtypes

    _, cfg, got = lib
    monkeypatch.setattr(dtypes, "_arch_gen", 14)
    assert _pull(cfg, "--yes") == 1
    assert capsys.readouterr().err == (
        "error: MiniMax-H3 needs an M3 or newer, and this Mac has an M2 "
        "chip; pass --force to download anyway\n") and got == []
    assert _pull(cfg, "--yes", "--force") == 0 and len(got) == 7


def test_a_mac_too_small_for_any_clip_is_refused_unless_forced(
        lib, monkeypatch, capsys):
    from gmlx.serve import capacity

    _, cfg, got = lib
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: 1 << 20)
    assert _pull(cfg, "--yes") == 1
    err = capsys.readouterr().err
    assert err.startswith("error: the smallest MiniMax-H3 clip does not fit "
                          "in memory: encode ")
    assert err.endswith(" against a working budget of 0.00 GiB, with the "
                        "encode, the denoise, the audio and the decode over "
                        "it; pass --force to download anyway\n")
    assert got == []
    assert _pull(cfg, "--dry-run") == 1
    monkeypatch.setenv("GMLX_OVERCOMMIT", "1")
    assert _pull(cfg, "--dry-run") == 0
    monkeypatch.delenv("GMLX_OVERCOMMIT")
    assert _pull(cfg, "--yes", "--force") == 0 and len(got) == 7


def test_the_checks_run_chip_then_memory_then_files(lib, monkeypatch,
                                                    capsys):
    """On a small M2 the chip refusal shows. On a small M3 the memory
    refusal shows before a file in the way, under --dry-run too."""
    from gmlx.load import dtypes
    from gmlx.serve import capacity

    root, cfg, got = lib
    dest = root / "lightx2v__Minimax-h3-Turbo" / REAL.lora.file
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"something else")
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: 1 << 20)
    monkeypatch.setattr(dtypes, "_arch_gen", 14)
    assert _pull(cfg, "--dry-run") == 1
    err = capsys.readouterr().err
    assert err.startswith("error: MiniMax-H3 needs an M3 or newer")
    assert "smallest" not in err
    monkeypatch.setattr(dtypes, "_arch_gen", 15)
    assert _pull(cfg, "--dry-run") == 1
    err = capsys.readouterr().err
    assert err.startswith("error: the smallest MiniMax-H3 clip")
    assert "is there" not in err and got == []


def _sparse(path, size: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.truncate(size)
    return path


@pytest.mark.parametrize("found, takes, pinned", [
    (["Q8_0"], None, False), (["Q6_K", "Q8_0"], None, False),
    (["a/Q8_0", "b/Q2_K", "c/Q3_K_M"], "c/Q3_K_M", False),
    (["Q8_0"], None, True)])
def test_found_files_too_large_for_this_mac_give_way(
        lib, monkeypatch, capsys, found, takes, pinned):
    """Larger quants of the encoder in a model folder keep every clip out
    of memory where the recipe's file fits: the pull takes the largest
    other quant it finds that fits, or downloads the recipe's. The DiT it
    takes stays, the recipe's own before a larger one, and so does a large
    vision tower, which the check does not price."""
    from gmlx.config import load_config, resolve_video_model
    from gmlx.serve import capacity

    root, cfg, got = lib
    mine = root / "mine"
    enc = {}
    for i, q in enumerate(found):
        d, _, quant = q.rpartition("/")
        enc[q] = _sparse(mine / d / f"Qwen3-VL-32B-Instruct-{quant}.gguf",
                         {"Q2_K": 1 << 20, "Q3_K_M": 2 << 20}.get(
                             quant, (3 + i) << 30))
    dit = mine / "minimax_h3_fl2va_pruned-Q8_0.gguf"
    dit.write_bytes(SMALL.dit.file.encode() * 4)
    if pinned:
        dit = mine / SMALL.dit.name
        dit.write_bytes(SMALL.dit.file.encode())
    tower = _sparse(mine / "mmproj-F32.gguf", 2 << 30)
    # With no room for the recipe's file either, the pull refuses.
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: 1 << 20)
    assert _pull(cfg, "--dry-run") == 1
    err = capsys.readouterr().err
    assert "note:" not in err and "the smallest MiniMax-H3 clip" in err
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: 3 << 30)
    assert _pull(cfg, "--yes") == 0
    how = (f"uses {enc[takes]}" if takes
           else f"downloads {SMALL.encoder.name}")
    err = capsys.readouterr().err.splitlines()
    assert [n for n in err if n.startswith("note:")] == [
        f"note: {enc[found[0]]} does not fit this Mac's memory, so the "
        f"pull {how}"]
    assert (SMALL.encoder.file in got) is (takes is None)
    assert len(got) == (4 if takes else 5)
    rm = resolve_video_model("h3", load_config(cfg))
    assert rm.path == str(dit)
    assert rm.encoder_mmproj == str(tower)
    assert rm.encoder == str(enc[takes] if takes else (
        root / "unsloth__Qwen3-VL-32B-Instruct-GGUF" / REAL.encoder.file))


def test_a_pull_that_adds_no_entry_keeps_the_files_it_finds(
        lib, monkeypatch, capsys):
    """On a chip the pull refuses, or with an entry on this DiT and LoRA
    already, the pull registers nothing, so it downloads nothing in place
    of a found file too large for this Mac."""
    from gmlx.load import dtypes
    from gmlx.serve import capacity

    root, cfg, got = lib
    _sparse(root / "mine" / "Qwen3-VL-32B-Instruct-Q8_0.gguf", 3 << 30)
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: 3 << 30)
    monkeypatch.setattr(dtypes, "_arch_gen", 14)
    assert _pull(cfg, "--dry-run") == 1
    err = capsys.readouterr().err
    assert err.startswith("error: MiniMax-H3 needs an M3")
    assert "note:" not in err
    monkeypatch.setattr(dtypes, "_arch_gen", 15)
    monkeypatch.setenv("GMLX_OVERCOMMIT", "1")
    assert _pull(cfg, "--yes") == 0 and len(got) == 6
    monkeypatch.delenv("GMLX_OVERCOMMIT")
    capsys.readouterr()
    assert _pull(cfg, "--yes") == 1 and len(got) == 6
    err = capsys.readouterr().err
    assert err.startswith("error: the smallest MiniMax-H3 clip")
    assert "note:" not in err


@pytest.mark.parametrize("name", sorted(catalog.RECIPES))
def test_every_recipe_needs_more_than_24_gb(name, monkeypatch):
    """Priced from the real pins: on a 24 GB Mac the text encoder alone
    is over the budget, and a 32 GB Mac runs the smallest clip."""
    from gmlx.serve import capacity

    r = catalog.RECIPES[name]
    plan = video_pull.Plan(r, [video_pull.Part(role, pin, "", False)
                               for role, pin in r.pins().items()])
    monkeypatch.setattr(capacity, "working_budget_bytes",
                        lambda: capacity.ram_budget_bytes(24 << 30))
    why = video_pull.memory_refusal(plan)
    assert why is not None and why.endswith(", with the encode over it")
    monkeypatch.setattr(capacity, "working_budget_bytes",
                        lambda: capacity.ram_budget_bytes(32 << 30))
    assert video_pull.memory_refusal(plan) is None


def test_json_carries_the_licenses_and_stays_json(lib, capsys):
    import json

    _, cfg, _ = lib
    assert _pull(cfg, "--yes", "--json") == 0
    io = capsys.readouterr()
    doc = json.loads(io.out)
    assert doc["license"] == "minimax-h3-community-license-agreement"
    assert (doc["encoder_license"], doc["vae_license"]) == (
        "apache-2.0", "minimax-h3-community-license-agreement")
    assert doc["download_bytes"] == sum(p.size for p in SMALL.pins().values())
    assert "registered h3" in io.err


def test_a_file_that_lands_during_the_pull_is_not_replaced(lib, monkeypatch,
                                                          capsys):
    root, cfg, _ = lib
    real = manage._hf_download

    def racing(repo, filename, revision, dest_dir):
        got = real(repo, filename, revision, dest_dir)
        if filename == REAL.dit.file:
            _dit_dest(root).write_bytes(b"theirs")
        return got

    monkeypatch.setattr(manage, "_hf_download", racing)
    assert _pull(cfg, "--yes") == 1
    assert "appeared during the pull" in capsys.readouterr().err
    assert _dit_dest(root).read_bytes() == b"theirs"
    assert not any(n.startswith(".gmlx-pull-")
                   for _, ds, _ in os.walk(root) for n in ds)


def test_a_download_outside_model_dirs_is_not_registered(lib, tmp_path,
                                                         capsys):
    _, cfg, _ = lib
    assert _pull(cfg, "--yes", "--dir", str(tmp_path / "elsewhere")) == 0
    assert "is not registered" in capsys.readouterr().err
    from gmlx.config import load_config

    assert load_config(cfg).video_models == {}


def test_the_scan_reads_first_shards_and_skips_hidden_folders(
        tmp_path, monkeypatch, capsys):
    seen = []

    def role(path, family):
        seen.append(os.path.relpath(path, tmp_path))
        if "bad" in path:
            raise ValueError("not a GGUF")
        return None

    monkeypatch.setattr(files, "_gguf_role", role)
    monkeypatch.setattr(files, "_vae_role", role)
    for name in ("a.gguf", "B.GGUF", "s-00001-of-00002.gguf",
                 "s-00002-of-00002.gguf", "v.safetensors", "bad.gguf",
                 "notes.txt", ".hidden/x.gguf"):
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_bytes(b"x")
    os.symlink(tmp_path / "a.gguf", tmp_path / "link.gguf")
    assert files.scan([str(tmp_path)], None) == []
    assert sorted(seen) == ["B.GGUF", "a.gguf", "bad.gguf",
                            "s-00001-of-00002.gguf", "v.safetensors"]
    assert "skipping" in capsys.readouterr().err


def test_an_image_pull_does_not_take_a_video_id():
    import types

    from gmlx.commands import image_pull
    from gmlx.config import build_config
    from gmlx.image import catalog as image_catalog

    r = next(iter(image_catalog.RECIPES.values()))
    cfg = build_config({"video_models": {r.name: {
        "path": "h3.gguf", "encoder": "e.gguf", "vae": "v.safetensors",
        "audio_vae": "a.safetensors"}}})
    got = image_pull._entry_id(r, types.SimpleNamespace(dit_quant="Q4_K_M"),
                               cfg)
    assert got != r.name and got.startswith(r.name)


@pytest.mark.parametrize("form", ["$GMLX_TEST_DL", "~/dl"])
def test_the_download_folder_expands_variables_and_home(
        lib, monkeypatch, tmp_path, form):
    _, cfg, _ = lib
    monkeypatch.setenv("GMLX_TEST_DL", str(tmp_path / "dl"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    assert _pull(cfg, "--yes", "--dir", form) == 0
    assert _dit_dest(tmp_path / "dl").exists()
    assert not (tmp_path / form).exists()


def test_a_pinned_file_is_not_taken_from_a_hidden_folder_or_an_unreadable_one(
        tmp_path, monkeypatch, capsys):
    from gmlx.image import catalog as image_catalog

    pin = SMALL.lora
    for d in (".gmlx-pull-0123456789abcdef", "a", "b"):
        (tmp_path / d).mkdir()
        (tmp_path / d / pin.name).write_bytes(pin.file.encode())
    real = image_catalog.sha256_file

    def sha(path):
        if path == str(tmp_path / "a" / pin.name):
            raise PermissionError(13, "Permission denied")
        return real(path)

    monkeypatch.setattr(image_catalog, "sha256_file", sha)
    assert files.find_pinned([str(tmp_path)], pin) == str(
        tmp_path / "b" / pin.name)
    assert "skipping" in capsys.readouterr().err


def test_gmlx_pull_of_a_video_recipe_points_to_video_pull(capsys):
    assert manage.cmd_pull(["h3"]) == 2
    assert "gmlx video pull h3" in capsys.readouterr().err


def test_rm_of_a_chat_model_keeps_the_files_a_video_entry_uses(tmp_path):
    from gmlx.config import load_config

    root = tmp_path / "lib"
    root.mkdir()
    for n in ("qwen3-vl.gguf", "mmproj.gguf", "h3.gguf", "v.safetensors",
              "a.safetensors"):
        (root / n).write_bytes(b"x")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"server:\n  model_dirs:\n    - {root}\n"
        "models:\n  qvl:\n    path: qwen3-vl.gguf\n    mmproj: mmproj.gguf\n"
        "video_models:\n  h3:\n    path: h3.gguf\n"
        "    encoder: qwen3-vl.gguf\n    encoder_mmproj: mmproj.gguf\n"
        "    vae: v.safetensors\n    audio_vae: a.safetensors\n")
    assert manage.cmd_rm(["qvl", "--config", str(cfg), "--yes",
                          "--no-reload"]) == 0
    assert (root / "qwen3-vl.gguf").exists() and (root / "mmproj.gguf").exists()
    assert "qvl" not in load_config(cfg).models
