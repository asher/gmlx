"""``gmlx image pack-vae`` on a tiny synthetic VAE: the round trip through
the umbrella, the JSON record, the recorded config, the refusals and their
exit codes. CPU."""
from __future__ import annotations

import hashlib
import json

import pytest
from test_image_vae import CONFIG_JSON, tiny_tensors, write_safetensors

from gmlx.commands.cli import umbrella_main

gguf = pytest.importorskip("gguf")


@pytest.fixture
def src(tmp_path):
    d = tmp_path / "vae"
    d.mkdir()
    (d / "config.json").write_text(CONFIG_JSON)
    return write_safetensors(d / "diffusion_pytorch_model.safetensors",
                             tiny_tensors())


def _field(reader, key):
    f = reader.fields[key]
    return bytes(f.parts[f.data[0]]).decode()


def test_pack_vae_round_trips_and_records_its_sources(src, tmp_path, capsys):
    out = tmp_path / "ae.gguf"
    rc = umbrella_main(["image", "pack-vae", src, "--out", str(out),
                        "--source-repo", "org/repo", "--source-revision",
                        "abc", "--json"])
    assert rc == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["path"] == str(out) and rec["family"] == "z_image"
    assert rec["lossless"] and rec["verified"] and rec["dtype"] == "source"
    r = gguf.GGUFReader(str(out))
    assert len(r.tensors) == rec["tensors"] == len(tiny_tensors())
    assert _field(r, "gmlx.vae.source_repo") == "org/repo"
    assert _field(r, "gmlx.vae.source_revision") == "abc"
    assert json.loads(_field(r, "gmlx.vae.config_json")) == \
        json.loads(CONFIG_JSON)
    with open(src, "rb") as f:
        assert rec["source_sha256"] == hashlib.sha256(f.read()).hexdigest()


def test_an_existing_out_is_refused_unless_forced(src, tmp_path, capsys):
    out = str(tmp_path / "ae.gguf")
    assert umbrella_main(["image", "pack-vae", src, "--out", out]) == 0
    assert "verified bit for bit" in capsys.readouterr().out
    assert umbrella_main(["image", "pack-vae", src, "--out", out]) == 1
    assert "exists" in capsys.readouterr().err
    assert umbrella_main(["image", "pack-vae", src, "--out", out,
                          "--force"]) == 0


def test_a_narrower_dtype_says_it_is_not_lossless(tmp_path, capsys):
    src = write_safetensors(tmp_path / "ae.safetensors", tiny_tensors(),
                            dtype="F32")
    out = tmp_path / "ae.gguf"
    assert umbrella_main(["image", "pack-vae", src, "--out", str(out),
                          "--dtype", "bf16"]) == 0
    assert "not lossless" in capsys.readouterr().err


def test_usage_errors_and_refusals_have_their_exit_codes(src, tmp_path,
                                                          capsys):
    assert umbrella_main(["image", "pack-vae", src]) == 2
    assert umbrella_main(["image", "nope"]) == 2
    assert "unknown action 'nope'" in capsys.readouterr().err
    gg = str(tmp_path / "a.gguf")
    assert umbrella_main(["image", "pack-vae", src, "--out", gg]) == 0
    assert umbrella_main(["image", "pack-vae", gg, "--out",
                          str(tmp_path / "b.gguf")]) == 1
    assert "already a GGUF" in capsys.readouterr().err
    assert umbrella_main(["image", "pack-vae", src, "--out",
                          str(tmp_path / "c.gguf"), "--family", "nope"]) == 1
    assert umbrella_main(["image", "pack-vae", src, "--out",
                          str(tmp_path / "d.gguf"), "--source-config",
                          str(tmp_path / "missing.json")]) == 1


def test_the_verb_and_its_actions_complete():
    from gmlx.commands.completion import _complete

    verbs = [c.split("\t")[0] for c in _complete([""])]
    assert "image" in verbs
    from gmlx.commands.image import HANDLERS

    assert [c.split("\t")[0] for c in _complete(["image", ""])] == \
        sorted(HANDLERS)
    flags = [c.split("\t")[0] for c in _complete(["image", "pack-vae", "-"])]
    assert {"--out", "--dtype", "--force", "--json"} <= set(flags)
    assert _complete(["image", "pack-vae", ""]) == ["::files"]


def test_help_names_the_actions(capsys):
    assert umbrella_main(["image", "--help"]) == 0
    assert "pack-vae" in capsys.readouterr().out
    assert umbrella_main(["help", "image"]) == 0
    assert umbrella_main(["image", "pack-vae", "--help"]) == 0
    assert "--source-config" in capsys.readouterr().out
