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


# The config.json at the root of BFL's FLUX.3 klein repository, beside its
# ae.safetensors, describes the pipeline, not the VAE.
PIPELINE_JSON = json.dumps({"format_version": 1, "model": "m",
                            "vae": {"path": "ae.safetensors"}})
DIT_JSON = json.dumps({"_class_name": "Flux2Transformer2DModel"})


@pytest.mark.parametrize("beside, told", [
    (PIPELINE_JSON, "names no Autoencoder class in _class_name"),
    (DIT_JSON, "is the config of a Flux2Transformer2DModel, not of a VAE"),
    ("{", "is not JSON")])
def test_a_config_beside_src_is_recorded_only_when_it_is_a_vae_config(
        tmp_path, capsys, beside, told):
    """A model folder can hold the VAE beside another config, which is
    left out with a note; --source-config records the VAE's."""
    d = tmp_path / "repo"
    d.mkdir()
    (d / "config.json").write_text(beside)
    src = write_safetensors(d / "ae.safetensors", tiny_tensors())
    out = tmp_path / "ae.gguf"
    assert umbrella_main(["image", "pack-vae", src, "--out", str(out)]) == 0
    err = capsys.readouterr().err
    assert f"{d / 'config.json'} {told}, so the GGUF records no config" in err
    assert _field(gguf.GGUFReader(str(out)), "gmlx.vae.config_json") == "{}"
    vae = tmp_path / "vae.json"
    vae.write_text(CONFIG_JSON)
    assert umbrella_main(["image", "pack-vae", src, "--out", str(out),
                          "--source-config", str(vae), "--force"]) == 0
    assert "records no config" not in capsys.readouterr().err
    assert json.loads(_field(gguf.GGUFReader(str(out)),
                             "gmlx.vae.config_json")) == json.loads(CONFIG_JSON)


def test_a_source_config_of_another_model_is_refused(src, tmp_path, capsys):
    other = tmp_path / "config.json"
    other.write_text(PIPELINE_JSON)
    out = tmp_path / "ae.gguf"
    assert umbrella_main(["image", "pack-vae", src, "--out", str(out),
                          "--source-config", str(other)]) == 1
    assert f"--source-config {other}: names no Autoencoder class" in \
        capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize("family", ["z-image", "Z-Image", "z_image"])
def test_family_takes_the_spellings_the_server_takes(src, tmp_path, capsys,
                                                     family):
    out = tmp_path / "ae.gguf"
    assert umbrella_main(["image", "pack-vae", src, "--out", str(out),
                          "--family", family, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["family"] == "z_image"


def test_family_names_the_family_it_reads_when_the_file_is_another(
        src, tmp_path, capsys):
    """flux-3-klein reads as flux3_klein, whose VAE the FLUX.1 file is
    not, and a name of no family is refused with the known ones."""
    out = str(tmp_path / "ae.gguf")
    assert umbrella_main(["image", "pack-vae", src, "--out", out,
                          "--family", "flux-3-klein"]) == 1
    assert "and flux3_klein decodes with" in capsys.readouterr().err
    assert umbrella_main(["image", "pack-vae", src, "--out", out,
                          "--family", "flux-4"]) == 1
    assert "no VAE is known for image family 'flux-4'" in \
        capsys.readouterr().err


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

    assert sorted(c.split("\t")[0] for c in _complete(["image", ""])) == \
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
