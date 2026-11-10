"""Where the EmbeddingGemma 2 service finds its media projector
(gmlx.serve.embeddings.resolve_media): server.embeddings_mmproj, the
preset's projector for this Mac, or one paired beside the model; the
config key and the --embeddings-mmproj flag; and the init pull line.
Header-only GGUFs, no model loads."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import gmlx.serve.embeddings as emb
from gmlx.config import ConfigError, MissingModelFile, build_config

WIDTH = 512


def _gguf(path, arch, kv):
    from gguf import GGUFWriter

    w = GGUFWriter(str(path), arch)
    for k, v in kv.items():
        if isinstance(v, bool):
            w.add_bool(k, v)
        elif isinstance(v, int):
            w.add_uint32(k, v)
        else:
            w.add_string(k, v)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


def _text(folder, name="embeddinggemma-2-Q8_0.gguf"):
    folder.mkdir(parents=True, exist_ok=True)
    return _gguf(folder / name, "gemma-embedding2",
                 {"gemma-embedding2.embedding_length": WIDTH})


def _mmproj(folder, name, dim=WIDTH, kind="gemma4v"):
    folder.mkdir(parents=True, exist_ok=True)
    return _gguf(folder / name, "clip", {
        "general.type": "mmproj", "clip.has_vision_encoder": True,
        "clip.vision.projector_type": kind, "clip.vision.projection_dim": dim})


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(emb, "_FAMILIES", {})


@pytest.fixture
def chip(monkeypatch):
    def set_chip(bf16: bool):
        monkeypatch.setattr("gmlx.load.dtypes.has_native_bfloat16", lambda: bf16)
    set_chip(True)
    return set_chip


def test_false_turns_media_off(tmp_path):
    model = _text(tmp_path / "m")
    assert emb.resolve_media(model, False) == emb.MediaSetting(
        reason="server.embeddings_mmproj is false")


def test_a_configured_projector_is_checked(tmp_path):
    model = _text(tmp_path / "m")
    good = _mmproj(tmp_path / "p", "mmproj-BF16.gguf")
    assert emb.resolve_media(model, good) == emb.MediaSetting(path=good)
    bad = _mmproj(tmp_path / "p", "mmproj-gemma4.gguf", dim=2560)
    assert emb.resolve_media(model, bad).reason == (
        "mmproj-gemma4.gguf is not a projector for this model: its vision "
        "projection_dim is 2560, and the text model's width is 512")
    other = _mmproj(tmp_path / "p", "mmproj-qwen.gguf", kind="qwen2vl_merger")
    assert "its vision projector is 'qwen2vl_merger'" in emb.resolve_media(
        model, other).reason


@pytest.mark.parametrize("bf16, want", [(True, "mmproj-BF16.gguf"),
                                        (False, "mmproj-F16.gguf")])
def test_a_projector_beside_the_model_pairs_by_chip(tmp_path, chip, bf16, want):
    folder = tmp_path / "unsloth__embeddinggemma-2-GGUF"
    model = _text(folder)
    for name in ("mmproj-BF16.gguf", "mmproj-F16.gguf", "mmproj-Q8_0.gguf"):
        _mmproj(folder, name)
    # A chat model's projector in the same folder projects to its own width.
    _mmproj(folder, "mmproj-gemma-4-F16.gguf", dim=2560)
    chip(bf16)
    assert emb.resolve_media(model).path == str(folder / want)


def test_a_chat_projector_alone_never_pairs(tmp_path):
    folder = tmp_path / "mixed"
    model = _text(folder)
    _mmproj(folder, "mmproj-F16.gguf", dim=2560)
    assert emb.resolve_media(model) == emb.MediaSetting(
        reason="no mmproj found; set server.embeddings_mmproj or "
               "--embeddings-mmproj")


def test_the_preset_projector_is_found_or_named(tmp_path, chip, monkeypatch):
    preset = emb.embedding_preset("embeddinggemma-2-gguf")
    model = _text(tmp_path / "models" / "unsloth__embeddinggemma-2-GGUF")
    mm = str(tmp_path / "models" / "unsloth__embeddinggemma-2-GGUF" / "mmproj-BF16.gguf")

    def resolve(ref, dirs):
        if ref == preset["quants"]["Q8_0"]:
            return model
        if ref.endswith("/mmproj-BF16.gguf") and (tmp_path / "ok").exists():
            return mm
        raise MissingModelFile(ref)

    monkeypatch.setattr("gmlx.config.resolve_path", resolve)
    assert emb.resolve_media(model).reason == (
        "no mmproj found; run gmlx pull hf:unsloth/embeddinggemma-2-GGUF/"
        "mmproj-BF16.gguf")
    (tmp_path / "ok").touch()
    _mmproj(tmp_path / "models" / "unsloth__embeddinggemma-2-GGUF", "mmproj-BF16.gguf")
    assert emb.resolve_media(model).path == mm


def test_a_model_without_media_takes_no_projector(tmp_path):
    qwen = _gguf(tmp_path / "Qwen3-Embedding-0.6B-Q8_0.gguf", "qwen3",
                 {"qwen3.embedding_length": 1024})
    assert emb.resolve_media(qwen) == emb.MediaSetting()
    with pytest.raises(ConfigError, match="takes text only"):
        emb.resolve_media(qwen, str(tmp_path / "mmproj.gguf"))


def test_a_safetensors_checkpoint_carries_its_own_towers(tmp_path):
    folder = tmp_path / "embeddinggemma-2-8bit"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({
        "model_type": "embedding_gemma2", "text_config": {},
        "vision_config": {"hidden_size": 32}}))
    assert emb.resolve_media(str(folder)) == emb.MediaSetting()
    with pytest.raises(ConfigError, match="carries its own towers"):
        emb.resolve_media(str(folder), "mmproj-BF16.gguf")
    (folder / "processor_config.json").write_text(json.dumps(
        {"video_processor": {"fps": 2}}))
    assert "video_processor.fps to 2" in emb.resolve_media(str(folder)).reason


def test_pull_hint_names_this_macs_projector(chip):
    ref = "hf:unsloth/embeddinggemma-2-GGUF/embeddinggemma-2-Q8_0.gguf"
    assert emb.pull_hint("embeddinggemma-2-gguf") == f"gmlx pull {ref} mmproj-BF16.gguf"
    chip(False)
    assert emb.pull_hint(ref) == f"gmlx pull {ref} mmproj-F16.gguf"
    assert emb.pull_hint(ref, projector=False) == f"gmlx pull {ref}"
    assert emb.pull_hint("qwen3-embed-0.6b") is None


def test_a_missing_preset_names_the_pull_with_its_projector(chip, monkeypatch):
    def resolve(ref, dirs):
        raise MissingModelFile(ref)

    monkeypatch.setattr("gmlx.config.resolve_path", resolve)
    with pytest.raises(MissingModelFile, match="gmlx pull hf:unsloth/embeddinggemma-2"
                                               "-GGUF/embeddinggemma-2-Q8_0.gguf "
                                               "mmproj-BF16.gguf"):
        emb.resolve_embeddings_model("embeddinggemma-2-gguf")


def test_the_init_default_is_embeddinggemma_2_and_true_stays_qwen():
    assert emb.INIT_EMBEDDINGS_ALIAS == "embeddinggemma-2-gguf"
    assert emb.DEFAULT_EMBEDDINGS_ALIAS == "qwen3-embed-0.6b"
    # The projector rule: 16-bit, by chip; never Q8_0.
    mm = emb.embedding_preset("embeddinggemma-2-gguf")["mmproj"]
    assert set(mm) == {"bf16", "f16"} and not any("Q8_0" in r for r in mm.values())


@pytest.mark.parametrize("value, want", [
    (False, False), (None, None), ("~/m/mmproj-F16.gguf", "~/m/mmproj-F16.gguf"),
    ("hf:unsloth/embeddinggemma-2-GGUF/mmproj-BF16.gguf",
     "hf:unsloth/embeddinggemma-2-GGUF/mmproj-BF16.gguf")])
def test_the_config_key(value, want):
    server = {"embeddings": "embeddinggemma-2-gguf"}
    if value is not None:
        server["embeddings_mmproj"] = value
    assert build_config({"server": server}).embeddings_mmproj == want


@pytest.mark.parametrize("value", [True, 3, ""])
def test_the_config_key_refuses_other_values(value):
    with pytest.raises(ConfigError, match="embeddings_mmproj must be"):
        build_config({"server": {"embeddings_mmproj": value}})


@pytest.mark.parametrize("flags, config, want", [
    ({}, "c.gguf", "c.gguf"),
    ({"embeddings_mmproj": "f.gguf"}, "c.gguf", "f.gguf"),
    ({"embeddings_mmproj": "false"}, "c.gguf", False),
    ({"embeddings_mmproj": "FALSE"}, None, False),
    # --embeddings names another model, so the config's projector goes.
    ({"embeddings": "embeddinggemma-2-gguf"}, "c.gguf", None),
    ({"embeddings": "x.gguf", "embeddings_mmproj": "f.gguf"}, "c.gguf", "f.gguf"),
])
def test_the_flag_over_the_config(flags, config, want):
    from gmlx.serve.server import _embeddings_mmproj_value

    a = SimpleNamespace(**{"embeddings": None, "embeddings_mmproj": None, **flags})
    cfg = SimpleNamespace(embeddings_mmproj=config)
    assert _embeddings_mmproj_value(a, cfg) == want


def test_pick_projector_prefers_the_chips_16_bit_type(chip):
    from gmlx.load.discovery import pick_projector

    paths = ["/a/mmproj-Q8_0.gguf", "/a/mmproj-F16.gguf", "/a/mmproj-BF16.gguf"]
    assert pick_projector(paths) == "/a/mmproj-BF16.gguf"
    chip(False)
    assert pick_projector(paths) == "/a/mmproj-F16.gguf"
    assert pick_projector(["/a/mmproj-Q8_0.gguf"]) == "/a/mmproj-Q8_0.gguf"
    assert pick_projector([]) is None


def test_scaffold_writes_the_projector_setting():
    from gmlx.load.discovery import scaffold_yaml

    for mmproj, line in ((False, "  embeddings_mmproj: false"),
                         ("/m/mmproj-F16.gguf", "  embeddings_mmproj: /m/mmproj-F16.gguf")):
        text = scaffold_yaml([], model_dirs=[], embeddings="embeddinggemma-2-gguf",
                             embeddings_mmproj=mmproj)
        assert line in text.splitlines()
    assert "embeddings_mmproj" not in scaffold_yaml(
        [], model_dirs=[], embeddings="embeddinggemma-2-gguf")


def test_print_config_round_trips_the_projector_setting():
    import yaml

    from gmlx.serve import server as srv

    for value in (False, "~/m/mmproj-F16.gguf"):
        cfg = build_config({"server": {"embeddings": "embeddinggemma-2-gguf",
                                       "embeddings_mmproj": value}})
        doc = yaml.safe_load(srv._dump_cfg_yaml(cfg))
        assert doc["server"]["embeddings_mmproj"] == value
        assert build_config(doc).embeddings_mmproj == value


def test_doctor_checks_a_configured_projector(tmp_path):
    from gmlx.commands import doctor

    model = _text(tmp_path / "m")
    cfg = SimpleNamespace(embeddings=model, rerank=None, stt=None, tts=None,
                          model_dirs=[], embeddings_mmproj=str(tmp_path / "gone.gguf"))
    row = doctor.check_services(cfg)
    assert row["status"] == "FAIL" and "embeddings_mmproj:" in row["detail"]
    cfg.embeddings_mmproj = _mmproj(tmp_path / "p", "mmproj-BF16.gguf")
    assert doctor.check_services(cfg)["status"] == "PASS"
    cfg.embeddings_mmproj = False
    assert doctor.check_services(cfg)["detail"] == "embeddings configured, files present"
