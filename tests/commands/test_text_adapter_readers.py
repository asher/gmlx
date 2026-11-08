"""Every reader of a text entry's adapters sees each file of an
``adapters:`` list: the container mounts, the files ``gmlx rm`` keeps, the
doctor's missing-file check, the encoders an image pull may share, the
launch note, and the upsampler's caption identity. CPU-only, no model
load."""
from __future__ import annotations

import pytest

from gmlx.config import build_config


def _cfg(tmp_path, adapters=None, **extra):
    for name in ("base.gguf", "a.gguf", "b.gguf"):
        (tmp_path / name).write_bytes(b"GGUF")
    entry = {"path": "base.gguf"}
    if adapters is not None:
        entry["adapters"] = adapters
    doc = {"server": {"model_dirs": [str(tmp_path)]},
           "models": {"m": entry, **extra}}
    return build_config(doc)


_TWO = [{"path": "a.gguf", "strength": 0}, "b.gguf"]


def test_container_mounts_every_adapter(tmp_path):
    from gmlx.container import settings

    paths = {p for p, _what, _when in settings._model_paths(_cfg(tmp_path, _TWO),
                                                           str(tmp_path))}
    assert {str(tmp_path / "a.gguf"), str(tmp_path / "b.gguf")} <= paths


def test_rm_keeps_an_adapter_another_entry_lists(tmp_path):
    from gmlx.commands import manage

    cfg = _cfg(tmp_path, ["a.gguf"], other={"path": "base.gguf", "adapters": _TWO})
    notes: list = []
    files, _size = manage._rm_deletion_plan(cfg, "m", cfg.models["m"], [],
                                            keep_files=False, notes=notes)
    assert str(tmp_path / "a.gguf") not in files
    assert "adapter kept: another model references it" in notes


def test_rm_deletes_each_adapter_nothing_else_lists(tmp_path):
    from gmlx.commands import manage

    cfg = _cfg(tmp_path, _TWO)
    notes: list = []
    files, _size = manage._rm_deletion_plan(cfg, "m", cfg.models["m"], [],
                                            keep_files=False, notes=notes)
    assert {str(tmp_path / "a.gguf"), str(tmp_path / "b.gguf")} <= set(files)


def test_doctor_checks_every_adapter_file(tmp_path):
    from gmlx.commands import doctor

    cfg = _cfg(tmp_path, _TWO)
    (tmp_path / "b.gguf").unlink()
    got = doctor.check_models(cfg)
    assert got["status"] == "FAIL" and "b.gguf" in got["detail"]


def test_an_image_pull_never_shares_an_adapted_entry(tmp_path, monkeypatch):
    from gmlx.commands import image_pull

    monkeypatch.setattr(image_pull.catalog, "encoder_matches", lambda p, f: True)
    cfg = _cfg(tmp_path, [{"path": "a.gguf", "strength": 0}],
               bare={"path": "base.gguf"})
    assert set(image_pull._chat_encoders(cfg, "z_image").values()) == {"bare"}


def test_the_launch_note_resolves_an_adapted_entry(tmp_path):
    from gmlx.commands import launch

    doc = {"models": {"m": {"path": "/abs/base.gguf", "adapters": _TWO}},
           "server": {"cache": {"enabled": False}}}
    assert launch._prompt_cache_off((None, doc), "m") is True


def test_the_caption_identity_follows_default_strengths(tmp_path, monkeypatch):
    pytest.importorskip("gmlx._vendor.mlx_vlm")
    import gmlx.serve.bridge_vlm as serving
    from gmlx.serve import image_upsample

    ids = {}
    for strength in (0.0, 0.5):
        cfg = _cfg(tmp_path, [{"path": "a.gguf", "strength": strength}])
        serving.clear_resolved_models()
        try:
            serving.register_resolved_models(cfg)
            ids[strength] = image_upsample._upsampler_identity("m")
        finally:
            serving.clear_resolved_models()
    assert ids[0.0] != ids[0.5]
    assert ids[0.0][0] == ids[0.5][0]              # one load signature
