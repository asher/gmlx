#!/usr/bin/env python3
"""The menu bar config editor's testable core: `validate_config_text` (the
server's own parser on a draft string) and `ConfigDraft` (load / conflict-checked
atomic save). No AppKit and no rumps - the `ConfigPanel` shell is GUI-only and
not unit-exercised, same split as the transcript panel."""
from __future__ import annotations

import os

from gmlx.menubar_config import ConfigDraft, validate_config_text


# validate_config_text
def test_validate_ok_minimal():
    ok, msg = validate_config_text("server:\n  port: 8080\n")
    assert ok is True
    assert "0 model(s)" in msg and "warning" not in msg


def test_validate_ok_empty_text():
    ok, _msg = validate_config_text("")
    assert ok is True                         # an empty config is a valid config


def test_validate_malformed_yaml():
    ok, msg = validate_config_text("server: [unclosed\n  port: 8080\n")
    assert ok is False and "malformed YAML" in msg
    assert "\n" not in msg                    # flattened for the status row


def test_validate_root_must_be_mapping():
    ok, msg = validate_config_text("- a\n- b\n")
    assert ok is False and "mapping" in msg


def test_validate_top_level_typo_is_an_error():
    ok, msg = validate_config_text("servre:\n  port: 8080\n")
    assert ok is False and "servre" in msg


def test_validate_bad_value_names_the_key():
    ok, msg = validate_config_text("server:\n  port: not-a-number\n")
    assert ok is False and "port" in msg


def test_validate_missing_model_path_is_a_warning(tmp_path):
    ok, msg = validate_config_text(
        f"server:\n  model_dirs: [{tmp_path}]\n"
        "models:\n  m:\n    path: nope.gguf\n")
    assert ok is True                         # the server would still start
    assert "1 warning" in msg and "'m'" in msg


def test_validate_resolving_model_path_is_clean(tmp_path):
    (tmp_path / "real.gguf").write_bytes(b"GGUF")
    ok, msg = validate_config_text(
        f"server:\n  model_dirs: [{tmp_path}]\n"
        "models:\n  m:\n    path: real.gguf\n")
    assert ok is True
    assert "1 model(s)" in msg and "warning" not in msg


def test_validate_soft_key_warning_is_reported(tmp_path):
    (tmp_path / "real.gguf").write_bytes(b"GGUF")
    ok, msg = validate_config_text(
        f"server:\n  model_dirs: [{tmp_path}]\n"
        "models:\n  m:\n    path: real.gguf\n"
        "profiles:\n  p:\n    sampling:\n      weird_knob: 1\n")
    assert ok is True and "warning" in msg and "weird_knob" in msg


# ConfigDraft
def test_draft_load_save_roundtrip(tmp_path):
    p = tmp_path / "gmlx.yaml"
    p.write_text("server:\n  port: 8080\n")
    d = ConfigDraft(str(p))
    assert d.load() == "server:\n  port: 8080\n"
    saved, msg = d.save("server:\n  port: 9090\n")
    assert saved is True and msg == "Saved."
    assert p.read_text() == "server:\n  port: 9090\n"
    assert d.changed_on_disk() is False       # our own save is not a conflict


def test_draft_save_preserves_mode(tmp_path):
    p = tmp_path / "gmlx.yaml"
    p.write_text("a: 1\n")
    os.chmod(p, 0o640)
    d = ConfigDraft(str(p))
    d.load()
    d.save("a: 2\n")
    assert (os.stat(p).st_mode & 0o777) == 0o640


def test_draft_conflict_refuses_then_force_overwrites(tmp_path):
    p = tmp_path / "gmlx.yaml"
    p.write_text("a: 1\n")
    d = ConfigDraft(str(p))
    d.load()
    p.write_text("a: outside-edit\n")         # someone else wrote the file
    os.utime(p, ns=(1, 10**15))               # force a distinct mtime
    assert d.changed_on_disk() is True
    saved, msg = d.save("a: 2\n")
    assert saved is False and "changed on disk" in msg
    assert p.read_text() == "a: outside-edit\n"   # nothing clobbered
    saved, _msg = d.save("a: 2\n", force=True)
    assert saved is True and p.read_text() == "a: 2\n"


def test_draft_save_creates_missing_file_0600(tmp_path):
    p = tmp_path / "new.yaml"
    d = ConfigDraft(str(p))
    assert d.changed_on_disk() is False       # never loaded: no baseline
    saved, _msg = d.save("server: {}\n")
    assert saved is True and p.read_text() == "server: {}\n"
    assert (os.stat(p).st_mode & 0o777) == 0o600  # configs may hold an api_key


def test_draft_deleted_underneath_is_not_a_conflict(tmp_path):
    p = tmp_path / "gmlx.yaml"
    p.write_text("a: 1\n")
    d = ConfigDraft(str(p))
    d.load()
    p.unlink()
    assert d.changed_on_disk() is False       # save() just recreates it
    saved, _msg = d.save("a: 2\n")
    assert saved is True and p.read_text() == "a: 2\n"
