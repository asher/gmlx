#!/usr/bin/env python3
"""``gmlx doctor`` - check assembly, statuses, exit codes, and the
conditional rows. CPU-only: every environment-touching check is a
module-level seam, monkeypatched per test; the end-to-end tests use a tmp
config and a minted GGUF."""
from __future__ import annotations

import json

import numpy as np
import pytest

from gguf import GGMLQuantizationType as GT  # noqa: E402
from gguf import GGUFWriter  # noqa: E402

from gmlx import doctor  # noqa: E402

# Bound before the autouse pin below so the server tests can call the real one.
_real_check_server = doctor.check_server
_real_check_launcher = doctor.check_launcher
_real_running_configs = doctor._running_configs
_real_check_agents = doctor.check_agents


def _mint(path):
    w = GGUFWriter(str(path), "llama")
    w.add_tensor("plain.f32", np.zeros((4, 16), dtype=np.float32),
                 raw_dtype=GT.F32)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def _cfg(tmp_path, body):
    lib = tmp_path / "lib"
    lib.mkdir(exist_ok=True)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(body.replace("<LIB>", str(lib)))
    return cfg, lib


_BASE = """
server:
  model_dirs:
    - <LIB>
models:
  m:
    path: m.gguf
"""


@pytest.fixture(autouse=True)
def _quiet_env(monkeypatch):
    """Pin the environment-dependent checks so tests are hermetic."""
    monkeypatch.setattr(doctor, "check_runtime",
                        lambda: doctor._check("runtime", "PASS", "pinned"))
    monkeypatch.setattr(doctor, "check_kernels",
                        lambda: doctor._check("kernels", "PASS", "pinned"))
    monkeypatch.setattr(doctor, "check_server",
                        lambda: doctor._check("server", "SKIP", "pinned"))
    monkeypatch.setattr(doctor, "check_hf_token",
                        lambda: doctor._check("hf token", "SKIP", "pinned"))
    monkeypatch.setattr(doctor, "check_launcher", lambda: None)
    monkeypatch.setattr(doctor, "check_agents", lambda: None)
    # Keep tests hermetic from whatever server the host machine is running.
    monkeypatch.setattr(doctor, "_running_configs", lambda path: [])


def test_all_pass_exit_zero(tmp_path, capsys):
    cfg, lib = _cfg(tmp_path, _BASE)
    _mint(lib / "m.gguf")
    rc = doctor.cmd_doctor(["--config", str(cfg)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "doctor" in out.splitlines()[0]
    assert "all checks passed." in out
    assert "PASS  config" in out and "PASS  models" in out
    assert "1 model, all paths present" in out


def test_missing_model_fails(tmp_path, capsys):
    cfg, _lib = _cfg(tmp_path, _BASE)          # m.gguf never minted
    rc = doctor.cmd_doctor(["--config", str(cfg)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL  models" in out and "m: missing path" in out
    assert "1 check failed." in out


def test_warns_do_not_fail(tmp_path, capsys):
    # An unknown family is a load-time warnings.warn, not a ConfigError; the
    # config row goes WARN and the exit code stays 0. (Dangling aliases and
    # defaults.model are hard ConfigErrors and fail the config check instead.)
    body = _BASE + "    family: martian\n"
    cfg, lib = _cfg(tmp_path, body)
    _mint(lib / "m.gguf")
    rc = doctor.cmd_doctor(["--config", str(cfg)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "WARN  config" in out and "1 warning" in out
    assert "unknown family" in out


def test_broken_config_fails(tmp_path, capsys):
    body = _BASE + "aliases:\n  ghost: nowhere\n"
    cfg, lib = _cfg(tmp_path, body)
    _mint(lib / "m.gguf")
    rc = doctor.cmd_doctor(["--config", str(cfg)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL  config" in out and "ghost" in out
    assert "SKIP  models" in out


def test_no_config_is_warn(tmp_path, monkeypatch, capsys):
    from gmlx import config as cfgmod
    monkeypatch.setattr(cfgmod, "default_config_paths",
                        lambda: [tmp_path / "absent.yaml"])
    rc = doctor.cmd_doctor([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "WARN  config" in out and "gmlx init" in out
    assert "SKIP  models" in out


def test_explicit_missing_config_is_usage_error(tmp_path, capsys):
    rc = doctor.cmd_doctor(["--config", str(tmp_path / "absent.yaml")])
    assert rc == 2
    assert "no config file" in capsys.readouterr().err


def test_conditional_rows_absent_without_features(tmp_path, capsys):
    cfg, lib = _cfg(tmp_path, _BASE)
    _mint(lib / "m.gguf")
    doctor.cmd_doctor(["--config", str(cfg)])
    out = capsys.readouterr().out
    assert "extras" not in out and "ffmpeg" not in out
    assert "mcp tools" not in out


def test_conditional_rows_present_with_features(tmp_path, monkeypatch, capsys):
    body = """
server:
  model_dirs:
    - <LIB>
  stt: whisper-turbo
models:
  m:
    path: m.gguf
talk:
  brain: assistant
assistant:
  mcp:
    - name: fs
      command: [definitely-not-a-real-binary, /tmp]
"""
    cfg, lib = _cfg(tmp_path, body)
    _mint(lib / "m.gguf")
    from gmlx import extras
    monkeypatch.setattr(extras, "extra_installed", lambda x: x != "assistant")
    monkeypatch.setattr(extras, "ffmpeg_present", lambda: False)
    rc = doctor.cmd_doctor(["--config", str(cfg)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL  extras" in out and "assistant" in out
    assert "FAIL  ffmpeg" in out and "brew install ffmpeg" in out
    assert "WARN  mcp tools" in out
    assert "fs: definitely-not-a-real-binary" in out


def test_assistant_exposure_warn_names_scoping(tmp_path, monkeypatch, capsys):
    from gmlx import extras
    monkeypatch.setattr(extras, "extra_installed", lambda x: True)
    body = """
server:
  model_dirs:
    - <LIB>
  host: 0.0.0.0
  assistant_allow_remote: true
  assistants:
    open-helper:
      model: m
    scoped-helper:
      model: m
      mcp:
        - name: w
          url: http://127.0.0.1:1/mcp
models:
  m:
    path: m.gguf
"""
    cfg, lib = _cfg(tmp_path, body)
    _mint(lib / "m.gguf")
    rc = doctor.cmd_doctor(["--config", str(cfg)])
    out = capsys.readouterr().out
    assert rc == 0                       # WARNs do not fail doctor
    assert "WARN  assistants" in out
    assert "non-loopback 0.0.0.0" in out
    assert "open-helper (inherits full assistant.mcp tools)" in out
    assert "scoped-helper (own mcp list, 1 server)" in out
    # loopback bind: no exposure row
    body2 = body.replace("host: 0.0.0.0", "host: 127.0.0.1")
    (tmp_path / "b").mkdir()
    cfg2, lib2 = _cfg(tmp_path / "b", body2)
    _mint(lib2 / "m.gguf")
    doctor.cmd_doctor(["--config", str(cfg2)])
    assert "WARN  assistants" not in capsys.readouterr().out


def test_mcp_row_scans_alias_lists_too(tmp_path, monkeypatch, capsys):
    from gmlx import extras
    monkeypatch.setattr(extras, "extra_installed", lambda x: True)
    body = """
server:
  model_dirs:
    - <LIB>
  assistants:
    helper:
      model: m
      mcp:
        - name: fs
          command: [definitely-not-a-real-binary, /tmp]
models:
  m:
    path: m.gguf
"""
    cfg, lib = _cfg(tmp_path, body)
    _mint(lib / "m.gguf")
    doctor.cmd_doctor(["--config", str(cfg)])
    out = capsys.readouterr().out
    assert "WARN  mcp tools" in out
    assert "fs: definitely-not-a-real-binary" in out


def test_deep_reads_headers(tmp_path, capsys):
    cfg, lib = _cfg(tmp_path, _BASE)
    _mint(lib / "m.gguf")
    rc = doctor.cmd_doctor(["--config", str(cfg), "--deep"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "headers ok" in out


def test_json_shape(tmp_path, capsys):
    cfg, lib = _cfg(tmp_path, _BASE)
    _mint(lib / "m.gguf")
    rc = doctor.cmd_doctor(["--config", str(cfg), "--json"])
    v = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert v["ok"] is True and v["version"]
    names = [c["name"] for c in v["checks"]]
    assert names[:5] == ["runtime", "kernels", "config", "models", "server"]
    assert all({"name", "status", "detail"} <= set(c) for c in v["checks"])


def test_doctor_prints_running_server_row(tmp_path, monkeypatch, capsys):
    # The server row a user actually reads: a healthy background server prints
    # PASS with the running-at host:port (pid) detail through the real
    # check_server, end to end in cmd_doctor (the autouse pin is overridden).
    from gmlx import lifecycle
    monkeypatch.setattr(doctor, "check_server", _real_check_server)
    monkeypatch.setattr(lifecycle, "list_runs",
                        lambda: [{"host": "127.0.0.1", "port": 8080,
                                  "pid": 1234}])
    monkeypatch.setattr(lifecycle, "identity_ok", lambda run: True)
    monkeypatch.setattr(lifecycle, "_health_ok", lambda h, p: True)
    cfg, lib = _cfg(tmp_path, _BASE)
    _mint(lib / "m.gguf")
    rc = doctor.cmd_doctor(["--config", str(cfg)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "PASS  server" in out
    assert "running at 127.0.0.1:8080 (pid 1234)" in out


def test_stale_server_warns(monkeypatch):
    from gmlx import lifecycle
    monkeypatch.setattr(lifecycle, "list_runs",
                        lambda: [{"host": "127.0.0.1", "port": 8080,
                                  "pid": 99999999}])
    monkeypatch.setattr(lifecycle, "identity_ok", lambda run: False)
    c = _real_check_server()
    assert c["status"] == "WARN"
    assert "1 stale run file [127.0.0.1:8080]" in c["detail"]


def test_healthy_server_passes(monkeypatch):
    from gmlx import lifecycle
    monkeypatch.setattr(lifecycle, "list_runs",
                        lambda: [{"host": "127.0.0.1", "port": 8080,
                                  "pid": 1234}])
    monkeypatch.setattr(lifecycle, "identity_ok", lambda run: True)
    monkeypatch.setattr(lifecycle, "_health_ok", lambda h, p: True)
    c = _real_check_server()
    assert c["status"] == "PASS"
    assert "running at 127.0.0.1:8080 (pid 1234)" in c["detail"]


def test_services_bare_repo_id_is_not_a_local_dir():
    """A bare HF repo id (org/name) contains a separator but is a repo
    reference; it must not FAIL as a missing local directory."""
    import types

    cfg = types.SimpleNamespace(
        embeddings="mlx-community/bge-m3-mlx-8bit",
        rerank=None, stt=None, tts=None, model_dirs=[])
    row = doctor.check_services(cfg)
    assert row["status"] == "PASS"


def test_services_missing_local_dir_still_fails(tmp_path):
    import types

    cfg = types.SimpleNamespace(
        embeddings=str(tmp_path / "not-there"),
        rerank=None, stt=None, tts=None, model_dirs=[])
    row = doctor.check_services(cfg)
    assert row["status"] == "FAIL"


# launcher: the renamed interpreter copy must actually execute - an
# interpreter swap under the venv can strand it (dyld abort before main()).
def test_launcher_pass_with_working_stub(monkeypatch):
    import sys as _sys

    if _sys.platform != "darwin":
        pytest.skip("launcher stub is macOS-only")
    from gmlx import procname
    monkeypatch.setattr(procname, "named_python", lambda: _sys.executable)
    c = _real_check_launcher()
    assert c["status"] == "PASS"


def test_launcher_fail_reports_dyld_line(monkeypatch, tmp_path):
    import sys as _sys

    if _sys.platform != "darwin":
        pytest.skip("launcher stub is macOS-only")
    from gmlx import procname
    bad = tmp_path / "gmlx"
    bad.write_text("#!/bin/sh\n"
                   "echo 'dyld[1]: Library not loaded: @executable_path/../"
                   "lib/libpython3.12.dylib' >&2\nexit 134\n")
    bad.chmod(0o755)
    monkeypatch.setattr(procname, "named_python", lambda: str(bad))
    c = _real_check_launcher()
    assert c["status"] == "FAIL"
    assert "Library not loaded" in c["detail"]


def test_launcher_warn_when_no_stub(monkeypatch):
    import sys as _sys

    if _sys.platform != "darwin":
        pytest.skip("launcher stub is macOS-only")
    from gmlx import procname
    monkeypatch.setattr(procname, "named_python", lambda: None)
    c = _real_check_launcher()
    assert c["status"] == "WARN"


# Extras follow the RUNNING server's config too: `serve --config talk.yaml`
# enables features the default-location config leaves commented out, and a
# doctor that only reads the latter would pass while wake mode degrades.
def test_extras_row_follows_running_server_config(tmp_path, monkeypatch, capsys):
    cfg, lib = _cfg(tmp_path, _BASE)               # no features configured
    _mint(lib / "m.gguf")
    talk_body = _BASE + "talk:\n  wake_word: hey\n"
    served = tmp_path / "served.yaml"
    served.write_text(talk_body.replace("<LIB>", str(lib)))

    from gmlx import config as cfgmod
    from gmlx import extras
    served_cfg = cfgmod.load_config(str(served))
    monkeypatch.setattr(doctor, "_running_configs",
                        lambda path: [(served_cfg, str(served))])
    monkeypatch.setattr(extras, "extra_installed", lambda x: False)
    monkeypatch.setattr(extras, "missing_extra_modules",
                        lambda x: ["sherpa_onnx"])
    monkeypatch.setattr(extras, "ffmpeg_present", lambda: True)
    rc = doctor.cmd_doctor(["--config", str(cfg)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL  extras" in out
    assert "talk (sherpa_onnx)" in out
    assert f"[server config {served}]" in out
    assert 'pip install "gmlx[talk]"' in out
    assert "PASS  ffmpeg" in out                   # audio need carried over too


def test_running_configs_skip_stale_and_primary(tmp_path, monkeypatch):
    from gmlx import lifecycle
    lib = tmp_path / "lib"
    lib.mkdir()
    primary = tmp_path / "cfg.yaml"
    other = tmp_path / "other.yaml"
    for p in (primary, other):
        p.write_text(_BASE.replace("<LIB>", str(lib)))
    monkeypatch.setattr(lifecycle, "list_runs", lambda: [
        {"config_abspath": str(primary), "pid": 1},   # doctor already has it
        {"config_abspath": str(other), "pid": 2},     # stale -> skipped
        {"config_abspath": None, "pid": 3},           # discovery start
    ])
    monkeypatch.setattr(lifecycle, "identity_ok", lambda run: False)
    assert _real_running_configs(str(primary)) == []
    monkeypatch.setattr(lifecycle, "identity_ok", lambda run: True)
    got = _real_running_configs(str(primary))
    assert [p for _cfg2, p in got] == [str(other)]


# launchd agents row
def test_agents_row_states(monkeypatch, tmp_path):
    import sys as _sys
    if _sys.platform != "darwin":
        pytest.skip("launchd is macOS-only")
    from gmlx import lifecycle
    plists = [tmp_path / "com.gmlx.menubar.plist",
              tmp_path / "com.gmlx.server.127-0-0-1-8080.plist"]
    for p in plists:
        p.write_text("x")
    monkeypatch.setattr(doctor, "_agent_plists", lambda: sorted(plists))
    monkeypatch.setattr(lifecycle, "agent_loaded", lambda label: True)
    c = _real_check_agents()
    assert c["status"] == "PASS" and "2 agents" in c["detail"]
    monkeypatch.setattr(lifecycle, "agent_loaded",
                        lambda label: label == "com.gmlx.menubar")
    c = _real_check_agents()
    assert c["status"] == "WARN"
    assert "com.gmlx.server.127-0-0-1-8080" in c["detail"]
    assert "gmlx service" in c["detail"]


def test_agents_row_absent_without_plists(monkeypatch):
    monkeypatch.setattr(doctor, "_agent_plists", lambda: [])
    assert _real_check_agents() is None
