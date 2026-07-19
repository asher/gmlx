#!/usr/bin/env python3
"""`gmlx launch <harness>`: the opencode config writer, server probe,
default-model pick, and exec/no-exec flow. CPU-only - the HTTP probe is faked, the
exec is a recording seam, so no server, no model, no real harness is run."""
from __future__ import annotations

import json
import time
import tomllib
from pathlib import Path

import pytest

from gmlx import launch, lifecycle  # noqa: E402


@pytest.fixture(autouse=True)
def _fake_home(monkeypatch, tmp_path_factory):
    # Default config paths land under ~; the launch module expanduser()s them at
    # call time, so pointing HOME at a temp dir keeps every test out of the real
    # home even if a config_path override is forgotten.
    monkeypatch.setenv("HOME", str(tmp_path_factory.mktemp("home")))


@pytest.fixture(autouse=True)
def _default_server_up(monkeypatch):
    # start-if-down probes _server_ready before dispatching to a harness. Default it to
    # "up" so harness-flow tests stay deterministic regardless of whether a real server
    # happens to be live on this box; the down-path tests below override it.
    monkeypatch.setattr(launch, "_server_ready", lambda base, api_key=None: True)


def _models():
    return [
        {"id": "qwen3.6-27b", "default": True, "vlm": False, "speculative": True},
        {"id": "gemma-e2b", "default": False, "vlm": False},
        {"id": "coder", "alias_of": "qwen3.6-27b", "profile": "coder",
         "default": False},
    ]


def _parse_launch_args(argv):
    """Run the REAL `gmlx launch` parser and capture its namespace, so parser
    drift (a renamed/removed option, a changed default) fails tests instead of
    diverging silently from a hand-built stub."""
    real = launch._ensure_server
    box = {}

    def grab(a):
        box["a"] = a
        return 0                                   # short-circuit before any harness work

    launch._ensure_server = grab
    try:
        rc = launch.cmd_launch(argv)
    finally:
        launch._ensure_server = real
    assert rc == 0 and "a" in box
    return box["a"]


def _args(**kw):
    a = _parse_launch_args(["opencode"])           # real parser: every flag + default
    overrides = dict(harness="opencode", model=None,
                     base_url="http://127.0.0.1:8080/v1",
                     host="127.0.0.1", port=8080, provider_id="gmlx",
                     config_path=None, config_only=False, api_key=None,
                     no_start=False, start_timeout=0.0)
    overrides.update(kw)
    for k, v in overrides.items():
        assert hasattr(a, k), f"launch parser lost option {k!r}"
        setattr(a, k, v)
    return a


# build_opencode_config (pure)
def test_build_opencode_config_shape():
    cfg = launch.build_opencode_config("http://127.0.0.1:8080/v1", _models(),
                                       default_model="qwen3.6-27b")
    prov = cfg["provider"]["gmlx"]
    assert prov["npm"] == "@ai-sdk/openai-compatible"
    assert prov["options"]["baseURL"] == "http://127.0.0.1:8080/v1"
    assert "apiKey" not in prov["options"]                  # local, unauthenticated
    assert set(prov["models"]) == {"qwen3.6-27b", "gemma-e2b", "coder"}  # + aliases
    assert cfg["model"] == "gmlx/qwen3.6-27b"           # default -> provider/model
    assert cfg["$schema"] == "https://opencode.ai/config.json"


def test_build_opencode_config_marks_alias_in_name():
    cfg = launch.build_opencode_config("http://x/v1", _models())
    name = cfg["provider"]["gmlx"]["models"]["coder"]["name"]
    assert "alias of qwen3.6-27b" in name and "coder" in name


def test_build_opencode_config_no_default_omits_model():
    cfg = launch.build_opencode_config("http://x/v1", _models())   # default_model None
    assert "model" not in cfg


def test_provider_id_override():
    cfg = launch.build_opencode_config("http://x/v1", _models(),
                                       provider_id="local", default_model="gemma-e2b")
    assert "local" in cfg["provider"]
    assert cfg["model"] == "local/gemma-e2b"


def _models_with_services():
    return _models() + [
        {"id": "whisper-1", "stt": True, "default": False},
        {"id": "text-embedding-3-small", "embeddings": True, "default": False},
        {"id": "reranker", "rerank": True, "default": False},
    ]


def test_builders_exclude_service_entries():
    # whisper-1 / text-embedding-3-small / reranker answer their own endpoints,
    # not /v1/chat/completions - offering them as chat models 404s on pick.
    models = _models_with_services()
    chat_ids = {"qwen3.6-27b", "gemma-e2b", "coder"}
    assert {m["id"] for m in launch.chat_models(models)} == chat_ids
    oc = launch.build_opencode_config("http://x/v1", models)
    assert set(oc["provider"]["gmlx"]["models"]) == chat_ids
    ac = launch.build_aichat_config("http://x/v1", models)
    assert {e["name"] for e in ac["clients"][0]["models"]} == chat_ids
    pi_models, _ = launch.build_pi_configs("http://x/v1", models)
    assert {m["id"] for m in pi_models["providers"]["gmlx"]["models"]} \
        == chat_ids
    omp_models, _ = launch.build_omp_configs("http://x/v1", models)
    assert {m["id"] for m in omp_models["providers"]["gmlx"]["models"]} \
        == chat_ids
    elia = launch.build_elia_config("http://x/v1", models)
    assert "text-embedding-3-small" not in elia and "whisper-1" not in elia


# _pick_default
def test_pick_default_uses_server_default():
    assert launch._pick_default(_models(), None) == "qwen3.6-27b"


def test_pick_default_explicit_model():
    assert launch._pick_default(_models(), "gemma-e2b") == "gemma-e2b"


def test_pick_default_unknown_model_raises():
    with pytest.raises(launch.LaunchError) as e:
        launch._pick_default(_models(), "nope")
    assert "available" in str(e.value)


def test_pick_default_accepts_profile_suffix_on_served_id():
    """`--model id@profile` passes when the head is served - the profile half is
    validated server-side (unknown -> 400 listing valid names)."""
    assert (launch._pick_default(_models(), "qwen3.6-27b@coding")
            == "qwen3.6-27b@coding")


def test_pick_default_profile_suffix_unknown_head_raises():
    with pytest.raises(launch.LaunchError):
        launch._pick_default(_models(), "nope@coding")


def test_pick_default_none_when_no_marker():
    models = [{"id": "a"}, {"id": "b"}]
    assert launch._pick_default(models, None) is None


# probe_models (faked HTTP)
def test_probe_models_ok(monkeypatch):
    seen = []

    def fake_get(url, timeout=5.0, headers=None):
        seen.append(url)
        if url.endswith("/health"):
            return {"status": "ok"}
        return {"object": "list", "data": _models()}

    monkeypatch.setattr(launch, "_http_get_json", fake_get)
    data = launch.probe_models("http://127.0.0.1:8080/v1")
    assert [m["id"] for m in data][:2] == ["qwen3.6-27b", "gemma-e2b"]
    assert seen[0].endswith(":8080/health")               # health on the root, no /v1
    assert seen[1].endswith("/v1/models")


def test_probe_models_server_down(monkeypatch):
    def fake_get(url, timeout=5.0, headers=None):
        raise OSError("connection refused")

    monkeypatch.setattr(launch, "_http_get_json", fake_get)
    with pytest.raises(launch.LaunchError) as e:
        launch.probe_models("http://127.0.0.1:8080/v1")
    assert "gmlx serve" in str(e.value)                   # tells you how to start one


def test_server_root_strips_v1():
    assert launch._server_root("http://h:8080/v1") == "http://h:8080"
    assert launch._server_root("http://h:8080/v1/") == "http://h:8080"
    assert launch._server_root("http://h:9000") == "http://h:9000"


def test_ensure_server_normalizes_explicit_base_url(monkeypatch):
    # A /v1-less --base-url must be normalized before the harness (talk's
    # audio routes exist only under /v1), for both the ready and the
    # not-ready explicit-endpoint paths.
    import types

    for ready in (True, False):
        monkeypatch.setattr(launch, "_server_ready",
                            lambda base, key, _r=ready: _r)
        a = types.SimpleNamespace(base_url="http://127.0.0.1:3000",
                                  host=None, port=None, api_key=None)
        assert launch._ensure_server(a) is None
        assert a.base_url == "http://127.0.0.1:3000/v1"


def test_ensure_server_takes_the_port_from_the_base_url(monkeypatch):
    """`--base-url http://host:3000/v1` left a.port at the 8080 default, so
    _launch_open_webui derived webui_port=3000 and collided with the server."""
    import types

    for ready in (True, False):
        monkeypatch.setattr(launch, "_server_ready",
                            lambda base, key, _r=ready: _r)
        a = types.SimpleNamespace(base_url="http://10.0.0.5:3000/v1",
                                  host=None, port=None, api_key=None)
        assert launch._ensure_server(a) is None
        assert (a.host, a.port) == ("10.0.0.5", 3000)

    # An explicit --port still wins over the URL.
    monkeypatch.setattr(launch, "_server_ready", lambda base, key: True)
    a = types.SimpleNamespace(base_url="http://127.0.0.1:3000/v1",
                              host=None, port=9999, api_key=None)
    assert launch._ensure_server(a) is None
    assert a.port == 9999


def test_ensure_server_tolerates_a_malformed_base_url_port(monkeypatch):
    """An out-of-range port in --base-url falls back to the default instead of
    an unhandled ValueError out of urlsplit().port; the probe reports the bad
    endpoint cleanly downstream."""
    import types

    monkeypatch.setattr(launch, "_server_ready", lambda base, key: True)
    a = types.SimpleNamespace(base_url="http://127.0.0.1:99999/v1",
                              host=None, port=None, api_key=None)
    assert launch._ensure_server(a) is None
    assert a.port == 8080


def test_open_webui_picks_a_free_port_against_a_3000_server(monkeypatch, tmp_path):
    """The server on 3000 must push Open WebUI off its own 3000 default."""
    import types

    monkeypatch.setattr(launch, "probe_models", lambda base, api_key=None: _models())
    monkeypatch.setattr(launch.shutil, "which", lambda b: "/bin/open-webui")
    seen: dict = {}

    def _exec(binary, argv, env):
        seen["port"] = env["PORT"]

    monkeypatch.setattr(launch, "_server_ready", lambda base, key: True)
    a = types.SimpleNamespace(base_url="http://127.0.0.1:3000/v1", host=None,
                              port=None, api_key=None, model=None,
                              config_only=False, config_path=str(tmp_path))
    assert launch._ensure_server(a) is None      # resolves a.port off the URL
    launch._launch_open_webui(a, exec_fn=_exec)
    assert seen["port"] == "3001"                # not 3000: the server holds it


# _launch_opencode flow (faked probe + recording exec)
def _fake_probe(monkeypatch):
    monkeypatch.setattr(launch, "probe_models", lambda base, api_key=None: _models())
    monkeypatch.setattr(launch.shutil, "which", lambda name: f"/usr/bin/{name}")


def test_launch_writes_config_and_execs(monkeypatch, tmp_path):
    _fake_probe(monkeypatch)
    cfg_path = tmp_path / "opencode.json"
    calls = {}

    def fake_exec(binary, argv, env):
        calls["binary"], calls["argv"], calls["env"] = binary, argv, env
        return 0

    rc = launch._launch_opencode(_args(config_path=str(cfg_path)), exec_fn=fake_exec)
    assert rc == 0
    assert calls["binary"] == "/usr/bin/opencode"
    assert calls["env"]["OPENCODE_CONFIG"] == str(cfg_path)   # injected, not the user file
    cfg = json.loads(cfg_path.read_text())
    assert cfg["model"] == "gmlx/qwen3.6-27b"


def test_launch_config_only_does_not_exec(monkeypatch, tmp_path, capsys):
    _fake_probe(monkeypatch)
    cfg_path = tmp_path / "opencode.json"
    execd = []
    rc = launch._launch_opencode(_args(config_path=str(cfg_path), config_only=True),
                                 exec_fn=lambda *a: execd.append(a) or 0)
    assert rc == 0
    assert execd == []                                       # never exec'd
    assert cfg_path.exists()
    assert "OPENCODE_CONFIG=" in capsys.readouterr().out      # prints the run command


def test_launch_missing_binary_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(launch, "probe_models", lambda base, api_key=None: _models())
    monkeypatch.setattr(launch.shutil, "which", lambda name: None)   # not installed
    rc = launch.cmd_launch(["opencode", "--config-path", str(tmp_path / "c.json")])
    assert rc == 1                                           # no auto-install, clean exit


def test_launch_config_only_works_without_binary(monkeypatch, tmp_path):
    # --config-only just writes the file, so a missing binary is fine.
    monkeypatch.setattr(launch, "probe_models", lambda base, api_key=None: _models())
    monkeypatch.setattr(launch.shutil, "which", lambda name: None)
    out = tmp_path / "c.json"
    rc = launch._launch_opencode(_args(config_path=str(out), config_only=True),
                                 exec_fn=lambda *a: 0)
    assert rc == 0 and out.exists()


def test_handler_prefers_base_url_over_host_port(monkeypatch, tmp_path):
    # _ensure_server can resolve a base_url that disagrees with host/port (e.g.
    # a config endpoint); every request must follow base_url, not the pair.
    probed = []
    monkeypatch.setattr(launch, "probe_models",
                        lambda base, api_key=None: probed.append(base) or _models())
    monkeypatch.setattr(launch.shutil, "which", lambda name: f"/usr/bin/{name}")
    cfg_path = tmp_path / "c.json"
    a = _args(base_url="http://10.9.8.7:9999/v1", host="127.0.0.1", port=8080,
              config_path=str(cfg_path))
    rc = launch._launch_opencode(a, exec_fn=lambda *x: 0)
    assert rc == 0
    assert probed == ["http://10.9.8.7:9999/v1"]
    cfg = json.loads(cfg_path.read_text())
    assert (cfg["provider"]["gmlx"]["options"]["baseURL"]
            == "http://10.9.8.7:9999/v1")
    # The keep request follows base_url too.
    posts = []
    monkeypatch.setattr(launch, "_http_post_json",
                        lambda url, body, **k: posts.append(url) or {})
    a.model = "gemma-e2b"
    launch._keep_model(a)
    assert posts == ["http://10.9.8.7:9999/v1/keep"]


def test_handler_base_url_none_falls_back_to_host_port(monkeypatch, tmp_path):
    probed = []
    monkeypatch.setattr(launch, "probe_models",
                        lambda base, api_key=None: probed.append(base) or _models())
    monkeypatch.setattr(launch.shutil, "which", lambda name: f"/usr/bin/{name}")
    a = _args(base_url=None, host="10.0.0.5", port=9001,
              config_path=str(tmp_path / "c.json"))
    rc = launch._launch_opencode(a, exec_fn=lambda *x: 0)
    assert rc == 0
    assert probed == ["http://10.0.0.5:9001/v1"]


# launch --model keeps the model resident (the keep tier)
def test_launch_model_fires_keep(monkeypatch, tmp_path):
    _fake_probe(monkeypatch)
    posts = []
    monkeypatch.setattr(launch, "_http_post_json",
                        lambda url, body, **kw: posts.append((url, body)) or {"status": "kept"})
    execd = []
    rc = launch.cmd_launch(
        ["opencode", "--model", "gemma-e2b", "--base-url", "http://127.0.0.1:8080/v1",
         "--config-path", str(tmp_path / "c.json")],
        exec_fn=lambda *a: execd.append(a) or 0)
    assert rc == 0
    assert posts == [("http://127.0.0.1:8080/v1/keep",
                      {"model": "gemma-e2b", "warm": True})]
    assert execd                                             # harness still execs


def test_launch_no_keep_fires_nothing(monkeypatch, tmp_path):
    _fake_probe(monkeypatch)
    posts = []
    monkeypatch.setattr(launch, "_http_post_json", lambda *a, **k: posts.append(a))
    rc = launch.cmd_launch(
        ["opencode", "--model", "gemma-e2b", "--no-keep",
         "--base-url", "http://127.0.0.1:8080/v1", "--config-path", str(tmp_path / "c.json")],
        exec_fn=lambda *a: 0)
    assert rc == 0
    assert posts == []                                       # opted out


def test_launch_without_model_fires_nothing(monkeypatch, tmp_path):
    _fake_probe(monkeypatch)
    posts = []
    monkeypatch.setattr(launch, "_http_post_json", lambda *a, **k: posts.append(a))
    rc = launch.cmd_launch(
        ["opencode", "--base-url", "http://127.0.0.1:8080/v1",
         "--config-path", str(tmp_path / "c.json")],
        exec_fn=lambda *a: 0)
    assert rc == 0
    assert posts == []                                       # no --model -> nothing to keep


def test_launch_config_only_fires_no_keep(monkeypatch, tmp_path):
    _fake_probe(monkeypatch)
    posts = []
    monkeypatch.setattr(launch, "_http_post_json", lambda *a, **k: posts.append(a))
    rc = launch.cmd_launch(
        ["opencode", "--model", "gemma-e2b", "--config-only",
         "--base-url", "http://127.0.0.1:8080/v1", "--config-path", str(tmp_path / "c.json")],
        exec_fn=lambda *a: 0)
    assert rc == 0
    assert posts == []                                       # config-only doesn't touch runtime


def test_launch_unknown_model_errors_before_keep(monkeypatch, tmp_path, capsys):
    # --model validation runs BEFORE the keep POST: one clean "not served"
    # error, exit 1, no contradictory "keeping X resident" line, no /v1/keep
    # call for an id the server doesn't serve.
    _fake_probe(monkeypatch)
    posts = []
    monkeypatch.setattr(launch, "_http_post_json",
                        lambda *a, **k: posts.append(a) or {})
    execd = []
    rc = launch.cmd_launch(
        ["opencode", "--model", "nope", "--base-url", "http://127.0.0.1:8080/v1",
         "--config-path", str(tmp_path / "c.json")],
        exec_fn=lambda *a: execd.append(a) or 0)
    cap = capsys.readouterr()
    assert rc == 1
    assert posts == []
    assert not execd
    assert "not served" in cap.err
    assert "keeping" not in cap.out and "keeping" not in cap.err


def test_launch_keep_unknown_model_404_body_notes_skip(monkeypatch, capsys):
    # Backstop inside _keep_model itself: a new server's 404 with the
    # unknown_model JSON body reads as "keep skipped", not "no /v1/keep route".
    import io
    import urllib.error

    def _raise(url, body, **kw):
        payload = io.BytesIO(b'{"status": "unknown_model", "model": "nope"}')
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, payload)

    monkeypatch.setattr(launch, "_http_post_json", _raise)
    a = _args(model="nope")
    launch._keep_model(a)
    out = capsys.readouterr().out
    assert "keep skipped" in out
    assert "no /v1/keep route" not in out


def test_launch_keep_old_server_404_still_execs(monkeypatch, tmp_path):
    import urllib.error

    _fake_probe(monkeypatch)

    def _raise(url, body, **kw):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(launch, "_http_post_json", _raise)
    execd = []
    rc = launch.cmd_launch(
        ["opencode", "--model", "gemma-e2b", "--base-url", "http://127.0.0.1:8080/v1",
         "--config-path", str(tmp_path / "c.json")],
        exec_fn=lambda *a: execd.append(a) or 0)
    assert rc == 0
    assert execd                                             # a route-less old server doesn't block


def test_cmd_launch_unknown_harness_argparse_errors():
    with pytest.raises(SystemExit) as e:
        launch.cmd_launch(["no-such-harness"])
    assert e.value.code == 2                                 # argparse choices guard


def test_cmd_launch_bare_prints_help(capsys):
    rc = launch.cmd_launch([])                               # no harness -> help, not error
    assert rc == 0
    out = capsys.readouterr().out
    assert "usage:" in out and "harness" in out             # long-form help, exit 0


def test_cmd_launch_menubar_routes(monkeypatch):
    seen = {}
    import gmlx.menubar as mb

    def fake_menubar(argv, prog=None):
        seen["argv"] = argv
        return 0

    monkeypatch.setattr(mb, "cmd_menubar", fake_menubar)
    rc = launch.cmd_launch(["menubar", "--interval", "9"])
    assert rc == 0
    assert seen["argv"] == ["--interval", "9"]              # menubar opts passed through


# pi: build_pi_configs (pure) - merge into the user's own ~/.pi/agent files
def test_build_pi_configs_shape():
    models_doc, settings_doc = launch.build_pi_configs(
        "http://127.0.0.1:8080/v1", _models(), default_model="qwen3.6-27b")
    prov = models_doc["providers"]["gmlx"]
    assert prov["baseUrl"] == "http://127.0.0.1:8080/v1"
    assert prov["api"] == "openai-completions"
    assert [m["id"] for m in prov["models"]] == ["qwen3.6-27b", "gemma-e2b", "coder"]
    assert settings_doc["defaultProvider"] == "gmlx"
    assert settings_doc["defaultModel"] == "qwen3.6-27b"


def test_build_pi_configs_no_default_omits_model():
    _, settings_doc = launch.build_pi_configs("http://x/v1", _models())
    assert "defaultModel" not in settings_doc
    assert settings_doc["defaultProvider"] == "gmlx"


def test_build_pi_configs_preserves_existing():
    existing_models = {"providers": {"openai": {"baseUrl": "https://api.openai.com/v1"}}}
    existing_settings = {"theme": "dark", "defaultModel": "gpt-4o"}
    models_doc, settings_doc = launch.build_pi_configs(
        "http://x/v1", _models(), default_model="gemma-e2b",
        existing_models=existing_models, existing_settings=existing_settings)
    assert "openai" in models_doc["providers"]               # other provider kept
    assert "gmlx" in models_doc["providers"]
    assert settings_doc["theme"] == "dark"                   # unrelated setting kept
    assert settings_doc["defaultProvider"] == "gmlx"     # we take over the default
    assert settings_doc["defaultModel"] == "gemma-e2b"       # overridden to a served id


# pi: _load_json
def test_load_json_absent_is_empty(tmp_path):
    assert launch._load_json(tmp_path / "nope.json") == {}


def test_load_json_empty_file_is_empty(tmp_path):
    p = tmp_path / "e.json"
    p.write_text("   \n")
    assert launch._load_json(p) == {}


def test_load_json_malformed_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    with pytest.raises(launch.LaunchError) as e:
        launch._load_json(p)
    assert "refusing to overwrite" in str(e.value)


# pi: _launch_pi flow (faked probe + recording exec)
def _fake_pi_probe(monkeypatch):
    monkeypatch.setattr(launch, "probe_models", lambda base, api_key=None: _models())
    monkeypatch.setattr(launch.shutil, "which", lambda name: f"/usr/bin/{name}")


def test_launch_pi_writes_both_files_and_execs(monkeypatch, tmp_path):
    _fake_pi_probe(monkeypatch)
    calls = {}

    def fake_exec(binary, argv, env):
        calls["binary"], calls["argv"] = binary, argv
        return 0

    rc = launch._launch_pi(_args(harness="pi", config_path=str(tmp_path)),
                           exec_fn=fake_exec)
    assert rc == 0
    assert calls["binary"] == "/usr/bin/pi" and calls["argv"] == ["pi"]
    models_doc = json.loads((tmp_path / "models.json").read_text())
    settings_doc = json.loads((tmp_path / "settings.json").read_text())
    assert "gmlx" in models_doc["providers"]
    assert settings_doc["defaultModel"] == "qwen3.6-27b"     # server default marker


def test_launch_pi_merges_existing_files(monkeypatch, tmp_path):
    _fake_pi_probe(monkeypatch)
    (tmp_path / "models.json").write_text(
        json.dumps({"providers": {"anthropic": {"baseUrl": "https://api.anthropic.com"}}}))
    (tmp_path / "settings.json").write_text(json.dumps({"theme": "solarized"}))
    rc = launch._launch_pi(_args(harness="pi", config_path=str(tmp_path)),
                           exec_fn=lambda *a: 0)
    assert rc == 0
    models_doc = json.loads((tmp_path / "models.json").read_text())
    settings_doc = json.loads((tmp_path / "settings.json").read_text())
    assert "anthropic" in models_doc["providers"]            # preserved
    assert "gmlx" in models_doc["providers"]
    assert settings_doc["theme"] == "solarized"              # preserved


def test_launch_pi_config_only_does_not_exec(monkeypatch, tmp_path, capsys):
    _fake_pi_probe(monkeypatch)
    execd = []
    rc = launch._launch_pi(_args(harness="pi", config_path=str(tmp_path),
                                 config_only=True),
                           exec_fn=lambda *a: execd.append(a) or 0)
    assert rc == 0 and execd == []
    assert (tmp_path / "models.json").exists()
    assert "run it with:  pi" in capsys.readouterr().out


def test_launch_pi_missing_binary_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(launch, "probe_models", lambda base, api_key=None: _models())
    monkeypatch.setattr(launch.shutil, "which", lambda name: None)
    rc = launch.cmd_launch(["pi", "--config-path", str(tmp_path)])
    assert rc == 1                                           # no auto-install, clean exit


def test_launch_pi_malformed_existing_refuses(monkeypatch, tmp_path):
    _fake_pi_probe(monkeypatch)
    (tmp_path / "models.json").write_text("{broken")
    rc = launch.cmd_launch(["pi", "--config-path", str(tmp_path)])
    assert rc == 1                                           # LaunchError -> clean exit, no clobber
    assert (tmp_path / "models.json").read_text() == "{broken"   # untouched


# omp (oh-my-pi): build_omp_configs (pure) - YAML, modelRoles.default
def test_build_omp_configs_shape():
    models_doc, config_doc = launch.build_omp_configs(
        "http://127.0.0.1:8080/v1", _models(), default_model="qwen3.6-27b")
    prov = models_doc["providers"]["gmlx"]
    assert prov["baseUrl"] == "http://127.0.0.1:8080/v1"
    assert prov["api"] == "openai-completions"
    assert prov["auth"] == "none"                            # local, unauthenticated
    assert [m["id"] for m in prov["models"]] == ["qwen3.6-27b", "gemma-e2b", "coder"]
    assert config_doc["modelRoles"]["default"] == "gmlx/qwen3.6-27b"   # role-pinned


def test_build_omp_configs_marks_alias_in_name():
    models_doc, _ = launch.build_omp_configs("http://x/v1", _models())
    names = {m["id"]: m["name"] for m in models_doc["providers"]["gmlx"]["models"]}
    assert "alias of qwen3.6-27b" in names["coder"]


def test_build_omp_configs_no_default_omits_role():
    _, config_doc = launch.build_omp_configs("http://x/v1", _models())
    assert "modelRoles" not in config_doc


def test_build_omp_configs_preserves_existing():
    existing_models = {"providers": {"openai": {"baseUrl": "https://api.openai.com/v1"}}}
    existing_config = {"theme": "dark", "modelRoles": {"smol": "openai/gpt-4o-mini"}}
    models_doc, config_doc = launch.build_omp_configs(
        "http://x/v1", _models(), default_model="gemma-e2b",
        existing_models=existing_models, existing_config=existing_config)
    assert "openai" in models_doc["providers"]               # other provider kept
    assert "gmlx" in models_doc["providers"]
    assert config_doc["theme"] == "dark"                     # unrelated setting kept
    assert config_doc["modelRoles"]["smol"] == "openai/gpt-4o-mini"   # other role kept
    assert config_doc["modelRoles"]["default"] == "gmlx/gemma-e2b"


# omp: _load_yaml
def test_load_yaml_absent_is_empty(tmp_path):
    assert launch._load_yaml(tmp_path / "nope.yml") == {}


def test_load_yaml_empty_file_is_empty(tmp_path):
    p = tmp_path / "e.yml"
    p.write_text("\n\n")
    assert launch._load_yaml(p) == {}


def test_load_yaml_non_mapping_raises(tmp_path):
    p = tmp_path / "list.yml"
    p.write_text("- a\n- b\n")
    with pytest.raises(launch.LaunchError) as e:
        launch._load_yaml(p)
    assert "refusing to overwrite" in str(e.value)


# omp: _launch_omp flow (faked probe + recording exec)
def test_launch_omp_writes_both_files_and_execs(monkeypatch, tmp_path):
    _fake_pi_probe(monkeypatch)                              # which() -> /usr/bin/<name>
    calls = {}

    def fake_exec(binary, argv, env):
        calls["binary"], calls["argv"] = binary, argv
        return 0

    rc = launch._launch_omp(_args(harness="omp", config_path=str(tmp_path)),
                            exec_fn=fake_exec)
    assert rc == 0
    assert calls["binary"] == "/usr/bin/omp" and calls["argv"] == ["omp"]
    import yaml as _yaml
    models_doc = _yaml.safe_load((tmp_path / "models.yml").read_text())
    config_doc = _yaml.safe_load((tmp_path / "config.yml").read_text())
    assert "gmlx" in models_doc["providers"]
    assert config_doc["modelRoles"]["default"] == "gmlx/qwen3.6-27b"


def test_launch_omp_merges_existing(monkeypatch, tmp_path):
    _fake_pi_probe(monkeypatch)
    import yaml as _yaml
    (tmp_path / "models.yml").write_text(
        _yaml.safe_dump({"providers": {"anthropic": {"baseUrl": "https://a"}}}))
    (tmp_path / "config.yml").write_text(_yaml.safe_dump({"theme": "gruvbox"}))
    rc = launch._launch_omp(_args(harness="omp", config_path=str(tmp_path)),
                            exec_fn=lambda *a: 0)
    assert rc == 0
    models_doc = _yaml.safe_load((tmp_path / "models.yml").read_text())
    config_doc = _yaml.safe_load((tmp_path / "config.yml").read_text())
    assert "anthropic" in models_doc["providers"] and "gmlx" in models_doc["providers"]
    assert config_doc["theme"] == "gruvbox"                  # preserved


def test_launch_omp_config_only_does_not_exec(monkeypatch, tmp_path, capsys):
    _fake_pi_probe(monkeypatch)
    execd = []
    rc = launch._launch_omp(_args(harness="omp", config_path=str(tmp_path),
                                  config_only=True),
                            exec_fn=lambda *a: execd.append(a) or 0)
    assert rc == 0 and execd == []
    assert (tmp_path / "models.yml").exists()
    assert "run it with:  omp" in capsys.readouterr().out


def test_launch_omp_missing_binary_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(launch, "probe_models", lambda base, api_key=None: _models())
    monkeypatch.setattr(launch.shutil, "which", lambda name: None)
    rc = launch.cmd_launch(["omp", "--config-path", str(tmp_path)])
    assert rc == 1                                           # no auto-install, clean exit


def test_launch_omp_malformed_existing_refuses(monkeypatch, tmp_path):
    _fake_pi_probe(monkeypatch)
    (tmp_path / "models.yml").write_text("- not\n- a mapping\n")
    rc = launch.cmd_launch(["omp", "--config-path", str(tmp_path)])
    assert rc == 1                                           # LaunchError -> clean exit
    assert (tmp_path / "models.yml").read_text() == "- not\n- a mapping\n"   # untouched


# hermes (NousResearch hermes-agent)
def test_build_hermes_config_shape():
    cfg = launch.build_hermes_config("http://127.0.0.1:8080/v1",
                                     default_model="qwen3.6-27b")
    assert cfg["inference"] == {"provider": "custom", "model": "qwen3.6-27b"}
    custom = cfg["providers"]["custom"]
    assert custom["base_url"] == "http://127.0.0.1:8080/v1"
    assert custom["api_key"] == "gmlx"


def test_build_hermes_config_preserves_existing():
    existing = {
        "inference": {"provider": "openrouter", "model": "x", "temperature": 0.6},
        "providers": {"openrouter": {"api_key": "sk-or-keep"},
                      "custom": {"api_key": "user-key"}},
        "gateway": {"telegram": True},
    }
    cfg = launch.build_hermes_config("http://h/v1", default_model="m",
                                     existing=existing)
    assert cfg["inference"]["provider"] == "custom"            # repointed
    assert cfg["inference"]["temperature"] == 0.6              # other keys kept
    assert cfg["providers"]["openrouter"] == {"api_key": "sk-or-keep"}
    assert cfg["providers"]["custom"]["api_key"] == "user-key"  # not clobbered
    assert cfg["gateway"] == {"telegram": True}
    assert existing["inference"]["provider"] == "openrouter"   # input not mutated


def test_launch_hermes_injects_config_env(monkeypatch, tmp_path):
    import yaml as _yaml
    _fake_probe(monkeypatch)
    monkeypatch.setenv("HERMES_CONFIG", str(tmp_path / "no-user-config.yaml"))
    out = tmp_path / "hermes-config.yaml"
    calls = {}

    def fake_exec(binary, argv, env):
        calls["binary"], calls["argv"], calls["env"] = binary, argv, env
        return 0

    rc = launch._launch_hermes(_args(harness="hermes", config_path=str(out)),
                               exec_fn=fake_exec)
    assert rc == 0
    assert calls["binary"] == "/usr/bin/hermes"
    assert calls["env"]["HERMES_CONFIG"] == str(out)
    assert calls["env"]["CUSTOM_BASE_URL"] == "http://127.0.0.1:8080/v1"
    cfg = _yaml.safe_load(out.read_text())
    assert cfg["inference"]["model"] == "qwen3.6-27b"


def test_launch_hermes_merges_user_config_without_touching_it(monkeypatch, tmp_path):
    import yaml as _yaml
    _fake_probe(monkeypatch)
    user = tmp_path / "user-hermes.yaml"
    user_text = "gateway:\n  discord: true\n"
    user.write_text(user_text)
    monkeypatch.setenv("HERMES_CONFIG", str(user))
    out = tmp_path / "ours.yaml"
    rc = launch._launch_hermes(
        _args(harness="hermes", config_path=str(out), config_only=True),
        exec_fn=lambda *a: 0)
    assert rc == 0
    assert user.read_text() == user_text                       # untouched
    assert _yaml.safe_load(out.read_text())["gateway"] == {"discord": True}


def test_launch_hermes_requires_default_model(monkeypatch, tmp_path):
    monkeypatch.setattr(launch, "probe_models",
                        lambda base, api_key=None: [{"id": "a"}, {"id": "b"}])  # no default mark
    monkeypatch.setattr(launch.shutil, "which", lambda name: f"/usr/bin/{name}")
    with pytest.raises(launch.LaunchError) as e:
        launch._launch_hermes(_args(harness="hermes",
                                    config_path=str(tmp_path / "h.yaml")),
                              exec_fn=lambda *a: 0)
    assert "--model" in str(e.value)


def test_launch_hermes_prints_context_note(monkeypatch, tmp_path, capsys):
    _fake_probe(monkeypatch)
    launch._launch_hermes(_args(harness="hermes",
                                config_path=str(tmp_path / "h.yaml"),
                                config_only=True),
                          exec_fn=lambda *a: 0)
    assert "64k" in capsys.readouterr().out


# goose (Block)
def test_build_goose_env_shape():
    env = launch.build_goose_env("http://127.0.0.1:8080/v1",
                                 default_model="qwen3.6-27b")
    assert env == {
        "GOOSE_PROVIDER": "openai",
        "GOOSE_MODEL": "qwen3.6-27b",
        "OPENAI_HOST": "http://127.0.0.1:8080",        # no /v1 - path goes below
        "OPENAI_BASE_PATH": "v1/chat/completions",
        "OPENAI_API_KEY": "gmlx",
    }


def test_launch_goose_execs_session_with_env(monkeypatch, tmp_path):
    import yaml as _yaml
    _fake_probe(monkeypatch)
    cfg_path = tmp_path / "config.yaml"
    calls = {}

    def fake_exec(binary, argv, env):
        calls["binary"], calls["argv"], calls["env"] = binary, argv, env
        return 0

    rc = launch._launch_goose(_args(harness="goose", config_path=str(cfg_path)),
                              exec_fn=fake_exec)
    assert rc == 0
    assert calls["argv"] == ["goose", "session"]
    assert calls["env"]["GOOSE_PROVIDER"] == "openai"
    assert calls["env"]["GOOSE_MODEL"] == "qwen3.6-27b"
    cfg = _yaml.safe_load(cfg_path.read_text())
    assert cfg["OPENAI_HOST"] == "http://127.0.0.1:8080"


def test_launch_goose_merges_existing_config(monkeypatch, tmp_path):
    import yaml as _yaml
    _fake_probe(monkeypatch)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("extensions:\n  developer:\n    enabled: true\n")
    rc = launch._launch_goose(
        _args(harness="goose", config_path=str(cfg_path), config_only=True),
        exec_fn=lambda *a: 0)
    assert rc == 0
    cfg = _yaml.safe_load(cfg_path.read_text())
    assert cfg["extensions"]["developer"]["enabled"] is True   # preserved
    assert cfg["GOOSE_MODEL"] == "qwen3.6-27b"


def test_launch_goose_requires_default_model(monkeypatch, tmp_path):
    monkeypatch.setattr(launch, "probe_models",
                        lambda base, api_key=None: [{"id": "a"}, {"id": "b"}])
    monkeypatch.setattr(launch.shutil, "which", lambda name: f"/usr/bin/{name}")
    with pytest.raises(launch.LaunchError) as e:
        launch._launch_goose(_args(harness="goose",
                                   config_path=str(tmp_path / "c.yaml")),
                             exec_fn=lambda *a: 0)
    assert "--model" in str(e.value)


def test_launch_goose_config_only_prints_env_line(monkeypatch, tmp_path, capsys):
    _fake_probe(monkeypatch)
    rc = launch._launch_goose(
        _args(harness="goose", config_path=str(tmp_path / "c.yaml"),
              config_only=True),
        exec_fn=lambda *a: 0)
    assert rc == 0
    out = capsys.readouterr().out
    assert "GOOSE_MODEL=qwen3.6-27b" in out and "goose session" in out
    assert "OPENAI_API_KEY=" in out                          # env line carries the key


def test_launch_goose_preserves_existing_openai_api_key(monkeypatch, tmp_path):
    # A real credential in the user's config.yaml must survive a launch; our
    # (placeholder) key travels only in the exec environment.
    import yaml as _yaml
    _fake_probe(monkeypatch)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("OPENAI_API_KEY: sk-real-openai-key\n")
    calls = {}

    def fake_exec(binary, argv, env):
        calls["env"] = env
        return 0

    rc = launch._launch_goose(_args(harness="goose", config_path=str(cfg_path)),
                              exec_fn=fake_exec)
    assert rc == 0
    cfg = _yaml.safe_load(cfg_path.read_text())
    assert cfg["OPENAI_API_KEY"] == "sk-real-openai-key"     # not clobbered
    assert calls["env"]["OPENAI_API_KEY"] == "gmlx"      # placeholder via env


def test_launch_goose_never_writes_api_key_to_yaml(monkeypatch, tmp_path):
    import yaml as _yaml
    _fake_probe(monkeypatch)
    cfg_path = tmp_path / "config.yaml"
    calls = {}

    def fake_exec(binary, argv, env):
        calls["env"] = env
        return 0

    rc = launch._launch_goose(_args(harness="goose", config_path=str(cfg_path),
                                    api_key="sk-served"),
                              exec_fn=fake_exec)
    assert rc == 0
    cfg = _yaml.safe_load(cfg_path.read_text())
    assert "OPENAI_API_KEY" not in cfg                       # secret stays out of YAML
    assert cfg["GOOSE_MODEL"] == "qwen3.6-27b"               # pointers persisted
    assert calls["env"]["OPENAI_API_KEY"] == "sk-served"     # real key via env


# claude-code

def test_build_claude_code_env_shape():
    env = launch.build_claude_code_env("http://127.0.0.1:8080/v1",
                                       default_model="qwen3.6-27b")
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8080"  # root, no /v1
    assert env["ANTHROPIC_MODEL"] == "qwen3.6-27b"
    assert env["ANTHROPIC_SMALL_FAST_MODEL"] == "qwen3.6-27b"
    assert env["ANTHROPIC_AUTH_TOKEN"]                           # never empty


def test_build_claude_code_env_carries_api_key():
    env = launch.build_claude_code_env("http://127.0.0.1:8080/v1",
                                       default_model="m", api_key="sk-local")
    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-local"


def test_launch_claude_code_requires_default_model(monkeypatch):
    monkeypatch.setattr(launch, "probe_models",
                        lambda base, api_key=None: [{"id": "a"}, {"id": "b"}])
    monkeypatch.setattr(launch.shutil, "which", lambda name: f"/usr/bin/{name}")
    with pytest.raises(launch.LaunchError) as e:
        launch._launch_claude_code(_args(harness="claude-code"),
                                   exec_fn=lambda *a: 0)
    assert "default model" in str(e.value)


def test_launch_claude_code_execs_with_env(monkeypatch, capsys):
    _fake_probe(monkeypatch)
    calls = {}

    def fake_exec(binary, argv, env):
        calls["argv"], calls["env"] = argv, env
        return 0

    rc = launch._launch_claude_code(
        _args(harness="claude-code", model="qwen3.6-27b"), exec_fn=fake_exec)
    assert rc == 0
    assert calls["argv"] == ["claude"]
    assert calls["env"]["ANTHROPIC_MODEL"] == "qwen3.6-27b"
    assert calls["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8080"


def test_launch_claude_code_env_drops_anthropic_api_key(monkeypatch):
    # An inherited real ANTHROPIC_API_KEY would outrank our ANTHROPIC_AUTH_TOKEN.
    _fake_probe(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real")
    calls = {}

    def fake_exec(binary, argv, env):
        calls["env"] = env
        return 0

    rc = launch._launch_claude_code(_args(harness="claude-code"), exec_fn=fake_exec)
    assert rc == 0
    assert "ANTHROPIC_API_KEY" not in calls["env"]
    assert calls["env"]["ANTHROPIC_AUTH_TOKEN"] == "gmlx"


def test_launch_claude_code_missing_binary_errors(monkeypatch):
    monkeypatch.setattr(launch, "probe_models", lambda base, api_key=None: _models())
    monkeypatch.setattr(launch.shutil, "which", lambda name: None)
    rc = launch.cmd_launch(["claude-code"])
    assert rc == 1                                           # no auto-install, clean exit


def test_launch_claude_code_config_only_prints_env(monkeypatch, capsys):
    _fake_probe(monkeypatch)
    execd = []
    rc = launch._launch_claude_code(
        _args(harness="claude-code", config_only=True),
        exec_fn=lambda *a: execd.append(a) or 0)
    assert rc == 0 and execd == []                           # never exec'd
    out = capsys.readouterr().out
    assert "ANTHROPIC_MODEL=qwen3.6-27b" in out
    assert "ANTHROPIC_BASE_URL=http://127.0.0.1:8080" in out
    assert out.rstrip().endswith("claude")                   # the run command


# aichat (sigoden/aichat) - chat-REPL + tools, AICHAT_CONFIG_DIR clean injection
def test_build_aichat_config_shape():
    cfg = launch.build_aichat_config("http://127.0.0.1:8080/v1", _models(),
                                     default_model="qwen3.6-27b")
    client = cfg["clients"][0]
    assert client["type"] == "openai-compatible"
    assert client["name"] == "gmlx"
    assert client["api_base"] == "http://127.0.0.1:8080/v1"
    assert "api_key" not in client                           # local, unauthenticated
    assert [m["name"] for m in client["models"]] == ["qwen3.6-27b", "gemma-e2b", "coder"]
    assert all(m["supports_function_calling"] for m in client["models"])  # tools on
    assert cfg["model"] == "gmlx:qwen3.6-27b"            # <client>:<id> default


def test_build_aichat_config_no_default_omits_model():
    cfg = launch.build_aichat_config("http://x/v1", _models())
    assert "model" not in cfg


def test_build_aichat_config_api_key():
    cfg = launch.build_aichat_config("http://x/v1", _models(), api_key="sk-x")
    assert cfg["clients"][0]["api_key"] == "sk-x"


def test_launch_aichat_writes_config_and_execs(monkeypatch, tmp_path):
    import yaml as _yaml
    _fake_probe(monkeypatch)
    calls = {}

    def fake_exec(binary, argv, env):
        calls["binary"], calls["argv"], calls["env"] = binary, argv, env
        return 0

    rc = launch._launch_aichat(_args(harness="aichat", config_path=str(tmp_path)),
                               exec_fn=fake_exec)
    assert rc == 0
    assert calls["binary"] == "/usr/bin/aichat" and calls["argv"] == ["aichat"]
    assert calls["env"]["AICHAT_CONFIG_DIR"] == str(tmp_path)  # injected, not user file
    cfg = _yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert cfg["model"] == "gmlx:qwen3.6-27b"
    assert cfg["clients"][0]["api_base"] == "http://127.0.0.1:8080/v1"


def test_launch_aichat_config_only_does_not_exec(monkeypatch, tmp_path, capsys):
    _fake_probe(monkeypatch)
    execd = []
    rc = launch._launch_aichat(_args(harness="aichat", config_path=str(tmp_path),
                                     config_only=True),
                               exec_fn=lambda *a: execd.append(a) or 0)
    assert rc == 0 and execd == []
    assert (tmp_path / "config.yaml").exists()
    assert "AICHAT_CONFIG_DIR=" in capsys.readouterr().out


def test_launch_aichat_missing_binary_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(launch, "probe_models", lambda base, api_key=None: _models())
    monkeypatch.setattr(launch.shutil, "which", lambda name: None)
    rc = launch.cmd_launch(["aichat", "--config-path", str(tmp_path)])
    assert rc == 1                                           # no auto-install, clean exit


# elia (darrenburns/elia) - chat TUI, XDG_CONFIG_HOME clean injection, TOML config
def test_build_elia_config_shape():
    doc = tomllib.loads(launch.build_elia_config(
        "http://127.0.0.1:8080/v1", _models(), default_model="qwen3.6-27b"))
    assert doc["default_model"] == "gmlx/qwen3.6-27b"    # the id lookup key
    by_id = {m["id"]: m for m in doc["models"]}
    assert set(by_id) == {"gmlx/qwen3.6-27b", "gmlx/gemma-e2b",
                          "gmlx/coder"}
    m = by_id["gmlx/qwen3.6-27b"]
    assert m["name"] == "openai/qwen3.6-27b"                 # litellm routing prefix
    assert m["api_base"] == "http://127.0.0.1:8080/v1"
    assert m["api_key"] == "gmlx"                        # placeholder, non-empty


def test_build_elia_config_no_default_omits_default_model():
    doc = tomllib.loads(launch.build_elia_config("http://x/v1", _models()))
    assert "default_model" not in doc
    assert len(doc["models"]) == 3


def test_build_elia_config_api_key():
    doc = tomllib.loads(launch.build_elia_config("http://x/v1", _models(),
                                                 api_key="sk-x"))
    assert all(m["api_key"] == "sk-x" for m in doc["models"])


def test_launch_elia_writes_config_and_execs(monkeypatch, tmp_path):
    _fake_probe(monkeypatch)
    calls = {}

    def fake_exec(binary, argv, env):
        calls["binary"], calls["argv"], calls["env"] = binary, argv, env
        return 0

    rc = launch._launch_elia(_args(harness="elia", config_path=str(tmp_path)),
                             exec_fn=fake_exec)
    assert rc == 0
    assert calls["binary"] == "/usr/bin/elia"
    assert calls["argv"] == ["elia", "-m", "gmlx/qwen3.6-27b"]  # default via -m
    assert calls["env"]["XDG_CONFIG_HOME"] == str(tmp_path)  # injected, not user file
    doc = tomllib.loads((tmp_path / "elia" / "config.toml").read_text())
    assert doc["default_model"] == "gmlx/qwen3.6-27b"


def test_launch_elia_config_only_does_not_exec(monkeypatch, tmp_path, capsys):
    _fake_probe(monkeypatch)
    execd = []
    rc = launch._launch_elia(_args(harness="elia", config_path=str(tmp_path),
                                   config_only=True),
                             exec_fn=lambda *a: execd.append(a) or 0)
    assert rc == 0 and execd == []
    assert (tmp_path / "elia" / "config.toml").exists()
    out = capsys.readouterr().out
    assert "XDG_CONFIG_HOME=" in out and "elia -m gmlx/qwen3.6-27b" in out


def test_launch_elia_missing_binary_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(launch, "probe_models", lambda base, api_key=None: _models())
    monkeypatch.setattr(launch.shutil, "which", lambda name: None)
    rc = launch.cmd_launch(["elia", "--config-path", str(tmp_path)])
    assert rc == 1


# open-webui (Open WebUI) - browser app, pure env injection + on-disk DATA_DIR
def test_build_open_webui_env_shape():
    env = launch.build_open_webui_env(
        "http://127.0.0.1:8080/v1", default_model="qwen3.6-27b",
        port=3000, data_dir="/home/u/.open-webui")
    assert env["OPENAI_API_BASE_URL"] == "http://127.0.0.1:8080/v1"   # keeps /v1
    assert env["OPENAI_API_KEY"] == "gmlx"                        # placeholder
    assert env["ENABLE_OLLAMA_API"] == "false"
    assert env["PORT"] == "3000"
    assert env["DATA_DIR"] == "/home/u/.open-webui"
    assert env["DEFAULT_MODELS"] == "qwen3.6-27b"
    # RAG points back at our server (lazy openai engine) - no boot-time embedder
    # download, and clean boot even with no cached embedder. HF_HUB_OFFLINE is NOT
    # set: it would hard-crash Open WebUI boot on a host without a cached embedder.
    assert env["RAG_EMBEDDING_ENGINE"] == "openai"
    assert env["RAG_OPENAI_API_BASE_URL"] == "http://127.0.0.1:8080/v1"
    assert env["RAG_OPENAI_API_KEY"] == "gmlx"
    assert env["RAG_EMBEDDING_MODEL"] == "text-embedding-3-small"  # our /v1/embeddings id
    assert env["RAG_EMBEDDING_MODEL_AUTO_UPDATE"] == "false"
    assert "HF_HUB_OFFLINE" not in env


def test_build_open_webui_env_no_default_omits_default_models():
    env = launch.build_open_webui_env(
        "http://x/v1", port=3000, data_dir="/d")          # default_model None
    assert "DEFAULT_MODELS" not in env


def test_build_open_webui_env_carries_api_key():
    env = launch.build_open_webui_env(
        "http://x/v1", api_key="sk-local", port=3000, data_dir="/d")
    assert env["OPENAI_API_KEY"] == "sk-local"
    assert env["RAG_OPENAI_API_KEY"] == "sk-local"        # RAG reuses the same key


def test_build_open_webui_env_no_audio_by_default():
    # Chat-only server: leave Open WebUI's built-in browser STT/TTS untouched.
    env = launch.build_open_webui_env("http://x/v1", port=3000, data_dir="/d")
    assert not any(k.startswith("AUDIO_") for k in env)


def test_build_open_webui_env_wires_audio_when_capable():
    env = launch.build_open_webui_env(
        "http://127.0.0.1:8080/v1", api_key="sk-local", port=3000,
        data_dir="/d", stt=True, tts=True)
    # STT -> /v1/audio/transcriptions
    assert env["AUDIO_STT_ENGINE"] == "openai"
    assert env["AUDIO_STT_OPENAI_API_BASE_URL"] == "http://127.0.0.1:8080/v1"
    assert env["AUDIO_STT_OPENAI_API_KEY"] == "sk-local"
    assert env["AUDIO_STT_MODEL"] == "whisper-1"
    # TTS -> /v1/audio/speech, with a Kokoro-valid voice (Open WebUI's "alloy" default
    # is an OpenAI voice Kokoro rejects).
    assert env["AUDIO_TTS_ENGINE"] == "openai"
    assert env["AUDIO_TTS_OPENAI_API_BASE_URL"] == "http://127.0.0.1:8080/v1"
    assert env["AUDIO_TTS_OPENAI_API_KEY"] == "sk-local"
    assert env["AUDIO_TTS_MODEL"] == "tts-1"
    assert env["AUDIO_TTS_VOICE"] == "af_heart"


def test_build_open_webui_env_audio_independent_toggles():
    stt_only = launch.build_open_webui_env(
        "http://x/v1", port=3000, data_dir="/d", stt=True)
    assert stt_only["AUDIO_STT_ENGINE"] == "openai"
    assert not any(k.startswith("AUDIO_TTS_") for k in stt_only)
    tts_only = launch.build_open_webui_env(
        "http://x/v1", port=3000, data_dir="/d", tts=True)
    assert tts_only["AUDIO_TTS_ENGINE"] == "openai"
    assert not any(k.startswith("AUDIO_STT_") for k in tts_only)


def test_launch_open_webui_detects_audio_from_models(monkeypatch, tmp_path):
    # Server advertises STT + TTS capability via /v1/models markers; the harness must
    # pick them up and route Open WebUI's audio engines at the server.
    audio_models = _models() + [
        {"id": "whisper-1", "stt": True},
        {"id": "tts-1", "tts": True},
    ]
    monkeypatch.setattr(launch.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(launch, "probe_models",
                        lambda base, api_key=None: audio_models)
    calls = {}
    rc = launch._launch_open_webui(
        _args(harness="open-webui", config_path=str(tmp_path)),
        exec_fn=lambda b, a, e: calls.update(env=e) or 0)
    assert rc == 0
    env = calls["env"]
    assert env["AUDIO_STT_ENGINE"] == "openai"
    assert env["AUDIO_TTS_ENGINE"] == "openai"
    assert env["AUDIO_TTS_VOICE"] == "af_heart"


def test_launch_open_webui_no_audio_when_chat_only(monkeypatch, tmp_path):
    _fake_probe(monkeypatch)            # _models() carries no stt/tts markers
    calls = {}
    rc = launch._launch_open_webui(
        _args(harness="open-webui", config_path=str(tmp_path)),
        exec_fn=lambda b, a, e: calls.update(env=e) or 0)
    assert rc == 0
    assert not any(k.startswith("AUDIO_") for k in calls["env"])


def test_launch_open_webui_execs_with_env(monkeypatch, tmp_path):
    _fake_probe(monkeypatch)
    calls = {}

    def fake_exec(binary, argv, env):
        calls["binary"], calls["argv"], calls["env"] = binary, argv, env
        return 0

    rc = launch._launch_open_webui(
        _args(harness="open-webui", config_path=str(tmp_path)), exec_fn=fake_exec)
    assert rc == 0
    assert calls["binary"] == "/usr/bin/open-webui"
    # `open-webui serve` ignores the PORT env var - the bind port MUST be the --port
    # CLI option, else the UI binds 8080 and collides with the gmlx server.
    assert calls["argv"] == ["open-webui", "serve", "--port", "3000"]
    env = calls["env"]
    assert env["OPENAI_API_BASE_URL"] == "http://127.0.0.1:8080/v1"
    assert env["PORT"] == "3000"                          # kept for self-URL construction
    assert env["DATA_DIR"] == str(tmp_path)               # chat history on host fs
    assert env["DEFAULT_MODELS"] == "qwen3.6-27b"


def test_launch_open_webui_avoids_server_port_collision(monkeypatch, tmp_path):
    _fake_probe(monkeypatch)
    calls = {}
    # Server itself on 3000 -> the UI must move off it, on the command line (the env
    # PORT is ignored by `open-webui serve`).
    rc = launch._launch_open_webui(
        _args(harness="open-webui", port=3000, base_url="http://127.0.0.1:3000/v1",
              config_path=str(tmp_path)),
        exec_fn=lambda b, a, e: calls.update(argv=a, env=e) or 0)
    assert rc == 0
    assert calls["argv"] == ["open-webui", "serve", "--port", "3001"]
    assert calls["env"]["PORT"] == "3001"


def test_launch_open_webui_drops_inherited_plural_openai_vars(monkeypatch, tmp_path):
    _fake_probe(monkeypatch)
    monkeypatch.setenv("OPENAI_API_BASE_URLS", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_API_KEYS", "sk-real")
    calls = {}
    rc = launch._launch_open_webui(
        _args(harness="open-webui", config_path=str(tmp_path)),
        exec_fn=lambda b, a, e: calls.update(env=e) or 0)
    assert rc == 0
    assert "OPENAI_API_BASE_URLS" not in calls["env"]     # our single endpoint wins
    assert "OPENAI_API_KEYS" not in calls["env"]


def test_launch_open_webui_config_only_does_not_exec(monkeypatch, tmp_path, capsys):
    _fake_probe(monkeypatch)
    execd = []
    rc = launch._launch_open_webui(
        _args(harness="open-webui", config_path=str(tmp_path), config_only=True),
        exec_fn=lambda *a: execd.append(a) or 0)
    assert rc == 0 and execd == []
    out = capsys.readouterr().out
    assert "OPENAI_API_BASE_URL=http://127.0.0.1:8080/v1" in out
    assert out.rstrip().endswith("open-webui serve --port 3000")


def test_launch_open_webui_missing_binary_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(launch, "probe_models", lambda base, api_key=None: _models())
    monkeypatch.setattr(launch.shutil, "which", lambda name: None)
    rc = launch.cmd_launch(["open-webui", "--config-path", str(tmp_path)])
    assert rc == 1                                        # no auto-install, clean exit


def test_probe_models_named_api_key_hint_on_401(monkeypatch):
    import urllib.error

    def fake_get(url, timeout=5.0, headers=None):
        if url.endswith("/health"):
            return {"status": "ok"}
        raise urllib.error.HTTPError(url, 401, "unauthorized", {}, None)

    monkeypatch.setattr(launch, "_http_get_json", fake_get)
    with pytest.raises(launch.LaunchError) as e:
        launch.probe_models("http://127.0.0.1:8080/v1")
    assert "--api-key" in str(e.value)


# start-if-down orchestration (auto-start a server when none is reachable)

class _FakeModel:
    def __init__(self, path="", pin=False):
        self.path, self.pin = path, pin


class _FakeDefaults:
    def __init__(self, model=None):
        self.model = model


class _FakeCfg:
    def __init__(self, *, host="127.0.0.1", port=8080, api_key=None, menubar=True,
                 models=None, model_dirs=None, default_model=None):
        self.host, self.port, self.api_key, self.menubar = host, port, api_key, menubar
        self.models = models or {}
        self.model_dirs = model_dirs or []
        self.defaults = _FakeDefaults(default_model)


class _FakePopen:
    """Reports 'alive' (poll() is None) for the first ``alive_polls`` calls, then exits."""
    def __init__(self, *, alive_polls=10**6, returncode=0):
        self._polls, self._alive = 0, alive_polls
        self.returncode, self.pid = returncode, 4242

    def poll(self):
        self._polls += 1
        return None if self._polls <= self._alive else self.returncode


def _down(monkeypatch):
    monkeypatch.setattr(launch, "_server_ready", lambda base, api_key=None: False)


def _no_real_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)


def _alive_log(monkeypatch, **kw):
    monkeypatch.setattr(lifecycle, "start_background_nowait",
                        lambda *a, **k: (_FakePopen(**kw), Path("/tmp/x.log")))
    monkeypatch.setattr(lifecycle, "gui_session_available", lambda: False)


# _preload_descr (pin -> default -> sole -> none; size best-effort)
def test_preload_descr_pin():
    cfg = _FakeCfg(models={"a": _FakeModel(pin=True), "b": _FakeModel()})
    assert launch._preload_descr(cfg)[0] == "a"


def test_preload_descr_default_marker():
    cfg = _FakeCfg(models={"a": _FakeModel(), "b": _FakeModel()}, default_model="b")
    assert launch._preload_descr(cfg)[0] == "b"


def test_preload_descr_sole_model():
    cfg = _FakeCfg(models={"only": _FakeModel()})
    assert launch._preload_descr(cfg)[0] == "only"


def test_preload_descr_none_when_no_preload():
    cfg = _FakeCfg(models={"a": _FakeModel(), "b": _FakeModel()})
    assert launch._preload_descr(cfg) == (None, None)


def test_preload_descr_size_when_file_exists(tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"x" * 2048)
    cfg = _FakeCfg(models={"a": _FakeModel(path=str(f))}, default_model="a")
    pid, label = launch._preload_descr(cfg)
    assert pid == "a" and "KB" in label                     # size rendered


def test_preload_descr_size_absent_when_missing(tmp_path):
    cfg = _FakeCfg(models={"a": _FakeModel(path=str(tmp_path / "missing.gguf"))},
                   default_model="a")
    assert launch._preload_descr(cfg) == ("a", "a")         # no size, just the id


# _ensure_server (endpoint resolution + start/guide decisions)
def test_ensure_server_up_skips_config(monkeypatch):
    # Reachable at the default endpoint: today's behavior - no config read, no spawn.
    read = []
    monkeypatch.setattr(launch, "_discover_config",
                        lambda: read.append(1) or (None, None))
    a = _args(base_url=None, host=None, port=None)
    assert launch._ensure_server(a) is None
    assert read == []                                       # config never consulted
    assert (a.base_url, a.host, a.port) == ("http://127.0.0.1:8080/v1",
                                            "127.0.0.1", 8080)


def test_ensure_server_bare_uses_managed_target(monkeypatch):
    # A managed server on a non-8080 port wins over the hardcoded default, so
    # launch never binds a harness to whatever happens to answer on 8080.
    monkeypatch.setattr(lifecycle, "auto_target", lambda h, p: ("127.0.0.1", 9345))
    seen = []
    monkeypatch.setattr(launch, "_server_ready",
                        lambda base, key=None: seen.append(base) or True)
    a = _args(base_url=None, host=None, port=None)
    assert launch._ensure_server(a) is None
    assert a.port == 9345
    assert seen[0] == "http://127.0.0.1:9345/v1"


def test_ensure_server_explicit_port_skips_auto_target(monkeypatch):
    # An explicit --port must never be second-guessed by runfile/config state.
    monkeypatch.setattr(lifecycle, "auto_target",
                        lambda h, p: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(launch, "_server_ready", lambda base, key=None: True)
    a = _args(base_url=None, host=None, port=7777)
    assert launch._ensure_server(a) is None
    assert a.base_url == "http://127.0.0.1:7777/v1"


def test_ensure_server_up_ignores_stray_config_api_key(monkeypatch):
    # A stray cwd config must not inject an api_key on the (reachable) up path.
    monkeypatch.setattr(launch, "_discover_config",
                        lambda: (_FakeCfg(api_key="sekret"), "./gmlx.yaml"))
    a = _args(base_url=None, host=None, port=None, api_key=None)
    launch._ensure_server(a)
    assert a.api_key is None


def test_ensure_server_explicit_base_url_down_no_spawn(monkeypatch):
    _down(monkeypatch)
    calls = []
    monkeypatch.setattr(lifecycle, "start_background_nowait",
                        lambda *a, **k: calls.append(1) or None)
    a = _args(base_url="http://127.0.0.1:8080/v1", host=None, port=None)
    assert launch._ensure_server(a) is None                 # proceed; harness raises the usual error
    assert calls == []                                      # never auto-started


def test_ensure_server_no_config_guides_init(monkeypatch, capsys):
    _down(monkeypatch)
    monkeypatch.setattr(launch, "_discover_config", lambda: (None, None))
    a = _args(base_url=None, host=None, port=None)
    assert launch._ensure_server(a) == 2
    assert "gmlx init" in capsys.readouterr().err


def test_ensure_server_malformed_config(monkeypatch, capsys):
    _down(monkeypatch)
    monkeypatch.setattr(launch, "_discover_config", lambda: (None, "/x/gmlx.yaml"))
    a = _args(base_url=None, host=None, port=None)
    assert launch._ensure_server(a) == 2
    assert "won't load" in capsys.readouterr().err


def test_ensure_server_no_start_down_with_config(monkeypatch, capsys):
    _down(monkeypatch)
    monkeypatch.setattr(launch, "_discover_config", lambda: (_FakeCfg(), "/x/c.yaml"))
    spawned = []
    monkeypatch.setattr(lifecycle, "start_background_nowait",
                        lambda *a, **k: spawned.append(1) or None)
    a = _args(base_url=None, host=None, port=None, no_start=True)
    assert launch._ensure_server(a) == 1
    assert spawned == [] and "--no-start" in capsys.readouterr().err


def test_ensure_server_no_start_down_no_config_guides(monkeypatch, capsys):
    _down(monkeypatch)
    monkeypatch.setattr(launch, "_discover_config", lambda: (None, None))
    a = _args(base_url=None, host=None, port=None, no_start=True)
    assert launch._ensure_server(a) == 2                    # no-config guidance precedes --no-start
    assert "gmlx init" in capsys.readouterr().err


def test_ensure_server_config_endpoint_already_up(monkeypatch):
    # Default :8080 down, but the config points at :9000 which IS up - resolve, no spawn.
    monkeypatch.setattr(launch, "_server_ready",
                        lambda base, api_key=None: base == "http://127.0.0.1:9000/v1")
    monkeypatch.setattr(launch, "_discover_config",
                        lambda: (_FakeCfg(port=9000), "/x/c.yaml"))
    spawned = []
    monkeypatch.setattr(lifecycle, "start_background_nowait",
                        lambda *a, **k: spawned.append(1) or None)
    a = _args(base_url=None, host=None, port=None)
    assert launch._ensure_server(a) is None
    assert spawned == [] and a.base_url == "http://127.0.0.1:9000/v1"


def test_ensure_server_launchd_restarting(monkeypatch, capsys):
    _down(monkeypatch)
    monkeypatch.setattr(launch, "_discover_config", lambda: (_FakeCfg(), "/x/c.yaml"))
    monkeypatch.setattr(lifecycle, "read_run", lambda h, p: {"managed_by": "launchd"})
    spawned = []
    monkeypatch.setattr(lifecycle, "start_background_nowait",
                        lambda *a, **k: spawned.append(1) or None)
    a = _args(base_url=None, host=None, port=None)
    assert launch._ensure_server(a) == 1
    assert spawned == [] and "launchd" in capsys.readouterr().err


def test_ensure_server_autostart_uses_config_endpoint(monkeypatch):
    _no_real_sleep(monkeypatch)
    states = iter([False, False, True])     # step1 down, step4 down, autostart poll ready
    monkeypatch.setattr(launch, "_server_ready", lambda base, api_key=None: next(states))
    cfg = _FakeCfg(host="127.0.0.1", port=9000, api_key="cfgkey",
                   models={"a": _FakeModel()}, default_model="a")
    monkeypatch.setattr(launch, "_discover_config", lambda: (cfg, "/abs/c.yaml"))
    monkeypatch.setattr(lifecycle, "read_run", lambda h, p: None)
    monkeypatch.setattr(lifecycle, "gui_session_available", lambda: False)
    seen = {}

    def fake_spawn(serve_args, *, host, port, config_abspath, api_key):
        seen.update(serve_args=serve_args, host=host, port=port, api_key=api_key)
        return (_FakePopen(), Path("/tmp/x.log"))

    monkeypatch.setattr(lifecycle, "start_background_nowait", fake_spawn)
    a = _args(base_url=None, host=None, port=None, api_key=None)
    assert launch._ensure_server(a) is None
    assert seen["serve_args"] == ["--config", "/abs/c.yaml"]
    assert seen["host"] == "127.0.0.1" and seen["port"] == 9000
    assert seen["api_key"] == "cfgkey"
    assert a.base_url == "http://127.0.0.1:9000/v1"


def test_ensure_server_autostart_no_preload_prints_cold_note(monkeypatch, capsys):
    _no_real_sleep(monkeypatch)
    states = iter([False, False, True])
    monkeypatch.setattr(launch, "_server_ready", lambda base, api_key=None: next(states))
    cfg = _FakeCfg(models={"a": _FakeModel(), "b": _FakeModel()})   # 2 models, no default -> no preload
    monkeypatch.setattr(launch, "_discover_config", lambda: (cfg, "/x/c.yaml"))
    monkeypatch.setattr(lifecycle, "read_run", lambda h, p: None)
    _alive_log(monkeypatch)
    a = _args(base_url=None, host=None, port=None)
    assert launch._ensure_server(a) is None
    assert "no model is preloaded" in capsys.readouterr().err


def test_ensure_server_autostart_preload_hot_no_cold_note(monkeypatch, capsys):
    _no_real_sleep(monkeypatch)
    states = iter([False, False, True])
    monkeypatch.setattr(launch, "_server_ready", lambda base, api_key=None: next(states))
    cfg = _FakeCfg(models={"a": _FakeModel()}, default_model="a")
    monkeypatch.setattr(launch, "_discover_config", lambda: (cfg, "/x/c.yaml"))
    monkeypatch.setattr(lifecycle, "read_run", lambda h, p: None)
    _alive_log(monkeypatch)
    a = _args(base_url=None, host=None, port=None)
    assert launch._ensure_server(a) is None
    assert "no model is preloaded" not in capsys.readouterr().err


# _autostart (spawn + spinner-poll outcomes)
def _cfg_one(path=""):
    return _FakeCfg(models={"m": _FakeModel(path=path)}, default_model="m")


def _call_autostart(start_timeout=0.0, config_only=False):
    return launch._autostart(
        base="http://127.0.0.1:8080/v1", host="127.0.0.1", port=8080, api_key=None,
        cfg=_cfg_one(), cfg_path="/x/c.yaml", start_timeout=start_timeout,
        config_only=config_only)


def test_autostart_ready_returns_preload_id(monkeypatch, capsys):
    _no_real_sleep(monkeypatch)
    seq = iter([False, False, True])
    monkeypatch.setattr(launch, "_server_ready", lambda base, api_key=None: next(seq))
    _alive_log(monkeypatch)
    assert _call_autostart() == (0, True, "m")
    assert "starting server - loading m" in capsys.readouterr().err   # spinner names the model


def test_autostart_child_dies_returns_one(monkeypatch, capsys):
    _no_real_sleep(monkeypatch)
    monkeypatch.setattr(launch, "_server_ready", lambda base, api_key=None: False)
    monkeypatch.setattr(lifecycle, "start_background_nowait",
                        lambda *a, **k: (_FakePopen(alive_polls=0, returncode=1),
                                         Path("/tmp/x.log")))
    monkeypatch.setattr(lifecycle, "_log_tail", lambda log, n: "boom\n")
    rc, ready, _ = _call_autostart()
    assert (rc, ready) == (1, False)
    assert "before it was ready" in capsys.readouterr().err


def test_autostart_port_in_use_names_the_port(monkeypatch, capsys):
    _no_real_sleep(monkeypatch)
    monkeypatch.setattr(launch, "_server_ready", lambda base, api_key=None: False)
    monkeypatch.setattr(lifecycle, "start_background_nowait",
                        lambda *a, **k: (_FakePopen(alive_polls=0, returncode=1),
                                         Path("/tmp/x.log")))
    monkeypatch.setattr(lifecycle, "_log_tail",
                        lambda log, n: "bind failed: address already in use\n")
    rc, ready, _ = _call_autostart()
    assert (rc, ready) == (1, False)
    err = capsys.readouterr().err
    assert "[launch] port 8080 on 127.0.0.1 is already in use" in err
    assert "before it was ready" not in err


def test_autostart_timeout_cap_returns_one(monkeypatch, capsys):
    _no_real_sleep(monkeypatch)
    monkeypatch.setattr(launch, "_server_ready", lambda base, api_key=None: False)
    _alive_log(monkeypatch)
    rc, ready, _ = _call_autostart(start_timeout=1e-9)      # positive cap, never ready
    assert (rc, ready) == (1, False)
    assert "still starting" in capsys.readouterr().err


def test_autostart_unlimited_waits_then_execs(monkeypatch):
    # start_timeout=0 must NOT give up early: a couple of False polls then True still execs.
    _no_real_sleep(monkeypatch)
    seq = iter([False, False, False, True])
    monkeypatch.setattr(launch, "_server_ready", lambda base, api_key=None: next(seq))
    _alive_log(monkeypatch)
    assert _call_autostart(start_timeout=0.0)[:2] == (0, True)


def test_autostart_keyboard_interrupt_returns_130(monkeypatch, capsys):
    _no_real_sleep(monkeypatch)

    def boom(base, api_key=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(launch, "_server_ready", boom)
    _alive_log(monkeypatch)
    rc, ready, _ = _call_autostart()
    assert (rc, ready) == (130, False)
    assert "interrupted" in capsys.readouterr().err


def test_autostart_spawn_refused_but_up(monkeypatch):
    monkeypatch.setattr(lifecycle, "start_background_nowait", lambda *a, **k: None)
    monkeypatch.setattr(launch, "_server_ready", lambda base, api_key=None: True)
    assert _call_autostart()[:2] == (0, True)


def test_autostart_config_only_skips_menubar(monkeypatch):
    _no_real_sleep(monkeypatch)
    monkeypatch.setattr(launch, "_server_ready", lambda base, api_key=None: True)
    monkeypatch.setattr(lifecycle, "start_background_nowait",
                        lambda *a, **k: (_FakePopen(), Path("/tmp/x.log")))
    monkeypatch.setattr(lifecycle, "gui_session_available", lambda: True)
    raised = []
    monkeypatch.setattr(lifecycle, "start_menubar", lambda **k: raised.append(True))
    assert _call_autostart(config_only=True)[:2] == (0, True)
    assert raised == []                                     # menu bar not raised under config-only


def test_autostart_raises_menubar_when_interactive(monkeypatch):
    _no_real_sleep(monkeypatch)
    monkeypatch.setattr(launch, "_server_ready", lambda base, api_key=None: True)
    monkeypatch.setattr(lifecycle, "start_background_nowait",
                        lambda *a, **k: (_FakePopen(), Path("/tmp/x.log")))
    monkeypatch.setattr(lifecycle, "gui_session_available", lambda: True)
    raised = []
    monkeypatch.setattr(lifecycle, "start_menubar", lambda **k: raised.append(True))
    assert _call_autostart(config_only=False)[:2] == (0, True)
    assert raised == [True]                                  # the one machine-wide bar


def test_toml_basic_string_escapes_control_chars():
    from gmlx.launch import _toml_basic_string
    s = _toml_basic_string("k\nevil=true")
    assert "\n" not in s
    assert tomllib.loads(f"x = {s}")["x"] == "k\nevil=true"  # parse-back proof


def test_load_json_on_directory_is_launch_error(tmp_path):
    from gmlx.launch import LaunchError, _load_json
    with pytest.raises(LaunchError):
        _load_json(tmp_path)


def test_write_text_atomic_leaves_no_tmp(tmp_path):
    from gmlx.launch import _write_text_atomic
    p = tmp_path / "cfg.json"
    _write_text_atomic(p, "{}")
    assert p.read_text() == "{}"
    assert not list(tmp_path.glob("*.tmp"))
