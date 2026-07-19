#!/usr/bin/env python3
"""The macOS menu-bar monitor's testable core: the `poll` snapshot (incl. the
401->key-required distinction), the pure `build_menu_model` row description, and the
rumps import guard. No rumps and no live server - urllib is faked, so this runs on any
platform. The rumps `App` wiring itself is GUI-only and not unit-exercised."""
from __future__ import annotations

import os
import types
import urllib.error

from gmlx import menubar as mb  # noqa: E402


class _Resp:
    def __init__(self, payload):
        self._b = payload

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_urlopen(routes):
    """routes: {path-suffix: payload-bytes | HTTPError | URLError}. Matches on the
    request URL ending with the suffix."""
    def _open(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else req
        for suffix, val in routes.items():
            if url.endswith(suffix):
                if isinstance(val, Exception):
                    raise val
                return _Resp(val)
        raise urllib.error.URLError("no route")
    return _open


# poll
def test_poll_up_with_resident(monkeypatch):
    metrics = b'{"server": {"resident_models": [{"ids": ["qwen3"], ' \
              b'"footprint_bytes": 4100000000}]}}'
    monkeypatch.setattr(mb.urllib.request, "urlopen",
                        _fake_urlopen({"/health": b"{}", "/v1/metrics": metrics}))
    snap = mb.poll("http://127.0.0.1:8080")
    assert snap["reachable"] is True
    assert snap["auth_required"] is False
    assert snap["resident"][0]["ids"] == ["qwen3"]


def test_poll_down(monkeypatch):
    monkeypatch.setattr(mb.urllib.request, "urlopen",
                        _fake_urlopen({"/health": urllib.error.URLError("refused")}))
    snap = mb.poll("http://127.0.0.1:8080")
    assert snap["reachable"] is False and snap["resident"] == []


def test_poll_401_is_up_key_required(monkeypatch):
    err = urllib.error.HTTPError("http://x/v1/metrics", 401, "unauth", {}, None)
    monkeypatch.setattr(mb.urllib.request, "urlopen",
                        _fake_urlopen({"/health": b"{}", "/v1/metrics": err}))
    snap = mb.poll("http://127.0.0.1:8080")
    assert snap["reachable"] is True          # /health answered -> up
    assert snap["auth_required"] is True       # 401 on metrics -> key required, not down


# build_menu_model
def _snap(**kw):
    base = dict(url="http://127.0.0.1:8080", reachable=True, auth_required=False,
                resident=[], error=None)
    base.update(kw)
    return base


def test_menu_down_offers_no_controls():
    m = mb.build_menu_model(_snap(reachable=False), None)
    assert m["state"] == "down" and m["title"].endswith("○")
    assert m["can_stop"] is False and m["can_reload"] is False
    assert m["can_start"] is False               # no runfile => nothing to relaunch from
    assert m["models_header"] is None            # header already says down - no models line


def test_menu_down_with_runfile_offers_start():
    # a crashed detached server leaves its runfile (with the relaunch argv) behind
    run = {"pid": 5, "port": 8080, "managed_by": "detach", "argv": ["x"],
           "log": "/tmp/s.log"}
    m = mb.build_menu_model(_snap(reachable=False), run)
    assert m["state"] == "down"
    assert m["can_start"] is True                # offer to (re)start it from the runfile
    assert m["can_restart"] is False             # "Start", not "Restart", while it's down
    assert m["can_stop"] is False                # nothing to stop
    assert m["models_header"] is None            # still no "Loaded models - server down"


def test_menu_up_lists_models_and_controls():
    snap = _snap(resident=[{"ids": ["qwen3"], "footprint_bytes": 4_100_000_000,
                            "pinned": True},
                           {"ids": ["gemma"], "footprint_bytes": 2_000_000_000}])
    run = {"pid": 99, "port": 8080, "managed_by": "detach", "argv": ["x"],
           "log": "/tmp/s.log"}
    m = mb.build_menu_model(snap, run)
    assert m["state"] == "up" and "pid 99" in m["header"]
    assert [x["id"] for x in m["models"]] == ["qwen3", "gemma"]
    # 4_100_000_000 bytes = 3.8 GiB - sizes render binary-GB everywhere now.
    assert "pinned" in m["models"][0]["label"] and "3.8 GB" in m["models"][0]["label"]
    assert m["can_reload"] and m["can_stop"] and m["can_restart"]
    assert m["log"] == "/tmp/s.log"


def test_menu_empty_resident_note():
    m = mb.build_menu_model(_snap(resident=[]),
                            {"pid": 1, "port": 8080, "managed_by": "detach"})
    assert m["models"] == [] and "No models resident" in m["models_header"]


def test_menu_key_required_state():
    m = mb.build_menu_model(_snap(auth_required=True),
                            {"pid": 1, "port": 8080, "managed_by": "detach"})
    assert m["state"] == "key-required" and m["title"].endswith("◐")
    assert m["can_reload"] is False
    assert "api-key" in m["models_header"]


def test_menu_launchd_restart_not_stop():
    run = {"pid": None, "port": 9000, "managed_by": "launchd",
           "label": "com.gmlx.server.x", "log": "/tmp/l.log"}
    m = mb.build_menu_model(_snap(url="http://127.0.0.1:9000"), run)
    assert m["can_stop"] is False           # launchd: stop is `service uninstall`
    assert m["can_restart"] is True         # but a kickstart restart is offered
    assert m["restart_kind"] == "launchd"


# api-key resolution: explicit wins; else the managed server's own config
def test_resolve_api_key_explicit_wins():
    assert mb.resolve_api_key("cli-key", {"api_key_set": True}) == "cli-key"


def test_resolve_api_key_skips_config_when_not_set():
    # api_key_set False => never even opens the config.
    assert mb.resolve_api_key(None, {"api_key_set": False,
                                     "config_abspath": "/nope.yaml"}) is None
    assert mb.resolve_api_key(None, None) is None


def test_resolve_api_key_reads_managed_config(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("server:\n  api_key: from-config\nmodels:\n  m:\n    path: /m.gguf\n")
    got = mb.resolve_api_key(None, {"api_key_set": True,
                                    "config_abspath": str(cfg)})
    assert got == "from-config"


# the rumps import guard (foreground GUI path): absent rumps => macOS-only msg, exit 2
def test_cmd_menubar_without_rumps(monkeypatch, capsys):
    import builtins
    real_import = builtins.__import__

    def _no_rumps(name, *a, **k):
        if name == "rumps":
            raise ImportError("no rumps")
        return real_import(name, *a, **k)

    from gmlx import lifecycle as lc
    monkeypatch.setattr(lc, "auto_target", lambda h, p: ("127.0.0.1", 8080))
    monkeypatch.setattr(lc, "write_menubar_run", lambda *a, **k: None)
    monkeypatch.setattr(builtins, "__import__", _no_rumps)
    rc = mb.cmd_menubar(["--foreground"])
    assert rc == 2
    assert "macOS" in capsys.readouterr().err


# default (no --foreground) detaches via the lifecycle helper - no rumps import here
def test_cmd_menubar_background_delegates(monkeypatch, capsys):
    seen = {}
    from gmlx import lifecycle as lc
    monkeypatch.setattr(lc, "gui_session_available", lambda: True)
    monkeypatch.setattr(lc, "start_menubar",
                        lambda extra=None: seen.update(extra=extra) or 0)
    rc = mb.cmd_menubar(["--interval", "9"])
    assert rc == 0
    # No explicit target -> the one bar tracks the primary; no --host/--port pinned.
    assert "--host" not in seen["extra"] and "--port" not in seen["extra"]
    assert seen["extra"][0] == "--interval" and float(seen["extra"][1]) == 9.0
    assert "monitoring the primary server" in capsys.readouterr().out


def test_cmd_menubar_background_explicit_url_pins(monkeypatch, capsys):
    """An explicit --url pins the bar to that server (passed through, and printed)."""
    seen = {}
    from gmlx import lifecycle as lc
    monkeypatch.setattr(lc, "gui_session_available", lambda: True)
    monkeypatch.setattr(lc, "start_menubar",
                        lambda extra=None: seen.update(extra=extra) or 0)
    rc = mb.cmd_menubar(["--url", "http://127.0.0.1:9000"])
    assert rc == 0
    assert "--url" in seen["extra"] and "http://127.0.0.1:9000" in seen["extra"]
    assert "http://127.0.0.1:9000" in capsys.readouterr().out


def test_cmd_menubar_background_needs_gui(monkeypatch, capsys):
    from gmlx import lifecycle as lc
    monkeypatch.setattr(lc, "gui_session_available", lambda: False)
    monkeypatch.setattr(lc, "auto_target", lambda h, p: ("127.0.0.1", 8080))
    rc = mb.cmd_menubar([])
    assert rc == 2
    assert "GUI session" in capsys.readouterr().err


# poll wires the Bearer header when a key is presented
def test_poll_sends_bearer(monkeypatch):
    seen = {}

    def _open(req, timeout=None):
        seen[req.full_url] = dict(req.header_items())
        return _Resp(b'{"server": {"resident_models": []}}')

    monkeypatch.setattr(mb.urllib.request, "urlopen", _open)
    mb.poll("http://127.0.0.1:8080", api_key="sekret")
    metrics_hdrs = next(h for u, h in seen.items() if u.endswith("/v1/metrics"))
    assert metrics_hdrs.get("Authorization") == "Bearer sekret"
    # /health is never authed
    health_hdrs = next(h for u, h in seen.items() if u.endswith("/health"))
    assert "Authorization" not in health_hdrs


# talk readiness (voice-chat item gating)
def test_poll_reads_talk_markers(monkeypatch):
    models = b'{"data": [{"id": "qwen3", "default": true}, ' \
             b'{"id": "whisper-turbo", "stt": true}, ' \
             b'{"id": "kokoro", "tts": true}]}'
    monkeypatch.setattr(mb.urllib.request, "urlopen", _fake_urlopen(
        {"/health": b"{}", "/v1/metrics": b"{}", "/v1/models": models}))
    snap = mb.poll("http://127.0.0.1:8080")
    assert snap["talk_ready"] is True
    assert snap["default_model"] == "qwen3"


def test_poll_talk_needs_both_services(monkeypatch):
    models = b'{"data": [{"id": "whisper-turbo", "stt": true}]}'  # stt only
    monkeypatch.setattr(mb.urllib.request, "urlopen", _fake_urlopen(
        {"/health": b"{}", "/v1/metrics": b"{}", "/v1/models": models}))
    assert mb.poll("http://127.0.0.1:8080")["talk_ready"] is False


def test_menu_can_talk_gating():
    m = mb.build_menu_model(_snap(talk_ready=True), None)
    assert m["can_talk"] is True
    assert mb.build_menu_model(_snap(), None)["can_talk"] is False
    assert mb.build_menu_model(
        _snap(talk_ready=True, auth_required=True), None)["can_talk"] is False
    assert mb.build_menu_model(
        _snap(talk_ready=True, reachable=False), None)["can_talk"] is False


def test_talk_label_names_the_model():
    # server default in the snapshot; talk.model override beats it; no model
    # known anywhere -> the generic label
    m = mb.build_menu_model(_snap(talk_ready=True, default_model="qwen3"), None)
    assert m["talk_label"] == "Talk to qwen3…"
    m = mb.build_menu_model(_snap(talk_ready=True, default_model="qwen3"),
                            None, talk_model="gemma-4")
    assert m["talk_label"] == "Talk to gemma-4…"
    assert mb.build_menu_model(_snap(talk_ready=True),
                               None)["talk_label"] == "Talk to model…"


def test_talk_model_from_config_reads_yaml(tmp_path):
    cfg = tmp_path / "gmlx.yaml"
    cfg.write_text("server: {port: 8080}\ntalk: {model: my-voice-model}\n")
    run = {"config_abspath": str(cfg)}
    assert mb.talk_model_from_config(run) == "my-voice-model"
    assert mb.talk_model_from_config(None) is None
    assert mb.talk_model_from_config({"config_abspath":
                                      str(tmp_path / "nope.yaml")}) is None


def test_talk_command_file_is_executable_stub(tmp_path):
    path = mb.talk_command_file("gmlx talk --base-url http://h:1/v1",
                                str(tmp_path))
    assert path.endswith(".command")
    assert os.access(path, os.X_OK)
    body = open(path).read()
    assert body.startswith("#!/bin/zsh\n")
    assert "exec gmlx talk --base-url http://h:1/v1" in body


def _fake_run(script):
    """subprocess.run stand-in: returncode looked up by argv head (`pgrep` or
    `open -a iTerm` or `open`), default 0. Records every argv."""
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        head = " ".join(argv[:3]) if argv[:2] == ["open", "-a"] else argv[0]
        return types.SimpleNamespace(returncode=script.get(head, 0))

    return run, calls


def test_open_talk_terminal_prefers_running_iterm(tmp_path):
    run, calls = _fake_run({"pgrep": 0})           # iTerm2 running
    mb.open_talk_terminal("gmlx talk", run=run, directory=str(tmp_path))
    assert calls[0][0] == "pgrep"
    assert calls[1][:3] == ["open", "-a", "iTerm"]  # handed to iTerm explicitly
    assert len(calls) == 2                         # no fallback needed


def test_open_talk_terminal_default_handler_and_fallback(tmp_path):
    run, calls = _fake_run({"pgrep": 1})           # iTerm2 not running
    mb.open_talk_terminal("gmlx talk", run=run, directory=str(tmp_path))
    assert [c[0] for c in calls] == ["pgrep", "open"]
    assert calls[1][1].endswith(".command")        # plain open -> default term

    run, calls = _fake_run({"pgrep": 0, "open -a iTerm": 1})   # -a failed
    mb.open_talk_terminal("gmlx talk", run=run, directory=str(tmp_path))
    assert calls[2][0] == "open" and calls[2][1].endswith(".command")


def test_voice_session_line_cases():
    assert mb.voice_session_line({"error": "boom"}) == "Voice chat: boom"
    assert mb.voice_session_line({"muted": True}) == "Voice chat: muted"
    assert mb.voice_session_line({"state": "idle", "wake": "hey assistant"}) \
        == 'Voice chat: say "hey assistant"'
    assert mb.voice_session_line({"state": "listening", "wake": "hey"}) \
        == 'Voice chat: say "hey"'
    # busy states name the state; no wake phrase -> plain state too
    assert mb.voice_session_line({"state": "thinking", "wake": "hey"}) \
        == "Voice chat: thinking"
    assert mb.voice_session_line({"state": "idle"}) == "Voice chat: idle"


def test_menu_model_voice_session():
    snap = _snap(talk_ready=True, default_model="qwen3")
    sess = {"state": "speaking", "muted": False, "wake": "hey assistant",
            "error": None, "busy": True}
    m = mb.build_menu_model(snap, None, session=sess, volume=0.6)
    assert m["title"].endswith("🔊")               # state glyph in the bar
    assert m["can_talk"] is False                  # start item replaced
    ts = m["talk_session"]
    assert ts["busy"] is True and ts["muted"] is False and ts["error"] is None
    assert ts["line"] == "Voice chat: speaking"
    assert ts["has_memory"] is False               # plain brain: no memory menu
    assert ts["volume"] == 0.6                     # slider rides the session
    m = mb.build_menu_model(snap, None, session=sess)
    assert m["talk_session"]["volume"] is None     # no volume -> no slider

    m = mb.build_menu_model(snap, None,
                            session={"state": "listening", "muted": True,
                                     "wake": "hey", "error": None,
                                     "busy": False, "has_memory": True})
    assert m["title"].endswith("🔇")
    assert m["talk_session"]["line"] == "Voice chat: muted"
    assert m["talk_session"]["has_memory"] is True  # assistant brain: menu shown

    # no session -> no glyph, start item offered as before
    m = mb.build_menu_model(snap, None)
    assert m["talk_session"] is None and m["can_talk"] is True
    assert m["title"] == "gmlx ●"


def test_voice_session_snapshot_lifecycle():
    sess = mb._VoiceSession()
    assert sess.alive()                            # starting counts as alive
    s = sess.snapshot()
    assert s["state"] == "starting" and not s["busy"]

    class FakeMachine:
        state = "thinking"
        muted = False
    sess.loop = types.SimpleNamespace(m=FakeMachine())
    sess.starting = False
    sess.wake = "hey assistant"
    s = sess.snapshot()
    assert s["state"] == "thinking" and s["busy"] and s["wake"] == "hey assistant"
    assert s["has_memory"] is False                # no brain on the fake loop

    sess.loop = types.SimpleNamespace(
        m=FakeMachine(), brain=types.SimpleNamespace(memory=object()))
    assert sess.snapshot()["has_memory"] is True

    sess.lines.append("you: hi")
    sess.lines.append("  hello")
    assert sess.transcript() == "you: hi\n  hello"
    assert not sess.alive()                        # no thread, not starting


def test_no_talk_flags_covers_merged_settings():
    """Every attr _merged_settings reads must exist (and be unset) so the
    menu bar resolves purely from YAML."""
    from gmlx.config import TalkCfg
    flags = mb._no_talk_flags()
    s = __import__("gmlx.talk", fromlist=["_merged_settings"])._merged_settings(
        flags, TalkCfg())
    assert s["mode"] == "wake" and s["wake_word"] == "hey assistant"
    assert s["chime"] is True                      # no_chime=False keeps chime


# busy signal, metrics-error honesty, row markers, copy-URL, death notification
def test_poll_reads_busy_and_queue(monkeypatch):
    metrics = (b'{"server": {"resident_models": [{"ids": ["qwen3"], "busy": 2,'
               b' "footprint_bytes": 1}], "request_queue_depth": 3}}')
    monkeypatch.setattr(mb.urllib.request, "urlopen",
                        _fake_urlopen({"/health": b"{}", "/v1/metrics": metrics}))
    snap = mb.poll("http://127.0.0.1:8080")
    assert snap["in_flight"] == 2 and snap["queue_depth"] == 3


def test_poll_shape_guards_survive_foreign_service(monkeypatch):
    # Valid JSON of the wrong shape (something else on the port) must not
    # kill the poll thread.
    monkeypatch.setattr(mb.urllib.request, "urlopen", _fake_urlopen(
        {"/health": b"{}", "/v1/metrics": b"[1, 2]", "/v1/models": b'{"data": 1}'}))
    snap = mb.poll("http://127.0.0.1:8080")
    assert snap["reachable"] is True
    assert snap["resident"] == [] and snap["in_flight"] == 0
    assert snap["talk_ready"] is False


def test_menu_busy_line_and_glyph():
    m = mb.build_menu_model(_snap(in_flight=1, queue_depth=2), None)
    assert m["busy_line"] == "1 generating, 2 queued"
    assert m["title"].endswith("◉") and m["state"] == "up"
    m = mb.build_menu_model(_snap(queue_depth=1), None)
    assert m["busy_line"] == "1 queued"
    m = mb.build_menu_model(_snap(), None)
    assert m["busy_line"] is None and m["title"] == "gmlx ●"


def test_menu_metrics_error_says_unknown_not_empty():
    m = mb.build_menu_model(_snap(error="metrics HTTP 500"), None)
    assert "unknown" in m["models_header"]
    assert "No models resident" not in m["models_header"]
    assert mb.build_menu_model(_snap(), None)["models_header"] \
        == "No models resident"


def test_menu_row_markers_and_eviction_countdown():
    snap = _snap(resident=[
        {"ids": ["a"], "footprint_bytes": 1_000_000_000, "ttl_s": 300,
         "idle_s": 40},
        {"ids": ["b"], "footprint_bytes": 1_000_000_000, "pinned": True,
         "ttl_s": 300, "idle_s": 40},
        {"ids": ["c"], "footprint_bytes": 1_000_000_000, "kept": True,
         "ttl_s": 300, "idle_s": 40},
        {"ids": ["d"], "footprint_bytes": 1_000_000_000, "ttl_s": 90,
         "idle_s": 0},
    ], default_model="a")
    labels = [x["label"] for x in mb.build_menu_model(snap, None)["models"]]
    la, lb, lc, ld = labels
    assert "[default]" in la and "evicts in 4m" in la      # 260s left -> minutes
    assert "[pinned]" in lb and "evicts" not in lb          # reaper-exempt
    assert "[kept]" in lc and "evicts" not in lc
    assert "evicts in 90s" in ld and "[default]" not in ld


def test_menu_carries_url_for_copy():
    assert mb.build_menu_model(_snap(), None)["url"] == "http://127.0.0.1:8080"


def test_menu_config_path_runfile_beats_fallback():
    run = {"managed_by": "detach", "pid": 1, "config_abspath": "/x/mine.yaml"}
    m = mb.build_menu_model(_snap(), run, fallback_config="/d/default.yaml")
    assert m["config_path"] == "/x/mine.yaml"
    # no runfile: the default-location fallback backs the Edit-config item,
    # including while the server is down (fixing the config IS the down task)
    m = mb.build_menu_model(_snap(reachable=False), None,
                            fallback_config="/d/default.yaml")
    assert m["config_path"] == "/d/default.yaml"
    assert mb.build_menu_model(_snap(), None)["config_path"] is None


def test_down_notifier_transitions():
    clk = [0.0]
    n = mb.DownNotifier(grace_s=10.0, clock=lambda: clk[0])
    assert n.observe(True) is False           # first sighting: no history
    clk[0] = 1.0
    assert n.observe(False) is True           # unexpected up->down: notify
    assert n.observe(False) is False          # steady down: only the flip
    assert n.observe(True) is False
    n.expect()                                # bar-initiated stop/restart
    clk[0] = 2.0
    assert n.observe(False) is False          # inside the grace window
    assert n.observe(True) is False
    clk[0] = 60.0
    assert n.observe(False) is True           # grace expired: a real death


# Unified logs panel: the pure tailer behind it
def test_log_merger_seeds_tail_and_prefixes(tmp_path):
    srv = tmp_path / "server.log"
    bar = tmp_path / "menubar.log"
    srv.write_text("old line\nboot ok\n")
    bar.write_text("bar up\n")
    m = mb.LogMerger()
    out = m.read_new([("server", str(srv)), ("menubar", str(bar))])
    assert out == [" server | old line", " server | boot ok",
                   "menubar | bar up"]
    assert m.read_new([("server", str(srv)), ("menubar", str(bar))]) == []


def test_log_merger_interleaves_increments(tmp_path):
    srv = tmp_path / "server.log"
    bar = tmp_path / "menubar.log"
    srv.write_text("")
    bar.write_text("")
    m = mb.LogMerger()
    m.read_new([("server", str(srv)), ("menubar", str(bar))])
    with open(srv, "a") as f:
        f.write("req 1\n")
    with open(bar, "a") as f:
        f.write("click\n")
    out = m.read_new([("server", str(srv)), ("menubar", str(bar))])
    assert out == [" server | req 1", "menubar | click"]


def test_log_merger_buffers_partial_lines(tmp_path):
    p = tmp_path / "a.log"
    p.write_text("")
    m = mb.LogMerger()
    m.read_new([("a", str(p))])
    with open(p, "a") as f:
        f.write("half")                        # no newline yet
    assert m.read_new([("a", str(p))]) == []
    with open(p, "a") as f:
        f.write(" done\nnext\n")
    assert m.read_new([("a", str(p))]) == ["      a | half done",
                                           "      a | next"]


def test_log_merger_resets_on_truncation(tmp_path):
    p = tmp_path / "a.log"
    p.write_text("one\ntwo\nthree\n")
    m = mb.LogMerger()
    m.read_new([("a", str(p))])
    p.write_text("fresh\n")                    # rotated/truncated shorter
    assert m.read_new([("a", str(p))]) == ["      a | fresh"]


def test_log_merger_missing_file_is_silent(tmp_path):
    m = mb.LogMerger()
    assert m.read_new([("a", str(tmp_path / "nope.log"))]) == []


# tap-to-talk hotkey: persisted settings + menu model
def test_menubar_settings_roundtrip(tmp_path, monkeypatch):
    from gmlx import lifecycle
    monkeypatch.setattr(lifecycle, "runtime_dir", lambda: tmp_path)
    assert mb.load_menubar_settings() == {"hotkey": "off", "autostart": None,
                                          "volume": 1.0}
    mb.save_menubar_settings({"hotkey": "on", "volume": 0.4})
    got = mb.load_menubar_settings()
    assert got["hotkey"] == "on" and got["volume"] == 0.4


def test_menubar_settings_tolerate_garbage(tmp_path, monkeypatch):
    from gmlx import lifecycle
    monkeypatch.setattr(lifecycle, "runtime_dir", lambda: tmp_path)
    (tmp_path / "menubar-settings.json").write_text("{not json")
    assert mb.load_menubar_settings()["hotkey"] == "off"
    (tmp_path / "menubar-settings.json").write_text(
        '{"hotkey": "globe-everything"}')
    assert mb.load_menubar_settings()["hotkey"] == "off"   # unknown coerces
    (tmp_path / "menubar-settings.json").write_text(
        '{"hotkey": "globe-space"}')                       # pre-modifier value
    assert mb.load_menubar_settings()["hotkey"] == "on"
    # volume: non-numeric falls to the default, out-of-range clamps
    for raw, want in (('"loud"', 1.0), ("null", 1.0), ("NaN", 1.0),
                      ("-3", 0.0), ("7", 1.0), ("0.25", 0.25)):
        (tmp_path / "menubar-settings.json").write_text(
            '{"volume": %s}' % raw)
        assert mb.load_menubar_settings()["volume"] == want


def test_menu_model_hotkey_toggle():
    m = mb.build_menu_model(_snap(), None, hotkey={
        "enabled": True, "available": True, "error": None,
        "label": "Right ⌘ + Space"})
    assert m["hotkey"] == {"enabled": True, "error": None,
                           "label": "Right ⌘ + Space"}
    # hidden when absent/unavailable...
    assert mb.build_menu_model(_snap(), None)["hotkey"] is None
    assert mb.build_menu_model(_snap(), None, hotkey={
        "enabled": False, "available": False})["hotkey"] is None
    # ...but server-down keeps the item (a local setting, not a control)
    down = mb.build_menu_model(_snap(reachable=False), None, hotkey={
        "enabled": True, "available": True,
        "error": "not active - needs permission"})
    assert down["hotkey"]["error"] == "not active - needs permission"
    assert down["hotkey"]["label"] == "\U0001f310 + Space"   # default label


def test_down_message_crash_vs_hang():
    detach = {"managed_by": "detach", "pid": 4242}
    msg = mb.down_message("http://h:1", detach, pid_dead=True)
    assert "crashed" in msg and "Open logs" in msg and "Start server" in msg
    # alive but unresponsive: don't claim a crash
    assert "stopped responding" in mb.down_message("http://h:1", detach,
                                                   pid_dead=False)
    # launchd runs have no usable pid and KeepAlive is already restarting
    launchd = {"managed_by": "launchd", "pid": None}
    assert "crashed" not in mb.down_message("http://h:1", launchd,
                                            pid_dead=True)
    assert mb.down_message(None, None, True) == "the server stopped responding"


def test_menubar_settings_autostart_roundtrip(tmp_path, monkeypatch):
    from gmlx import lifecycle
    monkeypatch.setattr(lifecycle, "runtime_dir", lambda: tmp_path)
    assert mb.load_menubar_settings()["autostart"] is None
    rec = {"argv": ["/stub", "-m", "gmlx", "serve", "--foreground"],
           "host": "127.0.0.1", "port": 8080,
           "config_abspath": "/abs/c.yaml", "api_key_set": True}
    mb.save_menubar_settings({"hotkey": "on", "autostart": rec})
    got = mb.load_menubar_settings()
    assert got["hotkey"] == "on" and got["autostart"] == rec
    # malformed records degrade to no-autostart instead of a boot crash
    for bad in ({"argv": []}, {"argv": ["/x"], "port": "eighty"},
                "yes", 7, {"host": "no-argv"}):
        mb.save_menubar_settings({"autostart": bad})
        assert mb.load_menubar_settings()["autostart"] is None


def _seed_autostart(tmp_path, monkeypatch, launched, *, boot="1234"):
    from gmlx import lifecycle, procname
    monkeypatch.setattr(lifecycle, "runtime_dir", lambda: tmp_path)
    monkeypatch.setattr(lifecycle, "boot_time", lambda: boot)
    monkeypatch.setattr(lifecycle, "read_run", lambda h, p: None)
    monkeypatch.setattr(lifecycle, "launch_detached",
                        lambda argv, **kw: (launched.append((argv, kw)), 0)[1])
    monkeypatch.setattr(procname, "named_python", lambda: "/fresh/gmlx")
    mb.save_menubar_settings({"hotkey": "off", "autostart": {
        "argv": ["/old/gmlx", "-m", "gmlx", "serve", "--foreground"],
        "host": "127.0.0.1", "port": 8080,
        "config_abspath": "/abs/c.yaml", "api_key_set": False}})


def test_autostart_runs_once_per_boot(tmp_path, monkeypatch):
    """The boot stamp is the don't-fight-the-user rule: a KeepAlive respawn
    of the menu bar mid-session must not resurrect a stopped server."""
    from gmlx import lifecycle
    launched = []
    _seed_autostart(tmp_path, monkeypatch, launched)
    mb._autostart_server_once()
    assert len(launched) == 1
    argv, kw = launched[0]
    assert argv[0] == "/fresh/gmlx"          # refreshed stub, not the recorded one
    assert argv[1:] == ["-m", "gmlx", "serve", "--foreground"]
    assert kw["config_abspath"] == "/abs/c.yaml"
    mb._autostart_server_once()               # same boot: skipped
    assert len(launched) == 1
    monkeypatch.setattr(lifecycle, "boot_time", lambda: "5678")
    mb._autostart_server_once()               # next login: runs again
    assert len(launched) == 2


def test_autostart_skips_running_server_and_empty_record(tmp_path, monkeypatch):
    from gmlx import lifecycle
    launched = []
    _seed_autostart(tmp_path, monkeypatch, launched)
    monkeypatch.setattr(lifecycle, "read_run",
                        lambda h, p: {"pid": 1, "managed_by": "detach"})
    monkeypatch.setattr(lifecycle, "identity_ok", lambda run: True)
    mb._autostart_server_once()
    assert launched == []                     # already up

    def no_boot_probe():
        raise AssertionError("no record: must return before boot_time")

    mb.save_menubar_settings({"hotkey": "off"})
    monkeypatch.setattr(lifecycle, "boot_time", no_boot_probe)
    mb._autostart_server_once()
    assert launched == []


def test_ptt_modifier_from_config(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("server: {model_dirs: []}\n"
                   "discover: [{dir: /tmp}]\n"
                   "talk:\n  push_to_talk_modifier: right-command\n")
    assert mb.ptt_modifier_from_config(
        {"config_abspath": str(cfg)}) == "right-command"
    assert mb.ptt_modifier_from_config(None) == "globe"
    assert mb.ptt_modifier_from_config(
        {"config_abspath": str(tmp_path / "absent.yaml")}) == "globe"
