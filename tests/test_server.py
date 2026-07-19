#!/usr/bin/env python3
"""Start-mode plumbing for the server CLI: single-model / discovery in-memory
config synthesis, the preload pick, mode mutual-exclusion, and `init` scaffold
write semantics (--force refusal, next-command hint). CPU-only - discovery is
faked so no GGUF is read and no server starts."""
from __future__ import annotations

import importlib
import logging
import sys
import types

import pytest

from gmlx import server as srv  # noqa: E402
from gmlx import server_patches as sp  # noqa: E402
from gmlx.server_patches import observability as sp_obs  # noqa: E402
from gmlx.config import ModelCfg, ServerCfg, ServerDefaults  # noqa: E402


def _ns(**kw):
    """An argparse-namespace stand-in with the serve defaults filled in."""
    base = dict(model=None, config=None, models_dir=None, recursive=False,
                hf_cache=False, mmproj=None, hf_source=None, speculative=False,
                draft_gguf=None, adapter=None, chat_template=None, host=None,
                port=None, budget_gb=None, max_models=None, pin=[], max_tokens=None,
                api_key=None, no_auth=False, stt=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


# single-model mode
def test_single_model_cfg_derives_id_and_pins(tmp_path):
    g = tmp_path / "Qwen3-0.6B-Q4_K_M.gguf"
    g.write_text("x")
    cfg = srv._single_model_cfg(_ns(model=str(g)))
    assert isinstance(cfg, ServerCfg)
    (mid, m), = cfg.models.items()
    assert mid == "qwen3.6-0.6b" or mid.startswith("qwen3")   # derived, slugified
    assert m.path == str(g.resolve()) or m.path == str(g)      # absolutised
    assert m.pin is True                                       # single model pinned
    assert cfg.defaults.model == mid                           # it is the default


def test_single_model_cfg_speculative_from_draft(tmp_path):
    g = tmp_path / "gemma-4-31B-it-Q6_K.gguf"
    d = tmp_path / "gemma-4-31B-it-assistant-Q8_0.gguf"
    g.write_text("x")
    d.write_text("x")
    cfg = srv._single_model_cfg(_ns(model=str(g), draft_gguf=str(d)))
    (_, m), = cfg.models.items()
    assert m.speculative is True
    assert m.draft_gguf == str(d.resolve()) or m.draft_gguf == str(d)


def test_single_model_cfg_vlm_mmproj(tmp_path):
    g = tmp_path / "gemma-4-E4B-it-Q6_K.gguf"
    mm = tmp_path / "mmproj-gemma-4-E4B-it-bf16.gguf"
    g.write_text("x")
    mm.write_text("x")
    cfg = srv._single_model_cfg(_ns(model=str(g), mmproj=str(mm)))
    (_, m), = cfg.models.items()
    assert m.mmproj and m.mmproj.endswith("mmproj-gemma-4-E4B-it-bf16.gguf")


def test_single_model_cfg_chat_template_rides_overrides(tmp_path):
    """--chat-template lands in the model's overrides and survives resolution onto
    the ResolvedModel (the same slot a config `overrides: {chat_template}` uses)."""
    from gmlx.config import resolve_model
    g = tmp_path / "Qwen3-0.6B-Q4_K_M.gguf"
    g.write_text("x")
    cfg = srv._single_model_cfg(_ns(model=str(g), chat_template="/tmpl/x.jinja"))
    (mid, m), = cfg.models.items()
    assert m.overrides == {"chat_template": "/tmpl/x.jinja"}
    assert resolve_model(mid, cfg).chat_template == "/tmpl/x.jinja"


def test_single_model_cfg_no_chat_template_has_empty_overrides(tmp_path):
    g = tmp_path / "Qwen3-0.6B-Q4_K_M.gguf"
    g.write_text("x")
    cfg = srv._single_model_cfg(_ns(model=str(g)))
    (_, m), = cfg.models.items()
    assert m.overrides == {}


def test_single_model_cfg_moe_expert_mass_passthrough(tmp_path):
    """--moe-expert-mass lands on the ModelCfg beside the placement and
    resolves onto the ResolvedModel - the single-model analog of a config
    `moe_expert_mass:`."""
    from gmlx.config import resolve_model
    g = tmp_path / "Qwen3-0.6B-Q4_K_M.gguf"
    g.write_text("x")
    cfg = srv._single_model_cfg(
        _ns(model=str(g), stream_experts=True, moe_expert_mass=0.9))
    (mid, m), = cfg.models.items()
    assert m.stream == "experts" and m.moe_expert_mass == 0.9
    assert resolve_model(mid, cfg).moe_expert_mass == 0.9


def test_single_model_cfg_adapter_passthrough(tmp_path):
    """--adapter lands on the ModelCfg and resolves onto the ResolvedModel (and so
    into its load_signature) - the single-model analog of a config `adapter:`."""
    from gmlx.config import resolve_model
    g = tmp_path / "Qwen3-0.6B-Q4_K_M.gguf"
    ad = tmp_path / "pirate.lora.gguf"
    g.write_text("x")
    ad.write_text("x")
    cfg = srv._single_model_cfg(_ns(model=str(g), adapter=str(ad)))
    (mid, m), = cfg.models.items()
    assert m.adapter == str(ad.resolve()) or m.adapter == str(ad)
    assert resolve_model(mid, cfg).adapter == m.adapter


# discovery mode (faked scan)
def test_discovery_cfg_wraps_scan(monkeypatch):
    fake = [ModelCfg(id="qwen", path="/m/qwen.gguf"),
            ModelCfg(id="gemma", path="/m/gemma.gguf")]
    monkeypatch.setattr(srv.discovery, "scan_dirs", lambda specs, dirs: fake)
    cfg = srv._discovery_cfg(["/m"], _ns(port=9000))
    assert set(cfg.models) == {"qwen", "gemma"}
    assert cfg.model_dirs == ["/m"]
    assert cfg.port == 9000


# preload pick: pinned > defaults.model > sole > none
def test_preload_prefers_pinned():
    cfg = ServerCfg(models={
        "a": ModelCfg(id="a", path="/a"),
        "b": ModelCfg(id="b", path="/b", pin=True),
    })
    assert srv._preload_id(cfg) == "b"


def test_preload_uses_default_model():
    cfg = ServerCfg(
        defaults=ServerDefaults(model="a"),
        models={"a": ModelCfg(id="a", path="/a"), "b": ModelCfg(id="b", path="/b")},
    )
    assert srv._preload_id(cfg) == "a"


def test_preload_sole_model():
    cfg = ServerCfg(models={"only": ModelCfg(id="only", path="/o")})
    assert srv._preload_id(cfg) == "only"


def test_preload_none_when_many_and_no_default():
    cfg = ServerCfg(models={
        "a": ModelCfg(id="a", path="/a"), "b": ModelCfg(id="b", path="/b")})
    assert srv._preload_id(cfg) is None


# _resolve_cfg dispatch
def test_resolve_cfg_config_mode(monkeypatch, tmp_path):
    cfg = ServerCfg(models={"x": ModelCfg(id="x", path="/x")})
    monkeypatch.setattr(srv, "load_config", lambda p: cfg)
    got, reload_fn = srv._resolve_cfg(_ns(config="/some/cfg.yaml"))
    assert got is cfg
    assert callable(reload_fn)                       # config mode wires a reloader


def test_resolve_cfg_single_model_no_reload(tmp_path):
    g = tmp_path / "m-Q4_K_M.gguf"
    g.write_text("x")
    got, reload_fn = srv._resolve_cfg(_ns(model=str(g)))
    assert len(got.models) == 1
    assert reload_fn is None


# mode mutual-exclusion
def test_serve_rejects_two_modes():
    with pytest.raises(SystemExit) as ei:           # ap.error -> sys.exit(2)
        srv._cmd_serve(["model.gguf", "--config", "c.yaml"])
    assert ei.value.code == 2


def test_serve_allows_vlm_and_speculative(tmp_path, capsys):
    # VLM (--mmproj) + MTP (--speculative) now coexist: text-only requests speculate,
    # media requests take the VLM forward. --print-config resolves without rejecting.
    g = tmp_path / "m.gguf"
    g.write_text("x")
    mm = tmp_path / "mm.gguf"
    mm.write_text("x")
    rc = srv._cmd_serve(
        [str(g), "--mmproj", str(mm), "--speculative", "--print-config"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "mmproj" in out and "speculative" in out


def test_print_config_reflects_cli_flags(tmp_path, capsys):
    # --print-config promises the EFFECTIVE config: flags beat the file, the
    # same precedence _serve applies at startup.
    g = tmp_path / "m.gguf"
    g.write_text("x")
    rc = srv._cmd_serve([str(g), "--print-config", "--port", "9317",
                         "--budget-gb", "12.5", "--embeddings",
                         "hf:org/repo/e.gguf"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "port: 9317" in out
    assert "budget_gb: 12.5" in out
    assert "hf:org/repo/e.gguf" in out


# default start mode: detach the server + raise the menu bar; never run _serve here
def test_serve_backgrounds_by_default(monkeypatch, tmp_path):
    g = tmp_path / "m.gguf"
    g.write_text("x")
    from gmlx import lifecycle as lc
    seen = {}

    def fake_bg(serve_args, **kw):
        seen["bg"] = kw
        return 0

    def fake_mb(**kw):
        seen["mb"] = True
        return 0

    def fake_serve(*a, **k):
        seen["fg"] = True
        return 99

    monkeypatch.setattr(lc, "start_background", fake_bg)
    monkeypatch.setattr(lc, "gui_session_available", lambda: True)
    monkeypatch.setattr(lc, "start_menubar", fake_mb)
    monkeypatch.setattr(srv, "_serve", fake_serve)
    rc = srv._cmd_serve([str(g)])
    assert rc == 0
    assert "bg" in seen and "fg" not in seen               # detached, not in-process
    assert seen["mb"] is True                               # the one machine-wide bar raised


def test_serve_foreground_flag_runs_in_place(monkeypatch, tmp_path):
    g = tmp_path / "m.gguf"
    g.write_text("x")
    from gmlx import lifecycle as lc
    seen = {}

    def fake_bg(*a, **k):
        seen["bg"] = True
        return 0

    def fake_serve(cfg, a, reload_fn):
        seen["fg"] = True
        return 0

    monkeypatch.setattr(lc, "start_background", fake_bg)
    monkeypatch.setattr(srv, "_import_serving", lambda: None)
    monkeypatch.setattr(srv, "_resolve_cfg", lambda a: (_one_model_cfg(), None))
    monkeypatch.setattr(srv, "_serve", fake_serve)
    rc = srv._cmd_serve([str(g), "-f"])
    assert rc == 0
    assert "fg" in seen and "bg" not in seen               # -f stays in the foreground


def test_serve_no_menubar_flag_skips_menu(monkeypatch, tmp_path):
    g = tmp_path / "m.gguf"
    g.write_text("x")
    from gmlx import lifecycle as lc
    seen = {}
    def fake_mb(*a, **k):
        seen["mb"] = True
        return 0

    monkeypatch.setattr(lc, "start_background", lambda *a, **k: 0)
    monkeypatch.setattr(lc, "gui_session_available", lambda: True)
    monkeypatch.setattr(lc, "start_menubar", fake_mb)
    rc = srv._cmd_serve([str(g), "--no-menubar"])
    assert rc == 0
    assert "mb" not in seen                                 # opted out of the menu bar


# init scaffold write semantics
def test_init_writes_and_refuses_overwrite(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(srv.discovery, "scan_dirs",
                        lambda specs, dirs, **kw: [ModelCfg(id="qwen",
                                                            path="/m/qwen.gguf")])
    out = tmp_path / "cfg.yaml"
    rc = srv._cmd_init(["--out", str(out), "--models-dir", "/m"])
    assert rc == 0
    assert out.exists()
    text = out.read_text()
    assert "qwen" in text and "models:" in text
    hint = capsys.readouterr().out
    assert "next:" in hint and str(out) in hint     # prints the next command

    rc2 = srv._cmd_init(["--out", str(out), "--models-dir", "/m"])
    assert rc2 == 1                                  # refuses to overwrite
    rc3 = srv._cmd_init(["--out", str(out), "--models-dir", "/m", "--force"])
    assert rc3 == 0                                  # --force overwrites


def test_init_validates_default_model(monkeypatch, tmp_path, capsys):
    # Discovery GENERATES the ids, so a hand-typed --default-model that matches
    # nothing must fail here (naming the real ids) - not exit 0 and write a
    # config every consumer rejects.
    monkeypatch.setattr(srv.discovery, "scan_dirs",
                        lambda specs, dirs, **kw: [ModelCfg(id="qwen-q4",
                                                            path="/m/q.gguf")])
    out = tmp_path / "cfg.yaml"
    rc = srv._cmd_init(["--out", str(out), "--models-dir", "/m",
                        "--default-model", "qwen"])
    assert rc == 2 and not out.exists()
    assert "qwen-q4" in capsys.readouterr().err
    rc = srv._cmd_init(["--out", str(out), "--models-dir", "/m",
                        "--default-model", "qwen-q4"])
    assert rc == 0 and out.exists()


def test_init_port_flag_lands_in_config(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(srv.discovery, "scan_dirs",
                        lambda specs, dirs, **kw: [ModelCfg(id="qwen",
                                                            path="/m/qwen.gguf")])
    out = tmp_path / "cfg.yaml"
    rc = srv._cmd_init(["--out", str(out), "--models-dir", "/m", "--port", "9090"])
    assert rc == 0
    assert "port: 9090" in out.read_text()
    from gmlx.config import load_config
    assert load_config(str(out)).port == 9090


def test_init_port_flag_rejects_out_of_range(monkeypatch, tmp_path, capsys):
    out = tmp_path / "cfg.yaml"
    with pytest.raises(SystemExit):
        srv._cmd_init(["--out", str(out), "--models-dir", "/m", "--port", "0"])
    assert "--port" in capsys.readouterr().err
    assert not out.exists()


def test_init_reloads_running_server(monkeypatch, tmp_path):
    """After init rewrites the config, it SIGHUPs a server already running from it."""
    monkeypatch.setattr(srv.discovery, "scan_dirs",
                        lambda specs, dirs, **kw: [ModelCfg(id="qwen",
                                                            path="/m/qwen.gguf")])
    out = tmp_path / "cfg.yaml"
    reloaded = {}
    monkeypatch.setattr(srv, "_reload_running",
                        lambda path, *, skip: reloaded.update(path=str(path), skip=skip))
    rc = srv._cmd_init(["--out", str(out), "--models-dir", "/m"])
    assert rc == 0
    assert reloaded["path"] == str(out) and reloaded["skip"] is False


def test_init_no_reload_flag_skips_reload(monkeypatch, tmp_path):
    monkeypatch.setattr(srv.discovery, "scan_dirs",
                        lambda specs, dirs, **kw: [ModelCfg(id="qwen",
                                                            path="/m/qwen.gguf")])
    out = tmp_path / "cfg.yaml"
    reloaded = {}
    monkeypatch.setattr(srv, "_reload_running",
                        lambda path, *, skip: reloaded.update(skip=skip))
    rc = srv._cmd_init(["--out", str(out), "--models-dir", "/m", "--no-reload"])
    assert rc == 0
    assert reloaded["skip"] is True                    # --no-reload threads through


def test_init_no_args_shows_help(capsys):
    rc = srv._cmd_init([])                            # no args -> help, not an error
    assert rc == 0
    out = capsys.readouterr().out
    assert "--models-dir" in out and "usage" in out.lower()


def test_init_requires_models_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(srv.discovery, "scan_dirs", lambda *a, **k: [])
    with pytest.raises(SystemExit) as ei:            # other args but no --models-dir
        srv._cmd_init(["--out", str(tmp_path / "c.yaml")])
    assert ei.value.code == 2


def test_init_defaults_to_write_path(monkeypatch, tmp_path):
    monkeypatch.setattr(srv.discovery, "scan_dirs", lambda *a, **k: [])
    default = tmp_path / "config.yaml"
    monkeypatch.setattr(srv, "default_config_write_path", lambda: default)
    rc = srv._cmd_init(["--models-dir", "/m"])       # no --out -> default location
    assert rc == 0 and default.exists()


def test_init_from_hf_cache_writes_portable_entries(monkeypatch, tmp_path):
    monkeypatch.setattr(
        srv.discovery, "scan_hf_cache",
        lambda **kw: [ModelCfg(id="m", path="hf:org/repo/m-Q4_K_M.gguf")])
    out = tmp_path / "c.yaml"
    rc = srv._cmd_init(["--from-hf-cache", "--out", str(out)])    # no --models-dir
    assert rc == 0
    from gmlx.config import load_config
    cfg = load_config(out)
    assert cfg.hf_cache is True
    assert cfg.models["m"].path == "hf:org/repo/m-Q4_K_M.gguf"


def test_init_requires_dir_or_cache(monkeypatch, tmp_path):
    with pytest.raises(SystemExit) as ei:           # neither --models-dir nor cache
        srv._cmd_init(["--out", str(tmp_path / "c.yaml")])
    assert ei.value.code == 2


def test_init_next_hint_omits_config_for_default_location(monkeypatch, tmp_path,
                                                          capsys):
    monkeypatch.setattr(srv.discovery, "scan_dirs", lambda *a, **k: [])
    out = tmp_path / "cfg.yaml"
    monkeypatch.setattr(srv, "default_config_paths", lambda: [out])
    rc = srv._cmd_init(["--models-dir", "/m", "--out", str(out)])
    assert rc == 0
    hint = capsys.readouterr().out
    assert "next: gmlx serve" in hint and "--config" not in hint
    if sys.platform == "darwin":   # the launchd start-at-login hint, bare here too
        assert "gmlx service install" in hint


def test_init_next_hint_keeps_config_for_nondefault_location(monkeypatch, tmp_path,
                                                             capsys):
    monkeypatch.setattr(srv.discovery, "scan_dirs", lambda *a, **k: [])
    monkeypatch.setattr(srv, "default_config_paths",
                        lambda: [tmp_path / "elsewhere.yaml"])
    out = tmp_path / "cfg.yaml"
    rc = srv._cmd_init(["--models-dir", "/m", "--out", str(out)])
    assert rc == 0
    hint = capsys.readouterr().out
    assert f"--config {out}" in hint
    if sys.platform == "darwin":   # the service-install hint carries --config too
        assert f"gmlx service install --config {out}" in hint


def test_main_dispatches_init(monkeypatch, tmp_path):
    called = {}
    monkeypatch.setattr(srv, "_cmd_init", lambda argv, prog=None: called.setdefault("init", argv) or 0)
    monkeypatch.setattr(srv, "_cmd_serve", lambda argv, prog=None: called.setdefault("serve", argv) or 0)
    srv.main(["init", "--models-dir", "/m"])
    assert called["init"] == ["--models-dir", "/m"]
    srv.main(["model.gguf"])
    assert called["serve"] == ["model.gguf"]


# sync - reconcile a config's models with what's on disk
def _sync_config(tmp_path, body):
    """Write a config whose model_dirs is <tmp_path>/lib (created), return its path."""
    lib = tmp_path / "lib"
    lib.mkdir(exist_ok=True)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(f"server:\n  model_dirs:\n    - {lib}\n" + body)
    return cfg_path, lib


def test_sync_adds_new_and_removes_gone(monkeypatch, tmp_path):
    cfg_path, lib = _sync_config(
        tmp_path,
        "models:\n  keep:\n    path: keep.gguf\n  gone:\n    path: gone.gguf\n")
    (lib / "keep.gguf").write_bytes(b"x")             # keep's file still exists
    from gmlx.config import ModelCfg as MC
    monkeypatch.setattr(
        srv.discovery, "scan_dirs",
        lambda specs, dirs, **kw: [MC(id="newbie", path=str(lib / "newbie.gguf"))])
    rc = srv._cmd_sync(["--config", str(cfg_path)])
    assert rc == 0
    from gmlx.config import load_config
    cfg = load_config(cfg_path)
    assert set(cfg.models) == {"keep", "newbie"}      # gone dropped, newbie added
    assert cfg.models["newbie"].path == "newbie.gguf"  # relative to model_dirs


def test_sync_never_drops_hf_entries_on_unreadable_cache(monkeypatch, tmp_path,
                                                          capsys):
    # An unreadable hf cache means hf: entries CANNOT BE VERIFIED - removing
    # them turns a transient env problem (HF_HOME set in another shell) into a
    # destructive config write.
    from gmlx.config import MissingModelFile, load_config
    cfg_path, lib = _sync_config(
        tmp_path, "models:\n  hfm:\n    path: hf:org/repo/f.gguf\n")
    real = srv.resolve_path

    def fake_resolve(p, dirs):
        if str(p).startswith("hf:"):
            raise MissingModelFile("hf model is not in the local cache")
        return real(p, dirs)

    monkeypatch.setattr(srv, "resolve_path", fake_resolve)
    monkeypatch.setattr(srv.discovery, "scan_dirs", lambda specs, dirs, **kw: [])
    monkeypatch.setattr(srv, "_hf_cache_readable", lambda: False)
    assert srv._cmd_sync(["--config", str(cfg_path)]) == 0
    assert "hfm" in load_config(cfg_path).models          # kept, not dropped
    err = capsys.readouterr().err
    assert "unreadable" in err
    # With a readable cache the same miss is a real removal.
    monkeypatch.setattr(srv, "_hf_cache_readable", lambda: True)
    assert srv._cmd_sync(["--config", str(cfg_path)]) == 0
    assert "hfm" not in load_config(cfg_path).models


def test_sync_keeps_relative_entries_when_root_missing(monkeypatch, tmp_path,
                                                       capsys):
    # A missing model_dirs root (unmounted disk) must not mass-drop the entries
    # under it - they are unverifiable, not gone.
    from gmlx.config import load_config
    gone_root = tmp_path / "unmounted"
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(f"server:\n  model_dirs:\n    - {gone_root}\n"
                        "models:\n  m1:\n    path: m1.gguf\n")
    monkeypatch.setattr(srv.discovery, "scan_dirs", lambda specs, dirs, **kw: [])
    assert srv._cmd_sync(["--config", str(cfg_path)]) == 0
    assert "m1" in load_config(cfg_path).models
    assert "model_dirs root" in capsys.readouterr().err


def test_sync_drops_dangling_absolute_paths(monkeypatch, tmp_path):
    # Absolute-path entries get the same the-file-is-gone reconciliation the
    # relative ones always had.
    from gmlx.config import load_config
    cfg_path, lib = _sync_config(
        tmp_path,
        f"models:\n  abs-gone:\n    path: {tmp_path}/gone-abs.gguf\n"
        "  keep:\n    path: keep.gguf\n")
    (lib / "keep.gguf").write_bytes(b"x")
    monkeypatch.setattr(srv.discovery, "scan_dirs", lambda specs, dirs, **kw: [])
    assert srv._cmd_sync(["--config", str(cfg_path)]) == 0
    cfg = load_config(cfg_path)
    assert "abs-gone" not in cfg.models and "keep" in cfg.models


def test_sync_dry_run_leaves_file_untouched(monkeypatch, tmp_path, capsys):
    cfg_path, lib = _sync_config(tmp_path, "models:\n  keep:\n    path: keep.gguf\n")
    (lib / "keep.gguf").write_bytes(b"x")
    original = cfg_path.read_text()
    from gmlx.config import ModelCfg as MC
    monkeypatch.setattr(
        srv.discovery, "scan_dirs",
        lambda specs, dirs, **kw: [MC(id="newbie", path=str(lib / "newbie.gguf"))])
    rc = srv._cmd_sync(["--config", str(cfg_path), "--dry-run"])
    assert rc == 0
    assert cfg_path.read_text() == original           # nothing written
    out = capsys.readouterr().out
    assert "add:" in out and "newbie" in out and "dry run" in out


def test_sync_preserves_comments(monkeypatch, tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "keep.gguf").write_bytes(b"x")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "# my hand-written header\n"
        f"server:\n  model_dirs:\n    - {lib}\n"
        "models:\n  keep:\n    path: keep.gguf   # do not touch\n")
    from gmlx.config import ModelCfg as MC
    monkeypatch.setattr(
        srv.discovery, "scan_dirs",
        lambda specs, dirs, **kw: [MC(id="newbie", path=str(lib / "newbie.gguf"))])
    rc = srv._cmd_sync(["--config", str(cfg_path)])
    assert rc == 0
    text = cfg_path.read_text()
    assert "# my hand-written header" in text and "# do not touch" in text
    assert "newbie" in text


def test_sync_long_path_stays_on_one_line(monkeypatch, tmp_path):
    """ruamel's default 80-col wrap used to fold long spliced values (hf: cache
    refs) onto a continuation line; the writer now never wraps."""
    cfg_path, lib = _sync_config(tmp_path, "models:\n  keep:\n    path: keep.gguf\n")
    (lib / "keep.gguf").write_bytes(b"x")
    long_ref = ("hf:unsloth/Qwen3.6-27B-Instruct-GGUF-with-a-very-long-repo-name/"
                "Qwen3.6-27B-Instruct-UD-Q6_K_XL-part-00001-of-00002.gguf")
    from gmlx.config import ModelCfg as MC
    monkeypatch.setattr(
        srv.discovery, "scan_dirs",
        lambda specs, dirs, **kw: [MC(id="longish", path=long_ref)])
    rc = srv._cmd_sync(["--config", str(cfg_path)])
    assert rc == 0
    assert f"path: {long_ref}\n" in cfg_path.read_text()


def test_sync_splices_family_comment(monkeypatch, tmp_path):
    """A spliced entry with a detected family carries the same trailing
    model-card comment the scaffold writes."""
    cfg_path, lib = _sync_config(tmp_path, "models:\n  keep:\n    path: keep.gguf\n")
    (lib / "keep.gguf").write_bytes(b"x")
    from gmlx.config import ModelCfg as MC
    monkeypatch.setattr(
        srv.discovery, "scan_dirs",
        lambda specs, dirs, **kw: [MC(id="qw", path=str(lib / "qw.gguf"),
                                      family="qwen3.6")])
    rc = srv._cmd_sync(["--config", str(cfg_path)])
    assert rc == 0
    text = cfg_path.read_text()
    assert "qw:" in text
    assert "# qwen3.6: t=1.0 top_p=0.95 top_k=20" in text


def test_sync_inserts_new_entry_before_trailing_comment_block(monkeypatch, tmp_path):
    """The scaffold ends the models block with a commented talk:/assistant: hint
    block that ruamel anchors to the LAST entry; appending would land new
    entries after those hints. Splices go at the top of the block instead."""
    cfg_path, lib = _sync_config(
        tmp_path,
        "models:\n  keep:\n    path: keep.gguf\n"
        "# talk:\n#   stt: whisper-large\n#   tts: kokoro\n")
    (lib / "keep.gguf").write_bytes(b"x")
    from gmlx.config import ModelCfg as MC
    monkeypatch.setattr(
        srv.discovery, "scan_dirs",
        lambda specs, dirs, **kw: [MC(id="newbie", path=str(lib / "newbie.gguf"))])
    assert srv._cmd_sync(["--config", str(cfg_path)]) == 0
    text = cfg_path.read_text()
    assert text.index("newbie:") < text.index("# talk:")
    from gmlx.config import load_config
    assert set(load_config(cfg_path).models) == {"keep", "newbie"}


def test_sync_no_config_errors(monkeypatch, capsys):
    monkeypatch.setattr(srv, "default_config_paths", lambda: [])
    rc = srv._cmd_sync([])
    assert rc == 2
    assert "no config found" in capsys.readouterr().err


def test_sync_already_in_sync_no_write(monkeypatch, tmp_path, capsys):
    cfg_path, lib = _sync_config(tmp_path, "models:\n  keep:\n    path: keep.gguf\n")
    (lib / "keep.gguf").write_bytes(b"x")
    original = cfg_path.read_text()
    monkeypatch.setattr(srv.discovery, "scan_dirs", lambda specs, dirs, **kw: [])
    rc = srv._cmd_sync(["--config", str(cfg_path)])
    assert rc == 0
    assert cfg_path.read_text() == original
    assert "already in sync" in capsys.readouterr().out


def test_main_dispatches_sync(monkeypatch):
    called = {}
    monkeypatch.setattr(srv, "_cmd_sync",
                        lambda argv, prog=None: called.setdefault("sync", argv) or 0)
    srv.main(["sync-models", "--config", "x.yaml"])
    assert called["sync"] == ["--config", "x.yaml"]


# _reload_running: after init/sync rewrites a config, SIGHUP a server running from it
def test_reload_running_signals_and_reports(monkeypatch, capsys):
    from gmlx import lifecycle as lc
    seen = {}

    def _fake_reload(p):
        seen["path"] = p
        return [("127.0.0.1", 8080, 42)]

    monkeypatch.setattr(lc, "reload_config", _fake_reload)
    srv._reload_running("/some/cfg.yaml", skip=False)
    assert seen["path"].endswith("cfg.yaml")           # expanded to an abspath
    out = capsys.readouterr().out
    assert "reloaded the running server at 127.0.0.1:8080 (pid 42)" in out


def test_reload_running_skip_is_noop(monkeypatch):
    from gmlx import lifecycle as lc
    called = {"reload": False}
    monkeypatch.setattr(lc, "reload_config",
                        lambda p: called.__setitem__("reload", True) or [])
    srv._reload_running("/some/cfg.yaml", skip=True)   # --no-reload
    assert called["reload"] is False                   # never consulted lifecycle


def test_reload_running_never_raises(monkeypatch, capsys):
    from gmlx import lifecycle as lc

    def _boom(p):
        raise RuntimeError("signalling blew up")

    monkeypatch.setattr(lc, "reload_config", _boom)
    srv._reload_running("/some/cfg.yaml", skip=False)  # a reload hiccup must not fail
    assert capsys.readouterr().out == ""               # swallowed, nothing printed


def test_reload_running_no_servers_is_quiet(monkeypatch, capsys):
    from gmlx import lifecycle as lc
    monkeypatch.setattr(lc, "reload_config", lambda p: [])
    srv._reload_running("/some/cfg.yaml", skip=False)
    assert capsys.readouterr().out == ""               # nothing running -> no note


def test_sync_reloads_running_server(monkeypatch, tmp_path):
    """The end-to-end wiring: a real `sync-models` that writes the config calls
    `_reload_running` with the config path so a live server picks up the change."""
    cfg_path, lib = _sync_config(tmp_path, "models:\n  keep:\n    path: keep.gguf\n")
    (lib / "keep.gguf").write_bytes(b"x")
    from gmlx.config import ModelCfg as MC
    monkeypatch.setattr(
        srv.discovery, "scan_dirs",
        lambda specs, dirs, **kw: [MC(id="newbie", path=str(lib / "newbie.gguf"))])
    reloaded = {}
    monkeypatch.setattr(srv, "_reload_running",
                        lambda path, *, skip: reloaded.update(path=str(path), skip=skip))
    rc = srv._cmd_sync(["--config", str(cfg_path)])
    assert rc == 0
    assert reloaded["path"] == str(cfg_path) and reloaded["skip"] is False


def test_sync_dry_run_does_not_reload(monkeypatch, tmp_path):
    """A dry run writes nothing, so there is nothing for a server to reload."""
    cfg_path, lib = _sync_config(tmp_path, "models:\n  keep:\n    path: keep.gguf\n")
    (lib / "keep.gguf").write_bytes(b"x")
    from gmlx.config import ModelCfg as MC
    monkeypatch.setattr(
        srv.discovery, "scan_dirs",
        lambda specs, dirs, **kw: [MC(id="newbie", path=str(lib / "newbie.gguf"))])
    called = {"reload": False}
    monkeypatch.setattr(srv, "_reload_running",
                        lambda *a, **k: called.__setitem__("reload", True))
    rc = srv._cmd_sync(["--config", str(cfg_path), "--dry-run"])
    assert rc == 0
    assert called["reload"] is False                   # no write => no reload


# per-request timing log (server_patches)
def _access_record(path):
    """A uvicorn access LogRecord: args = (client, method, full_path, ver, status)."""
    return logging.LogRecord(
        "uvicorn.access", logging.INFO, "", 0,
        '%s - "%s %s HTTP/%s" %s', ("127.0.0.1:1", "GET", path, "1.1", 200), None)


def test_access_noise_filter_drops_polls_and_timed_paths():
    f = sp_obs._AccessNoiseFilter()
    assert f.filter(_access_record("/health")) is False        # menu-bar poll noise
    assert f.filter(_access_record("/v1/metrics?x=1")) is False  # query stripped first
    assert f.filter(_access_record("/v1/chat/completions")) is False  # we log it richly
    assert f.filter(_access_record("/v1/models")) is False     # menu-bar/web-UI poll
    assert f.filter(_access_record("/v1/models?x=1")) is False
    assert f.filter(_access_record("/v1/reload")) is True


def test_access_noise_filter_keeps_malformed_record():
    # A record without the expected 5-tuple is passed through, never dropped/crashed.
    rec = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, "plain", None, None)
    assert sp_obs._AccessNoiseFilter().filter(rec) is True


def test_uvicorn_log_config_timestamps_and_filters():
    cfg = sp.uvicorn_log_config()
    assert "%(asctime)s" in cfg["formatters"]["access"]["fmt"]
    assert "%(asctime)s" in cfg["formatters"]["default"]["fmt"]
    assert "kq_access_noise" in cfg["filters"]
    assert "kq_access_noise" in cfg["loggers"]["uvicorn.access"]["filters"]
    # The filter's dotted "()" factory path must resolve, or serve dies at
    # startup with "Unable to configure filter 'kq_access_noise'" (dictConfig
    # imports it; a module move breaks the string without any import error).
    import importlib

    factory = cfg["filters"]["kq_access_noise"]["()"]
    mod, _, cls = factory.rpartition(".")
    assert callable(getattr(importlib.import_module(mod), cls))


def test_uvicorn_log_config_level_reaches_app_loggers():
    """--log-level must govern the whole server, not just uvicorn's own
    loggers: the level lands on the gmlx and mlx_vlm dictConfig entries
    (trace maps to DEBUG - stdlib logging has no TRACE tier)."""
    cfg = sp.uvicorn_log_config()
    assert cfg["loggers"]["gmlx"]["level"] == "INFO"
    assert cfg["loggers"]["mlx_vlm"]["level"] == "INFO"
    cfg = sp.uvicorn_log_config("debug")
    assert cfg["loggers"]["gmlx"]["level"] == "DEBUG"
    assert cfg["loggers"]["mlx_vlm"]["level"] == "DEBUG"
    cfg = sp.uvicorn_log_config("error")
    assert cfg["loggers"]["gmlx"]["level"] == "ERROR"
    cfg = sp.uvicorn_log_config("trace")
    assert cfg["loggers"]["gmlx"]["level"] == "DEBUG"



def test_format_timing_line_has_fields_and_handles_missing():
    line = sp_obs._format_timing_line({
        "timestamp_unix": 0, "endpoint": "/chat/completions", "model": "qwen3.6-27b",
        "prompt_tokens": 812, "generated_tokens": 240, "ttft_s": 0.34,
        "prefill_tok_s": 910.2, "decode_tok_s": 22.4, "request_elapsed_s": 11.0,
    })
    assert line.startswith("[req] ")
    for tok in ("/chat/completions", "qwen3.6-27b", "prompt=812", "gen=240",
                "ttft=0.34s", "prefill=910t/s", "decode=22.4t/s", "total=11.00s"):
        assert tok in line
    # Missing timing fields render as '-', not a crash.
    sparse = sp_obs._format_timing_line({"endpoint": "/v1/messages", "model": "m"})
    assert "ttft=-s" in sparse and "prompt=0" in sparse


def test_format_timing_line_appends_nondefault_finish():
    line = sp_obs._format_timing_line({"endpoint": "/x", "model": "m",
                                   "finish_reason": "length"})
    assert "finish=length" in line
    stopped = sp_obs._format_timing_line({"endpoint": "/x", "model": "m",
                                      "finish_reason": "stop"})
    assert "finish=" not in stopped                    # the common case stays terse


def _install_timing_on_fake(monkeypatch):
    """Swap mlx_vlm's metrics store for a fresh fake class (new each call so the
    in-place wrap never leaks across tests), then install the timing wrap on it."""
    class FakeStore:
        def __init__(self):
            self.ok = []
            self.fail = []

        def record_success(self, envelope):
            self.ok.append(envelope)

        def record_failure(self, *, endpoint, model, stream, error):
            self.fail.append((endpoint, model, error))

    gen = importlib.import_module("mlx_vlm.server.generation")
    monkeypatch.setattr(gen, "ServerMetricsStore", FakeStore)
    sp.install_request_timing_log()
    return FakeStore


def test_request_timing_log_prints_on_success(monkeypatch, capsys):
    store_cls = _install_timing_on_fake(monkeypatch)
    s = store_cls()
    s.record_success({"endpoint": "/chat/completions", "model": "m",
                      "generated_tokens": 5, "request_elapsed_s": 1.0})
    assert len(s.ok) == 1                              # original still ran
    out = capsys.readouterr().out
    assert "[req]" in out and "/chat/completions" in out and "gen=5" in out


def test_request_timing_log_prints_on_failure(monkeypatch, capsys):
    store_cls = _install_timing_on_fake(monkeypatch)
    s = store_cls()
    s.record_failure(endpoint="/v1/messages", model="m", stream=False, error="boom")
    assert s.fail == [("/v1/messages", "m", "boom")]   # original still ran
    out = capsys.readouterr().out
    assert "failed boom" in out and "/v1/messages" in out


def test_request_timing_log_is_idempotent(monkeypatch):
    store_cls = _install_timing_on_fake(monkeypatch)
    wrapped = store_cls.record_success
    sp.install_request_timing_log()                    # second call is a no-op
    assert store_cls.record_success is wrapped         # not double-wrapped


def test_request_timing_log_survives_bad_envelope(monkeypatch, capsys):
    # _format_timing_line raising must not propagate out of record_success.
    store_cls = _install_timing_on_fake(monkeypatch)
    monkeypatch.setattr(sp_obs, "_format_timing_line",
                        lambda e: (_ for _ in ()).throw(RuntimeError("nope")))
    s = store_cls()
    s.record_success({"endpoint": "/x"})               # must not raise
    assert len(s.ok) == 1                              # original still recorded


def test_sync_from_hf_cache_adds_and_sets_flag(monkeypatch, tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(f"server:\n  model_dirs:\n    - {lib}\nmodels: {{}}\n")
    monkeypatch.setattr(srv.discovery, "scan_dirs", lambda *a, **k: [])
    monkeypatch.setattr(
        srv.discovery, "scan_hf_cache",
        lambda **kw: [ModelCfg(id="cached", path="hf:org/repo/c-Q4_K_M.gguf")])
    rc = srv._cmd_sync(["--config", str(cfg_path), "--from-hf-cache"])
    assert rc == 0
    from gmlx.config import load_config
    cfg = load_config(cfg_path)
    assert cfg.hf_cache is True                       # flipped on for cache entries
    assert cfg.models["cached"].path == "hf:org/repo/c-Q4_K_M.gguf"


def test_sync_from_hf_cache_implied_by_config_flag(monkeypatch, tmp_path):
    """A config already carrying hf_cache: true syncs the cache without the flag."""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("server:\n  hf_cache: true\nmodels: {}\n")
    seen = {}

    def _scan(**kw):
        seen["called"] = True
        return [ModelCfg(id="c", path="hf:o/r/c-Q4_K_M.gguf")]

    monkeypatch.setattr(srv.discovery, "scan_hf_cache", _scan)
    rc = srv._cmd_sync(["--config", str(cfg_path)])    # no --from-hf-cache flag
    assert rc == 0
    assert seen.get("called") is True                  # cache scanned anyway


# _serve bind policy (api-key / --no-auth) + SIGHUP reload wiring
def _stub_serving_stack(monkeypatch):
    """Stub every install/registration _serve performs, plus uvicorn.run, so the
    policy logic runs without touching mlx-vlm state or binding a port.
    Returns the call-recorder dict."""
    import signal

    import uvicorn

    import gmlx.residency as residency_mod
    import gmlx.server_patches as sp_mod
    import gmlx.server_bridge_vlm as serving_mod

    calls = {}
    monkeypatch.setattr(serving_mod, "register_resolved_models", lambda cfg: None)
    monkeypatch.setattr(serving_mod, "resolved_models", lambda: {})
    monkeypatch.setattr(serving_mod, "install_gguf_server_bridge",
                        lambda: calls.__setitem__("bridge", True))
    monkeypatch.setattr(residency_mod, "install_gguf_residency_pool",
                        lambda **kw: calls.__setitem__("pool", kw))
    monkeypatch.setattr(sp_mod, "install_server_patches",
                        lambda cfg, reload_fn=None:
                        calls.__setitem__("patches_cfg", cfg))
    monkeypatch.setattr(sp_mod, "spawn_preload_warm",
                        lambda mid, extras=():
                        (calls.__setitem__("preload_warm", mid),
                         calls.__setitem__("preload_extras", list(extras))))
    monkeypatch.setattr(uvicorn, "run",
                        lambda *a, **kw: calls.__setitem__("uvicorn", kw))
    monkeypatch.setattr(signal, "signal",
                        lambda num, fn: calls.__setitem__("signal", (num, fn)))
    return calls


def _one_model_cfg(**server_kw):
    return ServerCfg(models={"m": ModelCfg(id="m", path="/m.gguf")}, **server_kw)


def test_serve_refuses_nonloopback_without_key(monkeypatch, capsys):
    calls = _stub_serving_stack(monkeypatch)
    monkeypatch.delenv("GMLX_API_KEY", raising=False)
    rc = srv._serve(_one_model_cfg(host="0.0.0.0"), _ns(), None)
    assert rc == 2
    assert "uvicorn" not in calls                     # never reached the bind
    err = capsys.readouterr().err
    # config is the sole key source now: the refusal points at `server.api_key`.
    assert "server.api_key" in err and "--no-auth" in err


def test_serve_nonloopback_no_auth_starts_with_warning(monkeypatch, capsys):
    calls = _stub_serving_stack(monkeypatch)
    monkeypatch.delenv("GMLX_API_KEY", raising=False)
    rc = srv._serve(_one_model_cfg(host="0.0.0.0"), _ns(no_auth=True), None)
    assert rc == 0
    assert "uvicorn" in calls
    assert "warning: binding" in capsys.readouterr().out


def test_serve_loopback_without_key_ok(monkeypatch, capsys):
    calls = _stub_serving_stack(monkeypatch)
    monkeypatch.delenv("GMLX_API_KEY", raising=False)
    rc = srv._serve(_one_model_cfg(), _ns(), None)
    assert rc == 0
    assert "uvicorn" in calls
    assert "WARNING" not in capsys.readouterr().out


def test_serve_api_key_config_only(monkeypatch, capsys):
    # `server.api_key` in the config is the sole server-side key source. A config key
    # satisfies the non-loopback policy ...
    calls = _stub_serving_stack(monkeypatch)
    monkeypatch.delenv("GMLX_API_KEY", raising=False)
    cfg = _one_model_cfg(host="0.0.0.0", api_key="from-config")
    assert srv._serve(cfg, _ns(), None) == 0
    assert cfg.api_key == "from-config"
    assert "uvicorn" in calls


def test_serve_env_api_key_is_not_a_server_source(monkeypatch):
    # The GMLX_API_KEY env var is a client-side convenience only - it no longer
    # satisfies the server's non-loopback bind policy (breaking change: config-only).
    _stub_serving_stack(monkeypatch)
    monkeypatch.setenv("GMLX_API_KEY", "from-env")
    cfg = _one_model_cfg(host="0.0.0.0")
    assert srv._serve(cfg, _ns(), None) == 2
    assert cfg.api_key is None


def test_serve_passes_resolved_bind_to_patches(monkeypatch):
    calls = _stub_serving_stack(monkeypatch)
    cfg = _one_model_cfg(host="0.0.0.0", api_key="k")
    rc = srv._serve(cfg, _ns(host="127.0.0.1", port=9999), None)
    assert rc == 0
    patched = calls["patches_cfg"]
    assert (patched.host, patched.port) == ("127.0.0.1", 9999)
    assert calls["uvicorn"]["host"] == "127.0.0.1"
    assert calls["uvicorn"]["port"] == 9999


def test_serve_preload_is_backgrounded_not_blocking(monkeypatch):
    # A sole/default/pinned model is preloaded OFF the startup path: the env that
    # makes mlx-vlm's lifespan block the bind is popped before uvicorn.run, and the
    # model is warmed in the background instead (so /health answers immediately).
    import os
    calls = _stub_serving_stack(monkeypatch)
    monkeypatch.delenv("MLX_VLM_PRELOAD_MODEL", raising=False)
    rc = srv._serve(_one_model_cfg(), _ns(), None)
    assert rc == 0
    assert "uvicorn" in calls
    assert calls.get("preload_warm") == "m"          # background warm fired
    assert "MLX_VLM_PRELOAD_MODEL" not in os.environ  # blocking-lifespan env popped


def test_serve_preload_extras_from_config(monkeypatch):
    # defaults.preload warms the named models after the primary; the primary
    # (default-model precedence) is excluded from the extras.
    calls = _stub_serving_stack(monkeypatch)
    cfg = ServerCfg(
        models={"m": ModelCfg(id="m", path="/m.gguf"),
                "e1": ModelCfg(id="e1", path="/e1.gguf"),
                "e2": ModelCfg(id="e2", path="/e2.gguf")},
        defaults=ServerDefaults(model="m", preload="all"),
    )
    assert srv._serve(cfg, _ns(), None) == 0
    assert calls.get("preload_warm") == "m"
    assert calls.get("preload_extras") == ["e1", "e2"]


def test_serve_sighup_triggers_reload_fn(monkeypatch, capsys):
    import signal as signal_mod

    calls = _stub_serving_stack(monkeypatch)
    reloads = []
    rc = srv._serve(_one_model_cfg(), _ns(),
                    lambda: reloads.append(1) or {"models": 1})
    assert rc == 0
    num, handler = calls["signal"]
    assert num == signal_mod.SIGHUP
    handler(num, None)
    assert reloads == [1]
    out = capsys.readouterr().out
    assert "SIGHUP config reload" in out and "kill -HUP" in out


def test_serve_no_sighup_without_reload_fn(monkeypatch):
    calls = _stub_serving_stack(monkeypatch)
    assert srv._serve(_one_model_cfg(), _ns(), None) == 0
    assert "signal" not in calls


# token-queue timeout: config value drives mlx-vlm's per-request env knob
def test_serve_sets_token_queue_timeout_env(monkeypatch, capsys):
    _stub_serving_stack(monkeypatch)
    monkeypatch.delenv("MLX_VLM_TOKEN_QUEUE_TIMEOUT", raising=False)  # undone on teardown
    cfg = _one_model_cfg(token_queue_timeout_s=1800.0)
    assert srv._serve(cfg, _ns(), None) == 0
    import os
    assert os.environ["MLX_VLM_TOKEN_QUEUE_TIMEOUT"] == "1800.0"
    assert "token-queue timeout: 1800s" in capsys.readouterr().out


def test_serve_token_queue_timeout_zero_disables(monkeypatch, capsys):
    _stub_serving_stack(monkeypatch)
    monkeypatch.delenv("MLX_VLM_TOKEN_QUEUE_TIMEOUT", raising=False)
    assert srv._serve(_one_model_cfg(token_queue_timeout_s=0.0), _ns(), None) == 0
    import os
    assert os.environ["MLX_VLM_TOKEN_QUEUE_TIMEOUT"] == "0.0"      # <=0 => wait forever
    assert "waits forever" in capsys.readouterr().out


# prefill step size: flag > config > exported env; sets the per-request env knob
def test_serve_sets_prefill_step_env_from_config(monkeypatch, capsys):
    _stub_serving_stack(monkeypatch)
    monkeypatch.delenv("PREFILL_STEP_SIZE", raising=False)
    assert srv._serve(_one_model_cfg(prefill_step_size=512), _ns(), None) == 0
    import os
    assert os.environ["PREFILL_STEP_SIZE"] == "512"
    assert "prefill step size: 512 tokens" in capsys.readouterr().out


def test_serve_prefill_step_flag_wins_over_config(monkeypatch, capsys):
    _stub_serving_stack(monkeypatch)
    monkeypatch.delenv("PREFILL_STEP_SIZE", raising=False)
    cfg = _one_model_cfg(prefill_step_size=1024)
    assert srv._serve(cfg, _ns(prefill_step_size=512), None) == 0
    import os
    assert os.environ["PREFILL_STEP_SIZE"] == "512"


def test_serve_prefill_step_absent_leaves_env(monkeypatch):
    _stub_serving_stack(monkeypatch)
    monkeypatch.setenv("PREFILL_STEP_SIZE", "256")   # exported env wins when unset
    assert srv._serve(_one_model_cfg(), _ns(), None) == 0
    import os
    assert os.environ["PREFILL_STEP_SIZE"] == "256"


def test_serve_prefill_step_nonpositive_ignored(monkeypatch, capsys):
    _stub_serving_stack(monkeypatch)
    monkeypatch.delenv("PREFILL_STEP_SIZE", raising=False)
    assert srv._serve(_one_model_cfg(prefill_step_size=0), _ns(), None) == 0
    import os
    assert "PREFILL_STEP_SIZE" not in os.environ
    assert "ignoring non-positive prefill step size" in capsys.readouterr().out


def test_serve_no_token_queue_timeout_defaults_1800(monkeypatch, capsys):
    # mlx-vlm's own 600s default is shorter than a deep-context dense prefill;
    # with no config value and no exported env we raise it to 1800s.
    _stub_serving_stack(monkeypatch)
    monkeypatch.delenv("MLX_VLM_TOKEN_QUEUE_TIMEOUT", raising=False)
    assert srv._serve(_one_model_cfg(), _ns(), None) == 0          # field defaults None
    import os
    assert os.environ["MLX_VLM_TOKEN_QUEUE_TIMEOUT"] == "1800"
    assert "token-queue timeout: 1800s (default)" in capsys.readouterr().out


def test_serve_no_token_queue_timeout_respects_exported_env(monkeypatch):
    _stub_serving_stack(monkeypatch)
    monkeypatch.setenv("MLX_VLM_TOKEN_QUEUE_TIMEOUT", "45")
    assert srv._serve(_one_model_cfg(), _ns(), None) == 0
    import os
    assert os.environ["MLX_VLM_TOKEN_QUEUE_TIMEOUT"] == "45"       # user env wins


# _resolve_service: missing file degrades, malformed value fails fast
def test_resolve_service_missing_file_degrades(capsys):
    from gmlx.config import MissingModelFile

    def missing(_v, _dirs):
        raise MissingModelFile("model path 'x.gguf' not found under "
                               "model_dirs: ['/m']")

    assert srv._resolve_service("server.embeddings", missing,
                                "x.gguf", ["/m"]) is None
    err = capsys.readouterr().err
    assert "server.embeddings disabled" in err and "x.gguf" in err


def test_resolve_service_malformed_value_fails_fast_naming_key():
    from gmlx.config import ConfigError

    def bad(_v, _dirs):
        raise ConfigError("must be a Qwen3-Reranker GGUF")

    with pytest.raises(ConfigError, match=r"server\.rerank: must be"):
        srv._resolve_service("server.rerank", bad, 5, [])
    ok = srv._resolve_service("k", lambda v, d: "/abs/x.gguf", "x", [])
    assert ok == "/abs/x.gguf"


def test_resolve_service_uses_model_dirs(tmp_path):
    """Regression shape (f2244c7): the shipped bug was a resolver that dropped
    model_dirs, so a model_dirs-relative service entry never resolved. Drive
    _resolve_service with the REAL resolvers and a relative path."""
    import os

    from gmlx.embeddings import resolve_embeddings_model
    from gmlx.rerank import resolve_rerank_model

    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "e.gguf").write_bytes(b"GGUF")
    expected = str((sub / "e.gguf").resolve())

    got = srv._resolve_service("server.embeddings", resolve_embeddings_model,
                               "sub/e.gguf", [str(tmp_path)])
    assert got == expected and os.path.isabs(got)

    got_r = srv._resolve_service("server.rerank", resolve_rerank_model,
                                 "sub/e.gguf", [str(tmp_path)])
    assert got_r == expected


# register_downloads: `gmlx pull` auto-register (a mini sync-models scoped
# to the downloaded paths)
def _reg_config(tmp_path, body="models:\n  old:\n    path: old.gguf\n"):
    lib = tmp_path / "lib"
    lib.mkdir(exist_ok=True)
    (lib / "old.gguf").write_bytes(b"x")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(f"server:\n  model_dirs:\n    - {lib}\n" + body)
    return cfg_path, lib


def test_register_downloads_adds_entry_and_reloads(monkeypatch, tmp_path, capsys):
    cfg_path, lib = _reg_config(tmp_path)
    sub = lib / "org__repo-GGUF"
    sub.mkdir()
    new = sub / "new.gguf"
    new.write_bytes(b"x")
    from gmlx.config import ModelCfg as MC
    monkeypatch.setattr(srv.discovery, "scan_dirs",
                        lambda specs, dirs, **kw: [MC(id="new-q4", path=str(new))])
    reloaded = []
    monkeypatch.setattr(srv, "_reload_running",
                        lambda path, skip: reloaded.append(path))
    srv.register_downloads([str(new)], str(cfg_path))
    from gmlx.config import load_config
    cfg = load_config(cfg_path)
    assert set(cfg.models) == {"old", "new-q4"}
    assert cfg.models["new-q4"].path == "org__repo-GGUF/new.gguf"  # relative
    assert reloaded == [cfg_path]
    assert "registered new-q4" in capsys.readouterr().out


def test_register_downloads_skips_outside_model_dirs(monkeypatch, tmp_path):
    cfg_path, _lib = _reg_config(tmp_path)
    elsewhere = tmp_path / "elsewhere" / "x.gguf"
    elsewhere.parent.mkdir()
    elsewhere.write_bytes(b"x")
    monkeypatch.setattr(srv.discovery, "scan_dirs",
                        lambda *args, **kw: (_ for _ in ()).throw(
                            AssertionError("must not scan")))
    before = cfg_path.read_text()
    srv.register_downloads([str(elsewhere)], str(cfg_path))
    assert cfg_path.read_text() == before          # untouched, no scan


def test_register_downloads_quiet_on_repull_and_no_config(monkeypatch, tmp_path,
                                                          capsys):
    # already-configured file (a re-pull) -> no write, no output
    cfg_path, lib = _reg_config(tmp_path)
    monkeypatch.setattr(srv.discovery, "scan_dirs", lambda *a, **kw: [])
    before = cfg_path.read_text()
    srv.register_downloads([str(lib / "old.gguf")], str(cfg_path))
    assert cfg_path.read_text() == before
    # no config at all -> silent no-op
    srv.register_downloads([str(lib / "old.gguf")], str(tmp_path / "nope.yaml"))
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""


def test_register_downloads_never_fails_pull(monkeypatch, tmp_path, capsys):
    cfg_path, lib = _reg_config(tmp_path)
    monkeypatch.setattr(srv.discovery, "scan_dirs",
                        lambda *a, **kw: (_ for _ in ()).throw(
                            RuntimeError("scanner exploded")))
    srv.register_downloads([str(lib / "old.gguf")], str(cfg_path))  # no raise
    err = capsys.readouterr().err
    assert "could not register" in err and "sync-models" in err


def test_sync_models_dir_override_lands_in_model_dirs(monkeypatch, tmp_path):
    """--models-dir entries are written relative to the override dir; the dir
    must join server.model_dirs or the new entries never resolve."""
    cfg_path, lib = _sync_config(tmp_path, "models: {}\n")
    other = tmp_path / "other"
    other.mkdir()
    (other / "newbie.gguf").write_bytes(b"x")
    from gmlx.config import ModelCfg as MC
    monkeypatch.setattr(
        srv.discovery, "scan_dirs",
        lambda specs, dirs, **kw: [MC(id="newbie",
                                      path=str(other / "newbie.gguf"))])
    rc = srv._cmd_sync(["--config", str(cfg_path),
                        "--models-dir", str(other), "--no-reload"])
    assert rc == 0
    from gmlx.config import load_config, resolve_path
    cfg = load_config(cfg_path)
    assert str(other) in cfg.model_dirs
    rp = resolve_path(cfg.models["newbie"].path, cfg.model_dirs)
    assert rp == str(other / "newbie.gguf")


def test_bg_serve_args_forwards_moe_expert_mass():
    """An explicit --moe-expert-mass must survive the background re-exec;
    dropping it would serve the child at the trained (lossless) fan-out."""
    import argparse
    ap = argparse.ArgumentParser()
    srv._add_serve_args(ap)
    a = ap.parse_args(["--stream-experts", "--moe-expert-mass", "0.9"])
    out = srv._bg_serve_args(a, None)
    assert out[out.index("--moe-expert-mass") + 1] == "0.9"
    assert "--moe-expert-mass" not in srv._bg_serve_args(ap.parse_args([]), None)
    """An explicit --no-decode-feeder/--no-prefill-feeder must survive the
    background re-exec; dropping it re-enables the feeder in the child."""
    import argparse
    ap = argparse.ArgumentParser()
    srv._add_serve_args(ap)
    for flags, want in [
        (["--no-decode-feeder"], ["--no-decode-feeder"]),
        (["--prefill-feeder"], ["--prefill-feeder"]),
        ([], []),
    ]:
        a = ap.parse_args(flags)
        out = srv._bg_serve_args(a, None)
        for w in want:
            assert w in out
        if not flags:
            assert not any("feeder" in x for x in out)
