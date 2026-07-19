#!/usr/bin/env python3
"""``gmlx doctor`` - one-pass environment self-check.

Each check function returns ``{"name", "status", "detail"}`` with status
PASS / WARN / FAIL / SKIP; ``cmd_doctor`` prints the aligned report and exits
1 when anything FAILs. Checks are module-level seams so tests can force any
outcome. Network-free by default; ``--deep`` additionally header-reads every
configured model.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import warnings

from .textfmt import plural_s as _s


def _check(name: str, status: str, detail: str) -> dict:
    return {"name": name, "status": status, "detail": detail}


def check_runtime() -> dict:
    try:
        import mlx.core as mx
    except ImportError as e:
        return _check("runtime", "FAIL", f"mlx not importable: {e}")
    try:
        import mlx_kquant  # noqa: F401
    except ImportError as e:
        return _check("runtime", "FAIL",
                      f"mlx-kquant not importable: {e} (pip install mlx-kquant)")
    from importlib.metadata import PackageNotFoundError, version
    vers = []
    for dist in ("mlx", "mlx-kquant", "mlx-lm", "gguf"):
        try:
            vers.append(f"{dist} {version(dist)}")
        except PackageNotFoundError:
            vers.append(f"{dist} ?")
    if not mx.metal.is_available():
        return _check("runtime", "WARN",
                      ", ".join(vers) + ", metal unavailable (CPU only)")
    return _check("runtime", "PASS", ", ".join(vers) + ", metal ok")


def check_kernels() -> dict:
    try:
        import mlx_kquant
    except ImportError:
        return _check("kernels", "SKIP", "mlx-kquant not importable")
    missing = [k for k in ("sdpa_vector", "sdpa_decode_gqa")
               if not hasattr(mlx_kquant, k)]
    if missing:
        return _check("kernels", "WARN",
                      "missing " + ", ".join(missing)
                      + " (falls back to MLX's default SDPA)")
    return _check("kernels", "PASS", "sdpa_vector, sdpa_decode_gqa")


def check_config(config_path):
    """Returns ``(check, cfg, path)`` - cfg/path are None when unloadable."""
    from . import config as cfgmod
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            cfg, path = cfgmod.load_cli_config(config_path)
        except cfgmod.ConfigError as e:
            return _check("config", "FAIL", str(e)), None, None
    if cfg is None:
        searched = ", ".join(str(p) for p in cfgmod.default_config_paths())
        return (_check("config", "WARN",
                       f"no config found (gmlx init); searched: {searched}"),
                None, None)
    if caught:
        first = str(caught[0].message)
        return (_check("config", "WARN",
                       f"loaded {path} with {len(caught)} "
                       f"warning{_s(len(caught))}: {first}"),
                cfg, path)
    return _check("config", "PASS", path), cfg, path


def check_models(cfg, *, deep: bool = False) -> dict:
    if cfg is None:
        return _check("models", "SKIP", "no config")
    if not cfg.models:
        if cfg.discover:
            return _check("models", "PASS",
                          "none configured (discover: scan active)")
        return _check("models", "WARN",
                      "no models: add entries under models: or a discover: scan")
    from .config import ConfigError, resolve_path
    from .manage import _shard_names
    misses: list[str] = []
    for mid, m in cfg.models.items():
        for label, p in (("path", m.path), ("mmproj", m.mmproj),
                         ("draft", m.draft_gguf), ("adapter", m.adapter)):
            if not p:
                continue
            try:
                rp = resolve_path(p, cfg.model_dirs)
            except ConfigError:
                misses.append(f"{mid}: missing {label} {p}")
                continue
            if label == "path":
                d, base = os.path.split(rp)
                gaps = [n for n in _shard_names(base)
                        if not os.path.exists(os.path.join(d, n))]
                if gaps:
                    misses.append(
                        f"{mid}: missing shard(s) {', '.join(gaps[:3])}")
                elif deep:
                    from .preflight import preflight
                    try:
                        preflight(rp)
                    except Exception as e:      # noqa: BLE001 - report, not raise
                        misses.append(f"{mid}: {e}")
    if misses:
        more = f" (+{len(misses) - 3} more)" if len(misses) > 3 else ""
        return _check("models", "FAIL", "; ".join(misses[:3]) + more
                      + " (gmlx pull to re-download, or gmlx sync-models to "
                        "drop gone entries)")
    # Dangling aliases / defaults.model are hard ConfigErrors at load time,
    # so they surface through the config check, not here.
    n = len(cfg.models)
    detail = f"{n} model{_s(n)}, all paths present"
    if deep:
        detail += ", headers ok"
    return _check("models", "PASS", detail)


def check_services(cfg):
    """One row over the configured service models (embeddings / rerank / stt /
    tts), or ``None`` when the config uses none. GGUF-backed services (and any
    local-path value) are resolved and stat'd - the server degrades a missing
    service GGUF at runtime, so doctor must see it too ("services in one
    pass"). Alias / HF-repo values have nothing to stat and just count."""
    if cfg is None:
        return None
    from .server_patches.routes import _service_file_on_disk
    svcs = [("embeddings", cfg.embeddings), ("rerank", cfg.rerank),
            ("stt", cfg.stt), ("tts", cfg.tts)]
    svcs = [(n, v) for n, v in svcs if v]
    if not svcs:
        return None
    misses = []
    for name, value in svcs:
        v = str(value)
        gguf_like = v.endswith(".gguf") or v.startswith("hf:")
        # Path-looking values only: a bare HF repo id (org/name) also
        # contains a separator but is a repo reference, not a local dir.
        local_dir = (v.startswith(("/", "~", "./", "../"))
                     or os.path.isdir(os.path.expanduser(v)))
        if gguf_like:
            if not _service_file_on_disk(v, cfg.model_dirs):
                misses.append(
                    f"{name}: {v} missing on disk - restore the file, update "
                    f"the config, or run `gmlx sync-models`")
        elif local_dir and not os.path.isdir(os.path.expanduser(v)):
            misses.append(f"{name}: model directory {v} not found")
    if misses:
        return _check("services", "FAIL", "; ".join(misses))
    return _check("services", "PASS",
                  ", ".join(n for n, _ in svcs) + " configured, files present")


def check_server() -> dict:
    from . import lifecycle
    runs = lifecycle.list_runs()
    if not runs:
        return _check("server", "SKIP",
                      "no background server (gmlx serve starts one)")
    parts: list[str] = []
    stale: list[str] = []
    status = "PASS"
    for run in runs:
        host, port, pid = run.get("host"), run.get("port"), run.get("pid")
        if not lifecycle.identity_ok(run):
            stale.append(f"{host}:{port}")
            status = "WARN"
        elif not lifecycle._health_ok(host, port):
            parts.append(f"{host}:{port} (pid {pid}) not answering /health")
            status = "WARN"
        else:
            parts.append(f"running at {host}:{port} (pid {pid})")
    if stale:
        shown = ", ".join(stale[:4]) + (", ..." if len(stale) > 4 else "")
        parts.append(f"{len(stale)} stale run file{_s(len(stale))} [{shown}] "
                     "(gmlx stop cleans up)")
    return _check("server", status, "; ".join(parts))


def _agent_plists() -> list:
    """The installed gmlx LaunchAgent plists. Module-level seam for tests."""
    from pathlib import Path
    return sorted(
        Path("~/Library/LaunchAgents").expanduser().glob("com.gmlx.*.plist"))


def check_agents():
    """None off macOS or with no gmlx LaunchAgent plists installed. Reports
    each agent's launchd load state: an installed-but-unloaded agent silently
    does nothing at login, so it gets a WARN with the re-load step."""
    if sys.platform != "darwin":
        return None
    from . import lifecycle
    plists = _agent_plists()
    if not plists:
        return None
    loaded, unloaded = [], []
    for pp in plists:
        label = pp.stem
        (loaded if lifecycle.agent_loaded(label) else unloaded).append(label)
    if unloaded:
        return _check(
            "launch agents", "WARN",
            "installed but not loaded: " + ", ".join(unloaded)
            + " (gmlx service status shows details; gmlx service install "
              "re-loads, gmlx service uninstall removes)")
    return _check("launch agents", "PASS",
                  ", ".join(loaded) + f" loaded ({len(loaded)} "
                  f"agent{_s(len(loaded))}; gmlx service status for details)")


def check_launcher():
    """None off macOS. Detached serve / menubar children exec a renamed copy
    of the interpreter (procname.py) so they show as "gmlx"; an interpreter
    swap under the venv can strand the copy (e.g. a relative-linked
    python-build-standalone binary), and then every detached start dies in
    dyld before any Python runs. Prove the copy executes."""
    if sys.platform != "darwin":
        return None
    from . import procname
    stub = procname.named_python()
    if stub is None:
        return _check("launcher", "WARN",
                      "no interpreter copy (detached starts fall back to "
                      'sys.executable and show as "Python")')
    try:
        p = subprocess.run([stub, "-c", "pass"], env=procname.child_env(),
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        return _check("launcher", "FAIL", f"{stub} won't run: {e}")
    if p.returncode != 0:
        err = (p.stderr or "").strip().splitlines()
        why = err[0] if err else f"exit code {p.returncode}"
        return _check("launcher", "FAIL", f"{stub} won't run: {why}")
    return _check("launcher", "PASS", stub)


def _running_configs(primary_path) -> list:
    """(cfg, path) for each live server whose runfile records a --config
    other than the file doctor is already checking. Extras and ffmpeg are
    properties of this machine, but the features that *need* them follow
    whatever config each server was actually started with - a
    `serve --config other.yaml` can enable stt/tts that the default-location
    config leaves commented out."""
    from . import config as cfgmod
    from . import lifecycle
    seen = set()
    if primary_path:
        seen.add(os.path.abspath(os.path.expanduser(str(primary_path))))
    out = []
    for run in lifecycle.list_runs():
        p = run.get("config_abspath")
        if not p:
            continue
        ap = os.path.abspath(os.path.expanduser(p))
        if ap in seen or not lifecycle.identity_ok(run):
            continue                    # duplicate, or a stale runfile
        seen.add(ap)
        try:
            out.append((cfgmod.load_config(ap), ap))
        except Exception:               # noqa: BLE001 - unreadable config:
            continue                    # the server row already covers it
    return out


def _needed_extras(cfg) -> list[str]:
    """Extras the config's features require (empty without a config)."""
    if cfg is None:
        return []
    from .config import TalkCfg
    need = []
    if cfg.stt:
        need.append("stt")
    if cfg.tts:
        need.append("tts")
    if cfg.talk != TalkCfg():           # any talk: key set in the YAML
        need.append("talk")
    if cfg.talk.brain == "assistant" or cfg.assistants:
        need.append("assistant")
    return need


def check_extras(cfg, running=()):
    """None when no configured feature needs an extra (row omitted).
    ``running`` is :func:`_running_configs` output; extras those configs need
    join the check, attributed to the server config that wants them."""
    need = list(_needed_extras(cfg))
    origin = {}
    for rcfg, rpath in running:
        for x in _needed_extras(rcfg):
            if x not in need:
                need.append(x)
                origin[x] = rpath
    if not need:
        return None
    from . import extras
    missing = [x for x in need if not extras.extra_installed(x)]
    if missing:
        def label(x):
            mods = extras.missing_extra_modules(x)
            out = f"{x} ({', '.join(mods)})" if mods else x
            if x in origin:
                out += f" [server config {origin[x]}]"
            return out
        pips = "; ".join(f'pip install "gmlx[{x}]"' for x in missing)
        return _check("extras", "FAIL",
                      "configured but not installed: "
                      f"{', '.join(label(x) for x in missing)} ({pips})")
    return _check("extras", "PASS", ", ".join(need) + " installed")


def check_ffmpeg(cfg, running=()):
    """None unless an audio feature (stt/tts/talk) is configured, here or on
    a running server's config."""
    from . import extras
    need = set(_needed_extras(cfg))
    for rcfg, _rpath in running:
        need.update(_needed_extras(rcfg))
    if not need & extras.FFMPEG_EXTRAS:
        return None
    if extras.ffmpeg_present():
        return _check("ffmpeg", "PASS", shutil.which("ffmpeg") or "on PATH")
    return _check("ffmpeg", "FAIL", "not on PATH (brew install ffmpeg)")


def _assistant_mcp_servers(cfg) -> list:
    """Every MCP server the assistant can reach: the shared assistant.mcp
    list (when the talk brain or an unscoped alias uses it) plus each
    alias's own scoped list. Deduped by name."""
    servers: list = []
    shared_used = (cfg.talk.brain == "assistant"
                   or any(a.mcp is None for a in cfg.assistants.values()))
    if shared_used:
        servers.extend(cfg.assistant.mcp)
    for alias in cfg.assistants.values():
        if alias.mcp:
            servers.extend(alias.mcp)
    seen: set = set()
    return [s for s in servers
            if s.name not in seen and not seen.add(s.name)]


def check_mcp(cfg):
    """None unless the assistant has stdio MCP servers configured."""
    if cfg is None:
        return None
    servers = _assistant_mcp_servers(cfg)
    if not servers:
        return None
    missing = [f"{srv.name}: {srv.command[0]}" for srv in servers
               if srv.command and shutil.which(srv.command[0]) is None]
    if missing:
        return _check("mcp tools", "WARN",
                      "missing binaries: " + ", ".join(missing))
    return _check("mcp tools", "PASS",
                  f"{len(servers)} server{_s(len(servers))}, commands on PATH")


def check_assistant_exposure(cfg):
    """None unless served assistants sit on a non-loopback bind. WARN names
    which aliases inherit the full shared tool list vs carry their own."""
    from .config import LOOPBACK_HOSTS
    if cfg is None or not cfg.assistants or cfg.host in LOOPBACK_HOSTS:
        return None
    parts = []
    for aid, alias in sorted(cfg.assistants.items()):
        if alias.mcp is None:
            parts.append(f"{aid} (inherits full assistant.mcp tools)")
        else:
            parts.append(f"{aid} (own mcp list, {len(alias.mcp)} "
                         f"server{_s(len(alias.mcp))})")
    return _check("assistants", "WARN",
                  f"served on non-loopback {cfg.host}: " + ", ".join(parts))


def check_hf_token() -> dict:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        try:
            from huggingface_hub import get_token
            tok = get_token()
        except Exception:                       # noqa: BLE001 - optional dep
            tok = None
    if tok:
        return _check("hf token", "PASS", "token present")
    return _check("hf token", "SKIP", "no token (needed only for gated repos)")


def check_disk(cfg) -> dict:
    root = os.path.expanduser("~")
    if cfg is not None:
        for d in cfg.model_dirs:
            p = os.path.expanduser(os.path.expandvars(d))
            if os.path.exists(p):
                root = p
                break
    free = shutil.disk_usage(root).free
    detail = f"{free / 1024**3:.1f} GB free at {root}"
    if free < 10 * 1024**3:
        return _check("disk", "WARN", detail)
    return _check("disk", "PASS", detail)


def _run_checks(config_path, *, deep: bool) -> list[dict]:
    cfg_check, cfg, path = check_config(config_path)
    running = _running_configs(path)
    checks = [check_runtime(), check_kernels(), cfg_check,
              check_models(cfg, deep=deep), check_server()]
    for c in (check_agents(), check_launcher(), check_services(cfg),
              check_extras(cfg, running), check_ffmpeg(cfg, running),
              check_mcp(cfg), check_assistant_exposure(cfg)):
        if c is not None:
            checks.append(c)
    checks += [check_hf_token(), check_disk(cfg)]
    return checks


def cmd_doctor(argv: list | None = None, prog: str = "gmlx doctor") -> int:
    ap = argparse.ArgumentParser(
        prog=prog,
        description="Check the runtime, config, model paths, background "
                    "server, and optional services in one pass, and name the "
                    "fix for anything that fails.")
    ap.add_argument("--config", default=None, metavar="FILE",
                    help="Config to check (default: the bare-start search "
                         "path, e.g. ~/.config/gmlx/gmlx.yaml).")
    ap.add_argument("--deep", action="store_true",
                    help="Also read each configured model's GGUF header.")
    ap.add_argument("--json", action="store_true",
                    help="Emit the checks as JSON.")
    a = ap.parse_args(argv)

    if a.config and not os.path.exists(os.path.expanduser(a.config)):
        print(f"error: no config file at {a.config}", file=sys.stderr)
        return 2

    import gmlx
    checks = _run_checks(a.config, deep=a.deep)
    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    if a.json:
        print(json.dumps({"version": gmlx.__version__, "checks": checks,
                          "ok": n_fail == 0}, indent=2))
        return 0 if n_fail == 0 else 1
    print(f"gmlx {gmlx.__version__} doctor")
    wid = max(len(c["name"]) for c in checks)
    for c in checks:
        print(f"  {c['status']:<4}  {c['name']:<{wid}}  {c['detail']}")
    print()
    n_warn = sum(1 for c in checks if c["status"] == "WARN")
    if n_fail:
        print(f"{n_fail} check{_s(n_fail)} failed.")
    elif n_warn:
        print(f"checks complete: {n_warn} warning{_s(n_warn)}.")
    else:
        print("all checks passed.")
    return 0 if n_fail == 0 else 1
