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

from gmlx.textfmt import plural_s as _s


def _check(name: str, status: str, detail: str) -> dict:
    return {"name": name, "status": status, "detail": detail}


# The mlx-kquant Metal library targets this release, which the wheel tag
# cannot say (scripts/brew_formula.py MACOS_KERNELS).
_MACOS_MIN = (26, 2)


def _macos_version() -> str:
    # sw_vers first: a Python built against an older SDK can see a
    # compatibility version instead of the real one.
    try:
        v = subprocess.run(["sw_vers", "-productVersion"], capture_output=True,
                           text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        v = ""
    if not v:
        import platform
        v = platform.mac_ver()[0]
    return v


def check_macos() -> dict:
    if sys.platform != "darwin":
        return _check("macos", "SKIP", "not macOS")
    want = ".".join(map(str, _MACOS_MIN))
    v = _macos_version()
    try:
        have = tuple(int(p) for p in v.split(".")[:2])
    except ValueError:
        have = ()
    if not have:
        return _check("macos", "WARN",
                      f"version unknown, gmlx needs macOS {want} or newer")
    if (have + (0,))[:2] < _MACOS_MIN:
        return _check("macos", "WARN",
                      f"macOS {v} is older than {want}, which the mlx-kquant "
                      "kernels need. Update macOS")
    return _check("macos", "PASS", f"macOS {v}")


def check_runtime() -> dict:
    try:
        import mlx.core as mx
    except ImportError as e:
        return _check("runtime", "FAIL", f"mlx not importable: {e}")
    try:
        import mlx_kquant  # noqa: F401
    except ImportError as e:
        from . import extras
        return _check("runtime", "FAIL",
                      f"mlx-kquant not importable: {e} "
                      f"({extras.repair_hint('mlx-kquant')})")
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
    if not hasattr(mlx_kquant, "sdpa_decode_gqa_kvarn"):
        return _check("kernels", "WARN",
                      "missing sdpa_decode_gqa_kvarn "
                      "(--kv-quant-scheme kvarn unavailable)")
    return _check("kernels", "PASS",
                  "sdpa_vector, sdpa_decode_gqa, sdpa_decode_gqa_kvarn")


def check_config(config_path):
    """Returns ``(check, cfg, path)`` - cfg/path are None when unloadable."""
    import gmlx.config as cfgmod
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
    from gmlx.config import ConfigError, resolve_path
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
                    from gmlx.load.preflight import preflight
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
    from gmlx.serve.patches.routes import _service_file_on_disk
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
    import gmlx.serve.lifecycle as lifecycle
    runs = lifecycle.list_runs()
    if not runs:
        return _check("server", "SKIP",
                      "no background server (gmlx serve starts one)")
    parts: list[str] = []
    stale: list[str] = []
    status = "PASS"
    for run in runs:
        host, port, pid = run.get("host"), run.get("port"), run.get("pid")
        where = lifecycle.host_port(host, port)
        # A headless agent's runfile records no pid, so its health is the
        # only sign, as lifecycle.stale_reason has it.
        launchd = run.get("managed_by") == "launchd"
        who = "managed by launchd" if launchd else f"pid {pid}"
        if not launchd and not lifecycle.identity_ok(run):
            stale.append(where)
            status = "WARN"
        elif not lifecycle._health_ok(host, port):
            parts.append(f"{where} ({who}) not answering /health")
            status = "WARN"
        else:
            parts.append(f"running at {where} ({who})")
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
    import gmlx.serve.lifecycle as lifecycle
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


def check_login_start():
    """None off macOS, and when every login start of ``gmlx serve`` can read a
    config. An older gmlx wrote such a start, as a menu bar autostart record or
    a headless agent, with no config, or with ``--config gmlx.yaml`` relative
    to the folder it ran in. A bare serve now needs a config, and launchd runs
    a login start in /, so either start exits at every login, and only its log
    says why."""
    if sys.platform != "darwin":
        return None
    import plistlib

    import gmlx.serve.lifecycle as lifecycle
    from gmlx.commands.menubar import load_menubar_settings

    starts = []                       # (name, argv, headless, host, port)
    auto = load_menubar_settings().get("autostart")
    if auto:
        starts.append(("the menu bar's server autostart", auto["argv"], False,
                       auto.get("host", "127.0.0.1"), auto.get("port", 8080)))
    menubar_item = False
    for pp in _agent_plists():
        if pp.stem == lifecycle.MENUBAR_AGENT_LABEL:
            menubar_item = True
        try:
            args = plistlib.loads(pp.read_bytes()).get("ProgramArguments") or []
        except Exception:  # noqa: BLE001 - check_agents reports a broken plist
            continue
        args = [str(x) for x in args]
        host = args[args.index("--host") + 1] if "--host" in args[:-1] else "127.0.0.1"
        port = args[args.index("--port") + 1] if "--port" in args[:-1] else 8080
        starts.append((pp.stem, args, True, host, port))
    bare = [(name, headless, host, port) for name, argv, headless, host, port in starts
            if lifecycle.starts_bare(argv)]
    found = [(name, why, headless, host, port)
             for name, argv, headless, host, port in starts
             if (why := lifecycle.login_config_problem(argv))]
    if not bare and not found:
        return None
    parts = []
    if bare:
        names = [name for name, *_ in bare]
        # Uninstall without --port acts on the server that gmlx stop would
        # pick, so a headless agent's step names its own port. Each uninstall
        # also removes the menu bar's login item and its record.
        drops = list(dict.fromkeys(
            "gmlx service uninstall" + ("" if host == "127.0.0.1" else f" --host {host}")
            + f" --port {port}" for _, headless, host, port in bare if headless))
        parts.append(f"{' and '.join(names)} start{'' if len(names) > 1 else 's'} gmlx "
                     "serve with no config, which exits at login.")
        # A headless agent that exits with success stays stopped until the
        # next login, also after gmlx init.
        kicks = [f"launchctl kickstart gui/{os.getuid()}/{name}"
                 for name, headless, *_ in bare if headless]
        if kicks:
            parts.append("Run gmlx init to create ~/.config/gmlx/gmlx.yaml. A headless "
                         "agent stays stopped until the next login, so after gmlx init, "
                         f"run {' and '.join(kicks)}, or log out and log in again. To "
                         f"remove the start instead, run {', then run '.join(drops)}.")
        else:
            parts.append("Run gmlx init to create ~/.config/gmlx/gmlx.yaml, or remove "
                         "the start with gmlx service uninstall.")
        if drops and menubar_item:
            parts.append("gmlx service uninstall also removes the menu bar's login item.")
    for name, why, *_ in found:
        parts.append(f"{name} starts gmlx serve with {why}, so the server does not "
                     "start at login.")
    if found:
        has_default = lifecycle.first_default_config() is not None
        steps = []
        for _, _, headless, host, port in found:
            # A headless agent's name holds its host and port, so a step
            # without them would add a second agent and leave this one.
            tgt = ("" if host == "127.0.0.1" else f" --host {host}") + (
                "" if str(port) == "8080" else f" --port {port}")
            if not headless:
                steps.append(f"gmlx stop{tgt}")
            steps.append(f"gmlx service install{' --headless' if headless else ''}{tgt}"
                         f"{' --config <full path>' if has_default else ''}")
        steps = list(dict.fromkeys(steps))
        if has_default:
            parts.append(f"Run {', then run '.join(steps)}, where <full path> names the "
                         "gmlx.yaml to start at login.")
        else:
            parts.append("Move the gmlx.yaml that the server should read at login to "
                         f"~/.config/gmlx/gmlx.yaml, then run {', then run '.join(steps)}.")
    return _check("login start", "WARN", " ".join(parts))


def check_launcher():
    """None off macOS. Detached serve / menubar children exec a renamed copy
    of the interpreter (procname.py) so they show as "gmlx"; an interpreter
    swap under the venv can strand the copy (e.g. a relative-linked
    python-build-standalone binary), and then every detached start dies in
    dyld before any Python runs. Prove the copy executes."""
    if sys.platform != "darwin":
        return None
    import gmlx.serve.procname as procname
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


def _sysctl_int(name: str) -> int | None:
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True,
                             timeout=5).stdout.strip()
        return int(out)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


# Seconds doctor waits for one container query.
DOCTOR_QUERY_TIMEOUT = 5.0

# The most files and folders doctor visits in the private homes, so a home
# that holds a large tree cannot make doctor slow.
_WALK_CAP = 100_000
# The most private homes doctor lists one per row.
_HOMES_LISTED = 10


def _folder_bytes(root, budget: list[int]) -> int:
    """The disk space of the files under ``root``. ``budget`` holds the
    number of files and folders still to visit; at zero the walk stops, and
    the total is then a lower bound."""
    total = 0
    for folder, _, files in os.walk(root):
        if budget[0] <= 0:
            return total
        budget[0] -= 1
        for name in files:
            if budget[0] <= 0:
                return total
            budget[0] -= 1
            try:
                total += os.lstat(os.path.join(folder, name)).st_blocks * 512
            except OSError:
                pass
    return total


def check_homes() -> list[dict]:
    """One row per private home of container mode, newest use first: the
    client, the project folder, the space on disk and the last launch.
    Homes past :data:`_HOMES_LISTED` share one row. Each home gets an equal
    part of the walk budget."""
    if sys.platform != "darwin":
        return []
    import time

    from gmlx.config import target_label
    from gmlx.container import session, settings

    homes = settings.private_homes()
    rows = []
    for home in homes[:_HOMES_LISTED]:
        budget = [_WALK_CAP // _HOMES_LISTED]
        size = session.gb(_folder_bytes(home.path, budget))
        size = f"at least {size}" if budget[0] <= 0 else size
        where = settings._tilde(home.folder) if home.folder else "default project"
        when = time.strftime("%Y-%m-%d", time.localtime(home.used)) if home.used else "unknown"
        detail = f"{target_label(home.client)}: {where}, {size}, last used {when}"
        rows.append(_check("home", "PASS", detail))
    rest = len(homes) - _HOMES_LISTED
    if rest > 0:
        rows.append(_check("home", "PASS", f"and {rest} more private home{_s(rest)} under "
                                            f"{settings._tilde(str(settings.data_path()))}"))
    return rows


def check_container():
    """None off macOS, and a SKIP row when container mode is neither
    configured nor installed. Otherwise the Apple container version and
    service, the packaged guest entry, file handles, and the volumes and
    images launch keeps on disk, with the delete command for the images no
    setting uses. :func:`check_homes` lists the private homes. Leftover
    launch containers warn, because their memory stays taken until they
    stop."""
    if sys.platform != "darwin":
        return None
    from gmlx.config import LAUNCH_CLIENTS, ConfigError, load_launch_settings
    from gmlx.container import cli
    try:
        launch_cfg = load_launch_settings(note_local=False)
        box = launch_cfg.container
        # An agent runs only in a container, so configuring one turns it on.
        enabled = (any(box.for_client(c).enabled for c in LAUNCH_CLIENTS)
                   or bool(launch_cfg.agents))
    except (ConfigError, OSError):
        launch_cfg, enabled = None, False    # the config row reports a broken file
    if cli.find() is None:
        if not enabled:
            return _check("container", "SKIP",
                          "Apple container is not installed (brew install container)")
        return _check("container", "FAIL",
                      "container mode is on, but Apple container is not installed "
                      "(brew install container)")
    # A service that does not answer costs doctor seconds, not minutes.
    with cli.query_timeout(DOCTOR_QUERY_TIMEOUT):
        return _container_row(enabled, launch_cfg)


def _open_servers() -> list[str]:
    """A line for each running server that listens on more than a loopback
    address and was started with no key. A container reaches such a server
    at the Mac's address on its network, past the session socket."""
    import gmlx.serve.lifecycle as lifecycle
    from gmlx.commands.launch_container import loopback_host

    return [f"the server at {lifecycle.host_port(run.get('host'), run.get('port'))} "
            "listens on more than loopback with no key, so a container can reach all "
            "of its routes (set server.api_key)"
            for run in lifecycle.classify_runs()[0]
            if not loopback_host(str(run.get("host") or "127.0.0.1"))
            and not run.get("api_key_set")]


def _container_row(enabled: bool, launch_cfg=None) -> dict:
    from gmlx.container import cli, images, localhost_domains, runtime, session, settings

    status, parts = "PASS", []

    def flag(level: str, text: str) -> None:
        nonlocal status
        if level == "FAIL" or status == "PASS":
            status = level
        parts.append(text)
    try:
        program = cli.find()
        version = cli.version(program)
        if program and (version is None or version < cli.CONTAINER_MIN):
            need = ".".join(map(str, cli.CONTAINER_MIN))
            shown = settings._tilde(program)
            what = (f"container {'.'.join(map(str, version))} at {shown} is older than "
                    f"{need}" if version else
                    f"container at {shown} gives no version number, and launch needs "
                    f"{need} or newer")
            # Launch refuses every container launch with such a version.
            flag("FAIL" if enabled else "WARN", what + cli.upgrade_steps(program)[1])
        elif version:
            parts.append("container " + ".".join(map(str, version)))
        if not runtime.entry_path().is_file():
            flag("FAIL" if enabled else "WARN",
                 f"the guest entry is not built ({runtime.BUILD_HINT})")
        if (note := localhost_domains.doctor_note()) is not None:
            flag("WARN", note)
        for text in _open_servers():
            if enabled:
                flag("WARN", text)
            else:                     # nothing needs it until container mode is on
                parts.append(text)
        found = cli.service()
        if not found.running:
            text = "the container service is stopped (container system start)"
            if enabled:
                flag("WARN", text)
            else:                     # nothing needs it until container mode is on
                parts.append(text)
            return _check("container", status, "; ".join(parts))
        if not cli.kernel_installed(found.app_root):
            flag("FAIL" if enabled else "WARN",
                 "the container service runs with no Linux kernel, so no container can start "
                 "(container system kernel set --recommended)")
        containers = cli.containers()
        files, limit = _sysctl_int("kern.num_files"), _sysctl_int("kern.maxfiles")
        per_process = _sysctl_int("kern.maxfilesperproc")
        if files is not None and limit:
            running = any(c.labels.get(cli.LAUNCH_LABEL) == "1" and c.state == "running"
                          for c in containers)
            text = f"{files:,} of {limit:,} open files ({per_process or 0:,} per process)"
            if running and files > limit // 2:
                flag("WARN", text + " while a launch container runs; narrow its shares")
            else:
                parts.append(text)
        volumes = [v for v in cli.volume_list() if v.labels.get(cli.LAUNCH_LABEL) == "1"]
        if volumes:
            parts.append("volumes " + ", ".join(
                f"{v.name} {session.gb(session.allocated_bytes(v.source))}"
                for v in volumes))
        count, layers, unused = images.disk_report(launch_cfg)
        if count:
            parts.append(f"{count} launch image{_s(count)}, {session.gb(layers)} of layers")
        if unused:
            parts.append(f"{len(unused)} image reference{_s(len(unused))} that no setting "
                         f"uses (container image delete {' '.join(unused)})")
        for c in session.leftover_containers(containers):
            memory = f", {session.gb(c.memory_bytes)}" if c.memory_bytes else ""
            flag("WARN", f"{c.name} is left over{memory} (container stop {c.name})")
        report = images.builder_report()
        if report is not None:
            line, stop_owed = report
            text = line.removeprefix("[launch] ").rstrip(".")
            if stop_owed:                 # a launch started it and could not stop it
                flag("WARN", text)
            else:                         # yours, or started before a launch looked
                parts.append(text)
    except cli.ContainerError as e:
        flag("WARN", str(e))
    return _check("container", status, "; ".join(parts))


def _running_configs(primary_path) -> list:
    """(cfg, path) for each live server whose runfile records a --config
    other than the file doctor is already checking. Extras and ffmpeg are
    properties of this machine, but the features that *need* them follow
    whatever config each server was actually started with - a
    `serve --config other.yaml` can enable stt/tts that the default-location
    config leaves commented out."""
    import gmlx.config as cfgmod
    import gmlx.serve.lifecycle as lifecycle
    seen = set()
    if primary_path:
        seen.add(os.path.abspath(os.path.expanduser(str(primary_path))))
    out = []
    for run in lifecycle.list_runs():
        if not run.get("config_abspath") or not lifecycle.identity_ok(run):
            continue                    # no config, or a stale runfile
        # An older gmlx recorded the path relative to the server's folder,
        # not to doctor's, and a path that stays relative names no file.
        ap = lifecycle.run_config_path(run)
        if not ap or not os.path.isabs(ap) or ap in seen:
            continue
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
    from gmlx.config import TalkCfg
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
        pips = "; ".join(dict.fromkeys(extras.install_hint(x) for x in missing))
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
    from gmlx.serve import media_programs, programs

    if not extras.ffmpeg_present():
        return _check("ffmpeg", "FAIL", media_programs.problem("ffmpeg")
                      or "The gmlx server finds no ffmpeg. Install it with `brew install ffmpeg`.")
    lookup = programs.look_up("ffmpeg")
    path = programs.tilde(lookup.path or "")
    # A skipped folder that holds an ffmpeg is the one the user can expect
    # the server to run.
    skips = programs.skips_that_hold(lookup.search, "ffmpeg")
    if skips:
        return _check("ffmpeg", "WARN", " ".join([f"The server runs {path}.", *skips]))
    return _check("ffmpeg", "PASS", path)


def _assistant_mcp_servers(cfg) -> list:
    """Every MCP server the assistant can reach: the shared assistant.mcp
    list, which `gmlx chat --assistant` always uses, plus each alias's own
    scoped list. Deduped by name."""
    servers: list = list(cfg.assistant.mcp)
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
    from gmlx.serve import programs

    missing, refused = [], []
    for srv in servers:
        if not srv.command:
            continue
        # The PATH that gmlx searches for the command, as assistant/mcp.py does.
        path = srv.env.get("PATH", os.environ.get("PATH", os.defpath))
        lookup = programs.look_up(srv.command[0], path)
        if lookup.path is None or shutil.which(lookup.path) is None:
            # A command that only a skipped PATH entry holds is on the
            # user's PATH, so the row says why gmlx does not find it.
            skips = [f"gmlx does not look in {shown}, because that PATH entry {why}"
                     for shown, why in programs.skipped_holders(lookup.search, srv.command[0])]
            missing.append(f"{srv.name}: {srv.command[0]}"
                           + (f" ({'; '.join(skips)})" if skips else ""))
        elif lookup.refusal is not None:
            refused.append(f"{srv.name}: {programs.tilde(lookup.path)}, which "
                           f"{lookup.refusal}")
    if missing or refused:
        parts = (["missing binaries: " + ", ".join(missing)] if missing else []) + (
            ["will not run " + "; ".join(refused)] if refused else [])
        return _check("mcp tools", "WARN", "; ".join(parts))
    return _check("mcp tools", "PASS",
                  f"{len(servers)} server{_s(len(servers))}, commands on PATH")


def check_assistant_exposure(cfg):
    """None unless served assistants sit on a non-loopback bind. WARN names
    which aliases inherit the full shared tool list vs carry their own."""
    from gmlx.config import LOOPBACK_HOSTS
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


def check_memory(cfg) -> dict:
    """RAM vs the configured models' file sizes. A model file bigger than the
    fit threshold that isn't set to stream gets a WARN naming it - the server
    would load it into memory pressure (or fail) on first request. Advisory:
    smaller models on the same config keep serving either way."""
    from gmlx.serve.lifecycle import human_gb
    from gmlx.load.memfit import classify_fit, total_ram_bytes

    ram = total_ram_bytes()
    if ram is None:
        return _check("memory", "SKIP", "could not read machine RAM")
    detail = f"{human_gb(ram, 0)} RAM"
    if cfg is None or not cfg.models:
        return _check("memory", "PASS", detail)
    from .manage import _model_size_bytes
    flagged: list[str] = []
    streamed: list[str] = []
    bad_stream: list[str] = []
    for mid, m in cfg.models.items():
        if getattr(m, "stream", None) == "experts":
            summary, ok = _stream_summary(m.path, cfg.model_dirs, ram)
            if summary:
                (streamed if ok else bad_stream).append(f"{mid}: {summary}")
            continue
        if getattr(m, "stream", None):
            continue                     # streamed on purpose: over-RAM is fine
        size = _model_size_bytes(m.path, cfg.model_dirs)
        if size and classify_fit(size, ram) == "over":
            flagged.append(f"{mid} ({human_gb(size)})")
    if flagged:
        more = f" (+{len(flagged) - 3} more)" if len(flagged) > 3 else ""
        return _check(
            "memory", "WARN",
            f"{detail}; larger than RAM: " + ", ".join(flagged[:3]) + more
            + " - a MoE model can set `stream: experts` "
              "(docs/streaming.md); a dense model needs a smaller quant")
    if bad_stream:
        return _check(
            "memory", "WARN",
            f"{detail}; cannot stream: " + "; ".join(bad_stream[:3])
            + " - the every-token weights must fit under the memory "
              "ceiling (docs/streaming.md)")
    tail = f"; streaming: {'; '.join(streamed[:3])}" if streamed else ""
    return _check("memory", "PASS", f"{detail}; configured models fit{tail}")


def _stream_summary(path: str, model_dirs: list[str], ram: int
                    ) -> tuple[str | None, bool]:
    """One clause on a ``stream: experts`` entry's plan on this Mac, and
    whether it streams. None for a dense file or an unreadable header."""
    import gmlx.config as cfgmod
    from gmlx.stream import plan as sp
    try:
        model = sp.model_plan(sp.scan_path(cfgmod.resolve_path(path, model_dirs)))
        if not model.streamable:
            return None, True
        box = sp.box_plan(model, ram_bytes=ram)
    except Exception:                                  # noqa: BLE001 - advisory
        return None, True
    if box is None:
        return None, True
    return sp.box_summary(model, box), box.verdict != sp.VERDICT_TOO_BIG


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
    checks = [check_macos(), check_runtime(), check_kernels(), cfg_check,
              check_models(cfg, deep=deep), check_server()]
    for c in (check_agents(), check_login_start(), check_launcher(), check_container(),
              *check_homes(),
              check_services(cfg), check_extras(cfg, running), check_ffmpeg(cfg, running),
              check_mcp(cfg), check_assistant_exposure(cfg)):
        if c is not None:
            checks.append(c)
    checks += [check_hf_token(), check_memory(cfg), check_disk(cfg)]
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
