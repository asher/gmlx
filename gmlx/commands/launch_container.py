"""``gmlx launch <client> --container``: the launch order in container mode.

The client's handler in ``launch.py`` still resolves the server, writes the
client's configuration and names the command, but it does so with ``HOME``
pointed at the client's private home, and its finish helper hands the
command back here instead of replacing the process. This module decides
container mode, takes the session lock, resolves the shares and the image,
and runs the supervisor in ``gmlx.container.session``.
"""

from __future__ import annotations

import contextlib
import os
import shlex
import sys
import urllib.parse
import webbrowser
from pathlib import Path

from gmlx.config import ConfigError, LaunchCfg, load_launch_settings
from gmlx.container import cli, confine, images, runtime, session, settings
from gmlx.container.cli import ContainerError
from gmlx.container.settings import Mount, SettingsError

# Flags that only mean something in container mode, by argparse dest.
CONTAINER_FLAGS = {"mount": "--mount", "mount_cwd": "--mount-cwd", "image": "--image",
                   "rebuild": "--rebuild", "network": "--network", "shell": "--shell"}
# What an attaching --shell accepts. Every other flag shapes a new session.
_ATTACH_DEFAULTS = {
    "model": None, "base_url": None, "host": None, "port": None, "api_key": None,
    "provider_id": "gmlx", "config_path": None, "config_only": False,
    "no_start": False, "start_timeout": 0.0, "no_keep": False, "dsh_profile": None,
    "mount": [], "mount_cwd": None, "image": None, "rebuild": False, "network": None,
}
_DSH_URL_LINE = r"dsh web: (\S+)"


def _say(line: str) -> None:
    print(line, flush=True)


def _flag_name(dest: str, value) -> str:
    """The flag as the user typed it. ``--no-mount-cwd`` sets False."""
    if dest == "mount_cwd" and value is False:
        return "--no-mount-cwd"
    return CONTAINER_FLAGS.get(dest) or "--" + dest.replace("_", "-")


def _flag_set(a, dest: str) -> bool:
    value = getattr(a, dest, None)
    if dest == "mount_cwd":           # --mount-cwd and --no-mount-cwd both count
        return value is not None
    return value not in (None, False, [])


def container_mode(a, ap) -> tuple[bool, LaunchCfg]:
    """Whether this launch runs in a container, from the flags and the
    ``enabled`` keys of the user-level config, and those settings. A broken
    ``launch`` block stops only a launch that asks for container mode on the
    command line. Any other launch runs on the Mac with one notice."""
    implied = [_flag_name(dest, getattr(a, dest, None))
               for dest in CONTAINER_FLAGS if _flag_set(a, dest)]
    if a.container is False and implied:
        ap.error(f"{implied[0]} applies only in container mode, so it cannot go with "
                 "--no-container")
    try:
        launch_cfg, notice = load_launch_settings()
    except ConfigError as e:
        if a.container or implied:
            raise
        print(f"[launch] ignoring the launch settings, so {a.harness} runs on the Mac: {e}",
              file=sys.stderr)
        return False, LaunchCfg()
    if notice:
        print(f"[launch] {notice}", file=sys.stderr)
    if a.container is not None:
        return a.container, launch_cfg
    if implied:
        return True, launch_cfg
    return bool(launch_cfg.container.for_client(a.harness).enabled), launch_cfg


@contextlib.contextmanager
def guest_home(home: Path):
    """Point ``HOME`` at the private home while a handler runs, and hide the
    variables that would send it to the user's own files."""
    hidden = ("DSH_HOME", "HERMES_HOME")
    saved = {k: os.environ.get(k) for k in ("HOME", *hidden)}
    os.environ["HOME"] = str(home)
    for key in hidden:
        os.environ.pop(key, None)
    try:
        # The guest can plant links in the private home, so every file the
        # handler touches is checked against them.
        with confine.confined(home):
            yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _server_endpoint(a) -> tuple[str, int]:
    """The server host and port this launch will target, before the server
    check runs."""
    from gmlx.serve import lifecycle

    if a.base_url:
        split = urllib.parse.urlsplit(a.base_url)
        try:
            port = split.port
        except ValueError:
            port = None
        return a.host or split.hostname or "127.0.0.1", int(
            a.port or port or (443 if split.scheme == "https" else 80))
    if a.host or a.port:
        return a.host or "127.0.0.1", int(a.port or 8080)
    host, port = lifecycle.auto_target(None, None)
    return host, int(port)


def guest_url(base_url: str) -> tuple[str, int | None, list]:
    """``(guest URL, API port, relay targets)`` for the server URL. An https
    URL passes through unchanged with no relay."""
    from gmlx.container.relay import resolve_targets

    split = urllib.parse.urlsplit(base_url)
    if split.scheme == "https":
        return base_url, None, []
    host = split.hostname or "127.0.0.1"
    port = split.port or 80
    if host in ("127.0.0.1", "0.0.0.0"):
        targets = [("127.0.0.1", port)]
    elif host in ("::1", "::"):
        targets = [("::1", port)]
    else:
        try:
            targets = resolve_targets(host, port)
        except OSError as e:
            raise SettingsError(f"cannot resolve the server host {host} ({e}).") from None
        if not targets:
            raise SettingsError(f"the server host {host} has no IPv4 or IPv6 address.")
    url = urllib.parse.urlunsplit(("http", f"127.0.0.1:{port}", split.path, split.query,
                                   split.fragment))
    return url, port, targets


# Step 3: the container command, the service and the guest entry

class _Prereqs:
    def __init__(self):
        self.binary = cli.find()
        self.version = None
        self.running = False
        self.entry = runtime.entry_path()
        if self.binary:
            try:
                self.version = cli.version()
                self.running = cli.system_running()
            except ContainerError:
                pass

    def report(self) -> list[str]:
        lines = []
        if not self.binary:
            lines.append(f"[launch] container is not installed. {cli.INSTALL_HINT}")
        else:
            v = ".".join(map(str, self.version)) if self.version else "unknown version"
            state = "running" if self.running else "stopped"
            lines.append(f"[launch] container {v}, service {state}")
            if self.version and self.version < cli.CONTAINER_MIN:
                lines.append(f"[launch] container {v} is older than the "
                             f"{'.'.join(map(str, cli.CONTAINER_MIN))} this mode needs")
        if not self.entry.is_file():
            lines.append(f"[launch] the guest entry {self.entry} is not built. Build it "
                         f"with: {runtime.BUILD_HINT}")
        return lines

    def require(self, say) -> bool:
        """Refuse what cannot run, and start a stopped service. Returns True
        when the service had to start, which marks a first run."""
        from gmlx.commands.launch import LaunchError

        if not self.binary:
            raise LaunchError(f"container mode needs Apple container. {cli.INSTALL_HINT}")
        if self.version is None or self.version < cli.CONTAINER_MIN:
            need = ".".join(map(str, cli.CONTAINER_MIN))
            have = ".".join(map(str, self.version)) if self.version else "an unknown version"
            raise LaunchError(f"container mode needs Apple container {need} or newer, and "
                              f"this Mac has {have}. Upgrade with: brew upgrade container")
        if not self.entry.is_file():
            raise LaunchError(f"the guest entry {self.entry} is not built. In a git "
                              f"checkout, build it with: {runtime.BUILD_HINT}")
        if self.running:
            return False
        if not session.stdin_is_tty():
            raise LaunchError("the container service is not running. Start it with: "
                              "container system start")
        say(f"[launch] step 1: start the container service. The first start asks to "
            f"install a Linux kernel and downloads about {cli.KERNEL_DOWNLOAD_MB} MB.")
        cli.system_start()
        return True


# --shell into a running session

def _attach(a, exec_fn, say) -> int:
    from gmlx.commands.launch import LaunchError

    for dest, default in _ATTACH_DEFAULTS.items():
        value = getattr(a, dest, default)
        if value != default:
            flag = _flag_name(dest, value)
            raise LaunchError(f"a {a.harness} session is already running, so --shell attaches "
                              f"to it, and {flag} applies only to a new session.")
    record = session.read_record(a.harness)
    name = (record or {}).get("name")
    running = [c for c in cli.list_launch_containers()
               if c.name == name and c.state == "running"
               and c.labels.get("gmlx.launch.client") == a.harness]
    if not record or not running:
        raise LaunchError(f"the {a.harness} session is still starting. Try again in a moment.")
    shares = [Mount(s["host"], s["guest"], bool(s.get("readonly"))) for s in record["shares"]]
    cwd = settings.guest_path(os.path.realpath(os.getcwd()), shares)
    say(f"[launch] attaching to {name} (working folder {record['workdir']})")
    if cwd is None:
        say("[launch] the current folder is not shared with this session, so the shell "
            "opens in its working folder.")
    command = [runtime.GUEST_ENTRY, *(["--clipboard"] if record.get("clipboard") else []),
               "--shell", "--", *a.passthrough]
    argv = cli.exec_argv(name, command, tty=session.stdin_is_tty(), cwd=cwd)
    return exec_fn(argv[0], argv, dict(os.environ))


# The launch order

def _client_env(client: str, plan, ready, command_cfg, web_port: int | None) -> dict:
    """Guest variables launch sets by value for one client."""
    env: dict[str, str] = {}
    if client == "claude-code":
        env.update({"IS_SANDBOX": "1", "DISABLE_AUTOUPDATER": "1"})
        if plan.network == "none":
            env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    if web_port is not None:
        env.update({"HOST": "127.0.0.1", "PORT": str(web_port)})
    if client == "open-webui" and command_cfg == "image":
        env["WEBUI_SECRET_KEY_FILE"] = str(plan.home / ".webui_secret_key")
    if client == "omp" and plan.clipboard == "images":
        # omp looks for a display before it runs a clipboard tool.
        env["WAYLAND_DISPLAY"] = "wayland-0"
    return env


def _split_env(entries: list[str]) -> tuple[list[str], dict[str, str]]:
    names, values = [], {}
    for entry in entries:
        name, sep, value = entry.partition("=")
        names.append(name)
        if sep:
            values[name] = value
    return names, values


def _dsh_profile_is_web(a, home: Path) -> bool:
    from gmlx.commands import launch as L

    profile = L._DSH_PROFILE if a.dsh_profile is None else a.dsh_profile
    manifest = home / ".dsh" / "profiles" / profile / "package.json"
    return L._dsh_runs_web_app(profile, manifest)


def _image_state(image_plan, rebuild: bool, running: bool) -> tuple[str, str]:
    """For the dry run: the reference the command would name and a line
    about the image, without building or pulling anything."""
    if not running:
        tag = (image_plan.ref if image_plan.kind == "image"
               else images.shipped_tag(image_plan.client, image_plan.packages)
               if image_plan.kind == "shipped" else images.build_repo(image_plan.client))
        return tag, f"[launch] image {tag}: its state is unknown until the service runs"
    if image_plan.kind == "image":
        tag = image_plan.ref
        info = cli.image_info(tag)
        verb = "pulled again" if rebuild else ("present" if info else "pulled")
    elif image_plan.kind == "shipped":
        tag = images.shipped_tag(image_plan.client, image_plan.packages)
        info = cli.image_info(tag)
        verb = "rebuilt" if rebuild else ("present" if info else "built")
    else:
        tag = images.build_repo(image_plan.client)
        info = None
        verb = "rebuilt" if rebuild else "checked, and built when its context changed"
    ref = f"{images.repository_of(info.name or tag)}@{info.digest}" if info else tag
    state = "present" if verb == "present" else f"would be {verb}"
    lines = [f"[launch] image {tag}: {state}"]
    for base in sorted(image_plan.bases) if image_plan.kind == "build" else []:
        present = cli.image_info(images.shipped_tag(
            base, image_plan.base_packages.get(base, []))) is not None
        base_state = ("would be rebuilt" if rebuild else "present" if present
                      else "would be built first")
        lines.append(f"[launch] base {images.base_ref(base)}: {base_state}")
    return ref, "\n".join(lines)


def run_container(a, launch_cfg: LaunchCfg, *, exec_fn) -> int:
    from gmlx.commands import launch as L

    say = _say
    client = a.harness
    cfg = launch_cfg.container.for_client(client)
    dry = bool(a.config_only)
    a.container_mode = True
    try:
        if a.config_path:
            raise L.LaunchError("--config-path does not apply in container mode, where the "
                                "client's configuration goes in its private home.")
        # Step 3
        prereqs = _Prereqs()
        first_run = False
        if dry:
            for line in prereqs.report():
                say(line)
        # Step 4
        lock = session.try_session_lock(client)
        if lock is None:
            if a.shell:                   # attaching refuses --config-only itself
                return _attach(a, exec_fn, say)
            record = session.read_record(client) or {}
            raise L.LaunchError(
                f"a container session of {client} is already running"
                + (f" ({record['name']})" if record.get("name") else "")
                + f". Open a shell in it with: gmlx launch {client} --shell")
        held = [lock]
        try:
            if not dry:
                first_run = prereqs.require(say)
            return _run_locked(a, launch_cfg, cfg, prereqs, first_run, held, exec_fn, say)
        finally:
            for item in reversed(held):
                item.release()
    except (L.LaunchError, SettingsError, ContainerError, ConfigError,
            confine.ConfinedError) as e:
        print(f"[launch] {e}", file=sys.stderr)
        return 1


def _run_locked(a, launch_cfg, cfg, prereqs, first_run, held, exec_fn, say) -> int:
    from gmlx.commands import launch as L

    client = a.harness
    dry = bool(a.config_only)
    # Step 5
    session.remove_record(client)
    # Step 6
    host, port = _server_endpoint(a)
    home = settings.private_home(client)
    web = client == "open-webui" or (client == "dsh" and _dsh_profile_is_web(a, home))
    web_port = L.web_port_for(client, port) if web else None
    plan = settings.resolve_plan(client, cfg, cwd=os.getcwd(), mount_cwd=a.mount_cwd,
                                 cli_mounts=a.mount, network=a.network, api_port=port,
                                 web_port=web_port)
    if client == "dsh" and a.dsh_profile in L._DSH_STDIO:
        raise L.LaunchError(f"the {a.dsh_profile} profile serves another program over stdio, "
                            "which a container session cannot hand over. Use it on the Mac "
                            "with --no-container.")
    image_plan = images.resolve_image(client, cfg, launch_cfg.container, image_override=a.image)
    for line in [*plan.warnings, *plan.notes, *image_plan.notices,
                 *settings.server_config_warnings(
                     settings.server_config_path(host, port,
                                                 autostart=not (a.base_url or a.no_start)),
                     plan.shares)]:
        say(line)
    running = prereqs.running
    ready = None
    runtime_dir = runtime.runtime_root() / (
        runtime.entry_digest() if prereqs.entry.is_file() else "<sha256>")
    if not dry:
        # Step 7
        session.cleanup_stale(client, keep_runtime=runtime_dir.name)
        containers = cli.containers()
        for line in session.orphan_notices(client, containers):
            say(line)
        # Step 8
        runtime_dir, runtime_lock = runtime.acquire_runtime()
        held.append(runtime_lock)
        held.extend(session.lock_volumes(plan.volumes))
        session.check_volumes_free(plan.volumes, containers)
        session.ensure_volumes(plan.volumes, say)
        step = "[launch] step 2: " if first_run else "[launch] "
        ready = images.ensure_image(image_plan, rebuild=a.rebuild,
                                    say=lambda line: say(line.replace("[launch] ", step, 1)))
        try:
            word = (cfg.command[0] if isinstance(cfg.command, list)
                    else images.image_command(ready, "image", [], [])[0][0]
                    if cfg.command == "image" else images.CLIENT_BINARY[client])
        except images.ImageError as e:
            if not a.shell:                # a shell is how you look into such an image
                raise
            say(f"[launch] warning: {e}")
        else:
            images.check_command(ready, word, str(runtime_dir), shell=a.shell, say=say)
    # Step 9
    rc = L._ensure_server(a)
    if rc is not None:
        return rc
    base = a.base_url or f"http://{a.host}:{a.port}/v1"
    a.guest_base_url, api_port, api_targets = guest_url(base)
    if api_port is None and plan.network == "none":
        raise L.LaunchError(f"network: none cannot reach {base}, which is not a local http "
                            "server. Use the default network for this server.")
    # Step 10
    if a.model and not a.no_keep and not dry:
        L._pick_default(L.probe_models(base, a.api_key), a.model)
        L._keep_model(a)
    # Step 11
    for line in settings.seed_home(plan.home, plan.seed):
        say(line)
    captured: dict = {}

    def sink(argv, pairs, extra):
        captured.update(argv=argv, pairs=pairs, extra=extra)
        return 0
    a.container_sink = sink
    with guest_home(plan.home):
        rc = L._HARNESSES[client](a, exec_fn=exec_fn)
    if rc != 0 or not captured:
        return rc
    if client == "open-webui":
        # The official image does not create its data folder, and SQLite
        # cannot open a database in a folder that does not exist.
        with confine.confined(plan.home):
            confine.mkdirs(Path(captured["pairs"]["DATA_DIR"]))
    # Step 12
    passthrough = captured["extra"]
    if a.shell:
        command, image_workdir = list(passthrough), None
    elif ready is not None:
        command, image_workdir = images.image_command(ready, cfg.command, captured["argv"],
                                                      passthrough)
    else:
        command = ([*cfg.command, *passthrough] if isinstance(cfg.command, list)
                   else [*captured["argv"], *passthrough])
        image_workdir = None
    names, values = _split_env(plan.env)
    env_values = {**settings.guest_env(plan.home),
                  **_client_env(client, plan, ready, cfg.command, web_port)}
    pair_names = [n for n in captured["pairs"] if n not in env_values]
    env_names = list(dict.fromkeys([*pair_names, *(n for n in names if n not in env_values)]))
    child_env = {**captured["pairs"], **values}
    # dsh prints its URL with a per-process login token, which the Mac
    # browser needs, so launch reads it from the client's output.
    token_url = client == "dsh" and web_port is not None and not a.shell
    if dry:
        tag, image_line = _image_state(image_plan, a.rebuild, prereqs.running)
        sess = session.Session(client, "xxxxxx", Path("<session folder>"))
        image_ref = tag
    else:
        sess = session.new_session(client, plan.forward)
        image_ref = ready.run_ref
    spec = session.RunSpec(
        session=sess, plan=plan, image_ref=image_ref, runtime_dir=runtime_dir,
        command=command, workdir=image_workdir or plan.workdir, env_values=env_values,
        env_names=env_names, child_env=child_env, api_port=api_port, web_port=web_port,
        tty=session.stdin_is_tty() and not token_url, interactive=not token_url,
        shell=a.shell, url_pattern=_DSH_URL_LINE if token_url else None,
        labels={"gmlx.launch.runtime": runtime_dir.name})
    summary = _summary_lines(plan, ready, a.shell, client)
    if dry:
        return _print_dry_run(spec, plan, image_line, summary, cfg, captured, running, say)
    record = {"name": sess.name, "workdir": spec.workdir, "clipboard": plan.clipboard == "images",
              "shares": [{"host": m.source, "guest": m.target, "readonly": m.readonly}
                         for m in plan.shares]}
    if first_run:
        summary.insert(0, f"[launch] step 3: start {client}")
    opener = webbrowser.open if (web_port and plan.open_browser) else None
    return session.supervise(spec, api_targets=api_targets, record=record, say=say,
                             opener=opener, summary=summary)


def _summary_lines(plan, ready, shell: bool, client: str) -> list[str]:
    lines = [images.describe(ready)] if ready is not None else []
    if ready is not None:
        note = images.image_age_note(ready)
        if note:
            lines.append(note)
    holder = None
    if plan.cwd_shared:
        guest = [m for m in plan.shares if m.target == plan.workdir
                 or plan.workdir.startswith(m.target.rstrip("/") + "/")]
        holder = max(guest, key=lambda m: len(m.target), default=None)
    for m in plan.shares:
        mode = "read-only" if m.readonly else "read-write"
        where = "" if m.target == m.source else f" at {m.target}"
        extra = ", working folder" if m is holder else ""
        if m.kind == "git":
            extra = f", {m.note}"
        lines.append(f"[launch] sharing {m.source}{where} ({mode}{extra})")
    if ready is not None:
        lines += session.volume_lines(plan.volumes)
    for port in plan.forward:
        lines.append(f"[launch] forwarding the guest's 127.0.0.1:{port} to Mac port {port}")
    if shell:
        lines.append(f"[launch] opening a shell instead of {client}")
    return lines


def _print_dry_run(spec, plan, image_line, summary, cfg, captured, running, say) -> int:
    say("[launch] container dry run: nothing is built, pulled or started.")
    say(image_line)
    say(f"[launch] runtime folder {spec.runtime_dir}")
    if plan.volumes:
        existing = {v.name for v in cli.volume_list()} if running else None
        for v in plan.volumes:
            if existing is None:
                state = "unknown until the service runs"
            else:
                state = "exists" if v.source in existing else "would be created"
            say(f"[launch] volume {v.source}: {state}")
    for line in summary:
        say(line)
    if isinstance(cfg.command, list) or cfg.command == "image":
        say("[launch] the command: setting replaces the client's own command, "
            f"{shlex.join(captured['argv'])}")
    argv = session.compose_run_argv(spec)
    say("[launch] the command, which needs the API socket that only a running launch "
        "provides:")
    print(shlex.join(argv))
    return 0
