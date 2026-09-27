"""What a container session shares with the client, and the checks on it.

:func:`resolve_plan` turns the launch config and flags into a
:class:`ContainerPlan`: the shares, the private home, the named volumes and
the forwarded ports, with the warnings launch prints before the run. It
refuses a plan that would share a system or credential folder by default,
mount two things at one guest path, or cover a path the guest entry uses.
Nothing here starts a container.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from gmlx.config import (LaunchClientCfg, parse_size_bytes, parse_volume_spec)

from .state import data_dir

# Clients whose built-in default shares no current folder.
NO_CWD_CLIENTS = frozenset({"open-webui", "elia"})
DEFAULT_VOLUME_SIZE = "32G"
# Folders launch never shares by default, whatever their contents.
SYSTEM_FOLDERS = frozenset({
    "/", "/Users", "/Volumes", "/private", "/tmp", "/var", "/opt", "/usr",
    "/Library", "/System", "/Applications", "/private/tmp", "/private/var"})
# Paths under $HOME that hold credentials or gmlx's own data.
SENSITIVE = (".ssh", ".gnupg", ".aws", ".azure", ".config/gcloud", ".kube",
             ".docker", ".password-store", "Library/Keychains", ".netrc",
             ".config/gh", ".npmrc", ".git-credentials", ".config/gmlx",
             ".cache/gmlx", ".local/share/gmlx")
# Guest paths no mount may cover.
RESERVED_TARGETS = ("/proc", "/sys", "/dev", "/opt/gmlx", "/var/host-services")
# Folders macOS guards with a privacy prompt for the container runtime.
PROTECTED = ("Desktop", "Documents", "Downloads", "Library/Mobile Documents")
CONFIG_READ_MAX = 1 << 20
MEMORY_WARN_FRACTION = 0.25


class SettingsError(ValueError):
    """The container settings cannot run. The message says what to change."""


@dataclass(frozen=True)
class Mount:
    """One mount of the session. ``source`` is a Mac realpath, or a volume
    name when ``kind`` is ``volume``."""
    source: str
    target: str
    readonly: bool = False
    kind: str = "share"               # share, git, home, runtime or volume
    note: str = ""
    size: str | None = None           # volumes only


@dataclass
class ContainerPlan:
    client: str
    home: Path
    mounts: list[Mount]
    workdir: str
    cwd_shared: bool
    forward: list[int]
    network: str
    cpus: int
    memory: str
    ssh_agent: bool
    env: list[str]
    open_browser: bool
    clipboard: str
    seed: list[str]
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def shares(self) -> list[Mount]:
        return [m for m in self.mounts if m.kind in ("share", "git")]

    @property
    def volumes(self) -> list[Mount]:
        return [m for m in self.mounts if m.kind == "volume"]


def _real(path: str | os.PathLike) -> str:
    return os.path.realpath(os.path.expanduser(str(path)))


def _inside(path: str, folder: str) -> bool:
    """True when ``path`` is ``folder`` or lies inside it, by whole path
    components."""
    if folder == "/":
        return path.startswith("/")
    return path == folder or path.startswith(folder.rstrip("/") + "/")


def _host_home() -> str:
    return _real(os.path.expanduser("~"))


def _tilde(path: str, home: str | None = None) -> str:
    home = home or _host_home()
    return "~" + path[len(home):] if _inside(path, home) else path


def sensitive_paths(home: str | None = None) -> list[str]:
    home = home or _host_home()
    out = [_real(os.path.join(home, p)) for p in SENSITIVE]
    for var, sub in (("XDG_CACHE_HOME", "gmlx"), ("XDG_DATA_HOME", "gmlx"),
                     ("XDG_CONFIG_HOME", "gmlx")):
        if os.environ.get(var):
            out.append(_real(os.path.join(os.environ[var], sub)))
    return list(dict.fromkeys(out))


def sensitive_hits(path: str, home: str | None = None) -> list[str]:
    """The sensitive paths that ``path`` is, contains or lies inside."""
    return [s for s in sensitive_paths(home) if _inside(path, s) or _inside(s, path)]


def auto_share_refusal(path: str, home: str | None = None) -> str | None:
    """Why launch will not share ``path`` by default, or None."""
    home = home or _host_home()
    if path in SYSTEM_FOLDERS:
        return f"{path} is a system folder"
    if _inside(home, path):
        return f"{_tilde(path, home)} is your home folder or holds it"
    if _inside(path, _real(data_dir())):
        return f"{_tilde(path, home)} holds container-mode data"
    hits = sensitive_hits(path, home)
    if hits:
        return (f"{_tilde(path, home)} holds or lies in a credential folder "
                f"({', '.join(_tilde(h, home) for h in hits)})")
    return None


def parse_mount_spec(spec: str) -> tuple[str, str | None, bool]:
    """``(source, target or None, readonly)`` for ``PATH[:DST][:ro]``."""
    parts = str(spec).split(":")
    readonly = False
    if len(parts) > 1 and parts[-1] in ("ro", "rw"):
        readonly = parts.pop() == "ro"
    if len(parts) > 2 or not parts[0]:
        raise SettingsError(f"mount {spec!r} is not PATH[:DST][:ro].")
    target = parts[1] if len(parts) == 2 else None
    if target is not None and not target.startswith("/"):
        raise SettingsError(f"mount {spec!r}: the guest path {target!r} must be absolute.")
    return os.path.expanduser(parts[0]), target, readonly


def _check_mount_chars(path: str, what: str) -> None:
    if "," in path or "=" in path:
        raise SettingsError(f"{what} {path} contains ',' or '=', which "
                            "`container run --mount` cannot take.")


def _explicit_mount(spec: str, plan_warnings: list[str], home: str) -> Mount:
    source, target, readonly = parse_mount_spec(spec)
    real = _real(source)
    if not os.path.exists(real):
        raise SettingsError(f"mount {spec!r}: {source} does not exist.")
    if not os.path.isdir(real):
        raise SettingsError(f"mount {spec!r}: {source} is not a folder. Share the "
                            "folder that holds the file.")
    hits = sensitive_hits(real, home)
    if hits:
        plan_warnings.append(
            f"[launch] warning: the mount {_tilde(real, home)} gives the client "
            f"{', '.join(_tilde(h, home) for h in hits)}.")
    return Mount(real, target or real, readonly)


def _volume_mount(spec: str) -> Mount:
    name, target, size = parse_volume_spec(spec)
    return Mount(name, target, kind="volume", size=size or DEFAULT_VOLUME_SIZE)


def normalize_mounts(mounts: list[Mount]) -> list[Mount]:
    """Drop duplicates, refuse two mounts at one guest path and any mount
    over or inside a reserved path, and order by guest path depth so a
    parent is mounted before anything inside it. A duplicate differs at most
    in its note, so ``--mount .`` beside the current-folder share is dropped."""
    out: list[Mount] = []
    seen: set[tuple] = set()
    by_target: dict[str, Mount] = {}
    for m in mounts:
        key = (m.source, os.path.normpath(m.target), m.readonly, m.kind, m.size)
        if key in seen:
            continue
        seen.add(key)
        target = os.path.normpath(m.target)
        if target == "/":
            raise SettingsError(f"{_label(m)} cannot be mounted at /.")
        for reserved in RESERVED_TARGETS:
            if _inside(target, reserved) or _inside(reserved, target):
                raise SettingsError(f"{_label(m)} cannot be mounted at {target}, which "
                                    f"covers {reserved}.")
        if m.kind != "volume":
            _check_mount_chars(m.source, "the folder")
        _check_mount_chars(target, "the guest path")
        other = by_target.get(target)
        if other is not None:
            raise SettingsError(f"{_label(other)} and {_label(m)} both mount at {target}.")
        by_target[target] = m
        out.append(m)
    return sorted(out, key=lambda m: (m.target.rstrip("/").count("/"), m.target))


def _label(m: Mount) -> str:
    if m.kind == "volume":
        return f"the volume {m.source}"
    if m.kind == "home":
        return "the private home"
    return _tilde(m.source)


def guest_path(host_path: str, mounts: list[Mount]) -> str | None:
    """Where ``host_path`` appears in the guest, through the share with the
    longest matching source, or None when no share holds it."""
    best = None
    for m in mounts:
        if m.kind in ("share", "git") and _inside(host_path, m.source):
            if best is None or len(m.source) > len(best.source):
                best = m
    if best is None:
        return None
    rest = host_path[len(best.source):].lstrip("/")
    return best.target.rstrip("/") + ("/" + rest if rest else "") or "/"


def forward_ports(ports: list[int], *, api_port: int | None,
                  web_port: int | None) -> list[int]:
    out = []
    for port in ports:
        if port == api_port:
            raise SettingsError(f"forward: {port} is the gmlx server's port, which the "
                                "guest already reaches at 127.0.0.1.")
        if port == web_port:
            raise SettingsError(f"forward: {port} is the browser app's own web port.")
        if port not in out:
            out.append(port)
    return out


def _git(cwd: str, *args: str) -> list[str] | None:
    try:
        proc = subprocess.run(["git", "-C", cwd, *args], capture_output=True, text=True,
                              timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.splitlines() if proc.returncode == 0 else None


def git_extra_mount(cwd: str, shares: list[Mount], home: str | None = None
                    ) -> tuple[Mount | None, list[str]]:
    """The git folder a linked worktree needs, when no share covers it, and
    the notes launch prints about git."""
    home = home or _host_home()
    out = _git(cwd, "rev-parse", "--path-format=absolute", "--show-toplevel",
               "--git-dir", "--git-common-dir")
    if not out or len(out) != 3:
        return None, []
    toplevel, git_dir, common = (_real(p) for p in out)
    if auto_share_refusal(toplevel, home) is not None:
        return None, []

    def covered(path):
        return any(_inside(path, m.source) for m in shares)
    if not covered(toplevel):
        if covered(cwd):
            return None, [f"[launch] git in the container needs the repository root, "
                          f"{_tilde(toplevel, home)}, which is not shared. Launch from it "
                          "to use git there."]
        return None, []
    if git_dir == common or covered(common):
        return None, []
    return Mount(common, common, kind="git", note="the git folder of this worktree"), []


def protected_folder_warnings(mounts: list[Mount], home: str | None = None) -> list[str]:
    """Shares in folders where macOS asks before the container runtime reads."""
    home = home or _host_home()
    guarded = [_real(os.path.join(home, p)) for p in PROTECTED] + ["/Volumes"]
    out = []
    for m in mounts:
        if m.kind not in ("share", "git"):
            continue
        hit = next((g for g in guarded if _inside(m.source, g)), None)
        if hit:
            out.append(f"[launch] {_tilde(m.source, home)} is in {_tilde(hit, home)}, which "
                       "macOS guards. macOS may ask once whether the container runtime can "
                       "read it, and the container waits until you answer.")
    return out


def private_home(client: str) -> Path:
    """The client's persistent home, shared at the same path in the guest."""
    home = data_dir() / client / "home"
    home.mkdir(parents=True, exist_ok=True)
    os.chmod(home, 0o700)
    return home


def _mount_cwd(client: str, flag: bool | None, cfg: LaunchClientCfg) -> bool:
    if flag is not None:
        return flag
    if cfg.mount_cwd is not None:
        return cfg.mount_cwd
    return client not in NO_CWD_CLIENTS


def memory_warning(memory: str) -> str | None:
    """A note when the container's memory is over a quarter of the Mac's."""
    size = parse_size_bytes(memory)
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        return None
    if size is None or size <= total * MEMORY_WARN_FRACTION:
        return None
    return (f"[launch] the container gets {memory} of the Mac's "
            f"{total / (1 << 30):.0f} GB, which the model server cannot use while it runs.")


def resolve_plan(client: str, cfg: LaunchClientCfg, *, cwd: str,
                 mount_cwd: bool | None = None, cli_mounts: list[str] = (),
                 network: str | None = None, api_port: int | None = None,
                 web_port: int | None = None) -> ContainerPlan:
    """The mounts, volumes and ports of one session, from the effective
    client config and the flags."""
    home = _host_home()
    warns: list[str] = []
    notes: list[str] = []
    mounts: list[Mount] = []
    cwd_real = _real(cwd)
    share_cwd = _mount_cwd(client, mount_cwd, cfg)
    if share_cwd:
        why = auto_share_refusal(cwd_real, home)
        if why is not None:
            raise SettingsError(f"launch will not share the current folder: {why}. Launch "
                                "from a project folder, or pass --no-mount-cwd.")
        mounts.append(Mount(cwd_real, cwd_real, note="working folder"))
    for spec in [*cfg.mounts, *cli_mounts]:
        mounts.append(_explicit_mount(spec, warns, home))
    git_mount, git_notes = (git_extra_mount(cwd_real, list(mounts), home)
                            if share_cwd else (None, []))
    if git_mount is not None:
        mounts.append(git_mount)
    notes.extend(git_notes)
    guest_home = private_home(client)
    mounts.append(Mount(str(guest_home), str(guest_home), kind="home"))
    mounts.extend(_volume_mount(v) for v in cfg.volumes)
    mounts = normalize_mounts(mounts)
    guest_cwd = guest_path(cwd_real, mounts)
    warns.extend(protected_folder_warnings(mounts, home))
    mem = memory_warning(cfg.memory or "4G")
    if mem:
        notes.append(mem)
    return ContainerPlan(
        client=client, home=guest_home, mounts=mounts,
        workdir=guest_cwd or str(guest_home), cwd_shared=guest_cwd is not None,
        forward=forward_ports(list(cfg.forward), api_port=api_port, web_port=web_port),
        network=network or cfg.network or "default", cpus=cfg.cpus or 4,
        memory=cfg.memory or "4G", ssh_agent=bool(cfg.ssh_agent), env=list(cfg.env),
        open_browser=cfg.open_browser is not False, clipboard=cfg.clipboard or "off",
        seed=list(cfg.seed), warnings=warns, notes=notes)


# The guest environment and the private home

def _host_tz() -> str | None:
    if os.environ.get("TZ"):
        return os.environ["TZ"]
    try:
        link = os.readlink("/etc/localtime")
    except OSError:
        return None
    marker = "zoneinfo/"
    return link.split(marker, 1)[1] if marker in link else None


def guest_env(home: Path) -> dict[str, str]:
    """The baseline the guest gets by value. None of it is secret."""
    term = os.environ.get("TERM", "")
    safe = term and all(c.isalnum() or c in "._+-" for c in term) and term != "dumb"
    env = {"HOME": str(home), "TERM": term if safe else "xterm-256color",
           "LANG": "C.UTF-8"}
    if os.environ.get("COLORTERM") in ("truecolor", "24bit"):
        env["COLORTERM"] = os.environ["COLORTERM"]
    tz = _host_tz()
    if tz:
        env["TZ"] = tz
    return env


def seed_home(home: Path, seeds: list[str]) -> list[str]:
    """Copy each seed into the private home once, at the same path relative
    to ``$HOME``, then add the host git identity where it is missing.
    Returns the warnings to print."""
    # Relative to $HOME as written, so a seed that is a link into a dotfiles
    # repository still lands at its own path.
    host_home = os.path.abspath(os.path.expanduser("~"))
    out = []
    for seed in seeds:
        src = os.path.abspath(os.path.expanduser(seed))
        if not _inside(src, host_home) or src == host_home:
            raise SettingsError(f"seed: {seed} is not inside your home folder.")
        if not os.path.exists(src):
            out.append(f"[launch] seed: {seed} does not exist, so nothing was copied.")
            continue
        hits = sensitive_hits(_real(src))
        if hits:
            out.append(f"[launch] warning: seed {_tilde(src, host_home)} copies "
                       f"{', '.join(_tilde(h) for h in hits)} into the private home.")
        dst = home / os.path.relpath(src, host_home)
        if dst.exists() or dst.is_symlink():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if os.path.isdir(src):
            shutil.copytree(src, dst, symlinks=True)
        else:
            shutil.copy2(src, dst)
    _seed_git_identity(home)
    return out


def _seed_git_identity(home: Path) -> None:
    gitconfig = str(home / ".gitconfig")
    for key in ("user.name", "user.email"):
        try:
            have = subprocess.run(["git", "config", "--file", gitconfig, "--get", key],
                                  capture_output=True, text=True, timeout=5)
            if have.returncode == 0:
                continue
            value = subprocess.run(["git", "config", "--global", "--get", key],
                                   capture_output=True, text=True, timeout=5)
            if value.returncode == 0 and value.stdout.strip():
                subprocess.run(["git", "config", "--file", gitconfig, key,
                                value.stdout.strip()], capture_output=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            return


# The server config the guest could change

def server_config_path(host: str, port: int) -> str | None:
    """The config file the target server runs with: the one in its runfile
    while it runs, else the one autostart would use."""
    from gmlx.config import default_config_paths
    from gmlx.serve import lifecycle

    run = lifecycle.read_run(host, port) or {}
    if run.get("config_abspath") and lifecycle.pid_alive(run.get("pid")):
        return str(run["config_abspath"])
    for path in default_config_paths():
        if path.exists():
            return str(path)
    return None


def _read_small_file(path: str) -> bytes:
    """The file's bytes, refusing anything but a regular file of at most
    1 MiB without ever blocking on it."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise OSError("not a regular file")
        if st.st_size > CONFIG_READ_MAX:
            raise OSError(f"larger than {CONFIG_READ_MAX >> 20} MiB")
        chunks, left = [], CONFIG_READ_MAX + 1
        while left > 0:
            chunk = os.read(fd, min(left, 65536))
            if not chunk:
                break
            chunks.append(chunk)
            left -= len(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _expand(path: str) -> str:
    return _real(os.path.expandvars(path))


def _model_paths(cfg) -> list[str]:
    """The files the config lists, resolved the way the server resolves them,
    without reading them."""
    out = []
    roots = [_expand(d) for d in cfg.model_dirs]
    for model in cfg.models.values():
        for p in (model.path, model.mmproj, model.draft_gguf, model.adapter):
            if not p or str(p).startswith("hf:"):
                continue
            p = os.path.expandvars(os.path.expanduser(str(p)))
            if os.path.isabs(p):
                out.append(_real(p))
                continue
            for root in roots:
                cand = os.path.join(root, p)
                if os.path.exists(cand):
                    out.append(_real(cand))
                    break
    return out


def server_config_warnings(config_path: str | None, shares: list[Mount]) -> list[str]:
    """Warnings for a server config, model folder or model file the client
    can change through a read-write share. Never scans a model folder and
    never blocks, so a planted file cannot stall the launch."""
    if not config_path:
        return []
    import yaml

    from gmlx.config import build_config

    home = _host_home()
    rw = [m for m in shares if m.kind in ("share", "git") and not m.readonly]
    real = _real(config_path)
    out = []
    in_share = next((m for m in rw if _inside(real, m.source)), None)
    if in_share is not None:
        out.append(f"[launch] warning: the server config {_tilde(real, home)} is inside the "
                   f"read-write share {_tilde(in_share.source, home)}. The client can change "
                   "it, and the server applies the change at its next reload.")
    try:
        doc = yaml.safe_load(_read_small_file(real))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = build_config(doc if isinstance(doc, dict) else {})
    except Exception as e:  # noqa: BLE001 - a broken config must never stop the launch
        out.append(f"[launch] could not check the server config {_tilde(real, home)} "
                   f"({str(e).splitlines()[0] if str(e) else type(e).__name__}).")
        return out
    for spec in cfg.discover:
        folders = [spec.dir] if spec.dir else list(cfg.model_dirs)
        for folder in folders:
            f = _expand(folder)
            for m in rw:
                if _inside(f, m.source) or (spec.recursive and _inside(m.source, f)):
                    out.append(f"[launch] warning: the server scans {_tilde(f, home)} for "
                               f"models, and the client can add files there through "
                               f"{_tilde(m.source, home)}.")
                    break
    for path in _model_paths(cfg):
        m = next((m for m in rw if _inside(path, m.source)), None)
        if m is not None:
            out.append(f"[launch] warning: the model file {_tilde(path, home)} is inside the "
                       "read-write share, and the server reads it again at its next load.")
    return list(dict.fromkeys(out))
