"""What a container session shares with the client, and the checks on it.

:func:`resolve_plan` turns the launch config and flags into a
:class:`ContainerPlan`: the shares, the private home, the named volumes and
the forwarded ports, with the warnings launch prints before the run. It
refuses a plan that would share a system or credential folder by default,
mount two things at one guest path, or cover a path the guest entry uses.
Nothing here starts a container.
"""

from __future__ import annotations

import contextlib
import os
import re
import secrets
import stat
import subprocess
import tempfile
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from gmlx.config import (LaunchClientCfg, normal_guest_target, parse_size_bytes,
                         parse_volume_spec)

from . import notices
from .notices import Once
from .state import canonical, data_dir, data_path, fd_path, path_inside, write_record

# Clients whose built-in default shares no current folder.
NO_CWD_CLIENTS = frozenset({"open-webui", "elia"})
DEFAULT_VOLUME_SIZE = "32G"
# Folders launch never shares by default, whatever their contents.
SYSTEM_FOLDERS = frozenset({
    "/", "/Users", "/Volumes", "/private", "/tmp", "/var", "/opt", "/usr",
    "/Library", "/System", "/Applications", "/private/tmp", "/private/var",
    "/System/Volumes", "/System/Volumes/Data"})
# The per-user temporary and cache folders of macOS. Launch never shares
# anything inside them by default.
TEMP_FOLDERS = "/private/var/folders"
# The folders gmlx keeps in /tmp when TMPDIR is unset: a server's session
# sockets and a launch session folder.
_GMLX_TEMP_FOLDER = re.compile(r"^gmlx-(?:sessions|launch)-", re.IGNORECASE)
# Paths that hold credentials, gmlx's own data, or programs and settings the
# Mac runs, by what they hold. Relative ones are under $HOME.
CREDENTIAL_PATHS = (".ssh", ".gnupg", ".aws", ".azure", ".config/gcloud", ".kube",
                    ".docker", ".password-store", "Library/Keychains", ".netrc",
                    ".config/gh", ".npmrc", ".git-credentials", ".config/gmlx",
                    ".cache/huggingface", ".codex")
GMLX_DATA_PATHS = (".cache/gmlx", ".local/share/gmlx")
RUN_PATHS = ("Library/LaunchAgents", ".config/git", ".local/bin", "bin",
             "Library/Application Support", ".cargo", "/opt/homebrew", "/usr/local")
SENSITIVE = CREDENTIAL_PATHS + GMLX_DATA_PATHS + RUN_PATHS
_HOLDS = {"credentials": CREDENTIAL_PATHS, "gmlx's own data": GMLX_DATA_PATHS,
          "files the Mac runs": RUN_PATHS}
# The folders where each client keeps its settings, history and sign-in on
# the Mac, under $HOME. A hook a guest adds there runs on the Mac, and the
# host-mode configs there hold the server key. Seeds may copy from them.
CLIENT_PATHS = {".claude": "claude-code", ".pi": "pi", ".omp": "omp", ".hermes": "hermes",
                ".open-webui": "open-webui", ".dsh": "dsh", ".config/goose": "goose",
                ".config/opencode": "opencode", ".local/share/opencode": "opencode",
                ".config/elia": "elia"}
# Client files that hold a sign-in token. Seeding one gives it to the client.
TOKEN_FILES = (".claude.json", ".claude/.credentials.json",
               ".local/share/opencode/auth.json", ".config/goose/secrets.yaml")
# Files that can hold a token, such as git's url.<base>.insteadOf with a
# token in the URL.
TOKEN_MAYBE_FILES = (".gitconfig",)
# How many read-write shares launch remembers for the seed check.
SHARED_HISTORY_MAX = 500
# The most a seed copies, so a planted or sparse file cannot fill the disk
# or the memory.
SEED_MAX_BYTES = 64 << 20
SEED_MAX_FILES = 10_000
SEED_MAX_DEPTH = 64
# Guest paths no mount may cover.
# The guest paths no mount may cover, each with what launch or Linux keeps
# there.
RESERVED_TARGETS = {
    "/proc": "which Linux in the container provides",
    "/sys": "which Linux in the container provides",
    "/dev": "which Linux in the container provides",
    "/opt/gmlx": "where launch keeps its own program",
    "/var/host-services": "where launch puts the sockets that reach the Mac",
    "/run/gmlx-session": "where launch keeps the state of a session",
}
# Folders macOS guards with a privacy prompt for the container runtime.
PROTECTED = ("Desktop", "Documents", "Downloads", "Library/Mobile Documents")
CONFIG_READ_MAX = 1 << 20
# The Mac's ~/.claude.json holds the history of every project, so launch
# reads a larger one for its theme.
CLAUDE_JSON_READ_MAX = 64 << 20
MEMORY_WARN_FRACTION = 0.25
# The memory that Apple container gives each virtual machine on top of the
# container's own, which Apple names guestMemoryOverhead.
VM_MEMORY_OVERHEAD = 128 << 20
# The key of a session that shares no current folder.
PROJECT_DEFAULT = "default"
PROJECT_NAME_MAX = 32
# The longest volume name Apple container takes.
VOLUME_NAME_MAX = 255
# The PATH of the programs launch runs by name: the folders of Homebrew and
# of the system, which launch never shares by default.
SYSTEM_PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"


class SettingsError(ValueError):
    """The container settings cannot run. The message says what to change."""


class Busy(SettingsError):
    """Something the session needs, such as a volume or a port, is in use,
    so a later launch can work."""


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
    worktree: str = ""                # git only: the project it serves


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
    project: str = "default"          # the project id the session keys
    new_home: bool = False            # the private home did not exist before
    ssh_socket: str | None = None     # the agent socket that ssh_agent names

    @property
    def shares(self) -> list[Mount]:
        return [m for m in self.mounts if m.kind in ("share", "git")]

    @property
    def volumes(self) -> list[Mount]:
        return [m for m in self.mounts if m.kind == "volume"]


def _real(path: str | os.PathLike) -> str:
    """The path with its links resolved, in the form macOS gives it, so a
    firmlink alias or another case never passes a check by path."""
    return canonical(path)


def _inside(path: str, folder: str) -> bool:
    """True when ``path`` is ``folder`` or lies inside it, by whole path
    components, ignoring case on a volume that ignores it."""
    return path_inside(path, folder)


def _same(a: str, b: str) -> bool:
    """One path, ignoring case on a volume that ignores it."""
    return _inside(a, b) and _inside(b, a)


def _host_home() -> str:
    return _real(os.path.expanduser("~"))


def _tilde(path: str, home: str | None = None) -> str:
    home = home or _host_home()
    return "~" + path[len(home):] if _inside(path, home) else path


def _sensitive_kinds(home: str) -> dict[str, str]:
    """Each sensitive path, with what it holds, such as "credentials"."""
    out: dict[str, str] = {}
    for what, paths in _HOLDS.items():
        for p in paths:
            out.setdefault(_real(os.path.join(home, p)), what)
    for var, what in (("XDG_CACHE_HOME", "gmlx's own data"),
                      ("XDG_DATA_HOME", "gmlx's own data"),
                      ("XDG_CONFIG_HOME", "credentials")):
        if os.environ.get(var):
            out.setdefault(_real(os.path.join(os.environ[var], "gmlx")), what)
    return out


def sensitive_paths(home: str | None = None) -> list[str]:
    return list(_sensitive_kinds(home or _host_home()))


def sensitive_hits(path: str, home: str | None = None) -> list[str]:
    """The sensitive paths that ``path`` is, contains or lies inside."""
    return [s for s in sensitive_paths(home) if _inside(path, s) or _inside(s, path)]


def temp_trees() -> list[str]:
    """The folders that hold the temporary and cache files of every program
    of this user, gmlx's fallback socket folders among them. /tmp is not
    one: every user keeps scratch folders there, and only gmlx's own folders
    in it are refused, by :func:`_gmlx_temp_refusal`."""
    trees = [_real(TEMP_FOLDERS)]
    for tmp in (os.environ.get("TMPDIR"), tempfile.gettempdir()):
        if tmp and not _same(_real(tmp), _real("/tmp")):
            trees.append(_real(tmp))
    return list(dict.fromkeys(trees))


def _gmlx_temp_refusal(path: str) -> str | None:
    """Why ``path`` is not shared by default when it is or lies in a folder
    that gmlx keeps in /tmp: the session sockets of a server, or a launch
    session folder."""
    root = _real("/tmp")
    if not _inside(path, root) or _same(path, root):
        return None
    top = os.path.relpath(path, root).split(os.sep)[0]
    if not _GMLX_TEMP_FOLDER.match(top):
        return None
    folder = os.path.join(root, top)
    verb = "is" if _same(path, folder) else "lies in"
    return f"{verb} {folder}, which holds the session sockets of gmlx"


def _gmlx_temp_share_refusal(path: str) -> str | None:
    """Why launch never shares ``path``, even when you name it: it is, holds
    or lies in a folder where gmlx keeps session sockets in a temporary
    folder, or it holds the temporary folder where gmlx makes them. A
    client could replace a socket path there with a link."""
    tmp = os.environ.get("TMPDIR") or "/tmp"
    active = {_real(tmp), _real(tempfile.gettempdir())}
    for root in dict.fromkeys([*active, _real("/tmp")]):
        if _inside(path, root) and not _same(path, root):
            top = os.path.relpath(path, root).split(os.sep)[0]
            if _GMLX_TEMP_FOLDER.match(top):
                folder = os.path.join(root, top)
                verb = "is" if _same(path, folder) else "lies in"
                return f"{verb} {folder}, which holds the session sockets of gmlx"
        elif _inside(root, path):
            try:
                held = sorted(n for n in os.listdir(root) if _GMLX_TEMP_FOLDER.match(n))
            except OSError:
                held = []
            if root in active or held:
                verb = "is" if _same(path, root) else "holds"
                return f"{verb} {root}, where gmlx keeps the session sockets of its servers"
    return None


def _temp_tree_relation(path: str) -> str | None:
    """How ``path`` meets a folder with the temporary files of your
    programs, as a phrase that follows the path, or None."""
    what = "the temporary files of your programs"
    for tree in temp_trees():
        if _same(path, tree):
            return f"holds {what}"
        if _inside(path, tree) or _inside(tree, path):
            verb = "lies in" if _inside(path, tree) else "holds"
            return f"{verb} {tree}, which holds {what}"
    return None


def auto_share_refusal(path: str, home: str | None = None) -> str | None:
    """Why launch will not share ``path`` by default, as a phrase that
    follows the path, such as "is your home folder", or None."""
    home = home or _host_home()
    if any(_same(path, f) for f in SYSTEM_FOLDERS):
        return "is a system folder"
    trees = [t for t in temp_trees() if _inside(path, t)]
    if trees:
        tree = max(trees, key=len)
        what = "holds the temporary files of your programs"
        return what if _same(path, tree) else f"lies in {tree}, which {what}"
    why = _gmlx_temp_refusal(path)
    if why is not None:
        return why
    if _same(path, home):
        return "is your home folder"
    if _inside(home, path):
        return "holds your home folder"
    return (_data_refusal(path, home) or _sensitive_refusal(path, home)
            or _client_refusal(path, home))


def _relation(path: str, folder: str, home: str, what: str) -> str:
    """``is FOLDER, WHAT``, ``lies in FOLDER, WHAT`` or ``holds FOLDER, WHAT``."""
    verb = ("is" if _same(path, folder) else "lies in" if _inside(path, folder)
            else "holds")
    return f"{verb} {_tilde(folder, home)}, {what}"


def _data_refusal(path: str, home: str) -> str | None:
    """Why ``path`` never reaches a container: it holds or lies in the
    folder with every private home and the guest entry."""
    data = _real(data_path())
    if _inside(path, data) or _inside(data, path):
        return _relation(path, data, home, "where launch keeps the private homes of the "
                                           "clients")
    return None


def _state_refusal(path: str, home: str) -> str | None:
    """Why ``path`` never reaches a container: it holds or lies in gmlx's
    settings or its server state. A client that writes the server's runfile
    there chooses the config file that later launches read."""
    folders = [os.path.join(home, ".config/gmlx"), os.path.join(home, ".cache/gmlx")]
    for var in ("XDG_CONFIG_HOME", "XDG_CACHE_HOME"):
        if os.environ.get(var):
            folders.append(os.path.join(os.environ[var], "gmlx"))
    for folder in dict.fromkeys(_real(f) for f in folders):
        if _inside(path, folder) or _inside(folder, path):
            return _relation(path, folder, home,
                             "where gmlx keeps its settings and server state")
    return None


def _client_folders(home: str) -> dict[str, str]:
    """Each folder where a client keeps its settings and history on the
    Mac, by real path, with the client's name."""
    out = {_real(os.path.join(home, rel)): client for rel, client in CLIENT_PATHS.items()}
    for var, client in (("HERMES_HOME", "hermes"), ("DSH_HOME", "dsh")):
        value = os.environ.get(var, "").strip()
        if value:
            out.setdefault(_real(os.path.expanduser(value)), client)
    return out


def _client_refusal(path: str, home: str) -> str | None:
    """How ``path`` meets a folder where a client keeps its settings and
    history on the Mac, as a phrase that follows the path, or None."""
    for folder, client in _client_folders(home).items():
        where = f"where {client} keeps its settings and history on the Mac"
        if _same(path, folder):
            return f"is {where}"
        if _inside(path, folder) or _inside(folder, path):
            verb = "lies in" if _inside(path, folder) else "holds"
            return f"{verb} {_tilde(folder, home)}, {where}"
    return None


def _sensitive_refusal(path: str, home: str) -> str | None:
    """How ``path`` meets the folders that hold credentials, gmlx's own data
    or files the Mac runs, as a phrase that follows the path, or None. It
    names only the kinds that apply."""
    kinds = _sensitive_kinds(home)
    hits = sensitive_hits(path, home)
    if not hits:
        return None
    same = [h for h in hits if _same(path, h)]
    if same:
        return f"holds {kinds[same[0]]}"
    outer = [h for h in hits if _inside(path, h)]
    if outer:
        folder = max(outer, key=len)
        return f"lies in {_tilde(folder, home)}, which holds {kinds[folder]}"
    what = list(dict.fromkeys(kinds[h] for h in hits))
    listed = what[0] if len(what) == 1 else f"{', '.join(what[:-1])} and {what[-1]}"
    return f"holds {', '.join(_tilde(h, home) for h in hits)}, which hold {listed}"


def parse_mount_spec(spec: str) -> tuple[str, str | None, bool]:
    """``(source, target or None, readonly)`` for ``PATH[:DST][:ro]``."""
    parts = str(spec).split(":")
    readonly = False
    if len(parts) > 1 and parts[-1] in ("ro", "rw"):
        readonly = parts.pop() == "ro"
    if len(parts) > 2 or not parts[0]:
        raise SettingsError(f"the share {spec} is not in the form PATH[:DST][:ro].")
    target = parts[1] if len(parts) == 2 else None
    if target is not None and not target.startswith("/"):
        raise SettingsError(f"the share {spec} names the container path {target}, which must "
                            "start with /.")
    return os.path.expanduser(parts[0]), target, readonly


def check_mount_chars(path: str, what: str) -> None:
    if "," in path or "=" in path:
        raise SettingsError(f"{what} {path} contains a comma or an equals sign, which "
                            "Apple container cannot take in a share.")


def _explicit_mount(spec: str, plan_warnings: list[str], home: str) -> Mount:
    source, target, readonly = parse_mount_spec(spec)
    real = _real(source)
    written = os.path.abspath(source)
    shown = _tilde(written, home)
    if not os.path.exists(real):
        raise SettingsError(f"the share {shown} does not exist.")
    if not os.path.isdir(real):
        raise SettingsError(f"the share {shown} is not a folder. Share the folder that "
                            "holds it.")
    # A folder a client could write in an earlier, wider share may now be a
    # link to somewhere else, so a mount is taken only by its real path.
    if not _same(written, real):
        raise SettingsError(f"the share {shown} is a symbolic link to "
                            f"{_tilde(real, home)}, or passes through one. Write the "
                            f"folder's real path, {_tilde(real, home)}, if you mean it.")
    why = (_data_refusal(real, home) or _state_refusal(real, home)
           or _gmlx_temp_share_refusal(real))
    if why is not None:
        raise SettingsError(f"will not share {shown}, because it {why}. Share a project "
                            "folder instead.")
    why = (_sensitive_refusal(real, home) or _client_refusal(real, home)
           or _temp_tree_relation(real))
    if why is not None:
        can = "read" if readonly else "read and change"
        plan_warnings.append(f"[launch] warning: the share {shown} {why}. The client can "
                             f"{can} every file in it.")
    return Mount(real, target or real, readonly)


def _volume_mount(spec: str, project: str | None = None) -> Mount:
    name, target, size = parse_volume_spec(spec)
    if project is not None:
        name = project_volume_name(name, project)
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
        target = normal_guest_target(m.target)
        m = replace(m, target=target)
        key = (m.source, target, m.readonly, m.kind, m.size)
        if key in seen:
            continue
        seen.add(key)
        if target == "/":
            raise SettingsError(f"{_label(m)} cannot use / in the container. Choose a folder "
                                "below it.")
        for reserved, what in RESERVED_TARGETS.items():
            if target == reserved:
                where = f"{target}, {what}"
            elif _inside(target, reserved):
                where = f"{target}, inside {reserved}, {what}"
            elif _inside(reserved, target):
                where = f"{target}, which would cover {reserved}, {what}"
            else:
                continue
            raise SettingsError(f"{_label(m)} cannot use {where}. Choose another path in "
                                "the container.")
        if m.kind != "volume":
            check_mount_chars(m.source, "the folder")
        check_mount_chars(target, "the container path")
        other = by_target.get(target)
        if other is not None:
            raise SettingsError(f"{_label(other)} and {_label(m)} both use {target} in the "
                                "container. Give one of them another path.")
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
    # By components, since the source may differ from the path in case.
    rest = "/".join(host_path.split("/")[len(best.source.rstrip("/").split("/")):])
    return best.target.rstrip("/") + ("/" + rest if rest else "") or "/"


def forward_ports(ports: list[int], *, api_port: int | None,
                  web_port: int | None) -> list[int]:
    out = []
    for port in ports:
        if port == api_port:
            raise SettingsError(f"forward lists {port}, the gmlx server's port, which the "
                                f"container reaches already. Remove {port} from forward.")
        if port == web_port:
            raise SettingsError(f"forward lists {port}, the web app's own port. Remove "
                                f"{port} from forward.")
        if port not in out:
            out.append(port)
    return out


def _system_env(**extra: str) -> dict[str, str]:
    """This process's environment with :data:`SYSTEM_PATH`, so a program
    that a client puts in a shared folder on PATH never runs in place of
    git or ssh-add."""
    return {**os.environ, "PATH": SYSTEM_PATH, **extra}


def _git(cwd: str, *args: str) -> list[str] | None:
    try:
        # The repository config is the guest's, so no command it names runs.
        proc = subprocess.run(["git", "-c", "core.fsmonitor=false", "-C", cwd, *args],
                              capture_output=True, text=True, timeout=5, env=_system_env())
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

    def covering(path):
        return max((m for m in shares if _inside(path, m.source)),
                   key=lambda m: len(m.source), default=None)
    if not covered(toplevel):
        if covered(cwd):
            return None, [f"[launch] git in the container needs the repository root, "
                          f"{_tilde(toplevel, home)}, which is not shared. Launch from it "
                          "to use git there."]
        return None, []
    if covered(common):
        return None, []
    # git finds the git folder through files in the share, which the guest
    # can change: the .git file, and a commondir file in a .git folder. So
    # an outside git folder is shared only when it names this project back,
    # from a file outside the share.
    # A folder the guest could write in an earlier, wider launch can be
    # made to look like a git folder that names this project back. Only a
    # folder with a git folder's own name counts, and never one that holds
    # the project.
    if not _git_folder_shape(common) or _inside(toplevel, common):
        return None, [f"[launch] git in the container cannot use {_tilde(common, home)} as "
                      f"a git folder, because it is not named like one (.git, a name "
                      f"ending in .git, .bare, or a folder in .git/modules) or it holds "
                      f"{_tilde(toplevel, home)}. Share it with --mount "
                      f"{_tilde(common, home)} if you intend to."]
    back = _git_back_reference(toplevel, git_dir, common)
    if back is None:
        return None, [f"[launch] git in the container cannot use the git folder "
                      f"{_tilde(common, home)}, because it does not name "
                      f"{_tilde(toplevel, home)} as one of its worktrees or submodules. "
                      f"Share it with --mount {_tilde(common, home)} if you intend to."]
    what, exact = back
    if not exact:
        fix = ("run git worktree repair there" if what == "worktree" else
               f"set core.worktree in {_tilde(os.path.join(git_dir, 'config'), home)} "
               "to its real path")
        return None, [f"[launch] git in the container cannot use the git folder "
                      f"{_tilde(common, home)}, because it names "
                      f"{_tilde(toplevel, home)} only through a symbolic link, so launch "
                      f"does not share it. If {_tilde(toplevel, home)} is a {what} of that "
                      f"repository, {fix}, and the next launch shares the git folder."]
    # Its hooks and config run on the Mac the next time you use git there,
    # so it gets the same checks as the current folder, and so does the main
    # worktree above a .git folder, such as a dotfiles repository at $HOME.
    owner = os.path.dirname(common) if os.path.basename(common) == ".git" else None
    for path in filter(None, (common, owner)):
        why = auto_share_refusal(path, home)
        if why is not None:
            return None, [f"[launch] git in the container cannot reach this repository's "
                          f"git folder {_tilde(common, home)}, because "
                          f"{_tilde(path, home)} {why}. Use git on the Mac for this "
                          "repository."]
    earlier = _forged_back_reference(toplevel, common)
    if earlier is not None:
        check = (f"Run git worktree list in {_tilde(_git_repository(common), home)}"
                 if what == "worktree" else
                 f"Check core.worktree in {_tilde(os.path.join(git_dir, 'config'), home)}")
        return None, [f"[launch] git in the container cannot use the git folder "
                      f"{_tilde(common, home)}, because it lies in {_tilde(earlier, home)}, "
                      f"which an earlier launch shared read-write, and so does "
                      f"{_tilde(toplevel, home)}. A client could have written the records "
                      f"that make it a {what} of that repository. {check} to see whether "
                      f"you made it, then share the git folder with --mount "
                      f"{_tilde(common, home)} if you intend to."]
    # A read-only share of the repository keeps its git folder read-only.
    root_share = covering(toplevel)
    return Mount(common, common, readonly=bool(root_share and root_share.readonly),
                 kind="git", note=f"the git folder of this {what} of "
                                  f"{_tilde(_git_repository(common), home)}",
                 worktree=toplevel), []


def _forged_back_reference(toplevel: str, common: str) -> str | None:
    """The folder an earlier launch shared read-write through which a client
    could have made ``toplevel`` name ``common`` as its git folder, or None.

    That takes writing the ``.git`` file of ``toplevel`` and the record in
    ``common`` that names it back, so both must lie in such folders. A
    ``.git`` file that named ``common`` when launch first shared the git
    folder for it was written before any client could write it, so that
    pair passes. This keeps worktrees beside a main checkout that a launch
    shared."""
    history = shared_history()
    earlier = [h for h in history if _inside(common, h)]
    if not earlier or not any(_inside(toplevel, h) for h in history):
        return None
    for top, named in worktree_history():
        if _same(top, toplevel) and _same(named, common):
            return None
    return max(earlier, key=len)


def _git_repository(common: str) -> str:
    """The repository folder a git folder belongs to: the folder that holds
    ``.git``, the folder that holds a ``.bare`` git folder, or a bare
    repository itself."""
    parts = common.split("/")
    for i, part in enumerate(parts):
        if part.casefold() == ".git" and i > 0:
            return "/".join(parts[:i]) or "/"
    if parts[-1].casefold() == ".bare":
        return os.path.dirname(common)
    return common


def _git_folder_shape(path: str) -> bool:
    """Whether ``path`` is named like a git folder: ``.git``, a name ending
    in ``.git`` such as a bare repository, ``.bare`` as in a bare repository
    with its worktrees beside it, or a submodule's folder under
    ``.git/modules``."""
    parts = [p for p in path.split("/") if p]
    if parts and (parts[-1].casefold().endswith(".git") or parts[-1].casefold() == ".bare"):
        return True
    return any(parts[i].casefold() == ".git" and parts[i + 1].casefold() == "modules"
               for i in range(len(parts) - 2))


def _read_small(path: str) -> str | None:
    """A small regular file's text, or None. Never blocks on a named pipe."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    except OSError:
        return None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > 4096:
            return None
        return os.read(fd, 4096).decode(errors="replace")
    except OSError:
        return None
    finally:
        os.close(fd)


def _git_back_reference(toplevel: str, git_dir: str, common: str
                        ) -> tuple[str, bool] | None:
    """How the git folder outside the share names ``toplevel`` back:
    ``"worktree"`` or ``"submodule"``, and whether the recorded path names
    it with no symbolic link followed. None when it does not name it.

    - A linked worktree's ``<common>/worktrees/<id>/gitdir`` names the
      project's ``.git`` file, as an absolute path or relative to that
      entry.
    - A submodule's ``core.worktree`` names the project folder, as an
      absolute path or relative to the submodule's git folder.
    - A ``.git`` folder inside the share never counts, because its
      ``commondir`` file can name any repository.

    Only the recorded path as written counts. A link on the way can lie in
    a folder the guest writes, in this launch or an earlier one, so a match
    through a link is reported and never shared. git records real paths, so
    every layout it creates matches as written."""
    dotgit = os.path.join(toplevel, ".git")
    if os.path.islink(dotgit) or not os.path.isfile(dotgit):
        return None
    if git_dir != common:
        if not _same(os.path.dirname(git_dir), os.path.join(common, "worktrees")):
            return None
        named = (_read_small(os.path.join(git_dir, "gitdir")) or "").strip()
        what, want = "worktree", dotgit
    else:
        out = _git(toplevel, "config", "--file", os.path.join(git_dir, "config"),
                   "--get", "core.worktree")
        named = out[0].strip() if out else ""
        what, want = "submodule", toplevel
    if not named:
        return None
    recorded = os.path.normpath(os.path.join(git_dir, named))
    if _same(recorded, want):
        return what, True
    if _same(_real(recorded), _real(want)):
        return what, False
    return None


def protected_folder_warnings(mounts: list[Mount], home: str | None = None) -> list[str]:
    """Shares in folders where macOS asks before the container runtime reads,
    one line for each such folder, which prints once, since macOS asks once."""
    home = home or _host_home()
    guarded = [_real(os.path.join(home, p)) for p in PROTECTED] + ["/Volumes"]
    out: list[str] = []
    for m in mounts:
        if m.kind not in ("share", "git"):
            continue
        hit = next((g for g in guarded if _inside(m.source, g)), None)
        if hit is None or any(isinstance(line, Once) and line.key == f"guarded:{hit}"
                              for line in out):
            continue
        where = (f"{_tilde(hit, home)} is a folder that macOS guards" if _same(m.source, hit)
                 else f"{_tilde(m.source, home)} is in {_tilde(hit, home)}, which macOS "
                      "guards")
        out.append(Once(f"[launch] {where}. macOS may ask once whether the container runtime "
                        "can read it, and the container waits until you answer.",
                        f"guarded:{hit}"))
    return out


# Projects and their private homes

def project_id(folder: str | None) -> str:
    """The id of the project a session keys. For a shared current folder it
    is the folder's name, cut to :data:`PROJECT_NAME_MAX` characters of
    ``[A-Za-z0-9._-]``, and the first 16 hex digits of the SHA-256 of its
    real path. A home that earlier versions made with 8 hex digits keeps
    its id while its project.json names this folder. A session that shares
    no current folder keys :data:`PROJECT_DEFAULT`.

    Two folders can still get one id, so a home whose project.json names
    another folder raises :class:`SettingsError`."""
    import hashlib

    if folder is None:
        return PROJECT_DEFAULT
    name = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(folder.rstrip("/")))
    digest = hashlib.sha256(os.fsencode(folder)).hexdigest()
    name = name[:PROJECT_NAME_MAX] or "folder"
    project, earlier = f"{name}-{digest[:16]}", f"{name}-{digest[:8]}"
    homes = _project_folders(project)
    for client, other in homes:
        if other is not None and not _same(other, folder):
            raise SettingsError(
                f"will not use the private home "
                f"{_tilde(str(private_home_path(client, project)))}, because it belongs to "
                f"{_tilde(other)}, and {_tilde(folder)} has the same project id. Rename or "
                "move one of the two folders, then launch again.")
    if not homes and any(other is not None and _same(other, folder)
                         for _, other in _project_folders(earlier)):
        return earlier
    return project


def _project_folders(project: str) -> list[tuple[str, str | None]]:
    """``(client, folder)`` for each client with a project folder of this
    id, with the folder its project.json names, or None when the record is
    missing or names none."""
    from gmlx.config import LAUNCH_CLIENTS

    out = []
    for client in LAUNCH_CLIENTS:
        if project_dir_path(client, project).is_dir():
            folder = read_project_record(client, project).get("folder")
            out.append((client, folder if isinstance(folder, str) else None))
    return out


def project_dir_path(client: str, project: str) -> Path:
    """The folder of one client's project: its private home and the
    records beside it. Nothing is created."""
    return data_path() / client / "projects" / project


def project_dir(client: str, project: str) -> Path:
    """:func:`project_dir_path`, created on first use."""
    d = data_dir() / client / "projects" / project
    d.mkdir(parents=True, exist_ok=True)
    return d


def private_home_path(client: str, project: str = PROJECT_DEFAULT) -> Path:
    """Where the private home of a client's project lives, without creating
    it."""
    return project_dir_path(client, project) / "home"


def private_home(client: str, project: str = PROJECT_DEFAULT) -> Path:
    """The persistent home of a client's project, shared at the same path in
    the guest, created on first use."""
    home = project_dir(client, project) / "home"
    home.mkdir(exist_ok=True)
    os.chmod(home, 0o700)
    return home


def shares_cwd(client: str, flag: bool | None, cfg: LaunchClientCfg) -> bool:
    """Whether a session of ``client`` shares the current folder, from the
    flag, the config and the client's own default."""
    if flag is not None:
        return flag
    if cfg.mount_cwd is not None:
        return cfg.mount_cwd
    return client not in NO_CWD_CLIENTS


def check_cwd_share(cwd_real: str, home: str | None = None) -> None:
    """Refuse to share the current folder when launch never shares it by
    default, such as your home folder."""
    home = home or _host_home()
    why = auto_share_refusal(cwd_real, home)
    if why is not None:
        raise SettingsError(f"will not share the current folder {_tilde(cwd_real, home)}, "
                            f"because it {why}. Launch from a project folder, or pass "
                            "--no-mount-cwd.")


def project_record_path(client: str, project: str) -> Path:
    """The record of a project beside its private home: the folder it keys
    and when it was last used."""
    return project_dir_path(client, project) / "project.json"


def read_project_record(client: str, project: str) -> dict:
    """The project record, or an empty one when it is missing or damaged."""
    import json

    try:
        fd = os.open(project_record_path(client, project),
                     os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return {}
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > CONFIG_READ_MAX:
            return {}
        doc = json.loads(os.read(fd, CONFIG_READ_MAX).decode())
    except (OSError, ValueError, RecursionError):
        return {}
    finally:
        os.close(fd)
    return doc if isinstance(doc, dict) else {}


def write_project_record(client: str, project: str, folder: str | None) -> None:
    """Record the project's folder and the time of this launch."""
    import json
    import time

    doc = {"folder": folder, "used": int(time.time())}
    write_record(project_dir(client, project) / "project.json", json.dumps(doc).encode())


def new_home_line(client: str, project: str) -> str:
    """The line for the first launch of a new private home."""
    scope = " for this project" if project != PROJECT_DEFAULT else ""
    return (f"[launch] {client} keeps its own history{scope} in the container, starting "
            "empty. Its history on the Mac stays on the Mac.")


def project_volume_name(name: str, project: str) -> str:
    """The name of a client's volume in one project: the configured name
    and 8 hex digits of the project id, within Apple's 255-character limit."""
    import hashlib

    digest = hashlib.sha256(project.encode()).hexdigest()[:8]
    return f"{name[:VOLUME_NAME_MAX - 9]}-{digest}"


@dataclass
class PrivateHome:
    """One private home on disk, for ``gmlx doctor`` and removal."""
    client: str
    project: str
    path: Path
    folder: str | None                # the project folder it keys, or None
    used: int | None                  # the last launch, in seconds since the epoch


def private_homes() -> list[PrivateHome]:
    """Every private home under the launch data folder, newest use first."""
    from gmlx.config import LAUNCH_CLIENTS

    out = []
    for client in LAUNCH_CLIENTS:
        root = data_path() / client / "projects"
        try:
            projects = sorted(os.listdir(root))
        except OSError:
            continue
        for project in projects:
            home = root / project / "home"
            if not home.is_dir() or home.is_symlink():
                continue
            doc = read_project_record(client, project)
            folder = doc.get("folder") if isinstance(doc.get("folder"), str) else None
            used = doc.get("used") if isinstance(doc.get("used"), int) else None
            out.append(PrivateHome(client, project, home, folder, used))
    return sorted(out, key=lambda h: -(h.used or 0))


def ready_home(client: str, home: Path) -> None:
    """Answer the first-run questions of a new private home that the
    client's config allows. For Claude Code, a home with no
    ``.claude.json`` gets one that marks the onboarding done, with the Mac's
    theme when the Mac's ``.claude.json`` sets one. Only that key is read,
    and the folder trust question is left to the client."""
    import json

    from . import confine

    if client != "claude-code":
        return
    target = home / ".claude.json"
    with confine.confined(home):
        try:
            if confine.exists(target):
                return
        except confine.ConfinedError:
            return                        # the guest put a link there, which it keeps
        doc: dict = {"hasCompletedOnboarding": True}
        theme = _mac_claude_theme()
        if theme is not None:
            doc["theme"] = theme
        confine.write_text(target, json.dumps(doc, indent=2) + "\n")


def _mac_claude_theme() -> str | None:
    import json

    path = os.path.join(_host_home(), ".claude.json")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > CLAUDE_JSON_READ_MAX:
            return None
        chunks = []
        while chunk := os.read(fd, 1 << 20):
            chunks.append(chunk)
        doc = json.loads(b"".join(chunks).decode())
    except (OSError, ValueError, RecursionError):
        return None
    finally:
        os.close(fd)
    theme = doc.get("theme") if isinstance(doc, dict) else None
    return theme if isinstance(theme, str) and theme.isprintable() and len(theme) <= 64 \
        else None


def agent_socket(value: bool | str | None, home: str,
                 shares: Sequence[str] = ()) -> str | None:
    """The real path of the SSH agent socket that a path in ``ssh_agent``
    names, or that SSH_AUTH_SOCK names for true. None for false, and for
    true without SSH_AUTH_SOCK. Only a named socket that this user owns is
    taken, since the container gets every key the agent holds.

    A client can put a link to another agent in a folder that it writes, so
    the path is refused when it, or any path that resolving it passes
    through, lies in ``shares``, in a folder an earlier session shared
    read-write, or in the folder with the private homes. Apple's relay
    opens the forwarded path again for each connection, so launch forwards
    the real path."""
    if value is True:
        env = os.environ.get("SSH_AUTH_SOCK")
        if not env:
            return None
        path = os.path.abspath(env)
        real = _real(path)
        why = _agent_refusal(path, real, shares, home)
        if why:
            raise SettingsError(f"SSH_AUTH_SOCK names {_tilde(path, home)}, which {why}. A "
                                "client can leave a link to another agent there, so set "
                                "ssh_agent to the path of an agent socket outside the shared "
                                "folders and the private homes.")
        return real
    if not isinstance(value, str):
        return None
    if "\0" in value:
        raise SettingsError("ssh_agent holds a NUL character, which no path can hold. Name "
                            "the socket of an SSH agent.")
    path = os.path.expanduser(value)
    # A ~user that does not exist stays as written.
    shown = _tilde(os.path.abspath(path), home) if os.path.isabs(path) else value
    try:
        st = os.stat(path) if os.path.isabs(path) else None
    except OSError:
        st = None
    if st is None:
        raise SettingsError(f"ssh_agent names {shown}, which does not exist. Start that agent, "
                            "or set ssh_agent to true to use the agent in SSH_AUTH_SOCK.")
    if not stat.S_ISSOCK(st.st_mode):
        raise SettingsError(f"ssh_agent names {shown}, which is not a socket. Name the socket "
                            "of an SSH agent.")
    if st.st_uid != os.getuid():
        raise SettingsError(f"ssh_agent names {shown}, which another user owns. Name the "
                            "socket of your own SSH agent.")
    real = _real(path)
    why = _agent_refusal(path, real, shares, home)
    if why:
        raise SettingsError(f"ssh_agent names {shown}, which {why}. A client can leave a link "
                            "to another agent there, so name a socket outside the shared "
                            "folders and the private homes.")
    return real


def _agent_refusal(path: str, real: str, shares: Sequence[str], home: str) -> str | None:
    """How ``path``, such as the agent socket, meets a folder that a client
    can write, as a phrase that follows the path, or None."""
    folders = [(_real(data_path()), "where launch keeps the private homes of the clients"),
               *((f, "a folder this launch shares") for f in shares),
               *((f, "a folder an earlier session shared read-write") for f in shared_history())]
    # The path as written, and with its folder resolved.
    given = [path, os.path.join(_real(os.path.dirname(path)), os.path.basename(path))]
    for p in [*given, *_resolution_paths(path), real]:
        for folder, what in folders:
            if _inside(p, folder):
                verb = "lies in" if p in given else "leads through"
                return f"{verb} {_tilde(folder, home)}, {what}"
    return None


def _resolution_paths(path: str) -> list[str]:
    """Each path that resolving the absolute ``path`` visits, every link
    among them, in the form macOS gives it. A link that a client can change
    anywhere on the way changes where the path leads."""
    visited: list[str] = []
    todo = [c for c in path.split("/") if c]
    done, links = "/", 0
    while todo:
        name = todo.pop(0)
        if name == ".":
            continue
        if name == "..":
            done = os.path.dirname(done)
            continue
        here = os.path.join(done, name)
        visited.append(here)
        try:
            target = os.readlink(here)
        except OSError:
            done = here
            continue
        links += 1
        if links > 32:              # the limit macOS sets, so the stat failed
            break
        if target.startswith("/"):
            done = "/"
        todo[:0] = [c for c in target.split("/") if c]
    return [os.path.join(_real(os.path.dirname(p)), os.path.basename(p)) for p in visited]


def check_program(path: str | None, shares: Sequence[str] = ()) -> None:
    """Refuse the ``container`` program at ``path`` when a client could
    replace it: it lies in a read-write share in ``shares``, in a folder an
    earlier session shared read-write or in the private homes, or a link
    on the way to it does. Launch runs that program on the Mac, also after
    the client exits."""
    if path is None:
        return
    home = _host_home()
    why = _agent_refusal(path, _real(path), shares, home)
    if why:
        raise SettingsError(f"launch found the container program at {_tilde(path, home)}, "
                            f"which {why}. A client could replace it, and launch runs it on "
                            "the Mac. Remove that folder from PATH, and launch again.")


AGENT_CHECK_TIMEOUT = 3.0


def _ssh_add_list(sock: str) -> int | None:
    """The exit code of ``ssh-add -l`` against ``sock``: 0 when the agent
    holds a key, 1 when it holds none, 2 when no agent answers. None when
    ssh-add cannot run or takes too long."""
    try:
        return subprocess.run(["ssh-add", "-l"], env=_system_env(SSH_AUTH_SOCK=sock),
                              stdin=subprocess.DEVNULL, capture_output=True,
                              timeout=AGENT_CHECK_TIMEOUT).returncode
    except (OSError, subprocess.TimeoutExpired):
        return None


def forwarded_agent(plan: ContainerPlan) -> str | None:
    """The agent socket that ``container run --ssh`` forwards: the real path
    of the socket in ``ssh_agent``, or of SSH_AUTH_SOCK when ``ssh_agent``
    is true. None when ``ssh_agent`` is off or names no agent, and launch
    then passes no ``--ssh``, with which the guest would get an
    SSH_AUTH_SOCK that leads nowhere."""
    if not plan.ssh_agent:
        return None
    return plan.ssh_socket or None


def agent_key_line(plan: ContainerPlan) -> str | None:
    """A line when ``ssh_agent`` is on but the agent it forwards holds no
    keys or does not answer. None when the agent holds a key, and when the
    check cannot tell."""
    if not plan.ssh_agent:
        return None
    sock = forwarded_agent(plan)
    if not sock:
        return ("[launch] ssh_agent is on, but SSH_AUTH_SOCK is not set, so the container "
                "gets no SSH agent.")
    code = _ssh_add_list(sock)
    if code == 1:
        env = os.environ.get("SSH_AUTH_SOCK")
        # ssh-add loads a key into the agent of SSH_AUTH_SOCK.
        if not env or not _same(_real(os.path.abspath(env)), sock):
            return (f"[launch] ssh_agent is on, but the SSH agent at {_tilde(sock)} holds no "
                    "keys, so ssh in the container cannot sign. Load a key into that agent.")
        return ("[launch] ssh_agent is on, but the SSH agent holds no keys, so ssh in the "
                "container cannot sign. Load one with ssh-add --apple-use-keychain "
                "~/.ssh/id_ed25519.")
    if code == 2:
        return (f"[launch] ssh_agent is on, but no SSH agent answers at {_tilde(sock)}, so ssh "
                "in the container cannot sign.")
    return None


def memory_warning(memory: str) -> str | None:
    """A note when the memory of the container's virtual machine is over a
    quarter of the Mac's."""
    size = parse_size_bytes(memory)
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        return None
    if size is None or size + VM_MEMORY_OVERHEAD <= total * MEMORY_WARN_FRACTION:
        return None
    return Once(f"[launch] the container gets {memory} of the Mac's "
                f"{total / (1 << 30):.0f} GB, and its virtual machine takes "
                f"{VM_MEMORY_OVERHEAD >> 20} MB more. The model server cannot use this memory "
                "while the container runs.", f"memory:{size}")


def resolve_plan(client: str, cfg: LaunchClientCfg, *, cwd: str,
                 mount_cwd: bool | None = None, cli_mounts: list[str] = (),
                 network: str | None = None, api_port: int | None = None,
                 web_port: int | None = None,
                 build_folders: dict[str, str] | None = None,
                 project: str = PROJECT_DEFAULT,
                 project_volumes: Sequence[str] = ()) -> ContainerPlan:
    """The mounts, volumes and ports of one session, from the effective
    client config and the flags. ``build_folders`` maps each client to its
    configured ``build:`` path, and no read-write share may overlap one.
    The private home is the one of ``project``, which the plan names
    without creating it, and each volume entry in ``project_volumes`` gets
    that project's name."""
    home = _host_home()
    warns: list[str] = []
    notes: list[str] = []
    mounts: list[Mount] = []
    cwd_real = _real(cwd)
    share_cwd = shares_cwd(client, mount_cwd, cfg)
    if share_cwd:
        check_cwd_share(cwd_real, home)
        mounts.append(Mount(cwd_real, cwd_real, note="working folder"))
    for spec in [*cfg.mounts, *cli_mounts]:
        mount = _explicit_mount(spec, warns, home)
        if share_cwd and (mount.source, normal_guest_target(mount.target)) == (
                cwd_real, cwd_real):
            # A mount of the current folder at its own path sets how it is
            # shared, such as read-only, in place of the default share.
            mounts = [m for m in mounts if m.note != "working folder"]
        mounts.append(mount)
    git_mount, git_notes = (git_extra_mount(cwd_real, list(mounts), home)
                            if share_cwd else (None, []))
    if git_mount is not None:
        mounts.append(git_mount)
    notes.extend(git_notes)
    # Before the private home is made, so a refused socket leaves none.
    ssh_socket = agent_socket(cfg.ssh_agent, home,
                              [m.source for m in mounts if m.kind in ("share", "git")])
    new_home = not private_home_path(client, project).is_dir()
    # By its real path, since the check just before the run takes a share
    # only by its real path.
    guest_home = Path(_real(private_home_path(client, project)))
    mounts.append(Mount(str(guest_home), str(guest_home), kind="home"))
    mounts.extend(_volume_mount(v, project if v in project_volumes else None)
                  for v in cfg.volumes)
    mounts = normalize_mounts(mounts)
    _refuse_build_folder_shares(mounts, build_folders or {}, home)
    _refuse_python_shares(mounts, home)
    warns.extend(_package_warnings(mounts, home))
    warns.extend(_path_warnings(mounts, home))
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
        seed=list(cfg.seed), warnings=warns, notes=notes, project=project,
        new_home=new_home, ssh_socket=ssh_socket)


def build_folder(build: str) -> str | None:
    """The build context a ``build:`` path names: the folder itself, or the
    folder that holds a named Containerfile. None for a relative path,
    which the image step refuses."""
    path = os.path.expanduser(build)
    if not os.path.isabs(path):
        return None
    # A named Containerfile builds from the folder that holds it. A path
    # that does not exist is taken as the folder, and the image step
    # reports it.
    if os.path.lexists(path) and not os.path.isdir(path):
        path = os.path.dirname(path)
    return _real(path)


def _refuse_build_folder_shares(mounts: list[Mount], build_folders: dict[str, str],
                                home: str) -> None:
    """A read-write share that holds or lies in a client's build folder
    lets this client change what that image runs at its next build, which
    has network access."""
    for client, build in sorted(build_folders.items()):
        folder = build_folder(build) if build else None
        if folder is None:
            continue
        for m in mounts:
            if m.readonly or m.kind not in ("share", "git"):
                continue
            if _inside(m.source, folder) or _inside(folder, m.source):
                raise SettingsError(
                    f"will not share {_tilde(m.source, home)} read-write, because the client "
                    f"could change the {client} build: folder {_tilde(folder, home)}.\n"
                    f"  Share it read-only with --mount {_tilde(m.source, home)}:ro, or move "
                    "the build folder.")


def _python_folders() -> list[tuple[str, str]]:
    """The folders of the Python that gmlx runs from, each with what it is.
    The Mac runs the code in them at the next gmlx command, and in a server
    that launch or launchd starts."""
    import site
    import sys

    from gmlx.serve.procname import stable_executable

    env = "the Python environment that gmlx runs from"
    out = [(sys.prefix, env), (sys.exec_prefix, env),
           (os.path.dirname(stable_executable()), "the folder of the Python that gmlx runs")]
    if site.ENABLE_USER_SITE:
        out.append((site.getusersitepackages(), "your user site-packages folder, which "
                                                "gmlx imports"))
    return [(_real(folder), what) for folder, what in out]


def _refuse_python_shares(mounts: list[Mount], home: str) -> None:
    """A read-write share that holds or lies in gmlx's Python environment
    lets the client change code that the Mac runs, such as a ``.pth`` file
    in site-packages."""
    folders = _python_folders()
    for m in mounts:
        if m.readonly or m.kind not in ("share", "git"):
            continue
        for folder, what in folders:
            if _inside(m.source, folder) or _inside(folder, m.source):
                raise SettingsError(
                    f"will not share {_tilde(m.source, home)} read-write, because it "
                    f"{_relation(m.source, folder, home, what)}. The client could change "
                    "code that the Mac runs.\n"
                    f"  Share it read-only with --mount {_tilde(m.source, home)}:ro, or move "
                    "the Python environment out of the folder.")


def _path_warnings(mounts: list[Mount], home: str) -> list[str]:
    """Warnings for the PATH entries that lead into a read-write share. A
    program that the client puts there runs on the Mac in place of a
    command of that name. An empty or relative entry names the current
    folder."""
    rw = [m for m in mounts if m.kind in ("share", "git") and not m.readonly]
    if not rw:
        return []
    entries = os.environ.get("PATH", os.defpath).split(os.pathsep)
    out = []
    if any(not os.path.isabs(e) for e in entries):
        out.append("[launch] warning: PATH has an empty or relative entry, which names the "
                   "current folder. A program the client writes in a read-write share would "
                   "run on the Mac when you run a command of that name from that folder. "
                   "Remove the entry from PATH.")
    for entry in dict.fromkeys(e for e in entries if os.path.isabs(e)):
        m = next((m for m in rw if _inside(entry, m.source) or _inside(_real(entry), m.source)),
                 None)
        if m is not None:
            out.append(f"[launch] warning: PATH holds {_tilde(entry, home)}, which lies in the "
                       f"read-write share {_tilde(m.source, home)}. A program the client puts "
                       "there runs on the Mac in place of a command of that name. Remove the "
                       "folder from PATH, or share the folder read-only.")
    return out


def _package_warnings(mounts: list[Mount], home: str) -> list[str]:
    """A warning when a read-write share holds or lies in the gmlx package
    folder, as an editable checkout puts it."""
    import gmlx

    package = _real(os.path.dirname(gmlx.__file__))
    for m in mounts:
        if m.readonly or m.kind not in ("share", "git"):
            continue
        if _inside(m.source, package) or _inside(package, m.source):
            return [f"[launch] warning: the share {_tilde(m.source, home)} "
                    f"{_relation(m.source, package, home, 'the gmlx package that the Mac runs')}"
                    ". The client can change gmlx's code, which the next gmlx command runs, "
                    "and the guest entry and Containerfile that later sessions and builds use."]
    return []


def recheck_sources(plan: ContainerPlan) -> None:
    """Check each folder the plan mounts again, just before the run. A
    client of another session can replace a shared folder with a link after
    the plan was made, and ``container run`` would follow it. Raises
    :class:`SettingsError` when a source is no longer the folder the plan
    checked. A swap after this check still reaches the run, so this narrows
    the window and cannot close it."""
    for m in plan.mounts:
        if m.kind == "volume":
            continue
        try:
            st = os.lstat(m.source)
        except OSError:
            st = None
        if st is None or not stat.S_ISDIR(st.st_mode) or not _same(_real(m.source), m.source):
            raise SettingsError(f"{_label(m)} changed after launch checked it, so it may be "
                                "a symbolic link now. Launch again to check it.")


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


def seed_record_path(home: Path) -> Path:
    """Where launch records the seeds it copied into ``home``. It lies
    beside the private home, never in it, so the client cannot change it."""
    return Path(home).parent / "seeded.json"


def _read_json_record(path: Path) -> dict:
    """The JSON object in a record file beside the private home, or an empty
    one when the file is missing, is not a regular file or does not parse."""
    import json

    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return {}
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > CONFIG_READ_MAX:
            return {}
        doc = json.loads(os.read(fd, CONFIG_READ_MAX).decode())
    except (OSError, ValueError, RecursionError):
        return {}
    finally:
        os.close(fd)
    return doc if isinstance(doc, dict) else {}


def _read_seed_record(path: Path) -> tuple[set[str], dict[str, dict]]:
    """The seeds launch copied, and for each the stamps of the Mac source
    and of the copy at the time of the copy. A record from before the
    stamps has none."""
    doc = _read_json_record(path)
    seeded = doc.get("seeded")
    done = {x for x in seeded if isinstance(x, str)} if isinstance(seeded, list) else set()
    raw = doc.get("stamps")
    stamps = {src: entry for src, entry in raw.items()
              if src in done and isinstance(entry, dict)
              and all(_stamp_ok(entry.get(k)) for k in ("source", "copy"))
              } if isinstance(raw, dict) else {}
    return done, stamps


def _stamp_ok(value) -> bool:
    return value is None or (isinstance(value, list) and len(value) == 3
                             and all(isinstance(n, int) for n in value))


def _host_stamp(real: str) -> list[int] | None:
    """The stamp of the seed source at ``real`` on the Mac, as
    :func:`confine.stamp_in` gives it."""
    from . import confine

    try:
        dir_fd = os.open(os.path.dirname(real), os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return None
    try:
        return confine.stamp_in(dir_fd, os.path.basename(real), SEED_MAX_FILES)
    finally:
        os.close(dir_fd)


def _seed_source_refusal(real: str, host_home: str) -> str | None:
    """Why launch will not copy the file or folder at ``real``, as a phrase
    that follows the path, or None."""
    if _same(real, host_home):
        return "is your home folder"
    if not _inside(real, host_home):
        return "lies outside your home folder"
    return _data_refusal(real, host_home) or _sensitive_refusal(real, host_home)


def _seed_refusal(shown: str, src: str, real: str, host_home: str) -> str:
    """The refusal of the seed ``shown`` at ``src``, whose real path is
    ``real``."""
    why = _seed_source_refusal(real, host_home)
    subject = "it" if _same(real, src) else f"it leads to {_tilde(real, host_home)}, which"
    return f"seed: will not copy {shown}, because {subject} {why}."


def shared_history_path() -> Path:
    """Where launch records the folders it shared read-write. A client can
    leave links in such a folder that outlive the session."""
    return data_path() / "shared.json"


def _read_history() -> dict:
    import json

    try:
        fd = os.open(shared_history_path(), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return {}
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > CONFIG_READ_MAX:
            return {}
        doc = json.loads(os.read(fd, CONFIG_READ_MAX).decode())
    except (OSError, ValueError, RecursionError):
        return {}
    finally:
        os.close(fd)
    return doc if isinstance(doc, dict) else {}


def shared_history() -> list[str]:
    """The folders launch shared read-write, newest first. A client can
    have left links or changed files there that outlive its session."""
    shared = _read_history().get("shared")
    return [x for x in shared if isinstance(x, str)] if isinstance(shared, list) else []


def worktree_history() -> list[tuple[str, str]]:
    """``(project, git folder)`` for each worktree or submodule whose git
    folder launch shared, as the project's ``.git`` file named it then."""
    pairs = _read_history().get("worktrees")
    if not isinstance(pairs, list):
        return []
    return [(p[0], p[1]) for p in pairs if isinstance(p, list) and len(p) == 2
            and all(isinstance(x, str) for x in p)]


def record_shares(plan: ContainerPlan) -> None:
    """Add this session's read-write shares, and the project each shared
    git folder serves, to the history, keeping the newest
    :data:`SHARED_HISTORY_MAX` of each. A lock keeps two launches that
    start together from dropping each other's shares."""
    import json

    from .state import FileLock

    now = [m.source for m in plan.mounts if not m.readonly and m.kind in ("share", "git")]
    bound = [(m.worktree, m.source) for m in plan.mounts if m.kind == "git" and m.worktree]
    with FileLock(data_dir() / "shared.lock"):
        old, old_pairs = shared_history(), worktree_history()
        merged = list(dict.fromkeys([*now, *old]))[:SHARED_HISTORY_MAX]
        pairs = list(dict.fromkeys([*bound, *old_pairs]))[:SHARED_HISTORY_MAX]
        if merged != old or pairs != old_pairs:
            write_record(shared_history_path(), json.dumps(
                {"shared": merged, "worktrees": [list(p) for p in pairs]}).encode())


def seed_writable(plan: ContainerPlan, cwd: str) -> list[str]:
    """The folders a client of this session can write, for the seed check:
    the read-write shares, and the current folder when a session could
    share it."""
    out = [m.source for m in plan.mounts if not m.readonly and m.kind in ("share", "git")]
    cwd_real = _real(cwd)
    if auto_share_refusal(cwd_real) is None:
        out.append(cwd_real)
    return list(dict.fromkeys(out))


def _seed_link_refusal(src: str, real: str, writable: list[str], home: str) -> str | None:
    """Why a seed whose path lies in a folder a client could write may not
    be copied. A client can replace the seed there with a link to any file
    of yours, so the real path must stay in that folder."""
    parent_real = os.path.join(_real(os.path.dirname(src)), os.path.basename(src))
    for folder in writable:
        if (_inside(src, folder) or _inside(parent_real, folder)) and not _inside(real, folder):
            return (f"it lies in {_tilde(folder, home)}, which a session shared "
                    f"read-write, and it leads to {_tilde(real, home)} outside that folder, "
                    "so a client may have replaced it with a symbolic link")
    return None


def seed_home(home: Path, seeds: list[str], *, reseed: bool = False,
              writable: Sequence[str] = ()) -> list[str]:
    """Copy each seed into the private home, at the same path relative to
    ``$HOME``, then add the host git identity where it is missing. Returns
    the lines to print, one for each seed copied. Every write is confined
    to the private home, since the guest can plant links there.

    Launch records each seed it copied beside the private home, with stamps
    of the Mac source and of the copy, so a copy the client deletes is not
    made again. A source that changed on the Mac is copied again when the
    copy is unchanged. When both changed, the copy stays and one line names
    ``--reseed``, which copies every seed again, replacing the copy in the
    private home. ``writable`` holds the folders this session shares
    read-write. With the folders earlier sessions shared, a seed in one of
    them must not lead out of it."""
    import json

    from . import confine

    # Relative to $HOME as written, so a seed that is a link into a dotfiles
    # repository still lands at its own path.
    host_home = os.path.abspath(os.path.expanduser("~"))
    host_real = _real(host_home)
    record = seed_record_path(home)
    done, stamps = _read_seed_record(record)
    copied, new_stamps = set(done), dict(stamps)
    guest_written = list(dict.fromkeys([*writable, *shared_history()]))
    out: list[str] = []
    with confine.confined(home):
        for seed in seeds:
            expanded = os.path.expanduser(seed)
            src = os.path.abspath(os.path.join(host_home, expanded))
            if not _inside(src, host_home) or src == host_home:
                raise SettingsError(f"seed: {seed} is not inside your home folder.")
            shown = _tilde(src, host_home)
            if not os.path.lexists(src):
                out.append(f"[launch] seed: {seed} does not exist, so nothing was copied.")
                continue
            dst = home / os.path.relpath(src, host_home)
            try:
                if reseed:
                    reason = " again for --reseed"
                elif not confine.exists(dst):
                    if src in done:
                        continue              # the client deleted the copy
                    reason = " into the private home"
                elif src not in done or src not in stamps:
                    # A copy made before the record, or before it kept
                    # stamps, counts from now.
                    copied.add(src)
                    new_stamps[src] = {"source": _host_stamp(_real(src)),
                                       "copy": confine.tree_stamp(dst, SEED_MAX_FILES)}
                    continue
                else:
                    source = _host_stamp(_real(src))
                    if source == stamps[src]["source"]:
                        continue
                    if confine.tree_stamp(dst, SEED_MAX_FILES) != stamps[src]["copy"]:
                        out += notices.due([Once(
                            f"[launch] seed: {shown} changed on the Mac and in the private "
                            "home, so launch kept the copy in the private home. --reseed "
                            "replaces it with the Mac file.",
                            f"seed:{record}:{src}:{source}")])
                        continue
                    reason = " again, because it changed on the Mac"
            except confine.ConfinedError as e:
                raise SettingsError(f"seed: {e}") from None
            try:
                new_stamps[src] = _copy_seed(src, dst, shown, out, host_real=host_real,
                                             guest_written=guest_written)
            except SettingsError as e:
                if reason.startswith(" again, because"):
                    # The earlier copy is still whole, so a source that
                    # cannot be copied now, such as one a client replaced
                    # with a link, never stops the launch.
                    out += notices.due([Once(f"[launch] {e} Launch kept the earlier copy.",
                                             f"seed:{record}:{src}:{_host_stamp(_real(src))}")])
                    continue
                raise
            copied.add(src)
            out.append(f"[launch] seed: copied {shown}{reason}")
        if copied != done or new_stamps != stamps:
            write_record(record, json.dumps({"seeded": sorted(copied),
                                             "stamps": new_stamps}).encode())
        try:
            out += _seed_git_identity(home)
        except confine.ConfinedError as e:
            # The guest owns the file, so a file launch cannot read or
            # replace costs only the identity, never the launch.
            out.append(f"[launch] warning: {e} Launch did not add your git identity to it.")
    return out


def _copy_seed(src: str, dst: Path, shown: str, out: list[str], *, host_real: str,
               guest_written: list[str]) -> dict:
    """Copy the seed at ``src`` to ``dst`` in the private home, adding its
    notes to ``out``, and return the stamps of the source and the copy."""
    from . import confine

    # The copy goes to a new name first and is renamed into place only when
    # it is whole, so a failed copy is never taken for a finished one at
    # the next launch.
    prefix = f".{dst.name}.gmlx-seed-"
    tmp = dst.with_name(prefix + secrets.token_hex(4))
    try:
        # Only a copy is checked. A link in the way may have been left by
        # a client in a folder an earlier launch shared, so the real path
        # decides.
        real = _real(src)
        if _seed_source_refusal(real, host_real) is not None:
            raise SettingsError(_seed_refusal(shown, src, real, host_real))
        why = _seed_link_refusal(src, real, guest_written, host_real)
        if why is not None:
            raise SettingsError(f"seed: will not copy {shown}, because {why}.")
        if not _same(real, src):
            out.append(f"[launch] seed: copying {shown} from "
                       f"{_tilde(real, host_real)}, where its symbolic link leads.")
        out.extend(_seed_token_warnings(shown, real, host_real))
        # The stamp comes first, so a change during the copy is copied at
        # the next launch.
        source = _host_stamp(real)
        # A copy that a killed launch left half done is removed first.
        for name in confine.listdir(dst.parent):
            if name.startswith(prefix):
                confine.remove_tree(dst.parent / name)
        _copy_confined(real, tmp)
        confine.remove_tree(dst)
        confine.rename(tmp, dst.name)
        return {"source": source, "copy": confine.tree_stamp(dst, SEED_MAX_FILES)}
    except BaseException as e:
        with contextlib.suppress(confine.ConfinedError, OSError):
            confine.remove_tree(tmp)
        if isinstance(e, confine.ConfinedError):
            raise SettingsError(f"seed: {e}") from None
        if isinstance(e, OSError):
            raise SettingsError(f"seed: cannot copy {shown} ({e}).") from None
        raise


def _seed_token_warnings(shown: str, real: str, host_real: str) -> list[str]:
    out = []
    for files, one, many, what in (
            (TOKEN_FILES, "holds", "hold", "a sign-in token"),
            (TOKEN_MAYBE_FILES, "can hold", "can hold",
             "a token, such as one in a url.<base>.insteadOf address")):
        hits = [t for t in files if _inside(os.path.join(host_real, t), real)]
        if not hits:
            continue
        if any(_same(os.path.join(host_real, t), real) for t in hits):
            subject, verb = f"seed {shown}", one
        else:
            subject = f"seed {shown} copies {', '.join('~/' + t for t in hits)}, which"
            verb = one if len(hits) == 1 else many
        out.append(f"[launch] warning: {subject} {verb} {what}. The client can read it.")
    return out


class _SeedBudget:
    """The bytes and entries one seed may still copy."""

    def __init__(self, src: str):
        self.src = src
        self.bytes = SEED_MAX_BYTES
        self.files = SEED_MAX_FILES

    def entry(self) -> None:
        self.files -= 1
        if self.files < 0:
            raise SettingsError(f"seed: {_tilde(self.src)} holds more than "
                                f"{SEED_MAX_FILES} files and folders. Seed a smaller "
                                "folder.")

    def take(self, n: int) -> None:
        self.bytes -= n
        if self.bytes < 0:
            raise SettingsError(f"seed: {_tilde(self.src)} is larger than "
                                f"{SEED_MAX_BYTES >> 20} MiB. Seed a smaller file or "
                                "folder.")


def _copy_confined(src: str, dst: Path) -> None:
    """Copy a file, a folder or a link into the private home with its mode.

    ``src`` is opened without following a link, and the path macOS gives
    the open file is checked again, so a link swapped in after the check
    cannot redirect the copy. Inside a folder, links are copied as links,
    named pipes and devices are skipped, and every file is opened the same
    way. The copy streams, and it stops at :data:`SEED_MAX_BYTES` and
    :data:`SEED_MAX_FILES`, counting a sparse file at its full size."""
    from . import confine

    budget = _SeedBudget(src)
    fd = os.open(src, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        opened = fd_path(fd)
        host_home = _real(os.path.expanduser("~"))
        if _seed_source_refusal(opened, host_home) is not None:
            raise SettingsError(_seed_refusal(_tilde(src), src, opened, host_home))
        if not _same(opened, src):
            raise SettingsError(f"seed: will not copy {_tilde(src)}, because it changed to "
                                f"{_tilde(opened)} while launch copied it.")
        st = os.fstat(fd)
        if stat.S_ISREG(st.st_mode):
            budget.entry()
            _copy_file(fd, st, dst, budget)
        elif stat.S_ISDIR(st.st_mode):
            budget.entry()
            confine.mkdirs(dst)
            _copy_folder(fd, dst, budget, depth=0)
        else:
            raise SettingsError(f"seed: {_tilde(src)} is not a file or a folder.")
    finally:
        os.close(fd)


def _copy_file(fd: int, st: os.stat_result, dst: Path, budget: _SeedBudget) -> None:
    from . import confine

    budget.take(st.st_size)             # before reading, so a sparse file counts whole

    def fill(out_fd: int) -> None:
        left = st.st_size
        while True:
            chunk = os.read(fd, min(1 << 20, max(left, 0) + 1))
            if not chunk:
                return
            left -= len(chunk)
            if left < 0:                # the file grew after the check
                budget.take(-left)
                left = 0
            confine.write_all(out_fd, chunk)
    confine.write_stream(dst, fill, stat.S_IMODE(st.st_mode))


def _copy_folder(dir_fd: int, dst: Path, budget: _SeedBudget, *, depth: int) -> None:
    from . import confine

    if depth >= SEED_MAX_DEPTH:
        raise SettingsError(f"seed: {_tilde(budget.src)} is nested more than "
                            f"{SEED_MAX_DEPTH} folders deep.")
    for name in sorted(os.listdir(dir_fd)):
        budget.entry()
        st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        if stat.S_ISLNK(st.st_mode):
            confine.symlink(os.readlink(name, dir_fd=dir_fd), dst / name)
            continue
        if not (stat.S_ISDIR(st.st_mode) or stat.S_ISREG(st.st_mode)):
            continue                    # named pipes, sockets and devices
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
        if stat.S_ISDIR(st.st_mode):
            flags |= os.O_DIRECTORY
        child = os.open(name, flags, dir_fd=dir_fd)
        try:
            cst = os.fstat(child)
            if stat.S_ISDIR(cst.st_mode):
                confine.mkdirs(dst / name)
                _copy_folder(child, dst / name, budget, depth=depth + 1)
            elif stat.S_ISREG(cst.st_mode):
                _copy_file(child, cst, dst / name, budget)
        finally:
            os.close(child)


def identity_record_path(home: Path) -> Path:
    """The record of the git identity launch wrote into ``home``, kept
    beside it where the client cannot change it."""
    return Path(home).parent / "git-identity.json"


def _git_get(where: list[str], key: str) -> str | None:
    value = subprocess.run(["git", "config", *where, "--get", key],
                           capture_output=True, text=True, timeout=5, env=_system_env())
    return (value.stdout.strip() or None) if value.returncode == 0 else None


def _seed_git_identity(home: Path) -> list[str]:
    """Add the host ``user.name`` and ``user.email`` to the private home's
    ``.gitconfig`` where they are missing, and follow a change on the Mac
    for a value launch wrote there. A value set in the container stays.
    Returns the line to print for a value that followed the Mac. git edits a
    copy outside the private home, since git follows a link at the file it
    writes, and the result goes back through the confined write."""
    import json
    import tempfile

    from . import confine

    gitconfig = home / ".gitconfig"
    record = identity_record_path(home)
    wrote = {k: v for k, v in _read_json_record(record).items() if isinstance(v, str)}
    known, updated = dict(wrote), []
    before = confine.read_text(gitconfig) or ""
    with tempfile.TemporaryDirectory() as tmp:
        work = os.path.join(tmp, "gitconfig")
        Path(work).write_text(before)
        for key in ("user.name", "user.email"):
            try:
                have = _git_get(["--file", work], key)
                mac = _git_get(["--global"], key)
                if mac is None:
                    continue
                if have == mac:
                    # The same value as the Mac's, so it follows the Mac
                    # from now on, also in a home from before the record.
                    known[key] = mac
                    continue
                if have is not None and wrote.get(key) != have:
                    continue                  # set in the container
                subprocess.run(["git", "config", "--file", work, key, mac],
                               capture_output=True, timeout=5, env=_system_env())
            except (OSError, subprocess.TimeoutExpired):
                return []
            known[key] = mac
            if have is not None:
                updated.append(key)
        after = Path(work).read_text()
    if after != before:
        confine.write_text(gitconfig, after)
    if known != wrote:
        write_record(record, json.dumps(known).encode())
    if not updated:
        return []
    return [f"[launch] updated the git {' and '.join(updated)} in the private home to "
            "match the Mac."]


# The server config the guest could change

def server_config_path(host: str, port: int, *, autostart: bool = True,
                       notes: list[str] | None = None) -> str | None:
    """The config file the target server runs with: the one in its runfile
    while it runs, else the one autostart would use. With ``autostart``
    False, as for ``--base-url``, only a runfile counts. A runfile from an
    older gmlx can hold a path relative to a folder launch cannot know, so
    that path counts as unknown and ``notes`` gets a line about it."""
    from gmlx.config import default_config_paths
    from gmlx.serve import lifecycle

    run = lifecycle.read_run(host, port) or {}
    if run and not run.get("config_abspath") and lifecycle.pid_alive(run.get("pid")):
        # Such a server may scan --models-dir, which the runfile does not
        # record.
        if notes is not None:
            notes.append(f"[launch] the server on port {port} has no config file, so launch "
                         "cannot check whether it scans a shared folder for models, where a "
                         "client could add a model file. Start the server from a config file "
                         "to have it checked.")
        return None
    if run.get("config_abspath") and lifecycle.pid_alive(run.get("pid")):
        path = str(run["config_abspath"])
        if os.path.isabs(path):
            return path
        if notes is not None:
            notes.append(f"[launch] the server on port {port} records its config as {path}, "
                         "relative to the folder it started from, so launch cannot check "
                         "whether that config is in a share. Restart the server with "
                         "gmlx restart to record the full path.")
        return None
    if not autostart:
        return None
    for path in default_config_paths():
        if path.exists():
            return str(path)
    return None


def _read_small_file(path: str) -> bytes:
    """The file's bytes, refusing a link and anything but a regular file of
    at most 1 MiB without ever blocking on it."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
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


def _expand(path: str, cwd: str) -> str:
    """``path`` as the server reads it: variables and ``~`` expanded, and a
    relative path taken from ``cwd``, the folder the server runs in."""
    return _real(os.path.join(cwd, os.path.expanduser(os.path.expandvars(path))))


def _model_paths(cfg, cwd: str) -> list[tuple[str, str, str]]:
    """The files the server config names that gmlx reads or runs on the Mac:
    model files, chat template files, the local models of the embeddings,
    rerank, tts and stt services, and the programs and path arguments of
    the stdio tool servers. They are resolved the way the server resolves
    them from ``cwd``, and never read. Each comes with a phrase that names
    it at ``{path}``, and with when gmlx next uses it. As in
    ``resolve_path``, a relative model path that no model folder holds is
    taken from the working folder."""
    load = "before the server's next load"
    out: list[tuple[str, str, str]] = []
    roots = [_expand(d, cwd) for d in cfg.model_dirs]

    def model_file(p: str) -> str:
        p = os.path.expandvars(os.path.expanduser(p))
        if os.path.isabs(p):
            return _real(p)
        return _real(next((c for c in (os.path.join(r, p) for r in roots)
                           if os.path.exists(c)), os.path.join(cwd, p)))

    for model in cfg.models.values():
        for p in (model.path, model.mmproj, model.draft_gguf, model.adapter):
            if p and not str(p).startswith("hf:"):
                out.append((model_file(str(p)), "the model file {path} is", load))
    # A chat template that holds no Jinja names a file, which the loader
    # reads from the server's folder as written.
    templates = [prof.chat_template for prof in cfg.profiles.values()]
    for model in cfg.models.values():
        templates.append((model.overrides or {}).get("chat_template"))
        templates += [(t or {}).get("chat_template") for t in (model.profiles or {}).values()
                      if isinstance(t, dict)]
    for t in templates:
        if isinstance(t, str) and t.strip() and "{" not in t:
            out.append((_real(os.path.join(cwd, t)), "the chat template file {path} is",
                        load))
    for key in ("embeddings", "rerank", "tts", "stt"):
        value = getattr(cfg, key, None)
        if not isinstance(value, str) or not value.strip():
            continue
        v = value.strip()
        if v.lower().startswith("hf:"):
            continue
        if v.lower().endswith(".gguf"):
            out.append((model_file(v), f"the {key} model {{path}} is", load))
            continue
        local = os.path.expanduser(v)
        if os.path.isabs(local) or os.path.isdir(os.path.join(cwd, local)):
            out.append((_real(os.path.join(cwd, local)), f"the {key} model {{path}} is",
                        load))
    servers = list(cfg.assistant.mcp)
    for alias in cfg.assistants.values():
        servers += alias.mcp or []
    for server in servers:
        if not server.command:
            continue
        start = "before gmlx next starts that tool server on the Mac"
        program = os.path.expanduser(server.command[0])
        if os.path.isabs(program) or "/" in program:
            out.append((_real(os.path.join(cwd, program)),
                        f"the tool server {server.name} runs {{path}}, which is", start))
        given = [*server.command[1:], *(e for v in server.env.values()
                                        for e in str(v).split(os.pathsep))]
        for arg in given:
            for p in (arg, arg.partition("=")[2]):
                if p.startswith(("/", "~")):
                    out.append((_real(os.path.expanduser(p)),
                                f"the tool server {server.name} uses {{path}}, which is",
                                start))
    return list(dict.fromkeys(out))


def pythonpath_warnings(shares: list[Mount]) -> list[str]:
    """Warnings when ``PYTHONPATH`` leads into a folder the client can write.
    An empty or relative entry puts the current folder on the import path.
    gmlx keeps that entry out of the processes it starts, but a ``gmlx``
    command you run yourself from the share imports the client's package
    before any gmlx code runs. An absolute entry in a read-write share
    reaches every gmlx process."""
    from gmlx.serve.procname import pythonpath_holds_cwd

    rw = [m for m in shares if m.kind in ("share", "git") and not m.readonly]
    if not rw:
        return []
    out = []
    if pythonpath_holds_cwd():
        out.append("[launch] warning: PYTHONPATH has an empty or relative entry, which puts "
                   "the current folder on Python's import path. A gmlx package the client "
                   "writes in a read-write share would run on the Mac the next time you run "
                   "gmlx from that folder. Remove the entry from PYTHONPATH.")
    home = _host_home()
    for entry in (os.environ.get("PYTHONPATH") or "").split(os.pathsep):
        if not os.path.isabs(entry):
            continue
        m = next((m for m in rw if _inside(entry, m.source) or _inside(_real(entry), m.source)),
                 None)
        if m is not None:
            out.append(f"[launch] warning: PYTHONPATH holds {_tilde(entry, home)}, which lies "
                       f"in the read-write share {_tilde(m.source, home)}. The client can add "
                       "a module there that the next gmlx command imports on the Mac. Remove "
                       "the entry from PYTHONPATH, or share the folder read-only.")
    return list(dict.fromkeys(out))


def server_config_warnings(config_path: str | None, shares: list[Mount]) -> list[str]:
    """Warnings for a server config, model folder or model file the client
    can change through a read-write share, and for a server key the client
    can read in any share. Never scans a model folder and never blocks, so
    a planted file cannot stall the launch."""
    if not config_path:
        return []
    import yaml

    from gmlx.config import build_config
    from gmlx.serve.lifecycle import server_cwd

    cwd = server_cwd(config_path)
    home = _host_home()
    rw = [m for m in shares if m.kind in ("share", "git") and not m.readonly]
    written, real = os.path.abspath(config_path), _real(config_path)
    out = []
    in_share = next((m for m in rw if _inside(written, m.source) or _inside(real, m.source)),
                    None)
    if in_share is not None:
        out.append(f"[launch] warning: the client can change the server config "
                   f"{_tilde(written, home)} in the read-write share "
                   f"{_tilde(in_share.source, home)}, and the server applies a change at its "
                   "next reload.")
    # A client can replace the config with a link to any file of yours, so
    # one that leads out of a folder a client could write is never read.
    for folder in dict.fromkeys([*(m.source for m in rw), *shared_history()]):
        if _inside(written, folder) and not _inside(real, folder):
            out.append(f"[launch] warning: the server config {_tilde(written, home)} leads "
                       f"to {_tilde(real, home)}, outside {_tilde(folder, home)}, which a "
                       "session shares or once shared read-write. A client may have replaced "
                       "it with a symbolic link, so launch did not read it. Check it before "
                       "the server reloads.")
            return out
    try:
        doc = yaml.safe_load(_read_small_file(real))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = build_config(doc if isinstance(doc, dict) else {})
    except Exception as e:  # noqa: BLE001 - a broken config must never stop the launch
        out.append(f"[launch] could not check the server config {_tilde(real, home)} "
                   f"({str(e).splitlines()[0] if str(e) else type(e).__name__}).")
        return out
    held = next((m for m in shares if m.kind in ("share", "git")
                 and (_inside(written, m.source) or _inside(real, m.source))), None)
    if cfg.api_key and held is not None:
        out.append(f"[launch] warning: the server config {_tilde(written, home)} sets "
                   f"server.api_key, and the client can read it in the share "
                   f"{_tilde(held.source, home)}. With the key, the client can call every "
                   "route of the server wherever it reaches the server's port. Move the "
                   "config out of the share.")
    for spec in cfg.discover:
        folders = [spec.dir] if spec.dir else list(cfg.model_dirs)
        for folder in folders:
            f = _expand(folder, cwd)
            for m in rw:
                if _inside(f, m.source) or (spec.recursive and _inside(m.source, f)):
                    out.append(f"[launch] warning: the server scans {_tilde(f, home)} for "
                               f"models, and the client can add files there through "
                               f"{_tilde(m.source, home)}.")
                    break
    for path, what, when in _model_paths(cfg, cwd):
        m = next((m for m in rw if _inside(path, m.source)), None)
        if m is not None:
            out.append(f"[launch] warning: {what.replace('{path}', _tilde(path, home))} "
                       f"inside the read-write share, so the client can replace it {when}.")
    return list(dict.fromkeys(out))
