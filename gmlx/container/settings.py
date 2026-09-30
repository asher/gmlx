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
import secrets
import stat
import subprocess
import tempfile
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from gmlx.config import (LaunchClientCfg, parse_size_bytes, parse_volume_spec)

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
# Paths that hold credentials, gmlx's own data, or programs and settings the
# Mac runs. Relative ones are under $HOME.
SENSITIVE = (".ssh", ".gnupg", ".aws", ".azure", ".config/gcloud", ".kube",
             ".docker", ".password-store", "Library/Keychains", ".netrc",
             ".config/gh", ".npmrc", ".git-credentials", ".config/gmlx",
             ".cache/gmlx", ".local/share/gmlx", "Library/LaunchAgents",
             ".config/git", ".local/bin", "bin", "Library/Application Support",
             ".cargo", ".cache/huggingface", ".codex", "/opt/homebrew", "/usr/local")
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


def temp_trees() -> list[str]:
    """The folders that hold the temporary and cache files of every program
    of this user, gmlx's fallback socket folders among them."""
    trees = [_real(tempfile.gettempdir()), _real(TEMP_FOLDERS)]
    if os.environ.get("TMPDIR"):
        trees.append(_real(os.environ["TMPDIR"]))
    return list(dict.fromkeys(trees))


def auto_share_refusal(path: str, home: str | None = None) -> str | None:
    """Why launch will not share ``path`` by default, or None."""
    home = home or _host_home()
    if any(_same(path, f) for f in SYSTEM_FOLDERS):
        return f"{path} is a system folder"
    trees = [t for t in temp_trees() if _inside(path, t)]
    if trees:
        tree = max(trees, key=len)
        where = path if _same(path, tree) else f"{path} lies in {tree}, which"
        return f"{where} holds the temporary files of your programs"
    if _inside(home, path):
        return f"{_tilde(path, home)} is your home folder or holds it"
    why = _data_refusal(path, home)
    if why is not None:
        return why
    hits = sensitive_hits(path, home)
    if hits:
        return (f"{_tilde(path, home)} holds or lies in a credential folder "
                f"({', '.join(_tilde(h, home) for h in hits)})")
    return None


def _data_refusal(path: str, home: str) -> str | None:
    """Why ``path`` never reaches a container: it holds or lies in the
    folder with every private home and the guest entry."""
    data = _real(data_path())
    if _inside(path, data) or _inside(data, path):
        return f"{_tilde(path, home)} holds or lies in container-mode data"
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
            return (f"{_tilde(path, home)} holds or lies in {_tilde(folder, home)}, "
                    "where gmlx keeps its settings and server state")
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


def check_mount_chars(path: str, what: str) -> None:
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
    # A folder a client could write in an earlier, wider share may now be a
    # link to somewhere else, so a mount is taken only by its real path.
    written = os.path.abspath(source)
    if not _same(written, real):
        raise SettingsError(f"mount {spec!r}: {source} is a symbolic link to "
                            f"{_tilde(real, home)}, or passes through one. Write the "
                            f"folder's real path, {_tilde(real, home)}, if you mean it.")
    why = _data_refusal(real, home) or _state_refusal(real, home)
    if why is not None:
        raise SettingsError(f"launch will not share {source}: {why}.")
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
        target = os.path.normpath(m.target)
        m = replace(m, target=target)
        key = (m.source, target, m.readonly, m.kind, m.size)
        if key in seen:
            continue
        seen.add(key)
        if target == "/":
            raise SettingsError(f"{_label(m)} cannot be mounted at /.")
        for reserved in RESERVED_TARGETS:
            if _inside(target, reserved) or _inside(reserved, target):
                raise SettingsError(f"{_label(m)} cannot be mounted at {target}, which "
                                    f"covers {reserved}.")
        if m.kind != "volume":
            check_mount_chars(m.source, "the folder")
        check_mount_chars(target, "the guest path")
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
    # By components, since the source may differ from the path in case.
    rest = "/".join(host_path.split("/")[len(best.source.rstrip("/").split("/")):])
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
        # The repository config is the guest's, so no command it names runs.
        proc = subprocess.run(["git", "-c", "core.fsmonitor=false", "-C", cwd, *args],
                              capture_output=True, text=True, timeout=5)
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
                          f"git folder {_tilde(common, home)}, because {why}. Use git on "
                          "the Mac for this repository."]
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


def private_home_path(client: str) -> Path:
    """Where the client's private home lives, without creating it."""
    return data_path() / client / "home"


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
                 web_port: int | None = None,
                 build_folders: dict[str, str] | None = None) -> ContainerPlan:
    """The mounts, volumes and ports of one session, from the effective
    client config and the flags. ``build_folders`` maps each client to its
    configured ``build:`` path, and no read-write share may overlap one."""
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
        mount = _explicit_mount(spec, warns, home)
        if share_cwd and (mount.source, os.path.normpath(mount.target)) == (cwd_real, cwd_real):
            # A mount of the current folder at its own path sets how it is
            # shared, such as read-only, in place of the default share.
            mounts = [m for m in mounts if m.note != "working folder"]
        mounts.append(mount)
    git_mount, git_notes = (git_extra_mount(cwd_real, list(mounts), home)
                            if share_cwd else (None, []))
    if git_mount is not None:
        mounts.append(git_mount)
    notes.extend(git_notes)
    guest_home = private_home(client)
    mounts.append(Mount(str(guest_home), str(guest_home), kind="home"))
    mounts.extend(_volume_mount(v) for v in cfg.volumes)
    mounts = normalize_mounts(mounts)
    _refuse_build_folder_shares(mounts, build_folders or {}, home)
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
                    f"launch will not share {_tilde(m.source, home)} read-write, because it "
                    f"overlaps {_tilde(folder, home)}, the build: folder of {client}. The "
                    "client could change what that image runs at its next build. Share it "
                    f"read-only with --mount {_tilde(m.source, home)}:ro, or move the build "
                    "folder.")


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
            raise SettingsError(f"{_label(m)} changed after launch checked it: it is no "
                                "longer the same folder, or it is a symbolic link now. "
                                "Launch again to check it.")


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


def _read_seed_record(path: Path) -> set[str]:
    import json

    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return set()
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > CONFIG_READ_MAX:
            return set()
        doc = json.loads(os.read(fd, CONFIG_READ_MAX).decode())
    except (OSError, ValueError, RecursionError):
        return set()
    finally:
        os.close(fd)
    seeded = doc.get("seeded") if isinstance(doc, dict) else None
    return {x for x in seeded if isinstance(x, str)} if isinstance(seeded, list) else set()


def _seed_source_refusal(real: str, host_home: str) -> str | None:
    """Why launch will not copy the file or folder at ``real``, or None."""
    if not _inside(real, host_home) or _same(real, host_home):
        return f"it is {_tilde(real, host_home)}, outside your home folder"
    why = _data_refusal(real, host_home)
    if why is not None:
        return why
    hits = sensitive_hits(real, host_home)
    if hits:
        return (f"{_tilde(real, host_home)} holds or lies in a credential folder "
                f"({', '.join(_tilde(h, host_home) for h in hits)})")
    return None


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
    :data:`SHARED_HISTORY_MAX` of each."""
    import json

    now = [m.source for m in plan.mounts if not m.readonly and m.kind in ("share", "git")]
    bound = [(m.worktree, m.source) for m in plan.mounts if m.kind == "git" and m.worktree]
    old, old_pairs = shared_history(), worktree_history()
    merged = list(dict.fromkeys([*now, *old]))[:SHARED_HISTORY_MAX]
    pairs = list(dict.fromkeys([*bound, *old_pairs]))[:SHARED_HISTORY_MAX]
    if merged != old or pairs != old_pairs:
        data_dir()
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
            return (f"{_tilde(src, home)} lies in {_tilde(folder, home)}, which a session "
                    f"shared read-write, and it leads to {_tilde(real, home)} outside that "
                    "folder, so a client may have replaced it with a symbolic link")
    return None


def seed_home(home: Path, seeds: list[str], *, reseed: bool = False,
              writable: Sequence[str] = ()) -> list[str]:
    """Copy each seed into the private home once, at the same path relative
    to ``$HOME``, then add the host git identity where it is missing.
    Returns the lines to print. Every write is confined to the private
    home, since the guest can plant links there.

    Launch records each seed it copied beside the private home, so a copy
    the client deletes is not made again. ``reseed`` copies every seed
    again, replacing the copy in the private home. ``writable`` holds the
    folders this session shares read-write. With the folders earlier
    sessions shared, a seed in one of them must not lead out of it."""
    import json

    from . import confine

    # Relative to $HOME as written, so a seed that is a link into a dotfiles
    # repository still lands at its own path.
    host_home = os.path.abspath(os.path.expanduser("~"))
    host_real = _real(host_home)
    record = seed_record_path(home)
    done = _read_seed_record(record)
    copied = set(done)
    guest_written = list(dict.fromkeys([*writable, *shared_history()]))
    out = []
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
            if src in done and not reseed:
                continue
            dst = home / os.path.relpath(src, host_home)
            # The copy goes to a new name first and is renamed into place
            # only when it is whole, so a failed copy is never taken for a
            # finished one at the next launch.
            prefix = f".{dst.name}.gmlx-seed-"
            tmp = dst.with_name(prefix + secrets.token_hex(4))
            try:
                if confine.exists(dst) and not reseed:
                    copied.add(src)           # a copy made before the record
                    continue
                # Only a copy is checked, so a link a client swapped in
                # after the copy never stops a later launch. A link in the
                # way may have been left by a client in a folder an earlier
                # launch shared, so the real path decides.
                real = _real(src)
                why = (_seed_source_refusal(real, host_real)
                       or _seed_link_refusal(src, real, guest_written, host_real))
                if why is not None:
                    raise SettingsError(f"seed: launch will not copy {shown}, because {why}.")
                if not _same(real, src):
                    out.append(f"[launch] seed: copying {shown} from "
                               f"{_tilde(real, host_real)}, where its symbolic link leads.")
                out.extend(_seed_token_warnings(shown, real, host_real))
                # A copy that a killed launch left half done is removed first.
                for name in confine.listdir(dst.parent):
                    if name.startswith(prefix):
                        confine.remove_tree(dst.parent / name)
                _copy_confined(real, tmp)
                confine.remove_tree(dst)
                confine.rename(tmp, dst.name)
                copied.add(src)
            except BaseException as e:
                with contextlib.suppress(confine.ConfinedError, OSError):
                    confine.remove_tree(tmp)
                if isinstance(e, confine.ConfinedError):
                    raise SettingsError(f"seed: {e}") from None
                if isinstance(e, OSError):
                    raise SettingsError(f"seed: cannot copy {shown} ({e}).") from None
                raise
        if copied != done:
            write_record(record, json.dumps({"seeded": sorted(copied)}).encode())
        try:
            _seed_git_identity(home)
        except confine.ConfinedError as e:
            # The guest owns the file, so a file launch cannot read or
            # replace costs only the identity, never the launch.
            out.append(f"[launch] warning: {e} Launch did not add your git identity to it.")
    return out


def _seed_token_warnings(shown: str, real: str, host_real: str) -> list[str]:
    out = []
    tokens = [t for t in TOKEN_FILES if _inside(os.path.join(host_real, t), real)]
    if tokens:
        out.append(f"[launch] warning: seed {shown} copies "
                   f"{', '.join('~/' + t for t in tokens)}, which holds a sign-in "
                   "token. The client can read it.")
    maybe = [t for t in TOKEN_MAYBE_FILES if _inside(os.path.join(host_real, t), real)]
    if maybe:
        out.append(f"[launch] warning: seed {shown} copies "
                   f"{', '.join('~/' + t for t in maybe)}, which can hold a token, such as "
                   "one in a url.<base>.insteadOf address. The client can read it.")
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
        why = _seed_source_refusal(opened, _real(os.path.expanduser("~")))
        if why is None and not _same(opened, src):
            why = f"it changed to {_tilde(opened)} while launch copied it"
        if why is not None:
            raise SettingsError(f"seed: launch will not copy {_tilde(src)}, because {why}.")
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


def _seed_git_identity(home: Path) -> None:
    """Add the host ``user.name`` and ``user.email`` to the private home's
    ``.gitconfig`` where they are missing. git edits a copy outside the
    private home, since git follows a link at the file it writes, and the
    result goes back through the confined write."""
    import tempfile

    from . import confine

    gitconfig = home / ".gitconfig"
    before = confine.read_text(gitconfig) or ""
    with tempfile.TemporaryDirectory() as tmp:
        work = os.path.join(tmp, "gitconfig")
        Path(work).write_text(before)
        for key in ("user.name", "user.email"):
            try:
                have = subprocess.run(["git", "config", "--file", work, "--get", key],
                                      capture_output=True, text=True, timeout=5)
                if have.returncode == 0:
                    continue
                value = subprocess.run(["git", "config", "--global", "--get", key],
                                       capture_output=True, text=True, timeout=5)
                if value.returncode == 0 and value.stdout.strip():
                    subprocess.run(["git", "config", "--file", work, key,
                                    value.stdout.strip()], capture_output=True, timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                return
        after = Path(work).read_text()
    if after != before:
        confine.write_text(gitconfig, after)


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
        # Such a server scans the folder it started from, or --models-dir,
        # which the runfile does not record.
        if notes is not None:
            notes.append(f"[launch] the server on port {port} runs without a config file, "
                         "so launch cannot check whether the folders it scans for models "
                         "are shared. A client could add a model file to a shared folder "
                         "it scans. Start the server with --config to have it checked.")
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


def _model_paths(cfg, cwd: str) -> list[str]:
    """The files the config lists, resolved the way the server resolves them
    from ``cwd``, without reading them. As in ``resolve_path``, a relative
    path that no model folder holds is taken from the working folder."""
    out = []
    roots = [_expand(d, cwd) for d in cfg.model_dirs]
    for model in cfg.models.values():
        for p in (model.path, model.mmproj, model.draft_gguf, model.adapter):
            if not p or str(p).startswith("hf:"):
                continue
            p = os.path.expandvars(os.path.expanduser(str(p)))
            if os.path.isabs(p):
                out.append(_real(p))
                continue
            cand = next((c for c in (os.path.join(r, p) for r in roots)
                         if os.path.exists(c)), os.path.join(cwd, p))
            out.append(_real(cand))
    return out


def pythonpath_warnings(shares: list[Mount]) -> list[str]:
    """A warning when ``PYTHONPATH`` puts the current folder on the import
    path and the client can write a shared folder. gmlx keeps that entry out
    of the processes it starts, but a ``gmlx`` command you run yourself from
    the share imports the client's package before any gmlx code runs."""
    from gmlx.serve.procname import pythonpath_holds_cwd

    if not pythonpath_holds_cwd() or not any(
            m.kind in ("share", "git") and not m.readonly for m in shares):
        return []
    return ["[launch] warning: PYTHONPATH has an empty or relative entry, which puts the "
            "current folder on Python's import path. A gmlx package the client writes in "
            "a read-write share would run on the Mac the next time you run gmlx from that "
            "folder. Remove the entry from PYTHONPATH."]


def server_config_warnings(config_path: str | None, shares: list[Mount]) -> list[str]:
    """Warnings for a server config, model folder or model file the client
    can change through a read-write share. Never scans a model folder and
    never blocks, so a planted file cannot stall the launch."""
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
        out.append(f"[launch] warning: the server config {_tilde(written, home)} is inside "
                   f"the read-write share {_tilde(in_share.source, home)}. The client can "
                   "change it, and the server applies the change at its next reload.")
    # A client can replace the config with a link to any file of yours, so
    # one that leads out of a folder a client could write is never read.
    for folder in dict.fromkeys([*(m.source for m in rw), *shared_history()]):
        if _inside(written, folder) and not _inside(real, folder):
            out.append(f"[launch] warning: the server config {_tilde(written, home)} lies in "
                       f"{_tilde(folder, home)}, which a session shares or once shared "
                       f"read-write, and it leads to {_tilde(real, home)} outside that "
                       "folder, so a client may have replaced it with a symbolic link. "
                       "Launch did not read it. Check it before the server reloads.")
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
    for path in _model_paths(cfg, cwd):
        m = next((m for m in rw if _inside(path, m.source)), None)
        if m is not None:
            out.append(f"[launch] warning: the model file {_tilde(path, home)} is inside the "
                       "read-write share, so the client can replace it before the server's "
                       "next load.")
    return list(dict.fromkeys(out))
