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
import threading
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from gmlx.config import (AGENT_DEPS_TARGET, LaunchClientCfg, agent_name, config_key,
                         normal_guest_target, parse_size_bytes, parse_volume_spec,
                         target_label)

from . import notices
from .notices import Once
from gmlx.safe_path import folded, same_name

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
# Mac runs, by what they hold. Relative ones are under $HOME. The claude
# program in ~/.local/bin leads to a file in ~/.local/share/claude.
CREDENTIAL_PATHS = (".ssh", ".gnupg", ".aws", ".azure", ".config/gcloud", ".kube",
                    ".docker", ".password-store", "Library/Keychains", ".netrc",
                    ".config/gh", ".npmrc", ".git-credentials", ".config/gmlx",
                    ".cache/huggingface", ".codex")
GMLX_DATA_PATHS = (".cache/gmlx", ".local/share/gmlx")
RUN_PATHS = ("Library/LaunchAgents", ".config/git", ".local/bin", "bin",
             "Library/Application Support", ".cargo", "/opt/homebrew", "/usr/local",
             ".local/share/claude")
# The startup files of zsh, which ZDOTDIR can move.
ZSH_FILES = (".zshenv", ".zprofile", ".zshrc", ".zlogin", ".zlogout")
# The settings of git, the shells and the editors. git, a new shell or the
# editor runs the commands in them on the Mac. A dotfiles folder often holds
# the real files, and the files in $HOME are links to them. Vim, tmux and
# Emacs also read their settings in ~/.config, and Neovim loads the plugins
# in its data folder.
COMMAND_PATHS = (".gitconfig", *ZSH_FILES, ".bashrc", ".bash_profile", ".bash_login",
                 ".bash_logout", ".profile", ".config/fish", ".vimrc", ".vim", ".exrc",
                 ".config/vim", ".config/nvim", ".local/share/nvim", ".tmux.conf",
                 ".config/tmux", ".emacs", ".emacs.el", ".emacs.d", ".config/emacs")
SENSITIVE = CREDENTIAL_PATHS + GMLX_DATA_PATHS + RUN_PATHS + COMMAND_PATHS
# The folders of RUN_PATHS where the shell finds a command that you type.
# CARGO_HOME moves the last one. A link in one of them runs only when you
# run its name, as a program in a PATH folder does. So a share that holds
# the file that such a link leads to gets a warning, not a refusal.
PROGRAM_PATHS = (".local/bin", "bin", ".cargo/bin")
_CREDENTIALS = "credentials"
_COMMANDS = "commands the Mac runs"
_OWN_DATA = "gmlx's own data"
# The kind of a link in a folder of PROGRAM_PATHS. No check refuses a share
# for it, and :func:`_program_link_warnings` names it.
_PROGRAM_LINK = "a program that you run by name"
_HOLDS = {_CREDENTIALS: CREDENTIAL_PATHS, _OWN_DATA: GMLX_DATA_PATHS,
          "files the Mac runs": RUN_PATHS, _COMMANDS: COMMAND_PATHS}
# A folder of the tables can hold a link to a file outside it, such as
# ~/.ssh/config or ~/.claude/settings.json that leads to a dotfiles folder.
# Launch looks for such links in the folder and in all its subfolders. It
# reads at most LINK_WALK_MAX entries for each folder of the tables, and
# each folder that it opens counts as LINK_WALK_OPEN entries. Each
# subfolder gets an equal part of the entries that are left, and the part
# that a small subfolder does not use goes to the subfolders after it. So
# a large folder of state, such as ~/.claude/projects, cannot use the part
# of a small folder beside it, such as ~/.claude/hooks.
LINK_WALK_MAX = 4096
LINK_WALK_OPEN = 16
# The paths of the tables where launch does not look for such links: the
# data of every app, and the package installations, whose many links lead
# into them. gmlx's own data is not searched either, because the clients
# write their private homes there.
LINK_WALK_SKIP = ("Library/Application Support", "/opt/homebrew", "/usr/local")
# The variables that move a path of the tables above to another folder or
# file, each with the path it moves. The tool then keeps its credentials,
# or reads the settings that run commands, at the path that the variable
# names. KUBECONFIG names a list of files. ZDOTDIR names the folder of the
# zsh startup files.
SENSITIVE_PATH_VARS = (("GNUPGHOME", ".gnupg"), ("AWS_CONFIG_FILE", ".aws"),
                       ("AWS_SHARED_CREDENTIALS_FILE", ".aws"), ("AZURE_CONFIG_DIR", ".azure"),
                       ("CLOUDSDK_CONFIG", ".config/gcloud"), ("KUBECONFIG", ".kube"),
                       ("DOCKER_CONFIG", ".docker"), ("PASSWORD_STORE_DIR", ".password-store"),
                       ("GH_CONFIG_DIR", ".config/gh"), ("NPM_CONFIG_USERCONFIG", ".npmrc"),
                       ("HF_HOME", ".cache/huggingface"),
                       ("HF_TOKEN_PATH", ".cache/huggingface"), ("CODEX_HOME", ".codex"),
                       ("CARGO_HOME", ".cargo"), ("GIT_CONFIG_GLOBAL", ".gitconfig"),
                       ("ZDOTDIR", ".zshrc"))
# The folders where each client keeps its settings, history and sign-in on
# the Mac, under $HOME. A hook a guest adds there runs on the Mac, and the
# host-mode configs there hold the server key. Seeds may copy from them.
# opencode installs its plugins in its cache folder and imports them from
# there.
CLIENT_PATHS = {".claude": "claude-code", ".pi": "pi", ".omp": "omp", ".hermes": "hermes",
                ".open-webui": "open-webui", ".dsh": "dsh", ".config/goose": "goose",
                ".config/opencode": "opencode", ".local/share/opencode": "opencode",
                ".cache/opencode": "opencode", ".opencode": "opencode", ".config/elia": "elia",
                "Library/Application Support/aichat": "aichat", ".config/aichat": "aichat"}
# The variables that move a client's folder, or a file of its settings, to
# another path, each with the client. The client then reads its settings,
# and the hooks, plugins and tools in them, from that path. aichat runs the
# tools in its functions folder.
CLIENT_PATH_VARS = (("HERMES_HOME", "hermes"), ("DSH_HOME", "dsh"),
                    ("CLAUDE_CONFIG_DIR", "claude-code"), ("PI_CODING_AGENT_DIR", "pi"),
                    ("OPENCODE_CONFIG_DIR", "opencode"), ("OPENCODE_CONFIG", "opencode"),
                    ("AICHAT_CONFIG_DIR", "aichat"),
                    ("AICHAT_CONFIG_FILE", "aichat"), ("AICHAT_ENV_FILE", "aichat"),
                    ("AICHAT_FUNCTIONS_DIR", "aichat"))
# The XDG variables that move the client folders in CLIENT_PATHS under
# these folders of $HOME.
CLIENT_XDG_VARS = {".config": "XDG_CONFIG_HOME", ".local/share": "XDG_DATA_HOME",
                   ".cache": "XDG_CACHE_HOME"}
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
# The links that most Linux images hold, the shipped images and Debian,
# Ubuntu and Alpine among them. Apple container resolves a mount's target
# inside the image and follows them, so the reserved checks apply where a
# target leads.
IMAGE_LINKS = {"/var/run": "/run"}
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
# The folders of SYSTEM_PATH on the read-only system volume of macOS. No
# client can change a program there, also when a session shares the folder.
SEALED_PATH = ("/usr/bin", "/bin")
# The shims in SEALED_PATH that launch runs, each with the path of its
# program in the active developer folder. /usr/bin/git runs the git of that
# folder, which xcrun finds through DEVELOPER_DIR, TOOLCHAINS, SDKROOT and a
# cache in the user's temporary folder, and xcrun can first run xcodebuild
# from that folder. So launch runs the program of the developer folder
# itself, and the share checks cover that folder.
DEVELOPER_SHIMS = {"/usr/bin/git": "usr/bin/git"}
# The program that names the active developer folder. It is on the
# read-only system volume, and it runs no program of that folder.
XCODE_SELECT = "/usr/bin/xcode-select"
# The folders of an installation such as /opt/homebrew that a program from
# it does not read when it runs: data and logs, headers for a build,
# Homebrew's own code, which runs only when you run brew, and the casks.
# An earlier share of one of them does not refuse the program, unless the
# program itself leads through it.
INSTALLATION_UNREAD = ("var", "include", "Library", "Homebrew", "Caskroom")
# The Homebrew package of each program that launch runs, where its name is
# not the name of the program.
BREW_FORMULAS = {"ssh-add": "openssh"}


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
    # A runtime agent's project folder in the guest, and whether the share
    # that holds it is read-only.
    source_guest: str | None = None
    source_readonly: bool = False

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


def _and_list(items: Sequence[str]) -> str:
    """``a``, ``a and b`` or ``a, b and c``."""
    items = list(items)
    if len(items) < 2:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


class _Resolver:
    """The real paths and the resolution paths of one build of the tables.
    The links that the walk finds share their parent folders, so each
    folder is named by macOS once, and each path is read as a link once."""

    def __init__(self):
        self._folders: dict[str, str] = {}
        self._real: dict[str, str] = {}
        self._links: dict[str, str | None] = {}

    def folder(self, path: str) -> str:
        """:func:`_real` of the folder ``path``."""
        hit = self._folders.get(path)
        if hit is None:
            hit = self._folders[path] = _real(path)
        return hit

    def real(self, path: str) -> str:
        """:func:`_real` of ``path``: a folder as macOS names it, else the
        name in the folder that holds it, as :func:`canonical` gives it."""
        hit = self._real.get(path)
        if hit is None:
            resolved = os.path.realpath(path)
            parent, name = os.path.split(resolved)
            if os.path.isdir(resolved) or not name or parent == resolved:
                hit = self.folder(resolved)
            else:
                hit = os.path.join(self.folder(parent), name)
            self._real[path] = hit
        return hit

    def readlink(self, path: str) -> str:
        """``os.readlink`` of ``path``, which raises OSError when it is not
        a link."""
        if path not in self._links:
            try:
                self._links[path] = os.readlink(path)
            except OSError:
                self._links[path] = None
        target = self._links[path]
        if target is None:
            raise OSError(f"{path} is not a link")
        return target


def _links_out(folder: str, home: str, resolver: _Resolver | None = None) -> list[str]:
    """Each link in ``folder`` and in its subfolders, as written through
    ``folder``, whose real path lies outside the folder, such as
    ~/.ssh/config when it leads to a file in a dotfiles folder. The Mac
    reads that file as a part of the folder, so a share that holds the file
    holds a part of the folder. A link whose real path is or holds the home
    folder is left out, because every folder in it would then be a part.
    The entries of a folder come before those of its subfolders, and
    :data:`LINK_WALK_MAX` sets how many entries launch reads. ``resolver``
    keeps the real path of each link for the build of the tables."""
    resolver = resolver or _Resolver()
    real = resolver.real(folder)
    if not os.path.isdir(real):
        return []
    home = resolver.folder(home)
    out: list[str] = []

    def walk(path: str, budget: int) -> int:
        """Read ``path`` and its subfolders with at most ``budget``
        entries, and return how many of them it used."""
        try:
            with os.scandir(path) as it:
                entries = sorted((e for _, e in zip(range(budget - LINK_WALK_OPEN), it)),
                                 key=lambda e: e.name)
        except OSError:
            return LINK_WALK_OPEN
        used = LINK_WALK_OPEN + len(entries)
        below: list[str] = []
        for entry in entries:
            try:
                if entry.is_symlink():
                    target = resolver.real(entry.path)
                    if not _inside(target, real) and not _inside(home, target):
                        out.append(entry.path)
                elif entry.is_dir(follow_symlinks=False):
                    below.append(entry.path)
            except OSError:
                continue
        for i, sub in enumerate(below):
            share = (budget - used) // (len(below) - i)
            if share > LINK_WALK_OPEN:
                used += walk(sub, share)
        return used

    walk(folder, LINK_WALK_MAX)
    return out


class _LinkWhy(str):
    """Why launch does not share a path, as a phrase that names a link in
    a protected folder. ``link`` is the link, ``folder`` is the protected
    folder that holds it, and ``what`` follows the folder to say what it
    holds, such as "which holds credentials". ``own`` is true when the
    protected path is itself the link, such as ~/bin that leads to a
    folder in a repository, or a link below the home folder leads through
    it, and ``folder`` is then the protected path too. A phrase
    that names several links has ``links``, the first link among them, and
    ``own`` is true when any protected path is itself its link.
    ``secret`` holds the links that lead to credentials or to a sign-in
    token. For one link, a phrase whose ``what`` says that the folder holds
    credentials makes the link secret too."""

    link: str
    folder: str
    what: str
    own: bool
    links: tuple[str, ...]
    secret: tuple[str, ...]

    def __new__(cls, text: str, link: str, folder: str, what: str, own: bool = False,
                links: Sequence[str] = (), secret: Sequence[str] = ()):
        why = super().__new__(cls, text)
        why.link, why.folder, why.what, why.own = link, folder, what, own
        why.links = tuple(dict.fromkeys(links or [link]))
        credentials = not links and what == f"which holds {_CREDENTIALS}"
        why.secret = tuple(dict.fromkeys(secret or ([link] if credentials else [])))
        return why


class _LinkOnWay(str):
    """Why launch does not share a path, as a phrase that names a link in
    it on the way to a protected folder outside it. A read-only share keeps
    the client from changing where the link leads."""


# Each path of the tables as the Mac finds it, with what it holds or the
# client's name, and the folder of the tables that holds it, or None.
_Written = list[tuple[str, str, str | None]]
# A path of the tables by real path, with what it holds, its link and its
# folder, from :func:`_named_link`.
_Kinds = dict[str, tuple[str, str | None, str | None]]


@dataclass
class _Tables:
    """The sensitive paths and the client folders of one home, as written,
    each with its real path and the paths that resolving it visits. One
    build serves every check that the build is given to, so that a check
    compares strings and makes no system call for most paths. The folded
    forms come first in each pair, for :func:`_near`."""
    sensitive: _Written
    clients: _Written
    real: dict[str, tuple[str, str]] = field(default_factory=dict)
    reach: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    kinds: _Kinds | None = None
    folders: _Kinds | None = None


class _LaunchMemo:
    """The answers :func:`launch_memo` keeps: the tables of each home, the
    reason of :func:`auto_share_refusal` for each path, and the developer
    folder. Each answer has the environment in its key, because the
    variables move the paths that a check looks at. Only the thread that
    started the block reads them."""

    def __init__(self):
        self.owner = threading.get_ident()
        self.tables: dict[tuple, _Tables] = {}
        self.reasons: dict[tuple, str | None] = {}
        self.developer: dict[tuple, str | None] = {}


_launch_memo: _LaunchMemo | None = None


@contextlib.contextmanager
def launch_memo():
    """Answer a repeated check of one launch in the block from its first
    answer: the tables of :func:`_tables`, the reason of
    :func:`auto_share_refusal` for each path, and the developer folder of
    :func:`_developer_folder`. A launch checks the current folder up to
    four times, and each check walks the folders of the tables. The answers
    hold only for the block, because the links and folders on disk can
    change between launches. A block inside a block of the same thread
    uses the outer answers."""
    global _launch_memo
    if _launch_memo_here() is not None:
        yield
        return
    _launch_memo = _LaunchMemo()
    try:
        yield
    finally:
        _launch_memo = None


def _launch_memo_here() -> _LaunchMemo | None:
    memo = _launch_memo
    return memo if memo is not None and memo.owner == threading.get_ident() else None


def _environment() -> tuple:
    """The environment, as a key of the answers of :func:`launch_memo`."""
    return tuple(sorted(os.environ.items()))


def _tables(home: str) -> _Tables:
    """Build :class:`_Tables` for ``home``, with one walk of the folders.
    In :func:`launch_memo`, the first build for ``home`` serves the
    block."""
    memo = _launch_memo_here()
    key = (home, _environment())
    if memo is not None and key in memo.tables:
        return memo.tables[key]
    tables = _build_tables(home)
    if memo is not None:
        memo.tables[key] = tables
    return tables


def _build_tables(home: str) -> _Tables:
    resolver = _Resolver()
    sensitive = [(os.path.abspath(p), w, f) for p, w, f in _sensitive_written(home, resolver)]
    clients = [(os.path.abspath(p), c, f) for p, c, f in _client_written(home, resolver)]
    tables = _Tables(sensitive, clients)
    for path, _, _ in [*sensitive, *clients]:
        if path not in tables.real:
            real = resolver.real(path)
            tables.real[path] = (folded(real), real)
            tables.reach[path] = [(folded(p), p) for p in dict.fromkeys(
                [path, *_resolution_paths(path, resolver)])]
    return tables


def _near(fpath: str, ffolder: str) -> bool:
    """Whether the folded ``fpath`` is or lies in the folded ``ffolder``,
    which a path that :func:`_inside` takes always is."""
    return fpath == ffolder or fpath.startswith(ffolder.rstrip("/") + "/")


def _sensitive_written(home: str, resolver: _Resolver | None = None) -> _Written:
    """Each sensitive path as the Mac finds it, with its links, what it
    holds, such as "credentials", and None. A path in ~/.config,
    ~/.local/share or ~/.cache also has its form in the folder that the XDG
    variable names, because gmlx, git, gh, claude and Hugging Face look
    there when the variable is set. A variable in
    :data:`SENSITIVE_PATH_VARS` adds the path it names, unless that path is
    or holds the home folder, whose own files the tables name. Each link in
    a folder of these that leads out of it, from :func:`_links_out`, holds
    what the folder holds, and comes with that folder in place of None. A
    link that is an entry of a folder of :data:`PROGRAM_PATHS` holds
    :data:`_PROGRAM_LINK` instead. ``resolver`` serves the walk."""
    out: _Written = [
        (os.path.join(home, p), what, None) for what, paths in _HOLDS.items() for p in paths]
    for what, paths in _HOLDS.items():
        for p in paths:
            root, _, name = p.rpartition("/")
            var = CLIENT_XDG_VARS.get(root)
            if var is not None and os.environ.get(var):
                out.append((os.path.join(os.environ[var], name), what, None))
    kinds = {p: what for what, paths in _HOLDS.items() for p in paths}
    for var, rel in SENSITIVE_PATH_VARS:
        raw = os.environ.get(var, "")
        for value in raw.split(os.pathsep) if var == "KUBECONFIG" else [raw]:
            value = value.strip()
            path = os.path.abspath(os.path.expanduser(value)) if value else ""
            if path and not _inside(_real(home), _real(path)):
                out.append((path, kinds[rel], None))
    skip = {os.path.join(home, p) for p in LINK_WALK_SKIP}
    programs = {os.path.join(home, p) for p in PROGRAM_PATHS}
    cargo = os.environ.get("CARGO_HOME", "").strip()
    if cargo:
        programs.add(os.path.join(os.path.abspath(os.path.expanduser(cargo)), "bin"))
    for path, what, _ in list(dict.fromkeys(out)):
        if what != _OWN_DATA and path not in skip:
            out += [(link, _PROGRAM_LINK if os.path.dirname(link) in programs else what, path)
                    for link in _links_out(path, home, resolver)]
    return out


def _sensitive_kinds(home: str, tables: _Tables | None = None) -> _Kinds:
    """Each sensitive path, by real path, with what it holds, such as
    "credentials". For the real path of a link from :func:`_links_out`, the
    link and its folder follow. A sensitive path that is itself a link,
    such as ~/.gitconfig that leads to a dotfiles folder, or that a link
    below the home folder leads through, comes with itself twice. Else None
    and None follow. ``tables`` is :func:`_tables`, when
    the caller has it. When one real path has several kinds, the first
    that is not :data:`_PROGRAM_LINK` wins."""
    tables = _tables(home) if tables is None else tables
    if tables.kinds is None:
        out: _Kinds = {}
        for path, what, folder in tables.sensitive:
            real = tables.real[path][1]
            have = out.get(real)
            if have is None or (have[0] == _PROGRAM_LINK and what != _PROGRAM_LINK):
                out[real] = _named_link(path, what, folder, home)
        tables.kinds = out
    return tables.kinds


def _named_link(path: str, what: str, folder: str | None, home: str
                ) -> tuple[str, str | None, str | None]:
    """``(what, link, folder)`` for a path of the tables: the link and its
    folder for a link from :func:`_links_out`, the path twice for a path
    of the tables that is itself a link, else None and None. A path whose
    folder below ``home`` is a link, such as ~/.config/gh when ~/.config
    leads to a dotfiles folder, counts as a link too, so that the phrase
    names the path that leads to its real path."""
    if folder is not None:
        return what, path, folder
    if os.path.islink(path) or _linked_folder(path, home):
        return what, path, path
    return what, None, None


def _linked_folder(path: str, home: str) -> bool:
    """Whether a folder that holds ``path`` and lies below ``home`` is a
    link. The home folder and the folders above it do not count. The walk
    stops at the root folder, which is its own parent, so it also ends
    when the home folder is "/"."""
    top = home.rstrip("/") + "/"
    folder = os.path.dirname(path)
    while folder.startswith(top) and folder != top:
        if os.path.islink(folder):
            return True
        parent = os.path.dirname(folder)
        if parent == folder:
            break
        folder = parent
    return False


def sensitive_paths(home: str | None = None) -> list[str]:
    return [p for p, kind in _sensitive_kinds(home or _host_home()).items()
            if kind[0] != _PROGRAM_LINK]


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
    if _same(path, folder):
        return "holds the session sockets of gmlx"
    return f"lies in {folder}, which holds the session sockets of gmlx"


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
                if _same(path, folder):
                    return "holds the session sockets of gmlx"
                return f"lies in {folder}, which holds the session sockets of gmlx"
        elif _inside(root, path):
            try:
                held = sorted(n for n in os.listdir(root) if _GMLX_TEMP_FOLDER.match(n))
            except OSError:
                held = []
            if root in active or held:
                where = "where gmlx keeps the session sockets of its servers"
                return f"is {where}" if _same(path, root) else f"holds {root}, {where}"
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
    follows the path, such as "is your home folder", or None. In
    :func:`launch_memo`, the first answer for ``path`` serves the block."""
    home = home or _host_home()
    memo = _launch_memo_here()
    if memo is None:
        return _auto_share_refusal(path, home)
    key = (path, home, _environment(), tempfile.gettempdir())
    if key not in memo.reasons:
        memo.reasons[key] = _auto_share_refusal(path, home)
    return memo.reasons[key]


def _auto_share_refusal(path: str, home: str) -> str | None:
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
    # The tables come with the walk of their folders, so the checks share them.
    tables = _tables(home)
    return (_data_refusal(path, home) or _table_refusal(path, home, tables)
            or _link_refusal(path, home, tables))


def _table_refusal(path: str, home: str, tables: _Tables) -> str | None:
    """How ``path`` meets the sensitive paths and the client folders, from
    :func:`_sensitive_refusal` and :func:`_client_refusal`, as one phrase
    that follows the path, or None. When both apply, the phrase names both,
    so that the step names every link. A sensitive path that the path holds
    as itself, not through a link, is enough, and the phrase names only it.
    Beside a sensitive path, only the client folders that exist count."""
    first = _sensitive_refusal(path, home, tables=tables)
    if first is not None and not isinstance(first, _LinkWhy):
        return first
    then = _client_refusal(path, home, tables, existing=first is not None)
    if first is None or then is None:
        return first or then
    also = f"is also {then[3:]}" if then.startswith("is ") else f"also {then}"
    text = f"{first}. It {also}"
    if isinstance(then, _LinkWhy):
        return _LinkWhy(text, first.link, first.folder, first.what,
                        own=first.own or then.own, links=[*first.links, *then.links],
                        secret=[*first.secret, *then.secret])
    return text


def _link_refusal(path: str, home: str, tables: _Tables | None = None) -> str | None:
    """How ``path`` holds a link on the way to a folder that launch never
    shares by default, a sensitive one or a client's, as a phrase that
    follows the path, or None. The Mac finds such a folder by its path as
    written, so a client that changes the link chooses the folder that the
    Mac reads in its place. A path that holds or lies in the folder itself
    gets the check by its real path. For a link from :func:`_links_out`,
    the phrase says what its folder holds. When the link is the protected
    path itself, the phrase names where it leads. ``tables`` is
    :func:`_tables`, when the caller has it."""
    tables = _tables(home) if tables is None else tables
    fpath = folded(path)
    # Each folder whose path as written leads through ``path``, with the
    # paths in ``path`` that resolving it visits. String tests come first,
    # so a check makes system calls only for these.
    hits: list[tuple[str, str, str, list[str]]] = []
    entries = [*((p, w, t, False) for p, w, t in tables.sensitive if w != _PROGRAM_LINK),
               *((p, c, t, True) for p, c, t in tables.clients)]
    for folder, kind, top, client in entries:
        freal, real = tables.real[folder]
        if ((_near(freal, fpath) or _near(fpath, freal))
                and (_inside(real, path) or _inside(path, real))):
            continue
        reached = [p for fp, p in tables.reach[folder] if _near(fp, fpath) and _inside(p, path)]
        if not reached:
            continue
        if client:
            where = f"where {kind} keeps its settings and history on the Mac"
            what = where if top is None else f"and {_tilde(top, home)} is {where}"
        else:
            what = (f"which holds {kind}" if top is None
                    else f"and {_tilde(top, home)} holds {kind}")
        hits.append((folder, what, real, reached))
    if not hits:
        return None
    # A folder that exists comes first, so the phrase names one that you have.
    folder, what, real, reached = next((h for h in hits if os.path.exists(h[0])), hits[0])
    link = next((p for p in reached if os.path.islink(p)), reached[0])
    return _LinkOnWay(f"holds {_tilde(link, home)}, {_link_way(link, folder, real, home)}, "
                      f"{what}")


def _link_way(link: str, folder: str, real: str, home: str) -> str:
    """``a link to REAL`` when ``link`` is the protected path ``folder``
    itself, whose real path is ``real``, else ``a link on the way to
    FOLDER``."""
    if _same(link, folder):
        return f"a link to {_tilde(real, home)}"
    return f"a link on the way to {_tilde(folder, home)}"


def _relation(path: str, folder: str, home: str, what: str) -> str:
    """``is WHAT``, ``lies in FOLDER, WHAT`` or ``holds FOLDER, WHAT``. The
    phrase follows the path, so the folder that the path is goes unnamed."""
    if _same(path, folder):
        return f"is {what}"
    verb = "lies in" if _inside(path, folder) else "holds"
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


def _refuse_state_links(mounts: list[Mount], home: str) -> None:
    """A read-write share that holds a link on the way to gmlx's settings,
    its server state or the folder of the private homes lets the client
    point the link at a folder of its own. gmlx finds these folders by the
    paths as written, so it would then read the client's server config,
    server records or session records as its own. A share that holds or
    lies in one of the folders themselves is refused by its real path, in
    :func:`_state_refusal` and :func:`_data_refusal`."""
    state = "where gmlx keeps its settings and server state"
    folders = [(str(data_path()), "where launch keeps the private homes of the clients"),
               (os.path.expanduser("~/.config/gmlx"), state),
               (os.path.expanduser("~/.cache/gmlx"), state)]
    for var in ("XDG_CONFIG_HOME", "XDG_CACHE_HOME"):
        if os.environ.get(var):
            folders.append((os.path.join(os.environ[var], "gmlx"), state))
    for m in mounts:
        if m.readonly or m.kind not in ("share", "git"):
            continue
        shown = _tilde(m.source, home)
        for folder, what in folders:
            folder = os.path.abspath(folder)
            real = _real(folder)
            link = _link_in(m.source, folder)
            if link is None or _inside(real, m.source) or _inside(m.source, real):
                continue
            raise SettingsError(
                f"will not share {shown} read-write, because it holds {_tilde(link, home)}, "
                f"{_link_way(link, folder, real, home)}, {what}. The client could change "
                "where it leads, and gmlx would take the client's files there for its own.\n"
                f"  Share it read-only with --mount {shown}:ro.")


def _client_written(home: str, resolver: _Resolver | None = None) -> _Written:
    """Each folder where a client keeps its settings and history on the
    Mac, as the client finds it, with its links, the client's name and
    None. That is the folder in $HOME, and the folder or file that an
    environment variable such as CLAUDE_CONFIG_DIR, AICHAT_CONFIG_FILE or
    XDG_CONFIG_HOME moves it to. Each link in such a folder that leads out
    of it, from :func:`_links_out`, such as ~/.claude/settings.json in a
    dotfiles folder, counts as a part of the folder, and comes with that
    folder in place of None. ``resolver`` serves the walk."""
    out: _Written = [
        (os.path.join(home, rel), client, None) for rel, client in CLIENT_PATHS.items()]
    moved = [(var, "", client) for var, client in CLIENT_PATH_VARS]
    for rel, client in CLIENT_PATHS.items():
        root, _, name = rel.rpartition("/")
        if root in CLIENT_XDG_VARS:
            moved.append((CLIENT_XDG_VARS[root], name, client))
    for var, name, client in moved:
        value = os.environ.get(var, "").strip()
        if value:
            out.append((os.path.abspath(os.path.join(os.path.expanduser(value), name)), client,
                        None))
    for path, client, _ in list(dict.fromkeys(out)):
        out += [(link, client, path) for link in _links_out(path, home, resolver)]
    return out


def _client_folders(home: str, tables: _Tables | None = None) -> _Kinds:
    """Each folder where a client keeps its settings and history on the
    Mac, by real path, with the client's name, from :func:`_client_written`
    in ``tables``. For the real path of a link from :func:`_links_out`, the
    link and its folder follow, and for a client folder that is itself a
    link, or that a link below the home folder leads through, the folder
    twice; else None and None."""
    tables = _tables(home) if tables is None else tables
    if tables.folders is None:
        out: _Kinds = {}
        for path, client, folder in tables.clients:
            real = tables.real[path][1]
            if real not in out:
                out[real] = _named_link(path, client, folder, home)
        tables.folders = out
    return tables.folders


def _hits(path: str, kinds: _Kinds) -> list[str]:
    """The paths of ``kinds`` that ``path`` is, holds or lies in, in their
    order. The folded forms are compared first."""
    fpath = folded(path)
    return [k for k in kinds
            if (_near(fpath, fk := folded(k)) or _near(fk, fpath))
            and (_inside(path, k) or _inside(k, path))]


def _token_link(link: str, folder: str, client: str) -> bool:
    """Whether ``link`` in ``folder``, a folder of ``client``, is a file of
    :data:`TOKEN_FILES` by its name in the folder, such as
    ~/.config/goose/secrets.yaml. A variable such as CLAUDE_CONFIG_DIR can
    move the folder, so the name in the folder is what counts. Each name
    is compared as the folder that holds it compares names, so when
    ``folder`` is on a volume that ignores case, SECRETS.yaml is the file
    that the client opens as secrets.yaml. The volume of the file that the
    link leads to does not count."""
    names = os.path.relpath(link, folder).split("/")
    for rel, owner in CLIENT_PATHS.items():
        if owner != client:
            continue
        for token in TOKEN_FILES:
            want = token[len(rel) + 1:].split("/") if token.startswith(f"{rel}/") else []
            if len(want) == len(names) and all(
                    same_name(name, other, os.path.join(folder, *names[:i]))
                    for i, (name, other) in enumerate(zip(names, want))):
                return True
    return False


def _client_refusal(path: str, home: str, tables: _Tables | None = None,
                    existing: bool = False) -> str | None:
    """How ``path`` meets a folder where a client keeps its settings and
    history on the Mac, as a phrase that follows the path, or None. For the
    real path of a link in such a folder, the phrase names the link.
    ``tables`` is :func:`_tables`, when the caller has it. When ``path``
    holds several such folders, the phrase names each, with its link. A
    link to a sign-in token, from :func:`_token_link`, is secret. A folder
    that ``path`` holds and that does not exist counts only when no folder
    that it holds exists and ``existing`` is false."""
    folders = _client_folders(home, tables)
    hits = _hits(path, folders)
    if not hits:
        return None

    def where(clients: Sequence[str]) -> str:
        if len(clients) == 1:
            return f"where {clients[0]} keeps its settings and history on the Mac"
        return f"where {_and_list(clients)} keep their settings and history on the Mac"

    def one(folder: str, verb: str) -> str:
        client, link, top = folders[folder]
        named = verb if verb == "is" else f"{verb} {_tilde(folder, home)},"
        if link is None or top is None:
            return f"{named} {where([client])}"
        if link == top:
            return _LinkWhy(f"{named} the real path of {_tilde(link, home)}, "
                            f"{where([client])}", link, top, where([client]), own=True)
        return _LinkWhy(f"{named} where the link {_tilde(link, home)} leads, and "
                        f"{_tilde(top, home)} is {where([client])}", link, top, where([client]),
                        secret=[link] if _token_link(link, top, client) else [])

    same = [f for f in hits if _same(path, f)]
    if same:
        return one(same[0], "is")
    outer = [f for f in hits if _inside(path, f)]
    if outer:
        return one(max(outer, key=len), "lies in")
    hits = [f for f in hits if os.path.lexists(f)] or ([] if existing else hits)
    if not hits:
        return None
    if len(hits) == 1:
        return one(hits[0], "holds")
    text = (f"holds {', '.join(_named_hit(f, folders[f][1], folders[f][2], home) for f in hits)}"
            f", {where(list(dict.fromkeys(folders[f][0] for f in hits)))}")
    named = [(client, link, top) for client, link, top in (folders[f] for f in hits)
             if link is not None and top is not None]
    if len(named) < len(hits):
        return text
    first = named[0]
    return _LinkWhy(text, first[1], first[2], where([first[0]]),
                    own=any(link == top for _, link, top in named),
                    links=[link for _, link, _ in named],
                    secret=[link for owner, link, top in named
                            if link != top and _token_link(link, top, owner)])


def _named_hit(hit: str, link: str | None, top: str | None, home: str) -> str:
    """``hit`` for a list of several, with the link that makes it a part of
    a protected folder, or with the protected path that it is the real path
    of."""
    if link is None:
        return _tilde(hit, home)
    if link == top:
        return f"{_tilde(hit, home)} (the real path of {_tilde(link, home)})"
    return f"{_tilde(hit, home)} (where the link {_tilde(link, home)} leads)"


def _sensitive_refusal(path: str, home: str, copy: bool = False,
                       tables: _Tables | None = None) -> str | None:
    """How ``path`` meets the folders and files that hold credentials, gmlx's
    own data, files the Mac runs or commands the Mac runs, as a phrase that
    follows the path, or None. It names only the kinds that apply. For a
    ``copy``, such as a seed, the settings in :data:`COMMAND_PATHS` do not
    count: the client can change only its copy. For the real path of a link
    in such a folder, the phrase names the link and its folder. ``tables``
    is :func:`_tables`, when the caller has it. A link in a folder of
    :data:`PROGRAM_PATHS` does not count here."""
    skip = (_PROGRAM_LINK, _COMMANDS) if copy else (_PROGRAM_LINK,)
    kinds = _sensitive_kinds(home, tables)
    hits = [h for h in _hits(path, kinds) if kinds[h][0] not in skip]
    if not hits:
        return None

    def one(hit: str, verb: str) -> str:
        """The phrase for one hit, where ``verb`` is "is", "lies in" or
        "holds"."""
        what, link, top = kinds[hit]
        named = verb if verb == "is" else f"{verb} {_tilde(hit, home)},"
        if link is None or top is None:
            return f"holds {what}" if verb == "is" else f"{named} which holds {what}"
        if link == top:
            return _LinkWhy(f"{named} the real path of {_tilde(link, home)}, which holds {what}",
                            link, top, f"which holds {what}", own=True)
        return _LinkWhy(f"{named} where the link {_tilde(link, home)} leads, and "
                        f"{_tilde(top, home)} holds {what}", link, top, f"which holds {what}")

    same = [h for h in hits if _same(path, h)]
    if same:
        return one(same[0], "is")
    outer = [h for h in hits if _inside(path, h)]
    if outer:
        return one(max(outer, key=len), "lies in")
    # A path that does not exist still counts, since the client could make
    # it, but the phrase names the ones that you have when there are any.
    hits = [h for h in hits if os.path.lexists(h)] or hits
    if len(hits) == 1:
        return one(hits[0], "holds")
    what = list(dict.fromkeys(kinds[h][0] for h in hits))
    text = (f"holds {', '.join(_named_hit(h, kinds[h][1], kinds[h][2], home) for h in hits)}, "
            f"which hold {_and_list(what)}")
    named = [(kind, link, top) for kind, link, top in (kinds[h] for h in hits)
             if link is not None and top is not None]
    if len(named) < len(hits):
        return text
    first = named[0]
    return _LinkWhy(text, first[1], first[2], f"which holds {first[0]}",
                    own=any(link == top for _, link, top in named),
                    links=[link for _, link, _ in named],
                    secret=[link for kind, link, _ in named if kind == _CREDENTIALS])


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
    tables = _tables(home)
    why = _table_refusal(real, home, tables) or _temp_tree_relation(real)
    if why is not None:
        can = "read" if readonly else "read and change"
        plan_warnings.append(f"[launch] warning: the share {shown} {why}. The client can "
                             f"{can} every file in it.")
    link = None if readonly or why is not None else _link_refusal(real, home, tables)
    if link is not None:
        plan_warnings.append(f"[launch] warning: the share {shown} {link}. The client can "
                             "change where the link leads, so that the Mac reads the client's "
                             f"files in place of yours. To prevent this, share {shown} "
                             "read-only.")
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
        reach = _image_target(target)
        for path in dict.fromkeys([target, reach]):
            lead = target if path == target else f"{target}, which leads to {path} in most images"
            for reserved, what in RESERVED_TARGETS.items():
                if path == reserved:
                    where = f"{lead}, {what}"
                elif _inside(path, reserved):
                    where = f"{lead}, inside {reserved}, {what}"
                elif _inside(reserved, path):
                    where = (f"{target}, which would cover {reserved}, {what}" if path == target
                             else f"{lead} and would cover {reserved}, {what}")
                else:
                    continue
                raise SettingsError(f"{_label(m)} cannot use {where}. Choose another path in "
                                    "the container.")
        if m.kind != "volume":
            check_mount_chars(m.source, "the folder")
        check_mount_chars(target, "the container path")
        other = by_target.get(reach)
        if other is not None:
            if other.target == target:
                clash = f"{_label(other)} and {_label(m)} both use {target} in the container"
            else:
                # Name each path as written, so that the one written in
                # the /var/run form is found.
                clash = ", and ".join(
                    f"{_label(x)} uses {x.target}"
                    + (f", which leads to {reach} in most images" if x.target != reach else "")
                    for x in (other, m))
            raise SettingsError(f"{clash}. Give one of them another path.")
        by_target[reach] = m
        out.append(m)
    return sorted(out, key=lambda m: (_image_target(m.target).rstrip("/").count("/"),
                                      _image_target(m.target)))


def _image_target(target: str) -> str:
    """Where the normal-form ``target`` leads in most images, through the
    links in :data:`IMAGE_LINKS`. Apple container follows such a link when
    it mounts a share, so ``/var/run`` covers ``/run``."""
    for link, dest in IMAGE_LINKS.items():
        if target == link or target.startswith(link + "/"):
            return dest + target[len(link):]
    return target


def _label(m: Mount) -> str:
    if m.kind == "volume":
        return f"the volume {m.source}"
    if m.kind == "home":
        return "the private home"
    return _tilde(m.source)


def holding_share(host_path: str, mounts: list[Mount]) -> Mount | None:
    """The share that holds ``host_path``, the one with the longest matching
    source when several do, or None."""
    best = None
    for m in mounts:
        if m.kind in ("share", "git") and _inside(host_path, m.source):
            if best is None or len(m.source) > len(best.source):
                best = m
    return best


def guest_path(host_path: str, mounts: list[Mount]) -> str | None:
    """Where ``host_path`` appears in the guest, through the share with the
    longest matching source, or None when no share holds it."""
    best = holding_share(host_path, mounts)
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
        proc = subprocess.run([_git_program(), "-c", "core.fsmonitor=false", "-C", cwd, *args],
                              stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              timeout=5, env=_system_env())
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.splitlines() if proc.returncode == 0 else None


def git_extra_mount(cwd: str, shares: list[Mount], home: str | None = None
                    ) -> tuple[Mount | None, list[str]]:
    """The git folder a linked worktree needs, when no share covers it, and
    the notes launch prints about git. A read-write ``--mount`` of exactly
    that git folder comes back as the git mount for this worktree, when it
    passes the checks other than the one for a forged record, so that the
    next launch takes the pair from the share history. Launch prints no
    note for a git folder that a share covers."""
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
    # The user's own read-write share of exactly the git folder vouches for
    # the records that make this a worktree of it.
    vouched = next((m for m in shares if m.kind == "share" and not m.readonly
                    and m.source == common and normal_guest_target(m.target) == common), None)
    if covered(common) and vouched is None:
        return None, []

    def note(line: str) -> tuple[None, list[str]]:
        return None, ([] if covered(common) else [line])
    # git finds the git folder through files in the share, which the guest
    # can change: the .git file, and a commondir file in a .git folder. So
    # an outside git folder is shared only when it names this project back,
    # from a file outside the share.
    # A folder the guest could write in an earlier, wider launch can be
    # made to look like a git folder that names this project back. Only a
    # folder with a git folder's own name counts, and never one that holds
    # the project.
    if not _git_folder_shape(common) or _inside(toplevel, common):
        return note(f"[launch] git in the container cannot use {_tilde(common, home)} as "
                    f"a git folder, because it is not named like one (.git, a name "
                    f"ending in .git, .bare, or a folder in .git/modules) or it holds "
                    f"{_tilde(toplevel, home)}. Share it with --mount "
                    f"{_tilde(common, home)} if you intend to.")
    back = _git_back_reference(toplevel, git_dir, common)
    if back is None:
        return note(f"[launch] git in the container cannot use the git folder "
                    f"{_tilde(common, home)}, because it does not name "
                    f"{_tilde(toplevel, home)} as one of its worktrees or submodules. "
                    f"Share it with --mount {_tilde(common, home)} if you intend to.")
    what, exact = back
    if not exact:
        fix = ("run git worktree repair there" if what == "worktree" else
               f"set core.worktree in {_tilde(os.path.join(git_dir, 'config'), home)} "
               "to its real path")
        return note(f"[launch] git in the container cannot use the git folder "
                    f"{_tilde(common, home)}, because it names "
                    f"{_tilde(toplevel, home)} only through a symbolic link, so launch "
                    f"does not share it. If {_tilde(toplevel, home)} is a {what} of that "
                    f"repository, {fix}, and the next launch shares the git folder.")
    # Its hooks and config run on the Mac the next time you use git there,
    # so it gets the same checks as the current folder, and so does the main
    # worktree above a .git folder, such as a dotfiles repository at $HOME.
    owner = os.path.dirname(common) if os.path.basename(common) == ".git" else None
    for path in filter(None, (common, owner)):
        why = auto_share_refusal(path, home)
        if why is not None:
            return note(f"[launch] git in the container cannot reach this repository's "
                        f"git folder {_tilde(common, home)}, because "
                        f"{_tilde(path, home)} {why}. Use git on the Mac for this "
                        "repository.")
    if vouched is not None:
        return replace(vouched, kind="git", note=f"the git folder of this {what} of "
                                                 f"{_tilde(_git_repository(common), home)}",
                       worktree=toplevel), []
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
                      f"{_tilde(common, home)} if you intend to. Later launches from "
                      f"{_tilde(toplevel, home)} then share it too."]
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


def launch_targets_on_disk() -> list[str]:
    """The client keys, then every agent key that has a folder under the
    launch data folder, so the state of an agent that is no longer
    configured still counts."""
    from gmlx.config import LAUNCH_CLIENTS

    try:
        names = sorted(os.listdir(data_path()))
    except OSError:
        names = []
    return [*LAUNCH_CLIENTS, *(n for n in names if agent_name(n) is not None
                               and (data_path() / n).is_dir())]


def drop_empty_target(client: str) -> None:
    """Remove an agent's folder under the launch data folder when no
    project is left in it, so that launch_targets_on_disk stops listing an
    agent whose last home is gone. A folder that holds anything stays."""
    if agent_name(client) is None:
        return
    root = data_path() / client
    for folder in (root / "projects", root):
        try:
            folder.rmdir()
        except OSError:
            return


def _project_folders(project: str) -> list[tuple[str, str | None]]:
    """``(target key, folder)`` for each launch target with a project folder
    of this id, with the folder its project.json names, or None when the
    record is missing or names none."""
    out = []
    for client in launch_targets_on_disk():
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
    if why is None:
        return
    shown = _tilde(cwd_real, home)
    step = "Launch from a project folder, or pass --no-mount-cwd."
    # A project folder that a link in a protected folder leads to stays a
    # project folder, so the step names the link and a read-only share. A
    # read-only share still gives the client the credentials that a link
    # leads to, and removing such a link stops the tool on the Mac.
    if isinstance(why, _LinkOnWay):
        step = ("The client could change where the link leads. To share it read-only, pass "
                f"--no-mount-cwd --mount {shown}:ro.")
    elif isinstance(why, _LinkWhy) and why.secret:
        secret = [_tilde(link, home) for link in why.secret]
        step = (f"A read-only share also lets the client read what {_links_named(secret)} "
                f"{'leads' if len(secret) == 1 else 'lead'} to. {step}")
    elif isinstance(why, _LinkWhy) and not why.own:
        links = [_tilde(link, home) for link in why.links]
        step = (f"To share it read-only, pass --no-mount-cwd --mount {shown}:ro, or remove "
                f"the {'link' if len(links) == 1 else 'links'} {_links_named(links)}.")
    raise SettingsError(f"will not share the current folder {shown}, because it {why}. {step}")


def _links_named(links: Sequence[str]) -> str:
    """The first :data:`LINKS_NAMED` of ``links`` and how many more there
    are, such as ``a, b, c and 2 more``, or all of them when there are no
    more than that."""
    more = len(links) - LINKS_NAMED
    if more > 0:
        return f"{', '.join(links[:LINKS_NAMED])} and {more} more"
    return _and_list(links)


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
    return (f"[launch] {target_label(client)} keeps its own history{scope} in the container, "
            "starting "
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
    """Every private home under the launch data folder, newest use first,
    keyed by target key, so an agent's home is under ``agent-<name>``."""
    out = []
    for client in launch_targets_on_disk():
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


def _resolution_paths(path: str, resolver: _Resolver | None = None) -> list[str]:
    """Each path that resolving the absolute ``path`` visits, every link
    among them, in the form macOS gives it. A link that a client can change
    anywhere on the way changes where the path leads. ``resolver`` keeps
    the links and the folders it reads for a build of the tables."""
    readlink = os.readlink if resolver is None else resolver.readlink
    folder = _real if resolver is None else resolver.folder
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
            target = readlink(here)
        except OSError:
            done = here
            continue
        links += 1
        if links > 32:              # the limit macOS sets, so the stat failed
            break
        if target.startswith("/"):
            done = "/"
        todo[:0] = [c for c in target.split("/") if c]
    return [os.path.join(folder(os.path.dirname(p)), os.path.basename(p)) for p in visited]


def _link_in(folder: str, path: str, reached: Sequence[str] | None = None) -> str | None:
    """A path in ``folder`` that the absolute ``path`` leads through, as
    written or while it resolves, a link first, or None. A client that
    writes ``folder`` can change such a link, and with it where ``path``
    leads. ``reached`` is :func:`_reach` of ``path``, when the caller has
    it for several folders."""
    hits = [p for p in (_reach(path) if reached is None else reached) if _inside(p, folder)]
    return next((p for p in hits if os.path.islink(p)), hits[0] if hits else None)


def _reach(path: str) -> list[str]:
    """``path`` and each path that resolving it visits, once each."""
    return list(dict.fromkeys([path, *_resolution_paths(path)]))


def _share_reach(path: str, shares: list[Mount], home: str) -> tuple[str, Mount] | None:
    """How the absolute ``path`` reaches a read-write share in ``shares``,
    as a phrase that follows the path, with that share, or None: it lies in
    the share as written or resolved, or it leads through a link in the
    share that the client can change."""
    for m in shares:
        if _inside(path, m.source) or _inside(_real(path), m.source):
            return f"lies in the read-write share {_tilde(m.source, home)}", m
    for m in shares:
        link = _link_in(m.source, path)
        if link is not None:
            return (f"leads through {_tilde(link, home)} in the read-write share "
                    f"{_tilde(m.source, home)}"), m
    return None


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
        raise SettingsError(f"launch found the container command at {_tilde(path, home)}, "
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
                 project_volumes: Sequence[str] = (),
                 source: str | None = None, runtime: bool = False) -> ContainerPlan:
    """The mounts, volumes and ports of one session, from the effective
    client config and the flags. ``build_folders`` maps each client to its
    configured ``build:`` path, and no read-write share may overlap one.
    The private home is the one of ``project``, which the plan names
    without creating it, and each volume entry in ``project_volumes`` gets
    that project's name. ``runtime`` marks an agent whose dependencies uv
    installs from ``source``, or from the current folder when ``source`` is
    None. A source that no share holds is shared read-only at its own path,
    and no share may use the dependency folder."""
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
    source_real = cwd_real
    if runtime and source is not None:
        # The source goes through the share rules, so a link or a sensitive
        # folder gets the same answer as a mounts: entry.
        source_mount = _explicit_mount(f"{source}:ro", warns, home)
        source_real = source_mount.source
        if holding_share(source_real, mounts) is None:
            mounts.append(replace(source_mount, note="source folder"))
    # Before git runs, so a git that a share let the client put in its
    # place never runs. The git folder gets the check below.
    _refuse_program_shares(mounts, home)
    git_mount, git_notes = (git_extra_mount(cwd_real, list(mounts), home)
                            if share_cwd else (None, []))
    if git_mount is not None:
        # In place of the explicit share of the git folder, when it is one.
        # Another share of the folder, such as one at another guest path,
        # stays.
        same = (git_mount.source, normal_guest_target(git_mount.target), git_mount.readonly)
        mounts = [m for m in mounts if not (
            m.kind == "share" and (m.source, normal_guest_target(m.target), m.readonly) == same)]
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
    _refuse_state_links(mounts, home)
    _refuse_program_shares(mounts, home)
    warns.extend(_package_warnings(mounts, home))
    warns.extend(_path_warnings(mounts, home))
    warns.extend(_program_link_warnings(mounts, home))
    guest_cwd = guest_path(cwd_real, mounts)
    warns.extend(protected_folder_warnings(mounts, home))
    source_guest, source_readonly = None, False
    if runtime:
        _refuse_deps_folder_shares(mounts)
        holder = holding_share(source_real, mounts)
        if holder is None:
            raise SettingsError(
                "the current folder is not shared, so uv has no project to install. Launch "
                f"with --mount-cwd, or set {config_key(client, 'source')} to the project "
                "folder.")
        source_guest, source_readonly = guest_path(source_real, mounts), holder.readonly
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
        new_home=new_home, ssh_socket=ssh_socket, source_guest=source_guest,
        source_readonly=source_readonly)


def _refuse_deps_folder_shares(mounts: list[Mount]) -> None:
    """A runtime agent's environment lives on the volume at the dependency
    folder, so no share may be there or inside it. A volume at the folder is
    the environment itself, and a share over the folder is refused already,
    since it would cover /opt/gmlx too."""
    what = "where uv keeps the agent's environment"
    for m in mounts:
        if m.kind == "volume":
            continue
        if m.target == AGENT_DEPS_TARGET:
            where = f"{m.target}, {what}"
        elif _inside(m.target, AGENT_DEPS_TARGET):
            where = f"{m.target}, inside {AGENT_DEPS_TARGET}, {what}"
        else:
            continue
        raise SettingsError(f"{_label(m)} cannot use {where}. Choose another path in the "
                            "container.")


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
    has network access. So does a share that holds a link on the way to
    it, because the build reads the path as written."""
    for client, build in sorted(build_folders.items()):
        folder = build_folder(build) if build else None
        if folder is None:
            continue
        written = os.path.abspath(os.path.expanduser(build))
        for m in mounts:
            if m.readonly or m.kind not in ("share", "git"):
                continue
            if _inside(m.source, folder) or _inside(folder, m.source):
                raise SettingsError(
                    f"will not share {_tilde(m.source, home)} read-write, because the client "
                    f"could change the {target_label(client)} build: folder "
                    f"{_tilde(folder, home)}.\n"
                    f"  Share it read-only with --mount {_tilde(m.source, home)}:ro, or move "
                    "the build folder.")
            link = _link_in(m.source, written)
            if link is not None:
                raise SettingsError(
                    f"will not share {_tilde(m.source, home)} read-write, because the "
                    f"{target_label(client)} build: path leads through {_tilde(link, home)}, "
                    "and the client could change where it leads.\n"
                    f"  Share it read-only with --mount {_tilde(m.source, home)}:ro, or set "
                    "build: to a path that does not go through the folder.")


_ENV_STEPS = ("move the Python environment out of the folder",
              "run gmlx by a path that does not go through the folder")


def _python_folders() -> list[tuple[str, str, tuple[str | None, str | None]]]:
    """The paths of the Python that gmlx runs from, and of the gmlx program
    that runs it, as written. Each comes with what it is, and with the step
    besides a read-only share for a share that holds it and for a share
    that holds a link on the way to it. The Mac runs the code in them at
    the next gmlx command, and in a server that launch or launchd starts.
    The paths that earlier gmlx commands recorded, such as the Python of
    the launchd agents, come from :func:`_recorded_pythons`."""
    import shutil
    import site
    import sys

    import gmlx
    from gmlx.serve.procname import _app_dir, _proc_dir, stable_executable

    env = "the Python environment that gmlx runs from"
    exe = stable_executable()
    out: list[tuple[str, str, tuple[str | None, str | None]]] = [
        (sys.prefix, env, _ENV_STEPS), (sys.exec_prefix, env, _ENV_STEPS),
        (os.path.dirname(exe), "the folder of the Python that gmlx runs", _ENV_STEPS),
        (exe, "the Python that gmlx runs", _ENV_STEPS)]
    if site.ENABLE_USER_SITE:
        out.append((site.getusersitepackages(), "your user site-packages folder, which "
                                                "gmlx imports", _ENV_STEPS))
    # A console script names the real Python on its first line, so a script
    # run through a linked .venv puts no linked path in sys.prefix. The path
    # it was run by is in sys.argv[0]. With python -m gmlx, that is a file in
    # the gmlx package, which only warns.
    script = sys.argv[0] if sys.argv else ""
    package = os.path.dirname(gmlx.__file__)
    if script not in ("", "-c", "-m") and os.path.isfile(script):
        script = os.path.abspath(script)
        if not (_inside(script, package) or _inside(_real(script), _real(package))):
            out.append((script, "the gmlx program that you ran",
                        (_ENV_STEPS[1], _ENV_STEPS[1])))
    found = shutil.which("gmlx")
    if found and os.path.isabs(found):
        drop = f"remove {_tilde(os.path.dirname(found))} from PATH"
        out.append((found, "the gmlx program that PATH finds", (drop, drop)))
    # The copies of the Python that the launchd agents, the menu bar and the
    # server run as. gmlx copies one again only when its stamp beside it
    # changes, and a client that can write the folder keeps the stamp.
    out.append((str(_app_dir()), "the folder of the gmlx app, whose Python and agent "
                                 "script the launchd agents and the menu bar run", (None, None)))
    out.append((str(_proc_dir()), "the folder of the copy of Python that the gmlx server "
                                  "runs as", (None, None)))
    return list(dict.fromkeys((os.path.abspath(p), what, steps) for p, what, steps in out))


# The line of the agent script in the gmlx app that names the Python it
# runs when the copy of Python beside it does not start.
_AGENT_PYTHON = re.compile(r'^PY="([^"\n]+)"$', re.MULTILINE)


def _target_flags(host: str, port) -> str:
    """The ``--host`` and ``--port`` flags of a gmlx command that acts on
    the server at ``host`` and ``port``, empty for the default server."""
    return (("" if host == "127.0.0.1" else f" --host {host}")
            + ("" if str(port) == "8080" else f" --port {port}"))


def _agent_script_pythons(path: str) -> list[str]:
    """The Python that the agent script at ``path`` names on its ``PY=``
    line, or an empty list when ``path`` is not such a script."""
    try:
        data = _read_small_file(path)
    except OSError:
        return []
    if not data.startswith(b"#!"):
        return []
    return [p for p in _AGENT_PYTHON.findall(data.decode(errors="replace"))
            if os.path.isabs(p)]


def _recorded_pythons() -> list[tuple[str, str, tuple[str | None, str | None]]]:
    """The paths of Python and of gmlx that earlier gmlx commands recorded,
    as written, in the form that :func:`_python_folders` gives. They are
    the program and the first PATH folder of each gmlx login agent, the
    Python that the agent script in the gmlx app names, the program of each
    server record, which gmlx restart runs, and the program of the menu
    bar's server autostart. The Mac runs them at login and at a restart,
    also when this gmlx command runs from another path."""
    import plistlib

    out: list[tuple[str, str, tuple[str | None, str | None]]] = []

    def steps(command: str) -> tuple[str, str]:
        return (f"{command} from a gmlx outside the folder",
                f"{command} by a path that does not go through the folder")

    agents = os.path.expanduser("~/Library/LaunchAgents")
    try:
        plists = sorted(n for n in os.listdir(agents)
                        if n.startswith("com.gmlx.") and n.endswith(".plist"))
    except OSError:
        plists = []
    for name in plists:
        try:
            doc = plistlib.loads(_read_small_file(os.path.join(agents, name)))
        except Exception:  # noqa: BLE001 - launchd does not run a plist that does not parse
            continue
        if not isinstance(doc, dict) or not isinstance(doc.get("ProgramArguments"), list):
            continue
        args = [str(a) for a in doc["ProgramArguments"]]
        label = name[:-len(".plist")]
        if label == "com.gmlx.commands.menubar":
            command = "run gmlx service install again"
        else:
            host = args[args.index("--host") + 1] if "--host" in args[:-1] else "127.0.0.1"
            port = args[args.index("--port") + 1] if "--port" in args[:-1] else "8080"
            command = f"run gmlx service install --headless{_target_flags(host, port)} again"
        agent = f"the gmlx login agent {label}"
        if args and os.path.isabs(args[0]):
            out.append((args[0], f"the program that {agent} runs", steps(command)))
            out += [(py, f"the Python that {agent} runs", steps(command))
                    for py in _agent_script_pythons(args[0])]
        env = doc.get("EnvironmentVariables")
        first = str(env.get("PATH") or "").split(os.pathsep)[0] if isinstance(env, dict) else ""
        if os.path.isabs(first):
            out.append((first, f"the first folder in the PATH of {agent}", steps(command)))
    cache = os.path.join(os.path.expanduser(os.environ.get("XDG_CACHE_HOME") or "~/.cache"),
                         "gmlx")
    try:
        runs = sorted(n for n in os.listdir(cache)
                      if n.startswith("run-") and n.endswith(".json"))
    except OSError:
        runs = []
    for name in runs:
        run = _read_json_record(Path(cache, name))
        argv = run.get("argv")
        # A launchd agent's record names the agent's program, which its
        # plist gives above.
        if run.get("managed_by") == "launchd" or not isinstance(argv, list) or not argv:
            continue
        flags = _target_flags(str(run.get("host") or "127.0.0.1"), run.get("port") or 8080)
        if isinstance(argv[0], str) and os.path.isabs(argv[0]):
            out.append((argv[0], f"the Python that gmlx restart{flags} runs",
                        steps(f"stop the server with gmlx stop{flags}, and start it again")))
    auto = _read_json_record(Path(cache, "menubar-settings.json")).get("autostart")
    argv = auto.get("argv") if isinstance(auto, dict) else None
    if isinstance(auto, dict) and isinstance(argv, list) and argv \
            and isinstance(argv[0], str) and os.path.isabs(argv[0]):
        flags = _target_flags(str(auto.get("host") or "127.0.0.1"), auto.get("port") or 8080)
        out.append((argv[0], "the Python that the menu bar's server autostart runs",
                    steps(f"stop the server with gmlx stop{flags}, and run gmlx service "
                          f"install{flags} again")))
    return list(dict.fromkeys((os.path.abspath(p), what, s) for p, what, s in out))


def _refuse_python_shares(mounts: list[Mount], home: str) -> None:
    """A read-write share that holds or lies in gmlx's Python environment
    lets the client change code that the Mac runs, such as a ``.pth`` file
    in site-packages. So does a share that holds a link on the way to it,
    such as a project's ``.venv`` that leads to another folder, because the
    client can point the link at an environment of its own. The gmlx
    program that you ran, and the one that PATH finds, get the same check,
    and so do the paths of Python that earlier gmlx commands recorded."""
    folders = list(dict.fromkeys([*_python_folders(), *_recorded_pythons()]))
    for m in mounts:
        if m.readonly or m.kind not in ("share", "git"):
            continue
        shown = _tilde(m.source, home)
        for path, what, (held, linked) in folders:
            real = _real(path)
            if _inside(m.source, real) or _inside(real, m.source):
                raise SettingsError(
                    f"will not share {shown} read-write, because it "
                    f"{_relation(m.source, real, home, what)}. The client could change "
                    "code that the Mac runs.\n"
                    f"  Share it read-only with --mount {shown}:ro"
                    f"{f', or {held}' if held else ''}.")
            link = _link_in(m.source, path)
            if link is not None:
                verb = "is" if _same(link, m.source) else "holds"
                raise SettingsError(
                    f"will not share {shown} read-write, because it {verb} "
                    f"{_tilde(link, home)}, which leads to {what}, {_tilde(real, home)}. "
                    "The client could change where it leads, and the Mac would run the "
                    "client's code.\n"
                    f"  Share it read-only with --mount {shown}:ro"
                    f"{f', or {linked}' if linked else ''}.")


def _system_program(name: str) -> str | None:
    """The path of the program ``name`` that a run by name with
    :data:`SYSTEM_PATH` finds, or None."""
    return next((p for p in (os.path.join(f, name) for f in SYSTEM_PATH.split(os.pathsep))
                 if os.path.isfile(p) and os.access(p, os.X_OK)), None)


def _developer_folder() -> str | None:
    """The active developer folder, as xcode-select names it with
    :func:`_system_env`: the folder that DEVELOPER_DIR or ``xcode-select
    --switch`` chose, or the default. The path is as written, with its
    links. None when xcode-select names none. In :func:`launch_memo`, the
    first answer serves the block, so one launch runs xcode-select once."""
    memo = _launch_memo_here()
    key = _environment()
    if memo is not None and key in memo.developer:
        return memo.developer[key]
    folder = _xcode_select_folder()
    if memo is not None:
        memo.developer[key] = folder
    return folder


def _xcode_select_folder() -> str | None:
    try:
        proc = subprocess.run([XCODE_SELECT, "-p"], stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=5, env=_system_env())
    except (OSError, subprocess.TimeoutExpired):
        return None
    folder = proc.stdout.strip()
    return folder if proc.returncode == 0 and os.path.isabs(folder) else None


def _launch_program(name: str) -> tuple[str | None, str | None, str | None]:
    """The program ``name`` that :data:`SYSTEM_PATH` finds, the program that
    launch runs for it, and the developer folder of that program. For a
    shim in :data:`DEVELOPER_SHIMS`, launch runs the program of the active
    developer folder, which is None when xcode-select names no folder. For
    another program, launch runs the one it finds, and there is no folder."""
    found = _system_program(name)
    rel = DEVELOPER_SHIMS.get(found or "")
    if rel is None:
        return found, found, None
    folder = _developer_folder()
    return found, (os.path.join(folder, rel) if folder is not None else None), folder


def _git_program() -> str:
    """The git that launch runs, from :func:`_launch_program`. Raises
    FileNotFoundError when there is none, as a run of a missing program
    does."""
    git = _launch_program("git")[1]
    if git is None:
        raise FileNotFoundError("launch found no git to run")
    return git


def _refuse_program_shares(mounts: list[Mount], home: str) -> None:
    """A read-write share that holds git or ssh-add as launch finds them in
    :data:`SYSTEM_PATH`, or a link on the way to them, lets the client
    replace a program that launch runs on the Mac. So does a share that
    holds a folder that the search looks in first, or a link on the way to
    one, such as a share of /opt/homebrew on a Mac that runs /usr/bin/git:
    a git that the client puts there runs in place of /usr/bin/git. For
    /usr/bin/git, launch runs the git of the developer folder, so a share
    that holds or lies in that folder, or holds a link on the way to it, is
    refused too. A program that launch finds off the sealed volume, such as
    Homebrew's git, loads libraries and reads settings from the
    installation that holds its search folder, such as /opt/homebrew/opt
    and /opt/homebrew/etc. So a share that holds or lies in that
    installation, or holds a link on the way to it, is refused. A folder an
    earlier session shared read-write, and the private homes, get the check
    in :func:`_refuse_program_history`, :func:`_refuse_developer_history`
    and :func:`_refuse_installation_history`."""
    folders = [f for f in SYSTEM_PATH.split(os.pathsep) if os.path.isabs(f)]
    # Each path, what it is, and whether a share that lies in it is refused.
    checks: list[tuple[str, str, bool]] = []
    for name in ("git", "ssh-add"):
        found, path, developer = _launch_program(name)
        _refuse_program_history(name, found, folders, home)
        if found is not None and developer is not None:
            _refuse_developer_history(name, found, developer, home)
        installation = _installation(found)
        if found is not None and installation is not None:
            _refuse_installation_history(name, found, installation, home)
        # No client can change a program in SEALED_PATH, so neither such a
        # program nor such a folder is a reason to refuse a share.
        if path is not None and os.path.dirname(path) not in SEALED_PATH:
            checks.append((path, f"the {name} that launch runs on the Mac. The client could "
                                 "replace it", False))
        if found is not None and developer is not None:
            checks.append((developer, f"the developer folder that launch runs {name} from in "
                                      f"place of {_tilde(found, home)}. The client could "
                                      f"change that {name} or the files it reads", True))
        if found is not None and installation is not None:
            checks.append((installation, f"the installation that {_tilde(found, home)} comes "
                                         "from. The client could change the libraries and "
                                         f"settings that this {name} loads from it", True))
        before = f" before {_tilde(found, home)}" if found is not None else ""
        for folder in folders:
            if os.path.join(folder, name) == found:
                break
            if folder in SEALED_PATH:
                continue
            checks.append((folder, f"where launch looks for {name}{before}. The client could "
                                   f"put its own {name} there, which launch would run on the "
                                   "Mac", False))
    for m in mounts:
        if m.readonly or m.kind not in ("share", "git"):
            continue
        shown = _tilde(m.source, home)
        for path, what, within in checks:
            real = _real(path)
            # The share that is the path itself goes unnamed in the phrase.
            if _same(real, m.source):
                where = f"is {what}"
            elif _inside(real, m.source):
                where = f"holds {_tilde(real, home)}, {what}"
            elif within and _inside(m.source, real):
                where = f"lies in {_tilde(real, home)}, {what}"
            elif (link := _link_in(m.source, path)) is not None:
                where = (f"{'is' if _same(link, m.source) else 'holds'} {_tilde(link, home)}, "
                         f"which leads to {_tilde(real, home)}, {what}")
            else:
                continue
            raise SettingsError(f"will not share {shown} read-write, because it {where}.\n"
                                f"  Share it read-only with --mount {shown}:ro.")


def _program_history_refusal(path: str, home: str) -> tuple[str, str] | None:
    """Why launch will not run a program by name that it looks for at
    ``path``, as a phrase that follows the path, with the step that clears
    it, or None. A client could have left a file at ``path`` in a folder an
    earlier session shared read-write or in the private homes, or a link on
    the way to ``path`` that leads to a place it can write later."""
    real = _real(path)
    if os.path.lexists(path):
        why = _agent_refusal(path, real, (), home)
        return (why, f"Remove {_tilde(path, home)}") if why is not None else None
    # Nothing is at the path, so only a link on the way can lead to a
    # program that a client writes later.
    visited = _resolution_paths(path)
    data = _real(data_path())
    if any(_inside(p, data) for p in [*visited, real]):
        link = next((p for p in visited if os.path.islink(p)), path)
        return (f"leads through {_tilde(link, home)} to {_tilde(data, home)}, where launch "
                "keeps the private homes of the clients", f"Remove the link {_tilde(link, home)}")
    for folder in shared_history():
        if any(_inside(p, folder) for p in visited) and not _inside(real, folder):
            link = next((p for p in visited if _inside(p, folder) and os.path.islink(p)), path)
            return (f"leads through {_tilde(link, home)} in {_tilde(folder, home)}, a folder "
                    f"an earlier session shared read-write, to {_tilde(real, home)}",
                    f"Remove the link {_tilde(link, home)}")
    return None


def _refuse_program_history(name: str, found: str | None, folders: list[str],
                            home: str) -> None:
    """Refuse to run ``name`` when a client could have put its own program
    where launch looks for it in ``folders``, up to the ``found`` one: in a
    folder an earlier session shared read-write or in the private homes, or
    through a link that leads to such a place."""
    for folder in folders:
        path = os.path.join(folder, name)
        hit = None if folder in SEALED_PATH else _program_history_refusal(path, home)
        if hit is not None:
            why, step = hit
            shown = _tilde(path, home)
            # The reason is about the path, so it follows the path, also
            # when launch found the program in a later folder.
            if path == found:
                first = f"launch found {name} at {shown}, which {why}."
            elif found:
                first = (f"launch looks for {name} at {shown} before it looks at "
                         f"{_tilde(found, home)}. {shown} {why}.")
            else:
                first = f"launch looks for {name} at {shown}, which {why}."
            raise SettingsError(f"{first} A client could have put its own {name} there, and "
                                f"launch would run it on the Mac.\n  {step}, and launch again.")
        if path == found:
            return


def _installation(found: str | None) -> str | None:
    """The installation that the program ``found`` comes from: the folder
    that holds its folder of :data:`SYSTEM_PATH`, such as /opt/homebrew
    for /opt/homebrew/bin/git. None for no program, or for a program on the
    sealed volume."""
    if found is None or os.path.dirname(found) in SEALED_PATH:
        return None
    return os.path.dirname(os.path.dirname(found))


def _folder_history_refusal(folder: str, home: str,
                            unread: Sequence[str] = ()) -> str | None:
    """How ``folder`` meets a place a client could write, as a phrase that
    follows the folder, or None. A program that launch runs reads its files
    in ``folder``. The folder lies in or holds a folder an earlier session
    shared read-write, or it lies in the private homes, or a link on the way
    to it does. An earlier share in one of the ``unread`` folders does not
    count, because the program reads no file there."""
    real = _real(folder)
    why = _agent_refusal(folder, real, (), home)
    held = next((f for f in shared_history() if _inside(f, real)
                 and not any(_inside(f, u) for u in unread)), None)
    if why is None and held is not None:
        why = f"holds {_tilde(held, home)}, a folder an earlier session shared read-write"
    return why


def _refuse_installation_history(name: str, found: str, folder: str, home: str) -> None:
    """Refuse to run ``name`` from ``found`` when a client could have
    changed the installation ``folder`` that it loads its libraries and
    settings from, as :func:`_folder_history_refusal` finds. The folders
    of :data:`INSTALLATION_UNREAD` do not count, unless ``found`` leads
    through one."""
    real = _real(folder)
    reached = [found, *_resolution_paths(found), _real(found)]
    unread = [p for p in (os.path.join(real, sub) for sub in INSTALLATION_UNREAD)
              if not any(_inside(r, p) for r in reached)]
    why = _folder_history_refusal(folder, home, unread)
    if why is not None:
        raise SettingsError(f"launch runs {name} from {_tilde(found, home)}, and its "
                            f"installation {_tilde(folder, home)} {why}. A client could have "
                            f"changed the libraries or settings that this {name} loads from "
                            f"it, and launch would run {name} with them on the Mac.\n"
                            f"  Remove {_tilde(found, home)}, for example with brew uninstall "
                            f"{BREW_FORMULAS.get(name, name)}, so that launch runs another "
                            f"{name}. Then launch again.")


def _refuse_developer_history(name: str, found: str, folder: str, home: str) -> None:
    """Refuse to run ``name`` from the developer ``folder``, in place of the
    shim ``found``, when a client could have changed that folder: it lies in
    or holds a folder an earlier session shared read-write, or it lies in
    the private homes, or a link on the way to it does."""
    why = _folder_history_refusal(folder, home)
    if why is not None:
        raise SettingsError(f"launch runs {name} from the developer folder "
                            f"{_tilde(folder, home)}, which {why}. A client could have "
                            f"changed the {name} there or the files it reads, and launch "
                            "would run it on the Mac.\n"
                            f"  Install {name} with Homebrew, which launch runs in place of "
                            f"{_tilde(found, home)}, or choose other developer tools with "
                            "sudo xcode-select --switch, and launch again.")


def _path_warnings(mounts: list[Mount], home: str) -> list[str]:
    """Warnings for the PATH entries that lead into a read-write share, one
    line for each share. A program that the client puts there runs on the
    Mac in place of a command of that name. When the share holds the
    Python environment that VIRTUAL_ENV names, what the client writes in it
    stays after the session and runs when you use the environment again,
    so the step keeps the environment outside the share. An empty or
    relative entry names the current folder."""
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
    venv = os.environ.get("VIRTUAL_ENV", "").strip()
    venv = os.path.abspath(os.path.expanduser(venv)) if venv else ""
    held: dict[str, list[tuple[str, str]]] = {}
    for entry in dict.fromkeys(e for e in entries if os.path.isabs(e)):
        reach = _share_reach(entry, rw, home)
        if reach is not None:
            held.setdefault(reach[1].source, []).append((entry, reach[0]))
    for source, found in held.items():
        shown = _tilde(source, home)
        names = [_tilde(entry, home) for entry, _ in found]
        if len(found) == 1:
            line = f"PATH holds {names[0]}, which {found[0][1]}."
        elif all(why.startswith("lies in ") for _, why in found):
            line = f"PATH holds {_and_list(names)}, which lie in the read-write share {shown}."
        else:
            line = "PATH holds " + ", and ".join(
                f"{name}, which {why}" for name, (_, why) in zip(names, found)) + "."
        line += (" A program the client puts there runs on the Mac in place of a command of "
                 "that name." if len(found) == 1 else " A program the client puts in one of "
                 "them runs on the Mac in place of a command of that name.")
        env_bin = venv and os.path.join(venv, "bin")
        in_env = [entry for entry, _ in found
                  if env_bin and (_same(entry, env_bin) or _same(_real(entry), _real(env_bin)))]
        if in_env and (_inside(venv, source) or _inside(_real(venv), source)):
            others = len(found) > len(in_env)
            line += (f" The client can also change the Python environment {_tilde(venv, home)} "
                     "that VIRTUAL_ENV names. Such a change stays after the session, and runs "
                     "when you use the environment or activate it again. To prevent this, "
                     "keep the Python environment outside the share"
                     f"{' and remove the other folders from PATH' if others else ''}, or share "
                     f"{shown} read-only.")
        else:
            folders = "the folder" if len(found) == 1 else "these folders"
            line += f" To prevent this, remove {folders} from PATH, or share {shown} read-only."
        out.append(f"[launch] warning: {line}")
    return out


# The most links, or names of programs, that one warning of
# :func:`_program_link_warnings` or one step of :func:`check_cwd_share`
# names. The rest are counted.
LINKS_NAMED = 3


def _program_link_warnings(mounts: list[Mount], home: str,
                           tables: _Tables | None = None) -> list[str]:
    """Warnings for each read-write share that holds the file that a link
    in a folder of :data:`PROGRAM_PATHS` leads to, or a link on the way to
    it. A change that the client makes there runs on the Mac when you run
    the link's name. ``tables`` is :func:`_tables`, when the caller has
    it."""
    rw = [m for m in mounts if m.kind in ("share", "git") and not m.readonly]
    if not rw:
        return []
    tables = _tables(home) if tables is None else tables
    links = list(dict.fromkeys(p for p, what, _ in tables.sensitive if what == _PROGRAM_LINK))
    out = []
    for m in rw:
        # The link, and how the share meets it: a verb and the rest.
        held: list[tuple[str, str, str]] = []
        for link in links:
            real = tables.real[link][1]
            if _inside(real, m.source):
                if _same(real, m.source):
                    held.append((link, "is", f"where the link {_tilde(link, home)} leads"))
                else:
                    held.append((link, "holds", f"{_tilde(real, home)}, where the link "
                                                f"{_tilde(link, home)} leads"))
                continue
            hit = _link_in(m.source, link, [p for _, p in tables.reach[link]])
            if hit is not None and os.path.islink(hit):
                held.append((link, "is" if _same(hit, m.source) else "holds",
                             f"{_tilde(hit, home)}, a link on the way to {_tilde(link, home)}"))
        if not held:
            continue
        shown = _tilde(m.source, home)
        named = [(verb, rest) for _, verb, rest in held[:LINKS_NAMED]]
        more = len(held) - len(named)
        if more:
            named.append(("holds", f"{more} more {'file' if more == 1 else 'files'} where such "
                                   "links lead"))
        names = list(dict.fromkeys(os.path.basename(link) for link, _, _ in held))
        others = len(names) - LINKS_NAMED
        if others > 0:
            listed = (f"{', '.join(names[:LINKS_NAMED])} or {others} other "
                      f"{'program' if others == 1 else 'programs'}")
        else:
            listed = names[0] if len(names) == 1 else f"{', '.join(names[:-1])} or {names[-1]}"
        one = len(held) == 1
        out.append(f"[launch] warning: the share {shown} {_phrase_list(named)}. The client can "
                   f"change the {'program that runs' if one else 'programs that run'} on the "
                   f"Mac when you run {listed}. To prevent this, remove the "
                   f"{'link' if one else 'links'}, or share {shown} read-only.")
    return out


def _phrase_list(parts: Sequence[tuple[str, str]]) -> str:
    """One phrase from ``(verb, rest)`` parts that each name a path: the
    verb once when all parts share it, with a comma before the last ``and``
    because each part holds a comma."""
    verbs = {verb for verb, _ in parts}
    items = ([rest for _, rest in parts] if len(verbs) == 1
             else [f"{verb} {rest}" for verb, rest in parts])
    joined = items[0] if len(items) == 1 else f"{', '.join(items[:-1])}, and {items[-1]}"
    return f"{parts[0][0]} {joined}" if len(verbs) == 1 else joined


def server_path(mounts: Sequence[Mount]) -> str:
    """The PATH for the gmlx server and the menu bar that launch starts.
    The server finds ffmpeg and the MCP tool servers on its PATH, and the
    menu bar starts the server again with its own PATH. The server skips
    the folders that a client can write each time it looks for a program
    (:mod:`gmlx.serve.programs`), and this PATH leaves them out before the
    server starts. It is the PATH of this process without its empty and
    relative entries, and without each entry that lies in or leads through
    a folder that a client can write. Such a folder is a read-write share in
    ``mounts``, a folder that an earlier session shared read-write, or a
    private home. With no entry left, it is
    :data:`SEALED_PATH`, which no client can change, also when a session
    shares the folder."""
    shares = [m.source for m in mounts if not m.readonly and m.kind in ("share", "git")]
    home = _host_home()
    kept = [e for e in os.environ.get("PATH", os.defpath).split(os.pathsep)
            if os.path.isabs(e) and _agent_refusal(e, _real(e), shares, home) is None]
    return os.pathsep.join(kept) or os.pathsep.join(SEALED_PATH)


def _editable_checkouts() -> list[tuple[str, str]]:
    """The name and folder of each package that gmlx's Python environment
    holds as an editable install, from the ``direct_url.json`` file that pip
    and uv write for it. gmlx itself is not in the list, because the
    package warning names it."""
    import json
    from importlib import metadata
    from urllib.parse import unquote, urlsplit

    out: list[tuple[str, str]] = []
    for dist in metadata.distributions():
        # Only an editable install needs its name, so the METADATA file of
        # every other one is never parsed.
        try:
            info = json.loads(dist.read_text("direct_url.json") or "null")
        except Exception:  # noqa: BLE001 - a broken install must never stop the launch
            continue
        if not isinstance(info, dict) or not isinstance(info.get("dir_info"), dict):
            continue
        if info["dir_info"].get("editable") is not True:
            continue
        url = urlsplit(str(info.get("url", "")))
        path = unquote(url.path)
        if not (url.scheme == "file" and url.netloc in ("", "localhost") and os.path.isabs(path)):
            continue
        try:
            name = str(dist.metadata["Name"] or "")
        except Exception:  # noqa: BLE001 - a broken install must never stop the launch
            continue
        if name.lower() != "gmlx":
            out.append((name, path))
    return list(dict.fromkeys(out))


def _package_warnings(mounts: list[Mount], home: str) -> list[str]:
    """Warnings when a read-write share holds or lies in code that gmlx
    runs from outside its Python environment, or holds a link on the way to
    it: the gmlx package folder, as an editable checkout puts it, the Python
    installation that the environment comes from, and the editable checkouts
    of other packages in the environment."""
    import sys

    import gmlx

    rw = [m for m in mounts if m.kind in ("share", "git") and not m.readonly]
    if not rw:
        return []
    paths = [(os.path.dirname(gmlx.__file__), "the gmlx package that the Mac runs",
              "The client can change gmlx's code, which the next gmlx command runs, and the "
              "guest entry and Containerfile that later sessions and builds use.")]
    paths += [(base, "the Python installation that gmlx's environment comes from",
               "The client can change Python and its standard library, which the next gmlx "
               "command runs on the Mac.") for base in (sys.base_prefix, sys.base_exec_prefix)]
    paths += [(folder, f"the editable checkout of {name} in gmlx's Python environment",
               "The client can change code that gmlx's Python can import on the Mac.")
              for name, folder in _editable_checkouts()]
    out = []
    for path, what, then in dict.fromkeys((os.path.abspath(p), w, t) for p, w, t in paths):
        real = _real(path)
        for m in rw:
            if _inside(m.source, real) or _inside(real, m.source):
                where = _relation(m.source, real, home, what)
            else:
                link = _link_in(m.source, path)
                if link is None:
                    continue
                verb = "is" if _same(link, m.source) else "holds"
                where = f"{verb} {_tilde(link, home)}, which leads to {what}, {_tilde(real, home)}"
            shown = _tilde(m.source, home)
            out.append(f"[launch] warning: the share {shown} {where}. {then} To prevent this, "
                       f"share {shown} read-only.")
            break
    return out


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
    return _data_refusal(real, host_home) or _sensitive_refusal(real, host_home, copy=True)


def _seed_refusal(shown: str, src: str, real: str, host_home: str) -> str:
    """The refusal of the seed ``shown`` at ``src``, whose real path is
    ``real``."""
    why = _seed_source_refusal(real, host_home)
    # A seed that is itself the link lies in the protected folder, or is
    # the protected path.
    if isinstance(why, _LinkWhy) and shown == _tilde(why.link, host_home):
        if why.own:
            return (f"seed: will not copy {shown}, because it leads to "
                    f"{_tilde(real, host_home)}, {why.what}.")
        return (f"seed: will not copy {shown}, because it lies in "
                f"{_tilde(why.folder, host_home)}, {why.what}.")
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
    value = subprocess.run([_git_program(), "config", *where, "--get", key],
                           stdin=subprocess.DEVNULL, capture_output=True, text=True,
                           timeout=5, env=_system_env())
    return (value.stdout.strip() or None) if value.returncode == 0 else None


def _git_failure() -> list[str]:
    """The line to print, once a day, when the git that launch runs does not
    run, such as the git of a developer folder without the command line
    tools. Nothing when git runs. Without git, launch cannot add the git
    identity to the private home or share the git folder of a worktree."""
    folders = SYSTEM_PATH.split(os.pathsep)
    listed = f"{', '.join(folders[:-1])} and {folders[-1]}"
    found, git, _ = _launch_program("git")
    if found is None:
        what = f"[launch] git is in none of {listed}, the folders launch runs git from."
    elif git is None:
        what = (f"[launch] {found} runs the git of the developer folder that xcode-select "
                "names, and xcode-select names none.")
    else:
        try:
            proc = subprocess.run([git, "--version"], stdin=subprocess.DEVNULL,
                                  capture_output=True, text=True, timeout=5,
                                  env=_system_env())
            if proc.returncode == 0:
                return []
            said = next((s.strip() for s in (proc.stderr + proc.stdout).splitlines()
                         if s.strip()), "")
        except subprocess.TimeoutExpired:
            said = "it did not answer in 5 seconds"
        except OSError as e:
            said = e.strerror or str(e)
        said = f" ({said[:200]})" if said else ""
        what = f"[launch] {git} did not run{said}. Launch runs git only from {listed}."
        if git != found:
            what += f" For {found}, it runs the git of the developer folder that xcode-select names."
    return notices.due([Once(f"{what} Without git, launch adds no git name and email to "
                             "the private home, and shares no git folder for a linked "
                             "worktree. Run xcode-select --install, or install git with "
                             "Homebrew.", f"git:{git or found}", notices.DAY)])


def _seed_git_identity(home: Path) -> list[str]:
    """Add the host ``user.name`` and ``user.email`` to the private home's
    ``.gitconfig`` where they are missing, and follow a change on the Mac
    for a value launch wrote there. A value set in the container stays.
    Returns the line to print for a value that followed the Mac, or for a
    git that does not run. git edits a copy outside the private home, since
    git follows a link at the file it writes, and the result goes back
    through the confined write."""
    import json
    import tempfile

    from . import confine

    gitconfig = home / ".gitconfig"
    record = identity_record_path(home)
    wrote = {k: v for k, v in _read_json_record(record).items() if isinstance(v, str)}
    known, updated, found = dict(wrote), [], False
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
                found = True
                if have == mac:
                    # The same value as the Mac's, so it follows the Mac
                    # from now on, also in a home from before the record.
                    known[key] = mac
                    continue
                if have is not None and wrote.get(key) != have:
                    continue                  # set in the container
                subprocess.run([_git_program(), "config", "--file", work, key, mac],
                               stdin=subprocess.DEVNULL, capture_output=True, timeout=5,
                               env=_system_env())
            except (OSError, subprocess.TimeoutExpired):
                return _git_failure()
            known[key] = mac
            if have is not None:
                updated.append(key)
        after = Path(work).read_text()
    if after != before:
        confine.write_text(gitconfig, after)
    if known != wrote:
        write_record(record, json.dumps(known).encode())
    # git config exits with 1 for a missing key, and so does a git that
    # cannot run. So when the Mac gives no identity, launch checks that git
    # runs.
    out = [] if found else _git_failure()
    if updated:
        out.append(f"[launch] updated the git {' and '.join(updated)} in the private home "
                   "to match the Mac.")
    return out


# The server config the guest could change

def server_config_path(host: str, port: int, *, autostart: bool = True,
                       notes: list[str] | None = None) -> str | None:
    """The config file the target server runs with: the one in its runfile
    while it runs or launchd manages it, as its start named it, else the one
    autostart would use.
    The named path can be a link, and the share check needs to see it. With
    ``autostart`` False, as for ``--base-url``, only a runfile counts. A runfile from an
    older gmlx can hold a path relative to a folder launch cannot know, so
    that path counts as unknown and ``notes`` gets a line about it. A start
    that names no config, model or model folder reads the first default
    config, also when its runfile does not record that file."""
    from gmlx.config import default_config_paths
    from gmlx.serve import lifecycle

    run = lifecycle.read_run(host, port) or {}
    # launchd starts a headless agent again at each login and after a crash,
    # and its runfile records no pid, so the runfile counts while it exists.
    live = run.get("managed_by") == "launchd" or lifecycle.pid_alive(run.get("pid"))
    # An older gmlx installed a headless agent with no --config when no
    # config existed. After gmlx init, that agent reads the default config.
    bare = (live and not run.get("config_abspath")
            and lifecycle.serves_default_config(run.get("argv") or []))
    if run and not run.get("config_abspath") and live and not bare:
        # Such a server may scan --models-dir, which the runfile does not
        # record.
        if notes is not None:
            notes.append(f"[launch] the server on port {port} has no config file, so launch "
                         "cannot check whether it scans a shared folder for models, where a "
                         "client could add a model file. Start the server from a config file "
                         "to have it checked.")
        return None
    if run.get("config_abspath") and live:
        path = str(run["config_abspath"])
        given = run.get("config_given")
        if isinstance(given, str) and os.path.isabs(given):
            # The server reads its config again through the path its start
            # named. That path can be a link in a share to the file it records.
            path = given
        if os.path.isabs(path):
            return path
        if notes is not None:
            # gmlx restart refuses a launchd agent, and doctor gives the
            # steps to install such an agent again.
            step = ("Run gmlx doctor for the steps to install it again with the full path."
                    if run.get("managed_by") == "launchd"
                    else "Restart the server with gmlx restart to record the full path.")
            notes.append(f"[launch] the server on port {port} records its config as {path}, "
                         "relative to the folder it started from, so launch cannot check "
                         f"whether that config is in a share. {step}")
        return None
    if not autostart and not bare:
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
    relative path taken from ``cwd``, the folder the server runs in. Links
    stay in it, because the server follows them each time it reads it."""
    return os.path.abspath(os.path.join(cwd, os.path.expanduser(os.path.expandvars(path))))


def _model_paths(cfg, cwd: str) -> list[tuple[str, str, str]]:
    """The files the server config names that gmlx reads or runs on the Mac:
    model files, chat template files, the local models of the embeddings,
    rerank, tts and stt services, and the programs and path arguments of
    the stdio tool servers. They are made absolute the way the server
    resolves them from ``cwd``, with their links kept, and never read. Each
    comes with a phrase that names it at ``{path}``, and with when gmlx next
    uses it. As in ``resolve_path``, a relative model path that no model
    folder holds is taken from the working folder."""
    load = "before the server's next load"
    out: list[tuple[str, str, str]] = []
    roots = [_expand(d, cwd) for d in cfg.model_dirs]

    def model_file(p: str) -> str:
        p = os.path.expandvars(os.path.expanduser(p))
        if os.path.isabs(p):
            return os.path.abspath(p)
        return os.path.abspath(next((c for c in (os.path.join(r, p) for r in roots)
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
            out.append((os.path.abspath(os.path.join(cwd, t)),
                        "the chat template file {path} is", load))
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
            out.append((os.path.abspath(os.path.join(cwd, local)),
                        f"the {key} model {{path}} is", load))
    servers = list(cfg.assistant.mcp)
    for alias in cfg.assistants.values():
        servers += alias.mcp or []
    for server in servers:
        if not server.command:
            continue
        start = "before gmlx next starts that tool server on the Mac"
        program = os.path.expanduser(server.command[0])
        if os.path.isabs(program) or "/" in program:
            out.append((os.path.abspath(os.path.join(cwd, program)),
                        f"the tool server {server.name} runs {{path}}, which is", start))
        given = [*server.command[1:], *(e for v in server.env.values()
                                        for e in str(v).split(os.pathsep))]
        for arg in given:
            for p in (arg, arg.partition("=")[2]):
                if p.startswith(("/", "~")):
                    out.append((os.path.abspath(os.path.expanduser(p)),
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
        reach = _share_reach(entry, rw, home)
        if reach is not None:
            out.append(f"[launch] warning: PYTHONPATH holds {_tilde(entry, home)}, which "
                       f"{reach[0]}. The client can add a module there that the next gmlx "
                       "command imports on the Mac. To prevent this, remove the entry from "
                       f"PYTHONPATH, or share {_tilde(reach[1].source, home)} read-only.")
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
        shown = _tilde(in_share.source, home)
        out.append(f"[launch] warning: the client can change the server config "
                   f"{_tilde(written, home)} in the read-write share {shown}, and the server "
                   "applies a change at its next reload. To prevent this, move the config out "
                   f"of the share, or share {shown} read-only.")
    # A client can replace the config, or a link on the way to it, with a
    # link to any file of yours, so one that leads out of a folder a client
    # could write is never read. The config path resolves once for all the
    # folders.
    reached = _reach(written)
    for folder in dict.fromkeys([*(m.source for m in rw), *shared_history()]):
        if _inside(real, folder):
            continue
        if _inside(written, folder):
            out.append(f"[launch] warning: the server config {_tilde(written, home)} leads "
                       f"to {_tilde(real, home)}, outside {_tilde(folder, home)}, which a "
                       "session shares or once shared read-write. A client may have replaced "
                       "it with a symbolic link, so launch did not read it. Check it before "
                       "the server reloads.")
            return out
        link = _link_in(folder, written, reached)
        if link is not None:
            out.append(f"[launch] warning: the server config {_tilde(written, home)} is "
                       f"reached through {_tilde(link, home)}, a link in "
                       f"{_tilde(folder, home)}, which a session shares or once shared "
                       "read-write. A client can change where the link leads, and the server "
                       "then reads a config that the client chooses, so launch did not read "
                       "it. Check where the link leads, and start the server with --config "
                       "and a path that does not go through the link.")
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
    # The guest opens the file a link leads to, so only that file counts. A
    # link in a share that leads out of every share gives the client no key.
    held = next((m for m in shares if m.kind in ("share", "git")
                 and _inside(real, m.source)), None)
    if cfg.api_key and held is not None:
        out.append(f"[launch] warning: the server config {_tilde(written, home)} sets "
                   f"server.api_key, and the client can read it in the share "
                   f"{_tilde(held.source, home)}. With the key, the client can call every "
                   "route of the server wherever it reaches the server's port. Move the "
                   "config out of the share.")
    for spec in cfg.discover:
        folders = [spec.dir] if spec.dir else list(cfg.model_dirs)
        for folder in folders:
            written = _expand(folder, cwd)
            f = _real(written)
            for m in rw:
                if _inside(f, m.source) or (spec.recursive and _inside(m.source, f)):
                    shown = _tilde(m.source, home)
                    out.append(f"[launch] warning: the server scans {_tilde(f, home)} for "
                               f"models, and the client can add files there through {shown}. "
                               "To prevent this, keep the share out of the folders that the "
                               f"server scans, or share {shown} read-only.")
                    break
                link = _link_in(m.source, written)
                if link is not None:
                    shown = _tilde(m.source, home)
                    out.append(f"[launch] warning: the server scans {_tilde(f, home)} for "
                               f"models through {_tilde(link, home)}, a link in the "
                               f"read-write share {shown}. The client can change where the "
                               "link leads, and the server then scans a folder that the client "
                               f"chooses. To prevent this, write {_tilde(f, home)} for the "
                               f"folder in the server config, or share {shown} read-only.")
                    break
    for path, what, when in _model_paths(cfg, cwd):
        real = _real(path)
        m = next((m for m in rw if _inside(real, m.source)), None)
        if m is not None:
            shown = _tilde(m.source, home)
            out.append(f"[launch] warning: {what.replace('{path}', _tilde(real, home))} "
                       f"inside the read-write share, so the client can replace it {when}. "
                       f"To prevent this, move it out of the share, or share {shown} "
                       "read-only.")
            continue
        hit = next(((m, link) for m in rw for link in [_link_in(m.source, path)]
                    if link is not None), None)
        if hit is not None:
            shown = _tilde(hit[0].source, home)
            out.append(f"[launch] warning: {what.replace('{path}', _tilde(path, home))} "
                       f"reached through {_tilde(hit[1], home)} in the read-write share "
                       f"{shown}, so the client can change where it leads {when}. To prevent "
                       "this, name it in the server config by a path that does not go through "
                       f"the share, or share {shown} read-only.")
    return list(dict.fromkeys(out))
