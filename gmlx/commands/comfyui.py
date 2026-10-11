"""``gmlx comfyui``: put the gmlx node pack into ComfyUI and keep it current.

``install`` finds ComfyUI (a clone, comfy-cli's workspace, an install of
Comfy Desktop or of the old ComfyUI desktop app, or the folder
``--comfyui-dir`` names), refuses one older than the pack's floor, copies the
pack to a folder gmlx owns and links that copy into ComfyUI's
``custom_nodes`` as ``comfyui-gmlx``. It records the folders, the ``gmlx``
command and how gmlx was installed in the settings file the pack reads.
``uninstall`` removes the links, the copy and the record. ``status`` shows
them.

The copy lives at ``$XDG_DATA_HOME/gmlx/comfyui/comfyui-gmlx``, a link to a
versioned folder beside it, so a refresh swaps the whole pack at once. A
stamp in the folder holds the gmlx version and a hash of the files;
``gmlx serve``, ``gmlx launch comfyui`` and ``install`` refresh the copy when
the stamp differs from the running gmlx.

gmlx never imports the pack, which runs in ComfyUI's Python: this module
reads the pack's version file as text.

Exit codes: 0 on success, 1 for a refusal (no ComfyUI, one too old), 2 for
a usage error such as several ComfyUI folders and no ``--comfyui-dir``.
"""
from __future__ import annotations

import argparse
import ast
import configparser
import hashlib
import json
import os
import re
import shlex
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PACK_SOURCE = Path(__file__).resolve().parent.parent / "comfyui_pack"
LINK_NAME = "comfyui-gmlx"
STAMP = "gmlx-stamp.json"
DESKTOP_DOWNLOAD = "https://comfy.org/download"
CLONE_URL = "https://github.com/comfyanonymous/ComfyUI"

_ACTIONS = ("install", "uninstall", "status")
_ACTION_DESC = {
    "install": "put the gmlx nodes into ComfyUI, or refresh them",
    "uninstall": "remove the gmlx nodes from ComfyUI",
    "status": "show where the gmlx nodes are installed",
}
# Comfy Desktop's userData folder: the package name first, then the
# product name.
_DESKTOP_DATA = ("comfyui-desktop-2", "Comfy Desktop")
_VENVS = (".venv", "venv", ".env", "env")
_IGNORE = ("__pycache__", "*.pyc", ".DS_Store", STAMP)


class ComfyError(Exception):
    """A refusal with its exit code."""

    def __init__(self, message: str, code: int = 1):
        super().__init__(message)
        self.code = code


# Paths.

def _home() -> Path:
    return Path(os.path.expanduser("~"))


def data_root() -> Path:
    """The folder gmlx owns for the pack: the copy and its versions."""
    base = os.environ.get("XDG_DATA_HOME") or str(_home() / ".local" / "share")
    return Path(base) / "gmlx" / "comfyui"


def copy_dir() -> Path:
    """The pack copy that ComfyUI's custom_nodes link to."""
    return data_root() / LINK_NAME


def settings_path() -> Path:
    """The settings file the pack reads, which holds the install record."""
    return Path(os.environ.get("GMLX_COMFY_SETTINGS")
                or _home() / ".config" / "gmlx" / "comfyui.json")


def _app_support() -> Path:
    return _home() / "Library" / "Application Support"


def _short(p) -> str:
    """A path with the home folder written as ~."""
    s, home = str(p), str(_home())
    return "~" + s[len(home):] if s == home or s.startswith(home + os.sep) else s


# Versions.

def pack_meta(source: Path = PACK_SOURCE) -> dict:
    """The literals of the pack's version file, read without importing it."""
    out = {}
    for node in ast.parse((source / "version.py").read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


def version_tuple(text) -> tuple:
    """``"0.37.4"`` as ``(0, 37, 4)``; the leading numbers only."""
    nums = re.findall(r"\d+", str(text or ""))[:3]
    return tuple(int(n) for n in nums) + (0,) * (3 - len(nums)) if nums else ()


def comfyui_version(code) -> str | None:
    """ComfyUI's version from ``comfyui_version.py`` in its code folder,
    read as text, or None."""
    if code is None:
        return None
    try:
        text = (Path(code) / "comfyui_version.py").read_text()
    except OSError:
        return None
    m = re.search(r"__version__\s*=\s*[\"']([^\"']+)[\"']", text)
    return m.group(1) if m else None


def gmlx_version() -> str:
    import gmlx

    return str(gmlx.__version__)


# How gmlx runs.

def gmlx_install() -> str:
    """How the running gmlx was installed: homebrew, uv or pip."""
    prefix = Path(sys.prefix)
    if "Cellar" in prefix.resolve().parts:
        return "homebrew"
    if (prefix / "uv-receipt.toml").is_file():
        return "uv"
    return "pip"


def gmlx_command() -> str:
    """The absolute path of the ``gmlx`` command of this install: the one on
    PATH when it is this install's, since Homebrew and uv keep that path
    across upgrades, else the script beside the interpreter."""
    script = Path(sys.prefix) / "bin" / "gmlx"
    on_path = shutil.which("gmlx")
    if on_path and script.exists() and \
            os.path.realpath(on_path) == os.path.realpath(script):
        return os.path.abspath(on_path)
    if script.exists():
        return str(script)
    if os.path.basename(sys.argv[0] or "") == "gmlx":
        return os.path.abspath(sys.argv[0])
    return os.path.abspath(on_path) if on_path else "gmlx"


# ComfyUI folders.

@dataclass
class Target:
    """One ComfyUI: the custom_nodes folder the link goes in, the code
    folder with main.py and comfyui_version.py, and how it was installed."""
    custom_nodes: Path
    code: Path | None
    kind: str
    python: Path | None = None
    label: str = ""
    roots: list = field(default_factory=list)
    policy: Path | None = None

    def key(self) -> str:
        return os.path.realpath(self.custom_nodes)

    def describe(self) -> str:
        names = {"clone": "a ComfyUI folder", "comfy-cli": "comfy-cli's ComfyUI",
                 "desktop": "Comfy Desktop", "desktop-adopted": "Comfy Desktop",
                 "desktop-build": "Comfy Desktop", "desktop-git": "Comfy Desktop",
                 "desktop-legacy": "the old ComfyUI desktop app",
                 "folder": "a ComfyUI folder"}
        what = names.get(self.kind, "ComfyUI")
        if self.label and self.kind.startswith("desktop") and self.kind != "desktop-legacy":
            what += f" install {self.label!r}"
        return what

    def matches(self, path: Path) -> bool:
        want = os.path.realpath(path)
        cands = [self.custom_nodes.parent, *self.roots]
        if self.code is not None:
            cands.append(self.code)
        return any(os.path.realpath(c) == want for c in cands)

    def to_record(self) -> dict:
        return {"custom_nodes": str(self.custom_nodes),
                "comfyui": str(self.code) if self.code else None,
                "kind": self.kind,
                "python": str(self.python) if self.python else None,
                "label": self.label}

    @classmethod
    def from_record(cls, rec: dict) -> "Target":
        return cls(custom_nodes=Path(rec["custom_nodes"]),
                   code=Path(rec["comfyui"]) if rec.get("comfyui") else None,
                   kind=str(rec.get("kind") or "folder"),
                   python=Path(rec["python"]) if rec.get("python") else None,
                   label=str(rec.get("label") or ""))


def _venv_python(folder: Path) -> Path | None:
    """The interpreter of the first venv in ``folder``."""
    for name in _VENVS:
        venv = folder / name
        for exe in ("python3", "python"):
            if (venv / "pyvenv.cfg").is_file() and os.access(venv / "bin" / exe, os.X_OK):
                return venv / "bin" / exe
    for name in _VENVS:
        for exe in ("python", "python3"):
            if os.access(folder / name / "bin" / exe, os.X_OK):
                return folder / name / "bin" / exe
    return None


def _clone(code: Path, kind: str) -> Target:
    return Target(code / "custom_nodes", code, kind, _venv_python(code), roots=[code])


def _clone_targets() -> list:
    out = []
    home = _home()
    if (home / "ComfyUI" / "main.py").is_file():
        out.append(_clone(home / "ComfyUI", "clone"))
    for ws in _comfy_cli_workspaces():
        for code in (ws, ws / "ComfyUI"):
            if (code / "main.py").is_file():
                out.append(_clone(code, "comfy-cli"))
                break
    return out


def _comfy_cli_workspaces() -> list:
    """comfy-cli's default workspace, and the one its config names."""
    out = [_home() / "comfy"]
    ini = _app_support() / "comfy-cli" / "config.ini"
    if ini.is_file():
        cp = configparser.ConfigParser()
        try:
            cp.read(ini)
            ws = cp.get("DEFAULT", "default_workspace", fallback="")
        except (configparser.Error, OSError):
            ws = ""
        if ws:
            out.append(Path(os.path.expanduser(ws)))
    return out


def _desktop_records() -> list:
    """The records of Comfy Desktop's installations.json, or []."""
    for name in _DESKTOP_DATA:
        path = _app_support() / name / "installations.json"
        if not path.is_file():
            continue
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            return []
        return [r for r in doc if isinstance(r, dict)] if isinstance(doc, list) else []
    return []


def desktop_target(rec: dict) -> Target | None:
    """The ComfyUI a Comfy Desktop record runs, per its source, or None for
    a record without a local ComfyUI."""
    source = rec.get("sourceId")
    install = rec.get("installPath")
    if source in ("cloud", "remote") or not isinstance(install, str) or not install:
        return None
    if rec.get("status") in ("installing", "failed"):
        return None
    root = Path(install)
    label = str(rec.get("name") or "")
    if rec.get("adopted") is True and rec.get("adoptedBaseDir"):
        # Desktop runs a fresh ComfyUI from installPath with
        # --base-directory at the old app's folder, where custom_nodes stay.
        base = Path(str(rec["adoptedBaseDir"]))
        py = rec.get("adoptedPythonPath")
        return Target(base / "custom_nodes", root / "ComfyUI", "desktop-adopted",
                      Path(py) if isinstance(py, str) and py else None, label,
                      roots=[root, base])
    if source == "standalone":
        code = root / "ComfyUI"
        return Target(code / "custom_nodes", code, "desktop", code / ".venv" / "bin" / "python3",
                      label, roots=[root])
    if source == "comfybuilder":
        code = root / "ComfyUI"
        policy = code / "governance" / "policy.signed.json"
        return Target(code / "custom_nodes", code, "desktop-build",
                      root / "venv" / "bin" / "python3", label, roots=[root],
                      policy=policy if policy.is_file() else None)
    if source == "git":
        for code in (root, root / "ComfyUI"):
            if (code / "main.py").is_file():
                venv, py = rec.get("venvPath"), None
                if isinstance(venv, str) and venv:
                    for exe in ("python3", "python"):
                        if os.access(Path(venv) / "bin" / exe, os.X_OK):
                            py = Path(venv) / "bin" / exe
                            break
                return Target(code / "custom_nodes", code, "desktop-git",
                              py or _venv_python(code), label, roots=[root])
        return None
    if source == "desktop":
        legacy = _legacy_target()
        if legacy is not None:
            legacy.label, legacy.roots = label, [*legacy.roots, root]
            return legacy
        return Target(root / "custom_nodes", None, "desktop-legacy",
                      root / ".venv" / "bin" / "python", label, roots=[root])
    for code in (root / "ComfyUI", root):
        if (code / "main.py").is_file():
            return Target(code / "custom_nodes", code, "folder", _venv_python(code), label,
                          roots=[root])
    return None


def _desktop_targets() -> list:
    recs = _desktop_records()
    # The old app's record last, so an adopted install that shares its
    # folder wins.
    recs.sort(key=lambda r: r.get("sourceId") == "desktop")
    return [t for t in (desktop_target(r) for r in recs) if t is not None]


def _legacy_target() -> Target | None:
    """The old ComfyUI desktop app: the custom_nodes that the
    ``comfyui_desktop`` entry of its extra_models_config.yaml names. The
    ``desktop_extensions`` entry is inside the app, which install never
    writes."""
    folder = _app_support() / "ComfyUI"
    path = folder / "extra_models_config.yaml"
    if not path.is_file():
        return None
    import yaml

    try:
        doc = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return None
    entry = doc.get("comfyui_desktop") if isinstance(doc, dict) else None
    if not isinstance(entry, dict) or not entry.get("base_path"):
        return None
    base = Path(os.path.expanduser(str(entry["base_path"])))
    nodes = Path(os.path.expanduser(str(entry.get("custom_nodes") or "custom_nodes")))
    nodes = nodes if nodes.is_absolute() else base / nodes
    code = None
    ext = doc.get("desktop_extensions")
    if isinstance(ext, dict) and ext.get("custom_nodes"):
        cand = Path(str(ext["custom_nodes"])).parent
        if (cand / "main.py").is_file():
            code = cand
    if code is None:
        cand = Path("/Applications/ComfyUI.app/Contents/Resources/ComfyUI")
        code = cand if (cand / "main.py").is_file() else None
    return Target(nodes, code, "desktop-legacy", base / ".venv" / "bin" / "python",
                  roots=[base])


def detect() -> list:
    """Every ComfyUI found on this Mac whose folder exists: the clone at
    ~/ComfyUI, comfy-cli's workspace, Comfy Desktop's installs and the old
    desktop app. One entry per custom_nodes folder."""
    found = [*_clone_targets(), *_desktop_targets()]
    legacy = _legacy_target()
    if legacy is not None:
        found.append(legacy)
    out, seen = [], set()
    for t in found:
        if not t.custom_nodes.parent.is_dir() or t.key() in seen:
            continue
        seen.add(t.key())
        out.append(t)
    return out


def target_for_dir(path) -> Target:
    """The ComfyUI that ``--comfyui-dir`` names: a detected one whose
    folder it is, else a folder with main.py, else a base folder with
    custom_nodes."""
    d = Path(os.path.abspath(os.path.expanduser(str(path))))
    for t in detect():
        if t.matches(d):
            return t
    for code in (d, d / "ComfyUI"):
        if (code / "main.py").is_file():
            return _clone(code, "clone")
    if (d / "custom_nodes").is_dir():
        return Target(d / "custom_nodes", None, "folder", _venv_python(d), roots=[d])
    raise ComfyError(f"{_short(d)} is not a ComfyUI folder: it has no main.py and no "
                     f"custom_nodes folder. {_where_hint()}")


def _where_hint() -> str:
    from gmlx import DOCS_URL

    return f"The ComfyUI guide says where each kind of install keeps it:\n  {DOCS_URL}comfyui.html"


def no_comfyui_help() -> str:
    """How to get ComfyUI, for a Mac without one."""
    from gmlx import DOCS_URL

    return ("no ComfyUI found in ~/ComfyUI, in comfy-cli's ~/comfy/ComfyUI or among the "
            "installs of Comfy Desktop. Get ComfyUI one of two ways:\n\n"
            f"Comfy Desktop, the app: download it from {DESKTOP_DOWNLOAD}, open it and "
            "create an install, then run gmlx comfyui install.\n\n"
            "A ComfyUI folder that gmlx launch comfyui runs:\n"
            "  brew install uv\n"
            f"  git clone {CLONE_URL} ~/ComfyUI\n"
            "  cd ~/ComfyUI\n"
            "  uv venv --python 3.12 .venv\n"
            "  uv pip install -r requirements.txt\n\n"
            "For a ComfyUI in another folder, pass --comfyui-dir. The ComfyUI guide:\n"
            f"  {DOCS_URL}comfyui.html")


def several_help(found: list, verb: str = "gmlx comfyui install") -> str:
    lines = [f"found {len(found)} ComfyUI installs. Run {verb} with --comfyui-dir and one "
             "of these folders, once for each that should have the gmlx nodes:"]
    for t in found:
        lines.append(f"  {_short(t.custom_nodes.parent)}  ({t.describe()})")
    return "\n".join(lines)


def upgrade_comfyui_steps(t: Target) -> str:
    """The steps that update this kind of ComfyUI."""
    if t.kind == "desktop-legacy":
        return ("This is the old ComfyUI desktop app, whose ComfyUI stays older than the "
                f"gmlx nodes need. Install Comfy Desktop from {DESKTOP_DOWNLOAD}, then run "
                "gmlx comfyui install again.")
    if t.kind in ("desktop", "desktop-adopted", "desktop-build"):
        return ("Update ComfyUI in Comfy Desktop, or update the app, then run "
                "gmlx comfyui install again.")
    if t.kind == "comfy-cli" and t.code is not None:
        return (f"Update it:\n  comfy --workspace {shlex.quote(str(t.code))} update comfy\n"
                "then run gmlx comfyui install again.")
    if t.code is not None and (t.code / ".git").exists():
        d = shlex.quote(str(t.code))
        py = shlex.quote(str(t.python)) if t.python else f"{d}/.venv/bin/python"
        return (f"Update it:\n  git -C {d} pull\n"
                f"  {py} -m pip install -r {d}/requirements.txt\n"
                "then run gmlx comfyui install again.")
    return "Update it the way you installed it, then run gmlx comfyui install again."


def check_version(t: Target, meta: dict) -> tuple[str | None, list]:
    """ComfyUI's version and the notes about it. Raises ComfyError below
    the pack's floor."""
    v = comfyui_version(t.code)
    floor, tested = meta["COMFYUI_FLOOR"], meta["COMFYUI_TESTED"]
    notes = []
    if v is None:
        notes.append(f"the ComfyUI version in {_short(t.custom_nodes.parent)} is unknown; the "
                     f"gmlx nodes need {floor} or newer")
    elif version_tuple(v) < version_tuple(floor):
        raise ComfyError(f"ComfyUI {v} in {_short(t.code)} is older than {floor}, the oldest "
                         f"the gmlx nodes run on. {upgrade_comfyui_steps(t)}")
    elif version_tuple(v) > version_tuple(tested):
        notes.append(f"ComfyUI {v} is newer than {tested}, the newest this gmlx was tested on")
    if t.policy is not None:
        notes.append(f"this Comfy Desktop build carries a signed policy at {_short(t.policy)}, "
                     "which can keep ComfyUI from loading custom nodes it does not list")
    return v, notes


# The record.

def load_record() -> dict:
    try:
        doc = json.loads(settings_path().read_text())
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


_RECORD_KEYS = ("installs", "comfyui_dir", "gmlx_command", "gmlx_install", "gmlx_python",
                "gmlx_version", "default_image_model", "pack_dir", "pack_source")


def save_record(doc: dict) -> None:
    """Write the settings file atomically, private to the user."""
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def recorded_targets(doc: dict) -> list:
    out = []
    for rec in doc.get("installs") or []:
        if isinstance(rec, dict) and rec.get("custom_nodes"):
            out.append(Target.from_record(rec))
    return out


def default_image_model() -> str | None:
    """The config's default image model: server.defaults.image_model, else
    its only image model."""
    try:
        import gmlx.config as config

        for p in config.default_config_paths(note_local=False):
            if p.exists():
                cfg = config.load_config(p)
                if cfg.defaults.image_model:
                    return cfg.defaults.image_model
                return next(iter(cfg.image_models)) if len(cfg.image_models) == 1 else None
    except Exception:  # noqa: BLE001 - a config that does not load names no model
        return None
    return None


# The copy.

def content_hash(source: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(source.rglob("*")):
        rel = p.relative_to(source)
        if p.is_dir() or any(_ignored(part) for part in rel.parts):
            continue
        h.update(rel.as_posix().encode() + b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _ignored(name: str) -> bool:
    import fnmatch

    return any(fnmatch.fnmatch(name, pat) for pat in _IGNORE)


def read_stamp(folder: Path) -> dict | None:
    try:
        doc = json.loads((folder / STAMP).read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def want_stamp(source: Path = PACK_SOURCE) -> dict:
    meta = pack_meta(source)
    return {"gmlx_version": gmlx_version(), "pack_version": meta["__version__"],
            "content_hash": content_hash(source), "comfyui_floor": meta["COMFYUI_FLOOR"],
            "comfyui_tested": meta["COMFYUI_TESTED"]}


@dataclass
class Sync:
    state: str            # "current" or "refreshed"
    old: dict | None
    new: dict
    warning: str | None = None


def sync_copy(source: Path = PACK_SOURCE) -> Sync:
    """Make the copy match the running gmlx. The files go to a temporary
    folder, which a rename makes the new version, and a link swapped in one
    rename points the copy at it, so a ComfyUI that starts meanwhile loads
    either the old pack or the new one, never a mix."""
    want = want_stamp(source)
    link = copy_dir()
    old = read_stamp(link)
    if old is not None and all(old.get(k) == want[k] for k in ("gmlx_version", "content_hash")):
        return Sync("current", old, old)
    warning = None
    if old and version_tuple(old.get("gmlx_version")) > version_tuple(want["gmlx_version"]):
        warning = (f"replaced the gmlx nodes from gmlx {old.get('gmlx_version')} with those of "
                   f"the older gmlx {want['gmlx_version']} that runs here")
    versions = data_root() / "versions"
    versions.mkdir(parents=True, exist_ok=True)
    name = f"{want['gmlx_version']}-{want['content_hash'][:12]}"
    dest = versions / name
    if read_stamp(dest) != want:
        tmp = versions / f".tmp-{name}-{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(source, tmp, ignore=shutil.ignore_patterns(*_IGNORE))
        (tmp / STAMP).write_text(json.dumps(want, indent=1, sort_keys=True) + "\n")
        if dest.exists():
            shutil.rmtree(dest)
        os.rename(tmp, dest)
    if link.exists() and not link.is_symlink():
        # A real folder from elsewhere: kept aside, never deleted.
        os.rename(link, versions / f"moved-{time.strftime('%Y%m%d%H%M%S')}-{os.getpid()}")
    swap = data_root() / f".{LINK_NAME}.{os.getpid()}.tmp"
    if swap.is_symlink() or swap.exists():
        swap.unlink()
    os.symlink(os.path.join("versions", name), swap)
    os.replace(swap, link)
    for other in versions.iterdir():
        if other.name != name and not other.name.startswith("moved-"):
            shutil.rmtree(other, ignore_errors=True)
    return Sync("refreshed", old, want, warning)


# The link.

def _points_at(link: Path, dest: Path) -> bool:
    try:
        return os.path.realpath(link) == os.path.realpath(dest)
    except OSError:
        return False


def link_into(t: Target, dest: Path) -> tuple[bool, str | None]:
    """Link ``custom_nodes/comfyui-gmlx`` to ``dest``. A real folder of that
    name, in any case, moves into ``custom_nodes/.disabled``, where ComfyUI
    and its managers keep disabled packs. Returns whether anything changed
    and where a folder moved, or None."""
    cn = t.custom_nodes
    cn.mkdir(exist_ok=True)
    moved, current = None, False
    want = cn / LINK_NAME
    for entry in sorted(cn.iterdir()):
        if entry.name.lower() != LINK_NAME:
            continue
        if entry.is_symlink():
            if entry.name == LINK_NAME and os.readlink(entry) == str(dest):
                current = True
            else:
                entry.unlink()
            continue
        disabled = cn / ".disabled"
        disabled.mkdir(exist_ok=True)
        stem = f"{LINK_NAME}-{time.strftime('%Y%m%d')}"
        aside, n = disabled / stem, 2
        while aside.exists() or aside.is_symlink():
            aside, n = disabled / f"{stem}-{n}", n + 1
        os.rename(entry, aside)
        moved = str(aside)
    if current:
        return moved is not None, moved
    # The temporary name ends in .disabled, which ComfyUI skips, so a
    # crash here never leaves a second copy of the nodes for it to load.
    tmp = cn / f".{LINK_NAME}.{os.getpid()}.tmp.disabled"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    os.symlink(str(dest), tmp)
    os.replace(tmp, want)
    return True, moved


def link_state(t: Target, doc: dict) -> str:
    """"linked", "missing", "elsewhere" or "folder" for a target's link."""
    entry = t.custom_nodes / LINK_NAME
    if not entry.is_symlink():
        return "folder" if entry.exists() else "missing"
    dest = PACK_SOURCE if doc.get("pack_source") else copy_dir()
    return "linked" if _points_at(entry, dest) and entry.exists() else "elsewhere"


# Install, uninstall, status.

@dataclass
class Installed:
    target: Target
    version: str | None
    notes: list
    changed: bool
    moved: str | None


def install(targets: list, *, link_source: bool = False, model: str | None = None,
            remember: bool = True) -> tuple[list, Sync | None]:
    """Check, copy, link and record ``targets``. Raises ComfyError on a
    refusal, before anything is written."""
    meta = pack_meta()
    checked = [(t, *check_version(t, meta)) for t in targets]
    sync = None if link_source else sync_copy()
    dest = PACK_SOURCE if link_source else copy_dir()
    done = [Installed(t, v, notes, *link_into(t, dest)) for t, v, notes in checked]
    doc = load_record()
    keep = [r for r in doc.get("installs") or []
            if isinstance(r, dict) and r.get("custom_nodes")
            and os.path.realpath(r["custom_nodes"]) not in {t.key() for t in targets}]
    doc["installs"] = keep + [t.to_record() for t in targets]
    doc.update(gmlx_command=gmlx_command(), gmlx_install=gmlx_install(),
               gmlx_python=sys.executable, gmlx_version=gmlx_version(),
               pack_dir=str(dest), pack_source=bool(link_source))
    model = model or default_image_model() or doc.get("default_image_model")
    if model:
        doc["default_image_model"] = model
    if remember:
        for t in targets:
            if t.code is not None and not t.kind.startswith("desktop"):
                doc["comfyui_dir"] = str(t.code)
    save_record(doc)
    return done, sync


def resolve_targets(comfyui_dir, doc: dict, verb: str = "gmlx comfyui install") -> list:
    """The ComfyUI folders to install into: --comfyui-dir, else the recorded
    folders that exist, else detection."""
    if comfyui_dir:
        return [target_for_dir(comfyui_dir)]
    recorded = [t for t in recorded_targets(doc) if t.custom_nodes.parent.is_dir()]
    if recorded:
        return recorded
    found = detect()
    if not found:
        raise ComfyError(no_comfyui_help(), 1)
    if len(found) > 1:
        raise ComfyError(several_help(found, verb), 2)
    return found


def install_lines(done: list, sync: Sync | None, *, prefix: str = "") -> list:
    out = []
    if sync is not None and sync.warning:
        out.append(f"{prefix}warning: {sync.warning}")
    for d in done:
        where = _short(d.target.custom_nodes / LINK_NAME)
        ver = f", ComfyUI {d.version}" if d.version else ""
        out.append(f"{prefix}installed the gmlx nodes in {where} ({d.target.describe()}{ver})")
        if d.moved:
            out.append(f"{prefix}  the folder that was there moved to {_short(d.moved)}")
        for n in d.notes:
            out.append(f"{prefix}  note: {n}")
    return out


def _cmd_install(argv: list, prog: str) -> int:
    ap = argparse.ArgumentParser(
        prog=prog,
        description="Put the gmlx nodes into ComfyUI, or refresh them. Without "
                    "--comfyui-dir it uses the folders installed before, else the "
                    "ComfyUI it finds: ~/ComfyUI, comfy-cli's ~/comfy/ComfyUI, the "
                    "installs of Comfy Desktop and the old ComfyUI desktop app.")
    ap.add_argument("--comfyui-dir", metavar="DIR",
                    help="The ComfyUI folder, its custom_nodes parent or a Comfy Desktop "
                         "install folder. Run once per ComfyUI that should have the nodes.")
    ap.add_argument("--link-source", action="store_true",
                    help="Link to the pack in this gmlx source tree instead of a copy, for "
                         "work on the pack.")
    ap.add_argument("--model", metavar="ID",
                    help="The image model a new loader node starts on (default: the "
                         "config's default image model).")
    a = ap.parse_args(argv)
    doc = load_record()
    try:
        targets = resolve_targets(a.comfyui_dir, doc)
        done, sync = install(targets, link_source=a.link_source, model=a.model)
    except ComfyError as e:
        print(f"error: {e}", file=sys.stderr)
        return e.code
    for line in install_lines(done, sync):
        print(line)
    if sync is not None and sync.state == "refreshed" and sync.old:
        print(f"updated the gmlx nodes from gmlx {sync.old.get('gmlx_version')} to "
              f"{sync.new['gmlx_version']}")
    print("Restart ComfyUI, or Comfy Desktop, if it is running, so it loads the nodes.")
    return 0


def _remove_link(cn: Path, dests: list) -> bool:
    entry = cn / LINK_NAME
    if entry.is_symlink() and (not entry.exists()
                               or any(_points_at(entry, d) for d in dests)):
        entry.unlink()
        return True
    return False


def _cmd_uninstall(argv: list, prog: str) -> int:
    ap = argparse.ArgumentParser(
        prog=prog,
        description="Remove the gmlx nodes from ComfyUI: the links in custom_nodes, the "
                    "copy gmlx keeps and the record. A folder that is not gmlx's link "
                    "stays.")
    ap.add_argument("--comfyui-dir", metavar="DIR",
                    help="Remove the nodes from this ComfyUI only.")
    a = ap.parse_args(argv)
    doc = load_record()
    dests = [copy_dir(), PACK_SOURCE]
    recorded = recorded_targets(doc)
    if a.comfyui_dir:
        d = Path(os.path.abspath(os.path.expanduser(a.comfyui_dir)))
        chosen = [t for t in recorded if t.matches(d)]
        if not chosen:
            try:
                chosen = [target_for_dir(d)]
            except ComfyError as e:
                print(f"error: {e}", file=sys.stderr)
                return e.code
    else:
        chosen = list(recorded)
        keys = {t.key() for t in chosen}
        chosen += [t for t in detect() if t.key() not in keys]
    removed = [t for t in chosen if _remove_link(t.custom_nodes, dests)]
    for t in removed:
        print(f"removed {_short(t.custom_nodes / LINK_NAME)}")
    gone = {t.key() for t in chosen}
    left = [r for r in doc.get("installs") or []
            if isinstance(r, dict) and r.get("custom_nodes")
            and os.path.realpath(r["custom_nodes"]) not in gone]
    if left:
        doc["installs"] = left
        save_record(doc)
        print(f"the gmlx nodes stay in {len(left)} other ComfyUI folder"
              f"{'s' if len(left) > 1 else ''}")
        return 0
    if data_root().exists():
        shutil.rmtree(data_root())
        print(f"removed {_short(data_root())}")
    for k in _RECORD_KEYS:
        doc.pop(k, None)
    if settings_path().exists():
        if doc:
            save_record(doc)
        else:
            settings_path().unlink()
            print(f"removed {_short(settings_path())}")
    if not removed:
        print("no gmlx node links to remove")
    return 0


def status(doc: dict | None = None) -> dict:
    """What is installed, as data: the copy, the record and each folder."""
    doc = load_record() if doc is None else doc
    meta = pack_meta()
    stamp = read_stamp(copy_dir())
    running = gmlx_version()
    folders = []
    for t in recorded_targets(doc):
        v = comfyui_version(t.code)
        exists = t.custom_nodes.parent.is_dir()
        state = link_state(t, doc) if exists else "folder missing"
        note = None
        if v and version_tuple(v) < version_tuple(meta["COMFYUI_FLOOR"]):
            note = f"older than {meta['COMFYUI_FLOOR']}, which the nodes need"
        elif v and version_tuple(v) > version_tuple(meta["COMFYUI_TESTED"]):
            note = f"newer than {meta['COMFYUI_TESTED']}, the newest this gmlx was tested on"
        folders.append({"custom_nodes": str(t.custom_nodes), "kind": t.kind,
                        "describe": t.describe(), "comfyui_version": v, "link": state,
                        "version_note": note})
    source = bool(doc.get("pack_source"))
    if source:
        copy = "source"
    elif stamp is None:
        copy = "missing"
    elif stamp.get("gmlx_version") != running:
        copy = "stale"
    else:
        copy = "current"
    return {"gmlx_version": running, "pack_version": meta["__version__"],
            "comfyui_floor": meta["COMFYUI_FLOOR"], "comfyui_tested": meta["COMFYUI_TESTED"],
            "copy": copy, "copy_dir": str(PACK_SOURCE if source else copy_dir()),
            "copy_gmlx_version": (stamp or {}).get("gmlx_version"),
            "gmlx_command": doc.get("gmlx_command"), "gmlx_install": doc.get("gmlx_install"),
            "default_image_model": doc.get("default_image_model"), "folders": folders,
            "installed": bool(folders)}


def status_problems(s: dict) -> list:
    out = []
    if s["copy"] == "missing":
        out.append("the pack copy is missing; run gmlx comfyui install")
    elif s["copy"] == "stale":
        out.append(f"the pack copy is from gmlx {s['copy_gmlx_version']}; run "
                   "gmlx comfyui install, or start gmlx serve")
    for f in s["folders"]:
        if f["link"] != "linked":
            out.append(f"{_short(f['custom_nodes'])}: {f['link']}; run "
                       "gmlx comfyui install")
        elif f["version_note"] and "older" in f["version_note"]:
            out.append(f"{_short(f['custom_nodes'])}: ComfyUI {f['comfyui_version']} is "
                       f"{f['version_note']}")
    return out


def _cmd_status(argv: list, prog: str) -> int:
    ap = argparse.ArgumentParser(prog=prog,
                                 description="Show where the gmlx nodes are installed and "
                                             "whether each ComfyUI fits them.")
    ap.add_argument("--json", action="store_true", help="Print the status as JSON.")
    a = ap.parse_args(argv)
    s = status()
    problems = status_problems(s) if s["installed"] else ["not installed"]
    if a.json:
        print(json.dumps({**s, "problems": problems}, indent=2))
        return 0 if not problems else 1
    if not s["installed"]:
        print("The gmlx nodes are not installed in any ComfyUI. Run gmlx comfyui install.")
        found = detect()
        if found:
            print("ComfyUI found here:")
            for t in found:
                print(f"  {_short(t.custom_nodes.parent)}  ({t.describe()})")
        return 1
    print(f"gmlx {s['gmlx_version']}, gmlx nodes {s['pack_version']}, for ComfyUI "
          f"{s['comfyui_floor']} or newer, tested to {s['comfyui_tested']}")
    copy = {"current": "current", "source": "the gmlx source tree",
            "missing": "missing",
            "stale": f"from gmlx {s['copy_gmlx_version']}"}[s["copy"]]
    print(f"pack: {_short(s['copy_dir'])} ({copy})")
    if s["gmlx_command"]:
        print(f"gmlx command: {s['gmlx_command']} ({s['gmlx_install']})")
    if s["default_image_model"]:
        print(f"default image model: {s['default_image_model']}")
    print("ComfyUI folders:")
    for f in s["folders"]:
        ver = f", ComfyUI {f['comfyui_version']}" if f["comfyui_version"] else ""
        note = f"; {f['version_note']}" if f["version_note"] else ""
        print(f"  {_short(f['custom_nodes'])}: {f['link']} ({f['describe']}{ver}){note}")
    print("The nodes are managed by gmlx and update with it.")
    for p in status_problems(s):
        print(f"warning: {p}")
    return 0 if not status_problems(s) else 1


# Callers.

def refresh_for_serve() -> str | None:
    """Refresh the copy at server start, only when an install record
    exists and every recorded ComfyUI folder exists on this filesystem.
    Returns a line to print, or None. Never raises."""
    try:
        doc = load_record()
        targets = recorded_targets(doc)
        if not targets or doc.get("pack_source"):
            return None
        if not all(t.custom_nodes.parent.is_dir() for t in targets):
            return None
        sync = sync_copy()
        cmd = doc.get("gmlx_command")
        if cmd and not os.path.exists(cmd):
            doc["gmlx_command"] = gmlx_command()
            save_record(doc)
    except Exception as e:  # noqa: BLE001 - the server starts whatever the pack's state
        return f"could not refresh the ComfyUI node pack: {e}"
    if sync.state == "current":
        return None
    old = (sync.old or {}).get("gmlx_version")
    line = (f"updated the ComfyUI node pack to gmlx {sync.new['gmlx_version']}"
            + (f" from {old}" if old else "") + "; restart ComfyUI to load it")
    return line + (f" ({sync.warning})" if sync.warning else "")


def remembered_dir(doc: dict | None = None) -> Path | None:
    """The ComfyUI folder that launch used or install linked last."""
    doc = load_record() if doc is None else doc
    d = doc.get("comfyui_dir")
    return Path(d) if isinstance(d, str) and (Path(d) / "main.py").is_file() else None


def _print_help(prog: str) -> None:
    lines = [f"usage: {prog} <action> [options]", "",
             "put the gmlx node pack into ComfyUI and keep it current.", "", "actions:"]
    for a in _ACTIONS:
        lines.append(f"  {a:<10} {_ACTION_DESC[a]}")
    lines += ["", f"run `{prog} <action> --help` for an action's options."]
    print("\n".join(lines))


HANDLERS = {"install": _cmd_install, "uninstall": _cmd_uninstall, "status": _cmd_status}


def cmd_comfyui(argv: list[str], prog: str = "gmlx comfyui") -> int:
    """Dispatch ``gmlx comfyui <action> ...``."""
    if not argv or argv[0] in ("-h", "--help", "help"):
        _print_help(prog)
        return 0
    action, rest = argv[0], argv[1:]
    if action not in HANDLERS:
        print(f"error: unknown action {action!r}; one of {', '.join(_ACTIONS)}",
              file=sys.stderr)
        return 2
    return HANDLERS[action](rest, prog=f"{prog} {action}")
