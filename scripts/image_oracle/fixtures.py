#!/usr/bin/env python3
"""Oracle fixtures for the image acceptance checks: writer, reader and
manifest.

A fixture is one ``<name>.npz`` of named arrays and one
``<name>.manifest.json`` in the directory that GMLX_IMAGE_FIXTURES names.
The manifest records the oracle's package versions and revisions, the
model files by name, size and sha256, the inputs (prompts, noise seeds,
sigmas and sizes), the shape, dtype and sha256 of each array, the
oracle's own measurements and the calibrated tolerances. ``manifest.json``
in the same directory indexes every fixture there.

The repository keeps a copy of each manifest that a test reads, under
``tests/fixtures/image/oracle/``. ``load`` checks every array against the
manifest it is given, so a test runs only on the arrays that the
committed manifest describes.

This module needs only numpy and the standard library, so both the oracle
environment and the gmlx tests import it.

  python fixtures.py list
  python fixtures.py verify <name> [--manifest <committed copy>]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import fcntl
import hashlib
import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path

import numpy as np

ENV = "GMLX_IMAGE_FIXTURES"
INDEX = "manifest.json"
SCHEMA = 1

_DTYPES = frozenset({"float64", "float32", "float16", "int64", "int32",
                     "int16", "int8", "uint8", "uint16", "uint32", "bool"})
_ORACLE_PACKAGES = ("torch", "diffusers", "transformers", "tokenizers",
                    "accelerate", "huggingface-hub", "gguf", "safetensors",
                    "numpy")


class FixtureError(RuntimeError):
    pass


def fixture_root(path: str | os.PathLike | None = None, *,
                 create: bool = False) -> Path:
    """The fixture directory: ``path``, else GMLX_IMAGE_FIXTURES."""
    raw = path if path is not None else os.environ.get(ENV)
    if not raw:
        raise FixtureError(f"set {ENV} to the fixture directory")
    root = Path(raw).expanduser()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    elif not root.is_dir():
        raise FixtureError(f"{ENV}={raw!r} is not a directory")
    return root


def array_sha256(a: np.ndarray) -> str:
    """sha256 of the array's C-order little-endian bytes."""
    a = np.asarray(a)
    if a.dtype.byteorder == ">":
        a = a.astype(a.dtype.newbyteorder("<"))
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def file_identity(path: str | os.PathLike) -> dict:
    """Name, size and sha256 of a file."""
    p = Path(path).expanduser().resolve()
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return {"name": p.name, "size": p.stat().st_size,
            "sha256": h.hexdigest()}


def _revision(dist: metadata.Distribution) -> str | None:
    """The commit or archive URL a package was installed from, if any."""
    raw = dist.read_text("direct_url.json")
    if not raw:
        return None
    d = json.loads(raw)
    commit = d.get("vcs_info", {}).get("commit_id")
    return commit or d.get("url")


def package_versions(names=_ORACLE_PACKAGES) -> dict:
    out = {}
    for n in names:
        try:
            dist = metadata.distribution(n)
        except metadata.PackageNotFoundError:
            out[n] = None
            continue
        entry = {"version": dist.version}
        rev = _revision(dist)
        if rev:
            entry["source"] = rev
        out[n] = entry
    return out


def environment() -> dict:
    """The interpreter, platform and oracle packages of this process."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": package_versions(),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
    }


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def portable_argv(argv) -> list[str]:
    """argv with every path reduced to its file name, so a manifest holds
    no local directory."""
    out = []
    for a in map(str, argv):
        if a.startswith("-") and "=" in a:
            k, v = a.split("=", 1)
            out.append(f"{k}={Path(v).name}" if os.sep in v else a)
        else:
            out.append(Path(a).name if os.sep in a else a)
    return out


def _describe(a: np.ndarray) -> dict:
    return {"shape": list(a.shape), "dtype": a.dtype.name,
            "sha256": array_sha256(a)}


class FixtureWriter:
    """Collects arrays and manifest fields for one fixture, then writes the
    ``.npz``, its manifest and the index entry in one call."""

    def __init__(self, name: str, *, check: int | str, family: str,
                 script: str | None = None, argv=None):
        if not name or "/" in name or name.startswith("."):
            raise FixtureError(f"bad fixture name {name!r}")
        self.name = name
        self.arrays: dict[str, np.ndarray] = {}
        self.manifest: dict = {
            "schema": SCHEMA,
            "name": name,
            "check": check,
            "family": family,
            "script": script,
            "argv": portable_argv(sys.argv if argv is None else argv),
            "oracle": environment(),
            "files": {},
            "inputs": {},
            "results": {},
            "tolerances": {},
            "notes": [],
        }

    @property
    def inputs(self) -> dict:
        return self.manifest["inputs"]

    @property
    def results(self) -> dict:
        return self.manifest["results"]

    @property
    def tolerances(self) -> dict:
        return self.manifest["tolerances"]

    def note(self, text: str) -> None:
        self.manifest["notes"].append(text)

    def add_file(self, role: str, path: str | os.PathLike) -> dict:
        ident = file_identity(path)
        self.manifest["files"][role] = ident
        return ident

    def add(self, key: str, array) -> None:
        a = np.asarray(array)
        if a.dtype.name not in _DTYPES:
            raise FixtureError(f"{key}: dtype {a.dtype} cannot go in a "
                               "fixture; store float32 or float16")
        if key in self.arrays:
            raise FixtureError(f"{key} is already in {self.name}")
        # np.require keeps a 0-d array 0-d; ascontiguousarray would not.
        self.arrays[key] = np.require(a, requirements="C")

    def write(self, root: str | os.PathLike | None = None) -> Path:
        root = fixture_root(root, create=True)
        self.manifest["created"] = _now()
        self.manifest["npz"] = f"{self.name}.npz"
        self.manifest["tensors"] = {k: _describe(a)
                                    for k, a in sorted(self.arrays.items())}
        npz = root / f"{self.name}.npz"
        tmp = root / f".{self.name}.npz.tmp"
        with open(tmp, "wb") as f:
            np.savez(f, **self.arrays)
        os.replace(tmp, npz)
        path = root / f"{self.name}.manifest.json"
        _write_json(path, self.manifest)
        _update_index(root, self.name, self.manifest)
        return path


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=False) + "\n")
    os.replace(tmp, path)


def _update_index(root: Path, name: str, manifest: dict) -> None:
    """Record one fixture in the index under a lock, since oracle scripts
    for different checks can finish at the same time."""
    with open(root / ".manifest.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        index_path = root / INDEX
        index = (json.loads(index_path.read_text())
                 if index_path.exists() else {"schema": SCHEMA,
                                              "fixtures": {}})
        index["fixtures"][name] = {
            "manifest": f"{name}.manifest.json",
            "npz": manifest["npz"],
            "check": manifest["check"],
            "family": manifest["family"],
            "created": manifest["created"],
            "tensors": len(manifest["tensors"]),
        }
        _write_json(index_path, index)


def read_manifest(name: str, *, root=None, manifest=None) -> dict:
    """The manifest for ``name``: a dict or path given as ``manifest``,
    else the one written beside the fixture."""
    if isinstance(manifest, dict):
        return manifest
    path = (Path(manifest) if manifest is not None
            else fixture_root(root) / f"{name}.manifest.json")
    if not path.is_file():
        raise FixtureError(f"no manifest at {path}")
    return json.loads(path.read_text())


def load(name: str, *, root=None, manifest=None
         ) -> tuple[dict[str, np.ndarray], dict]:
    """Arrays of fixture ``name`` and its manifest. Every array the
    manifest lists must be present with the same shape, dtype and sha256,
    and the file must hold no other array."""
    m = read_manifest(name, root=root, manifest=manifest)
    npz = fixture_root(root) / m.get("npz", f"{name}.npz")
    if not npz.is_file():
        raise FixtureError(f"{npz} is missing. Run the oracle script "
                           f"{m.get('script')} to write it")
    with np.load(npz, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    want = m.get("tensors", {})
    bad = []
    for k, d in want.items():
        a = arrays.get(k)
        if a is None:
            bad.append(f"{k} missing")
        elif (list(a.shape) != d["shape"] or a.dtype.name != d["dtype"]
              or array_sha256(a) != d["sha256"]):
            bad.append(f"{k} differs")
    bad += [f"{k} not in the manifest" for k in sorted(set(arrays) - set(want))]
    if bad:
        shown = ", ".join(bad[:5]) + (f" (+{len(bad) - 5} more)"
                                      if len(bad) > 5 else "")
        raise FixtureError(f"{npz.name} does not match its manifest: {shown}")
    return arrays, m


def rel_l2(got, ref) -> float:
    g = np.asarray(got, np.float64)
    r = np.asarray(ref, np.float64)
    return float(np.linalg.norm(g - r) / np.linalg.norm(r))


def cosine(got, ref) -> float:
    g = np.asarray(got, np.float64).ravel()
    r = np.asarray(ref, np.float64).ravel()
    return float(g @ r / (np.linalg.norm(g) * np.linalg.norm(r)))


def max_abs(got, ref) -> float:
    return float(np.max(np.abs(np.asarray(got, np.float64)
                               - np.asarray(ref, np.float64))))


def psnr(got, ref, peak: float) -> float:
    err = np.mean((np.asarray(got, np.float64)
                   - np.asarray(ref, np.float64)) ** 2)
    return float("inf") if err == 0 else float(10 * np.log10(peak ** 2 / err))


def compare(got, ref) -> dict:
    return {"rel_l2": rel_l2(got, ref), "cosine": cosine(got, ref),
            "max_abs": max_abs(got, ref)}


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", help=f"fixture directory (default ${ENV})")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list the fixtures in the index")
    v = sub.add_parser("verify", help="check a fixture's arrays against a "
                       "manifest")
    v.add_argument("name")
    v.add_argument("--manifest", help="manifest to check against (default: "
                   "the one beside the fixture)")
    a = ap.parse_args(argv)
    root = fixture_root(a.root)
    if a.cmd == "list":
        index = json.loads((root / INDEX).read_text())
        for n, e in sorted(index["fixtures"].items()):
            print(f"{n}  check {e['check']}  {e['family']}  "
                  f"{e['tensors']} arrays  {e['created']}")
        return 0
    arrays, m = load(a.name, root=root, manifest=a.manifest)
    print(f"{a.name}: {len(arrays)} arrays match the manifest "
          f"(created {m.get('created')})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(_main())
    except FixtureError as e:
        sys.exit(f"error: {e}")
