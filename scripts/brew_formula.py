#!/usr/bin/env python3
"""Generate the Homebrew formula for a released gmlx version.

Resolves ``gmlx[all]==VERSION`` for macOS arm64 with uv, picks one PyPI file
per pinned package, and renders ``gmlx.rb``: the gmlx sdist as the formula
source and every other package as a hashed resource. The formula installs
the resources into a private venv with an offline pip install, so each
formula version carries its complete dependency set and an upgrade builds a
new venv instead of changing packages in place.

  python scripts/brew_formula.py 0.4.16                 # print the formula
  python scripts/brew_formula.py 0.4.16 -o Formula/gmlx.rb

Needs uv on PATH and network access to PyPI. The version must already be on
PyPI.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

DIST = "gmlx"
EXTRAS = "all"
TAP = "asher/gmlx"
MACOS_FLOOR = (26, 0)          # mlx-kquant publishes macosx_26_0 wheels only
MACOS_SYMBOL = ":tahoe"
# The mlx-kquant Metal library targets 26.2, which the wheel tag cannot say.
MACOS_KERNELS = "26.2"
# Build requirement for the sdists (gmlx itself, docopt, rumps). The formula
# builds them with --no-build-isolation, so setuptools must be in the set.
BUILD_REQUIREMENTS = ["setuptools>=77"]

_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)")
_MACOS_PLAT = re.compile(r"^macosx_(\d+)_(\d+)_(arm64|universal2)$")


def resolve(version: str, python: str) -> list[tuple[str, str]]:
    """``(name, version)`` for every package ``gmlx[all]==version`` needs on
    macOS arm64 with the given CPython, gmlx included."""
    with tempfile.TemporaryDirectory() as tmp:
        req = Path(tmp) / "requirements.in"
        req.write_text("\n".join([f"{DIST}[{EXTRAS}]=={version}",
                                  *BUILD_REQUIREMENTS]) + "\n")
        env = dict(os.environ,
                   MACOSX_DEPLOYMENT_TARGET="%d.%d" % MACOS_FLOOR)
        out = subprocess.run(
            ["uv", "pip", "compile", str(req), "--quiet", "--refresh",
             "--no-header", "--no-annotate",
             "--python-version", python,
             "--python-platform", "aarch64-apple-darwin"],
            env=env, check=True, capture_output=True, text=True).stdout
    return parse_pins(out)


def parse_pins(text: str) -> list[tuple[str, str]]:
    """``name==version`` lines of a compiled requirements file."""
    pins = []
    for line in text.splitlines():
        m = _PIN.match(line.strip())
        if m:
            pins.append((m.group(1), m.group(2)))
    return pins


def _wheel_tags(filename: str) -> tuple[list[str], list[str], list[str]] | None:
    """Python, ABI and platform tags of a wheel filename, or None."""
    if not filename.endswith(".whl"):
        return None
    parts = filename[:-4].split("-")
    if len(parts) not in (5, 6):
        return None
    py, abi, plat = parts[-3:]
    return py.split("."), abi.split("."), plat.split(".")


def wheel_rank(filename: str, python: str) -> tuple | None:
    """Sort key for a wheel usable on macOS arm64 at the floor with CPython
    ``python`` (for example ``"3.13"``), higher is better, or None when the
    wheel cannot be installed there.

    The order is a native wheel for this CPython, then an abi3 wheel, then a
    pure-Python wheel. Among native wheels the newest macOS tag wins, since
    it is built against the SDK the formula requires."""
    tags = _wheel_tags(filename)
    if tags is None:
        return None
    pys, abis, plats = tags
    major, minor = (int(x) for x in python.split("."))
    cp = f"cp{major}{minor}"

    plat_vers = [(0, 0)] if "any" in plats else []
    for plat in plats:
        m = _MACOS_PLAT.match(plat)
        if m and (int(m.group(1)), int(m.group(2))) <= MACOS_FLOOR:
            plat_vers.append((int(m.group(1)), int(m.group(2))))
    if not plat_vers:
        return None

    cp_minors = [int(p[len(f"cp{major}"):]) for p in pys
                 if p.startswith(f"cp{major}") and p[len(f"cp{major}"):].isdigit()]
    if cp in pys and cp in abis:
        kind = 3
    elif "abi3" in abis and any(m <= minor for m in cp_minors):
        kind = 2
    elif "none" in abis and any(p in (f"py{major}", f"py{major}{minor}")
                                for p in pys):
        kind = 1
    else:
        return None
    return kind, max(plat_vers)


def pick_file(files: list[dict], python: str) -> dict | None:
    """The PyPI file to use for one release: the best-ranked wheel, else the
    sdist, else None. ``files`` are entries of the PyPI JSON ``urls`` list."""
    ranked = [(wheel_rank(f["filename"], python), f) for f in files]
    ranked = [(r, f) for r, f in ranked if r is not None]
    if ranked:
        return max(ranked, key=lambda rf: rf[0])[1]
    for f in files:
        if f.get("packagetype") == "sdist":
            return f
    return None


def _release_files(name: str, version: str, *, attempts: int = 5) -> list[dict]:
    """The PyPI JSON ``urls`` list for one release. Retries on 404 with a
    growing delay, because a release just published can take a few minutes
    to appear in the JSON API."""
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.load(resp)["urls"]
        except urllib.error.HTTPError as exc:
            if exc.code != 404 or attempt == attempts - 1:
                raise
        time.sleep(15 * (attempt + 1))
    raise AssertionError("unreachable")


def collect(pins: list[tuple[str, str]], python: str) -> tuple[dict, list[dict]]:
    """``(gmlx sdist, [resource, ...])``, each a dict with name, url and
    sha256. Exits when any package has no usable file."""
    source, resources, missing = None, [], []
    for name, version in pins:
        files = _release_files(name, version)
        if name.lower() == DIST:
            chosen = next((f for f in files if f.get("packagetype") == "sdist"),
                          None)
        else:
            chosen = pick_file(files, python)
        if chosen is None:
            missing.append(f"{name}=={version}")
            continue
        entry = {"name": name.lower(), "url": chosen["url"],
                 "sha256": chosen["digests"]["sha256"]}
        if name.lower() == DIST:
            source = entry
        else:
            resources.append(entry)
    if missing or source is None:
        sys.exit("no usable file on PyPI for: "
                 + ", ".join(missing or [DIST + " sdist"]))
    return source, resources


def render(source: dict, resources: list[dict], python: str) -> str:
    """The formula text."""
    py_formula = f"python@{python}"
    blocks = "\n".join(
        f'  resource "{r["name"]}" do\n'
        f'    url "{r["url"]}"\n'
        f'    sha256 "{r["sha256"]}"\n'
        f"  end\n"
        for r in sorted(resources, key=lambda r: r["name"]))
    return f'''\
# Generated by scripts/brew_formula.py in the gmlx repository. Do not edit.
class Gmlx < Formula
  include Language::Python::Virtualenv

  desc "Run, serve and fine-tune GGUF models natively on MLX"
  homepage "https://github.com/asher/gmlx"
  url "{source["url"]}"
  sha256 "{source["sha256"]}"
  license all_of: ["BUSL-1.1", "MIT", "Apache-2.0"]

  depends_on arch: :arm64
  depends_on "ffmpeg"
  depends_on "libyaml"
  depends_on macos: {MACOS_SYMBOL}
  depends_on "{py_formula}"

  # Wheels carry prebuilt extension modules with @rpath install names, some
  # without header room for a longer one. Rewriting them to opt/ paths fails.
  preserve_rpath

{blocks}
  def install
    odie "gmlx needs macOS {MACOS_KERNELS} or newer" if MacOS.full_version < "{MACOS_KERNELS}"

    venv = virtualenv_create(libexec, "python{python}", system_site_packages: false)
    # Keep the venv isolated from Python packages of other formulae.
    deps_pth = venv.site_packages/"homebrew_deps.pth"
    rm deps_pth if deps_pth.exist?

    dist = buildpath/"homebrew-dist"
    dist.mkpath
    resources.each {{ |r| cp r.cached_download, dist/r.downloader.basename }}

    # An offline install from the pinned files: resolution fails if the set is
    # incomplete, so no package can come from anywhere else.
    python = formula_opt_libexec("{py_formula}")/"bin/python"
    pip = [python, "-m", "pip", "--python=#{{libexec}}/bin/python"]
    install = ["install", "--no-index", "--find-links=#{{dist}}", "--no-build-isolation"]
    system(*pip, *install, "setuptools")
    system(*pip, *install, "#{{buildpath}}[{EXTRAS}]")
    system(*pip, "check")

    bin.install_symlink libexec/"bin/gmlx"
  end

  def caveats
    <<~EOS
      A gmlx server or login item that is running during an upgrade keeps the
      old version. Restart it after `brew upgrade gmlx`:
        gmlx restart
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{{bin}}/gmlx --version")
    system libexec/"bin/python", "-c", "import mlx_kquant, mlx_vlm, gmlx.commands.cli"
  end
end
'''


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("version", help="gmlx release on PyPI, for example 0.4.16")
    ap.add_argument("--python", default="3.13",
                    help="Homebrew python@X.Y the venv uses (default 3.13)")
    ap.add_argument("-o", "--output", type=Path,
                    help="write the formula here instead of stdout")
    args = ap.parse_args(argv)

    source, resources = collect(resolve(args.version, args.python), args.python)
    text = render(source, resources, args.python)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"wrote {args.output} ({len(resources)} resources)",
              file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
