"""The mlx version gate, and the versions of the copied packages.

gmlx runs on stock mlx and carries mlx-vlm and mlx-lm as a copy
(``gmlx._vendor``). A venv resolves its requirements once, so an mlx older
than the copy needs can outlive a gmlx upgrade. ``check_mlx_version`` runs
at CLI entry and refuses it. ``copied_versions`` reads the copy's version
files without importing the packages, for ``gmlx doctor``.
"""
from __future__ import annotations

import os
import re
import sys

# The mlx floor the copied mlx-vlm declares. tests/upstream/test_versions.py
# holds it equal to the manifest's record of that requirement.
MLX_FLOOR = "0.32.2"
# gmlx's mlx-kquant requirement. Each mlx-kquant release pins the mlx its
# kernels were built against, so mlx moves with it. The test holds it equal
# to the pyproject dependency.
KQUANT_REQUIREMENT = "mlx-kquant>=0.5.8,<0.6"

_COPY_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "_vendor")
_VERSION_FILES = {"mlx-vlm": "mlx_vlm/version.py",
                  "mlx-lm": "mlx_lm/_version.py"}
_VERSION = re.compile(r"""^__version__\s*=\s*["']([^"']+)["']""", re.M)


def _parse_version(v: str) -> tuple[int, ...]:
    m = re.match(r"(\d+(?:\.\d+)*)", v)
    if not m:
        return ()
    return tuple(int(p) for p in m.group(1).split("."))


_WARNED = False


def check_mlx_version(quiet: bool = False) -> list[str]:
    """Raise RuntimeError when the installed mlx is below ``MLX_FLOOR``.
    Missing or unparsable metadata (a source install) warns and never
    raises. Returns the warning lines, printed once unless ``quiet``."""
    global _WARNED
    import importlib.metadata as md

    warnings = []
    try:
        found = md.version("mlx")
    except md.PackageNotFoundError:
        found = None
        warnings.append(
            f"[versions] mlx: no package metadata, as from a source "
            f"install; the supported floor >= {MLX_FLOOR} is not verified")
    if found is not None:
        found_t = _parse_version(found)
        if not found_t:
            warnings.append(
                f"[versions] mlx: unparsable version {found!r}; the "
                f"supported floor >= {MLX_FLOOR} is not verified")
        elif found_t < _parse_version(MLX_FLOOR):
            raise RuntimeError(
                f"mlx {found} is below the supported floor {MLX_FLOOR}; "
                f"{_upgrade_hint()}")
    if warnings and not quiet and not _WARNED:
        _WARNED = True
        for w in warnings:
            print(w, file=sys.stderr, flush=True)
    return warnings


def _upgrade_hint() -> str:
    """How to bring mlx up to the floor, by the route that installed gmlx.
    Only a pip install takes pip; a keg or a tool install is reinstalled.
    pip upgrades mlx through mlx-kquant, which pins the mlx its kernels
    were built against; an upgrade of mlx alone breaks that pin."""
    from gmlx.commands.extras import ROUTE_PIP, install_route, repair_hint

    cmd = repair_hint(f"-U '{KQUANT_REQUIREMENT}'")
    if install_route() == ROUTE_PIP:
        return (f"upgrade mlx-kquant, which brings the mlx it pins ({cmd}), "
                f"or reinstall gmlx into a fresh venv")
    return f"reinstall gmlx to upgrade it ({cmd})"


def copied_versions() -> dict[str, str]:
    """The upstream release each copied package was taken from, by
    distribution name; "?" for a version file that is missing or holds
    no version."""
    out = {}
    for dist, path in _VERSION_FILES.items():
        try:
            with open(os.path.join(_COPY_DIR, *path.split("/")),
                      encoding="utf-8") as f:
                m = _VERSION.search(f.read())
        except OSError:
            m = None
        out[dist] = m.group(1) if m else "?"
    return out
