"""Every relative import in the gmlx tree names a module that exists.

Vendored files are the motivating class: a module copied out of another
package keeps its package-relative imports, which then resolve inside gmlx
instead (models/vlm_text_only.py, vendored from mlx-vlm 0.6.4
models/text_only.py, once kept `from .cache import KVCache`). The hazard has
sharpened since the package restructure: gmlx.cache is now a real package,
so a stale relative import at the wrong depth can silently bind an existing
gmlx module instead of failing. When the import sits inside a function only
a live server exercises, no unit test trips it - so this resolves the target
purely on the filesystem, without importing anything: function-local imports
and modules with heavy optional dependencies (the misaki vendor tree) are
all checked the same way.
"""
import ast
from pathlib import Path

import gmlx

_ROOT = Path(gmlx.__file__).parent


def _target_exists(base_dir: Path, dotted: str) -> bool:
    p = base_dir.joinpath(*dotted.split("."))
    return p.with_suffix(".py").is_file() or (p / "__init__.py").is_file()


def test_relative_import_targets_exist():
    bad = []
    for py in sorted(_ROOT.rglob("*.py")):
        pkg_dir = py.parent
        tree = ast.parse(py.read_text(), filename=str(py))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ImportFrom) and node.level):
                continue
            base = pkg_dir
            for _ in range(node.level - 1):
                base = base.parent
            if node.module:
                # from .mod import name / from ..pkg.mod import name
                if not _target_exists(base, node.module):
                    bad.append(f"{py.relative_to(_ROOT.parent)}:{node.lineno}"
                               f" -> {node.module!r} not under {base}")
            else:
                # from . import name: each name must be a submodule or an
                # attribute of the package; accept either (attributes can't
                # be checked without importing), but the package must exist.
                if not (base / "__init__.py").is_file():
                    bad.append(f"{py.relative_to(_ROOT.parent)}:{node.lineno}"
                               f" -> package {base} has no __init__.py")
    assert not bad, "unresolvable relative imports:\n" + "\n".join(bad)


def test_every_upstream_graft_goes_through_the_owned_table():
    """A module binds an upstream import name only through
    ``gmlx.models.owned.install``, and only for a row of ``OWNED_MODULES``
    that names it as the owner. Every row has its install call.

    A direct ``sys.modules`` write is how an upstream-first graft hides: it
    stays out of the table, so the resolution guard in
    tests/models/test_owned_modules.py never checks it. Scanned on the
    filesystem, so a graft inside a function counts.
    """
    import re

    from gmlx.models.owned import OWNED_MODULES

    # A module writing itself, or anything, under an mlx_lm / mlx_vlm name.
    direct = (
        re.compile(r'sys\.modules\[[^\]]+\]\s*=\s*sys\.modules\[__name__\]'),
        re.compile(r'sys\.modules\.setdefault\(\s*f?"mlx_(?:vlm|lm)\.'),
        re.compile(r'sys\.modules\[\s*f?"mlx_(?:vlm|lm)\.[^"]+"\s*\]\s*=[^=]'),
    )
    install = re.compile(r'owned\.install\(\s*f?"([^"]+)",\s*__name__\s*\)')
    found = {}
    bad = []
    for py in sorted(_ROOT.rglob("*.py")):
        mod = "gmlx." + str(
            py.relative_to(_ROOT).with_suffix("")).replace("/", ".")
        mod = mod.removesuffix(".__init__")
        text = py.read_text()
        for pattern in direct:
            for m in pattern.finditer(text):
                bad.append(f"{mod}: direct sys.modules graft {m.group(0)!r}")
        for name in install.findall(text):
            # The two VLM containers install f"mlx_vlm.models.{MODEL_TYPE}".
            if "{MODEL_TYPE}" in name:
                name = [n for n, o in OWNED_MODULES.items() if o == mod]
                name = name[0] if name else f"<no row for {mod}>"
            found[name] = mod
            if OWNED_MODULES.get(name) != mod:
                bad.append(f"{mod}: installs {name}, which it does not own")
    for name, owner in OWNED_MODULES.items():
        if found.get(name) != owner:
            bad.append(f"{owner}: no owned.install call for {name}")
    assert not bad, "\n".join(bad)
