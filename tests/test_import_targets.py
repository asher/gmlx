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
