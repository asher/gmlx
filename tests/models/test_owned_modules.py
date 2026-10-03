"""Every upstream name gmlx owns resolves to the gmlx module.

``gmlx.models.owned.OWNED_MODULES`` lists the mlx-lm / mlx-vlm import names
whose gmlx module the GGUF path needs. An upstream release may ship a module
under any of them: mlx-vlm 0.6.15 ships muse_glimmer, and mlx-lm 0.32.0 ships
muse_glimmer, kimi_k3 and deepseek_v41. With an upstream-first registration
that module won silently, and the loader built a class that could not encode a
GGUF image, refused the drafter, or decoded the GGUF weights into garbage.

Two cases per name. The first imports the name fresh, so the installed
upstream package is found on disk when it ships one, and resolves it through
the same upstream function the loader calls. The second puts a stand-in
upstream module in ``sys.modules`` first, so the guard holds for every name
whether or not the installed upstream version ships it yet. CPU-only.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types

import pytest

pytest.importorskip("mlx.core")

from gmlx.models.owned import OWNED_MODULES  # noqa: E402


def _upstream_ships(name: str) -> bool:
    """True when the installed upstream package has ``name`` on disk. Asks the
    path finder directly, so a module already in sys.modules does not count."""
    parent, _, leaf = name.rpartition(".")
    try:
        pkg = importlib.import_module(parent)
    except ImportError:
        return False
    return importlib.machinery.PathFinder.find_spec(leaf, pkg.__path__) is not None


_IDS = [n + (" (upstream ships)" if _upstream_ships(n) else "")
        for n in OWNED_MODULES]


def _resolve(name: str):
    """The module the upstream loader gets for ``name``."""
    parent, _, leaf = name.rpartition(".")
    if parent == "mlx_lm.models":
        from mlx_lm.utils import _get_classes

        model_cls, _ = _get_classes({"model_type": leaf})
        return sys.modules[model_cls.__module__]
    if parent == "mlx_vlm.models":
        from mlx_vlm.utils import get_model_and_args

        module, _ = get_model_and_args({"model_type": leaf})
        return module
    from mlx_vlm.tool_parsers import load_tool_module

    return load_tool_module(leaf)


def _forget(monkeypatch, name: str) -> None:
    parent, _, leaf = name.rpartition(".")
    monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delattr(importlib.import_module(parent), leaf, raising=False)


@pytest.mark.parametrize("name", list(OWNED_MODULES), ids=_IDS)
def test_upstream_loader_resolves_the_owned_module(monkeypatch, name):
    owner = importlib.import_module(OWNED_MODULES[name])
    _forget(monkeypatch, name)
    owner.ensure_registered()
    assert _resolve(name) is owner


@pytest.mark.parametrize("name", list(OWNED_MODULES), ids=_IDS)
def test_owned_module_replaces_an_imported_upstream_module(monkeypatch, name):
    owner = importlib.import_module(OWNED_MODULES[name])
    parent, _, leaf = name.rpartition(".")
    upstream = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, upstream)
    monkeypatch.setattr(importlib.import_module(parent), leaf, upstream,
                        raising=False)
    owner.ensure_registered()
    assert importlib.import_module(name) is owner
    assert getattr(importlib.import_module(parent), leaf) is owner
    assert _resolve(name) is owner
