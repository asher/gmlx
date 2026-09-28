"""Graft a gmlx module into an upstream package namespace."""

from __future__ import annotations

import importlib
import sys


def graft_module(name: str, module) -> None:
    """Make ``import <name>`` resolve to ``module``, ahead of an upstream
    module of that name, whether or not the upstream one was imported first.

    Binds ``sys.modules[name]`` and the attribute on the parent package,
    which ``from <parent> import <leaf>`` reads. Submodules of an upstream
    package under ``name`` stop resolving unless already imported.
    Idempotent. gmlx.upstream.seams lists every graft whose name upstream
    also ships, with the release that shipped it.
    """
    sys.modules[name] = module
    parent, _, leaf = name.rpartition(".")
    setattr(importlib.import_module(parent), leaf, module)
