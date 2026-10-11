#!/usr/bin/env python3
"""Tripwire for `docs/long-context.md`: the architecture table lists the
architectures the rope scaling options reach, and the config example builds
the override it shows. CPU-only; no model."""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest

pytest.importorskip("yaml")
import yaml  # noqa: E402

from gmlx import config  # noqa: E402
from gmlx.load import rope_scaling as rs  # noqa: E402

_DOC = Path(__file__).resolve().parent.parent / "docs" / "long-context.md"


def _section(title: str) -> str:
    return _DOC.read_text().split(f"\n## {title}\n", 1)[1].split("\n## ", 1)[0]


def test_arch_table_lists_the_wired_arches():
    """A load with the options refuses on an arch outside WIRED_ARCHES, so
    the table names exactly those."""
    listed = set()
    for line in _section("Supported architectures").splitlines():
        if line.startswith("| `"):
            listed |= set(re.findall(r"`([^`]+)`", line.split("|")[1]))
    assert listed == set(rs.WIRED_ARCHES)


def test_config_example_builds_the_override():
    block = re.search(r"```yaml\n(.*?)```", _DOC.read_text(), re.DOTALL)[1]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = config.build_config(yaml.safe_load(block))
    ov = cfg.models["qwen3.8-flash-next-512k"].rope_scaling
    assert (ov.rope_scaling, ov.rope_scale, ov.yarn_orig_ctx) == ("yarn", 2.0, 262144)
