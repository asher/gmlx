#!/usr/bin/env python3
"""Tripwire for `docs/python.md`: every name in the package's stable surface
(`gmlx.__all__`) must appear in the doc, and every kwarg the doc tables
list for `load_model` / `generate` / `bench` must exist in the live signature,
so a renamed export or parameter cannot silently drift the reference.
CPU-only; no model."""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

_DOC = Path(__file__).resolve().parent.parent / "docs" / "python.md"

# First backticked cell of a kwarg-table row; compound rows split on " / ".
_ROW = re.compile(r"^\| `([A-Za-z_0-9]+)`(?: / `([A-Za-z_0-9]+)`)?", re.M)


def _section(doc: str, start: str, end: str) -> str:
    body = doc.split(start, 1)[1]
    return body.split(end, 1)[0]


def _documented_kwargs(section: str) -> set:
    names = set()
    for m in _ROW.finditer(section):
        names.update(n for n in m.groups() if n)
    return names


def test_doc_exists():
    assert _DOC.is_file(), f"missing {_DOC}"


def test_every_export_documented():
    import gmlx

    doc = _DOC.read_text()
    missing = [name for name in gmlx.__all__ if name not in doc]
    assert not missing, f"exports missing from docs/python.md: {missing}"


@pytest.mark.parametrize(
    "start,end,func_name",
    [
        ("## Load a model", "## Generate", "load_model"),
        ("## Generate", "## Benchmark", "generate"),
        ("## Benchmark", "## Preflight", "bench"),
    ],
)
def test_documented_kwargs_exist(start, end, func_name):
    """Every kwarg named in the doc's tables is a real parameter today."""
    pytest.importorskip("mlx_lm")
    import gmlx

    func = getattr(gmlx, func_name)
    params = set(inspect.signature(func).parameters)
    documented = _documented_kwargs(_section(_DOC.read_text(), start, end))
    assert documented, f"no kwarg rows found under {start!r}"
    unknown = documented - params
    assert not unknown, (
        f"docs/python.md lists kwargs {sorted(unknown)} that {func_name}() "
        "no longer has"
    )


# Default cells that are prose, not a literal (defaults resolved at runtime).
_NON_LITERAL_DEFAULTS = {"detected", "from GGUF", "model-aware"}

# Full kwarg-table row: names cell + default cell.
_ROW_WITH_DEFAULT = re.compile(
    r"^\| `([A-Za-z_0-9]+)`(?: / `([A-Za-z_0-9]+)`)? \| ([^|]+) \|", re.M)


@pytest.mark.parametrize(
    "start,end,func_name",
    [
        ("## Load a model", "## Generate", "load_model"),
        ("## Generate", "## Benchmark", "generate"),
        ("## Benchmark", "## Preflight", "bench"),
    ],
)
def test_python_doc_defaults_match_signature(start, end, func_name):
    """Every literal default the doc tables state equals the live signature's
    default (a compound `a` / `b` row states one default for both names)."""
    import ast

    pytest.importorskip("mlx_lm")
    import gmlx

    params = inspect.signature(getattr(gmlx, func_name)).parameters
    section = _section(_DOC.read_text(), start, end)
    checked = 0
    for m in _ROW_WITH_DEFAULT.finditer(section):
        cell = m.group(3).strip()
        if cell in _NON_LITERAL_DEFAULTS:
            continue
        lit = re.fullmatch(r"`([^`]+)`", cell)
        assert lit, f"{func_name}: unparseable default cell {cell!r}"
        doc_default = ast.literal_eval(lit.group(1))
        for name in (m.group(1), m.group(2)):
            if name is None:
                continue
            assert params[name].default == doc_default, (
                f"docs/python.md says {func_name}({name}={doc_default!r}) "
                f"but the signature default is {params[name].default!r}"
            )
            checked += 1
    assert checked, f"no literal default cells found under {start!r}"


def test_preflight_fields_documented():
    """The Preflight dataclass fields named in the doc still exist."""
    pytest.importorskip("mlx_lm")
    from gmlx.preflight import Preflight

    doc_named = {"arch", "shards", "codec_histogram", "n_tensors", "n_params"}
    fields = set(Preflight.__dataclass_fields__)
    assert doc_named <= fields, f"drifted: {sorted(doc_named - fields)}"


def test_doc_is_ascii():
    raw = _DOC.read_bytes()
    bad = [i for i, b in enumerate(raw) if b > 0x7F]
    assert not bad, f"non-ASCII bytes in docs/python.md at offsets {bad[:5]}"
