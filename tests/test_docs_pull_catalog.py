"""The docs name the pull catalog: the table in docs/cli.md lists every
catalog model with its params, license and blurb, and the quickstart's
memory table names the first RECOMMENDED pick of every memory class. Both
fail when the catalog moves on. CPU-only."""
from __future__ import annotations

import re
from pathlib import Path

from gmlx.load import catalog

_DOCS = Path(__file__).resolve().parent.parent / "docs"


def _table_rows(text: str, header: str) -> list:
    """The cells of each row of the table whose header line is ``header``."""
    at = text.index(header)
    rows = []
    for line in text[at:].splitlines()[2:]:
        if not line.startswith("|"):
            break
        rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows


def test_cli_doc_lists_every_catalog_model():
    rows = _table_rows((_DOCS / "cli.md").read_text(), "| Name | Params | License | Good for |")
    got = {r[0].strip("`"): r[1:] for r in rows}
    assert list(got) == list(catalog.MODELS), "the table follows the catalog's order"
    for name, m in catalog.MODELS.items():
        assert got[name] == [m.params, catalog.license_text(m), m.blurb], name


def test_quickstart_names_the_first_pick_of_each_memory_class():
    rows = _table_rows((_DOCS / "quickstart.md").read_text(), "| Mac memory | Model | Command |")
    by_class = {}
    for label, title, command in rows:
        for cls in re.findall(r"\d+", label):
            by_class[int(cls)] = (title, command)
    assert set(by_class) == set(catalog.RECOMMENDED)
    for cls, picks in catalog.RECOMMENDED.items():
        m = catalog.MODELS[picks[0]]
        assert by_class[cls] == (m.title, f"`gmlx pull {m.name}`"), cls
