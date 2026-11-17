"""The docs name the pull catalog: the table in docs/cli.md lists every
catalog model with its params, license and blurb, the quickstart's
memory table names the first RECOMMENDED pick of every memory class, and
docs/images.md gives each image model's download as pull prints it. Each
fails when the catalog moves on. CPU-only."""
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
        assert got[name] == [m.params, catalog.license_text(m), m.use], name


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


def test_images_doc_gives_each_image_download_as_pull_does():
    """The Choose a model table in docs/images.md and its sentence on the
    model that needs more memory give each download as `gmlx pull --list`
    prints it on that Mac: every file of the class's quants, the encoder's
    vision tower with it."""
    from gmlx.image import catalog as image

    text = (_DOCS / "images.md").read_text()
    header = "| Model | Download, 16 GB Mac | Download, 24 GB Mac |"
    classes = [int(n) for n in re.findall(r"(\d+) GB Mac", header)]
    rows = _table_rows(text, header)
    got = {r[0].strip("`"): r[1:3] for r in rows}
    assert set(got) == set(image.RECIPES)
    for name, cells in got.items():
        r = image.RECIPES[name]
        want = [f"Needs {r.least_class} GB" if r.least_class > c else
                catalog.human_gb(image.download_bytes(r, c << 30))
                for c in classes]
        assert cells == want, name
    big = [r for r in image.RECIPES.values() if r.least_class > classes[-1]]
    assert big
    flat = " ".join(text.split())
    for r in big:
        said = re.search(rf"{re.escape(r.title)} needs a Mac with (\d+) GB or "
                         rf"more\. Its download is ([\d.]+ GB) there, and "
                         rf"([\d.]+ GB) on a Mac with (\d+) GB or more\.",
                         flat)
        assert said, r.name
        least, there, more, upper = said.groups()
        assert int(least) == r.least_class, r.name
        for c in image.MEMORY_CLASSES:
            if c >= r.least_class:
                assert catalog.human_gb(image.download_bytes(r, c << 30)) == (
                    there if c < int(upper) else more), (r.name, c)
