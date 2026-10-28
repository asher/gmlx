"""The assistant tools in examples/assistant-tools: a script that a doc
page prints in full matches the file, and the folder's README lists every
script. CPU-only; no model."""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_TOOLS = _ROOT / "examples" / "assistant-tools"
# Doc section -> the script it prints.
_PRINTED = {("docs/assistant.md", "### Decisions as a tool"): "decide-tool.py"}


def _scripts():
    return sorted(p.name for p in _TOOLS.glob("*.py"))


@pytest.mark.parametrize("page,heading", list(_PRINTED))
def test_printed_script_matches_the_file(page, heading):
    after = (_ROOT / page).read_text().split(heading, 1)[1]
    code = re.search(r"```python\n(.*?)```", after, re.S)
    assert code, f"{page} {heading}: no python block"
    assert code.group(1) == (_TOOLS / _PRINTED[(page, heading)]).read_text()


@pytest.mark.parametrize("name", _scripts())
def test_script_parses_and_declares_its_dependencies(name):
    text = (_TOOLS / name).read_text()
    ast.parse(text)
    assert text.startswith("# /// script\n"), name
    assert text.isascii(), name


def test_readme_lists_every_script():
    readme = (_TOOLS / "README.md").read_text()
    assert readme.isascii()
    assert [n for n in _scripts() if f"`{n}`" not in readme] == []
