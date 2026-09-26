"""scripts/mkdocs_hooks.py: the page description taken from a page's opening
paragraph."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mkdocs_hooks.py"


def _load():
    spec = importlib.util.spec_from_file_location("_mkdocs_hooks", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hooks = _load()


def test_description_is_the_first_paragraph_as_plain_text():
    html = ('<h1 id="x">X<a class="headerlink" href="#x">&para;</a></h1>\n'
            "<p>gmlx runs <code>GGUF</code> files, as <a href=\"a.html\">A</a> "
            "says. It is fast &amp; exact.</p>\n<p>Later.</p>")
    assert hooks.page_description(html) == \
        "gmlx runs GGUF files, as A says. It is fast & exact."


def test_description_drops_a_sentence_that_leads_into_a_block():
    html = "<p>One sentence. Then install it:</p><pre>brew</pre>"
    assert hooks.page_description(html) == "One sentence."


def test_description_skips_a_paragraph_with_no_sentence():
    html = "<p><img src=\"a.png\"></p><p>Install it:</p><p>Real text.</p>"
    assert hooks.page_description(html) == "Real text."


def test_description_stops_at_a_sentence_end_under_the_cap():
    first = "a" * 200 + "."
    html = f"<p>{first} {'b' * 150}.</p>"
    assert hooks.page_description(html) == first


def test_no_paragraph_gives_none():
    assert hooks.page_description("<h1>Only a title</h1>") is None
