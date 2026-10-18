"""Static parse of chat templates that use the renderer's extension tags."""

from __future__ import annotations

from pathlib import Path

import pytest

_K2 = Path(__file__).resolve().parents[1] / "fixtures" / "k2_horizon_template.jinja"

_GENERATION = (
    "{% for m in messages %}{% if m.role == 'assistant' %}"
    "{% generation %}{{ m.content }}{% endgeneration %}"
    "{% else %}{{ m.content }}{% if loop.index > 9 %}{% break %}{% endif %}"
    "{% endif %}{% endfor %}")


def test_plain_jinja_rejects_what_the_parser_accepts():
    import jinja2

    from gmlx.load.template_parse import parse_chat_template

    with pytest.raises(jinja2.TemplateSyntaxError):
        jinja2.Environment().parse(_GENERATION)
    parse_chat_template(_GENERATION)


def test_generation_body_stays_visible_to_meta():
    from jinja2 import meta

    from gmlx.load.template_parse import parse_chat_template

    names = meta.find_undeclared_variables(parse_chat_template(
        "{% generation %}{{ reasoning_effort }}{% endgeneration %}"))
    assert names == {"reasoning_effort"}


def test_parser_still_rejects_bad_jinja():
    import jinja2

    from gmlx.load.template_parse import parse_chat_template

    with pytest.raises(jinja2.TemplateSyntaxError):
        parse_chat_template("{% generation %}unclosed")


def test_chat_template_override_accepts_the_generation_tag(tmp_path):
    from gmlx.load.loader import _resolve_chat_template

    text = _K2.read_text()
    assert "{% generation %}" in text
    p = tmp_path / "k2.jinja"
    p.write_text(text)
    assert _resolve_chat_template(str(p)) == text
    assert _resolve_chat_template(_GENERATION) == _GENERATION
