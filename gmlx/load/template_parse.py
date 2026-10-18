# SPDX-License-Identifier: MIT
"""Parse a chat template with the tags its renderer accepts.

transformers renders chat templates with ``loopcontrols`` and its own
``{% generation %}`` block (the assistant-mask tracker). A plain
``jinja2.Environment`` rejects both, so a static parse of such a template
fails although the template renders. This parser accepts the same tags.
The generation block parses as its body, so the variables inside it stay
visible to ``jinja2.meta``.
"""

from __future__ import annotations

import functools


@functools.cache
def _environment():
    import jinja2
    from jinja2.ext import Extension

    class _Generation(Extension):
        tags = {"generation"}

        def parse(self, parser):
            next(parser.stream)
            return parser.parse_statements(
                ("name:endgeneration",), drop_needle=True)

    return jinja2.Environment(
        extensions=[_Generation, "jinja2.ext.loopcontrols"])


def parse_chat_template(source: str):
    """The Jinja AST of ``source``. Raises ``jinja2.TemplateSyntaxError``
    on a template the renderer would also reject."""
    return _environment().parse(source)
