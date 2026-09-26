"""MkDocs hooks for the docs site built from docs/ (see mkdocs.yml).

A relative link from a page to a file outside docs/, such as ../README.md,
works on GitHub but has no page on the site. The hook points each one at
the file on GitHub, at the release tag when the build runs for a tag.

A long page opens with a list of links to its own sections for readers on
GitHub. The site shows the same list beside the page, so the hook removes
the list from the page body.

A `# doctest: build` line that opens a code block marks the block for
tests/test_docs_config.py. The hook removes the marker from the site page.

Each page opens with the sentences that say what it covers. The hook makes
them the page description, which the search snippet and the Open Graph
card in overrides/main.html show, and drops a closing sentence that leads
into a list or a code block.
"""

from __future__ import annotations

import html
import os
import posixpath
import re

_REPO_URL = "https://github.com/asher/gmlx"
_FENCE = re.compile(r"^\s*(```|~~~)")
_DOCTEST_MARKER = "# doctest: build"
_CONTENTS_ITEM = re.compile(r"[-*] \[[^\]]+\]\(#[^)\s]+\)", re.DOTALL)
_LINK = re.compile(r"(\]\(|\b(?:src|href)=\")([^)\"\s#]+)(#[^)\"\s]*)?")
_PARAGRAPH = re.compile(r"<p>(.*?)</p>", re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_DESCRIPTION_MAX = 300


def _ref() -> str:
    if os.environ.get("GITHUB_REF_TYPE") == "tag":
        return os.environ["GITHUB_REF_NAME"]
    return "main"


def _outside_url(page_src: str, target: str, docs_dir: str) -> str | None:
    """The GitHub URL for a relative target outside docs/, else None."""
    if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("/"):
        return None
    repo_root = os.path.dirname(docs_dir)
    path = posixpath.normpath(posixpath.join("docs", posixpath.dirname(page_src), target))
    if path == "docs" or path.startswith("docs/"):
        return None
    kind = "tree" if os.path.isdir(os.path.join(repo_root, path)) else "blob"
    return f"{_REPO_URL}/{kind}/{_ref()}/{path.rstrip('/')}"


def _drop_contents_list(markdown: str) -> str:
    """Remove the first list before the first section heading whose items
    all link to an anchor on the same page."""
    blocks = markdown.split("\n\n")
    for i, block in enumerate(blocks):
        if block.startswith("## "):
            break
        items = re.split(r"\n(?=[-*] )", block)
        if all(_CONTENTS_ITEM.fullmatch(item) for item in items):
            return "\n\n".join(blocks[:i] + blocks[i + 1 :])
    return markdown


def on_page_markdown(markdown, page, config, files):
    markdown = _drop_contents_list(markdown)
    docs_dir = config["docs_dir"]
    out = []
    in_fence = False
    fence_opened = False
    for line in markdown.split("\n"):
        if fence_opened and line.strip() == _DOCTEST_MARKER:
            fence_opened = False
            continue
        fence_opened = False
        if _FENCE.match(line):
            in_fence = not in_fence
            fence_opened = in_fence
        elif not in_fence:

            def repl(m):
                url = _outside_url(page.file.src_uri, m.group(2), docs_dir)
                if url is None:
                    return m.group(0)
                return m.group(1) + url + (m.group(3) or "")

            line = _LINK.sub(repl, line)
        out.append(line)
    return "\n".join(out)


def page_description(page_html: str) -> str | None:
    """The first paragraph of text as plain sentences, without a sentence
    that ends in a colon, cut at a sentence end to fit _DESCRIPTION_MAX
    characters. None when the page has no such text."""
    for m in _PARAGRAPH.finditer(page_html):
        text = " ".join(html.unescape(_TAG.sub("", m.group(1))).split())
        sentences = [x for x in _SENTENCE_END.split(text) if x and not x.endswith(":")]
        if not sentences:
            continue
        out = sentences[0]
        for sentence in sentences[1:]:
            if len(out) + 1 + len(sentence) > _DESCRIPTION_MAX:
                break
            out += " " + sentence
        return out
    return None


def on_page_content(page_html, page, config, files):
    if not page.meta.get("description"):
        description = page_description(page_html)
        if description:
            page.meta["description"] = description
    return page_html
