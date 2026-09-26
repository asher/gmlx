"""MkDocs hooks for the docs site built from docs/ (see mkdocs.yml).

A relative link from a page to a file outside docs/, such as ../README.md,
works on GitHub but has no page on the site. The hook points each one at
the file on GitHub, at the release tag when the build runs for a tag.

A long page opens with a list of links to its own sections for readers on
GitHub. The site shows the same list beside the page, so the hook removes
the list from the page body.

A `# doctest: build` line that opens a code block marks the block for
tests/test_docs_config.py. The hook removes the marker from the site page.
"""

from __future__ import annotations

import os
import posixpath
import re

_REPO_URL = "https://github.com/asher/gmlx"
_FENCE = re.compile(r"^\s*(```|~~~)")
_DOCTEST_MARKER = "# doctest: build"
_CONTENTS_ITEM = re.compile(r"[-*] \[[^\]]+\]\(#[^)\s]+\)", re.DOTALL)
_LINK = re.compile(r"(\]\(|\b(?:src|href)=\")([^)\"\s#]+)(#[^)\"\s]*)?")


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
