"""gmlx/container/ignore.py against the cases of moby's patternmatcher tests,
so drift from BuildKit's matching shows up here."""

from __future__ import annotations

import pytest

from gmlx.container import ignore

# (pattern, path, excluded) from TestMatches in moby/patternmatcher.
MOBY_CASES = [
    ("**", "file", True),
    ("**", "file/", True),
    ("**/", "file", True),
    ("**/", "file/", True),
    ("**", "/", True),
    ("**/", "/", True),
    ("**", "dir/file", True),
    ("**/", "dir/file", True),
    ("**", "dir/file/", True),
    ("**/", "dir/file/", True),
    ("**/**", "dir/file", True),
    ("**/**", "dir/file/", True),
    ("dir/**", "dir/file", True),
    ("dir/**", "dir/file/", True),
    ("dir/**", "dir/dir2/file", True),
    ("dir/**", "dir/dir2/file/", True),
    ("**/dir", "dir", True),
    ("**/dir", "dir/file", True),
    ("**/dir2/*", "dir/dir2/file", True),
    ("**/dir2/*", "dir/dir2/file/", True),
    ("**/dir2/**", "dir/dir2/dir3/file", True),
    ("**/dir2/**", "dir/dir2/dir3/file/", True),
    ("**file", "file", True),
    ("**file", "dir/file", True),
    ("**/file", "dir/file", True),
    ("**file", "dir/dir/file", True),
    ("**/file", "dir/dir/file", True),
    ("**/file*", "dir/dir/file", True),
    ("**/file*", "dir/dir/file.txt", True),
    ("**/file*txt", "dir/dir/file.txt", True),
    ("**/file*.txt", "dir/dir/file.txt", True),
    ("**/file*.txt*", "dir/dir/file.txt", True),
    ("**/**/*.txt", "dir/dir/file.txt", True),
    ("**/**/*.txt2", "dir/dir/file.txt", False),
    ("**/*.txt", "file.txt", True),
    ("**/**/*.txt", "file.txt", True),
    ("a**/*.txt", "a/file.txt", True),
    ("a**/*.txt", "a/dir/file.txt", True),
    ("a**/*.txt", "a/dir/dir/file.txt", True),
    ("a/*.txt", "a/dir/file.txt", False),
    ("a/*.txt", "a/file.txt", True),
    ("a/*.txt**", "a/file.txt", True),
    ("a[b-d]e", "ae", False),
    ("a[b-d]e", "ace", True),
    ("a[b-d]e", "aae", False),
    ("a[^b-d]e", "aze", True),
    (".*", ".foo", True),
    (".*", "foo", False),
    ("abc.def", "abcdef", False),
    ("abc.def", "abc.def", True),
    ("abc.def", "abcZdef", False),
    ("abc?def", "abcZdef", True),
    ("abc?def", "abcdef", False),
    ("**/foo/bar", "foo/bar", True),
    ("**/foo/bar", "dir/foo/bar", True),
    ("**/foo/bar", "dir/dir2/foo/bar", True),
    ("abc/**", "abc", False),
    ("abc/**", "abc/def", True),
    ("abc/**", "abc/def/ghi", True),
    ("**/.foo", ".foo", True),
    ("**/.foo", "bar.foo", False),
    ("a(b)c/def", "a(b)c/def", True),
    ("a(b)c/def", "a(b)c/xyz", False),
    ("a.|)$(}+{bc", "a.|)$(}+{bc", True),
    ("dist/proxy.py-2.4.0rc3.dev36+g08acad9-py3-none-any.whl",
     "dist/proxy.py-2.4.0rc3.dev36+g08acad9-py3-none-any.whl", True),
    ("dist/*.whl", "dist/proxy.py-2.4.0rc3.dev36+g08acad9-py3-none-any.whl", True),
]

# (patterns, path, excluded) from the multi-pattern cases.
MOBY_MULTI = [
    (["**", "!util/docker/web"], "util/docker/web/foo", False),
    (["**", "!util/docker/web", "util/docker/web/foo"], "util/docker/web/foo", True),
    (["**", "!dist/proxy.py-2.4.0rc3.dev36+g08acad9-py3-none-any.whl"],
     "dist/proxy.py-2.4.0rc3.dev36+g08acad9-py3-none-any.whl", False),
    (["**", "!dist/*.whl"], "dist/proxy.py-2.4.0rc3.dev36+g08acad9-py3-none-any.whl", False),
]


@pytest.mark.parametrize("pattern, path, excluded", MOBY_CASES)
def test_moby_pattern_cases(pattern, path, excluded):
    assert ignore.Matcher([pattern]).excluded(path) is excluded


@pytest.mark.parametrize("patterns, path, excluded", MOBY_MULTI)
def test_moby_multi_pattern_cases(patterns, path, excluded):
    assert ignore.Matcher(patterns).excluded(path) is excluded


def test_parent_folder_match_excludes_the_files_below():
    m = ignore.Matcher(["node_modules"])
    assert m.excluded("node_modules/react/index.js")
    assert not m.excluded("src/node_modules.txt")


def test_exception_reincludes_a_file():
    m = ignore.Matcher(["*.md", "!README.md"])
    assert m.excluded("CHANGES.md")
    assert not m.excluded("README.md")
    assert m.has_exclusions


@pytest.mark.parametrize("pattern", ["a\\*b", "a[b", "x[[:alpha:]]", "!"])
def test_unsupported_patterns_are_refused(pattern):
    with pytest.raises(ignore.UnsupportedPattern):
        ignore.Matcher([pattern])


def test_read_patterns_follows_readall():
    text = "﻿# comment\n\n  /build/  \n! /dist/x \nsrc/../out\n#!keep\n"
    assert ignore.read_patterns(text) == ["build", "!dist/x", "out"]


def test_named_ignore_file_wins_over_the_context_one(tmp_path):
    cf = tmp_path / "images" / "Containerfile"
    cf.parent.mkdir()
    cf.write_text("FROM x\n")
    (tmp_path / ".dockerignore").write_text("a\n")
    assert ignore.ignore_file(cf, tmp_path) == tmp_path / ".dockerignore"
    named = cf.with_name("Containerfile.dockerignore")
    named.write_text("b\n")
    assert ignore.ignore_file(cf, tmp_path) == named


def test_load_reports_an_unsupported_pattern(tmp_path):
    cf = tmp_path / "Containerfile"
    cf.write_text("FROM x\n")
    (tmp_path / ".dockerignore").write_text("node_modules\nfoo\\*bar\n")
    matcher, notice = ignore.load(cf, tmp_path)
    assert matcher is None
    assert "every context file counts" in notice
