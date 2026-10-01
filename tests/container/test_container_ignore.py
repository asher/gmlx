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
    text = "\ufeff# comment\n\n  /build/  \n! /dist/x \nsrc/../out\n#!keep\n"
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


def test_an_invalid_range_is_unsupported_not_a_crash(tmp_path):
    with pytest.raises(ignore.UnsupportedPattern, match="not a valid pattern"):
        ignore.Matcher(["a[z-a]b"])
    (tmp_path / "Containerfile").write_text("FROM x\n")
    (tmp_path / ".dockerignore").write_text("a[z-a]b\n")
    matcher, notice = ignore.load(tmp_path / "Containerfile", tmp_path)
    assert matcher is None and "every context file counts" in notice


@pytest.mark.parametrize("pattern", ["notes,old", "!keep,this", "[,]x"])
def test_a_pattern_with_a_comma_is_unsupported(tmp_path, pattern):
    # Apple's builder splits a pattern on its commas, so the hash counts
    # every file rather than skip one that the build copies.
    with pytest.raises(ignore.UnsupportedPattern, match="comma"):
        ignore.Matcher([pattern])
    (tmp_path / "Containerfile").write_text("FROM x\n")
    (tmp_path / ".dockerignore").write_text(f"node_modules\n{pattern}\n")
    matcher, notice = ignore.load(tmp_path / "Containerfile", tmp_path)
    assert matcher is None and "every context file counts" in notice


def test_a_pattern_ends_at_the_end_of_the_text_as_in_go():
    assert not ignore.Matcher(["*.txt"]).excluded("a.txt\n")
    assert ignore.Matcher(["*.txt"]).excluded("a.txt")


def test_a_double_star_does_not_cross_a_newline_as_in_go():
    matcher = ignore.Matcher(["a/**/b"])
    assert matcher.excluded("a/x/y/b")
    assert not matcher.excluded("a/x\ny/b")


def test_lines_split_on_newlines_only():
    assert ignore.read_patterns("a\r\nb\rc\n\x0cd\n") == ["a", "b\rc", "d"]


@pytest.mark.parametrize("patterns, folder, prune", [
    (["node_modules"], "node_modules", True),
    (["node_modules", "!keep.txt"], "node_modules", True),        # one component
    (["build", "!build/keep"], "build", False),                   # below the folder
    (["build", "!src/keep"], "build", False),                      # two components
    (["a/build", "!src/keep"], "a/build", True),
    (["build", "!**/keep"], "build", False),
    (["build", "![k]eep"], "build", False),
    (["build", "!build"], "build", False),                         # not excluded
    (["other"], "build", False),
])
def test_excludes_all_below(patterns, folder, prune):
    assert ignore.Matcher(patterns).excludes_all_below(folder) is prune


# Hostile ignore files: the file can lie in a folder the client writes.

def test_the_step_matcher_agrees_with_the_regular_expression():
    """Patterns without a character class are matched without re. Both
    forms must give the same answer."""
    import random
    rng = random.Random(7)
    pieces = ["a", "b", "/", ".", "*", "**", "**/", "?", "a/b", "+"]
    compared = 0
    for _ in range(3000):
        pattern = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 7)))
        try:
            matcher = ignore.Matcher([pattern])
        except ignore.UnsupportedPattern:
            continue
        for pat in matcher.patterns:
            if pat.steps is None:
                continue
            for _ in range(20):
                path = "".join(rng.choice("ab/.\n+") for _ in range(rng.randint(0, 10)))
                assert ignore._run_steps(pat.steps, path) == (pat.regex.match(path) is not None), \
                    (pattern, path)
                compared += 1
    assert compared > 10000


def test_a_star_heavy_pattern_matches_a_long_name_quickly():
    import time
    matcher = ignore.Matcher(["*a" * 14 + "b"])
    started = time.monotonic()
    assert not matcher.excluded("a" * 120)
    assert not matcher.excluded(("a" * 250 + "/") * 16)
    assert time.monotonic() - started < 1.0


def test_a_character_class_with_many_stars_is_unsupported():
    ignore.Matcher(["*[ab]*"])
    with pytest.raises(ignore.UnsupportedPattern, match="more than 2 stars"):
        ignore.Matcher(["*[ab]*a*"])


def test_too_many_patterns_are_unsupported():
    ignore.Matcher([f"f{i}" for i in range(200)])
    with pytest.raises(ignore.UnsupportedPattern, match="more than 200"):
        ignore.Matcher([f"f{i}" for i in range(201)])


def test_matching_past_the_work_budget_stops():
    matcher = ignore.Matcher(["*a" * 14 + "b"] * 50, work_max=100_000)
    with pytest.raises(ignore.TooMuchWork, match="takes too long"):
        for i in range(1000):
            matcher.excluded(f"src/{'a' * 200}{i}")
    assert matcher.work > 100_000


def test_the_default_budget_bounds_the_worst_ignore_file():
    """The largest file the caps allow, of the slowest patterns, stops
    within a few seconds of matching work."""
    import time
    matcher = ignore.Matcher(["*a" * 14 + "b"] * ignore.PATTERNS_MAX)
    started = time.monotonic()
    with pytest.raises(ignore.TooMuchWork):
        # Bounded, so a matcher with no budget fails here instead of hanging.
        for i in range(30):
            matcher.excluded(("a" * 60 + "/") * 16 + str(i))
    assert time.monotonic() - started < 15


def test_a_typical_ignore_file_stays_far_under_the_budget():
    matcher = ignore.Matcher(["node_modules", "**/node_modules", "*.log", "dist", ".git",
                              "**/*.pyc", "!keep.log", "coverage", ".env*"])
    matcher.excluded("packages/web-app/src/components/Button/index.test.tsx")
    assert matcher.work * 20_000 < ignore.WORK_MAX         # 20,000 such files fit


def _build_folder(tmp_path):
    (tmp_path / "Containerfile").write_text("FROM x\n")
    return tmp_path / "Containerfile"


def test_a_large_sparse_ignore_file_is_not_read(tmp_path):
    cf = _build_folder(tmp_path)
    with open(tmp_path / ".dockerignore", "wb") as f:
        f.truncate(768 << 20)
    matcher, notice = ignore.load(cf, tmp_path)
    assert matcher is None and "larger than 1 MiB" in notice


def test_an_ignore_file_that_is_a_link_or_a_pipe_is_not_read(tmp_path):
    import os
    cf = _build_folder(tmp_path)
    (tmp_path / "real").write_text("node_modules\n")
    named = tmp_path / "Containerfile.dockerignore"
    named.symlink_to(tmp_path / "real")
    matcher, notice = ignore.load(cf, tmp_path)
    assert matcher is None and "symbolic link" in notice
    named.unlink()
    os.mkfifo(named)
    matcher, notice = ignore.load(cf, tmp_path)      # returns at once, no writer
    assert matcher is None and "not a regular file" in notice
