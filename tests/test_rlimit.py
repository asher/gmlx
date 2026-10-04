"""gmlx/rlimit.py: the open-file limit of the server and the launch
supervisor."""

from __future__ import annotations

import ast
import inspect
import resource

from gmlx import rlimit


def test_the_soft_limit_is_raised_toward_the_target_and_never_past_the_hard_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, hard))
        rlimit.raise_nofile_limit(1024)
        want = 1024 if hard == resource.RLIM_INFINITY else min(1024, hard)
        assert resource.getrlimit(resource.RLIMIT_NOFILE) == (want, hard)
        rlimit.raise_nofile_limit(512)             # never lowers it
        assert resource.getrlimit(resource.RLIMIT_NOFILE)[0] == want
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def test_the_server_raises_its_limit_before_it_serves():
    """Idle connections through a launch relay each hold one of the
    server's descriptors, so the server must not stay at a soft limit of
    256."""
    from gmlx.serve import server
    src = inspect.getsource(server._serve)
    tree = ast.parse(src.lstrip() if src.startswith(" ") else src)
    lines = {ast.unparse(n.func): n.lineno for n in ast.walk(tree)
             if isinstance(n, ast.Call)}
    assert "raise_nofile_limit" in lines
    assert lines["raise_nofile_limit"] < lines["uvicorn.run"]


def test_the_limit_in_effect_is_returned_and_a_low_one_warns():
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, hard))
        got = rlimit.raise_nofile_limit(1024)
        assert got == resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    assert rlimit.low_limit_warning(10240, "the server") is None
    assert rlimit.low_limit_warning(None, "the server") is None
    line = rlimit.low_limit_warning(2560, "the server")
    assert line is not None and "only 2560 files" in line


def test_the_server_warns_when_its_limit_stays_low():
    from gmlx.serve import server
    src = inspect.getsource(server._serve)
    assert "low_limit_warning(raise_nofile_limit()" in src
    # The warning is printed to stderr when there is one.
    tree = ast.parse(src.lstrip() if src.startswith(" ") else src)
    blocks = [n for n in ast.walk(tree) if isinstance(n, ast.If)
              and isinstance(n.test, ast.Name) and n.test.id == "low"]
    assert any(isinstance(c, ast.Call) and ast.unparse(c.func) == "print"
               and "sys.stderr" in ast.unparse(c) and "low" in ast.unparse(c.args)
               for n in blocks for stmt in n.body for c in ast.walk(stmt))


def test_the_low_limit_warning_names_no_hard_limit_and_quotes_the_commands():
    """A failed raise leaves the soft limit low while the hard limit may be
    unlimited, so the line never calls the limit the hard one."""
    line = rlimit.low_limit_warning(256, "the server")
    assert line is not None
    assert "hard" not in line
    assert "`ulimit -n`" in line and "`launchctl limit maxfiles`" in line
