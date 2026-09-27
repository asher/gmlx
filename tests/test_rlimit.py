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
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert "raise_nofile_limit" in calls
    assert calls.index("raise_nofile_limit") < calls.index("uvicorn.run")
