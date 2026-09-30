"""The claims table: every name gmlx claims exists in the copy, and after
the serve startup and the load-time installers each claim took. A failure
names the row, so a fold or an edit of the copy that leaves a row behind
shows here."""
from __future__ import annotations

import contextlib
import functools
import os
import subprocess
import sys
import types

import pytest

import gmlx.upstream.claims as uc
from gmlx.upstream.claims import Claim

_COPY = "gmlx._vendor.mlx_vlm.prompt_utils"


def _only(monkeypatch, *claims):
    monkeypatch.setattr(uc, "CLAIMS", claims)


def test_every_claimed_name_is_in_the_copy():
    problems = uc.check_claims()
    assert not problems, "\n".join(problems)


_CHILD = r"""
import os
import sys

import gmlx.upstream.claims as uc
import mlx.core as mx

if os.environ.get("KQUANT_FORCE_CPU"):
    mx.set_default_device(mx.cpu)
sys.exit(uc.main(["--installed"]))
"""


def test_every_claim_takes_after_install():
    """The installers rebind names for the whole process, so they run in a
    child."""
    r = subprocess.run([sys.executable, "-c", _CHILD], capture_output=True,
                       text=True, timeout=600, env=dict(os.environ))
    assert r.returncode == 0, (r.stdout + r.stderr)[-6000:]
    assert r.stdout.strip().splitlines()[-1] == f"{len(uc.CLAIMS)} claims OK"


def test_a_name_missing_from_the_copy_is_named(monkeypatch):
    _only(monkeypatch, Claim(_COPY, "definitely_not_a_thing", "test"),
          Claim("gmlx._vendor.mlx_vlm.no_such_module", "x", "test"))
    problems = uc.check_claims()
    assert any("definitely_not_a_thing: missing attribute in the copy" in p
               for p in problems)
    assert any("no_such_module:x: missing module in the copy" in p
               for p in problems)


def test_a_name_outside_the_copy_is_named(monkeypatch):
    _only(monkeypatch, Claim("gmlx.upstream.patching", "root", "test"))
    assert uc.check_claims() == [
        "gmlx.upstream.patching:root: not a module of the copy"]


def test_two_rows_for_one_name_are_named(monkeypatch):
    row = Claim(_COPY, "apply_chat_template", "test")
    _only(monkeypatch, row, row)
    assert any("two rows" in p for p in uc.check_claims())


def test_a_claim_the_install_left_on_the_copy_is_named(monkeypatch):
    import gmlx._vendor.mlx_vlm.prompt_utils as pu

    # No installer claims this name.
    _only(monkeypatch, Claim(_COPY, "_content_media_count", "test"))
    assert uc.check_claims() == []
    problems = uc.check_claims(installed=True)
    assert problems and "still the copy's after install" in problems[0]
    from gmlx.upstream.patching import root
    monkeypatch.setattr(pu, "_content_media_count", root)
    assert uc.check_claims(installed=True) == []


def test_origin_looks_through_a_decorator_written_elsewhere():
    from gmlx._vendor.mlx_vlm.generate import common
    from gmlx.upstream import wired_limit_cpu_fix as wl

    stock = getattr(common.wired_limit, "_gmlx_cpu_guard", common.wired_limit)
    # contextlib's wrapper over the copy's function and over gmlx's.
    assert uc._origin(stock) == "copy"
    assert uc._origin(wl._guard(stock)) == "gmlx"
    assert uc._origin(wl.cpu_wired_limit) == "gmlx"
    assert uc._origin(contextlib.nullcontext) == "other"
    assert uc._origin(functools.partial(wl._guard, stock)) == "gmlx"
    # No code of its own: the module of its class, or of the class itself.
    assert uc._origin(uc.Claim) == "gmlx"
    assert uc._origin(uc.Claim("m", "a", "u")) == "gmlx"
    assert uc._origin(types.SimpleNamespace()) == "other"


def test_a_route_claim_names_a_path_with_no_route(monkeypatch):
    _only(monkeypatch, Claim("gmlx._vendor.mlx_vlm.server.openai",
                             "chat_completions_endpoint", "test",
                             how="route", paths=("/no/such/route",)))
    assert uc.check_claims() == []
    problems = uc.check_claims(installed=True)
    assert any("no POST /no/such/route route" in p for p in problems)


def test_a_subclass_that_does_not_derive_is_named(monkeypatch):
    _only(monkeypatch, Claim("gmlx._vendor.mlx_vlm.apc",
                             "APCManager.lookup_prefix", "test",
                             how="subclass",
                             by="gmlx.cache.kv_cache:KVCache"))
    problems = uc.check_claims()
    assert any("does not derive from the copy's class" in p
               for p in problems)


def test_a_row_names_an_installer_that_is_gone_or_takes_arguments(
        monkeypatch):
    _only(monkeypatch,
          Claim(_COPY, "apply_chat_template", "test",
                install="gmlx.upstream.patching:no_such_installer"),
          Claim(_COPY, "get_chat_template", "test",
                install="gmlx.serve.patches:install_server_patches"))
    problems = uc.check_claims()
    assert any("no_such_installer missing attribute" in p for p in problems)
    assert any("takes arguments other than the model" in p
               for p in problems)


@pytest.mark.parametrize("row", [
    Claim(_COPY, "apply_chat_template", "test", how="patched"),
    Claim(_COPY, "apply_chat_template", "test", paths=("/x",)),
    Claim(_COPY, "apply_chat_template", "test", how="route"),
    Claim(_COPY, "apply_chat_template", "test", by="gmlx.cache.kv_cache:KVCache"),
    Claim(_COPY, "apply_chat_template", "test", how="subclass",
          install="gmlx.serve.bridge_lm:install_gguf_bridge"),
])
def test_a_malformed_row_is_named(monkeypatch, row):
    _only(monkeypatch, row)
    assert uc.check_claims()
