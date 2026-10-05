"""The claims table: every name gmlx claims exists in the copy, and after
the serve startup and the load-time installers each claim took. A failure
names the row, so a fold or an edit of the copy that leaves a row behind
shows here."""
from __future__ import annotations

import contextlib
import functools
import json
import os
import re
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


def test_a_row_may_name_a_root_package_of_the_copy(monkeypatch):
    _only(monkeypatch, Claim("gmlx._vendor.mlx_vlm", "prepare_inputs", "test",
                             wrappers=("gmlx.serve.media_sinks",)))
    assert uc.check_claims() == []


def test_two_rows_for_one_name_are_named(monkeypatch):
    row = Claim(_COPY, "apply_chat_template", "test")
    _only(monkeypatch, row, row)
    assert any("two rows" in p for p in uc.check_claims())


def test_a_claim_the_install_left_on_the_copy_is_named(monkeypatch):
    import gmlx._vendor.mlx_vlm.prompt_utils as pu

    # No installer claims this name.
    _only(monkeypatch, Claim(_COPY, "_content_media_count", "test",
                             wrappers=("gmlx.upstream.patching",)))
    assert uc.check_claims() == []
    problems = uc.check_claims(installed=True)
    assert problems and "still the copy's after install" in problems[0]
    from gmlx.upstream.patching import root
    monkeypatch.setattr(pu, "_content_media_count", root)
    assert uc.check_claims(installed=True) == []


_OUTER = """
def outer(fn):
    def unlinked(*a, **k):  # holds fn, links nothing
        return fn(*a, **k)
    return unlinked
"""


def _written_in(rel: str):
    """``outer`` with its code read as written in gmlx/``rel``."""
    ns: dict = {}
    exec(compile(_OUTER, os.path.join(uc._GMLX_DIR, rel), "exec"), ns)
    return ns["outer"]


def test_the_chain_follows_links_and_what_an_unlinked_wrapper_holds():
    from gmlx.upstream.patching import link, root

    top = _written_in("serve/cb_phase.py")(root)
    both = {"gmlx.serve.cb_phase", "gmlx.upstream.patching"}
    assert uc.chain_modules(top) == both

    def mine(*a, **k):  # not gmlx code: counts for nothing, links on
        return top(*a, **k)

    assert uc.chain_modules(link(mine, top)) == both
    # A linked wrapper is followed by its link, not by what it holds.
    assert uc.chain_modules(link(top, uc.check_claims)) == {
        "gmlx.serve.cb_phase", "gmlx.upstream.claims"}
    # No code: the module of its class.
    assert uc.chain_modules(uc.Claim("m", "a", "u")) == {
        "gmlx.upstream.claims"}


def test_a_listed_wrapper_missing_from_the_chain_is_named(monkeypatch):
    import gmlx._vendor.mlx_vlm.prompt_utils as pu
    from gmlx.upstream.patching import root

    _only(monkeypatch, Claim(_COPY, "_content_media_count", "test",
                             wrappers=("gmlx.upstream.claims",
                                       "gmlx.upstream.patching")))
    monkeypatch.setattr(pu, "_content_media_count", root)
    assert uc.check_claims(installed=True) == [
        f"{_COPY}:_content_media_count: no wrapper from gmlx.upstream.claims "
        "in its chain after install; the serve startup did not claim it "
        "(claimed by test)"]
    monkeypatch.setattr(pu, "_content_media_count",
                        _written_in("upstream/claims.py")(root))
    assert uc.check_claims(installed=True) == []


def test_a_wrap_row_lists_its_wrappers(monkeypatch):
    _only(monkeypatch, Claim(_COPY, "apply_chat_template", "test"))
    assert any("wrappers go with how='wrap', which needs them" in p
               for p in uc.check_claims())
    _only(monkeypatch, Claim(_COPY, "apply_chat_template", "test",
                             wrappers=("gmlx._vendor.mlx_vlm.apc",)))
    assert any("wrappers name gmlx modules outside the copy" in p
               for p in uc.check_claims())


_NOOP_CHILD = r"""
import importlib
import os
import sys

import mlx.core as mx

if os.environ.get("KQUANT_FORCE_CPU"):
    mx.set_default_device(mx.cpu)
module, name = sys.argv[1].split(":")
setattr(importlib.import_module(module), name, lambda *a, **k: None)
import gmlx.upstream.claims as uc  # noqa: E402

sys.exit(uc.main(["--installed"]))
"""


@pytest.mark.parametrize("installer", [
    "gmlx.serve.admit_gate:install_admit_headroom_gate",
    "gmlx.serve.batch_sched:install_decode_priority_sched",
])
def test_an_installer_that_installs_nothing_fails_the_installed_check(
        installer):
    """Other wrappers stay on BatchGenerator._next, so the outermost is
    still gmlx's: only the walk of the chain finds the missing one."""
    r = subprocess.run([sys.executable, "-c", _NOOP_CHILD, installer],
                       capture_output=True, text=True, timeout=600,
                       env=dict(os.environ))
    module = installer.partition(":")[0]
    assert r.returncode == 1, (r.stdout + r.stderr)[-4000:]
    assert (f"BatchGenerator._next: no wrapper from {module} in its chain "
            "after install") in r.stdout, r.stdout[-4000:]


def test_the_generate_step_stream_generate_calls_is_claimed():
    """media_spans wraps dispatch's binding, the one stream_generate calls,
    so an installer that installs nothing leaves that row on the copy."""
    installer = "gmlx.gen.media_spans:install_block_aware_generate_step"
    r = subprocess.run([sys.executable, "-c", _NOOP_CHILD, installer],
                       capture_output=True, text=True, timeout=600,
                       env=dict(os.environ))
    assert r.returncode == 1, (r.stdout + r.stderr)[-4000:]
    assert ("gmlx._vendor.mlx_vlm.generate.dispatch:generate_step: still "
            "the copy's after install") in r.stdout, r.stdout[-4000:]


def _gmlx_modules() -> set[str]:
    """Every gmlx module and package outside the copy, dotted."""
    out = set()
    for root, dirs, files in os.walk(uc._GMLX_DIR):
        dirs[:] = [d for d in dirs if d not in ("_vendor", "__pycache__")]
        rel = os.path.relpath(root, uc._GMLX_DIR)
        base = "gmlx" if rel == "." else "gmlx." + rel.replace(os.sep, ".")
        out.add(base)
        out.update(f"{base}.{f[:-3]}" for f in files
                   if f.endswith(".py") and f != "__init__.py")
    return out


_DOTTED = re.compile(r"(?<![\w.])[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")
_SNAKE = re.compile(r"(?<![\w.])[a-z][a-z0-9]*(?:_[a-z0-9]+)+(?![\w.(])")


def _outside_the_chain(claim: Claim, mods: set[str]) -> list[str]:
    """The references to gmlx code in a row's used_by that name neither
    its wrappers nor its installer. A dotted name means the modules that
    its longest leading part ends; a bare snake_case word, the modules it
    names."""
    named = {}
    for ref in _DOTTED.findall(claim.used_by):
        parts = ref.split(".")
        for k in range(len(parts), 0, -1):
            tail = ".".join(parts[:k])
            hit = {m for m in mods if m == tail or m.endswith("." + tail)}
            if hit:
                named[ref] = hit
                break
    for word in _SNAKE.findall(claim.used_by):
        hit = {m for m in mods if m.endswith("." + word)}
        if hit:
            named[word] = hit
    own = set(claim.wrappers) | {claim.install.partition(":")[0]} - {""}
    return sorted(ref for ref, hit in named.items()
                  if not any(m in own or any(o.startswith(m + ".") for o in own)
                             for m in hit))


def test_a_wrap_row_names_only_code_in_its_chain():
    """The installed check reports a row as claimed by its used_by, so a
    wrap row names its wrappers and its installer, and no other code."""
    mods = _gmlx_modules()
    outside = {uc._key(c): _outside_the_chain(c, mods)
               for c in uc.CLAIMS if c.how == "wrap"}
    outside = {k: v for k, v in outside.items() if v}
    assert not outside, outside


def test_code_outside_the_chain_is_found():
    row = Claim(_COPY, "apply_chat_template",
                "render.install_faithful_history (in the chain); "
                "estimate._estimate_bound (reads it); single_bos_fix (so "
                "does this); hash_image_payload(x) is no module",
                wrappers=("gmlx.serve.patches.render",))
    assert _outside_the_chain(row, _gmlx_modules()) == [
        "estimate._estimate_bound", "single_bos_fix"]


def test_origin_looks_through_a_decorator_written_elsewhere():
    from gmlx._vendor.mlx_vlm.generate import common
    from gmlx.stream import wired_limit as wl

    # contextlib's wrapper over the copy's function and over gmlx's.
    assert uc._origin(common.wired_limit) == "copy"
    assert uc._origin(wl.cpu_wired_limit) == "gmlx"
    assert uc._origin(contextlib.nullcontext) == "other"
    assert uc._origin(functools.partial(wl.cpu_wired_limit, None)) == "gmlx"
    assert uc._origin(functools.partial(common.wired_limit, {})) == "copy"
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


_LIST_CHILD = r"""
import os
import sys

import gmlx.upstream.claims as uc
import mlx.core as mx

if os.environ.get("KQUANT_FORCE_CPU"):
    mx.set_default_device(mx.cpu)
uc.install_claims()
print("\n".join(sorted(n for n in sys.modules if n.startswith("gmlx._vendor."))))
"""

_CENSUS_CHILD = r"""
import importlib
import json
import os
import sys

import mlx.core as mx

if os.environ.get("KQUANT_FORCE_CPU"):
    mx.set_default_device(mx.cpu)


def kinds(v):
    return callable(v) or isinstance(v, (staticmethod, classmethod, property))


def slots(mods):
    out = {}
    for name in mods:
        m = sys.modules.get(name)
        for k, v in list(vars(m).items()) if m is not None else ():
            if kinds(v):
                out[f"{name}:{k}"] = id(v)
            if isinstance(v, type) and v.__module__ == name:
                for ck, cv in list(vars(v).items()):
                    if kinds(cv):
                        out[f"{name}:{k}.{ck}"] = id(cv)
    return out


mods = sys.stdin.read().split()
for n in mods:
    importlib.import_module(n)
before = slots(mods)
import gmlx.upstream.claims as uc  # noqa: E402

uc.install_claims()
after = slots(mods)
claimed = {f"{c.module}:{c.attr}" for c in uc.CLAIMS}
rebound = {}
for key in before:
    if key in after and after[key] != before[key]:
        _status, obj = uc._resolve(*key.split(":"))
        rebound[key] = [sorted(uc.chain_modules(obj)), key in claimed]
unlisted = {}
for c in uc.CLAIMS:
    if c.how == "wrap":
        _status, obj = uc._resolve(c.module, c.attr)
        extra = sorted(uc.chain_modules(obj) - set(c.wrappers))
        if extra:
            unlisted[f"{c.module}:{c.attr}"] = extra
print(json.dumps({"rebound": rebound, "unlisted": unlisted}))
"""


@pytest.fixture(scope="module")
def rebinds():
    """The names of the copy that the serve startup and the installers
    rebind, each with the gmlx modules in its chain and whether a row
    claims it, and the wrap rows whose chain holds a module they do not
    list. Two children: one finds the copy modules the installs load, the
    other imports them, installs, and compares."""
    env = dict(os.environ)
    r = subprocess.run([sys.executable, "-c", _LIST_CHILD], capture_output=True,
                       text=True, timeout=600, env=env)
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    mods = [line for line in r.stdout.splitlines()
            if line.startswith("gmlx._vendor.")]
    r = subprocess.run([sys.executable, "-c", _CENSUS_CHILD], input="\n".join(mods),
                       capture_output=True, text=True, timeout=600, env=env)
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


# Installers whose every rebind of the copy has a row, and whose wrap rows
# list them among their wrappers.
_CENSUSED = (
    "gmlx.cache.apc_pooling",
    "gmlx.serve.cb_phase",
    "gmlx.serve.media_sinks",
    "gmlx.serve.mem_preflight",
    "gmlx.serve.patches._common",
    "gmlx.serve.patches.api_contract",
    "gmlx.serve.patches.chat_behavior",
    "gmlx.serve.patches.media_gate",
    "gmlx.serve.patches.routes",
    "gmlx.serve.patches.spec_processors",
    "gmlx.serve.residency",
    "gmlx.spec.admission",
    "gmlx.spec.ckpt",
    "gmlx.spec.engine",
    "gmlx.spec.kv_quant",
)


@pytest.mark.parametrize("installer", _CENSUSED)
def test_each_name_an_installer_rebinds_has_a_row_that_lists_it(rebinds,
                                                                installer):
    unclaimed = sorted(k for k, (mods, claimed) in rebinds["rebound"].items()
                       if installer in mods and not claimed)
    unlisted = sorted(k for k, mods in rebinds["unlisted"].items()
                      if installer in mods)
    assert not unclaimed and not unlisted, (
        f"no row: {unclaimed}; row without {installer} in wrappers: "
        f"{unlisted}")
