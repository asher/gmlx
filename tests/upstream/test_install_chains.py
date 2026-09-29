"""Static rules for installers that wrap an upstream function, so a
reinstall finds its own layer under another installer's wrap.

- unwrap: reading ``_gmlx_orig`` or ``__wrapped__`` one level down to find
  the original. A wrap on top is dropped or taken for the original; use
  ``patching.under``.
- guard: reading the installer's flag with ``getattr`` on the function it
  rebinds. Under another wrap the flag is missed and the installer wraps
  twice; use ``patching.wrapped_with``.
- link: a wrapper that carries a flag, or that goes on a name more than one
  installer wraps, without a link to what it wraps. The walk stops there;
  use ``patching.link``. A replacement that calls nothing under it is not
  a wrapper, and a link would claim a chain that is gone; those are listed
  in ``_REPLACEMENTS``.
"""

from __future__ import annotations

import ast
import collections
import importlib
import pathlib

import pytest

import gmlx
from gmlx.upstream.patching import _walk, link

_ROOT = pathlib.Path(gmlx.__file__).parent
_LINKS = {"_gmlx_orig", "__wrapped__"}
_SKIP = {"upstream/patching.py"}

# Whole-body replacements on a name another installer also wraps, as
# "path::installer::def". Neither calls what it replaces.
_REPLACEMENTS = {
    # Installs before the cascade stamp wraps it (serve patches order).
    "cache/apc_pooling.py::install_batched_cachelist_admission::_extend_cache",
    # SpeculativeGenerationBatch.extend raises in stock. The cascade wraps
    # the extend of a cache, which is another name.
    "spec/admission.py::install_continuous_batch_admission::_buffered_extend",
}

# Offenders not yet moved to the chain helpers, as "path::function".
_KNOWN = {
    "guard": {
        "eval_guard.py::install_stream_registry",
        "models/deepseek_v4/chat.py::install_string_content",
        "models/deepseek_v41/tools.py::install_message_normalizer",
        "serve/patches/_common.py::_install_gen_args_transform",
        "serve/patches/apc.py::install_retire_render_capture",
        "serve/patches/chat_behavior.py::install_chat_template_kwargs",
        "serve/patches/chat_behavior.py::install_ignore_eos",
        "serve/patches/chat_behavior.py::install_openai_stop_sequences",
        "serve/patches/chat_behavior.py::install_role_normalization",
        "serve/patches/chat_behavior.py::install_stream_thinking_seed",
        "serve/patches/chat_behavior.py::install_stream_timings",
        "serve/patches/chat_behavior.py::install_thinking_budget_fix",
        "serve/patches/chat_behavior.py::install_vanilla_stream_chunks",
        "serve/patches/mtp_thinking.py::_install_criteria",
        "serve/patches/mtp_thinking.py::_install_defer",
        "serve/patches/mtp_thinking.py::_install_transport",
        "serve/patches/observability.py::install_request_timing_log",
        "serve/patches/render.py::install_chat_image_placement",
        "serve/patches/render.py::install_faithful_history",
        "serve/patches/routes.py::install_hf_download_gate",
        "serve/patches/routes.py::install_runtime_snapshot_enrichment",
        "serve/patches/sampling.py::install_fast_sampler",
        "serve/patches/sampling.py::install_until_eos_default",
        "stream/wired_limit.py::_install_wired_limit_warn_once",
        "stream/wired_limit.py::_neutralize_wired_limit_sweep",
    },
    "link": {
        "eval_guard.py::install_stream_registry",
        "models/deepseek_v4/chat.py::install_string_content",
        "models/deepseek_v41/tools.py::install_message_normalizer",
        "serve/patches/_common.py::_install_gen_args_transform",
        "serve/patches/apc.py::install_retire_render_capture",
        "serve/patches/chat_behavior.py::install_chat_template_kwargs",
        "serve/patches/chat_behavior.py::install_ignore_eos",
        "serve/patches/chat_behavior.py::install_thinking_budget_fix",
        "serve/patches/chat_behavior.py::install_vanilla_stream_chunks",
        "serve/patches/mtp_thinking.py::_install_criteria",
        "serve/patches/mtp_thinking.py::_install_defer",
        "serve/patches/mtp_thinking.py::_install_transport",
        "serve/patches/observability.py::install_request_timing_log",
        "serve/patches/routes.py::install_hf_download_gate",
        "serve/patches/routes.py::install_runtime_snapshot_enrichment",
        "serve/patches/sampling.py::install_fast_sampler",
        "serve/patches/sampling.py::install_until_eos_default",
        "stream/wired_limit.py::_install_wired_limit_warn_once",
        "stream/wired_limit.py::_neutralize_wired_limit_sweep",
        "stream/wired_limit.py::configure_cpu_device",
        "tui/chat.py::_backend_mtp_text",
        "tui/chat.py::_backend_plain_text",
    },
}


def _norm(node) -> str:
    """``getattr(x, "name", ...)`` reads the same slot as ``x.name``."""
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "getattr" and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)):
        return f"{ast.unparse(node.args[0])}.{node.args[1].value}"
    return ast.unparse(node)


def _rebinds(fn):
    """(slot, value) for every attribute rebind in ``fn``."""
    for n in ast.walk(fn):
        if (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Attribute)):
            slot, value = ast.unparse(n.targets[0]), n.value
        elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
              and n.func.id == "setattr" and len(n.args) == 3
              and isinstance(n.args[1], ast.Constant)
              and isinstance(n.args[1].value, str)):
            slot, value = f"{ast.unparse(n.args[0])}.{n.args[1].value}", \
                n.args[2]
        else:
            continue
        attr = slot.rsplit(".", 1)[-1]
        if attr.startswith(("_gmlx", "_kq")) or attr.startswith("__") \
                and attr not in ("__init__", "__call__"):
            continue
        yield slot, value


def _functions():
    for path in sorted(_ROOT.rglob("*.py")):
        rel = path.relative_to(_ROOT).as_posix()
        if rel in _SKIP:
            continue
        for fn in ast.walk(ast.parse(path.read_text())):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield rel, fn


def _wrappers(fn):
    """(slot, def) for local defs ``fn`` rebinds onto a slot."""
    defs = {d.name: d for d in fn.body
            if isinstance(d, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for slot, value in _rebinds(fn):
        name = ast.unparse(value)
        if name in defs:
            yield slot, defs[name]


def _marks(fn, name):
    """(flagged, linked) for the local def ``name`` in ``fn``."""
    flagged = linked = False
    for n in ast.walk(fn):
        text = ast.unparse(n)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "link" \
                and n.args and ast.unparse(n.args[0]) == name:
            linked = True
            flagged |= len(n.args) > 2
        elif isinstance(n, ast.Assign):
            tgt = ast.unparse(n.targets[0])
            if tgt.startswith((f"{name}.", f"{name}.__dict__[")):
                if any(k in tgt for k in _LINKS):
                    linked = True
                elif text.endswith("= True"):
                    flagged = True
        elif (isinstance(n, ast.Call) and text.startswith(
                (f"setattr({name}, ", f"{name}.__dict__.update("))):
            if any(k in text for k in _LINKS):
                linked = True
            if "True" in text:
                flagged = True
    return flagged, linked


def _flag_reads(test, alias, slots):
    """``getattr(slot, flag, False)`` calls in an if test, split into the
    plain ones and the ones under ``not``."""
    plain, negated = [], []

    def visit(node, neg):
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            visit(node.operand, not neg)
            return
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "getattr" and len(node.args) == 3
                and isinstance(node.args[2], ast.Constant)
                and node.args[2].value is False):
            obj = ast.unparse(node.args[0])
            if obj in slots or alias.get(obj, obj) in slots:
                (negated if neg else plain).append(node)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, neg)

    visit(test, False)
    return plain, negated


def _guard_reads_slot(node, alias, slots) -> bool:
    """An install guard: the flag read on the slot either skips the
    install (the branch exits) or gates it (``if not ...``)."""
    plain, negated = _flag_reads(node.test, alias, slots)
    exits = any(isinstance(s, (ast.Return, ast.Continue)) for s in node.body)
    return bool(negated) or (bool(plain) and exits)


def _scan():
    found = collections.defaultdict(set)
    installers = collections.defaultdict(set)
    wrappers = []
    for rel, fn in _functions():
        key = f"{rel}::{fn.name}"
        alias = {}
        for n in ast.walk(fn):
            if (isinstance(n, ast.Assign) and len(n.targets) == 1
                    and isinstance(n.targets[0], ast.Name)):
                alias.setdefault(n.targets[0].id, _norm(n.value))
        slots = {slot for slot, _ in _rebinds(fn)}
        for n in ast.walk(fn):
            if isinstance(n, ast.If) and _guard_reads_slot(n, alias, slots):
                found["guard"].add(key)
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == "getattr" and len(n.args) >= 2):
                name = n.args[1]
                if isinstance(name, ast.Constant) and name.value in _LINKS:
                    found["unwrap"].add(key)
            elif (isinstance(n, ast.Attribute) and n.attr in _LINKS
                  and isinstance(n.ctx, ast.Load)
                  and ast.unparse(n.value) != "self"):
                found["unwrap"].add(key)
        for slot, d in _wrappers(fn):
            installers[slot.rsplit(".", 1)[-1]].add(key)
            wrappers.append((key, slot, d, _marks(fn, d.name)))
    for key, slot, d, (flagged, linked) in wrappers:
        wrapped = any("wraps" in ast.unparse(x) for x in d.decorator_list)
        shared = len(installers[slot.rsplit(".", 1)[-1]]) > 1
        if (flagged or shared) and not (linked or wrapped) \
                and f"{key}::{d.name}" not in _REPLACEMENTS:
            found["link"].add(key)
    return found


def test_no_single_level_unwrap():
    assert sorted(_scan()["unwrap"]) == []


def test_guards_walk_the_chain():
    assert sorted(_scan()["guard"] - _KNOWN["guard"]) == []


def test_wrappers_link_what_they_wrap():
    assert sorted(_scan()["link"] - _KNOWN["link"]) == []


def test_known_offenders_still_offend():
    """Keep the lists honest: a fixed site leaves them."""
    found = _scan()
    for rule, known in _KNOWN.items():
        assert sorted(known - found[rule]) == [], rule


_ROUTES = {
    "cascade": ("gmlx.upstream.cascade_sdpa", "install_cascade_sdpa",
                "_installed_route", "_gmlx_cascade_route"),
    "sparse": ("gmlx.upstream.sparse_sdpa", "install_sparse_sdpa",
               "_installed", "_gmlx_sparse_route"),
    "g4": ("gmlx.models.gemma4.batched_sdpa", "install_gemma4_batched_sdpa",
           "_installed", "_gmlx_g4_route"),
}


@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_route_reinstall_under_another_wrap_adds_no_layer(monkeypatch, route):
    """The SDPA route installers read their flag on the outermost function,
    so a reinstall under another installer's wrap added a second route."""
    modname, install, installed, flag = _ROUTES[route]
    mod = importlib.import_module(modname)
    monkeypatch.setenv("GMLX_SPARSE_ATTN", "1")
    monkeypatch.delenv("GMLX_CASCADE_SDPA", raising=False)
    monkeypatch.delenv("GMLX_G4_BATCHED_SDPA", raising=False)
    if route == "g4":
        monkeypatch.setattr(mod.attn_hd512, "_installed", True)
    seams = []
    for pkg, name in mod._MODULES:
        try:
            seam = importlib.import_module(f"{pkg}.{name}")
        except ImportError:
            continue
        monkeypatch.setattr(seam, "scaled_dot_product_attention",
                            seam.scaled_dot_product_attention)
        seams.append(seam)

    def layers():
        return [sum(bool(getattr(f, flag, False)) for f in _walk(
            s.scaled_dot_product_attention)) for s in seams]

    monkeypatch.setattr(mod, installed, False)
    assert getattr(mod, install)()
    before = layers()
    assert max(before) == 1
    for s in seams:
        cur = s.scaled_dot_product_attention
        s.scaled_dot_product_attention = link(
            lambda *a, _cur=cur, **k: _cur(*a, **k), cur, "_test_other")
    monkeypatch.setattr(mod, installed, False)
    assert getattr(mod, install)()
    assert layers() == before


_TICK = (
    ("gmlx.serve.batch_sched", "install_decode_priority_sched",
     "_kq_gguf_decode_priority_sched"),
    ("gmlx.serve.admit_gate", "install_admit_headroom_gate",
     "_kq_gguf_admit_headroom"),
    ("gmlx.cache.fresh_gate", "install_fresh_admission_gate",
     "_kq_gguf_fresh_gate"),
    ("gmlx.serve.queue_cap", "_install_census", "_kq_gguf_queue_census"),
    ("gmlx.serve.tick_guard", "install_tick_guard", "_kq_gguf_tick_guard"),
)


def test_tick_installers_rerun_in_another_order_add_no_layer(monkeypatch):
    """The installers on BatchGenerator._next read their flag on the
    outermost function, so running them again in another order put a
    second layer of each on top of the others."""
    ar = pytest.importorskip("mlx_vlm.generate.ar")
    for name in ("GMLX_ADMIT_HEADROOM", "GMLX_APC_FRESH_WAIT_MS",
                 "GMLX_TICK_GUARD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(ar.BatchGenerator, "_next", ar.BatchGenerator._next)
    installs = [getattr(importlib.import_module(m), fn) for m, fn, _ in _TICK]
    for install in installs:
        install()
    top = ar.BatchGenerator._next
    for install in installs[::-1] + installs[1::2] + installs[::2]:
        install()
        assert ar.BatchGenerator._next is top, install.__name__
    for *_, flag in _TICK:
        assert sum(bool(f.__dict__.get(flag)) for f in _walk(top)) == 1, flag
