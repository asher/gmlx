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
  use ``patching.link``.
"""

from __future__ import annotations

import ast
import collections
import pathlib

import gmlx

_ROOT = pathlib.Path(gmlx.__file__).parent
_LINKS = {"_gmlx_orig", "__wrapped__"}
_SKIP = {"upstream/patching.py"}

# Offenders not yet moved to the chain helpers, as "path::function".
_KNOWN = {
    "guard": {
        "cache/apc_manager.py::_install_fork_free_free_ram",
        "cache/apc_media.py::install_causal_serve_media",
        "cache/apc_pooling.py::install_batched_cachelist_admission",
        "cache/apc_pooling.py::install_pooled_prefill_batch_gate",
        "cache/compat.py::ensure_runtime_origin_make_cache",
        "cache/fresh_gate.py::install_fresh_admission_gate",
        "cache/kvarn_serve.py::_install_apc_gate",
        "cache/kvarn_serve.py::_install_make_cache",
        "cache/kvarn_serve.py::_install_ppb_fastpath",
        "eval_guard.py::install_stream_registry",
        "gen/generation.py::harden_mtp_rollback",
        "gen/media_spans.py::install_block_aware_generate_step",
        "models/deepseek_v4/chat.py::install_string_content",
        "models/deepseek_v41/tools.py::install_message_normalizer",
        "models/gemma4/batched_sdpa.py::install_gemma4_batched_sdpa",
        "serve/admit_gate.py::install_admit_headroom_gate",
        "serve/batch_sched.py::install_decode_priority_sched",
        "serve/engine_jobs.py::_install_diffusion_jobs",
        "serve/engine_jobs.py::_install_step_jobs",
        "serve/governor.py::install_governor",
        "serve/live_requests.py::install_live_requests",
        "serve/mem_preflight.py::install_memory_preflight",
        "serve/memtrace.py::install_serve_memtrace",
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
        "serve/patches/row_failed.py::install_row_failed_bridge",
        "serve/patches/sampling.py::install_fast_sampler",
        "serve/patches/sampling.py::install_until_eos_default",
        "serve/queue_cap.py::_install_census",
        "serve/seed_rows.py::install_per_request_seed",
        "serve/step_timing.py::install_step_timing",
        "serve/tick_guard.py::install_tick_guard",
        "spec/ckpt.py::_install_ckpt_checkpoint_store",
        "spec/ckpt.py::_install_exact_anchor_pick",
        "spec/ckpt.py::_install_plain_ckpt_decode",
        "spec/ckpt.py::_install_restored_prompt_progress",
        "spec/engine.py::_install_apc_manager_stash",
        "spec/engine.py::install_owned_spec_engine",
        "stream/wired_limit.py::_install_wired_limit_warn_once",
        "stream/wired_limit.py::_neutralize_wired_limit_sweep",
        "tune/attention.py::install_training_attention",
        "tune/checkpoint.py::checkpoint_layers",
        "upstream/arrays_cache_fix.py::install_arrays_cache_fix",
        "upstream/cascade_sdpa.py::_mask_stamp_on_filter",
        "upstream/cascade_sdpa.py::_merge_stamp_on_extend",
        "upstream/cascade_sdpa.py::install_cascade_sdpa",
        "upstream/cascade_sdpa.py::install_cascade_stamp",
        "upstream/quantized_cache_pack_fix.py::_wrap",
        "upstream/quantized_sdpa_fix.py::_patch_make_mask",
        "upstream/quantized_sdpa_fix.py::install_quantized_sdpa_mask_fix",
        "upstream/sparse_sdpa.py::install_sparse_sdpa",
    },
    "link": {
        "cache/apc_pooling.py::install_batched_cachelist_admission",
        "cache/apc_pooling.py::install_pooled_prefill_batch_gate",
        "cache/apc_pooling.py::install_pooled_prompt_kv_quant",
        "cache/apc_pooling.py::install_pooling_apc_support",
        "cache/apc_qsa.py::install_qsa_apc_support",
        "cache/compat.py::ensure_runtime_origin_make_cache",
        "cache/fresh_gate.py::install_fresh_admission_gate",
        "cache/kvarn_apc.py::install_kvarn_apc",
        "cache/kvarn_serve.py::_install_apc_gate",
        "cache/kvarn_serve.py::_install_make_cache",
        "cache/kvarn_serve.py::_install_ppb_fastpath",
        "eval_guard.py::install_stream_registry",
        "gen/generation.py::harden_mtp_rollback",
        "gen/media_spans.py::install_span_aware_prompt_step",
        "gen/prefill_decay.py::install_prefill_decay",
        "gen/prefill_tail.py::install_prefill_tail",
        "lora_rows.py::install_row_channel",
        "models/deepseek_v4/cache.py::_wrap_rotating",
        "models/deepseek_v4/chat.py::install_string_content",
        "models/deepseek_v41/tools.py::install_message_normalizer",
        "serve/admit_gate.py::install_admit_headroom_gate",
        "serve/batch_sched.py::install_decode_priority_sched",
        "serve/governor.py::install_governor",
        "serve/live_requests.py::install_live_requests",
        "serve/mem_preflight.py::install_memory_preflight",
        "serve/memtrace.py::install_serve_memtrace",
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
        "serve/patches/row_failed.py::install_row_failed_bridge",
        "serve/patches/sampling.py::install_fast_sampler",
        "serve/patches/sampling.py::install_until_eos_default",
        "serve/queue_cap.py::_install_census",
        "serve/seed_rows.py::install_per_request_seed",
        "serve/step_timing.py::install_step_timing",
        "serve/tick_guard.py::install_tick_guard",
        "spec/admission.py::install_continuous_batch_admission",
        "spec/ckpt.py::_install_ckpt_checkpoint_store",
        "spec/ckpt.py::_install_exact_anchor_pick",
        "spec/ckpt.py::_install_plain_ckpt_decode",
        "spec/ckpt.py::_install_restored_prompt_progress",
        "spec/engine.py::_install_apc_manager_stash",
        "spec/engine.py::install_owned_spec_engine",
        "spec/mtp_prefill.py::install_full_prompt_mtp_prefill",
        "stream/wired_limit.py::_install_wired_limit_warn_once",
        "stream/wired_limit.py::_neutralize_wired_limit_sweep",
        "stream/wired_limit.py::configure_cpu_device",
        "tui/chat.py::_backend_mtp_text",
        "tui/chat.py::_backend_plain_text",
        "upstream/cascade_sdpa.py::_mask_stamp_on_filter",
        "upstream/cascade_sdpa.py::_merge_stamp_on_extend",
        "upstream/gdn_patches.py::_patch_qwen35_empty_sequence_guard",
        "upstream/quantized_cache_pack_fix.py::_wrap",
        "upstream/rotating_cache_fix.py::_install_notrim",
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
        if (flagged or shared) and not (linked or wrapped):
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
