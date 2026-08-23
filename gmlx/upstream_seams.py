"""Contract registry for every upstream symbol gmlx patches or deep-imports.

gmlx monkeypatches ~30 private symbols across mlx-vlm and mlx-lm and
deep-imports model internals. Those seams are guarded structurally (try/except
ImportError + idempotence flags), so upstream renames or rewrites fail
silently - stock behavior quietly returns, or a confusing error surfaces far
downstream (mlx-vlm 0.6.4 vendoring switch_layers turned into a gather_mm
shape error at MoE prefill). This module makes the contract explicit:

- ``SEAMS`` declares each (module, attr) we touch, why, and whether a missing
  seam must abort install (``critical``) or only costs an acceleration.
- ``upstream_seams.json`` pins a source fingerprint for each seam, captured
  at the qualified upstream versions. ``tests/test_upstream_seams.py`` fails
  with the symbol's name when upstream drifts under the pin.
- ``check_upstream_versions()`` is the runtime gate: below-floor upstream
  versions raise with an actionable message (stale venvs resolve pyproject
  constraints exactly once - a floor pin alone cannot catch them); versions
  newer than the qualified set warn once.

On a deliberate upstream bump: re-audit the drifted seams, then
``python -m gmlx.upstream_seams --regen`` (see docs/upstream-upgrades.md).
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import re
import sys
from dataclasses import dataclass

_JSON_PATH = os.path.join(os.path.dirname(__file__), "upstream_seams.json")

# Hard floors: below these, core wiring (switch_layers geometry, cache
# protocol, server engine) predates what gmlx targets; refuse to run.
FLOORS = {
    "mlx": "0.31",
    "mlx-lm": "0.31",
    "mlx-vlm": "0.6.3",
}


@dataclass(frozen=True)
class Seam:
    module: str            # upstream import path
    attr: str              # dotted attribute path; "" = module existence only
    used_by: str           # our patch/import site (for the failure message)
    critical: bool = False  # True: a missing seam must abort, not degrade
    module_optional: bool = False  # module absent on some qualified versions
    attr_optional: bool = False    # attr absent on some qualified versions


SEAMS: tuple[Seam, ...] = (
    # --- batched serve scheduler + ragged decode (batch_sched / ragged_decode) ---
    Seam("mlx_vlm.generate.ar", "BatchGenerator._next",
         "batch_sched.install_decode_priority_sched (decode-first tick, "
         "prompt-arm structure, _prompt_time_counter contract); "
         "admit_gate.install_admit_headroom_gate (admission-arm gating "
         "via the pending-list stash); serve_memtrace (tick bracket)",
         critical=True),
    Seam("mlx_vlm.generate.ar", "BatchGenerator.insert",
         "batch_sched arrival-merge (_unprocessed_sequences append/rebind)"),
    # --- qwen3_5 owned forwards (qwen35_owned/gdn/attn/layers/rope/
    #     verify_linear) ---
    # The owned qwen3.5 forward surface carries no pins: its copies and
    # mirrors are certified by source-equality, construction-pair, and
    # identity tests. But ownership lands at construction only on the
    # text MTP load (loader._mtp_target_classes); load_vlm_mtp_model's
    # target is built stock by mlx_vlm.utils, so the multimodal path
    # still installs the patched regime and its symbols stay pinned
    # below.
    Seam("mlx_vlm.models.qwen3_5.gated_delta", "gated_delta_ops",
         "gdn_patches._patch_mlxvlm_gated_delta_tiled_v (stock-built "
         "trees only: vlm MTP targets + the GMLX_QWEN_OWNED=0 text "
         "fallback; owned GDN routes through mlx-lm gated_delta)",
         critical=True),
    Seam("mlx_vlm.models.qwen3_5.gated_delta", "gated_delta_kernel",
         "gdn_patches._patch_mlxvlm_gated_delta_tiled_v (stock-built "
         "trees only)", critical=True),
    Seam("mlx_vlm.models.qwen3_5.gated_delta", "_gated_delta_with_states_ops",
         "gdn_patches._patch_mlxvlm_gated_delta_tiled_v (stock-built "
         "trees only)", critical=True),
    Seam("mlx_vlm.models.qwen3_5.gated_delta", "_gated_delta_state_ops",
         "gdn_patches._patch_mlxvlm_gated_delta_tiled_v (stock-built "
         "trees only)", critical=True),
    # --- qwen3_5 stock-path verify patches (vlm MTP targets) ---
    Seam("mlx_vlm.models.qwen3_5.language",
         "_qwen3_5_ragged_decode_attention",
         "ragged_decode.install_unified_ragged_plan (vlm MTP stock path; "
         "the owned attention calls the in-tree dispatch directly)"),
    Seam("mlx_vlm.models.qwen3_5.language",
         "_target_verify_left_padded_attention",
         "gdn_patches._patch_batched_verify_sdpa / qwen35_verify_fold "
         "(vlm MTP stock path; the owned attention carries the composed "
         "verify routes natively)"),
    Seam("mlx_vlm.models.qwen3_5.language", "_target_verify_linear",
         "gdn_patches._patch_bf16_verify_linear (vlm MTP stock path; the "
         "owned verify-linear family folds the lever into its dispatch)"),
    Seam("mlx_vlm.models.qwen3_5.language", "scaled_dot_product_attention",
         "qwen35_verify_fold (B>=2 left-padded fold; vlm MTP stock path "
         "- the owned attention folds at its own dispatch and calls the "
         "base module symbol)"),
    Seam("mlx_vlm.models.qwen3_5.language", "Qwen3_5GatedDeltaNet.__call__",
         "gdn_patches._patch_gated_delta_fused_verify (vlm MTP stock "
         "path; the owned subclass carries the fused dispatch natively)"),
    # Mirror pin: _owned_model_call deliberately diverges from this
    # body (B=1 shortcut removed, padded prefill hoisted, S=0 guard),
    # so it cannot be substitution-normalized; the fingerprint forces a
    # re-mirror review on upstream change.
    Seam("mlx_vlm.models.qwen3_5.language", "Qwen3_5Model.__call__",
         "qwen35_owned._owned_model_call (forward body mirrored with "
         "deliberate divergences; re-mirror on upstream change)",
         critical=True),
    # --- gemma4 host-sync-free masks/offsets (gemma4_sync carries bodies).
    # Owned gemma4_text MTP trees (gemma4_owned) shadow all three rows by
    # subclass override and mirror the same bodies; the pins stay live for
    # stock-built trees (multimodal construction, GMLX_GEMMA_OWNED=0). ---
    Seam("mlx_vlm.models.gemma4.language", "Gemma4TextModel._make_masks",
         "gemma4_sync.install_gemma4_nosync (body carried, offset probe); "
         "gemma4_owned._owned_make_masks (mirror)"),
    Seam("mlx_vlm.models.gemma4.language", "Attention.__call__",
         "gemma4_sync.install_gemma4_nosync (body carried, offset wrap); "
         "gemma4_owned._owned_attention_call (mirror)"),
    Seam("mlx_vlm.models.gemma4.language", "scaled_dot_product_attention",
         "gemma4_batched_sdpa (hd512 B>1 row route; left-pad tail slices); "
         "gemma4_owned._sdpa_dispatch calls the claim directly"),
    # --- quantized-KV SDPA batch-mask fix (both base modules) ---
    Seam("mlx_lm.models.base", "quantized_scaled_dot_product_attention",
         "quantized_sdpa_fix (5D grouped scores vs 4D batch mask)"),
    Seam("mlx_vlm.models.base", "quantized_scaled_dot_product_attention",
         "quantized_sdpa_fix (same body as the mlx_lm copy)"),
    Seam("mlx_vlm.models.cache", "BatchQuantizedKVCache.make_mask",
         "quantized_sdpa_fix._patch_make_mask (starts registration for "
         "the fused decode route; left_padding attr contract)"),
    # --- gemma4 text-only load path (mlx_lm module wrapped in text_only) ---
    Seam("mlx_lm.models.gemma4_text", "scaled_dot_product_attention",
         "gemma4_batched_sdpa (same row route at the text-only seam)"),
    Seam("mlx_lm.models.gemma4_text", "Attention.__call__",
         "gemma4_batched_sdpa (module-global sdpa call, cache kwarg)"),
    Seam("mlx_lm.models.gemma4_text", "Gemma4TextModel._make_masks",
         "gemma4_batched_sdpa (one mask object per layer type; the "
         "producer->consumer pad relay keys on that identity)"),
    # --- shared-prefix cascade decode route + APC warm-batch stamp ---
    Seam("mlx_lm.models.llama", "scaled_dot_product_attention",
         "cascade_sdpa (stamped shared-prefix batched decode)"),
    Seam("mlx_vlm.generate.ar", "PromptProcessingBatch.__init__",
         "cascade_sdpa.install_cascade_stamp (token-prefix stamp at batch "
         "formation; rows + caches coexist in matching order here)",
         critical=True),
    Seam("mlx_vlm.generate.ar", "_extend_cache",
         "cascade_sdpa.install_cascade_stamp (stamp carry across the "
         "B=1-to-batch merge lift on admission)", critical=True),
    # --- speculative / AR batch engine (spec_engine owns these methods) ---
    Seam("mlx_vlm.generate.ar", "BatchGenerator.__init__",
         "spec_engine._install_apc_manager_stash", critical=True),
    Seam("mlx_vlm.generate.ar",
         "PromptProcessingBatch._store_apc_exact_checkpoints",
         "spec_engine._install_ckpt_checkpoint_store (ckpt cursor rides "
         "the stock store)", critical=True),
    Seam("mlx_vlm.generate.ar", "PromptProcessingBatch.__init__",
         "spec_engine.install_full_prompt_mtp_prefill (prefill-step "
         "restore + stock-path ckpt arming)", critical=True),
    Seam("mlx_vlm.generate.ar", "GenerationBatch._step",
         "spec_engine._install_plain_ckpt_decode (token accounting + "
         "decode-time snapshots)", critical=True),
    Seam("mlx_vlm.generate.ar", "GenerationBatch.filter",
         "spec_engine._install_plain_ckpt_decode (B=1 retirement at row "
         "exit)", critical=True),
    Seam("mlx_vlm.generate.ar", "PromptProcessingBatch.prompt_step",
         "spec_engine.install_full_prompt_mtp_prefill", critical=True),
    Seam("mlx_vlm.generate.ar", "PromptProcessingBatch.generate",
         "spec_engine.install_full_prompt_mtp_prefill", critical=True),
    Seam("mlx_vlm.generate.ar", "SpeculativeGenerationBatch.next",
         "spec_engine.install_continuous_batch_admission", critical=True),
    Seam("mlx_vlm.generate.ar", "SpeculativeGenerationBatch.filter",
         "spec_engine._filter_with_release (per-row release rides the "
         "mark-finished contract; the rounds generator sheds via "
         "stop_check)", critical=True),
    Seam("mlx_vlm.generate.ar", "SpeculativeGenerationBatch.__len__",
         "spec_engine._len_with_promotion (patched semantics are D-3's "
         "hazard; decision modules count rows via _orig_len only)",
         critical=True),
    Seam("mlx_vlm.generate.ar", "GenerationBatch._eval_pending_state",
         "governor shed ordering (filter under pressure requires "
         "eval before the gather)", critical=True),
    Seam("mlx_vlm.generate.ar", "BatchGenerator.remove",
         "governor retire rung + tick_guard rebuild (queue pop / "
         "prompt-batch clear / decode filter semantics)", critical=True),
    Seam("mlx_vlm.generate.ar", "run_speculative_server_rounds",
         "spec_engine.install_owned_spec_engine", critical=True),
    Seam("mlx_vlm.speculative.utils", "make_speculative_prompt_cache",
         "spec_engine.install_spec_kv_quant (B=1 KV_BITS)", critical=True),
    Seam("mlx_vlm.generate.ar", "BatchGenerator._apc_pick_for",
         "spec_engine._bind_l1_view (L1 APC helpers)"),
    Seam("mlx_vlm.generate.ar", "BatchGenerator._apc_exact_checkpoint_len",
         "spec_engine._bind_l1_view (L1 APC helpers)"),
    Seam("mlx_vlm.generate.ar", "BatchGenerator._apc_extra_hash",
         "spec_engine._bind_l1_view (L1 APC helpers)"),
    # --- server engine ---
    Seam("mlx_vlm.server.generation", "run_speculative_server_rounds",
         "spec_engine.install_owned_spec_engine", critical=True),
    Seam("mlx_vlm.server.generation", "load_model_resources",
         "server_bridge_vlm (GGUF model resource loader)", critical=True),
    Seam("mlx_vlm.server.generation", "ResponseGenerator._make_sampler",
         "server_patches.install_fast_sampler"),
    Seam("mlx_vlm.server.generation", "ResponseGenerator._step",
         "server_patches.row_failed (permanently failed rows delivered "
         "to their response queues on the engine thread)", critical=True),
    Seam("mlx_vlm.server.generation", "GenerationArguments.to_template_kwargs",
         "server_patches (chat_template_kwargs transform)"),
    Seam("mlx_vlm.server.generation", "ResponseGenerator._preprocess_request",
         "server_patches.install_retire_render_capture (ids hop + "
         "tokenize path for the next-turn retirement key)"),
    Seam("mlx_vlm.server.openai", "apply_chat_template",
         "server_patches.install_retire_render_capture (render-context "
         "memo, module attr) + render.install_faithful_history (inner "
         "key-merge wrap)"),
    Seam("mlx_vlm.server.anthropic", "apply_chat_template",
         "server_patches._common._render_target_modules (faithful "
         "history, retire capture, and thinking seed wrap every "
         "captured render binding)"),
    Seam("mlx_vlm.server.app", "_protocol_deps",
         "server_patches._common._render_target_modules (deps captures "
         "the openai function object at namespace construction)"),
    Seam("mlx_vlm.prompt_utils", "apply_chat_template",
         "server_patches.render.install_faithful_history "
         "(return_messages contract: one rebuilt message per readable "
         "input item; tool passthrough branch)"),
    Seam("mlx_vlm.prompt_utils", "get_chat_template",
         "server_patches.render.install_faithful_history (owned render "
         "tail mirrors the stock function's)"),
    Seam("mlx_vlm.server.openai", "_split_thinking",
         "retire_key.build_assistant_message (response-shape mirror)"),
    Seam("mlx_vlm.server.openai", "process_tool_calls",
         "retire_key.build_assistant_message (response-shape mirror)"),
    Seam("mlx_vlm.server.openai", "_infer_tool_parser_from_processor",
         "retire_key.build_assistant_message (response-shape mirror)"),
    Seam("mlx_vlm.server.openai", "load_tool_module",
         "retire_key.build_assistant_message (response-shape mirror)"),
    Seam("mlx_vlm.server.responses_state",
         "ThinkingStreamState._build_open_close_markers",
         "retire_key._thinking_markers (splitter default marker pairs "
         "for the mid-decode virtual closer)"),
    Seam("mlx_vlm.server.app", "_split_thinking_text",
         "server_patches.chat_behavior.install_stream_thinking_seed "
         "(non-stream truncated-thinking classification, module attr "
         "looked up per call by app._split_thinking)"),
    Seam("mlx_vlm.server.runtime", "runtime",
         "residency._RuntimeProxy wrap", critical=True),
    Seam("mlx_vlm.server.app", "app",
         "server_patches route surgery / server.py uvicorn root",
         critical=True),
    Seam("mlx_vlm.speculative.drafters", "load_drafter",
         "server_bridge_vlm (in-memory MTP drafter injection)", critical=True),
    Seam("mlx_vlm.utils", "get_model_path",
         "server_patches (HF download gate)", critical=True),
    Seam("mlx_vlm.utils", "StoppingCriteria.__call__",
         "server_patches (ignore-EOS)"),
    # --- tool-parser registry (hy_v3_tools / muse_glimmer_tools) ---
    Seam("mlx_vlm.tool_parsers", "_TEMPLATE_MARKERS",
         "hy_v3_tools / muse_glimmer_tools ensure_registered (marker prepend)"),
    Seam("mlx_vlm.tool_parsers", "load_tool_module",
         "hy_v3_tools / muse_glimmer_tools (sys.modules graft resolves "
         "through it)"),
    # --- APC internals (lone-harvest patch, gmlx manager subclass, apc_pooling) ---
    Seam("mlx_vlm.apc", "harvest_blocks_from_batch_cache",
         "server_patches.install_apc_lone_harvest", critical=True),
    Seam("mlx_vlm.apc", "_clone_layer_major_kv_cache_for_apc",
         "apc_manager.GmlxAPCManager.store_kv_blocks", critical=True),
    Seam("mlx_vlm.apc", "_sequence_hash",
         "apc_manager.GmlxAPCManager.store_kv_blocks", critical=True),
    Seam("mlx_vlm.apc", "APCExactCacheEntry",
         "apc_manager.GmlxAPCManager.store_kv_blocks", critical=True),
    Seam("mlx_vlm.apc", "_hash_tokens",
         "apc_manager.GmlxAPCManager.store_kv_blocks", critical=True),
    Seam("mlx_vlm.apc", "_DiskLayerMajorBlock",
         "apc_manager.GmlxAPCManager.store_kv_blocks", critical=True),
    Seam("mlx_vlm.apc", "_copy_mlx_array",
         "apc_manager.GmlxAPCManager.store_kv_blocks", critical=True),
    Seam("mlx_vlm.apc", "SEED_PARENT_HASH",
         "apc_manager.GmlxAPCManager.store_kv_blocks", critical=True),
    Seam("mlx_vlm.apc", "DEFAULT_BLOCK_SIZE",
         "apc_manager.build_apc_manager (from_env mirror)", critical=True),
    Seam("mlx_vlm.apc", "DEFAULT_NUM_BLOCKS",
         "apc_manager.build_apc_manager (from_env mirror)", critical=True),
    Seam("mlx_vlm.apc", "DiskBlockStore",
         "apc_manager.build_apc_manager (from_env mirror)", critical=True),
    Seam("mlx_vlm.apc", "_cache_entry_supports_exact_apc",
         "apc_pooling (PoolingCache exact-APC predicate)", critical=True),
    Seam("mlx_vlm.apc", "_merge_exact_cache_entries",
         "apc_pooling (PoolingCache merge arm)", critical=True),
    Seam("mlx_vlm.apc", "_read_safetensors_tensor",
         "apc_pooling (disk-tier zero-width spill)", critical=True),
    Seam("mlx_vlm.apc", "_clone_cache_entry_for_apc",
         "apc_pooling", critical=True),
    Seam("mlx_vlm.generate.ar", "BatchGenerator.__init__",
         "apc_pooling.install_pooled_prefill_batch_gate (prompt batches "
         "stay B=1 on pooling-cache models; to_batch_cache has no pooled "
         "arm)", critical=True),
    Seam("mlx_vlm.generate.ar", "_extend_cache",
         "apc_pooling.install_batched_cachelist_admission (promotion test "
         "looks through CacheList so an already-batched one is not merged "
         "twice)", critical=True),
    Seam("mlx_vlm.apc", "_safetensors_dtype_info",
         "apc_pooling", critical=True),
    # <= 0.6.3 the wrapper delegates to this mlx-lm alias; 0.6.4 inlined it.
    Seam("mlx_vlm.generate.common", "mlx_maybe_quantize_kv_cache",
         "apc_pooling (rotating-safe KV-quant replacement, <=0.6.3 arm)",
         critical=True, attr_optional=True),
    Seam("mlx_vlm.generate", "maybe_quantize_kv_cache",
         "apc_pooling (rotating-safe KV-quant replacement)", critical=True),
    # --- cache classes (dual-origin via gmlx.cache_compat; on <= 0.6.3
    # the vlm module re-exports mlx-lm's classes, so these fingerprint the
    # same source) ---
    Seam("mlx_vlm.models.cache", "KVCache",
         "cache_compat (ckpt sidecar tails, snapshots)", critical=True),
    Seam("mlx_vlm.models.cache", "RotatingKVCache",
         "cache_compat (ds4 make_cache, rollback attach, spec_helpers)",
         critical=True),
    Seam("mlx_vlm.models.cache", "CacheList",
         "cache_compat (ds4 make_cache, prefix_cache snapshots)",
         critical=True),
    Seam("mlx_vlm.models.cache", "ArraysCache",
         "cache_compat (ckpt_supported, prefix_cache snapshots) / "
         "arrays_cache_fix (prepare wrap)"),
    Seam("mlx_vlm.models.cache", "BatchKVCache",
         "mtp_drafter / cache_snapshot row round-trip"),
    Seam("mlx_vlm.models.cache", "BatchKVCache.filter",
         "spec_engine per-row release + governor retire (physical row "
         "drop through the cache's own filter)", critical=True),
    Seam("mlx_vlm.models.cache", "BatchKVCache.extract",
         "governor orange retire (contiguous single-row extract before "
         "filter)", critical=True),
    Seam("mlx_vlm.models.cache", "ArraysCache.extract",
         "governor orange retire (GDN hybrid row extract)"),
    Seam("mlx_vlm.models.cache", "_BaseCache.nbytes",
         "governor green sampling + registered-cache bytes() protocol",
         critical=True),
    Seam("mlx_vlm.models.cache", "BatchRotatingKVCache",
         "cache_compat (rollback attach, safe KV-quant exclusion)"),
    Seam("mlx_vlm.models.cache", "BufferedRotatingKVCache",
         "spec_helpers rollback-slack wrap", critical=True),
    Seam("mlx_vlm.models.cache", "make_prompt_cache",
         "chat MTP cache construction", critical=True),
    # 0.6.4 replaced the runtime's model_cache dict with a registry;
    # residency and the wire tests speak both shapes.
    Seam("mlx_vlm.server.runtime", "ModelCacheRegistry",
         "residency pool / test_wire_contract fake loader",
         attr_optional=True),
    # --- vendored switch_layers (exists only on mlx-vlm >= 0.6.4) ---
    Seam("mlx_vlm.models.switch_layers", "SwitchLinear.__call__",
         "modules.switch_layer_types (dual-origin leaf swap)",
         critical=True, module_optional=True),
    Seam("mlx_vlm.models.switch_layers", "SwitchGLU.__call__",
         "modules.switch_layer_types (dual-origin fused GLU)",
         critical=True, module_optional=True),
    # --- mlx-lm seams ---
    Seam("mlx_lm.models.switch_layers", "SwitchLinear.__call__",
         "modules leaf swap + fused subclass signature contract",
         critical=True),
    Seam("mlx_lm.models.switch_layers", "SwitchGLU.__call__",
         "modules.install_fused_moe_glu / loader.install_expert_streaming",
         critical=True),
    Seam("mlx_lm.server", "ModelProvider._load",
         "server_bridge_lm (GGUF ModelProvider)", critical=True),
    Seam("mlx_lm.generate", "maybe_quantize_kv_cache",
         "apc_pooling", critical=True),
    Seam("mlx_lm.models.cache", "RotatingKVCache",
         "rotating_cache_fix / prefix_cache snapshots"),
    Seam("mlx_lm.models.cache", "ArraysCache",
         "arrays_cache_fix (prepare wrap: lengths must clear the stale "
         "left_padding that merge() seeds, or right-padded ragged prefill "
         "runs unmasked through conv/SSM state)"),
    Seam("mlx_lm.models.base", "create_causal_mask", "rotating_cache_fix"),
    Seam("mlx_lm.models.hunyuan", "MoeBlock",
         "loader._patch_hunyuan_norm_topk"),
    Seam("mlx_lm.models.deepseek_v32", "MoEGate",
         "dsv32_patches (fp32 router patch)"),
    # --- kimi_k3 vendored-module imports (kimi_k3_model) ---
    # The vendored Kimi-K3 class reuses mlx-lm's per-key-channel-decay
    # gated-delta scan (KDA's exact recurrence), the kimi_linear ShortConv1d
    # semantics (reimplemented in-module; the class itself is not imported),
    # and the MLA MultiLinear (absorbed embed_q/unembed_out).
    Seam("mlx_lm.models.gated_delta", "gated_delta_kernel",
         "kimi_k3_model.KimiK3DeltaAttention (metal scan dispatch)",
         critical=True),
    Seam("mlx_lm.models.gated_delta", "gated_delta_ops",
         "kimi_k3_model.KimiK3DeltaAttention (CPU/fallback scan)",
         critical=True),
    Seam("mlx_lm.models.mla", "MultiLinear",
         "kimi_k3_model.KimiK3MLAAttention (embed_q/unembed_out -> "
         "KQuantMultiLinear)", critical=True),
    Seam("mlx_lm.models.kimi_linear", "ShortConv1d",
         "kimi_k3_model.ShortConv1d (semantics mirror; re-mirror review "
         "on upstream change)"),
)


def _resolve(seam: Seam):
    """('ok', obj) | ('missing-module', None) | ('missing-attr', None).

    Modules resolve through import; ``sys.modules`` wins for already-loaded
    ones, which matters for e.g. mlx_vlm.server.runtime (the package
    __init__ shadows the submodule attribute with the runtime instance).
    """
    try:
        mod = sys.modules.get(seam.module) or importlib.import_module(
            seam.module)
    except ImportError:
        return "missing-module", None
    obj = mod
    for part in seam.attr.split(".") if seam.attr else ():
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return "missing-attr", None
    return "ok", obj


def _fingerprint(obj) -> str | None:
    """sha256 of the object's source, or None when unhashable (constants,
    kernel objects, module-level values)."""
    try:
        src = inspect.getsource(obj)
    except (TypeError, OSError):
        return None
    return hashlib.sha256(src.encode()).hexdigest()


def _rebound(seam: Seam, obj) -> bool:
    """True when the seam currently holds a gmlx replacement - one of
    our installers already ran in this interpreter. Source comparison is
    then meaningless (the patch being present proves the install succeeded);
    existence checks still apply. Upstream's own cross-package from-imports
    (e.g. qwen3_5.gated_delta re-exporting mlx-lm functions) keep their
    original __module__ and are fingerprinted normally."""
    mod = getattr(obj, "__module__", None) or getattr(
        type(obj), "__module__", "")
    return bool(mod) and mod.split(".")[0] == "gmlx"


def _key(seam: Seam) -> str:
    return f"{seam.module}:{seam.attr}"


def collect_fingerprints() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for seam in SEAMS:
        status, obj = _resolve(seam)
        if status == "ok":
            if _rebound(seam, obj):
                raise RuntimeError(
                    f"{_key(seam)} already holds a gmlx patch - "
                    f"regenerate fingerprints in a fresh interpreter")
            out[_key(seam)] = _fingerprint(obj)
        elif ((status == "missing-module" and seam.module_optional)
              or (status == "missing-attr" and seam.attr_optional)):
            continue  # not present on this qualified version
        else:
            raise RuntimeError(
                f"cannot fingerprint {_key(seam)}: {status} "
                f"(used by {seam.used_by})")
    return out


def load_pinned() -> dict:
    with open(_JSON_PATH) as f:
        return json.load(f)


def check_seams() -> list[str]:
    """Problems found against the pinned fingerprints; [] when clean."""
    pinned = load_pinned()
    fps = pinned["fingerprints"]
    problems = []
    for seam in SEAMS:
        status, obj = _resolve(seam)
        if status == "missing-module":
            if not seam.module_optional:
                problems.append(
                    f"{_key(seam)}: module missing (used by {seam.used_by})")
            continue
        if status == "missing-attr":
            if not seam.attr_optional:
                problems.append(
                    f"{_key(seam)}: attribute missing - upstream renamed or "
                    f"removed it (used by {seam.used_by})")
            continue
        if _rebound(seam, obj):
            continue  # our installer already replaced it in this process
        want = fps.get(_key(seam), "<unpinned>")
        got = _fingerprint(obj)
        if want == "<unpinned>":
            problems.append(
                f"{_key(seam)}: no pinned fingerprint - run "
                f"`python -m gmlx.upstream_seams --regen`")
        elif got != want:
            problems.append(
                f"{_key(seam)}: source changed under the pin - re-audit "
                f"{seam.used_by}, then regenerate "
                f"(pinned for {pinned.get('generated_with')})")
    return problems


def vendored_upstream_collisions() -> list[str]:
    """Vendored mlx-lm model modules that upstream now ships natively.

    Our vendored modules register themselves into ``sys.modules`` under the
    mlx_lm.models namespace, so an upstream module of the same name would be
    silently shadowed; flag it so the vendored copy gets reconciled/dropped.
    """
    from .arch_table import _VENDORED_MLX_LM_MODULES
    import mlx_lm.models as lm_models
    root = os.path.dirname(lm_models.__file__)
    hits = []
    for mod_name in _VENDORED_MLX_LM_MODULES:
        leaf = mod_name.rsplit(".", 1)[-1]
        if os.path.exists(os.path.join(root, f"{leaf}.py")):
            hits.append(
                f"{mod_name}: upstream mlx-lm now ships this module; the "
                f"vendored copy shadows it - reconcile and drop the vendor "
                f"entry (arch_table._VENDORED_MLX_LM_MODULES)")
    hits += _vendored_vlm_collisions()
    return hits


# gmlx module -> the mlx-vlm namespace its ensure_registered() grafts into.
# Each is a package directory upstream, so a native arrival shows up as either
# a <leaf>.py module or a <leaf>/ package.
VENDORED_MLX_VLM_MODULES = {
    # muse_glimmer model: shipped upstream in mlx-vlm 0.6.15; the graft is
    # upstream-first so gmlx.muse_glimmer_vlm_model is dead code under this
    # pin. Delete the module at the vendoring review.
    "gmlx.hy_v3_tools": "mlx_vlm.tool_parsers.hy_v3",
    "gmlx.muse_glimmer_tools": "mlx_vlm.tool_parsers.muse_glimmer",
}


def _vendored_vlm_collisions() -> list[str]:
    """Same check on the mlx-vlm side: our grafts are upstream-first at import
    time, but a native module arriving under a name we also register is the
    signal to drop the vendored copy rather than keep shadowing it."""
    hits = []
    for mod_name, target in VENDORED_MLX_VLM_MODULES.items():
        pkg, _, leaf = target.rpartition(".")
        try:
            parent = importlib.import_module(pkg)
        except ImportError:
            continue
        root = os.path.dirname(parent.__file__)
        if (os.path.exists(os.path.join(root, f"{leaf}.py"))
                or os.path.isdir(os.path.join(root, leaf))):
            hits.append(
                f"{mod_name}: upstream mlx-vlm now ships {target}; the "
                f"vendored copy is only a fallback - reconcile and drop the "
                f"vendor entry (upstream_seams.VENDORED_MLX_VLM_MODULES)")
    return hits


def _parse_version(v: str) -> tuple[int, ...]:
    m = re.match(r"(\d+(?:\.\d+)*)", v)
    if not m:
        return ()
    return tuple(int(p) for p in m.group(1).split("."))


_VERSION_WARNED = False


def check_upstream_versions(quiet: bool = False) -> list[str]:
    """Runtime gate for the mlx / mlx-lm / mlx-vlm environment.

    Below a floor: raises RuntimeError (a stale venv resolved pyproject
    constraints once and never again - pip pins cannot catch it). Newer than
    the qualified versions recorded at the last --regen: warns once. Missing
    or unparsable metadata (source installs): warns, never raises. Returns
    the warning lines (also printed unless ``quiet``).
    """
    global _VERSION_WARNED
    import importlib.metadata as md
    try:
        tested = load_pinned().get("generated_with", {})
    except (OSError, json.JSONDecodeError):
        tested = {}
    warnings, errors = [], []
    for pkg, floor in FLOORS.items():
        try:
            found = md.version(pkg)
        except md.PackageNotFoundError:
            warnings.append(
                f"[versions] {pkg}: no package metadata (source install?) - "
                f"cannot verify the supported floor >= {floor}")
            continue
        found_t = _parse_version(found)
        if not found_t:
            warnings.append(
                f"[versions] {pkg}: unparsable version {found!r} - cannot "
                f"verify the supported floor >= {floor}")
            continue
        floor_t = _parse_version(floor)
        if found_t < floor_t:
            errors.append(
                f"{pkg} {found} is below the supported floor {floor}; "
                f"upgrade it (pip install -U '{pkg}>={floor}') or reinstall "
                f"gmlx into a fresh venv")
            continue
        qual = tested.get(pkg)
        if qual and found_t > _parse_version(qual):
            warnings.append(
                f"[versions] {pkg} {found} is newer than the qualified "
                f"{qual}; untested - see docs/upstream-upgrades.md")
    if errors:
        raise RuntimeError(
            "unsupported upstream package versions:\n  " + "\n  ".join(errors))
    if warnings and not quiet and not _VERSION_WARNED:
        _VERSION_WARNED = True
        for w in warnings:
            print(w, file=sys.stderr, flush=True)
    return warnings


def regen(path: str = _JSON_PATH) -> dict:
    import importlib.metadata as md
    generated_with = {}
    for pkg in FLOORS:
        try:
            generated_with[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            generated_with[pkg] = None
    data = {
        "generated_with": generated_with,
        "fingerprints": collect_fingerprints(),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
        f.write("\n")
    return data


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="Check (default) or regenerate the upstream seam "
                    "fingerprints.")
    p.add_argument("--regen", action="store_true",
                   help="re-capture fingerprints from the installed "
                        "upstream versions (do this only after a deliberate, "
                        "audited upstream bump)")
    args = p.parse_args(argv)
    if args.regen:
        data = regen()
        n = len(data["fingerprints"])
        print(f"pinned {n} seam fingerprints for {data['generated_with']}")
        return 0
    problems = check_seams() + vendored_upstream_collisions()
    try:
        problems += check_upstream_versions(quiet=True)
    except RuntimeError as e:
        problems.append(str(e))
    if problems:
        print("\n".join(problems))
        return 1
    print(f"{len(SEAMS)} seams OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
