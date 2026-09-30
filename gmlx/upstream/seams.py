"""Contract registry for every upstream symbol gmlx patches or deep-imports.

gmlx monkeypatches well over a hundred symbols across mlx-vlm and mlx-lm
and deep-imports model internals. Those seams are guarded structurally (try/except
ImportError + idempotence flags), so upstream renames or rewrites fail
silently - stock behavior quietly returns, or a confusing error surfaces far
downstream (mlx-vlm 0.6.4 vendoring switch_layers turned into a gather_mm
shape error at MoE prefill). This module makes the contract explicit:

- ``SEAMS`` declares each (module, attr) we touch, why, and whether a missing
  seam must abort install (``critical``) or only costs an acceleration.
- ``seams.json`` pins a fingerprint for each seam, captured at the
  qualified upstream versions: the symbol's source, or its value where it
  has none, and for a forwarding lambda the implementation it calls too.
  An identity row pins nothing: it names the gmlx object a stock name must
  resolve to once the cache binding ran (gmlx.cache.bind).
  ``tests/upstream/test_upstream_seams.py`` fails with the symbol's name
  when upstream drifts under the pin.
- ``check_upstream_versions()`` is the runtime gate: below-floor upstream
  versions raise with an actionable message (stale venvs resolve pyproject
  constraints exactly once - a floor pin alone cannot catch them); versions
  newer than the qualified set warn once.

On a deliberate upstream bump: re-audit the drifted seams, then
``python -m gmlx.upstream.seams --regen`` (see docs/internals/upstream-upgrades.md).
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace

_JSON_PATH = os.path.join(os.path.dirname(__file__), "seams.json")

# Hard floors: below these, core wiring (switch_layers geometry, cache
# protocol, server engine) predates what gmlx targets; refuse to run.
FLOORS = {
    "mlx": "0.31",
    "mlx-lm": "0.31",
    "mlx-vlm": "0.7.3",
}


@dataclass(frozen=True)
class Seam:
    module: str            # upstream import path
    attr: str              # dotted attribute path; "" = the module source
    used_by: str           # our patch/import site (for the failure message)
    critical: bool = False  # True: a missing seam must abort, not degrade
    module_optional: bool = False  # module absent on some qualified versions
    attr_optional: bool = False    # attr absent on some qualified versions
    impl: str = ""  # "module:attr" a forwarding lambda calls; pinned with it
    # The text to pin, for a value whose generic pin says too little.
    pin: Callable[[object], str] | None = None
    # "module:attr" of the gmlx object the stock name must resolve to. An
    # identity row pins no text: gmlx owns the object, and check_seams
    # holds only that the name still leads to it.
    owned: str = ""


def _stock_parser_specs(table) -> str:
    """The stock entries of a tool-parser table (the SPECS tuple in order,
    or the _BY_NAME dict by name) as (name, markers, priority) lines,
    specs that gmlx registered left out. An added or re-ranked upstream
    spec changes which parser a template routes to; the table's type
    alone would pin nothing."""
    from gmlx.upstream.tool_parsers import registered_names

    ours = registered_names()
    specs = (sorted(table.values(), key=lambda spec: spec.name)
             if isinstance(table, dict) else list(table))
    return "\n".join(
        repr((spec.name, getattr(spec, "template_markers", None),
              getattr(spec, "priority", None)))
        for spec in specs if spec.name not in ours)


def _in_place_image_rows(table) -> str:
    """The format rows of the model types whose images render in place
    (``render._IN_PLACE_IMAGES``). The two DeepSeek rows the loader adds
    on load are left out, so a load earlier in the process does not move
    the pin. A row that moves to another format changes what the rebuild
    hands the in-place guard."""
    from gmlx.serve.patches.render import _IN_PLACE_IMAGES

    added = ("deepseek_v4_vl", "deepseek_v41_vl")
    return "\n".join(f"{name}: {table.get(name)}"
                     for name in sorted(_IN_PLACE_IMAGES)
                     if name not in added)


def _defined_on(fn) -> str:
    """Where a method is defined, so an override in a subclass moves the
    pin."""
    return f"{fn.__module__}.{fn.__qualname__}"


SEAMS: tuple[Seam, ...] = (
    # --- batched serve scheduler + ragged decode (batch_sched / ragged_decode) ---
    Seam("gmlx._vendor.mlx_vlm.generate.ar", "BatchGenerator._next",
         "batch_sched.install_decode_priority_sched (decode-first tick, "
         "prompt-arm structure, _prompt_time_counter contract); "
         "admit_gate.install_admit_headroom_gate (admission-arm gating "
         "via the pending-list stash); serve_memtrace (tick bracket); "
         "mtp_prefill._install_l0_tenant_rows (publishes the generator's "
         "tenant rows for the step; the prompt batch is built and first "
         "stepped inside _next)",
         critical=True),
    Seam("gmlx._vendor.mlx_vlm.generate.ar", "PromptProcessingBatch.needs_processing",
         "batch_sched._last_column_only (chunks run while more than one "
         "_input_ids column remains, and generate() takes the last one: "
         "that tick is neither paced nor read as a chunk cost); "
         "prefill_decay.install_prefill_decay (reads the step only to "
         "test it for None, so it stays unwrapped)"),
    Seam("gmlx._vendor.mlx_vlm.generate.ar", "BatchGenerator.insert",
         "batch_sched arrival-merge (_unprocessed_sequences append/rebind); "
         "lora_rows (per-row prompt_kwargs carry the APC salt, positional "
         "order prompts, max_tokens, prompt_kwargs); seed_rows (returned "
         "uids); mtp_prefill._install_l0_tenant_rows (each row's "
         "_apc_tenant kwarg, None included, on the generator by returned "
         "uid)"),
    # --- qwen3_5 vendored text model (gmlx.models.qwen35) ---
    # gmlx runs mlx-vlm 0.6.15's qwen3.5 text code from vendored copies
    # (language, gated_delta, config, moe_language, moe_config) and
    # patches nothing in mlx_vlm.models.qwen3_5*. These rows pin the
    # installed upstream modules whole, so any upstream change to the
    # text code shows as drift: diff it against the vendored copy, port
    # bug fixes into the copy, and record perf or structural changes as
    # A/B candidates before regenerating.
    Seam("gmlx._vendor.mlx_vlm.models.qwen3_5.language", "",
         "drift watch for gmlx.models.qwen35.language (vendored 0.6.15)"),
    Seam("gmlx._vendor.mlx_vlm.models.qwen3_5.gated_delta", "",
         "drift watch for gmlx.models.qwen35.gated_delta (vendored 0.6.15)"),
    Seam("gmlx._vendor.mlx_vlm.models.qwen3_5.config", "",
         "drift watch for gmlx.models.qwen35.config (vendored 0.6.15 "
         "TextConfig; ModelConfig is used from here)"),
    Seam("gmlx._vendor.mlx_vlm.models.qwen3_5_moe.language", "",
         "drift watch for gmlx.models.qwen35.moe_language (vendored 0.6.15)"),
    Seam("gmlx._vendor.mlx_vlm.models.qwen3_5_moe.config", "",
         "drift watch for gmlx.models.qwen35.moe_config (vendored 0.6.15 "
         "TextConfig; ModelConfig is used from here)"),
    # --- gemma4 host-sync-free masks/offsets (gemma4_sync carries bodies).
    # Owned gemma4_text MTP trees (gemma4_owned) shadow all three rows by
    # subclass override and mirror the same bodies; the pins stay live for
    # stock-built trees (multimodal construction, GMLX_GEMMA_OWNED=0). ---
    Seam("gmlx._vendor.mlx_vlm.models.gemma4.language", "Gemma4TextModel._make_masks",
         "gemma4_sync.install_gemma4_nosync (body carried and mirror-tested, "
         "offset probe; the carried body, or the stock one under "
         "GMLX_G4_NOSYNC=0, runs under image_masks); "
         "gemma4_owned._owned_make_masks (mirror)"),
    # A correctness fix installed by the same call, ahead of its kill switch.
    Seam("gmlx._vendor.mlx_vlm.models.gemma4.language", "LanguageModel.chunked_prefill_policy",
         "gemma4_sync._install_image_chunking_policy (wrap: under "
         "bidirectional image attention answers as for a causal model, "
         "since the prefill loops keep image blocks whole; the owned MTP "
         "target inherits it)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.models.gemma4_unified.gemma4_unified",
         "Model._should_disable_chunked_prefill",
         "gemma4_sync._install_image_chunking_policy (wrap: never declines, "
         "so the per-request stamp stays off)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.models.gemma4.language", "Attention.__call__",
         "gemma4_sync.install_gemma4_nosync (body carried, offset wrap); "
         "gemma4_owned._owned_attention_call (mirror)"),
    Seam("gmlx._vendor.mlx_vlm.models.gemma4.language", "scaled_dot_product_attention",
         "gemma4_batched_sdpa (hd512 B>1 row route; left-pad tail slices); "
         "gemma4_owned._sdpa_dispatch calls the claim directly",
         owned="gmlx.cache.attention:scaled_dot_product_attention"),
    # The stock exact verifier bypasses the language module's SDPA, which
    # the gemma4 batched, cascade and sparse rebinds cover. It runs only
    # when the config sets exact_speculative_verify (default False, never
    # set by gmlx).
    Seam("gmlx._vendor.mlx_vlm.models.gemma4.config", "TextConfig",
         "gemma4 verifier gate (exact_speculative_verify default)"),
    Seam("gmlx._vendor.mlx_vlm.models.gemma4.language", "LanguageModel.__call__",
         "gemma4 verifier gate (speculative_verify routing); gemma4_owned "
         "inherits it"),
    Seam("gmlx._vendor.mlx_vlm.models.gemma4.language", "Gemma4TextModel.__call__",
         "gemma4_owned inherits it (owned masks and attention take the "
         "per-layer mask, shared KV and offset it passes, including the "
         "logits_to_keep trim before the KV-shared layers)"),
    # --- gemma4 text-only load path (mlx_lm module wrapped in text_only) ---
    Seam("gmlx._vendor.mlx_lm.models.gemma4_text", "scaled_dot_product_attention",
         "gemma4_batched_sdpa (same row route at the text-only seam)",
         owned="gmlx.cache.attention:scaled_dot_product_attention"),
    Seam("gmlx._vendor.mlx_lm.models.gemma4_text", "Attention.__call__",
         "gemma4_batched_sdpa (module-global sdpa call, cache kwarg)"),
    Seam("gmlx._vendor.mlx_lm.models.gemma4_text", "Gemma4TextModel._make_masks",
         "gemma4_batched_sdpa (one mask object per layer type; the "
         "producer->consumer pad relay keys on that identity)"),
    # --- shared-prefix cascade decode route + APC warm-batch stamp ---
    Seam("gmlx._vendor.mlx_lm.models.llama", "scaled_dot_product_attention",
         "cascade_sdpa (stamped shared-prefix batched decode)",
         owned="gmlx.cache.attention:scaled_dot_product_attention"),
    Seam("gmlx._vendor.mlx_vlm.generate.ar", "PromptProcessingBatch.__init__",
         "cascade_sdpa.install_cascade_stamp (token-prefix stamp at batch "
         "formation; rows + caches coexist in matching order here) + "
         "kvarn_serve._install_ppb_fastpath (B=1 fast path skips "
         "_make_cache; kvarn rebuilds the empty cache)",
         critical=True),
    Seam("gmlx._vendor.mlx_vlm.generate.ar", "_extend_cache",
         "cascade_sdpa.install_cascade_stamp (stamp carry across the "
         "B=1-to-batch merge lift on admission)", critical=True),
    # --- speculative / AR batch engine (the spec package owns these methods) ---
    Seam("gmlx._vendor.mlx_vlm.generate.ar", "BatchGenerator.__init__",
         "spec_engine._install_apc_manager_stash + kvarn_serve APC gate",
         critical=True),
    Seam("gmlx._vendor.mlx_vlm.generate.ar", "_make_cache",
         "kvarn_serve._install_make_cache (batch kvarn KV construction); "
         "cache.refusals (to_batch is tried before the isinstance chain)",
         critical=True),
    Seam("gmlx._vendor.mlx_vlm.generate.ar",
         "PromptProcessingBatch._store_apc_exact_checkpoints",
         "ckpt._install_ckpt_checkpoint_store (ckpt cursor rides "
         "the stock store)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.generate.ar", "PromptProcessingBatch.__init__",
         "mtp_prefill.install_full_prompt_mtp_prefill (prefill-step "
         "restore + stock-path ckpt arming + packed warm_cache kwarg "
         "recovered to fp16 before init)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.generate.ar", "PromptProcessingBatch.prompt_progress",
         "ckpt._install_restored_prompt_progress (a gmlx-owned restore "
         "reported as the row's cached tokens)"),
    Seam("gmlx._vendor.mlx_vlm.generate.ar", "GenerationBatch._step",
         "ckpt._install_plain_ckpt_decode (token accounting + "
         "decode-time snapshots)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.generate.ar", "GenerationBatch.filter",
         "ckpt._install_plain_ckpt_decode (B=1 retirement at row "
         "exit)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.generate.ar", "PromptProcessingBatch.prompt_step",
         "mtp_prefill.install_full_prompt_mtp_prefill; "
         "media_spans.install_span_aware_prompt_step (pins the block "
         "step; a chunk past right-padded row ends rides the _suffix_lens "
         "end clip and the _speculative_prefill.append output)",
         critical=True),
    Seam("gmlx._vendor.mlx_vlm.generate.ar", "generate_step",
         "media_spans.install_block_aware_generate_step (block starts "
         "ride prompt_cache_checkpoint_lengths through a pass-through "
         "prompt_cache_checkpoint; _TailForward reads the chunk call's "
         "inputs, inputs_embeds and n_to_process keywords, and the final "
         "_step call's positional ids, from model.language_model)",
         critical=True),
    Seam("gmlx._vendor.mlx_vlm.speculative.utils", "SpeculativePrefill.append",
         "media_spans._TailForward (keeps the held chunk's hidden-state "
         "list by reference; the final forward fills it)"),
    Seam("gmlx._vendor.mlx_vlm.speculative.utils", "SpeculativePrefill.finish",
         "media_spans._TailForward (joins the kept lists with the final "
         "output's hidden states, which keep the last column)"),
    Seam("gmlx._vendor.mlx_vlm.generate.ar", "PromptProcessingBatch.generate",
         "mtp_prefill.install_full_prompt_mtp_prefill; "
         "seed_rows.install_per_request_seed (row-uid publish); "
         "server_patches.mtp_thinking (thinking-hook transport, outermost)",
         critical=True),
    Seam("gmlx._vendor.mlx_vlm.generate.ar", "SpeculativeGenerationBatch.next",
         "admission.install_continuous_batch_admission", critical=True),
    Seam("gmlx._vendor.mlx_vlm.generate.ar", "SpeculativeGenerationBatch.filter",
         "admission._filter_with_release (per-row release rides the "
         "mark-finished contract; the rounds generator sheds via "
         "stop_check)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.generate.ar", "SpeculativeGenerationBatch.__len__",
         "admission._len_with_promotion (patched semantics are D-3's "
         "hazard; decision modules count rows via _orig_len only)",
         critical=True),
    Seam("gmlx._vendor.mlx_vlm.generate.ar", "GenerationBatch._eval_pending_state",
         "governor shed ordering (filter under pressure requires "
         "eval before the gather)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.generate.ar", "BatchGenerator.remove",
         "governor retire rung + tick_guard rebuild (queue pop / "
         "prompt-batch clear / decode filter semantics)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.generate.ar", "run_speculative_server_rounds",
         "spec_engine.install_owned_spec_engine", critical=True),
    Seam("gmlx._vendor.mlx_vlm.speculative.utils", "make_speculative_prompt_cache",
         "spec_engine.install_owned_b1_spec_cache (B=1 MTP single-stream "
         "caches) + kv_quant.install_spec_kv_quant (B=1 KV_BITS/kvarn)",
         critical=True),
    Seam("gmlx._vendor.mlx_vlm.generate.ar", "BatchGenerator._apc_pick_for",
         "spec_engine._bind_l1_view (L1 APC helpers)"),
    Seam("gmlx._vendor.mlx_vlm.generate.ar", "BatchGenerator._apc_exact_checkpoint_len",
         "spec_engine._bind_l1_view (L1 APC helpers)"),
    Seam("gmlx._vendor.mlx_vlm.generate.ar", "BatchGenerator._apc_extra_hash",
         "spec_engine._bind_l1_view (L1 APC helpers) + "
         "media_salt.install_media_salt (a MediaSalt passes whole)"),
    Seam("gmlx._vendor.mlx_vlm.generate.ar", "BatchGenerator._apc_media_token_ids",
         "spec_engine._bind_l1_view (L1 APC helpers) + "
         "load.vlm._mirror_media_token_ids (the guard reads self.model's "
         "config, the language model under serve) + "
         "apc_media.install_causal_serve_media (CausalMedia for qwen3.5)"),
    Seam("gmlx._vendor.mlx_vlm.generate.ar", "BatchGenerator._apc_safe_prefix_lookup_min",
         "spec_engine._bind_l1_view (L1 APC helpers) + ckpt restore floors"),
    Seam("gmlx._vendor.mlx_vlm.generate.ar", "BatchGenerator._apc_suffix_is_text_only",
         "spec_engine._bind_l1_view (L1 APC helpers) + ckpt restore gates"),
    Seam("gmlx._vendor.mlx_vlm.generate.ar", "BatchGenerator._apc_prefix_has_media_tokens",
         "spec_engine._bind_l1_view (L1 APC helpers)"),
    Seam("gmlx._vendor.mlx_vlm.generate.ar", "BatchGenerator._build_mixed_prompt_batch",
         "media_salt + apc_media (each row picks its prefix through "
         "_apc_pick_for with its own MediaSalt; a warm row's inputs_embeds "
         "and sequence-aligned kwargs, mrope position ids included, slice "
         "at its prefix length; APC-private keys never reach the model)"),
    Seam("gmlx._vendor.mlx_vlm.generate.ar",
         "PromptProcessingBatch._capture_rope_deltas_from_prompt_kwargs",
         "apc_media.install_causal_serve_media (a restored Qwen3.5 row "
         "decodes with the rope deltas of its prompt kwargs, else the "
         "language model's)"),
    Seam("gmlx._vendor.mlx_vlm.generate.ar", "APC_PRIVATE_PROMPT_KEYS",
         "media_salt.install_media_salt (_apc_semantic_hash, which carries "
         "the MediaSalt, is dropped before the model call)"),
    Seam("gmlx._vendor.mlx_vlm.apc", "apc_lookup_plan",
         "spec_engine._L1View (_apc_pick_for takes the legacy branch: the "
         "view has no coordinator)"),
    Seam("gmlx._vendor.mlx_vlm.generate.ar", "_slice_sequence_aligned_prompt_kwarg",
         "spec_engine._trim_prompt_columns (mrope position ids slice on "
         "their own sequence axis)", critical=True),
    # --- server engine ---
    Seam("gmlx._vendor.mlx_vlm.server.generation", "load_model_resources",
         "server_bridge_vlm (GGUF model resource loader)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.server.generation", "ResponseGenerator._make_sampler",
         "server_patches.install_fast_sampler"),
    Seam("gmlx._vendor.mlx_vlm.server.generation", "_check_configured_context_budget",
         "server_patches.install_context_overflow_wording (replaced; "
         "same check, overflow text clients match)"),
    Seam("gmlx._vendor.mlx_vlm.server.generation", "get_configured_context_limit",
         "server_patches.install_context_overflow_wording (wrapped; "
         "a per-model max_kv_size wins over the process-wide limit)"),
    Seam("gmlx._vendor.mlx_vlm.server.generation", "ResponseGenerator.generate",
         "mem_preflight.install_memory_preflight; "
         "server_patches.mtp_thinking (thinking_budget deferral); "
         "media_salt.install_media_salt (a request scope: the salt is "
         "hash_image_payload(pixel_values=raw['pixel_values']) after "
         "_preprocess_request, then semantic_extra_hash, before the "
         "request queues)",
         critical=True),
    Seam("gmlx._vendor.mlx_vlm.server.generation",
         "ResponseGenerator._make_thinking_budget_criteria",
         "chat_behavior.install_thinking_budget_fix (rebind); "
         "server_patches.mtp_thinking (restore + hook attach; runtime "
         "chain mtp -> tbfix)",
         critical=True),
    Seam("gmlx._vendor.mlx_vlm.server.generation",
         "ResponseGenerator._make_logits_processors",
         "seed_rows.install_per_request_seed; lora_rows.install_row_channel "
         "(engine-thread handoff to the insert that takes the result as "
         "an argument)",
         critical=True),
    Seam("gmlx._vendor.mlx_vlm.server.generation", "ResponseGenerator._step",
         "server_patches.row_failed (permanently failed rows delivered "
         "to their response queues on the engine thread)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.server.generation", "GenerationArguments.to_template_kwargs",
         "server_patches (chat_template_kwargs transform)"),
    Seam("gmlx._vendor.mlx_vlm.server.generation", "ResponseGenerator._preprocess_request",
         "server_patches.install_retire_render_capture (ids hop + "
         "tokenize path for the next-turn retirement key) + "
         "media_salt.install_media_salt (hands the request's inputs to "
         "its image hash)"),
    Seam("gmlx._vendor.mlx_vlm.server.generation", "ResponseGenerator._initialize_model",
         "media_salt.install_media_salt (the image hash reads the "
         "model, processor and apc_mode this sets)"),
    Seam("gmlx._vendor.mlx_vlm.server.generation", "ResponseGenerator._cpu_preprocess",
         "single_bos_fix (calls the prepare_inputs binding with prompts= "
         "and a model-type add_special_tokens)"),
    Seam("gmlx._vendor.mlx_vlm.server.generation", "prepare_inputs",
         "single_bos_fix (module attr: a prompt that opens with BOS is "
         "encoded without special tokens)"),
    # The route modules' render, parser and tool-module bindings are
    # _protocol_deps lambdas that look the mlx_vlm.server attribute up per
    # call; impl pins that attribute with the binding.
    Seam("gmlx._vendor.mlx_vlm.server.openai", "apply_chat_template",
         "server_patches.install_retire_render_capture (render-context "
         "memo, module attr) + render.install_faithful_history (inner "
         "key-merge wrap)", impl="gmlx._vendor.mlx_vlm.server:apply_chat_template"),
    Seam("gmlx._vendor.mlx_vlm.server.openai", "_normalize_instruction_messages",
         "server_patches.install_role_normalization (wrapped: a call with "
         "the messages alone, the chat completions one, keeps developer "
         "messages in place) + estimate._estimate_bound (follows the "
         "chat route's merge)"),
    Seam("gmlx._vendor.mlx_vlm.server.openai", "chat_completions_endpoint",
         "render.install_chat_image_placement (route wrap: a user "
         "message's content with an image part passes through the route's "
         "text extraction call, which keeps it)"),
    Seam("gmlx._vendor.mlx_vlm.server.openai", "extract_text_from_content",
         "render.install_chat_image_placement (wrapped: marked content "
         "returns its text parts and one image marker per image, which "
         "apply_chat_template counts per message)"),
    Seam("gmlx._vendor.mlx_vlm.prompt_utils", "extract_text_from_content",
         "render.install_chat_image_placement (the stock extraction of "
         "text-only content, behind the openai binding's wrap) + "
         "estimate._estimate_bound (text of each message) + "
         "render._images_in_place (a run of text parts joins and strips as "
         "the extraction joins and strips a message's text)"),
    Seam("gmlx._vendor.mlx_vlm.prompt_utils", "_content_media_count",
         "render.install_chat_image_placement (apply_chat_template counts "
         "the image markers of each message)"),
    Seam("gmlx._vendor.mlx_vlm.prompt_utils", "MODEL_CONFIG",
         "render._images_in_place (each in-place model type rebuilds a user "
         "message as a list of one text part and its image parts) + "
         "render._tool_images_to_user (a model type outside the table takes "
         "the text path of apply_chat_template, which has no image path)",
         pin=_in_place_image_rows),
    Seam("gmlx._vendor.mlx_vlm.prompt_utils", "MessageFormatter._format_list_with_image",
         "render._images_in_place (a list format rebuild: one text part, "
         "the images before it or after it, audio parts last)"),
    Seam("gmlx._vendor.mlx_vlm.prompt_utils",
         "MessageFormatter._format_list_with_image_type",
         "render._images_in_place (a typed list rebuild: one text part, "
         "the images before it, audio parts last)"),
    Seam("gmlx._vendor.mlx_vlm.prompt_utils", "MessageBuilder",
         "render._images_in_place (the text part it copies for each run of "
         "text, and the image part it places)"),
    Seam("gmlx._vendor.mlx_vlm.prompt_utils", "get_message_json",
         "system_content_fix (a list-format model type gets its system and "
         "developer text as a list of text parts)"),
    Seam("gmlx._vendor.mlx_vlm.server.anthropic", "_preflight_stream_context_budget",
         "server_patches.install_context_overflow_wording (wrapped: the "
         "overflow 400 is recorded for the endpoint wrapper)"),
    Seam("gmlx._vendor.mlx_vlm.server.anthropic", "anthropic_messages_endpoint",
         "server_patches.install_context_overflow_wording (the streaming "
         "preflight's 400 reaches the catch-all 500; wrapper answers 400)"),
    Seam("gmlx._vendor.mlx_vlm.server.anthropic", "apply_chat_template",
         "server_patches._common._render_target_modules (faithful "
         "history, retire capture, thinking seed, and developer-role "
         "normalization wrap every captured render binding)",
         impl="gmlx._vendor.mlx_vlm.server:apply_chat_template"),
    # The Messages route's conversion step: gmlx claims it and calls the
    # stock helpers as the stock body does.
    Seam("gmlx._vendor.mlx_vlm.server.anthropic", "_anthropic_messages_to_internal",
         "messages_route.install_messages_route (replaced; the stock body, "
         "with a user message that holds an image as its text runs and "
         "image markers in block order, and its tool results ahead of its "
         "text)"),
    Seam("gmlx._vendor.mlx_vlm.server.anthropic",
         "_anthropic_content_blocks_to_text_and_tools",
         "messages_route.install_messages_route (each message's text, tool "
         "calls and tool results; appends every image ref in block order)"),
    Seam("gmlx._vendor.mlx_vlm.server.anthropic", "_anthropic_image_source_to_ref",
         "messages_route._placed (an image block the conversion keeps)"),
    Seam("gmlx._vendor.mlx_vlm.server.anthropic", "_anthropic_tool_result_content_to_openai",
         "render._tool_images_to_user (a tool result with an image converts "
         "to text parts and one image marker per image; without one, to "
         "its text joined with newlines)"),
    Seam("gmlx._vendor.mlx_vlm.server.anthropic", "_as_plain_dict",
         "messages_route._placed (reads each block as the stock "
         "conversion reads it)"),
    Seam("gmlx._vendor.mlx_vlm.server.anthropic", "_anthropic_system_text",
         "messages_route.install_messages_route (the system message)"),
    Seam("gmlx._vendor.mlx_vlm.server.anthropic", "_anthropic_tools_to_openai",
         "messages_route.install_messages_route (the request's tools)"),
    Seam("gmlx._vendor.mlx_vlm.server.anthropic", "_anthropic_tool_choice_to_openai",
         "messages_route.install_messages_route (the request's tool "
         "choice)"),
    Seam("gmlx._vendor.mlx_vlm.server.anthropic", "_prepare_chat_tool_choice",
         "messages_route.install_messages_route (enforces the tool choice "
         "on the converted messages)"),
    # The Responses route builds its messages in place: the in-place
    # render and the Messages tool image placement follow it.
    Seam("gmlx._vendor.mlx_vlm.server.responses_state", "_append_response_item_to_prompt",
         "render._images_in_place (a message's text parts and one image "
         "marker per image, in content order) + render._tool_images_to_user "
         "(a tool output's images go to a user message of their own right "
         "after it)"),
    Seam("gmlx._vendor.mlx_vlm.server.openai", "responses_endpoint",
         "render.install_faithful_history (renders the messages and image "
         "list _response_items_to_chat builds, in the order it builds "
         "them)"),
    Seam("gmlx._vendor.mlx_vlm.server.app", "_protocol_deps",
         "server_patches._common._render_target_modules (the route "
         "modules copy their bindings from these members at route "
         "registration; gmlx wraps the member beside both copies)"),
    Seam("gmlx._vendor.mlx_vlm.prompt_utils", "apply_chat_template",
         "server_patches.render.install_faithful_history "
         "(return_messages contract: one rebuilt message per readable "
         "input item; tool passthrough branch) + render._images_in_place "
         "(each user message gets as many images as it holds markers, "
         "and the rest go to the last user message) + "
         "render._TOOL_IMAGES (a tool message's image markers count toward "
         "the rest, so the wrapper leaves them out of num_images) + "
         "render._install_nested_in_place (wrapped: the same, for the "
         "rebuild a processor runs inside a render that keeps them)"),
    Seam("gmlx._vendor.mlx_vlm.prompt_utils", "_normalize_tool_message",
         "render._TOOL_IMAGES (a tool message passes through with its "
         "content list, image markers included)"),
    Seam("gmlx._vendor.mlx_vlm.prompt_utils", "get_chat_template",
         "server_patches.render.install_faithful_history (owned render "
         "tail mirrors the stock function's) + "
         "reasoning.install_template_default_thinking (absent "
         "enable_thinking seeded as Jinja Undefined so the 0.6.15 False "
         "injection stands down) + system_content_fix (a text-only system "
         "or developer message reaches the template as a string) + "
         "render._install_nested_in_place (wrapped: within an in-place "
         "render, the images of a nested rebuild go back in place)"),
    Seam("gmlx._vendor.mlx_vlm.server.openai", "_split_thinking",
         "retire_key.build_assistant_message (response-shape mirror)"),
    Seam("gmlx._vendor.mlx_vlm.server.openai", "process_tool_calls",
         "retire_key.build_assistant_message (response-shape mirror)"),
    Seam("gmlx._vendor.mlx_vlm.server.openai", "_infer_tool_parser_from_processor",
         "retire_key.build_assistant_message (response-shape mirror)",
         impl="gmlx._vendor.mlx_vlm.server:_infer_tool_parser_from_processor"),
    Seam("gmlx._vendor.mlx_vlm.server", "_infer_tool_parser_from_processor",
         "server_patches.apc.install_retire_render_capture (wrapped: "
         "records the request's tool_parser override for retire_key; the "
         "_protocol_deps lambda looks the package attribute up per call)"),
    Seam("gmlx._vendor.mlx_vlm.server.openai", "load_tool_module",
         "retire_key.build_assistant_message (response-shape mirror)",
         impl="gmlx._vendor.mlx_vlm.server:load_tool_module"),
    Seam("gmlx._vendor.mlx_vlm.server.responses_state",
         "ThinkingStreamState._build_open_close_markers",
         "retire_key._thinking_markers (splitter default marker pairs "
         "for the mid-decode virtual closer)"),
    Seam("gmlx._vendor.mlx_vlm.server.app", "_split_thinking_text",
         "server_patches.chat_behavior.install_stream_thinking_seed "
         "(non-stream truncated-thinking classification, module attr "
         "looked up per call by app._split_thinking)"),
    Seam("gmlx._vendor.mlx_vlm.server.runtime", "runtime",
         "residency._RuntimeProxy wrap", critical=True),
    Seam("gmlx._vendor.mlx_vlm.server.app", "app",
         "server_patches route surgery / server.py uvicorn root",
         critical=True),
    Seam("gmlx._vendor.mlx_vlm.speculative.drafters", "load_drafter",
         "server_bridge_vlm (in-memory MTP drafter injection)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.utils", "get_model_path",
         "server_patches (HF download gate)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.utils", "StoppingCriteria.__call__",
         "server_patches (ignore-EOS)"),
    Seam("gmlx._vendor.mlx_vlm.utils", "process_inputs",
         "single_bos_fix (module attr: the flag reaches only a processor "
         "that names it; inside a generation call that drops the "
         "duplicate BOS, forwarded to a known type through **kwargs)"),
    Seam("gmlx._vendor.mlx_vlm.utils", "process_inputs_with_fallback",
         "single_bos_fix (calls the process_inputs module attr with the "
         "add_special_tokens it was given)"),
    Seam("gmlx._vendor.mlx_vlm.utils", "prepare_inputs",
         "single_bos_fix (the body every prepare_inputs binding runs: a "
         "text prompt reaches the tokenizer with add_special_tokens, a "
         "media prompt reaches process_inputs_with_fallback with it)"),
    Seam("gmlx._vendor.mlx_vlm.server.embeddings", "_embed_image",
         "single_bos_fix (embeddings encode through their own stock "
         "prepare_inputs binding)"),
    Seam("gmlx._vendor.mlx_vlm.server.reranking", "_score_vl_batch",
         "single_bos_fix (reranking encodes through its own stock "
         "prepare_inputs binding)"),
    Seam("gmlx._vendor.mlx_vlm.generate.dispatch", "_prepare_generation_inputs",
         "single_bos_fix (calls the prepare_inputs binding with prompts= "
         "and a model-type add_special_tokens)"),
    Seam("gmlx._vendor.mlx_vlm.generate.dispatch", "prepare_inputs",
         "single_bos_fix (module attr, generate path)"),
    Seam("gmlx._vendor.mlx_vlm.server.anthropic", "anthropic_count_tokens_endpoint",
         "single_bos_fix (the fallback count calls the prepare_inputs "
         "binding with prompts= and no add_special_tokens, after "
         "get_cached_model filled the model cache)"),
    Seam("gmlx._vendor.mlx_vlm.server.anthropic", "prepare_inputs",
         "single_bos_fix (module attr: the count adds special tokens as "
         "generation does, and drops a duplicate BOS)"),
    Seam("gmlx._vendor.mlx_vlm.server.openai", "responses_input_tokens_endpoint",
         "single_bos_fix (the fallback count calls the prepare_inputs "
         "binding with prompts= and no add_special_tokens, after "
         "get_cached_model filled the model cache)"),
    Seam("gmlx._vendor.mlx_vlm.server.openai", "prepare_inputs",
         "single_bos_fix (module attr: the count adds special tokens as "
         "generation does, and drops a duplicate BOS)"),
    Seam("gmlx._vendor.mlx_vlm.utils", "should_add_special_tokens",
         "single_bos_fix (the count default: the model type rule "
         "generation applies; _template_owns_markers: the types whose "
         "markers the chat template writes, laguna among them)"),
    Seam("gmlx._vendor.mlx_vlm.models.laguna.processing_laguna", "LagunaProcessor.__init__",
         "single_bos_fix._template_owns_markers (the processor names its "
         "model_type)"),
    Seam("gmlx._vendor.mlx_vlm.server.app", "get_cached_model",
         "single_bos_fix (the count reads the processor and config this "
         "stores in the runtime's model cache)"),
    Seam("gmlx._vendor.mlx_vlm.models.gemma4.processing_gemma4", "Gemma4Processor.__call__",
         "single_bos_fix (a known forwarding type: **kwargs reach "
         "self.tokenizer) + render._IN_PLACE_IMAGES (expands the image "
         "markers in prompt order, one image each)"),
    Seam("gmlx._vendor.mlx_vlm.models.gemma4.processing_gemma4",
         "Gemma4Processor.apply_chat_template",
         "render._install_nested_in_place (renders by running the stock "
         "rebuild once more, which renders through the prompt_utils "
         "get_chat_template and is looked up on prompt_utils per call)"),
    Seam("gmlx._vendor.mlx_vlm.models.gemma4_unified.processing_gemma4_unified",
         "Gemma4UnifiedProcessor.apply_chat_template",
         "render._install_nested_in_place (the 12B processor renders with "
         "the Gemma4Processor method, so through the same nested rebuild)",
         pin=_defined_on),
    Seam("gmlx._vendor.mlx_vlm.models.gemma4.processing_gemma4",
         "Gemma4Processor._extract_media_from_messages",
         "render._install_nested_in_place (the rebuild it runs counts the "
         "image items of every message, a tool message's too)"),
    Seam("gmlx._vendor.mlx_vlm.models.pixtral.processing_pixtral",
         "PixtralProcessor.__call__",
         "single_bos_fix (a known forwarding type: **kwargs reach "
         "self.tokenizer) + render._IN_PLACE_IMAGES (expands the image "
         "markers in prompt order, one image each)"),
    Seam("gmlx._vendor.mlx_vlm.models.mistral3.processing_mistral3",
         "Mistral3Processor.__call__",
         "render._IN_PLACE_IMAGES (expands the image markers in prompt "
         "order, one image each)"),
    Seam("gmlx._vendor.mlx_vlm.models.qwen3_omni_moe.processing_qwen3_omni_moe",
         "Qwen3OmniMoeProcessor.replace_multimodal_special_tokens",
         "render._IN_PLACE_IMAGES (expands the image markers in prompt "
         "order, one image each)"),
    Seam("gmlx._vendor.mlx_vlm.models.qwen3_vl.processing_qwen3_vl",
         "Qwen3VLProcessor.__call__",
         "media_salt._expected_tokens (an image expands to t*h*w of its "
         "image_grid_thw row over merge_size**2) + render._IN_PLACE_IMAGES "
         "(expands the image markers in prompt order, one image each)"),
    Seam("gmlx._vendor.mlx_vlm.models.gemma4_unified.processing_gemma4_unified",
         "Gemma4UnifiedImageProcessor.preprocess",
         "media_salt._expected_tokens (the 12B rule: one token per patch "
         "row of image_position_ids)"),
    Seam("gmlx._vendor.mlx_vlm.models.gemma4_unified.processing_gemma4_unified",
         "_pad_patches",
         "media_salt._expected_tokens (the 12B rule: padding position rows "
         "are -1)"),
    Seam("gmlx._vendor.mlx_vlm.models.gemma4.processing_gemma4",
         "Gemma4ImageProcessor.preprocess",
         "media_salt._expected_tokens (a [C, H, W] image expands to one "
         "token per pooling_kernel_size**2 patches of patch_size pixels)"),
    Seam("gmlx._vendor.mlx_vlm.models.kimi_k25.processing_kimi_k25",
         "KimiK25Processor.__call__",
         "media_salt._expected_tokens (an image expands to h*w of its "
         "image_grid_hws row over the merge kernel's area) + "
         "render._IN_PLACE_IMAGES (expands the image markers in prompt "
         "order, one image each)"),
    Seam("gmlx._vendor.mlx_vlm.models.kimi_k25.processing_kimi_k25",
         "KimiK25ImageProcessor.preprocess",
         "media_salt._image_rows (each image's patches are its image_grid_hws "
         "row's h*w rows of the concatenated pixel_values)"),
    Seam("gmlx._vendor.mlx_vlm.models.kimi_k25.processing_kimi_k25",
         "KimiK25ImageProcessor.patchify",
         "media_salt._image_rows (an image patchifies to one row per cell "
         "of its (H // patch_size, W // patch_size) grid)"),
    Seam("gmlx._vendor.mlx_vlm.models.llava.processing_llava", "LlavaProcessor.__call__",
         "media_salt._expected_tokens (every image expands to the first "
         "image's (H // patch_size) * (W // patch_size) patches plus "
         "num_additional_image_tokens, less one under the default "
         "strategy)"),
    # --- tool-parser registry (upstream.tool_parsers.register_tool_parser,
    # used by the hy_v3, hy_v4, deepseek_v41, muse_glimmer and
    # function_gemma tools) ---
    Seam("gmlx._vendor.mlx_vlm.tools.registry", "ParserSpec",
         "tool_parsers.register_tool_parser (spec built from name, marker "
         "groups and priority)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.tools.registry", "SPECS",
         "tool_parsers.register_tool_parser (priority-200 spec prepended; "
         "module global rebound)", critical=True, pin=_stock_parser_specs),
    Seam("gmlx._vendor.mlx_vlm.tools.registry", "_BY_NAME",
         "tool_parsers.register_tool_parser (name lookup; an upstream spec "
         "of the same name keeps its markers)", critical=True,
         pin=_stock_parser_specs),
    Seam("gmlx._vendor.mlx_vlm.tools.registry", "_infer_tool_parser",
         "tool_parsers.register_tool_parser (reads the SPECS module global "
         "per call, highest priority first)"),
    Seam("gmlx._vendor.mlx_vlm.tools.registry", "load_tool_module",
         "tool_parsers.register_tool_parser (reads the GMLX_PARSERS table "
         "before the parsers package)"),
    Seam("gmlx._vendor.mlx_vlm.tools", "SPECS",
         "tool_parsers.register_tool_parser (the request schemas validate "
         "a tool_parser override against this re-export)",
         pin=_stock_parser_specs),
    # --- APC internals (lone-harvest patch, gmlx manager subclass, apc_pooling) ---
    Seam("gmlx._vendor.mlx_vlm.apc", "harvest_blocks_from_batch_cache",
         "server_patches.install_apc_lone_harvest", critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc", "_clone_layer_major_kv_cache_for_apc",
         "apc_manager.GmlxAPCManager.store_kv_blocks", critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc", "_sequence_hash",
         "apc_manager.GmlxAPCManager.store_kv_blocks", critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc", "APCExactCacheEntry",
         "apc_manager.GmlxAPCManager.store_kv_blocks", critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc", "_hash_tokens",
         "apc_manager.GmlxAPCManager.store_kv_blocks + "
         "apc_manager.block_chain", critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc", "_DiskLayerMajorBlock",
         "apc_manager.GmlxAPCManager.store_kv_blocks", critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc", "_copy_mlx_array",
         "apc_manager.GmlxAPCManager.store_kv_blocks", critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc", "SEED_PARENT_HASH",
         "apc_manager.GmlxAPCManager.store_kv_blocks + "
         "apc_manager.block_chain (the seed of every chain it walks) + "
         "fresh_gate._covered_len (through block_chain)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc", "DEFAULT_BLOCK_SIZE",
         "apc_manager.build_apc_manager (from_env mirror)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc", "DEFAULT_NUM_BLOCKS",
         "apc_manager.build_apc_manager (from_env mirror)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc", "DiskBlockStore",
         "apc_manager.build_apc_manager (from_env mirror)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc", "from_env",
         "apc_manager.build_apc_manager (mirror: env knobs and their "
         "parsing; disk tier stays opt-in)"),
    Seam("gmlx._vendor.mlx_vlm.apc", "APCManager.store_kv_blocks",
         "apc_manager.GmlxAPCManager.store_kv_blocks (mirror: re-sync the "
         "copy on change; each block hashes under the salt at its end)",
         critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc", "_cache_nbytes",
         "apc_manager.GmlxAPCManager.store_kv_blocks", critical=True,
         owned="gmlx.cache.kv_cache:cache_nbytes"),
    Seam("gmlx._vendor.mlx_vlm.apc", "APCBlock",
         "apc_manager.GmlxAPCManager._grow_pool", critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc", "APCManager._make_room",
         "apc_manager.GmlxAPCManager._prefill_reserve_bytes (reads 0 after "
         "autosize; a renamed reserve field would escape the property), "
         "GmlxAPCManager._memory_headroom (the memory_evictions count and "
         "the evict, clear, read-again order) and "
         "GmlxAPCManager._make_room (scopes the headroom marks to the "
         "call)"),
    Seam("gmlx._vendor.mlx_vlm.apc", "APCManager._memory_headroom",
         "apc_manager.GmlxAPCManager._memory_headroom (drops a cached "
         "free-RAM reading after an eviction or a cache clear inside "
         "_make_room)"),
    Seam("gmlx._vendor.mlx_vlm.apc", "APCManager.prepare_prefill",
         "apc_manager.GmlxAPCManager._prefill_reserve_bytes (disk flush "
         "stays stock; the reserve it sets reads 0 after autosize)"),
    Seam("gmlx._vendor.mlx_vlm.apc", "_free_ram_bytes",
         "apc_manager._install_fork_free_free_ram (disk-restore gates and "
         "the _make_room headroom read, rebound to the in-process mach "
         "read, else cached; stock forks vm_stat)"),
    # The prefix-cache plan and the clone and merge arms live in
    # apc_adapters; the apc functions below import them per call.
    Seam("gmlx._vendor.mlx_vlm.apc_adapters", "register_capability",
         "apc_pooling + kvarn_apc (CHECKPOINT rows for the pooling and "
         "kvarn cache classes and their batch classes)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc_adapters", "Capability",
         "apc_pooling + kvarn_apc (CHECKPOINT member)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc_adapters", "clone_cache_entry",
         "apc_pooling + kvarn_apc (wrapped: pooling and kvarn clone arms; "
         "CacheList members recurse through the module global)",
         critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc_adapters", "merge_cache_entries",
         "apc_pooling + kvarn_apc (wrapped: pooling and kvarn merge arms)",
         critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc_adapters", "build_prefix_cache_plan",
         "kvarn_apc plan wrap (a stamped kvarn model pages nothing, so its "
         "strategy is checkpoint)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc_coordinator", "build_prefix_cache_plan",
         "kvarn_apc plan wrap (the coordinator's own binding, rebound "
         "beside the adapter's)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc", "_merge_exact_cache_entries",
         "apc_pooling + kvarn_apc merge arms (imports "
         "apc_adapters.merge_cache_entries per call)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc", "_read_safetensors_tensor",
         "apc_pooling (disk-tier zero-width spill; wrapped reader)",
         critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc", "_clone_cache_entry_for_apc",
         "apc_pooling + kvarn_apc clone arms (imports "
         "apc_adapters.clone_cache_entry per call)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc", "multimodal_token_ids_from_config",
         "apc_media.install_media_token_ids (config media_token_ids fold "
         "for expanded image blocks) + load.vlm._mirror_media_token_ids "
         "(gemma-4 container ids onto a bidirectional text config)",
         critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc", "media_safe_prefix_min",
         "apc_media.install_media_token_ids (zero floor for "
         "WholeMediaBlocks)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc", "prefix_leaves_text_only_suffix",
         "apc_media.install_media_token_ids (a WholeMediaBlocks prefix "
         "ends outside every run)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc", "adjust_prefix_to_text_suffix_boundary",
         "apc_media.install_media_token_ids (a WholeMediaBlocks column "
         "inside a run moves to its end)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc", "media_token_spans",
         "apc_media.safe_prefix and the WholeMediaBlocks guards (half-open "
         "runs of media ids, in prompt order)"),
    Seam("gmlx._vendor.mlx_vlm.apc", "prefix_contains_media_tokens",
         "apc_media (the stock pick's prefix_has_media test reads a "
         "CausalMedia or WholeMediaBlocks set as a plain id set)"),
    Seam("gmlx._vendor.mlx_vlm.apc", "_tensor_content_hash",
         "media_salt.install_media_salt (hash_image_payload outside a "
         "request keeps the stock content hash)"),
    Seam("gmlx._vendor.mlx_vlm.apc", "hash_image_payload",
         "media_salt.install_media_salt (a request's images hash once, "
         "per image, into spans; a list of images of different sizes "
         "hashes by content, not as 0)"),
    Seam("gmlx._vendor.mlx_vlm.apc", "semantic_extra_hash",
         "media_salt.install_media_salt (spans and the text salt make "
         "the MediaSalt)"),
    Seam("gmlx._vendor.mlx_vlm.server.generation", "ResponseGenerator._gpu_embed",
         "media_salt.install_media_salt (the MediaSalt reaches "
         "gen_kwargs['_apc_semantic_hash'] unchanged)"),
    Seam("gmlx._vendor.mlx_vlm.apc", "APCManager.lookup_exact_cache",
         "apc_manager.GmlxAPCManager.lookup_exact_cache (one call per "
         "MediaSalt window: min exclusive, a positive max inclusive)"),
    Seam("gmlx._vendor.mlx_vlm.apc", "APCManager.store_exact_cache",
         "apc_manager.GmlxAPCManager.store_exact_cache (keys on the salt "
         "at the entry length)"),
    Seam("gmlx._vendor.mlx_vlm.apc", "APCManager.lookup_prefix",
         "apc_manager.GmlxAPCManager.lookup_prefix (mirror for a "
         "MediaSalt: the stock walk over block_chain, one hit or miss; "
         "re-sync on change) + fresh_gate._covered_len (the same walk)",
         critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc", "APCManager.lookup_prefix_disk_cache",
         "apc_manager.GmlxAPCManager.lookup_prefix_disk_cache (mirror for "
         "a MediaSalt: the stock restore over block_chain, each shard "
         "block's salt checked at its end; re-sync on change)",
         critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc", "make_warm_kv_cache_from_layers",
         "apc_manager.GmlxAPCManager.lookup_prefix_disk_cache (the "
         "MediaSalt mirror builds the warm cache from the loaded prefix)",
         critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc", "DiskBlockStore.load_layer_major_prefix",
         "apc_manager.GmlxAPCManager.lookup_prefix_disk_cache (the "
         "MediaSalt mirror: keys, values and per-block metadata with "
         "token_ids and extra_hash)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc", "DiskBlockStore.prefix_cache_bytes",
         "apc_manager.GmlxAPCManager.lookup_prefix_disk_cache (the "
         "MediaSalt mirror sizes its _make_room call)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.generate.ar", "BatchGenerator.__init__",
         "apc_pooling.install_pooled_prefill_batch_gate (prompt batches "
         "stay B=1 on pooling-cache models; to_batch_cache has no pooled "
         "arm)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.generate.ar", "_extend_cache",
         "apc_pooling.install_batched_cachelist_admission (promotion test "
         "looks through CacheList so an already-batched one is not merged "
         "twice)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc", "_safetensors_dtype_info",
         "apc_pooling + kvarn_apc + apc_qsa shard readers (the stock "
         "decoder reads the packed uint32 and uint8 planes)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc", "model_apc_mode",
         "residency + spec_engine L1 probe; kvarn_apc wraps it (a stamped "
         "model answers from the plan, as the coordinator does)",
         critical=True),
    Seam("gmlx._vendor.mlx_vlm.generate", "maybe_quantize_kv_cache",
         "apc_pooling (rotating-safe KV-quant replacement); cache.refusals "
         "(the hybrid and TurboQuant arms skip preserve_auxiliary_kv_state)",
         critical=True),
    # --- owned cache binding (gmlx.cache.bind) ---
    # Stock modules that import cache names by name before a gmlx wrapper
    # could reach them. The identity rows hold that each name is the owned
    # object; the fingerprint rows pin stock code the binding relies on.
    Seam("gmlx._vendor.mlx_vlm.speculative.cache_state", "RotatingKVCache",
         "rollback transaction (rotating branch on the owned class)",
         owned="gmlx.cache.kv_cache:RotatingKVCache"),
    Seam("gmlx._vendor.mlx_vlm.speculative.cache_state", "BatchRotatingKVCache",
         "rollback transaction (rotating branch on the owned batch class)",
         owned="gmlx.cache.kv_cache:BatchRotatingKVCache"),
    Seam("gmlx._vendor.mlx_vlm.speculative.cache_state", "CacheList",
         "rollback transaction (recurses into owned cache lists)",
         owned="gmlx.cache.kv_cache:CacheList"),
    Seam("gmlx._vendor.mlx_lm.generate", "_make_cache",
         "cache.bind (type(c) is KVCache batches the owned KVCache and "
         "rejects its subclasses)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc", "_resolve_checkpoint_class",
         "cache.checkpoint_claim (wrapped: the prefix check also accepts "
         "gmlx.cache.kv_cache)", critical=True),
    Seam("gmlx._vendor.mlx_vlm.apc_adapters", "_has_explicit_snapshot_contract",
         "cache.bind (compares a class's members against the owned "
         "_BaseCache.__dict__)"),
    Seam("gmlx._vendor.mlx_vlm.apc_adapters", "_custom_state_contract",
         "cache.bind (same _BaseCache.__dict__ comparison)"),
    Seam("gmlx._vendor.mlx_vlm.apc_adapters", "_apc_type_tables",
         "cache.bind late path (resets _APC_EXACT_TYPES and "
         "_APC_BLOCK_TYPES so they refill with owned classes)"),
    Seam("gmlx._vendor.mlx_lm.models.cache", "load_prompt_cache",
         "cache.bind.register (resolves each registered gmlx cache class "
         "by name in the module globals)"),
    Seam("gmlx._vendor.mlx_vlm.server.runtime", "ModelCacheRegistry",
         "residency pool / test_wire_contract fake loader"),
    # --- vendored switch_layers ---
    Seam("gmlx._vendor.mlx_vlm.models.switch_layers", "SwitchLinear.__call__",
         "modules.switch_layer_types (dual-origin leaf swap)",
         critical=True),
    Seam("gmlx._vendor.mlx_vlm.models.switch_layers", "SwitchGLU.__call__",
         "modules.switch_layer_types (dual-origin fused GLU); "
         "modules.install_switch_glu_block_routing (a block of 2 to "
         "DECODE_BLOCK_SIZE positions loops per token when the exact "
         "kernels decline)",
         critical=True),
    Seam("gmlx._vendor.mlx_vlm.models.switch_layers", "SwitchGLU._combine",
         "modules._BlockRoutedSwitchGLU (block call, then the stock "
         "weights, shared and residual combine)",
         critical=True),
    # --- structured reads on DiffusionGemma (systemone) ---
    # The read presents a batch-N prompt K/V to the stock decoder through
    # per-layer views that carry keys, offset and decoder_state.
    Seam("gmlx._vendor.mlx_vlm.models.diffusion_gemma.language", "Attention.__call__",
         "systemone.engine (decoder reads _cache_state, SDPA without cache)"),
    Seam("gmlx._vendor.mlx_vlm.models.diffusion_gemma.language", "DecoderModel.__call__",
         "systemone.engine (canvas forward with cache views)"),
    Seam("gmlx._vendor.mlx_vlm.models.diffusion_gemma.language",
         "DecoderModel._make_decoder_masks",
         "systemone.engine (views make both masks None)"),
    Seam("gmlx._vendor.mlx_vlm.models.diffusion_gemma.language", "DecoderModel._embed_canvas",
         "systemone.engine (self-conditioning input contract)"),
    Seam("gmlx._vendor.mlx_vlm.models.diffusion_gemma.language", "SelfConditioning.__call__",
         "systemone.engine (soft embedding shape)"),
    Seam("gmlx._vendor.mlx_vlm.models.diffusion_gemma.language", "_cache_state",
         "systemone.engine (view K/V source, decoder_state attribute)"),
    Seam("gmlx._vendor.mlx_vlm.models.diffusion_gemma.language", "_cache_offset",
         "systemone.engine (view offset attribute)"),
    Seam("gmlx._vendor.mlx_vlm.models.diffusion_gemma.language", "EncoderModel.make_cache",
         "systemone.engine (prompt cache layer types)"),
    Seam("gmlx._vendor.mlx_vlm.models.diffusion_gemma.language",
         "DiffusionGemma4Backbone.diffusion_prefill_cache",
         "systemone.engine.StructuredReader.prefill"),
    Seam("gmlx._vendor.mlx_vlm.models.diffusion_gemma.language",
         "DiffusionGemma4Backbone.diffusion_update_cache",
         "systemone.engine.PromptCache.extend"),
    Seam("gmlx._vendor.mlx_vlm.models.diffusion_gemma.language",
         "DiffusionGemma4Backbone.diffusion_decoder_masks",
         "systemone.engine (mask contract the views rely on)"),
    Seam("gmlx._vendor.mlx_vlm.models.diffusion_gemma.language", "make_compiled_softcap",
         "systemone.engine (model._softcap on the read logits)"),
    Seam("gmlx._vendor.mlx_vlm.models.diffusion_gemma.diffusion_gemma", "Model.__init__",
         "systemone.engine (owner of _softcap)"),
    Seam("gmlx._vendor.mlx_vlm.generate.common", "generation_stream",
         "systemone.engine.engine_scope"),
    Seam("gmlx._vendor.mlx_vlm.generate.diffusion", "stream_diffusion_generate",
         "systemone.engine.StructuredReader.think"),
    Seam("gmlx._vendor.mlx_vlm.generate.diffusion", "_diffusion_entropy_transfer_mask",
         "systemone.engine.entropy_transfer_mask (owned copy; re-mirror "
         "on upstream change)"),
    Seam("gmlx._vendor.mlx_lm.generate", "wired_limit",
         "systemone.engine.engine_scope (the gmlx-patched policy) / "
         "wired_limit_cpu_fix (wrapped: no limit on the CPU device)"),
    Seam("gmlx._vendor.mlx_vlm.generate.common", "wired_limit",
         "wired_limit_cpu_fix (wrapped: the stock context reads the working "
         "set key from the default device's info)"),
    Seam("gmlx._vendor.mlx_vlm.generate.ar", "wired_limit",
         "wired_limit_cpu_fix (the binding BatchGenerator enters, wrapped "
         "the same)"),
    Seam("gmlx._vendor.mlx_vlm.server.generation", "ResponseGenerator._generate_diffusion",
         "engine_jobs.install_engine_jobs (wrapped: a queued job runs in "
         "place of a generation)"),
    Seam("gmlx._vendor.mlx_vlm.server.generation", "ResponseGenerator._run_diffusion",
         "engine_jobs (one request at a time, context put before the call)"),
    Seam("gmlx._vendor.mlx_vlm.server.generation", "ResponseGenerator._log_prefill_started",
         "engine_jobs (Prefill started line reads prompt_tokens)"),
    Seam("gmlx._vendor.mlx_vlm.server.generation", "ResponseGenerator._drain_cancellations",
         "engine_jobs (job should_stop)"),
    Seam("gmlx._vendor.mlx_vlm.server.generation", "ResponseGenerator._collect_pending_requests",
         "engine_jobs.install_engine_jobs (wrapped: step jobs run between "
         "batch steps; active=True never blocks, capacity 0 reads nothing)"),
    Seam("gmlx._vendor.mlx_vlm.server.generation", "ResponseGenerator._run_impl",
         "engine_jobs (the batch loop calls _collect_pending_requests between "
         "decode steps and drains cancellations after it)"),
    Seam("gmlx._vendor.mlx_vlm.server.generation", "QueuedGenerationRequest",
         "engine_jobs.run_on_engine (queued job shape)"),
    Seam("gmlx._vendor.mlx_vlm.server.generation", "GenerationContext",
         "engine_jobs.run_on_engine (first queue item)"),
    Seam("gmlx._vendor.mlx_vlm.server.generation", "get_token_queue_timeout",
         "patches.systemone (decision deadline)"),
    Seam("gmlx._vendor.mlx_vlm.server.generation", "_build_metrics_envelope",
         "patches.systemone (request log line)"),
    # --- mlx-lm seams ---
    Seam("gmlx._vendor.mlx_lm.models.switch_layers", "SwitchLinear.__call__",
         "modules leaf swap + fused subclass signature contract",
         critical=True),
    Seam("gmlx._vendor.mlx_lm.models.switch_layers", "SwitchGLU.__call__",
         "modules.install_fused_moe_glu / loader.install_expert_streaming",
         critical=True),
    Seam("gmlx._vendor.mlx_lm.server", "ModelProvider._load",
         "server_bridge_lm (GGUF ModelProvider)", critical=True),
    Seam("gmlx._vendor.mlx_lm.generate", "maybe_quantize_kv_cache",
         "apc_pooling", critical=True),
    Seam("gmlx._vendor.mlx_lm.models.hunyuan", "MoeBlock",
         "loader._patch_hunyuan_norm_topk"),
    Seam("gmlx._vendor.mlx_lm.models.deepseek_v32", "MoEGate",
         "dsv32_patches (fp32 router patch)"),
    # --- kimi_k3 vendored-module imports (kimi_k3_model) ---
    # The vendored Kimi-K3 class reuses mlx-lm's per-key-channel-decay
    # gated-delta scan (KDA's exact recurrence), the kimi_linear ShortConv1d
    # semantics (reimplemented in-module; the class itself is not imported),
    # and the MLA MultiLinear (absorbed embed_q/unembed_out).
    Seam("gmlx._vendor.mlx_lm.models.gated_delta", "gated_delta_kernel",
         "kimi_k3_model.KimiK3DeltaAttention (metal scan dispatch)",
         critical=True),
    Seam("gmlx._vendor.mlx_lm.models.gated_delta", "gated_delta_ops",
         "kimi_k3_model.KimiK3DeltaAttention (CPU/fallback scan)",
         critical=True),
    Seam("gmlx._vendor.mlx_lm.models.mla", "MultiLinear",
         "kimi_k3_model.KimiK3MLAAttention (embed_q/unembed_out -> "
         "KQuantMultiLinear)", critical=True),
    Seam("gmlx._vendor.mlx_lm.models.kimi_linear", "ShortConv1d",
         "kimi_k3_model.ShortConv1d (semantics mirror; re-mirror review "
         "on upstream change)"),
)


def _resolve_path(module: str, attr: str):
    """('ok', obj) | ('missing-module', None) | ('missing-attr', None).

    Modules resolve through import; ``sys.modules`` wins for already-loaded
    ones, which matters for e.g. mlx_vlm.server.runtime (the package
    __init__ shadows the submodule attribute with the runtime instance).
    """
    try:
        mod = sys.modules.get(module) or importlib.import_module(module)
    except ImportError:
        return "missing-module", None
    obj = mod
    for part in attr.split(".") if attr else ():
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return "missing-attr", None
    return "ok", obj


def _resolve(seam: Seam):
    return _resolve_path(seam.module, seam.attr)


def _resolve_impl(seam: Seam):
    module, _, attr = seam.impl.partition(":")
    return _resolve_path(module, attr)


def _identity_problem(seam: Seam, obj) -> str | None:
    """A problem line when an identity row's stock name no longer leads to
    its owned object. gmlx wrappers may sit on the stock name and on the
    owned name alike (a per-arch SDPA claim, say), so both sides are
    followed down the ``_gmlx_orig`` and ``__wrapped__`` links those
    wrappers leave, to the object that wraps nothing."""
    from gmlx.upstream.patching import root

    module, _, attr = seam.owned.partition(":")
    status, owned = _resolve_path(module, attr)
    if status != "ok":
        return (f"{_key(seam)}: owned object {seam.owned} {status} "
                f"(used by {seam.used_by})")
    if root(obj) is root(owned):
        return None
    return (f"{_key(seam)}: does not resolve to the owned {seam.owned}; "
            f"the cache binding (gmlx.cache.bind) did not reach it (used "
            f"by {seam.used_by})")


_UPSTREAM_ROOTS = ("gmlx._vendor.mlx_vlm", "gmlx._vendor.mlx_lm")


def _source(obj) -> str | None:
    try:
        return inspect.getsource(obj)
    except (TypeError, OSError):
        return None


def _type_name(obj) -> str:
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _pin_text(obj) -> str:
    """The text a seam pins: the source of a function, class or module; the
    accessor sources of a property; the repr of a plain constant; the member
    names and types of a namespace (gmlx rebinds members in place); the
    class source of an instance of an upstream class; else the qualified
    name of the object or its type."""
    if isinstance(obj, property):
        return "\n".join(
            _source(f) or "" for f in (obj.fget, obj.fset, obj.fdel)
            if f is not None)
    if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
        return repr(obj)
    if isinstance(obj, SimpleNamespace):
        return "\n".join(f"{name}: {_type_name(value)}"
                         for name, value in sorted(vars(obj).items()))
    src = _source(obj)
    if src is not None:
        return src
    name = getattr(obj, "__qualname__", None)
    if isinstance(name, str):
        return f"{getattr(obj, '__module__', '')}.{name}"
    if type(obj).__module__.startswith(tuple(r + "." for r in _UPSTREAM_ROOTS)):
        src = _source(type(obj))
        if src is not None:
            return src
    return _type_name(obj)


def _fingerprint(obj) -> str:
    """sha256 of what the seam pins for ``obj`` (see ``_pin_text``)."""
    return hashlib.sha256(_pin_text(obj).encode()).hexdigest()


def _is_lambda(obj) -> bool:
    return getattr(obj, "__name__", None) == "<lambda>"


def _rebound(seam: Seam, obj) -> bool:
    """True when the seam currently holds a gmlx replacement - one of
    our installers already ran in this interpreter. Source comparison is
    then meaningless (the patch being present proves the install succeeded);
    existence checks still apply. Upstream's own cross-package from-imports
    (e.g. qwen3_5.gated_delta re-exporting mlx-lm functions) keep their
    original __module__ and are fingerprinted normally."""
    mod = getattr(obj, "__module__", None) or getattr(
        type(obj), "__module__", "")
    # The copied packages live under gmlx._vendor and are not replacements.
    return (bool(mod) and mod.split(".")[0] == "gmlx"
            and not mod.startswith("gmlx._vendor."))


def _key(seam: Seam) -> str:
    return f"{seam.module}:{seam.attr}"


def _row_fingerprint(seam: Seam, obj, impl) -> str:
    """The seam's fingerprint. A forwarding seam pins the binding and the
    implementation it calls together, so a change to either shows. A row
    with its own pin function pins what that returns."""
    text = seam.pin(obj) if seam.pin is not None else _pin_text(obj)
    if seam.impl:
        text += "\0" + _pin_text(impl)
    return hashlib.sha256(text.encode()).hexdigest()


def _unforwarded_lambda(seam: Seam, obj) -> str | None:
    """A problem line when the seam resolves to a lambda and names no
    implementation: the lambda's own source does not change when the
    function it forwards to moves or changes."""
    if seam.impl or not _is_lambda(obj):
        return None
    return (f"{_key(seam)}: resolves to a forwarding lambda - set impl to "
            f"the module:attr it calls (used by {seam.used_by})")


def collect_fingerprints() -> dict[str, str]:
    out: dict[str, str] = {}
    for seam in SEAMS:
        if seam.owned:
            continue  # an identity row pins no text
        status, obj = _resolve(seam)
        if status == "ok":
            impl = None
            if seam.impl:
                istatus, impl = _resolve_impl(seam)
                if istatus != "ok":
                    raise RuntimeError(
                        f"cannot fingerprint {_key(seam)}: implementation "
                        f"{seam.impl} {istatus} (used by {seam.used_by})")
            if _rebound(seam, obj) or (seam.impl and _rebound(seam, impl)):
                raise RuntimeError(
                    f"{_key(seam)} already holds a gmlx patch - "
                    f"regenerate fingerprints in a fresh interpreter")
            lam = _unforwarded_lambda(seam, obj)
            if lam:
                raise RuntimeError(lam)
            out[_key(seam)] = _row_fingerprint(seam, obj, impl)
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
        if seam.owned:
            problem = _identity_problem(seam, obj)
            if problem:
                problems.append(problem)
            continue
        if _rebound(seam, obj):
            continue  # our installer already replaced it in this process
        lam = _unforwarded_lambda(seam, obj)
        if lam:
            problems.append(lam)
            continue
        impl = None
        if seam.impl:
            istatus, impl = _resolve_impl(seam)
            if istatus != "ok":
                problems.append(
                    f"{_key(seam)}: implementation {seam.impl} missing - "
                    f"upstream moved or removed it (used by {seam.used_by})")
                continue
            if _rebound(seam, impl):
                continue
        want = fps.get(_key(seam), "<unpinned>")
        got = _row_fingerprint(seam, obj, impl)
        if want == "<unpinned>":
            problems.append(
                f"{_key(seam)}: no pinned fingerprint - run "
                f"`python -m gmlx.upstream.seams --regen`")
        elif got != want:
            problems.append(
                f"{_key(seam)}: source changed under the pin - re-audit "
                f"{seam.used_by}, then regenerate "
                f"(pinned for {pinned.get('generated_with')})")
    rows = {_key(seam) for seam in SEAMS}
    for key in sorted(set(fps) - rows):
        problems.append(
            f"{key}: pinned with no row in SEAMS; restore the row, or run "
            f"`python -m gmlx.upstream.seams --regen` if it went on purpose")
    return problems


def vendored_upstream_collisions() -> list[str]:
    """Owned model types the vendored packages also ship, not yet recorded
    in ``KNOWN_UPSTREAM_ARRIVALS``.

    The vendored loaders resolve an owned type to its gmlx module before
    their own packages, so an arrival changes nothing at run time. Flagging
    it gets the upstream version reviewed as a candidate to A/B against the
    gmlx one.
    """
    from gmlx.models.owned import OWNED_LM_MODELS
    import gmlx._vendor.mlx_lm.models as lm_models
    root = os.path.dirname(lm_models.__file__)
    hits = []
    for mod_name, gmlx_mod in OWNED_LM_MODELS.items():
        if gmlx_mod in KNOWN_UPSTREAM_ARRIVALS:
            continue
        leaf = mod_name.rsplit(".", 1)[-1]
        if os.path.exists(os.path.join(root, f"{leaf}.py")):
            hits.append(
                f"{mod_name}: upstream mlx-lm now ships this module and "
                f"{gmlx_mod} stays registered ahead of it. Review upstream's "
                f"version and record the release in KNOWN_UPSTREAM_ARRIVALS")
    hits += _vendored_vlm_collisions()
    return hits


# gmlx module -> the mlx-vlm name it stands for.
# Each is a package directory upstream, so a native arrival shows up as either
# a <leaf>.py module or a <leaf>/ package.
VENDORED_MLX_VLM_MODULES = {
    "gmlx.models.deepseek_v41.tools": "gmlx._vendor.mlx_vlm.tools.parsers.deepseek_v41",
    "gmlx.models.function_gemma": "gmlx._vendor.mlx_vlm.tools.parsers.function_gemma",
    "gmlx.models.hy_v3.tools": "gmlx._vendor.mlx_vlm.tools.parsers.hy_v3",
    "gmlx.models.hy_v4.tools": "gmlx._vendor.mlx_vlm.tools.parsers.hy_v4",
    "gmlx.models.muse_glimmer.tools": "gmlx._vendor.mlx_vlm.tools.parsers.muse_glimmer",
    "gmlx.models.deepseek_v4.vlm_model": "gmlx._vendor.mlx_vlm.models.deepseek_v4_vl",
    "gmlx.models.deepseek_v41.vlm_model": "gmlx._vendor.mlx_vlm.models.deepseek_v41_vl",
    "gmlx.models.glm5_next.vlm_model": "gmlx._vendor.mlx_vlm.models.glm5_next",
    "gmlx.models.muse_glimmer.vlm_model": "gmlx._vendor.mlx_vlm.models.muse_glimmer",
    "gmlx.models.qwen4_exp.vlm_model": "gmlx._vendor.mlx_vlm.models.qwen4_exp",
}

# Vendored modules whose name upstream also ships, with the upstream release
# in which the arrival was recorded. The gmlx module still resolves first;
# the upstream one is a candidate to A/B against it. The collision checks
# skip these entries and flag any other arrival.
KNOWN_UPSTREAM_ARRIVALS = {
    "gmlx.models.muse_glimmer.vlm_model": "mlx-vlm 0.6.15",
    "gmlx.models.glm5_next.vlm_model": "mlx-vlm 0.7.3",
    "gmlx.models.qwen4_exp.vlm_model": "mlx-vlm 0.7.3",
}


def _vendored_vlm_collisions() -> list[str]:
    """Same check on the mlx-vlm side."""
    hits = []
    for mod_name, target in VENDORED_MLX_VLM_MODULES.items():
        if mod_name in KNOWN_UPSTREAM_ARRIVALS:
            continue
        pkg, _, leaf = target.rpartition(".")
        try:
            parent = importlib.import_module(pkg)
        except ImportError:
            continue
        if parent.__file__ is None:
            continue
        root = os.path.dirname(parent.__file__)
        if (os.path.exists(os.path.join(root, f"{leaf}.py"))
                or os.path.isdir(os.path.join(root, leaf))):
            hits.append(
                f"{mod_name}: upstream mlx-vlm now ships {target} and the "
                f"gmlx module stays registered ahead of it. Review "
                f"upstream's version and record the release in "
                f"KNOWN_UPSTREAM_ARRIVALS")
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
                f"{qual}; untested - see docs/internals/upstream-upgrades.md")
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
