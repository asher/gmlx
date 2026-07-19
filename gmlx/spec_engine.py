"""Owned speculative-decoding engine seams for the serve path.

Routes all MTP batch sizes through the owned engine: B=1 through the scalar
single-stream round (``owned_server_rounds``), B>1 through the vectorized
batched round (``owned_server_rounds_batch``). Non-MTP draft kinds delegate
to stock ``run_speculative_server_rounds`` unchanged.

Installation is a late-bound monkeypatch (same no-fork pattern as
`server_patches` / `server_bridge_vlm`): `mlx_vlm.generate.ar` binds
`run_speculative_server_rounds` by name at import time and calls it as a module
global, so reassigning `ar.run_speculative_server_rounds` at server-boot time --
after `ar` is imported -- redirects the live serve path.
"""

from __future__ import annotations

import logging
import os
import sys

import mlx.core as mx

from . import prefill_decay
from .envflags import env_int

_log = logging.getLogger(__name__)

_OWNED_MTP_ROUND_FLAG = "_kq_gguf_owned_mtp_round"
_FULL_PREFILL_FLAG = "_kq_gguf_full_prompt_mtp_prefill"

_SPEC_APC_DISABLED = os.environ.get("GMLX_SPEC_APC", "1") == "0"
# Retirement store (prompt + generated -> shared APC at request finish) is a
# beyond-stock multi-turn win; killable on its own or via the global switch.
_SPEC_APC_RETIRE_DISABLED = (
    _SPEC_APC_DISABLED or os.environ.get("GMLX_SPEC_APC_RETIRE", "1") == "0"
)
_SPEC_APC_SIDECAR_DISABLED = (
    _SPEC_APC_DISABLED
    or os.environ.get("GMLX_SPEC_APC_SIDECAR", "1") == "0"
)
_SPEC_APC_CKPT_DISABLED = (
    _SPEC_APC_DISABLED
    or os.environ.get("GMLX_SPEC_APC_CKPT", "1") == "0"
)
_MTP_DEBUG = os.environ.get("GMLX_MTP_DEBUG", "0") not in ("", "0")


def _debug_note(msg: str) -> None:
    """Engine-internals one-shot notices; opt in with GMLX_MTP_DEBUG=1
    (they would otherwise open every server log)."""
    if _MTP_DEBUG:
        print(msg, file=sys.stderr, flush=True)



def _get_spec_prefix_cache(model):
    """Lazy-create a SpecPrefixCache on the model, or return None if disabled."""
    if _SPEC_APC_DISABLED:
        return None
    cache = getattr(model, "_spec_prefix_cache", None)
    if cache is None:
        from .prefix_cache import SpecPrefixCache
        max_entries = env_int("GMLX_SPEC_APC_ENTRIES", 4)
        budget_mb = env_int("GMLX_SPEC_APC_BUDGET_MB", 8192)
        cache = SpecPrefixCache(max_entries=max_entries,
                                max_bytes=budget_mb << 20)
        model._spec_prefix_cache = cache
    return cache


# L1: the shared APCManager (same block pool / exact LRU / disk namespace the
# stock non-speculative path uses)

class _L1View:
    """Minimal duck-typed receiver for BatchGenerator's APC lookup helpers.

    Upstream's lookup ladder (``_apc_pick_for``: exact -> blocks -> disk,
    longest match wins, media-token guards, release-on-reject) and its hash
    salting (``_apc_extra_hash``) are reused verbatim by binding the unbound
    methods onto this attribute surface, so the owned MTP path can never
    drift from the stock path's matching semantics.
    """

    def __init__(self, model, apc_manager, apc_mode):
        self.model = model
        self.apc_manager = apc_manager
        self.apc_mode = apc_mode


_L1_VIEW_METHODS = (
    "_apc_extra_hash",
    "_apc_media_token_ids",
    "_apc_safe_prefix_lookup_min",
    "_apc_suffix_is_text_only",
    "_apc_prefix_has_media_tokens",
    "_apc_exact_checkpoint_len",
    "_apc_pick_for",
)
_L1_BOUND = [False]
_L1_MODE_UNSET = object()


def _bind_l1_view() -> None:
    """Graft BatchGenerator's APC helpers onto _L1View (idempotent)."""
    if _L1_BOUND[0]:
        return
    from mlx_vlm.generate.ar import BatchGenerator
    try:
        for name in _L1_VIEW_METHODS:
            setattr(_L1View, name, getattr(BatchGenerator, name))
    except AttributeError as e:
        _log.warning("APC L1 disabled: upstream helper missing: %s", e)
        return
    _L1_BOUND[0] = True


_APC_STASH_FLAG = "_kq_gguf_apc_manager_stash"


def _install_apc_manager_stash() -> None:
    """Stash the serve-time APCManager on the model object for the owned path.

    Upstream ``BatchGenerator.__init__`` nulls its ``apc_manager`` whenever a
    draft model is configured (the stock prefill APC machinery assumes the
    non-speculative generate flow). The owned MTP engine integrates on its
    own terms, so capture the manager before that gate: in the construction
    call itself, where the true manager and the model are both in scope on
    the generation worker thread (residency's build-scratch ContextVar does
    not cross into that thread, so ``runtime.apc_manager`` cannot be read
    reliably from here). Assigns on every speculative construction --
    including None -- so a BatchGenerator built without APC clears a stale
    stash instead of inheriting one. Idempotent.
    """
    from mlx_vlm.generate.ar import BatchGenerator
    if getattr(BatchGenerator.__init__, _APC_STASH_FLAG, False):
        return
    _orig_init = BatchGenerator.__init__

    def _init_with_stash(self, model, processor, **kwargs):
        if kwargs.get("draft_model") is not None:
            # Kill switch (re-read per call): with spec APC off, stock ar.py
            # must not see the manager either -- since mlx-vlm 0.6.4 its own
            # post-prefill exact store handles B=1 MTP caches (older versions
            # silently declined them), so a stashed-but-disabled manager
            # would still collect stores.
            if _SPEC_APC_DISABLED:
                kwargs["apc_manager"] = None
            try:
                model._kq_apc_manager = kwargs.get("apc_manager")
            except Exception:
                pass
        _orig_init(self, model, processor, **kwargs)

    _init_with_stash.__dict__[_APC_STASH_FLAG] = True
    BatchGenerator.__init__ = _init_with_stash


def _resolve_l1(model):
    """Return (manager, apc_mode) for the shared APC tier, or (None, None)."""
    if _SPEC_APC_DISABLED or not _L1_BOUND[0]:
        return None, None
    manager = getattr(model, "_kq_apc_manager", None)
    if manager is None:
        return None, None
    mode = getattr(model, "_kq_apc_mode", _L1_MODE_UNSET)
    if mode is _L1_MODE_UNSET:
        from mlx_vlm import apc as _apc
        # Probe the bare language model: model_apc_mode falls back to
        # "block" when make_cache is missing, which would misclassify a
        # hybrid reached through a wrapper without make_cache.
        lm = getattr(model, "language_model", None) or model
        try:
            mode = _apc.model_apc_mode(lm)
        except Exception:
            _log.warning("APC L1: model_apc_mode probe failed", exc_info=True)
            mode = None
        try:
            model._kq_apc_mode = mode
        except Exception:
            pass
    if mode is None:
        return None, None
    return manager, mode


def _ckpt_active(model, mode) -> bool:
    """True when the checkpoint tier (attn-KV blocks + recurrent-state
    sidecar) replaces the exact tier for this model: a gated-delta hybrid
    cache shape, exact mode, kill switch open. Shape probed once per model;
    the module flag is re-read every call so benches can toggle in-process.
    """
    if _SPEC_APC_CKPT_DISABLED or mode != "exact":
        return False
    flag = getattr(model, "_kq_apc_ckpt", None)
    if flag is None:
        from .cache_snapshot import ckpt_supported
        lm = getattr(model, "language_model", None) or model
        try:
            flag = bool(ckpt_supported(lm.make_cache()))
        except Exception:
            flag = False
        try:
            model._kq_apc_ckpt = flag
        except Exception:
            pass
    return bool(flag)


def _l1_lookup_and_arm_store(batch, manager, mode, l0_prefix) -> int:
    """Consult the shared APCManager below L0 and arm the stock post-prefill
    store (mid-prefill exact checkpoints + post-prefill exact store / block
    harvest, all owned by stock ``PromptProcessingBatch.generate``) by
    populating ``_apc_manager`` / ``_apc_mode`` / ``_apc_meta``.

    Returns the restored L1 prefix length (0 on miss, or when L0 already
    restored -- L0 carries full-prompt hidden and is always preferred).

    ``meta["prefix_len"]`` stays 0 by design: the owned prefill keeps
    ``_processed_prompt_columns`` in absolute token space (it trims
    ``_input_ids`` in place, unlike stock warm batches which are constructed
    with suffix-only rows), and ``_row_real_tokens_processed`` -- which gates
    the mid-prefill checkpoint store -- is only correct in that space with a
    zero meta prefix. The one cost is that a block-tier harvest re-walks
    restored prefix blocks, but ``store_kv_blocks`` dedups by hash chain
    (acquire+release of existing blocks, no data copies).
    """
    view = _L1View(batch.model, manager, mode)
    ids_list = [int(t) for t in batch._mtp_full_input_ids[0].tolist()]
    prompt_kwargs = batch._prompt_kwargs or {}
    extra_hash = view._apc_extra_hash(prompt_kwargs)
    ckpt = _ckpt_active(batch.model, mode)
    held_blocks = []
    l1_prefix = 0
    if l0_prefix == 0 and len(ids_list) >= 2:
        warm = None
        blocks = []
        prefix_len = 0
        tier = "exact"
        pick = view._apc_pick_for((0, ids_list, 0, prompt_kwargs, None, None))
        if pick is not None:
            warm = pick.get("warm_cache")
            blocks = list(pick.get("matched_blocks") or ())
            prefix_len = int(pick.get("prefix_len") or 0)
            extra_hash = int(pick.get("extra_hash", extra_hash))
            if warm is None and blocks:
                from mlx_vlm import apc as _apc
                warm = _apc.make_warm_kv_cache(
                    blocks, min_capacity_tokens=len(ids_list) + 1)
                tier = "block"
        if ckpt:
            # Checkpoint tier: the longest salted sidecar + block chain
            # wins only when strictly longer than the exact-tier pick.
            # Media guards mirror the stock exact probe.
            from .cache_snapshot import ckpt_lookup
            min_p = max(prefix_len,
                        view._apc_safe_prefix_lookup_min(ids_list))
            cw, cp = ckpt_lookup(
                manager, ids_list, extra_hash=extra_hash,
                min_prefix_tokens=min_p)
            if (cw is not None and cp > prefix_len
                    and view._apc_suffix_is_text_only(ids_list, cp)):
                if blocks:
                    manager.release(blocks)
                    blocks = []
                warm, prefix_len, tier = cw, cp, "ckpt"
        if warm and 0 < prefix_len < len(ids_list):
            batch.prompt_cache = warm
            # Matched blocks stay acquired until the stock post-prefill
            # harvest releases them (the warm-cache concatenation is
            # lazy; the pool must not recycle these blocks before it
            # materializes).
            held_blocks = blocks
            l1_prefix = prefix_len
            _log.info(
                "APC L1 hit: prefix=%d suffix=%d tier=%s",
                prefix_len, len(ids_list) - prefix_len, tier,
            )
            # Drafter-KV sidecar: a plain L1 hit restores target KV but
            # not hidden, so the drafter would re-seed from suffix-only
            # hidden at the wrong positions (acceptance erodes at depth).
            # A sidecar covering exactly the restored prefix hands the
            # owned round a warm drafter start. Stash rides the first
            # cache entry, same discipline as the retirement context.
            if not _SPEC_APC_SIDECAR_DISABLED:
                from .cache_snapshot import drafter_sidecar_lookup
                side = drafter_sidecar_lookup(
                    manager, ids_list, prefix_len, extra_hash)
                if side:
                    batch.prompt_cache[0]._kq_apc_drafter_warm = side
                    _log.info(
                        "APC sidecar hit: prefix=%d", prefix_len)
        elif blocks:
            manager.release(blocks)
    batch._mtp_l1_prefix_len = l1_prefix
    batch._apc_manager = manager
    batch._apc_mode = mode
    batch._apc_meta = [{
        "full_input_ids": ids_list,
        "prefix_len": 0,
        "extra_hash": extra_hash,
        "apc_blocks": held_blocks,
        "checkpoint_len": view._apc_exact_checkpoint_len(ids_list),
    }]
    if ckpt:
        # The checkpoint tier replaces the stock exact-tier stores: the
        # post-prefill full-cache clone is suppressed here, and the
        # mid-prefill checkpoint store is superseded in _mtp_prompt_step
        # (same aligned column, marks checkpoint_done so the stock store
        # is a no-op). Column alignment itself still runs on the stock
        # machinery, which requires _apc_mode == "exact".
        batch._apc_harvest_enabled = False
        batch._kq_ckpt_armed = True
    return l1_prefix


def _ckpt_mid_prefill_store(batch) -> None:
    """Checkpoint-tier replacement for the stock mid-prefill exact store.

    Fires at the same aligned column (the stock ``_next_apc_checkpoint_
    column`` machinery forces the chunk boundary there), then marks
    ``checkpoint_done`` so the stock store -- a full multi-GB hybrid clone
    -- becomes a no-op. Marked done even when the store fails: falling back
    to the stock clone would defeat the tier's purpose.
    """
    if not getattr(batch, "_kq_ckpt_armed", False):
        return
    manager = getattr(batch, "_apc_manager", None)
    meta_list = getattr(batch, "_apc_meta", None) or []
    if manager is None or not meta_list or meta_list[0] is None:
        return
    meta = meta_list[0]
    if meta.get("checkpoint_done"):
        return
    checkpoint_len = int(meta.get("checkpoint_len") or 0)
    if checkpoint_len <= 0:
        return
    if batch._row_real_tokens_processed(0) != checkpoint_len:
        return
    from .cache_snapshot import ckpt_store
    ckpt_store(
        manager, meta["full_input_ids"][:checkpoint_len],
        batch.prompt_cache, extra_hash=int(meta.get("extra_hash", 0)))
    meta["checkpoint_done"] = True


def _mtp_prefill_init(batch) -> None:
    """One-time APC lookup + prefix trim for an MTP prompt batch.

    Runs on the first ``prompt_step`` call, or directly from ``generate()``
    when the prompt is short enough that chunked prefill never fires.
    Lookup ladder: L0 (SpecPrefixCache: whole-prompt KV + full-prompt
    hidden, the only tier the drafter can teacher-force from without a cold
    start) then L1 (shared APCManager: exact / block / disk KV, no hidden).
    Also arms the stock post-prefill store whenever a manager is reachable,
    regardless of which tier (if any) hit.
    """
    if hasattr(batch, "_mtp_full_input_ids"):
        return
    batch._mtp_full_input_ids = batch._input_ids
    batch._mtp_chunk_hiddens = []
    batch._mtp_l1_prefix_len = 0

    if batch._inputs_embeds is None:
        return

    # Gated to B=1 because PromptProcessingBatch prefills one request at a
    # time today. The restored single-row cache (with its offset) later
    # merges into the live B>1 decode batch via BatchKVCache.extend during
    # continuous-batch injection -- so APC absolutely works in a B>1
    # serving context; the gate is about prefill granularity, not decode
    # batch size. If mlx-vlm ever coalesces prefills into a multi-row
    # PromptProcessingBatch, this guard silently disables APC for those
    # rows. The warning below makes that visible.
    b = int(batch._input_ids.shape[0])
    if b > 1:
        if not _SPEC_APC_DISABLED:
            _log.warning(
                "APC skipped: prefill batch B=%d > 1 "
                "(owned-path APC requires single-request prefill)", b)
        return

    restored = 0
    spec_cache = _get_spec_prefix_cache(batch.model)
    if spec_cache is not None:
        hit = spec_cache.lookup(batch._input_ids)
        if hit is not None:
            restored, entry = hit
            spec_cache.restore(entry, batch.prompt_cache)
            batch._mtp_chunk_hiddens = [entry.hidden]
            _log.info(
                "APC hit: prefix=%d suffix=%d",
                restored, int(batch._input_ids.shape[1]) - restored,
            )

    manager, mode = _resolve_l1(batch.model)
    if manager is not None:
        try:
            l1_prefix = _l1_lookup_and_arm_store(batch, manager, mode, restored)
            restored = max(restored, l1_prefix)
        except Exception:
            _log.warning("APC L1 failed; continuing cold", exc_info=True)

    # Stash the retirement context so the owned B=1 round can store this
    # request's full context (prompt + generated) into the shared APC when it
    # finishes. Keyed on the original full ids (pre-trim) -- the serve-layer
    # prompt_tokens is suffix-only on a warm turn, so it can't be the key.
    # The stash lives on the request's first cache entry, not on the model:
    # the server closes a finished rounds generator lazily (sometimes after
    # the next request's prefill), so a model-level stash races and retires
    # under the wrong key. Must run after the L1 block above -- an exact-tier
    # hit replaces batch.prompt_cache wholesale. B=1 only (this init is gated
    # to B=1); B>1 retirement is handled per-row at the batch decode's
    # finish seam.
    if (manager is not None and not _SPEC_APC_RETIRE_DISABLED
            and batch.prompt_cache):
        meta = (batch._apc_meta or [{}])[0] or {}
        batch.prompt_cache[0]._kq_apc_retire = {
            "full_ids": [int(t) for t in batch._mtp_full_input_ids[0].tolist()],
            "extra_hash": int(meta.get("extra_hash", 0)),
            "mode": ("ckpt" if _ckpt_active(batch.model, mode) else mode),
            "checkpoint_len": int(meta.get("checkpoint_len", 0) or 0),
        }

    if restored > 0:
        batch._input_ids = batch._input_ids[:, restored:]
        batch._inputs_embeds = batch._inputs_embeds[:, restored:]
        batch._processed_prompt_columns = restored
        for k in batch._prompt_length_aware_keys:
            batch._prompt_kwargs[k] = batch._prompt_kwargs[k][:, restored:, ...]
        batch._mtp_apc_prefix_len = restored


def install_full_prompt_mtp_prefill() -> None:
    """Retain full-prompt hidden through the BatchGenerator MTP prefill so the
    native head teacher-forces the whole prompt into its KV (llama parity).

    mlx-vlm's ``PromptProcessingBatch`` chunks prefill: intermediate chunks
    (``prompt_step``) discard the model output (only KV-cache side-effects
    survive), then ``generate()`` runs the final chunk with
    ``return_hidden=True``.  The MTP drafter thus only sees hidden for that
    last chunk -- often 1 token -- and acceptance erodes at depth.

    This patch makes ``prompt_step`` also request ``return_hidden=True`` on
    MTP batches, accumulating per-chunk hidden in ``_mtp_chunk_hiddens``.
    ``generate()`` then concatenates them with the final chunk's hidden so
    ``speculative_hidden_state`` returns full-prompt hidden to the drafter.

    Also installs the owned-path APC surface: the L0 SpecPrefixCache
    (whole-prompt KV + hidden, in-memory) plus the L1 shared APCManager
    (exact / block / disk tiers -- the same manager the stock
    non-speculative path uses, reached via ``model._kq_apc_manager``, which
    ``_install_apc_manager_stash`` captures at BatchGenerator construction).
    Kill switch for both tiers: ``GMLX_SPEC_APC=0``.

    Idempotent.  Only MTP batches (``self.draft_kind == "mtp"``) are affected;
    eagle3 / dflash keep the stock path.
    """
    from mlx_vlm.generate.ar import PromptProcessingBatch

    # L1 plumbing is idempotent on its own flags, so it installs (or
    # repairs) even when the prefill override is already in place.
    _bind_l1_view()
    _install_apc_manager_stash()

    if getattr(PromptProcessingBatch, _FULL_PREFILL_FLAG, False):
        return

    _orig_prompt_step = PromptProcessingBatch.prompt_step
    _orig_generate = PromptProcessingBatch.generate
    _orig_init = PromptProcessingBatch.__init__

    def _resolve_mtp_prefill_step() -> int:
        # Honor the serve path's PREFILL_STEP_SIZE env override
        # (mlx_vlm.server.generation.get_prefill_step_size) so MTP prefill
        # can be chunked smaller to cap peak memory.
        from mlx_vlm.generate.ar import DEFAULT_PREFILL_STEP_SIZE
        return int(os.environ.get(
            "PREFILL_STEP_SIZE", DEFAULT_PREFILL_STEP_SIZE))

    def _mtp_init(self, *args, **kwargs) -> None:
        _orig_init(self, *args, **kwargs)
        # Re-enable chunked prefill.  Stock mlx-vlm nulls prefill_step_size
        # for speculative models because intermediate chunks discard hidden;
        # our prompt_step captures it, so the gate no longer applies.
        # Restoring at construction (not first prompt_step) matters: the
        # scheduler consults needs_processing() first, and with a None step
        # an APC-less deep prompt would one-shot the whole prefill.
        if (getattr(self, "draft_kind", None) == "mtp"
                and self.prefill_step_size is None):
            self.prefill_step_size = _resolve_mtp_prefill_step()

    def _mtp_prompt_step(self) -> int:
        if self.draft_kind != "mtp":
            return _orig_prompt_step(self)

        if not hasattr(self, "_mtp_full_input_ids"):
            if self.prefill_step_size is None:
                self.prefill_step_size = _resolve_mtp_prefill_step()
            # APC lookup (L0 then L1) + prefix trim + store arming.
            _mtp_prefill_init(self)

        if not self.needs_processing():
            return 0

        # Depth-decayed step: shrink only when this chunk's score transient
        # would exceed the cap (see prefill_decay; keeps MoE weight
        # amortization at shallow depth instead of a global small step).
        step = (prefill_decay.decayed_for_batch(self)
                or self._inputs_embeds.shape[1])
        n = min(step, self._inputs_embeds.shape[1] - 1)

        if not hasattr(self, "_mtp_padding_widened"):
            self._mtp_padding_widened = True
            for c in self.prompt_cache:
                lp = getattr(c, "left_padding", None)
                if isinstance(lp, mx.array) and lp.ndim > 0 and lp.size > 1:
                    max_lp = int(lp.max().item())
                    if max_lp >= n:
                        n = min(max_lp + 1, self._inputs_embeds.shape[1] - 1)
                    break

        checkpoint_col = self._next_apc_checkpoint_column()
        if checkpoint_col is not None:
            n = min(n, checkpoint_col - self._processed_prompt_columns)
        if n <= 0:
            return 0
        prompt_kwargs = self._prompt_kwargs_for_step(n)
        out = self.model(
            self._input_ids[:, :n],
            cache=self.prompt_cache,
            inputs_embeds=self._inputs_embeds[:, :n],
            n_to_process=n,
            return_hidden=True,
            **prompt_kwargs,
        )
        chunk_hidden = out.hidden_states[-1]
        # Teacher-forcing drafters (native MTP heads) seed their KV from the
        # whole prompt hidden, so every chunk is retained. Shared-KV drafters
        # (gemma-4 assistant) read only the last position: keeping just the
        # newest chunk caps capture memory at O(chunk) instead of O(prompt)
        # -- GBs at deep context.
        if callable(getattr(self.draft_model, "prefill_from_target_hidden", None)):
            self._mtp_chunk_hiddens.append(chunk_hidden)
            # Window-limited heads can't use context beyond the trailing
            # hidden_capture_limit positions; an uncapped capture pins the
            # whole prompt's hidden (GBs at deep context). The drafter's
            # teacher-force self-aligns to the trailing h_len positions.
            limit = getattr(self.draft_model, "hidden_capture_limit", None)
            if limit:
                total = sum(int(h.shape[1]) for h in self._mtp_chunk_hiddens)
                if total > limit:
                    merged = (self._mtp_chunk_hiddens[0]
                              if len(self._mtp_chunk_hiddens) == 1
                              else mx.concatenate(self._mtp_chunk_hiddens, axis=1))
                    self._mtp_chunk_hiddens = [merged[:, -limit:]]
        else:
            self._mtp_chunk_hiddens = [chunk_hidden]
        mx.eval([c.state for c in self.prompt_cache] + [chunk_hidden])
        self._processed_prompt_columns += n
        _ckpt_mid_prefill_store(self)
        self._store_apc_exact_checkpoints()
        self._inputs_embeds = self._inputs_embeds[:, n:]
        self._input_ids = self._input_ids[:, n:]
        for k in self._prompt_length_aware_keys:
            self._prompt_kwargs[k] = self._prompt_kwargs[k][:, n:, ...]
        mx.clear_cache()
        return n

    def _mtp_generate(self, sampler, stop_criteria,
                      compute_logprobs=True, top_logprobs_k=0):
        if self.draft_kind == "mtp":
            # Short prompts never enter prompt_step (chunked prefill is not
            # needed), so the APC lookup/store arming runs here instead.
            _mtp_prefill_init(self)
        result = _orig_generate(
            self, sampler, stop_criteria,
            compute_logprobs=compute_logprobs,
            top_logprobs_k=top_logprobs_k,
        )
        from mlx_vlm.generate.ar import SpeculativeGenerationBatch
        if (
            self.draft_kind != "mtp"
            or not isinstance(result, SpeculativeGenerationBatch)
        ):
            return result
        chunk_hiddens = getattr(self, "_mtp_chunk_hiddens", None)
        if not chunk_hiddens:
            # No captured chunks: the whole (remaining) prompt went through
            # the final generate forward, so stock prompt_tokens/hidden are
            # already an aligned pair (suffix-only on an L1 hit).
            return result
        parts = chunk_hiddens + [result.hidden]
        full_hidden = mx.concatenate(parts, axis=1)
        result.hidden = full_hidden
        full_ids = getattr(self, "_mtp_full_input_ids", None)
        l1_prefix = int(getattr(self, "_mtp_l1_prefix_len", 0) or 0)
        if full_ids is not None:
            # On an L1 hit the captured hidden covers only the forwarded
            # suffix, so hand the drafter the matching suffix tokens: the
            # teacher-forcing (token, hidden) pair must stay positionally
            # aligned. The missing prefix can only affect draft acceptance,
            # never correctness -- verify catches every draft.
            result.prompt_tokens = (
                full_ids[:, l1_prefix:] if l1_prefix > 0 else full_ids
            )

        # APC L0 store: cache this request's target KV + hidden so a
        # future request sharing this token prefix skips re-prefill.
        # Uses result.prompt_cache (SpecBatch owns the cache now),
        # not self.prompt_cache (empty after _orig_generate).
        #
        # B=1 only -- same prefill-granularity gate as the lookup.
        # The stored single-row snapshot is valid for injection into
        # a B>1 batch: SpecPrefixCache.restore writes into a fresh
        # single-row prompt_cache, and BatchKVCache.extend merges
        # it at the correct per-row offset.
        #
        # Skipped on an L1 hit: hidden covers only the suffix, and L0
        # entries pair full-prompt keys with full-prompt hidden.
        b = int(full_hidden.shape[0]) if full_ids is not None else 0
        spec_cache = (
            _get_spec_prefix_cache(self.model)
            if b == 1 and l1_prefix == 0 else None
        )
        if spec_cache is not None and full_ids is not None:
            spec_cache.store(full_ids, result.prompt_cache, full_hidden)
            _log.info(
                "APC store: tokens=%d layers=%d",
                int(full_ids.shape[1]), len(result.prompt_cache),
            )

        return result

    PromptProcessingBatch.__init__ = _mtp_init
    PromptProcessingBatch.prompt_step = _mtp_prompt_step
    PromptProcessingBatch.generate = _mtp_generate
    setattr(PromptProcessingBatch, _FULL_PREFILL_FLAG, True)
    if _SPEC_APC_DISABLED:
        apc_status = "off"
    elif _L1_BOUND[0]:
        apc_status = "on: L0+L1"
    else:
        apc_status = "on: L0 only"
    _debug_note(f"[mtp] serve prefill: full-prompt hidden capture installed "
                f"(APC {apc_status})")


_CONTINUOUS_BATCH_FLAG = "_kq_gguf_continuous_batch"
_RELEASED_FLAG = "_kq_gguf_spec_released"
_RELEASE_PENDING_FLAG = "_kq_gguf_spec_release_pending"


def install_continuous_batch_admission() -> None:
    """Let new requests prefill and inject during speculative decode.

    Without this, mlx-vlm's ``is_speculative`` gate blocks all prefills while
    speculative decode is in-flight, and ``extend()`` raises on non-empty
    speculative batches. This installs five patches:

    1. Disables the ``is_speculative`` admission gate (lets prefills run
       during decode).
    2. Overrides ``extend()`` to buffer new batches instead of raising.
    3. Overrides ``__len__()`` to auto-promote buffered batches when the
       current batch finishes.
    4. Overrides ``next()`` to process pending injections - updates outer
       tracking state, emits first tokens, queues for the generator.
    5. Releases a finished batch's request state (target KV, captured
       hidden, shared KV, drafter KV) the moment its last row finishes.

    The generator-side injection (extending caches + drafter mid-flight)
    happens in ``_owned_decode_rounds_batch`` via ``model._generator_injections``.
    """
    from mlx_vlm.generate import ar as _ar

    SpecBatch = _ar.SpeculativeGenerationBatch
    if getattr(SpecBatch, _CONTINUOUS_BATCH_FLAG, False):
        return

    # 1. Remove admission gate
    SpecBatch.is_speculative = False

    _orig_len = SpecBatch.__len__

    # 5. Release request state at finish. BatchGenerator parks the finished
    # batch in _generation_batch until the next request's prefill completes
    # (only PromptProcessingBatch.generate's extend replaces it), so every
    # heavy attr -- the full target KV, the captured full-prompt hidden, the
    # prefill shared-KV, the rounds generator (whose delegation frame re-pins
    # all of the above), and the drafter's own head KV -- survives that whole
    # prefill window. At deep context that stacks two requests' footprints
    # for many minutes (d200k gemma-4-31b: ~65 GB across an ~18-minute
    # prefill) and runs the box to the wire ceiling. Drop it all on the
    # finishing step instead.
    def _release_heavy_state(self) -> bool:
        """Drop request state from a finished batch. Returns False when the
        rounds generator is mid-step on another thread (a client abort racing
        the engine); ``__len__`` retries on the engine thread."""
        if getattr(self, _RELEASED_FLAG, False):
            return True
        rounds = getattr(self, "_rounds_iter", None)
        if rounds is not None:
            try:
                # Terminal-token finishes already ran the inner loop's own
                # cleanup; close() is then a no-op resume. Aborted requests
                # close here, firing the mid-round rollback + retirement.
                rounds.close()
            except ValueError:
                setattr(self, _RELEASE_PENDING_FLAG, True)
                return False
            except Exception:
                _log.warning("spec batch release: rounds close failed",
                             exc_info=True)
        self._rounds_iter = None
        self.prompt_cache = []
        self.hidden = None
        self.shared_kv_states = None
        self.prompt_tokens = None
        self.first_tokens = None
        if getattr(self, "draft_kind", None) == "mtp":
            drafter = getattr(self, "draft_model", None)
            model = getattr(self, "model", None)
            if drafter is not None and model is not None:
                try:
                    drafter.reset(model)  # drops the head's request KV
                except Exception:
                    _log.warning("spec batch release: drafter reset failed",
                                 exc_info=True)
        setattr(self, _RELEASED_FLAG, True)
        setattr(self, _RELEASE_PENDING_FLAG, False)
        mx.clear_cache()
        return True

    def _release_if_finished(self) -> None:
        if _orig_len(self) == 0:
            _release_heavy_state(self)

    # 2. Buffer extend() instead of raising
    def _buffered_extend(self, other):
        active = sum(not d for d in self._finished)
        if active == 0:
            pending = getattr(self, "_pending_injections", [])
            self.__dict__.update(other.__dict__)
            self._pending_injections = pending
            setattr(self, _RELEASED_FLAG, False)
            setattr(self, _RELEASE_PENDING_FLAG, False)
            return
        if not hasattr(self, "_pending_injections"):
            self._pending_injections = []
        self._pending_injections.append(other)

    SpecBatch.extend = _buffered_extend

    # 3. Auto-promote buffered batches when current is done
    def _len_with_promotion(self):
        if getattr(self, _RELEASE_PENDING_FLAG, False) and _orig_len(self) == 0:
            _release_heavy_state(self)
        active = _orig_len(self)
        if active == 0:
            pending = getattr(self, "_pending_injections", None)
            if pending:
                other = pending.pop(0)
                remaining = pending[:]
                self.__dict__.update(other.__dict__)
                self._pending_injections = remaining
                setattr(self, _RELEASED_FLAG, False)
                setattr(self, _RELEASE_PENDING_FLAG, False)
                return _orig_len(self)
        return active

    SpecBatch.__len__ = _len_with_promotion

    _orig_filter = SpecBatch.filter

    def _filter_with_release(self, keep):
        _orig_filter(self, keep)
        _release_if_finished(self)

    SpecBatch.filter = _filter_with_release

    # 4. Process pending injections in next() before advancing the generator
    _orig_next = SpecBatch.next

    def _next_with_injection(self):
        pending = getattr(self, "_pending_injections", None)
        # Mid-flight adoption works only when the batch rounds generator is
        # running: it drains model._generator_injections at its round
        # boundaries. The scalar (B=1) generator never does, so merging uids
        # into a scalar batch strands the entry -- the injected request's
        # continuation then re-dispatches from the wrong state (the finished
        # row's cache) and its stream is silently truncated. Leave scalar
        # injections buffered; _len_with_promotion adopts them wholesale
        # (their own cache/hidden/first token) once the current request ends.
        # `_all_uids` is an mlx-vlm generator internal (stable under the
        # ==0.6.3 pin); re-verify this batch-vs-scalar signal on a pin lift.
        if pending and len(self._all_uids) > 1:
            responses = []
            gen_inj = getattr(self.model, "_generator_injections", None)
            if gen_inj is None:
                self.model._generator_injections = []
                gen_inj = self.model._generator_injections

            for other in pending:
                B_new = len(other._all_uids)
                base_row = len(self._all_uids)
                self._all_uids.extend(other._all_uids)
                self._num_tokens.extend([0] * B_new)
                self._finished.extend([False] * B_new)
                self.max_tokens.extend(other.max_tokens)

                mx.eval(other.first_tokens)
                first_list = other.first_tokens.tolist()
                for row in range(B_new):
                    abs_row = base_row + row
                    tok = int(first_list[row])
                    self._num_tokens[abs_row] = 1
                    finish = self._finish_reason(abs_row, tok)
                    if finish is not None:
                        self._finished[abs_row] = True
                    responses.append(self.Response(
                        uid=other._all_uids[row], token=tok,
                        token_logprob=0.0, finish_reason=finish))

                gen_inj.append({
                    "uids": list(other._all_uids),
                    "prompt_cache": other.prompt_cache,
                    "hidden": other.hidden,
                    "shared_kv_states": other.shared_kv_states,
                    "prompt_tokens": other.prompt_tokens,
                    "first_tokens": other.first_tokens,
                    "first_tokens_list": first_list,
                })

            pending.clear()
            self._refresh_uids()

            more = _orig_next(self)
            responses.extend(more)
            _release_if_finished(self)
            return responses

        responses = _orig_next(self)
        _release_if_finished(self)
        return responses

    SpecBatch.next = _next_with_injection
    setattr(SpecBatch, _CONTINUOUS_BATCH_FLAG, True)
    _debug_note("[mtp] continuous batch: admission gate removed, mid-flight "
                "injection enabled")


def install_owned_spec_engine() -> None:
    """Route serve-path MTP through owned engine: B=1 scalar, B>1 batch.

    Idempotent. Non-mtp draft kinds delegate to the stock
    ``run_speculative_server_rounds`` unchanged. B=1 stays on the exact scalar
    path (``owned_server_rounds``); B>1 routes through
    ``owned_server_rounds_batch``.
    """
    from mlx_vlm.generate import ar as _ar

    _orig = _ar.run_speculative_server_rounds
    if getattr(_orig, _OWNED_MTP_ROUND_FLAG, False):
        return

    from gmlx.speculative import (
        owned_server_rounds,
        owned_server_rounds_batch,
    )

    _first_use_b1 = [False]
    _first_use_batch = [False]

    def _owned_server_rounds(
        model,
        draft_model,
        prompt_cache,
        hidden,
        *,
        draft_kind,
        first_bonus,
        max_tokens,
        sampler,
        draft_block_size=None,
        token_dtype=mx.int32,
        stop_check=None,
        greedy_sampling=False,
        shared_kv_states=None,
        eos_token_ids=None,
        prompt_tokens=None,
        row_ids=None,
        **_extra,
    ):
        batch_size = int(first_bonus.shape[0]) if first_bonus.ndim > 0 else 1
        if draft_kind == "mtp":
            if batch_size == 1:
                if not _first_use_b1[0]:
                    _debug_note("[mtp] owned round: B=1 scalar path")
                    _first_use_b1[0] = True
                rounds = owned_server_rounds(
                    model,
                    draft_model,
                    prompt_cache,
                    hidden,
                    first_bonus=first_bonus,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    shared_kv_states=shared_kv_states,
                    prompt_tokens=prompt_tokens,
                    draft_block_size=draft_block_size,
                    greedy_sampling=greedy_sampling,
                    stop_check=stop_check,
                    eos_token_ids=eos_token_ids,
                )
                # This delegation frame outlives the request (the server
                # abandons finished generators suspended at their last
                # yield), and its args would re-pin the request KV + hidden
                # the inner loop nulls on the terminal token. Keep only the
                # inner generator.
                del prompt_cache, hidden, shared_kv_states, prompt_tokens
                del first_bonus, _extra
                yield from rounds
                return

            if not _first_use_batch[0]:
                _debug_note(f"[mtp] owned round: B={batch_size} batch path")
                _first_use_batch[0] = True
            rounds = owned_server_rounds_batch(
                model,
                draft_model,
                prompt_cache,
                hidden,
                first_bonus=first_bonus,
                max_tokens=max_tokens,
                sampler=sampler,
                shared_kv_states=shared_kv_states,
                prompt_tokens=prompt_tokens,
                draft_block_size=draft_block_size,
                greedy_sampling=greedy_sampling,
                stop_check=stop_check,
                eos_token_ids=eos_token_ids,
                row_ids=row_ids,
            )
            del prompt_cache, hidden, shared_kv_states, prompt_tokens
            del first_bonus, _extra
            yield from rounds
            return

        # Non-mtp draft kind: stock path.
        yield from _orig(
            model,
            draft_model,
            prompt_cache,
            hidden,
            draft_kind=draft_kind,
            first_bonus=first_bonus,
            max_tokens=max_tokens,
            sampler=sampler,
            draft_block_size=draft_block_size,
            token_dtype=token_dtype,
            stop_check=stop_check,
            greedy_sampling=greedy_sampling,
            shared_kv_states=shared_kv_states,
            eos_token_ids=eos_token_ids,
            prompt_tokens=prompt_tokens,
            row_ids=row_ids,
            **_extra,
        )

    _owned_server_rounds.__dict__[_OWNED_MTP_ROUND_FLAG] = True
    _ar.run_speculative_server_rounds = _owned_server_rounds
    from mlx_vlm.server import generation as _gen
    _gen.run_speculative_server_rounds = _owned_server_rounds
    _debug_note("[mtp] serve round: owned engine installed (B=1 + B>1)")


_SPEC_KV_QUANT_FLAG = "_kq_gguf_spec_kv_quant"
_SPEC_KV_QUANT_WIDTHS = (2, 3, 4, 6, 8)  # mx.quantize affine widths


def _spec_kv_quant_params():
    """(bits, group_size) when serve's KV_BITS asks for an affine width the
    single-stream cache can honor, else None. Fractional widths and
    non-uniform schemes (turboquant) have no trimmable B=1 cache."""
    if os.environ.get("GMLX_SPEC_KV_QUANT", "1") == "0":
        return None
    raw = os.environ.get("KV_BITS", "")
    if not raw:
        return None
    try:
        bits = float(raw)
    except ValueError:
        return None
    if bits <= 0:
        return None
    scheme = os.environ.get("KV_QUANT_SCHEME", "uniform")
    if (scheme != "uniform" or bits != int(bits)
            or int(bits) not in _SPEC_KV_QUANT_WIDTHS):
        _log.warning(
            "KV_BITS=%s scheme=%s: no trimmable single-stream cache; "
            "B=1 MTP target KV stays fp16", raw, scheme)
        return None
    return int(bits), int(os.environ.get("KV_GROUP_SIZE", "64"))


def install_spec_kv_quant() -> None:
    """Honor KV_BITS on the B=1 MTP serve path.

    Stock ``make_speculative_prompt_cache`` returns plain fp16 caches for
    ``draft_kind == "mtp", batch_size == 1``, discarding the engine's
    kv_bits: ``BatchQuantizedKVCache`` cannot trim, and MTP rollback must
    trim the target. The single-stream ``QuantizedKVCache`` can trim --
    packing is per-token along head_dim, so trim is an offset move -- and
    the model rollback already goes through ``is_trimmable()``/``trim()``.
    Each plain KVCache converts at construction (empty, so conversion is
    free); SSM / linear-attention / pooled caches pass through untouched.
    Sliding-window stacks drop the flag (parity with the plain path, which
    cannot quantize rotating caches). B>1 MTP keeps stock behavior with a
    one-shot warning (ragged rollback on packed rows is unsupported).
    No-op unless KV_BITS is set at server boot.
    Kill switch: GMLX_SPEC_KV_QUANT=0."""
    from mlx_vlm.generate import ar as _ar
    from mlx_vlm.server import generation as _gen
    from mlx_vlm.speculative import utils as _su

    if getattr(_su.make_speculative_prompt_cache, _SPEC_KV_QUANT_FLAG, False):
        return
    params = _spec_kv_quant_params()
    if params is None:
        return
    bits, group = params

    from .cache_compat import cache_types

    plain_kv = cache_types("KVCache")
    rotating = (cache_types("RotatingKVCache")
                + cache_types("BatchRotatingKVCache"))
    _orig = _su.make_speculative_prompt_cache
    _noted = [False]
    _warned_batch = [False]
    _warned_rotating = [False]

    def _quantizing_spec_cache(lm, *, draft_kind, batch_size, left_padding,
                               make_cache):
        caches = _orig(
            lm,
            draft_kind=draft_kind,
            batch_size=batch_size,
            left_padding=left_padding,
            make_cache=make_cache,
        )
        if draft_kind != "mtp":
            return caches
        if batch_size != 1:
            if not _warned_batch[0]:
                _warned_batch[0] = True
                _log.warning(
                    "KV_BITS with MTP at batch size %d: packed batch "
                    "rollback is unsupported; batched rows keep the stock "
                    "cache", batch_size)
            return caches
        if any(isinstance(c, rotating) for c in caches):
            if not _warned_rotating[0]:
                _warned_rotating[0] = True
                _log.warning(
                    "KV_BITS dropped on the MTP path: sliding-window "
                    "cache stack cannot quantize")
            return caches
        out = []
        n = 0
        for c in caches:
            if type(c) in plain_kv:
                out.append(c.to_quantized(group_size=group, bits=bits))
                n += 1
            else:
                out.append(c)
        if n and not _noted[0]:
            _noted[0] = True
            print(
                f"[kv] MTP spec path: {n}-layer target KV quantized "
                f"({bits}-bit, group {group})",
                flush=True,
            )
        return out

    _quantizing_spec_cache.__dict__[_SPEC_KV_QUANT_FLAG] = True
    _su.make_speculative_prompt_cache = _quantizing_spec_cache
    _ar.make_speculative_prompt_cache = _quantizing_spec_cache
    _gen.make_speculative_prompt_cache = _quantizing_spec_cache
    _debug_note(f"[mtp] spec cache: KV_BITS={bits} group={group} armed (B=1)")
