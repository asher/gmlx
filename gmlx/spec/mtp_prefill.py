"""Full-prompt MTP prefill and the seed stream.

Split out of ``gmlx.spec.engine``; the install flags
(``_FULL_PREFILL_FLAG``, ``_SEED_STREAM_DISABLED``) stay on the engine,
which the stays band also reads.
"""

from __future__ import annotations

import logging
import os

import mlx.core as mx

import gmlx.lora_rows as lora_rows
import gmlx.gen.prefill_decay as prefill_decay
from gmlx.envflags import env_int
from gmlx.spec.ckpt import (
    _install_ckpt_checkpoint_store,
    _install_exact_anchor_pick,
    _install_plain_ckpt_decode,
    _l1_lookup_and_arm_store,
    _plain_anchor_init,
    _plain_ckpt_init,
    _snap_fields,
)
from gmlx.spec.engine import (
    _FULL_PREFILL_FLAG,
    _L1_BOUND,
    _SEED_STREAM_DISABLED,
    _SPEC_APC_DISABLED,
    _SPEC_APC_RETIRE_DISABLED,
    _bind_l1_view,
    _ckpt_active,
    _debug_note,
    _get_spec_prefix_cache,
    _install_apc_manager_stash,
    _resolve_l1,
)

_log = logging.getLogger(__name__)


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
        _log.info("KQDBG mtp_prefill_init: inputs_embeds None, ladder skipped")
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
                "(owned-path APC requires single-request prefill)",
                b,
            )
        return

    # Serve wraps make_cache so mlx-lm-origin entries carry the mlx-vlm
    # runtime's class identities; embedded and test users reach this init
    # without that wrapper, and the L1 exact tiers dispatch on the vlm
    # classes (an mlx-lm ArraysCache misses every adapter rule). Rebind
    # here so both paths see the same identities. No-op when the entries
    # are already vlm-origin.
    from gmlx.cache.compat import rebind_to_runtime_origin
    rebind_to_runtime_origin(batch.prompt_cache)

    # Upstream admission already restored a prefix and built this batch
    # suffix-only: the owned ladder's keys (L0 and L1 both) are full-prompt
    # token ids, so every lookup and store here would run in the wrong
    # space -- a suffix-keyed L0 entry cross-hits a later turn's suffix and
    # its restore clobbers the upstream warm cache. Leave these batches to
    # the stock machinery, which owns their meta and store schedule.
    up_meta = getattr(batch, "_apc_meta", None) or []
    if up_meta and isinstance(up_meta[0], dict) \
            and int(up_meta[0].get("prefix_len") or 0) > 0:
        batch._mtp_upstream_warm = True
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
                restored,
                int(batch._input_ids.shape[1]) - restored,
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
    if manager is not None and not _SPEC_APC_RETIRE_DISABLED and batch.prompt_cache:
        meta = (batch._apc_meta or [{}])[0] or {}
        full_ids = [int(t) for t in batch._mtp_full_input_ids[0].tolist()]
        from gmlx.cache.retire_key import lookup_render_ctx
        batch.prompt_cache[0]._kq_apc_retire = {
            "full_ids": full_ids,
            "extra_hash": int(meta.get("extra_hash", 0)),
            "mode": (
                "ckpt"
                if _ckpt_active(batch.model, mode, int(manager.block_size))
                else mode
            ),
            "checkpoint_len": int(meta.get("checkpoint_len", 0) or 0),
            # Live reference: the sidecar keys on ckpt_last_stored, not
            # the cursor value frozen above.
            "apc_meta": meta,
            # Render context for the next-turn LCP key (None off the server
            # path or on a media prompt; retirement then keys as before).
            "render_ctx": lookup_render_ctx(full_ids),
            **_snap_fields(batch, manager),
        }

    if restored > 0:
        batch._input_ids = batch._input_ids[:, restored:]
        batch._inputs_embeds = batch._inputs_embeds[:, restored:]
        batch._processed_prompt_columns = restored
        for k in batch._prompt_length_aware_keys:
            batch._prompt_kwargs[k] = batch._prompt_kwargs[k][:, restored:, ...]
        batch._mtp_apc_prefix_len = restored


def _mtp_seed_stream_init(batch) -> None:
    """Arm per-chunk drafter seeding for this request, if eligible.

    Cold full prefill only (v1): any restored prefix (L0/L1/upstream) or warm
    drafter sidecar keeps the deferred one-shot seed -- correctness identical,
    seeding then still runs after the first token. Eligibility here plus the
    per-chunk B re-check in prompt_step; a mid-request stop keeps the partial
    seed KV (adopted at its true offset) and defers only the remainder.

    The seed KV is request-scoped (built via drafter.make_cache, ridden on
    batch state and handed over via a prompt_cache[0] stash exactly like the
    drafter warm sidecar), never the drafter's own _cache: another request's
    live decode round owns that object.
    """
    if hasattr(batch, "_mtp_seed_ctx"):
        return
    batch._mtp_seed_ctx = None
    drafter = getattr(batch, "draft_model", None)
    if (
        _SEED_STREAM_DISABLED
        or drafter is None
        or not callable(getattr(drafter, "seed_chunk", None))
        or getattr(drafter, "hidden_capture_limit", None) is not None
        or int(batch._input_ids.shape[0]) != 1
        or getattr(batch, "_mtp_upstream_warm", False)
        or getattr(batch, "_mtp_chunk_hiddens", None)
        or int(getattr(batch, "_mtp_l1_prefix_len", 0) or 0) != 0
        or int(getattr(batch, "_processed_prompt_columns", 0) or 0) != 0
        or not batch.prompt_cache
        or getattr(batch.prompt_cache[0], "_kq_apc_drafter_warm", None)
            is not None
    ):
        return
    lp = getattr(batch.prompt_cache[0], "left_padding", None)
    if isinstance(lp, mx.array) and lp.size and int(lp.max().item()) > 0:
        return
    try:
        drafter.bind(batch.model)
        seed_kv = drafter.make_cache()
    except Exception:
        _log.warning("seed streaming unavailable for this drafter; "
                     "deferred seed", exc_info=True)
        return
    ctx = {
        "kv": seed_kv,
        "len": 0,
        "active": True,
        # Retain chunk hiddens alongside streaming whenever an L0 store can
        # arm: the store needs full-prompt hidden. APC off => no retention
        # while streaming (the capture-memory win lands in that config).
        "retain": _get_spec_prefix_cache(batch.model) is not None,
        "retained_from": 0,
    }
    batch._mtp_seed_ctx = ctx
    batch.prompt_cache[0]._kq_seed_stream = ctx


def _zero_pad_rows(arr, rows: int):
    pad = mx.zeros((rows - arr.shape[0],) + tuple(arr.shape[1:]), dtype=arr.dtype)
    return mx.concatenate([arr, pad], axis=0)


def _widen_prompt_rope_state(batch, prompt_kwargs: dict) -> dict:
    """Continuous-batch admission can grow the spec prompt batch (and decode
    forwards run at other widths) between chunks; the target caches text
    mrope deltas at the old width and only slices down, never widens, so the
    next chunk forward dies on offsets(B) + rope_deltas(B_old) broadcast.
    Text rows have delta 0, so zero-pad both delta sources to the live width
    (decode-loop twin of this guard: speculative.py injection path)."""
    b = batch._input_ids.shape[0]
    rd = prompt_kwargs.get("rope_deltas")
    if rd is not None and rd.shape[0] < b:
        prompt_kwargs = dict(prompt_kwargs)
        prompt_kwargs["rope_deltas"] = _zero_pad_rows(rd, b)
    lm = getattr(batch.model, "language_model", batch.model)
    rd = getattr(lm, "_rope_deltas", None)
    if rd is not None and rd.shape[0] < b:
        lm._rope_deltas = _zero_pad_rows(rd, b)
    return prompt_kwargs


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
    # The L1 disk tier serializes through mlx-vlm's DiskBlockStore, which
    # has no arm for QSAKVCache and refuses the whole exact snapshot.
    # Installed here as well as in serve patches so embedded/test users of
    # the spec engine get disk APC.
    from gmlx.cache.apc_qsa import install_qsa_apc_support
    install_qsa_apc_support()
    _install_apc_manager_stash()
    _install_ckpt_checkpoint_store()
    _install_plain_ckpt_decode()
    _install_exact_anchor_pick()

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

        return int(os.environ.get("PREFILL_STEP_SIZE", DEFAULT_PREFILL_STEP_SIZE))

    def _mtp_init(self, *args, **kwargs) -> None:
        _orig_init(self, *args, **kwargs)
        # Re-enable chunked prefill.  Stock mlx-vlm nulls prefill_step_size
        # for speculative models because intermediate chunks discard hidden;
        # our prompt_step captures it, so the gate no longer applies.
        # Restoring at construction (not first prompt_step) matters: the
        # scheduler consults needs_processing() first, and with a None step
        # an APC-less deep prompt would one-shot the whole prefill.
        if (
            getattr(self, "draft_kind", None) == "mtp"
            and self.prefill_step_size is None
        ):
            self.prefill_step_size = _resolve_mtp_prefill_step()
        # Stock (non-speculative) batches get the checkpoint tier here:
        # lookup, prefix trim, cursor arming, retirement stash.
        if getattr(self, "draft_kind", None) is None and not _SPEC_APC_DISABLED:
            try:
                # Ckpt-active hybrids under kvarn convert their stock B=1
                # single-stream caches in place before arming, so the
                # layout signature, lookup, and every store see the same
                # kvarn classes. Must run here: the outer batch rebuild
                # (kvarn_serve) would install batch classes the tier is
                # blind to; after conversion its shared decline predicate
                # trips instead. kwargs is load-bearing -- stock init
                # consumes and drops the scheme/bits constructor params.
                manager, mode = _resolve_l1(self.model)
                if manager is not None:
                    from gmlx.cache.kvarn_serve import ensure_ppb_kvarn

                    ensure_ppb_kvarn(
                        self, kwargs,
                        ckpt_active=_ckpt_active(
                            self.model, mode, int(manager.block_size)))
            except Exception:
                _log.warning(
                    "kvarn ckpt cache conversion failed; continuing stock",
                    exc_info=True,
                )
            try:
                _plain_ckpt_init(self)
                _plain_anchor_init(self)
            except Exception:
                _log.warning(
                    "APC plain ckpt init failed; continuing stock", exc_info=True
                )

    def _mtp_prompt_step(self) -> int:
        if self.draft_kind != "mtp":
            return _orig_prompt_step(self)
        # cb_phase flips fine prefill caps by wrapping the stock
        # prompt_step, but this body replaces it for MTP batches, so the
        # flip must happen here too: a multi-thousand-token chunk under
        # the coarse decode caps keeps every layer's transients live in
        # one command buffer and OOMs the GPU on deep prompts.
        if os.environ.get("GMLX_CB_PHASE", "1") != "0":
            from gmlx.serve.cb_phase import flip
            flip("prefill")

        if not hasattr(self, "_mtp_full_input_ids"):
            if self.prefill_step_size is None:
                self.prefill_step_size = _resolve_mtp_prefill_step()
            # APC lookup (L0 then L1) + prefix trim + store arming.
            _mtp_prefill_init(self)
            _mtp_seed_stream_init(self)

        if not self.needs_processing():
            return 0

        # Depth-decayed step: shrink only when this chunk's score transient
        # would exceed the cap (see prefill_decay; keeps MoE weight
        # amortization at shallow depth instead of a global small step).
        step = prefill_decay.decayed_for_batch(self) or self._inputs_embeds.shape[1]
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
        # Media requests ride this body too: keep image blocks whole (a
        # boundary inside a block moves to its edge, see media_spans).
        from gmlx.gen.media_spans import span_aware_prompt_n
        n = span_aware_prompt_n(self, n)
        # A final chunk under ~3 simdgroup tiles routes the projections
        # through the skinny-M kernels, whose accumulation order seeds fp
        # noise that stacked recurrent (GDN) layers amplify into
        # first-token divergence. Absorb such a tail into this chunk so
        # every chunk stays in the wide-GEMM regime. Checkpoint columns
        # stay exact.
        min_tail = env_int("GMLX_PREFILL_MIN_TAIL", 48)
        if checkpoint_col is None and min_tail > 0:
            rem1 = self._inputs_embeds.shape[1] - 1
            tail = rem1 - n
            if 0 < tail < min_tail:
                n = rem1        # absorb: overshoot bounded by min_tail-1
        if n <= 0:
            return 0
        prompt_kwargs = self._prompt_kwargs_for_step(n)
        prompt_kwargs = _widen_prompt_rope_state(self, prompt_kwargs)
        with lora_rows.published(getattr(self, "uids", [])):
            out = self.model(
                self._input_ids[:, :n],
                cache=self.prompt_cache,
                inputs_embeds=self._inputs_embeds[:, :n],
                n_to_process=n,
                return_hidden=True,
                **prompt_kwargs,
            )
        chunk_hidden = out.hidden_states[-1]
        # Seed streaming: teacher-force this chunk into the request-scoped
        # head KV at the head's running offset. The shifted span for
        # columns [c0, c0+n) is prompt[c0+1 : c0+n+1], always in range
        # because generate() keeps at least one residual column (the n-1
        # cap above). A failure or a widened batch stops streaming but
        # keeps the partial KV: the owned round adopts it at its true
        # offset and seeds only the remainder.
        seed_ctx = getattr(self, "_mtp_seed_ctx", None)
        streamed = False
        if seed_ctx is not None and seed_ctx["active"]:
            if int(self._input_ids.shape[0]) != 1:
                seed_ctx["active"] = False
            else:
                c0 = int(self._processed_prompt_columns)
                try:
                    self.draft_model.seed_chunk(
                        self._mtp_full_input_ids[:, c0 + 1:c0 + n + 1],
                        chunk_hidden, seed_ctx["kv"])
                    seed_ctx["len"] += n
                    streamed = True
                except Exception:
                    _log.warning("seed streaming failed at column %d; "
                                 "deferred seed for the remainder", c0,
                                 exc_info=True)
                    seed_ctx["active"] = False
        # Teacher-forcing drafters (native MTP heads) seed their KV from the
        # whole prompt hidden, so every chunk is retained except when the
        # chunk just streamed and no L0 store is armed (nothing downstream
        # reads it). Shared-KV drafters (gemma-4 assistant) read only the
        # last position: keeping just the newest chunk caps capture memory
        # at O(chunk) instead of O(prompt), GBs at deep context.
        if callable(getattr(self.draft_model, "prefill_from_target_hidden", None)):
            if streamed and not seed_ctx["retain"]:
                pass
            else:
                if (seed_ctx is not None and not seed_ctx["retain"]
                        and not self._mtp_chunk_hiddens):
                    # Streaming stopped mid-request with no retention so
                    # far: the retained span starts here, not at column 0.
                    seed_ctx["retained_from"] = int(
                        self._processed_prompt_columns)
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
        mx.eval([c.state for c in self.prompt_cache] + [chunk_hidden]
                + ([c.state for c in seed_ctx["kv"]] if streamed else []))
        self._processed_prompt_columns += n
        # The ckpt cursor rides the wrapped stock store (see
        # _install_ckpt_checkpoint_store).
        self._store_apc_exact_checkpoints()
        self._inputs_embeds = self._inputs_embeds[:, n:]
        self._input_ids = self._input_ids[:, n:]
        for k in self._prompt_length_aware_keys:
            self._prompt_kwargs[k] = self._prompt_kwargs[k][:, n:, ...]
        mx.clear_cache()
        return n

    def _mtp_generate(
        self, sampler, stop_criteria, compute_logprobs=True, top_logprobs_k=0
    ):
        if self.draft_kind == "mtp":
            # Short prompts never enter prompt_step (chunked prefill is not
            # needed), so the APC lookup/store arming runs here instead.
            _mtp_prefill_init(self)
        result = _orig_generate(
            self,
            sampler,
            stop_criteria,
            compute_logprobs=compute_logprobs,
            top_logprobs_k=top_logprobs_k,
        )
        from mlx_vlm.generate.ar import SpeculativeGenerationBatch

        if self.draft_kind != "mtp" or not isinstance(
            result, SpeculativeGenerationBatch
        ):
            # Stock-path ckpt batches store the full prompt here, the
            # moment the MTP path stores it at rounds entry: prefill just
            # finished, the first token is out, its KV not yet appended.
            if (
                getattr(self, "_kq_ckpt_armed", False)
                and getattr(self, "draft_kind", None) is None
            ):
                try:
                    cache = getattr(result, "prompt_cache", None) or []
                    stash = getattr(cache[0], "_kq_apc_retire", None) if cache else None
                    if stash is not None and stash.get("mode") == "ckpt":
                        from gmlx.cache.snapshot import (
                            ckpt_full_store_redundant,
                            ckpt_store,
                        )
                        m = stash.get("apc_meta")
                        if ckpt_full_store_redundant(m):
                            _log.info("APC ckpt post-prefill store "
                                      "skipped: render-stable boundary "
                                      "landed")
                        elif ckpt_store(
                                stash["manager"], stash["full_ids"], cache,
                                extra_hash=int(stash.get("extra_hash", 0))):
                            if m is not None:
                                m.setdefault(
                                    "ckpt_stored_boundaries", []
                                ).append(len(stash["full_ids"]))
                except Exception:
                    _log.warning(
                        "APC plain post-prefill store failed; continuing", exc_info=True
                    )
            return result
        chunk_hiddens = getattr(self, "_mtp_chunk_hiddens", None)
        full_ids = getattr(self, "_mtp_full_input_ids", None)
        l1_prefix = int(getattr(self, "_mtp_l1_prefix_len", 0) or 0)
        if not chunk_hiddens:
            # No captured chunks: the whole (remaining) prompt went through
            # the final generate forward, so stock prompt_tokens/hidden are
            # already an aligned pair (suffix-only on an L1 hit) and
            # result.hidden needs no rebuild; with seed streaming and no
            # retention, result.hidden is already the residual unstreamed
            # tail (retention accompanies an armed L0 store, so none can
            # fire here). The L0 store below must still run for the
            # single-shot case: arch prefill profiles can raise the step
            # past typical prompt lengths (qwen4exp defaults to 8192), so
            # sub-step prompts land here and still need their warm-start
            # entry.
            full_hidden = result.hidden
        else:
            parts = chunk_hiddens + [result.hidden]
            full_hidden = mx.concatenate(parts, axis=1)
        seed_ctx = getattr(self, "_mtp_seed_ctx", None)
        seed_len = int(seed_ctx["len"]) if seed_ctx else 0
        if chunk_hiddens:
            if seed_len > 0:
                # Columns [0, seed_len) are already teacher-forced into
                # the streamed head KV; hand the owned round only the
                # residual hidden so its seed call covers exactly the
                # unstreamed tail at the adopted offset. full_hidden (the
                # retained span) still feeds the L0 store below, which
                # needs the whole prompt.
                rfrom = int(seed_ctx.get("retained_from") or 0)
                result.hidden = full_hidden[:, seed_len - rfrom:]
            else:
                result.hidden = full_hidden
        if chunk_hiddens and full_ids is not None:
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
        # With streaming, full_hidden covers the whole prompt only when
        # retention ran from column 0: after a mid-request streaming stop
        # the retained span starts past column 0, and with no retention at
        # all full_hidden is just the residual tail. Neither must ever be
        # stored as a full-prompt entry.
        full_covers_prompt = seed_len == 0 or (
            bool(chunk_hiddens)
            and int(seed_ctx.get("retained_from") or 0) == 0)
        spec_cache = (
            _get_spec_prefix_cache(self.model)
            if b == 1 and l1_prefix == 0 and full_covers_prompt
            and not getattr(self, "_mtp_upstream_warm", False) else None
        )
        if spec_cache is not None and full_ids is not None:
            # Window-limited heads only use the trailing capture window;
            # chunked prefill already trimmed, single-shot must match (an
            # uncapped entry pins the whole prompt's hidden for nothing).
            limit = getattr(self.draft_model, "hidden_capture_limit", None)
            store_hidden = (full_hidden if not limit
                            else full_hidden[:, -int(limit):])
            spec_cache.store(full_ids, result.prompt_cache, store_hidden)
            _log.info(
                "APC store: tokens=%d layers=%d",
                int(full_ids.shape[1]),
                len(result.prompt_cache),
            )
        else:
            _log.debug(
                "APC store skipped: b=%d l1_prefix=%d upstream_warm=%s "
                "full_ids=%s",
                b, l1_prefix,
                getattr(self, "_mtp_upstream_warm", False),
                "set" if full_ids is not None else "None",
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
    _debug_note(
        f"[mtp] serve prefill: full-prompt hidden capture installed (APC {apc_status})"
    )
