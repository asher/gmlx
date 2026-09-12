"""Continuous-batch admission for the owned speculative engine.

Split out of ``gmlx.spec.engine``.
"""

from __future__ import annotations

import logging

import mlx.core as mx

from gmlx.envflags import env_bool
from gmlx.spec.engine import _debug_note
from gmlx.spec.kv_quant import batch_liftable, lift_single_cache

_log = logging.getLogger(__name__)


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
                _log.warning("spec batch release: rounds close failed", exc_info=True)
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
                    _log.warning(
                        "spec batch release: drafter reset failed", exc_info=True
                    )
        setattr(self, _RELEASED_FLAG, True)
        setattr(self, _RELEASE_PENDING_FLAG, False)
        mx.clear_cache()
        return True

    def _release_if_finished(self) -> None:
        if _orig_len(self) == 0:
            _release_heavy_state(self)
            return
        _shed_finished_attr_rows(self)

    def _shed_finished_attr_rows(self) -> None:
        """Per-row release of the batch-held start-time snapshots.

        The live rounds generator sheds a finished or filtered row's KV,
        drafter state, and its own hidden/shared_kv slices at the next
        round boundary; the batch object's prefill-time copies (hidden,
        shared_kv_states, prompt_tokens, first_tokens) stayed resident
        until the whole batch finished. Slice them by the surviving rows
        instead. Injected rows carry no snapshot here (their state rides
        the injection queue into the generator), so the snapshot covers
        the first first_tokens.shape[0] physical rows only. Slices are
        lazy and ride the tick's eval; nothing here forces a sync.

        Runs only once the rounds generator holds the state: pre-start,
        _start_rounds still needs the snapshots row-aligned with the
        caches (finished rows included; the generator stop_checks them
        out itself), so a first-token finish must not slice here."""
        if self._rounds_iter is None:
            return
        ft = getattr(self, "first_tokens", None)
        if ft is None or getattr(self, _RELEASED_FLAG, False):
            return
        rows = getattr(self, "_kq_attr_rows", None)
        if rows is None:
            try:
                rows = self._kq_attr_rows = list(range(ft.shape[0]))
            except Exception:
                return
        keep = [p for p in rows
                if p < len(self._finished) and not self._finished[p]]
        if len(keep) == len(rows):
            return
        if not keep:
            self.hidden = None
            self.shared_kv_states = None
            self.prompt_tokens = None
            self.first_tokens = None
            self._kq_attr_rows = []
            return
        keep_set = set(keep)
        pos = [i for i, p in enumerate(rows) if p in keep_set]
        idx = mx.array(pos, dtype=mx.int32)
        for name in ("hidden", "prompt_tokens", "first_tokens"):
            arr = getattr(self, name, None)
            if arr is not None:
                setattr(self, name, arr[idx])
        kv = getattr(self, "shared_kv_states", None)
        if isinstance(kv, dict) and kv:
            # New dict, new arrays: the generator may still hold (and
            # slice) the originals; never mutate a possibly shared dict.
            self.shared_kv_states = {
                k: (K[idx], V[idx]) for k, (K, V) in kv.items()}
        self._kq_attr_rows = keep

    # 2. Buffer extend() instead of raising
    def _buffered_extend(self, other):
        active = sum(not d for d in self._finished)
        if active == 0:
            pending = getattr(self, "_pending_injections", [])
            self.__dict__.pop("_kq_attr_rows", None)
            self.__dict__.update(other.__dict__)
            self._pending_injections = pending
            setattr(self, _RELEASED_FLAG, False)
            setattr(self, _RELEASE_PENDING_FLAG, False)
            return
        if not hasattr(self, "_pending_injections"):
            self._pending_injections = []
        self._pending_injections.append(other)
        _debug_note(f"[mtp] extend buffered: +{len(other._all_uids)} rows "
                    f"(pending={len(self._pending_injections)}, "
                    f"active={active})")

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
                self.__dict__.pop("_kq_attr_rows", None)
                self.__dict__.update(other.__dict__)
                self._pending_injections = remaining
                setattr(self, _RELEASED_FLAG, False)
                setattr(self, _RELEASE_PENDING_FLAG, False)
                return _orig_len(self)
        return active

    SpecBatch.__len__ = _len_with_promotion

    _orig_filter = SpecBatch.filter

    def _compact_prestart_rows(self, keep) -> None:
        """Physically drop rows from a batch whose rounds generator has
        not started: filter the caches through their own filter (lifting
        host caches first) and slice snapshots plus bookkeeping to the
        same keep list. Pre-start, the batch object owns all state, so
        the drop frees the rows' bytes immediately instead of marking
        them finished and waiting for a generator that has no round
        boundary yet."""
        idx = mx.array(keep, dtype=mx.int32)
        self.prompt_cache = [_lift_host_cache(c) for c in self.prompt_cache]
        for c in self.prompt_cache:
            c.filter(idx)
        for name in ("hidden", "prompt_tokens", "first_tokens"):
            arr = getattr(self, name, None)
            if arr is not None:
                setattr(self, name, arr[idx])
        kv = getattr(self, "shared_kv_states", None)
        if isinstance(kv, dict) and kv:
            self.shared_kv_states = {
                k: (K[idx], V[idx]) for k, (K, V) in kv.items()}
        self._all_uids = [self._all_uids[i] for i in keep]
        self.uids = list(self._all_uids)
        self.max_tokens = [self.max_tokens[i] for i in keep]
        self._num_tokens = [self._num_tokens[i] for i in keep]
        self._finished = [False] * len(keep)
        self.__dict__.pop("_kq_attr_rows", None)

    def _filter_with_release(self, keep):
        # Pre-start strict subset (a cancel or a governor retire landing
        # before the first tick): compact physically. Live or degenerate
        # cases keep the upstream mark-finished contract; the running
        # generator sheds the row at its next round boundary and the
        # snapshot shed below covers the batch-held copies.
        if (len(keep) < len(self.uids)
                and keep
                and self._rounds_iter is None
                and not getattr(self, _RELEASED_FLAG, False)
                and getattr(self, "first_tokens", None) is not None
                and self.uids == self._all_uids
                and not any(self._finished)
                and all(batch_liftable(c) for c in self.prompt_cache)):
            _compact_prestart_rows(self, list(keep))
            return
        _orig_filter(self, keep)
        _release_if_finished(self)

    SpecBatch.filter = _filter_with_release

    # 4. Process pending injections in next() before advancing the generator
    _orig_next = SpecBatch.next

    def _note_last_tokens(self, responses) -> None:
        # Last delivered token per uid: the bonus a preempt rebuild restarts
        # from (its KV is not yet in the cache at a round boundary).
        stash = getattr(self, "_kq_last_tokens", None)
        if stash is None:
            stash = self._kq_last_tokens = {}
        for r in responses:
            if r.token is not None:
                stash[r.uid] = int(r.token)

    def _lift_host_cache(c):
        """Promote a single-sequence host cache to its batch class so the
        rebuilt batch generator can extend/filter it (same lift the
        injection path applies to incoming caches)."""
        if hasattr(c, "filter") and hasattr(c, "extend"):
            return c
        return lift_single_cache(c)

    def _preempt_scalar(self) -> bool:
        """Preempt a live scalar (B=1) spec generation so queued rows can
        join: close the generator, deliver the closed round's undelivered
        tail (the scalar path yields one token per next(), so a close
        usually lands mid-round; those tokens are verified and their KV
        stays in the cache), lift the caches to batch classes, and mark
        the batch armless (hidden=None); _start_rounds then rebuilds it on
        the batch loop, whose first injection drain admits the waiters.
        The rebuild resumes from the round's bonus token, whose KV is not
        in the cache. GMLX_MTP_PREEMPT=0 leaves the old drain-wait
        behavior.

        The rebuilt row carries no APC retirement context (batch-loop rows
        start with retire_ctxs None), so the preempted request's prefix is
        not offered back to the prompt cache when it finishes."""
        if not env_bool("GMLX_MTP_PREEMPT", True):
            return False
        if not getattr(self, "_sent_first", False):
            return False
        last = getattr(self, "_kq_last_tokens", {}).get(self._all_uids[0])
        if last is None:
            return False
        # Every cache must be batch-liftable before the generator
        # closes. A quantized or kvarn B=1 cache lifts to fp16. Anything
        # else unliftable declines into the drain-wait.
        if not all(batch_liftable(c) for c in self.prompt_cache):
            return False
        it = self._rounds_iter
        captured = []
        if it is not None:
            self._rounds_iter = None
            self.model._kq_preempt_capture = captured
            try:
                it.close()
            finally:
                try:
                    del self.model._kq_preempt_capture
                except AttributeError:
                    pass
        responses = []
        uid = self._all_uids[0]
        for tok in captured:
            if self._finished[0]:
                break
            tok = int(tok)
            self._num_tokens[0] += 1
            finish = self._finish_reason(0, tok)
            if finish is not None:
                self._finished[0] = True
            responses.append(self.Response(
                uid=uid, token=tok, token_logprob=0.0, finish_reason=finish))
            last = tok
        self._kq_preempt_responses = responses
        if self._finished[0]:
            # The captured tail finished the row; nothing to rebuild. The
            # pending injections promote through __len__ once drained.
            self._refresh_uids()
            return False
        self.prompt_cache = [_lift_host_cache(c) for c in self.prompt_cache]
        self.first_tokens = mx.array([int(last)], dtype=self.token_dtype)
        self.hidden = None
        self.shared_kv_states = None
        self.prompt_tokens = None
        self.model._kq_rebuild_emitted = [int(self._num_tokens[0])]
        _debug_note("[mtp] preempt: scalar generation rebuilt for "
                    "continuous batching")
        return True

    def _next_with_injection(self):
        pending = getattr(self, "_pending_injections", None)
        # Physical-row uids for the owned rounds loop (it has no batch
        # object): read once at generator start, injected rows carry theirs.
        try:
            self.model._kq_row_uids = list(self._all_uids)
        except AttributeError:      # attribute-less model stand-ins
            pass
        # Mid-flight adoption works only when the batch rounds generator is
        # running: it drains model._generator_injections at its round
        # boundaries. The scalar (B=1) generator never does, so a live
        # scalar host is preempted first: its generator closes at the round
        # boundary and the batch is rebuilt armless on the batch loop.
        # `_all_uids` is an mlx-vlm generator internal (stable under the
        # ==0.6.3 pin); re-verify this batch-vs-scalar signal on a pin lift.
        preempted = False
        if pending and len(self._all_uids) == 1:
            preempted = _preempt_scalar(self)
        # The preempt capture: verified tokens the closed round had not yet
        # delivered. They precede everything this call returns.
        pre_responses = self.__dict__.pop("_kq_preempt_responses", None) or []
        if pending and (len(self._all_uids) > 1 or preempted):
            responses = list(pre_responses)
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
                    responses.append(
                        self.Response(
                            uid=other._all_uids[row],
                            token=tok,
                            token_logprob=0.0,
                            finish_reason=finish,
                        )
                    )

                gen_inj.append(
                    {
                        "uids": list(other._all_uids),
                        "prompt_cache": other.prompt_cache,
                        "hidden": other.hidden,
                        "shared_kv_states": other.shared_kv_states,
                        "prompt_tokens": other.prompt_tokens,
                        "first_tokens": other.first_tokens,
                        "first_tokens_list": first_list,
                        # The running generator froze max(max_tokens) at
                        # start; injected rows carry their own budgets.
                        "max_tokens": list(other.max_tokens),
                    }
                )

            pending.clear()
            self._refresh_uids()

            more = _orig_next(self)
            responses.extend(more)
            _note_last_tokens(self, responses)
            _release_if_finished(self)
            return responses

        responses = pre_responses + _orig_next(self)
        _note_last_tokens(self, responses)
        _release_if_finished(self)
        return responses

    SpecBatch.next = _next_with_injection
    setattr(SpecBatch, _CONTINUOUS_BATCH_FLAG, True)
    _debug_note(
        "[mtp] continuous batch: admission gate removed, mid-flight injection enabled"
    )
