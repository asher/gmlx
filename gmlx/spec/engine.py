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

from gmlx.envflags import env_int

_log = logging.getLogger(__name__)

_OWNED_MTP_ROUND_FLAG = "_kq_gguf_owned_mtp_round"
_FULL_PREFILL_FLAG = "_kq_gguf_full_prompt_mtp_prefill"

_SPEC_APC_DISABLED = os.environ.get("GMLX_SPEC_APC", "1") == "0"
# Seed streaming: teacher-force the native MTP head chunk-by-chunk during
# target prefill. 0 defers seeding to one whole-prompt pass after the
# first token, which stalls the stream for seconds at depth.
_SEED_STREAM_DISABLED = os.environ.get("GMLX_MTP_SEED_STREAM", "1") == "0"
# Retirement store (prompt + generated -> shared APC at request finish) is a
# beyond-stock multi-turn win; killable on its own or via the global switch.
_SPEC_APC_RETIRE_DISABLED = (
    _SPEC_APC_DISABLED or os.environ.get("GMLX_SPEC_APC_RETIRE", "1") == "0"
)
_SPEC_APC_SIDECAR_DISABLED = (
    _SPEC_APC_DISABLED or os.environ.get("GMLX_SPEC_APC_SIDECAR", "1") == "0"
)
_SPEC_APC_CKPT_DISABLED = (
    _SPEC_APC_DISABLED or os.environ.get("GMLX_SPEC_APC_CKPT", "1") == "0"
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
        from gmlx.cache.prefix_cache import SpecPrefixCache
        max_entries = env_int("GMLX_SPEC_APC_ENTRIES", 4)
        budget_mb = env_int("GMLX_SPEC_APC_BUDGET_MB", 8192)
        cache = SpecPrefixCache(max_entries=max_entries, max_bytes=budget_mb << 20)
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
    reliably from here). Assigns on every construction -- including None --
    so a BatchGenerator built without APC clears a stale stash instead of
    inheriting one. Idempotent.
    """
    from mlx_vlm.generate.ar import BatchGenerator

    if getattr(BatchGenerator.__init__, _APC_STASH_FLAG, False):
        return
    _orig_init = BatchGenerator.__init__

    def _init_with_stash(self, model, processor, **kwargs):
        # upstream server never passes completion_batch_size; inject ours
        if "completion_batch_size" not in kwargs:
            from gmlx.serve.decode_batch import decode_batch
            kwargs["completion_batch_size"] = decode_batch()
        # Kill switch (re-read per call): with spec APC off, stock ar.py
        # must not see the manager on the speculative path either -- since
        # mlx-vlm 0.6.4 its own post-prefill exact store handles B=1 MTP
        # caches (older versions silently declined them), so a
        # stashed-but-disabled manager would still collect stores.
        if kwargs.get("draft_model") is not None and _SPEC_APC_DISABLED:
            kwargs["apc_manager"] = None
        try:
            model._kq_apc_manager = kwargs.get("apc_manager")
        except Exception:
            pass
        _orig_init(self, model, processor, **kwargs)
        # Stock admission forms a prompt batch only when free slots >=
        # prefill_batch_size. Stock pairs 32/8 (24 slots stay open); the
        # injected width cap pairs 8/8, where a full prefill group equals
        # the whole batch and no request can join while any row decodes:
        # serving degrades to FIFO. Groups of 1 keep insertion live at
        # every width; B>1 prompt batching is no throughput win (see the
        # ckpt formation gate below).
        pbs = getattr(self, "prefill_batch_size", None)
        cbs = getattr(self, "completion_batch_size", None)
        if pbs is not None and cbs is not None and pbs >= cbs:
            self.prefill_batch_size = 1
        # APC arrived armed but upstream's quantized-KV opt-out dropped it
        # (ar.py nulls the manager whenever kv_bits is set; no tier serves
        # quantized caches). The mode probe still reads "block" for these
        # models, so without this line the server boots silent and every
        # request prefills cold. Draft-model batches are excluded: upstream
        # nulls their manager by design and the owned ladder resolves (and
        # warns) through _resolve_l1.
        if (kwargs.get("apc_manager") is not None
                and kwargs.get("kv_bits") is not None
                and kwargs.get("draft_model") is None
                and getattr(self, "apc_manager", None) is None):
            _log.warning(
                "APC OFF: KV quantization (kv_bits=%s) opts out of the "
                "block APC tier upstream -- every request prefills cold",
                kwargs.get("kv_bits"))
        # Ckpt-tier models form prompt batches one request at a time: the
        # owned APC declines B>1 prefill, so a coalesced burst would go
        # all-cold, and B>1 prompt batching is not a throughput win anyway
        # (gemma-31b 2x27k: 130s batched vs 120s serialized). Applies to
        # the stock path too: its ckpt arming is B=1-gated the same way.
        if not _SPEC_APC_DISABLED:
            try:
                manager, mode = _resolve_l1(model)
                if manager is not None and _ckpt_active(
                    model, mode, int(manager.block_size)
                ):
                    self.prefill_batch_size = 1
            except Exception:
                _log.warning(
                    "APC ckpt formation gate failed; continuing", exc_info=True
                )

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
        if mode is None:
            # A manager was built and wired, then silently dropped here --
            # without this line a dead cache is indistinguishable from an
            # idle one (minimax-m3 with the MSA indexer armed).
            kinds = []
            try:
                kinds = sorted({type(c).__name__ for c in lm.make_cache()})
            except Exception:
                pass
            _log.warning(
                "APC OFF for this model: no tier serves its cache stack "
                "(%s) -- every request prefills cold",
                ", ".join(kinds) or "unprobeable")
        try:
            model._kq_apc_mode = mode
        except Exception:
            pass
    if mode is None:
        return None, None
    return manager, mode


def _ckpt_active(model, mode, block_size: int = 16) -> bool:
    """True when the checkpoint tier (chain-backed attn/rotating KV +
    recurrent-state sidecar) replaces the exact tier for this model: a
    supported hybrid cache shape (gated-delta or sliding-window),
    exact mode, kill switch open. Shape probed once per model;
    the module flag is re-read every call so benches can toggle in-process.
    """
    if _SPEC_APC_CKPT_DISABLED or mode != "exact":
        return False
    flag = getattr(model, "_kq_apc_ckpt", None)
    if flag is None:
        from gmlx.cache.snapshot import ckpt_layout
        lm = getattr(model, "language_model", None) or model
        try:
            tags = ckpt_layout(lm.make_cache(), block_size)
        except Exception:
            tags = None
        flag = tags is not None
        if flag:
            _log.info(
                "APC tier: ckpt (layers: %d kv / %d qsa / %d rot / "
                "%d arr / %d kvarn)",
                tags.count("kv"),
                sum(1 for t in tags if t.startswith("qsa")),
                sum(1 for t in tags if t.startswith("rot")),
                tags.count("arr"),
                sum(1 for t in tags if t.startswith("kvarn")))
        try:
            model._kq_apc_ckpt = flag
        except Exception:
            pass
    return bool(flag)


def _ckpt_layout_for(model, block_size: int = 16):
    """The model's stock per-layer tags (make_cache probe), cached on the
    model object. Fallback signature source only -- live batches sign via
    _ckpt_layout_live, which sees per-request cache conversion."""
    tags = getattr(model, "_kq_apc_ckpt_layout", None)
    if tags is None:
        from gmlx.cache.snapshot import ckpt_layout
        lm = getattr(model, "language_model", None) or model
        try:
            tags = tuple(ckpt_layout(lm.make_cache(), block_size) or ())
        except Exception:
            tags = ()
        try:
            model._kq_apc_ckpt_layout = tags
        except Exception:
            pass
    return tags or None


def _ckpt_layout_expected(model, block_size: int = 16):
    """The signature a live request on ``model`` signs with, for probes
    that hold no cache stack (the estimate dry-run). The stock probe,
    with the layers the stamped single-stream kvarn policy converts
    retagged at its wire config; a stock or affine boot is the stock
    probe unchanged (affine caches tag ``kv`` either way)."""
    tags = _ckpt_layout_for(model, block_size)
    if not tags:
        return tags
    try:
        from gmlx.cache.kvarn_serve import (
            _serve_widths_and_tail,
            stamped_single_policy,
        )
        from gmlx.cache.snapshot import kvarn_layout_tag

        single = stamped_single_policy(model)
        if (single is None or getattr(single, "scheme", None) != "kvarn"
                or single.verdict not in ("full", "partial")
                or len(single.per_layer) != len(tags)):
            return tags
        k_bits, v_bits, tail = _serve_widths_and_tail(model)
        tag = kvarn_layout_tag(k_bits, v_bits, tail)
        return tuple(
            tag if t == "kv" and plan.quantize else t
            for t, plan in zip(tags, single.per_layer)
        )
    except Exception:
        return tags


# Signature for a live stack the layout probe rejects: refuses every
# record instead of skipping the check (None) or borrowing the stock
# probe's answer, either of which could adopt a record the live caches
# cannot carry.
_LAYOUT_UNSUPPORTED = ("unsupported",)


def _ckpt_layout_live(batch, block_size: int = 16):
    """Per-batch layout signature from the LIVE prompt cache.

    Cache conversion is per-request (kvarn declines on the MTP path for
    reasons a model-level probe cannot see, and the stock path converts
    between batch construction and arming), so a model-cached signature
    from one request would key every later one wrongly. One pass over the
    cache list, no memoization; the stock make_cache probe serves only
    batches that carry no caches yet."""
    caches = getattr(batch, "prompt_cache", None)
    if caches:
        from gmlx.cache.snapshot import ckpt_layout

        try:
            tags = tuple(ckpt_layout(caches, block_size) or ())
        except Exception:
            tags = ()
        return tags or _LAYOUT_UNSUPPORTED
    return _ckpt_layout_for(getattr(batch, "model", None), block_size)


def _live_kv_quant_config(model=None):
    """The serve KV quant policy as a warm-merge config, or None.

    The batched verdict stamped on the model rules the warm merge:
    a warm hit joins a batch, and under uniform MTP batches run fp16 KV.
    No stamp means None. The only caller is the MTP prefill init, where fp16
    is the only correct merge. Key/value split overrides stay None."""
    stamped = getattr(model, "_gmlx_kv_policy", None)
    if stamped is None:
        return None
    batched = getattr(stamped, "batched", None)
    if batched is None or batched.verdict not in ("full", "partial"):
        return None
    if getattr(batched, "scheme", "uniform") != "uniform":
        # Only affine has a merge config upstream understands. kvarn
        # serves warm prefixes from the fp16 exact tier, so a float merge
        # is the correct one -- but say so rather than describing kvarn
        # records with an affine config.
        _log.debug("warm merge stays float: %s KV has no affine config",
                   batched.scheme)
        return None
    bits = float(batched.bits)
    group = int(batched.group_size or 64)
    try:
        from mlx_vlm.kv_quant import from_legacy
        pol = from_legacy(bits, None, group)
        return pol.to_config() if pol is not None else None
    except Exception:
        _log.warning("KV quant policy resolve failed; warm merge stays "
                     "float", exc_info=True)
        return None

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

    from gmlx.spec.speculative import (
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
            # hidden=None marks a preempted scalar generation rebuilt for
            # continuous batching: it must run the batch loop (arm-from-
            # capture entry), never the scalar fast path.
            if batch_size == 1 and hidden is not None:
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
