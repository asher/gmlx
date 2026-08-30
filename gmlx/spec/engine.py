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

import gmlx.lora_rows as lora_rows
import gmlx.gen.prefill_decay as prefill_decay
from gmlx.envflags import env_bool, env_int

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
        from gmlx.cache.prefix_cache import SpecPrefixCache
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
                        model, mode, int(manager.block_size)):
                    self.prefill_batch_size = 1
            except Exception:
                _log.warning("APC ckpt formation gate failed; continuing",
                             exc_info=True)

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
                "APC tier: ckpt (layers: %d kv / %d qsa / %d rot / %d arr)",
                tags.count("kv"),
                sum(1 for t in tags if t.startswith("qsa")),
                sum(1 for t in tags if t.startswith("rot")),
                tags.count("arr"))
        try:
            model._kq_apc_ckpt = flag
        except Exception:
            pass
    return bool(flag)


def _ckpt_layout_for(model, block_size: int = 16):
    """The live model's per-layer tags for the lookup-side signature check
    (compared against the freshly constructed model, never a stored
    entry). Cached on the model object."""
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


def _live_kv_quant_config():
    """The serve KV quant policy as a warm-merge config, or None.

    Env-sourced (KV_BITS / KV_QUANT_SCHEME / KV_GROUP_SIZE) like the
    owned B=1 spec path: serve config feeds these vars and the engine
    reads the same channel, so the merged warm batch matches the live
    ``_make_cache`` layer types. Key/value split overrides are not
    exposed in gmlx config and stay None."""
    raw = os.environ.get("KV_BITS", "")
    if not raw:
        return None
    try:
        bits = float(raw)
    except ValueError:
        return None
    if bits <= 0:
        return None
    try:
        from mlx_vlm.kv_quant import from_legacy
        pol = from_legacy(
            bits, os.environ.get("KV_QUANT_SCHEME") or None,
            int(os.environ.get("KV_GROUP_SIZE", "64") or 64))
        return pol.to_config() if pol is not None else None
    except Exception:
        _log.warning("KV quant policy resolve failed; warm merge stays "
                     "float", exc_info=True)
        return None


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
    ckpt = _ckpt_active(batch.model, mode, int(manager.block_size))
    held_blocks = []
    l1_prefix = 0
    if l0_prefix == 0 and len(ids_list) >= 2:
        warm = None
        blocks = []
        prefix_len = 0
        tier = "exact"
        pick = view._apc_pick_for((0, ids_list, 0, prompt_kwargs, None, None))
        # Same trivial-pick floor as the admission wrapper: a sub-block
        # exact restore saves nothing and its nonzero l1_prefix would skip
        # the L0 hidden store for this request.
        if (pick is not None and not pick.get("matched_blocks")
                and 0 < int(pick.get("prefix_len") or 0)
                < int(manager.block_size)):
            pick = None
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
            from gmlx.cache.snapshot import ckpt_lookup
            min_p = max(prefix_len,
                        view._apc_safe_prefix_lookup_min(ids_list))
            cw, cp = ckpt_lookup(
                manager, ids_list, extra_hash=extra_hash,
                min_prefix_tokens=min_p,
                layout=_ckpt_layout_for(batch.model,
                                        int(manager.block_size)))
            if (cw is not None and cp > prefix_len
                    and view._apc_suffix_is_text_only(ids_list, cp)):
                if blocks:
                    manager.release(blocks)
                    blocks = []
                warm, prefix_len, tier = cw, cp, "ckpt"
        elif mode == "exact":
            # Exact-tier anchor: the shared-system-prefix clone in the
            # gmlx anchor LRU wins only when strictly longer than the
            # stock exact pick. Media guards mirror the stock probe.
            from gmlx.cache.snapshot import anchor_exact_lookup
            min_p = max(prefix_len,
                        view._apc_safe_prefix_lookup_min(ids_list))
            aw, ap = anchor_exact_lookup(
                manager, ids_list, extra_hash=extra_hash,
                min_prefix_tokens=min_p)
            if (aw is not None and ap > prefix_len
                    and view._apc_suffix_is_text_only(ids_list, ap)):
                if blocks:
                    manager.release(blocks)
                    blocks = []
                warm, prefix_len, tier = aw, ap, "anchor"
        if warm and 0 < prefix_len < len(ids_list) and tier in (
                "exact", "anchor"):
            # Same batch-aware merge admission applies to its picks: raw
            # exact/anchor clones carry single-row leaves (left_padding
            # None, scalar offsets) and crash the batch cache classes'
            # update path (mx.depends on a None) when the suffix forwards.
            # kv_quant_config re-quantizes the float snapshot to the live
            # _make_cache layer types under serve kv_bits (stored exact
            # entries stay float; a float row joining a quantized batch
            # breaks the update path).
            from mlx_vlm import apc as _apc
            warm, _ = _apc.make_warm_batch_exact_cache_multi(
                [warm], prefix_lens=[prefix_len],
                kv_quant_config=_live_kv_quant_config())
        if warm and 0 < prefix_len < len(ids_list):
            batch.prompt_cache = warm
            # Matched blocks stay acquired until the stock post-prefill
            # harvest releases them (the warm-cache concatenation is
            # lazy; the pool must not recycle these blocks before it
            # materializes).
            held_blocks = blocks
            l1_prefix = prefix_len
            # Observability only: the live request view reads the
            # restored prefix from here (meta keeps prefix_len 0 so the
            # stock machinery does not account it twice).
            batch._kq_apc_restored = (int(prefix_len), str(tier))
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
                from gmlx.cache.snapshot import drafter_sidecar_lookup
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
    guard = int(view._apc_exact_checkpoint_len(ids_list) or 0)
    meta = {
        "full_input_ids": ids_list,
        "prefix_len": 0,
        "extra_hash": extra_hash,
        "apc_blocks": held_blocks,
        "checkpoint_len": guard,
    }
    batch._apc_meta = [meta]
    if ckpt:
        # The checkpoint tier replaces the stock exact-tier stores: the
        # post-prefill full-cache clone is suppressed here, and the
        # mid-prefill checkpoint store is superseded by the cursor riding
        # the wrapped stock store (_install_ckpt_checkpoint_store; the
        # stock body is suppressed by the cursor's advance). Column
        # alignment itself still runs on the stock machinery, which
        # requires _apc_mode == "exact".
        _ckpt_arm_schedule(batch, meta, guard,
                           max(l0_prefix, l1_prefix),
                           int(manager.block_size))
        batch._apc_harvest_enabled = False
        batch._kq_ckpt_armed = True
        from gmlx.cache.snapshot import ckpt_note_armed
        ckpt_note_armed(manager)
    elif mode == "exact":
        _exact_anchor_arm(batch, meta, guard,
                          max(l0_prefix, l1_prefix))
    return l1_prefix


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _ckpt_unit(batch, block_size: int) -> int:
    """The natural chunk grid: lcm(prefill_step_size, block_size)."""
    step = int(getattr(batch, "prefill_step_size", 0) or 0)
    return block_size if step <= 0 else \
        step * block_size // _gcd(step, block_size)


def _ckpt_cursor_init(batch, guard: int, restored: int,
                      block_size: int) -> tuple[list, int, int]:
    """Boundary schedule for the checkpoint cursor: an ordered
    ``[(position, kind), ...]`` list plus ``(terminal, interval)``.

    Boundaries sit on the natural chunk grid, lcm(prefill_step_size,
    block_size) -- an off-grid boundary truncates a chunk, and gated-delta
    state is chunk-shape sensitive (certified: any grid change drifts).
    Interval points above the restored prefix, then the terminal (the
    grid point at or below the stock guard column). GMLX_APC_CKPT_INTERVAL
    tokens, default 4096, snapped up to the grid; 0 = terminal-only.
    Later stages add store positions by appending boundaries here, never
    by new store mechanisms.
    """
    unit = _ckpt_unit(batch, block_size)
    terminal = (guard // unit) * unit
    if terminal <= max(0, restored):
        return [], 0, 0
    raw = env_int("GMLX_APC_CKPT_INTERVAL", 4096)
    interval = 0 if raw <= 0 else max(unit, (raw // unit) * unit)
    bounds = []
    if interval:
        b = ((max(0, restored) // interval) + 1) * interval
        while b < terminal:
            bounds.append((b, "boundary"))
            b += interval
    bounds.append((terminal, "boundary"))
    return bounds, terminal, interval


def _ckpt_replay_boundary(batch, meta, restored: int,
                          block_size: int) -> int | None:
    """N-1 replay boundary, or None when it cannot earn its pause.

    An identical resend can only adopt a record strictly below the
    query, and the interval/terminal schedule never places one there
    for prompts under one interval (the depth e2e's bug 1); N-1 is the
    deepest position that is both adoptable and drift-free (the warm
    turn forwards exactly one token), and the pause is free on the cold
    side -- both prefill loops already stop at N-1 to feed the first
    decode step, so the boundary lands on a natural chunk edge and
    perturbs no chunk shape. arr layouts gate on a minimum N:
    recurrent state is prompt-length-independent (>100 MB per record on
    27B-class), and short-prompt records would churn the LRU out of the
    deep-conversation records it exists to protect. Rotating layouts
    need N-1 at or past the window (below it the store's grid gate
    declines). Kill switch: GMLX_APC_CKPT_REPLAY=0.
    """
    if env_int("GMLX_APC_CKPT_REPLAY", 1) == 0:
        return None
    n = len(meta.get("full_input_ids") or ())
    replay = n - 1
    if replay < 2 or replay <= max(0, restored):
        return None
    tags = _ckpt_layout_for(getattr(batch, "model", None), block_size) or ()
    if "arr" in tags and n < env_int("GMLX_APC_CKPT_REPLAY_MIN", 1024):
        return None
    for t in tags:
        if t.startswith("rot") and replay < int(t.split(":")[1]):
            return None
    return replay


def _ckpt_turn_boundaries(batch, meta, restored: int,
                          block_size: int) -> list[int]:
    """Render-stable turn boundary positions for the schedule.

    p_stable is the deepest prompt position a next-turn re-render keeps;
    the gen-prompt/think tail past it is re-rendered away, so records
    stored only above it can never serve turn 2 (how multi-turn adoption
    silently died). Every layout gets the grid point at or below
    p_stable (drift-free for chunk-shape-sensitive state); rot-only
    layouts also pause exactly at p_stable (attention splits exactly;
    needs the window wrapped). GMLX_APC_CKPT_TURN=0 disables these and
    with them the p=N drop gate.
    """
    if env_int("GMLX_APC_CKPT_TURN", 1) == 0:
        return []
    ids = meta.get("full_input_ids") or ()
    unit = _ckpt_unit(batch, block_size)
    tags = _ckpt_layout_for(getattr(batch, "model", None), block_size) or ()
    ws = [int(t.split(":")[1]) for t in tags if t.startswith("rot")]
    # Cheapest boundary this layout could arm: the grid needs one unit
    # of stable prefix; rot-only layouts can also pause exactly at
    # p_stable once the window wraps. Below that no boundary can land,
    # so skip the render+tokenize prediction entirely.
    need = unit if ("arr" in tags or not ws) else min(unit, max(ws))
    if len(ids) - 1 < need:
        return []
    from gmlx.cache.retire_key import lookup_render_ctx, prompt_stable_lcp
    ctx = lookup_render_ctx(ids)
    p_stable = prompt_stable_lcp(ctx, ids) if ctx else None
    if not p_stable or p_stable < 2:
        return []
    p_stable = min(int(p_stable), len(ids) - 1)
    meta["ckpt_p_stable"] = p_stable
    floor = max(0, restored)
    out = []
    grid = (p_stable // unit) * unit
    if grid > floor:
        out.append(grid)
    if ws and "arr" not in tags and p_stable != grid \
            and p_stable > floor and p_stable >= max(ws):
        out.append(p_stable)
    return out


def _ckpt_sys_boundary(batch, meta, restored: int,
                       block_size: int) -> int | None:
    """Anchor stop at the end of the shared system prefix.

    Sibling fan-out requests share the system prompt and tool schemas
    and diverge at the first user message, generally between grid
    points, so the interval schedule alone wastes up to one interval of
    sibling recompute, and strip-on-extend removes the early boundary
    the siblings need as the chain deepens (the anchor exemption in
    _record_insert keeps this one). arr layouts snap the stop down to
    the chunk grid (off-grid chunking drifts GDN state) and keep the
    replay byte floor (recurrent state is prompt-length-independent, so
    a tiny anchor costs the same >100 MB clone as a deep one);
    attention layouts snap to the block grid, which also satisfies the
    rotating store's below-window grid gate. GMLX_APC_CKPT_SYS=0
    disables; GMLX_APC_CKPT_SYS_MIN floors the position (a sub-floor
    shared prefix re-prefills in milliseconds and is not worth a
    record).
    """
    if env_int("GMLX_APC_CKPT_SYS", 1) == 0:
        return None
    ids = meta.get("full_input_ids") or ()
    tags = _ckpt_layout_for(getattr(batch, "model", None), block_size) or ()
    floor_min = max(block_size, env_int("GMLX_APC_CKPT_SYS_MIN", 256))
    if "arr" in tags:
        floor_min = max(floor_min,
                        env_int("GMLX_APC_CKPT_REPLAY_MIN", 1024))
    # Below the floor no anchor can land; skip the render+tokenize
    # prediction entirely (same rule as the turn boundaries).
    if len(ids) - 1 < floor_min:
        return None
    from gmlx.cache.retire_key import lookup_render_ctx, system_prefix_lcp
    ctx = lookup_render_ctx(ids)
    lcp = system_prefix_lcp(ctx, ids) if ctx else None
    if not lcp:
        return None
    unit = _ckpt_unit(batch, block_size) if "arr" in tags else block_size
    pos = (min(int(lcp), len(ids) - 1) // unit) * unit
    if pos < floor_min or pos <= max(0, restored):
        return None
    meta["ckpt_sys_bound"] = pos
    return pos


def _exact_anchor_boundary(batch, meta, guard: int,
                           restored: int) -> int | None:
    """Anchor position for an exact-tier (non-ckpt) model: the sibling
    divergence point, ungridded (exact clones restore at any position).
    Clamped to the stock guard column, so the prefill pauses at most
    twice: once for the anchor, once for the stock guard store.
    GMLX_APC_CKPT_SYS=0 disables (one switch for both tiers);
    GMLX_APC_CKPT_SYS_MIN floors the position (a sub-floor shared
    prefix re-prefills in milliseconds and is not worth a clone).
    """
    if env_int("GMLX_APC_CKPT_SYS", 1) == 0:
        return None
    ids = meta.get("full_input_ids") or ()
    floor_min = max(2, env_int("GMLX_APC_CKPT_SYS_MIN", 256))
    if len(ids) - 1 < floor_min:
        return None
    from gmlx.cache.retire_key import lookup_render_ctx, system_prefix_lcp
    ctx = lookup_render_ctx(ids)
    lcp = system_prefix_lcp(ctx, ids) if ctx else None
    if not lcp:
        _log.info("APC anchor declined: no measurable system prefix "
                  "(render ctx %s)", "present" if ctx else "missing")
        return None
    pos = min(int(lcp), len(ids) - 1)
    if guard > 0:
        pos = min(pos, guard)
    if pos < floor_min or pos <= max(0, restored):
        return None
    return pos


def _exact_anchor_arm(batch, meta, guard: int, restored: int) -> None:
    """Schedule the anchor pause by mirroring its position into
    ``checkpoint_len`` (the key the stock column truncation reads).
    ``_exact_anchor_store`` hands the column back to the stock guard
    after firing, so the stock store still runs exactly as unarmed."""
    pos = _exact_anchor_boundary(batch, meta, guard, restored)
    if pos is None:
        return
    meta["anchor_len"] = pos
    meta["anchor_guard"] = guard
    if pos != guard:
        meta["checkpoint_len"] = pos
    batch._kq_anchor_armed = True
    _log.info("APC anchor armed: pos=%d guard=%d", pos, guard)


def _exact_anchor_store(batch) -> None:
    """Anchor store for exact-tier models: one whole-prefix clone at the
    sibling divergence, into the gmlx anchor LRU. Runs from the wrapped
    stock store immediately before the stock body; after firing it
    restores ``checkpoint_len`` to the stock guard column without
    latching ``checkpoint_done``, so the stock guard store (and its
    latch) fire untouched."""
    manager = getattr(batch, "_apc_manager", None)
    meta_list = getattr(batch, "_apc_meta", None) or []
    if manager is None or not meta_list or meta_list[0] is None:
        return
    meta = meta_list[0]
    pos = int(meta.get("anchor_len") or 0)
    if pos <= 0 or meta.get("anchor_done"):
        return
    if batch._row_real_tokens_processed(0) != pos:
        return
    meta["anchor_done"] = True
    guard = int(meta.get("anchor_guard") or 0)
    if int(meta.get("checkpoint_len") or 0) == pos and pos != guard:
        meta["checkpoint_len"] = guard
    cache = batch._apc_prompt_cache_for_store(0)
    if cache is None:
        return
    from gmlx.cache.snapshot import anchor_exact_store
    anchor_exact_store(manager, meta["full_input_ids"][:pos], cache,
                       extra_hash=int(meta.get("extra_hash", 0)))


def _sched_insert(bounds: list, pos: int, kind: str, *,
                  upgrade: bool = False) -> None:
    """Insert (pos, kind) keeping order. On collision the existing entry
    keeps its kind: a colliding position is always grid-aligned or an
    exact turn boundary, where a plain boundary record adopts freely --
    identical resend included -- while flipping it to replay would gate
    turn-2 and branch adoption out on recurrent layouts (and satisfy the
    p=N drop with a record turn 2 cannot use). ``upgrade`` lets an
    anchor replace a plain boundary at the same position (strictly more
    retention, same free adoption), never a replay."""
    import bisect

    pts = [b for b, _ in bounds]
    i = bisect.bisect_left(pts, pos)
    if i < len(pts) and pts[i] == pos:
        if upgrade and bounds[i][1] == "boundary":
            bounds[i] = (pos, kind)
        return
    bounds.insert(i, (pos, kind))


def _ckpt_arm_schedule(batch, meta, guard: int, restored: int,
                       block_size: int) -> None:
    """Publish the boundary schedule into the request meta. The head
    mirrors into ``checkpoint_len`` (an int) because the stock
    checkpoint-column truncation and store reads exactly that key.
    ``ckpt_stored_boundaries`` collects every boundary whose store
    landed (record verified in the index) -- the settled variable the
    post-prefill p=N decision and the sidecar key set both read;
    ``ckpt_p_stable_bounds`` is the qualifying set for the p=N drop."""
    bounds, terminal, interval = _ckpt_cursor_init(
        batch, guard, restored, block_size)
    turn = _ckpt_turn_boundaries(batch, meta, restored, block_size)
    for pos in turn:
        _sched_insert(bounds, pos, "boundary")
    sysb = _ckpt_sys_boundary(batch, meta, restored, block_size)
    if sysb is not None:
        _sched_insert(bounds, sysb, "anchor", upgrade=True)
    replay = _ckpt_replay_boundary(batch, meta, restored, block_size)
    if replay is not None:
        # Colliding with the anchor keeps the anchor (default no-upgrade):
        # it adopts identical resends freely, replay semantics add nothing.
        _sched_insert(bounds, replay, "replay")
    meta["ckpt_boundaries"] = bounds
    meta["checkpoint_len"] = int(bounds[0][0]) if bounds else 0
    meta["ckpt_terminal"] = terminal
    meta["ckpt_interval"] = interval
    meta["ckpt_last_stored"] = 0
    meta["ckpt_stored_boundaries"] = []
    meta["ckpt_p_stable_bounds"] = turn


def _ckpt_mid_prefill_store(batch) -> None:
    """Checkpoint-tier replacement for the stock mid-prefill exact store.

    Fires at the schedule head, pops it, and mirrors the next head into
    ``checkpoint_len``, latching ``checkpoint_done`` when the schedule
    empties. The advance is what suppresses the stock store;
    ``_install_ckpt_checkpoint_store`` wraps the stock method so the
    cursor always runs immediately before it -- the ordering is
    structural, not positional. Advances past failed stores;
    ``ckpt_last_stored`` records only boundaries that landed.
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
    terminal = int(meta.get("ckpt_terminal") or 0)
    bounds = meta.get("ckpt_boundaries") or []
    kind = "boundary"
    if bounds and int(bounds[0][0]) == checkpoint_len:
        kind = str(bounds.pop(0)[1])
    # GDN skeletons inline >100 MB of state; boundaries superseded within
    # the same prefill do not earn disk, only the terminal does -- and a
    # replay skeleton would buy restart-repair of an identical resend
    # only, which does not earn it either.
    layout = _ckpt_layout_for(getattr(batch, "model", None),
                              int(manager.block_size)) or ()
    skel = "arr" not in layout or (kind != "replay"
                                   and checkpoint_len >= terminal)
    from gmlx.cache.snapshot import ckpt_store
    if ckpt_store(
            manager, meta["full_input_ids"][:checkpoint_len],
            batch.prompt_cache, extra_hash=int(meta.get("extra_hash", 0)),
            skeleton_disk=skel, kind=kind):
        meta["ckpt_last_stored"] = checkpoint_len
        meta.setdefault("ckpt_stored_boundaries", []).append(checkpoint_len)
    if bounds:
        meta["checkpoint_len"] = int(bounds[0][0])
    else:
        meta["checkpoint_done"] = True


_CKPT_STORE_FLAG = "_kq_ckpt_cursor_store"


def _install_ckpt_checkpoint_store() -> None:
    """Wrap the stock mid-prefill checkpoint store so the cursor runs
    immediately before it on armed batches (both the owned MTP prefill
    and the stock prompt_step call the stock method, so one wrap covers
    both paths). The cursor's advance of ``checkpoint_len`` is what
    suppresses the stock store -- wrapping makes that ordering
    structural. Exact-tier anchor batches ride the same wrap with their
    own single-stop hook. Idempotent."""
    from mlx_vlm.generate.ar import PromptProcessingBatch
    if getattr(PromptProcessingBatch._store_apc_exact_checkpoints,
               _CKPT_STORE_FLAG, False):
        return
    _orig = PromptProcessingBatch._store_apc_exact_checkpoints

    def _store_with_ckpt_cursor(self):
        if getattr(self, "_kq_ckpt_armed", False):
            _ckpt_mid_prefill_store(self)
        elif getattr(self, "_kq_anchor_armed", False):
            _exact_anchor_store(self)
        _orig(self)

    _store_with_ckpt_cursor.__dict__[_CKPT_STORE_FLAG] = True
    PromptProcessingBatch._store_apc_exact_checkpoints = \
        _store_with_ckpt_cursor


def _snap_fields(batch, manager) -> dict:
    """Decode-time snapshot ring parameters for a retirement stash.

    ``snap_grid`` anchors snapshot positions to the prefill chunk grid
    (lcm of step and block size), so a restore replays chunk-exact --
    but only while one grid unit fits inside the snapshot interval; a
    serve-sized step (2048) would otherwise push the first snapshot far
    past prompt end + interval, so it falls back to the block size (the
    off-grid restore is the scoped-benign case). ``snap_align`` is the
    block alignment a rotating window store requires below the window;
    ``snap_offgrid_min`` (= W) is where the store gate stops caring --
    a wrapped window is whole blocks at any p.
    """
    import math
    from gmlx.cache.snapshot import _DECODE_CKPT_DEFAULT
    bs = int(manager.block_size)
    tags = _ckpt_layout_for(batch.model, bs) or ()
    step = int(getattr(batch, "prefill_step_size", 0) or 0)
    grid = math.lcm(step, bs) if step > 0 else bs
    if grid > env_int("GMLX_APC_DECODE_CKPT", _DECODE_CKPT_DEFAULT):
        grid = bs
    rot_w = 0
    for t in tags:
        if t.startswith("rot"):
            rot_w = int(t.split(":")[1])
            break
    return {
        "snap_ok": bool(tags),
        "snap_grid": grid,
        "snap_align": bs if rot_w else 1,
        "snap_offgrid_min": rot_w,
    }


def _plain_ckpt_init(batch) -> None:
    """Checkpoint-tier lookup + arming for a stock (non-speculative)
    prompt batch.

    The stock path reaches the tier only here: exact-tier stores are
    suppressed on ckpt models, so admission's own lookup ladder misses
    and every ckpt-tier request arrives as a cold single-request batch.
    Lookup and in-place prefix trim mirror the owned MTP prefill
    (single-row caches throughout; the batched warm-merge machinery
    never runs). B=1 unbatched batches only; anything else stays stock.
    """
    manager = getattr(batch, "_apc_manager", None)
    mode = getattr(batch, "_apc_mode", None)
    meta_list = getattr(batch, "_apc_meta", None) or []
    if (manager is None or mode != "exact" or len(meta_list) != 1
            or meta_list[0] is None or len(batch.uids) != 1
            or batch._right_pad_per_row is not None
            or batch._inputs_embeds is None):
        return
    bs = int(manager.block_size)
    if not _ckpt_active(batch.model, mode, bs):
        return
    meta = meta_list[0]
    if int(meta.get("prefix_len") or 0):
        return                          # stock warm row: leave it stock
    ids_list = [int(t) for t in meta["full_input_ids"]]
    if len(ids_list) < 2:
        return
    extra_hash = int(meta.get("extra_hash", 0))
    view = _L1View(batch.model, manager, mode)
    restored = 0
    from gmlx.cache.snapshot import ckpt_lookup
    warm, cp = ckpt_lookup(
        manager, ids_list, extra_hash=extra_hash,
        min_prefix_tokens=view._apc_safe_prefix_lookup_min(ids_list),
        layout=_ckpt_layout_for(batch.model, bs))
    if (warm is not None and 0 < cp < len(ids_list)
            and view._apc_suffix_is_text_only(ids_list, cp)):
        batch.prompt_cache = warm
        batch._input_ids = batch._input_ids[:, cp:]
        batch._inputs_embeds = batch._inputs_embeds[:, cp:]
        batch._processed_prompt_columns = cp
        for k in batch._prompt_length_aware_keys:
            batch._prompt_kwargs[k] = batch._prompt_kwargs[k][:, cp:, ...]
        restored = cp
        batch._kq_apc_restored = (int(cp), "ckpt")     # live request view
        _log.info("APC L1 hit: prefix=%d suffix=%d tier=ckpt",
                  cp, len(ids_list) - cp)
    guard = int(meta.get("checkpoint_len") or 0)
    _ckpt_arm_schedule(batch, meta, guard, restored, bs)
    batch._apc_harvest_enabled = False
    batch._kq_ckpt_armed = True
    from gmlx.cache.snapshot import ckpt_note_armed
    ckpt_note_armed(manager)
    if not _SPEC_APC_RETIRE_DISABLED and batch.prompt_cache:
        from gmlx.cache.retire_key import lookup_render_ctx
        batch.prompt_cache[0]._kq_apc_retire = {
            "full_ids": ids_list,
            "extra_hash": extra_hash,
            "mode": "ckpt",
            "checkpoint_len": int(meta.get("checkpoint_len") or 0),
            "apc_meta": meta,
            "render_ctx": lookup_render_ctx(ids_list),
            "manager": manager,
            "gen": [],
            **_snap_fields(batch, manager),
        }


def _plain_anchor_init(batch) -> None:
    """Arm the exact-tier anchor stop on a stock prompt batch (non-ckpt
    exact models: DeepSeek-V4-class pooling stacks).

    Restores come from the admission pick (_install_exact_anchor_pick),
    so this only schedules the store. Warm and right-padded rows are
    included: a restored prefix is usually far short of the divergence
    (a bare bos match off some unrelated request), and upstream's
    checkpoint column and row extraction handle both shapes. Refusing
    them would skip every row that rides a warm batch, which on a busy
    server is nearly all of them. The restored prefix becomes the
    boundary floor, so a row already past the divergence arms nothing.
    """
    manager = getattr(batch, "_apc_manager", None)
    mode = getattr(batch, "_apc_mode", None)
    meta_list = getattr(batch, "_apc_meta", None) or []
    if (manager is None or mode != "exact" or len(meta_list) != 1
            or meta_list[0] is None or len(batch.uids) != 1
            or batch._inputs_embeds is None):
        return
    if _ckpt_active(batch.model, mode, int(manager.block_size)):
        return                          # ckpt tier owns these models
    meta = meta_list[0]
    if len(meta.get("full_input_ids") or ()) < 2:
        return
    _exact_anchor_arm(batch, meta, int(meta.get("checkpoint_len") or 0),
                      int(meta.get("prefix_len") or 0))
    # Retirement stash, independent of the anchor outcome: exact-tier
    # rows retire their full post-decode row at filter (the per-turn
    # store the post-prefill exact store cannot cover), warm rows
    # included -- the decode cache holds the full sequence either way.
    if not _SPEC_APC_RETIRE_DISABLED and batch.prompt_cache:
        from gmlx.cache.retire_key import lookup_render_ctx
        ids_list = [int(t) for t in meta["full_input_ids"]]
        batch.prompt_cache[0]._kq_apc_retire = {
            "full_ids": ids_list,
            "extra_hash": int(meta.get("extra_hash", 0)),
            "mode": "exact",
            "manager": manager,
            "render_ctx": lookup_render_ctx(ids_list),
            "gen": [],
        }


_ANCHOR_PICK_FLAG = "_kq_exact_anchor_pick"


def _install_exact_anchor_pick() -> None:
    """Consult the anchor LRU inside the stock admission pick.

    The pick is where a warm prefix belongs: admission builds the batch
    from it (suffix rows, right padding, warm-cache merge) and every
    downstream path treats an anchor restore exactly like a stock exact
    one. The anchor wins only when strictly longer than the stock pick,
    so it never shortens a restore. Idempotent.
    """
    from mlx_vlm.generate.ar import BatchGenerator
    if getattr(BatchGenerator._apc_pick_for, _ANCHOR_PICK_FLAG, False):
        return
    _orig = BatchGenerator._apc_pick_for

    def _pick_with_anchor(self, sequence):
        pick = _orig(self, sequence)
        try:
            if _SPEC_APC_DISABLED or getattr(self, "apc_mode", None) != "exact":
                return pick
            manager = getattr(self, "apc_manager", None)
            if manager is None or _ckpt_active(
                    getattr(self, "model", None), "exact",
                    int(manager.block_size)):
                return pick
            _uid, ids_list, _mt, prompt_kwargs, _lps, _crit = sequence
            if not ids_list or len(ids_list) < 2:
                return pick
            # Floor trivial exact picks: a sub-block restore (a bare-BOS
            # match off an unrelated request) saves nothing but suffix-
            # constructs the batch, knocking the spec path's ids out of
            # render space (anchor + retirement keys). Real warm picks are
            # thousands of tokens and pass untouched.
            if (pick is not None and not pick.get("matched_blocks")
                    and 0 < int(pick.get("prefix_len") or 0)
                    < int(manager.block_size)):
                pick = None
            have = int((pick or {}).get("prefix_len") or 0)
            extra_hash = self._apc_extra_hash(prompt_kwargs or {})
            floor = max(have, self._apc_safe_prefix_lookup_min(ids_list))
            from gmlx.cache.snapshot import anchor_exact_lookup
            warm, ap = anchor_exact_lookup(
                manager, ids_list, extra_hash=extra_hash,
                min_prefix_tokens=floor)
            if warm is None or ap <= have or ap >= len(ids_list):
                return pick
            if not self._apc_suffix_is_text_only(ids_list, ap):
                return pick
            if pick and pick.get("matched_blocks"):
                manager.release(pick["matched_blocks"])
            _log.info("APC L1 hit: prefix=%d suffix=%d tier=anchor",
                      ap, len(ids_list) - ap)
            return {
                "matched_blocks": [],
                "warm_cache": warm,
                "prefix_len": ap,
                "extra_hash": extra_hash,
                "full_input_ids": list(ids_list),
            }
        except Exception:
            _log.warning("APC anchor pick failed; continuing",
                         exc_info=True)
            return pick

    _pick_with_anchor.__dict__[_ANCHOR_PICK_FLAG] = True
    BatchGenerator._apc_pick_for = _pick_with_anchor


_PLAIN_DECODE_FLAG = "_kq_ckpt_plain_decode"


def _retire_rows(gb) -> dict:
    """uid -> retire-stash registry on a generation batch.

    Stashes arm on the B=1 prompt batch's cache object (the only stable
    home before the decode batch exists); the first decode-side touch
    lifts them here so they survive ``extend`` rebuilding the cache
    objects at continuous-batch injection."""
    reg = getattr(gb, "_kq_apc_retire_rows", None)
    if reg is None:
        reg = {}
        gb._kq_apc_retire_rows = reg
    return reg


def _lift_cache_stash(gb) -> None:
    if not getattr(gb, "prompt_cache", None) or len(gb.uids) != 1:
        return
    stash = getattr(gb.prompt_cache[0], "_kq_apc_retire", None)
    if stash is not None:
        gb.prompt_cache[0]._kq_apc_retire = None
        _retire_rows(gb)[gb.uids[0]] = stash


def _plain_step_tick(gb, out) -> None:
    """Per-token accounting + snapshot tick for stock-path retire rows.

    Rows are tracked per uid so accounting survives ``extend`` merges.
    Runs per step, so a deterministic failure disables the hook for that
    row on first strike instead of emitting a traceback per token;
    dropping ``gen`` also quiets retirement (its offset check would skip
    anyway on a broken count). The decode-time snapshot ring stays B=1
    (its clones ride the live single-row caches); rows in a B>1 batch
    retire snapshot-free, under their verbatim key or an LCP cap the
    tier arm can serve without a ring."""
    try:
        _lift_cache_stash(gb)
        reg = getattr(gb, "_kq_apc_retire_rows", None)
    except Exception:
        _log.warning("APC plain decode hook failed; continuing",
                     exc_info=True)
        return
    if not reg:
        return
    # _step returns (tokens, lps, top_idx, top_lp); slot 0 is the flat
    # per-row token list.
    rows = out[0] if isinstance(out, tuple) else out
    if rows is None:
        return
    solo = len(gb.uids) == 1
    for i, uid in enumerate(gb.uids):
        stash = reg.get(uid)
        if stash is None or "gen" not in stash:
            continue
        tok = rows[i] if i < len(rows) else None
        if tok is None:
            continue                    # no emission for this row this tick
        try:
            if isinstance(tok, (list, tuple)):
                tok = tok[0]
            stash["gen"].append(int(tok))
            if solo and stash.get("mode") == "ckpt":
                from gmlx.cache.snapshot import decode_ckpt_tick
                decode_ckpt_tick(stash, gb.prompt_cache, stash["gen"])
        except Exception:
            stash.pop("gen", None)
            stash["snap_ok"] = False
            _log.warning("APC plain decode hook failed; disabled for "
                         "this request", exc_info=True)


def _plain_retire(stash: dict, prompt_cache: list) -> None:
    """Retire a finished stock-path row off a single-row cache list.

    Offset invariants mirror ``speculative._retire_b1``: the stock step
    loop forwards each token as it emits it, so a clean finish leaves
    ``offset == len(seq)`` (an abort between steps leaves the same).
    ``stash["mode"]`` picks the tier arm: "ckpt" stores blocks +
    sidecar, "exact" a whole-row snapshot (DeepSeek-V4-class pooling
    stacks).
    """
    try:
        manager = stash.get("manager")
        if manager is None:
            return
        gen = [int(t) for t in stash.get("gen") or ()]
        if not gen:
            return
        seq = [int(t) for t in stash["full_ids"]] + gen
        from gmlx.cache.snapshot import _cache_offset_max, retirement_store
        offset = _cache_offset_max(prompt_cache)
        if offset == len(seq) - 1:
            seq = seq[:-1]
        elif offset != len(seq):
            _log.info("APC retire skipped: cache offset %d != tokens %d",
                      offset, len(seq))
            return
        lcp = None
        if os.environ.get("GMLX_APC_RETIRE_LCP") != "0":
            from gmlx.cache.retire_key import next_turn_lcp
            lcp = next_turn_lcp(stash.get("render_ctx"), seq, gen)
        max_len = lcp if lcp is not None and lcp < len(seq) else None
        _log.info("APC retire: seq=%d ctx=%s lcp=%s cap=%s",
                  len(seq), stash.get("render_ctx") is not None,
                  lcp, max_len)
        ok = retirement_store(
            manager, stash.get("mode") or "ckpt", seq, prompt_cache,
            row=0,
            extra_hash=int(stash.get("extra_hash", 0)), max_len=max_len,
            decode_snaps=stash.get("snaps"))
        if ok:
            _log.info("APC retire store: tokens=%d", ok)
    except Exception:
        _log.warning("APC retire failed; continuing", exc_info=True)


def _install_plain_ckpt_decode() -> None:
    """Stock-path decode hooks for the retirement store (ckpt + exact).

    Token accounting rides ``_step``; retirement fires from ``filter``
    for every leaving row (finish or client abort). A lone row retires
    off its live single-row caches; a row leaving a B>1 batch is first
    extracted via ``row_snapshot`` (padding-trimmed clones with row-true
    offsets), so retirement survives concurrency instead of firing only
    when the batch happens to drain to one row. Stashes live in a
    uid-keyed registry lifted across ``extend`` (the seam that rebuilds
    cache objects at continuous-batch injection).
    GMLX_APC_RETIRE_BATCH=0 restores the lone-row-only v1 scope.
    Idempotent."""
    from mlx_vlm.generate.ar import GenerationBatch
    if getattr(GenerationBatch._step, _PLAIN_DECODE_FLAG, False):
        return
    _orig_step = GenerationBatch._step
    _orig_filter = GenerationBatch.filter
    _orig_extend = GenerationBatch.extend

    def _step_with_ckpt(self):
        out = _orig_step(self)
        _plain_step_tick(self, out)
        return out

    def _filter_with_ckpt(self, keep):
        try:
            _lift_cache_stash(self)
            reg = getattr(self, "_kq_apc_retire_rows", None)
            if reg and self.prompt_cache:
                keep_set = set(keep)
                solo = len(self.uids) == 1
                batched_ok = os.environ.get(
                    "GMLX_APC_RETIRE_BATCH") != "0"
                for i, uid in enumerate(self.uids):
                    if i in keep_set:
                        continue
                    stash = reg.pop(uid, None)
                    if stash is None:
                        continue
                    if solo:
                        _plain_retire(stash, self.prompt_cache)
                    elif batched_ok:
                        from gmlx.cache.snapshot import row_snapshot
                        rows = row_snapshot(self.prompt_cache, i)
                        if rows is None:
                            _log.info("APC retire skipped: row %d "
                                      "extract unavailable", i)
                        else:
                            _plain_retire(stash, rows)
        except Exception:
            _log.warning("APC plain retire hook failed; continuing",
                         exc_info=True)
        _orig_filter(self, keep)

    def _extend_with_ckpt(self, other):
        try:
            _lift_cache_stash(self)
            _lift_cache_stash(other)
            other_reg = getattr(other, "_kq_apc_retire_rows", None)
            if other_reg:
                _retire_rows(self).update(other_reg)
                other._kq_apc_retire_rows = {}
        except Exception:
            _log.warning("APC retire stash carry failed; continuing",
                         exc_info=True)
        _orig_extend(self, other)

    _step_with_ckpt.__dict__[_PLAIN_DECODE_FLAG] = True
    _filter_with_ckpt.__dict__[_PLAIN_DECODE_FLAG] = True
    _extend_with_ckpt.__dict__[_PLAIN_DECODE_FLAG] = True
    GenerationBatch._step = _step_with_ckpt
    GenerationBatch.filter = _filter_with_ckpt
    GenerationBatch.extend = _extend_with_ckpt


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
                "(owned-path APC requires single-request prefill)", b)
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
        full_ids = [int(t) for t in batch._mtp_full_input_ids[0].tolist()]
        from gmlx.cache.retire_key import lookup_render_ctx
        batch.prompt_cache[0]._kq_apc_retire = {
            "full_ids": full_ids,
            "extra_hash": int(meta.get("extra_hash", 0)),
            "mode": ("ckpt" if _ckpt_active(
                batch.model, mode, int(manager.block_size)) else mode),
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
    pad = mx.zeros((rows - arr.shape[0],) + tuple(arr.shape[1:]),
                   dtype=arr.dtype)
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
        # Stock (non-speculative) batches get the checkpoint tier here:
        # lookup, prefix trim, cursor arming, retirement stash.
        if getattr(self, "draft_kind", None) is None \
                and not _SPEC_APC_DISABLED:
            try:
                _plain_ckpt_init(self)
                _plain_anchor_init(self)
            except Exception:
                _log.warning("APC plain ckpt init failed; continuing "
                             "stock", exc_info=True)

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
            # Stock-path ckpt batches store the full prompt here, the
            # moment the MTP path stores it at rounds entry: prefill just
            # finished, the first token is out, its KV not yet appended.
            if getattr(self, "_kq_ckpt_armed", False) \
                    and getattr(self, "draft_kind", None) is None:
                try:
                    cache = getattr(result, "prompt_cache", None) or []
                    stash = getattr(cache[0], "_kq_apc_retire", None) \
                        if cache else None
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
                    _log.warning("APC plain post-prefill store failed; "
                                 "continuing", exc_info=True)
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
                int(full_ids.shape[1]), len(result.prompt_cache),
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
                and all(hasattr(c, "filter") or hasattr(type(c), "merge")
                        for c in self.prompt_cache)):
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
        lifted = type(c).merge([c])
        stamp = getattr(c, "_gmlx_cascade", None)
        if stamp is not None:
            lifted._gmlx_cascade = stamp
        return lifted

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
        # Every cache must be batch-liftable before the generator closes.
        # A quantized single-stream cache (KV_BITS MTP arm) has no merge;
        # decline and keep the drain-wait behavior, so the B>1 rebuild
        # goes through make_speculative_prompt_cache and its fp16 swap.
        if not all(
            (hasattr(c, "filter") and hasattr(c, "extend"))
            or hasattr(type(c), "merge")
            for c in self.prompt_cache
        ):
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
                    # The running generator froze max(max_tokens) at
                    # start; injected rows carry their own budgets.
                    "max_tokens": list(other.max_tokens),
                })

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

    from gmlx.cache.compat import cache_types

    plain_kv = cache_types("KVCache")
    rotating = (cache_types("RotatingKVCache")
                + cache_types("BatchRotatingKVCache"))
    _orig = _su.make_speculative_prompt_cache
    _noted = [False]
    _warned_batch = [False]
    _warned_rotating = [False]
    _warned_stock = [False]

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
            # Force fp16 batch KV. Under KV_BITS the stock builder returns
            # BatchQuantizedKVCache, which the stock rollback misfiles as
            # an SSM cache (not trimmable, no zero_row_tail): rejected
            # draft tokens are never trimmed and the state pairing shifts.
            # Walk into CacheList entries: to_batch_cache's recursive arm
            # quantizes nested subcaches too (and without the layer gate).
            from mlx_vlm.models.cache import BatchKVCache

            batch_quant = cache_types("BatchQuantizedKVCache")

            def _swap(c):
                if isinstance(c, batch_quant):
                    return BatchKVCache(left_padding), 1
                inner = getattr(c, "caches", None)
                if inner is None:
                    return c, 0
                subs = [_swap(s) for s in inner]
                n = sum(k for _, k in subs)
                if n:
                    c.caches = tuple(s for s, _ in subs)
                return c, n

            swapped = 0
            for e, c in enumerate(caches):
                caches[e], n_sw = _swap(c)
                swapped += n_sw
            if swapped and not _warned_batch[0]:
                _warned_batch[0] = True
                _log.warning(
                    "KV_BITS with MTP at batch size %d: packed batch "
                    "rollback is unsupported; %d layers run fp16 KV",
                    batch_size, swapped)
            return caches
        if any(isinstance(c, rotating) for c in caches):
            if not _warned_rotating[0]:
                _warned_rotating[0] = True
                _log.warning(
                    "KV_BITS dropped on the MTP path: sliding-window "
                    "cache stack cannot quantize")
            return caches
        if ("qwen3_5" in type(lm).__module__
                and not env_bool("GMLX_QWEN_OWNED", True)):
            # The bare-stock text fallback has no verify patches; its
            # verify fallback slices keys as raw arrays and crashes on
            # quantized tuples (issue #104 second symptom).
            if not _warned_stock[0]:
                _warned_stock[0] = True
                _log.warning(
                    "KV_BITS dropped on the MTP path: the GMLX_QWEN_OWNED=0 "
                    "stock fallback cannot verify on a quantized KV cache")
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
