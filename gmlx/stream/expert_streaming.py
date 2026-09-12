"""MoE expert streaming: CPU offload install, GPU residency, prefill step."""
from __future__ import annotations

import os
import random
import time

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from gmlx.envflags import env_bool, env_int
from gmlx.gen.prefill_decay import deduct_untracked_weights
from gmlx.load import loadlog
from gmlx.load.loader import (
    _PHASE,
    _STREAM_GPU_TOKENS_DEFAULT,
    _STREAM_PREFETCH_MIN_TOKENS,
    _STREAMING_PREFILL_STEP,
    _STREAMING_PREFILL_STEP_BY_MODEL_TYPE,
    _arena_split_max_tokens,
    _arena_stage_max_tokens,
    _kq_expert_gpu_ok,
    _lookahead_default,
    _phase_token,
    _resolve_feeder_defaults,
    _stream_gpu_tokens,
    _switch_num_experts,
    moe_streaming_active,
)

from .budget import _decode_arena_bytes, _prefill_ring_reason, _ram_floor_bytes
from .wired_limit import _neutralize_wired_limit_sweep, configure_cpu_device


# MoE expert CPU offload (hybrid GPU+CPU inference)
#
# On unified memory the GPU constraint is the wired limit, not a separate
# VRAM pool: Metal-resident buffers must be wired, while CPU-consumed mmap
# pages ride the page cache (evictable, can exceed RAM). For fine-grained MoE
# the routed expert stacks are ~90-95% of the bytes but each expert is read
# with probability top_k/n_experts per token, while the every-token layers
# (attention, norms, routers, shared experts, embeddings, KV cache) are read
# every token.
# Running the SwitchGLU expert containers on the CPU stream therefore keeps
# those hot layers + KV on GPU while the expert wire bytes stay file-backed in
# the page cache when the GPU is idle, and the kquant gather op executes on
# its threaded CPU path. MLX's cross-stream dependency tracking handles the
# GPU->CPU->GPU handoff inside each MoE layer (zero-copy - same pages).
#
# Residency (measured): Metal wires only what GPU work references or what
# sits in MLX's residency set - unreferenced file-backed buffers stay
# evictable page cache even under full memory pressure. The one hazard is
# mlx-lm's generation-time wired-limit bump (see
# _neutralize_wired_limit_sweep): MLX services a raised wired limit by
# sweeping every live buffer into the residency set, offloaded experts
# included. Models larger than the wired budget therefore run in streaming
# mode: the sweep is neutralized and GPU prefill routing is forced off, so
# the GPU never references (and never wires) expert bytes, and the page
# cache streams them from disk.
#
# Prefill staging: at decode each expert sees ~top_k/n_experts of one token,
# but a prefill chunk makes every expert hot with tens of rows each - a GEMM
# workload where the CPU (~1.5 TFLOP/s) is the wrong device. Calls with at
# least GMLX_STREAM_GPU_TOKENS tokens therefore run on the default (GPU)
# stream against the same zero-copy buffers - no copies, no staging; the
# driver wires the touched expert bytes for the duration of the work and
# releases them when the GPU goes idle. Threshold 0 disables GPU routing
# (pure CPU experts, the conservative choice when the model is far larger
# than RAM and prefill-wiring every expert is undesirable).
#
# Cost model (measured): offloaded decode pays a per-layer surcharge of
# genuine CPU dot compute plus per-layer stream fences and CPU-pool
# wake-from-idle (3 wakes per layer, one per gather; the wake cost grows
# when the pool sits idle between layers while the GPU runs the
# every-token layers).
# Routing every call to the GPU stream instead (GMLX_STREAM_GPU_TOKENS=1,
# no CPU hop) runs ~3.7x faster on a fits-in-RAM MoE, so in-RAM the CPU
# offload is for the over-budget regime, not the fast path.

_CPU_OFFLOAD_CLASS_CACHE: dict = {}


def configure_stream_cpu(
    model,
    gguf_path: str | None = None,
    feeder_prefill: bool | None = None,
    feeder_decode: bool | None = None,
):
    """Whole-model CPU streaming (``--stream-cpu``): run the model on the CPU
    device with the streaming-expert machinery always engaged.

    ``--stream-cpu`` is an explicit opt-in into the CPU/over-RAM path, so it
    forces streaming (``force_stream=True``) regardless of model size - experts
    run on the CPU stream whether or not the model fits the wired budget (a
    fits-in-RAM model is then served from the page cache rather than faulting
    from disk; for the faster all-GPU path on a model that fits, omit
    ``--stream-cpu``). The GPU
    working-set budget is still captured before switching the default device to
    CPU so the over-/under-budget log line stays accurate (the CPU device would
    otherwise report a budget that hides the condition). Returns
    ``(n_wrapped, offloaded_bytes)``.
    """
    try:
        gpu_info = dict(mx.device_info())
    except Exception:
        gpu_info = None
    configure_cpu_device()
    if gpu_info and "max_recommended_working_set_size" in gpu_info:
        mx.device_info = lambda: gpu_info
    return install_expert_streaming(
        model,
        gguf_path=gguf_path,
        force_stream=True,
        feeder_prefill=feeder_prefill,
        feeder_decode=feeder_decode,
    )


def _install_gpu_residency(model, moe_modules, *,
                           skip_ids=frozenset(),
                           include_expert_stacks: bool = False) -> None:
    """Wire every non-expert weight buffer into the Metal residency set,
    so command buffers stop re-wiring the every-token weights' pages on
    every use (the
    per-use wiring is what an unswept streaming install pays instead of
    the neutralized wire-everything sweep).

    ``skip_ids``: arrays that must NOT be inserted - streamed lookup
    tables (a residency insert wires the buffer as surely as a GPU op).
    ``include_expert_stacks``: table-only streaming keeps the experts
    resident, so the GB-scale-stack belt is lifted and they are wired
    with everything else."""
    import mlx_kquant as kq

    if not getattr(kq, "residency_insert", None):
        print("[stream] gpu-resident weights unavailable "
              "(mlx-kquant lacks residency ops)")
        return
    skip = set(skip_ids)
    for mods in moe_modules.values():
        for m in mods:
            for attr in ("gate_proj", "up_proj", "down_proj"):
                w = getattr(getattr(m, attr, None), "weight", None)
                if w is not None:
                    skip.add(id(w))
    inserted = []
    nbytes = 0
    for _, a in tree_flatten(model.parameters()):
        if id(a) in skip:
            continue
        if (not include_expert_stacks
                and a.ndim == 3 and a.nbytes > (1 << 30)):
            continue  # belt: any GB-scale stack is an expert container
        if kq.residency_insert(a):
            inserted.append(a)
            nbytes += a.nbytes
    kq.residency_commit()
    model._kq_resident_arrays = inserted
    n = len(inserted)
    print(f"[stream] gpu-resident weights: {n} buffers "
          f"({nbytes / 1e9:.1f} GB) in the Metal residency set "
          "(GMLX_GPU_RESIDENT=0 disables)")


def install_expert_streaming(
    model,
    n_layers: int | None = None,
    gguf_path: str | None = None,
    force_stream: bool = False,
    feeder_prefill: bool | None = None,
    feeder_decode: bool | None = None,
    stats_verbose: bool | None = None,
):
    """Run routed-expert stacks (SwitchGLU) on the CPU stream.

    Wraps each ``SwitchGLU`` in the first ``n_layers`` decoder layers (all
    layers when None) so its forward - the expert gather matmuls - executes
    under ``mx.stream(mx.cpu)`` at decode shapes, and on the default (GPU)
    stream for prefill-sized calls (see the staging note above). Per-instance
    ``__class__`` swap; routers, shared experts, attention, and the KV cache
    stay on the default (GPU) stream. Returns ``(n_wrapped, offloaded_bytes)``.

    ``gguf_path`` (the loaded checkpoint) enables sequential expert prefetch
    for streaming-mode models - see ``gmlx.stream.prefetch``. Without it,
    over-budget prefill demand-faults expert bytes at random-read bandwidth.
    """
    from gmlx.load.modules import switch_layer_types

    _, glu_types = switch_layer_types()

    layers = getattr(model, "layers", None)
    if layers is None:
        layers = model.model.layers

    # Streaming mode: neutralize the generation-time residency sweep (which
    # would otherwise wire the whole model - see _neutralize_wired_limit_sweep)
    # and run every expert call on the CPU stream. Engaged when the model is
    # over the wired budget (it must stream) or when force_stream is set:
    # --stream-cpu (configure_stream_cpu) passes force_stream so the flag does
    # what it says - experts on CPU regardless of model size; on a fits-in-RAM
    # model the page cache then serves those bytes from RAM rather than faulting
    # from disk. --stream-experts keeps the budget-keyed decision, so below the
    # budget it still routes prefill-sized calls to the GPU stream
    # (GMLX_STREAM_GPU_TOKENS) - the fast path in-RAM.
    params = getattr(model, "parameters", None)
    total_bytes = sum(a.nbytes for _, a in tree_flatten(params())) if params else 0
    try:
        budget = int(0.9 * mx.device_info()["max_recommended_working_set_size"])
    except Exception:
        budget = None
    over_budget = budget is not None and total_bytes > budget

    # Selection ladder step 1 (docs/streaming.md): archs with a
    # declared streamable lookup table (e.g. qwen4exp's 26.8 GiB PLE
    # n-gram table) stream it instead of the experts when it alone brings
    # the resident set under budget - table gathers touch ~1.4 KB/token
    # against the experts' every-MoE-layer surcharge. The table wrap runs
    # its row gather on a dedicated CPU stream so the buffer is never a
    # GPU-stream input (a single GPU reference would wire all of it).
    # When the post-table estimate is still over budget, v1 falls back to
    # expert streaming with the table resident: streaming both at once
    # (compose) needs the hot-row arena and is not shipped.
    table_offloaded = 0
    if not force_stream:
        from gmlx.stream.table_stream import (
            install_table_streaming,
            table_bytes,
            table_stream_selected,
        )

        compose = False
        if table_stream_selected(model, total_bytes, budget):
            post = total_bytes - table_bytes(model)
            # Step 2 default: over budget even post-table streams both
            # (compose). GMLX_STREAM_PLE_COMPOSE=0 keeps the table
            # resident; the selection test already honors it in auto
            # mode, so this only fires under GMLX_STREAM_PLE=1.
            if budget is not None and post > budget:
                if env_bool("GMLX_STREAM_PLE_COMPOSE", True):
                    compose = True
                    table_offloaded, table_names = (
                        install_table_streaming(model))
                else:
                    print(
                        "[stream] table stays resident "
                        "(GMLX_STREAM_PLE_COMPOSE=0); experts stream"
                    )
            else:
                table_offloaded, table_names = install_table_streaming(model)
        if table_offloaded and compose:
            loadlog.info(
                f"[stream] compose: streamable table "
                f"{'+'.join(table_names)} "
                f"({table_offloaded / 2**30:.1f} GiB) on the CPU stream "
                "AND experts streamed"
            )
            key = getattr(model, "_kq_weights_key", None)
            from gmlx.gen.prefill_decay import (
                note_streamed_tracked_bytes,
                untracked_weight_bytes_for,
            )
            tracked = max(
                0.0, total_bytes - untracked_weight_bytes_for(key))
            credit = min(float(table_offloaded), tracked)
            if credit > 0:
                note_streamed_tracked_bytes(
                    credit, key, source="table", cap=tracked)
            deduct_untracked_weights(table_offloaded, key)
        elif table_offloaded:
            # The selection test admits the table only when the remainder
            # clears the budget (or streaming is forced on a fits model),
            # so experts are resident from here on.
            over_budget = False
            base = ("" if budget is None
                    else f" of {budget / 2**30:.1f} GiB budget")
            loadlog.info(
                f"[stream] streamable table {'+'.join(table_names)} "
                f"({table_offloaded / 2**30:.1f} GiB) stays file-backed on "
                "the CPU stream; experts resident (post-deduction "
                f"{(total_bytes - table_offloaded) / 2**30:.1f} GiB{base})"
            )
            key = getattr(model, "_kq_weights_key", None)
            from gmlx.gen.prefill_decay import (
                note_streamed_tracked_bytes,
                untracked_weight_bytes_for,
            )
            tracked = max(
                0.0, total_bytes - untracked_weight_bytes_for(key))
            credit = min(float(table_offloaded), tracked)
            if credit > 0:
                note_streamed_tracked_bytes(
                    credit, key, source="table", cap=tracked)
            deduct_untracked_weights(table_offloaded, key)

    streaming = force_stream or over_budget
    prefetcher = None
    cast_dead_bytes = 0
    held_wired = 0
    if streaming:
        _neutralize_wired_limit_sweep()
        # Reclaim the wired bytes of released streaming models (feeder and
        # MoE modules reference each other, so unwiring waits for a
        # collection), then charge what is still held against this weight
        # pin. Wired pages are invisible to jetsam, so two pins that each
        # size against the whole machine wire it solid. The arena needs no
        # charge; see _decode_arena_bytes.
        from gmlx.stream import installs as _installs

        freed = _installs.reclaim_dead()
        held_wired = _installs.live_wired_bytes()
        if freed:
            loadlog.info(
                f"[stream] reclaimed {freed / 1e9:.1f} GB wired from a "
                "released streaming model")
        if held_wired:
            print(
                f"[stream] another live streaming install holds "
                f"{held_wired / 1e9:.1f} GB wired; this model sizes against "
                "what is left (release the other model first for the full "
                "budget)")
        from gmlx.stream.prefetch import maybe_make_prefetcher

        prefetcher = maybe_make_prefetcher(gguf_path)
        if prefetcher is not None:
            object.__setattr__(model, "_kq_prefetcher", prefetcher)
        # Wire the every-token weights before the decode feeder sizes its
        # arena: pinned every-token pages come out of the same wired budget.
        from gmlx.stream.pin_weights import cast_copies, maybe_pin_weights
        from gmlx.stream.table_stream import streamable_tables_for

        # Declared streamable components never enter the pin set, streamed
        # or resident: mlocking them starves the expert page cache. Cast
        # tensors go too - their wire bytes have no view left to keep.
        casts = cast_copies(model, gguf_path)
        cast_dead_bytes = casts.dead_bytes
        pin_exclude = frozenset(
            t.gguf_name for t, _ in streamable_tables_for(model)
        ) | casts.names
        weights_pin = maybe_pin_weights(
            gguf_path, exclude_names=pin_exclude, reserved_bytes=held_wired)
        if weights_pin is not None:
            object.__setattr__(model, "_kq_weights_pin", weights_pin)
            _installs.record(model, weights_pin.pinned_bytes)

    def _wrapped_class(cls):
        sub = _CPU_OFFLOAD_CLASS_CACHE.get(cls)
        if sub is None:
            # A fused base consumes routing scores itself (mix seam); a
            # stock base (unrecognized activation, e.g. minimax-m3's
            # SwiGLUOAI) takes (x, indices) only. The wrapper still
            # advertises _kq_scores_sink so blocks hand scores over for
            # miss-shed; it strips them before forwarding and applies
            # the shed mix python-side.
            _fwd_scores = bool(getattr(cls, "_kq_mix_scores", False))

            class _CPUOffload(cls):
                _kq_scores_sink = True

                def __call__(self, x, indices, *args, **kwargs):
                    # Extra args pass through untouched (e.g. deepseek-v4
                    # hands the fused SwitchGLU its routing scores). A base
                    # without the mix seam takes (x, indices) only: keep the
                    # scores for the miss-shed hook and strip them from what
                    # gets forwarded.
                    scores_arg = args[0] if args else None
                    if args and not _fwd_scores:
                        args = args[1:]
                    # Threshold read per call (cheap; once per MoE layer per
                    # forward) so env changes A/B without a reload. Streaming
                    # mode pins everything to CPU: a GPU expert call would
                    # wire the buffers it references, which an over-budget
                    # model cannot afford.
                    cpu_only = getattr(self, "_kq_cpu_only", False)
                    gpu_tokens = _stream_gpu_tokens(
                        getattr(
                            self, "_kq_gpu_tokens_default", _STREAM_GPU_TOKENS_DEFAULT
                        )
                    )
                    n_tokens = indices.size // indices.shape[-1]
                    pf = getattr(self, "_kq_prefetcher", None)
                    fdr = getattr(self, "_kq_feeder", None)
                    dfr = getattr(self, "_kq_decode_feeder", None)
                    small = n_tokens <= _arena_stage_max_tokens()
                    la = getattr(self, "_kq_lookahead", None)
                    la_pred = None
                    ph = _PHASE
                    if ph is not None:
                        _phase_token(
                            ph, getattr(self, "_kq_li", None), n_tokens)
                        if n_tokens != 1:
                            ph = None
                    lsp = getattr(self, "_kq_layer_shed", None)
                    if lsp is not None and cpu_only and n_tokens == 1:
                        rng = getattr(self, "_kq_shed_rng", None)
                        if rng is None:
                            # per-layer seed: reproducible shed pattern
                            rng = random.Random(
                                0x5EED ^ (getattr(self, "_kq_li", 0) or 0))
                            object.__setattr__(self, "_kq_shed_rng", rng)
                        if rng.random() < lsp:
                            # Skip the routed path entirely (gather, stage
                            # and this layer's eval fence). The unmixed
                            # zeros return makes the block mix nothing and
                            # still add its shared expert.
                            if dfr is not None:
                                dfr._layer_shed_n += 1
                            return mx.zeros(
                                (*x.shape[:-1], indices.shape[-1],
                                 x.shape[-1]), dtype=x.dtype)
                    gt = getattr(self, "_kq_gpu_token", None)
                    gt_live = (
                        gt is not None
                        and gt._route_shed is not None
                        and cpu_only
                        and n_tokens == 1
                        and dfr is not None
                        and dfr.covers(self._kq_li)
                    )
                    if gt_live:
                        dfr.ensure_wired()
                        # Token tick for EVERY covered decode layer, stage
                        # path included: boundary detection and the
                        # adaptive hot-set refresh live here.
                        gt.on_layer_entry(
                            self._kq_li,
                            None if getattr(self, "_kq_in_split", False)
                            else getattr(self, "_kq_miss_shed", None))
                    if (
                        gt_live
                        and scores_arg is not None
                        and not dfr.wedged_at(self._kq_li)
                        and gt.layer_autonomous(self._kq_li)
                    ):
                        # GPU-autonomous layer (gpu-dispatch Tier 2): no
                        # per-layer eval. route_shed remaps ids to arena
                        # slots and sheds non-resident experts on the GPU;
                        # the graph flushes at the next stage-path layer's
                        # eval or the logits, and the host consumes the
                        # recorded misses at the token boundary
                        # (popularity + prestage + fresh slot tables) - see
                        # gpu_token.py for the fence argument. In adaptive
                        # mode only layers with a measured hit rate above
                        # GMLX_AUTO_HOT_HIT run here, so the shed cost per
                        # layer is near zero.
                        tbl = gt.table(self._kq_li)
                        self._kq_cpu_only = False
                        try:
                            with dfr.swapped(self._kq_li):
                                with mx.stream(mx.gpu):
                                    sc_f32 = scores_arg.astype(mx.float32)
                                    slots, mix, m_ids, m_sc = (
                                        gt._route_shed(
                                            indices.astype(mx.uint32),
                                            sc_f32, tbl))
                                    mix_c = mix.astype(x.dtype)
                                    if _fwd_scores:
                                        y = super().__call__(
                                            x, slots, mix_c,
                                            *args[1:], **kwargs)
                                    else:
                                        y = super().__call__(
                                            x, slots, *args, **kwargs)
                                        if y.ndim == x.ndim + 1:
                                            y = (y * mix_c[..., None]).sum(
                                                axis=-2)
                                    gt.record(
                                        self._kq_li, indices, sc_f32,
                                        m_ids, m_sc, y)
                            return y
                        finally:
                            self._kq_cpu_only = True
                    if la is not None and cpu_only and n_tokens == 1:
                        # Decode only: prefill prestage would fault the
                        # cold arena while the ring holds the wired budget.
                        # Lookahead: run the NEXT MoE layer's router on this
                        # layer's input and evaluate it together with the
                        # router read below (one sync either way). The
                        # prediction feeds nothing downstream - it only
                        # records recall (probe) or drives prestage reads.
                        # Latent-MoE blocks (kimi-k3) hand the full-width
                        # router input over out of band; x here is the
                        # expert container's latent-width input.
                        x_la = getattr(self, "_kq_la_input", None)
                        if x_la is None:
                            x_la = x
                        if ph is not None:
                            t_la = time.perf_counter()
                            la_pred = la.on_call(x_la, indices)
                            ph["la"] += time.perf_counter() - t_la
                        else:
                            la_pred = la.on_call(x_la, indices)
                    if (
                        dfr is not None
                        and cpu_only
                        and small
                        and dfr.covers(self._kq_li)
                    ):
                        # Decode feeder: the routed experts are served from
                        # this layer's wired GPU arena; misses are pread from
                        # the GGUF into evicted slots first. Small prefill
                        # chunks take this path too when their routed set
                        # fits - the arena persists across requests, which is
                        # what makes repeat short-prompt TTFT cheap. The eval
                        # is both the router read and the arena-overwrite
                        # safety fence (see decode_feeder.py). ``stage``
                        # returns None when the call routes to more distinct
                        # experts than the arena has slots - fall through.
                        t0 = time.perf_counter() if ph is not None else 0.0
                        if n_tokens == 1:
                            dfr.ensure_wired()
                        # Miss-shed is decode-only: a single-token leaf of an
                        # arena token split is prefill work, and a shedding
                        # leaf would return a mixed rank-3 output next to a
                        # clean leaf's per-expert rank-4 - the reassembly
                        # concatenate cannot take both.
                        ms = (None if getattr(self, "_kq_in_split", False)
                              else getattr(self, "_kq_miss_shed", None))
                        sc_f32 = None
                        if (ms is not None and scores_arg is not None
                                and n_tokens == 1):
                            # Shed reads the scores host-side; fold them into
                            # the router eval so the hook adds a small D2H
                            # copy, not a second per-layer graph flush.
                            sc_f32 = scores_arg.astype(mx.float32)
                            mx.eval(indices, sc_f32)
                        else:
                            mx.eval(indices)
                        if ph is not None:
                            t1 = time.perf_counter()
                            ph["ev"] += t1 - t0
                            wait0 = getattr(dfr, "_t_demand", 0.0)
                        ids = np.array(indices)
                        shed_args = None
                        shed_mix = None
                        if sc_f32 is not None:
                            sc = np.asarray(sc_f32).reshape(-1)
                            keep = dfr.shed_misses(
                                self._kq_li, ids.reshape(-1), sc, ms)
                            if keep is not None:
                                # Arena-path only: the overflow fallback
                                # below keeps the original routed set.
                                kept = ids.reshape(-1)[keep]
                                shp = ids.shape[:-1] + (kept.size,)
                                ids = np.ascontiguousarray(kept.reshape(shp))
                                scn = sc[keep]
                                # survivors keep the token's full mass
                                scn = scn * (sc.sum() / max(scn.sum(), 1e-20))
                                sc_mx = mx.array(scn.reshape(shp)).astype(
                                    scores_arg.dtype)
                                if _fwd_scores:
                                    shed_args = (sc_mx,) + args[1:]
                                else:
                                    # Stock base returns per-expert outputs;
                                    # the block's weights still cover the
                                    # full routed set, so mix the shed
                                    # survivors here instead.
                                    shed_mix = sc_mx
                        slots = dfr.stage(self._kq_li, ids)
                        if ph is not None:
                            t2 = time.perf_counter()
                            w = getattr(dfr, "_t_demand", 0.0) - wait0
                            ph["stage_wait"] += w
                            ph["stage_book"] += (t2 - t1) - w
                        if la_pred:
                            # This layer's demand misses have joined
                            # (stage returned); the predicted layers'
                            # misses now read in the background while this
                            # layer's gather and the next layers' every-token
                            # work compute - speculation never competes with
                            # demand traffic for the SSD.
                            la_keep = (
                                ms if getattr(
                                    self, "_kq_prestage_keepers", False)
                                else None)
                            for _dst, (_ids, _sc) in la_pred.items():
                                if la_keep is not None:
                                    dfr.prestage(
                                        _dst, _ids, keep_mass=la_keep,
                                        pred_scores=_sc)
                                else:
                                    dfr.prestage(_dst, _ids)
                            if ph is not None:
                                ph["prestage"] += time.perf_counter() - t2
                        if slots is not None:
                            # arena call: weights are wired GPU views for
                            # this scope, so lift the streaming CPU pin
                            # and let the fused kq kernels run
                            if shed_args is not None:
                                args = shed_args
                            self._kq_cpu_only = False
                            try:
                                t3 = (time.perf_counter()
                                      if ph is not None else 0.0)
                                with dfr.swapped(self._kq_li):
                                    with mx.stream(mx.gpu):
                                        y = super().__call__(
                                            x, mx.array(slots),
                                            *args, **kwargs)
                                        if (shed_mix is not None
                                                and y.ndim == x.ndim + 1):
                                            y = (y * shed_mix[..., None]).sum(
                                                axis=-2)
                                if ph is not None:
                                    ph["build"] += time.perf_counter() - t3
                                return y
                            finally:
                                self._kq_cpu_only = True
                    if (
                        dfr is not None
                        and cpu_only
                        and 1 < n_tokens <= _arena_split_max_tokens()
                        and not kwargs
                        and dfr.covers(self._kq_li)
                        and dfr.can_stage_smaller(self._kq_li)
                    ):
                        # The chunk routes more distinct experts than the
                        # arena has slots (stage refused above, or the chunk
                        # is over the stage-size gate and was never tried).
                        # Halve along the token axis and recurse: pieces
                        # whose routed union fits are served from the wired
                        # arena's read pool, so a turn-transition prefill or
                        # a wide verify batch never drops to the CPU
                        # page-cache gather. Bottoms out at n_tokens == 1,
                        # which always takes a non-split path.
                        ax = x.ndim - 2
                        orig = ((scores_arg,) + args
                                if scores_arg is not None and not _fwd_scores
                                else args)
                        sliceable = (
                            x.shape[ax] == n_tokens
                            and indices.ndim == x.ndim
                            and all(
                                isinstance(a, mx.array)
                                and a.ndim == x.ndim
                                and a.shape[ax] == n_tokens
                                for a in orig)
                        )
                        if sliceable:
                            half = n_tokens // 2
                            parts = []
                            prev_split = getattr(
                                self, "_kq_in_split", False)
                            object.__setattr__(
                                self, "_kq_in_split", True)
                            try:
                                for sl in (slice(0, half),
                                           slice(half, n_tokens)):
                                    t = tuple(
                                        [slice(None)] * ax + [sl])
                                    parts.append(self.__call__(
                                        x[t], indices[t],
                                        *[a[t] for a in orig]))
                                    # The pieces share one precomputed
                                    # routing. Thus the stage-time eval of a
                                    # later piece's indices does not wait for
                                    # an earlier piece's gather. Staging
                                    # could overwrite (or resize away) arena
                                    # slots that the unexecuted gather
                                    # references. Execute each piece before
                                    # the next piece stages.
                                    mx.eval(parts[-1])
                            finally:
                                object.__setattr__(
                                    self, "_kq_in_split", prev_split)
                            return mx.concatenate(parts, axis=ax)
                    wedged = dfr is not None and dfr.wedged_at(self._kq_li)
                    if wedged and dfr.has_dead(self._kq_li):
                        # A wedged read poisoned part of this layer's file
                        # range: no fallback below (mmap gather, advisory
                        # prefetch, prefill staging) may touch a dead
                        # expert's bytes - rewrite the routing ids first.
                        mx.eval(indices)
                        indices = mx.array(dfr.redirect_dead(
                            self._kq_li, np.array(indices)))
                    if (
                        fdr is not None
                        and not wedged
                        and small
                        and n_tokens >= _STREAM_PREFETCH_MIN_TOKENS
                        and fdr.covers(self._kq_li)
                    ):
                        # Router-aware partial staging: a short chunk routes
                        # to a fraction of the experts, so stage only those
                        # slices into the ring slot instead of the whole
                        # layer (see feeder.prefill_partial_call).
                        mx.eval(indices)
                        ids = np.unique(np.array(indices)).tolist()
                        with fdr.prefill_partial_call(self, self._kq_li, ids):
                            with mx.stream(mx.gpu):
                                return super().__call__(
                                    x, indices, *args, **kwargs)
                    if (
                        fdr is not None
                        and not wedged
                        and n_tokens >= _STREAM_PREFETCH_MIN_TOKENS
                        and fdr.covers(self._kq_li)
                    ):
                        # Feeder prefill: this layer's expert stacks are
                        # staged straight from the GGUF into GPU-visible
                        # ring slots and the GEMM runs on the GPU stream
                        # from the slot - the page cache never sees the
                        # bytes. The eval is the ring protocol's slot-free
                        # proof (previous layer's compute has finished);
                        # see feeder.py. Wedged layers skip this (and the
                        # whole-layer advisory below): both sweep the full
                        # expert range, poisoned bytes included.
                        mx.eval(x)
                        with fdr.prefill_call(self, self._kq_li):
                            with mx.stream(mx.gpu):
                                return super().__call__(
                                    x, indices, *args, **kwargs)
                    if (
                        pf is not None
                        and not wedged
                        and pf.enabled
                        and n_tokens >= _STREAM_PREFETCH_MIN_TOKENS
                    ):
                        # Streaming prefill: materialize the lazy graph up to
                        # this layer so the advisory window advances at
                        # execution pace. Build-time would fire every layer's
                        # advisory at once, and an over-RAM advisory storm
                        # evicts its own earlier reads.
                        mx.eval(x)
                        pf.on_layer(self._kq_li)
                    elif (
                        pf is not None
                        and pf.enabled
                        and cpu_only
                        and env_bool("GMLX_DECODE_PREFETCH", True)
                    ):
                        # Streaming decode: the router's top-k is tiny and
                        # the gather needs it anyway - evaluate it now and
                        # pull the selected experts' slices into the page
                        # cache at queue depth (on_decode) instead of
                        # demand-faulting 16 KB clusters from inside the
                        # gemv. GMLX_DECODE_PREFETCH=0 disables.
                        mx.eval(indices)
                        pf.on_decode(
                            self._kq_li,
                            np.unique(np.array(indices)).tolist(),
                        )
                    if gpu_tokens > 0 and n_tokens >= gpu_tokens and not cpu_only:
                        # Prefill regime: GEMM on the GPU stream, same
                        # zero-copy buffers.
                        return super().__call__(x, indices, *args, **kwargs)
                    with mx.stream(mx.cpu):
                        return super().__call__(x, indices, *args, **kwargs)

            _CPUOffload.__name__ = cls.__name__ + "_CPUOffload"
            _CPU_OFFLOAD_CLASS_CACHE[cls] = sub = _CPUOffload
        return sub

    n_wrapped = 0
    offloaded = 0
    n_cpu_only_codec = 0
    moe_modules: dict[int, list] = {}
    for li, layer in enumerate(layers):
        if n_layers is not None and li >= n_layers:
            break
        for m in layer.modules():
            if not isinstance(m, glu_types):
                continue
            if m.__class__ in _CPU_OFFLOAD_CLASS_CACHE.values():
                continue  # already wrapped (idempotent)
            gpu_ok = _kq_expert_gpu_ok(m)
            if not gpu_ok:
                n_cpu_only_codec += 1
            m.__class__ = _wrapped_class(m.__class__)
            if streaming:
                m._kq_cpu_only = True
                object.__setattr__(m, "_kq_li", li)
                if gpu_ok:
                    moe_modules.setdefault(li, []).append(m)
                if prefetcher is not None:
                    object.__setattr__(m, "_kq_prefetcher", prefetcher)
            elif gpu_ok:
                # All-GPU auto-policy: in-RAM, the residency sweep wires the
                # whole model regardless of where expert calls run, so the
                # CPU hop has no memory benefit and a large decode cost
                # (measured ~4-5x). Route every call to the GPU stream; an
                # explicit GMLX_STREAM_GPU_TOKENS (e.g. 0) overrides.
                object.__setattr__(m, "_kq_gpu_tokens_default", 1)
            else:
                m._kq_cpu_only = True
            offloaded += sum(a.nbytes for _, a in tree_flatten(m.parameters()))
            n_wrapped += 1
    if over_budget and offloaded:
        # Streamed expert bytes are page cache, never wired, and must not
        # tax headroom_bytes() or the admission gate and request preflight
        # starve every request. Which side of the accounting they sit on
        # depends on how the load materialized them: registered untracked
        # (zero-copy walk, small tracked delta) they need deducting; but a
        # load whose arrays landed allocator-tracked (untracked registered
        # ~0) has them inside mx.get_active_memory instead, and headroom
        # needs the add-back credit. tracked = total - untracked splits
        # the two regimes; the credit is clamped to the expert share.
        # Same 0.9 x working-set budget test as _warm_touch_pass.
        key = getattr(model, "_kq_weights_key", None)
        from gmlx.gen.prefill_decay import (
            note_streamed_tracked_bytes,
            untracked_weight_bytes_for,
        )
        tracked = max(0.0, total_bytes - untracked_weight_bytes_for(key))
        credit = min(float(offloaded), tracked)
        if credit > 0:
            note_streamed_tracked_bytes(
                credit, key, source="experts", cap=tracked)
            print(
                f"[stream] headroom credits {credit / 1e9:.1f} GB of "
                "allocator-tracked expert bytes as reclaimable page cache"
            )
        deduct_untracked_weights(offloaded, key)
    if n_cpu_only_codec:
        print(
            f"[stream] {n_cpu_only_codec} expert stacks use a CPU-only codec "
            "(no Metal matmul kernels yet): feeder/arena staging and GPU "
            "prefill routing off - every expert call runs on the CPU stream"
        )
    # Non-expert weights + KV run on the default device: CPU for --stream-cpu
    # (configure_stream_cpu sets the default to CPU before this call), GPU for
    # --stream-experts.
    base_dev = "CPU" if "cpu" in str(mx.default_device()).lower() else "GPU"
    if streaming:
        head = (
            f"model {total_bytes / 1e9:.0f} GB > ~{budget / 1e9:.0f} GB "
            "wired budget"
            if over_budget
            else f"model {total_bytes / 1e9:.0f} GB, streaming forced"
        )
        loadlog.info(
            f"[stream] streaming: {head} - {n_wrapped} MoE layers' experts "
            f"({offloaded / 1e9:.1f} GB) stay file-backed; rest of the model "
            f"+ KV on {base_dev}"
        )
    feeder_prefill, feeder_decode = _resolve_feeder_defaults(
        feeder_prefill, feeder_decode
    )
    feeder = None
    dfeeder = None
    room = arena = None
    if streaming and prefetcher is not None and moe_modules:
        from gmlx.stream.budget import kv_room_bytes
        from gmlx.stream.table_stream import streamed_table_bytes

        pin = getattr(model, "_kq_weights_pin", None)
        room = kv_room_bytes(gguf_path)
        arena_kw = dict(
            room_bytes=room.bytes,
            pinned_bytes=getattr(pin, "pinned_bytes", 0),
            streamable_bytes=streamed_table_bytes(model),
            cast_dead_bytes=cast_dead_bytes)
        arena = _decode_arena_bytes(
            total_bytes, prefetcher.offsets, budget, **arena_kw)
    ring = 0
    if (
        streaming
        and prefetcher is not None
        and moe_modules
        and feeder_prefill
    ):
        from gmlx.stream.prefill_feeder import (
            maybe_make_prefill_feeder,
            ring_bytes,
        )

        # No working-set budget (the CPU device): the ring is not judged.
        reason = _prefill_ring_reason(
            prefetcher.offsets, arena if budget is not None else None)
        if reason:
            print(f"[stream] feeder prefill unavailable ({reason}); "
                  "falling back to page-cache prefetch")
        else:
            feeder = maybe_make_prefill_feeder(
                prefetcher.offsets, moe_modules)
        if feeder is not None and arena is not None:
            # The ring keeps its room for the process lifetime: a later
            # prefill rebuilds it there, with no lend out of the arena.
            ring = ring_bytes(prefetcher.offsets)
            arena = _decode_arena_bytes(
                total_bytes, prefetcher.offsets, budget, ring_bytes=ring,
                **arena_kw)
        if feeder is not None:
            n_cov = sum(feeder.covers(li) for li in moe_modules)
            for li, mods in moe_modules.items():
                if feeder.covers(li):
                    for m in mods:
                        object.__setattr__(m, "_kq_feeder", feeder)
            object.__setattr__(model, "_kq_feeder", feeder)
            cov = (
                "" if n_cov == len(moe_modules)
                else f" on {n_cov}/{len(moe_modules)} layers"
            )
            loadlog.info(
                "[stream] feeder prefill: expert stacks staged straight "
                f"from GGUF through 2 x {feeder.slot_bytes / 1e9:.1f} GB "
                f"GPU-visible ring slots{cov} (--no-prefill-feeder disables)"
            )
    if (
        streaming
        and prefetcher is not None
        and moe_modules
        and feeder_decode
    ):
        from gmlx.stream.decode_feeder import maybe_make_decode_feeder

        from gmlx.stream.budget import ceiling_bytes

        # The ring's room is out of the arena's budget already, so the
        # two never sum past the ceiling; the lend (DecodeFeeder
        # .lend_for_ring) stays as the fallback for a box whose free RAM
        # is gone when a later prefill rebuilds the ring.
        dfeeder = maybe_make_decode_feeder(
            prefetcher.offsets, moe_modules, arena, stats_verbose)
        if dfeeder is not None:
            dfeeder._room_bytes = room.bytes
            n_cov = sum(dfeeder.covers(li) for li in moe_modules)
            for li, mods in moe_modules.items():
                if dfeeder.covers(li):
                    for m in mods:
                        object.__setattr__(m, "_kq_decode_feeder", dfeeder)
            object.__setattr__(model, "_kq_decode_feeder", dfeeder)
            # Committed from here, not from the first decode: the arena
            # wires itself the moment this model decodes, and a second
            # install that sized against the unwired window would find the
            # memory gone before it ever ran.
            _installs.record(model, dfeeder.nominal_bytes)
            _installs.record_arena(dfeeder)
            if feeder is not None:
                # The first decode call frees the ring (DecodeFeeder
                # .ensure_wired). A later prefill pass rebuilds it in its
                # own room; the rebuild asks the arena to lend only when
                # the box has lost that room.
                dfeeder._release_ring = feeder.release_slots
                feeder._lend_hook = dfeeder.lend_for_ring
            wired = (
                "fully wired at first decode"
                if dfeeder._mlock_deferred
                else f"{dfeeder.locked_bytes / 1e9:.1f} GB wired"
            )
            cov = (
                "" if n_cov == len(moe_modules)
                else f" on {n_cov}/{len(moe_modules)} layers"
            )
            loadlog.info(
                f"[stream] decode feeder: {dfeeder.nominal_bytes / 1e9:.1f} GB "
                f"popularity-managed expert arena ({wired}){cov} "
                "(--no-decode-feeder disables, GMLX_DECODE_ARENA_GB sizes)"
            )
            ceiling = ceiling_bytes() or budget
            expert_bytes = sum(
                r[2] for rs in prefetcher.offsets.values() for r in rs)
            room_how = (
                f"{room.depth} tokens x {room.width}: kv "
                f"{room.kv_bytes / 1e9:.1f} + prefill "
                f"{room.transient_bytes / 1e9:.1f} + reserve "
                f"{room.reserve_bytes / 1e9:.1f}"
                if room.priced else "flat GMLX_DECODE_KV_RESERVE_GB")
            # Always visible, like the pin line: the one line a memory
            # report needs.
            try:
                floor = _ram_floor_bytes(int(mx.device_info()["memory_size"]))
            except Exception:
                floor = _ram_floor_bytes(None)
            print(
                f"[stream] memory budget: ceiling {ceiling / 1e9:.1f} GB = "
                f"every-token {(total_bytes - expert_bytes) / 1e9:.1f} + "
                f"arena {dfeeder.nominal_bytes / 1e9:.1f} + ring "
                f"{ring / 1e9:.1f} + kv room {room.bytes / 1e9:.1f} "
                f"({room_how}) + floor {floor / 1e9:.1f}; "
                "GMLX_STREAM_KV_CTX sizes the room"
            )
            rate = getattr(dfeeder, "_probe_bps", 0.0)
            measured = (
                f"drive reads {rate / 1e9:.1f} GB/s"
                if rate else "fast-disk recipe forced")
            if dfeeder._fast_disk:
                loadlog.info(
                    f"[stream] decode feeder: {measured} - prefetch takes the"
                    " bandwidth (predictions evict by popularity, the barrier"
                    " joins only what the call routes to, prestage reads at"
                    " normal disk priority; --stream-fast-disk off restores"
                    " the conservative recipe)"
                )
            elif rate:
                loadlog.info(
                    f"[stream] decode feeder: {measured} - demand misses have"
                    " the bandwidth, prefetch stays out of their way"
                    " (--stream-fast-disk on overrides,"
                    " GMLX_DECODE_FAST_DISK_GBPS sets the bar)"
                )
    if (streaming or table_offloaded) and env_bool("GMLX_GPU_RESIDENT", True):
        tskip = frozenset()
        if table_offloaded:
            from gmlx.stream.table_stream import streamed_table_array_ids

            tskip = streamed_table_array_ids(model)
        _install_gpu_residency(
            model, moe_modules, skip_ids=tskip,
            include_expert_stacks=bool(table_offloaded) and not streaming)
    if streaming and dfeeder is not None:
        import gmlx.stream.gpu_token as gpu_token

        if gpu_token.autonomous_enabled():
            if gpu_token.route_shed_op() is None:
                print(
                    "[stream] gpu-autonomous: requested but the installed "
                    "mlx_kquant has no route_shed op; falling back to "
                    "per-layer staging"
                )
            else:
                gt = gpu_token.GpuTokenState(dfeeder)
                gpu_token.register_exit_stats(gt)
                for li, mods in moe_modules.items():
                    if dfeeder.covers(li):
                        for m in mods:
                            object.__setattr__(m, "_kq_gpu_token", gt)
                object.__setattr__(model, "_kq_gpu_token", gt)
                mode_note = (
                    "all covered layers syncless (shed-heavy diagnostic)"
                    if gpu_token.autonomous_mode() == "all"
                    else "adaptive: layers above GMLX_AUTO_HOT_HIT go "
                    "syncless, the rest keep per-layer staging"
                )
                loadlog.info(
                    "[stream] gpu-autonomous token: route_shed remaps + "
                    f"sheds on GPU; {mode_note}; misses prestage at "
                    "token boundaries (GMLX_GPU_AUTONOMOUS=1|all)"
                )
    if streaming and dfeeder is not None and env_bool(
            "GMLX_GPU_KEEPWARM", True):
        import gmlx.stream.keepwarm as keepwarm

        keepwarm.start()
        loadlog.info(
            "[stream] gpu keep-warm: background heartbeat holds GPU "
            "clocks between per-layer decode bursts, parked while no "
            "decode is running (lossless, costs power only during "
            "decode; GMLX_GPU_KEEPWARM=0 disables)"
        )
    la_probe = env_bool("GMLX_DECODE_LOOKAHEAD_PROBE", False)
    # Lookahead's replica router folds into the per-layer sync; whether its
    # stall savings cover that tax is a per-family measurement. On
    # glm_moe_dsa (GLM-5.2, 75 layers, top-8) it measured net negative
    # (~40ms/tok sync for ~18ms of stalls), so those families default off.
    # An explicit GMLX_DECODE_LOOKAHEAD always wins.
    la_default = _lookahead_default(model)
    la_prefetch = (
        env_bool("GMLX_DECODE_LOOKAHEAD", la_default) and dfeeder is not None)
    if (streaming and dfeeder is not None and not la_default
            and "GMLX_DECODE_LOOKAHEAD" not in os.environ):
        loadlog.info(
            "[stream] lookahead prestage: off by family default (replica-"
            "router sync tax measured above its stall savings; "
            "GMLX_DECODE_LOOKAHEAD=1 enables)"
        )
    if streaming and (la_probe or la_prefetch):
        from gmlx.stream.lookahead import install_lookahead

        n_la = install_lookahead(
            model, layers, probe=la_probe, prefetch=la_prefetch,
            stats_verbose=stats_verbose,
        )
        la_depth = max(1, min(3, env_int("GMLX_DECODE_LOOKAHEAD_DEPTH", 1)))
        la_what = (
            "next-layer router predictions"
            if la_depth == 1
            else f"router predictions {la_depth} layers deep"
        )
        if n_la and la_prefetch:
            loadlog.info(
                f"[stream] lookahead prestage: {la_what} pre-read arena "
                f"misses on {n_la} MoE layer pairs (lossless; "
                "GMLX_DECODE_LOOKAHEAD=0 disables)"
            )
        if n_la and la_probe:
            loadlog.info(
                f"[stream] lookahead probe: recording {la_what} recall "
                f"on {n_la} MoE layer pairs (lossless; table at exit)"
            )
    if streaming:
        # The context line printed above and the feeder lines cover the
        # normal story; what remains is the fallback mechanics for
        # whatever the feeders don't handle.
        fallback = []
        if dfeeder is None:
            fallback.append(
                "decode streams expert bytes from disk through the page "
                "cache (disk-bound)"
                if over_budget
                else "decode reads experts through the page cache on the "
                "CPU stream"
            )
        if feeder is None:
            fallback.append(
                "prefill uses sequential page-cache prefetch"
                if prefetcher is not None
                else "prefill demand-faults (no gguf_path)"
            )
        if fallback:
            loadlog.info(f"[stream] {'; '.join(fallback)}")
        if not over_budget:
            b = f"~{budget / 1e9:.0f} GB" if budget else "unknown"
            print(
                f"[stream] --stream-cpu streams experts even though the "
                f"{total_bytes / 1e9:.0f} GB model fits the wired budget "
                f"({b}) - omit --stream-cpu for the faster all-GPU path on "
                "a model that fits"
            )
    else:
        gpu_tokens = _stream_gpu_tokens(1)
        if gpu_tokens == 1:
            staging = (
                "model fits the wired budget - decode auto-routed to the "
                "GPU stream (GMLX_STREAM_GPU_TOKENS=0 forces CPU decode)"
            )
        elif gpu_tokens > 0:
            staging = f"prefill calls >={gpu_tokens} tokens routed to GPU"
        else:
            staging = "GPU prefill routing disabled"
        if table_offloaded:
            # Table-only mode: the experts are GPU-resident and wired (the
            # streamed table made room); "file-backed" would be wrong.
            loadlog.info(
                f"[stream] routed experts resident on GPU across "
                f"{n_wrapped} layers ({offloaded / 1e9:.1f} GB wired; "
                f"{staging})"
            )
        else:
            loadlog.info(
                f"[stream] routed experts -> CPU stream on {n_wrapped} "
                f"layers ({offloaded / 1e9:.1f} GB stays file-backed; rest "
                f"of the model + KV on {base_dev}; {staging})"
            )
    return n_wrapped, offloaded


def _resolve_prefill_step(model, requested: int | None) -> tuple[int | None, bool]:
    """Pick the prefill chunk width: an explicit request always wins; a
    streaming-mode model defaults to ``_STREAMING_PREFILL_STEP``, or to its
    model_type's narrower entry; everything else keeps mlx-lm's own
    default. Returns ``(step_or_none, defaulted)``."""
    if requested is not None or not moe_streaming_active(model):
        return requested, False
    mt = getattr(model, "model_type", None) or getattr(
        getattr(model, "args", None), "model_type", None)
    return _STREAMING_PREFILL_STEP_BY_MODEL_TYPE.get(
        mt, _STREAMING_PREFILL_STEP), True


def install_moe_experts_override(model, k: int) -> int:
    """Experiment, lossy: route every token to ``k`` experts instead of the
    trained top-k, on MoE blocks whose experts ``install_expert_streaming``
    wrapped (and only those - the knob exists to probe how router fan-out
    shapes offloaded prefill/decode traffic, not as a general sampler).

    The override rewrites the router's own top-k attribute (``top_k`` /
    ``num_experts_per_tok`` on the block, and on a DeepSeek-style gate
    submodule when present - named ``gate``, or ``router`` on hy_v3), so
    expert selection and the arch's weight renormalization run unchanged
    at the new k. Outputs differ from the trained model by design; parity
    gates will fail. Returns the number of MoE blocks overridden; raises
    on k < 1 or k > the expert count.
    """
    if k < 1:
        raise ValueError(f"MoE top-k override must be >= 1, got {k}")
    layers = getattr(model, "layers", None)
    if layers is None:
        layers = model.model.layers
    overridden = 0
    trained_k = None
    for layer in layers:
        for owner in layer.modules():
            glu = None
            for child in owner.children().values():
                candidates = child if isinstance(child, (list, tuple)) else [child]
                for c in candidates:
                    if type(c).__name__.endswith("_CPUOffload"):
                        glu = c
                        break
                if glu is not None:
                    break
            if glu is None:
                continue
            n_experts = _switch_num_experts(glu)
            if n_experts and k > n_experts:
                raise ValueError(
                    f"MoE top-k override {k} exceeds the {n_experts}-expert "
                    "stack on an offloaded layer"
                )
            hit = False
            for target in (
                owner,
                getattr(owner, "gate", None),
                getattr(owner, "router", None),
            ):
                if target is None:
                    continue
                for attr in ("top_k", "num_experts_per_tok"):
                    current = getattr(target, attr, None)
                    if isinstance(current, int):
                        if trained_k is None:
                            trained_k = current
                        setattr(target, attr, k)
                        hit = True
            if hit:
                overridden += 1
    if overridden:
        print(
            f"[stream] MoE top-k override: {trained_k}->{k} experts/token "
            f"on {overridden} offloaded MoE layers (lossy - outputs differ "
            "from the trained router)"
        )
    else:
        print(
            "[stream] MoE top-k override found no offloaded MoE block "
            "with a router top-k attribute - no effect"
        )
    return overridden
