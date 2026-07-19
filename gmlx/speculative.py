"""MTP speculative decoding: single-stream (B=1) and batched (B>1).

Owns the decode loop, the prefill, and the rejection walk; rides the model's
verify forward, the drafter, and the stock KV cache. Generic across drafter
kinds: a native MTP-head drafter and an assistant-model drafter differ only in
which optional hooks they expose, never in this loop's control flow.

The walk samples every verify position into one deferred graph and takes the
accepted-prefix length from an on-device reduction, so a round costs a single
host sync rather than one per position. Draft and target sampling share a
per-round RNG start state, so a position's draft and target draws agree whenever
their distributions do, which keeps acceptance high.

A native MTP head carries its own KV cache, seeded and extended through the
drafter's prefill_from_target_hidden and accept_verified_tokens hooks; an
assistant-model drafter exposes neither and skips them. rollback_speculative_cache
restores a target's hybrid (e.g. gated-delta) cache state where it needs it and is
a no-op on a dense cache.
"""

from __future__ import annotations

import atexit
import logging
import math
import os
import sys
import time
from typing import Any
from collections.abc import Callable, Iterator

import mlx.core as mx
from .envflags import env_bool, env_int

# Speculative round helpers + the draft/target RNG coupler, owned here (see
# spec_helpers) instead of mlx-vlm's private speculative API. The drafter model
# classes and mlx_vlm.models.cache are still consumed from mlx-vlm by design.
from .spec_helpers import (
    _SpeculativeSamplerRNG,
    _buffer_mtp_target_cache,
    _dflash_block_total,
    _mtp_cache_offset_max,
    _mtp_draft_hidden,
    _mtp_draft_position,
    _mtp_next_block_size,
    _mtp_verify_target,
    _record_speculative_round,
    _slice_shared_kv_after_reject,
    generation_stream,
)

_log = logging.getLogger(__name__)

_PREFILL_CHUNK = 2048

_round_log_state = {"fh": None, "path": None, "idx": 0}


def _round_log_open():
    # Env-gated per-round timing for the serve-vs-in-process depth study.
    # Reopens when GMLX_ROUND_LOG changes so one process can write several
    # files (e.g. one per depth in-process).
    path = os.environ.get("GMLX_ROUND_LOG")
    if not path:
        return None
    st = _round_log_state
    if st["path"] != path:
        if st["fh"] is not None:
            st["fh"].close()
        st["fh"] = open(path, "a")
        if _ROUND_PROFILE:
            st["fh"].write("# idx\tdraft_ms\tverify_ms\twalk_ms\temit_ms\tbook_ms\ttotal_ms\taccepted\tbs\tgap_ms\n")
        else:
            st["fh"].write("# idx\tcompute_ms\temit_ms\tbook_ms\ttotal_ms\taccepted\tbs\tgap_ms\n")
        st["path"] = path
    return st


@atexit.register
def _round_log_close():
    fh = _round_log_state["fh"]
    if fh is not None:
        try:
            fh.close()
        except Exception:
            pass


def _round_log_session(kv_offset, max_tokens):
    # Delimit one generation's rounds; kv_offset ~= prompt depth.
    st = _round_log_open()
    if st is None:
        return
    st["fh"].write(f"# session kv={int(kv_offset)} maxtok={int(max_tokens)}\n")
    st["fh"].flush()
    st["idx"] = 0


def _round_log(compute_ms, emit_ms, book_ms, accepted, bs,
               draft_ms=None, verify_ms=None, walk_ms=None,
               gap_ms=None):
    st = _round_log_open()
    if st is None:
        return
    gap_col = f"\t{gap_ms:.3f}" if gap_ms is not None else ""
    if _ROUND_PROFILE and draft_ms is not None:
        total = draft_ms + verify_ms + walk_ms + emit_ms + book_ms
        st["fh"].write(
            f"{st['idx']}\t{draft_ms:.3f}\t{verify_ms:.3f}\t{walk_ms:.3f}\t"
            f"{emit_ms:.3f}\t{book_ms:.3f}\t{total:.3f}\t{accepted}\t{bs}"
            f"{gap_col}\n")
    else:
        total = compute_ms + emit_ms + book_ms
        st["fh"].write(
            f"{st['idx']}\t{compute_ms:.3f}\t{emit_ms:.3f}\t{book_ms:.3f}\t"
            f"{total:.3f}\t{accepted}\t{bs}{gap_col}\n")
    st["fh"].flush()
    st["idx"] += 1


def _argmax_sampler(logits: mx.array) -> mx.array:
    # Greedy-draft fallback for drafters without a built-in argmax path; keeps
    # the caller's leading shape (e.g. [B, 1] for [B, 1, V]).
    return mx.argmax(logits, axis=-1)


# Draft strategy under a sampling target (temp>0). Two lossless schemes (output is
# always the target's sampled token, so both preserve the target distribution):
#   greedy (default): draft the head's argmax, sample the target independently,
#     accept iff equal. Matches llama.cpp's draft-mtp scheme. Deterministic draft
#     consumes no RNG, so coupling overhead is zero.
#   coupled (GMLX_MTP_COUPLED_DRAFT=1): sample the draft + couple the
#     draft/target RNG. Sub-maximal with sorted-CDF samplers (sort-order mismatch
#     between draft/target distributions makes common random numbers disagree even
#     when the modes match). Measured 61% vs 67% greedy on Qwen3.6-27B ultrachat.
_FORCE_GREEDY_DRAFT = not env_bool("GMLX_MTP_COUPLED_DRAFT", False)
_ROUND_PROFILE = env_bool("GMLX_ROUND_PROFILE", False)
# Walk diagnostics (B=1 loop only). GMLX_WALK_PROFILE=1 splits the walk
# into graph-build / eval / host tail and prints medians at session end.
# =2 additionally stages the eval to split lm_head projection from the
# sampler/accept tail (adds one sync; use =1 for representative totals).
# GMLX_MTP_TOP2_LOG=1 additionally logs how often the draft head's second
# choice equals the target token at the first rejected position (the
# tree-verify branch rescue rate); it disables the drafter's internal-argmax
# fast path so the draft sampler sees logits.
_WALK_PROFILE = env_int("GMLX_WALK_PROFILE", 0)
_TOP2_LOG = env_bool("GMLX_MTP_TOP2_LOG", False)
# GMLX_MTP_PQ_LOG=1 measures, per draft position, the exact-match
# acceptance ceiling (max target prob) and the counterfactual p/q rejection-
# sampling acceptance sum(min(p, q_sharp)) for a sweep of draft-proposal
# sharpness settings, without changing emitted tokens. It forces the drafter's
# logits path (same protocol as the top2 log) and folds the reductions into
# the walk's one sync. Sampling-target (B=1) rounds only; greedy rounds are
# counted as skipped. GMLX_MTP_PQ_TARGET describes the run's target
# sampler as temp:top_k:top_p[:min_p] and must match the run's actual sampling
# flags; the transform mirrors mlx_lm make_sampler (top-p, min-p, then top-k
# masks on untempered logprobs, temperature at the categorical), so profile
# sampling like 1.0:20:0.95 replicates exactly. xtc is not modeled.
_PQ_LOG = env_bool("GMLX_MTP_PQ_LOG", False)
# Stochastic MTP acceptance replaces exact-match acceptance with Leviathan
# rejection sampling under a sampling target (B=1 owned rounds): drafts are
# sampled from a sharpened proposal q (GMLX_MTP_DRAFT_SHARP, temp:top_k:
# top_p), accepted with prob min(1, p/q) against the target's effective
# sampling distribution p, and the first rejection emits a residual
# max(p-q, 0) sample. Output distribution equals plain target sampling;
# tokens are not bit-identical to a non-speculative run. Greedy rounds and
# B>1 keep exact-match. Requires a sampler whose effective distribution the
# walk can reconstruct (the serve _FastPositionedSampler, or a make_sampler
# closure annotated with gmlx_sampling_params); opaque samplers fall
# back to exact-match with a one-time warning.
# Off by default: default MTP stays token-identical to non-speculative
# decoding. Opt in per process with run/chat --stochastic-mtp or the server
# config `stochastic_mtp: true` (set_stoch_accept); GMLX_MTP_STOCH_ACCEPT
# presets the default for A/B runs. Acceptance gains are measured in
# docs/performance.md.
_STOCH_ACCEPT = env_bool("GMLX_MTP_STOCH_ACCEPT", False)


def set_stoch_accept(enabled: bool) -> None:
    """Process-wide switch for stochastic MTP acceptance (--stochastic-mtp /
    server config `stochastic_mtp`). Overrides the env preset."""
    global _STOCH_ACCEPT
    _STOCH_ACCEPT = bool(enabled)


def use_owned_engine(drafter, temp: float) -> bool:
    """Whether a stock MTP entry point must route to the owned engine: the
    drafter's contract demands it, or stochastic acceptance is on for a
    sampled run (the stochastic walk lives only in the owned rounds)."""
    return bool(getattr(drafter, "requires_owned_engine", False)
                or (temp > 0.0 and _STOCH_ACCEPT))


_walk_stats = {"rows": [], "reject": 0, "rescue": 0, "rounds": 0,
               "misaligned": 0}
_pq_stats = {"count": [], "match": [], "ceil": [], "cf": [],
             "rounds": 0, "skipped": 0, "misaligned": 0}


def _pq_parse(spec: str, *, single: bool = False):
    settings = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        fields = part.split(":")
        temp = float(fields[0])
        top_k = int(fields[1]) if len(fields) > 1 else 0
        top_p = float(fields[2]) if len(fields) > 2 else 1.0
        min_p = float(fields[3]) if len(fields) > 3 else 0.0
        if temp <= 0.0:
            raise ValueError(f"pq-log setting needs temp > 0: {part!r}")
        if not 0.0 <= min_p < 1.0:
            raise ValueError(f"pq-log setting needs 0 <= min_p < 1: {part!r}")
        settings.append((temp, top_k, top_p, min_p))
    if not settings:
        raise ValueError(f"empty pq-log setting spec: {spec!r}")
    return settings[0] if single else settings


def _pq_parse_env(name: str, default: str, *, single: bool = False):
    """Parse a pq-spec env var, degrading to the default on a malformed value
    (the envflags contract: a bad env string must not crash an import)."""
    try:
        return _pq_parse(os.environ.get(name, default), single=single)
    except ValueError as e:
        _log.warning("%s: %s; using default %r", name, e, default)
        return _pq_parse(default, single=single)


_PQ_TARGET = _pq_parse_env("GMLX_MTP_PQ_TARGET", "1.0:0:1.0", single=True)
_PQ_SWEEP = _pq_parse_env(
    "GMLX_MTP_PQ_SWEEP", "0.6:20:0.95,0.8:20:0.95,1.0:20:0.95,1.0:0:1.0")
# Proposal sharpness for stochastic acceptance. 0.6:20:0.95 measured best
# or tied across coding, chat, and creative workloads.
_STOCH_DRAFT = _pq_parse_env(
    "GMLX_MTP_DRAFT_SHARP", "0.6:20:0.95", single=True)


def _pq_probs(logits: mx.array, temp: float, top_k: int, top_p: float,
              min_p: float = 0.0) -> mx.array:
    """The effective sampling distribution of mlx_lm make_sampler(temp, top_p,
    min_p, top_k), rows [n, V]: top-p, min-p, then top-k masks on untempered
    logprobs (make_sampler's chain order), temperature at the categorical."""
    x = logits.astype(mx.float32)
    lp = x - mx.logsumexp(x, axis=-1, keepdims=True)
    v = lp.shape[-1]
    if 0.0 < top_p < 1.0:
        sorted_idx = mx.argsort(lp, axis=-1)
        sorted_probs = mx.take_along_axis(mx.exp(lp), sorted_idx, axis=-1)
        cum = mx.cumsum(sorted_probs, axis=-1)
        inverse_idx = mx.put_along_axis(
            mx.zeros_like(sorted_idx), sorted_idx,
            mx.arange(v, dtype=sorted_idx.dtype), axis=-1)
        cum = mx.take_along_axis(cum, inverse_idx, axis=-1)
        lp = mx.where(cum > 1 - top_p, lp, -float("inf"))
    if min_p > 0.0:
        # apply_min_p with min_tokens_to_keep=1: the max survives its own
        # threshold, so no explicit keep-floor is needed.
        scaled = mx.max(lp, axis=-1, keepdims=True) + math.log(min_p)
        lp = mx.where(lp < scaled, -float("inf"), lp)
    if 0 < top_k < v:
        mask_idx = mx.argpartition(-lp, kth=top_k - 1, axis=-1)[..., top_k:]
        lp = mx.put_along_axis(
            lp, mask_idx, mx.array(-float("inf"), lp.dtype), axis=-1)
    return mx.softmax(lp / temp, axis=-1)


def _pq_graph(target_logits: mx.array, q_rows: list[mx.array]) -> mx.array:
    """[1 + len(sweep), n_draft] stats: row 0 = max target prob (exact-match
    ceiling), rows 1.. = sum(min(p, q_sharp)) per sweep setting."""
    q = mx.stack([r.astype(mx.float32) for r in q_rows])
    p = _pq_probs(target_logits, *_PQ_TARGET)
    rows = [p.max(axis=-1)]
    for setting in _PQ_SWEEP:
        rows.append(mx.minimum(p, _pq_probs(q, *setting)).sum(axis=-1))
    return mx.stack(rows)


def _pq_accumulate(stats_rows: list[list[float]], tgt: list[int],
                   drf: list[int], n_draft: int) -> None:
    st = _pq_stats
    st["rounds"] += 1
    while len(st["count"]) < n_draft:
        st["count"].append(0)
        st["match"].append(0.0)
        st["ceil"].append(0.0)
        st["cf"].append([0.0] * len(_PQ_SWEEP))
    for j in range(n_draft):
        st["count"][j] += 1
        st["match"][j] += 1.0 if tgt[j] == drf[j] else 0.0
        st["ceil"][j] += stats_rows[0][j]
        for s in range(len(_PQ_SWEEP)):
            st["cf"][j][s] += stats_rows[1 + s][j]


def _pq_expected_tokens(rates: list[float]) -> float:
    expected, run = 1.0, 1.0
    for a in rates:
        run *= a
        expected += run
    return expected


def _pq_report():
    st = _pq_stats
    if not _PQ_LOG:
        return
    if not st["count"]:
        if st["rounds"] or st["skipped"] or st["misaligned"]:
            print(f"[pq-log] no data (rounds={st['rounds']} "
                  f"skipped={st['skipped']} misaligned={st['misaligned']})",
                  file=sys.stderr)
            st["rounds"], st["skipped"], st["misaligned"] = 0, 0, 0
        return
    def _name(s):
        return f"{s[0]}:{s[1]}:{s[2]}" + (f":{s[3]}" if s[3] else "")

    names = [_name(s) for s in _PQ_SWEEP]
    lines = [f"[pq-log] rounds={st['rounds']} skipped={st['skipped']} "
             f"misaligned={st['misaligned']} target={_name(_PQ_TARGET)}",
             "  pos      n  match   ceil  " + "  ".join(f"{n:>14s}" for n in names)]
    match_rates, ceil_rates = [], []
    cf_rates = [[] for _ in _PQ_SWEEP]
    for j, n in enumerate(st["count"]):
        match_rates.append(st["match"][j] / n)
        ceil_rates.append(st["ceil"][j] / n)
        row = f"  {j:>3d} {n:>6d}  {match_rates[j]:.3f}  {ceil_rates[j]:.3f}  "
        cells = []
        for s in range(len(_PQ_SWEEP)):
            cf_rates[s].append(st["cf"][j][s] / n)
            cells.append(f"{cf_rates[s][j]:>14.3f}")
        lines.append(row + "  ".join(cells))
    e_match = _pq_expected_tokens(match_rates)
    parts = [f"match={e_match:.2f}", f"ceil={_pq_expected_tokens(ceil_rates):.2f}"]
    for s, name in enumerate(names):
        e = _pq_expected_tokens(cf_rates[s])
        parts.append(f"{name}={e:.2f}({(e / e_match - 1) * 100:+.1f}%)")
    lines.append("  E[tok/round]: " + "  ".join(parts))
    print("\n".join(lines), file=sys.stderr)
    st["count"], st["match"], st["ceil"], st["cf"] = [], [], [], []
    st["rounds"], st["skipped"], st["misaligned"] = 0, 0, 0


if _PQ_LOG:
    # The CLI can exit without ever closing the decode generator, so the
    # walk-report finally is not a reliable flush; the report is
    # reset-on-print, so a double fire prints nothing new.
    import atexit

    atexit.register(_pq_report)


def _walk_report():
    ws = _walk_stats
    if _WALK_PROFILE and ws["rows"]:
        med = [sorted(c)[len(c) // 2] for c in zip(*ws["rows"])]
        split = f"head={med[3]:.3f} rest={med[1] - med[3]:.3f} " if med[3] else ""
        print(
            f"[walk-profile] rounds={len(ws['rows'])} median ms: "
            f"build={med[0]:.3f} eval={med[1]:.3f} {split}host={med[2]:.3f}",
            file=sys.stderr,
        )
    if _TOP2_LOG and ws["rounds"]:
        r = ws["rescue"] / ws["reject"] if ws["reject"] else float("nan")
        skew = f" MISALIGNED={ws['misaligned']}" if ws["misaligned"] else ""
        print(
            f"[top2-log] rounds={ws['rounds']} rejects={ws['reject']} "
            f"top2-rescues={ws['rescue']} r={r:.3f}{skew}",
            file=sys.stderr,
        )
    ws["rows"], ws["reject"], ws["rescue"], ws["rounds"] = [], 0, 0, 0
    ws["misaligned"] = 0
    _pq_report()


# stochastic (p/q) acceptance

def annotate_sampling_params(sampler, *, temp, top_p, top_k, min_p) -> None:
    """Expose a make_sampler closure's params so the stochastic walk can
    reconstruct its effective distribution. No-op for greedy (None)."""
    if sampler is not None:
        sampler.gmlx_sampling_params = {
            "temp": float(temp), "top_p": float(top_p),
            "top_k": int(top_k), "min_p": float(min_p)}


def _stoch_supported_sampler(sampler) -> bool:
    if sampler is None:
        return False
    if hasattr(sampler, "_filtered") and hasattr(sampler, "temperature"):
        return True
    return getattr(sampler, "gmlx_sampling_params", None) is not None


def _stoch_target_probs(sampler, logits: mx.array) -> mx.array:
    """The sampler's effective distribution over raw logits rows [n, V].

    A serve _FastPositionedSampler reconstructs p through its own _filtered
    pipeline (llama.cpp filter order, min_p included), so p is exact by
    construction; a make_sampler closure is replicated from its
    gmlx_sampling_params annotation (mlx_lm filter order, min_p == 0)."""
    if hasattr(sampler, "_filtered") and hasattr(sampler, "temperature"):
        lp = logits.astype(mx.float32)
        lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        if not getattr(sampler, "_has_filter", True):
            return mx.softmax(lp * (1.0 / sampler.temperature), axis=-1)
        masked, part, order = sampler._filtered(lp)
        p = mx.zeros_like(lp)
        # masked is in rank space (sorted desc by prob); part is in
        # argpartition's arbitrary candidate order. Map rank -> vocab id via
        # order before scattering. (Metal's argpartition returns the top-k
        # pre-sorted, so order == identity hid a direct part scatter on GPU;
        # the CPU backend's does not.)
        ids = mx.take_along_axis(part, order, axis=-1)
        return mx.put_along_axis(p, ids, mx.softmax(masked, axis=-1), axis=-1)
    params = sampler.gmlx_sampling_params
    return _pq_probs(logits, params["temp"], params["top_k"], params["top_p"],
                     min_p=params.get("min_p", 0.0))


def _stoch_draft_sampler(stash: list):
    """Sharpened-proposal draft sampler: sample from q = sharpened head
    logits and stash the q row, aligned [seed, rollouts...] like the pq/top2
    stashes."""
    def sampler(logits):
        row = logits.reshape(-1, logits.shape[-1])[-1:]
        q = _pq_probs(row, *_STOCH_DRAFT)
        tok = mx.random.categorical(mx.log(q), axis=-1)
        stash.append(q[0])
        return tok.reshape(logits.shape[:-1])
    return sampler


def _stochastic_walk(lm, verify, draft_tokens: mx.array, sampler, budget: int,
                     q_rows: list[mx.array]):
    """Leviathan rejection walk with a single host sync.

    Accept draft j with prob min(1, p_j(d_j) / q_j(d_j)); the first rejection
    emits a residual max(p - q, 0) sample, full acceptance emits a bonus
    sample from p. Same (accepted, new_tokens) contract as _coupled_walk."""
    n_draft = int(draft_tokens.shape[1])
    with mx.stream(generation_stream):
        logits = lm.speculative_logits_from_hidden(verify.hidden)
        if logits.ndim == 3:
            logits = logits[0]
        p = _stoch_target_probs(sampler, logits)              # [n_draft+1, V]
        draft_row = draft_tokens.reshape(-1)
        p_n = p[:n_draft]
        q = mx.stack(q_rows)                                  # [n_draft, V]
        p_at = mx.take_along_axis(p_n, draft_row[:, None], axis=-1)[:, 0]
        q_at = mx.take_along_axis(q, draft_row[:, None], axis=-1)[:, 0]
        u = mx.random.uniform(shape=(n_draft,))
        acc = (u * q_at < p_at).astype(mx.int32)
        accepted = mx.cumprod(acc).sum()
        res = mx.maximum(p_n - q, 0.0)
        z = res.sum(axis=-1, keepdims=True)
        res = mx.where(z > 0, res / z, p_n)
        res_tokens = mx.random.categorical(mx.log(res), axis=-1)  # [n_draft]
        bonus = mx.random.categorical(mx.log(p[n_draft:]), axis=-1)  # [1]
    mx.eval(accepted, res_tokens, bonus, draft_row)               # the one sync
    acc_i = int(accepted.item())
    drf = draft_row.tolist()
    if acc_i == n_draft:
        new = drf + bonus.tolist()
    else:
        new = drf[:acc_i] + [int(res_tokens[acc_i].item())]
    return acc_i, new[:budget]


# The master switch (GMLX_SPEC_APC=0) folds in so every sidecar site in
# this module honors it directly, not only via spec_engine's gated arming.
_SIDECAR_DISABLED = (
    not env_bool("GMLX_SPEC_APC", True)
    or not env_bool("GMLX_SPEC_APC_SIDECAR", True)
)


def _coupled_walk(lm, verify, draft_tokens: mx.array, sampler, budget: int,
                  top2=None, pq=None):
    """Rejection walk with a single host sync.

    Sample every verify position into one deferred graph (sequentially, so the
    per-position RNG draws stay coupled to the drafter's), then take the
    accepted-prefix length from an on-device cumprod of leading matches. All
    positions share one lm_head projection. Returns (accepted, new_tokens),
    where new_tokens is the accepted drafts plus the bonus token at the first
    rejection (or the natural next token if every draft is accepted), clamped to
    budget. top2 optionally carries the draft head's per-position second
    choices for the rescue-rate log, pq the head's per-position logits rows for
    the p/q counterfactual log; each must align 1:1 with draft positions
    ([seed, rollouts...]), so a length mismatch is counted, never scored.
    """
    n_draft = int(draft_tokens.shape[1])
    _tb0 = time.perf_counter() if _WALK_PROFILE else 0.0
    head_out = None
    pq_arr = None
    with mx.stream(generation_stream):
        if verify.target_tokens is not None:
            target = verify.target_tokens.reshape(-1)                 # [n_pos]
            if pq is not None:
                _pq_stats["skipped"] += 1
        else:
            logits = lm.speculative_logits_from_hidden(verify.hidden)
            if logits.ndim == 3:
                logits = logits[0]
            if _WALK_PROFILE == 2:
                head_out = logits
            if pq is not None and n_draft > 0:
                if len(pq) == n_draft:
                    pq_arr = _pq_graph(logits[:n_draft], pq)
                else:
                    _pq_stats["misaligned"] += 1
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            if sampler is None:
                target = mx.argmax(logprobs, axis=-1)                 # [n_pos]
            else:
                target = sampler(logprobs).reshape(-1)                # [n_pos]
        draft_row = draft_tokens.reshape(-1)
        if n_draft > 0:
            match = (target[:n_draft] == draft_row).astype(mx.int32)
            accepted = mx.cumprod(match).sum()
        else:
            accepted = mx.array(0, dtype=mx.int32)
    _tb1 = time.perf_counter() if _WALK_PROFILE else 0.0
    head_ms = 0.0
    if head_out is not None:
        mx.eval(head_out)                          # profile=2: stage the head
        head_ms = (time.perf_counter() - _tb1) * 1e3
    extras = list(top2) if top2 else []
    if pq_arr is not None:
        extras.append(pq_arr)
    if extras:
        mx.eval(target, accepted, *extras)                            # the one sync
    else:
        mx.eval(target, accepted)                                     # the one sync
    _tb2 = time.perf_counter() if _WALK_PROFILE else 0.0
    acc = int(accepted.item())
    tgt = target.tolist()
    drf = draft_row.tolist()
    if pq_arr is not None:
        _pq_accumulate(pq_arr.tolist(), tgt, drf, n_draft)
    new = drf[:acc] + [tgt[acc]]
    ws = _walk_stats
    ws["rounds"] += 1
    if top2:
        if len(top2) != n_draft:
            ws["misaligned"] += 1
        elif acc < n_draft:
            ws["reject"] += 1
            if int(top2[acc].item()) == tgt[acc]:
                ws["rescue"] += 1
    if _WALK_PROFILE:
        ws["rows"].append(((_tb1 - _tb0) * 1e3, (_tb2 - _tb1) * 1e3,
                           (time.perf_counter() - _tb2) * 1e3, head_ms))
    return acc, new[:budget]


def _coupled_walk_batch(
    lm,
    verify,
    draft_tokens: mx.array,
    sampler,
    budgets: list[int],
) -> tuple[list[int], list[list[int]]]:
    """Batched rejection walk with a single host sync.

    Per-row cumprod over matches yields an accepted count per row [B], all in
    one deferred graph evaluated with a single mx.eval. Returns
    (accepted_list, new_tokens_list) where each row's new_tokens is the
    accepted drafts plus the bonus at the first rejection, clamped to that
    row's budget.
    """
    B = int(draft_tokens.shape[0])
    n_draft = int(draft_tokens.shape[1])
    n_pos = n_draft + 1
    with mx.stream(generation_stream):
        if verify.target_tokens is not None:
            target = verify.target_tokens                                # [B, n_pos]
        else:
            logits = lm.speculative_logits_from_hidden(verify.hidden)    # [B, n_pos, V]
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            if sampler is None:
                target = mx.argmax(logprobs, axis=-1)                    # [B, n_pos]
            else:
                flat = logprobs.reshape(-1, logprobs.shape[-1])          # [B*n_pos, V]
                target = sampler(flat).reshape(B, n_pos)                 # [B, n_pos]
        if n_draft > 0:
            match = (target[:, :n_draft] == draft_tokens).astype(mx.int32)
            accepted = mx.cumprod(match, axis=1).sum(axis=1)             # [B]
        else:
            accepted = mx.zeros(B, dtype=mx.int32)
    mx.eval(target, accepted)                                            # the one sync
    acc_list = accepted.tolist()
    tgt = target.tolist()
    drf = draft_tokens.tolist()
    new_tokens_list: list[list[int]] = []
    for i in range(B):
        a = acc_list[i]
        new = drf[i][:a] + [tgt[i][a]]
        new_tokens_list.append(new[:budgets[i]])
    return acc_list, new_tokens_list


def _owned_decode_rounds(
    model,
    drafter,
    lm,
    prompt_cache: list,
    *,
    hidden: mx.array,
    b: int,
    shared_kv,
    seed_tokens: mx.array | None,
    emitted: int,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array] | None,
    draft_block_size: int | None,
    drafter_warm: list | None = None,
    sidecar_ctx: dict | None = None,
) -> Iterator[int]:
    """Owned MTP decode loop, shared by the CLI prefill+decode path and the
    serve decode-only path.

    Consumes the captured target hidden (full captured length; reduced to the
    last slot here), the first sampled token b, the prefill shared_kv, and
    seed_tokens -- the token ids whose hidden seeds a native MTP head (the
    captured prompt slice on the CLI path, the prompt tokens on the serve path).
    emitted counts tokens the caller already yielded (the first bonus), so the
    loop yields only the tokens it produces. Greedy iff sampler is None.
    """
    token_dtype = mx.int32
    greedy = sampler is None
    # greedy_draft: draft the head's argmax even under a sampling target (llama
    # parity). The target is still sampled (sampler passed to the walk) unless the
    # whole round is greedy. A deterministic draft consumes no RNG, so coupling is
    # moot -- disable it to skip the per-round save/restore.
    greedy_draft = greedy or _FORCE_GREEDY_DRAFT

    block_total = _dflash_block_total(drafter, draft_block_size)
    configured_block_total = int(getattr(drafter.config, "block_size", block_total))
    drafter.reset(model)

    stoch = _STOCH_ACCEPT and not greedy and _stoch_supported_sampler(sampler)
    if _STOCH_ACCEPT and not greedy and not stoch:
        _log.warning(
            "GMLX_MTP_STOCH_ACCEPT: sampler's effective distribution is "
            "not reconstructable (opaque sampler); using exact-match")

    # Draft and target share a per-round RNG start state. (Gating on a
    # positioned-target sampler would decouple them and lower acceptance.)
    # The stochastic path needs no coupling: min(1, p/q) acceptance only
    # requires the accept uniforms to be independent of the draft draws.
    sampler_rng = _SpeculativeSamplerRNG(
        drafter, enabled=not greedy_draft and not stoch)

    stoch_stash: list[mx.array] = []
    if stoch:
        draft_sampler = _stoch_draft_sampler(stoch_stash)
        draft_kwargs = {}
    else:
        draft_sampler = _argmax_sampler if greedy_draft else sampler
        draft_kwargs = ({"greedy": True}
                        if greedy_draft and getattr(drafter, "supports_greedy_draft_argmax", False)
                        else {})

    top2_stash: list | None = None
    if _TOP2_LOG:
        # Stash each draft-head sampling's second choice for the rescue-rate
        # log. Forces the logits path (no internal-argmax fast path) so the
        # wrapper sees every pick; the drafted tokens themselves are
        # unchanged. Alignment protocol: draft position 0 is the seed, whose
        # pick happens inside the previous round's accept (or the prefill)
        # hook, so the stash is cleared only after the walk consumes it --
        # the accept hook then deposits the next round's seed entry and
        # draft_block appends the rollout entries behind it.
        top2_stash = []
        _inner_sampler = draft_sampler

        def draft_sampler(logits, _inner=_inner_sampler, _stash=top2_stash):
            with mx.stream(generation_stream):
                kth = logits.shape[-1] - 2
                k2 = mx.argpartition(logits, kth=kth, axis=-1)[..., -2:]
                v2 = mx.take_along_axis(logits, k2, axis=-1)
                second = mx.take_along_axis(
                    k2, mx.argmin(v2, axis=-1, keepdims=True), axis=-1)
            _stash.append(second.reshape(-1)[0])
            return _inner(logits)

        draft_kwargs = {}

    pq_stash: list | None = None
    if _PQ_LOG and stoch:
        # The counterfactual log measures the gain over exact-match from the
        # head's raw logits; under stochastic acceptance both premises are
        # gone, so the two modes are mutually exclusive.
        _log.warning("GMLX_MTP_PQ_LOG disabled: stochastic acceptance on")
    elif _PQ_LOG:
        # Same alignment protocol as the top2 stash: entry 0 is the seed pick
        # deposited by the previous round's accept hook (or the prefill),
        # rollout picks append behind it, the walk consumes+clears per round.
        pq_stash = []
        _pq_inner_sampler = draft_sampler

        def draft_sampler(logits, _inner=_pq_inner_sampler, _stash=pq_stash):
            _stash.append(logits.reshape(-1, logits.shape[-1])[-1])
            return _inner(logits)

        draft_kwargs = {}

    # Sidecar warm start: adopt the restored head KV (prefix rows) so the
    # suffix-only hidden below teacher-forces at its true positions instead
    # of position 0 (the acceptance-at-depth erosion an L1-only hit causes).
    if (drafter_warm and not _SIDECAR_DISABLED
            and getattr(drafter, "supports_kv_sidecar", False)):
        drafter.restore_kv(drafter_warm)

    # Native MTP head: seed the head's KV from the captured target hidden.
    prefill_draft = getattr(drafter, "prefill_from_target_hidden", None)
    if callable(prefill_draft) and seed_tokens is not None:
        sampler_rng.draft_call(prefill_draft, seed_tokens, hidden, b,
                               draft_sampler, token_dtype, **draft_kwargs)

    _sidecar_post_prefill(drafter, sidecar_ctx)

    if hidden.shape[1] > 1:
        hidden = hidden[:, -1:, :]
    hidden = _mtp_draft_hidden(lm, hidden)
    kv_offset = _mtp_cache_offset_max(prompt_cache)
    drafter.set_shared_kv(shared_kv, kv_offset,
                          position=_mtp_draft_position(kv_offset), kv_valid_len=kv_offset)

    _accept_fn = getattr(drafter, "accept_verified_tokens", None)
    _has_accept = callable(_accept_fn)
    _rollback_fn = getattr(lm, "rollback_speculative_cache", None)
    _has_rollback = callable(_rollback_fn)
    _draft_hidden_fn = getattr(lm, "speculative_draft_hidden", None)
    _has_draft_hidden = callable(_draft_hidden_fn)
    _walk_sampler = None if greedy else sampler
    _needs_shared_kv = getattr(drafter, "uses_shared_kv", True)
    _draft_block = drafter.draft_block
    _prefer_fixed_bs = getattr(drafter, "prefer_requested_block_size", False)

    _round_log_session(kv_offset, max_tokens)
    _prev_end = time.perf_counter()
    _last_clear = emitted
    try:
        while emitted < max_tokens:
            _t0 = time.perf_counter()
            _gap = (_t0 - _prev_end) * 1e3 if _prev_end else 0.0
            if _prefer_fixed_bs:
                bs = min(block_total, max_tokens - emitted + 1)
            else:
                bs = _mtp_next_block_size(drafter, block_total, configured_block_total,
                                          max_tokens - emitted + 1)
            if bs <= 1:
                break
            draft_tokens = sampler_rng.draft_tokens(
                _draft_block, b, hidden, None, bs, draft_sampler, token_dtype,
                **draft_kwargs)
            # The drafter may return fewer drafts than requested (e.g. the
            # deepseek-v4 confidence-gated rollout); every downstream bs
            # consumer (rollback width, accept bookkeeping, finish seams)
            # must see the actual width or rejection math trims valid
            # tokens from the cache.
            bs = int(draft_tokens.shape[1]) + 1

            if _ROUND_PROFILE:
                mx.eval(draft_tokens)
            _td = time.perf_counter()

            with mx.stream(generation_stream):
                verify_input = mx.concatenate(
                    [mx.array([[b]], dtype=token_dtype), draft_tokens], axis=1)
                verify = _mtp_verify_target(lm, verify_input, prompt_cache, sampler,
                                            sample_target_tokens=greedy)

            if _ROUND_PROFILE:
                mx.eval(verify.hidden)
            _tv = time.perf_counter()

            if stoch and len(stoch_stash) == draft_tokens.shape[1]:
                accepted, new_tokens = _stochastic_walk(
                    lm, verify, draft_tokens, sampler,
                    max_tokens - emitted, list(stoch_stash))
            else:
                # Exact-match walk; also the per-round fallback when a drafter
                # returns fewer drafts than it sampled (stash misaligned) --
                # accepting a sampled draft iff it equals the target's own
                # sample stays lossless, just lower-acceptance.
                accepted, new_tokens = _coupled_walk(
                    lm, verify, draft_tokens, _walk_sampler,
                    max_tokens - emitted,
                    top2=list(top2_stash) if top2_stash else None,
                    pq=list(pq_stash) if pq_stash else None)
            stoch_stash.clear()
            if top2_stash is not None:
                # Consumed; the accept hook below re-seeds entry 0 for the
                # next round.
                top2_stash.clear()
            if pq_stash is not None:
                pq_stash.clear()
            sampler_rng.target_sampled(sync_draft=True)
            _record_speculative_round(drafter, accepted, bs - 1)
            _t1 = time.perf_counter()

            n_new = len(new_tokens)
            budget_left = max_tokens - emitted
            if n_new > budget_left:
                new_tokens = new_tokens[:budget_left]
                n_new = budget_left
            delivered = 0
            try:
                for tok in new_tokens:
                    delivered += 1
                    yield tok
            except GeneratorExit:
                # Consumer stopped mid-round (EOS / stop string). Roll the target
                # cache back to exactly the delivered tokens so the finish seam
                # sees KV consistent with what was consumed (APC retirement
                # depends on this; without it the final round leaves rejected
                # drafts and unconsumed accepts in the cache).
                k = min(delivered, accepted)
                if _has_rollback and k < bs - 1:
                    with mx.stream(generation_stream):
                        _rollback_fn(prompt_cache, verify.gdn_states, k, bs)
                # Mirror the rollback into the drafter head: ingest exactly the
                # delivered tokens so its KV pairs row-for-row with the retired
                # target prefix (the retirement-time sidecar depends on this;
                # without it the head lags by the final round and every
                # retirement sidecar is skipped as unfaithful).
                if _has_accept and delivered > 0:
                    _accept_fn(verify.hidden, draft_tokens, k,
                               new_tokens[:delivered] if delivered > accepted
                               else [],
                               draft_sampler, token_dtype, **draft_kwargs)
                raise
            emitted += n_new
            if emitted >= max_tokens:
                # Budget exhausted: same finish-seam contract as the mid-round
                # close above -- drop this round's rejected-draft KV tail (and,
                # when the budget truncated the round, the undelivered accepts),
                # and top the drafter head up with the delivered tokens.
                k = min(delivered, accepted)
                if _has_rollback and k < bs - 1:
                    with mx.stream(generation_stream):
                        _rollback_fn(prompt_cache, verify.gdn_states, k, bs)
                if _has_accept and delivered > 0:
                    _accept_fn(verify.hidden, draft_tokens, k,
                               new_tokens if delivered > accepted else [],
                               draft_sampler, token_dtype, **draft_kwargs)
                return
            _t2 = time.perf_counter()

            if _has_accept:
                sampler_rng.draft_call(_accept_fn, verify.hidden, draft_tokens,
                                       accepted, new_tokens, draft_sampler, token_dtype,
                                       **draft_kwargs)

            hidden = (_draft_hidden_fn(verify.hidden[:, accepted:accepted + 1, :])
                      if _has_draft_hidden
                      else verify.hidden[:, accepted:accepted + 1, :])
            b = new_tokens[-1] if new_tokens else b
            if accepted < bs - 1 and _has_rollback:
                with mx.stream(generation_stream):
                    _rollback_fn(prompt_cache, verify.gdn_states, accepted, bs)
            if _needs_shared_kv:
                next_shared_kv = _slice_shared_kv_after_reject(
                    verify.shared_kv_states, bs - (accepted + 1))
                kv_offset += accepted + 1
                drafter.set_shared_kv(next_shared_kv, kv_offset,
                                      position=_mtp_draft_position(kv_offset),
                                      kv_valid_len=kv_offset)
            else:
                kv_offset += accepted + 1
            if emitted - _last_clear >= 256:
                mx.clear_cache()
                _last_clear = emitted
            _round_log((_t1 - _t0) * 1e3, (_t2 - _t1) * 1e3,
                       (time.perf_counter() - _t2) * 1e3, accepted, bs,
                       draft_ms=(_td - _t0) * 1e3 if _ROUND_PROFILE else None,
                       verify_ms=(_tv - _td) * 1e3 if _ROUND_PROFILE else None,
                       walk_ms=(_t1 - _tv) * 1e3 if _ROUND_PROFILE else None,
                       gap_ms=_gap)
            _prev_end = time.perf_counter()

    finally:
        _walk_report()


def stream_speculative(
    model,
    drafter,
    prompt: mx.array,
    *,
    prompt_cache: list,
    max_tokens: int = 256,
    sampler: Callable[[mx.array], mx.array] | None = None,
    draft_block_size: int | None = None,
    prefill_chunk: int = _PREFILL_CHUNK,
) -> Iterator[int]:
    """Yield generated token ids one at a time.

    sampler is None for greedy (argmax throughout, no RNG coupling) or a callable
    logprobs[B, V] -> token[B] for temperature sampling.
    """
    lm = model.language_model if hasattr(model, "language_model") else model
    if not hasattr(lm, "rollback_speculative_cache"):
        raise RuntimeError(
            "MTP speculative decoding requires the target to expose "
            "rollback_speculative_cache")
    greedy = sampler is None

    if prompt.ndim == 1:
        prompt = prompt[None]
    n = int(prompt.shape[1])

    # Chunk through all prompt tokens, retaining each chunk's pre-norm hidden, so
    # the native MTP head can teacher-force the whole prompt into its own KV. This
    # matches llama.cpp's draft-mtp, whose process() hook fires on every prefill
    # ubatch and teacher-forces the full shifted prompt into the MTP draft KV
    # (common/speculative.cpp). Truncating the capture to the last token starved
    # the head of prompt context at depth (d4096/d16384) and eroded acceptance;
    # full prompt context recovers it (measured +0.2-0.4 mean-accepted on
    # Qwen3.6-27B). The full prompt seeds the head; the retained hidden is the
    # target signal it teacher-forces against.
    hiddens = []
    # Shared-KV drafters read only the last position; retaining every chunk's
    # hidden is required only when the drafter teacher-forces its own KV from
    # the full prompt (see the note above).
    keep_all_hiddens = callable(
        getattr(drafter, "prefill_from_target_hidden", None))
    # Window-limited heads (deepseek_v4: sliding_window=128, relative RoPE)
    # can't use context beyond hidden_capture_limit trailing positions, so
    # cap the retained capture -- V4's raw 4D hidden is 32 KB/token, and an
    # uncapped 32k-prompt capture would pin ~1 GB for a 128-token window.
    capture_limit = (getattr(drafter, "hidden_capture_limit", None)
                     if keep_all_hiddens else None)
    i = 0
    with mx.stream(generation_stream):
        while i < n:
            take = min(prefill_chunk, n - i)
            last = (i + take >= n)
            out = lm(prompt[:, i:i + take], cache=prompt_cache,
                     return_hidden=True, return_shared_kv=last)
            if keep_all_hiddens:
                hiddens.append(out.hidden_states[-1])
                if capture_limit:
                    total = sum(int(h.shape[1]) for h in hiddens)
                    if total > capture_limit:
                        merged = (hiddens[0] if len(hiddens) == 1
                                  else mx.concatenate(hiddens, axis=1))
                        hiddens = [merged[:, -capture_limit:]]
            else:
                hiddens = [out.hidden_states[-1]]
            i += take
            if not last:
                mx.eval([c.state for c in prompt_cache] + [hiddens[-1]])
                mx.clear_cache()
        first_logits = out.logits[:, -1, :]
        first = (mx.argmax(first_logits, axis=-1) if greedy
                 else sampler(first_logits))
    hidden = hiddens[0] if len(hiddens) == 1 else mx.concatenate(hiddens, axis=1)
    shared_kv = out.shared_kv_states
    b = int(first.item())

    _buffer_mtp_target_cache(prompt_cache, drafter, draft_block_size)
    yield b
    if max_tokens <= 1:
        return

    yield from _owned_decode_rounds(
        model, drafter, lm, prompt_cache,
        hidden=hidden, b=b, shared_kv=shared_kv, seed_tokens=prompt,
        emitted=1, max_tokens=max_tokens, sampler=sampler,
        draft_block_size=draft_block_size)


def owned_server_rounds(
    model,
    drafter,
    prompt_cache: list,
    hidden: mx.array,
    *,
    first_bonus,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array] | None,
    shared_kv_states,
    prompt_tokens,
    draft_block_size: int | None = None,
    greedy_sampling: bool = False,
    stop_check: Callable[[int, int], bool] | None = None,
    eos_token_ids: set | None = None,
    **_extra,
) -> Iterator[tuple]:
    """Decode-only owned MTP round for the serve path (batch size 1).

    Matches mlx-vlm's run_speculative_server_rounds contract: the server has
    already prefilled prompt_cache and emitted first_bonus, and passes the
    captured full-prompt hidden, the prefill shared_kv_states, and the prompt
    tokens (the head seed). Yields (per-row [tok], None); the bonus is never
    re-yielded. Routes through the OWNED decode loop, not mlx-vlm's _mtp_rounds,
    so the serve path is the same engine the CLI uses and is batching-ready.
    """
    lm = model.language_model if hasattr(model, "language_model") else model
    b = (int(first_bonus.reshape(-1).item())
         if isinstance(first_bonus, mx.array) else int(first_bonus))
    # Pop the retirement context before buffering: _buffer_mtp_target_cache
    # replaces rotating entries with BufferedRotatingKVCache objects, and the
    # stash attr dies with the swapped-out entry (gemma-4's first layer is
    # rotating). A generator local is immune to cache-entry swaps.
    retire_ctx = _pop_retire_ctx(prompt_cache)
    drafter_warm = _pop_drafter_warm(prompt_cache)
    sidecar_ctx = None
    if retire_ctx is not None:
        sidecar_ctx = {
            "full_ids": retire_ctx["full_ids"],
            "extra_hash": int(retire_ctx.get("extra_hash", 0)),
            "checkpoint_len": int(retire_ctx.get("checkpoint_len", 0) or 0),
            "manager": getattr(model, "_kq_apc_manager", None),
        }
        if retire_ctx.get("mode") == "ckpt":
            _ckpt_post_prefill(model, prompt_cache, retire_ctx)
    _buffer_mtp_target_cache(prompt_cache, drafter, draft_block_size)
    eff_sampler = None if greedy_sampling else sampler
    generated = [b]
    rounds = _owned_decode_rounds(
        model, drafter, lm, prompt_cache,
        hidden=hidden, b=b, shared_kv=shared_kv_states,
        seed_tokens=prompt_tokens, emitted=1, max_tokens=max_tokens,
        sampler=eff_sampler, draft_block_size=draft_block_size,
        drafter_warm=drafter_warm, sidecar_ctx=sidecar_ctx)
    try:
        for tok in rounds:
            generated.append(tok)
            # Finish eagerly on the token that ends the request: the server
            # abandons finished generators (close fires only at GC, often
            # after the next request has prefilled), so (a) a deferred
            # retirement would store after the follow-up turn's cache lookup
            # and miss it, and (b) this frame stays suspended at the final
            # yield -- at 32k depth the request KV + captured hidden held in
            # its locals is ~1.6 GB pinned until process exit. Close the
            # round loop first (its mid-round rollback trims the target
            # cache to the delivered tokens), retire, then drop every heavy
            # local so the abandoned frame pins nothing.
            if ((eos_token_ids is not None and tok in eos_token_ids)
                    or (stop_check is not None and stop_check(0, tok))
                    or len(generated) >= max_tokens):
                rounds.close()
                if retire_ctx is not None:
                    _retire_b1(model, prompt_cache, generated, retire_ctx,
                               drafter=drafter, sidecar_ctx=sidecar_ctx)
                    retire_ctx = None
                rounds = None
                prompt_cache = hidden = shared_kv_states = None
                drafter_warm = first_bonus = prompt_tokens = None
                sidecar_ctx = None
            yield [tok], None
    finally:
        # Non-terminal finishes (client disconnect mid-stream): close the
        # round loop first -- its mid-round-close rollback must trim the
        # target cache before the retirement snapshot reads it (implicit
        # close would only fire at frame teardown, after this finally).
        # The terminal-token path above already closed, retired, and nulled
        # the locals, so this can't double-store. See _retire_b1.
        if rounds is not None:
            rounds.close()
        if retire_ctx is not None:
            _retire_b1(model, prompt_cache, generated, retire_ctx,
                       drafter=drafter, sidecar_ctx=sidecar_ctx)
            retire_ctx = None
        # Always-on per-request acceptance summary: the aggregate record a
        # serve run keeps by default (GMLX_ROUND_LOG for per-round detail).
        al = list(getattr(drafter, "accept_lens", None) or ())
        if al:
            dl = list(getattr(drafter, "draft_lens", None) or ())
            drafted = sum(dl) if len(dl) == len(al) else 0
            rate = f" rate={sum(al) / drafted:.3f}" if drafted else ""
            print(f"[spec] rounds={len(al)} drafted={drafted} "
                  f"accepted={sum(al):g}{rate}", file=sys.stderr, flush=True)


def _ckpt_post_prefill(model, prompt_cache: list, retire_ctx: dict) -> None:
    """Full-prompt checkpoint-tier store, replacing the stock post-prefill
    exact store that ckpt arming suppresses.

    Runs once at rounds entry, before the target cache is buffered: the
    first token is already out, so the store cost (only-new blocks plus the
    recurrent-state sidecar) lands on the gap before the second token, the
    same place the drafter prefill already sits. This is the key a
    continuation turn hits when no retirement happened. Best-effort; never
    raises.
    """
    try:
        manager = getattr(model, "_kq_apc_manager", None)
        if manager is None:
            return
        from .cache_snapshot import ckpt_store
        ckpt_store(manager, retire_ctx["full_ids"], prompt_cache,
                   extra_hash=int(retire_ctx.get("extra_hash", 0)))
    except Exception:
        _log.warning("APC ckpt post-prefill failed; continuing",
                     exc_info=True)


def _sidecar_post_prefill(drafter, sidecar_ctx: dict | None) -> None:
    """Store the freshly seeded head KV under the target's post-prefill keys.

    Runs right after the drafter prefill. Only a faithful head may be stored:
    its KV must cover every prompt position (cold full-hidden prefill, or a
    sidecar warm start plus the suffix) -- a head seeded from suffix-only
    hidden has its rows at the wrong positions and would poison future turns
    if stored under a full-prefix key. The coverage verdict is also recorded
    on the drafter for the retirement-time sidecar store. Two keys mirror the
    stock exact-mode target stores: the guard-trimmed checkpoint length (what
    an identical re-sent prompt hits) and the full prompt (what a continuation
    hits when no retirement happened). Best-effort; never raises.
    """
    # Reset the coverage verdict and request nonce unconditionally: a stale
    # nonce from an earlier request would let that request's lazy retirement
    # export this request's head KV under the old key (silent poison).
    if drafter is not None:
        try:
            drafter._kq_head_covered = False
            drafter._kq_head_request = None
        except Exception:
            pass  # slotted/frozen drafter forbids ad-hoc attrs
    if sidecar_ctx is None or _SIDECAR_DISABLED:
        return
    if not getattr(drafter, "supports_kv_sidecar", False):
        return
    try:
        full_ids = sidecar_ctx["full_ids"]
        n = len(full_ids)
        caches = drafter.export_kv()
        if not caches:
            return
        offset = min(int(getattr(c, "offset", 0) or 0) for c in caches)
        if offset != n:
            _log.debug(
                "sidecar post-prefill skipped: head offset %d != prompt %d",
                offset, n)
            return
        drafter._kq_head_covered = True
        # Identity nonce: retirement only stores the sidecar if the drafter
        # still belongs to this request (the server drops finished
        # generators lazily, so a deferred retire can observe a drafter
        # reseeded by a later request).
        drafter._kq_head_request = sidecar_ctx
        manager = sidecar_ctx.get("manager")
        if manager is None:
            return
        from .cache_snapshot import drafter_sidecar_store
        extra_hash = int(sidecar_ctx.get("extra_hash", 0))
        checkpoint_len = int(sidecar_ctx.get("checkpoint_len", 0) or 0)
        stored = []
        if 0 < checkpoint_len <= n:
            if drafter_sidecar_store(
                    manager, drafter, full_ids, checkpoint_len, extra_hash):
                stored.append(checkpoint_len)
        if n != checkpoint_len:
            if drafter_sidecar_store(
                    manager, drafter, full_ids, n, extra_hash):
                stored.append(n)
        if stored:
            _log.info("APC sidecar store: tokens=%s", stored)
    except Exception:
        _log.warning("APC sidecar post-prefill failed; continuing",
                     exc_info=True)


def _pop_drafter_warm(prompt_cache: list) -> list | None:
    """Detach the restored drafter-KV sidecar from the prompt cache.

    Stashed by the prefill's L1 lookup on the request's first cache entry
    (same request-scoped discipline as the retirement context; a model-level
    stash races with the next request's prefill). Popped before buffering
    can swap the entry out from under the attr.
    """
    if not prompt_cache:
        return None
    warm = getattr(prompt_cache[0], "_kq_apc_drafter_warm", None)
    if warm is not None:
        try:
            prompt_cache[0]._kq_apc_drafter_warm = None
        except Exception:
            pass  # slotted/frozen cache forbids ad-hoc attrs
    return warm


def _pop_retire_ctx(prompt_cache: list) -> dict | None:
    """Detach the request-scoped retirement context from the prompt cache.

    The prefill stashes it on the request's first cache entry (request-scoped;
    a model-level stash races with the next request's prefill because the
    server closes finished generators lazily). Popped once, before any code
    can swap cache entries out from under the attr.
    """
    if not prompt_cache:
        return None
    retire = getattr(prompt_cache[0], "_kq_apc_retire", None)
    if retire is not None:
        try:
            prompt_cache[0]._kq_apc_retire = None
        except Exception:
            pass  # slotted/frozen cache forbids ad-hoc attrs
    return retire


def _retire_b1(model, prompt_cache: list, generated: list[int],
               retire: dict | None, drafter=None,
               sidecar_ctx: dict | None = None) -> None:
    """Store a finished B=1 request's full KV into the shared APC.

    Guarded on an exact offset match: only stores when the target cache holds
    exactly ``len(full_ids) + len(generated)`` tokens (or one fewer -- the
    pending-token round-boundary state), so a cache carrying stale
    rejected-draft KV (or trimmed by rotation) is skipped, never stored under
    a key it does not faithfully cover. A drafter whose head KV faithfully
    covers the prompt (see _sidecar_post_prefill) gets its KV stored alongside
    under the same key, so the follow-up turn's warm start keeps full drafter
    context (acceptance parity) instead of a suffix-only seed. When
    ``sidecar_ctx`` is given, the sidecar store additionally requires the
    drafter's request nonce to be this very context object: the target
    prompt_cache is closure-owned and safe under a lazy (GC-time) retire, but
    the drafter is model-shared and may have been reseeded by a later
    request. Best-effort; never raises into the generator's finally.
    """
    if not retire:
        return
    try:
        manager = getattr(model, "_kq_apc_manager", None)
        if manager is None:
            return
        full_ids = retire["full_ids"]
        seq = full_ids + [int(t) for t in generated]
        offset = _mtp_cache_offset_max(prompt_cache)
        # Two clean finish states. offset == len(seq)-1: the newest sampled
        # token's KV pends the next verify (round-boundary invariant) -- store
        # everything but it (usually the EOS; costs one warm token). offset ==
        # len(seq): a mid-round close where the last delivered token was a
        # verify input, so its KV is already committed.
        if offset == len(seq) - 1:
            seq = seq[:-1]
        elif offset != len(seq):
            # Anything else is a rejected-draft KV tail, rotation, or a hidden
            # trim. Skip rather than store a mismatched entry; the count tells
            # us which arches need a fix.
            _log.info(
                "APC retire skipped: cache offset %d != tokens %d",
                offset, len(seq))
            return
        from .cache_snapshot import retirement_store
        ok = retirement_store(
            manager, retire.get("mode"), seq, prompt_cache,
            row=0, extra_hash=int(retire.get("extra_hash", 0)))
        if ok:
            _log.info("APC retire store: tokens=%d", len(seq))
        if (ok and drafter is not None and not _SIDECAR_DISABLED
                and getattr(drafter, "_kq_head_covered", False)
                and (sidecar_ctx is None
                     or getattr(drafter, "_kq_head_request", None)
                     is sidecar_ctx)):
            from .cache_snapshot import drafter_sidecar_store
            if drafter_sidecar_store(
                    manager, drafter, seq, len(seq),
                    int(retire.get("extra_hash", 0))):
                _log.info("APC sidecar store (retire): tokens=%d", len(seq))
    except Exception:
        _log.warning("APC retire failed; continuing", exc_info=True)


def _retire_batch_row(model, prompt_cache: list, slot: int,
                      retire: dict, gen_row: list[int],
                      position: int) -> None:
    """Store a finished batch row's KV into the shared APC (block mode only).

    Exact-mode retirement is a full per-row cache clone -- never taken at
    B>1 (it stalls the co-resident lanes; the drafter-state sidecar is the
    planned fix). Block mode harvests only not-yet-cached 16-token blocks,
    which is cheap enough to run between rounds. ``position`` is the row's
    absolute token offset in the batch cache; content is aligned with the
    token sequence up to ``len(seq) - 1`` (the newest token's KV pends the
    next verify), so anything the guard passes is faithfully covered.
    Best-effort; never raises into the decode loop.
    """
    try:
        manager = getattr(model, "_kq_apc_manager", None)
        if manager is None:
            return
        if retire.get("mode") != "block":
            # exact = full row clone (stalls co-resident lanes); ckpt =
            # affordable but unwired at B>1 in v1 -- the eligible fleet is
            # thinking-template models whose retirement keys structurally
            # miss on chat turns, so the stall buys nothing yet.
            _log.debug(
                "APC retire skipped at B>1: %s-mode store not taken in a "
                "live batch", retire.get("mode"))
            return
        seq = retire["full_ids"] + [int(t) for t in gen_row]
        store_len = len(seq) - 1
        if position < store_len:
            _log.info(
                "APC retire skipped: row position %d < tokens %d",
                position, store_len)
            return
        from .cache_snapshot import retirement_store
        ok = retirement_store(
            manager, "block", seq[:store_len], prompt_cache,
            row=slot, extra_hash=int(retire.get("extra_hash", 0)))
        if ok:
            _log.info("APC retire store (row): tokens=%d", store_len)
    except Exception:
        _log.warning("APC retire failed; continuing", exc_info=True)


# Batched B>1 owned MTP round

def _owned_decode_rounds_batch(
    model,
    drafter,
    lm,
    prompt_cache: list,
    *,
    hidden: mx.array,
    b: list[int],
    shared_kv: dict,
    seed_tokens: mx.array | None,
    emitted: list[int],
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array] | None,
    draft_block_size: int | None,
    stop_check: Callable[[int, int], bool] | None = None,
    eos_token_ids: set | None = None,
    row_ids: list[int] | None = None,
) -> Iterator[tuple[list[int | None], Any]]:
    """Owned batched MTP decode loop (B >= 2).

    Structurally parallel to the B=1 _owned_decode_rounds but tracks per-row
    state (bonus token, KV offset, emitted count, finished flag). Uses
    _coupled_walk_batch for the single-sync walk, accept_verified_tokens_batch
    for ragged drafter KV management, and the existing batch-aware
    rollback_speculative_cache.

    Coupled-draft RNG is forced off at B>1 (greedy-draft only).

    Continuous-batch injection: between rounds, checks
    ``model._generator_injections`` for new rows queued by the scheduler.
    New rows' target KV caches are extended into ``prompt_cache``, the
    drafter is prefilled in isolation then merged, and the per-row loop
    state grows to include the newcomers.
    """
    token_dtype = mx.int32
    greedy = sampler is None
    B_orig = len(b)
    row_ids = list(range(B_orig)) if row_ids is None else list(row_ids)

    # Per-row APC retirement state. Rows born in a batched prefill carry no
    # retire context (APC is gated to single-request prefill); rows injected
    # from a B=1 prefill carry theirs on the injected cache's first entry.
    retire_ctxs: list[dict | None] = [None] * B_orig
    gen_rows: list[list[int]] = [[int(t)] for t in b]
    retired = [False] * B_orig

    block_total = _dflash_block_total(drafter, draft_block_size)
    configured_block_total = int(
        getattr(drafter.config, "block_size", block_total))
    # Batched rounds reseed the shared drafter without the B=1 sidecar seam;
    # clear the request nonce so no earlier request's lazy retirement can
    # export this batch's head KV under its key.
    try:
        drafter._kq_head_covered = False
        drafter._kq_head_request = None
    except Exception:
        pass  # slotted/frozen drafter forbids ad-hoc attrs
    drafter.reset(model, left_padding=[0] * len(b))
    sampler_rng = _SpeculativeSamplerRNG(drafter, enabled=False)

    draft_kwargs = {}
    if getattr(drafter, "supports_greedy_draft_argmax", False):
        draft_kwargs["greedy"] = True
    draft_sampler = _argmax_sampler

    prefill_draft = getattr(drafter, "prefill_from_target_hidden", None)
    if callable(prefill_draft) and seed_tokens is not None:
        sampler_rng.draft_call(
            prefill_draft, seed_tokens, hidden, mx.array(b, dtype=token_dtype),
            draft_sampler, token_dtype, **draft_kwargs)

    if hidden.shape[1] > 1:
        hidden = hidden[:, -1:, :]
    hidden = _mtp_draft_hidden(lm, hidden)

    L_prefill = _mtp_cache_offset_max(prompt_cache)
    positions = [L_prefill] * len(b)
    drafter.set_shared_kv(
        shared_kv, kv_offset=L_prefill,
        position=_mtp_draft_position(mx.array(positions)),
        kv_valid_len=mx.array(positions),
        left_padding=_batch_left_padding(prompt_cache))

    finished = [False] * B_orig
    active_idx = list(range(B_orig))

    _accept_batch_fn = getattr(drafter, "accept_verified_tokens_batch", None)
    _has_accept_batch = callable(_accept_batch_fn)
    _rollback_fn = getattr(lm, "rollback_speculative_cache", None)
    _has_rollback = callable(_rollback_fn)
    _draft_hidden_fn = getattr(lm, "speculative_draft_hidden", None)
    _has_draft_hidden = callable(_draft_hidden_fn)
    _walk_sampler = None if greedy else sampler
    _needs_shared_kv = getattr(drafter, "uses_shared_kv", True)
    _draft_block = drafter.draft_block

    def _drain_injections():
        # continuous-batch injection
        nonlocal hidden, B_orig
        gen_inj = getattr(model, "_generator_injections", None)
        if gen_inj:
            for inj in gen_inj:
                B_new = len(inj["uids"])
                for i, cache in enumerate(prompt_cache):
                    extend_fn = getattr(cache, "extend", None)
                    if callable(extend_fn):
                        other = inj["prompt_cache"][i]
                        if hasattr(cache, "_idx") and not hasattr(other, "_idx"):
                            other = type(other).merge([other])
                        extend_fn(other)

                inject_fn = getattr(drafter, "inject_rows", None)
                if callable(inject_fn):
                    inject_fn(
                        inj["prompt_tokens"], inj["hidden"],
                        inj["first_tokens"], draft_sampler,
                        token_dtype, greedy=True)

                inj_hidden = inj["hidden"]
                if inj_hidden.shape[1] > 1:
                    inj_hidden = inj_hidden[:, -1:, :]
                inj_hidden = _mtp_draft_hidden(lm, inj_hidden)
                hidden = mx.concatenate([hidden, inj_hidden], axis=0)

                # A single-request injection carries its APC retirement
                # context on its (un-merged) cache's first entry; pop it
                # before the injected cache object is discarded. Multi-row
                # injections were prefilled batched, where APC is off.
                # The drafter-KV sidecar stash is popped and dropped: the
                # batched drafter's KV is shared across lanes, so a per-row
                # warm restore needs extend machinery this loop doesn't have.
                inj_ctx = (_pop_retire_ctx(inj["prompt_cache"])
                           if B_new == 1 else None)
                _pop_drafter_warm(inj["prompt_cache"])

                inj_offset = _mtp_cache_offset_max(inj["prompt_cache"])
                for row in range(B_new):
                    b.append(int(inj["first_tokens_list"][row]))
                    positions.append(inj_offset)
                    emitted.append(1)
                    finished.append(False)
                    active_idx.append(B_orig)
                    retire_ctxs.append(inj_ctx if row == 0 else None)
                    gen_rows.append([int(inj["first_tokens_list"][row])])
                    retired.append(False)
                    B_orig += 1

                if _needs_shared_kv and inj.get("shared_kv_states"):
                    for k in shared_kv:
                        K_old, V_old = shared_kv[k]
                        if k in inj["shared_kv_states"]:
                            K_new, V_new = inj["shared_kv_states"][k]
                            shared_kv[k] = (
                                mx.concatenate([K_old, K_new], axis=0),
                                mx.concatenate([V_old, V_new], axis=0))
            # Injection grew B; the target caches text mrope deltas at the old
            # width and only handles too-WIDE (slices down), not too-narrow --
            # verify then dies on offsets(B) + rope_deltas(B_old) broadcast.
            # Text-only rows have delta 0, so zero-pad to the new width.
            _rd = getattr(lm, "_rope_deltas", None)
            if _rd is not None and _rd.shape[0] < B_orig:
                _pad = mx.zeros(
                    (B_orig - _rd.shape[0],) + tuple(_rd.shape[1:]),
                    dtype=_rd.dtype)
                lm._rope_deltas = mx.concatenate([_rd, _pad], axis=0)
            gen_inj.clear()
        # end injection

    _round_log_session(max(positions) if positions else 0, max_tokens)
    _prev_end = time.perf_counter()
    _last_clear = sum(emitted)
    while len(active_idx) > 0:
        _drain_injections()
        n_active = len(active_idx)
        remaining = [
            max(1, max_tokens - emitted[active_idx[j]] + 1)
            for j in range(n_active)]
        bs = _mtp_next_block_size(
            drafter, block_total, configured_block_total, min(remaining))
        if bs <= 1:
            break

        _t0 = time.perf_counter()
        _gap = (_t0 - _prev_end) * 1e3 if _prev_end else 0.0
        b_active = [b[active_idx[j]] for j in range(n_active)]
        b_arr = mx.array(b_active, dtype=token_dtype)

        draft_tokens = sampler_rng.draft_tokens(
            _draft_block, b_arr, hidden, None, bs, draft_sampler, token_dtype,
            **draft_kwargs)
        # Actual draft width (drafters may return fewer than requested);
        # see the scalar round.
        bs = int(draft_tokens.shape[1]) + 1
        if _ROUND_PROFILE:
            mx.eval(draft_tokens)
        _td = time.perf_counter()

        with mx.stream(generation_stream):
            verify_input = mx.concatenate(
                [b_arr[:, None], draft_tokens], axis=1)
            verify = _mtp_verify_target(
                lm, verify_input, prompt_cache, sampler,
                sample_target_tokens=greedy)
        if _ROUND_PROFILE:
            mx.eval(verify.hidden)
        _tv = time.perf_counter()

        budgets = [max_tokens - emitted[active_idx[j]] for j in range(n_active)]
        accepted_list, new_tokens_list = _coupled_walk_batch(
            lm, verify, draft_tokens, _walk_sampler, budgets)
        _t1 = time.perf_counter()
        sampler_rng.target_sampled(sync_draft=True)
        _record_speculative_round(
            drafter,
            sum(accepted_list) / len(accepted_list),
            bs - 1)

        max_a = max(accepted_list)

        if _has_accept_batch:
            sampler_rng.draft_call(
                _accept_batch_fn, verify.hidden, draft_tokens,
                accepted_list, new_tokens_list, draft_sampler, token_dtype,
                **draft_kwargs)

        if max_a < bs - 1 or any(a < max_a for a in accepted_list):
            row_idx = mx.arange(n_active)
            col_idx = mx.array(accepted_list)
            hidden = verify.hidden[row_idx, col_idx, :][:, None, :]
        else:
            hidden = verify.hidden[:, -1:, :]
        hidden = _mtp_draft_hidden(lm, hidden)

        max_new = max(len(nt) for nt in new_tokens_list) if new_tokens_list else 0
        for pos in range(max_new):
            tokens_out: list[int | None] = [None] * B_orig
            for j in range(n_active):
                orig = active_idx[j]
                if pos < len(new_tokens_list[j]) and not finished[orig]:
                    tok = new_tokens_list[j][pos]
                    tokens_out[orig] = tok
                    gen_rows[orig].append(tok)
                    emitted[orig] += 1
                    if emitted[orig] >= max_tokens:
                        finished[orig] = True
                    if eos_token_ids is not None and tok in eos_token_ids:
                        finished[orig] = True
                    if stop_check is not None and stop_check(orig, tok):
                        finished[orig] = True
            yield tokens_out, {"round_pos": pos, "round_len": max_new}
        _t2 = time.perf_counter()

        for j in range(n_active):
            orig = active_idx[j]
            if new_tokens_list[j]:
                b[orig] = new_tokens_list[j][-1]
            positions[orig] = positions[orig] + accepted_list[j] + 1

        if any(a < bs - 1 for a in accepted_list) and _has_rollback:
            with mx.stream(generation_stream):
                _rollback_fn(prompt_cache, verify.gdn_states,
                             accepted_list, bs)

        if _needs_shared_kv:
            rejected_global = bs - (max_a + 1)
            next_shared_kv = _slice_shared_kv_batch(
                verify.shared_kv_states, rejected_global, accepted_list, max_a)
        else:
            next_shared_kv = shared_kv

        # Retire finished rows into the shared APC while their KV is still in
        # the batch cache (the filter below drops it). Runs after this
        # round's rollback, so per-row content is clean up to positions[orig].
        for j in range(n_active):
            orig = active_idx[j]
            if finished[orig] and not retired[orig]:
                retired[orig] = True
                if retire_ctxs[orig] is not None:
                    _retire_batch_row(
                        model, prompt_cache, j, retire_ctxs[orig],
                        gen_rows[orig], positions[orig])
                    retire_ctxs[orig] = None

        if all(finished[active_idx[j]] for j in range(n_active)):
            # The yield above is the generator's only suspension point, so an
            # injection queued during this round would otherwise be stranded:
            # its request stalls after its first token and the stale entry is
            # adopted by an unrelated later batch's first-round drain. Drop
            # the finished rows and adopt the injections as the new batch.
            if not getattr(model, "_generator_injections", None) or not all(
                    hasattr(c, "filter") for c in prompt_cache):
                break
            empty = mx.array([], dtype=mx.int32)
            for c in prompt_cache:
                c.filter(empty)
            _filter_drafter = getattr(drafter, "filter_batch", None)
            if callable(_filter_drafter):
                _filter_drafter(empty)
            hidden = hidden[empty]
            if _needs_shared_kv:
                for k in next_shared_kv:
                    K_next, V_next = next_shared_kv[k]
                    next_shared_kv[k] = (K_next[empty], V_next[empty])
                shared_kv = next_shared_kv
            active_idx = []
            _drain_injections()
            if not active_idx:
                break
            continue
        cache_filterable = all(hasattr(c, "filter") for c in prompt_cache)
        if cache_filterable:
            keep_slots = [
                j for j in range(n_active) if not finished[active_idx[j]]]
            if len(keep_slots) < n_active:
                keep_mx = mx.array(keep_slots, dtype=mx.int32)
                for c in prompt_cache:
                    c.filter(keep_mx)
                filter_drafter = getattr(drafter, "filter_batch", None)
                if callable(filter_drafter):
                    filter_drafter(keep_mx)
                hidden = hidden[keep_mx]
                for k in next_shared_kv:
                    K_next, V_next = next_shared_kv[k]
                    next_shared_kv[k] = (K_next[keep_mx], V_next[keep_mx])
                active_idx = [active_idx[j] for j in keep_slots]

        positions_active = [positions[i] for i in active_idx]
        new_kv_offset = max(positions_active) if positions_active else 0
        drafter.set_shared_kv(
            next_shared_kv, kv_offset=new_kv_offset,
            position=_mtp_draft_position(mx.array(positions_active)),
            kv_valid_len=mx.array(positions_active),
            left_padding=_batch_left_padding(prompt_cache))

        if sum(emitted) - _last_clear >= 256:
            mx.clear_cache()
            _last_clear = sum(emitted)

        _round_log((_t1 - _t0) * 1e3, (_t2 - _t1) * 1e3,
                   (time.perf_counter() - _t2) * 1e3,
                   round(sum(accepted_list) / len(accepted_list), 2), bs,
                   draft_ms=(_td - _t0) * 1e3 if _ROUND_PROFILE else None,
                   verify_ms=(_tv - _td) * 1e3 if _ROUND_PROFILE else None,
                   walk_ms=(_t1 - _tv) * 1e3 if _ROUND_PROFILE else None,
                   gap_ms=_gap)
        _prev_end = time.perf_counter()


def _slice_shared_kv_batch(
    shared_kv_states: dict, rejected_global: int,
    accepted_list: list[int], max_a: int,
) -> dict:
    """Slice and per-row tail-zero shared_kv after batched reject."""
    next_shared_kv = {}
    for k, kv in shared_kv_states.items():
        K, V = kv
        valid = K.shape[-2] - rejected_global
        if valid >= K.shape[-2]:
            K_next, V_next = K, V
        elif valid <= 0:
            K_next = K[..., :1, :]
            V_next = V[..., :1, :]
        else:
            K_next = K[..., :valid, :]
            V_next = V[..., :valid, :]
        if any(a < max_a for a in accepted_list):
            mask_rows = mx.arange(K_next.shape[-2])
            keep_lens = mx.array(
                [valid - max_a + a for a in accepted_list], dtype=mx.int32)
            keep_mask = mask_rows[None, :] < keep_lens[:, None]
            keep_f = keep_mask.astype(K_next.dtype)[:, None, :, None]
            K_next = K_next * keep_f
            V_next = V_next * keep_f
        next_shared_kv[k] = (K_next, V_next)
    return next_shared_kv


def _batch_left_padding(prompt_cache: list) -> mx.array | None:
    """Extract left-padding from the first BatchKVCache in the prompt cache."""
    for c in prompt_cache:
        lp = getattr(c, "left_padding", None)
        if lp is not None:
            return lp
    return None


def owned_server_rounds_batch(
    model,
    drafter,
    prompt_cache: list,
    hidden: mx.array,
    *,
    first_bonus: mx.array,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array] | None,
    shared_kv_states: dict,
    prompt_tokens: mx.array | None,
    draft_block_size: int | None = None,
    greedy_sampling: bool = False,
    stop_check: Callable[[int, int], bool] | None = None,
    eos_token_ids: set | None = None,
    row_ids: list[int] | None = None,
    **_extra,
) -> Iterator[tuple[list[int | None], Any]]:
    """Decode-only owned MTP round for the serve path (batch size >= 2).

    Matches the run_speculative_server_rounds contract. B=1 serve uses
    owned_server_rounds (scalar fast-path); this is the batched counterpart.
    """
    lm = model.language_model if hasattr(model, "language_model") else model
    B = int(first_bonus.shape[0])
    b = first_bonus.reshape(-1).tolist()
    _buffer_mtp_target_cache(prompt_cache, drafter, draft_block_size)
    eff_sampler = None if greedy_sampling else sampler
    yield from _owned_decode_rounds_batch(
        model, drafter, lm, prompt_cache,
        hidden=hidden, b=b, shared_kv=shared_kv_states,
        seed_tokens=prompt_tokens,
        emitted=[1] * B, max_tokens=max_tokens,
        sampler=eff_sampler, draft_block_size=draft_block_size,
        stop_check=stop_check, eos_token_ids=eos_token_ids,
        row_ids=row_ids)
