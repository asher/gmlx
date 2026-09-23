"""Evaluation of a student against a view: bits per byte on held-out slices,
the paired downstream tasks, the chat sanity checks, and the Markdown and
JSON reports."""
from __future__ import annotations

import math
import re
import sys
from typing import Iterable

import numpy as np

from gmlx.load.tokenizer import hf_inner, token_bytes, whitespace_start_mask

from .corpus import nfc, per_turn_rows
from .format import ROUTES_FIELD, pin_routes
from .frames import (
    cut_windows,
    fit_conversation,
    fit_reply,
    render_frame,
    render_row,
    row_render_args,
    shared_boundaries_spans,
    target_mask,
)
from .tokens import adds_bos, bos_id, encode_with_byte_ends

# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

def window_hashes(data: bytes, w: int = 64, block: int = 1 << 20) -> np.ndarray:
    """Polynomial hash of every w-byte window, uint64 wraparound. Hashed in
    blocks of ``block`` bytes (overlapping by w - 1) so the expanded window
    view never exceeds w * block * 8 bytes at once."""
    b = np.frombuffer(data, dtype=np.uint8)
    if len(b) < w:
        return np.zeros(0, dtype=np.uint64)
    P = np.uint64(1099511628211)
    powers = np.ones(w, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for i in range(1, w):
            powers[i] = powers[i - 1] * P
    powers = powers[::-1].copy()
    out = []
    for start in range(0, len(b) - w + 1, block):
        seg = b[start:start + block + w - 1]
        view = np.lib.stride_tricks.sliding_window_view(seg, w).astype(np.uint64)
        with np.errstate(over="ignore"):
            out.append(view @ powers)
    return np.concatenate(out) if len(out) > 1 else out[0]


def decontam_fractions(slices: dict[str, bytes], corpus_texts: Iterable[bytes], w: int = 64) -> dict[str, float]:
    """Per slice, the fraction of its w-byte windows present in the corpus,
    from one pass over the corpus texts: each text is hashed once and its
    windows looked up in every slice's sorted window hashes."""
    sh = {name: np.unique(window_hashes(b, w)) for name, b in slices.items()}
    hit = {name: np.zeros(h.shape[0], dtype=bool) for name, h in sh.items()}
    for t in corpus_texts:
        ch = window_hashes(t, w)
        if ch.size == 0:
            continue
        for name, h in sh.items():
            if h.size == 0:
                continue
            pos = np.searchsorted(h, ch)
            pos[pos == h.size] = 0
            hit[name][pos[h[pos] == ch]] = True
    return {name: (float(hit[name].mean()) if sh[name].size else 0.0) for name in slices}


def decontam_fraction(slice_bytes: bytes, corpus_texts: Iterable[bytes], w: int = 64) -> float:
    """Fraction of the slice's w-byte windows present in the corpus."""
    return decontam_fractions({"slice": slice_bytes}, corpus_texts, w)["slice"]


def _target_logprobs(logits, targets: np.ndarray) -> np.ndarray:
    """[B, T-1] float32 log-probabilities of targets[b, t] (the token at
    t + 1) under logits[b, t], one row at a time: the target logit is
    gathered before the row's log-sum-exp, so no full-vocabulary float32
    array outlives its row."""
    import mlx.core as mx
    tgt = mx.array(np.asarray(targets, dtype=np.int32))
    rows = []
    for b in range(int(logits.shape[0])):
        row = logits[b, :-1]
        zt = mx.take_along_axis(row, tgt[b][:, None], axis=-1)[:, 0].astype(mx.float32)
        lp = zt - mx.logsumexp(row.astype(mx.float32), axis=-1)
        mx.eval(lp)
        rows.append(np.asarray(lp))
    return np.stack(rows)


def cache_kld(model, reader, *, max_rows: int | None = None, tokenizer=None,
              replay_layers: list[int] | None = None) -> dict:
    """Sparse KL of the student against a same-tokenizer teacher cache, the
    bucketed form at the cache's K: sum over the top-K of p (log p - log q)
    plus (1 - M)(log(1 - M) - log Q_tail), per position where the cache has
    an on-path target. The student is forwarded on the cache's token ids.
    Framed rows are re-rendered through ``tokenizer`` (the student's own
    template) when it is given; where that render differs from the cached
    one, positions are paired at equal offsets inside the content spans
    and kept only where the next token agrees on both sides, so the sparse
    KL stays exact. With ``replay_layers`` (the cache's MoE layer list, for
    the cache's own teacher) every row forwarded on its cached ids replays
    the cached routes, so the KL measures elementwise noise only; a
    re-rendered row cannot replay and is counted. Returns the position
    mean, a clustered SE over rows, top-1 agreement, the counts, the rows
    re-rendered and the rows replayed."""
    import mlx.core as mx
    n_rows = len(reader) if max_rows is None else min(max_rows, len(reader))
    replayed = 0
    frame = (reader.manifest.get("gmlx_distill") or {}).get("frame") if tokenizer is not None else None
    stb = token_bytes(tokenizer, int(reader.manifest["vocab_size"])) if frame else None
    row_means, row_top1, n_pos, rerendered = [], [], 0, 0
    for r in range(n_rows):
        arrs, _text, meta = reader.row(r)
        t_ids = arrs["token_ids"].astype(np.int32)
        n = len(t_ids)
        if n < 2:
            continue
        valid = arrs["onpath_mask"][:n - 1].astype(bool)
        ids, t_pos, s_pos = t_ids, None, None
        if frame and meta.get("messages"):
            try:
                stext, s_spans = render_row(tokenizer, meta["messages"], **row_render_args(meta.get("frame")))
            except ValueError:
                continue
            assert stb is not None
            s_ids, s_ends, _flag = encode_with_byte_ends(tokenizer, stext, stb, add_special_tokens=False)
            s_ids = s_ids.astype(np.int32)
            if len(s_ids) != n or not np.array_equal(s_ids, t_ids):
                t_spans = [tuple(x) for x in meta["spans"]]
                al = shared_boundaries_spans(arrs["token_end_byte"].astype(np.int64), s_ends.astype(np.int64),
                                             t_spans, s_spans)
                tp, sp = al.t_pos, al.s_pos
                keep = (tp + 1 < n) & (sp + 1 < len(s_ids))
                tp, sp = tp[keep], sp[keep]
                keep = valid[tp] & (t_ids[tp + 1] == s_ids[sp + 1])
                t_pos, s_pos, ids = tp[keep], sp[keep], s_ids
                rerendered += 1
        if t_pos is None:
            if not valid.any():
                continue
            t_pos = s_pos = np.nonzero(valid)[0]
        elif t_pos.size == 0:
            continue
        assert s_pos is not None
        # the scored positions are gathered before the float32 cast and the
        # softmax, so no full-row full-vocab float32 array is built
        sel_pos = mx.array(s_pos)
        if replay_layers is not None and ids is t_ids and ROUTES_FIELD in arrs:
            with pin_routes(model, arrs[ROUTES_FIELD], replay_layers):
                out = model(mx.array(ids[None]))
                out = out.logits if hasattr(out, "logits") else out
                sel = out[0][sel_pos].astype(mx.float32)
                mx.eval(sel)
            replayed += 1
        else:
            out = model(mx.array(ids[None]))
            out = out.logits if hasattr(out, "logits") else out
            sel = out[0][sel_pos].astype(mx.float32)
        lsm = sel - mx.logsumexp(sel, axis=-1, keepdims=True)
        pos = t_pos
        idx = mx.array(arrs["top_k_indices"][pos].astype(np.int32))
        lq = mx.take_along_axis(lsm, idx, axis=-1)
        top1 = mx.argmax(lsm, axis=-1)
        mx.eval(lq, top1)
        lq = np.asarray(lq).astype(np.float64)
        lp = arrs["top_k_log_softmax"][pos].astype(np.float64)
        p = np.exp(lp)
        M = p.sum(axis=1)
        kl = (p * (lp - lq)).sum(axis=1)
        q_tail = np.maximum(1.0 - np.exp(lq).sum(axis=1), 2.0 ** -126)
        rest = np.maximum(1.0 - M, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            tail = np.where(rest > 0, rest * (np.log(rest) - np.log(q_tail)), 0.0)
        kl = kl + tail
        row_means.append(float(kl.mean()))
        row_top1.append(float((np.asarray(top1) == arrs["top_k_indices"][pos, 0]).mean()))
        n_pos += len(pos)
    rm = np.array(row_means)
    return {"mean_kld_nats": float(rm.mean()) if rm.size else None,
            "clustered_se": float(rm.std(ddof=1) / np.sqrt(rm.size)) if rm.size > 1 else None,
            "top1_agreement": float(np.mean(row_top1)) if row_top1 else None,
            "rows": int(rm.size), "positions": int(n_pos), "K": int(reader.manifest["top_k"]),
            "rerendered_rows": int(rerendered), "replayed_rows": int(replayed)}


def bits_per_byte(model, tokenizer, text: str, *, max_len: int = 512, batch_size: int = 8,
                  tb=None, window_prefix: str | None = None) -> dict:
    """Teacher-forced bits per byte over max_len windows. Every window is
    forwarded behind BOS (when the tokenizer adds one) and, with
    window_prefix, behind that text (a chat wrapper for instruct models);
    the prefix tokens are context only, excluded from the NLL and the byte
    count. A window's first token is scored when something precedes it
    (BOS or the prefix), so for a tokenizer without BOS the prefix adds one
    scored token per window. Returns nll_nats, bytes, tokens, bpb (None
    when no byte was scored)."""
    import mlx.core as mx
    inner = hf_inner(tokenizer)
    text_b = nfc(text).encode("utf-8")
    if tb is None:
        tb = token_bytes(tokenizer)
    ids, ends, _ = encode_with_byte_ends(tokenizer, text_b, tb, add_special_tokens=True)
    special = set(inner.all_special_ids)
    n_prefix = 0
    while n_prefix < len(ids) and int(ids[n_prefix]) in special:
        n_prefix += 1
    ws = whitespace_start_mask(tokenizer, len(tb), tb)
    windows = cut_windows(ids, ws, max_len, n_prefix)
    bos = bos_id(tokenizer) if adds_bos(tokenizer) else None
    prefix_ids = [int(t) for t in inner.encode(window_prefix, add_special_tokens=False)] if window_prefix else []
    head = ([bos] if bos is not None else []) + prefix_ids
    total_nll = 0.0
    total_bytes = 0
    total_tokens = 0
    seqs = []
    for (s, e) in windows:
        while s < e and int(ids[s]) in special:
            s += 1
        if s >= e:
            continue
        prev_end = int(ends[s - 1]) if s > 0 else 0
        seq = head + ids[s:e].tolist()
        starts = [prev_end] * len(head) + ends[s:e].tolist()
        seqs.append((seq, starts, len(head)))
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        L = max(len(s[0]) for s in chunk)
        arr = np.zeros((len(chunk), L), dtype=np.int32)
        for j, (seq, _, _) in enumerate(chunk):
            arr[j, :len(seq)] = seq
        logits = model(mx.array(arr))
        if hasattr(logits, "logits"):
            logits = logits.logits
        lp = _target_logprobs(logits, arr[:, 1:])
        del logits
        for j, (seq, ends_j, n_head) in enumerate(chunk):
            n = len(seq)
            for t in range(max(n_head, 1), n):
                if int(seq[t]) in special:
                    continue
                nb = int(ends_j[t]) - int(ends_j[t - 1])
                total_nll += -float(lp[j, t - 1])
                total_bytes += nb
                total_tokens += 1
    bpb = total_nll / total_bytes / math.log(2.0) if total_bytes else None
    return {"nll_nats": total_nll, "bytes": total_bytes, "tokens": total_tokens, "bpb": bpb}


_REFUSAL_RE = re.compile(r"\b(I can(?:'|no)t|I cannot|I won(?:'|no)t|I'm not able|I am not able|I'm unable|I am unable|not able to help|can't help with|cannot help with|I must decline|I'm sorry|I am sorry)\b", re.IGNORECASE)


_TEMPLATE_LEAK_RE = re.compile(r"<\|?(start_of_turn|end_of_turn|im_start|im_end|eot_id|start_header_id)\|?>|^(user|model|assistant|system)\s*$", re.MULTILINE)


def _is_repetitive(text: str, n: int = 12, times: int = 3) -> bool:
    """True when some run of n whitespace-delimited words repeats at least
    `times` times in the reply (a degenerate loop, not a list)."""
    w = text.split()
    if len(w) < n * times:
        return False
    seen: dict[tuple, int] = {}
    for i in range(len(w) - n + 1):
        k = tuple(w[i:i + n])
        seen[k] = seen.get(k, 0) + 1
        if seen[k] >= times:
            return True
    return False


def _span_rows(tokenizer, convs: list, *, max_len: int, per_turn: bool = False, last_only: bool = False,
               positions: dict | None = None, reason_target: bool = False) -> tuple[list, int]:
    """Scorable rows (ids, target mask, target bytes, id) from conversations.
    Each entry of convs is a message list or a {id, messages,
    student_messages} row; the student's list is used when present. With
    per_turn every assistant turn becomes its own reply row (per_turn_rows);
    with last_only (or per_turn) only the final turn is a target and the
    row is fitted by fit_reply, else by fit_conversation. positions, a
    {id: [[start, end], ...]} map of byte ranges relative to the reply's
    content start, restricts the targets to positions whose predicted
    token starts inside a range (rows whose id is absent score nothing).
    Returns the rows and the count dropped."""
    tb = token_bytes(tokenizer)
    rows = []
    dropped = 0
    for c in convs:
        if isinstance(c, dict):
            rid = str(c.get("id", len(rows)))
            msgs = c.get("student_messages") or c.get("messages") or []
        else:
            rid, msgs = str(len(rows)), c
        variants = per_turn_rows(msgs) if per_turn else [msgs]
        for k, m in enumerate(variants):
            fit = (fit_reply(tokenizer, m, max_len, tb, reason_target=reason_target) if (per_turn or last_only)
                   else fit_conversation(tokenizer, m, max_len, tb))
            if fit is None:
                dropped += 1
                continue
            ids, ends, _text, _m, spans, _f = fit
            e = ends.astype(np.int64)
            tm = target_mask(e, spans)
            if positions is not None:
                ranges = positions.get(rid) if not per_turn else positions.get(f"{rid}:{k}")
                keep = np.zeros_like(tm)
                if ranges:
                    b0 = int(spans[-1][0])
                    starts = e[:-1] - b0
                    for a, z in ranges:
                        keep[:-1] |= (starts >= int(a)) & (starts < int(z))
                tm = tm & keep
            nbytes = int((e[1:] - e[:-1])[tm[:-1]].sum())
            if nbytes > 0:
                rows.append((ids, tm, nbytes, rid if not per_turn else f"{rid}:{k}"))
    return rows, dropped


def _score_span_rows(model, rows: list, *, batch_tokens: int) -> dict:
    """Sum of negative log-likelihood over each row's target mask; batches
    hold at most batch_tokens padded tokens so a long row runs alone. bpb
    and nll_per_token are None when nothing was scored."""
    import mlx.core as mx
    rows = sorted(rows, key=lambda r: len(r[0]))
    ln2 = math.log(2)
    nll = 0.0
    ntok = 0
    nbytes_total = 0
    per_row = []
    items = []
    batches = []
    cur: list = []
    for r in rows:
        L = len(r[0])
        if cur and (len(cur) + 1) * L > batch_tokens:
            batches.append(cur)
            cur = []
        cur.append(r)
    if cur:
        batches.append(cur)
    for chunk in batches:
        L = max(len(r[0]) for r in chunk)
        arr = np.zeros((len(chunk), L), dtype=np.int32)
        for j, r in enumerate(chunk):
            arr[j, :len(r[0])] = r[0]
        logits = model(mx.array(arr))
        if hasattr(logits, "logits"):
            logits = logits.logits
        lp = _target_logprobs(logits, arr[:, 1:])
        del logits
        for j, (ids, tm, nb, rid) in enumerate(chunk):
            m = tm[:len(ids) - 1]
            s = -float(lp[j, :len(ids) - 1][m].sum())
            nll += s
            ntok += int(m.sum())
            nbytes_total += nb
            per_row.append(s / ln2 / nb)
            items.append({"id": rid, "nll": s, "tokens": int(m.sum()), "bytes": nb})
    n = len(per_row)
    return {"bpb": nll / ln2 / nbytes_total if nbytes_total else None,
            "nll_per_token": nll / ntok if ntok else None,
            "row_bpb_se": float(np.std(per_row) / math.sqrt(n)) if n > 1 else None,
            "tokens": ntok, "bytes": nbytes_total, "rows": n, "items": items}


def chat_slice_nll(model, tokenizer, convs: list[list[dict]], *, max_len: int = 2048,
                   batch_tokens: int = 4096, per_turn: bool = False) -> dict:
    """Bits per byte and nats per token over the assistant spans of held-out
    conversations rendered through the model's own chat template (every
    other turn is context). A conversation longer than max_len tokens
    loses trailing turns; one that never fits is dropped and counted.
    With per_turn each assistant turn is scored as the final turn of its
    own prefix (the reply-row render), and a row that does not fit loses
    leading turns instead. Batches hold at most batch_tokens padded
    tokens, so a long conversation runs alone. The SE is over rows."""
    rows, dropped = _span_rows(tokenizer, convs, max_len=max_len, per_turn=per_turn)
    r = _score_span_rows(model, rows, batch_tokens=batch_tokens)
    r.pop("items")
    r["dropped"] = dropped
    r["per_turn"] = per_turn
    return r


def reply_slice_nll(model, tokenizer, rows: list[dict], *, max_len: int = 2048, batch_tokens: int = 4096,
                    positions: dict | None = None, reason_target: bool = False) -> dict:
    """Bits per byte and nats per token over the reply of held-out reply
    rows ({id, messages, student_messages}) under the student's own list,
    restricted to the byte ranges in positions (a census high-delta map)
    when given. Per-row items are returned for paired comparisons."""
    srows, dropped = _span_rows(tokenizer, rows, max_len=max_len, last_only=True, positions=positions,
                                reason_target=reason_target)
    r = _score_span_rows(model, srows, batch_tokens=batch_tokens)
    r["dropped"] = dropped
    r["restricted"] = positions is not None
    return r


def chat_sanity(model, tokenizer, items: list[dict], *, refs: dict | None = None, max_tokens: int = 256) -> dict:
    """Chat sanity set for instruct students. Each item is {"id", "messages":
    [user/assistant turns ending on a user turn], "kind": "task" | "refuse"}.
    Per item: a greedy reply through the chat template, whether it kept the
    turn structure (template compliance: ended on end-of-turn, or was cut by
    the budget without leaking a template marker or looping), whether the
    budget cut it (truncated), and whether it reads as a refusal (phrase
    match). With refs, a {id: reply} map taken
    from the untouched student's own replies, also the nats per token of that
    reference reply under this model (drift from the untouched behaviour).
    Aggregates: compliance, refusal_rate over "refuse" items, task_refusal_rate
    over "task" items, ref_nll_nats mean over scored items."""
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    generate_guard()
    inner = hf_inner(tokenizer)
    sampler = make_sampler(temp=0.0)
    per = []
    for it in items:
        msgs = it["messages"]
        kind = it.get("kind", "task")
        # tokenize=False then encode: some tokenizer wrappers return text
        # for tokenize=True; the template text carries its own BOS
        prompt_text = render_frame(tokenizer, msgs)
        prefix = [int(t) for t in inner.encode(prompt_text, add_special_tokens=False)]
        rec = {"id": it.get("id"), "kind": kind, "n_prefix": len(prefix)}
        text, finish = "", None
        for resp in stream_generate(model, tokenizer, prompt=prefix, max_tokens=max_tokens, sampler=sampler):
            text += resp.text
            finish = resp.finish_reason
        rec["reply"] = text
        rec["finish_reason"] = finish
        rec["truncated"] = finish != "stop"
        # template compliance: the reply keeps the turn structure. A reply
        # that ends on end-of-turn complies; one cut by the budget complies
        # unless it leaked a template marker or fell into repetition
        rec["leaked"] = bool(_TEMPLATE_LEAK_RE.search(text))
        rec["repetitive"] = _is_repetitive(text)
        rec["compliant"] = finish == "stop" or (finish == "length" and not rec["leaked"] and not rec["repetitive"])
        rec["refused"] = bool(_REFUSAL_RE.search(text))
        ref = (refs or {}).get(it.get("id"))
        if ref:
            ref_ids = inner.encode(ref, add_special_tokens=False)
            if ref_ids:
                arr = np.array([prefix + list(ref_ids)], dtype=np.int32)
                out = model(mx.array(arr))
                out = out.logits if hasattr(out, "logits") else out
                lpv = _target_logprobs(out, arr[:, 1:])[0, len(prefix) - 1:]
                rec["ref_nll_nats"] = float(-lpv.mean())
                rec["n_ref_tokens"] = int(len(lpv))
        per.append(rec)
    nll = [r["ref_nll_nats"] for r in per if "ref_nll_nats" in r]
    trunc = [r["truncated"] for r in per]
    refuse = [r["refused"] for r in per if r["kind"] == "refuse"]
    task_refuse = [r["refused"] for r in per if r["kind"] == "task"]
    return {"items": per, "n": len(per),
            "compliance": float(np.mean([r["compliant"] for r in per])) if per else None,
            "truncated_rate": float(np.mean(trunc)) if per else None,
            "refusal_rate": float(np.mean(refuse)) if refuse else None,
            "task_refusal_rate": float(np.mean(task_refuse)) if task_refuse else None,
            "ref_nll_nats": float(np.mean(nll)) if nll else None}


def continuation_logprob(model, tokenizer, context: str, continuations: list[str]) -> list[float]:
    """Sum log-prob of each continuation given the context (multiple-choice
    scoring by continuation log-likelihood)."""
    import mlx.core as mx
    inner = hf_inner(tokenizer)
    ctx_ids = inner.encode(context, add_special_tokens=True)
    out = []
    for cont in continuations:
        full = inner.encode(context + cont, add_special_tokens=True)
        n_ctx = len(ctx_ids)
        # tokens may merge across the boundary; score from the first differing id
        k = 0
        while k < min(n_ctx, len(full)) and full[k] == ctx_ids[k]:
            k += 1
        arr = np.array([full], dtype=np.int32)
        logits = model(mx.array(arr))
        if hasattr(logits, "logits"):
            logits = logits.logits
        lp = _target_logprobs(logits, arr[:, 1:])[0]
        out.append(float(lp[k - 1:].sum()))
    return out


def score_multiple_choice(model, tokenizer, items: list[dict]) -> list[dict]:
    """items: {id, query, choices, gold}. Returns per-item {id, pred, gold,
    correct, scores}."""
    res = []
    for it in items:
        scores = continuation_logprob(model, tokenizer, it["query"], it["choices"])
        pred = int(np.argmax(scores))
        res.append({"id": it["id"], "pred": pred, "gold": int(it["gold"]),
                    "correct": int(pred == int(it["gold"])), "scores": scores})
    return res


def gsm8k_extract(text: str) -> str | None:
    import re
    m = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m[-1].replace(",", "")
    m = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return m[-1].replace(",", "") if m else None


def generate_guard() -> None:
    """mlx-lm's generation wraps every call in its wired-limit context, which
    reads a Metal-only device key; on the CPU device (smoke tests) that
    raises, so the context is replaced by a no-op there."""
    import contextlib

    import mlx.core as mx
    import mlx_lm.generate  # noqa: F401  (the package re-exports the function under the same name)
    g = sys.modules["mlx_lm.generate"]
    if "max_recommended_working_set_size" not in mx.device_info():
        setattr(g, "wired_limit", lambda *_a, **_k: contextlib.nullcontext())


def score_gsm8k(model, tokenizer, items: list[dict], shots: list[dict], *, max_tokens: int = 384) -> list[dict]:
    """Greedy generation, 8-shot prompt, exact match on the #### answer."""
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    generate_guard()
    prefix = "".join(f"Question: {s['question']}\nAnswer: {s['answer']}\n\n" for s in shots)
    sampler = make_sampler(temp=0.0)
    res = []
    for it in items:
        prompt = prefix + f"Question: {it['question']}\nAnswer:"
        text = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler, verbose=False)
        text = text.split("\n\nQuestion:")[0]
        pred = gsm8k_extract(text)
        gold = gsm8k_extract(it["answer"])
        res.append({"id": it["id"], "answer": text, "pred": pred, "gold": gold,
                    "correct": int(pred is not None and gold is not None and float(pred) == float(gold))})
    return res


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 1) -> dict:
    """Mean difference a - b with a percentile CI over paired resamples."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    d = a - b
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return {"diff": float(d.mean()), "ci_lo": float(np.percentile(means, 2.5)),
            "ci_hi": float(np.percentile(means, 97.5)), "n": int(len(d)),
            "p_one_sided_regress": float(np.mean(means >= 0))}


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm step-down: name -> significant."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    stop = False
    for i, (name, p) in enumerate(items):
        if stop or p > alpha / (m - i):
            stop = True
            out[name] = False
        else:
            out[name] = True
    return out


def welch_one_sided(x: np.ndarray, y: np.ndarray) -> dict:
    """Pooled-variance one-sided two-sample t-test that mean(x) > mean(y)
    (equal variances, dof n_x + n_y - 2)."""
    from math import sqrt
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    sp2 = ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / dof
    se = sqrt(sp2 * (1 / nx + 1 / ny))
    t = (x.mean() - y.mean()) / se if se > 0 else float("inf")
    try:
        from scipy.stats import t as tdist
        p = float(1 - tdist.cdf(t, dof))
    except ImportError:
        p = _t_sf(t, dof)
    return {"t": float(t), "dof": dof, "p": p, "pooled_sd": float(sqrt(sp2))}


def _t_sf(t: float, dof: int) -> float:
    """Survival function of Student's t by numerical integration (no scipy)."""
    if math.isinf(t):
        return 0.0 if t > 0 else 1.0
    from math import gamma, pi, sqrt
    c = gamma((dof + 1) / 2) / (sqrt(dof * pi) * gamma(dof / 2))
    def f(u):
        return c * (1 + u * u / dof) ** (-(dof + 1) / 2)
    hi = max(abs(t), 1.0) + 60.0
    xs = np.linspace(t, hi, 200001)
    ys = np.array([f(u) for u in xs[::100]])
    xs2 = xs[::100]
    return float(np.trapezoid(ys, xs2))
