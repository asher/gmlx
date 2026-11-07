"""Raw token logprobs for the serve engine.

The values are f32 ``log_softmax`` of the model's logits before any logits
processor (bias, penalties, grammars), as llama.cpp and vLLM report them.
Sampling still reads the processed logits.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import mlx.core as mx

from gmlx.spec.topk import HIER_MIN_V, HIER_ROW, topk_ids

# (token logprob, top ids in descending order, their logprobs)
LogprobEntry = tuple[float, list[int] | None, list[float] | None]

JSON_FLOOR = -9999.0


def raw_logprobs(logits: mx.array) -> mx.array:
    """``logits`` [..., V] in any dtype -> f32 log_softmax [..., V]."""
    x = logits.astype(mx.float32)
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def top_logprobs(lp: mx.array, k: int) -> tuple[mx.array, mx.array]:
    """The ``k`` largest values of each row of ``lp`` [N, V], descending:
    (ids [N, k] int32, values [N, k]). A large vocabulary takes the
    hierarchical selection, so no sort spans the whole row."""
    v = int(lp.shape[-1])
    k = max(1, min(int(k), v))
    if v >= HIER_MIN_V and k <= -(-v // HIER_ROW):
        ids = topk_ids(lp, k)
        vals = mx.take_along_axis(lp, ids, axis=-1)
        order = mx.argsort(-vals, axis=-1)
        ids = mx.take_along_axis(ids, order, axis=-1)
        vals = mx.take_along_axis(vals, order, axis=-1)
    else:
        ids = mx.argsort(-lp, axis=-1)[:, :k]
        vals = mx.take_along_axis(lp, ids, axis=-1)
    return ids.astype(mx.int32), vals


def json_logprob(x: float) -> float:
    """A logprob safe for JSON: non-finite values and values below -9999.0
    become -9999.0, as vLLM clamps them."""
    x = float(x)
    if not math.isfinite(x) or x < JSON_FLOOR:
        return JSON_FLOOR
    return x


class RoundLogprobs:
    """Logprobs of one target forward for the rows that want them.

    ``logits`` [W, n_pos, V] are the raw rows (before any row processor) of
    the W wanted rows. The f32 logprobs and their top-k are dispatched on
    build. ``drafts`` [W, n_draft], when given, puts the logprob of each
    draft id in the same graph, so a round whose emitted ids are all drafts
    or top-k ids needs no further sync.
    """

    def __init__(self, logits: mx.array, k: int, drafts: mx.array | None = None):
        lp = raw_logprobs(logits)
        if lp.ndim == 2:
            lp = lp[None]
        self._lp = lp
        w, n_pos, v = (int(s) for s in lp.shape)
        self.k = max(0, min(int(k), v))
        targets: list[mx.array] = []
        self._ids = self._vals = None
        if self.k > 0:
            ids, vals = top_logprobs(lp.reshape(w * n_pos, v), self.k)
            self._ids = ids.reshape(w, n_pos, self.k)
            self._vals = vals.reshape(w, n_pos, self.k)
            targets += [self._ids, self._vals]
        self._drafts = self._draft_vals = None
        if drafts is not None and int(drafts.shape[-1]) > 0:
            d = drafts.reshape(w, -1).astype(mx.int32)
            n_d = min(int(d.shape[1]), n_pos)
            d = d[:, :n_d]
            self._drafts = d
            self._draft_vals = mx.take_along_axis(
                lp[:, :n_d, :], d[:, :, None], axis=-1)[:, :, 0]
            targets += [self._drafts, self._draft_vals]
        if targets:
            mx.async_eval(*targets)

    def select(self, rows: Sequence[int]) -> RoundLogprobs:
        """The same round for a subset of its rows, in the order given."""
        idx = mx.array(list(rows), dtype=mx.int32)
        out = RoundLogprobs.__new__(RoundLogprobs)
        out.k = self.k
        out._lp = self._lp[idx]
        out._ids = None if self._ids is None else self._ids[idx]
        out._vals = None if self._vals is None else self._vals[idx]
        out._drafts = None if self._drafts is None else self._drafts[idx]
        out._draft_vals = None if self._draft_vals is None else self._draft_vals[idx]
        return out

    def arrays(self) -> list[mx.array]:
        """The arrays a caller may hand to ``mx.async_eval``."""
        return [a for a in (self._ids, self._vals, self._drafts, self._draft_vals)
                if a is not None]

    def entries(self, tokens_rows: Sequence[Sequence[int]]) -> list[list[LogprobEntry]]:
        """One entry per emitted token: ``tokens_rows[w][i]`` is the id the
        row emitted at verify position ``i``."""
        host_ids = host_vals = host_d = host_dv = None
        if self._ids is not None:
            host_ids = self._ids.tolist()
            host_vals = self._vals.tolist()
        if self._drafts is not None:
            host_d = self._drafts.tolist()
            host_dv = self._draft_vals.tolist()
        values: list[list[float | None]] = []
        missing: list[tuple[int, int, int]] = []
        for w, toks in enumerate(tokens_rows):
            row: list[float | None] = []
            for i, tok in enumerate(toks):
                tok = int(tok)
                val = None
                if host_d is not None and i < len(host_d[w]) and host_d[w][i] == tok:
                    val = host_dv[w][i]
                elif host_ids is not None and i < len(host_ids[w]):
                    try:
                        val = host_vals[w][i][host_ids[w][i].index(tok)]
                    except ValueError:
                        val = None
                if val is None:
                    missing.append((w, i, tok))
                row.append(val)
            values.append(row)
        if missing:
            idx_w = mx.array([m[0] for m in missing], dtype=mx.int32)
            idx_p = mx.array([m[1] for m in missing], dtype=mx.int32)
            idx_t = mx.array([m[2] for m in missing], dtype=mx.int32)
            got = self._lp[idx_w, idx_p, idx_t].tolist()
            for (w, i, _), val in zip(missing, got):
                values[w][i] = val
        out: list[list[LogprobEntry]] = []
        for w, toks in enumerate(tokens_rows):
            row_entries: list[LogprobEntry] = []
            for i in range(len(toks)):
                ids = host_ids[w][i] if host_ids is not None else None
                vals = host_vals[w][i] if host_vals is not None else None
                row_entries.append((float(values[w][i]), ids, vals))
            out.append(row_entries)
        return out
