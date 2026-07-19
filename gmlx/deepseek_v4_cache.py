# SPDX-License-Identifier: Apache-2.0
"""PoolingCache + BatchPoolingCache for DeepSeek V4 (vendored).

Copied 1:1 from mlx-lm PR 1192
https://github.com/Blaizzy/mlx-lm/blob/5c10538136b9038b9626c134612b08afc18d697a/mlx_lm/models/cache.py
(lines 903-1447), by way of omlx's ``patches/deepseek_v4/cache_extras.py``
port. ``deepseek_v4_model.ensure_registered()`` injects both classes into
the ``mlx_lm.models.cache`` namespace (upstream wins if it already ships
them) so cache-type resolution by module attribute keeps working.

Delete this file once installed mlx-lm ships PoolingCache natively.

The one-update undo log (``trim(n)``, n <= 2, after verify writes that may
complete pool windows) is load-bearing for MTP draft rejection - see
gmlx/deepseek_v4_mtp.py. The rotating-cache undo wrap at the bottom of
this file is the matching piece for ``RotatingKVCache`` (port of omlx
patches/mlx_lm_mtp/cache_rollback.py, extended to 3-wide verifies for S=3
rounds).
"""
import sys
import threading
from typing import List

import mlx.core as mx

from mlx_lm.models.cache import _BaseCache


class PoolingCache(_BaseCache):
    """Cache for pooled (compressed) KV tokens with a remainder buffer.

    Stores two things:
      1. A growing pool of compressed tokens (step-allocated).
      2. A small remainder buffer of tokens not yet forming a full window.
    """

    POOL_STEP = 1024  # rows per capacity growth of the pooled buffer

    # Eligible for --kv-bits pooled-row storage. The indexer pool opts out at
    # construction: its rows are read in full every step by the indexer score
    # kernel, so at-rest packing buys no resident-memory headroom worth the
    # per-step dequant traffic (see task #38).
    quantizable = True

    def __init__(self, ratio: int):
        self.ratio = ratio

        self.buf_kv = None
        self.buf_gate = None
        self.remainder = 0

        # Affine-quantized at-rest storage for pooled rows (--kv-bits): the
        # fp16 _pbuf is replaced by packed/scales/biases triples with the
        # same step-allocated watermark discipline. Rows are quantized once
        # on append; fetches dequantize (full pool for prefill/compressed
        # reads, gathered top-k rows for sparse decode). Trim/undo move
        # watermarks only, so they are storage-agnostic.
        self._qbits = None
        self._qgroup = 64

        # Pooled rows live in a step-allocated buffer with a valid-length
        # watermark: appends write in place (donation keeps them O(rows
        # appended)), so window completions stop paying a full-pool copy
        # (the old per-completion concatenate was ~19 MB x 21 layers every
        # 4th token at 64k context). Rollback/replay become watermark
        # moves; rows above the watermark stay intact until overwritten.
        self._pbuf = None
        self._plen = 0
        self._undo = None

        # Cross-call linkage for the ratio-4 overlap compressor: the raw
        # kv/gate projections of the last COMPLETED window. accumulate_windows
        # prepends them to the ready windows so the compressor's cross-window
        # shift links each call's first window to its true predecessor (the
        # ds4.c reference carries the same rows in persistent state); the
        # compressor drops the prepended window's recomputed pooled row.
        self._prev_kv = None
        self._prev_gate = None

    def _lookback_rows(self, B, D1, D2, dt_kv, dt_gate):
        if self._prev_kv is not None:
            return self._prev_kv, self._prev_gate
        # No predecessor yet: zero kv with -inf gate reproduces the
        # compressor's first-window masking of the shifted a-half lanes.
        return (
            mx.zeros((B, self.ratio, D1), dtype=dt_kv),
            mx.full((B, self.ratio, D2), -mx.inf, dtype=dt_gate),
        )

    def quantize_storage(self, group_size: int = 64, bits: int = 8):
        """Arm packed at-rest storage. Free on a fresh cache; with pooled
        rows already landed (serve's conversion-at-start hook) the existing
        rows are packed in place. Idempotent once armed."""
        if self._qbits is not None:
            return
        self._qgroup = int(group_size)
        self._qbits = int(bits)
        if self._plen > 0:
            rows = self._pbuf[:, : self._plen]
            if rows.shape[-1] % self._qgroup != 0 or rows.shape[-1] < 32:
                self._qbits = None  # incompatible width; stay fp16
                return
            self._pbuf = None
            self._plen = 0
            self._undo = None  # replay assumes one storage mode per round
            self.append_pooled(rows)
        else:
            self._pbuf = None

    @property
    def is_quantized(self):
        return self._qbits is not None

    @property
    def pooled(self):
        if self._plen == 0:
            return None
        if self._qbits is not None:
            pk, sc, bi = self._pbuf
            B, P = pk.shape[0], self._plen
            return mx.dequantize(
                pk[:, :P].reshape(B * P, -1),
                sc[:, :P].reshape(B * P, -1),
                bi[:, :P].reshape(B * P, -1),
                group_size=self._qgroup,
                bits=self._qbits,
            ).reshape(B, P, -1)
        if self._pbuf.shape[1] == self._plen:
            return self._pbuf
        return self._pbuf[:, : self._plen]

    @pooled.setter
    def pooled(self, v):
        if v is None:
            self._pbuf = None
            self._plen = 0
        elif self._qbits is not None:
            self._pbuf = None
            self._plen = 0
            self.append_pooled(v)
        else:
            self._pbuf = v
            self._plen = v.shape[1]

    @property
    def offset(self):
        return self._plen

    def accumulate_windows(self, kv: mx.array, gate: mx.array, offset):
        B, L, D1 = kv.shape
        _, _, D2 = gate.shape

        if self.buf_kv is None:
            self.buf_kv = mx.zeros((B, self.ratio, D1), dtype=kv.dtype)
            self.buf_gate = mx.zeros((B, self.ratio, D2), dtype=gate.dtype)

        # One-update undo log for MTP draft rejection: trim() needs the
        # pre-update state plus this update's raw inputs to undo the last
        # token(s) when they completed a pool window. Only decode / MTP-verify
        # sized updates (L <= 4; block-total-B rounds verify B-1+1 wide) are
        # ever trimmed; skipping the stash for prompt chunks avoids pinning
        # large prefill projections. Buffer slices are taken before any
        # mutation, so they reference the pre-update array node.
        # update_and_fetch extends the tuple with the post-append pooled
        # tensor so a confirmed-prefix replay that re-completes a window can
        # slice its pooled row back instead of recompressing (the compressor
        # inputs are gone).
        if L <= 4:
            self._undo = (
                self.buf_kv[:, : self.remainder] if self.remainder > 0 else None,
                self.buf_gate[:, : self.remainder] if self.remainder > 0 else None,
                self.remainder,
                self._plen,
                kv,
                gate,
                self._prev_kv,
                self._prev_gate,
            )
        else:
            self._undo = None

        # Prompt mode
        if L > 1:
            total = L + self.remainder
            usable = (total // self.ratio) * self.ratio
            new_remainder = total % self.ratio

            if usable > 0:
                r_kv = mx.concatenate(
                    [
                        self.buf_kv[:, : self.remainder],
                        kv[:, : (usable - self.remainder)],
                    ],
                    axis=1,
                )
                r_gate = mx.concatenate(
                    [
                        self.buf_gate[:, : self.remainder],
                        gate[:, : (usable - self.remainder)],
                    ],
                    axis=1,
                )
                r_base = offset - self.remainder
                self.remainder = 0
            else:
                r_kv = mx.zeros((B, 0, D1), dtype=kv.dtype)
                r_gate = mx.zeros((B, 0, D2), dtype=gate.dtype)
                r_base = 0

            if new_remainder > 0:
                self.buf_kv[:, self.remainder : new_remainder] = kv[:, -new_remainder:]
                self.buf_gate[:, self.remainder : new_remainder] = gate[
                    :, -new_remainder:
                ]
            self.remainder = new_remainder

            if self.ratio == 4 and r_kv.shape[1] > 0:
                pv_kv, pv_gate = self._lookback_rows(
                    B, D1, D2, kv.dtype, gate.dtype)
                self._prev_kv = r_kv[:, -self.ratio :]
                self._prev_gate = r_gate[:, -self.ratio :]
                r_kv = mx.concatenate([pv_kv, r_kv], axis=1)
                r_gate = mx.concatenate([pv_gate, r_gate], axis=1)
                r_base = r_base - self.ratio

            return r_kv, r_gate, r_base

        # Decode mode
        else:
            self.buf_kv[:, self.remainder : self.remainder + 1] = kv
            self.buf_gate[:, self.remainder : self.remainder + 1] = gate
            self.remainder = (self.remainder + 1) % self.ratio

            if self.remainder == 0:
                r_kv = self.buf_kv
                r_gate = self.buf_gate
                r_base = offset - self.ratio + 1
                if self.ratio == 4:
                    pv_kv, pv_gate = self._lookback_rows(
                        B, D1, D2, kv.dtype, gate.dtype)
                    # Derived nodes, not the buffer object: the next update's
                    # __setitem__ must not mutate the retained window.
                    self._prev_kv = mx.contiguous(self.buf_kv)
                    self._prev_gate = mx.contiguous(self.buf_gate)
                    r_kv = mx.concatenate([pv_kv, r_kv], axis=1)
                    r_gate = mx.concatenate([pv_gate, r_gate], axis=1)
                    r_base = r_base - self.ratio
            else:
                r_kv = mx.zeros((B, 0, D1), dtype=kv.dtype)
                r_gate = mx.zeros((B, 0, D2), dtype=gate.dtype)
                r_base = 0

            return r_kv, r_gate, r_base

    def update_and_fetch(self, px: mx.array):
        if px.shape[1] == 0:
            if self._plen == 0:
                return mx.zeros((px.shape[0], 0, px.shape[-1]), dtype=px.dtype)
            return self.pooled
        self.append_pooled(px)
        return self.pooled

    def append_pooled(self, px: mx.array):
        """Append pooled rows without the dense fetch. Sparse decode uses
        this directly, then reads back only its top-k rows via
        gather_pooled -- under quantized storage a full-pool dequantize per
        step would otherwise erase the memory win's bandwidth side."""
        B, n, D = px.shape
        need = self._plen + n
        if self._qbits is not None and self._pbuf is None and (
            D % self._qgroup != 0 or D < 32
        ):
            # Row width incompatible with mx.quantize (group >= 32, dividing
            # D). Disarm before anything lands; the pool stays fp16.
            import sys

            print(
                f"warning: pooled rows ({D}-wide) cannot pack at group size "
                f"{self._qgroup}; pool stays fp16",
                file=sys.stderr,
            )
            self._qbits = None
        if self._qbits is not None:
            pk, sc, bi = mx.quantize(
                px.reshape(B * n, D),
                group_size=self._qgroup,
                bits=self._qbits,
            )
            new = tuple(t.reshape(B, n, -1) for t in (pk, sc, bi))
            cap = 0 if self._pbuf is None else self._pbuf[0].shape[1]
            if need > cap:
                step = self.POOL_STEP
                grow = ((need - cap + step - 1) // step) * step
                pads = tuple(
                    mx.zeros((B, grow, t.shape[-1]), dtype=t.dtype) for t in new
                )
                self._pbuf = (
                    pads
                    if self._pbuf is None
                    else tuple(
                        mx.concatenate([b, p], axis=1)
                        for b, p in zip(self._pbuf, pads)
                    )
                )
            for b, t in zip(self._pbuf, new):
                b[:, self._plen : need] = t
        else:
            cap = 0 if self._pbuf is None else self._pbuf.shape[1]
            if need > cap:
                step = self.POOL_STEP
                grow = ((need - cap + step - 1) // step) * step
                pad = mx.zeros((B, grow, D), dtype=px.dtype)
                self._pbuf = (
                    pad if self._pbuf is None
                    else mx.concatenate([self._pbuf, pad], axis=1)
                )
            self._pbuf[:, self._plen : need] = px
        self._plen = need
        if self._undo is not None and len(self._undo) == 8:
            # Post-append watermark, for undo replays that re-complete a
            # window (see accumulate_windows): the appended rows stay in
            # the buffer through a rollback, so replay is a watermark move.
            self._undo = self._undo + (need,)

    def gather_pooled(self, topk: mx.array):
        """Dequantized gather of pooled rows by ``(B, L, K)`` indices,
        shaped ``(B, 1, L, K, D)`` to match _sparse_topk_gather."""
        pk, sc, bi = self._pbuf
        B, L, K = topk.shape
        idx = topk.reshape(B, L * K)[..., None]
        parts = [
            mx.take_along_axis(
                t[:, : self._plen],
                mx.broadcast_to(idx, (B, L * K, t.shape[-1])),
                axis=1,
            ).reshape(B * L * K, -1)
            for t in (pk, sc, bi)
        ]
        deq = mx.dequantize(
            parts[0],
            parts[1],
            parts[2],
            group_size=self._qgroup,
            bits=self._qbits,
        )
        return deq.reshape(B, L, K, -1)[:, None]

    def make_mask(self, L: int = 1, offset: int = 0):
        """Build a causal validity mask for pooled positions.

        Query at absolute position ``offset + j`` can attend to pooled token
        ``i`` iff ``i < (offset + j) // ratio``.

        Returns ``(N, P)`` bool mask, or ``None`` when every pooled position
        is visible to every query (common during decode).
        """
        if self._plen == 0 or L == 1:
            return None

        pool_idx = mx.arange(self._plen)
        query_idx = mx.arange(offset + 1, offset + L + 1)
        return pool_idx < query_idx[:, None] // self.ratio

    @property
    def state(self):
        buf_kv = self.buf_kv[:, : self.remainder] if self.remainder > 0 else None
        buf_gate = self.buf_gate[:, : self.remainder] if self.remainder > 0 else None
        return (buf_kv, buf_gate, self.pooled, self._prev_kv, self._prev_gate)

    @state.setter
    def state(self, v):
        if len(v) == 5:
            buf_kv, buf_gate, pooled, prev_kv, prev_gate = v
        else:  # pre-lookback snapshot
            buf_kv, buf_gate, pooled = v
            prev_kv = prev_gate = None
        self.remainder = 0
        self.buf_kv = self.buf_gate = None
        self._prev_kv = prev_kv
        self._prev_gate = prev_gate
        if buf_kv is not None:
            self.accumulate_windows(buf_kv, buf_gate, 0)
        self.pooled = pooled
        self._undo = None

    @property
    def meta_state(self):
        return self.ratio

    @meta_state.setter
    def meta_state(self, v):
        self.ratio = v

    def is_trimmable(self):
        # Trim-by-1 contract (MTP draft rejection): possible while the last
        # token still sits in the remainder buffer, or via the one-update
        # undo log when it completed a pool window.
        if self._plen == 0 or self.remainder >= 1:
            return True
        return self._can_undo(1)

    def _can_trim(self, n):
        """n-aware trimmability probe (MTP rollback two-phase check)."""
        if self._plen == 0 or n <= self.remainder:
            return True
        return self._can_undo(n)

    def _can_undo(self, n):
        undo = self._undo
        if undo is None:
            return False
        k = undo[4].shape[1] - n
        if k < 0:
            return False
        if undo[2] + k < self.ratio:
            # The replayed confirmed prefix stays inside the buffer.
            return True
        # The replay re-completes window(s); their pooled rows are sliced
        # back from the post-append pooled stash (present iff the original
        # update completed at least as many windows, which k < L implies).
        return len(undo) >= 9

    def trim(self, n):
        if n <= self.remainder:
            self.remainder -= n
            self._undo = None
            return n
        if not self._can_undo(n):
            return 0
        undo = self._undo
        buf_kv, buf_gate, rem_prev, plen_prev, kv, gate = undo[:6]
        prev_kv_s, prev_gate_s = undo[6], undo[7]
        self._undo = None
        k = kv.shape[1] - n
        total = rem_prev + k
        if total < self.ratio:
            self._plen = plen_prev
            self.remainder = rem_prev
            self._prev_kv = prev_kv_s
            self._prev_gate = prev_gate_s
            if buf_kv is not None:
                self.buf_kv[:, :rem_prev] = buf_kv
                self.buf_gate[:, :rem_prev] = buf_gate
            if k > 0:
                # Replay the confirmed prefix; it stays in the buffer, so no
                # window is recompressed.
                self.accumulate_windows(kv[:, :k], gate[:, :k], 0)
                self._undo = None
            return n
        # The confirmed prefix re-completes w window(s). Their pooled rows
        # are exactly the first w rows the original update appended
        # (identical inputs) and still sit above the rolled-back watermark,
        # so replay is a watermark move; the remainder buffer is rebuilt
        # from the stashed raw inputs -- no recompression.
        w = total // self.ratio
        self._plen = plen_prev + w
        new_rem = total % self.ratio
        if rem_prev > 0:
            seq_kv = mx.concatenate([buf_kv, kv[:, :k]], axis=1)
            seq_gate = mx.concatenate([buf_gate, gate[:, :k]], axis=1)
        else:
            seq_kv, seq_gate = kv[:, :k], gate[:, :k]
        if self.ratio == 4:
            # The lookback window is the LAST window the replay leaves
            # completed, not the one the undone update installed.
            self._prev_kv = seq_kv[:, (w - 1) * self.ratio : w * self.ratio]
            self._prev_gate = seq_gate[:, (w - 1) * self.ratio : w * self.ratio]
        if new_rem > 0:
            self.buf_kv[:, :new_rem] = seq_kv[:, -new_rem:]
            self.buf_gate[:, :new_rem] = seq_gate[:, -new_rem:]
        self.remainder = new_rem
        return n

    def size(self):
        return self._plen

    def empty(self):
        return self._plen == 0 and self.remainder == 0

    @property
    def nbytes(self):
        total = 0
        if self.buf_kv is not None:
            total += self.buf_kv.nbytes + self.buf_gate.nbytes
        if self._pbuf is not None:
            bufs = self._pbuf if isinstance(self._pbuf, tuple) else (self._pbuf,)
            total += sum(b.nbytes for b in bufs)
        return total

    @classmethod
    def merge(cls, caches):
        return BatchPoolingCache.merge(caches)


class BatchPoolingCache(_BaseCache):
    """Batched pooling cache with per-element variable-length tracking."""

    def __init__(self, ratio: int, left_padding: List[int]):
        self.ratio = ratio

        if not all(p == 0 for p in left_padding):
            raise RuntimeError("BatchPoolingCache does not support left padding")

        batch_size = len(left_padding)

        self.buf_kv = None
        self.buf_gate = None
        self.remainder = [0] * batch_size

        self.pooled = None
        self._pool_lengths = [0] * batch_size

        self._lengths = [2**31] * batch_size
        self._processed = [0] * batch_size
        self._undo = None

        # Per-row lookback window for the ratio-4 overlap compressor (see
        # PoolingCache). Rows that have not completed a window yet hold
        # zero kv with -inf gate (the first-window masking).
        self._prev_kv = None
        self._prev_gate = None

    @property
    def offset(self):
        return mx.array(self._pool_lengths, dtype=mx.int32)

    def prepare(self, *, lengths=None, right_padding=None, left_padding=None):
        if left_padding is not None:
            raise RuntimeError("BatchPoolingCache does not support left padding")
        if lengths is not None:
            self._lengths = [p + n for p, n in zip(self._processed, lengths)]

    def finalize(self):
        self._lengths = [2**31] * len(self._pool_lengths)

    @property
    def is_quantized(self):
        # Batch pools always hold dense rows: merge/extend read through the
        # scalar caches' dequantizing ``pooled`` property.
        return False

    def accumulate_windows(self, kv: mx.array, gate: mx.array, offset):
        B, L, D1 = kv.shape
        _, _, D2 = gate.shape
        ratio = self.ratio

        if self.buf_kv is None:
            self.buf_kv = mx.zeros((B, ratio, D1), dtype=kv.dtype)
            self.buf_gate = mx.zeros((B, ratio, D2), dtype=gate.dtype)

        # One-update undo log for MTP draft rejection (see PoolingCache).
        # The buffer references are only consulted when a window completed,
        # in which case this method rebinds self.buf_* to fresh arrays and
        # the stashed objects keep the pre-update contents. The pooled
        # tensor needs no snapshot: update_and_fetch only writes beyond the
        # old _pool_lengths, so restoring the length lists is enough.
        if L <= 2:
            self._undo = (
                self.buf_kv,
                self.buf_gate,
                list(self.remainder),
                list(self._pool_lengths),
                list(self._processed),
                kv,
                gate,
                self._prev_kv,
                self._prev_gate,
            )
        else:
            self._undo = None

        valid_lengths = [
            min(n - p, L) for n, p in zip(self._lengths, self._processed)]
        if max(valid_lengths) != L:
            raise RuntimeError()
        for i in range(B):
            self._processed[i] += valid_lengths[i]

        totals = [vl + r for vl, r in zip(valid_lengths, self.remainder)]
        usable = [(t // ratio) * ratio for t in totals]
        max_usable = max(usable)
        new_remainder = [t % ratio for t in totals]

        # No sequence produced a full window yet
        if max_usable == 0:
            for i in range(B):
                r = self.remainder[i]
                vl = valid_lengths[i]
                self.buf_kv[i, r : r + vl] = kv[i, :vl]
                self.buf_gate[i, r : r + vl] = gate[i, :vl]
            self.remainder = new_remainder

            r_kv = mx.zeros((B, 0, D1), dtype=kv.dtype)
            r_gate = mx.zeros((B, 0, D2), dtype=gate.dtype)
            r_base = 0
            return r_kv, r_gate, r_base

        # At least one sequence completed a window
        r_kv = mx.zeros((B, max_usable, D1), dtype=kv.dtype)
        r_gate = mx.zeros((B, max_usable, D2), dtype=gate.dtype)
        r_base = [0] * B

        new_buf_kv = mx.zeros_like(self.buf_kv)
        new_buf_gate = mx.zeros_like(self.buf_gate)

        for i in range(B):
            r = self.remainder[i]
            vl = valid_lengths[i]
            u = usable[i]
            nr = new_remainder[i]

            if u > 0:
                # Tokens from the buffer (the leftover from last call)
                if r > 0:
                    r_kv[i, :r] = self.buf_kv[i, :r]
                    r_gate[i, :r] = self.buf_gate[i, :r]

                # Tokens from the new input that complete full windows
                consume = u - r
                r_kv[i, r : r + consume] = kv[i, :consume]
                r_gate[i, r : r + consume] = gate[i, :consume]

                r_base[i] = (
                    offset[i] - r if isinstance(offset, mx.array) else offset - r
                )

            # Fill new remainder buffer from the tail of the input
            if nr > 0:
                if u > 0:
                    # Old remainder was consumed into usable output;
                    # new remainder is purely from the tail of new input.
                    new_buf_kv[i, :nr] = kv[i, vl - nr : vl]
                    new_buf_gate[i, :nr] = gate[i, vl - nr : vl]
                else:
                    # No full window produced: carry over old buffer and
                    # append any new valid tokens.
                    if r > 0:
                        new_buf_kv[i, :r] = self.buf_kv[i, :r]
                        new_buf_gate[i, :r] = self.buf_gate[i, :r]
                    if vl > 0:
                        new_buf_kv[i, r : r + vl] = kv[i, :vl]
                        new_buf_gate[i, r : r + vl] = gate[i, :vl]

        self.buf_kv = new_buf_kv
        self.buf_gate = new_buf_gate
        self.remainder = new_remainder

        if ratio == 4:
            if self._prev_kv is None:
                pv_kv = mx.zeros((B, ratio, D1), dtype=kv.dtype)
                pv_gate = mx.full((B, ratio, D2), -mx.inf, dtype=gate.dtype)
            else:
                pv_kv, pv_gate = self._prev_kv, self._prev_gate
            new_prev_kv = mx.zeros((B, ratio, D1), dtype=kv.dtype)
            new_prev_gate = mx.full((B, ratio, D2), -mx.inf, dtype=gate.dtype)
            for i in range(B):
                u = usable[i]
                if u > 0:
                    new_prev_kv[i] = r_kv[i, u - ratio : u]
                    new_prev_gate[i] = r_gate[i, u - ratio : u]
                else:
                    new_prev_kv[i] = pv_kv[i]
                    new_prev_gate[i] = pv_gate[i]
            r_kv = mx.concatenate([pv_kv, r_kv], axis=1)
            r_gate = mx.concatenate([pv_gate, r_gate], axis=1)
            r_base = [rb - ratio for rb in r_base]
            self._prev_kv = new_prev_kv
            self._prev_gate = new_prev_gate

        r_base = mx.array(r_base)
        return r_kv, r_gate, r_base

    def update_and_fetch(self, px: mx.array):
        B, N, D = px.shape

        if N == 0:
            if self.pooled is None:
                return mx.zeros((B, 0, D), dtype=px.dtype)
            return self.pooled

        # Derive how many new pooled tokens each sequence actually produced.
        new_counts = [
            (self._processed[i] - self.remainder[i]) // self.ratio
            - self._pool_lengths[i]
            for i in range(B)
        ]
        max_new = max(new_counts)
        if max_new == 0:
            if self.pooled is None:
                return mx.zeros((B, 0, D), dtype=px.dtype)
            return self.pooled

        max_pool = max(self._pool_lengths) + max_new

        if self.pooled is None:
            self.pooled = mx.zeros((B, max_pool, D), dtype=px.dtype)
        elif self.pooled.shape[1] < max_pool:
            pad = mx.zeros((B, max_pool - self.pooled.shape[1], D), dtype=px.dtype)
            self.pooled = mx.concatenate([self.pooled, pad], axis=1)

        for i in range(B):
            nc = new_counts[i]
            if nc > 0:
                pl = self._pool_lengths[i]
                self.pooled[i, pl : pl + nc] = px[i, :nc]
                self._pool_lengths[i] = pl + nc

        return self.pooled

    def make_mask(self, L: int = 1, offset=0):
        if self.pooled is None:
            return None

        B, P, _ = self.pooled.shape
        pool_lengths = mx.array(self._pool_lengths)

        # Length based mask
        pool_idx = mx.arange(P)[None, None, :]
        valid = pool_idx < pool_lengths[:, None, None]

        # Decode so no need for causal masking
        if L == 1:
            if all(pl == P for pl in self._pool_lengths):
                return None
            return valid

        # Prompt so we need to combine with causal
        if isinstance(offset, mx.array):
            query_pos = offset[:, None] + mx.arange(1, L + 1)
        else:
            query_pos = offset + mx.arange(1, L + 1)[None]

        causal = pool_idx < (query_pos[..., None] // self.ratio)
        mask = causal & valid
        return mask

    @property
    def state(self):
        return (self.buf_kv, self.buf_gate, self.pooled,
                self._prev_kv, self._prev_gate)

    @state.setter
    def state(self, v):
        if len(v) == 5:
            (self.buf_kv, self.buf_gate, self.pooled,
             self._prev_kv, self._prev_gate) = v
        else:  # pre-lookback snapshot
            self.buf_kv, self.buf_gate, self.pooled = v
            self._prev_kv = self._prev_gate = None
        self._undo = None

    @property
    def meta_state(self):
        return (self.ratio, self.remainder, self._pool_lengths, self._processed)

    @meta_state.setter
    def meta_state(self, v):
        self.ratio, self.remainder, self._pool_lengths, self._processed = v

    def is_trimmable(self):
        # Trim-by-1 contract (MTP draft rejection): possible while every
        # row's last token still sits in the remainder buffer, or via the
        # one-update undo log when a row completed a pool window.
        if self.pooled is None or min(self.remainder) >= 1:
            return True
        return self._can_undo(1)

    def _can_undo(self, n):
        undo = self._undo
        if undo is None:
            return False
        k = undo[5].shape[1] - n
        # The replayed confirmed prefix must stay inside the buffer for
        # every row (a replay that pools again cannot be reconstructed).
        return k >= 0 and all(r + k < self.ratio for r in undo[2])

    def trim(self, n):
        if n <= min(self.remainder):
            for i in range(len(self.remainder)):
                self.remainder[i] -= n
                self._processed[i] -= n
            self._undo = None
            return n
        if not self._can_undo(n):
            return 0
        (buf_kv, buf_gate, remainder, pool_lengths, processed, kv, gate,
         prev_kv, prev_gate) = self._undo
        self._undo = None
        k = kv.shape[1] - n
        # The undo path only triggers when some row completed a window,
        # which rebinds self.buf_* to fresh arrays - the stashed objects
        # still hold the pre-update contents. The pooled tensor keeps any
        # extra written rows; restoring _pool_lengths masks them out.
        self.buf_kv = buf_kv
        self.buf_gate = buf_gate
        self._prev_kv = prev_kv
        self._prev_gate = prev_gate
        self.remainder = list(remainder)
        self._pool_lengths = list(pool_lengths)
        self._processed = list(processed)
        if k > 0:
            # Replay the confirmed prefix; _can_undo guarantees it stays in
            # the buffer, so no window is recompressed.
            self.accumulate_windows(kv[:, :k], gate[:, :k], 0)
            self._undo = None
        return n

    def size(self):
        return 0 if self.pooled is None else self.pooled.shape[1]

    def empty(self):
        return self.pooled is None and all(r == 0 for r in self.remainder)

    @property
    def nbytes(self):
        total = 0
        if self.buf_kv is not None:
            total += self.buf_kv.nbytes + self.buf_gate.nbytes
        if self.pooled is not None:
            total += self.pooled.nbytes
        return total

    def filter(self, batch_indices):
        if isinstance(batch_indices, mx.array):
            idx_list = batch_indices.tolist()
        else:
            idx_list = list(batch_indices)

        if self.buf_kv is not None:
            self.buf_kv = self.buf_kv[batch_indices]
            self.buf_gate = self.buf_gate[batch_indices]
        if self.pooled is not None:
            self.pooled = self.pooled[batch_indices]
        if self._prev_kv is not None:
            self._prev_kv = self._prev_kv[batch_indices]
            self._prev_gate = self._prev_gate[batch_indices]

        self.remainder = [self.remainder[i] for i in idx_list]
        self._pool_lengths = [self._pool_lengths[i] for i in idx_list]
        self._lengths = [self._lengths[i] for i in idx_list]
        self._processed = [self._processed[i] for i in idx_list]

    def extend(self, other):
        # Merge the remainder buffers
        if self.buf_kv is None and other.buf_kv is None:
            pass
        elif self.buf_kv is not None and other.buf_kv is not None:
            self.buf_kv = mx.concatenate([self.buf_kv, other.buf_kv], axis=0)
            self.buf_gate = mx.concatenate([self.buf_gate, other.buf_gate], axis=0)
        elif self.buf_kv is None:
            B = len(self.remainder)
            D1 = other.buf_kv.shape[2]
            D2 = other.buf_gate.shape[2]
            self.buf_kv = mx.concatenate(
                [mx.zeros((B, self.ratio, D1), dtype=other.buf_kv.dtype), other.buf_kv],
                axis=0,
            )
            self.buf_gate = mx.concatenate(
                [
                    mx.zeros((B, self.ratio, D2), dtype=other.buf_gate.dtype),
                    other.buf_gate,
                ],
                axis=0,
            )
        else:
            B2 = len(other.remainder)
            D1 = self.buf_kv.shape[2]
            D2 = self.buf_gate.shape[2]
            self.buf_kv = mx.concatenate(
                [self.buf_kv, mx.zeros((B2, self.ratio, D1), dtype=self.buf_kv.dtype)],
                axis=0,
            )
            self.buf_gate = mx.concatenate(
                [
                    self.buf_gate,
                    mx.zeros((B2, self.ratio, D2), dtype=self.buf_gate.dtype),
                ],
                axis=0,
            )

        # Merge the pooled buffers
        if self.pooled is None and other.pooled is None:
            pass
        else:
            B1 = len(self.remainder)
            B2 = len(other.remainder)
            P1 = 0 if self.pooled is None else self.pooled.shape[1]
            P2 = 0 if other.pooled is None else other.pooled.shape[1]
            max_P = max(P1, P2)

            if max_P > 0:
                if self.pooled is not None:
                    D = self.pooled.shape[2]
                else:
                    D = other.pooled.shape[2]
                dt = (self.pooled if self.pooled is not None else other.pooled).dtype

                def pad_pool(pooled, B, P):
                    if pooled is None:
                        return mx.zeros((B, max_P, D), dtype=dt)
                    if P < max_P:
                        pad = mx.zeros((pooled.shape[0], max_P - P, D), dtype=dt)
                        return mx.concatenate([pooled, pad], axis=1)
                    return pooled

                self.pooled = mx.concatenate(
                    [pad_pool(self.pooled, B1, P1), pad_pool(other.pooled, B2, P2)],
                    axis=0,
                )

        # Merge the lookback windows (rows without one hold the zero-kv /
        # -inf-gate first-window masking).
        if self._prev_kv is not None or other._prev_kv is not None:
            B1 = len(self.remainder)
            B2 = len(other.remainder)
            src = self._prev_kv if self._prev_kv is not None else other._prev_kv
            src_g = (self._prev_gate if self._prev_gate is not None
                     else other._prev_gate)

            def pad_prev(prev, prev_gate, B):
                if prev is not None:
                    return prev, prev_gate
                return (
                    mx.zeros((B,) + src.shape[1:], dtype=src.dtype),
                    mx.full((B,) + src_g.shape[1:], -mx.inf, dtype=src_g.dtype),
                )

            pk1, pg1 = pad_prev(self._prev_kv, self._prev_gate, B1)
            pk2, pg2 = pad_prev(other._prev_kv, other._prev_gate, B2)
            self._prev_kv = mx.concatenate([pk1, pk2], axis=0)
            self._prev_gate = mx.concatenate([pg1, pg2], axis=0)

        self.remainder = self.remainder + other.remainder
        self._pool_lengths = self._pool_lengths + other._pool_lengths
        self._lengths = self._lengths + other._lengths
        self._processed = self._processed + other._processed

    def extract(self, idx):
        cache = PoolingCache(self.ratio)
        pl = self._pool_lengths[idx]
        r = self.remainder[idx]

        if self.pooled is not None and pl > 0:
            cache.pooled = mx.contiguous(self.pooled[idx : idx + 1, :pl])

        if self.buf_kv is not None and r > 0:
            cache.buf_kv = mx.contiguous(self.buf_kv[idx : idx + 1])
            cache.buf_gate = mx.contiguous(self.buf_gate[idx : idx + 1])
            cache.remainder = r

        if self._prev_kv is not None and pl > 0:
            cache._prev_kv = mx.contiguous(self._prev_kv[idx : idx + 1])
            cache._prev_gate = mx.contiguous(self._prev_gate[idx : idx + 1])

        return cache

    @classmethod
    def merge(cls, caches):
        """Merge a list of PoolingCache instances into a BatchPoolingCache."""
        B = len(caches)
        if not all(c.ratio == caches[0].ratio for c in caches):
            raise ValueError(
                "BatchPoolingCache can only merge caches with the same ratio"
            )
        ratio = caches[0].ratio
        batch_cache = cls(ratio, [0] * B)

        # Check if all caches are empty
        if all(c.empty() for c in caches):
            return batch_cache

        # Merge pooled buffers
        pool_sizes = [c.size() for c in caches]
        max_pool = max(pool_sizes)
        if max_pool > 0:
            D = next(c.pooled.shape[2] for c in caches if c.pooled is not None)
            dt = next(c.pooled.dtype for c in caches if c.pooled is not None)
            pooled = mx.zeros((B, max_pool, D), dtype=dt)
            for i, c in enumerate(caches):
                if c.pooled is not None:
                    ps = c.pooled.shape[1]
                    pooled[i, :ps] = c.pooled[0]
            batch_cache.pooled = pooled

        batch_cache._pool_lengths = pool_sizes
        batch_cache.remainder = [c.remainder for c in caches]
        batch_cache._processed = [
            c.remainder + ps * ratio for c, ps in zip(caches, pool_sizes)
        ]

        # Merge remainder buffers
        has_buf = any(c.buf_kv is not None for c in caches)
        if has_buf:
            D1 = next(c.buf_kv.shape[2] for c in caches if c.buf_kv is not None)
            D2 = next(c.buf_gate.shape[2] for c in caches if c.buf_gate is not None)
            dt = next(c.buf_kv.dtype for c in caches if c.buf_kv is not None)
            buf_kv = mx.zeros((B, ratio, D1), dtype=dt)
            buf_gate = mx.zeros((B, ratio, D2), dtype=dt)
            for i, c in enumerate(caches):
                if c.buf_kv is not None and c.remainder > 0:
                    buf_kv[i, : c.remainder] = c.buf_kv[0, : c.remainder]
                    buf_gate[i, : c.remainder] = c.buf_gate[0, : c.remainder]
            batch_cache.buf_kv = buf_kv
            batch_cache.buf_gate = buf_gate

        # Merge lookback windows
        if any(c._prev_kv is not None for c in caches):
            src = next(c._prev_kv for c in caches if c._prev_kv is not None)
            src_g = next(c._prev_gate for c in caches if c._prev_gate is not None)
            prev_kv = mx.zeros((B,) + src.shape[1:], dtype=src.dtype)
            prev_gate = mx.full((B,) + src_g.shape[1:], -mx.inf,
                                dtype=src_g.dtype)
            for i, c in enumerate(caches):
                if c._prev_kv is not None:
                    prev_kv[i] = c._prev_kv[0]
                    prev_gate[i] = c._prev_gate[0]
            batch_cache._prev_kv = prev_kv
            batch_cache._prev_gate = prev_gate

        return batch_cache


# MTP rotating-cache undo log (port of omlx patches/mlx_lm_mtp/
# cache_rollback.py, verbatim semantics).
#
# A rotated RotatingKVCache is not trimmable (the evicted token's slot has
# been overwritten), so an MTP draft rejection could not roll back the
# verify write and the rejected token(s) would stay in the cache as
# phantoms, progressively corrupting output (hit on DeepSeek-V4-Flash,
# sliding_window=128). Verify-width (2- or 3-token; S<=3 rounds) updates
# always take the ``_update_concat`` path, which only rebinds
# ``keys``/``values`` (no in-place setitem), so stashing the pre-update
# attribute references plus the update's inputs gives an exact undo;
# ``trim(n)`` then replays the confirmed prefix as a normal update.
# Stashing is armed ONLY around the V4 MTP verify forward
# (DeepseekV4SpecLM.speculative_verify_hidden) so non-MTP flows keep stock
# trim semantics (protects prefix-cache trim paths from phantom
# trimmability).

# Thread-local so concurrent engine steps of other models never see the
# flag armed by an MTP verify forward running on a different thread.
_UNDO_ARMED = threading.local()


def set_undo_armed(flag: bool) -> None:
    """Arm/disarm the rotating-cache undo stash (MTP verify forwards only)."""
    _UNDO_ARMED.value = bool(flag)


def _is_undo_armed() -> bool:
    # Resolve the flag through sys.modules: the wrapped methods live on
    # foreign (mlx-lm) classes and outlive this module object if it is ever
    # re-imported (e.g. tests that patch.dict sys.modules), so a closure
    # over this instance's _UNDO_ARMED could go stale.
    mod = sys.modules.get(__name__)
    armed = getattr(mod, "_UNDO_ARMED", None) if mod is not None else None
    if armed is None:
        armed = _UNDO_ARMED
    return getattr(armed, "value", False)


def _wrap_rotating(cls, fields) -> None:
    """Wrap update_and_fetch / is_trimmable / trim with the MTP undo log."""
    if getattr(cls, "_gmlx_mtp_undo_attached", False):
        return

    orig_update = cls.update_and_fetch
    orig_is_trimmable = cls.is_trimmable
    orig_trim = cls.trim

    def update_and_fetch(self, keys, values):
        # Only armed verify-sized updates (qL 2-4) are undoable: S == 1
        # uses the in-place ring write (setitem mutates the wrapper, which
        # would invalidate reference snapshots) and prompt chunks have no
        # rollback consumer. S >= 2 always routes through the concat path,
        # which REBINDS keys/values to fresh wrappers, so plain references
        # keep the pre-update value -- no detach copy (a full-ring copy per
        # field per layer, ~200 MB/round across 43 caches).
        if keys.shape[2] in (2, 3, 4) and _is_undo_armed():
            self._mtp_undo = ({f: getattr(self, f) for f in fields}, keys, values)
        else:
            self._mtp_undo = None
        return orig_update(self, keys, values)

    def is_trimmable(self):
        if orig_is_trimmable(self):
            return True
        return getattr(self, "_mtp_undo", None) is not None

    def trim(self, n):
        if orig_is_trimmable(self):
            self._mtp_undo = None
            return orig_trim(self, n)
        undo = getattr(self, "_mtp_undo", None)
        self._mtp_undo = None
        if undo is None:
            return 0
        snap, keys, values = undo
        k = keys.shape[2] - n
        if k < 0:
            return 0
        for f, v in snap.items():
            setattr(self, f, v)
        if k > 0:
            # Replay the confirmed prefix as a normal decode-sized update.
            orig_update(self, keys[..., :k, :], values[..., :k, :])
            self._mtp_undo = None
        return n

    cls.update_and_fetch = update_and_fetch
    cls.is_trimmable = is_trimmable
    cls.trim = trim
    cls._mtp_undo = None
    cls._gmlx_mtp_undo_attached = True


def ensure_rollback_attached() -> None:
    """Attach the MTP undo log to every loaded origin's rotating caches
    (idempotent per class; mlx-vlm vendored its own since 0.6.4)."""
    from .cache_compat import cache_types

    for cls in cache_types("RotatingKVCache"):
        _wrap_rotating(cls, ("keys", "values", "offset", "_idx"))
    try:
        batch_classes = cache_types("BatchRotatingKVCache")
    except AttributeError:
        return
    for cls in batch_classes:
        _wrap_rotating(
            cls,
            ("keys", "values", "offset", "_offset", "_idx", "rotated",
             "left_padding"),
        )
