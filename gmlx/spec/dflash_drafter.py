# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
# Portions copyright (c) 2026 Prince Canuma (module tree after mlx-vlm's
# DFlashDraftModel, MIT)
"""Owned DFlash drafters: the block-diffusion base and the DFlash 2 extras.

DFlash denoises a whole block at once. Its decoder runs in two modes
(llama.cpp ``src/models/dflash.cpp``):

- inject: the target's captured residuals go through the encoder
  ``hidden_norm(fc(captures))``, and each layer projects that straight to
  K/V (k_norm + RoPE, no ``input_layernorm``, no query, no FFN) and writes it
  into the draft KV ring at the committed tokens' positions.
- draft: ``[last_bonus, MASK x (block-1)]`` is embedded with the target's
  ``tok_embd``, positioned at ``n_past + i``, and run through the layers with
  non-causal attention over the ring plus the block's own transient K/V.
  Drafts are read from rows 1..block-1, and the target's ``lm_head``
  produces the logits.

gmlx's engine splits the two modes across ``prefill_from_target_hidden`` /
``accept_verified_tokens`` (inject) and ``draft_block`` (draft).

DFlash 2 adds a grouped dynamic causal convolution around every attention
and MLP sublayer of the draft path and a candidate selector that walks a path
through the top-k candidates of each block position.

DSpark on the DFlash backbone (the Ternary Bonsai drafters) keeps the plain
layers and adds two heads over the draft rows: a rank-``markov_rank`` bigram
head that biases each row's logits with ``markov_w2(markov_w1[prev])``,
chained through the block by the row picks, and a confidence head
``sigmoid(conf_proj([row; markov_w1[prev]]))`` that can cut the block to its
confident prefix. The anchor row drafts too, so a block of ``rows`` rows
proposes ``rows`` tokens. A drafter trained with log-SNR conditioning adds a
per-row noise-level embedding to the token embedding before the layers.

Sliding layers keep a temporal ring of ``sliding_window - 1`` context rows
(plus rollback slack) and mask the block the way the reference does: a block
row at ring offset ``q`` sees context key ``k`` only while ``q - k <
sliding_window``. A missing ``attention.causal`` key means a non-causal block,
which is llama.cpp's behavior.

A drafter with a context window (``attention.context_window``) bounds its
full layers too: every block row sees the last ``context_window`` committed
rows, the same span for each row. A ``rope_pattern`` entry of false makes a
layer NoPE on both paths.

Positions are relative: seeding only the trailing window of the prompt
shifts context and block queries by the same amount, which RoPE is invariant
to. Correctness never rests on the drafter regardless: the verify walk emits
the target's own tokens, so the drafter moves acceptance, never output.

Stochastic acceptance invariant: the q row a drafter stashes must be built
from the same probability array its token was drawn from. Both drafters are
lossless under that rule. The v1 drafter draws block positions independently
given the forward (which sees only the anchor and MASK tokens), so position
j's row is its proposal conditional on any accepted prefix. DFlash 2's row j
is a genuine conditional on the realized predecessor (the lattice row of the
token drawn at j-1), which is the proposal the verifier divides by.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from gmlx._vendor.mlx_vlm.models.activations import swiglu
from gmlx._vendor.mlx_vlm.models.rope_utils import initialize_rope

from . import topk
from .drafter_protocol import DraftStash, native_block_size
from gmlx.cache.kv_cache import BufferedRotatingKVCache, KVCache
from gmlx.envflags import env_bool, env_float, env_int

_LAYER_TYPES = ("full_attention", "sliding_attention")
# Rollback slack rows the temporal ring keeps beyond its window.
_RING_SLACK = max(16, env_int("GMLX_DFLASH_RING_SLACK", 512))
# Draft attention reads the ring's buffer in place (DFlashAttention.
# _ring_in_place). GMLX_DFLASH_RING_IN_PLACE=0 copies the ring per layer.
_RING_IN_PLACE = env_bool("GMLX_DFLASH_RING_IN_PLACE", True)
# A counted draft (GMLX_SPEC_ACCEPT_COUNTED) runs its attention in float32.
_COUNTED_F32 = env_bool("GMLX_DFLASH_COUNTED_F32", True)
# A counted draft reads the ring by row index in the plain draft's key
# order (1), which keeps the plain attention kernel; 0 is the array mask
# over the buffer rows.
_COUNTED_GATHER = env_bool("GMLX_DFLASH_COUNTED_GATHER", True)
# The counted draft on a ring still shorter than the sliding window (a
# context under about one window): masked rows in place of the plain key
# count, equal to the plain draft to rounding only.
_COUNTED_SHORT = env_bool("GMLX_DFLASH_COUNTED_SHORT", True)
# Projections that share an input (q/k/v, gate/up) run as one dispatch per
# codec at block widths. GMLX_DFLASH_QMM_MULTI=0 runs one per weight.
_QMM_MULTI = env_bool("GMLX_DFLASH_QMM_MULTI", True)
# The selector walk (picks, proposal rows, row cut) runs as one kernel.
# GMLX_DFLASH_WALK_KERNEL=0 keeps the op chain.
_WALK_KERNEL = env_bool("GMLX_DFLASH_WALK_KERNEL", True)
# The walk kernel also scores the lattice rows of the path it takes.
# GMLX_DFLASH_LATTICE_WALK=0 builds the whole lattice first.
_LATTICE_WALK = env_bool("GMLX_DFLASH_LATTICE_WALK", True)
# A verified round's commit starts on the GPU when it is built, so it runs
# while the host books the round and builds the next draft.
# GMLX_DFLASH_COMMIT_ASYNC=0 leaves it to run ahead of the next draft.
_COMMIT_ASYNC = env_bool("GMLX_DFLASH_COMMIT_ASYNC", True)
# The lattice takes its candidates from the hierarchical top-k.
# GMLX_DFLASH_LATTICE_TOPK=0 runs a full-width argpartition.
_LATTICE_TOPK = env_bool("GMLX_DFLASH_LATTICE_TOPK", True)
# DFlash 2 drafts skip the head rows of the longest run of token ids that
# the request's context has not used (DFlash2Drafter._head_skip).
# GMLX_DFLASH_HEAD_SUBSET=0 reads the whole head every round.
_HEAD_SUBSET = env_bool("GMLX_DFLASH_HEAD_SUBSET", True)
# GMLX_SPEC_APC_LONGER=1: the drafter keeps the target hidden of the rows
# its ring was fed, so a finished request can leave the last
# hidden_capture_limit of them beside its retire record.
_KEEP_TAIL = env_bool("GMLX_SPEC_APC_LONGER", False)
# Head rows are tracked in blocks of this many token ids.
_HEAD_BLOCK = 128
# A run shorter than this share of the vocabulary is not skipped.
_HEAD_SKIP_MIN = 1 / 8


class _Ring(list):
    """A drafter context: the per-layer caches plus the token-id blocks the
    context holds (``occ``, None until the prompt ids are known) and the id
    range the head skips (``skip``)."""

    occ: Optional[np.ndarray] = None
    skip: Optional[tuple] = None


def _note_tokens(ring, tokens) -> None:
    """Mark the blocks of ``tokens`` used in ``ring``. Drops the skipped run
    when a token lands inside it, so the next draft picks a new one."""
    occ = getattr(ring, "occ", None)
    if occ is None:
        return
    blk = np.asarray(tokens, dtype=np.int64).reshape(-1) // _HEAD_BLOCK
    blk = blk[(blk >= 0) & (blk < occ.shape[0])]
    if blk.size == 0:
        return
    skip = ring.skip                    # () once no run is long enough
    if skip and bool(((blk * _HEAD_BLOCK >= skip[0])
                                  & (blk * _HEAD_BLOCK < skip[1])).any()):
        ring.skip = None
    occ[blk] = True


class HiddenTail:
    """The packed target hidden behind a ring's last ``limit`` rows: the
    prompt capture, then each accept's committed rows, as (array, rows)
    pieces that are joined only when a request retires."""

    def __init__(self, limit: int):
        self.limit = int(limit)
        self.parts: list = []
        self.rows = 0
        self.round_rows = 0

    def _add(self, hidden: mx.array, rows: int) -> None:
        self.parts.append([hidden, int(rows)])
        self.rows += int(rows)
        while len(self.parts) > 1 and self.rows - self.parts[0][1] >= self.limit:
            self.rows -= self.parts.pop(0)[1]

    def prompt(self, hidden: mx.array) -> None:
        self.parts, self.rows, self.round_rows = [], 0, 0
        self._add(hidden, int(hidden.shape[1]))

    def accept(self, hidden: mx.array, rows: int) -> None:
        self._add(hidden, rows)
        self.round_rows += int(rows)

    def trim(self, dead: int) -> None:
        """Take ``dead`` rows of the last accept back."""
        dead = int(dead)
        if dead <= 0 or not self.parts:
            return
        if dead > self.parts[-1][1]:
            # Not the shape of one accept: nothing here can be trusted.
            self.parts, self.rows, self.round_rows = [], 0, -1
            return
        self.parts[-1][1] -= dead
        self.rows -= dead
        self.round_rows -= dead

    def copy(self) -> "HiddenTail":
        """A copy the drafter's later requests leave as it is."""
        t = HiddenTail(self.limit)
        t.parts = [[h, n] for h, n in self.parts]
        t.rows, t.round_rows = self.rows, self.round_rows
        return t

    def take(self, prompt_len: int, end: int) -> Optional[mx.array]:
        """The hidden of the positions ending at ``end`` (at most ``limit``
        of them), given the prompt's length, or None when the pieces do
        not reach ``end``."""
        if self.round_rows < 0 or not self.parts:
            return None
        drop = int(prompt_len) + self.round_rows - int(end)
        if drop < 0 or drop >= self.rows:
            return None
        pieces = [h[:, :n] for h, n in self.parts if n > 0]
        out = pieces[0] if len(pieces) == 1 else mx.concatenate(pieces, axis=1)
        if drop:
            out = out[:, :-drop]
        return mx.contiguous(out[:, -self.limit:])


def _open_ring(ring, vocab: int, tokens) -> None:
    """Start the occupancy of ``ring`` from a prompt's token ids."""
    if tokens is None:
        return
    if isinstance(tokens, mx.array):
        tokens = np.array(tokens)
    ring.occ = np.zeros((vocab + _HEAD_BLOCK - 1) // _HEAD_BLOCK, dtype=bool)
    ring.skip = None
    _note_tokens(ring, tokens)


def _longest_free_run(occ: np.ndarray, vocab: int) -> Optional[tuple]:
    """``(lo, hi)`` token ids of the longest run of unused blocks, or None
    when it is shorter than ``_HEAD_SKIP_MIN`` of the vocabulary."""
    free = np.concatenate([[False], ~occ, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(free))
    if edges.size == 0:
        return None
    starts, ends = edges[0::2], edges[1::2]
    i = int(np.argmax(ends - starts))
    lo, hi = int(starts[i]) * _HEAD_BLOCK, min(vocab, int(ends[i]) * _HEAD_BLOCK)
    return (lo, hi) if hi - lo >= _HEAD_SKIP_MIN * vocab else None


def head_rows(head, x: mx.array, lo: int, hi: int) -> Optional[mx.array]:
    """``head(x)`` without the output columns ``[lo, hi)``: the rows before
    and after the range, joined. None for a head form it cannot slice,
    which then runs whole. Weight row slices are views, so no weight is
    copied."""
    obj = getattr(head, "__self__", None)
    if obj is not None and getattr(head, "__name__", "") == "as_linear":
        mod = obj                      # a tied head: the embedding's rows
    elif isinstance(head, nn.Module):
        mod = head
    else:
        return None
    names = set(mod.keys()) if hasattr(mod, "keys") else set()
    w = mod.get("weight") if "weight" in names else None
    if w is None or w.ndim != 2:
        return None
    n = int(w.shape[0])
    if not 0 <= lo < hi <= n:
        return None
    parts = [(0, lo), (hi, n)]
    kind = type(mod).__name__
    if kind in ("KQuantLinear", "KQuantEmbedding"):
        import mlx_kquant as kq

        if names - {"weight", "scales", "bias", "out_scale"}:
            return None
        bias, osc = mod.get("bias"), mod.get("out_scale")

        def run(a, b):  # pyright: ignore[reportRedeclaration]
            kw = {}
            if osc is not None:
                kw["out_scale"] = osc if osc.size == 1 else osc[a:b]
            y = kq.quantized_matmul(x, w[a:b], mod["scales"], mod.kquant_type,
                                    transpose=True, **kw)
            return y if bias is None else y + bias[a:b]
    elif kind in ("QuantizedLinear", "QuantizedEmbedding"):
        if names - {"weight", "scales", "biases", "bias"}:
            return None
        sc, bi, bias = mod["scales"], mod.get("biases"), mod.get("bias")

        def run(a, b):  # pyright: ignore[reportRedeclaration]
            y = mx.quantized_matmul(
                x, w[a:b], scales=sc[a:b],
                biases=None if bi is None else bi[a:b], transpose=True,
                group_size=mod.group_size, bits=mod.bits, mode=mod.mode)  # pyright: ignore[reportArgumentType]
            return y if bias is None else y + bias[a:b]
    elif kind in ("Linear", "Embedding") and mx.issubdtype(w.dtype, mx.floating):
        if names - {"weight", "bias"}:
            return None
        bias = mod.get("bias")

        def run(a, b):
            y = x @ w[a:b].T
            return y if bias is None else y + bias[a:b]
    else:
        return None
    return mx.concatenate([run(a, b) for a, b in parts if b > a], axis=-1)


def _shared_input(linears, x: mx.array) -> list:
    """Each of ``linears`` applied to ``x``. Quantized weights of one codec
    go through one multi-weight matmul where mlx-kquant has a route for the
    row count; every other member runs on its own."""
    if _QMM_MULTI:
        from gmlx.upstream.gdn_patches import kquant_multi

        out = kquant_multi(linears, x)
        if out is not None:
            return [lin(x) if o is None else o for lin, o in zip(linears, out)]
    return [lin(x) for lin in linears]


@dataclass
class DFlashConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    rope_theta: float
    block_size: int
    mask_token_id: int
    target_layer_ids: List[int]
    num_target_layers: int
    rope_scaling: Optional[dict] = None
    # The rope scaling of the drafter's own header (gmlx.load.rope_scaling),
    # present only when it changes the rope.
    gmlx_rope_scaling: Optional[dict] = None
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    # The checkpoint's trained diffusion block: the deepest block the drafter
    # can produce. None leaves block_size as the ceiling.
    native_block_size: Optional[int] = None
    # The block total to draft when the run asks for none; None drafts
    # block_size.
    runtime_block_size: Optional[int] = None
    layer_types: List[str] = field(default_factory=list)
    sliding_window: Optional[int] = None
    # None = non-causal block (llama.cpp); the converter's attention.causal
    # key sets it explicitly.
    is_causal: Optional[bool] = None
    final_logit_softcapping: Optional[float] = None
    output_multiplier: float = 1.0
    input_embedding_scale: float = 1.0
    # DFlash 2 only; selector_top_k == 0 means a v1 drafter.
    conv_kernel_size: int = 0
    conv_group_size: int = 0
    selector_rank: int = 0
    selector_top_k: int = 0
    # DSpark heads on the DFlash backbone; markov_rank == 0 means none.
    markov_rank: int = 0
    confidence_head: bool = False
    # (min, max) log-SNR of a drafter trained with noise-level conditioning.
    log_snr_range: Optional[tuple] = None
    # The anchor row proposes a token as well (llama.cpp's default).
    sample_from_anchor: bool = True
    # Rows of the target's head and token tables when they pad past
    # vocab_size (Inkling 2); None means vocab_size.
    padded_vocab_size: Optional[int] = None
    # Per-head attention sinks in every draft layer (Nemotron drafters).
    attention_sinks: bool = False
    # Per layer, whether it rotates; empty means every layer does.
    rope_layers: List[bool] = field(default_factory=list)
    # Context rows every block row of a full layer sees; None is unbounded.
    context_window: Optional[int] = None
    # The LiLiCorr lattice head; lilicorr_layers == 0 means none.
    lilicorr_layers: int = 0
    lilicorr_hidden_size: int = 0
    lilicorr_top_k: int = 0
    lilicorr_heads: int = 0
    lilicorr_factor_dim: int = 0
    lilicorr_mlp_size: int = 0
    lilicorr_vector_eps: float = 1e-4
    lilicorr_logit_scale: float = 1.0

    def __post_init__(self):
        n = int(self.num_hidden_layers)
        if not self.layer_types:
            self.layer_types = ["full_attention"] * n
        self.layer_types = list(self.layer_types)
        if len(self.layer_types) != n:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries for "
                f"{n} layers")
        unknown = set(self.layer_types) - set(_LAYER_TYPES)
        if unknown:
            raise ValueError(f"unsupported draft layer_types {sorted(unknown)}")
        if "sliding_attention" in self.layer_types and not self.sliding_window:
            raise ValueError("sliding_attention layers need sliding_window")
        self.rope_layers = [bool(v) for v in self.rope_layers]
        if self.rope_layers and len(self.rope_layers) != n:
            raise ValueError(
                f"rope_layers has {len(self.rope_layers)} entries for "
                f"{n} layers")
        if self.context_window is not None and int(self.context_window) <= 0:
            raise ValueError("context_window must be positive")

    @property
    def is_dflash2(self) -> bool:
        return int(self.selector_top_k) > 0

    @property
    def is_dspark(self) -> bool:
        return int(self.markov_rank) > 0


def block_attention_mask(ctx_len: int, block: int, window: Optional[int],
                         causal: bool, context: Optional[int] = None):
    """Bool mask ``[block, ctx_len + block]`` for a draft block over a ring.

    Ring rows are indexed as time (the ring is temporal), so a block row at
    ``q = ctx_len + i`` keeps context key ``k`` while ``q - k < window``; the
    block term is all-true, or lower-triangular when causal. ``context``
    keeps the last ``context`` ring rows for every block row alike. None when
    no key would be masked. Hand-rolled: ``BufferedRotatingKVCache.make_mask``
    is causal and sizes its window from a planned insert of the query rows,
    which the draft block never is.
    """
    if not causal:
        open_window = window is None or ctx_len + block - 1 < window
        open_context = context is None or ctx_len <= context
        if open_window and open_context:
            return None
    q = ctx_len + mx.arange(block)[:, None]
    k = mx.arange(ctx_len + block)[None, :]
    in_block = k >= ctx_len
    if causal:
        in_block = in_block & (k <= q)
    in_ctx = k < ctx_len
    if window is not None:
        in_ctx = in_ctx & (q - k < window)
    if context is not None:
        in_ctx = in_ctx & (k >= ctx_len - context)
    return in_block | in_ctx


_LOG_SNR_FEATURES = 128


def log_snr_features(rows: int) -> mx.array:
    """Sinusoidal noise-level features ``[rows, 128]`` for one draft block.
    The anchor row sits at the trained maximum log-SNR and every masked row
    at the minimum, so the normalized level is 1000 for the anchor and 0
    elsewhere whatever the range, before the ``10000^(-i/64)`` frequencies."""
    half = _LOG_SNR_FEATURES // 2
    level = mx.array([1000.0] + [0.0] * (rows - 1), dtype=mx.float32)
    freq = mx.exp(-mx.log(mx.array(10000.0)) * mx.arange(half) / half)
    angle = level[:, None] * freq[None, :]
    return mx.concatenate([mx.sin(angle), mx.cos(angle)], axis=-1)


# One side of a grouped dynamic convolution in a single dispatch: thread
# (b, l, h) walks the taps. Each product and each add rounds to InT in the
# op chain's order (static term, then dynamic term, per tap), so 16-bit
# results match the chain bit for bit.
_CONV_SRC = r"""
    uint gid = thread_position_in_grid.x;       // (b*L + l)*H + h
    uint h = gid % H;
    uint bl = gid / H;
    uint l = bl % L;
    auto d_ = dyn + (size_t)bl * (2 * K * G) + (size_t)SIDE * K * G + h / GS;
    auto b_ = base + (size_t)SIDE * K * H + h;
    InT acc = (InT)0.0f;
    for (int t = 0; t < K; ++t) {
        float xs = l >= (uint)t ? (float)x[gid - (size_t)t * H] : 0.0f;
        acc = (InT)((float)acc
                    + (float)(InT)((float)(InT)(float)b_[(size_t)t * H] * xs));
        acc = (InT)((float)acc + (float)(InT)((float)d_[(size_t)t * G] * xs));
    }
"""
_CONV_DTYPES = (mx.float16, mx.bfloat16, mx.float32)
_conv_kerns = None


def _conv_kernels():
    """The grouped-conv kernels (plain, with a residual add), or None.
    ``GMLX_DFLASH_CONV_KERNEL=0`` keeps the op chain."""
    global _conv_kerns
    if _conv_kerns is None:
        _conv_kerns = False
        if (env_bool("GMLX_DFLASH_CONV_KERNEL", True)
                and mx.metal.is_available()
                and mx.default_device().type == mx.DeviceType.gpu):
            _conv_kerns = (
                mx.fast.metal_kernel(
                    name="gmlx_dflash_conv",
                    input_names=["x", "dyn", "base"], output_names=["y"],
                    source=_CONV_SRC + "    y[gid] = acc;\n"),
                mx.fast.metal_kernel(
                    name="gmlx_dflash_conv_res",
                    input_names=["x", "dyn", "base", "res"],
                    output_names=["y"],
                    source=_CONV_SRC
                    + "    y[gid] = (InT)((float)res[gid] + (float)acc);\n"),
            )
    return _conv_kerns or None


class GroupedDynamicConv(nn.Module):
    """Two-sided grouped dynamic causal convolution over block positions.

    ``base_kernel[side][tap]`` is a per-channel kernel; ``kernel_projection``
    adds a per-position, per-group dynamic term. Taps run over the block
    positions only, zero padded at the block start, so the convolution is
    block-local. ``prepare`` convolves a sublayer input with side 0 and hands
    back the projected kernels; ``finish`` convolves the sublayer output
    with side 1 and adds the residual.

    On the GPU each side is one kernel dispatch. The op chain runs for any
    other device, for a dtype the kernel does not take, and when the
    projection's dtype differs from the input's.
    """

    def __init__(self, hidden_size: int, kernel_size: int, group_size: int):
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"hidden_size {hidden_size} is not a multiple of "
                f"conv_group_size {group_size}")
        self.kernel_size = int(kernel_size)
        self.group_size = int(group_size)
        groups = hidden_size // self.group_size
        self.base_kernel = mx.zeros((2, self.kernel_size, hidden_size))
        self.kernel_projection = nn.Linear(
            hidden_size, 2 * self.kernel_size * groups, bias=False)

    def _convolve(self, x: mx.array, dynamic: mx.array, base: mx.array) -> mx.array:
        """x [B, L, H]; dynamic [B, L, taps, groups]; base [taps, H]."""
        B, L, H = x.shape
        groups = H // self.group_size
        xb = x.reshape(B, L, groups, self.group_size)
        # Same accumulation order as the reference (static term, then the
        # dynamic term, per tap) so bf16 rounding matches it bit for bit.
        out = mx.zeros_like(xb)
        for tap in range(self.kernel_size):
            shifted = xb if tap == 0 else mx.pad(
                xb, [(0, 0), (tap, 0), (0, 0), (0, 0)])[:, :L]
            kernel = base[tap].reshape(1, 1, groups, self.group_size).astype(x.dtype)
            out = out + kernel * shifted
            out = out + dynamic[:, :, tap][..., None] * shifted
        return out.reshape(B, L, H)

    def _fused(self, x: mx.array, proj: mx.array, side: int,
               residual: Optional[mx.array]) -> Optional[mx.array]:
        """One side through the kernel, or None when the op chain must run."""
        kerns = _conv_kernels()
        if (kerns is None or x.ndim != 3 or x.dtype not in _CONV_DTYPES
                or proj.dtype != x.dtype
                or self.base_kernel.dtype not in _CONV_DTYPES
                or (residual is not None and (residual.dtype != x.dtype
                                              or residual.shape != x.shape))):
            return None
        B, L, H = x.shape
        n = B * L * H
        if n == 0:
            return None
        inputs = [x, proj, self.base_kernel]
        if residual is not None:
            inputs.append(residual)
        return kerns[residual is not None](  # pyright: ignore[reportIndexIssue, reportCallIssue]
            inputs=inputs,
            template=[("InT", x.dtype), ("H", H), ("L", L),
                      ("K", self.kernel_size), ("G", H // self.group_size),
                      ("GS", self.group_size), ("SIDE", side)],
            grid=(n, 1, 1), threadgroup=(min(256, n), 1, 1),
            output_shapes=[x.shape], output_dtypes=[x.dtype])[0]

    def _side(self, x: mx.array, proj: mx.array, side: int) -> mx.array:
        B, L, H = x.shape
        dynamic = proj.reshape(B, L, 2, self.kernel_size, H // self.group_size)
        return self._convolve(x, dynamic[:, :, side], self.base_kernel[side])

    def prepare(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """``(side-0 convolution of x, projected kernels for finish)``."""
        proj = self.kernel_projection(x)
        y = self._fused(x, proj, 0, None)
        return (self._side(x, proj, 0) if y is None else y), proj

    def finish(self, y: mx.array, proj: mx.array,
               residual: Optional[mx.array] = None) -> mx.array:
        """Side-1 convolution of ``y``, plus ``residual`` when given."""
        out = self._fused(y, proj, 1, residual)
        if out is None:
            out = self._side(y, proj, 1)
            if residual is not None:
                out = residual + out
        return out


class CandidateSelector(nn.Module):
    """DFlash 2 path selector over the top-k candidates of each block position.

    An edge from predecessor token ``a`` to candidate ``b`` at position ``p``
    scores ``logits[p][b] + <pred[a] * project(h[p]), succ[b]>``. The drafter
    walks the block sequentially from the anchor, one candidate per position.
    """

    def __init__(self, hidden_size: int, vocab_size: int, rank: int, top_k: int):
        super().__init__()
        self.top_k = int(top_k)
        self.hidden_projection = nn.Linear(hidden_size, rank, bias=False)
        self.predecessor_codebook = nn.Embedding(vocab_size, rank)
        self.successor_codebook = nn.Embedding(vocab_size, rank)

    def lattice(self, hidden: mx.array, logits: mx.array, anchor: mx.array,
                skip: Optional[tuple] = None):
        """Edge scores for one block.

        ``hidden`` [L, H] and ``logits`` [L, V] are the rows that draft tokens
        (the anchor row excluded); ``anchor`` is the token the block starts
        from. ``skip`` ``(lo, hi)`` says the logits lack the ids in
        ``[lo, hi)`` (``head_rows``). Returns ``(cands [L, k], first [k],
        edges [L-1, k, k])``, cands as token ids:
        ``first`` scores position 0 from the anchor; ``edges[p-1][i][j]``
        scores candidate ``j`` of position ``p`` after candidate ``i`` of
        position ``p-1``. All lazy; no host sync.
        """
        k = self.top_k
        cands, unary, hp, succ, pred0 = self._parts(hidden, logits, anchor, skip)
        first = unary[0] + succ[0] @ (pred0[0] * hp[0])                     # [k]
        if cands.shape[0] > 1:
            preds = self.predecessor_codebook(cands[:-1]).astype(unary.dtype)
            edges = (preds * hp[1:, None, :]) @ succ[1:].transpose(0, 2, 1)
            edges = unary[1:, None, :] + edges                              # [L-1, k, k]
        else:
            edges = mx.zeros((0, k, k), dtype=first.dtype)
        return cands, first, edges

    def _parts(self, hidden, logits, anchor, skip):
        """Candidates [L, k], their logits, the projected hidden [L, r] and
        the successor rows [L, k, r] and anchor predecessor row [1, r]."""
        k = self.top_k
        if _LATTICE_TOPK and logits.shape[-1] >= topk.HIER_MIN_V:
            # Same ids as the flat argpartition but for ties at the k-th
            # value, without the full-width sort.
            cols = topk.topk_ids(logits, k)
        else:
            cols = mx.argpartition(logits, -k, axis=-1)[..., -k:]
        unary = mx.take_along_axis(logits, cols, axis=-1)
        cands = cols
        if skip is not None:
            lo, hi = skip
            cands = cols + (cols >= lo).astype(cols.dtype) * (hi - lo)
        hp = self.hidden_projection(hidden).astype(unary.dtype)
        succ = self.successor_codebook(cands).astype(unary.dtype)          # [L, k, r]
        pred0 = self.predecessor_codebook(anchor.reshape(1)).astype(unary.dtype)
        return cands, unary, hp, succ, pred0

    def walk(self, hidden: mx.array, logits: mx.array, anchor: mx.array,
             skip: Optional[tuple] = None, floor: float = 0.0, proposal=None):
        """``lattice`` and ``selector_walk`` in one: the walk kernel scores
        only the lattice row of each realized predecessor. Returns
        ``(cands, tokens, columns, pick log-probabilities, keep, q)`` as
        ``selector_walk`` gives them, or None where the two steps run."""
        kerns = _lattice_walk_kernels() if _WALK_KERNEL and _LATTICE_WALK else None
        L, k = int(logits.shape[0]), self.top_k
        if (kerns is None or L < 2 or not 2 <= k <= 64
                or logits.dtype not in _CONV_DTYPES):
            return None
        sample = proposal is not None
        temp, top_k, top_p, min_p = proposal if sample else (1.0, 0, 0.0, 0.0)
        if sample and not temp > 0.0:
            return None
        cands, unary, hp, succ, pred0 = self._parts(hidden, logits, anchor, skip)
        rank = int(hp.shape[-1])
        if rank > _LWALK_MAX_RANK:
            return None
        preds = self.predecessor_codebook(cands[:-1]).astype(unary.dtype)  # [L-1, k, r]
        par = mx.array([temp, top_p, min_p,
                        math.log(floor) if floor > 0 else -math.inf],
                       dtype=mx.float32)
        inputs = [unary, hp, succ, pred0, preds, cands, par]
        shapes: list[tuple[int, ...]] = [(L,), (L,), (L,), (1,)]
        dtypes = [cands.dtype, mx.int32, mx.float32, mx.int32]
        if sample:
            inputs.append(mx.random.uniform(shape=(L,)))
            shapes.append((L, k))
            dtypes.append(mx.float32)
        out = kerns[sample](  # pyright: ignore[reportIndexIssue, reportCallIssue]
            inputs=inputs,
            template=[("T", unary.dtype), ("L", L), ("K", k), ("R", rank),
                      ("TOPK", int(top_k)), ("TG", _LWALK_TG)],
            grid=(_LWALK_TG, 1, 1), threadgroup=(_LWALK_TG, 1, 1),
            output_shapes=shapes, output_dtypes=dtypes)
        return (cands, out[0], out[1], out[2], out[3],
                out[4] if sample else None)


def greedy_path(first: mx.array, edges: mx.array) -> mx.array:
    """Candidate column of each position on the argmax path through a
    lattice, as a lazy scalar chain. Returns [L]."""
    sel = [mx.argmax(first)]
    for p in range(edges.shape[0]):
        sel.append(mx.argmax(edges[p][sel[-1]]))
    return mx.stack(sel)


def greedy_walk(cands: mx.array, first: mx.array, edges: mx.array) -> mx.array:
    """Argmax path through a lattice. Returns the drafted tokens [L]."""
    sel = greedy_path(first, edges)
    return mx.take_along_axis(cands, sel[:, None], axis=-1)[:, 0]


def path_estimate(first: mx.array, edges: mx.array, sel: mx.array) -> mx.array:
    """Tokens a round is expected to emit from the path ``sel``, as a lazy
    float32 scalar.

    Each pick counts as accepted with its probability under a softmax over
    its row's k candidates, and a draft is accepted only when every draft
    before it is. The estimate is 1 for the token the round always emits
    plus, for each position, the product of the pick probabilities up to
    and including that position.
    """
    return 1.0 + mx.exp(mx.cumsum(path_logprobs(first, edges, sel))).sum()


def path_logprobs(first: mx.array, edges: mx.array, sel: mx.array) -> mx.array:
    """Log-probability of each pick on the path ``sel`` under a softmax
    over its row's k candidates, as a lazy float32 [L]."""
    f = first.astype(mx.float32)
    lp = (f[sel[0]] - mx.logsumexp(f))[None]
    n = int(edges.shape[0])
    if n:
        rows = edges.astype(mx.float32)[mx.arange(n), sel[:-1]]         # [L-1, k]
        picked = mx.take_along_axis(rows, sel[1:, None], axis=-1)[:, 0]
        lp = mx.concatenate([lp, picked - mx.logsumexp(rows, axis=-1)])
    return lp


# The selector walk on one GPU thread: per position the lattice row of the
# realized predecessor, a pick, the pick's log-probability under a softmax
# over the row, and the row cut. The draw step samples the sharpened proposal
# (_pq_probs on the k-wide row: nucleus, min-p and top-k masks on untempered
# log-probabilities, temperature at the draw) by inverse CDF on a uniform,
# and writes the proposal row the draw used.
_WALK_ROW = r"""
    if (thread_position_in_grid.x != 0) return;
    const float log_floor = par[3];
    float cum = 0.0f;
    int kept = 0, prev = 0;
    for (int p = 0; p < L; ++p) {
        float row[K];
        float top = -INFINITY;
        for (int j = 0; j < K; ++j) {
            row[j] = p == 0 ? (float)first[j]
                : (float)edges[((size_t)(p - 1) * K + prev) * K + j];
            top = max(top, row[j]);
        }
        float s = 0.0f;
        for (int j = 0; j < K; ++j) s += exp(row[j] - top);
        const float lse = top + log(s);
        int pick = 0;
"""
_WALK_ARGMAX = r"""
        for (int j = 1; j < K; ++j)
            if (row[j] > row[pick]) pick = j;
"""
_WALK_DRAW = r"""
        const float temp = par[0], top_p = par[1], min_p = par[2];
        int ord[K];             // columns by ascending score, stable
        bool live[K];
        for (int j = 0; j < K; ++j) {
            int a = j;
            while (a > 0 && row[ord[a - 1]] > row[j]) {
                ord[a] = ord[a - 1];
                --a;
            }
            ord[a] = j;
        }
        float c = 0.0f;
        for (int a = 0; a < K; ++a) {
            const int j = ord[a];
            c += exp(row[j] - lse);
            live[j] = !(top_p > 0.0f && top_p < 1.0f) || c > 1.0f - top_p;
            if (min_p > 0.0f && row[j] - top < log(min_p)) live[j] = false;
            if (TOPK > 0 && TOPK < K && a < K - TOPK) live[j] = false;
        }
        live[ord[K - 1]] = true;
        float z = 0.0f;
        for (int j = 0; j < K; ++j)
            if (live[j]) z += exp((row[j] - top) / temp);
        float acc = 0.0f;
        bool drawn = false;
        for (int j = 0; j < K; ++j) {
            const float qj = live[j] ? exp((row[j] - top) / temp) / z : 0.0f;
            qc[(size_t)p * K + j] = qj;
            acc += qj;
            if (live[j] && !drawn) {
                pick = j;
                drawn = u[p] < acc;
            }
        }
"""
_WALK_STEP = r"""
        const float lpick = row[pick] - lse;
        cum += lpick;
        if (cum >= log_floor) ++kept;
        toks[p] = cands[(size_t)p * K + pick];
        sel[p] = pick;
        lp[p] = lpick;
        prev = pick;
    }
    keep[0] = kept;
"""
# The walk with the lattice rows scored inside: per position, a threadgroup
# scores each candidate against the realized predecessor, rounding as the
# op chain does (the codebook product, the dot product, the logit add), and
# its first thread runs the walk step.
_LWALK_ROW = r"""
    const uint tid = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    threadgroup float m[R];
    threadgroup float srow[K];
    threadgroup int sprev[1];
    if (tid == 0) sprev[0] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float log_floor = par[3];
    float cum = 0.0f;
    int kept = 0;
    for (int p = 0; p < L; ++p) {
        const int prev = sprev[0];
        for (uint t = tid; t < (uint)R; t += TG) {
            const float a = p == 0 ? (float)pred0[t]
                : (float)preds[((size_t)(p - 1) * K + prev) * R + t];
            m[t] = (float)(T)(a * (float)hp[(size_t)p * R + t]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = sg; j < (uint)K; j += TG / 32) {
            float d = 0.0f;
            for (uint t = lane; t < (uint)R; t += 32)
                d += (float)succ[((size_t)p * K + j) * R + t] * m[t];
            d = simd_sum(d);
            if (lane == 0)
                srow[j] = (float)(T)((float)unary[(size_t)p * K + j] + (float)(T)d);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float row[K];
            float top = -INFINITY;
            for (int j = 0; j < K; ++j) {
                row[j] = srow[j];
                top = max(top, row[j]);
            }
            float s = 0.0f;
            for (int j = 0; j < K; ++j) s += exp(row[j] - top);
            const float lse = top + log(s);
            int pick = 0;
"""
_LWALK_STEP = r"""
            const float lpick = row[pick] - lse;
            cum += lpick;
            if (cum >= log_floor) ++kept;
            toks[p] = cands[(size_t)p * K + pick];
            sel[p] = pick;
            lp[p] = lpick;
            sprev[0] = pick;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) keep[0] = kept;
"""
_LWALK_TG = 256
_LWALK_MAX_RANK = 4096
_walk_kerns = None
_lwalk_kerns = None


def _lattice_walk_kernels():
    """(greedy, sampled) lattice walk kernels, or None."""
    global _lwalk_kerns
    if _lwalk_kerns is None:
        _lwalk_kerns = False
        if (mx.metal.is_available()
                and mx.default_device().type == mx.DeviceType.gpu):
            names = ["unary", "hp", "succ", "pred0", "preds", "cands", "par"]
            outs = ["toks", "sel", "lp", "keep"]
            _lwalk_kerns = (
                mx.fast.metal_kernel(
                    name="gmlx_dflash_lattice_walk", input_names=names,
                    output_names=outs,
                    source=_LWALK_ROW + _WALK_ARGMAX + _LWALK_STEP),
                mx.fast.metal_kernel(
                    name="gmlx_dflash_lattice_walk_sample",
                    input_names=names + ["u"], output_names=outs + ["qc"],
                    source=_LWALK_ROW + _WALK_DRAW + _LWALK_STEP),
            )
    return _lwalk_kerns or None


def _walk_kernels():
    """(greedy, sampled) selector walk kernels, or None."""
    global _walk_kerns
    if _walk_kerns is None:
        _walk_kerns = False
        if (mx.metal.is_available()
                and mx.default_device().type == mx.DeviceType.gpu):
            _walk_kerns = (
                mx.fast.metal_kernel(
                    name="gmlx_dflash_walk",
                    input_names=["first", "edges", "cands", "par"],
                    output_names=["toks", "sel", "lp", "keep"],
                    source=_WALK_ROW + _WALK_ARGMAX + _WALK_STEP),
                mx.fast.metal_kernel(
                    name="gmlx_dflash_walk_sample",
                    input_names=["first", "edges", "cands", "par", "u"],
                    output_names=["toks", "sel", "lp", "keep", "qc"],
                    source=_WALK_ROW + _WALK_DRAW + _WALK_STEP),
            )
    return _walk_kerns or None


def selector_walk(cands: mx.array, first: mx.array, edges: mx.array,
                  floor: float, proposal=None):
    """One block's walk through a lattice in one dispatch, or None when the
    op chain must run (no GPU, a one-row block, a wide row).

    ``proposal`` None walks the argmax path. ``(temp, top_k, top_p, min_p)``
    draws each pick from the sharpened proposal of its row. Returns
    ``(tokens [L], columns [L], pick log-probabilities [L], keep [1], q)``:
    ``keep`` counts the picks whose prefix clears ``floor`` (all of them at
    floor 0) and ``q`` [L, k] is the proposal row of each draw in candidate
    column order, None on the argmax path.
    """
    kerns = _walk_kernels() if _WALK_KERNEL else None
    L, k = int(cands.shape[0]), int(cands.shape[1])
    if (kerns is None or L < 2 or not 2 <= k <= 64
            or first.dtype not in _CONV_DTYPES or edges.dtype != first.dtype):
        return None
    sample = proposal is not None
    temp, top_k, top_p, min_p = proposal if sample else (1.0, 0, 0.0, 0.0)
    if sample and not temp > 0.0:
        return None
    par = mx.array([temp, top_p, min_p,
                    math.log(floor) if floor > 0 else -math.inf],
                   dtype=mx.float32)
    inputs = [first, edges, cands, par]
    shapes: list[tuple[int, ...]] = [(L,), (L,), (L,), (1,)]
    dtypes = [cands.dtype, mx.int32, mx.float32, mx.int32]
    if sample:
        inputs.append(mx.random.uniform(shape=(L,)))
        shapes.append((L, k))
        dtypes.append(mx.float32)
    out = kerns[sample](  # pyright: ignore[reportIndexIssue, reportCallIssue]
        inputs=inputs,
        template=[("L", L), ("K", k), ("TOPK", int(top_k))],
        grid=(1, 1, 1), threadgroup=(1, 1, 1),
        output_shapes=shapes, output_dtypes=dtypes)
    return out[0], out[1], out[2], out[3], (out[4] if sample else None)


def _note_cut_draft(stash, toks: mx.array, n: int) -> None:
    """Leave the first draft a row cut at ``n`` dropped, with the proposal
    row it was drawn from, in ``stash.cut`` for the walk. The cut keeps a
    draft when the path chance through the draft's own pick clears the
    floor, so the draft at the cut is dropped because of what it drew; the
    walk has to test it like any drawn draft, or the token there is no
    longer a sample of the target row."""
    if stash is None:
        return
    q = stash.q
    if (q is None or stash.pq is not None or stash.top2 is not None
            or n >= len(q) or n >= toks.shape[1]):
        stash.cut = None
        return
    # The block is evaluated by now (the cut count came with it), so this
    # reads host memory and builds no op.
    stash.cut = (int(np.asarray(toks)[0, n]), q[n])


def _cut_path(path: tuple, n: int) -> tuple:
    """A round path ``(first, edges, sel, lp)`` cut to its first ``n``
    picks. ``first`` and ``edges`` are None when the walk kernel scored the
    path; ``lp`` is None when no kernel gave the pick log-probabilities."""
    first, edges, sel, lp = path
    return (first, None if edges is None else edges[: n - 1], sel[:n],
            None if lp is None else lp[:n])


def rows_over_floor(lp: mx.array, floor: float) -> mx.array:
    """How many leading picks keep the chance that the whole prefix is
    accepted at or over ``floor``, as a lazy int scalar. ``lp`` [L] holds
    each pick's log-probability."""
    return (mx.cumsum(lp) >= math.log(floor)).sum()


# --- draw / stash -------------------------------------------------------------

def _pick(support: Optional[mx.array], idx: mx.array) -> mx.array:
    if support is None:
        return idx
    return mx.take_along_axis(support, idx[:, None], axis=-1).reshape(-1)


def _scatter(vals: mx.array, support: Optional[mx.array], fill: float,
             vocab: int) -> mx.array:
    """Rows [n, W] on support [n, W] -> full-width rows [n, vocab]. A
    ``put_along_axis`` set: the top-k ids are distinct, and an add onto a
    ``-inf`` base would stay ``-inf``."""
    if support is None:
        return vals
    base = mx.full((vals.shape[0], vocab), fill, dtype=vals.dtype)
    return mx.put_along_axis(base, support, vals, axis=-1)


def _second_choice(rows: mx.array) -> mx.array:
    """Index of each row's second-highest entry, [n]."""
    kth = rows.shape[-1] - 2
    k2 = mx.argpartition(rows, kth=kth, axis=-1)[..., -2:]
    v2 = mx.take_along_axis(rows, k2, axis=-1)
    return mx.take_along_axis(
        k2, mx.argmin(v2, axis=-1, keepdims=True), axis=-1).reshape(-1)


def _sample_rows(sampler, rows: mx.array, support: Optional[mx.array], vocab: int):
    """Draw one token per row with the engine's sampler. Full-width rows go
    to the sampler as-is. Compact rows (a support) cannot: mlx_lm's
    make_sampler bounds-checks top_k against the row width, so the draw is
    rebuilt from the sampler's annotated params; a serve sampler guards its
    own width and is called directly; an opaque sampler gets the scattered
    row. Same precedence as speculative._stoch_supported_sampler. The
    compiled sampler must never see a compact row: its ValueError is raised
    mid-trace and leaves ``mx.random.state`` holding a tracer."""
    if support is None:
        toks = sampler(rows).reshape(-1)
        return toks, toks
    if hasattr(sampler, "_filtered") and hasattr(sampler, "temperature"):
        idx = sampler(rows).reshape(-1)
    elif getattr(sampler, "gmlx_sampling_params", None) is not None:
        from .speculative import _pq_probs

        p = sampler.gmlx_sampling_params
        q = _pq_probs(rows, p["temp"], min(int(p["top_k"]), int(rows.shape[-1])),
                      p["top_p"], min_p=p.get("min_p", 0.0))
        idx = mx.random.categorical(mx.log(q), axis=-1)
    else:
        toks = sampler(_scatter(rows, support, float("-inf"), vocab)).reshape(-1)
        return toks, mx.argmax(support == toks[:, None], axis=-1)
    return _pick(support, idx), idx


def draw_rows(rows: mx.array, support: Optional[mx.array], *, vocab: int,
              greedy: bool, sampler, stash: Optional[DraftStash]):
    """Draw one draft token per row and record the round's instrument rows.

    ``rows`` [n, W] are log-domain draft scores; ``support`` [n, W] maps row
    columns to token ids (None when W is the vocab). Greedy -> argmax;
    stochastic (``stash.q`` present) -> a draw from the sharpened proposal,
    which is also the q row stashed; otherwise the engine's sampler. Logging
    never changes the draw. Returns ``(tokens [n], column index [n])``.
    """
    stoch = stash is not None and stash.q is not None
    if greedy:
        idx = mx.argmax(rows, axis=-1)
        toks = _pick(support, idx)
    elif stoch:
        from .speculative import _STOCH_DRAFT, _sharp_draw

        idx, q = _sharp_draw(rows, *_STOCH_DRAFT)
        toks = _pick(support, idx)
        stash.q.extend(_scatter(q, support, 0.0, vocab))
    else:
        toks, idx = _sample_rows(sampler, rows, support, vocab)
    if stash is not None:
        if stash.pq is not None:
            stash.pq.extend(_scatter(rows, support, float("-inf"), vocab))
        if stash.top2 is not None:
            stash.top2.extend(_pick(support, _second_choice(rows)))
    return toks, idx


# --- module tree --------------------------------------------------------------

class DFlashAttention(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None
        self.use_rope = (config.rope_layers[layer_idx] if config.rope_layers
                         else True)
        self.context_window = (None if self.is_sliding
                               else config.context_window)
        # The most context rows this layer ever reads. A longer inject only
        # projects the trailing ones, which leaves each layer's relative
        # positions as they were.
        if self.sliding_window is not None:
            self.ring_rows = int(self.sliding_window) - 1
        else:
            self.ring_rows = self.context_window
        self.is_causal = bool(config.is_causal) if config.is_causal is not None else False
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        if config.attention_sinks:
            self.sinks = mx.zeros((self.n_heads,))

    def inject(self, h_ctx: mx.array, rope, cache, kv=None) -> None:
        """Write the context rows' K/V into the ring at the ring's positions.
        ``kv`` holds the rows' (keys, values) projections when the caller
        ran them."""
        if self.ring_rows is not None and int(h_ctx.shape[1]) > self.ring_rows:
            h_ctx, kv = h_ctx[:, -self.ring_rows:], None
        B, S, _ = h_ctx.shape
        keys, values = kv or _shared_input((self.k_proj, self.v_proj), h_ctx)
        keys = self.k_norm(keys.reshape(B, S, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, S, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if self.use_rope:
            keys = rope(keys, offset=cache.offset)
        cache.update_and_fetch(keys, values)

    def _ring_in_place(self, cache, q, k, v, sinks):
        """Keys and values for a draft block read from the ring's own buffer.

        The block's K/V go into the buffer's spare rows after the ring rows.
        They never count as ring rows: the next commit writes over them. The
        attention then reads one view of the buffer, with no copy of the
        ring. Returns ``(q, keys, values, mask, reversed)`` or None when the
        cache is not a slack-backed ring with room and matching dtypes.

        A full non-causal sliding ring also drops the array mask. Block row
        ``i`` sees ring rows from ``lo + i`` up, so the view is laid out as
        the rows every block row sees, the block, then the ``L - 1`` ring
        rows only some block rows see, newest of them first. With the block
        rows reversed that is a causal mask aligned to the end, which the
        attention kernels take without a mask array. ``reversed`` tells the
        caller to reverse the output rows back.
        """
        if (not _RING_IN_PLACE or type(cache) is not BufferedRotatingKVCache
                or cache.keep or cache.keys is None):
            return None
        rk, rv = cache.keys, cache.values
        if (rk.dtype != k.dtype or rv.dtype != v.dtype
                or rk.shape[:2] != k.shape[:2] or rk.shape[3:] != k.shape[3:]
                or rv.shape[:2] != v.shape[:2] or rv.shape[3:] != v.shape[3:]):
            return None
        idx, L, cap = int(cache._idx), int(k.shape[2]), int(rk.shape[2])
        window = self.sliding_window
        lo = idx - (int(window) - 1) if window is not None else -1
        if (lo >= 0 and L > 1 and not self.is_causal and sinks is None
                and self.context_window is None and idx + 2 * L - 1 <= cap):
            def rows(ring, block):
                return mx.concatenate(
                    [block, ring[..., lo:lo + L - 1, :][..., ::-1, :]], axis=2)

            cache.keys[..., idx:idx + 2 * L - 1, :] = rows(rk, k)
            cache.values[..., idx:idx + 2 * L - 1, :] = rows(rv, v)
            a, b = lo + L - 1, idx + 2 * L - 1
            return (q[:, :, ::-1, :], cache.keys[..., a:b, :],
                    cache.values[..., a:b, :], "causal", True)
        if idx + L > cap:
            return None
        cache.keys[..., idx:idx + L, :] = k
        cache.values[..., idx:idx + L, :] = v
        mask = block_attention_mask(idx, L, window, self.is_causal,
                                    self.context_window)
        return (q, cache.keys[..., :idx + L, :], cache.values[..., :idx + L, :],
                mask, False)

    def ring_counted_ok(self, cache, block: int, most: int,
                        ahead: int = 0) -> bool:
        """Whether ``draft(counted=...)`` can run on ``cache`` for a block
        of ``block`` rows with up to ``most`` trailing ring rows not live,
        once ``ahead`` more rows are in the ring."""
        if (not _RING_IN_PLACE or type(cache) is not BufferedRotatingKVCache
                or cache.keep or cache.keys is None or self.is_causal
                or "sinks" in self or self.context_window is not None
                or self.sliding_window is None):
            return False
        idx, cap = int(cache._idx) + ahead, int(cache.keys.shape[2])
        if idx - most - (int(self.sliding_window) - 1) < 0:
            # A ring shorter than the window: the masked form, behind its
            # lever.
            return _COUNTED_SHORT and idx - most >= 1
        # Read by row index the block is not written into the buffer.
        return _COUNTED_GATHER or idx + block <= cap

    def _ring_counted(self, cache, q, k, v, counted):
        """Keys, values and mask for a block drafted while the count of
        live ring rows is still on the GPU.

        The commit ahead wrote ``most + 1`` rows and the last ``dead`` of
        them (an array, 0 to ``most``) are not live. The view is the
        buffer's rows every such count can see, then the block in the spare
        rows after them, at host-known places; ``dead`` only enters the
        mask. Block row ``i`` sees ring rows ``[live - (window - 1) + i,
        live)`` with ``live = idx - dead``, and every block row."""
        dead, most = counted
        idx, L = int(cache._idx), int(k.shape[2])
        w1 = int(self.sliding_window or 1) - 1
        if (cache.keys.dtype != k.dtype or cache.values.dtype != v.dtype):
            return None
        if idx - most - w1 < 0:
            # A ring shorter than the window (GMLX_DFLASH_COUNTED_SHORT):
            # the plain draft's keys in its order, every ring row then the
            # block, with the ``dead`` rows before the block masked out.
            # The plain draft has ``dead`` keys fewer, so this equals it to
            # rounding only. The ring is not written.
            if not _COUNTED_SHORT:
                return None
            live = idx - dead
            col = mx.arange(idx + L)[None, :]
            row = mx.arange(L)[:, None]
            mask = (((col < live) & (live + row - col < w1 + 1))
                    | (col >= idx))
            return (q, mx.concatenate([cache.keys[..., :idx, :], k], axis=2),
                    mx.concatenate([cache.values[..., :idx, :], v], axis=2),
                    mask, False)
        if _COUNTED_GATHER and L > 1 and w1 >= L - 1:
            # The plain layout of _ring_in_place at ring index
            # ``idx - dead``, read by row index: the rows every block row
            # sees, the block, then the L - 1 oldest rows newest first.
            # Same keys in the same order as the plain draft, so the same
            # attention kernel and the same bits; the ring is not written.
            lo = (idx - w1) - dead
            rows = mx.concatenate([lo + mx.arange(L - 1, w1),
                                   lo + mx.arange(L - 2, -1, -1)])
            n_a = w1 - (L - 1)

            def view(ring, block):
                got = mx.take(ring, rows, axis=2)
                return mx.concatenate(
                    [got[..., :n_a, :], block, got[..., n_a:, :]], axis=2)

            return (q[:, :, ::-1, :], view(cache.keys, k),
                    view(cache.values, v), "causal", True)
        cache.keys[..., idx:idx + L, :] = k
        cache.values[..., idx:idx + L, :] = v
        lo = idx - most - w1
        n_ring = w1 + most
        col = mx.arange(n_ring + L)[None, :]
        row = mx.arange(L)[:, None]
        first = (most - dead) + row                 # live - w1 + i - lo
        mask = ((col >= first) & (col < n_ring - dead)) | (col >= n_ring)
        keys = cache.keys[..., lo:idx + L, :]
        values = cache.values[..., lo:idx + L, :]
        if _COUNTED_F32:
            # An array mask takes the stock attention, which keeps the
            # stored dtype through the softmax; the causal form's kernel
            # does not. float32 here keeps the draft's accept rate.
            q, keys, values = (t.astype(mx.float32) for t in (q, keys, values))
        return (q, keys, values, mask, False)

    def draft(self, x: mx.array, rope, cache, counted=None) -> mx.array:
        """Block rows attend the ring plus their siblings. The block's own
        K/V never enters the ring. ``counted`` is ``(dead, most)`` of
        ``_ring_counted``."""
        B, L, _ = x.shape
        q, k, v = _shared_input((self.q_proj, self.k_proj, self.v_proj), x)
        q = self.q_norm(q.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        k = self.k_norm(k.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if self.use_rope:
            at = (cache.offset if counted is None
                  else int(cache.offset) - counted[0])
            q = rope(q, offset=at)
            k = rope(k, offset=at)
        sinks = self["sinks"].astype(q.dtype) if "sinks" in self else None
        if counted is not None:
            ring = self._ring_counted(cache, q, k, v, counted)
            if ring is None:
                raise RuntimeError("counted draft on a ring that cannot take it")
        else:
            ring = self._ring_in_place(cache, q, k, v, sinks)
        if ring is not None:
            q, k, v, mask, flipped = ring
        else:
            flipped = False
            ck, cv = cache.state
            ctx_len = 0
            if ck is not None:
                ctx_len = int(ck.shape[2])
                k = mx.concatenate([ck, k], axis=2)
                v = mx.concatenate([cv, v], axis=2)
            mask = block_attention_mask(ctx_len, L, self.sliding_window,
                                        self.is_causal, self.context_window)
        o = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask, sinks=sinks)
        if flipped:
            o = o[:, :, ::-1, :]
        if o.dtype != x.dtype:
            o = o.astype(x.dtype)
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1))


class Qwen3MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = _shared_input((self.gate_proj, self.up_proj), x)
        return self.down_proj(swiglu(gate, up))


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        self.self_attn = DFlashAttention(config, layer_idx)
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def inject(self, h_ctx: mx.array, rope, cache, kv=None) -> None:
        self.self_attn.inject(h_ctx, rope, cache, kv)

    def draft(self, h: mx.array, rope, cache, counted=None) -> mx.array:
        h = h + self.self_attn.draft(self.input_layernorm(h), rope, cache,
                                     counted)
        return h + self.mlp(self.post_attention_layernorm(h))


class DFlash2DecoderLayer(DFlashDecoderLayer):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.attention_conv = GroupedDynamicConv(
            config.hidden_size, config.conv_kernel_size, config.conv_group_size)
        self.mlp_conv = GroupedDynamicConv(
            config.hidden_size, config.conv_kernel_size, config.conv_group_size)

    def draft(self, h: mx.array, rope, cache, counted=None) -> mx.array:
        x, kernel = self.attention_conv.prepare(self.input_layernorm(h))
        h = self.attention_conv.finish(
            self.self_attn.draft(x, rope, cache, counted), kernel, h)
        x, kernel = self.mlp_conv.prepare(self.post_attention_layernorm(h))
        return self.mlp_conv.finish(self.mlp(x), kernel, h)


# --- drafters -----------------------------------------------------------------

class DFlashDrafter(nn.Module):
    """DFlash drafter following gmlx's BatchDrafterProtocol (owned engine)."""

    layer_class = DFlashDecoderLayer
    kind_label = "dflash"
    supports_greedy_draft_argmax = True
    prefer_requested_block_size = False
    cap_at_configured_depth = True
    uses_shared_kv = False
    supports_kv_sidecar = False
    # The stock MTP round doesn't know the packed-hidden target hooks.
    requires_owned_engine = True
    supports_q_stash = True
    # v1 rows are independent given the anchor, so a row-wise draw is not a
    # proposal the p/q walk can accept against; exact-match stays.
    stochastic_draft = False

    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config
        hidden = int(config.hidden_size)
        concat_dim = len(config.target_layer_ids) * hidden
        self.fc = nn.Linear(concat_dim, hidden, bias=False)
        self.hidden_norm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        self.layers = [self.layer_class(config, i)
                       for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        if config.gmlx_rope_scaling:
            # llama.cpp's table and magnitude; the drafter has one rope, so
            # its sliding layers rotate as its full ones (llama.cpp dflash
            # loader sets the sliding scale to the trained one).
            from gmlx.load import rope_scaling as rs
            from gmlx.load.rope_swap import scaled_rope

            scaling = rs.RopeScaling(**config.gmlx_rope_scaling)
            self.rope = scaled_rope(
                int(config.head_dim), scaling.freq_base or config.rope_theta,
                scaling)
        else:
            self.rope = initialize_rope(
                dims=config.head_dim, base=config.rope_theta,
                traditional=False, scaling_config=config.rope_scaling,
                max_position_embeddings=config.max_position_embeddings)
        self.embed_tokens = None
        self.embed_scale = 1.0
        self.lm_head = None
        self.accept_lens: List[int] = []
        self.draft_lens: List[int] = []
        self.gated_rounds = 0
        self._native_block_size = (
            native_block_size(config) or int(config.block_size))
        self._hidden = hidden
        self._n_targets = len(config.target_layer_ids)
        # Only the trailing ring-capacity rows of the prompt capture are
        # usable; the engine trims the retained prefill hidden to this.
        window = int(config.sliding_window or 0)
        if config.context_window is not None:
            self.hidden_capture_limit = max(
                int(config.context_window), window - 1)
        else:
            self.hidden_capture_limit = (window - 1) if window else None
        self._cache: List[Any] = []

    def capture_held_bytes(self, prompt_tokens: int) -> int:
        """Peak bytes of the prompt capture as the ring seeds: the packed
        target hidden, its tap slice and the ``fc`` output, over the rows
        the drafter keeps. 0 when the capture has no row limit."""
        limit = self.hidden_capture_limit
        if not limit:
            return 0
        rows = min(int(prompt_tokens), int(limit))
        return rows * self._hidden * (2 * self._n_targets + 2) * 2

    # --- lifecycle ----------------------------------------------------------

    def bind(self, target_model):
        if hasattr(target_model, "embed_tokens"):
            inner = target_model
        elif hasattr(getattr(target_model, "model", None), "embed_tokens"):
            inner = target_model.model
        elif hasattr(getattr(getattr(target_model, "language_model", None),
                             "model", None), "embed_tokens"):
            inner = target_model.language_model.model
        else:
            raise AttributeError(
                f"cannot find embed_tokens in {type(target_model).__name__}")
        from gmlx.load.modules import lora_base

        # The base table, as llama.cpp drafts: a draft call publishes no
        # row vector.
        self.embed_tokens = lora_base(inner.embed_tokens)
        # The raw table times the drafter's own scale, as llama.cpp reads it.
        # The converter writes a target's embed_scale (Gemma's sqrt(hidden))
        # into that scale.
        self.embed_scale = float(self.config.input_embedding_scale)
        lm = getattr(target_model, "language_model", target_model)

        # The base head: an output.weight adapter's wrapper reads a row
        # vector, and a draft call publishes none.
        self.lm_head = (lora_base(getattr(target_model, "lm_head", None)
                                  or getattr(lm, "lm_head", None))
                        or self.embed_tokens.as_linear)
        return self

    def make_cache(self, left_padding: Optional[List[int]] = None) -> list:
        if left_padding is not None:
            raise NotImplementedError(f"{type(self).__name__} is B=1 only")
        caches = []
        for layer_type in self.config.layer_types:
            if layer_type == "sliding_attention":
                if self.config.sliding_window is None:
                    raise ValueError(
                        "sliding_attention layer without a sliding_window")
                # Temporal and slack-backed: the draft path reads cache.state
                # directly and indexes its rows as time.
                caches.append(BufferedRotatingKVCache(
                    max_size=int(self.config.sliding_window) - 1,
                    buffer_size=_RING_SLACK))
            elif self.config.context_window is not None:
                # One row over the window: after a compaction the ring keeps
                # max_size - 1 rows, so the mask always has the full window.
                caches.append(BufferedRotatingKVCache(
                    max_size=int(self.config.context_window) + 1,
                    buffer_size=_RING_SLACK))
            else:
                caches.append(KVCache())
        return caches

    def reset(self, target_model, left_padding: Optional[List[int]] = None) -> list:
        if left_padding is not None:
            raise NotImplementedError(
                f"{type(self).__name__} is B=1 only: batched rounds need "
                "per-row cache offsets in the inject path")
        self.bind(target_model)
        self.accept_lens = []
        self.draft_lens = []
        self.gated_rounds = 0
        self._cache = self.make_cache()
        limit = self.hidden_capture_limit
        object.__setattr__(
            self, "_tail", HiddenTail(limit) if _KEEP_TAIL and limit else None)
        return self._cache

    def draft_eval_state(self) -> List[Any]:
        return [c.state for c in self._cache]

    def set_shared_kv(self, *args, **kwargs) -> None:
        return None

    # --- heads ---------------------------------------------------------------

    def _embed_input_tokens(self, tokens: mx.array) -> mx.array:
        return self.embed_tokens(tokens) * self.embed_scale

    def _logits(self, hidden: mx.array) -> mx.array:
        logits = self.lm_head(hidden)
        if self.config.output_multiplier != 1.0:
            logits = logits * self.config.output_multiplier
        cap = self.config.final_logit_softcapping
        if cap:
            logits = mx.tanh(logits / cap) * cap
        return logits

    # --- inject mode --------------------------------------------------------

    def _captures(self, packed: mx.array) -> mx.array:
        """Trailing ``n_targets*hidden`` of the packed target hidden."""
        want = self._hidden * (1 + self._n_targets)
        if int(packed.shape[-1]) != want:
            raise ValueError(
                f"packed hidden width {packed.shape[-1]} != {want}; the "
                "target's _dflash_capture wiring is missing")
        # Materialize: this feeds the quantized fc, whose kernel reads the
        # buffer directly and would otherwise see the packed strides.
        return mx.contiguous(packed[..., self._hidden:])

    def append_context(self, captures: mx.array, cache=None) -> None:
        """Encode committed positions and inject their K/V into the ring.
        ``cache`` names a parked row's ring instead of the live one."""
        h_ctx = self.hidden_norm(self.fc(captures))
        kv = self._context_kv(h_ctx)
        for layer, c, pair in zip(self.layers, self._cache if cache is None else cache, kv):
            layer.inject(h_ctx, self.rope, c, pair)

    def _context_kv(self, h_ctx: mx.array) -> list:
        """Every layer's (keys, values) projections of ``h_ctx`` from one
        multi-weight matmul per codec, or None per layer where the layer
        projects its own (a ring shorter than ``h_ctx``)."""
        attn = [layer.self_attn for layer in self.layers]
        S = int(h_ctx.shape[1])
        if not _QMM_MULTI or len(attn) < 2 or any(
                a.ring_rows is not None and S > a.ring_rows for a in attn):
            return [None] * len(attn)
        out = _shared_input([p for a in attn for p in (a.k_proj, a.v_proj)], h_ctx)
        return [(out[2 * i], out[2 * i + 1]) for i in range(len(attn))]

    # --- draft mode ---------------------------------------------------------

    def _draft_hidden(self, tokens: mx.array, counted=None) -> mx.array:
        h = self._embed_input_tokens(tokens)
        for layer, c in zip(self.layers, self._cache):
            h = (layer.draft(h, self.rope, c) if counted is None
                 else layer.draft(h, self.rope, c, counted))
        return self.norm(h)

    def _block_tokens(self, last_bonus, block_size: int, token_dtype) -> mx.array:
        if block_size > self._native_block_size:
            raise RuntimeError(
                f"{type(self).__name__} drafts at most "
                f"{self._native_block_size - 1} token(s)/round; got "
                f"block_size={block_size} - cap_at_configured_depth should "
                "have clamped it")
        if not self._cache:
            raise RuntimeError("reset(target_model) must run before draft_block()")
        mask_id = int(self.config.mask_token_id)
        bonus = (int(last_bonus) if isinstance(last_bonus, int)
                 else int(last_bonus.reshape(-1)[0].item()))
        return mx.array([[bonus] + [mask_id] * (block_size - 1)], dtype=token_dtype)

    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache,
        block_size: int,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
        stash: Optional[DraftStash] = None,
    ) -> mx.array:
        """One round: ``[bonus, MASK x (block_size-1)]`` denoised in a single
        forward; drafts are rows 1..block_size-1."""
        del hidden, cache
        block = self._block_tokens(last_bonus, block_size, token_dtype)
        logits = self._logits(self._draft_hidden(block)[:, 1:])[0]
        toks, _ = draw_rows(logits, None, vocab=int(self.config.vocab_size),
                            greedy=greedy, sampler=sampler, stash=stash)
        return toks[None]

    # --- commit -------------------------------------------------------------

    def prefill_from_target_hidden(
        self,
        input_ids: mx.array,
        hidden: mx.array,
        bonus_token,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
        stash: Optional[DraftStash] = None,
    ) -> None:
        """Seed the ring from the trailing prompt hiddens. DFlash needs no
        draft seed: rounds start from the engine-passed bonus token."""
        del input_ids, bonus_token, sampler, token_dtype, greedy, stash
        if int(hidden.shape[1]) == 0:
            return
        limit = self.hidden_capture_limit
        if limit:
            hidden = hidden[:, -limit:]
        if getattr(self, "_tail", None) is not None:
            self._tail.prompt(hidden)
        self.append_context(self._captures(hidden))

    def accept_verified_tokens(
        self,
        verify_hidden: mx.array,
        draft_tokens: mx.array,
        accepted: int,
        new_tokens: List[int],
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
        stash: Optional[DraftStash] = None,
    ) -> None:
        """Inject the committed positions' captures. ``verify_hidden[:, p]`` is
        the target hidden at verify position ``p``; 0..accepted were committed
        (the accepted drafts plus the row the new bonus was sampled from),
        matching the rolled-back target 1:1."""
        del draft_tokens, sampler, token_dtype, greedy, stash
        if new_tokens:
            _note_tokens(self._cache, new_tokens)
        if getattr(self, "_tail", None) is not None:
            self._tail.accept(verify_hidden, int(accepted) + 1)
        self.append_context(self._captures(verify_hidden[:, : int(accepted) + 1]))
        if _COMMIT_ASYNC:
            mx.async_eval([x for c in self._cache for x in (c.keys, c.values)
                           if x is not None])

    def accept_mark(self, rows: int):
        """What ``accept_undo`` needs to take back the next accept of
        ``rows`` positions, or None when a trim cannot: a ring that would
        compact or grow for them, or a cache that is no ring."""
        ring = self._cache
        steps = []
        for layer, c in zip(self.layers, ring):
            if type(c) is BufferedRotatingKVCache:
                n = min(int(rows), layer.self_attn.ring_rows or int(rows))
                if (c.keep or c.keys is None or c._planned_drop(n)
                        or c._idx + n > int(c.keys.shape[2])):
                    return None
            elif type(c) is KVCache:
                n = int(rows)
            else:
                return None
            steps.append((c, n))
        occ = getattr(ring, "occ", None)
        return (ring, steps, None if occ is None else occ.copy(),
                getattr(ring, "skip", None))

    def accept_undo(self, mark) -> None:
        """Take back the one accept made since ``accept_mark``."""
        ring, steps, occ, skip = mark
        if getattr(self, "_tail", None) is not None and steps:
            self._tail.trim(max(n for _, n in steps))
        for c, n in steps:
            if c.trim(n) != n:
                raise RuntimeError(
                    f"{type(c).__name__}.trim({n}) refused; the drafter "
                    "context is now inconsistent")
        if occ is not None:
            ring.occ[:] = occ
            ring.skip = skip

    def discard_block(self) -> None:
        """Forget a drafted block the engine never verifies."""

    def counted_ok(self, block: int, most: int, ahead: int = 0) -> bool:
        """Whether a ``draft_block(counted=(dead, most))`` of ``block`` rows
        can run on the ring once ``ahead`` more rows are committed."""
        return (bool(self._cache) and isinstance(self.rope, nn.RoPE)
                and all(layer.self_attn.ring_counted_ok(c, block, most, ahead)
                        for layer, c in zip(self.layers, self._cache)))

    def accept_trim(self, mark, dead: int):
        """Drop the last ``dead`` rows of the one accept made since
        ``mark``. Returns the mark that takes back what is left."""
        ring, steps, occ, skip = mark
        if getattr(self, "_tail", None) is not None and steps:
            self._tail.trim(min(int(dead), max(n for _, n in steps)))
        out = []
        for c, n in steps:
            d = min(int(dead), n)
            if d and c.trim(d) != d:
                raise RuntimeError(
                    f"{type(c).__name__}.trim({d}) refused; the drafter "
                    "context is now inconsistent")
            out.append((c, n - d))
        return (ring, out, occ, skip)

    # --- parked row contexts (mixed batches) ---------------------------------

    def row_context_open(self, input_ids, hidden: mx.array, bonus_token,
                         kv=None) -> list:
        """A parked context for a row the engine decodes plain: a ring of its
        own, seeded from the row's prompt capture as the live ring is."""
        del input_ids, bonus_token, kv
        ring = self.make_cache()
        if int(hidden.shape[1]):
            limit = self.hidden_capture_limit
            self.append_context(
                self._captures(hidden[:, -limit:] if limit else hidden), ring)
        return ring

    def row_context_ingest(self, ctx: list, verify_hidden: mx.array,
                           new_tokens: List[int], accepted: int) -> None:
        """Inject a plain round's committed positions into a parked ring."""
        del new_tokens
        self.append_context(
            self._captures(verify_hidden[:, : int(accepted) + 1]), ctx)

    def row_context_state(self, ctx: list) -> List[Any]:
        return [c.state for c in ctx]

    def row_context_adopt(self, ctx: list, sampler, token_dtype=mx.int32,
                          greedy: bool = False, stash=None) -> None:
        """Make a parked ring the live one. The drafter holds no live row."""
        del sampler, token_dtype, greedy, stash
        self._cache = ctx


class DFlash2Drafter(DFlashDrafter):
    """DFlash 2: conv-wrapped draft layers and a selector-walked block."""

    layer_class = DFlash2DecoderLayer
    kind_label = "dflash2"
    stochastic_draft = True

    def __init__(self, config: DFlashConfig):
        if not config.is_dflash2:
            raise ValueError("DFlash2Drafter needs selector_top_k > 0")
        if config.conv_kernel_size <= 0 or config.conv_group_size <= 0:
            raise ValueError("DFlash2Drafter needs conv_kernel_size and conv_group_size")
        super().__init__(config)
        self.candidate_selector = CandidateSelector(
            config.hidden_size, config.vocab_size, config.selector_rank,
            config.selector_top_k)
        # Outside the parameter tree: the lattice and path of the last greedy
        # draft.
        object.__setattr__(self, "_round_path", None)
        # Row cut: a round verifies the leading drafts whose whole prefix
        # the selector gives at least this chance, and always the first.
        # 0 verifies the whole block. The loader sets it per target.
        object.__setattr__(self, "row_floor", 0.0)
        # A draft_block(defer_cut=True) leaves its cut here: the lazy keep
        # count and the round's stash.
        object.__setattr__(self, "_pending_cut", None)

    # --- head over the used token ids --------------------------------------------

    def make_cache(self, left_padding: Optional[List[int]] = None) -> list:
        return _Ring(super().make_cache(left_padding))

    def prefill_from_target_hidden(self, input_ids, hidden, bonus_token,
                                   sampler, token_dtype=mx.int32,
                                   greedy=False, stash=None) -> None:
        _open_ring(self._cache, int(self.config.vocab_size), input_ids)
        return super().prefill_from_target_hidden(
            input_ids, hidden, bonus_token, sampler, token_dtype, greedy, stash)

    def row_context_open(self, input_ids, hidden, bonus_token, kv=None) -> list:
        ring = super().row_context_open(input_ids, hidden, bonus_token, kv)
        _open_ring(ring, int(self.config.vocab_size), input_ids)
        return ring

    def row_context_ingest(self, ctx, verify_hidden, new_tokens, accepted) -> None:
        _note_tokens(ctx, new_tokens)
        return super().row_context_ingest(ctx, verify_hidden, new_tokens, accepted)

    def _head_skip(self) -> Optional[tuple]:
        """The token-id range the next draft's head skips, or None for the
        whole head: the longest run of 128-id blocks no token of this
        context (prompt, commits, anchors) falls in, while it is at least
        an eighth of the vocabulary. Unknown until the prompt ids arrive."""
        ring: Any = self._cache
        if not _HEAD_SUBSET or getattr(ring, "occ", None) is None:
            return None
        if ring.skip is None:
            ring.skip = _longest_free_run(ring.occ, int(self.config.vocab_size)) or ()
        return ring.skip or None

    def _draft_logits(self, h: mx.array):
        """Draft-row logits and the id range they lack (None: all ids)."""
        skip = self._head_skip()
        if skip is not None:
            logits = head_rows(self.lm_head, h, *skip)
            if logits is not None:
                if self.config.output_multiplier != 1.0:
                    logits = logits * self.config.output_multiplier
                cap = self.config.final_logit_softcapping
                if cap:
                    logits = mx.tanh(logits / cap) * cap
                return logits, skip
        return self._logits(h), None

    @property
    def round_estimate(self) -> Optional[mx.array]:
        """path_estimate of the last draft, None unless it was greedy. Built
        on read, so a round the engine does not gate builds nothing."""
        path = self._round_path
        if path is None:
            return None
        first, edges, sel, lp = path
        if lp is None:
            return path_estimate(first, edges, sel)
        return 1.0 + mx.exp(mx.cumsum(lp)).sum()

    supports_deferred_row_cut = True
    # draft_block(lazy_bonus=True) and note_bonus.
    supports_lazy_bonus = True

    def note_bonus(self, token) -> bool:
        """Report the bonus of a ``lazy_bonus`` block, or every token of its
        round when the commit noted none. False when one falls in the id
        range that block's head skipped: a block drafted from the known
        tokens would have used another range."""
        ring: Any = self._cache
        before = getattr(ring, "skip", None)
        _note_tokens(ring, [int(token)] if isinstance(token, int)
                     else [int(t) for t in token])
        return getattr(ring, "skip", None) == before

    @property
    def row_cut_pending(self) -> bool:
        """Whether the last block left its row cut to the caller."""
        return self._pending_cut is not None

    def pending_row_cut(self) -> int:
        """Drafts the pending row cut keeps of the last ``defer_cut``
        block. Waits for the draft."""
        return max(1, int(self._pending_cut[0].item()))

    def pending_row_cut_array(self) -> mx.array:
        """The pending row cut's keep count as drafted, not read."""
        return self._pending_cut[0]

    def discard_block(self) -> None:
        object.__setattr__(self, "_round_path", None)
        object.__setattr__(self, "_pending_cut", None)

    def apply_row_cut(self, n: int) -> None:
        """Close the pending cut at ``n`` drafts: the round's stash rows and
        path are trimmed to the drafts the caller verifies."""
        pend = self._pending_cut
        object.__setattr__(self, "_pending_cut", None)
        if pend is None:
            return
        stash = pend[1]
        _note_cut_draft(stash, pend[2], n)
        # The engine pairs one stash row with each draft it verifies.
        for rows in ((stash.q, stash.pq, stash.top2) if stash is not None else ()):
            if rows is not None:
                del rows[n:]
        path = self._round_path
        if path is not None:
            object.__setattr__(self, "_round_path", _cut_path(path, n))

    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache,
        block_size: int,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
        stash: Optional[DraftStash] = None,
        defer_cut: bool = False,
        lazy_bonus: bool = False,
        counted=None,
    ) -> mx.array:
        """``defer_cut`` returns the whole block unevaluated and leaves the
        row cut to the caller (``pending_row_cut``, ``apply_row_cut``), so
        the caller can build the verify while the draft runs.

        ``lazy_bonus`` takes ``last_bonus`` as an array that is not read
        here, so the block can be submitted behind the graph that yields the
        bonus. The caller reports the token through ``note_bonus`` once it
        is known.

        ``counted`` is ``(dead, most)``: the commit ahead wrote ``most + 1``
        ring rows and the last ``dead`` (an array not read here) are not
        live; the caller trims them once the count is known
        (``counted_ok``, ``accept_trim``). Same math as a block drafted
        after the trim, in another key order."""
        del hidden, cache
        object.__setattr__(self, "_round_path", None)
        object.__setattr__(self, "_pending_cut", None)
        if stash is not None:
            stash.cut = None
        if lazy_bonus:
            block = mx.concatenate(
                [last_bonus.reshape(1, 1).astype(token_dtype),
                 self._block_tokens(0, block_size, token_dtype)[:, 1:]], axis=1)
        else:
            block = self._block_tokens(last_bonus, block_size, token_dtype)
            # An int bonus is read as it came: .item() on the block waits for
            # the stream.
            _note_tokens(self._cache, [last_bonus if isinstance(last_bonus, int)
                                       else int(block[0, 0].item())])  # pyright: ignore[reportArgumentType]
        h = (self._draft_hidden(block) if counted is None
             else self._draft_hidden(block, counted))[:, 1:]
        logits, skip = self._draft_logits(h)
        selector = self.candidate_selector
        rows = (h[0], logits[0], block[0, 0], skip)
        floor = float(self.row_floor or 0.0)
        if not greedy and stash is not None and (stash.pq is not None
                                                 or stash.top2 is not None):
            # The sampled walk that feeds the p/q and top-2 logs has no test
            # of a cut draft, so a cut there would bias the token at the cut.
            # A greedy draft keeps its cut.
            floor = 0.0
        if greedy and stash is None:
            walked = selector.walk(*rows, floor)
            first = edges = None
            if walked is not None:
                cands, toks, sel, lp, keep = walked[:5]
                toks = toks[None]
            else:
                cands, first, edges = selector.lattice(*rows)
                walked = selector_walk(cands, first, edges, floor)
                if walked is not None:
                    toks, sel, lp, keep = walked[0][None], walked[1], walked[2], walked[3]
                else:
                    sel = greedy_path(first, edges)
                    toks = mx.take_along_axis(cands, sel[:, None], axis=-1).reshape(1, -1)
                    lp = keep = None
            path = (first, edges, sel, lp)
            if floor > 0:
                if keep is None:
                    keep = rows_over_floor(path_logprobs(first, edges, sel), floor)  # pyright: ignore[reportArgumentType]
                if defer_cut:
                    mx.async_eval(toks, keep)
                    object.__setattr__(self, "_round_path", path)
                    object.__setattr__(self, "_pending_cut", (keep, None, toks))
                    return toks
                mx.eval(toks, keep)
                n = max(1, int(keep.item()))  # pyright: ignore[reportArgumentType]
                toks, path = toks[:, :n], _cut_path(path, n)
            object.__setattr__(self, "_round_path", path)
            return toks
        # Sequential: each row is the lattice row of the realized predecessor.
        # Draws happen on the compact k-wide row; only stash rows are widened.
        vocab = int(self.config.vocab_size)
        walked = None
        stoch = (not greedy and stash is not None and stash.q is not None
                 and stash.pq is None and stash.top2 is None)
        from .speculative import _STOCH_DRAFT

        if stoch:
            fused = selector.walk(*rows, floor, _STOCH_DRAFT)
            if fused is not None:
                cands, walked = fused[0], fused[1:]
        if walked is None:
            cands, first, edges = selector.lattice(*rows)
            if stoch:
                walked = selector_walk(cands, first, edges, floor, _STOCH_DRAFT)
        if walked is not None:
            # The proposal rows the draws used, widened for the verifier.
            stash.q.extend(_scatter(walked[4], cands, 0.0, vocab))  # pyright: ignore[reportArgumentType, reportPossiblyUnboundVariable]
            out, keep = walked[0][None], walked[3]
        else:
            toks, lps = [], []
            row, support = first[None], cands[0][None]  # pyright: ignore[reportPossiblyUnboundVariable]
            for p in range(cands.shape[0]):  # pyright: ignore[reportPossiblyUnboundVariable]
                tok, idx = draw_rows(row, support, vocab=vocab, greedy=greedy,
                                     sampler=sampler, stash=stash)
                toks.append(tok[0])
                if floor > 0:
                    r = row[0].astype(mx.float32)
                    lps.append(r[idx[0]] - mx.logsumexp(r))
                if p + 1 < cands.shape[0]:  # pyright: ignore[reportPossiblyUnboundVariable]
                    row, support = edges[p][idx[0]][None], cands[p + 1][None]  # pyright: ignore[reportPossiblyUnboundVariable]
            out = mx.stack(toks)[None]
            keep = rows_over_floor(mx.stack(lps), floor) if floor > 0 else None
        if floor > 0:
            if defer_cut:
                mx.async_eval(out, keep)
                object.__setattr__(self, "_pending_cut", (keep, stash, out))
                return out
            mx.eval(out, keep)
            n = max(1, int(keep.item()))  # pyright: ignore[reportArgumentType]
            _note_cut_draft(stash, out, n)
            out = out[:, :n]
            # The engine pairs one stash row with each draft it verifies.
            for rows in ((stash.q, stash.pq, stash.top2) if stash is not None else ()):
                if rows is not None:
                    del rows[n:]
        return out


# --- target side --------------------------------------------------------------

if TYPE_CHECKING:
    # The hooks are mixed in ahead of this class; giving pyright the real
    # base resolves the super() calls against the vendored method set.
    from gmlx.models.qwen35.language import LanguageModel as _CaptureBase
else:
    _CaptureBase = object


class DFlashCaptureHooks(_CaptureBase):
    """Packed-hidden capture for owned qwen3.5 LanguageModels.

    While armed, every hidden the engine sees is ``[trunk | cap ...]``: the
    final normed hidden followed by the raw outputs of the captured layers,
    in layer order. The stock ``__call__`` (mrope position resolution, head)
    stays inherited; this wrapper only installs the owned forward's capture
    sink around it and repacks ``hidden_states[-1]``. The logits hooks slice
    the trunk back out, materialized, before the quantized head.

    ``speculative_verify_hidden`` and ``speculative_verify_logits`` both go
    through ``self(...)`` with ``return_hidden`` set, so both return the
    packed hidden (with correct logits): the contract the drafter wants, and
    independent of which hook ``_mtp_verify_target`` tries first.
    """

    _dflash_capture: Optional[tuple] = None

    def set_dflash_capture(self, layer_ids) -> None:
        # Around nn.Module.__setattr__, which would route a tuple into the
        # parameter tree.
        object.__setattr__(self, "_dflash_capture",
                           tuple(int(i) for i in layer_ids) or None)

    def _dflash_trunk(self, hidden: mx.array) -> mx.array:
        if self._dflash_capture is None:
            return hidden
        # The trunk lead is a strided view of the packed hidden, and the head
        # is a quantized kernel that reads the buffer directly; slicing
        # lazily hands it the packed strides. Materialize first.
        return mx.contiguous(hidden[..., : int(self.args.hidden_size)])

    def __call__(self, inputs, inputs_embeds=None, mask=None, cache=None, **kwargs):
        ids = self._dflash_capture
        if ids is None or not kwargs.get("return_hidden"):
            return super().__call__(inputs, inputs_embeds=inputs_embeds,
                                    mask=mask, cache=cache, **kwargs)
        sink: list = []
        object.__setattr__(self.model, "_dflash_sink", sink)
        object.__setattr__(self.model, "_dflash_layers", ids)
        try:
            out = super().__call__(inputs, inputs_embeds=inputs_embeds,
                                   mask=mask, cache=cache, **kwargs)
        finally:
            object.__setattr__(self.model, "_dflash_sink", None)
        final = out.hidden_states[-1]
        if len(sink) != len(ids):
            if int(final.shape[1]) == 0:
                return out
            raise RuntimeError(
                f"dflash capture collected {len(sink)} of {len(ids)} layers")
        out.hidden_states = [*out.hidden_states[:-1],
                             mx.concatenate([final, *sink], axis=-1)]
        return out

    def speculative_logits_from_hidden(self, hidden: mx.array) -> mx.array:
        return super().speculative_logits_from_hidden(self._dflash_trunk(hidden))

    def speculative_argmax_from_hidden(self, hidden: mx.array):
        return super().speculative_argmax_from_hidden(self._dflash_trunk(hidden))



class DSparkDFlashDrafter(DFlashDrafter):
    """DFlash backbone with the DSpark bigram and confidence heads."""

    kind_label = "dspark"

    def __init__(self, config: DFlashConfig):
        super().__init__(config)
        if not config.is_dspark:
            raise ValueError("DSparkDFlashDrafter needs markov_rank > 0")
        hidden = int(config.hidden_size)
        rank = int(config.markov_rank)
        self.markov_w1 = nn.Embedding(config.vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, config.vocab_size, bias=False)
        if config.confidence_head:
            self.conf_proj = nn.Linear(hidden + rank, 1, bias=True)
        if config.log_snr_range is not None:
            self.log_snr_fc1 = nn.Linear(_LOG_SNR_FEATURES, hidden, bias=True)
            self.log_snr_fc2 = nn.Linear(hidden, hidden, bias=True)
        # Confidence below this cuts the block; 0 keeps every row.
        self._confidence_tau = env_float("GMLX_DSPARK_CONF", 0.0)
        # Outside the parameter tree: a per-block-size memo of the
        # noise-level embedding, which is constant for the model.
        object.__setattr__(self, "_snr_embed", {})

    def _rows(self, block_size: int) -> int:
        return int(block_size) - 1 if self.config.sample_from_anchor else int(block_size)

    def _block_tokens(self, last_bonus, block_size: int, token_dtype) -> mx.array:
        if block_size > self._native_block_size:
            raise RuntimeError(
                f"{type(self).__name__} drafts at most "
                f"{self._native_block_size - 1} token(s)/round; got "
                f"block_size={block_size} - cap_at_configured_depth should "
                "have clamped it")
        if not self._cache:
            raise RuntimeError("reset(target_model) must run before draft_block()")
        mask_id = int(self.config.mask_token_id)
        bonus = (int(last_bonus) if isinstance(last_bonus, int)
                 else int(last_bonus.reshape(-1)[0].item()))
        rows = self._rows(block_size)
        return mx.array([[bonus] + [mask_id] * (rows - 1)], dtype=token_dtype)

    def _log_snr_embed(self, rows: int, dtype) -> mx.array:
        cached = self._snr_embed.get(rows)
        if cached is None or cached.dtype != dtype:
            feat = log_snr_features(rows)
            cached = self.log_snr_fc2(nn.silu(self.log_snr_fc1(feat))).astype(dtype)
            self._snr_embed[rows] = cached
        return cached

    def _embed_input_tokens(self, tokens: mx.array) -> mx.array:
        h = super()._embed_input_tokens(tokens)
        if self.config.log_snr_range is not None:
            h = h + self._log_snr_embed(int(tokens.shape[1]), h.dtype)[None]
        return h

    def chain(self, hidden: mx.array, logits: mx.array, anchor: mx.array):
        """Bias every row of one block by its predecessor's pick.

        ``hidden`` [R, H] are the normed draft rows, ``logits`` [R, V] their
        head outputs, ``anchor`` the token the block starts from. Row i's
        predecessor is the anchor for i == 0 and the argmax of row i-1's
        biased logits after that. Returns ``(rows [R, V] float32, confs [R]
        or None)``; the mask token is excluded from every row.
        """
        vocab = int(self.config.vocab_size)
        mask = mx.zeros((vocab,), dtype=mx.float32)
        mask = mask.at[int(self.config.mask_token_id)].add(float("-inf"))
        conf_on = self.config.confidence_head and self._confidence_tau > 0.0
        prev = anchor.reshape(1)
        rows, confs = [], []
        for i in range(int(logits.shape[0])):
            m = self.markov_w1(prev)                                    # [1, R]
            bias = self.markov_w2(m)[0].astype(mx.float32)
            row = logits[i].astype(mx.float32) + bias + mask
            rows.append(row)
            if conf_on:
                feat = mx.concatenate(
                    [hidden[i : i + 1].astype(mx.float32), m.astype(mx.float32)],
                    axis=-1)
                confs.append(mx.sigmoid(self.conf_proj(feat))[0, 0])
            prev = mx.argmax(row).reshape(1)
        return mx.stack(rows), (mx.stack(confs) if conf_on else None)

    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache,
        block_size: int,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
        greedy: bool = False,
        stash: Optional[DraftStash] = None,
    ) -> mx.array:
        """One round: the block's rows through the layers, then the bigram
        chain over every row (the anchor row included) and, with a confidence
        threshold set, a cut to the confident prefix of at least one row."""
        del hidden, cache
        block = self._block_tokens(last_bonus, block_size, token_dtype)
        h = self._draft_hidden(block)
        if not self.config.sample_from_anchor:
            h = h[:, 1:]
        rows, confs = self.chain(h[0], self._logits(h)[0], block[0, 0])
        if confs is not None:
            keep = 0
            values: Any = confs.tolist()
            for c in values:
                if c < self._confidence_tau:
                    break
                keep += 1
            rows = rows[:max(1, keep)]
        toks, _ = draw_rows(rows, None, vocab=int(self.config.vocab_size),
                            greedy=greedy, sampler=sampler, stash=stash)
        return toks[None]
