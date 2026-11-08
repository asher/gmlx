"""KL divergence over a text, as llama.cpp's llama-perplexity computes it
(tools/perplexity/perplexity.cpp, MIT).

The text is cut into chunks of ``n_ctx`` tokens. Each chunk runs as one
forward, and the positions from ``n_ctx // 2`` to ``n_ctx - 2`` are
scored. At each scored position the reference log-probs are held as
uint16 over ``[max - 16, max]``, with a float scale and floor in front,
in the record layout of llama-perplexity's ``--kl-divergence-base`` file.
The KL sum runs in float64 over the reference tokens whose log-prob is
above -16. The elementwise steps run in float32, as llama.cpp's do.

This module needs only numpy.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import BinaryIO

import numpy as np

MAGIC = b"_logits_"
LOGP_FLOOR = np.float32(-16.0)
# Rows per numpy pass, which bounds the float64 temporaries.
_SLICE = 32


def record_width(n_vocab: int) -> int:
    """uint16 values per scored position: two float32 (scale, floor), the
    vocabulary, padded to an even count."""
    return 2 * ((n_vocab + 1) // 2) + 4


def scored_span(n_ctx: int) -> tuple[int, int]:
    """(first, count): the first scored position of a chunk and how many
    positions are scored."""
    first = n_ctx // 2
    return first, n_ctx - 1 - first


def chunk_count(n_tokens: int, n_ctx: int, chunks: int | None = None) -> int:
    """The chunks a text of ``n_tokens`` runs: all whole chunks, at most
    ``chunks``. A text under ``2 * n_ctx`` tokens is refused."""
    if n_tokens < 2 * n_ctx:
        raise ValueError(
            f"the text tokenizes to {n_tokens} tokens, and --ctx {n_ctx} "
            f"needs at least {2 * n_ctx}")
    n = n_tokens // n_ctx
    return n if chunks is None else min(chunks, n)


def chunk_ids(tokens: np.ndarray, chunk: int, n_ctx: int,
              bos: int | None = None) -> np.ndarray:
    """The input ids of one chunk, with BOS in place of its first token
    when ``bos`` is given."""
    ids = np.array(tokens[chunk * n_ctx:(chunk + 1) * n_ctx], dtype=np.int32)
    if bos is not None:
        ids[0] = bos
    return ids


def read_text(path: str) -> str:
    """The text file as llama.cpp's ``-f`` reads it: one trailing newline
    is dropped."""
    with open(path, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"{path} is not UTF-8 text: {e}") from None
    return text[:-1] if text.endswith("\n") else text


def _log_sum_exp(logits: np.ndarray, max_logit: np.ndarray) -> np.ndarray:
    sum_exp = np.exp(logits - max_logit[:, None]).sum(axis=1, dtype=np.float64)
    return np.log(sum_exp).astype(np.float32)


def quantize_logprobs(logits: np.ndarray) -> np.ndarray:
    """uint16 records [n, record_width(V)] of float32 logits [n, V]."""
    logits = np.ascontiguousarray(logits, dtype=np.float32)
    n, V = logits.shape
    out = np.zeros((n, record_width(V)), dtype=np.uint16)
    for r0 in range(0, n, _SLICE):
        z = logits[r0:r0 + _SLICE]
        max_logit = z.max(axis=1)
        min_logit = np.maximum(z.min(axis=1), max_logit - np.float32(16))
        lse = _log_sum_exp(z, max_logit)
        min_log_prob = (min_logit - max_logit) - lse
        scale = (max_logit - min_logit) / np.float32(65535)
        head = np.stack([scale, min_log_prob], axis=1).astype("<f4")
        out[r0:r0 + _SLICE, :4] = head.view("<u2")
        # A row with scale 0 has no logit above its floor, so it stays 0.
        inv = np.float32(1) / np.where(scale != 0, scale, np.float32(1))
        d = np.where(z > min_logit[:, None], z - min_logit[:, None],
                     np.float32(0))
        # C converts the int to uint16 modulo 2^16.
        q = np.rint(inv[:, None] * d).astype(np.int64) & 0xFFFF
        out[r0:r0 + _SLICE, 4:4 + V] = q.astype(np.uint16)
    return out


def _split(records: np.ndarray, V: int):
    head = np.ascontiguousarray(records[:, :4]).view("<f4")
    return head[:, 0], head[:, 1], records[:, 4:4 + V]


def kld_rows(logits: np.ndarray,
             records: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(KL per row in float64, whether the argmax agrees) for float32
    logits [n, V] against the reference records [n, record_width(V)]."""
    logits = np.ascontiguousarray(logits, dtype=np.float32)
    n, V = logits.shape
    if records.shape != (n, record_width(V)):
        raise ValueError(f"records {records.shape} do not match logits "
                         f"{logits.shape}")
    kld = np.empty(n, dtype=np.float64)
    same = np.empty(n, dtype=bool)
    for r0 in range(0, n, _SLICE):
        z = logits[r0:r0 + _SLICE]
        scale, min_log_prob, q = _split(records[r0:r0 + _SLICE], V)
        max_logit = z.max(axis=1)
        lse = _log_sum_exp(z, max_logit)
        p_log_base = scale[:, None] * q.astype(np.float32) + min_log_prob[:, None]
        same[r0:r0 + _SLICE] = z.argmax(axis=1) == p_log_base.argmax(axis=1)
        top = max_logit + lse
        term = np.exp(p_log_base) * ((p_log_base - z) + top[:, None])
        term[~(p_log_base > LOGP_FLOOR)] = 0
        kld[r0:r0 + _SLICE] = term.sum(axis=1, dtype=np.float64)
    return kld, same


def _mean_and_uncertainty(s: float, s2: float, count: int) -> tuple[float, float]:
    if count < 1:
        return 0.0, 0.0
    f = s / count
    df = s2 / count - f * f
    df = float(np.sqrt(df / (count - 1))) if df > 0 and count > 10 else 0.0
    return f, df


@dataclass
class KldSums:
    """The running sums llama-perplexity reports from."""

    sum_kld: float = 0.0
    sum_kld2: float = 0.0
    n_same_top: int = 0
    count: int = 0

    def add(self, logits: np.ndarray, records: np.ndarray) -> None:
        kld, same = kld_rows(logits, records)
        self.sum_kld += float(kld.sum())
        self.sum_kld2 += float((kld * kld).sum())
        self.n_same_top += int(same.sum())
        self.count += int(kld.shape[0])

    def mean_kld(self) -> tuple[float, float]:
        return _mean_and_uncertainty(self.sum_kld, self.sum_kld2, self.count)

    def same_top(self) -> tuple[float, float]:
        """The share of positions where the argmax agrees, from 0 to 1."""
        if self.count < 2:
            return (self.n_same_top / self.count if self.count else 0.0), 0.0
        p = self.n_same_top / self.count
        return p, float(np.sqrt(p * (1 - p) / (self.count - 1)))


def reference_bytes(n_ctx: int, n_vocab: int, n_chunk: int) -> int:
    """The size of a reference file."""
    _, n = scored_span(n_ctx)
    return (len(MAGIC) + 12 + 4 * n_ctx * n_chunk
            + 2 * n_chunk * n * record_width(n_vocab))


def write_reference_header(f: BinaryIO, n_ctx: int, n_vocab: int,
                           n_chunk: int, tokens: np.ndarray) -> None:
    """The header of a reference file: the magic, n_ctx, n_vocab, n_chunk
    and the ids of every chunk, as llama-perplexity writes them."""
    f.write(MAGIC)
    f.write(struct.pack("<Iii", n_ctx, n_vocab, n_chunk))
    f.write(np.asarray(tokens[:n_ctx * n_chunk], dtype="<i4").tobytes())


def read_reference_header(f: BinaryIO) -> tuple[int, int, int, np.ndarray]:
    """(n_ctx, n_vocab, n_chunk, tokens) of a reference file."""
    if f.read(len(MAGIC)) != MAGIC:
        raise ValueError("not a reference log-prob file")
    n_ctx, n_vocab, n_chunk = struct.unpack("<Iii", f.read(12))
    raw = f.read(4 * n_ctx * n_chunk)
    if len(raw) != 4 * n_ctx * n_chunk:
        raise ValueError("the reference file ends inside its token list")
    return n_ctx, n_vocab, n_chunk, np.frombuffer(raw, dtype="<i4").copy()


def read_reference_records(f: BinaryIO, n: int, n_vocab: int) -> np.ndarray:
    """The next ``n`` records [n, record_width(n_vocab)]."""
    w = record_width(n_vocab)
    raw = f.read(2 * n * w)
    if len(raw) != 2 * n * w:
        raise ValueError("the reference file ends before its last record")
    return np.frombuffer(raw, dtype="<u2").reshape(n, w)
