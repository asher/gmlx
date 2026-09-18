"""DeepSeek-V4.1-Flash engram constants against the reference formulas.

The GGUF stores the hash multipliers, the bucket primes and their offsets
as metadata, but the reference derives all three
(``inference/engram.py``: ``compute_hash_multipliers`` and
``EngramLayout.from_args``). A converter that seeded an RNG differently,
or drew the primes per layer instead of from one shared set, would ship
constants that hash into the wrong rows and only show as degraded output.
This regenerates them and checks each table's row count is the sum of its
own primes.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

_FIXTURE = (pathlib.Path(__file__).resolve().parents[1]
            / "fixtures" / "deepseek41_engram_constants.json")


@pytest.fixture(scope="module")
def gguf() -> dict:
    return json.loads(_FIXTURE.read_text())


def _small_primes(limit: int) -> list[int]:
    sieve = bytearray([1]) * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i * i:: i] = bytearray(len(sieve[i * i:: i]))
    return [i for i, ok in enumerate(sieve) if ok]


_TRIAL = _small_primes(4096)


def _is_prime(n: int) -> bool:
    for p in _TRIAL:
        if p * p > n:
            return True
        if n % p == 0:
            return n == p
    return True


def _next_prime(start: int, seen: set[int]) -> int:
    """``engram.find_next_prime``: least prime above ``start``, not reused."""
    c = start + 1
    while not _is_prime(c) or c in seen:
        c += 1
    return c


def test_multipliers_regenerate_from_the_reference_rng(gguf):
    """``compute_hash_multipliers``: one odd draw per (layer, look-back)
    from ``default_rng(10007 * layer)``, bounded so ``id * multiplier``
    cannot overflow int64."""
    comp = gguf["reference_config"]["engram_compressed_vocab_size"]
    bound = max(1, (np.iinfo(np.int64).max // comp) // 2)
    got: list[int] = []
    for layer_id in gguf["layer_ids"]:
        rng = np.random.default_rng(10007 * layer_id)
        draw = rng.integers(low=0, high=bound,
                            size=(gguf["max_ngram_size"],), dtype=np.int64)
        got.extend((draw * 2 + 1).tolist())
    assert got == gguf["multipliers"]


def test_primes_come_from_one_set_shared_across_layers(gguf):
    """``EngramLayout.from_args`` restarts the search at
    ``engram_vocab_size - 1`` for every (layer, n-gram size) group but
    carries one ``seen`` set, so the draws never repeat and the two tables
    differ in size on purpose."""
    start = gguf["reference_config"]["engram_vocab_size"] - 1
    n_ngram = gguf["max_ngram_size"] - 1
    got, seen = [], set()
    for _ in gguf["layer_ids"]:
        for _ in range(n_ngram):
            cur = start
            for _ in range(gguf["n_heads"]):
                cur = _next_prime(cur, seen)
                seen.add(cur)
                got.append(cur)
    assert got == gguf["primes"]
    assert len(seen) == len(gguf["primes"]), "a prime was drawn twice"


def test_offsets_tile_each_table_and_the_rows_are_their_sum(gguf):
    """Offsets are the per-layer cumulative sum, so the bucket ranges are
    disjoint and cover the table exactly."""
    per_layer = gguf["n_heads"] * (gguf["max_ngram_size"] - 1)
    for i, rows in enumerate(gguf["table_rows"]):
        primes = gguf["primes"][i * per_layer:(i + 1) * per_layer]
        offsets = gguf["offsets"][i * per_layer:(i + 1) * per_layer]
        assert offsets == np.cumsum([0, *primes[:-1]]).tolist()
        assert max(o + p for o, p in zip(offsets, primes)) == rows
        assert sum(primes) == rows
