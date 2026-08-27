"""Pin a streaming model's every-token (non-expert) weights in RAM.

--stream-experts assumes only the routed-expert stacks stream from disk;
attention, routers, shared experts, norms and the lm head (the every-token
weights) are meant to stay resident (see the prefetch module's design
note). Nothing enforced that: they live in evictable file-backed mmap
pages, and on a box where model plus page cache exceed RAM the kernel
evicts them between uses. Each decode token then re-faults them all from
disk,
which on a large MoE saturates the SSD before the experts read a byte
(Kimi-K3: ~50 GB/token of every-token-weight page-ins vs ~7 GB of
expert misses, a hard
~9 s/token floor on a 6 GB/s NVMe).

This module makes the residency assumption true. Each shard is mapped once
(read-only, shared) and every non-expert tensor range is mlocked through
that mapping; the unified buffer cache shares physical pages across
mappings, so the runtime's own zero-copy views hit wired pages and the
only per-token disk traffic left is expert misses. Pinning runs before the
decode feeder sizes its arena, so arena auto-sizing sees the reduced
budget.

Opt-out with GMLX_PIN_WEIGHTS=0. A partial pin (wire-limit refusal mid-way)
is announced and left partial: unpinned ranges just stay cache-managed.
"""

from __future__ import annotations

import ctypes
import os
from concurrent.futures import ThreadPoolExecutor

from gmlx.envflags import env_bool

# Every-token set = everything the expert prefetcher does not stream
# (keep the two definitions in lockstep via the shared regex).
from .prefetch import _EXPS_RE

_PAGE = os.sysconf("SC_PAGESIZE") if hasattr(os, "sysconf") else 16384
_MAP_FAILED = ctypes.c_void_p(-1).value

# Parallel first-touch: mlock faults its range in synchronously, and a
# single call runs at one thread's demand-fault bandwidth. Chunked across
# a pool the initial pin runs at sequential-read speed instead.
_CHUNK = 256 << 20
_WORKERS = 8

# Refuse a pin that would leave the box no working room: wired
# every-token weights + decode arena + KV + anon must fit under RAM with margin.
_MAX_FRACTION = 0.6


def _libc():
    lc = ctypes.CDLL(None, use_errno=True)
    lc.mmap.restype = ctypes.c_void_p
    lc.mmap.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int64,
    ]
    lc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lc.munlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    return lc


def every_token_ranges(gguf_path: str) -> dict[str, list[tuple[int, int]]]:
    """``shard_path -> [(offset, nbytes), ...]``: merged, page-aligned byte
    ranges of every non-expert tensor. Headers and metadata are read once
    at load and stay unpinned."""
    from gmlx.load.headerscan import scan_gguf
    from gmlx.load.preflight import find_split_shards

    out: dict[str, list[tuple[int, int]]] = {}
    for path in find_split_shards(gguf_path):
        scan = scan_gguf(path, array_limit=0)
        raw = []
        for t in scan.tensors:
            if _EXPS_RE.fullmatch(t.name):
                continue
            start = scan.data_offset + t.offset
            raw.append((start, start + t.nbytes))
        raw.sort()
        merged: list[list[int]] = []
        for a, b in raw:
            a &= ~(_PAGE - 1)
            b = (b + _PAGE - 1) & ~(_PAGE - 1)
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        out[path] = [(a, b - a) for a, b in merged]
    return out


class WeightsPin:
    """One model's pinned every-token weights: per-shard mappings plus their mlocked
    ranges. ``close()`` (or GC) unlocks and unmaps."""

    def __init__(self, ranges: dict[str, list[tuple[int, int]]]):
        self._libc = _libc()
        self._maps: list[tuple[int, int]] = []  # (base, map_len)
        self._locked: list[tuple[int, int]] = []  # (addr, len)
        self.total_bytes = sum(n for r in ranges.values() for _, n in r)
        self.pinned_bytes = 0
        self._refused = False
        tasks: list[tuple[int, int]] = []
        try:
            for path, rs in ranges.items():
                size = os.stat(path).st_size
                fd = os.open(path, os.O_RDONLY)
                try:
                    base = self._libc.mmap(None, size, 0x1, 0x1, fd, 0)
                finally:
                    os.close(fd)
                if base is None or base == _MAP_FAILED:
                    raise OSError(
                        ctypes.get_errno(), f"mmap failed for {path}")
                self._maps.append((base, size))
                for off, n in rs:
                    for c in range(0, n, _CHUNK):
                        tasks.append((base + off + c, min(_CHUNK, n - c)))
            with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
                for (addr, n), ok in zip(
                        tasks, pool.map(self._lock_one, tasks)):
                    if ok:
                        self._locked.append((addr, n))
                        self.pinned_bytes += n
                    else:
                        self._refused = True
        except BaseException:
            self.close()
            raise

    def _lock_one(self, task: tuple[int, int]) -> bool:
        if self._refused:  # first refusal stops the rest (wire limit hit)
            return False
        addr, n = task
        return self._libc.mlock(ctypes.c_void_p(addr), n) == 0

    def close(self) -> None:
        lc = getattr(self, "_libc", None)
        if lc is None:
            return
        for addr, n in self._locked:
            lc.munlock(ctypes.c_void_p(addr), n)
        self._locked = []
        for base, n in self._maps:
            lc.munmap(ctypes.c_void_p(base), n)
        self._maps = []

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass  # GC-time cleanup must never raise


def maybe_pin_weights(gguf_path: str | None) -> WeightsPin | None:
    """A WeightsPin over the model's non-expert ranges, or None with a printed
    reason. Called only for streaming-mode installs."""
    if gguf_path is None:
        return None
    if not env_bool("GMLX_PIN_WEIGHTS", True):
        print("[stream] weight pin off (GMLX_PIN_WEIGHTS=0); every-token "
              "weights stay cache-managed")
        return None
    try:
        ranges = every_token_ranges(gguf_path)
        total = sum(n for rs in ranges.values() for _, n in rs)
        try:
            ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGESIZE")
        except (ValueError, OSError):
            ram = 0
        if ram and total > _MAX_FRACTION * ram:
            print(
                f"[stream] weight pin skipped: every-token weights "
                f"({total / 1e9:.1f} GB) exceed {int(_MAX_FRACTION * 100)}% "
                f"of RAM ({ram / 1e9:.0f} GB); staying cache-managed")
            return None
        pin = WeightsPin(ranges)
        if pin._refused:
            print(
                f"[stream] weight pin partial: wired "
                f"{pin.pinned_bytes / 1e9:.1f} of {pin.total_bytes / 1e9:.1f} "
                "GB (mlock refused); the rest stays cache-managed")
        else:
            print(
                f"[stream] every-token weights pinned: "
                f"{pin.pinned_bytes / 1e9:.1f} GB wired "
                "(GMLX_PIN_WEIGHTS=0 disables)")
        return pin
    except Exception as e:
        print(f"[stream] weight pin unavailable ({e}); every-token weights "
              "stay cache-managed")
        return None
