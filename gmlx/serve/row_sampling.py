"""Per-row sampler and logprob settings for the batch engine.

Each request carries its own sampler, seed and logprob settings to its
batch row through ``BatchGenerator.insert(row_settings=...)``. The
generator keeps them by uid in one dict for the row's life, so a greedy
request that joins a sampled batch stays greedy, and a request that wants
logprobs gets them in any batch.

Lifetime. ``remove(uid)`` keeps the entry: the tick guard requeues the
same uid, and a cancelled speculative row stays in the rounds until its
stop check fires. The entry goes when a response that finishes the uid
leaves ``BatchGenerator.next``, and the uid joins the generator's retired
set. Past ``MAX_ROWS`` entries an insert drops every uid that is not live,
never a live one, and the retired marks of uids that are not live.

Speech. A spoken reply's record is the ``speech`` field. It is mutable
and never requeues, so ``remove(uid)``, the finish and a failed requeue
each close it (:mod:`gmlx.serve.speech_rows`).

Seeds. Each row's sampler is built without the request's seed, so its base
seed is the stock default. The seed rides in the settings and is
registered on the row's own sampler instance. A seed guarantees the row's
own key stream; batched matmul reduction order still shifts logits at
float tolerance, so outputs match only across runs with the same batch
composition.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

MAX_ROWS = 4096

GREEDY_KEY = ("greedy",)


@dataclass(frozen=True)
class RowSettings:
    """One row's sampling and logprob settings. New per-row state adds a
    field here and rides the same insert argument and lifetime."""

    sampler: Callable | None = None   # None is greedy; this row's own instance
    seed: int | None = None
    logprobs: bool = False
    top_k: int = 0                    # logprob alternatives, 0..cap
    key: tuple = GREEDY_KEY           # sampler config without the seed
    lora: Any = None                  # gmlx.serve.lora_request.RowLora, salted
    speech: Any = None                # gmlx.serve.speech_rows.SpeechRow


def sampler_key(args) -> tuple:
    """The sampler config of a request's ``GenerationArguments``, without
    its seed. Rows with equal keys draw through one sampler call."""
    if float(getattr(args, "temperature", 0.0) or 0.0) == 0.0:
        return GREEDY_KEY
    return tuple(getattr(args, name, None) for name in (
        "temperature", "top_p", "top_k", "min_p", "top_n_sigma", "p_less",
        "typical_p"))


def takes_row_seeds(sampler) -> bool:
    return isinstance(getattr(sampler, "_kq_row_seeds", None), dict)


def register_seed(sampler, uid, seed) -> None:
    if seed is not None and takes_row_seeds(sampler):
        sampler._kq_row_seeds[uid] = int(seed)


def _draw(sampler, lp: mx.array, row_ids, positions) -> mx.array:
    target = getattr(sampler, "sample_target", None)
    if callable(target) and row_ids is not None and positions is not None:
        return target(lp, row_ids=row_ids, positions=positions)
    return sampler(lp)


@contextlib.contextmanager
def _seeded(sampler, uids: list, seeds: dict) -> Iterator[None]:
    """Publish the draw's row uids and seeds on ``sampler`` for one call."""
    if not seeds or not takes_row_seeds(sampler):
        yield
        return
    saved_rows = getattr(sampler, "_kq_rows", None)
    saved_seeds = sampler._kq_row_seeds
    sampler._kq_rows = list(uids)
    sampler._kq_row_seeds = dict(seeds)
    try:
        yield
    finally:
        sampler._kq_rows = saved_rows
        sampler._kq_row_seeds = saved_seeds


class RowSampler:
    """A sampler over a batch whose rows each carry their own settings.

    ``bind(uids)`` names the physical rows, in order, for the draws inside
    it. A draw over ``R * n`` rows (a flattened ``[R, n, V]`` verify) maps
    row ``j`` to ``uids[j // n]``. Rows group by ``RowSettings.key``:
    greedy rows take the argmax, and each other group draws once on its
    slice through one of its samplers, with the group's seeds published
    around the draw.
    """

    def __init__(self, settings: Mapping[Any, RowSettings], default: RowSettings):
        self._settings = settings
        self._default = default
        self._uids: list | None = None

    def row(self, uid) -> RowSettings:
        if uid is None:
            return self._default
        return self._settings.get(uid, self._default)

    @contextlib.contextmanager
    def bind(self, uids) -> Iterator[RowSampler]:
        saved = self._uids
        self._uids = list(uids)
        try:
            yield self
        finally:
            self._uids = saved

    def rebind(self, uids) -> None:
        """Bind ``uids`` until the next bind, for a caller that cannot hold
        a context across its rounds."""
        self._uids = list(uids)

    def rows(self) -> list[RowSettings]:
        return [self.row(u) for u in (self._uids or ())]

    def effective(self) -> RowSampler | Callable | None:
        """None when every bound row is greedy; the one sampler when every
        row shares its config and no row is seeded; else this sampler."""
        rows = self.rows()
        if all(r.sampler is None for r in rows):
            return None
        if len({r.key for r in rows}) == 1 and all(r.seed is None for r in rows):
            return rows[0].sampler
        return self

    def __call__(self, lp: mx.array) -> mx.array:
        return self._sample(lp, None, None)

    def sample_target(self, lp: mx.array, *, row_ids, positions) -> mx.array:
        return self._sample(lp, row_ids, positions)

    def _sample(self, lp: mx.array, row_ids, positions) -> mx.array:
        uids = self._uids or []
        n, m = len(uids), int(lp.shape[0])
        if n == 0 or m % n:
            raise RuntimeError(f"row sampler bound to {n} rows, draw has {m}")
        rep = m // n
        row_uids = [uids[j // rep] for j in range(m)]
        groups: dict[tuple, list[int]] = {}
        for j, u in enumerate(row_uids):
            groups.setdefault(self.row(u).key, []).append(j)
        parts = []
        for key, js in groups.items():
            sub = lp if len(js) == m else lp[mx.array(js, dtype=mx.int32)]
            if key == GREEDY_KEY:
                parts.append((js, mx.argmax(sub, axis=-1)))
                continue
            g_uids = [row_uids[j] for j in js]
            sampler = self.row(g_uids[0]).sampler
            if sampler is None:
                parts.append((js, mx.argmax(sub, axis=-1)))
                continue
            seeds = {u: self.row(u).seed for u in g_uids
                     if self.row(u).seed is not None}
            g_rows = None if row_ids is None else [row_ids[j] for j in js]
            g_pos = None if positions is None else [positions[j] for j in js]
            with _seeded(sampler, g_uids, seeds):
                parts.append((js, _draw(sampler, sub, g_rows, g_pos).reshape(-1)))
        if len(parts) == 1:
            return parts[0][1]
        order = [j for js, _ in parts for j in js]
        inverse = [0] * m
        for pos, j in enumerate(order):
            inverse[j] = pos
        joined = mx.concatenate([t.astype(mx.int32) for _, t in parts])
        return joined[mx.array(inverse, dtype=mx.int32)]


def live_uids(gen) -> set:
    """Every uid a ``BatchGenerator`` still serves: queued, in the prompt
    batch, in the generation batch, or in a speculative batch's rows and
    pending injections."""
    live = {s[0] for s in getattr(gen, "_unprocessed_sequences", ())}
    pb = getattr(gen, "_prompt_batch", None)
    if pb is not None:
        live.update(getattr(pb, "uids", ()) or ())
    gb = getattr(gen, "_generation_batch", None)
    if gb is not None:
        live.update(getattr(gb, "uids", ()) or ())
        live.update(getattr(gb, "_all_uids", ()) or ())
        for other in getattr(gb, "_pending_injections", ()) or ():
            live.update(getattr(other, "_all_uids", ()) or ())
    return live


def prune(settings: dict, gen, retired: set | None = None) -> None:
    """Drop every uid that is not live once ``settings`` or ``retired``
    (the uids whose settings went with their finish response) passes
    MAX_ROWS."""
    if len(settings) <= MAX_ROWS and (retired is None or len(retired) <= MAX_ROWS):
        return
    live = live_uids(gen)
    for uid in [u for u in settings if u not in live]:
        del settings[uid]
    if retired is not None:
        retired.intersection_update(live)
