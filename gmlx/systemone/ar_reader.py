"""Letter reads on an autoregressive model.

A decision prompt is the state followed by one question. Every pass of a
request shares the state, so the reader prefills that prefix once and runs
each pass's tail against it. Tails run in batches through broadcast views of
the prefix cache: a view hands every row the same prefix state and never
stores the tail, so one prefill serves any number of tails. The answer is
the score of each option letter at the last prompt position, computed in
float32 from the head's letter rows.

The reader is a generator that yields after each forward, so the server can
run chat steps between the forwards of a long decision. Every size it uses
is fixed, never taken from load, so a decision repeats bit for bit."""

from __future__ import annotations

import contextlib
import copy
import time
from typing import Any, Callable, Generator, Optional

import mlx.core as mx
import mlx.nn as nn

from gmlx.gen.generation import encode_prompt

from . import letters

# Tokens one forward may carry: a prefix chunk, or a bucket of padded tails.
# One forward is the longest a chat stream waits for a decision.
FORWARD_TOKENS = 128
# Cap on the per-layer copy the views make: rows x (prefix + tail) keys and
# values, plus rows x the recurrent state.
VIEW_BYTES = 512 << 20
# A forward's fixed cost in tokens, for the bucket plan.
FORWARD_OVERHEAD = 32
READER_VERSION = 1

Step = Generator[None, None, Any]


def _logits(out):
    return getattr(out, "logits", out)


def text_model(model):
    """The mlx-lm model under the server's text-only wrapper, if any."""
    lm = getattr(model, "language_model", None)
    inner = getattr(lm, "_model", None)
    return inner if inner is not None else model


def _trunk_and_head(model) -> tuple[Any, Any]:
    """``(trunk, head)``: the module that returns the final hidden states and
    the one that maps them to logits, or None where the model has no such
    split."""
    m = text_model(model)
    text = getattr(m, "language_model", m)
    trunk = getattr(text, "model", None)
    if trunk is None or not callable(trunk):
        return None, None
    head = getattr(text, "lm_head", None)
    if head is None:
        embed = getattr(trunk, "embed_tokens", None)
        head = getattr(embed, "as_linear", None)
    return trunk, head


def _letter_rows(head, ids: list[int]):
    """The head's rows for ``ids`` as float32 ``(len(ids), hidden)`` plus a
    float32 bias or None, or None for a head this cannot read exactly."""
    from gmlx.load.hadamard_modules import fold_of

    module = getattr(head, "__self__", head)   # a tied head's embedding
    if fold_of(module) is not None:
        return None
    idx = mx.array(ids, dtype=mx.uint32)
    bias = module["bias"][idx].astype(mx.float32) if "bias" in module else None
    if getattr(module, "mode", None) == "kquant":
        import mlx_kquant as kq

        rows = kq.dequantize(module["weight"][idx], module["scales"],
                             module.kquant_type)
    elif isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
        biases = module.get("biases")
        rows = mx.dequantize(
            module["weight"][idx], module["scales"][idx],
            biases[idx] if biases is not None else None,
            group_size=module.group_size, bits=module.bits,
            mode=getattr(module, "mode", "affine"))
    elif isinstance(module, (nn.Linear, nn.Embedding)):
        rows = module["weight"][idx]
    else:
        return None
    return rows.astype(mx.float32), bias


class _KVView:
    """One attention layer's prefix keys and values, broadcast to the rows
    of a forward. The rows' own keys and values join them for this forward
    and are not stored."""

    __slots__ = ("keys", "values", "offset")

    def __init__(self, keys, values, offset: int):
        self.keys = keys
        self.values = values
        self.offset = offset

    def update_and_fetch(self, keys, values):
        b = keys.shape[0]
        pk = mx.broadcast_to(self.keys, (b,) + tuple(self.keys.shape[1:]))
        pv = mx.broadcast_to(self.values, (b,) + tuple(self.values.shape[1:]))
        return mx.concatenate([pk, keys], axis=2), mx.concatenate([pv, values], axis=2)

    def make_mask(self, n: int, return_array: bool = False, window_size=None):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(n, self.offset, return_array, window_size)


class _StateView:
    """One recurrent layer's prefix state, broadcast to the rows of a
    forward. Writes are dropped, so the prefix state stays as it was."""

    __slots__ = ("cache", "lengths", "left_padding")

    def __init__(self, arrays, rows: int):
        self.cache = [None if a is None
                      else mx.broadcast_to(a, (rows,) + tuple(a.shape[1:]))
                      for a in arrays]
        self.lengths = None
        self.left_padding = None

    def __getitem__(self, i):
        return self.cache[i]

    def __setitem__(self, i, value):
        pass

    def advance(self, n: int) -> None:
        pass

    def make_mask(self, n: int):
        return None

    @property
    def state(self):
        return self.cache


def _view_kinds(cache) -> Optional[list[str]]:
    """``"kv"`` or ``"arr"`` per layer when broadcast views can stand in for
    every layer, else None."""
    from gmlx.cache.compat import cache_types

    kv, arr = cache_types("KVCache"), cache_types("ArraysCache")
    kinds = []
    for c in cache:
        if type(c) in kv:
            kinds.append("kv")
        elif type(c) in arr:
            kinds.append("arr")
        else:
            return None
    return kinds


def _fork(cache) -> Optional[list]:
    """A bitwise single-row copy of ``cache`` that a forward can extend, or
    None when a layer cannot be copied."""
    from gmlx.cache.snapshot import _clone_row_faithful

    out = []
    for c in cache:
        f = _clone_row_faithful(c)
        if f is None:
            return None
        out.append(f)
    return out


def plan_buckets(lengths: list[int], fits: Callable[[int, int], bool],
                 overhead: int = FORWARD_OVERHEAD) -> list[list[int]]:
    """Indices of ``lengths`` grouped into forwards: sorted by length, split
    where padding plus a forward's overhead costs least, and every group of
    more than one row within ``fits(rows, width)``."""
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    width = [lengths[i] for i in order]
    n = len(order)
    best = [0.0] + [float("inf")] * n
    cut = [0] * (n + 1)
    for j in range(1, n + 1):
        for i in range(j - 1, -1, -1):
            rows = j - i
            if rows > 1 and not fits(rows, width[j - 1]):
                break
            cost = best[i] + rows * width[j - 1] + overhead
            if cost < best[j]:
                best[j], cut[j] = cost, i
    groups, j = [], n
    while j > 0:
        groups.append(order[cut[j]:j])
        j = cut[j]
    return groups[::-1]


class LetterTokens:
    """The chat prompt of a letter pass: the pass text as the only user
    message, thinking off, the generation prompt on. Ids are kept per text,
    so a request can tokenize its passes before the reads start."""

    def __init__(self, processor, *, lock=None):
        self.wrapper = getattr(processor, "_wrapper", processor)
        self.lock = lock if lock is not None else contextlib.nullcontext()
        self._ids: dict[str, list[int]] = {}

    def render(self, text: str) -> str:
        with self.lock:
            out: Any = self.wrapper.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
        return str(out)

    def ids(self, text: str) -> list[int]:
        got = self._ids.get(text)
        if got is None:
            rendered = self.render(text)
            with self.lock:
                got = self._ids[text] = list(encode_prompt(self.wrapper, rendered))
        return got

    def letter_ids(self) -> list[int]:
        with self.lock:
            ids = [self.wrapper.encode(L, add_special_tokens=False)
                   for L in letters.LETTERS]
        if any(len(i) != 1 for i in ids) or len({i[0] for i in ids}) != len(ids):
            raise ValueError("the tokenizer does not give each option letter "
                             "its own token")
        return [i[0] for i in ids]


def split_point(tokens: LetterTokens, state: str) -> tuple[list[int], int]:
    """The prefix ids every pass on ``state`` shares, and their length: the
    common start of the prompt that ends at the state and of a pass prompt.
    Every pass continues the state with the same text, so any pass gives
    the same split. Tokens that merge across the state's end stay in the
    tails."""
    head = tokens.ids(letters.prefix_text(state))
    full = tokens.ids(letters.prompt_text(state, "", []))
    n = 0
    for a, b in zip(head, full):
        if a != b:
            break
        n += 1
    return full[:n], n


def _no_rows(rows: int):
    return contextlib.nullcontext()


class LetterReader:
    """Letter reads on one autoregressive model. One reader serves every
    decision on the model; ``bind`` gives a decision its own copy for the
    per-request adapter scales."""

    def __init__(self, model, letter_ids: list[int], *,
                 rows_scope: Optional[Callable[[int], Any]] = None):
        self.model = model
        self.letter_ids = list(letter_ids)
        self.rows_scope = rows_scope or _no_rows
        self.trunk: Any
        self.head: Any
        self.trunk, self.head = _trunk_and_head(model)
        self.rows: Optional[tuple[Any, Any]] = None
        self.path = None          # "rows" or "logits", set by the self-check
        self.forwards = 0

    def bind(self, rows_scope: Optional[Callable[[int], Any]] = None) -> "LetterReader":
        """A copy that shares the checked read path and publishes adapter
        scales through ``rows_scope`` around each forward."""
        out = copy.copy(self)
        out.rows_scope = rows_scope or _no_rows
        out.forwards = 0
        return out

    # -- model calls ---------------------------------------------------------

    def make_cache(self):
        from mlx_lm.models.cache import make_prompt_cache

        m = self.model
        if not hasattr(m, "make_cache") and not hasattr(m, "layers"):
            m = getattr(m, "language_model", m)
        return make_prompt_cache(m)

    def _hidden(self, x, cache):
        self.forwards += 1
        with self.rows_scope(x.shape[0]):
            return self.trunk(x, cache=cache)

    def _full(self, x, cache):
        self.forwards += 1
        with self.rows_scope(x.shape[0]):
            return _logits(self.model(x, cache=cache))

    def _advance(self, x, cache) -> None:
        """Run ``x`` into ``cache`` without reading scores."""
        (self._hidden if self.path == "rows" else self._full)(x, cache)
        mx.eval([c.state for c in cache])

    def _scores(self, x, cache, last: list[int]) -> mx.array:
        """``(rows, 52)`` float32 letter scores at position ``last[r]``."""
        pick = mx.array(last, dtype=mx.int32)
        rows = mx.arange(x.shape[0])
        if self.path == "rows" and self.rows is not None:
            h = self._hidden(x, cache)[rows, pick].astype(mx.float32)
            weights, bias = self.rows
            out = h @ weights.T
            return out + bias if bias is not None else out
        logits = self._full(x, cache)[rows, pick]
        return logits[:, mx.array(self.letter_ids)].astype(mx.float32)

    def check(self, rows_scope: Optional[Callable[[int], Any]] = None) -> str:
        """Choose how letter scores are read. The float32 head rows serve
        when the model's logits are exactly the head of its trunk and the
        rows reproduce the head's letter scores; otherwise the full logits
        do. Runs on a short probe, again after an adapter replaces the head."""
        trunk, head = _trunk_and_head(self.model)
        if getattr(head, "__self__", head) is not getattr(self.head, "__self__", self.head):
            self.trunk, self.head, self.path, self.rows = trunk, head, None, None
        if self.path is not None:
            return self.path
        self.path = "logits"
        if self.trunk is None or self.head is None:
            return self.path
        rows = _letter_rows(self.head, self.letter_ids)
        if rows is None:
            return self.path
        x = mx.array([self.letter_ids[:8]], dtype=mx.int32)
        letter = mx.array(self.letter_ids)
        try:
            with (rows_scope or self.rows_scope)(1):
                full = _logits(self.model(x, cache=self.make_cache()))[0, -1]
                h = self.trunk(x, cache=self.make_cache())
                split = self.head(h)[0, -1]
        except Exception:  # noqa: BLE001 - a trunk that will not run alone
            return self.path
        weights, bias = rows
        exact = h[0, -1].astype(mx.float32) @ weights.T
        if bias is not None:
            exact = exact + bias
        same = mx.array_equal(full, split)
        close = mx.max(mx.abs(exact - split[letter].astype(mx.float32))) <= 0.25
        mx.eval(same, close)
        if bool(same) and bool(close):
            self.rows, self.path = rows, "rows"
        return self.path

    # -- prefix --------------------------------------------------------------

    def prefill(self, ids: list[int], cache=None) -> Step:
        """Run ``ids`` into ``cache`` (a fresh one by default) in fixed
        chunks, yielding after each. Returns the cache."""
        cache = self.make_cache() if cache is None else cache
        for i in range(0, len(ids), FORWARD_TOKENS):
            self._advance(mx.array([ids[i:i + FORWARD_TOKENS]], dtype=mx.int32), cache)
            yield
        return cache

    # -- tails ---------------------------------------------------------------

    def _views(self, prefix, kinds, rows: int, offset: int):
        views = []
        for c, kind in zip(prefix, kinds):
            if kind == "kv":
                keys, values = c.state
                views.append(_KVView(keys, values, offset))
            else:
                views.append(_StateView(c.state, rows))
        return views

    def _fits(self, prefix, kinds, offset: int):
        per_token, state = 0, 0
        for c, kind in zip(prefix, kinds):
            if kind == "kv":
                keys, values = c.state
                per_token = max(per_token, (keys.nbytes + values.nbytes) // max(offset, 1))
            else:
                state = max(state, sum(a.nbytes for a in c.state if a is not None))

        def fits(rows: int, width: int) -> bool:
            if rows * width > FORWARD_TOKENS:
                return False
            return rows * ((offset + width) * per_token + state) <= VIEW_BYTES
        return fits

    def _one(self, prefix, prefix_ids: list[int], tail: list[int]) -> Step:
        """One tail on its own copy of the prefix, in fixed chunks. A prefix
        that cannot be copied is computed again."""
        cache = _fork(prefix)
        if cache is None:
            cache = yield from self.prefill(prefix_ids)
        body, last = tail[:-1], tail[-1:]
        for i in range(0, len(body), FORWARD_TOKENS):
            self._advance(mx.array([body[i:i + FORWARD_TOKENS]], dtype=mx.int32), cache)
            yield
        score = self._scores(mx.array([last], dtype=mx.int32), cache, [0])
        mx.eval(score)
        yield
        return score[0]

    def tails(self, prefix, prefix_ids: list[int], tails: list[list[int]]) -> Step:
        """Letter scores for each tail after the prefix, as float32 host
        lists of 52. Tails share forwards where the cache allows it."""
        out: list[Any] = [None] * len(tails)
        offset = len(prefix_ids)
        kinds = _view_kinds(prefix)
        if kinds is None:
            for i, t in enumerate(tails):
                out[i] = yield from self._one(prefix, prefix_ids, t)
            return [s.tolist() for s in out]
        fits = self._fits(prefix, kinds, offset)
        for group in plan_buckets([len(t) for t in tails], fits):
            width = max(len(tails[i]) for i in group)
            if len(group) == 1 and width > FORWARD_TOKENS:
                out[group[0]] = yield from self._one(prefix, prefix_ids, tails[group[0]])
                continue
            pad = [tails[i] + [0] * (width - len(tails[i])) for i in group]
            x = mx.array(pad, dtype=mx.int32)
            views = self._views(prefix, kinds, len(group), offset)
            scores = self._scores(x, views, [len(tails[i]) - 1 for i in group])
            mx.eval(scores)
            for r, i in enumerate(group):
                out[i] = scores[r]
            yield
        return [s.tolist() for s in out]


def decide_letters(reader: LetterReader, tokens: LetterTokens, schema, state: str,
                   *, prefixes=None,
                   should_stop: Optional[Callable[[], bool]] = None,
                   rows_scope: Optional[Callable[[int], Any]] = None) -> Step:
    """One letter decision, as a generator that yields after each forward.
    ``prefixes`` (a ``prefixes.Prefixes``) finds and keeps the state prefix
    between requests. ``rows_scope(rows)`` publishes the request's adapter
    scales around each forward. Returns the decision body: answers and
    diagnostics."""
    from .prefixes import Prefixes

    prefixes = prefixes if prefixes is not None else Prefixes()
    started = time.time()
    reader.check(rows_scope)
    reader = reader.bind(rows_scope)
    read = letters.decision(schema, state)
    out: dict[str, Any] = {}
    try:
        batch = next(read)
    except StopIteration as stop:
        batch, out = [], stop.value
    ids, split = split_point(tokens, state)
    cache, reused, stored = yield from _prefix(reader, prefixes, ids)
    computed = 0 if reused else split
    passes, prompt_tokens = 0, 0
    while batch:
        if should_stop is not None and should_stop():
            raise letters_cancelled()
        full = [tokens.ids(p.text) for p in batch]
        got: list[Any] = [None] * len(batch)
        shared = [i for i, f in enumerate(full) if f[:split] == ids and len(f) > split]
        if shared:
            scores = yield from reader.tails(cache, ids, [full[i][split:] for i in shared])
            for i, sc in zip(shared, scores):
                got[i] = sc
        for i in (i for i in range(len(batch)) if got[i] is None):
            # A pass whose tokens do not start with the prefix reads alone.
            own = yield from reader.prefill(full[i][:-1])
            got[i] = (yield from reader.tails(own, full[i][:-1], [full[i][-1:]]))[0]
            computed += len(full[i]) - len(full[i][split:])
        passes += len(batch)
        prompt_tokens += sum(len(f) for f in full)
        computed += sum(len(f) - split for f in full)
        try:
            batch = read.send([g[:p.letters] for g, p in zip(got, batch)])
        except StopIteration as stop:
            batch, out = [], stop.value
    return {
        "answers": out["answers"],
        "diagnostics": {
            "engine": "gmlx",
            "readout": "letters",
            "stages": out["stages"],
            "skipped": out["skipped"],
            "orderings": schema["orderings"],
            "prefix": {"tokens": split, "reused": reused, "stored": stored,
                       "tier": prefixes.tier},
            "passes": passes,
            "computed_tokens": computed,
            "prompt_tokens": prompt_tokens,
            "path": reader.path,
            "timing": {"total_ms": (time.time() - started) * 1e3,
                       "reads": reader.forwards},
        },
    }


def _prefix(reader: LetterReader, prefixes, ids: list[int]) -> Step:
    """The prefix cache for ``ids``: a kept one, or a fresh prefill that is
    then kept. Returns ``(cache, reused, stored)``."""
    cache = prefixes.lookup(ids)
    if cache is not None:
        return cache, True, False
    cache = yield from reader.prefill(ids)
    return cache, False, prefixes.store(ids, cache)


def prewarm(reader: LetterReader, tokens: LetterTokens, state: str, *, prefixes=None,
            rows_scope: Optional[Callable[[int], Any]] = None) -> Step:
    """Prefill the prefix of ``state`` and keep it, so the next decision on
    the state reads only its tails. Returns the prefix diagnostics."""
    from .prefixes import Prefixes

    prefixes = prefixes if prefixes is not None else Prefixes()
    ids, split = split_point(tokens, state)
    if prefixes.tier in ("off", "unsupported"):
        return {"tokens": split, "reused": False, "stored": False, "tier": prefixes.tier}
    reader.check(rows_scope)
    reader = reader.bind(rows_scope)
    _cache, reused, stored = yield from _prefix(reader, prefixes, ids)
    return {"tokens": split, "reused": reused, "stored": stored, "tier": prefixes.tier}


def letters_cancelled():
    from .reads import Cancelled

    return Cancelled("decision cancelled")


def run_to_end(step: Step):
    """Drive a step generator to its return value."""
    try:
        while True:
            next(step)
    except StopIteration as stop:
        return stop.value
