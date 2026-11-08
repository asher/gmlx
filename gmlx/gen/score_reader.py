"""Prompt-token logprobs for scoring, read as an engine step job.

A scoring request gives the logprob of every prompt token after the first,
given the tokens before it. The reader runs the prompt on a cache of its
own, one forward per step, so it never meets the prompt cache, admission
or the batch prefill. Each step calls the language model, the object chat
calls, with token ids; the engine job seeds a text row's rope state, so
each chunk is numbered from its cache offset.

On a resident model the logits are the model's own forward. Only the
float32 reductions run on sub-chunks of the forward's columns. A streamed
mixture-of-experts model runs a wide trunk and the head on sub-chunks,
when that split reproduces the model's own logits bit for bit on a probe;
otherwise it takes the full path too.

Every width depends on the model, the box and the depth, never on load,
so a score repeats bit for bit on one box."""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Generator, Optional, cast

import mlx.core as mx

from gmlx.gen.logprobs import raw_logprobs, top_logprobs
from gmlx.gen.prefill_plan import BYTES_PER_V_ELEMENT, GB, head_step

Step = Generator[None, None, Any]
# (logprob of the prompt token, top ids, their logprobs)
Entry = tuple[float, Optional[list[int]], Optional[list[float]]]

# Forward width on the full path before the caps, prefill_plan's dense trunk.
FORWARD_TOKENS = 512
# Logits of one forward at most min(LOGITS_CAP, 5 percent of the working set).
LOGITS_CAP = 0.5 * GB
# Reductions of one sub-chunk at most min(REDUCE_CAP, 5 percent).
REDUCE_CAP = 1.0 * GB
WS_SHARE = 0.05
# Ids of the probes run when the reader is built.
PROBE_IDS = (1, 2, 3, 4, 5, 6, 7, 8)
_READER_ATTR = "_kq_score_reader"

NO_COLUMNS = "this model does not return logits for every prompt token"
EMBEDS_DIFFER = ("model {model!r} reads token ids and embeddings differently, "
                 "so a score would not match what chat reads; scoring is not "
                 "supported on this model")


class ScoringRefused(Exception):
    """The model cannot be scored. The message is for the client."""


def _logits(out):
    return getattr(out, "logits", out)


def working_set() -> Optional[float]:
    from gmlx.device import metal_value

    try:
        ws = float(metal_value("max_recommended_working_set_size"))
    except Exception:  # noqa: BLE001 - no Metal: no working set
        return None
    return ws if ws > 0 else None


def _share(cap: float, ws: Optional[float]) -> float:
    return cap if ws is None else min(cap, WS_SHARE * ws)


def reduce_width(vocab: int, ws: Optional[float]) -> int:
    """Columns of one float32 reduction sub-chunk."""
    return head_step(vocab, _share(REDUCE_CAP, ws) / GB, BYTES_PER_V_ELEMENT)


def logits_width(vocab: int, itemsize: int, ws: Optional[float],
                 base: Optional[int] = None) -> int:
    """The largest halving tier of ``base`` (``FORWARD_TOKENS``) whose
    logits fit the cap."""
    cap = _share(LOGITS_CAP, ws)
    w = int(FORWARD_TOKENS if base is None else base)
    while w > 1 and w * vocab * itemsize > cap:
        w //= 2
    return w


def plan(width: int, tokens: int, heads: int, ws: Optional[float]) -> list[tuple[int, int]]:
    """``(depth, width)`` of each forward over ``tokens`` tokens: ``width``
    capped at each depth by the attention score transient, as chat
    prefill caps its chunks (``prefill_decay.decayed_step``) at the box's
    fixed cap, never the live headroom."""
    from gmlx.gen import prefill_decay

    cap = prefill_decay.cap_bytes_for(ws if ws is not None else 0.0)
    out, depth = [], 0
    while depth < tokens:
        w = min(int(prefill_decay.decayed_step(width, depth, heads, cap=cap)),
                tokens - depth)
        out.append((depth, w))
        depth += w
    return out


def attention_bytes(steps: list[tuple[int, int]], heads: int) -> float:
    """The largest score transient of a plan, ``decayed_step``'s formula."""
    return max((heads * w * (d + w) * 2.0 for d, w in steps), default=0.0)


def transient_bytes(tokens: int, *, vocab: int, itemsize: int, width: int,
                    reduce: int, split: bool, heads: int,
                    ws: Optional[float]) -> float:
    """The transient of scoring ``tokens`` positions at these widths: the
    logits of one forward (of one head sub-chunk on the split path), the
    float32 reductions of one sub-chunk, and the largest attention score
    transient over the plan."""
    tokens = max(1, int(tokens))
    logits = min(reduce if split else width, tokens) * vocab * itemsize
    reduce_bytes = min(reduce, width, tokens) * vocab * BYTES_PER_V_ELEMENT
    attn = attention_bytes(plan(width, tokens, heads, ws), heads)
    return float(logits + reduce_bytes + attn)


def price_bytes(model, tokens: int, reader: Optional["ScoreReader"] = None,
                ws: Optional[float] = None) -> Optional[float]:
    """The transient of a scoring job of ``tokens`` positions on ``model``.
    With a built reader, at its widths. Before that the logits dtype and the
    path are not known, so the price is the largest over both itemsizes
    and, on a streamed model, both paths. None when the vocabulary size
    cannot be read."""
    if reader is not None and reader.built and reader.vocab:
        return transient_bytes(
            tokens, vocab=reader.vocab, itemsize=reader.itemsize,
            width=reader.width, reduce=reader.reduce,
            split=reader.split is not None, heads=reader.heads, ws=reader.ws)
    from gmlx.stream.expert_streaming import experts_streamed

    vocab = config_value(model, "vocab_size")
    if not isinstance(vocab, int) or vocab <= 0:
        return None
    if ws is None:
        ws = working_set()
    heads = attention_heads(model)
    reduce = reduce_width(vocab, ws)
    shapes = [(size, logits_width(vocab, size, ws), False) for size in (2, 4)]
    if experts_streamed(model):
        shapes += [(size, adopted_step(model), True) for size in (2, 4)]
    return max(
        transient_bytes(tokens, vocab=vocab, itemsize=size, width=width,
                        reduce=reduce, split=split, heads=heads, ws=ws)
        for size, width, split in shapes)


def config_value(model, name: str, default=None):
    from gmlx.serve.mem_preflight import _get, _lm_config

    return _get(_lm_config(model), name, default)


def attention_heads(model) -> int:
    return int(config_value(model, "num_attention_heads", 32) or 32)


def adopted_step(model) -> int:
    """The prefill step a streamed model adopted at load, read as the MTP
    prefill body reads it."""
    from gmlx._vendor.mlx_vlm.generate.ar import DEFAULT_PREFILL_STEP_SIZE
    from gmlx.stream.installs import wrapper_chain

    if "PREFILL_STEP_SIZE" in os.environ:
        return int(os.environ["PREFILL_STEP_SIZE"])
    for m in wrapper_chain(model):
        step = getattr(m, "_gmlx_adopted_prefill_step", None)
        if step:
            return int(step)
    return int(DEFAULT_PREFILL_STEP_SIZE)


def _softcap(text) -> Optional[Callable[[Any], Any]]:
    """The model's own final logit softcap, in its dtype, or None."""
    cap = getattr(text, "final_logit_softcapping", None)
    if cap is None:
        cap = getattr(getattr(text, "args", None), "final_logit_softcapping", None)
    if not cap:
        return None
    fn = getattr(sys.modules.get(type(text).__module__), "logit_softcap", None)
    if callable(fn):
        return lambda z, c=cap, f=fn: f(c, z)
    return lambda z, c=float(cap): mx.tanh(z / c) * c


def _no_rows(rows: int):
    import contextlib

    return contextlib.nullcontext()


def _seed(model) -> None:
    from gmlx.serve.engine_jobs import seed_text_rope

    seed_text_rope(model)


class ScoreReader:
    """Prompt-token logprobs on one model. Built by the first scoring job,
    on the engine thread, and kept for the model."""

    def __init__(self, model, *, ws: Optional[float] = None,
                 streamed: Optional[bool] = None):
        from gmlx.stream.expert_streaming import experts_streamed

        self.model = model
        self.lm = getattr(model, "language_model", model)
        self.ws = working_set() if ws is None else ws
        self.streamed = experts_streamed(model) if streamed is None else bool(streamed)
        self.heads = attention_heads(model)
        self.vocab: Optional[int] = None
        self.dtype = None
        self.refusal: Optional[str] = None
        self.split: Optional[tuple] = None   # (trunk, head, step, softcap)
        self._head_seen: Any = None
        self.built = False
        self.forwards = 0

    @property
    def itemsize(self) -> int:
        return int(self.dtype.size) if self.dtype is not None else 4

    @property
    def reduce(self) -> int:
        return reduce_width(int(self.vocab or 0), self.ws)

    @property
    def width(self) -> int:
        """The forward width before the depth cap."""
        if self.split is not None:
            return adopted_step(self.model)
        return logits_width(int(self.vocab or 0), self.itemsize, self.ws)

    def plan(self, tokens: int) -> list[tuple[int, int]]:
        return plan(self.width, tokens, self.heads, self.ws)

    def make_cache(self):
        from gmlx.cache.kv_cache import make_prompt_cache

        return make_prompt_cache(self.lm)

    def build(self, rows_scope=None) -> Step:
        """The one-token forward that records the logits dtype, the probe of
        ids against embeddings and, on a streamed model, the split check."""
        scope = rows_scope or _no_rows
        _seed(self.model)
        with scope(1):
            out = _logits(self.lm(mx.array([PROBE_IDS[:1]], dtype=mx.int32),
                                  cache=self.make_cache()))
        mx.eval(out)
        self.dtype, self.vocab = out.dtype, int(out.shape[-1])
        yield
        if not self._ids_match_embeddings(scope):
            self.refusal = EMBEDS_DIFFER
        yield
        if self.streamed:
            self._check_split(scope)
            yield
        self.built = True

    def _ids_match_embeddings(self, scope) -> bool:
        """Whether the language model gives the same logits for ids as for
        the outer embeddings of those ids and the inputs chat passes with
        them, bit for bit. True for a model with no outer embedding."""
        embed = getattr(self.model, "get_input_embeddings", None)
        if not callable(embed) or self.lm is self.model:
            return True
        x = mx.array([PROBE_IDS], dtype=mx.int32)
        _seed(self.model)
        with scope(1):
            ids = _logits(self.lm(x, cache=self.make_cache()))
            mx.eval(ids)
        _seed(self.model)
        with scope(1):
            got = embed(x, None)
            to_dict = getattr(got, "to_dict", None)
            fields: dict[str, Any] = (cast(dict, to_dict()) if callable(to_dict)
                                      else {"inputs_embeds": got})
            extra = {str(k): v for k, v in fields.items() if v is not None}
            embeds = extra.pop("inputs_embeds")
            chat = _logits(self.lm(x, cache=self.make_cache(),
                                   inputs_embeds=embeds, **extra))
            mx.eval(chat)
        _seed(self.model)
        return ids.shape == chat.shape and bool(mx.array_equal(ids, chat).item())

    def _check_split(self, scope) -> None:
        """Take the trunk and head split when, on a probe of one reduction
        width, its logits equal the model's own at every column for one
        choice of step after the head."""
        from gmlx.eval_guard import guard
        from gmlx.systemone.ar_reader import _head_steps, _trunk_and_head, text_model

        trunk, head = _trunk_and_head(self.model)
        self._head_seen = getattr(head, "__self__", head)
        self.split = None
        if trunk is None or head is None:
            return
        inner = text_model(self.model)
        cap = _softcap(getattr(inner, "language_model", inner))
        n = max(1, min(self.reduce, 4096))
        x = mx.array([[PROBE_IDS[i % len(PROBE_IDS)] for i in range(n)]],
                     dtype=mx.int32)
        try:
            _seed(self.model)
            with scope(1):
                full = _logits(self.lm(x, cache=self.make_cache()))
                z = head(trunk(x, cache=self.make_cache()))
            guard.eval(full, z, site="score-split-check", owner="scratch")
        except Exception:  # noqa: BLE001 - a trunk that will not run alone
            return
        finally:
            _seed(self.model)
        for step in _head_steps(self.model):
            got = z if step is None else step(z)
            if cap is not None:
                got = cap(got)
            if got.shape == full.shape and mx.array_equal(full, got).item():
                self.split = (trunk, head, step, cap)
                return

    def _recheck(self, scope) -> None:
        """Check the split again when an adapter replaced the head."""
        if not self.streamed:
            return
        from gmlx.systemone.ar_reader import _trunk_and_head

        _, head = _trunk_and_head(self.model)
        if getattr(head, "__self__", head) is not self._head_seen:
            self._check_split(scope)

    def score(self, ids: list[int], k: int, rows_scope=None,
              should_stop: Optional[Callable[[], bool]] = None) -> Step:
        """The entry of every prompt position after the first: the
        logprob of ``ids[i]`` given ``ids[:i]``, and the ``k`` most likely
        ids with their logprobs when ``k`` > 0. Yields after each forward."""
        from gmlx.systemone.reads import Cancelled

        scope = rows_scope or _no_rows
        if self.refusal is not None:
            raise ScoringRefused(self.refusal)
        self._recheck(scope)
        cache = self.make_cache()
        r = max(1, self.reduce)
        out: list[Entry] = []
        trunk = head = step = cap = hidden = logits = None
        for depth, w in self.plan(len(ids) - 1):
            if should_stop is not None and should_stop():
                raise Cancelled("the scoring job was cancelled")
            x = mx.array([ids[depth:depth + w]], dtype=mx.int32)
            targets = ids[depth + 1:depth + w + 1]
            self.forwards += 1
            if self.split is not None:
                trunk, head, step, cap = self.split
                with scope(1):
                    hidden = trunk(x, cache=cache)
                columns = hidden.shape[1]
            else:
                hidden = None
                with scope(1):
                    logits = _logits(self.lm(x, cache=cache))
                columns = logits.shape[1] if logits.ndim == 3 else 0
            if columns != w:
                self.refusal = NO_COLUMNS
                raise ScoringRefused(NO_COLUMNS)
            for a in range(0, w, r):
                b = min(a + r, w)
                if hidden is not None and head is not None:
                    with scope(1):
                        z = head(hidden[:, a:b])
                    z = z if step is None else step(z)
                    z = z if cap is None else cap(z)
                    z = z[0]
                else:
                    assert logits is not None
                    z = logits[0, a:b]
                lp = raw_logprobs(z)
                t = mx.array(targets[a:b], dtype=mx.int32)[:, None]
                chosen = mx.take_along_axis(lp, t, axis=-1)[:, 0]
                if k > 0:
                    tid, tval = top_logprobs(lp, k)
                    mx.eval(chosen, tid, tval)
                    out.extend(zip(cast(list, chosen.tolist()), cast(list, tid.tolist()),
                                   cast(list, tval.tolist())))
                else:
                    mx.eval(chosen)
                    out.extend((c, None, None) for c in cast(list, chosen.tolist()))
            mx.eval([c.state for c in cache])
            yield
        return out


def reader_for(rg) -> Optional[ScoreReader]:
    """The reader kept on ``rg``'s engine, or None before the first job."""
    return getattr(rg, "__dict__", {}).get(_READER_ATTR)


def reader_of(rg) -> ScoreReader:
    """The reader of ``rg``'s engine, made on first use."""
    reader = reader_for(rg)
    if reader is None:
        reader = ScoreReader(rg.model)
        rg.__dict__[_READER_ATTR] = reader
    return reader
