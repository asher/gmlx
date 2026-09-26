"""Structured reads on a DiffusionGemma model.

A read seeds the decoder canvas with an answer template, runs the decoder
over it against a prefilled prompt, and reports the label log-probabilities
at each answer slot. The prompt is prefilled once. The decoder never writes
the prompt cache, so one prefill serves any number of reads, and a batch of
samples reads the same cache through broadcast views. Past one step, a read
runs the denoise loop in ``denoise.py``."""

from __future__ import annotations

import contextlib
import importlib
import time
from typing import Any, Callable, Optional

import mlx.core as mx
from mlx_vlm.generate.common import generation_stream
from mlx_vlm.generate.diffusion import stream_diffusion_generate
from mlx_vlm.models.diffusion_gemma.language import _cache_state

from gmlx.gen.generation import encode_prompt

from .denoise import DenoiseConfig, denoise_pass, host_list, slot_reads, unembed
from .reads import (
    Cancelled,
    ReadRequest,
    ReadResult,
    SampleRead,
    build_canvas,
    label_id_union,
)


class _ReadCacheView:
    """One layer's prompt K/V as the decoder attention reads it, broadcast to
    a batch. ``keys`` only marks the view as non-empty."""

    __slots__ = ("keys", "offset", "decoder_state")

    def __init__(self, keys, values, offset: int):
        self.keys = keys
        self.offset = offset
        self.decoder_state = (keys, values)


class PromptCache:
    """A prefilled prompt: its ids and the model's per-layer cache."""

    def __init__(self, model, ids: list[int], cache: list):
        self._model = model
        self.ids = list(ids)
        self.cache = cache
        self._views: dict[int, list] = {}

    def extend(self, ids: list[int]) -> None:
        """Append ``ids`` to the prompt with a causal update of the cache."""
        if not ids:
            return
        self.cache = self._model.diffusion_update_cache(
            mx.array([list(ids)], dtype=mx.int32), cache=self.cache)
        mx.eval([c.state for c in self.cache])
        self.ids.extend(int(i) for i in ids)
        self._views.clear()

    def views_for(self, n: int) -> list:
        views = self._views.get(n)
        if views is not None:
            return views
        text_config = self._model.config.text_config
        window = max(int(text_config.sliding_window) - 1, 0)
        offset = len(self.ids)
        views = []
        for layer_type, c in zip(text_config.layer_types, self.cache):
            state = _cache_state(c)
            if state is None:
                raise RuntimeError("structured read over an empty prompt cache")
            keys, values = state
            if layer_type == "sliding_attention" and window \
                    and keys.shape[2] > window:
                keys = keys[:, :, -window:, :]
                values = values[:, :, -window:, :]
            if n != keys.shape[0]:
                keys = mx.broadcast_to(keys, (n,) + tuple(keys.shape[1:]))
                values = mx.broadcast_to(values, (n,) + tuple(values.shape[1:]))
            views.append(_ReadCacheView(keys, values, offset))
        self._views[n] = views
        return views


class StructuredReader:
    """Runs structured reads on one DiffusionGemma model."""

    def __init__(self, model, *, prefill_step_size: int):
        self.model = model
        # A step of 0 or less prefills in one pass, as mlx-vlm's lane does.
        step = int(prefill_step_size)
        self.prefill_step_size = step if step > 0 else None
        self.decoder = model.model.decoder
        self.embed = self.decoder.embed_tokens
        self.vocab = int(model.config.text_config.vocab_size)
        self.canvas_length = int(model.config.canvas_length)
        self.config = DenoiseConfig(model)

    def prefill(self, ids: list[int]) -> PromptCache:
        step = self.prefill_step_size
        cache = self.model.make_cache()
        cache = self.model.diffusion_prefill_cache(
            mx.array([list(ids)], dtype=mx.int32),
            attention_mask=None,
            cache=cache,
            pixel_values=None,
            mm_token_type_ids=None,
            prefill_step_size=step,
            chunk_prefill=step is not None and len(ids) > step,
        )
        mx.eval([c.state for c in cache])
        return PromptCache(self.model, ids, cache)

    def _label_rows(self, allowed: list[int]) -> mx.array:
        return self.embed(mx.array(allowed, dtype=mx.int32))

    def read(self, prompt: PromptCache, req: ReadRequest, *,
             should_stop: Optional[Callable[[], bool]] = None) -> ReadResult:
        started = time.perf_counter()
        width = int(req.width)
        allowed = label_id_union(req.slots)
        label_rows = self._label_rows(allowed) if req.constrained else None
        allowed_arr = mx.array(allowed, dtype=mx.int32)
        budget = int(req.rows_per_pass or self.canvas_length)
        per_pass = max(1, budget // width)
        seeds = list(req.seeds)
        samples: list[Optional[SampleRead]] = [None] * len(seeds)
        for start in range(0, len(seeds), per_pass):
            if should_stop is not None and should_stop():
                raise Cancelled("read cancelled")
            chunk = seeds[start:start + per_pass]
            reads = self._read_pass(prompt, req, chunk, width, allowed,
                                    allowed_arr, label_rows, should_stop)
            samples[start:start + len(chunk)] = reads
        return ReadResult(
            samples=tuple(s for s in samples if s is not None),
            prompt_len=len(prompt.ids),
            timing_ms={"read": (time.perf_counter() - started) * 1e3},
        )

    def _read_pass(self, prompt, req, seeds, width, allowed, allowed_arr,
                   label_rows, should_stop):
        n = len(seeds)
        slot_pos = [s.pos for s in req.slots]
        canvases = [build_canvas(req.template, req.slots, width, sd, self.vocab)
                    for sd in seeds]
        canvas = mx.array(canvases, dtype=mx.int32)
        constrained = label_rows is not None

        if req.steps <= 1:
            h = self.decoder(canvas, cache=prompt.views_for(n),
                             decoder_attention_mask=None)
            rows = h if req.trace else h[:, mx.array(slot_pos, dtype=mx.int32)]
            logits = unembed(self, rows, label_rows)
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            am = mx.argmax(logits, axis=-1)
            if constrained:
                am = allowed_arr[am]
            if req.trace:
                slot_lp = lp[:, mx.array(slot_pos, dtype=mx.int32)]
                slot_am = am[:, mx.array(slot_pos, dtype=mx.int32)]
            else:
                slot_lp, slot_am = lp, am
            mx.eval(slot_lp, slot_am, am)
            host_lp = host_list(slot_lp)
            host_am = host_list(slot_am)
            full_am = host_list(am) if req.trace else None
            out = []
            for i, sd in enumerate(seeds):
                trace = None
                if req.trace:
                    trace = (tuple(canvases[i]), tuple(int(t) for t in full_am[i]))
                out.append(SampleRead(
                    seed=sd, canvas_in=tuple(canvases[i]),
                    slots=slot_reads(host_lp[i], host_am[i], req.slots, allowed,
                                     constrained),
                    trace=trace))
            return out
        return denoise_pass(self, prompt, req, seeds, canvases, width, allowed,
                            allowed_arr, label_rows, should_stop)

    def think(self, prompt_ids: list[int], budget: int, *, stop_id: int,
              canvas_width: int, processor, backend,
              should_stop: Optional[Callable[[], bool]] = None
              ) -> tuple[list[int], dict]:
        """A thought: up to ``budget`` tokens from the mlx-vlm denoiser at
        temperature 1 and the served canvas width, cut at ``stop_id``.
        ``prompt_ids`` already ends with the open tag. Returns the thought
        ids and ``{"tokens", "closed", "ms"}``."""
        started = time.perf_counter()
        criteria = getattr(backend, "stopping_criteria", None)
        saved = list(criteria.eos_token_ids) if criteria is not None else None
        if criteria is not None:
            criteria.eos_token_ids.append(int(stop_id))
        ids: list[int] = []
        closed = False
        gen = stream_diffusion_generate(
            self.model, processor, backend,
            mx.array([list(prompt_ids)], dtype=mx.int32), None, None,
            max_tokens=int(budget),
            skip_special_token_ids=set(),
            temperature=1.0,
            diffusion_sampler="entropy-bound",
            diffusion_max_canvas_length=int(canvas_width),
            prefill_step_size=self.prefill_step_size,
        )
        try:
            for r in gen:
                if should_stop is not None and should_stop():
                    raise Cancelled("thought cancelled")
                if getattr(r, "is_draft", False) \
                        or getattr(r, "diffusion_block_complete", False):
                    continue
                if r.finish_reason is not None:
                    if r.finish_reason == "stop" and r.token is not None \
                            and len(ids) < r.generation_tokens:
                        if int(r.token) == int(stop_id):
                            closed = True
                        else:
                            # vLLM keeps an end-of-turn stop in the output.
                            ids.append(int(r.token))
                    break
                if r.token is not None and r.generation_tokens > len(ids):
                    ids.append(int(r.token))
        finally:
            gen.close()
            if criteria is not None and saved is not None:
                criteria.eos_token_ids = saved
        return ids, {"tokens": len(ids), "closed": closed,
                     "ms": (time.perf_counter() - started) * 1e3}


class ChatTokens:
    """Tokenizer access for one decision, in the shapes the decision logic
    takes: ``enc``, ``decode`` and ``chat_ids``. The prompt is the system
    text plus ``state_text`` as the user message, rendered by the model's
    chat template. When ``lock`` is given every call takes it, since a
    server runs some calls on the request thread and the rest on the engine
    thread."""

    def __init__(self, processor, state_text: str = "", *, lock=None):
        self.processor = processor
        self.state_text = state_text
        self.wrapper = getattr(processor, "_wrapper", processor)
        self.lock = lock if lock is not None else contextlib.nullcontext()

    def enc(self, text: str) -> list[int]:
        with self.lock:
            return list(self.wrapper.encode(text, add_special_tokens=False))

    def decode(self, ids) -> str:
        with self.lock:
            return self.wrapper.decode(list(ids))

    def render(self, sys_text: str, thinking: bool) -> str:
        """The prompt text, with each message's content as a string. The
        Gemma 4 template ends each text part of a list-form system message
        with a space, so content parts would change the prompt ids."""
        msgs = [{"role": "system", "content": sys_text},
                {"role": "user", "content": self.state_text}]
        with self.lock:
            out: Any = self.wrapper.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=bool(thinking))
        return str(out)

    def encode_prompt(self, text: str) -> list[int]:
        with self.lock:
            return list(encode_prompt(self.wrapper, text))

    def chat_ids(self, sys_text: str, thinking: bool) -> list[int]:
        return self.encode_prompt(self.render(sys_text, thinking))


class BoundReader:
    """A ``StructuredReader`` bound to a tokenizer pair and a stop check, in
    the ``ReadEngine`` shape the decision logic calls."""

    def __init__(self, reader: StructuredReader, *, processor, backend,
                 should_stop: Optional[Callable[[], bool]] = None):
        self.reader = reader
        self.processor = processor
        self.backend = backend
        self.should_stop = should_stop

    def prefill(self, ids):
        return self.reader.prefill(ids)

    def read(self, prompt, req):
        return self.reader.read(prompt, req, should_stop=self.should_stop)

    def think(self, prompt_ids, budget, *, stop_id, canvas_width):
        return self.reader.think(
            prompt_ids, budget, stop_id=stop_id, canvas_width=canvas_width,
            processor=self.processor, backend=self.backend,
            should_stop=self.should_stop)


@contextlib.contextmanager
def engine_scope(model, seed: int):
    """The setup every job needs: the RNG seeded for the thought, the
    generation stream, and the wired limit entered once. Clears the MLX
    cache on exit."""
    mx.random.seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
    wired: Any = contextlib.nullcontext()
    if mx.default_device() == mx.gpu and mx.metal.is_available():
        wired_limit = importlib.import_module("mlx_lm.generate").wired_limit
        wired = wired_limit(model, [generation_stream])
    try:
        with wired, mx.stream(generation_stream):
            yield
    finally:
        mx.clear_cache()
