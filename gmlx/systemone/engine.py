# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from vLLM examples/features/structured_diffusion (canvas seeding,
# width rule) and vllm/model_executor/models/diffusion_gemma.py (step loop)
# at ab3de6edf2 and modified for gmlx (Apache-2.0; see
# licenses/vllm-LICENSE).
"""Structured reads on a DiffusionGemma model.

A read seeds the decoder canvas with an answer template, runs the decoder
over it against a prefilled prompt, and reports the label log-probabilities
at each answer slot. The prompt is prefilled once. The decoder never writes
the prompt cache, so one prefill serves any number of reads, and a batch of
samples reads the same cache through broadcast views.

With more than one step the read runs vLLM's denoise loop: a scheduled
temperature, a Gumbel sample, entropy-bound acceptance, renoise from the
whole vocabulary, pinned template positions, self-conditioning and per-sample
convergence. A converged sample reports temperature-1 log-probabilities from
the step it converged on."""

from __future__ import annotations

import contextlib
import importlib
import time
from collections import deque
from typing import Any, Callable, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.generate.common import generation_stream
from mlx_vlm.generate.diffusion import stream_diffusion_generate
from mlx_vlm.models.diffusion_gemma.language import _cache_state

from gmlx.gen.generation import encode_prompt

from .reads import (
    Cancelled,
    ReadRequest,
    ReadResult,
    SampleRead,
    SlotRead,
    build_canvas,
    label_id_union,
    pinned_positions,
)

LOGPROB_FLOOR = -9999.0


def _as_dict(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    to_dict = getattr(value, "to_dict", None)
    raw: Any = to_dict() if callable(to_dict) else getattr(value, "__dict__", None)
    return dict(raw or {})


def _host(x: mx.array) -> list:
    """``x.tolist()`` typed as the list it is for any array of rank 1 or
    more."""
    out: Any = x.tolist()
    return out


class DenoiseConfig:
    """The schedule and stop parameters from the model's generation config."""

    def __init__(self, model):
        gen = _as_dict(getattr(model.config, "generation_config", None))
        temp = _as_dict(gen.get("linear_temperature_schedule_config"))
        self.t_min = float(temp.get("t_min", gen.get("t_min", 0.4)))
        self.t_max = float(temp.get("t_max", gen.get("t_max", 0.8)))
        self.max_denoising_steps = int(gen.get("max_denoising_steps") or 48)
        sampler = _as_dict(gen.get("sampler_config"))
        self.entropy_bound = float(sampler.get("entropy_bound", 0.1))
        stop = _as_dict(gen.get("diffusion_stopping_config"))
        self.confidence_threshold = float(
            stop.get("confidence_threshold", gen.get("confidence_threshold", 0.005)))
        stability = int(stop.get("stability_threshold",
                                 gen.get("stability_threshold", 1)))
        # vLLM keeps stability_threshold + 1 argmax canvases: the default of 1
        # means the current canvas matches the previous one.
        self.history = stability + 1

    def temperature(self, step: int) -> float:
        n = float(self.max_denoising_steps)
        remaining = max(n - step, 1.0)
        return self.t_min + (self.t_max - self.t_min) * (remaining / n)


def entropy_transfer_mask(entropy: mx.array, entropy_bound: float) -> mx.array:
    """Accept the lowest-entropy positions while their summed entropy, less
    the largest of them, stays within the bound."""
    order = mx.argsort(entropy, axis=-1)
    sorted_entropy = mx.take_along_axis(entropy, order, axis=-1)
    cumulative = mx.cumsum(sorted_entropy, axis=-1)
    cumulative_max = mx.cummax(sorted_entropy, axis=-1)
    sorted_mask = (cumulative - cumulative_max) <= entropy_bound
    return mx.put_along_axis(mx.zeros_like(sorted_mask), order, sorted_mask, axis=-1)


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


def _seed_key(seed: int) -> mx.array:
    return mx.random.key(int(seed) & 0xFFFFFFFFFFFFFFFF)


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

    def _unembed(self, h: mx.array, label_rows: Optional[mx.array]) -> mx.array:
        """Softcapped fp32 logits: over the label rows when constrained, else
        over the whole vocabulary. The GEMM runs in the activation dtype."""
        if label_rows is not None:
            logits = h @ label_rows.astype(h.dtype).T
        else:
            logits = self.embed.as_linear(h)
        return self.model._softcap(logits)

    def _soft_embedding(self, probs: mx.array, label_rows: Optional[mx.array]):
        """``probs @ embedding`` scaled like an input embedding: the
        self-conditioning signal for the next step."""
        embed = self.embed
        if label_rows is not None:
            out = probs.astype(label_rows.dtype) @ label_rows
        elif isinstance(embed, nn.QuantizedEmbedding):
            out = mx.quantized_matmul(
                probs.astype(mx.bfloat16), embed.weight, embed.scales,
                embed.biases, transpose=False, group_size=embed.group_size,
                bits=embed.bits, mode=getattr(embed, "mode", "affine"))
        else:
            out = probs.astype(embed.weight.dtype) @ embed.weight
        return out * self.decoder.embed_scale

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

    def _slot_reads(self, raw_rows, argmax_ids, req, allowed, constrained):
        """``raw_rows``: host fp32 logits [S, C] at the slots. ``argmax_ids``:
        vocabulary ids [S]. Returns one SlotRead per slot."""
        out = []
        for si, slot in enumerate(req.slots):
            row = raw_rows[si]
            am = int(argmax_ids[si])
            if constrained:
                top = {int(t): max(float(v), LOGPROB_FLOOR)
                       for t, v in zip(allowed, row)}
            else:
                top = {am: max(float(row[am]), LOGPROB_FLOOR)}
                for t in allowed:
                    top[int(t)] = max(float(row[t]), LOGPROB_FLOOR)
            out.append(SlotRead(argmax_id=am, top=top))
        return tuple(out)

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
            logits = self._unembed(rows, label_rows)
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
            host_lp = _host(slot_lp)
            host_am = _host(slot_am)
            full_am = _host(am) if req.trace else None
            out = []
            for i, sd in enumerate(seeds):
                trace = None
                if req.trace:
                    trace = (tuple(canvases[i]), tuple(int(t) for t in full_am[i]))
                out.append(SampleRead(
                    seed=sd, canvas_in=tuple(canvases[i]),
                    slots=self._slot_reads(host_lp[i], host_am[i], req, allowed,
                                           constrained),
                    trace=trace))
            return out
        return self._denoise_pass(prompt, req, seeds, canvases, width, allowed,
                                  allowed_arr, label_rows, should_stop)

    def _denoise_pass(self, prompt, req, seeds, canvases, width, allowed,
                      allowed_arr, label_rows, should_stop):
        cfg = self.config
        constrained = label_rows is not None
        pins = pinned_positions(req.slots, width, req.steps) if req.pinned else []
        pin_mask = mx.array([p in set(pins) for p in range(width)])[None, :]
        pin_keep = (~pin_mask)[..., None]
        slot_idx = mx.array([s.pos for s in req.slots], dtype=mx.int32)
        keys = [mx.random.split(_seed_key(sd), req.steps) for sd in seeds]

        active = list(range(len(seeds)))
        canvas = mx.array(canvases, dtype=mx.int32)
        seed_canvas = canvas
        sc = None
        histories = [deque(maxlen=cfg.history) for _ in seeds]
        traces: list[list[tuple[int, ...]]] = [[] for _ in seeds]
        results: list[Optional[SampleRead]] = [None] * len(seeds)
        step = 0
        while active:
            if should_stop is not None and should_stop():
                raise Cancelled("read cancelled")
            n = len(active)
            if req.trace:
                host_in = _host(canvas)
                for j, i in enumerate(active):
                    traces[i].append(tuple(int(t) for t in host_in[j]))
            h = self.decoder(canvas, cache=prompt.views_for(n),
                             self_conditioning_embeddings=sc,
                             decoder_attention_mask=None)
            logits = self._unembed(h, label_rows)          # [n, W, C] fp32
            scaled = logits / cfg.temperature(step)
            noise = []
            fresh = []
            for i in active:
                k_gumbel, k_noise = mx.random.split(keys[i][step])
                noise.append(mx.random.gumbel(shape=scaled.shape[1:], key=k_gumbel))
                fresh.append(mx.random.randint(0, self.vocab, (width,),
                                               dtype=mx.int32, key=k_noise))
            sampled = mx.argmax(scaled + mx.stack(noise), axis=-1)
            argmax = mx.argmax(scaled, axis=-1)
            if constrained:
                sampled = allowed_arr[sampled]
                argmax = allowed_arr[argmax]
            lp = scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)
            probs = mx.exp(lp)
            entropy = -(probs * lp).sum(axis=-1)
            mean_entropy = entropy.mean(axis=-1)
            accept = entropy_transfer_mask(entropy, cfg.entropy_bound)
            nxt = mx.where(accept, sampled.astype(mx.int32), mx.stack(fresh))
            nxt = mx.where(pin_mask, seed_canvas, nxt)
            raw_lp = logits[:, slot_idx]
            raw_lp = raw_lp - mx.logsumexp(raw_lp, axis=-1, keepdims=True)
            mx.eval(argmax, mean_entropy, nxt, raw_lp)
            step += 1
            host_argmax = _host(argmax)
            host_entropy = _host(mean_entropy)
            done = []
            for j, i in enumerate(active):
                hist = histories[i]
                hist.append(tuple(int(t) for t in host_argmax[j]))
                stable = len(hist) >= cfg.history and len(set(hist)) == 1
                confident = host_entropy[j] < cfg.confidence_threshold
                if (stable and confident) or step >= req.steps:
                    done.append(j)
            if done:
                host_lp = _host(raw_lp)
                for j in done:
                    i = active[j]
                    slot_am = [host_argmax[j][p] for p in (s.pos for s in req.slots)]
                    trace = None
                    if req.trace:
                        traces[i].append(tuple(int(t) for t in host_argmax[j]))
                        trace = tuple(traces[i])
                    results[i] = SampleRead(
                        seed=seeds[i], canvas_in=tuple(canvases[i]),
                        slots=self._slot_reads(host_lp[j], slot_am, req, allowed,
                                               constrained),
                        trace=trace)
            keep = [j for j in range(n) if j not in set(done)]
            if not keep:
                break
            soft = self._soft_embedding(probs, label_rows) * pin_keep
            if len(keep) < n:
                idx = mx.array(keep, dtype=mx.int32)
                nxt, seed_canvas, soft = nxt[idx], seed_canvas[idx], soft[idx]
                active = [active[j] for j in keep]
            canvas = nxt
            sc = soft
        return [r for r in results if r is not None]

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
