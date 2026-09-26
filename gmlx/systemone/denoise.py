# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from vllm/model_executor/models/diffusion_gemma.py (the unembedding,
# the slot log-probabilities and the multi-step structured-read loop) at
# ab3de6edf2 and modified for gmlx (Apache-2.0; see licenses/vllm-LICENSE).
# entropy_transfer_mask follows mlx-vlm's _diffusion_entropy_transfer_mask
# (MIT; see licenses/mlx-vlm-LICENSE).
"""The read rules taken from vLLM: the unembedding, the log-probabilities a
slot reports, and the denoise loop a read runs past one step.

The loop has a scheduled temperature, a Gumbel sample, entropy-bound
acceptance, renoise from the whole vocabulary, pinned template positions,
self-conditioning and per-sample convergence. A converged sample reports
temperature-1 log-probabilities from the step it converged on. The loop
takes the ``StructuredReader`` in ``engine.py`` for the model and the
prompt."""

from __future__ import annotations

from collections import deque
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .reads import Cancelled, SampleRead, SlotRead, pinned_positions

LOGPROB_FLOOR = -9999.0


def _as_dict(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    to_dict = getattr(value, "to_dict", None)
    raw: Any = to_dict() if callable(to_dict) else getattr(value, "__dict__", None)
    return dict(raw or {})


def host_list(x: mx.array) -> list:
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


def _seed_key(seed: int) -> mx.array:
    return mx.random.key(int(seed) & 0xFFFFFFFFFFFFFFFF)


def unembed(reader, h: mx.array, label_rows: Optional[mx.array]) -> mx.array:
    """Softcapped fp32 logits: over the label rows when constrained, else
    over the whole vocabulary. The GEMM runs in the activation dtype."""
    if label_rows is not None:
        logits = h @ label_rows.astype(h.dtype).T
    else:
        logits = reader.embed.as_linear(h)
    return reader.model._softcap(logits)


def soft_embedding(reader, probs: mx.array, label_rows: Optional[mx.array]):
    """``probs @ embedding`` scaled like an input embedding: the
    self-conditioning signal for the next step."""
    embed = reader.embed
    if label_rows is not None:
        out = probs.astype(label_rows.dtype) @ label_rows
    elif isinstance(embed, nn.QuantizedEmbedding):
        out = mx.quantized_matmul(
            probs.astype(mx.bfloat16), embed.weight, embed.scales,
            embed.biases, transpose=False, group_size=embed.group_size,
            bits=embed.bits, mode=getattr(embed, "mode", "affine"))
    else:
        out = probs.astype(embed.weight.dtype) @ embed.weight
    return out * reader.decoder.embed_scale


def slot_reads(raw_rows, argmax_ids, slots, allowed, constrained):
    """``raw_rows``: host fp32 logits [S, C] at the slots. ``argmax_ids``:
    vocabulary ids [S]. Returns one SlotRead per slot."""
    out = []
    for si in range(len(slots)):
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


def denoise_pass(reader, prompt, req, seeds, canvases, width, allowed,
                 allowed_arr, label_rows, should_stop):
    """The reads of one batch of samples past one step. ``reader`` is the
    ``StructuredReader`` and ``prompt`` its prefilled ``PromptCache``."""
    cfg = reader.config
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
            host_in = host_list(canvas)
            for j, i in enumerate(active):
                traces[i].append(tuple(int(t) for t in host_in[j]))
        h = reader.decoder(canvas, cache=prompt.views_for(n),
                         self_conditioning_embeddings=sc,
                         decoder_attention_mask=None)
        logits = unembed(reader, h, label_rows)        # [n, W, C] fp32
        scaled = logits / cfg.temperature(step)
        noise = []
        fresh = []
        for i in active:
            k_gumbel, k_noise = mx.random.split(keys[i][step])
            noise.append(mx.random.gumbel(shape=scaled.shape[1:], key=k_gumbel))
            fresh.append(mx.random.randint(0, reader.vocab, (width,),
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
        host_argmax = host_list(argmax)
        host_entropy = host_list(mean_entropy)
        done = []
        for j, i in enumerate(active):
            hist = histories[i]
            hist.append(tuple(int(t) for t in host_argmax[j]))
            stable = len(hist) >= cfg.history and len(set(hist)) == 1
            confident = host_entropy[j] < cfg.confidence_threshold
            if (stable and confident) or step >= req.steps:
                done.append(j)
        if done:
            host_lp = host_list(raw_lp)
            for j in done:
                i = active[j]
                slot_am = [host_argmax[j][p] for p in (s.pos for s in req.slots)]
                trace = None
                if req.trace:
                    traces[i].append(tuple(int(t) for t in host_argmax[j]))
                    trace = tuple(traces[i])
                results[i] = SampleRead(
                    seed=seeds[i], canvas_in=tuple(canvases[i]),
                    slots=slot_reads(host_lp[j], slot_am, req.slots, allowed,
                                     constrained),
                    trace=trace)
        keep = [j for j in range(n) if j not in set(done)]
        if not keep:
            break
        soft = soft_embedding(reader, probs, label_rows) * pin_keep
        if len(keep) < n:
            idx = mx.array(keep, dtype=mx.int32)
            nxt, seed_canvas, soft = nxt[idx], seed_canvas[idx], soft[idx]
            active = [active[j] for j in keep]
        canvas = nxt
        sc = soft
    return [r for r in results if r is not None]

