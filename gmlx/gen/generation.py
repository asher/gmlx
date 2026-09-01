"""Generation API over loaded kquant models.

``generate`` drives mlx-lm's engine with stop sequences, thinking budgets, KV
quantization and the over-generation probe; ``generate_speculative`` /
``stream_generate_speculative`` drive the MTP speculative round (mlx-vlm's
engine or the owned walk). ``StopScanner`` is the incremental stop-sequence
matcher shared with the chat REPL and serve streaming paths.
"""

from __future__ import annotations

import os
import sys
import time

import mlx.core as mx

import gmlx.load.loader as loader


# Tokens per target prefill forward on the speculative path only. mlx-vlm forces
# an unchunked prefill once a drafter is attached (prefill_step_size=None), so we
# chunk it ourselves with this manual loop (which evals cache state per chunk so
# the drafter can read the shared K/V back). For that loop a 16k-token MoE sweep
# peaks at 4096 (8k ~ tied, an unchunked single-shot ~10% slower + far more
# memory). The plain mlx-lm prefill engine is a different implementation and is
# fastest at its own 2048 default (4096 measured ~11% slower there) - do not
# reuse this value for it. Note 4096 tokens * top_k experts is >32767 gather rows
# at top_k=8, so this width relies on the kquant gather being correct past 32767.
_PREFILL_CHUNK = 4096


class StopScanner:
    """Incremental stop-sequence matcher for streamed text.

    ``feed(segment)`` returns ``(printable, hit)``: the text safe to emit now
    (everything that can no longer be the start of a stop sequence) and
    whether a stop sequence completed - in which case ``printable`` ends just
    before it. A tail of ``max(len(stop)) - 1`` chars is held back between
    feeds so a stop string split across segments still matches; call
    ``flush()`` at end-of-stream to release it."""

    def __init__(self, stops: list):
        self.stops = [s for s in stops if s]
        self.hold = max((len(s) for s in self.stops), default=1) - 1
        self.buf = ""

    def feed(self, segment: str) -> tuple:
        self.buf += segment
        hit_at = min(
            (i for i in (self.buf.find(s) for s in self.stops) if i != -1), default=-1
        )
        if hit_at != -1:
            out, self.buf = self.buf[:hit_at], ""
            return out, True
        if self.hold == 0:
            out, self.buf = self.buf, ""
            return out, False
        if len(self.buf) > self.hold:
            out, self.buf = self.buf[: -self.hold], self.buf[-self.hold :]
            return out, False
        return "", False

    def flush(self) -> str:
        out, self.buf = self.buf, ""
        return out


# Prompts shorter than this prefill in well under a second: no spinner flash.
_PREFILL_SPINNER_MIN_TOKENS = 512


def _prefill_progress_ui(stream=None):
    """``(callback, close)`` pair for a prefill spinner on ``stream`` (default
    stderr). The callback matches mlx-lm's ``prompt_progress_callback``
    ``(processed, total)`` shape: it starts the spinner on the first call of a
    long prefill, updates the label per chunk, and closes it on the final
    ``(total, total)`` call, which fires before the first decoded token.
    ``close`` is idempotent (also called from a ``finally``)."""
    from gmlx.spinner import Spinner

    state: dict = {}

    def close() -> None:
        sp = state.pop("spinner", None)
        if sp is not None:
            sp.__exit__(None, None, None)

    def cb(processed: int, total: int) -> None:
        if total < _PREFILL_SPINNER_MIN_TOKENS:
            return
        if processed >= total:
            close()
            return
        sp = state.get("spinner")
        if sp is None:
            sp = Spinner(f"prefill {processed}/{total} tok", stream=stream)
            sp.__enter__()
            state["spinner"] = sp
        else:
            sp.update(f"prefill {processed}/{total} tok")

    return cb, close


def _echo_think_tag(prompt, tokenizer):
    """The think-open tag to echo before a raw verbose stream, or None. A
    pre-fill chat template ends the rendered prompt inside an open thinking
    block, so the generated stream carries only the close tag; chat's renderer
    seeds this state, the raw printers here echo the tag instead."""
    from .thinking_budget import prompt_open_think_tag

    return prompt_open_think_tag(prompt, tokenizer=tokenizer)


def _verbose_emitter(prompt, tokenizer, reasoning):
    """``(write, close)`` for a verbose stream. ``reasoning`` show/hide styles
    a thinking model's chain-of-thought through :class:`~gmlx.tui.reasoning.
    StreamRenderer` (the chat REPL's rendering); anything else prints raw,
    echoing the prompt-opened think tag so the bare close marker still reads."""
    open_tag = _echo_think_tag(prompt, tokenizer)
    if reasoning in ("show", "hide"):
        from gmlx.tui.reasoning import StreamRenderer, prompt_opens_header

        r = StreamRenderer(reasoning, start_in_thinking=open_tag is not None,
                           start_in_header=prompt_opens_header(prompt))
        return r.write, r.close
    if open_tag is not None:
        print(open_tag, flush=True)

    def write(s):
        print(s, end="", flush=True)

    return write, (lambda: None)


def resolve_kvarn_policy(model, kv_bits, kv_tail_tokens, rotating_window,
                         stack, *, mode="single", mtp=False):
    """The kvarn policy for one built stack, with the model-shape decline
    resolved. The single entry point CLI, chat and the MTP paths share."""
    from gmlx.cache.kv_policy import resolve_kv_quant_policy
    from gmlx.cache.kvarn_cache import kvarn_unsupported, kvarn_widths

    k_bits, v_bits = kvarn_widths(kv_bits)
    return resolve_kv_quant_policy(
        stack,
        kv_bits=k_bits,
        value_bits=v_bits,
        scheme="kvarn",
        tail_tokens=int(kv_tail_tokens or 0),
        rotating_window=rotating_window,
        mode=mode,
        mtp=mtp,
        scheme_reason=kvarn_unsupported(model),
    )


def apply_kvarn_policy(prompt_cache, policy) -> int:
    """Convert in place the layers the policy takes. Returns the number of
    layers converted."""
    from gmlx.cache.kv_policy import arm_stack, quantize_kv_members

    arm_stack(prompt_cache, policy, hold=False)
    n = 0
    for i, plan in enumerate(policy.per_layer):
        if plan.quantize:
            prompt_cache[i], k = quantize_kv_members(prompt_cache[i], policy)
            n += k
    if n:
        from gmlx.cache.kvarn_sdpa import install_kvarn_sdpa

        install_kvarn_sdpa()
    return n


def convert_kvarn_cache(model, prompt_cache, kv_bits, kv_tail_tokens,
                        rotating_window=None):
    """Resolve the kvarn policy for one freshly built stack and apply it.
    Returns the policy; its verdict tells the caller what happened. The
    chat REPL rebuilds a cache per turn and re-resolves here, so every
    turn gets the same layer selection as the load-time banner."""
    policy = resolve_kvarn_policy(
        model, kv_bits, kv_tail_tokens, rotating_window, prompt_cache
    )
    if policy.verdict in ("full", "partial"):
        apply_kvarn_policy(prompt_cache, policy)
    return policy


def setup_kvarn_cache(
    model, kv_bits, kv_tail_tokens, max_kv_size, out=None, make_cache=None
):
    """Build the model's prompt cache with kvarn KV on every layer the
    policy takes, printing the [kv] banner. Returns the cache list, or
    None (with a warning) when the scheme cannot apply. ``make_cache``
    overrides the stock cache builder (the MTP paths build via
    mlx-vlm's)."""
    from gmlx.cache.kv_policy import kv_line
    from gmlx.cache.kvarn_cache import kvarn_rotating_window

    out = out if out is not None else sys.stderr
    window = (
        None if make_cache is not None else kvarn_rotating_window(model, max_kv_size)
    )
    if make_cache is not None:
        prompt_cache = make_cache()
    else:
        from mlx_lm.models.cache import make_prompt_cache as _mpc

        prompt_cache = _mpc(model, max_kv_size=max_kv_size)
    policy = convert_kvarn_cache(
        model, prompt_cache, kv_bits, kv_tail_tokens, window
    )
    if policy.verdict not in ("full", "partial"):
        print(f"warning: --kv-quant-scheme kvarn dropped: {policy.reason}",
              file=out)
        return None
    print(kv_line(None, policy), file=out)
    return prompt_cache


# The fused kvarn decode kernel walks verify blocks up to this query width;
# wider blocks fall back to a full materialize per round.
_KVARN_VERIFY_QL = 4


def setup_kvarn_mtp_cache(model, drafter, kv_bits, kv_tail_tokens, block, out=None):
    """setup_kvarn_cache for the MTP target. Declines drafters that read the
    target KV back (kvarn records are not raw K/V; ``uses_shared_kv`` defaults
    True so unknown drafters decline conservatively), warns when the verify
    width exceeds the fused route, and builds via mlx-vlm's cache maker so
    the arch's own make_cache hook applies. Returns the cache list or None."""
    out = out if out is not None else sys.stderr
    if getattr(drafter, "uses_shared_kv", True):
        print(
            "warning: --kv-quant-scheme kvarn dropped: this drafter reads "
            "the target KV back (kvarn records are not raw K/V)",
            file=out,
        )
        return None
    from mlx_vlm.models import cache as _cache

    lm = model.language_model if hasattr(model, "language_model") else model
    prompt_cache = setup_kvarn_cache(
        model,
        kv_bits,
        kv_tail_tokens,
        None,
        out=out,
        make_cache=lambda: _cache.make_prompt_cache(lm),
    )
    if prompt_cache is not None and block and block > _KVARN_VERIFY_QL:
        print(
            f"warning: verify width {block} exceeds the fused kvarn route "
            f"({_KVARN_VERIFY_QL}); verify rounds fall back to materialized "
            "attention",
            file=out,
        )
    if prompt_cache is not None:
        harden_mtp_rollback(model)
        if lm is not model:
            harden_mtp_rollback(lm)
    return prompt_cache


def harden_mtp_rollback(obj) -> None:
    """Instance-wrap ``rollback_speculative_cache`` with a trim pre-check:
    every ``_can_trim``-aware cache must accept the full trim before the
    stock body mutates any. Upstream rollbacks (qwen3_5, gemma4) ignore
    ``trim`` returns, so a cache that refused would silently desync layer
    offsets; the guard turns that into a loud error. Idempotent."""
    fn = getattr(obj, "rollback_speculative_cache", None)
    if fn is None or getattr(fn, "_gmlx_kvarn_guard", False):
        return

    def _guarded(caches, gdn_states, accepted, block_size):
        if isinstance(accepted, int):
            max_a = accepted
        elif hasattr(accepted, "tolist"):
            max_a = max(int(a) for a in accepted.reshape(-1).tolist())
        else:
            max_a = max(int(a) for a in accepted)
        trim = int(block_size) - max_a - 1
        if trim > 0:
            refused = [
                type(c).__name__
                for c in caches
                if c is not None and hasattr(c, "_can_trim") and not c._can_trim(trim)
            ]
            if refused:
                raise RuntimeError(
                    f"MTP rollback: {', '.join(refused)} refuse trim({trim}); "
                    "aborting before a partial rollback desyncs layer offsets"
                )
        return fn(caches, gdn_states, accepted, block_size)

    _guarded._gmlx_kvarn_guard = True
    obj.rollback_speculative_cache = _guarded


def generate(
    model,
    tokenizer,
    prompt: str | list[int],
    *,
    max_tokens: int = 64,
    temp: float = 0.0,
    top_p: float = 0.95,
    top_k: int = 0,
    min_p: float = 0.05,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.0,
    repetition_penalty: float = 0.0,
    repetition_context_size: int = 20,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    logit_bias: dict | None = None,
    stop: list | None = None,
    system_prompt: str | None = None,
    template_kwargs: dict | None = None,
    max_kv_size: int | None = None,
    kv_bits: int | None = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    kv_quant_scheme: str | None = None,
    kv_tail_tokens: int = 1024,
    prefill_step_size: int | None = None,
    thinking_budget: int | None = None,
    thinking_start_token: str | None = None,
    thinking_end_token: str | None = None,
    apply_chat_template: bool = True,
    prefill_progress: bool = False,
    over_generation: int = 0,
    inject_critique: str | None = None,
    inject_no_thinking: bool = False,
    over_temp: float | None = None,
    over_top_p: float | None = None,
    over_top_k: int | None = None,
    over_min_p: float | None = None,
    over_generation_log: str | None = None,
    over_label: str | None = None,
    verbose: bool = False,
    reasoning: str | None = None,
) -> str:
    """Generate text from a kquant model. Applies the tokenizer's chat template
    to a string prompt when present (set ``apply_chat_template=False`` for
    base models or pre-tokenized prompts). ``system_prompt`` and
    ``template_kwargs`` (extra ``apply_chat_template`` kwargs, e.g.
    ``{"enable_thinking": False}``) only apply on the templated path.
    ``stop`` sequences end the generation when they appear in the output (the
    matched text is trimmed). ``kv_bits``/``kv_group_size``/
    ``quantized_kv_start`` quantize the KV cache (mlx-lm ``QuantizedKVCache``).
    ``thinking_budget`` caps reasoning tokens for thinking models: once ~N
    thinking tokens are generated it forces ``</think>`` so the model answers.
    An explicit budget is honored even when ``enable_thinking`` is false (a model
    may still emit ``<think>``); it is a no-op only when no ``<think>`` is ever
    generated or the tokenizer lacks a ``</think>`` token. With or without a
    budget, ^T (see ``thinking_budget.install_finish_thinking_key``, armed by
    the CLI entry points) force-closes an open thinking block the same way.
    ``thinking_start_token`` / ``thinking_end_token`` name the reasoning
    markers for a model whose spelling the tokenizer/template probe does not
    recognize; unset, the probe decides.
    Returns the generated text. ``prefill_progress`` shows a stderr spinner during a long prefill
    (TTY only; cleared before the first token).
    ``reasoning`` shapes how a *verbose* stream displays a thinking model's
    chain-of-thought: ``"show"`` styles it under a label (the chat REPL's
    rendering), ``"hide"`` collapses it to the elapsed/token payoff, and
    ``None``/``"raw"`` (the default) streams verbatim. The returned text is
    always raw.
    ``over_generation``/``inject_critique`` route to an experimental
    two-phase over-generation probe (see overgen.py)."""
    import mlx_lm
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    xtc_kwargs = {}
    if xtc_probability > 0:
        xtc_kwargs = {
            "xtc_probability": xtc_probability,
            "xtc_threshold": xtc_threshold,
            "xtc_special_tokens": tokenizer.encode("\n")
            + list(tokenizer.eos_token_ids),
        }
    sampler = make_sampler(
        temp=temp, top_p=top_p, top_k=top_k, min_p=min_p, **xtc_kwargs
    )
    from gmlx.load.tokenizer import merge_suppressed_tokens
    logit_bias = merge_suppressed_tokens(logit_bias, tokenizer)
    logits_processors = make_logits_processors(
        logit_bias=logit_bias or None,
        repetition_penalty=repetition_penalty or None,
        repetition_context_size=repetition_context_size,
        presence_penalty=presence_penalty or None,
        frequency_penalty=frequency_penalty or None,
    )
    # Capture the raw prompt before templating; the over-generation log uses it
    # (or over_label) to pair free vs injected runs of the same input.
    _over_prompt = prompt if isinstance(prompt, str) else None
    if (
        isinstance(prompt, str)
        and apply_chat_template
        and tokenizer.chat_template is not None
    ):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **(template_kwargs or {}),
        )
    # Thinking-token cap: append last so its forced one-hot logits win over the
    # penalty/bias processors. Honored whenever a budget is set, regardless of
    # enable_thinking; seed the in-thinking state from whether the *rendered
    # prompt* actually opens a <think> block (a pre-fill model opens it in the
    # prompt; a generate model emits it, which the processor detects). The flag
    # only shapes the prompt via the template - it never gates the cap.
    tbp = None
    if thinking_budget is None or thinking_budget >= 0:
        from .thinking_budget import (
            make_thinking_budget_processor,
            prompt_opens_thinking,
        )

        # interruptible: with no budget set the processor never trips on its
        # own, but stays armed for the ^T finish-thinking key.
        tbp = make_thinking_budget_processor(
            tokenizer,
            thinking_budget,
            start_in_thinking=prompt_opens_thinking(
                prompt, thinking_start_token, thinking_end_token,
                tokenizer=tokenizer),
            verbose=verbose,
            interruptible=True,
            start_token=thinking_start_token,
            end_token=thinking_end_token,
        )
        if tbp is not None:
            logits_processors = list(logits_processors) + [tbp]

    # DiffusionGemma is non-autoregressive - route it to the block-diffusion
    # denoiser (mlx-vlm), which ignores the AR sampler/stop controls below.
    from .diffusion import is_diffusion_model

    if is_diffusion_model(model):
        from .diffusion import generate as _diffusion_generate

        return _diffusion_generate(
            model, tokenizer, prompt, max_tokens=max_tokens, verbose=verbose
        )

    prompt_cache = None
    if (kv_quant_scheme or "uniform").lower() == "kvarn":
        if inject_critique is not None or (over_generation and over_generation > 0):
            # The over-generation runner builds its own per-pass caches and
            # never sees a pre-armed prompt cache.
            print(
                "warning: --kv-quant-scheme kvarn dropped: over-generation/"
                "critique passes rebuild their own fp16 KV cache",
                file=sys.stderr,
            )
        else:
            prompt_cache = setup_kvarn_cache(
                model, kv_bits, kv_tail_tokens, max_kv_size
            )
        # Handled here whatever the outcome: the affine policy path must
        # never touch a kvarn stack.
        kv_bits = None
    elif kv_bits is not None:
        # Resolve on the constructed stack. A bare make_cache probe
        # misses what max_kv_size builds.
        from mlx_lm.models.cache import make_prompt_cache as _mpc

        from gmlx.cache.kv_policy import (arm_stack, kv_line,
                                          resolve_kv_quant_policy)

        prompt_cache = _mpc(model, max_kv_size=max_kv_size)
        policy = resolve_kv_quant_policy(
            prompt_cache, kv_bits=kv_bits, kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start, scheme=kv_quant_scheme,
            max_kv_size=max_kv_size)
        print(kv_line(None, policy), file=sys.stderr)
        if policy.verdict == "error":
            raise SystemExit(2)
        if policy.verdict == "dropped":
            prompt_cache = None
            kv_bits = None
        else:
            arm_stack(prompt_cache, policy)

    gen_kwargs = {
        "max_tokens": max_tokens,
        "sampler": sampler,
        "logits_processors": logits_processors,
        "max_kv_size": max_kv_size,
        "kv_bits": kv_bits,
        "kv_group_size": kv_group_size,
        "quantized_kv_start": quantized_kv_start,
    }
    if prompt_cache is not None:
        gen_kwargs["prompt_cache"] = prompt_cache
    # Module-attribute lookup so the monkeypatch seam
    # gmlx.load.loader._resolve_prefill_step stays live for this path.
    step, defaulted = loader._resolve_prefill_step(model, prefill_step_size)
    if defaulted and verbose:
        print(
            f"[prefill] streaming model: chunk size defaults to {step} "
            "(--prefill-step-size overrides)"
        )
    if step is not None:
        gen_kwargs["prefill_step_size"] = step

    if inject_critique is not None or (over_generation and over_generation > 0):
        over_sampler = make_sampler(
            temp=over_temp if over_temp is not None else temp,
            top_p=over_top_p if over_top_p is not None else top_p,
            top_k=over_top_k if over_top_k is not None else top_k,
            min_p=over_min_p if over_min_p is not None else min_p,
        )
        base_kwargs = {
            k: v
            for k, v in gen_kwargs.items()
            if k not in ("max_tokens", "sampler", "logits_processors",
                         "max_kv_size", "prompt_cache")
        }
        params = {
            "max_tokens": max_tokens, "temp": temp, "top_p": top_p,
            "top_k": top_k, "min_p": min_p, "over_generation": over_generation,
            "over_temp": over_temp, "over_top_p": over_top_p,
            "over_top_k": over_top_k, "over_min_p": over_min_p,
            "inject_no_thinking": inject_no_thinking,
        }
        # The injected critique can disable thinking independently of the main
        # turn (the phase-1 prompt is already templated above). The template
        # hint (enable_thinking=False) covers gen-prompt-gated models; a
        # budget-0 thinking cap on the critique force-closes channel-style
        # thinking the template can't gate.
        critique_tk = dict(template_kwargs or {})
        over_logits_processors = logits_processors
        if inject_critique is not None and inject_no_thinking:
            critique_tk["enable_thinking"] = False
            from .thinking_budget import make_thinking_budget_processor

            tbp = make_thinking_budget_processor(
                tokenizer, 0, start_in_thinking=False, verbose=verbose,
                # Seam detection needs the model's untouched stop behavior.
                eos_floor=False,
                start_token=thinking_start_token,
                end_token=thinking_end_token,
            )
            if tbp is not None:
                over_logits_processors = list(logits_processors) + [tbp]
        return _generate_over(
            model, tokenizer, prompt,
            main_sampler=sampler, over_sampler=over_sampler,
            logits_processors=logits_processors,
            over_logits_processors=over_logits_processors,
            base_kwargs=base_kwargs,
            max_kv_size=max_kv_size, max_tokens=max_tokens,
            window=over_generation, inject_critique=inject_critique,
            template_kwargs=critique_tk, log_path=over_generation_log,
            orig_prompt=_over_prompt, label=over_label,
            params=params, verbose=verbose, reasoning=reasoning,
        )

    close_progress = None
    if prefill_progress and sys.stderr.isatty():
        progress_cb, close_progress = _prefill_progress_ui()
        gen_kwargs["prompt_progress_callback"] = progress_cb

    try:
        if tbp is not None:
            from .thinking_budget import set_finish_key_target

            set_finish_key_target(tbp)
        stop = [s for s in (stop or []) if s]
        styled = verbose and reasoning in ("show", "hide")
        if not stop:
            if verbose and (styled
                            or _echo_think_tag(prompt, tokenizer) is not None):
                # Own the loop (same verbose format as mlx_lm.generate) when
                # the stream needs shaping: reasoning show/hide styling, or a
                # pre-fill template that ended the prompt inside an open
                # thinking block (the raw stream alone shows a bare close tag,
                # and mlx_lm.generate has no seam between its separator and
                # the first token).
                from mlx_lm.generate import stream_generate

                print("=" * 10)
                emit, close_emit = _verbose_emitter(prompt, tokenizer, reasoning)
                text, last = "", None
                for last in stream_generate(model, tokenizer, prompt,
                                            **gen_kwargs):
                    emit(last.text)
                    text += last.text
                close_emit()
                print()
                print("=" * 10)
                if not text:
                    print("No text generated for this prompt")
                else:
                    print(
                        f"Prompt: {last.prompt_tokens} tokens, "
                        f"{last.prompt_tps:.3f} tokens-per-sec"
                    )
                    print(
                        f"Generation: {last.generation_tokens} tokens, "
                        f"{last.generation_tps:.3f} tokens-per-sec"
                    )
                    print(f"Peak memory: {last.peak_memory:.3f} GB")
                return text
            return mlx_lm.generate(
                model, tokenizer, prompt, verbose=verbose, **gen_kwargs)

        # Stop sequences need the streamed text: scan with a held-back tail so
        # a stop string split across segments still matches, then end the
        # stream. The scanner sees the raw text; the emitter only displays.
        from mlx_lm.generate import stream_generate

        emit = close_emit = None
        if verbose:
            emit, close_emit = _verbose_emitter(prompt, tokenizer, reasoning)
        scanner = StopScanner(stop)
        pieces, last = [], None
        for r in stream_generate(model, tokenizer, prompt, **gen_kwargs):
            out, hit = scanner.feed(r.text)
            if out:
                pieces.append(out)
                if emit is not None:
                    emit(out)
            last = r
            if hit:
                break
        else:
            tail = scanner.flush()
            if tail:
                pieces.append(tail)
                if emit is not None:
                    emit(tail)
        if verbose:
            close_emit()
            print()
            if last is not None:
                print(
                    f"[generate] {last.generation_tokens} tok @ "
                    f"{last.generation_tps:.1f} tok/s"
                )
        return "".join(pieces)
    finally:
        if tbp is not None:
            from .thinking_budget import clear_finish_key_target

            clear_finish_key_target()
        if close_progress is not None:
            close_progress()


# Free over-generation cuts the window short after this many consecutive
# stop/special tokens: once the model only re-emits the stop it would have
# honored, further tokens carry no signal.
_OVERGEN_STOP_RUN = 5


def _generate_over(
    model,
    tokenizer,
    prompt,
    *,
    main_sampler,
    over_sampler,
    logits_processors,
    base_kwargs,
    max_kv_size,
    max_tokens,
    window,
    inject_critique,
    template_kwargs,
    log_path,
    params,
    verbose,
    over_logits_processors=None,
    orig_prompt=None,
    label=None,
    reasoning=None,
):
    """Over-generation probe (experimental). Phase 1 generates to the natural
    stop (the seam); phase 2 continues from phase 1's KV cache, either forcing
    ``window`` free tokens past the seam or injecting a follow-up critique turn
    and answering it. ``inject_critique`` selects the mode; when both it and
    ``window`` are set, ``window`` caps the injected reply. Returns the full
    text and appends a JSONL record to ``log_path`` when set. See overgen.py.
    ``reasoning`` styles phase 1's verbose stream like :func:`generate`'s;
    phase 2 prints raw (the forced continuation is the thing being probed)."""
    from mlx_lm.generate import stream_generate
    from mlx_lm.models.cache import make_prompt_cache

    from gmlx.spec.overgen import (
        append_log,
        build_critique_bridge,
        collect_interim_eos,
        seam_marker,
        suppressed_eos,
    )

    if over_logits_processors is None:
        over_logits_processors = logits_processors
    real_eos = set(getattr(tokenizer, "eos_token_ids", None) or [])
    cache = make_prompt_cache(model, max_kv_size=max_kv_size)

    # Phase 1: generate to the first EOG token. The EOG set is neutralized so
    # the seam is detected here (and its token held out of the kept text)
    # instead of stream_generate ending the call.
    pre, seam, last = [], None, None
    emit = close_emit = None
    if verbose:
        # Phase 1 is a normal reply: style it like the main verbose path
        # (thinking rendering included). Phase 2 stays raw: the forced
        # continuation is the thing being probed.
        emit, close_emit = _verbose_emitter(prompt, tokenizer, reasoning)
    with suppressed_eos(tokenizer):
        for r in stream_generate(
            model, tokenizer, prompt, max_tokens=max_tokens,
            sampler=main_sampler, logits_processors=logits_processors,
            prompt_cache=cache, **base_kwargs,
        ):
            last = r
            if int(r.token) in real_eos:
                seam = {
                    "step": r.generation_tokens,
                    "token_id": int(r.token),
                    "text": r.text,
                }
                break
            pre.append(r.text)
            if emit is not None:
                emit(r.text)
    if close_emit is not None:
        close_emit()
    pre_text = "".join(pre)
    phase1_last = last  # the seam response; phase 2 reassigns `last` below

    if seam is None:
        if verbose:
            print()
            print("[over-generation] no stop within max-tokens; nothing forced")
        return pre_text

    if not (seam["text"] or "").strip():
        try:
            seam["text"] = tokenizer.decode([seam["token_id"]])
        except Exception:
            seam["text"] = ""
    if verbose:
        print(seam_marker(seam), end="", flush=True)

    # Phase 2: continue from the same cache. Only the bridge (one seam token, or
    # the critique turn) is prefilled; phase 1's context stays cached.
    over_parts, over_tokens, early_stop = [], [], False
    if inject_critique is not None:
        mode = "inject"
        bridge = build_critique_bridge(
            tokenizer, seam["token_id"], inject_critique, template_kwargs
        )
        cap = window if window and window > 0 else max_tokens
        for r in stream_generate(
            model, tokenizer, bridge, max_tokens=cap, sampler=over_sampler,
            logits_processors=over_logits_processors, prompt_cache=cache,
            **base_kwargs,
        ):
            over_parts.append(r.text)
            over_tokens.append(int(r.token))
            if verbose:
                print(r.text, end="", flush=True)
            last = r
    else:
        mode = "free"
        # Once the model only re-emits stop/special tokens (the EOG it would
        # have honored, or other control tokens), the rest of the window is
        # noise; cut it short on a run of them.
        special_ids = set(real_eos) | set(
            getattr(tokenizer, "all_special_ids", None) or []
        )
        run = 0
        with suppressed_eos(tokenizer):
            for r in stream_generate(
                model, tokenizer, [seam["token_id"]], max_tokens=window,
                sampler=over_sampler, logits_processors=over_logits_processors,
                prompt_cache=cache, **base_kwargs,
            ):
                over_parts.append(r.text)
                over_tokens.append(int(r.token))
                if verbose:
                    print(r.text, end="", flush=True)
                last = r
                run = run + 1 if int(r.token) in special_ids else 0
                if run >= _OVERGEN_STOP_RUN:
                    early_stop = True
                    break
    over_text = "".join(over_parts)
    interim = collect_interim_eos(over_tokens, real_eos)
    # One turn, two stream_generate calls: report the whole turn's tokens, not
    # just phase 2's. generation_tps is decode-only in each phase, so combine by
    # decode time (tokens / tps) for an honest overall rate.
    p1_tok = phase1_last.generation_tokens if phase1_last is not None else 0
    p2_tok = len(over_tokens)
    total_tok = p1_tok + p2_tok

    if verbose:
        print()
        p1_tps = phase1_last.generation_tps if phase1_last is not None else 0.0
        p2_tps = last.generation_tps if (last is not None and p2_tok) else 0.0
        t1 = p1_tok / p1_tps if p1_tps else 0.0
        t2 = p2_tok / p2_tps if p2_tps else 0.0
        overall = total_tok / (t1 + t2) if (t1 + t2) else 0.0
        print(f"[generate] {total_tok} tok @ {overall:.1f} tok/s")
        extra = f", {len(interim)} further stop(s)" if interim else ""
        if early_stop:
            extra += " (cut short: repeated stop/special tokens)"
        print(
            f"[over-generation] mode={mode} seam id={seam['token_id']} "
            f"forced {p2_tok} token(s){extra}"
        )

    if log_path:
        append_log(log_path, {
            "mode": mode,
            "label": label,
            "prompt": orig_prompt,
            "seam": seam,
            "inject_critique": inject_critique,
            "pre_text": pre_text,
            "over_text": over_text,
            "interim_stops": interim,
            "phase1_tokens": p1_tok,
            "over_tokens": p2_tok,
            "total_tokens": total_tok,
            "early_stop": early_stop,
            "params": params,
        })

    return pre_text + over_text


def _chunked_prefill_cache(lm, input_ids, chunk, cache=None):
    """Populate a fresh prompt cache (or the given empty one) by prefilling
    ``input_ids[:, :-1]`` in ``chunk``-token steps, leaving the trailing token
    for the caller's capture forward. Bounds each expert-gather forward to a
    width proven free of the single-shot MoE gather memory bug. The drafter's
    shared K/V is read back from this (fully populated) cache, so chunking is
    loss-free. Returns the cache."""
    from mlx_vlm.models import cache as _cache

    for attr in ("_position_ids", "_rope_deltas"):
        if hasattr(lm, attr):
            setattr(lm, attr, None)
    c = cache if cache is not None else _cache.make_prompt_cache(lm)
    n_total = input_ids.shape[1]
    i = 0
    while i < n_total - 1:
        n = min(chunk, n_total - 1 - i)
        lm(input_ids[:, i : i + n], cache=c, n_to_process=n)
        mx.eval([cc.state for cc in c])
        i += n
    return c


_MTP_FINISH_WHY = "on the stock MTP engine (run with --no-mtp to use it)"


def _with_mtp_finish_key_notice(fn, *args, **kwargs):
    """Run ``fn`` with the ^T finish-thinking target armed as "unsupported":
    mlx-vlm's stock MTP round exposes no forced-close seam, so the key
    explains itself instead of silently doing nothing. The owned engine
    (the default; see the routing in ``_generate_speculative`` and
    ``_stream_generate_speculative``) re-arms with a real hook."""
    from .thinking_budget import (
        FinishKeyUnsupported,
        clear_finish_key_target,
        set_finish_key_target,
    )

    set_finish_key_target(FinishKeyUnsupported(_MTP_FINISH_WHY))
    try:
        return fn(*args, **kwargs)
    finally:
        clear_finish_key_target()


def generate_speculative(model, drafter, tokenizer, prompt, **kwargs) -> dict:
    """See :func:`_generate_speculative` (^T-notice shim)."""
    return _with_mtp_finish_key_notice(
        _generate_speculative, model, drafter, tokenizer, prompt, **kwargs
    )


def _generate_speculative(
    model,
    drafter,
    tokenizer,
    prompt: str | list[int],
    *,
    max_tokens: int = 128,
    temp: float = 0.0,
    top_p: float = 0.95,
    top_k: int = 0,
    min_p: float = 0.05,
    draft_block_size: int | None = None,
    apply_chat_template: bool = True,
    system_prompt: str | None = None,
    template_kwargs: dict | None = None,
    verbose: bool = False,
    reasoning: str | None = None,
    kv_bits: int | None = None,
    kv_group_size: int = 64,
    kv_quant_scheme: str | None = None,
    kv_tail_tokens: int = 1024,
    thinking_budget: int | None = None,
    thinking_start_token: str | None = None,
    thinking_end_token: str | None = None,
) -> dict:
    """Single-stream MTP speculative generation via mlx-vlm's engine.

    Drives ``mlx_vlm.generate.ar.generate_step`` with the in-memory (target,
    drafter) pair (``draft_kind="mtp"``). Returns a stats dict
    ``{text, tokens, elapsed_s, decode_tps, accept_rate, mean_accept_len,
    rounds}`` - ``accept_rate`` is fraction of drafted tokens accepted; greedy
    output must match the non-speculative path token-for-token (lossless gate).
    ``reasoning`` shapes the *verbose* stream like :func:`generate`'s; the
    returned ``text`` is always raw.
    """
    # The owned engine is the default (as it already is for the REPL, serve,
    # and bench): it rolls the cache back cleanly on early stops and carries
    # the ^T finish-thinking seam. GMLX_OWNED_ROUND=0 opts back to mlx-vlm's
    # stock round, except for drafters whose contract demands the owned
    # engine (deepseek_v4: 4D hidden + rotating-undo rollback; stochastic
    # acceptance also lives only in the owned walk).
    from gmlx.spec.speculative import use_owned_engine
    if (os.environ.get("GMLX_OWNED_ROUND") != "0"
            or use_owned_engine(drafter, temp)):
        return generate_speculative_owned(
            model, drafter, tokenizer, prompt,
            max_tokens=max_tokens, temp=temp, top_p=top_p, top_k=top_k,
            min_p=min_p, draft_block_size=draft_block_size,
            apply_chat_template=apply_chat_template,
            system_prompt=system_prompt, template_kwargs=template_kwargs,
            verbose=verbose, reasoning=reasoning,
            kv_bits=kv_bits, kv_group_size=kv_group_size,
            kv_quant_scheme=kv_quant_scheme,
            kv_tail_tokens=kv_tail_tokens,
            thinking_budget=thinking_budget,
            thinking_start_token=thinking_start_token,
            thinking_end_token=thinking_end_token,
        )

    kvarn_cache = None
    if kv_quant_scheme == "kvarn":
        block_note = draft_block_size or int(getattr(drafter.config, "block_size", 3))
        kvarn_cache = setup_kvarn_mtp_cache(
            model, drafter, kv_bits, kv_tail_tokens, block_note
        )
        # kvarn owns the width request; a declined scheme means fp16, not
        # a silent fall-through to another quantization.
        kv_bits = None
    if kv_bits is not None:
        # The stock mlx-vlm round has no KV-quantization hook and the models
        # routed here (gemma4/qwen3.x drafters) have no pooled caches.
        from gmlx.cache.kv_policy import dropped_policy, kv_line

        print(
            kv_line(None, dropped_policy(
                "stock MTP rounds have no KV quantization hook",
                kv_bits, kv_group_size, "single")),
            file=sys.stderr,
        )
    if thinking_budget is not None:
        # The forced-close seam lives in the owned rounds only.
        print(
            "warning: --thinking-budget not applied on the stock MTP engine",
            file=sys.stderr,
        )

    from mlx_lm.sample_utils import make_sampler
    from gmlx.spec.helpers import _resolve_block_total
    from mlx_vlm.generate.ar import generate_step

    if (
        isinstance(prompt, str)
        and apply_chat_template
        and tokenizer.chat_template is not None
    ):
        messages = (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        )
        messages.append({"role": "user", "content": prompt})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **(template_kwargs or {}),
        )
    if isinstance(prompt, str):
        # mlx_lm.stream_generate's single-BOS rule: don't add special tokens
        # when the (chat-templated) prompt already starts with the BOS string,
        # or the BOS post-processor would prepend a second one.
        add_special = tokenizer.bos_token is None or not prompt.startswith(
            tokenizer.bos_token
        )
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=add_special)
    else:
        prompt_ids = prompt
    input_ids = mx.array(prompt_ids, dtype=mx.int32)[None]

    # Greedy (temp==0): pass sampler=None so the engine's ``sampler_is_greedy``
    # path fires and the verify walk computes argmax over the whole draft block
    # in one batched op, instead of the per-position deferred walk (one lm_head
    # projection + CPU sync per position). Token-identical to the per-position
    # walk under greedy; ~6% faster when the verify forward is cheap (e.g. MoE
    # targets, where the per-round walk/drafter cost is a real fraction of the
    # round). A non-greedy sampler must be passed explicitly for temp>0.
    sampler = (
        None
        if temp == 0.0
        else make_sampler(temp=temp, top_p=top_p, top_k=top_k, min_p=min_p)
    )
    block = _resolve_block_total(drafter, draft_block_size)
    eos_ids = set(getattr(tokenizer, "eos_token_ids", None) or [tokenizer.eos_token_id])

    detok = tokenizer.detokenizer
    detok.reset()
    n = 0
    emit = close_emit = None
    if verbose:
        emit, close_emit = _verbose_emitter(prompt, tokenizer, reasoning)
    # Split prefill from decode at the first-token boundary, mirroring
    # mlx-lm's stream_generate: the first yielded token's wall time is the
    # (chunked) prefill cost; decode_tps is measured over the steady decode
    # window only, so it's directly comparable to the plain tg_tps A/B.
    tic = time.perf_counter()
    prefill_s = None
    # Chunk the target prefill (see _chunked_prefill_cache): hand generate_step
    # only the trailing token so no single expert-gather forward exceeds the safe
    # width. Short prompts prefill in one already-safe forward. The pre-prefill
    # wall time is inside the first-token window, so prefill_tps stays correct.
    feed, prompt_cache = input_ids, kvarn_cache
    if input_ids.shape[1] > _PREFILL_CHUNK:
        prompt_cache = _chunked_prefill_cache(
            model.language_model, input_ids, _PREFILL_CHUNK, cache=kvarn_cache
        )
        feed = input_ids[:, -1:]
    for tok, _logprobs in generate_step(
        feed,
        model,
        None,
        None,
        prompt_cache=prompt_cache,
        max_tokens=max_tokens,
        temperature=temp,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        sampler=sampler,
        draft_model=drafter,
        draft_kind="mtp",
        draft_block_size=block,
    ):
        if prefill_s is None:
            prefill_s = time.perf_counter() - tic
            tic = time.perf_counter()
        tok = int(tok[0]) if isinstance(tok, list) else int(tok)
        if tok in eos_ids:
            break
        detok.add_token(tok)
        n += 1
        if emit is not None:
            seg = detok.last_segment
            if seg:
                emit(seg)
        if n >= max_tokens:
            break
    detok.finalize()
    decode_s = time.perf_counter() - tic
    if prefill_s is None:
        prefill_s = 0.0
    if emit is not None:
        if detok.last_segment:
            emit(detok.last_segment)
        close_emit()
        print()

    accept = list(getattr(drafter, "accept_lens", []) or [])
    draft = list(getattr(drafter, "draft_lens", []) or [])
    total_a, total_d = sum(accept), sum(draft)
    return {
        "text": detok.text,
        "tokens": n,
        "elapsed_s": prefill_s + decode_s,
        "prefill_s": prefill_s,
        "decode_tps": n / decode_s if decode_s > 0 else 0.0,
        "accept_rate": (total_a / total_d) if total_d else 0.0,
        "mean_accept_len": (total_a / len(accept)) if accept else 0.0,
        "rounds": len(accept),
        "draft_n": total_d,
        "draft_n_accepted": total_a,
    }


def generate_speculative_owned(
    model,
    drafter,
    tokenizer,
    prompt: str | list[int],
    *,
    max_tokens: int = 128,
    temp: float = 0.0,
    top_p: float = 0.95,
    top_k: int = 0,
    min_p: float = 0.05,
    draft_block_size: int | None = None,
    apply_chat_template: bool = True,
    system_prompt: str | None = None,
    template_kwargs: dict | None = None,
    verbose: bool = False,
    reasoning: str | None = None,
    kv_bits: int | None = None,
    kv_group_size: int = 64,
    kv_quant_scheme: str | None = None,
    kv_tail_tokens: int = 1024,
    thinking_budget: int | None = None,
    thinking_start_token: str | None = None,
    thinking_end_token: str | None = None,
) -> dict:
    """Same contract as generate_speculative but drives the owned
    stream_speculative engine (engine/speculative.py) instead of mlx-vlm's
    generate_step. This is the bench_tg_depth default (matches serve);
    GMLX_OWNED_ROUND=0 opts back to mlx-vlm's generate_speculative."""
    from mlx_lm.sample_utils import make_sampler
    from gmlx.spec.helpers import _resolve_block_total
    from mlx_vlm.models import cache as _cache

    from gmlx.spec.speculative import annotate_sampling_params, stream_speculative

    from .thinking_budget import (
        clear_finish_key_target,
        make_mtp_finish_hook,
        prompt_opens_thinking,
        set_finish_key_target,
    )

    if (
        isinstance(prompt, str)
        and apply_chat_template
        and tokenizer.chat_template is not None
    ):
        messages = (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        )
        messages.append({"role": "user", "content": prompt})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **(template_kwargs or {}),
        )
    if isinstance(prompt, str):
        add_special = tokenizer.bos_token is None or not prompt.startswith(
            tokenizer.bos_token
        )
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=add_special)
    else:
        prompt_ids = prompt
    input_ids = mx.array(prompt_ids, dtype=mx.int32)

    sampler = (
        None
        if temp == 0.0
        else make_sampler(temp=temp, top_p=top_p, top_k=top_k, min_p=min_p)
    )
    annotate_sampling_params(
        sampler, temp=temp, top_p=top_p, top_k=top_k, min_p=min_p)
    block = _resolve_block_total(drafter, draft_block_size)
    eos_ids = set(getattr(tokenizer, "eos_token_ids", None) or [tokenizer.eos_token_id])

    lm = model.language_model if hasattr(model, "language_model") else model
    prompt_cache = None
    if kv_quant_scheme == "kvarn":
        prompt_cache = setup_kvarn_mtp_cache(
            model, drafter, kv_bits, kv_tail_tokens, block
        )
        # kvarn owns the width request; a declined scheme means fp16, not
        # a silent fall-through to the pooled packing.
        kv_bits = None
    if prompt_cache is None:
        prompt_cache = _cache.make_prompt_cache(lm)
    if kv_bits is not None:
        # These are the rounds serve quantizes on, so the CLI quantizes
        # the same layers: rollback trims, and affine packing is
        # per-token along head_dim, so a trim is an offset move. The
        # bare-stock GDN fallback is the one target that cannot verify
        # on packed tuples.
        from gmlx.cache.kv_policy import (arm_stack, kv_line,
                                          resolve_kv_quant_policy)
        from gmlx.spec.engine import mtp_kv_decline

        decline = mtp_kv_decline(lm)
        policy = resolve_kv_quant_policy(
            prompt_cache, kv_bits=kv_bits, kv_group_size=kv_group_size,
            mtp=True, scheme=kv_quant_scheme,
            can_quantize_kv=decline is None, no_kv_reason=decline)
        print(kv_line(None, policy), file=sys.stderr)
        if policy.verdict == "error":
            raise SystemExit(2)
        arm_stack(prompt_cache, policy, hold=False)

    detok = tokenizer.detokenizer
    detok.reset()
    n = 0
    emit = close_emit = None
    if verbose:
        emit, close_emit = _verbose_emitter(prompt, tokenizer, reasoning)
    # Thinking budget + ^T finish-thinking: the owned rounds honor a
    # forced-close hook; arm it (a None hook - unresolvable markers - drops
    # the budget with a note and leaves the key a silent no-op, matching the
    # plain path).
    hook = make_mtp_finish_hook(
        tokenizer,
        budget=thinking_budget,
        start_in_thinking=prompt_opens_thinking(
            prompt, thinking_start_token, thinking_end_token,
            tokenizer=tokenizer),
        verbose=verbose,
        start_token=thinking_start_token,
        end_token=thinking_end_token,
    )
    set_finish_key_target(hook)
    tic = time.perf_counter()
    prefill_s = None
    try:
        for tok in stream_speculative(
            model,
            drafter,
            input_ids,
            prompt_cache=prompt_cache,
            max_tokens=max_tokens,
            sampler=sampler,
            draft_block_size=block,
            thinking_hook=hook,
        ):
            if prefill_s is None:
                prefill_s = time.perf_counter() - tic
                tic = time.perf_counter()
            tok = int(tok)
            if tok in eos_ids:
                break
            detok.add_token(tok)
            n += 1
            if emit is not None:
                seg = detok.last_segment
                if seg:
                    emit(seg)
            if n >= max_tokens:
                break
    finally:
        clear_finish_key_target()
    detok.finalize()
    decode_s = time.perf_counter() - tic
    if prefill_s is None:
        prefill_s = 0.0
    if emit is not None:
        if detok.last_segment:
            emit(detok.last_segment)
        close_emit()
        print()

    accept = list(getattr(drafter, "accept_lens", []) or [])
    draft = list(getattr(drafter, "draft_lens", []) or [])
    total_a, total_d = sum(accept), sum(draft)
    return {
        "text": detok.text,
        "tokens": n,
        "elapsed_s": prefill_s + decode_s,
        "prefill_s": prefill_s,
        "decode_tps": n / decode_s if decode_s > 0 else 0.0,
        "accept_rate": (total_a / total_d) if total_d else 0.0,
        "mean_accept_len": (total_a / len(accept)) if accept else 0.0,
        "rounds": len(accept),
        "draft_n": total_d,
        "draft_n_accepted": total_a,
    }


def _prefill_into_cache(lm, input_ids, cache, chunk):
    """Prefill ``input_ids`` into an EXISTING prompt cache in ``chunk``-token steps
    (the cross-turn sibling of :func:`_chunked_prefill_cache`, which builds a fresh
    one). The chat REPL reuses one speculative cache across turns, so each turn's new
    tokens append here; bounding every expert-gather forward to the proven-safe width
    keeps the single-shot MoE gather correct. The caller leaves the trailing token for
    the drafter-capture forward (``generate_step`` with the drafter attached)."""
    for attr in ("_position_ids", "_rope_deltas"):
        if hasattr(lm, attr):
            setattr(lm, attr, None)
    n_total = input_ids.shape[1]
    i = 0
    while i < n_total:
        n = min(chunk, n_total - i)
        lm(input_ids[:, i : i + n], cache=cache, n_to_process=n)
        mx.eval([cc.state for cc in cache])
        i += n


class _MTPStreamResponse:
    """Minimal response object duck-typed for chat's ``_stream_reply``: it reads
    ``.text`` (the new segment), ``.generation_tokens`` and ``.generation_tps`` off
    the last chunk for the stat line, plus ``.prompt_tokens``/``.prompt_tps`` for
    the prefill half. Mirrors the fields mlx-lm's ``GenerationResponse`` exposes
    that the REPL actually consumes."""

    __slots__ = (
        "text",
        "generation_tokens",
        "generation_tps",
        "prompt_tokens",
        "prompt_tps",
    )

    def __init__(
        self,
        text: str,
        generation_tokens: int,
        generation_tps: float,
        prompt_tokens: int = 0,
        prompt_tps: float = 0.0,
    ):
        self.text = text
        self.generation_tokens = generation_tokens
        self.generation_tps = generation_tps
        self.prompt_tokens = prompt_tokens
        self.prompt_tps = prompt_tps


def _stream_generate_speculative_owned(
    model,
    drafter,
    tokenizer,
    prompt: str | list[int],
    *,
    prompt_cache,
    max_tokens: int = 256,
    temp: float = 0.0,
    top_p: float = 0.95,
    top_k: int = 0,
    min_p: float = 0.05,
    draft_block_size: int | None = None,
    thinking_budget: int | None = None,
    thinking_start_token: str | None = None,
    thinking_end_token: str | None = None,
    start_in_thinking: bool | None = None,
):
    """Owned-engine body of :func:`stream_generate_speculative` (the REPL
    default; see the routing note there). Same ``_MTPStreamResponse``
    surface as the stock round, but drives ``stream_speculative`` (which
    does its own chunked prefill through the persistent ``prompt_cache``).

    ``start_in_thinking`` seeds the ^T finish-thinking hook for a prompt that
    pre-opens a thinking block; None computes it from ``prompt`` when that is
    a string (chat passes token ids plus the explicit flag)."""
    from mlx_lm.sample_utils import make_sampler
    from gmlx.spec.helpers import _resolve_block_total

    from gmlx.spec.speculative import annotate_sampling_params, stream_speculative

    from .thinking_budget import (
        clear_finish_key_target,
        make_mtp_finish_hook,
        prompt_opens_thinking,
        set_finish_key_target,
    )

    if start_in_thinking is None:
        start_in_thinking = prompt_opens_thinking(
            prompt, thinking_start_token, thinking_end_token,
            tokenizer=tokenizer)
    if isinstance(prompt, str):
        add_special = tokenizer.bos_token is None or not prompt.startswith(
            tokenizer.bos_token
        )
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=add_special)
    else:
        prompt_ids = prompt
    input_ids = mx.array(prompt_ids, dtype=mx.int32)[None]

    sampler = (
        None
        if temp == 0.0
        else make_sampler(temp=temp, top_p=top_p, top_k=top_k, min_p=min_p)
    )
    annotate_sampling_params(
        sampler, temp=temp, top_p=top_p, top_k=top_k, min_p=min_p)
    block = _resolve_block_total(drafter, draft_block_size)
    eos_ids = set(getattr(tokenizer, "eos_token_ids", None) or [tokenizer.eos_token_id])

    detok = tokenizer.detokenizer
    detok.reset()
    n = 0
    n_prompt = input_ids.shape[1]
    prompt_tps = 0.0
    hook = make_mtp_finish_hook(
        tokenizer,
        budget=thinking_budget,
        start_in_thinking=start_in_thinking,
        start_token=thinking_start_token,
        end_token=thinking_end_token,
    )
    set_finish_key_target(hook)
    tic = time.perf_counter()
    prefill_done = False
    try:
        for tok in stream_speculative(
            model,
            drafter,
            input_ids,
            prompt_cache=prompt_cache,
            max_tokens=max_tokens,
            sampler=sampler,
            draft_block_size=block,
            thinking_hook=hook,
        ):
            if not prefill_done:
                prefill_done = True
                prefill_s = time.perf_counter() - tic
                prompt_tps = n_prompt / prefill_s if prefill_s > 0 else 0.0
                tic = time.perf_counter()
            tok = int(tok)
            if tok in eos_ids:
                break
            detok.add_token(tok)
            n += 1
            seg = detok.last_segment
            if seg:
                elapsed = time.perf_counter() - tic
                yield _MTPStreamResponse(
                    seg, n, n / elapsed if elapsed > 0 else 0.0, n_prompt, prompt_tps
                )
            if n >= max_tokens:
                break
    finally:
        clear_finish_key_target()
    detok.finalize()
    tail = detok.last_segment
    elapsed = time.perf_counter() - tic
    yield _MTPStreamResponse(
        tail or "", n, n / elapsed if elapsed > 0 else 0.0, n_prompt, prompt_tps
    )


def stream_generate_speculative(model, drafter, tokenizer, prompt, **kwargs):
    """See :func:`_stream_generate_speculative` (^T-notice shim)."""
    from .thinking_budget import (
        FinishKeyUnsupported,
        clear_finish_key_target,
        set_finish_key_target,
    )

    set_finish_key_target(FinishKeyUnsupported(_MTP_FINISH_WHY))
    try:
        yield from _stream_generate_speculative(
            model, drafter, tokenizer, prompt, **kwargs
        )
    finally:
        clear_finish_key_target()


def _stream_generate_speculative(
    model,
    drafter,
    tokenizer,
    prompt: str | list[int],
    *,
    prompt_cache,
    max_tokens: int = 256,
    temp: float = 0.0,
    top_p: float = 0.95,
    top_k: int = 0,
    min_p: float = 0.05,
    draft_block_size: int | None = None,
    thinking_budget: int | None = None,
    thinking_start_token: str | None = None,
    thinking_end_token: str | None = None,
    start_in_thinking: bool | None = None,
):
    """Streaming MTP speculative generation over a persistent prompt cache - the
    interactive-REPL sibling of :func:`generate_speculative`. Yields
    :class:`_MTPStreamResponse` chunks so ``chat``'s ``_stream_reply`` renders and
    times an MTP reply exactly like the plain text path.

    ``prompt`` is the already chat-templated new turn; its tokens append to
    ``prompt_cache`` (created once with mlx-vlm's ``make_prompt_cache`` and reused
    across turns - for single-stream ``draft_kind="mtp"`` that cache is a plain
    target KV cache the native drafter reads back, so cross-turn reuse is identical
    to the text path). Greedy output matches the non-speculative path token-for-token.

    Sampling is temp/top-p/top-k/min-p only - the MTP verify walk exposes no
    stop/penalty/bias hooks (same surface as :func:`generate_speculative`); the
    REPL's other ``/`` sampling controls don't reach this path. The ^T
    finish-thinking key does work on the owned engine, through the round loop's
    forced-close seam rather than a logits processor.
    """
    # The owned walk is the default for the REPL (as it already is for serve
    # and bench): unlike mlx-vlm's stock round it rolls the cache back to
    # exactly the delivered tokens when the consumer closes mid-round (Esc /
    # EOS / a stop string), so the persistent chat cache stays clean for the
    # next turn. GMLX_OWNED_ROUND=0 opts back to the stock round, except for
    # drafters whose contract demands the owned engine.
    from gmlx.spec.speculative import use_owned_engine
    if (os.environ.get("GMLX_OWNED_ROUND") != "0"
            or use_owned_engine(drafter, temp)):
        yield from _stream_generate_speculative_owned(
            model, drafter, tokenizer, prompt, prompt_cache=prompt_cache,
            max_tokens=max_tokens, temp=temp, top_p=top_p, top_k=top_k,
            min_p=min_p, draft_block_size=draft_block_size,
            thinking_budget=thinking_budget,
            thinking_start_token=thinking_start_token,
            thinking_end_token=thinking_end_token,
            start_in_thinking=start_in_thinking,
        )
        return

    if thinking_budget is not None:
        # The forced-close seam lives in the owned rounds only.
        print(
            "warning: thinking budget not applied on the stock MTP engine",
            file=sys.stderr,
        )

    from mlx_lm.sample_utils import make_sampler
    from gmlx.spec.helpers import _resolve_block_total
    from mlx_vlm.generate.ar import generate_step

    if isinstance(prompt, str):
        # Single-BOS rule, identical to mlx_lm.stream_generate + generate_speculative:
        # don't re-add specials when the templated turn already starts with the BOS
        # string, or the post-processor prepends a second one.
        add_special = tokenizer.bos_token is None or not prompt.startswith(
            tokenizer.bos_token
        )
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=add_special)
    else:
        prompt_ids = prompt
    input_ids = mx.array(prompt_ids, dtype=mx.int32)[None]

    sampler = (
        None
        if temp == 0.0
        else make_sampler(temp=temp, top_p=top_p, top_k=top_k, min_p=min_p)
    )
    block = _resolve_block_total(drafter, draft_block_size)
    eos_ids = set(getattr(tokenizer, "eos_token_ids", None) or [tokenizer.eos_token_id])

    detok = tokenizer.detokenizer
    detok.reset()

    # generate_step nulls prefill_step_size once a drafter is attached (it prefills
    # unchunked), so chunk the new turn's tokens into the persistent cache ourselves
    # when they exceed the safe width, then hand generate_step only the trailing token
    # for the drafter-capture forward. Short turns prefill in one already-safe forward.
    feed = input_ids
    if input_ids.shape[1] > _PREFILL_CHUNK:
        _prefill_into_cache(
            model.language_model, input_ids[:, :-1], prompt_cache, _PREFILL_CHUNK
        )
        feed = input_ids[:, -1:]

    n = 0
    # Split prefill from decode at the first token, as generate_speculative does, so
    # the reported tok/s is the steady decode rate (the prefill wall time is excluded).
    n_prompt = input_ids.shape[1]
    prompt_tps = 0.0
    tic = time.perf_counter()
    prefill_done = False
    for tok, _logprobs in generate_step(
        feed,
        model,
        None,
        None,
        prompt_cache=prompt_cache,
        max_tokens=max_tokens,
        temperature=temp,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        sampler=sampler,
        draft_model=drafter,
        draft_kind="mtp",
        draft_block_size=block,
    ):
        if not prefill_done:
            prefill_done = True
            prefill_s = time.perf_counter() - tic
            prompt_tps = n_prompt / prefill_s if prefill_s > 0 else 0.0
            tic = time.perf_counter()
        tok = int(tok[0]) if isinstance(tok, list) else int(tok)
        if tok in eos_ids:
            break
        detok.add_token(tok)
        n += 1
        seg = detok.last_segment
        if seg:
            elapsed = time.perf_counter() - tic
            yield _MTPStreamResponse(
                seg, n, n / elapsed if elapsed > 0 else 0.0, n_prompt, prompt_tps
            )
        if n >= max_tokens:
            break
    detok.finalize()
    tail = detok.last_segment
    elapsed = time.perf_counter() - tic
    yield _MTPStreamResponse(
        tail or "", n, n / elapsed if elapsed > 0 else 0.0, n_prompt, prompt_tps
    )

