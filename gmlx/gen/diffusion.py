"""Drive gmlx's text surfaces (``run`` / ``chat``) through the mlx-vlm
block-diffusion denoiser.

A DiffusionGemma checkpoint is an mlx-vlm model, not an mlx-lm one, so mlx-lm's
autoregressive ``generate`` / ``stream_generate`` can't drive it - it has no KV
loop, it denoises a fixed-length canvas over reverse-diffusion steps. mlx-vlm's
``stream_diffusion_generate`` does the denoising, but wants a ``processor``
exposing a copy-safe streaming detokenizer and a ``tokenizer`` exposing
``stopping_criteria`` + ``decode``. The server path already adapts a synthesized
GGUF tokenizer into exactly that shape (``server_bridge_vlm._make_text_processor``),
so we reuse it here and expose a ``stream_generate``-shaped generator the
``run`` / ``chat`` paths can call.

The ``serve`` path needs nothing from this module: mlx-vlm's batching server has
its own diffusion lane and is handed the same engine-ready processor.
"""

from __future__ import annotations

import mlx.core as mx


def is_diffusion_model(model) -> bool:
    """True if ``model`` denoises a canvas (mlx-vlm diffusion family) rather than
    decoding autoregressively. Falls back to the ``canvas_length`` config marker
    mlx-vlm's own dispatch keys on."""
    try:
        from mlx_vlm.generate.diffusion import is_diffusion_model as _is

        return bool(_is(model))
    except Exception:
        cfg = getattr(model, "config", None)
        return getattr(cfg, "canvas_length", None) is not None


def _diffusion_io(tokenizer):
    """Build the (processor, engine-tokenizer, skip-ids) triple the denoiser
    needs from a gmlx ``TokenizerWrapper`` - the same adapter the server
    uses, so run/chat/serve share one tokenizer seam."""
    from .server_bridge_vlm import _make_text_processor

    processor = _make_text_processor(tokenizer)
    backend = processor.tokenizer  # callable HF tokenizer carrying stopping_criteria
    skip_ids = set(getattr(backend, "all_special_ids", None) or [])
    if not skip_ids:
        skip_ids = set(getattr(tokenizer, "eos_token_ids", []) or [])
    return processor, backend, skip_ids


def stream(model, tokenizer, prompt, *, max_tokens: int = 256, **_ignored):
    """Yield the denoiser's ``GenerationResult`` chunks (each carries ``.text``
    and, on the last, ``.finish_reason``). ``prompt`` is a token-id list or a
    string. Autoregressive sampler controls (temperature, top-p, penalties,
    logit bias, stop strings) don't apply to the entropy-bound diffusion sampler
    and are ignored."""
    from mlx_vlm.generate.diffusion import stream_diffusion_generate

    processor, backend, skip_ids = _diffusion_io(tokenizer)
    ids = tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
    input_ids = mx.array(ids)[None]
    yield from stream_diffusion_generate(
        model,
        processor,
        backend,
        input_ids,
        None,
        None,
        max_tokens=max_tokens,
        skip_special_token_ids=skip_ids,
    )


def generate(
    model, tokenizer, prompt, *, max_tokens: int = 256, verbose: bool = False, **kw
) -> str:
    """Collect the full denoised completion as a string (the ``run`` path)."""
    parts: list[str] = []
    for r in stream(model, tokenizer, prompt, max_tokens=max_tokens, **kw):
        if r.text:
            parts.append(r.text)
            if verbose:
                print(r.text, end="", flush=True)
    if verbose:
        print()
    return "".join(parts)
