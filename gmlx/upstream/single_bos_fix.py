"""Upstream-bug fix: a rendered prompt that opens with BOS gets a second one.

The Gemma-4, Llama-3 and DiffusionGemma chat templates emit BOS, and their
GGUF tokenizers add one on encode by default. mlx-vlm picks
``add_special_tokens`` by model type, not by the prompt: the server's
``ResponseGenerator._cpu_preprocess`` and the generate path's
``should_add_special_tokens`` turn it off only for gemma3, gemma3n, gemma4
and gemma4_unified with a chat template. A text-only Gemma-4 GGUF serves as
gemma4_text, so its prompt, like a Llama-3 one, is encoded with two BOS.
On the media path ``process_inputs`` hands the flag only to a processor
that names it as a parameter. Gemma-4's processor takes it through
``**kwargs``, so a Gemma-4 image prompt gets two BOS as well.

The fix applies the rule of mlx-lm's ``stream_generate`` and gmlx's
``encode_prompt``: a prompt that opens with the tokenizer's BOS string is
encoded without special tokens. ``prepare_inputs`` is wrapped where the
server and the generate path bind it, and ``process_inputs`` passes the
flag to a processor that takes it only through ``**kwargs``. Idempotent.
"""

from __future__ import annotations

import importlib
import inspect

_FLAG = "_gmlx_single_bos"
# Modules whose prepare_inputs binding encodes a rendered chat prompt.
_BINDINGS = ("mlx_vlm.server.generation", "mlx_vlm.generate.dispatch")


def _bos_token(processor) -> str | None:
    """The BOS string of the processor's tokenizer. Reads the special
    tokens map first: transformers logs an error on ``bos_token`` when the
    tokenizer has none."""
    tokenizer = getattr(processor, "tokenizer", processor)
    for source in (tokenizer, processor):
        named = getattr(source, "special_tokens_map", None)
        bos = (named.get("bos_token") if isinstance(named, dict)
               else getattr(source, "bos_token", None))
        if isinstance(bos, str) and bos:
            return bos
    return None


def _opens_with_bos(processor, prompts) -> bool:
    texts = [prompts] if isinstance(prompts, str) else prompts
    if not isinstance(texts, (list, tuple)) or not texts:
        return False
    bos = _bos_token(processor)
    return bos is not None and all(
        isinstance(t, str) and t.startswith(bos) for t in texts)


def _forwarding(method):
    """A callable with ``method``'s signature plus ``add_special_tokens``,
    or None when ``method`` names the flag or takes no ``**kwargs``."""
    try:
        sig = inspect.signature(method)
    except (TypeError, ValueError):
        return None
    params = list(sig.parameters.values())
    if ("add_special_tokens" in sig.parameters or not params
            or params[-1].kind is not inspect.Parameter.VAR_KEYWORD):
        return None
    flag = inspect.Parameter("add_special_tokens",
                             inspect.Parameter.KEYWORD_ONLY, default=True)

    def call(*args, **kwargs):
        return method(*args, **kwargs)

    call.__signature__ = sig.replace(  # type: ignore[attr-defined]
        parameters=params[:-1] + [flag, params[-1]])
    return call


def _wrap_prepare(original):
    def prepare_inputs(processor, *args, **kwargs):
        if _opens_with_bos(processor, kwargs.get("prompts")):
            kwargs["add_special_tokens"] = False
        return original(processor, *args, **kwargs)

    prepare_inputs.__dict__[_FLAG] = True
    prepare_inputs.__wrapped__ = original  # type: ignore[attr-defined]
    return prepare_inputs


def _wrap_process(original):
    def process_inputs(processor, prompts, *args, **kwargs):
        if _opens_with_bos(processor, prompts):
            kwargs["add_special_tokens"] = False
            forward = _forwarding(getattr(processor, "process", processor))
            if forward is not None:
                processor = forward
        return original(processor, prompts, *args, **kwargs)

    process_inputs.__dict__[_FLAG] = True
    process_inputs.__wrapped__ = original  # type: ignore[attr-defined]
    return process_inputs


def install_single_bos_fix() -> None:
    """Wrap ``mlx_vlm.utils.process_inputs`` and the server and generate
    ``prepare_inputs`` bindings. Idempotent per binding."""
    utils = importlib.import_module("mlx_vlm.utils")
    if not getattr(utils.process_inputs, _FLAG, False):
        utils.process_inputs = _wrap_process(utils.process_inputs)
    for name in _BINDINGS:
        module = importlib.import_module(name)
        if not getattr(module.prepare_inputs, _FLAG, False):
            module.prepare_inputs = _wrap_prepare(module.prepare_inputs)
