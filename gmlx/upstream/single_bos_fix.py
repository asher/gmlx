"""Upstream-bug fix: a rendered prompt that opens with BOS gets a second one.

The Gemma-4, Llama-3, Mistral, Pixtral and DiffusionGemma chat templates
emit BOS, and their GGUF tokenizers add one on encode by default. mlx-vlm
picks ``add_special_tokens`` by model type, not by the prompt: the
server's ``ResponseGenerator._cpu_preprocess`` turns it off only for
gemma3, gemma3n, gemma4 and gemma4_unified with a chat template, and the
generate path's ``should_add_special_tokens`` for those and laguna. A
text-only Gemma-4 GGUF serves as gemma4_text, so its prompt, like a
Llama-3 one, is encoded with two BOS.
On the media path ``process_inputs`` hands the flag only to a processor
that names it as a parameter. Gemma-4's processor takes it through
``**kwargs``, so a Gemma-4 image prompt gets two BOS as well.

The fix applies the rule of mlx-lm's ``stream_generate`` and gmlx's
``encode_prompt`` to the one duplicate token. A prompt that opens with the
tokenizer's BOS string is encoded without special tokens when the only
special token the tokenizer adds to one sequence is a leading BOS
(``_adds_only_bos``). A tokenizer that also adds EOS or another token keeps
the stock flag, so the fix never drops more than the duplicate BOS.

The Laguna chat template writes its own BOS, and its processor names its
model type. When a processor names a type whose markers
``should_add_special_tokens`` leaves to the chat template, a prompt that
opens with the token the tokenizer adds in front is encoded without
special tokens, as on the generate path (``_template_owns_markers``). A
Laguna prompt on the server then holds the template's BOS alone, whatever
its tokenizer adds.

The rule applies where a generation prompt is encoded: the
``prepare_inputs`` bindings of the server's ``_cpu_preprocess`` and of the
generate path, and those of the Messages ``count_tokens`` and Responses
``input_tokens`` routes, whose counts then match the prompt the model
reads. A count also adds the special tokens generation adds: mlx-vlm's
model type rule (``should_add_special_tokens``) for the model the route
resolved, read from the runtime's model cache. Embeddings and reranking
call ``prepare_inputs`` through their own bindings and tokenize as stock.
Inside a call that drops the duplicate, ``process_inputs`` also passes
the flag to a processor of a type known to forward its keyword arguments
to its tokenizer: a stock class in ``_FORWARDS_TO_TOKENIZER`` or a
subclass, or a gmlx class that sets ``KWARGS_REACH_TOKENIZER``.
Idempotent.
"""

from __future__ import annotations

import contextvars
import importlib
import inspect
import sys
import weakref

from gmlx.upstream.patching import wrapped_with

_FLAG = "_gmlx_single_bos"
# Modules whose prepare_inputs binding encodes a rendered chat prompt for
# generation.
_BINDINGS = ("mlx_vlm.server.generation", "mlx_vlm.generate.dispatch")
# Modules whose prepare_inputs binding counts the tokens of a rendered
# generation prompt. Wrapped when the server has imported them.
_COUNT_BINDINGS = ("mlx_vlm.server.anthropic", "mlx_vlm.server.openai")
# Processor classes whose __call__ takes add_special_tokens through
# **kwargs and hands it to self.tokenizer. Subclasses count.
_FORWARDS_TO_TOKENIZER = frozenset({
    "mlx_vlm.models.gemma4.processing_gemma4:Gemma4Processor",
    "mlx_vlm.models.pixtral.processing_pixtral:PixtralProcessor",
})
# Set to True on a gmlx processor class whose __call__ does the same.
KWARGS_REACH_TOKENIZER = "_gmlx_kwargs_reach_tokenizer"
_PROBE = "a"
# True inside a prepare_inputs call that drops the duplicate BOS.
_DROPPING = contextvars.ContextVar("_gmlx_single_bos_dropping", default=False)
_ONLY_BOS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_LEAD: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _tokenizer(processor):
    return getattr(processor, "tokenizer", processor)


def _bos_token(processor) -> str | None:
    """The BOS string of the processor's tokenizer. Reads the special
    tokens map first: transformers logs an error on ``bos_token`` when the
    tokenizer has none."""
    tokenizer = _tokenizer(processor)
    for source in (tokenizer, processor):
        named = getattr(source, "special_tokens_map", None)
        bos = (named.get("bos_token") if isinstance(named, dict)
               else getattr(source, "bos_token", None))
        if isinstance(bos, str) and bos:
            return bos
    return None


def _opens_with(prompts, text: str | None) -> bool:
    texts = [prompts] if isinstance(prompts, str) else prompts
    if not text or not isinstance(texts, (list, tuple)) or not texts:
        return False
    return all(isinstance(t, str) and t.startswith(text) for t in texts)


def _opens_with_bos(processor, prompts) -> bool:
    return _opens_with(prompts, _bos_token(processor))


def _probe(tokenizer) -> tuple[list[int], list[int]] | None:
    """The probe text's ids without and with special tokens, or None."""
    try:
        bare = [int(t) for t in tokenizer.encode(_PROBE,
                                                 add_special_tokens=False)]
        full = [int(t) for t in tokenizer.encode(_PROBE,
                                                 add_special_tokens=True)]
    except Exception:  # noqa: BLE001  an unreadable tokenizer keeps stock
        return None
    return bare, full


def _memo(cache, tokenizer, probe):
    """``probe(tokenizer)``, once per tokenizer."""
    try:
        return cache[tokenizer]
    except (KeyError, TypeError):
        pass
    value = probe(tokenizer)
    try:
        cache[tokenizer] = value
    except TypeError:
        pass
    return value


def _probe_only_bos(tokenizer) -> bool:
    bos_id = getattr(tokenizer, "bos_token_id", None)
    ids = _probe(tokenizer) if isinstance(bos_id, int) else None
    return ids is not None and ids[1] == [bos_id] + ids[0]


def _adds_only_bos(tokenizer) -> bool:
    """Whether the only special token ``tokenizer`` adds to one sequence
    is a leading BOS. Probed once per tokenizer."""
    return _memo(_ONLY_BOS, tokenizer, _probe_only_bos)


def _probe_lead(tokenizer) -> str | None:
    ids = _probe(tokenizer)
    if ids is None:
        return None
    bare, full = ids
    if len(full) <= len(bare) or full[1:len(bare) + 1] != bare:
        return None
    try:
        text = tokenizer.convert_ids_to_tokens(full[0])
    except Exception:  # noqa: BLE001  an unreadable tokenizer keeps stock
        return None
    return text if isinstance(text, str) and text else None


def _template_owns_markers(processor, prompts) -> bool:
    """Whether mlx-vlm's model type rule (``should_add_special_tokens``)
    leaves the special tokens to the chat template for the model type
    ``processor`` names, and ``prompts`` open with the token the tokenizer
    adds in front, named or not. Probed once per tokenizer."""
    model_type = getattr(processor, "model_type", None)
    if not isinstance(model_type, str):
        return False
    utils = importlib.import_module("mlx_vlm.utils")
    if utils.should_add_special_tokens(model_type, processor):
        return False
    return _opens_with(prompts, _memo(_LEAD, _tokenizer(processor),
                                      _probe_lead))


def drops_duplicate_bos(processor, prompts) -> bool:
    """Whether encoding ``prompts`` without special tokens drops exactly
    the BOS the tokenizer would add in front of the template's own."""
    return (_opens_with_bos(processor, prompts)
            and _adds_only_bos(_tokenizer(processor)))


def _forwards_to_tokenizer(processor) -> bool:
    cls = type(processor)
    if getattr(cls, KWARGS_REACH_TOKENIZER, False) is True:
        return True
    return any(f"{base.__module__}:{base.__qualname__}"
               in _FORWARDS_TO_TOKENIZER for base in cls.__mro__)


def _forwarding(processor):
    """A callable with the signature of ``processor``'s process method plus
    ``add_special_tokens``, for a processor of a known forwarding type
    whose method takes the flag only through ``**kwargs``. Else None."""
    if not _forwards_to_tokenizer(processor):
        return None
    method = getattr(processor, "process", processor)
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


def _count_adds_special_tokens(module, processor) -> bool:
    """The ``add_special_tokens`` generation passes for the model a count
    route in ``module`` resolved. The route's ``get_cached_model`` fills
    the runtime's model cache with that model's processor and config.
    True, the rule's default, when the cache holds another processor."""
    cache = getattr(getattr(module, "runtime", None), "model_cache", None)
    get = getattr(cache, "get", None)
    if not callable(get) or get("processor") is not processor:
        return True
    config = get("config")
    model_type = (config.get("model_type") if isinstance(config, dict)
                  else getattr(config, "model_type", None))
    utils = importlib.import_module("mlx_vlm.utils")
    return bool(utils.should_add_special_tokens(model_type, processor))


def _wrap_prepare(original, *, count=None):
    """``count`` is the module of a count route's binding."""
    def prepare_inputs(processor, *args, **kwargs):
        if count is not None and "add_special_tokens" not in kwargs:
            # The count's stock default is off; generation adds the
            # tokenizer's special tokens unless the template owns them.
            kwargs["add_special_tokens"] = _count_adds_special_tokens(
                count, processor)
        prompts = kwargs.get("prompts")
        dropping = drops_duplicate_bos(processor, prompts)
        if dropping or _template_owns_markers(processor, prompts):
            kwargs["add_special_tokens"] = False
        token = _DROPPING.set(dropping)
        try:
            return original(processor, *args, **kwargs)
        finally:
            _DROPPING.reset(token)

    prepare_inputs.__dict__[_FLAG] = True
    prepare_inputs.__wrapped__ = original  # type: ignore[attr-defined]
    return prepare_inputs


def _wrap_process(original):
    def process_inputs(processor, prompts, *args, **kwargs):
        if _DROPPING.get():
            forward = _forwarding(processor)
            if forward is not None:
                processor = forward
        return original(processor, prompts, *args, **kwargs)

    process_inputs.__dict__[_FLAG] = True
    process_inputs.__wrapped__ = original  # type: ignore[attr-defined]
    return process_inputs


def install_single_bos_fix(count_routes: bool = False) -> None:
    """Wrap ``mlx_vlm.utils.process_inputs``, the server and generate
    ``prepare_inputs`` bindings, and the token count bindings of the
    server routes: with ``count_routes`` always, else when the server has
    imported them. Idempotent per binding."""
    utils = importlib.import_module("mlx_vlm.utils")
    if not wrapped_with(utils.process_inputs, _FLAG):
        utils.process_inputs = _wrap_process(utils.process_inputs)
    modules = [(importlib.import_module(name), False) for name in _BINDINGS]
    for name in _COUNT_BINDINGS:
        if count_routes:
            importlib.import_module(name)
        if sys.modules.get(name) is not None:
            modules.append((sys.modules[name], True))
    for module, count in modules:
        if not wrapped_with(module.prepare_inputs, _FLAG):
            module.prepare_inputs = _wrap_prepare(
                module.prepare_inputs, count=module if count else None)
