"""Prompt ids for context copy on MTP-drafted models.

Context copy (``gmlx.spec.context_copy``) drafts from the request's prompt,
and the speculative batch keeps only the prefill's last chunk of it. The
route mirrors the logits-processor transport:

* capture - ``PromptProcessingBatch.__init__`` indexes each row's full
  prompt (a ``PromptIndex``) when context copy is on for the model: the
  prefill input for a cold row, the APC entry's full ids for a warm row
  whose input is the suffix. The index is built before the prefill, so no
  decode round waits for it.
* build - ``PromptProcessingBatch.generate`` hangs them on the speculative
  batch it returns.
* transport - ``SpeculativeGenerationBatch._start_rounds`` stashes a lone
  row's index on the first cache entry, where the owned scalar rounds pop
  it. A batch of several rows runs no context copy.
"""

from __future__ import annotations

import inspect
import logging

from gmlx.upstream.patching import link, wrapped_with

_log = logging.getLogger(__name__)

_CAPTURE_FLAG = "_kq_copy_prompt_capture"
_BUILD_FLAG = "_kq_copy_prompt_build"
_START_FLAG = "_kq_copy_prompt_start"


def _innermost(fn):
    """The stock function under every wrapper, whose signature names the
    arguments the wrappers pass through."""
    for _ in range(64):
        nxt = (getattr(fn, "_gmlx_orig", None)
               or getattr(fn, "__wrapped__", None))
        if nxt is None:
            return fn
        fn = nxt
    return fn


def _full_prompts(input_ids, apc_meta) -> list:
    from gmlx.spec.context_copy import PromptIndex

    rows = []
    for i, ids in enumerate(input_ids if input_ids is not None else ()):
        meta = apc_meta[i] if apc_meta and i < len(apc_meta) else None
        full = meta.get("full_input_ids") if meta else None
        rows.append(PromptIndex(full if full is not None else ids))
    return rows


def _install_capture(ppb_cls) -> None:
    if wrapped_with(ppb_cls.__init__, _CAPTURE_FLAG):
        return
    _orig = ppb_cls.__init__
    sig = inspect.signature(_innermost(_orig))

    def __init__(self, *args, **kwargs):
        _orig(self, *args, **kwargs)
        from gmlx.spec.context_copy import context_copy_enabled

        if (getattr(self, "draft_kind", None) != "mtp"
                or getattr(self, "draft_model", None) is None
                or not context_copy_enabled(self.draft_model)):
            return
        try:
            bound = sig.bind(self, *args, **kwargs).arguments
            self._kq_copy_prompts = _full_prompts(
                bound.get("input_ids"), bound.get("apc_meta"))
        except Exception:
            _log.debug("context copy: prompt capture failed", exc_info=True)

    ppb_cls.__init__ = link(__init__, _orig, _CAPTURE_FLAG)


def _install_build(ppb_cls, spec_cls) -> None:
    if wrapped_with(ppb_cls.generate, _BUILD_FLAG):
        return
    _orig = ppb_cls.generate

    def _generate(self, *args, **kwargs):
        result = _orig(self, *args, **kwargs)
        prompts = getattr(self, "_kq_copy_prompts", None)
        if prompts and isinstance(result, spec_cls):
            result._kq_copy_prompts = prompts
        return result

    ppb_cls.generate = link(_generate, _orig, _BUILD_FLAG)


def _install_start(spec_cls) -> None:
    if wrapped_with(spec_cls._start_rounds, _START_FLAG):
        return
    _orig = spec_cls._start_rounds

    def _start_rounds(self):
        prompts = getattr(self, "_kq_copy_prompts", None)
        if (prompts and self._rounds_iter is None and self.prompt_cache
                and len(self._all_uids) == 1
                and getattr(self, "hidden", None) is not None):
            self.prompt_cache[0]._kq_copy_prompt = prompts[0]
        return _orig(self)

    spec_cls._start_rounds = link(_start_rounds, _orig, _START_FLAG)


def install_context_copy_transport() -> None:
    """Install the capture / build / transport wraps. Idempotent."""
    from gmlx._vendor.mlx_vlm.generate import ar as _ar

    _install_capture(_ar.PromptProcessingBatch)
    _install_build(_ar.PromptProcessingBatch, _ar.SpeculativeGenerationBatch)
    _install_start(_ar.SpeculativeGenerationBatch)
    _log.info("context copy transport installed")
