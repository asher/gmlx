"""Server ``response_format`` on MTP-drafted models.

mlx-vlm's ``ResponseGenerator.generate`` refuses request logits processors
(the structured-output grammar) for any speculative model, and its
speculative batch drops them after the first token. gmlx's owned MTP rounds
mask every verify position with the grammar instead (``gmlx.spec.grammar``),
so the server route is:

* defer - ``generate`` moves ``args.logits_processors`` aside for MTP
  models only, so the upstream refusal never fires. The engine thread's
  ``_make_logits_processors`` puts the value back before it builds the
  row's processors, and ``generate`` puts it back on return for the
  routes that read it afterwards.
* build - ``PromptProcessingBatch.generate`` reads each row's processors
  before the stock prefill clears them (the prefill applies them to the
  first token) and hangs a ``SpecGrammar`` per row on the speculative
  batch it returns.
* transport - ``SpeculativeGenerationBatch._start_rounds`` stashes the
  list on the first cache entry, where the owned rounds pop it, the same
  request-scoped discipline as the thinking-budget hook. A preempt
  rebuild starts the rounds again and stashes the same grammars, and
  continuous-batch admission carries an injected batch's list in its
  injection entry.

Non-MTP drafters keep the stock refusal.
"""

from __future__ import annotations

import logging

_log = logging.getLogger(__name__)

_DEFER_FLAG = "_kq_spec_grammar_defer"
_RESTORE_FLAG = "_kq_spec_grammar_restore"
_BUILD_FLAG = "_kq_spec_grammar_build"
_START_FLAG = "_kq_spec_grammar_start"
_DEFERRED_ATTR = "_kq_deferred_logits_processors"


def _is_mtp(obj) -> bool:
    return (getattr(obj, "draft_model", None) is not None
            and getattr(obj, "draft_kind", None) == "mtp")


def _restore(args) -> None:
    deferred = getattr(args, _DEFERRED_ATTR, None)
    if deferred is None:
        return
    args.logits_processors = deferred
    try:
        delattr(args, _DEFERRED_ATTR)
    except AttributeError:
        pass


def _install_defer(cls) -> None:
    if getattr(cls.generate, _DEFER_FLAG, False):
        return
    _orig = cls.generate

    def _generate(self, prompt, images=None, audio=None, args=None,
                  videos=None):
        if (_is_mtp(self) and args is not None
                and getattr(args, "logits_processors", None) is not None):
            setattr(args, _DEFERRED_ATTR, args.logits_processors)
            args.logits_processors = None
        try:
            return _orig(self, prompt, images=images, audio=audio, args=args,
                         videos=videos)
        finally:
            if args is not None:
                _restore(args)

    _generate.__dict__.update(_orig.__dict__)
    _generate.__dict__[_DEFER_FLAG] = True
    cls.generate = _generate


def _install_restore(cls) -> None:
    if getattr(cls._make_logits_processors, _RESTORE_FLAG, False):
        return
    _orig = cls._make_logits_processors

    def _make(self, args, *rest, **kwargs):
        _restore(args)
        return _orig(self, args, *rest, **kwargs)

    _make.__dict__.update(_orig.__dict__)
    _make.__dict__[_RESTORE_FLAG] = True
    cls._make_logits_processors = _make


def _row_grammars(procs, n_rows: int) -> list | None:
    from gmlx.spec.grammar import SpecGrammar

    rows = list(procs or ())[:n_rows]
    rows += [None] * (n_rows - len(rows))
    grammars = [SpecGrammar.from_processors(p) if p else None for p in rows]
    return grammars if any(g is not None for g in grammars) else None


def _install_build(ppb_cls, spec_cls) -> None:
    if getattr(ppb_cls.generate, _BUILD_FLAG, False):
        return
    _orig = ppb_cls.generate

    def _generate(self, *args, **kwargs):
        procs = list(getattr(self, "logits_processors", None) or ())
        n_rows = len(getattr(self, "uids", None) or ())
        result = _orig(self, *args, **kwargs)
        if (isinstance(result, spec_cls)
                and getattr(result, "draft_kind", None) == "mtp"
                and any(procs)):
            result._kq_grammars = _row_grammars(procs, n_rows)
        return result

    _generate.__dict__.update(_orig.__dict__)
    _generate.__dict__[_BUILD_FLAG] = True
    ppb_cls.generate = _generate


def _install_start(spec_cls) -> None:
    if getattr(spec_cls._start_rounds, _START_FLAG, False):
        return
    _orig = spec_cls._start_rounds

    def _start_rounds(self):
        grammars = getattr(self, "_kq_grammars", None)
        if (grammars and self._rounds_iter is None and self.prompt_cache
                and getattr(self, "draft_kind", None) == "mtp"):
            self.prompt_cache[0]._kq_spec_grammars = list(grammars)
        return _orig(self)

    _start_rounds.__dict__.update(_orig.__dict__)
    _start_rounds.__dict__[_START_FLAG] = True
    spec_cls._start_rounds = _start_rounds


def install_mtp_structured_output() -> None:
    """Install the defer / restore / build / transport wraps. Idempotent.

    Must run after the other ``_make_logits_processors`` and
    ``PromptProcessingBatch.generate`` wrappers (seed rows, LoRA rows, the
    owned MTP prefill), so the restore runs first and the build reads the
    processors before any inner wrapper clears them."""
    from mlx_vlm.generate import ar as _ar
    from mlx_vlm.server.generation import ResponseGenerator

    _install_defer(ResponseGenerator)
    _install_restore(ResponseGenerator)
    _install_build(_ar.PromptProcessingBatch, _ar.SpeculativeGenerationBatch)
    _install_start(_ar.SpeculativeGenerationBatch)
    _log.info("mtp structured output installed")
