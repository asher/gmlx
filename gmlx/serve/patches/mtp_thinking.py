"""Server ``thinking_budget`` on MTP-drafted models.

mlx-vlm's ``ResponseGenerator.generate`` hard-rejects ``thinking_budget``
for any speculative model, and its speculative batch construction drops the
thinking-budget criteria anyway. On gmlx's owned MTP rounds the budget is
enforced by the ``MTPFinishThinking`` hook instead (forced-close verify
rounds, the same seam the CLI ^T/budget path uses), so the server route is:

* defer - ``ResponseGenerator.generate`` moves ``args.thinking_budget``
  aside for MTP-drafted models only, so the upstream raise never fires.
  Non-MTP drafters keep the stock error, and the pre-generate readers
  (chat-template kwargs) ran before generate and saw the real value.
* restore + build - an outermost wrapper on
  ``_make_thinking_budget_criteria`` (over the seed and thinking-budget-fix
  wrappers) puts the value back before delegating, so every later reader
  sees it even when the delegate early-outs or raises, then attaches an
  ``MTPFinishThinking`` to the criteria object the request carries.
* transport - ``PromptProcessingBatch.generate`` moves the hook from the
  criteria row onto the request's prompt cache (request-scoped, the same
  discipline as the APC retirement context) for a single-row batch; the
  owned rounds pop it from there.

Batched MTP rounds cannot honor a per-request budget: the hook is dropped
with a note, at formation here or at admission in the round loop (see the
behavior matrix in docs/server-config.md).
"""

from __future__ import annotations

import logging

_log = logging.getLogger(__name__)

_DEFER_FLAG = "_kq_mtp_tb_defer"
_CRITERIA_FLAG = "_kq_mtp_tb_criteria"
_TRANSPORT_FLAG = "_kq_mtp_tb_transport"
_DEFERRED_ATTR = "_kq_deferred_thinking_budget"


def _is_mtp(obj) -> bool:
    return (getattr(obj, "draft_model", None) is not None
            and getattr(obj, "draft_kind", None) == "mtp")


class _HookCarrier:
    """Stand-in criteria when the stock builder returns None but an MTP hook
    must still ride the request. ``GenerationBatch._step`` calls both methods
    unconditionally on any non-None criteria entry, so a row that lands on
    the plain batch (drafter disabled, fallback) must not raise."""

    _kq_mtp_hook = None

    def __call__(self, tok):
        return None

    def pop_forced_token_id(self):
        return None


def _build_hook(rg, args, input_ids):
    from gmlx.gen.thinking_budget import make_mtp_finish_hook

    from .chat_behavior import _prompt_opens_thinking_tokens

    try:
        start_in = _prompt_opens_thinking_tokens(rg, args, input_ids)
    except Exception:  # noqa: BLE001
        start_in = False
    # Raw args marker values (not the DEFAULT_* fallbacks): the factory's
    # own template/vocab probe resolves non-<think> spellings.
    return make_mtp_finish_hook(
        rg.tokenizer,
        budget=args.thinking_budget,
        start_in_thinking=start_in,
        start_token=getattr(args, "thinking_start_token", None),
        end_token=getattr(args, "thinking_end_token", None))


def _install_defer(cls) -> None:
    if getattr(cls.generate, _DEFER_FLAG, False):
        return
    _orig = cls.generate

    def _generate(self, prompt, images=None, audio=None, args=None,
                  videos=None):
        if (_is_mtp(self) and args is not None
                and getattr(args, "thinking_budget", None) is not None):
            setattr(args, _DEFERRED_ATTR, args.thinking_budget)
            args.thinking_budget = None
        return _orig(self, prompt, images=images, audio=audio, args=args,
                     videos=videos)

    _generate.__dict__.update(_orig.__dict__)
    _generate.__dict__[_DEFER_FLAG] = True
    cls.generate = _generate


def _install_criteria(cls) -> None:
    if getattr(cls._make_thinking_budget_criteria, _CRITERIA_FLAG, False):
        return
    _orig = cls._make_thinking_budget_criteria

    def _make(self, args, input_ids):
        deferred = getattr(args, _DEFERRED_ATTR, None)
        if deferred is not None:
            # Restore before delegating: a delegate early-out or raise must
            # not strand the None the defer wrap wrote.
            args.thinking_budget = deferred
            try:
                delattr(args, _DEFERRED_ATTR)
            except AttributeError:
                pass
        crit = _orig(self, args, input_ids)
        if deferred is None or not _is_mtp(self):
            return crit
        hook = _build_hook(self, args, input_ids)
        if hook is None:
            _log.warning(
                "thinking_budget ignored: no thinking markers resolved for "
                "this model's tokenizer")
            return crit
        if crit is None:
            crit = _HookCarrier()
        crit._kq_mtp_hook = hook
        return crit

    _make.__dict__.update(_orig.__dict__)
    _make.__dict__[_CRITERIA_FLAG] = True
    cls._make_thinking_budget_criteria = _make


def _stash_hook(ppb, gen_batch) -> None:
    if not _is_mtp(ppb):
        return
    crits = getattr(ppb, "thinking_budget_criteria", None) or []
    hooks = [h for h in (getattr(c, "_kq_mtp_hook", None) for c in crits)
             if h is not None]
    if not hooks:
        return
    cache = getattr(gen_batch, "prompt_cache", None)
    uids = getattr(gen_batch, "uids", None)
    n_rows = len(uids) if uids is not None else len(crits)
    if len(crits) == 1 and n_rows == 1 and cache:
        cache[0]._kq_mtp_thinking_hook = hooks[0]
        return
    _log.warning(
        "thinking_budget dropped: not applied when MTP requests batch "
        "together")


def _install_transport(ppb_cls) -> None:
    if getattr(ppb_cls.generate, _TRANSPORT_FLAG, False):
        return
    _orig = ppb_cls.generate

    def _generate(self, *args, **kwargs):
        gen_batch = _orig(self, *args, **kwargs)
        try:
            _stash_hook(self, gen_batch)
        except Exception:  # noqa: BLE001
            _log.exception("mtp thinking-budget stash failed")
        return gen_batch

    _generate.__dict__.update(_orig.__dict__)
    _generate.__dict__[_TRANSPORT_FLAG] = True
    ppb_cls.generate = _generate


def install_mtp_thinking_budget() -> None:
    """Install the defer / restore+build / transport wraps. Idempotent.

    Must run after ``install_full_prompt_mtp_prefill`` (the transport wrap
    has to land outside ``_mtp_generate`` so the hook stash happens after
    the APC L0 store) and after the criteria-seam installers (thinking
    budget fix, per-request seed) so the restore wrapper is outermost."""
    from mlx_vlm.generate import ar as _ar
    from mlx_vlm.server.generation import ResponseGenerator

    from gmlx.spec.engine import _FULL_PREFILL_FLAG

    if not getattr(_ar.PromptProcessingBatch, _FULL_PREFILL_FLAG, False):
        _log.error(
            "mtp thinking-budget NOT installed: the owned MTP prefill is "
            "missing (install order regression in patches/__init__.py)")
        return

    _install_defer(ResponseGenerator)
    _install_criteria(ResponseGenerator)
    _install_transport(_ar.PromptProcessingBatch)
    _log.info("mtp thinking-budget installed")
