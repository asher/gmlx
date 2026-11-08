"""Server ``thinking_budget`` on MTP-drafted models.

mlx-vlm's ``ResponseGenerator.generate`` hard-rejects ``thinking_budget``
for any speculative model, and its speculative batch construction drops the
thinking-budget criteria anyway. On gmlx's owned MTP rounds the budget is
enforced by the ``MTPFinishThinking`` hook instead (forced-close verify
rounds, the same seam the CLI ^T/budget path uses), so the server route is:

* defer - ``ResponseGenerator.generate`` moves ``args.thinking_budget``
  aside for MTP-drafted models only, so the upstream raise never fires.
  On a separate drafter (a stock drafter of another kind) a budget from
  the config is dropped with a warning; a client budget was refused with
  a 400 on its route, and the stock error stays as a backstop. The
  pre-generate readers (chat-template kwargs) ran before generate and saw
  the real value.
* restore + build - an outermost wrapper on
  ``_make_thinking_budget_criteria`` (over the thinking-budget-fix
  rebind) puts the value back before delegating, so every later reader
  sees it even when the delegate early-outs or raises, then attaches an
  ``MTPFinishThinking`` to the criteria object the request carries.
* transport - ``PromptProcessingBatch.generate`` moves the hook from the
  criteria row onto the request's prompt cache (request-scoped, the same
  discipline as the APC retirement context) for a single-row batch; the
  owned rounds pop it from there. The batch keeps it as ``_kq_close_hook``,
  so a preempt waits for a close in flight.
* first token - when the prompt's reasoning already spends the budget, the
  close starts on the first token. On the MTP route the hook gives it, and
  only for a row that will not queue behind a running batch: a
  ``BatchGenerator._next`` wrap lets the site read the batch the extend
  will test.

Batched MTP rounds cannot honor a per-request budget: the hook is dropped
with a note, at formation here or at admission in the round loop. The
user-facing rule is under Sampling in docs/config.md.
"""

from __future__ import annotations

import contextvars
import logging
from typing import Any

from gmlx.upstream.patching import link, wrapped_with

from .sampling import separate_drafter_kind

_log = logging.getLogger(__name__)

_DEFER_FLAG = "_kq_mtp_tb_defer"
_CRITERIA_FLAG = "_kq_mtp_tb_criteria"
_TRANSPORT_FLAG = "_kq_mtp_tb_transport"
_QUEUE_PROBE_FLAG = "_kq_mtp_tb_queue_probe"
_DEFERRED_ATTR = "_kq_deferred_thinking_budget"

# The BatchGenerator whose step is running, for the first-token site.
_step_generator: contextvars.ContextVar = contextvars.ContextVar(
    "_kq_mtp_tb_step_generator", default=None)


def _is_mtp(obj) -> bool:
    return (getattr(obj, "draft_model", None) is not None
            and getattr(obj, "draft_kind", None) == "mtp")


class _HookCarrier:
    """Stand-in criteria when the stock builder returns None but an MTP hook
    must still ride the request. ``GenerationBatch._step`` calls both methods
    unconditionally on any non-None criteria entry, so a row that lands on
    the plain batch (drafter disabled, fallback) must not raise."""

    _kq_mtp_hook: Any = None

    def __call__(self, tok):
        return None

    def pop_forced_token_id(self):
        return None

    def first_forced_token_id(self, speculative=False):
        return _hook_first_token(self) if speculative else None

    def rewind(self, delivered):
        hook = self._kq_mtp_hook
        if hook is not None:
            hook.rewind(delivered)


def _will_queue() -> bool:
    """Whether a row this step inserts joins a running generation batch:
    the test ``BatchGenerator._extend_generation_batch`` makes. On an MTP
    batch ``len()`` may promote a waiting batch, as the extend's would."""
    gen = _step_generator.get()
    batch = getattr(gen, "_generation_batch", None) if gen is not None else None
    return batch is not None and len(batch) > 0


def _hook_first_token(crit):
    hook = getattr(crit, "_kq_mtp_hook", None)
    if hook is None or _will_queue():
        return None
    return hook.take_first()


def _route_first_token(crit) -> None:
    """A plain batch asks the criteria for its forced first id; a
    speculative one asks the hook."""
    own = getattr(crit, "first_forced_token_id", None)

    def first_forced_token_id(speculative=False):
        if speculative:
            return _hook_first_token(crit)
        return own(False) if own is not None else None

    crit.first_forced_token_id = first_forced_token_id


def _route_rewind(crit) -> None:
    """A requeued row rewinds the criteria and its hook. The hook is read
    at call time: a multi-row batch may have dropped it."""
    own = getattr(crit, "rewind", None)
    if own is None:
        return

    def rewind(delivered):
        own(delivered)
        hook = getattr(crit, "_kq_mtp_hook", None)
        if hook is not None:
            hook.rewind(delivered)

    crit.rewind = rewind


def _build_hook(rg, args, input_ids):
    from gmlx.gen.thinking_budget import make_mtp_finish_hook, prompt_tail_ids

    from .chat_behavior import _budget_markers, _prompt_opens_thinking_tokens

    try:
        start_in = _prompt_opens_thinking_tokens(rg, args, input_ids)
        pair = _budget_markers(rg, args, input_ids).pair
    except Exception:  # noqa: BLE001
        start_in, pair = False, None
    # The resolved pair, else the raw args values (not the DEFAULT_*
    # fallbacks): the factory's own vocab probe then runs.
    start, end = pair or (getattr(args, "thinking_start_token", None),
                          getattr(args, "thinking_end_token", None))
    return make_mtp_finish_hook(
        rg.tokenizer,
        budget=args.thinking_budget,
        start_in_thinking=start_in,
        start_token=start,
        end_token=end,
        prompt_ids=prompt_tail_ids(input_ids),
        tools=getattr(args, "_kq_has_tools", False),
        message=getattr(args, "thinking_budget_message", None),
        reasoning_ids=input_ids)


def _install_defer(cls) -> None:
    if wrapped_with(cls.generate, _DEFER_FLAG):
        return
    _orig = cls.generate

    def _generate(self, prompt, images=None, audio=None, args=None,
                  videos=None):
        if (args is not None
                and getattr(args, "thinking_budget", None) is not None
                and not getattr(args, "_kq_budget_field", None)
                and separate_drafter_kind(self)):
            if getattr(args, "_kq_off_budget", False):
                # Thinking off on a template with no switch is a budget of
                # 0, which this drafter cannot run. Thinking stays on.
                _log.warning("thinking off ignored: this model has no "
                             "thinking switch, and its drafter cannot run a "
                             "thinking budget")
            else:
                _log.warning("thinking_budget ignored: this model's drafter "
                             "cannot run a thinking budget")
            args.thinking_budget = None
        if (_is_mtp(self) and args is not None
                and getattr(args, "thinking_budget", None) is not None):
            setattr(args, _DEFERRED_ATTR, args.thinking_budget)
            args.thinking_budget = None
        return _orig(self, prompt, images=images, audio=audio, args=args,
                     videos=videos)

    _generate.__dict__.update(_orig.__dict__)
    _generate.__dict__[_DEFER_FLAG] = True
    cls.generate = link(_generate, _orig)


def _install_criteria(cls) -> None:
    if wrapped_with(cls._make_thinking_budget_criteria, _CRITERIA_FLAG):
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
        else:
            _route_first_token(crit)
            _route_rewind(crit)
        crit._kq_mtp_hook = hook
        return crit

    _make.__dict__.update(_orig.__dict__)
    _make.__dict__[_CRITERIA_FLAG] = True
    cls._make_thinking_budget_criteria = link(_make, _orig)


def _lone_hook(ppb):
    """The hook of a one-row MTP prompt batch, or None. Several rows drop
    every hook before the first token, so none forces it."""
    from gmlx.gen.thinking_budget import drop_mtp_hook

    if not _is_mtp(ppb):
        return None
    crits = getattr(ppb, "thinking_budget_criteria", None) or []
    hooks = [h for h in (getattr(c, "_kq_mtp_hook", None) for c in crits)
             if h is not None]
    if not hooks:
        return None
    uids = getattr(ppb, "uids", None)
    n_rows = len(uids) if uids is not None else len(crits)
    if len(crits) == 1 and n_rows == 1:
        return hooks[0]
    for c in crits:
        hook = getattr(c, "_kq_mtp_hook", None)
        if hook is not None:
            drop_mtp_hook(hook, warn=True)
            c._kq_mtp_hook = None
    if n_rows > 1:
        _log.warning(
            "thinking_budget dropped: not applied when MTP requests batch "
            "together (%d rows)", n_rows)
    else:
        _log.warning(
            "thinking_budget dropped: the MTP prompt batch has %d criteria "
            "for %d row(s)", len(crits), n_rows)
    return None


def _stash_hook(ppb, gen_batch, hook) -> None:
    if not _is_mtp(ppb):
        return
    cache = getattr(gen_batch, "prompt_cache", None)
    if hook is not None and not cache:
        hook = None
    if hook is not None:
        cache[0]._kq_mtp_thinking_hook = hook
    # Always set, so a batch merged in by __dict__ never keeps a stale one.
    gen_batch._kq_close_hook = hook


def _install_transport(ppb_cls) -> None:
    if wrapped_with(ppb_cls.generate, _TRANSPORT_FLAG):
        return
    _orig = ppb_cls.generate

    def _generate(self, *args, **kwargs):
        hook = None
        try:
            hook = _lone_hook(self)
        except Exception:  # noqa: BLE001
            _log.exception("mtp thinking-budget drop failed")
        gen_batch = _orig(self, *args, **kwargs)
        try:
            _stash_hook(self, gen_batch, hook)
        except Exception:  # noqa: BLE001
            _log.exception("mtp thinking-budget stash failed")
        return gen_batch

    _generate.__dict__.update(_orig.__dict__)
    _generate.__dict__[_TRANSPORT_FLAG] = True
    ppb_cls.generate = link(_generate, _orig)


def _install_queue_probe(gen_cls) -> None:
    if wrapped_with(gen_cls._next, _QUEUE_PROBE_FLAG):
        return
    _orig = gen_cls._next

    def _next(self, **kwargs):
        token = _step_generator.set(self)
        try:
            return _orig(self, **kwargs)
        finally:
            _step_generator.reset(token)

    _next.__dict__.update(getattr(_orig, "__dict__", {}))
    _next.__dict__[_QUEUE_PROBE_FLAG] = True
    gen_cls._next = link(_next, _orig)


def install_mtp_thinking_budget() -> None:
    """Install the defer / restore+build / transport / queue-probe wraps.
    Idempotent.

    Must run after ``install_full_prompt_mtp_prefill`` (the transport wrap
    has to land outside ``_mtp_generate`` so the hook stash happens after
    the APC L0 store) and after the criteria-seam installer (the thinking
    budget fix) so the restore wrapper is outermost."""
    from gmlx._vendor.mlx_vlm.generate import ar as _ar
    from gmlx._vendor.mlx_vlm.server.generation import ResponseGenerator

    from gmlx.spec.engine import _FULL_PREFILL_FLAG

    if not getattr(_ar.PromptProcessingBatch, _FULL_PREFILL_FLAG, False):
        _log.error(
            "mtp thinking-budget NOT installed: the owned MTP prefill is "
            "missing (install order regression in patches/__init__.py)")
        return

    _install_defer(ResponseGenerator)
    _install_criteria(ResponseGenerator)
    _install_transport(_ar.PromptProcessingBatch)
    _install_queue_probe(_ar.BatchGenerator)
    _log.info("mtp thinking-budget installed")
