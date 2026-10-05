"""APC (prefix-cache) engine patches: lone-sequence harvest and retirement
render capture. The batched store eval lives on the gmlx-owned manager
subclass (``gmlx.cache.apc_manager.GmlxAPCManager``), not as a method patch here."""

from __future__ import annotations

import contextvars
import importlib
import logging
import weakref

from gmlx.upstream.patching import link, wrapped_with

_log = logging.getLogger(__name__)
_CAPTURE_WARNED: set[str] = set()

# The request's ``tool_parser`` override (None: infer from the template),
# recorded where each protocol handler resolves its parser, ahead of the
# render the capture records.
_REQUEST_TOOL_PARSER: contextvars.ContextVar[str | None] = \
    contextvars.ContextVar("gmlx_request_tool_parser", default=None)


def _warn_capture_once(site: str) -> None:
    if site in _CAPTURE_WARNED:
        return
    _CAPTURE_WARNED.add(site)
    _log.warning("retirement %s capture failed; next-turn retirement keys "
                 "are off for such requests", site, exc_info=True)


def install_apc_lone_harvest() -> None:
    """Teach mlx-vlm's APC block harvest to read a lone request's plain ``KVCache``.

    The continuous-batching engine harvests prefill K/V into APC prefix blocks via
    ``apc.harvest_blocks_from_batch_cache``, which slices a row out of a batched
    cache by its ``_idx`` attribute. A single, non-concurrent request takes a fast
    path that builds a plain ``KVCache`` - which exposes ``offset`` (valid length),
    not ``_idx`` - so the stock harvest hits its ``idx is None`` guard, returns
    ``[]``, and the request stores nothing in memory or on the SSD tier. The result
    is that a purely sequential single-client session (the common coding-assistant
    case) never populates the prefix cache, while concurrent traffic does.

    This generalizes the harvest: when ``_idx`` is absent, fall back to ``offset``
    over the single row (no left padding), which is exactly the slice the stock
    code would take on a batched cache. An affine cache (``--kv-bits``) keeps
    ``keys`` as a tuple, which neither path can slice: it goes through stock
    ``layer_kv_for_apc``, which dequantizes it, so affine keeps block reuse.
    A batch row is extracted first, so only that row is dequantized. ``ar.py``
    calls the function by module attribute
    (``_apc.harvest_blocks_from_batch_cache``), so replacing it on the module
    is picked up at the call site without a fork.

    Signature contract: matches the 0.7.3 harvest exactly --
    ``full_token_ids`` third positional, ``batch_idx`` keyword-only
    (None = lone request). All callers use that convention."""
    import mlx.core as mx

    apc = importlib.import_module("gmlx._vendor.mlx_vlm.apc")
    if getattr(apc, "_kq_lone_harvest", False):
        return

    def harvest_blocks_from_batch_cache(
        apc_manager, batch_caches, full_token_ids,
        *, batch_idx=None, extra_hash=0, skip_first_n_tokens=0):
        row = 0 if batch_idx is None else int(batch_idx)
        layer_keys = []
        layer_values = []
        for c in batch_caches:
            keys = getattr(c, "keys", None)
            values = getattr(c, "values", None)
            if keys is None or values is None:
                return []
            idx = getattr(c, "_idx", None)
            if isinstance(keys, tuple):
                # Affine cache: dequantize as stock does, one row only.
                if idx is not None and hasattr(c, "extract"):
                    k, v = apc.layer_kv_for_apc(c.extract(row))
                else:
                    k, v = apc.layer_kv_for_apc(
                        c, batch_idx=None if idx is None else row)
                if k is None or v is None:
                    return []
                layer_keys.append(k)
                layer_values.append(v)
                continue
            left_padding = getattr(c, "left_padding", None)
            if idx is None:
                # Lone-request fast path: a plain KVCache has a scalar `offset`
                # and no `_idx`/`left_padding`. Harvest its one row over
                # [0, offset).
                offset = getattr(c, "offset", None)
                if offset is None or not isinstance(keys, mx.array):
                    return []
                idx = int(offset)
                left_padding = None
            if left_padding is not None:
                try:
                    lp = int(left_padding[row].item())
                except Exception:
                    lp = 0
            else:
                lp = 0
            layer_keys.append(keys[row:row + 1, :, lp:idx, :])
            layer_values.append(values[row:row + 1, :, lp:idx, :])
        return apc_manager.store_kv_blocks(
            full_token_ids, layer_keys, layer_values,
            extra_hash=extra_hash, skip_first_n_tokens=skip_first_n_tokens)

    apc.harvest_blocks_from_batch_cache = harvest_blocks_from_batch_cache
    apc._kq_lone_harvest = True


def install_retire_render_capture() -> None:
    """Record each chat request's render context for next-turn retirement keys.

    Two hops, both provenance: the openai module's ``apply_chat_template``
    records (processed messages, template kwargs, processor, config) keyed
    by the rendered prompt text, and ``ResponseGenerator._preprocess_request``
    re-keys the entry by the exact token ids it produces -- the same ids the
    spec engine later stashes as ``full_ids`` -- while capturing the
    generator's own text-to-ids path so the retirement render tokenizes
    identically. Media requests are not re-keyed (re-encoding text cannot
    reproduce expanded media token ids). The recorded context also carries
    the request's ``tool_parser`` override, so the retirement parses tool
    calls with the handler's parser. Consumed by speculative retirement
    via ``retire_key.next_turn_lcp``. Kill: ``GMLX_APC_RETIRE_LCP=0``.
    """
    import os

    if os.environ.get("GMLX_APC_RETIRE_LCP") == "0":
        return
    import gmlx.cache.retire_key as retire_key

    from ._common import _render_target_modules
    gen_mod = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")

    def _make_capture(orig_render):
        def apply_chat_template(processor, config, prompt, *a, **kw):
            out = orig_render(processor, config, prompt, *a, **kw)
            try:
                if isinstance(out, str) and isinstance(prompt, list):
                    retire_key.register_render(out, {
                        "messages": [dict(m) for m in prompt
                                     if isinstance(m, dict)],
                        "kw": dict(kw),
                        "processor": processor,
                        "config": config,
                        "render": orig_render,
                        "tool_parser": _REQUEST_TOOL_PARSER.get(),
                        "media": bool(kw.get("num_images")
                                      or kw.get("num_audios")
                                      or kw.get("video")),
                    })
            except Exception:
                _warn_capture_once("render")
            return out

        apply_chat_template._kq_retire_capture = True
        return apply_chat_template

    # Every captured binding (openai, anthropic, _protocol_deps), so
    # retirement keys cover all protocol routes, not just OpenAI.
    for target in _render_target_modules():
        fn = getattr(target, "apply_chat_template", None)
        if fn is not None and not wrapped_with(fn, "_kq_retire_capture"):
            target.apply_chat_template = _make_capture(fn)

    # The handlers resolve their parser through the package attribute
    # (``_protocol_deps`` looks it up at call time), before they render.
    pkg = importlib.import_module("gmlx._vendor.mlx_vlm.server")
    infer = getattr(pkg, "_infer_tool_parser_from_processor", None)
    if infer is not None and not wrapped_with(infer, "_kq_retire_capture"):
        def _infer_tool_parser_from_processor(processor, override=None):
            _REQUEST_TOOL_PARSER.set(override)
            return infer(processor, override=override)

        _infer_tool_parser_from_processor._kq_retire_capture = True
        pkg._infer_tool_parser_from_processor = _infer_tool_parser_from_processor

    cls = gen_mod.ResponseGenerator
    orig_pre = cls._preprocess_request
    if wrapped_with(orig_pre, "_kq_retire_capture"):
        return

    def _preprocess_request(self, prompt, images=None, audio=None,
                            videos=None):
        raw = orig_pre(self, prompt, images, audio, videos)
        try:
            if isinstance(prompt, str) and not images and not audio \
                    and not videos and isinstance(raw, dict):
                ids = raw.get("input_ids")
                if ids is not None:
                    row = ids.tolist() if hasattr(ids, "tolist") else list(ids)
                    if row and isinstance(row[0], list):
                        row = row[0]
                    # Weak: a memo entry must never pin the generator (and
                    # through it the model weights) past a pool unload.
                    ref = weakref.ref(self)

                    def preprocess(text, _ref=ref):
                        gen = _ref()
                        if gen is None:
                            raise RuntimeError(
                                "retire render: generator unloaded")
                        return orig_pre(gen, text, None, None, None)

                    retire_key.register_ids(prompt, row, preprocess)
        except Exception:
            _warn_capture_once("ids")
        return raw

    _preprocess_request._kq_retire_capture = True
    cls._preprocess_request = link(_preprocess_request, orig_pre)
