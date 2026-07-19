"""Shared helpers for the server-patch modules."""

from __future__ import annotations

import importlib
import inspect
import sys


_PATCH_FLAG = "_kq_gguf_server_patches"


def _install_gen_args_transform(flag: str, transform) -> None:
    """Wrap ``_build_gen_args`` with ``transform(args, request, processor)``.

    The openai/anthropic route modules copy the function into their own
    ``_build_gen_args`` global at registration, so all captured references are
    swapped (plus ``app`` and the ``_protocol_deps`` namespace). Stacks: each
    install wraps the current function and carries earlier patch flags forward,
    so every transform stays idempotent under its own ``flag``."""
    app = importlib.import_module("mlx_vlm.server.app")
    if getattr(app._build_gen_args, flag, False):
        return
    original = app._build_gen_args

    def build_gen_args(request, processor=None, tenant_id=None):
        args = original(request, processor, tenant_id)
        return transform(args, request, processor)

    build_gen_args.__dict__.update(original.__dict__)   # earlier patch flags
    build_gen_args.__dict__[flag] = True
    app._build_gen_args = build_gen_args
    deps = getattr(app, "_protocol_deps", None)
    if deps is not None:
        deps.build_gen_args = build_gen_args
    for modname in ("mlx_vlm.server.openai", "mlx_vlm.server.anthropic"):
        mod = sys.modules.get(modname)
        if mod is not None and getattr(mod, "_build_gen_args", None) is original:
            mod._build_gen_args = build_gen_args


def _get_pool():
    pkg = sys.modules.get("mlx_vlm.server")
    return getattr(pkg, "_kq_residency_pool", None) if pkg else None


def _remove_routes(app, *paths) -> None:
    keep = [r for r in app.router.routes
            if getattr(r, "path", None) not in paths]
    app.router.routes[:] = keep


def _wrap_post_routes(app, paths, flag, make_endpoint) -> None:
    """Replace each existing POST route at ``paths`` with
    ``make_endpoint(original)``, stamping the idempotency ``flag`` and copying
    the stock ``__signature__`` (FastAPI's body injection depends on it).

    No-op for a route that is absent or already wrapped (``flag`` set), so
    installs stay idempotent. ``make_endpoint`` takes the original endpoint and
    returns the bare async replacement; this helper owns the signature/flag copy
    and the remove+re-add, so the boilerplate lives in one place."""
    for path in paths:
        route = next(
            (r for r in app.router.routes
             if getattr(r, "path", None) == path
             and "POST" in (getattr(r, "methods", None) or ())),
            None)
        if route is None or getattr(route.endpoint, flag, False):
            continue
        original = route.endpoint
        endpoint = make_endpoint(original)
        endpoint.__signature__ = inspect.signature(original)
        endpoint.__dict__[flag] = True
        _remove_routes(app, path)
        app.add_api_route(path, endpoint, methods=["POST"],
                          include_in_schema=False)


_CHAT_PATHS = ("/chat/completions", "/v1/chat/completions")


# Anthropic's error taxonomy is fixed by its API; map from HTTP status.
_ANTHROPIC_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    529: "overloaded_error",
}


def _wants_anthropic_error(path: str) -> bool:
    return path.startswith("/v1/messages")


def _error_content(path: str, status: int, err_type: str, message: str,
                   **extra) -> dict:
    """The error body for a request to ``path``, shaped for its dialect.

    OpenAI-style routes get ``{"error": {"type", "message", ...}}``; the
    Anthropic Messages route gets ``{"type": "error", "error": {...}}`` with
    the type drawn from Anthropic's fixed taxonomy. ``extra`` fields (e.g.
    ``available_models``) ride inside the error object in both shapes."""
    if _wants_anthropic_error(path):
        a_type = _ANTHROPIC_ERROR_TYPES.get(
            status, "api_error" if status >= 500 else "invalid_request_error")
        return {"type": "error",
                "error": {"type": a_type, "message": message, **extra}}
    return {"error": {"type": err_type, "message": message, **extra}}
