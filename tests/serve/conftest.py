"""Shared serve-suite fixtures.

The seam snapshot/restore fixture guards every test in this directory;
it is a no-op when mlx-vlm is not installed."""
from __future__ import annotations

import importlib
import sys

import pytest

try:
    import gmlx.serve.bridge_vlm as serving
    from gmlx.serve.patches import _common as sp_common
    from gmlx.serve.patches import hardening as sp_hardening
    _APP = importlib.import_module("mlx_vlm.server.app")
    _UTILS = importlib.import_module("mlx_vlm.utils")
    _PKG = importlib.import_module("mlx_vlm.server")
except ImportError:
    _APP = None


_PATCH_MODULES = {
    "test_server_patches",
    "test_patches_chat_behavior",
    "test_patches_routes",
    "test_patches_sampling",
}


@pytest.fixture(autouse=True)
def _restore_mlxvlm(request):
    if _APP is None or request.module.__name__ not in _PATCH_MODULES:
        yield
        return
    """Snapshot every mlx-vlm seam these patches mutate, restore after each test."""
    fastapi_app = _APP.app
    saved = {
        "build_gen_args": _APP._build_gen_args,
        "snapshot": _APP._server_runtime_snapshot,
        "get_model_path": _UTILS.get_model_path,
        "routes": sp_common._snapshot_routes(fastapi_app),
        "handlers": dict(fastapi_app.exception_handlers),
        "deps": getattr(getattr(_APP, "_protocol_deps", None), "build_gen_args", None),
        "pool": getattr(_PKG, "_kq_residency_pool", None),
        # Middleware installs (auth / host guard) append to user_middleware and
        # set app.state flags; CORS hardening mutates a Middleware's kwargs in
        # place - snapshot all three or one test's auth poisons the rest.
        "middleware": list(fastapi_app.user_middleware),
        "mw_kwargs": [(m, dict(getattr(m, "kwargs", {}) or {}))
                      for m in fastapi_app.user_middleware],
    }
    openai = sys.modules.get("mlx_vlm.server.openai")
    anthropic = sys.modules.get("mlx_vlm.server.anthropic")
    saved["openai_bga"] = getattr(openai, "_build_gen_args", None)
    saved["anthropic_bga"] = getattr(anthropic, "_build_gen_args", None)
    apc = sys.modules.get("mlx_vlm.apc") or importlib.import_module("mlx_vlm.apc")
    saved["apc_harvest"] = apc.harvest_blocks_from_batch_cache
    saved["apc_lone_flag"] = getattr(apc, "_kq_lone_harvest", False)
    gen = importlib.import_module("mlx_vlm.server.generation")
    saved["to_template_kwargs"] = gen.GenerationArguments.to_template_kwargs
    pu = importlib.import_module("mlx_vlm.prompt_utils")
    saved["get_chat_template"] = pu.get_chat_template
    saved["make_sampler"] = gen.ResponseGenerator._make_sampler
    saved["make_tb_criteria"] = gen.ResponseGenerator._make_thinking_budget_criteria
    schemas = importlib.import_module("mlx_vlm.server.schemas")
    saved["stream_chunk_dump"] = schemas.ChatStreamChunk.model_dump_json
    saved["stopping_call"] = _UTILS.StoppingCriteria.__call__
    serving.clear_resolved_models()
    yield
    _UTILS.StoppingCriteria.__call__ = saved["stopping_call"]
    apc.harvest_blocks_from_batch_cache = saved["apc_harvest"]
    apc._kq_lone_harvest = saved["apc_lone_flag"]
    gen.GenerationArguments.to_template_kwargs = saved["to_template_kwargs"]
    pu.get_chat_template = saved["get_chat_template"]
    gen.ResponseGenerator._make_sampler = saved["make_sampler"]
    gen.ResponseGenerator._make_thinking_budget_criteria = saved["make_tb_criteria"]
    schemas.ChatStreamChunk.model_dump_json = saved["stream_chunk_dump"]
    _APP._build_gen_args = saved["build_gen_args"]
    _APP._server_runtime_snapshot = saved["snapshot"]
    _UTILS.get_model_path = saved["get_model_path"]
    sp_common._restore_routes(fastapi_app, saved["routes"])
    fastapi_app.exception_handlers.clear()
    fastapi_app.exception_handlers.update(saved["handlers"])
    if getattr(_APP, "_protocol_deps", None) is not None and saved["deps"] is not None:
        _APP._protocol_deps.build_gen_args = saved["deps"]
    if openai is not None:
        openai._build_gen_args = saved["openai_bga"]
    if anthropic is not None:
        anthropic._build_gen_args = saved["anthropic_bga"]
    if saved["pool"] is None:
        if hasattr(_PKG, "_kq_residency_pool"):
            delattr(_PKG, "_kq_residency_pool")
    else:
        _PKG._kq_residency_pool = saved["pool"]
    fastapi_app.user_middleware[:] = saved["middleware"]
    for m, kw in saved["mw_kwargs"]:
        if getattr(m, "kwargs", None) is not None:
            m.kwargs.clear()
            m.kwargs.update(kw)
    fastapi_app.middleware_stack = None       # force a rebuild from the restored list
    for flag in (sp_hardening._AUTH_FLAG, sp_hardening._HOST_GUARD_FLAG,
                 sp_hardening._JSON_CT_FLAG):
        if hasattr(fastapi_app.state, flag):
            delattr(fastapi_app.state, flag)
    serving.clear_resolved_models()
