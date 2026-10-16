"""Shared serve-suite fixtures.

The seam snapshot/restore fixtures guard every test in this directory;
they are no-ops when mlx-vlm is not installed."""
from __future__ import annotations

import importlib
import sys
import types

import pytest

try:
    import gmlx.lora_rows as lora_rows
    from gmlx.cache import media_salt
    import gmlx.serve.bridge_vlm as serving
    from gmlx.serve.patches import _common as sp_common
    from gmlx.serve.patches import hardening as sp_hardening
    from gmlx.serve.patches import media_gate
    from gmlx.serve.patches import routes as sp_routes
    from gmlx.serve import media_sinks
    from gmlx.serve import server as srv
    _APP = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")
    _UTILS = importlib.import_module("gmlx._vendor.mlx_vlm.utils")
    _PKG = importlib.import_module("gmlx._vendor.mlx_vlm.server")
except ImportError:
    _APP = None


_PATCH_MODULES = {
    "test_server_patches",
    "test_patches_chat_behavior",
    "test_patches_routes",
    "test_image_routes",
    "test_image_steps",
    "test_patches_sampling",
}


@pytest.fixture(autouse=True, scope="module")
def _restore_batch_next():
    """The tick installers wrap BatchGenerator._next, and a test module
    that runs one leaves its layer for every later test in the process."""
    try:
        ar = importlib.import_module("gmlx._vendor.mlx_vlm.generate.ar")
    except ImportError:
        yield
        return
    saved = ar.BatchGenerator._next
    yield
    ar.BatchGenerator._next = saved


@pytest.fixture(autouse=True, scope="module")
def _restore_render():
    """install_faithful_history wraps the prompt_utils render functions,
    and a test module that runs it leaves its layers for every later test
    in the process."""
    try:
        pu = importlib.import_module("gmlx._vendor.mlx_vlm.prompt_utils")
    except ImportError:
        yield
        return
    saved = (pu.get_chat_template, pu.apply_chat_template)
    yield
    pu.get_chat_template, pu.apply_chat_template = saved


def _engine_classes() -> list:
    """The mlx-vlm engine classes that the server patches wrap."""
    ar = importlib.import_module("gmlx._vendor.mlx_vlm.generate.ar")
    gen = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
    names = ("BatchGenerator", "GenerationBatch", "PromptProcessingBatch",
             "SpeculativeGenerationBatch")
    return [getattr(ar, n) for n in names if hasattr(ar, n)] + [gen.ResponseGenerator]


def _restore_class(cls, attrs: dict) -> None:
    """Give ``cls`` the attributes in ``attrs`` again, its install flags too."""
    for name in [n for n in vars(cls) if n not in attrs]:
        delattr(cls, name)
    for name, value in attrs.items():
        if vars(cls).get(name) is not value:
            setattr(cls, name, value)


_UNSET = object()


def _snapshot_engine_seams() -> dict:
    """The engine classes, and each process flag that stops a second
    install of a patch on them."""
    # The LoRA row channel wraps methods of several engine classes, and a
    # process flag stops a second install. Restore the classes and the flag
    # together, or a later install finds the flag set and does nothing.
    # The media salt wraps three methods of the engine classes and two apc
    # functions, and keeps its flag on the apc module. Restore them
    # together for the same reason.
    apc = importlib.import_module("gmlx._vendor.mlx_vlm.apc")
    salt = ("hash_image_payload", "semantic_extra_hash",
            media_salt._INSTALLED_FLAG)
    return {"classes": [(cls, dict(vars(cls))) for cls in _engine_classes()],
            "lora_channel": lora_rows._channel_installed,
            "media_salt": {n: vars(apc).get(n, _UNSET) for n in salt}}


def _restore_engine_seams(saved: dict) -> None:
    for cls, attrs in saved["classes"]:
        _restore_class(cls, attrs)
    lora_rows._channel_installed = saved["lora_channel"]
    apc = importlib.import_module("gmlx._vendor.mlx_vlm.apc")
    for name, value in saved["media_salt"].items():
        if value is _UNSET:
            vars(apc).pop(name, None)
        else:
            setattr(apc, name, value)


@pytest.fixture(scope="module")
def module_engine_seams():
    """Restore the engine seams when the module ends. For a module whose
    fixture installs the server patches once for all of its tests."""
    if _APP is None:
        yield
        return
    saved = _snapshot_engine_seams()
    yield
    _restore_engine_seams(saved)


@pytest.fixture
def engine_seams():
    """The snapshot and restore of the engine seams that ``_restore_mlxvlm``
    runs around each test of the patch modules."""
    return types.SimpleNamespace(snapshot=_snapshot_engine_seams,
                                 restore=_restore_engine_seams)


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
    openai = sys.modules.get("gmlx._vendor.mlx_vlm.server.openai")
    anthropic = sys.modules.get("gmlx._vendor.mlx_vlm.server.anthropic")
    saved["openai_bga"] = getattr(openai, "_build_gen_args", None)
    saved["anthropic_bga"] = getattr(anthropic, "_build_gen_args", None)
    apc = sys.modules.get("gmlx._vendor.mlx_vlm.apc") or importlib.import_module("gmlx._vendor.mlx_vlm.apc")
    saved["apc_harvest"] = apc.harvest_blocks_from_batch_cache
    saved["apc_lone_flag"] = getattr(apc, "_kq_lone_harvest", False)
    gen = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
    saved["to_template_kwargs"] = gen.GenerationArguments.to_template_kwargs
    pu = importlib.import_module("gmlx._vendor.mlx_vlm.prompt_utils")
    saved["get_chat_template"] = pu.get_chat_template
    saved["engine_seams"] = _snapshot_engine_seams()
    schemas = importlib.import_module("gmlx._vendor.mlx_vlm.server.schemas")
    saved["stream_chunk_dump"] = schemas.ChatStreamChunk.model_dump_json
    saved["stopping_call"] = _UTILS.StoppingCriteria.__call__
    serving.clear_resolved_models()
    yield
    _UTILS.StoppingCriteria.__call__ = saved["stopping_call"]
    apc.harvest_blocks_from_batch_cache = saved["apc_harvest"]
    apc._kq_lone_harvest = saved["apc_lone_flag"]
    gen.GenerationArguments.to_template_kwargs = saved["to_template_kwargs"]
    pu.get_chat_template = saved["get_chat_template"]
    _restore_engine_seams(saved["engine_seams"])
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
                 sp_hardening._JSON_CT_FLAG, media_gate._FLAG,
                 sp_routes._UNCONFIGURED_FLAG):
        if hasattr(fastapi_app.state, flag):
            delattr(fastapi_app.state, flag)
    media_sinks.uninstall()
    media_gate.set_media_root(None)
    sp_routes.clear_missing_services()
    srv._LIVE_SERVICES.clear()
    serving.clear_resolved_models()
