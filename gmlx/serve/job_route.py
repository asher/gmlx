"""Shared pieces of the routes that run a request as an engine job.

A job route holds its model, reads the request's adapter scales, and runs
its forwards on the model's engine thread (``engine_jobs``). The systemone
and scoring routes use these helpers."""

from __future__ import annotations

import importlib
from typing import Callable

from fastapi import HTTPException

import gmlx.serve.bridge_vlm as serving


def request_lora(spec, body):
    """The request's adapter scales, or a 400 for a malformed field."""
    from gmlx.serve.lora_request import AdapterRequestError
    from gmlx.serve.lora_request import request_lora as parse

    try:
        return parse(spec, body)
    except AdapterRequestError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None


def salted(lora, rg):
    """``lora`` salted from the adapters ``rg``'s copy loaded; a missing
    identity is a server defect, a 500."""
    from gmlx.serve.bridge_vlm import loaded_adapters_of
    from gmlx.serve.lora_request import AdapterIdentityMissing
    from gmlx.serve.lora_request import salted as salt

    try:
        return salt(lora, loaded_adapters_of(rg))
    except AdapterIdentityMissing as e:
        raise HTTPException(status_code=500, detail=str(e)) from None


def resolve_engine(pick: Callable[[], str], profile, runtime):
    """Load the model ``pick()`` names and return ``(rg, hold, spec,
    model_str)``: the engine, its residency hold, the model spec and the
    model string. The caller releases ``hold``."""
    from gmlx.serve.residency import _http_from_resolver_error

    app_mod = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")
    # The residency seam resolves the model string with the body profile
    # from this context variable.
    token = serving.set_request_profile(profile)
    try:
        try:
            model_str = pick()
        except (serving.ModelNotFound, serving.ModelFileMissing,
                serving.UnknownProfile, serving.NoModelSpecified) as e:
            raise _http_from_resolver_error(e) from e
        app_mod.get_cached_model(model_str)
        spec = serving.get_active_spec()
    finally:
        serving.reset_request_profile(token)
    guard = runtime.response_generator
    rg = getattr(guard, "_rg", guard)
    hold = getattr(guard, "_hold", None)
    if rg is None:
        if hold is not None:
            hold.release()
        raise HTTPException(status_code=500, detail=(
            "the model engine is unavailable; restart the server"))
    return rg, hold, spec, model_str
