"""Per-request adapter strengths on text routes.

A text model entry loads a list of adapters, each with a default strength.
A request sets the strength of any of them for itself alone, with the gmlx
``adapters`` field (by path or file name) or the llama-server ``lora``
field (by config index). The result is a :class:`RowLora`: one scale per
slot of the resident entry's adapter union. The gen-args seam parses it on
the request thread; the engine salts it at insert from the identities of
the adapters its own copy loaded, and the row carries it in its
``RowSettings``.
"""
from __future__ import annotations

import math
import os
from typing import Any, NamedTuple

import numpy as np

_STRENGTH_BOUND = 100.0


class RowLora(NamedTuple):
    scales: tuple            # one per union slot, slot order
    row: np.ndarray          # the same scales as float32, built once
    salt: int | None         # 0 when every slot is 0; None until salted
    model: str = ""          # the model id, for messages


class AdapterRequestError(ValueError):
    """A request's adapter fields are malformed or name no adapter of its
    model. A client error."""


class AdapterIdentityMissing(RuntimeError):
    """An adapted row runs on a copy with no loaded adapter identities. A
    server defect."""


def _present(request, name: str):
    if isinstance(request, dict):
        return request.get(name)
    return getattr(request, name, None)


def _strength(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)) \
            or not -_STRENGTH_BOUND <= float(value) <= _STRENGTH_BOUND:
        raise AdapterRequestError(
            f"{where} must be a finite number from -100 to 100")
    return float(value)


def _model_dirs() -> list:
    import gmlx.serve.bridge_vlm as serving

    cfg = serving.server_config()
    return list(getattr(cfg, "model_dirs", None) or ())


def _refuse_outside_roots(aid: str) -> None:
    if os.path.isabs(aid) or aid.startswith("~") or ".." in aid.split("/") \
            or "\\" in aid:
        raise AdapterRequestError(
            f"adapter {aid!r}: an id is a configured adapter or a path inside "
            "a model folder, never an absolute path")


class _Index:
    """The ways a request names an entry's adapters, built once per
    request: the written relative path, the file name and its stem."""

    def __init__(self, items):
        self.items = items
        self.written: dict = {}
        self.names: dict = {}
        for i, ref in enumerate(items):
            if not os.path.isabs(ref.written):
                self.written.setdefault(os.path.normpath(ref.written), i)
            base = os.path.basename(ref.path)
            for name in {base, os.path.splitext(base)[0]}:
                self.names.setdefault(name, []).append(i)


def _match(aid: str, index: _Index, model_id: str, model_dirs) -> int:
    """The config index ``aid`` names among the entry's adapters."""
    _refuse_outside_roots(aid)
    at = index.written.get(os.path.normpath(aid))
    if at is not None:
        return at
    hits = index.names.get(aid, [])
    if len(hits) > 1:
        raise AdapterRequestError(
            f"adapter {aid!r} matches adapters {hits[0]} and {hits[1]} of model "
            f"{model_id!r}; name it by its path in the config, or use lora "
            "with its index")
    if hits:
        return hits[0]
    real = {os.path.realpath(ref.path): i for i, ref in enumerate(index.items)}
    for root in model_dirs or ():
        root = os.path.realpath(os.path.expanduser(os.path.expandvars(root)))
        cand = os.path.realpath(os.path.join(root, aid))
        if cand.startswith(root + os.sep) and cand in real:
            return real[cand]
    raise AdapterRequestError(
        f"adapter {aid!r} is not an adapter of model {model_id!r}; a request "
        "sets the strength of the adapters its model entry loads (see "
        "/v1/models)")


def _from_adapters(value, items, model_id: str, model_dirs) -> dict:
    if not isinstance(value, list):
        raise AdapterRequestError("adapters must be a list of {id, strength}")
    out: dict = {}
    index = _Index(items)
    for n, item in enumerate(value):
        if isinstance(item, str):
            item = {"id": item}
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) \
                or not item["id"].strip():
            raise AdapterRequestError(
                "each adapter is {id, strength}, with id a string")
        unknown = set(item) - {"id", "strength"}
        if unknown:
            raise AdapterRequestError(
                f"adapter fields {sorted(unknown)} are unknown; an adapter is "
                "{id, strength}")
        st = item.get("strength")
        st = 1.0 if st is None else _strength(st, f"adapters[{n}].strength")
        aid = item["id"].strip()
        i = _match(aid, index, model_id, model_dirs)
        if i in out:
            raise AdapterRequestError(f"adapter {aid!r} is listed twice")
        out[i] = st
    return out


def _from_lora(value, items, model_id: str) -> dict:
    if not isinstance(value, list):
        raise AdapterRequestError("lora must be a list of {id, scale}")
    out: dict = {}
    count = len(items)
    for n, item in enumerate(value):
        if not isinstance(item, dict):
            raise AdapterRequestError("lora must be a list of {id, scale}")
        lid = item.get("id")
        if isinstance(lid, bool) or not isinstance(lid, int):
            raise AdapterRequestError(f"lora[{n}].id must be an integer")
        if not 0 <= lid < count:
            raise AdapterRequestError(
                f"lora id {lid} is out of range; model {model_id!r} has "
                f"{count} adapters (ids 0 to {count - 1})")
        if lid in out:
            raise AdapterRequestError(f"lora id {lid} is listed twice")
        out[lid] = _strength(item.get("scale"), f"lora[{n}].scale")
    return out


def request_lora(spec, request, *, model_dirs=None) -> RowLora:
    """The request's scales over its entry's adapter union. Parse and match
    only; raises :class:`AdapterRequestError` for a client error. The salt
    is 0 for a row with every slot at 0 and None otherwise, until
    :func:`salted` sets it on the engine. A spec whose adapter union lacks
    one of its own adapters raises :class:`AdapterIdentityMissing`."""
    from gmlx.serve.patches._common import refuse_both_spellings

    adapters = _present(request, "adapters")
    lora = _present(request, "lora")
    refuse_both_spellings({k for k, v in (("adapters", adapters), ("lora", lora))
                           if v is not None}, "adapters", "lora")
    model_id = getattr(spec, "id", "") if spec is not None else ""
    items = tuple(getattr(spec, "adapter_list", ()) or ()) if spec is not None else ()
    union = tuple(getattr(spec, "adapters", ()) or ()) if spec is not None else ()
    slots = tuple(getattr(spec, "adapter_slots", ()) or ()) if spec is not None else ()
    if not items:
        if (adapters is not None and adapters != []) or (lora is not None and lora != []):
            if not isinstance(adapters if adapters is not None else lora, list):
                raise AdapterRequestError(
                    "adapters must be a list of {id, strength}" if adapters is not None
                    else "lora must be a list of {id, scale}")
            raise AdapterRequestError(
                f"model {model_id!r} loads no adapters; adapters and lora apply "
                "only to a model entry with adapters")
        scales = (0.0,) * len(union)
        return RowLora(scales, np.zeros((len(scales),), np.float32), 0, model_id)
    if len(slots) != len(items):
        effective = getattr(spec, "effective_adapters", None)
        if callable(effective):
            union = tuple(effective())
        for ref in items:
            if ref.path not in union:
                raise AdapterIdentityMissing(
                    f"model {model_id!r}: adapter {ref.path!r} is not in its "
                    "adapter union; this is a gmlx defect")
        slots = tuple(union.index(ref.path) for ref in items)
    by_index = {i: ref.strength for i, ref in enumerate(items)}
    if adapters is not None:
        dirs = _model_dirs() if model_dirs is None else model_dirs
        by_index.update(_from_adapters(adapters, items, model_id, dirs))
    elif lora is not None:
        got = _from_lora(lora, items, model_id)
        if got:
            by_index = {i: got.get(i, 0.0) for i in range(len(items))}
    vals = [0.0] * len(union)
    for i, st in by_index.items():
        vals[slots[i]] = float(st)
    scales = tuple(vals)
    row = np.asarray(scales, np.float32).reshape(-1)
    return RowLora(scales, row, 0 if not row.any() else None, model_id)


def salted(lora: RowLora, identities) -> RowLora:
    """``lora`` with its salt set from the loaded adapter identities of the
    copy that runs the row. Raises :class:`AdapterIdentityMissing` for an
    adapted row on a copy with no identities, or with identities of
    another union; the salt never falls back to 0 for an adapted row."""
    if lora.salt is not None:
        return lora
    from gmlx import lora_rows

    if identities is None or len(identities) != len(lora.scales):
        raise AdapterIdentityMissing(
            f"model {lora.model!r} has no loaded adapter identities; this is a "
            "gmlx defect")
    return lora._replace(salt=lora_rows.row_salt(identities, lora.scales))


_GEN_ARGS_FLAG = "_kq_gguf_lora_gen_args"


def install_lora_gen_args() -> None:
    """Request thread: set ``args._kq_lora`` from the resolved spec and the
    request's ``adapters`` or ``lora`` field. A malformed field is a 400."""
    from fastapi import HTTPException

    from gmlx.serve.patches._common import _install_gen_args_transform
    import gmlx.serve.bridge_vlm as serving

    def _stamp(args, request, _processor):
        try:
            args._kq_lora = request_lora(serving.get_active_spec(), request)
        except AdapterRequestError as e:
            raise HTTPException(400, detail=str(e)) from None
        return args

    _install_gen_args_transform(_GEN_ARGS_FLAG, _stamp)


def request_fields_error(spec, request: Any) -> str | None:
    """The message of a request's malformed adapter fields, or None. For
    callers that check before the seam runs."""
    from fastapi import HTTPException

    try:
        request_lora(spec, request)
    except AdapterRequestError as e:
        return str(e)
    except HTTPException as e:
        return str(e.detail)
    return None
