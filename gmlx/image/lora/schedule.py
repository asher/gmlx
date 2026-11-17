"""Adapter schedules: the sigmas, head rule and default size an adapter
was distilled to sample with, which gmlx applies whenever the adapter
runs.

An adapter declares its schedule in a sidecar next to its file,
``<file>.gmlx.json`` (``SIDECAR_SUFFIX``), which ``gmlx image generate
--save-adapter-schedule`` writes. A serve config's adapter entry carries
the same fields, and ``KNOWN`` holds the schedules of published adapters
by the sha256 of their files. Each field of an adapter's schedule is its
entry's, else its sidecar's, else its row of ``KNOWN`` (``declared``).
Each field of a request's schedule is then the request's own setting,
else the adapters' (which must agree), else the family's (``resolve``).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from dataclasses import dataclass

SIDECAR_SUFFIX = ".gmlx.json"
SCHEMA = "gmlx.adapter_schedule/1"
# How a step reads a bank of step heads: head i at step i, as diffusers
# reads them, or the heads its span reads, as ComfyUI does.
HEADS = ("in_order", "span")
FIELDS = ("sigmas", "heads", "size")


class ScheduleError(ValueError):
    """An adapter schedule that is not one, or adapters whose schedules
    disagree."""


@dataclass(frozen=True)
class AdapterSchedule:
    """``sigmas`` the schedule, from the first sigma to the last; ``heads``
    one of ``HEADS``; ``size`` the default (width, height). Each is None
    where the adapter declares nothing."""
    sigmas: tuple | None = None
    heads: str | None = None
    size: tuple | None = None

    def fields(self) -> dict:
        """The declared fields as a sidecar writes them."""
        out: dict = {}
        if self.sigmas is not None:
            out["sigmas"] = list(self.sigmas)
        if self.heads is not None:
            out["heads"] = self.heads
        if self.size is not None:
            out["size"] = f"{self.size[0]}x{self.size[1]}"
        return out


@dataclass(frozen=True)
class Declared:
    """The schedule of one adapter, under the ``name`` a message shows,
    and where it comes from: ``source`` names every place it takes a
    field from, and ``fields`` maps a field to its own place where they
    differ."""
    name: str
    source: str
    schedule: AdapterSchedule
    fields: tuple = ()

    def source_of(self, field: str) -> str:
        """Where the schedule's ``field`` comes from."""
        return dict(self.fields).get(field, self.source)


@dataclass(frozen=True)
class Resolved:
    """A request's schedule: ``sigmas`` None for the family's schedule,
    ``heads`` "in_order" or None for the DiT's own rule, ``size`` None for
    the family's default, and ``line`` what to print, None when no
    adapter sets anything. ``size_from`` names the adapter and the place
    a size the request did not give comes from."""
    sigmas: tuple | None = None
    heads: str | None = None
    size: tuple | None = None
    line: str | None = None
    size_from: str | None = None


@dataclass(frozen=True)
class Known:
    name: str
    source: str
    schedule: AdapterSchedule


# Published adapters gmlx knows the schedule of, by the sha256 of the file.
KNOWN = {
    # linoyts/MiniMax-H3-Image-pDMD-2steps c07d96e5635311ed007d44654c2f98cc8d6e0a1e,
    # pytorch_lora_weights.safetensors: the card's two steps at sigmas 1,
    # 12/13 and 0, with the head of each step read in order, at 1024x1024.
    "c0a88d9c1de401e50894424a6b9a73e46b4f0d977e181e48642fdcfe98ede435": Known(
        "pDMD-2steps", "linoyts/MiniMax-H3-Image-pDMD-2steps",
        AdapterSchedule(sigmas=(1.0, 12 / 13, 0.0), heads="in_order",
                        size=(1024, 1024))),
}

_SHA_LOCK = threading.Lock()
_SHA: dict = {}


def file_sha256(path: str) -> str:
    """The sha256 of the file at ``path``, kept by its real path, size and
    modification time, so a file is read once while it stays the same."""
    path = os.path.realpath(os.path.expanduser(path))
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime_ns)
    with _SHA_LOCK:
        hit = _SHA.get(key)
    if hit is not None:
        return hit
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(16 << 20):
            h.update(chunk)
    digest = h.hexdigest()
    with _SHA_LOCK:
        _SHA[key] = digest
    return digest


def sidecar_path(path: str) -> str:
    """Where the sidecar of the adapter at ``path`` sits."""
    return os.path.expanduser(path) + SIDECAR_SUFFIX


def check_sigmas(value, where: str) -> tuple:
    """A schedule of two or more finite sigmas that does not increase, as
    floats."""
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise ScheduleError(f"{where}: sigmas is a list of numbers, not "
                            f"{value!r}")
    out = []
    for v in value:
        if isinstance(v, bool) or not isinstance(v, (int, float)) \
                or not math.isfinite(v):
            raise ScheduleError(f"{where}: sigmas holds finite numbers, not "
                                f"{v!r}")
        out.append(float(v))
    if len(out) < 2:
        raise ScheduleError(f"{where}: sigmas holds two values or more")
    if any(b > a for a, b in zip(out, out[1:])):
        raise ScheduleError(f"{where}: sigmas must not increase, not "
                            f"{out}")
    return tuple(out)


def check_size(value, where: str) -> tuple:
    """A "WxH" size, or a [width, height] pair, of positive integers."""
    try:
        if isinstance(value, str):
            w, h = (int(t) for t in value.lower().split("x"))
        elif isinstance(value, (list, tuple)) and len(value) == 2 and not \
                any(isinstance(v, bool) or not isinstance(v, int)
                    for v in value):
            w, h = value
        else:
            raise ValueError
    except ValueError:
        raise ScheduleError(f"{where}: size is WxH, such as 1024x1024, not "
                            f"{value!r}") from None
    if w <= 0 or h <= 0:
        raise ScheduleError(f"{where}: size is WxH of positive integers, "
                            f"not {value!r}")
    return int(w), int(h)


def parse(fields, where: str, *, allowed=FIELDS) -> AdapterSchedule:
    """The schedule a mapping declares: ``sigmas``, ``heads`` and
    ``size``, each optional, at least one given. Any other key of
    ``fields`` outside ``allowed`` is refused."""
    if not isinstance(fields, dict):
        raise ScheduleError(f"{where}: expected a mapping of "
                            f"{', '.join(FIELDS)}, got {fields!r}")
    unknown = sorted(set(fields) - set(allowed))
    if unknown:
        raise ScheduleError(f"{where}: unknown key {unknown[0]!r} (it takes "
                            f"{', '.join(FIELDS)})")
    sigmas = heads = size = None
    if fields.get("sigmas") is not None:
        sigmas = check_sigmas(fields["sigmas"], where)
    if fields.get("heads") is not None:
        heads = fields["heads"]
        if heads not in HEADS:
            raise ScheduleError(f"{where}: heads is one of "
                                f"{', '.join(HEADS)}, not {heads!r}")
    if fields.get("size") is not None:
        size = check_size(fields["size"], where)
    if sigmas is None and heads is None and size is None:
        raise ScheduleError(f"{where}: declares none of {', '.join(FIELDS)}")
    return AdapterSchedule(sigmas, heads, size)


def read_sidecar(path: str) -> AdapterSchedule | None:
    """The schedule the sidecar of the adapter at ``path`` declares, None
    without a sidecar. A sidecar that does not parse, or that names the
    sha256 of another file, is refused with its path."""
    side = sidecar_path(path)
    if not os.path.exists(side):
        return None
    try:
        with open(side, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError) as e:
        raise ScheduleError(f"the adapter schedule {side} does not read as "
                            f"JSON: {e}") from None
    if not isinstance(raw, dict):
        raise ScheduleError(f"the adapter schedule {side} is not a JSON "
                            f"object")
    schema = raw.get("schema")
    if schema != SCHEMA:
        raise ScheduleError(f"the adapter schedule {side} has schema "
                            f"{schema!r}, not {SCHEMA!r}")
    sched = parse(raw, f"the adapter schedule {side}",
                  allowed=FIELDS + ("schema", "sha256"))
    want = raw.get("sha256")
    if want is not None and want != file_sha256(path):
        raise ScheduleError(
            f"the adapter schedule {side} was written for another file "
            f"(sha256 {str(want)[:12]}), not {os.path.basename(path)}; "
            f"write it again with --save-adapter-schedule or delete it")
    return sched


def write_sidecar(path: str, sched: AdapterSchedule) -> str:
    """Write the sidecar of the adapter at ``path``, bound to the file by
    its sha256, and return its path. A folder gmlx cannot write to is
    refused."""
    side = sidecar_path(path)
    folder = os.path.dirname(os.path.abspath(side))
    if not os.access(folder, os.W_OK):
        raise ScheduleError(f"cannot write the adapter schedule {side}: the "
                            f"folder {folder} is not writable")
    body = {"schema": SCHEMA, "sha256": file_sha256(path), **sched.fields()}
    tmp = f"{side}.tmp-{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(body, f, indent=1)
            f.write("\n")
        os.replace(tmp, side)
    except OSError as e:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise ScheduleError(f"cannot write the adapter schedule {side}: "
                            f"{e}") from None
    return side


def declared(path: str, entry: dict | None = None, *,
             sidecar: bool = True) -> Declared | None:
    """The schedule the adapter at ``path`` declares, field by field: its
    config entry's (``entry``), else its sidecar's, else its row of
    ``KNOWN``; None when none declares a field. ``sidecar`` False leaves
    the sidecar unread, for the run that replaces it."""
    name = os.path.basename(path)
    layers = []
    if entry:
        layers.append(("its entry in the config",
                       parse(entry, f"the config's adapter {name}")))
    side = read_sidecar(path) if sidecar else None
    if side is not None:
        layers.append((f"its sidecar {sidecar_path(path)}", side))
    row = KNOWN.get(file_sha256(path))
    if row is not None:
        name = row.name
        layers.append((f"gmlx's table of known adapters, {row.source}",
                       row.schedule))
    values, places = {}, {}
    for field in FIELDS:
        for place, sched in layers:
            if getattr(sched, field) is not None:
                values[field], places[field] = getattr(sched, field), place
                break
    if not values:
        return None
    used = [place for place, _ in layers if place in places.values()]
    source = " and ".join(used)
    return Declared(name, source, AdapterSchedule(**values),
                    tuple(places.items()) if len(used) > 1 else ())


def _shown(field: str, value) -> str:
    if field == "sigmas":
        return ", ".join(f"{v:.6g}" for v in value)
    if field == "size":
        return f"{value[0]}x{value[1]}"
    return {"in_order": "in order", "span": "by their span"}[value]


def _agreed(decls: list, field: str, given: str, *, who: bool = False):
    """The value the adapters agree on for ``field``, None when none
    declares it; adapters that disagree are refused. With ``who``, the
    value and the first adapter that declares it, with its place."""
    vals = [(d, getattr(d.schedule, field)) for d in decls
            if getattr(d.schedule, field) is not None]
    for d, v in vals[1:]:
        if v != vals[0][1]:
            a = vals[0][0]
            raise ScheduleError(
                f"adapters {a.name} ({a.source_of(field)}) and {d.name} "
                f"({d.source_of(field)}) declare different {field}: "
                f"{_shown(field, vals[0][1])} and {_shown(field, v)}; give "
                f"{given} to choose")
    value = vals[0][1] if vals else None
    if not who:
        return value
    d = vals[0][0] if vals else None
    return value, (None if d is None else f"{d.name} ({d.source_of(field)})")


def resolve(decls, *, sigmas=None, steps=None, size=None,
            names=None, size_from: str | None = None) -> Resolved:
    """The schedule of a request whose adapters declare ``decls`` (None
    for an adapter that declares nothing), under the request's own
    ``sigmas``, ``steps`` and ``size``, each None where it gives none.
    The request's sigmas read heads in order, and its step count runs
    the family's schedule; else the adapters' sigmas and heads apply,
    heads in order where they declare sigmas and no heads. ``names``
    maps sigmas, steps and size to what a message calls the request's
    setting, and ``size_from`` names where a size not from that setting
    came from."""
    names = {"sigmas": "sigmas", "steps": "steps", "size": "size",
             **(names or {})}
    decls = [d for d in decls if d is not None]
    given = []
    if sigmas is not None:
        out_sigmas, heads = tuple(float(v) for v in sigmas), "in_order"
        given.append(names["sigmas"])
    elif steps is not None:
        out_sigmas = heads = None
        given.append(names["steps"])
    else:
        out_sigmas = _agreed(decls, "sigmas", names["sigmas"])
        heads = _agreed(decls, "heads", names["sigmas"])
        if heads is None and out_sigmas is not None:
            heads = "in_order"
    if heads == "span":
        heads = None
    size_src = None
    if size is not None:
        out_size = tuple(size)
        given.append(size_from or names["size"])
    else:
        out_size, size_src = _agreed(decls, "size", names["size"], who=True)
    applied = []
    if sigmas is None and steps is None:
        if out_sigmas is not None:
            applied.append(f"sigmas {_shown('sigmas', out_sigmas)}")
        if any(d.schedule.heads or d.schedule.sigmas for d in decls):
            applied.append("heads " + _shown(
                "heads", "in_order" if heads else "span"))
    if size is None and out_size is not None:
        applied.append(f"default size {_shown('size', out_size)}")
    line = None
    if decls:
        who = ", ".join(f"{d.name} ({d.source})" for d in decls)
        what = ", ".join(applied) if applied else "nothing"
        line = f"adapter schedule of {who}: {what}"
        if given:
            line += (f"; {' and '.join(given)} given, which "
                     f"{'wins' if len(given) == 1 else 'win'}")
    return Resolved(out_sigmas, heads, out_size, line, size_src)
