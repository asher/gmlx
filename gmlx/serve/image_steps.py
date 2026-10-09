"""The image step contract: one denoise step at a time for a client whose
sampler runs elsewhere, such as the ComfyUI node pack.

The routes live under ``/internal/image-steps/v1`` (``patches.image_steps``)
and this module holds what they share: the caches of uploaded tensors and
adapter matrices, the patch tables each client defined, the runs, the
family facts a client builds its model from, and the work each route runs
on the image worker.

A run is a cache scope and a busy marker. It holds the image model in the
pool until it ends or sits idle for ``RUN_IDLE_S``, and its end sheds the
prefix states of the conditionings it stepped and stops a step of it that
is still running. Steps of every client queue on the image worker in the
order they come, between the jobs of the OpenAI image routes.

Under ``convention: "comfy"`` a step takes what ComfyUI hands its DiT, the
latents in model space and a timestep per row, and gives back what
ComfyUI expects, which ``CONVENTIONS`` holds per family. ComfyUI's
Lumina2 forward, which runs Z-Image, takes ``1 - timestep`` as the time of
the native model and negates its output; its Qwen-Image 2.1 forward takes
the timestep as it is and returns the native output. Both give, for the
flow schedule, the velocity ``d x / d sigma`` at ``sigma = timestep``,
which ``denoise_step`` returns.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .image_wire import WireError, WireTensor, short_dtype

_log = logging.getLogger(__name__)

RUN_IDLE_S = 120.0
MAX_ROWS = 16
MAX_TENSOR_BYTES = 1 << 30
MAX_ADAPTER_BYTES = 1 << 30
FEATURES = ("prefix_cache", "noise_mask", "rgba_vae", "tiled_vae")


@dataclass(frozen=True)
class Convention:
    """How a step reads a timestep and returns its output under one
    client convention: ``sigma = scale * timestep + shift`` and
    ``out = sign * velocity``."""
    scale: float = 1.0
    shift: float = 0.0
    sign: float = 1.0


CONVENTIONS = {
    "comfy": {"z_image": Convention(), "qwen_image21": Convention()},
}

# What a client builds its model from, per family: ComfyUI's model class
# and image model name, the latent format and the VAE's.
COMFY = {
    "z_image": {"model_class": "ZImage", "image_model": "lumina2",
                "latent": {"channels": 16, "stride": 8, "patch": 2},
                "vae": {"channels_out": 3, "stride": 8}},
    "qwen_image21": {"model_class": "QwenImage21",
                     "image_model": "qwen_image21",
                     "latent": {"channels": 64, "stride": 16, "patch": 1},
                     "vae": {"channels_out": 4, "stride": 16}},
}


class StepError(Exception):
    """A step contract request refused with its status and a body."""

    def __init__(self, status: int, message: str, **extra):
        super().__init__(message)
        self.status = status
        self.extra = extra

    def body(self) -> dict:
        return {"error": str(self), **self.extra}


class StepInterrupted(Exception):
    """The run of a step was interrupted or ended while it ran."""


class Missing(StepError):
    def __init__(self, hashes):
        missing = sorted(set(hashes))
        super().__init__(409, f"{len(missing)} tensor(s) or table(s) are not "
                         "held by the server; upload them and retry",
                         missing=missing)


def _wire_copy(t: WireTensor) -> WireTensor:
    return WireTensor(t.dtype, t.shape, memoryview(bytes(t.raw)))


class _Bytes:
    """A least recently used map of WireTensors bounded in bytes."""

    def __init__(self, max_bytes: int):
        from collections import OrderedDict

        self.max_bytes = int(max_bytes)
        self._items: OrderedDict = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            t = self._items.get(key)
            if t is not None:
                self._items.move_to_end(key)
            return t

    def __contains__(self, key) -> bool:
        with self._lock:
            return key in self._items

    def put(self, key, t: WireTensor) -> None:
        if t.nbytes > self.max_bytes:
            raise StepError(413, f"a tensor of {t.nbytes} bytes is over the "
                            f"cache of {self.max_bytes} bytes")
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self._bytes -= old.nbytes
            self._items[key] = t
            self._bytes += t.nbytes
            while self._bytes > self.max_bytes and len(self._items) > 1:
                _, gone = self._items.popitem(last=False)
                self._bytes -= gone.nbytes

    @property
    def bytes(self) -> int:
        return self._bytes

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0


@dataclass(eq=False)
class Run:
    id: str
    model: str
    hold: Any
    began: float
    last: float
    gen: int = 0                       # an interrupt bumps it
    ended: bool = False
    steps: int = 0
    rows: int = 0
    forward_s: float = 0.0
    cache_hits: int = 0
    conds: set = field(default_factory=set)         # keys, to shed at end
    cond_objs: dict = field(default_factory=dict)   # the run's state scope


class StepsState:
    """The state the step routes share. ``configure`` sets the cache
    sizes from ``server.image``."""

    def __init__(self):
        self.tensors = _Bytes(1024 << 20)
        self.adapters = _Bytes(2048 << 20)
        self.runs: dict = {}
        # model id -> (pipeline, {contract table id: core table id})
        self.tables: dict = {}
        self.lock = threading.Lock()
        self.settings = None
        self.clock = time.monotonic
        # The server writes a log line per run and per refusal; the bench's
        # loopback routes write none, so its stdout stays JSON.
        self.log_requests = True

    def configure(self, settings) -> None:
        self.settings = settings
        tmb = settings.tensor_cache_mb << 20
        amb = settings.adapter_cache_mb << 20
        if tmb != self.tensors.max_bytes:
            self.tensors = _Bytes(tmb)
        if amb != self.adapters.max_bytes:
            self.adapters = _Bytes(amb)

    # Tensors and adapters.

    def put_tensor(self, store: _Bytes, want: str, tensors: dict,
                   limit: int) -> str:
        if len(tensors) != 1:
            raise StepError(400, f"an upload holds one tensor, not "
                            f"{len(tensors)}")
        (t,) = tensors.values()
        if t.nbytes > limit:
            raise StepError(413, f"a tensor of {t.nbytes} bytes is over the "
                            f"limit of {limit}")
        got = t.hash()
        if got != want:
            raise StepError(400, "the tensor's hash is not the one in the "
                            "path", got=got)
        store.put(got, _wire_copy(t))
        return got

    # Runs.

    def begin(self, model: str, hold) -> Run:
        now = self.clock()
        run = Run(id=uuid.uuid4().hex[:16], model=model, hold=hold,
                  began=now, last=now)
        with self.lock:
            self.runs[run.id] = run
        return run

    def get_run(self, run_id: str | None) -> Run | None:
        if not run_id:
            return None
        with self.lock:
            run = self.runs.get(run_id)
        if run is None:
            raise StepError(404, f"no run {run_id!r}; it ended or sat idle "
                            f"past {RUN_IDLE_S:.0f} seconds")
        run.last = self.clock()
        return run

    def end(self, run: Run, reason: str = "end") -> None:
        with self.lock:
            if self.runs.pop(run.id, None) is None:
                return
        run.ended = True
        run.gen += 1
        try:
            if run.hold is not None:
                run.hold.release()
        except Exception:  # noqa: BLE001 - a release must not fail the end
            _log.debug("run hold release failed", exc_info=True)
        if run.conds:
            served = getattr(getattr(run.hold, "_entry", None), "pipeline",
                             None)
            if served is not None and served.pipe is not None:
                from .image_service import WORKER

                WORKER.submit(shed_states, served.pipe, set(run.conds))
        if not self.log_requests:
            return
        from .patches.routes import _log_aux_request

        _log_aux_request("/internal/image-steps/v1/runs", run.model,
                         run.began, status="ok" if reason == "end"
                         else reason, steps=run.steps, rows=run.rows,
                         forward=f"{run.forward_s:.2f}s",
                         prefix_hits=run.cache_hits)

    def sweep(self) -> list:
        """End the runs idle past ``RUN_IDLE_S``. Returns their ids."""
        now = self.clock()
        with self.lock:
            idle = [r for r in self.runs.values()
                    if now - r.last > RUN_IDLE_S]
        for r in idle:
            self.end(r, reason="idle")
        return [r.id for r in idle]

    def interrupt(self, run_id: str | None) -> int:
        """Interrupt the in-flight step of a run, or of every run when
        ``run_id`` is None. Returns how many runs it reached."""
        with self.lock:
            runs = list(self.runs.values()) if not run_id else [
                r for r in self.runs.values() if r.id == run_id]
        for r in runs:
            r.gen += 1
        return len(runs)

    # Patch tables.

    def tables_of(self, model: str, pipe) -> dict:
        with self.lock:
            held = self.tables.get(model)
            if held is None or held[0] is not pipe:
                held = (pipe, {})
                self.tables[model] = held
            return held[1]

    def reset(self) -> None:
        for r in list(self.runs.values()):
            self.end(r, reason="reset")
        self.tensors.clear()
        self.adapters.clear()
        with self.lock:
            self.tables.clear()


STATE = StepsState()


def shed_states(pipe, conds: set) -> None:
    """Drop the held prefix states of the conditionings whose keys are
    ``conds``."""
    cache = pipe.captions
    with cache._lock:
        keep = {k for k in cache._items
                if not (isinstance(k, tuple) and k and k[0] in conds)}
    cache.shed_others(keep, 0)


# Family facts and payloads.

def family_of(spec) -> tuple:
    """``(family, variant)`` of a configured image model, from its
    header."""
    from gmlx.image.loader import identify_dit

    ident = identify_dit(spec.path)
    return ident.family, ident.variant


def status(spec, resident: bool, plan=None) -> dict:
    from gmlx.image.dtypes import image_activation

    fam, variant = family_of(spec)
    facts = COMFY.get(fam.name)
    if facts is None:
        raise StepError(400, f"{fam.title} has no step contract adapter")
    from gmlx.image.api import _header_bytes

    mem = {"dit_bytes": _header_bytes(spec.path, "dit"),
           "encoder_bytes": (_header_bytes(spec.encoder, "encoder")
                             if spec.encoder else 0),
           "vae_bytes": _header_bytes(spec.vae, "vae")}
    if plan is not None:
        weights = sum(mem.values())
        mem["activation_reserve_bytes"] = max(0, plan.footprint - weights)
        mem["resident_bytes"] = plan.footprint if resident else 0
    act = image_activation(fam.name)
    return {"model": spec.id, "family": fam.name, "variant": variant,
            "resident": resident,
            "comfy": {"model_class": facts["model_class"],
                      "image_model": facts["image_model"]},
            "latent": dict(facts["latent"]), "vae": dict(facts["vae"]),
            "edit_capable": fam.edit is not None,
            "sampling": sampling(fam, variant),
            "compute_dtype": str(act.dit).rsplit(".", 1)[-1].replace(
                "bfloat16", "bf16").replace("float16", "f16").replace(
                "float32", "f32"),
            "memory": mem}


def sampling(fam, variant: str) -> dict | None:
    """The variant's own sampling: its steps and cfg, and for a variant
    distilled to a fixed schedule, those sigmas with the final zero, which
    a client samples on in place of its own schedule."""
    row = fam.sampling_row(variant)
    if row is None:
        return None
    return {"steps": row.steps, "cfg": row.cfg, "shift": row.shift,
            "sigmas": [*map(float, row.sigmas), 0.0] if row.sigmas else None}


def _torch_shape(kv: dict, name: str, ne) -> list:
    orig = kv.get(f"comfy.gguf.orig_shape.{name}")
    if isinstance(orig, (list, tuple)):
        return [int(v) for v in orig]
    return [int(v) for v in reversed(list(ne))]


_DTYPE_CLASS = {"F32": "f32", "F16": "f16", "BF16": "bf16"}


def manifest(spec) -> dict:
    """The DiT's tensor names in GGUF key naming, their shapes in PyTorch
    order and their dtype class, and the values of the tensors a client's
    model detection reads by value."""
    import base64

    import gguf
    import numpy as np

    fam, _variant = family_of(spec)
    reader = gguf.GGUFReader(spec.path)
    kv = {}
    for f in reader.fields.values():
        if f.name.startswith("comfy.gguf.orig_shape."):
            kv[f.name] = [int(f.parts[i][0]) for i in f.data]
        elif f.name == "general.architecture":
            kv[f.name] = bytes(f.parts[f.data[0]]).decode()
    tensors = []
    by_name = {}
    for t in reader.tensors:
        ggml = t.tensor_type.name
        tensors.append({"name": t.name,
                        "shape": _torch_shape(kv, t.name, t.shape),
                        "dtype": _DTYPE_CLASS.get(ggml, "quant"),
                        "ggml_type": ggml})
        by_name[t.name] = t
    inline = {}
    for name in _inline_names(fam.name, [t["name"] for t in tensors]):
        t = by_name.get(name)
        if t is not None and t.tensor_type.name == "F32":
            a = np.asarray(t.data, np.float32).reshape(
                _torch_shape(kv, name, t.shape))
            inline[name] = {"dtype": "f32", "shape": list(a.shape),
                            "b64": base64.b64encode(
                                a.astype("<f4").tobytes()).decode()}
    return {"model": spec.id, "family": fam.name, "naming": "gguf",
            "arch": kv.get("general.architecture"), "count": len(tensors),
            "tensors": tensors, "inline": inline}


def _inline_names(family: str, names: list) -> list:
    """Tensors whose values ComfyUI's detection reads. For Z-Image, the
    spread of ``layers.{n-2}.ffn_norm1.weight`` sets ``allow_fp16``."""
    if family != "z_image":
        return []
    layers = {int(n.split(".")[1]) for n in names
              if n.startswith("layers.") and n.split(".")[1].isdigit()}
    return [f"layers.{max(layers) - 1}.ffn_norm1.weight"] if layers else []


def capabilities() -> dict:
    from gmlx import __version__
    from gmlx.image.registry import families

    from .image_wire import CONTRACT

    return {"contract": {"major": CONTRACT[0], "minor": CONTRACT[1]},
            "min_client": {"major": 1, "minor": 0},
            "gmlx_version": __version__,
            "wire": {"container": "safetensors",
                     "dtypes": ["f32", "f16", "bf16", "u8"]},
            "families": sorted(f.name for f in families()
                               if f.name in COMFY),
            "conventions": sorted(CONVENTIONS),
            "patch_kinds": ["lora"],
            "features": list(FEATURES),
            "limits": {"max_tensor_bytes": MAX_TENSOR_BYTES,
                       "max_adapter_bytes": MAX_ADAPTER_BYTES,
                       "max_rows": MAX_ROWS}}


# Work on the image worker.

def _f32(t: WireTensor):
    import mlx.core as mx

    return t.mx().astype(mx.float32)


def _cond_key(h: str) -> str:
    """The cache key of a conditioning that arrived by hash: its own
    namespace, so it never meets a key the core gives an encoded prompt."""
    return "wire:" + h


def row_hashes(row) -> list:
    """The tensor hashes a step row names: its cond, its reference
    latents and its attention mask."""
    if not isinstance(row, dict) or not isinstance(row.get("cond"), str):
        raise StepError(400, "each row is an object with a cond hash")
    refs = row.get("ref_latents") or []
    if not isinstance(refs, list) or not all(isinstance(r, str)
                                             for r in refs):
        raise StepError(400, "a row's ref_latents is a list of hashes")
    mask = row.get("attention_mask")
    if mask is not None and not isinstance(mask, str):
        raise StepError(400, "a row's attention_mask is a hash")
    return [row["cond"], *refs, *([mask] if mask else [])]


# Families whose conditioning splices reference latents into the text at
# image slots. On the wire a slot is ComfyUI's: the text row a reference
# goes in before, one per reference, and the cond holds text rows only.
# The core's encoder layout instead gives each reference a run of slot
# rows in the features, one per SLOT_TOKENS latent tokens.
SLOT_FAMILIES = frozenset({"qwen_image21"})


def _slot_counts(ref_shapes) -> list:
    from gmlx.image.families.qwen_image21.layout import SLOT_TOKENS

    counts = []
    for h, w in ref_shapes:
        if (int(h) * int(w)) % SLOT_TOKENS:
            raise StepError(400, f"a reference latent of {h}x{w} tokens "
                            f"does not fill whole image slots of "
                            f"{SLOT_TOKENS} tokens")
        counts.append(int(h) * int(w) // SLOT_TOKENS)
    return counts


def slots_to_wire(n_rows: int, slots, ref_shapes) -> tuple:
    """The core's encoder layout as the wire's: the text rows of the
    features, and the text row each reference goes in before."""
    import numpy as np

    slots = [int(s) for s in slots]
    counts = _slot_counts(ref_shapes)
    if sum(counts) != len(slots):
        raise StepError(500, f"{len(slots)} image slots for references of "
                        f"{sum(counts)} slots")
    is_slot = np.zeros(n_rows, bool)
    is_slot[slots] = True
    before, at = [], 0
    for n in counts:
        block = slots[at:at + n]
        if block != list(range(block[0], block[0] + n)):
            raise StepError(500, "a reference's image slots are not "
                            "consecutive rows")
        before.append(block[0] - at)
        at += n
    return np.flatnonzero(~is_slot), before


def slots_from_wire(x, before, ref_shapes) -> tuple:
    """Text rows ``x`` (T, D) and the wire's slots as the core's layout:
    each reference's slot rows put in before its text row. A reference
    without a slot goes after the text, as in ComfyUI."""
    import mlx.core as mx

    n_text = int(x.shape[0])
    counts = _slot_counts(ref_shapes)
    before = ([int(s) for s in before] + [n_text] * len(counts))[:len(counts)]
    if before != sorted(before) or any(not 0 <= s <= n_text for s in before):
        raise StepError(400, f"image slots {before} are not ascending text "
                        f"rows of the cond's {n_text}")
    parts, slots, prev = [], [], 0
    for s, n in zip(before, counts):
        parts.append(x[prev:s])
        start = len(slots) + s
        parts.append(mx.zeros((n, x.shape[1]), x.dtype))
        slots.extend(range(start, start + n))
        prev = s
    parts.append(x[prev:])
    return mx.concatenate(parts, axis=0), tuple(slots)


def conditioning(row: dict, tensors: _Bytes, family: str = ""):
    """The Conditioning of a step row from the tensors held, for a model
    of ``family``. Raises :class:`Missing` for a tensor the cache no
    longer holds."""
    from gmlx.image.conditioning import Conditioning

    held = {h: tensors.get(h) for h in row_hashes(row)}
    gone = [h for h, t in held.items() if t is None]
    if gone:
        raise Missing(gone)
    h = row["cond"]
    feat = held[h]
    refs = [(r, held[r]) for r in row.get("ref_latents") or []]
    mask_h = row.get("attention_mask")
    mask = held[mask_h] if mask_h else None
    x = _f32(feat)
    if x.ndim == 3:
        if x.shape[0] != 1:
            raise StepError(400, f"a row's cond is (1, L, D) or (L, D), not "
                            f"{tuple(x.shape)}")
        x = x[0]
    if x.ndim != 2:
        raise StepError(400, f"a row's cond is (1, L, D) or (L, D), not "
                        f"{tuple(x.shape)}")
    n = row.get("num_tokens")
    if mask is not None:
        n = _mask_tokens(mask, n)
    if n is not None:
        n = int(n)
        if not (0 < n <= x.shape[0] or n == x.shape[0]):
            raise StepError(400, f"num_tokens {n} is outside the cond's "
                            f"{x.shape[0]} rows")
        x = x[:n]
    key = _cond_key(h) if n is None or n == feat.shape[-2] else \
        _cond_key(f"{h}:{n}")
    if not row.get("prefix_cache", True):
        key = ""
    latents = []
    for _r, t in refs:
        z = _f32(t)
        latents.append(z[0] if z.ndim == 4 else z)
    slots = tuple(int(s) for s in row.get("image_slots") or ())
    if family in SLOT_FAMILIES and latents:
        x, slots = slots_from_wire(x, slots,
                                   [tuple(z.shape[-2:]) for z in latents])
    return Conditioning(x, key, ref_latents=tuple(latents),
                        image_slots=slots,
                        ref_hashes=tuple(r for r, _ in refs))


def _mask_tokens(mask: WireTensor, n) -> int:
    """The token count of an attention mask whose ones are a prefix: the
    core encodes prompts unpadded, so a mask over padding is the count of
    real tokens, and any other mask is refused."""
    import numpy as np

    m = np.asarray(mask.numpy(), np.float32).reshape(-1)
    ones = int((m > 0).sum())
    if not (m[:ones] > 0).all() or (m[ones:] > 0).any():
        raise StepError(400, "an attention mask must keep a prefix of the "
                        "tokens; the server encodes prompts unpadded")
    if n is not None and int(n) != ones:
        raise StepError(400, f"num_tokens {n} disagrees with the attention "
                        f"mask's {ones} tokens")
    return ones


def step_pacer(run: Run | None, gen: int):
    """The pacer of a step: it stops the step when its run is interrupted
    or ended, and gives chat its share of the GPU."""
    from .gpu_share import pacer

    chat = pacer()

    def fn(arrays):
        if run is not None and (run.gen != gen or run.ended):
            raise StepInterrupted()
        if chat is not None:
            chat(arrays)

    return fn


def run_step(served, x, sigmas, conds, table, run, gen) -> tuple:
    """One ``denoise_step`` over the rows on the image worker. Returns
    (velocity f32 evaluated, forward seconds, prefix cache answer)."""
    import mlx.core as mx

    from gmlx.image import api
    from gmlx.image.pacing import paced

    pipe = served.pipe
    if run is not None and (run.gen != gen or run.ended):
        raise StepInterrupted()
    cache = _prefix_answer(pipe, conds, table, x.shape[2:])
    t0 = time.perf_counter()
    with paced(step_pacer(run, gen)):
        v = api.denoise_step(pipe, x, sigmas, conds, patch_table=table)
        v = v.astype(mx.float32)
        mx.eval(v)
    return v, time.perf_counter() - t0, cache


def _prefix_answer(pipe, conds, table, size) -> str:
    from gmlx.image import api

    if pipe.family.step is None:
        return "none"
    if any(not c.key for c in conds):
        return "off"
    tid = api._table_id(pipe, table)
    keys = [api._state_key(pipe, c, tid, size) for c in conds]
    return "hit" if all(pipe.captions.peek(k) is not None
                        for k in keys) else "miss"


def define_table(served, body: dict, adapters: _Bytes) -> str:
    """The core table of a contract table body, with its pairs registered
    on the pipeline. Raises :class:`Missing` for adapter matrices not
    held."""
    from gmlx.image import api
    from gmlx.image.lora import patch_table as pt
    from gmlx.image.lora.reader import AdapterError

    pipe = served.pipe
    entries = body.get("entries")
    if not isinstance(entries, list):
        raise StepError(400, "a patch table body needs a list of entries")
    missing = []
    for e in entries:
        for p in (e.get("patches") or []) if isinstance(e, dict) else []:
            for side in ("up", "down"):
                h = p.get(side)
                if not isinstance(h, str) or h not in adapters:
                    missing.append(str(h))
    if missing:
        raise Missing(missing)
    rows = []
    pairs: dict = {}
    try:
        for e in entries:
            row = {"key": e.get("key"), "band": e.get("band"), "patches": []}
            for p in e.get("patches") or []:
                unknown = set(p) - {"adapter", "up", "down", "alpha",
                                    "strength", "strength_model", "kind"}
                if unknown:
                    raise StepError(400, f"patch fields {sorted(unknown)} "
                                    "are not read by this server")
                pk = (p["up"], p["down"], p.get("alpha"))
                if pk not in pairs:
                    up = adapters.get(p["up"]).mx()
                    down = adapters.get(p["down"]).mx()
                    pairs[pk] = pt.register_patch(pipe, up, down,
                                                  p.get("alpha"))
                row["patches"].append({
                    "adapter": pairs[pk], "strength": p.get("strength", 1.0),
                    "strength_model": p.get("strength_model", 1.0),
                    **({"kind": p["kind"]} if "kind" in p else {})})
            rows.append(row)
        plans = pt.plans_from_patches(pipe.family, rows)
        return api.define_patch_table(pipe, plans)
    except (AdapterError, ValueError, KeyError, TypeError) as e:
        if isinstance(e, StepError):
            raise
        raise StepError(400, f"patch table refused: {e}") from None


def table_id(body: bytes) -> str:
    """The id of a patch table: the sha256 of its canonical JSON."""
    try:
        doc = json.loads(body)
    except ValueError:
        raise StepError(400, "a patch table body is JSON") from None
    canon = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canon).hexdigest()


def encode(served, text: str, images: list, borrow=None,
           ref_latents: bool = True) -> dict:
    """Encode ``text``, with reference images (H, W, 3) uint8 for a family
    that edits, on the image worker. Each image keeps its own area, so a
    client that sizes its references feeds the encoder and its VAE the
    same pixels. Returns the conditioning features f32 (1, L, D), the
    reference latents in model space (1, C, h, w), none when
    ``ref_latents`` is false, and the image slots."""
    import mlx.core as mx

    from gmlx.image import api

    from .image_service import _encode

    pipe = served.pipe
    refs: tuple = ()
    if images:
        edit = pipe.family.edit
        if edit is None:
            raise StepError(400, f"{pipe.family.title} takes no reference "
                            "images")
        try:
            refs = api.references(pipe, [
                api.references(pipe, [im],
                               area=im.shape[0] * im.shape[1])[0]
                for im in images[:edit.max_refs + 1]])
        except ValueError as e:
            raise StepError(400, str(e)) from None
    cond, _ = _encode(served, text, None, refs, borrow)
    enc = pipe.encoder
    if enc is not None and enc.owned and not enc.released and \
            not served.plan.keep_encoder:
        enc.release()
    if refs and ref_latents:
        cond = api.encode_references(pipe, cond)
    feats = cond.features.astype(mx.float32)
    slots = [int(s) for s in cond.image_slots]
    if refs and pipe.family.name in SLOT_FAMILIES:
        lat = COMFY[pipe.family.name]["latent"]
        px = lat["stride"] * lat["patch"]
        text, slots = slots_to_wire(
            int(feats.shape[0]), slots,
            [(r.pixels.shape[0] // px, r.pixels.shape[1] // px)
             for r in refs])
        feats = feats[mx.array(text)]
    feats = feats[None]
    lats = [z.astype(mx.float32)[None] for z in cond.ref_latents] \
        if ref_latents else []
    mx.eval(feats, *lats)
    return {"cond": _host(feats), "refs": [_host(z) for z in lats],
            "image_slots": slots}


def _host(a):
    """An evaluated MLX array as a numpy array. MLX streams belong to the
    thread that made them, so arrays leave the image worker as numpy."""
    import numpy as np

    return np.array(a, copy=True)


def vae_decode(served, raw: WireTensor, alpha: bool):
    """VAE-space latents (B, C, h, w) to an image (B, H, W, C) f32 in
    [0, 1], as numpy."""
    import mlx.core as mx

    from gmlx.image import api

    pipe = served.pipe
    vae = pipe.vae
    z = _f32(raw).transpose(0, 2, 3, 1)
    model = vae.to_model_space(z)
    img = api.vae_decode(pipe, model, alpha=alpha)
    mx.eval(img)
    return _host(img.astype(mx.float32))


def vae_encode(served, image: WireTensor):
    """An image (B, H, W, C) f32 in [0, 1] to VAE-space latents
    (B, C, h, w) f32, as numpy."""
    import mlx.core as mx

    from gmlx.image import api

    pipe = served.pipe
    img = _f32(image)
    model = api.vae_encode(pipe, img[..., :3] if pipe.vae.config.in_channels
                           == 3 else img)
    raw = pipe.vae.to_vae_space(model).transpose(0, 3, 1, 2)
    mx.eval(raw)
    return _host(raw.astype(mx.float32))


def wire_dtype_name(dtype: str) -> str:
    return short_dtype(dtype)


def os_path_under(path: str, roots) -> str | None:
    """``path`` resolved, when it lies in one of ``roots``."""
    real = os.path.realpath(os.path.expanduser(path))
    for root in roots or ():
        root = os.path.realpath(os.path.expanduser(root))
        if real.startswith(root + os.sep) and os.path.isfile(real):
            return real
    return None


__all__ = ["STATE", "StepError", "StepInterrupted", "WireError", "Missing",
           "CONVENTIONS", "COMFY", "capabilities", "status", "manifest"]
