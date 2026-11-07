"""The image step contract: one denoise step at a time for a client whose
sampler runs elsewhere, such as the ComfyUI node pack.

The routes live under ``/internal/image-steps/v1`` (``patches.image_steps``)
and this module holds what they share: the caches of uploaded tensors and
adapter matrices, the patch tables each client defined, the runs, the
family facts a client builds its model from, and the work each route runs
on the image worker.

A run is a cache scope and a busy marker. It holds the image model in the
pool until it ends or sits idle for ``RUN_IDLE_S``, which a timer checks,
and its end sheds the prefix states of the conditionings it stepped. An
interrupt or an end of a run discards the step of it that is running when
that step's evaluation finishes, and every later step of the run is
refused. Steps of every client queue on the image worker in the order
they come, between the jobs of the OpenAI image routes.

Under ``convention: "comfy"`` a step takes what ComfyUI hands its DiT, the
latents in model space and a timestep per row, and gives back what
ComfyUI expects, which ``CONVENTIONS`` holds per family. ComfyUI's
Lumina2 forward, which runs Z-Image, takes ``1 - timestep`` as the time of
the native model and negates its output; its Qwen-Image 2.1 forward takes
the timestep as it is and returns the native output, as its Krea 2,
FLUX.2 and FLUX.3 forwards do. All give, for the flow schedule, the
velocity ``d x / d sigma`` at ``sigma = timestep``, which
``denoise_step`` returns. ComfyUI's FLUX.2 forward embeds the fp32 time,
where BFL's pipeline, which gmlx samples on, embeds its bf16 time, so a
FLUX.2 step takes the DiT's "fp32" time rule. ComfyUI's FLUX.3 forward
embeds the time in the model's dtype, as BFL's does. Under
``convention: "reference"`` a step embeds the time by the DiT's reference
rule, as ``sample`` does, and is otherwise the comfy step.

A FLUX.3 flow map DiT also reads the time its step goes to. ComfyUI's
forward takes it from the schedule its sampler runs on: the entry after
the one nearest the timestep, the last at most. A step on a flow map
holds that schedule as ``sample_sigmas`` and the route takes the target
of each row by the same rule. ComfyUI's FLUX.2 and FLUX.3 latents are
the DiT's own model space, the VAE's latents in 2x2 patches and
normalized, so their VAE routes take and give model-space latents.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import threading
import time
import uuid
import weakref
from dataclasses import dataclass, field, replace
from typing import Any

from .image_wire import WireError, WireTensor, short_dtype

_log = logging.getLogger(__name__)

RUN_IDLE_S = 120.0
# How often the timer ends idle runs, and the most runs held at once.
SWEEP_S = 5.0
MAX_RUNS = 32
MAX_ROWS = 16
MAX_TENSOR_BYTES = 1 << 30
MAX_ADAPTER_BYTES = 1 << 30
# The bytes of contract table bodies a loaded pipeline keeps, to define a
# client's table again when the pipeline let its core table go.
CLIENT_TABLE_BYTES = 16 << 20
# The bound of a store with no cap, which its bookings bound.
UNBOUNDED = 1 << 62
FEATURES = ("prefix_cache", "noise_mask", "rgba_vae", "tiled_vae",
            "ref_cache", "skip_blocks", "av_step", "anchors",
            "keyframe_upsampling")


@dataclass(frozen=True)
class Convention:
    """How a step reads a timestep and returns its output under one
    client convention: ``sigma = scale * timestep + shift``, the DiT's
    time rule ``time`` (``denoise_step``) and ``out = sign * velocity``."""
    scale: float = 1.0
    shift: float = 0.0
    sign: float = 1.0
    time: str = "reference"


CONVENTIONS = {
    "comfy": {"z_image": Convention(), "qwen_image21": Convention(),
              "krea2": Convention(), "flux2_klein": Convention(time="fp32"),
              "flux2_klein_9b": Convention(time="fp32"),
              "flux2_klein_9b_kv": Convention(time="fp32"),
              "flux3_klein": Convention(),
              "moss_image": Convention(scale=0.001)},
}
# Under "reference" a step reads the timestep and returns its output as
# under "comfy", and embeds the time by the DiT's reference rule, as a
# direct run of gmlx does. ``gmlx image bench --via-steps`` steps on it.
CONVENTIONS["reference"] = {fam: replace(c, time="reference")
                            for fam, c in CONVENTIONS["comfy"].items()}

# What a client builds its model from, per family: ComfyUI's model class
# and image model name, the latent format and the VAE's. The VAE's
# latents on the wire are in the VAE's own space unless its ``space`` is
# "model".
COMFY = {
    "z_image": {"model_class": "ZImage", "image_model": "lumina2",
                "latent": {"channels": 16, "stride": 8, "patch": 2},
                "vae": {"channels_out": 3, "stride": 8}},
    "qwen_image21": {"model_class": "QwenImage21",
                     "image_model": "qwen_image21",
                     "latent": {"channels": 64, "stride": 16, "patch": 1},
                     "vae": {"channels_out": 4, "stride": 16}},
    "krea2": {"model_class": "Krea2", "image_model": "krea2",
              "latent": {"channels": 16, "stride": 8, "patch": 2},
              "vae": {"channels_out": 3, "stride": 8}},
    "flux2_klein": {"model_class": "Flux2", "image_model": "flux2",
                    "latent": {"channels": 128, "stride": 16, "patch": 1},
                    "vae": {"channels_out": 3, "stride": 16,
                            "space": "model"}},
    # ComfyUI builds the 9B and KV checkpoints as its Flux2 too, at the
    # width and depth its detection reads from the weights.
    "flux2_klein_9b": {"model_class": "Flux2", "image_model": "flux2",
                       "latent": {"channels": 128, "stride": 16, "patch": 1},
                       "vae": {"channels_out": 3, "stride": 16,
                               "space": "model"}},
    "flux2_klein_9b_kv": {"model_class": "Flux2", "image_model": "flux2",
                          "latent": {"channels": 128, "stride": 16,
                                     "patch": 1},
                          "vae": {"channels_out": 3, "stride": 16,
                                  "space": "model"}},
    "flux3_klein": {"model_class": "Flux3", "image_model": "flux3",
                    "latent": {"channels": 128, "stride": 16, "patch": 1},
                    "vae": {"channels_out": 3, "stride": 16,
                            "space": "model"}},
    # ComfyUI has no MOSS-Image model of its own: the pack registers
    # MOSSImage, on the FLUX.2 latent format, as the OpenMOSS plugin does.
    "moss_image": {"model_class": "MOSSImage", "image_model": "moss_image",
                   "latent": {"channels": 128, "stride": 16, "patch": 1},
                   "vae": {"channels_out": 3, "stride": 16,
                           "space": "model"}},
    # MiniMax-H3 is a video model with a sound track: ComfyUI's
    # MiniMaxH3 samples a nested latent of video and audio, which an AV
    # step carries. Both VAE routes carry the DiT's model space.
    "h3": {"model_class": "MiniMaxH3", "image_model": "minimax_h3",
           "latent": {"channels": 24, "stride": 16, "patch": 2,
                      "temporal": 4},
           # A long guide clip is encoded in windows of at most 7 pieces
           # of 17 frames, 124 frames at the clip's end: the longest clip
           # the video VAE's measured encode cells hold.
           "vae": {"channels_out": 3, "stride": 16, "space": "model",
                   "encode_pieces": 7},
           "audio": {"channels": 32, "stereo": 2, "sample_rate": 32000,
                     "latents_per_second": 40}},
}
# The families whose step is an AV step: a video and an audio stream.
AV_FAMILIES = frozenset({"h3"})


def _model_space(family: str) -> bool:
    """Whether the VAE routes of ``family`` carry model-space latents."""
    return COMFY.get(family, {}).get("vae", {}).get("space") == "model"


def flow_targets(sample_sigmas, timesteps) -> list:
    """The time each row's step goes to on a flow map DiT, as ComfyUI's
    FLUX.3 forward takes it: the entry of the client's schedule after the
    one nearest the row's timestep, the last at most."""
    import numpy as np

    s = np.asarray(sample_sigmas, np.float32).reshape(-1)
    if s.size < 2 or not np.isfinite(s).all():
        raise StepError(400, "sample_sigmas is a schedule of 2 or more "
                        "finite sigmas")
    out = []
    for t in timesteps:
        i = int(np.abs(s - np.float32(t)).argmin())
        out.append(float(s[min(i + 1, s.size - 1)]))
    return out


class StepError(Exception):
    """A step contract request refused with its status and a body."""

    # The Retry-After seconds of a 503, or None for a deferred load's.
    retry_after: int | None = None
    # A 503 that ends by itself, as models in use finish.
    transient: bool = False

    def __init__(self, status: int, message: str, **extra):
        super().__init__(message)
        self.status = status
        self.extra = extra

    def body(self) -> dict:
        return {"error": str(self), **self.extra}


class StepInterrupted(Exception):
    """The run of a step was interrupted or ended while it ran."""


class Missing(StepError):
    """A 409 naming what the server does not hold. ``hint`` says why, when
    the LoRA matrices of a table gave way to memory."""

    def __init__(self, hashes, hint: str | None = None):
        missing = sorted(set(hashes))
        text = (f"{len(missing)} tensor(s) or table(s) are not held by the "
                "server; upload them and retry")
        super().__init__(409, f"{text}. {hint}" if hint else text,
                         missing=missing)
        self.hint = hint


# The subjects of the step routes' LoRA refusals.
UPLOAD = "the LoRA matrices of this upload"
PAIRS = "the LoRA pairs of this patch table"
FACTORS = "the LoRA factors of this step"
ENCODE = "the LoRA bytes of this encode"
# A body with no Content-Length is booked this many bytes at a time.
BODY_STEP = 64 << 20

_WIRE_DTYPES = {"F32": "float32", "F16": "float16", "BF16": "bfloat16"}


def adapter_cap() -> int | None:
    """server.image.adapter_cache_mb in bytes, or None when unset."""
    from .image_service import cache_bytes

    return cache_bytes()[1]


def wire_pair_key(up: WireTensor, down: WireTensor) -> str | None:
    """The pair hash ``patch_pair`` gives the arrays of ``up`` and
    ``down``, from their wire bytes, or None for a dtype it takes not."""
    from gmlx.image.lora import install

    names = [_WIRE_DTYPES.get(t.dtype) for t in (down, up)]
    if None in names:
        return None

    def flat(t):
        s = tuple(int(d) for d in t.shape)
        return s if len(s) <= 2 else (s[0], math.prod(s[1:]))
    return install.pair_key((names[0], flat(down), down.raw),
                            (names[1], flat(up), up.raw))


def body_pairs(body: dict, adapters) -> dict:
    """{pair key: bytes} of the pairs of a contract table ``body`` whose
    matrices ``adapters`` holds."""
    out: dict = {}
    for e in body.get("entries") or []:
        for p in (e.get("patches") or []) if isinstance(e, dict) else []:
            if not isinstance(p, dict):
                continue
            up, down = (adapters.peek(h) if isinstance(h, str) else None
                        for h in (p.get("up"), p.get("down")))
            if up is None or down is None:
                continue
            key = wire_pair_key(up, down)
            if key is not None:
                out[key] = up.nbytes + down.nbytes
    return out


def pairs_due(served, pairs: dict) -> dict:
    """The pairs of ``pairs`` the pipeline holds not and that are not the
    entry's own: what a define of them puts."""
    from .image_service import _pair_keys

    own = _pair_keys(served.adapters)
    return {k: n for k, n in pairs.items()
            if k not in own and served.pipe.adapters.peek(k) is None}


def request_matrices(served, body: dict, adapters) -> dict:
    """{hash: bytes} of the matrices ``adapters`` holds of the pairs of a
    contract table ``body`` that are not the entry's own: what the
    request's LoRA holds at its forward, from the wire bytes."""
    from .image_service import _pair_keys

    own = _pair_keys(served.adapters)
    out: dict = {}
    for e in body.get("entries") or []:
        for p in (e.get("patches") or []) if isinstance(e, dict) else []:
            if not isinstance(p, dict):
                continue
            hs = [h for h in (p.get("up"), p.get("down"))
                  if isinstance(h, str)]
            ts = [adapters.peek(h) for h in hs]
            if len(ts) != 2 or None in ts:
                continue
            if wire_pair_key(ts[0], ts[1]) in own:
                continue
            out.update({h: t.nbytes for h, t in zip(hs, ts)})
    return out


def matrix_sizes(body: dict, adapters) -> dict:
    """{hash: bytes} of the matrices a contract table ``body`` names."""
    return {h: adapters.nbytes(h)
            for e in body.get("entries") or [] if isinstance(e, dict)
            for p in e.get("patches") or [] if isinstance(p, dict)
            for h in (p.get("up"), p.get("down"))
            if isinstance(h, str) and h in adapters}


def _wire_copy(t: WireTensor) -> WireTensor:
    return WireTensor(t.dtype, t.shape, memoryview(bytes(t.raw)))


def _mib(n: int, up: bool) -> str:
    """``n`` bytes in MiB at 1 decimal, rounded up when ``up`` and down
    otherwise. A 413 rounds the bytes it refuses up and its bound down, so
    they never read at or under the bound."""
    x = n / 2**20 * 10
    return f"{(math.ceil(x) if up else math.floor(x)) / 10:.1f} MiB"


class _Bytes:
    """A least recently used map of WireTensors bounded in bytes. Its 413
    names it by ``name`` and the server.image key that configures it."""

    def __init__(self, max_bytes: int | None, name: str, key: str):
        from collections import OrderedDict

        self.name, self.key = name, key
        # None bounds the store by nothing but its bookings.
        self.max_bytes = UNBOUNDED if max_bytes is None else int(max_bytes)
        # The configured bound, of which the models loaded leave the
        # cache max_bytes, or None.
        self.configured = None if max_bytes is None else self.max_bytes
        self._items: OrderedDict = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        # The keys a define or a step holds, by count: neither a shed nor
        # the bound lets them go.
        self._held: dict = {}

    def _shed_over(self, keep_one: bool) -> None:
        """Drop the oldest tensors not held while the store is over its
        bound, the newest kept with ``keep_one``."""
        newest = next(reversed(self._items), None) if keep_one else None
        for k in [k for k in self._items
                  if k not in self._held and k != newest]:
            if self._bytes <= self.max_bytes:
                return
            self._bytes -= self._items.pop(k).nbytes

    @contextlib.contextmanager
    def holding(self, keys):
        """Hold ``keys`` for the block: a shed and the bound skip them."""
        keys = list(keys)
        with self._lock:
            for k in keys:
                self._held[k] = self._held.get(k, 0) + 1
        try:
            yield
        finally:
            with self._lock:
                for k in keys:
                    n = self._held.get(k, 0) - 1
                    if n > 0:
                        self._held[k] = n
                    else:
                        self._held.pop(k, None)

    def get(self, key):
        with self._lock:
            t = self._items.get(key)
            if t is not None:
                self._items.move_to_end(key)
            return t

    def __contains__(self, key) -> bool:
        with self._lock:
            return key in self._items

    def peek(self, key):
        """The tensor at ``key``, or None, without a use."""
        with self._lock:
            return self._items.get(key)

    def nbytes(self, key) -> int:
        """Bytes of the tensor at ``key``, 0 without one, without a use."""
        with self._lock:
            t = self._items.get(key)
            return 0 if t is None else t.nbytes

    def put(self, key, t: WireTensor) -> None:
        self.check(t.nbytes)
        with self._lock:
            old = self._items.pop(key, None)
            self._items[key] = t
            # One write: the pool reads the count with no lock.
            self._bytes += t.nbytes - (0 if old is None else old.nbytes)
            self._shed_over(True)

    def shed(self, nbytes: int, hold=()) -> int:
        """Drop the least recently used tensors but those at ``hold`` and
        those held until ``nbytes`` are gone, and return the bytes
        dropped."""
        hold = set(hold)
        with self._lock:
            dropped = 0
            for k in [k for k in self._items
                      if k not in hold and k not in self._held]:
                if dropped >= nbytes:
                    break
                t = self._items.pop(k)
                self._bytes -= t.nbytes
                dropped += t.nbytes
            return dropped

    @property
    def bytes(self) -> int:
        return self._bytes

    def check(self, nbytes: int) -> None:
        """Refuse ``nbytes`` over the bound with a 413 that gives its
        remedy: the share the models loaded leave the cache, when the
        configured bound holds them, or else the configured bound. A
        store with no bound refuses nothing: the pool's room bounds it."""
        if self.configured is None or nbytes <= self.max_bytes:
            return
        configured = f"{self.configured / 2**20:.0f} MiB"
        if nbytes > self.configured:
            raise StepError(413, f"a tensor of {_mib(nbytes, True)} is over the "
                            f"configured {configured} of {self.name}; raise "
                            f"server.image.{self.key}")
        raise StepError(413, f"a tensor of {_mib(nbytes, True)} is over "
                        f"{self.name} of {_mib(self.max_bytes, False)}: the "
                        "models loaded leave "
                        f"it {self.max_bytes / self.configured:.0%} of its "
                        f"configured {configured}. A smaller defaults.size or "
                        "defaults.seconds on the tightest entry gives it more.")

    def bound(self, configured: int | None, max_bytes: int) -> None:
        """Set the bound at ``max_bytes`` of the ``configured`` one, or at
        none for a ``configured`` of None, and shed the oldest tensors
        over it."""
        with self._lock:
            if configured is None:
                self.configured, self.max_bytes = None, UNBOUNDED
            else:
                self.configured = int(configured)
                self.max_bytes = int(max_bytes)
            self._shed_over(False)

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
    gen: int = 0                       # an interrupt or end bumps it
    ended: bool = False
    interrupted: bool = False
    steps: int = 0
    rows: int = 0
    forward_s: float = 0.0
    cache_hits: int = 0
    conds: set = field(default_factory=set)         # keys, to shed at end
    cond_objs: dict = field(default_factory=dict)   # the run's state scope
    ref_cache: dict = field(default_factory=dict)   # denoise_step's, per run


class ClientTables:
    """The contract tables clients defined on one pipeline: each id's
    core table id and the body, as sent, it was defined from. A step on a
    core table the pipeline let go defines it again from its body. The
    least recently used ids go first past ``max_bytes`` of bodies, the
    newest one staying, and a step that names an id gone is a 409 the
    client answers by sending its table again. A body over ``max_bytes``
    is not kept: its id answers that 409 once its core table goes."""

    def __init__(self, max_bytes: int | None = None):
        from collections import OrderedDict

        self.max_bytes = int(CLIENT_TABLE_BYTES if max_bytes is None
                             else max_bytes)
        self._items: OrderedDict = OrderedDict()   # id -> (core, body)
        # id -> {matrix hash: bytes} of the body, from its define.
        self._sizes: dict = {}
        self._bytes = 0
        self._lock = threading.Lock()

    def put(self, tid: str, core: str, body: bytes | None = None,
            sizes: dict | None = None) -> None:
        """Keep ``tid`` as ``core``, with the body it was defined from and
        the bytes of the matrices it names. A body over the bound is not
        kept, so its table is not defined again at a step."""
        if body is not None and len(body) > self.max_bytes:
            body = None
        with self._lock:
            old = self._items.pop(tid, None)
            if old is not None and old[1] is not None:
                self._bytes -= len(old[1])
            self._items[tid] = (core, body)
            if sizes is not None:
                self._sizes[tid] = dict(sizes)
            self._bytes += len(body) if body is not None else 0
            while self._bytes > self.max_bytes and len(self._items) > 1:
                gid, (_core, gone) = self._items.popitem(last=False)
                self._bytes -= len(gone) if gone is not None else 0
                self._sizes.pop(gid, None)

    def __setitem__(self, tid: str, core: str) -> None:
        self.put(tid, core)

    def get(self, tid, default=None):
        with self._lock:
            got = self._items.get(tid)
            if got is None:
                return default
            self._items.move_to_end(tid)
            return got[0]

    def __getitem__(self, tid) -> str:
        got = self.get(tid)
        if got is None:
            raise KeyError(tid)
        return got

    def __contains__(self, tid) -> bool:
        with self._lock:
            return tid in self._items

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def body(self, tid) -> bytes | None:
        """The body ``tid`` was defined from, as sent, or None."""
        with self._lock:
            got = self._items.get(tid)
        return None if got is None else got[1]

    def sizes(self, tid) -> dict:
        """{matrix hash: bytes} of the body of ``tid``, as defined."""
        with self._lock:
            return dict(self._sizes.get(tid) or {})

    @property
    def bytes(self) -> int:
        return self._bytes


def _mb(mb: int | None) -> int | None:
    return None if mb is None else int(mb) << 20


def _tensor_cache(max_bytes: int) -> _Bytes:
    return _Bytes(max_bytes, "the step routes' tensor cache",
                  "tensor_cache_mb")


def _adapter_cache(max_bytes: int | None) -> _Bytes:
    return _Bytes(max_bytes, "the step routes' adapter cache",
                  "adapter_cache_mb")


class StepsState:
    """The state the step routes share. ``configure`` sets the cache
    sizes from ``server.image``."""

    def __init__(self):
        from gmlx.config import ImageServerCfg

        cfg = ImageServerCfg()
        self.tensors = _tensor_cache(cfg.tensor_cache_mb << 20)
        self.adapters = _adapter_cache(_mb(cfg.adapter_cache_mb))
        self._bounds = (self.tensors.max_bytes, self.adapters.configured)
        # The smallest cache share of the models loaded, and the largest
        # room one of them keeps in the tensor cache for a cond.
        self._share, self._tensor_floor = 1.0, 0
        self.runs: dict = {}
        # model id -> (weak reference to its pipeline, ClientTables)
        self.tables: dict = {}
        self.lock = threading.Lock()
        self.settings = None
        self.clock = time.monotonic
        # The server writes a log line per run and per refusal; the bench's
        # loopback routes write none, so its stdout stays JSON.
        self.log_requests = True
        self.sweep_s = SWEEP_S
        self._sweeper: threading.Thread | None = None

    def configure(self, settings) -> None:
        self.settings = settings
        tmb = settings.tensor_cache_mb << 20
        amb = _mb(settings.adapter_cache_mb)
        if tmb != self.tensors.max_bytes:
            self.tensors = _tensor_cache(tmb)
        if amb != self.adapters.configured:
            self.adapters = _adapter_cache(amb)
        self._bounds = (tmb, amb)

    def scale_caches(self, share: float) -> None:
        """Bound the two caches at ``share`` of their configured sizes, for
        the models loaded whose plans leave their caches only that room.
        The tensor cache keeps its floor."""
        self._share = float(share)
        self._bound()

    def floor_tensors(self, nbytes: int) -> None:
        """Keep the tensor cache at ``nbytes`` at any share, the room a
        loaded video model books there for one cond."""
        self._tensor_floor = int(nbytes)
        self._bound()

    def _bound(self) -> None:
        """The tensor store at the share, and at no less than its floor,
        which stands past the configured bound for the cond of a video
        entry's upsampler caption; the upload store at the cap alone, as
        its bookings bound it."""
        tensor, adapter = self._bounds
        self.tensors.bound(tensor, max(int(tensor * self._share),
                                       self._tensor_floor))
        self.adapters.bound(adapter, UNBOUNDED if adapter is None
                            else adapter)

    # Tensors and adapters.

    def put_tensor(self, store: _Bytes, want: str, tensors: dict,
                   limit: int, copy: bool = True) -> str:
        """Keep the one tensor of an upload under its hash. ``copy=False``
        keeps a view of the body, which an adapter upload books whole."""
        if len(tensors) != 1:
            raise StepError(400, f"an upload holds one tensor, not "
                            f"{len(tensors)}")
        (t,) = tensors.values()
        if t.nbytes > limit:
            raise StepError(413, f"a tensor of {_mib(t.nbytes, True)} is over "
                            f"the upload limit of {_mib(limit, False)}")
        store.check(t.nbytes)
        got = t.hash()
        if got != want:
            raise StepError(400, "the tensor's hash is not the one in the "
                            "path", got=got)
        store.put(got, _wire_copy(t) if copy else t)
        return got

    # Runs.

    def begin(self, model: str, hold) -> Run:
        """A new run on ``hold``. Refuses with 429 while ``MAX_RUNS`` runs
        are held, so clients that never end their runs cannot hold the
        model past ``RUN_IDLE_S``."""
        self.sweep()
        now = self.clock()
        run = Run(id=uuid.uuid4().hex[:16], model=model, hold=hold,
                  began=now, last=now)
        with self.lock:
            full = len(self.runs) >= MAX_RUNS
            if not full:
                self.runs[run.id] = run
        if full:
            raise StepError(429, f"{MAX_RUNS} runs are held; end a run or "
                            f"wait for one to sit idle past "
                            f"{RUN_IDLE_S:.0f} seconds")
        self._start_sweeper()
        return run

    def _start_sweeper(self) -> None:
        with self.lock:
            if self._sweeper is not None and self._sweeper.is_alive():
                return
            th = threading.Thread(target=self._sweep_loop, daemon=True,
                                  name="gmlx-image-runs")
            self._sweeper = th
        th.start()

    def _sweep_loop(self) -> None:
        stop = threading.Event()
        while True:
            stop.wait(self.sweep_s)
            try:
                self.sweep()
            except Exception:  # noqa: BLE001 - the timer must keep running
                _log.debug("run sweep failed", exc_info=True)
            with self.lock:
                if not self.runs:
                    self._sweeper = None
                    return

    def get_run(self, run_id: str | None) -> Run | None:
        if not run_id:
            return None
        with self.lock:
            run = self.runs.get(run_id)
        if run is None:
            raise StepError(404, f"no run {run_id!r}; it ended or sat idle "
                            f"past {RUN_IDLE_S:.0f} seconds")
        run.last = self.clock()
        if run.interrupted:
            raise StepInterrupted()
        return run

    def end(self, run: Run, reason: str = "end") -> None:
        with self.lock:
            if self.runs.pop(run.id, None) is None:
                return
        run.ended = True
        run.gen += 1
        run.ref_cache = {}
        # The hold's release drops its entry, so the states are shed first.
        if run.conds:
            served = getattr(getattr(run.hold, "_entry", None), "pipeline",
                             None)
            if served is not None and served.pipe is not None:
                from .image_service import WORKER

                WORKER.submit(shed_states, served.pipe, set(run.conds))
        try:
            if run.hold is not None:
                run.hold.release()
        except Exception:  # noqa: BLE001 - a release must not fail the end
            _log.debug("run hold release failed", exc_info=True)
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
        """Interrupt the run ``run_id``: its step in flight is discarded
        and its later steps are refused. A request that names no run
        interrupts nothing. Returns how many runs it reached."""
        if not run_id:
            return 0
        with self.lock:
            run = self.runs.get(run_id)
        if run is None:
            return 0
        run.interrupted = True
        run.gen += 1
        return 1

    # Patch tables.

    def tables_of(self, model: str, pipe) -> ClientTables:
        """The contract tables clients defined on ``pipe``, the loaded
        pipeline of ``model``. A pipeline loaded again starts an empty
        map. The map holds the pipeline weakly, so a model the pool
        releases leaves no pipeline alive here, and the next lookup drops
        its map with the table bodies kept in it."""
        with self.lock:
            for dead in [m for m, (ref, _t) in self.tables.items()
                         if ref() is None]:
                del self.tables[dead]
            held = self.tables.get(model)
            if held is None or held[0]() is not pipe:
                held = (weakref.ref(pipe), ClientTables())
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

def is_video(spec) -> bool:
    """Whether a resolved entry is a video model's."""
    return hasattr(spec, "audio_vae")


def family_of(spec) -> tuple:
    """``(family, variant)`` of a configured image or video model, from
    its header."""
    if is_video(spec):
        from gmlx.video.loader import identify_dit as identify_video

        vid = identify_video(spec.path, variant=getattr(spec, "variant", None))
        return vid.family, vid.variant
    from gmlx.image.loader import identify_dit

    ident = identify_dit(spec.path, variant=getattr(spec, "variant", None))
    return ident.family, ident.variant


def video_status(spec, resident: bool, plan=None) -> dict:
    """The status of a video model: its ComfyUI facts, the entry's own
    steps and shifts, the video sigmas of a variant trained on fixed
    timesteps at those steps, and its memory."""
    from gmlx.video import schedule
    from gmlx.video.memfit import file_bytes

    fam, variant = family_of(spec)
    facts = COMFY[fam.name]
    mem = {"dit_bytes": file_bytes(spec.path),
           "encoder_bytes": file_bytes(spec.encoder),
           "vae_bytes": file_bytes(spec.vae),
           "audio_vae_bytes": file_bytes(spec.audio_vae)}
    sampling = None
    if plan is not None:
        mem["resident_bytes"] = plan.footprint if resident else 0
        rungs = fam.rungs(variant, plan.steps)
        # ComfyUI's sampler steps the video sigma; its forward derives the
        # audio sigma from it at the shifts in force.
        sigmas = (list(map(float, schedule.sigmas(
            plan.steps, plan.shifts["video"], rungs))) if rungs else None)
        sampling = {"steps": plan.steps, "cfg": 1.0,
                    "shift": plan.shifts["video"],
                    "audio_shift": plan.shifts["audio"], "sigmas": sigmas,
                    "schedule": None, "neg_skip": []}
    return {"model": spec.id, "family": fam.name, "variant": variant,
            "resident": resident,
            "comfy": {"model_class": facts["model_class"],
                      "image_model": facts["image_model"]},
            "latent": dict(facts["latent"]), "vae": dict(facts["vae"]),
            "audio": dict(facts["audio"]), "edit_capable": False,
            "sampling": sampling, "compute_dtype": "bf16", "memory": mem}


def status(spec, resident: bool, plan=None) -> dict:
    from gmlx.image.dtypes import image_activation

    if is_video(spec):
        return video_status(spec, resident, plan)
    fam, variant = family_of(spec)
    facts = COMFY[fam.name]
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
            # A step takes the reference latents the client encoded on
            # the VAE route, so a DiT that reads references edits here.
            "edit_capable": fam.edit is not None,
            "sampling": sampling(fam, variant),
            "compute_dtype": str(act.dit).rsplit(".", 1)[-1].replace(
                "bfloat16", "bf16").replace("float16", "f16").replace(
                "float32", "f32"),
            "memory": mem}


def _qwen_image21_rule(fam) -> dict:
    c = fam.constants.values
    return {"rule": "qwen_image21", "base_len": c["base_image_seq_len"],
            "max_len": c["max_image_seq_len"], "base_shift": c["base_shift"],
            "max_shift": c["max_shift"], "terminal": c["shift_terminal"]}


# The schedule rule of each family whose schedule follows the image size,
# from which a client forms the sigmas of an image's tokens as the
# family's own schedule does: FLUX.2 klein's of get_schedule, and
# Qwen-Image 2.1's dynamic shift with its scheduler's constants.
SIZE_SCHEDULES = {"flux2_klein": lambda fam: {"rule": "flux2"},
                  "flux2_klein_9b": lambda fam: {"rule": "flux2"},
                  "flux2_klein_9b_kv": lambda fam: {"rule": "flux2"},
                  "qwen_image21": _qwen_image21_rule}


def sampling(fam, variant: str) -> dict | None:
    """The variant's own sampling: its steps and cfg, and for a variant
    distilled to a fixed schedule, those sigmas with the final zero, which
    a client samples on in place of its own schedule. ``schedule`` names
    the rule of a schedule that follows the image size, or is None."""
    row = fam.sampling_row(variant)
    if row is None:
        return None
    rule = SIZE_SCHEDULES.get(fam.name)
    return {"steps": row.steps, "cfg": row.cfg, "shift": row.shift,
            "sigmas": [*map(float, row.sigmas), 0.0] if row.sigmas else None,
            "schedule": rule(fam) if rule and not row.sigmas else None,
            "neg_skip": [int(i) for i in row.neg_skip]}


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
    from gmlx.video.registry import families as video_families

    from .image_wire import CONTRACT, MAX_BODY_BYTES

    return {"contract": {"major": CONTRACT[0], "minor": CONTRACT[1]},
            "min_client": {"major": 1, "minor": 0},
            "gmlx_version": __version__,
            "wire": {"container": "safetensors",
                     "dtypes": ["f32", "f16", "bf16", "u8"]},
            "families": sorted(f.name for f in (*families(),
                                                *video_families())
                               if f.name in COMFY),
            "conventions": sorted(CONVENTIONS),
            "patch_kinds": ["lora"],
            "features": list(FEATURES),
            "limits": {"max_tensor_bytes": min(MAX_TENSOR_BYTES,
                                               STATE.tensors.max_bytes),
                       "max_adapter_bytes": min(
                           MAX_ADAPTER_BYTES,
                           STATE.adapters.configured or MAX_ADAPTER_BYTES),
                       "max_rows": MAX_ROWS, "max_runs": MAX_RUNS,
                       "max_body_bytes": MAX_BODY_BYTES}}


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


def run_step(served, x, sigmas, conds, table, run, gen,
             targets=None, time_rule: str = "reference",
             ref_cache: bool = False, skip=()) -> tuple:
    """One ``denoise_step`` over the rows on the image worker, to the
    ``targets`` of a flow map DiT, on the DiT's ``time_rule``, with
    ``ref_cache`` on the run's cache of reference K and V, and with every
    row leaving out the blocks of ``skip``. Returns (velocity f32
    evaluated, forward seconds, prefix cache answer)."""
    import mlx.core as mx

    from gmlx.image import api
    from gmlx.image.pacing import paced

    pipe = served.pipe
    if run is not None and (run.gen != gen or run.ended):
        raise StepInterrupted()
    held = run.ref_cache if ref_cache else None
    cache = (_ref_answer(conds, held) if held is not None
             else _prefix_answer(pipe, conds, table, x.shape[2:], time_rule))
    t0 = time.perf_counter()
    with paced(step_pacer(run, gen)):
        v = api.denoise_step(pipe, x, sigmas, conds, patch_table=table,
                             target=targets, time=time_rule, ref_cache=held,
                             skip=skip)
        v = v.astype(mx.float32)
        mx.eval(v)
    # Without chat the step is one evaluation, which an interrupt or an end
    # cannot cut short: its result is discarded here.
    if run is not None and (run.gen != gen or run.ended):
        raise StepInterrupted()
    return v, time.perf_counter() - t0, cache


def _prefix_answer(pipe, conds, table, size,
                   time_rule: str = "reference") -> str:
    from gmlx.image import api

    if pipe.family.step is None or time_rule != "reference":
        return "none"
    if any(not c.key for c in conds):
        return "off"
    tid = api._table_id(pipe, table)
    keys = [api._state_key(pipe, c, tid, size) for c in conds]
    return "hit" if all(pipe.captions.peek(k) is not None
                        for k in keys) else "miss"


def _ref_answer(conds, held: dict) -> str:
    """The cache answer of a step on a run's reference K and V: "hit" when
    the run holds them for every row with references."""
    from gmlx.image import api

    refs = [api._ref_key(c, i) for i, c in enumerate(conds) if c.ref_latents]
    if not refs:
        return "none"
    return "hit" if all(k in held for k in refs) else "miss"


def _pair_room(served, need: int) -> None:
    """Refuse the pairs of a table, ``need`` bytes held at once, over the
    optional cap server.image.adapter_cache_mb with a 413 that names it.
    With no cap the pool books the pairs, and refuses them there."""
    cap = adapter_cap()
    if cap is None or need <= cap:
        return
    raise StepError(413, f"the LoRA pairs of this patch table take "
                    f"{_mib(need, True)}, over the {_mib(cap, False)} the "
                    f"pair cache of {served.spec.id!r} keeps for a request's "
                    f"pairs; raise server.image.adapter_cache_mb to "
                    f"{math.ceil(need / 2**20)} or more")


def _with_own(served, entries: list) -> list:
    """Table ``entries`` with the entry's own adapters stacked on, as the
    images API stacks a request's adapters on them. An own adapter that
    the entries name a pair of is left out, so the table's strength and
    alpha for it apply, once. ComfyUI reads no adapter header, so a pair
    whose file holds no alpha comes with none, where the entry's reader
    gives it the header's: the table's alpha is the one that applies."""
    from gmlx.image.lora import patch_table as pt

    if not served.adapters:
        return entries
    named = {p[0] for _m, _b, patches in entries for p in patches}
    own = [(ad, st) for ad, st in served.adapters
           if not any(r[-1] in named for r in ad.dit)]
    if not own:
        return entries
    rows = {(m, b): list(patches) for m, b, patches in entries}
    for m, b, patches in pt.table_entries(own):
        rows.setdefault((m, b), []).extend(patches)
    return [(m, b, patches) for (m, b), patches in rows.items()]


def define_table(served, body: dict, adapters: _Bytes) -> str:
    """The core table of a contract table body, with its pairs registered
    on the pipeline and the entry's own adapters stacked on. Raises
    :class:`Missing` for adapter matrices not held, and a 413 for pairs
    the entry's pair cache cannot hold at once beside its own."""
    from gmlx.image import api
    from gmlx.image.lora import install
    from gmlx.image.lora import patch_table as pt
    from gmlx.image.lora.reader import AdapterError

    from . import video_service as vs
    from .image_service import _pair_keys, ensure_dit

    # The table is defined on the DiT, which an encode lets go.
    if is_video(served.spec):
        vs.ensure_dit(served)
    else:
        ensure_dit(served)
    pipe = served.pipe
    entries = body.get("entries")
    if not isinstance(entries, list):
        raise StepError(400, "a patch table body needs a list of entries")
    missing = []
    held: dict = {}
    for e in entries:
        for p in (e.get("patches") or []) if isinstance(e, dict) else []:
            alpha = p.get("alpha")
            if alpha is not None and (isinstance(alpha, bool) or not
                                      isinstance(alpha, (int, float))):
                raise StepError(400, f"a patch's alpha is a number or null, "
                                f"not {json.dumps(alpha)}")
            got = [adapters.get(h) if isinstance(h, str) else None
                   for h in (p.get("up"), p.get("down"))]
            missing += [str(p.get(side)) for side, t in zip(("up", "down"),
                                                            got) if t is None]
            if None not in got:
                held[(p["up"], p["down"])] = got
    if missing:
        raise Missing(missing)
    # The cap's check, from the wire bytes, before a pair is made.
    own = _pair_keys(served.adapters)
    sizes = {wire_pair_key(up, down): up.nbytes + down.nbytes
             for up, down in held.values()}
    _pair_room(served, sum(n for k, n in sizes.items() if k not in own))
    try:
        # Each pair as the pipeline keeps it, under the content hash of
        # its down and up: a pair named twice, under any alphas, or one of
        # the entry's own takes no room.
        pairs = {pk: pt.patch_pair(up.mx(), down.mx())
                 for pk, (up, down) in held.items()}
    except (AdapterError, ValueError, TypeError) as e:
        raise StepError(400, f"patch table refused: {e}") from None
    rows = []
    keys: dict = {}
    try:
        for e in entries:
            row = {"key": e.get("key"), "band": e.get("band"), "patches": []}
            for p in e.get("patches") or []:
                unknown = set(p) - {"adapter", "up", "down", "alpha",
                                    "strength", "strength_model", "kind"}
                if unknown:
                    raise StepError(400, f"patch fields {sorted(unknown)} "
                                    "are not read by this server")
                pk = (p["up"], p["down"])
                if pk not in keys:
                    down, up, _alpha, key = pairs[pk]
                    keys[pk] = install.register_pair(
                        pipe, down, up, key=key, hold=keys.values())
                row["patches"].append({
                    "adapter": keys[pk], "alpha": p.get("alpha"),
                    "strength": p.get("strength", 1.0),
                    "strength_model": p.get("strength_model", 1.0),
                    **({"kind": p["kind"]} if "kind" in p else {})})
            rows.append(row)
        plans = pt.plans_from_patches(pipe.family, rows)
        return api.define_patch_table(pipe, _with_own(served, plans))
    except (AdapterError, ValueError, KeyError, TypeError) as e:
        if isinstance(e, StepError):
            raise
        if isinstance(e, install.PairRoomError):
            raise StepError(413, f"patch table refused: {e}") from None
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
           ref_latents: bool = True, typed: str | None = None) -> dict:
    """Encode ``text``, with reference images (H, W, 3) uint8 for a family
    that edits, on the image worker. Each image keeps its own area, so a
    client that sizes its references feeds the encoder and its VAE the
    same pixels. Returns the conditioning features f32 (1, L, D), the
    reference latents in model space (1, C, h, w), none when
    ``ref_latents`` is false, and the image slots. ``typed``, the prompt
    as typed beside an upsampled ``text``, adds its features as
    ``typed``."""
    import mlx.core as mx

    from gmlx.image import api

    from .image_service import _encode, ensure_vae

    pipe = served.pipe
    refs: tuple = ()
    if images:
        edit = pipe.family.edit
        if edit is None:
            raise StepError(400, f"{pipe.family.title} takes no reference "
                            "images")
        if edit.text_only:
            raise StepError(400, f"the {pipe.family.title} encoder reads the "
                            "prompt alone; a step takes the reference "
                            "latents the VAE route encodes")
        try:
            refs = api.references(pipe, [
                api.references(pipe, [im],
                               area=im.shape[0] * im.shape[1])[0]
                for im in images[:edit.max_refs + 1]])
        except ValueError as e:
            raise StepError(400, str(e)) from None
    cond, other = _encode(served, text, typed, refs, borrow)
    enc = pipe.encoder
    if enc is not None and enc.owned and not enc.released and \
            not served.plan.keep_encoder:
        enc.release()
    if refs and ref_latents:
        # A staged entry's encode let the VAE go.
        ensure_vae(served)
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
    extra = [] if other is None else [other.features.astype(mx.float32)[None]]
    mx.eval(feats, *lats, *extra)
    got = {"cond": _host(feats), "refs": [_host(z) for z in lats],
           "image_slots": slots}
    if extra:
        got["typed"] = _host(extra[0])
    return got


def _host(a):
    """An evaluated MLX array as a numpy array. MLX streams belong to the
    thread that made them, so arrays leave the image worker as numpy."""
    import numpy as np

    return np.array(a, copy=True)


def vae_decode(served, raw: WireTensor, alpha: bool):
    """Latents (B, C, h, w) in the VAE's space, or the model's for a
    family whose VAE facts say so, to an image (B, H, W, C) f32 in
    [0, 1], as numpy. The images decode one at a time, as the routes plan
    them."""
    import mlx.core as mx
    import numpy as np

    from gmlx.image import api

    from .image_service import ensure_vae

    ensure_vae(served)
    pipe = served.pipe
    vae = pipe.vae
    model = _model_space(pipe.family.name)
    z = _f32(raw) if model else _f32(raw).transpose(0, 2, 3, 1)
    out = []
    for i in range(z.shape[0]):
        one = z[i:i + 1]
        img = api.vae_decode(pipe, one if model else vae.to_model_space(one),
                             alpha=alpha)
        mx.eval(img)
        out.append(_host(img.astype(mx.float32)))
    return np.concatenate(out, axis=0)


def vae_encode(served, image: WireTensor):
    """An image (B, H, W, C) f32 in [0, 1] to latents (B, C, h, w) f32 in
    the VAE's space, or the model's for a family whose VAE facts say so,
    as numpy."""
    import mlx.core as mx

    from gmlx.image import api

    import numpy as np

    from .image_service import ensure_vae

    ensure_vae(served)
    pipe = served.pipe
    img = _f32(image)
    if pipe.vae.config.in_channels == 3:
        img = img[..., :3]
    keep = _model_space(pipe.family.name)
    out = []
    for i in range(img.shape[0]):
        model = api.vae_encode(pipe, img[i:i + 1])
        raw = model if keep else pipe.vae.to_vae_space(model).transpose(
            0, 3, 1, 2)
        mx.eval(raw)
        out.append(_host(raw.astype(mx.float32)))
    return np.concatenate(out, axis=0)


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
