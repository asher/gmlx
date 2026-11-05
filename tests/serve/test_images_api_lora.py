"""The images API's LoRA bytes, booked with the residency pool: a
reservation on the route before the request's budget is read, and two
turns in the job, the pairs before a file is read and the factors after
the encode. A tiny Z-Image pipeline is the pool's image entry; the other
models are stand-ins with a footprint and no weights."""
from __future__ import annotations

import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")
mx = pytest.importorskip("mlx.core")

from fastapi.testclient import TestClient  # noqa: E402

from gmlx.config import ImageServerCfg, build_config  # noqa: E402
from gmlx.image import api  # noqa: E402
from gmlx.serve import bridge_vlm as serving  # noqa: E402
from gmlx.serve import capacity, residency  # noqa: E402
from gmlx.serve import image_service as svc  # noqa: E402
from gmlx.serve import image_steps as steps  # noqa: E402
from gmlx.serve.lora_books import STORE  # noqa: E402
from gmlx.serve.patches import images  # noqa: E402
from gmlx.serve.queue_cap import _RETRY_MIN_S  # noqa: E402
from gmlx.serve.residency import IMAGE_KIND, _BusyHold  # noqa: E402
from test_image_routes import _APP, _DOC  # noqa: E402
from test_image_steps import _client as _steps_client  # noqa: E402
from test_step_routes_lora import (GB, _crowd, _model, _mount,  # noqa: E402
                                   _pool, _PoolHost, _racing_dry_run,
                                   _store)

MB = 10**6


def _spec(**kw):
    base = dict(id="zit", path="/dit.gguf", family=None, variant=None,
                defaults={}, adapters=(), vae=None, encoder=None,
                encoder_model=None, encoder_mmproj=None,
                encoder_release="auto")
    return SimpleNamespace(**(base | kw))


def _cond():
    rng = np.random.default_rng(0)
    return api.Conditioning(
        mx.array(rng.standard_normal((5, 16)).astype(np.float32)), "k1")


def _own_lora() -> dict:
    """The entry's own LoRA, in bfloat16 as LoRA files ship: rank 16 on q,
    k and v of both blocks, three bands of each block's fused qkv leaf.
    Its kernel pairs build at the DiT's float32 and take more than the
    footprint's own term leaves beside its pairs."""
    from image_fixtures import TINY_Z_CONFIG as c
    from lora_fixtures import make_pair

    dim = c["dim"]
    rng = np.random.default_rng(1)
    return {f"layers.{i}.attention.to_{x}": make_pair(rng, dim, dim, 16, 16.0)
            for i in (0, 1) for x in "qkv"}


def _entry(request, monkeypatch, tmp_path, *, budget: int = 4 * GB,
           fp: int = GB, own: bool = True, staged: bool = False,
           cap_mb=None, dit=None):
    """Image entry "zit" of a tiny Z-Image DiT, or of ``dit()``, of ``fp``
    bytes in a pool of ``budget``, with a LoRA of its own unless ``own``
    is False, served on the images API with request LoRAs a and b in its
    model folder and the adapter cap ``cap_mb``. A staged entry's encode
    lets its DiT go, as a staged encode does. Returns a namespace of the
    pool, the entry, its pipeline, the model folder, the encodes run and
    each fit check's budget with the entry's pending bytes then."""
    from image_fixtures import tiny_z_model
    from lora_fixtures import dialect_keys, tiny_z_adapter, write_adapter

    from gmlx.image import loader, vae
    from gmlx.image.registry import get_family

    root = tmp_path / "models"
    root.mkdir()
    mx.save_safetensors(str(root / "own.safetensors"), {
        k: mx.array(v).astype(mx.bfloat16)
        for k, v in dialect_keys("kohya", _own_lora()).items()})
    for name, seed in (("a", 11), ("b", 12)):
        write_adapter(root / f"{name}.safetensors", "kohya",
                      tiny_z_adapter(seed, 4.0, wide=True))
    monkeypatch.setattr(svc, "_CACHE_BYTES", dict(svc._CACHE_BYTES))
    monkeypatch.setattr(svc, "_SHARES", {})
    make = dit or (lambda: tiny_z_model()[0])
    monkeypatch.setattr(api, "load", lambda path, **kw: api.ImagePipeline(
        dit=make(), family=get_family("z_image"), variant="turbo"))
    monkeypatch.setattr(loader, "load_dit", lambda path, **k: (make(), None))
    monkeypatch.setattr(vae, "load_vae", lambda path, **k: None)
    monkeypatch.setattr(svc, "entry_plan", lambda spec: svc.EntryPlan(
        footprint=fp, keep_encoder=True, family="z_image", variant="turbo",
        size=(64, 48), cfg=1.0, staged=staged))
    pool = _pool(monkeypatch, budget)
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: budget)
    cfg = SimpleNamespace(image=ImageServerCfg(adapter_cache_mb=cap_mb),
                          model_dirs=(str(root),), port=None)
    monkeypatch.setattr(svc, "default_store_root",
                        lambda port=None: str(tmp_path / "files"))
    svc.configure_caches(cfg.image)
    spec = _spec(adapters=((str(root / "own.safetensors"), 1.0),)
                 if own else ())
    served = svc._load(spec)
    request.addfinalizer(lambda: svc._release([served]))
    e = pool.acquire(spec.path, None, IMAGE_KIND, loaded_as="zit",
                     build_spec=SimpleNamespace(fp=fp, stream=None,
                                                path=spec.path, pipe=served))
    pool.release(e)
    rec = SimpleNamespace(pool=pool, e=e, pipe=served.pipe, root=root,
                          served=served, encodes=0, at_encode=None,
                          latents=[], budgets=[])

    def acquire(model):
        with pool._lock:
            e.busy += 1
        return _BusyHold(pool, e)

    def encode(served_, text, neg, refs, borrow):
        rec.encodes += 1
        if staged:
            svc._drop_dit(served_)
        if rec.at_encode is not None:
            rec.at_encode()
        return _cond(), None

    serving.register_resolved_models(build_config(_DOC))
    request.addfinalizer(serving.clear_resolved_models)
    monkeypatch.setattr(residency, "acquire_image_model", acquire)
    monkeypatch.setattr(svc, "check_family", lambda req, s: None)
    monkeypatch.setattr(svc, "target_size", lambda req, s: (64, 48))
    monkeypatch.setattr(svc, "fit_check", lambda req, s, size, b:
                        rec.budgets.append((b, pool.lora_held(e)[1])))
    monkeypatch.setattr(svc, "_encode", encode)
    monkeypatch.setattr(api, "plan_request", lambda *a, **k: SimpleNamespace(
        release_encoder=False, rows=1))
    monkeypatch.setattr(api, "vae_decode", lambda pipe, x, *, alpha, plan:
                        mx.zeros((1, 48, 64, 3)))
    sample = api.sample

    def sampled(*a, **k):
        x = sample(*a, **k)
        rec.latents.append(np.array(x))
        return x

    monkeypatch.setattr(api, "sample", sampled)
    # The endpoint binds the job's checks as it is installed.
    images.install_image_routes(cfg)
    return rec


def _post(adapters=(), **fields):
    """One images API request; returns the status, the answer and the
    headers. The VAE is a stub: the job's latents are in the entry's
    ``latents``."""
    body = {"prompt": "a lighthouse", "steps": 2, "seed": 7,
            "adapters": [{"id": a, "strength": s} for a, s in adapters],
            **fields}
    r = TestClient(_APP.app, base_url="http://127.0.0.1").post(
        "/v1/images/generations", json=body)
    return r.status_code, r.json(), r.headers


def _books_hold(pool, e, *, budget: bool = True) -> None:
    """The entry's ledger is the keys its caches hold beyond its own, at
    their bytes, and its own keys are the load's and its default table's,
    their excess over the footprint's own term booked. Nothing is
    pending, no byte is past its booking, and the pool is within its
    budget."""
    pipe = e.pipeline.pipe
    with pool._lock:
        pool.books.drain()
        rec = pool.books.rec(e)
        own, ledger = dict(rec.own), dict(rec.ledger)
        pending, excess, held = rec.pending, rec.excess, rec.held
        term = rec.own_term
    cached = {k: c.nbytes(k) for c in (pipe.adapters, pipe.tables)
              for k in c.keys(lambda k: True)}
    assert {k: n for k, n in cached.items() if k not in own} == ledger
    assert {k: n for k, n in cached.items() if k in own} == own
    assert excess == max(0, sum(own.values()) - term)
    assert (held, pending) == (sum(ledger.values()) + excess, 0)
    assert pool.overruns == []
    assert pool._resident_bytes() <= pool._budget or not budget


def _crowd_in_use(t) -> None:
    """``_crowd`` while the entry is in use, as another request of it
    holds it: the crowd evicts no model."""
    with t.pool._lock:
        t.e.busy += 1
    _crowd(t.pool)
    t.pool.release(t.e)


def _on_worker(fn):
    return svc.WORKER.submit(fn).result()


def _after_route(monkeypatch, fn) -> list:
    """``fn`` on the image worker once, right after a request's route has
    booked it and read its budget, as another job would run there.
    Returns the route's bookings."""
    book, budget, seen = images._book, images._budget, []

    def booked(hold, served, req):
        got = book(hold, served, req)
        seen.append(got)
        return got

    def then(hold, pairs=0):
        got = budget(hold, pairs)
        if len(seen) == 1:
            seen.append("ran")
            _on_worker(fn)
        return got

    monkeypatch.setattr(images, "_book", booked)
    monkeypatch.setattr(images, "_budget", then)
    return seen


def _big_lora(path, nbytes: int) -> str:
    """A safetensors file whose header names one LoRA factor of ``nbytes``
    bytes, written sparse: a refusal from its header reads none of it."""
    head = json.dumps({"lora_unet_layers_0_feed_forward_w1.lora_down.weight":
                       {"dtype": "F32", "shape": [nbytes // 4],
                        "data_offsets": [0, nbytes]}}).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(head)) + head)
        f.truncate(8 + len(head) + nbytes)
    return str(path)


# T1: a LoRA over the old 256 MiB bound, with the cap unset.

def _wide_cfg(dim: int) -> dict:
    """A Z-Image config of one block of each kind at width ``dim``, with a
    narrow feed-forward: its attention leaves are those of a full DiT,
    and its weights a few hundred MB."""
    from image_fixtures import TINY_Z_CONFIG

    head = dim // 30
    return dict(TINY_Z_CONFIG, dim=dim, n_heads=30, n_kv_heads=30,
                head_dim=head, axes_dims=[head // 4, 3 * head // 8,
                                          3 * head // 8],
                n_layers=1, n_refiner_layers=1, ffn_dim=256)


_STEMS = [f"{b}.attention.to_{x}" for b in
          ("layers.0", "noise_refiner.0", "context_refiner.0")
          for x in ("q", "k", "v", "out.0")]


def _wide_dit(dim: int):
    """A DiT of ``_wide_cfg(dim)`` in bfloat16."""
    from gmlx.image.families.z_image.model import ZImageDiT

    m = ZImageDiT(_wide_cfg(dim))
    m.set_dtype(mx.bfloat16)
    mx.eval(m.parameters())
    mx.clear_cache()
    m.eval()
    return m


def _wide_lora(path, dim: int, rank: int) -> int:
    """A bfloat16 LoRA of ``rank`` on q, k, v and the output of each
    block's attention at width ``dim``; returns its pair bytes."""
    keys = {}
    for i, stem in enumerate(_STEMS):
        k = mx.random.key(i)
        name = "lora_unet_" + stem.replace(".", "_")
        keys[name + ".lora_down.weight"] = (mx.random.normal(
            (rank, dim), key=k) * 0.01).astype(mx.bfloat16)
        keys[name + ".lora_up.weight"] = (mx.random.normal(
            (dim, rank), key=mx.random.split(k)[0]) * 0.01).astype(
            mx.bfloat16)
        keys[name + ".alpha"] = mx.array(float(rank))
    mx.save_safetensors(str(path), keys, metadata={"format": "pt"})
    n = sum(int(v.nbytes) for k, v in keys.items()
            if k.endswith(".weight"))
    del keys
    mx.clear_cache()
    return n


T1_DIM, T1_RANK = 3840, 1536


def _over_the_old_bound(request, monkeypatch, tmp_path, dim, rank):
    t = _entry(request, monkeypatch, tmp_path, budget=16 * GB, own=False,
               dit=lambda: _wide_dit(dim))
    pairs = _wide_lora(t.root / "wide.safetensors", dim, rank)
    status, _doc, _h = _post([("wide.safetensors", 1.0)])
    assert status == 200 and np.isfinite(t.latents[-1]).all()
    assert t.pipe.adapters.bytes == pairs and t.pipe.tables.bytes > 0
    _books_hold(t.pool, t.e)
    return pairs


def test_a_lora_over_256_mb_runs_on_the_images_api_with_the_cap_unset(
        request, monkeypatch, tmp_path):
    """A request LoRA of 270 MiB of pairs on attention leaves of a full
    DiT's width runs on the images API with server.image.adapter_cache_mb
    unset and the memory free: its pairs and built factors are booked as
    they arrive, the books equal what the caches hold, and no byte is
    past its booking."""
    pairs = _over_the_old_bound(request, monkeypatch, tmp_path, T1_DIM,
                                T1_RANK)
    assert pairs > 256 << 20


def test_a_lora_over_a_cap_of_256_mb_gets_the_refusal_naming_it(
        request, monkeypatch, tmp_path):
    """With server.image.adapter_cache_mb set to 256, the same LoRA gets
    the images API's refusal that names the setting, from its file's
    size before it is read: no encode, nothing booked. The cap adds
    nothing to the entry's footprint (T3's plan test, at caps None, 1 and
    256)."""
    t = _entry(request, monkeypatch, tmp_path, budget=16 * GB, own=False,
               cap_mb=256, dit=lambda: _wide_dit(T1_DIM))
    _wide_lora(t.root / "wide.safetensors", T1_DIM, T1_RANK)
    status, doc, _h = _post([("wide.safetensors", 1.0)])
    assert (status, doc["error"]["param"]) == (400, "adapters")
    assert doc["error"]["message"] == (
        "the request's adapters take 270.1 MiB, more than the 256.0 MiB "
        "this image model keeps for request adapters; raise "
        "server.image.adapter_cache_mb")
    assert t.encodes == 0 and t.pool.lora_held(t.e) == (0, 0, {})
    assert t.pipe.adapters.bytes == 0


def _over_the_old_bound_on_the_steps(request, monkeypatch, dim, rank):
    """A client's LoRA of ``rank`` at width ``dim`` in float32, as the pack
    sends one, through the step routes: its matrices uploaded, its table
    defined and one step on it. Returns its matrices' bytes."""
    from gmlx.image.families.z_image.lora_map import lora_map
    from gmlx.image.registry import get_family
    from gmlx.image.samplers import noise_from_seed
    from test_image_steps import SHAPE, _features, _rows
    from test_step_routes_lora import _books_hold as _step_books_hold

    pool = _pool(monkeypatch, 16 * GB)
    pipe = _on_worker(lambda: api.ImagePipeline(
        dit=_wide_dit(dim), family=get_family("z_image"), variant="turbo"))
    cfg = ImageServerCfg(adapter_cache_mb=None)
    monkeypatch.setattr(svc, "_CACHE_BYTES", dict(svc._CACHE_BYTES))
    svc.configure_caches(cfg)
    host = _mount(request, _PoolHost({"wide": pipe}, pool), cfg)
    targets = lora_map(_wide_cfg(dim)).targets
    c = _steps_client()
    rng = np.random.default_rng(0)
    entries, sent = [], 0
    for stem in _STEMS:
        module, band = targets[stem]
        down = (rng.standard_normal((rank, dim)) * 0.01).astype(np.float32)
        up = (rng.standard_normal((dim, rank)) * 0.01).astype(np.float32)
        patch = {"up": c.put(up, kind="adapters"),
                 "down": c.put(down, kind="adapters"), "alpha": None,
                 "strength": 1.0}
        sent += up.nbytes + down.nbytes
        del up, down
        entries.append({"key": f"diffusion_model.{module}.weight",
                        "band": None if band is None else list(band),
                        "patches": [patch]})
    tid = c.define_table("wide", {"entries": entries})
    v = c.step("wide", None, noise_from_seed(3, SHAPE), [0.75],
               _rows(c, _features()), patch_table=tid)
    assert np.isfinite(v).all()
    assert pipe.adapters.bytes == sent and pipe.tables.bytes > 0
    _step_books_hold(pool, host.entry("wide"))
    return sent


def test_a_lora_over_256_mb_runs_on_the_step_routes_with_the_cap_unset(
        request, monkeypatch):
    """The same over the step routes, as the pack sends it: 270 MiB of
    float32 matrices at rank 768 uploaded, defined and stepped with the
    cap unset and the memory free, the books equal to what the store and
    the caches hold."""
    sent = _over_the_old_bound_on_the_steps(request, monkeypatch, T1_DIM,
                                            T1_RANK // 2)
    assert sent > 256 << 20


# T2: the refusals.

def test_a_lora_no_unload_makes_room_for_is_413_with_its_gb(
        request, monkeypatch, tmp_path):
    """A 3 GB LoRA file, priced with its factors at 9 GB, on a server
    whose budget leaves 7.4 GB beside the entry: 413 from the file's
    header, with both figures in decimal GB. Nothing is read or booked."""
    t = _entry(request, monkeypatch, tmp_path, budget=8_400_000_000,
               own=False)
    _big_lora(t.root / "big.safetensors", 3 * GB)
    status, doc, _h = _post([("big.safetensors", 1.0)])
    assert (status, doc["error"]["message"]) == (
        413, "the request's adapters need 9.0 GB of memory, over the 7.4 GB "
        "the server's memory budget leaves beside 'zit' and the pinned "
        "models")
    assert doc["error"]["param"] == "adapters"
    assert t.encodes == 0 and t.pool.lora_held(t.e) == (0, 0, {})


def test_a_lora_a_busy_model_holds_the_room_for_is_503_with_retry_after(
        request, monkeypatch, tmp_path):
    """The same LoRA beside a model in use that leaves 8.6 GB: a 503 that
    names the models in use, with Retry-After as a deferred load has it.
    Once that model is idle, a request of LoRA a runs beside it."""
    t = _entry(request, monkeypatch, tmp_path, budget=20 * GB, own=False)
    _big_lora(t.root / "big.safetensors", 3 * GB)
    busy = _model(t.pool, "qwen3-32b", 10_400_000_000, busy=True)
    status, doc, headers = _post([("big.safetensors", 1.0)])
    assert (status, doc["error"]["message"]) == (
        503, "the request's adapters need 9.0 GB of memory, and 8.6 GB is "
        "free beside the models in use (zit, qwen3-32b); retry when one is "
        "idle")
    assert headers["retry-after"] == str(_RETRY_MIN_S)
    assert t.encodes == 0 and t.pool.lora_held(t.e) == (0, 0, {})
    t.pool.release(busy)
    status, doc, _h = _post([("a.safetensors", 1.0)])
    assert status == 200 and doc["data"]
    _books_hold(t.pool, t.e)


def test_an_idle_model_holding_the_room_is_evicted_at_the_route(
        request, monkeypatch, tmp_path):
    """A request LoRA that fits only once an idle model goes: the route's
    reservation evicts that model, as a cold load does, and the request
    runs on the room it booked."""
    import os

    t = _entry(request, monkeypatch, tmp_path, own=False)
    size = os.path.getsize(t.root / "a.safetensors")
    other = _model(t.pool, "qwen3-32b", 3 * GB - size // 2, busy=False)
    assert other in t.pool._entries.values()
    status, doc, _h = _post([("a.safetensors", 1.0)])
    assert status == 200 and doc["data"]
    assert other not in t.pool._entries.values()
    assert t.pipe.adapters.bytes > 0
    _books_hold(t.pool, t.e)


def test_the_route_reads_its_budget_after_its_reservation(
        request, monkeypatch, tmp_path):
    """The route books a request's LoRA before it reads its budget: the
    fit check gets the working budget less the reservation's factors,
    with the pairs left to the plan, which counts them once they load."""
    t = _entry(request, monkeypatch, tmp_path, own=False)
    pairs = []
    book = images._book

    def booked(hold, served, req):
        got = book(hold, served, req)
        pairs.append(got[1])
        return got

    monkeypatch.setattr(images, "_book", booked)
    assert _post([("a.safetensors", 1.0)])[0] == 200
    (budget, pending), = t.budgets
    factors = pending - pairs[0]
    assert pairs[0] > 0 and factors > 0
    assert budget == 4 * GB - factors
    _books_hold(t.pool, t.e)


# T3 at the pool.

@pytest.mark.parametrize("cap_mb", [None, 256])
def test_a_model_with_no_lora_books_no_lora_memory(request, monkeypatch,
                                                   tmp_path, cap_mb):
    """An entry with no LoRA, and a request with none, book nothing: the
    ledger is empty after the load and after the job, and the pool's
    resident bytes are the footprints, with or without a cap."""
    t = _entry(request, monkeypatch, tmp_path, own=False, cap_mb=cap_mb)
    assert t.pool.lora_held(t.e) == (0, 0, {})
    status, _doc, _h = _post()
    assert status == 200
    assert t.pool.lora_held(t.e) == (0, 0, {})
    assert t.pool._resident_bytes() == sum(
        e.footprint for e in t.pool._entries.values()) == GB
    _books_hold(t.pool, t.e)


# T24: the turn bookings of the images API.

def test_a_request_with_no_adapters_books_the_own_tables_excess(
        request, monkeypatch, tmp_path):
    """The entry's own table runs a request with no adapters. Its pairs
    and factors past the footprint's own term are booked, the same
    request again books nothing more, and the books equal what the caches
    hold. With the memory taken, it answers 503 and builds nothing."""
    t = _entry(request, monkeypatch, tmp_path)
    status, _doc, _h = _post()
    assert status == 200 and t.pipe.tables.bytes > 0
    excess = t.pool.books.excess(t.e)
    assert excess > 0
    _books_hold(t.pool, t.e)
    assert _post()[0] == 200
    assert np.array_equal(t.latents[-1], t.latents[0])
    assert t.pool.books.excess(t.e) == excess
    _books_hold(t.pool, t.e)
    # Another job's claim takes the built table, and a model in use the
    # memory free.
    _on_worker(lambda: svc.shed_lora(t.e, t.pipe.tables.bytes, own=True))
    assert t.pipe.tables.bytes == 0
    _crowd_in_use(t)
    status, doc, headers = _post()
    assert status == 503 and headers["retry-after"] == str(_RETRY_MIN_S)
    assert doc["error"]["message"].startswith(
        "the request's adapters need 0.1 GB of memory, and 0.0 GB is free "
        "beside the models in use (")
    assert t.pipe.tables.bytes == 0
    _books_hold(t.pool, t.e, budget=False)


def test_the_own_adapter_at_another_strength_is_a_table_booked_in_full(
        request, monkeypatch, tmp_path):
    """A request that names the entry's own LoRA at 0.6 runs a table of
    its own: its factors are booked in full beside the own table's, the
    same request again books nothing more, and it gives another image
    than the own table at 1.0."""
    t = _entry(request, monkeypatch, tmp_path)
    assert _post()[0] == 200
    assert _post([("own", 0.6)])[0] == 200
    plain, weaker = t.latents
    assert not np.allclose(plain, weaker)
    _held, _pending, ledger = t.pool.lora_held(t.e)
    assert ledger and all(isinstance(k, tuple) for k in ledger)
    _books_hold(t.pool, t.e)
    assert _post([("own", 0.6)])[0] == 200
    assert np.array_equal(t.latents[-1], weaker)
    assert t.pool.lora_held(t.e)[2] == ledger
    _books_hold(t.pool, t.e)


@pytest.mark.parametrize("room", [True, False])
def test_a_staged_entrys_encode_drops_its_tables_and_its_turn_books_them(
        request, monkeypatch, tmp_path, room):
    """On a staged entry the encode lets the DiT and its built tables go
    inside the job. A request of LoRA a again prices its factors in full
    on its route and builds them after the encode, with the same image
    and the books equal to what the caches hold. With the memory taken
    after a first request let the DiT go, its route answers 503 and the
    job never runs."""
    t = _entry(request, monkeypatch, tmp_path, staged=True)
    assert _post([("a.safetensors", 1.0)])[0] == 200
    built = t.pipe.tables.bytes
    assert built > 0
    _books_hold(t.pool, t.e)
    if not room:
        _on_worker(lambda: svc._drop_dit(t.served))
        _crowd_in_use(t)
        encodes = t.encodes
        status, doc, headers = _post([("a.safetensors", 1.0)])
        assert status == 503 and headers["retry-after"]
        assert t.encodes == encodes and t.pipe.tables.bytes == 0
        _books_hold(t.pool, t.e, budget=False)
        return
    assert _post([("a.safetensors", 1.0)])[0] == 200
    assert t.encodes == 2
    assert np.array_equal(t.latents[-1], t.latents[0])
    # A rebuilt kernel table counts a little less than its first build.
    assert 0 < t.pipe.tables.bytes <= built
    _books_hold(t.pool, t.e)


@pytest.mark.parametrize("room", [True, False])
def test_a_pair_held_at_the_route_and_shed_before_the_job_is_booked_again(
        request, monkeypatch, tmp_path, room):
    """The route of a request of LoRA a finds its pairs and its built
    table held and books nothing; another job sheds both before the job
    runs. The job books the pairs at its first turn and the factors at its
    second, and gives the same image, the books equal to what the caches
    hold. With the memory taken it answers 503 at its first turn and reads
    no pair."""
    t = _entry(request, monkeypatch, tmp_path)
    assert _post([("a.safetensors", 1.0)])[0] == 200
    own = sum(t.pipe.adapters.nbytes(k) for k in t.pipe.adapters.kept)
    pairs, built = t.pipe.adapters.bytes, t.pipe.tables.bytes
    assert pairs > own and built > 0

    def shed():
        svc.shed_lora(t.e, t.pipe.adapters.bytes + t.pipe.tables.bytes)
        assert t.pipe.adapters.bytes == own
        if not room:
            _crowd(t.pool)

    booked = _after_route(monkeypatch, shed)
    status, doc, headers = _post([("a.safetensors", 1.0)])
    assert booked == [(None, 0), "ran"]
    if room:
        assert status == 200
        assert np.array_equal(t.latents[-1], t.latents[0])
        assert t.pipe.adapters.bytes == pairs
        _books_hold(t.pool, t.e)
        return
    assert status == 503 and headers["retry-after"] == str(_RETRY_MIN_S)
    assert doc["error"]["message"].startswith(
        "the request's adapters need 0.1 GB of memory, and 0.0 GB is free ")
    assert t.pipe.adapters.bytes == own
    _books_hold(t.pool, t.e, budget=False)


@pytest.mark.parametrize("gone", ["pair", "dit"])
def test_a_request_whose_table_goes_during_its_routes_dry_run_is_priced(
        request, monkeypatch, tmp_path, gone):
    """Another job sheds a request table's pairs and built table, or a
    staged entry's encode lets the DiT go, on the worker while the
    request's route prices the table by the dry run. The route prices it
    at twice its pairs, as any other request table, the job's turns book
    it exactly, and it gives the same image."""
    t = _entry(request, monkeypatch, tmp_path, staged=gone == "dit")
    assert _post([("a.safetensors", 1.0)])[0] == 200
    if gone == "pair":
        def fn():
            svc.shed_lora(t.e, t.pipe.adapters.bytes + t.pipe.tables.bytes)
    else:
        def fn():
            svc._drop_dit(t.served)
    ran = _racing_dry_run(monkeypatch, fn)
    status, doc, _headers = _post([("a.safetensors", 1.0)])
    assert ran and status == 200, doc
    assert np.array_equal(t.latents[-1], t.latents[0])
    _books_hold(t.pool, t.e)


# T20: the request budget.

def _fit_entry(pipe_bytes: int):
    """A served entry for a request plan, whose pair and tables caches
    hold ``pipe_bytes``."""
    nb = SimpleNamespace(bytes=0)
    pipe = SimpleNamespace(
        encoder=None, dit=SimpleNamespace(act_dtype="bfloat16"),
        weights={"dit": 40 * MB, "vae": 8 * MB},
        adapters=SimpleNamespace(bytes=pipe_bytes), tables=nb,
        family=SimpleNamespace(name="z_image", edit=None, title="Z-Image",
                               step=None),
        vae=None, conditionings=nb, modulations=nb, size_multiple=16)
    return SimpleNamespace(
        pipe=pipe, plan=SimpleNamespace(keep_encoder=True, cfg=1.0),
        spec=SimpleNamespace(id="flux2-klein"),
        facts=svc.DitFacts(act_dtype="bfloat16", vae_dtype=None,
                           size_multiple=16, flow_map=False, ref_cache=False,
                           blocks=None))


def test_the_request_budget_counts_each_byte_once_and_fits_exactly(
        request, monkeypatch):
    """A request plan on an entry with held request pairs, held and
    pending uploads, a pending step on it, a cold build in flight and the
    request's own reservation of pairs and factors: its budget is the
    working budget less the other models, the pending uploads, the
    pending step, the build and its own factors. The plan counts the
    entry's weights, its held pairs with the request's once they load,
    and the held uploads itself, so a plan that fills that budget to the
    byte fits, and one a byte over does not."""
    from gmlx.image import memfit

    pool = _pool(monkeypatch, 100 * GB)
    _mount(request, _PoolHost({}, pool))
    up = np.ones((1, 1 << 18), np.float32)          # 1 MiB held upload
    _steps_client().put(up, kind="adapters")
    other = _model(pool, "qwen3-32b", 900 * MB, busy=False)
    o = _model(pool, "flux2-klein", 60 * MB, busy=True)
    held = pool.reserve_lora(o, 3 * MB, subject=steps.PAIRS)
    held.enter("pair", 3 * MB)
    held.cancel()
    pending_up = pool.reserve_lora(STORE, 2 * MB, subject=steps.UPLOAD)
    step = pool.reserve_lora(o, 5 * MB, subject=steps.FACTORS)
    pairs, factors = 7 * MB, 11 * MB
    own = pool.reserve_lora(o, pairs + factors, subject=svc.ADAPTERS)
    pool._building = 13 * MB
    assert pool.lora_held(o)[0] == 3 * MB
    assert pool.lora_held(STORE)[:2] == (up.nbytes, 2 * MB) == (
        sum(_store().values()), 2 * MB)
    monkeypatch.setattr(capacity, "overcommit", lambda: False)
    hold = SimpleNamespace(_pool=pool, _entry=o)
    beside = other.footprint + 2 * MB + 5 * MB + 13 * MB + factors

    # The plan once the request's pairs load: the held pairs and its own.
    served = _fit_entry(3 * MB + pairs)
    req = svc.parse_request("generations", {"prompt": "x",
                                            "size": "256x256"})
    peaks = []
    real = memfit.plan_memory

    def spy(**kw):
        got = real(**kw)
        peaks.append(got.peak)
        return got

    monkeypatch.setattr(memfit, "plan_memory", spy)
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: 10**15)
    svc.fit_check(req, served, (256, 256), images._budget(hold, pairs))
    peak = peaks[-1]
    for total, fits in ((beside + peak, True), (beside + peak - 1, False)):
        monkeypatch.setattr(capacity, "working_budget_bytes",
                            lambda total=total: total)
        assert images._budget(hold, pairs) == total - beside
        if fits:
            svc.fit_check(req, served, (256, 256), total - beside)
        else:
            with pytest.raises(svc.ImageRequestError):
                svc.fit_check(req, served, (256, 256), total - beside)
    # With GMLX_OVERCOMMIT set a request plans on no budget.
    monkeypatch.setattr(capacity, "overcommit", lambda: True)
    assert images._budget(hold, pairs) is None
    for b in (pending_up, step, own):
        b.cancel()


# T21: the optional cap at startup.

def test_a_set_lora_cap_logs_one_line_at_startup_and_none_shifts_unbounded(
        request, monkeypatch, tmp_path, caplog):
    """The image routes installed at startup with server.image.
    adapter_cache_mb set log one line that names the cap and its remedy,
    and none without it. With no cap the step routes' upload store keeps
    no bound at any share and refuses no size; a cap bounds it at the cap
    alone, at any share."""
    import logging

    monkeypatch.setattr(svc, "_CACHE_BYTES", dict(svc._CACHE_BYTES))
    monkeypatch.setattr(svc, "default_store_root",
                        lambda port=None: str(tmp_path / "files"))

    def start(cap_mb):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=svc.__name__):
            images.install_image_routes(SimpleNamespace(
                image=ImageServerCfg(adapter_cache_mb=cap_mb),
                model_dirs=(), port=None))
        return [r.getMessage() for r in caplog.records
                if "adapter_cache_mb" in r.getMessage()]

    request.addfinalizer(lambda: start(None))
    assert start(256) == [
        "server.image.adapter_cache_mb: 256 caps each image model's "
        "request LoRAs at 256 MiB; remove the key to let LoRAs use free "
        "memory"]
    assert start(None) == []
    state = steps.StepsState()
    for cap_mb, bound in ((None, steps.UNBOUNDED), (256, 256 << 20)):
        state.configure(ImageServerCfg(adapter_cache_mb=cap_mb))
        state.scale_caches(0.25)
        assert state.adapters.max_bytes == bound
        if cap_mb is None:
            state.adapters.check(1 << 40)
        else:
            with pytest.raises(steps.StepError) as e:
                state.adapters.check(bound + 1)
            assert e.value.status == 413
            assert "server.image.adapter_cache_mb" in str(e.value)
