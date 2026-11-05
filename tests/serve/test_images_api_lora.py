"""The images API's LoRA bytes, booked with the residency pool at two
turns of the job on the image worker: the pairs before a file is read and
the factors after the encode. The route reads the request's file headers
for its 413 and its budget from the pool's figures, and books nothing. A
tiny Z-Image pipeline is the pool's image entry; the other models are
stand-ins with a footprint and no weights."""
from __future__ import annotations

import json
import os
import struct
import threading
import time
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
from gmlx.serve.patches import images  # noqa: E402
from gmlx.serve.queue_cap import _RETRY_MIN_S  # noqa: E402
from gmlx.serve.residency import IMAGE_KIND, _BusyHold, gb  # noqa: E402
from test_image_routes import _APP, _DOC  # noqa: E402
from test_image_steps import _client as _steps_client  # noqa: E402
from test_step_routes_lora import (GB, _crowd, _held, _model,  # noqa: E402
                                   _mount, _pool, _PoolHost, _set_budget,
                                   _store)

MB = 10**6
# Seconds a lock holder waits for the route before it lets go.
LOCK_WAIT = 5


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
    # Retry-After as a deferred load has it.
    pool._retry_fn = None
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
                        rec.budgets.append((b, 0 if e.lora is None
                                            else e.lora.pending)))
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
    """The entry's ledger is the request keys its caches hold, at their
    bytes, and the excess of its own keys over the footprint's own term.
    Nothing is pending, no byte is past its booking, and the pool and the
    uploads are within the budget."""
    keys, own = svc.lora_figures(e)
    assert pool.lora_ledger(e) == (0, keys)
    assert e.lora.own == own and e.lora.excess == max(0, own - e.lora.own_term)
    assert pool._incoming == 0
    assert pool.overruns == []
    assert _held(pool) <= pool._budget or not budget


def _settle(t) -> None:
    """The entry's ledger set to its caches on the worker, as the next
    job's turn would, after a test let LoRA bytes go outside a job."""
    _on_worker(lambda: svc.lora_settle(t.pool, t.e, "another job"))


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
    read its budget, as another job would run there before this one.
    Returns a list that gets "ran" when it did."""
    budget, seen = images._budget, []

    def then(hold):
        got = budget(hold)
        if not seen:
            seen.append("ran")
            _on_worker(fn)
        return got

    monkeypatch.setattr(images, "_budget", then)
    return seen


def _chat(pool, name: str, fp: int):
    """A chat model in use of ``fp`` bytes, put in the pool as a cold load
    would leave it."""
    e = residency._Entry(cache_key=(f"/m/{name}.gguf", None, "chat"),
                         model_path=f"/m/{name}.gguf", footprint=int(fp),
                         busy=1, loaded_as=name)
    with pool._lock:
        pool._entries[e.cache_key] = e
        pool._touch(e)
        pool._publish()
    return e


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
    assert t.encodes == 0 and t.pool.lora_ledger(t.e) == (0, {})
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
    """A 3 GB LoRA file on a server whose budget leaves 2.4 GB beside the
    entry: 413 on the route from the file's header, with both figures in
    decimal GB. Nothing is read or booked."""
    t = _entry(request, monkeypatch, tmp_path, budget=3_400_000_000,
               own=False)
    _big_lora(t.root / "big.safetensors", 3 * GB)
    status, doc, _h = _post([("big.safetensors", 1.0)])
    assert (status, doc["error"]["message"]) == (
        413, "the request's adapters need 3.0 GB of memory, over the 2.4 GB "
        "the server's memory budget leaves beside 'zit' and the pinned "
        "models")
    assert doc["error"]["param"] == "adapters"
    assert t.encodes == 0 and t.pool.lora_ledger(t.e) == (0, {})


def test_a_lora_whose_factors_no_unload_makes_room_for_is_413_at_its_turn(
        request, monkeypatch, tmp_path):
    """A request LoRA whose pairs fit beside the entry, on a budget that
    leaves half its factors' room past them: the route and the pair turn
    let it through, and the factor turn after the encode answers 413 for
    its pairs and factors, which no unload makes room for. No factor is
    built, and the books equal the pairs the caches hold."""
    t = _entry(request, monkeypatch, tmp_path, own=False)
    assert _post([("a.safetensors", 1.0)])[0] == 200
    pairs, factors = t.pipe.adapters.bytes, t.pipe.tables.bytes
    assert pairs > 0 and factors > 0
    _on_worker(lambda: svc.shed_lora(t.e, pairs + factors))
    _settle(t)
    budget = t.e.footprint + pairs + factors // 2
    _set_budget(t.pool, budget)
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: budget)
    priced, real = [], svc.turn_figures

    def spy(*a):
        priced.append(real(*a))
        return priced[-1]

    monkeypatch.setattr(svc, "turn_figures", spy)
    status, doc, _h = _post([("a.safetensors", 1.0)])
    (need, built), = priced
    assert built == 0 and need > factors // 2
    assert (status, doc["error"]["message"]) == (
        413, f"the request's adapters need {gb(need + pairs, True)} GB "
        f"of memory, over the {gb(budget - t.e.footprint, False)} GB the "
        "server's memory budget leaves beside 'zit' and the pinned models")
    assert t.encodes == 2 and t.pipe.tables.bytes == 0
    assert t.pipe.adapters.bytes == pairs
    _books_hold(t.pool, t.e)


def test_a_factor_turn_counts_the_encoder_its_job_holds_as_its_own(
        request, monkeypatch, tmp_path):
    """The factor turn of a model that shares a chat model's encoder, on a
    budget that leaves half the factors' room beside the model, the
    encoder its job holds and the pairs: the turn answers 413 at once
    naming both as the job's own, and never waits on the encoder as a
    chat model in use. The encoder is held for the job only."""
    t = _entry(request, monkeypatch, tmp_path, own=False)
    assert _post([("a.safetensors", 1.0)])[0] == 200
    pairs, factors = t.pipe.adapters.bytes, t.pipe.tables.bytes
    _on_worker(lambda: svc.shed_lora(t.e, pairs + factors))
    _settle(t)
    enc = _chat(t.pool, "enc-qwen3", 2 * GB)
    t.pool.release(enc)

    def borrow_for(spec):
        with t.pool._lock:
            enc.busy += 1
        return (lambda pipe, run, stop, timeout_s: run()), \
            _BusyHold(t.pool, enc)

    monkeypatch.setattr(images, "_borrow_for", borrow_for)
    monkeypatch.setattr(t.served.spec, "encoder_model", "enc-qwen3")
    kept = t.e.footprint + enc.footprint
    budget = kept + pairs + factors // 2
    _set_budget(t.pool, budget)
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: budget)
    priced, real = [], svc.turn_figures

    def spy(*a):
        priced.append(real(*a))
        return priced[-1]

    monkeypatch.setattr(svc, "turn_figures", spy)
    start = time.monotonic()
    status, doc, _h = _post([("a.safetensors", 1.0)])
    assert time.monotonic() - start < _RETRY_MIN_S
    (need, _built), = priced
    assert (status, doc["error"]["message"]) == (
        413, f"the request's adapters need {gb(need + pairs, True)} GB "
        f"of memory, over the {gb(budget - kept, False)} GB the server's "
        "memory budget leaves beside 'zit', 'enc-qwen3' and the pinned "
        "models")
    assert t.encodes == 2 and t.pipe.tables.bytes == 0
    assert enc in t.pool._entries.values() and enc.busy == 0
    _books_hold(t.pool, t.e)


def test_a_job_that_fails_after_its_turns_leaves_nothing_pending(
        request, monkeypatch, tmp_path):
    """A request of LoRA a whose plan fails after its turns booked its
    pairs and factors: it answers the plan's 400, and the settle in the
    job's finally ends the booking, the books equal to what the caches
    hold."""
    t = _entry(request, monkeypatch, tmp_path, own=False)

    def refused(*a, **k):
        raise svc.ImageRequestError(400, "the plan does not fit",
                                    param="size")

    monkeypatch.setattr(api, "plan_request", refused)
    status, doc, _h = _post([("a.safetensors", 1.0)])
    assert (status, doc["error"]["message"]) == (400, "the plan does not fit")
    assert t.pipe.adapters.bytes > 0 and t.pipe.tables.bytes == 0
    _books_hold(t.pool, t.e)


def test_a_lora_a_chat_model_in_use_holds_the_room_for_waits_then_is_503(
        request, monkeypatch, tmp_path):
    """The 3 GB LoRA beside a chat model in use that leaves 1.6 GB: the
    job's pair turn waits for that model up to the deadline, then answers
    503 naming it, with Retry-After as a deferred load has it. Nothing is
    read or booked. Once that model is idle, a request of LoRA a runs
    beside it."""
    t = _entry(request, monkeypatch, tmp_path, budget=20 * GB, own=False)
    _big_lora(t.root / "big.safetensors", 3 * GB)
    busy = _chat(t.pool, "qwen3-32b", 17_400_000_000)
    start = time.monotonic()
    status, doc, headers = _post([("big.safetensors", 1.0)])
    assert time.monotonic() - start >= _RETRY_MIN_S
    assert (status, doc["error"]["message"]) == (
        503, "the request's adapters need 3.0 GB of memory, and 1.6 GB is "
        "free beside the models in use (qwen3-32b); retry when one is "
        "idle")
    assert headers["retry-after"] == str(_RETRY_MIN_S)
    assert t.encodes == 0 and t.pool.lora_ledger(t.e) == (0, {})
    assert t.pipe.adapters.bytes == 0
    t.pool.release(busy)
    status, doc, _h = _post([("a.safetensors", 1.0)])
    assert status == 200 and doc["data"]
    assert busy in t.pool._entries.values()
    _books_hold(t.pool, t.e)


def test_an_idle_model_holding_the_room_is_evicted_at_the_jobs_turn(
        request, monkeypatch, tmp_path):
    """A request LoRA that fits only once an idle model goes: the job's
    pair turn evicts that model, as a cold load does, and the request
    runs on the room it booked."""
    t = _entry(request, monkeypatch, tmp_path, own=False)
    size = os.path.getsize(t.root / "a.safetensors")
    other = _model(t.pool, "qwen3-32b", 3 * GB - size // 2, busy=False)
    assert other in t.pool._entries.values()
    status, doc, _h = _post([("a.safetensors", 1.0)])
    assert status == 200 and doc["data"]
    assert other not in t.pool._entries.values()
    assert t.pipe.adapters.bytes > 0
    _books_hold(t.pool, t.e)


def test_the_route_reads_its_lora_and_its_budget_without_the_pools_lock(
        request, monkeypatch, tmp_path):
    """Another thread holds the pool's lock as the route checks the
    request's LoRA bytes and reads its budget: both read the pool's
    figures alone and book nothing. The fit check gets the working budget
    less the other model, with nothing pending, and the job's turns book
    the request's LoRA."""
    t = _entry(request, monkeypatch, tmp_path, own=False)
    _model(t.pool, "qwen3-32b", 500 * MB, busy=False)
    hopeless, budget, out = images._hopeless, images._budget, []

    def checked(hold, served, req):
        locked, free = threading.Event(), threading.Event()

        def holder():
            with t.pool._lock:
                locked.set()
                out.append(free.wait(LOCK_WAIT))
        th = threading.Thread(target=holder, daemon=True)
        th.start()
        assert locked.wait(LOCK_WAIT)
        hopeless(hold, served, req)
        out.append(budget(hold))
        free.set()
        th.join()

    monkeypatch.setattr(images, "_hopeless", checked)
    assert _post([("a.safetensors", 1.0)])[0] == 200
    assert out == [4 * GB - 500 * MB, True], "the route waited on the lock"
    assert t.budgets == [(4 * GB - 500 * MB, 0)]
    assert t.pipe.adapters.bytes > 0
    _books_hold(t.pool, t.e)


def test_a_requests_estimate_is_its_files_whatever_the_pipeline_holds(
        request, monkeypatch, tmp_path):
    """The route counts the tensor bytes of the request's files from their
    headers, whether the pipeline holds their pairs or a shed let them
    go: it reads no cache or memo."""
    from gmlx.image.vae.reader import safetensors_header

    t = _entry(request, monkeypatch, tmp_path, own=False)
    path = t.root / "a.safetensors"
    tensors = os.path.getsize(path) - safetensors_header(str(path))[1]
    req = svc.parse_request("generations", {
        "prompt": "x", "adapters": [{"id": "a.safetensors",
                                     "strength": 1.0}]})

    def estimate():
        return svc.request_estimate(t.served, req.adapters, (str(t.root),))

    assert estimate() == tensors > 0
    assert _post([("a.safetensors", 1.0)])[0] == 200
    assert t.pipe.adapters.bytes > 0 and estimate() == tensors
    _on_worker(lambda: svc.shed_lora(t.e, t.pipe.adapters.bytes
                                     + t.pipe.tables.bytes))
    assert t.pipe.adapters.bytes == 0 and estimate() == tensors


# T3 at the pool.

@pytest.mark.parametrize("cap_mb", [None, 256])
def test_a_model_with_no_lora_books_no_lora_memory(request, monkeypatch,
                                                   tmp_path, cap_mb):
    """An entry with no LoRA, and a request with none, book nothing: the
    ledger is empty after the load and after the job, and the pool's
    resident bytes are the footprints, with or without a cap."""
    t = _entry(request, monkeypatch, tmp_path, own=False, cap_mb=cap_mb)
    assert t.pool.lora_ledger(t.e) == (0, {}) and t.e.lora.held == 0
    status, _doc, _h = _post()
    assert status == 200
    assert t.pool.lora_ledger(t.e) == (0, {}) and t.e.lora.held == 0
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
    excess = t.e.lora.excess
    assert excess > 0
    _books_hold(t.pool, t.e)
    assert _post()[0] == 200
    assert np.array_equal(t.latents[-1], t.latents[0])
    assert t.e.lora.excess == excess
    _books_hold(t.pool, t.e)
    # Another job's turn sheds the built table, and a model in use takes
    # the memory free.
    _on_worker(lambda: svc.shed_lora(t.e, t.pipe.tables.bytes, own=True))
    assert t.pipe.tables.bytes == 0
    _settle(t)
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
    _pending, ledger = t.pool.lora_ledger(t.e)
    assert ledger and all(isinstance(k, tuple) for k in ledger)
    _books_hold(t.pool, t.e)
    assert _post([("own", 0.6)])[0] == 200
    assert np.array_equal(t.latents[-1], weaker)
    assert t.pool.lora_ledger(t.e)[1] == ledger
    _books_hold(t.pool, t.e)


@pytest.mark.parametrize("room", [True, False])
def test_a_staged_entrys_encode_drops_its_tables_and_its_turn_books_them(
        request, monkeypatch, tmp_path, room):
    """On a staged entry the encode lets the DiT and its built tables go
    inside the job. A request of LoRA a again books its factors in full
    at its turn after the encode and builds them, with the same image and
    the books equal to what the caches hold. With the memory taken after
    a first request let the DiT go, that turn answers 503 and builds
    nothing."""
    t = _entry(request, monkeypatch, tmp_path, staged=True)
    assert _post([("a.safetensors", 1.0)])[0] == 200
    built = t.pipe.tables.bytes
    assert built > 0
    _books_hold(t.pool, t.e)
    if not room:
        _on_worker(lambda: svc._drop_dit(t.served))
        _settle(t)
        _crowd_in_use(t)
        encodes = t.encodes
        status, doc, headers = _post([("a.safetensors", 1.0)])
        assert status == 503 and headers["retry-after"] == str(_RETRY_MIN_S)
        assert t.encodes == encodes + 1 and t.pipe.tables.bytes == 0
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
    """Another job sheds the pairs and the built table of a request of
    LoRA a after its route, which books nothing, and before its job runs.
    The job books the pairs at its first turn and the factors at its
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
        svc.lora_settle(t.pool, t.e, "another job")
        if not room:
            _crowd(t.pool)

    ran = _after_route(monkeypatch, shed)
    status, doc, headers = _post([("a.safetensors", 1.0)])
    assert ran == ["ran"]
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
    """A request plan on an entry with held request pairs and a booking of
    a step's turn, beside held uploads, an upload body in flight, another
    model and a cold build in flight: its budget is the working budget
    less the other model, the step's booking, the body in flight and the
    build. The plan counts the entry's weights, its held pairs and the
    held uploads itself, so a plan that fills that budget to the byte
    fits, and one a byte over does not."""
    from gmlx.image import memfit

    pool = _pool(monkeypatch, 100 * GB)
    _mount(request, _PoolHost({}, pool))
    up = np.ones((1, 1 << 18), np.float32)          # 1 MiB held upload
    _steps_client().put(up, kind="adapters")
    assert pool.upload_admit(2 * MB, None)
    request.addfinalizer(lambda: pool.upload_done(2 * MB))
    other = _model(pool, "qwen3-32b", 900 * MB, busy=False)
    o = _model(pool, "flux2-klein", 60 * MB, busy=True)
    with pool._lock:
        o.lora = residency._Ledger()
        o.lora.set(({"pair": 3 * MB}, 0))
        o.lora.pending = 5 * MB
        pool._building = 13 * MB
        pool._publish()
    assert pool.lora_ledger(o) == (5 * MB, {"pair": 3 * MB})
    assert steps.STATE.adapters.bytes + pool.figures.incoming == (
        up.nbytes + 2 * MB) == sum(_store().values()) + 2 * MB
    monkeypatch.setattr(capacity, "overcommit", lambda: False)
    hold = SimpleNamespace(_pool=pool, _entry=o)
    beside = other.footprint + 5 * MB + 2 * MB + 13 * MB

    # The plan reads the held pairs from the pipeline.
    served = _fit_entry(3 * MB)
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
    svc.fit_check(req, served, (256, 256), images._budget(hold))
    peak = peaks[-1]
    for total, fits in ((beside + peak, True), (beside + peak - 1, False)):
        monkeypatch.setattr(capacity, "working_budget_bytes",
                            lambda total=total: total)
        assert images._budget(hold) == total - beside
        if fits:
            svc.fit_check(req, served, (256, 256), total - beside)
        else:
            with pytest.raises(svc.ImageRequestError):
                svc.fit_check(req, served, (256, 256), total - beside)
    # With GMLX_OVERCOMMIT set a request plans on no budget.
    monkeypatch.setattr(capacity, "overcommit", lambda: True)
    assert images._budget(hold) is None


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
