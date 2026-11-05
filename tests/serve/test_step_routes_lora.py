"""The step routes' LoRA bytes, booked with the residency pool: an
upload is admitted before its body is read, a define and a step book at
their turns on the worker, and memory the models hold is refused with
its figures.
A tiny Z-Image pipeline is the pool's image entry; the other models are
stand-ins with a footprint and no weights."""
from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")
mx = pytest.importorskip("mlx.core")

from fastapi.testclient import TestClient  # noqa: E402

from gmlx.config import ImageServerCfg  # noqa: E402
from gmlx.image.samplers import noise_from_seed  # noqa: E402
from gmlx.serve import image_service, residency  # noqa: E402
from gmlx.serve import image_steps as steps  # noqa: E402
from gmlx.serve import image_wire as wire  # noqa: E402
from gmlx.serve.image_steps_client import (MissingTensors,  # noqa: E402
                                           StepsClientError, array_hash)
from gmlx.serve.patches import image_steps as routes  # noqa: E402
from gmlx.serve.residency import (IMAGE_KIND, _BusyHold,  # noqa: E402
                                  _ResidencyPool)
from test_image_steps import (_APP, SHAPE, _client, _features,  # noqa: E402
                              _header_alpha_own, _Host, _pack_mats,
                              _pack_sync, _rows)

GB = 10**9
P = wire.PREFIX
V = {wire.CONTRACT_HEADER: "1.0"}


class _Pool(_ResidencyPool):
    overruns: list


def _pool(monkeypatch, budget: int) -> _Pool:
    """A pool of ``budget`` bytes whose LoRA sheds are the server's own.
    Each byte past a booking is recorded in ``pool.overruns``."""
    from gmlx.serve import capacity as cap

    monkeypatch.setattr(cap, "preload_gate", lambda *a, **k: None)
    monkeypatch.delenv("GMLX_OVERCOMMIT", raising=False)
    monkeypatch.setattr(residency, "_holds_image_worker", lambda: True)
    pool = _Pool(
        SimpleNamespace(_last_entry=None), None, None, int(budget), (),
        image_fns=(lambda spec: spec.fp, lambda spec: spec.pipe,
                   lambda box: box.pop()))
    pool._worker_for = lambda *a, **k: contextlib.nullcontext()
    pool._headroom_fn = lambda: None
    pool._retry_fn = lambda: 1
    pool.overruns = []
    pool.overrun_hook = lambda s, n: pool.overruns.append((s, n))
    return pool


def _set_budget(pool, budget: int) -> None:
    """Set the pool's budget, and the figures the routes read."""
    with pool._lock:
        pool._budget = int(budget)
        pool._publish()


def _held(pool) -> int:
    """What the budget counts: the pool and the uploads."""
    return pool._resident_bytes() + steps.STATE.adapters.live()


def _model(pool, name: str, fp: int, *, busy: bool):
    """A stand-in model of ``fp`` bytes, in use or idle."""
    e = pool.acquire(f"/m/{name}.gguf", None, IMAGE_KIND,
                     build_spec=SimpleNamespace(fp=int(fp), stream=None,
                                                path=f"/m/{name}.gguf",
                                                pipe=SimpleNamespace()),
                     loaded_as=name)
    if not busy:
        pool.release(e)
    return e


class _PoolHost(_Host):
    """The tiny pipelines as entries of ``pool``, each of ``fp`` bytes."""

    def __init__(self, pipes: dict, pool, fp: int = GB):
        super().__init__(pipes)
        self.pool, self.fp = pool, int(fp)

    def lora_pool(self):
        return self.pool

    def entry(self, model):
        served = self.served[self.resolve(model)[1].id]
        e = self.pool.acquire(
            served.spec.path, None, IMAGE_KIND, loaded_as=served.spec.id,
            build_spec=SimpleNamespace(fp=self.fp, stream=None,
                                       path=served.spec.path, pipe=served))
        self.pool.release(e)
        return e

    def acquire(self, model):
        e = self.entry(model)
        with self.pool._lock:
            e.busy += 1
        return _BusyHold(self.pool, e)


def _mount(request, host, cfg=None, model_dirs=()):
    routes.mount(_APP.app, cfg or ImageServerCfg(), host,
                 model_dirs=model_dirs)

    def done():
        steps.STATE.reset()
        routes.mount(_APP.app, ImageServerCfg(steps_api=False))
    request.addfinalizer(done)
    return host


def _asgi(method: str, path: str, chunks=(), headers=None):
    """``method`` on ``path`` through the app as a server reads a body:
    one chunk a receive. Returns the status, the JSON answer and the
    lengths of the chunks the app read."""
    read: list = []
    it = iter(chunks)

    async def receive():
        try:
            c = next(it)
        except StopIteration:
            return {"type": "http.request", "body": b"", "more_body": False}
        read.append(len(c))
        return {"type": "http.request", "body": c, "more_body": True}

    out: dict = {"body": b""}

    async def send(m):
        if m["type"] == "http.response.start":
            out["status"] = m["status"]
        elif m["type"] == "http.response.body":
            out["body"] += m.get("body", b"")

    hdrs = {"host": "127.0.0.1", **V, **(headers or {})}
    scope = {"type": "http", "asgi": {"version": "3.0"},
             "http_version": "1.1", "method": method, "scheme": "http",
             "path": path, "raw_path": path.encode(), "query_string": b"",
             "root_path": "",
             "headers": [(k.lower().encode(), str(v).encode())
                         for k, v in hdrs.items()],
             "client": ("127.0.0.1", 50000), "server": ("127.0.0.1", 80)}
    asyncio.run(_APP.app(scope, receive, send))
    return out["status"], json.loads(out["body"] or b"{}"), read


def _store() -> dict:
    with steps.STATE.adapters._lock:
        return {k: t.nbytes for k, t in steps.STATE.adapters._items.items()}


def _books_hold(pool, *entries, budget: bool = True) -> None:
    """Each entry's ledger is the request keys its caches hold, at their
    bytes, with nothing pending and no byte past its booking; the store
    counts no body in flight, and the pool and the uploads are within
    the budget."""
    for e in entries:
        keys, _own = image_service.lora_figures(e)
        assert pool.lora_ledger(e) == (0, keys)
    assert steps.STATE.adapters.incoming == 0
    assert _held(pool) <= pool._budget or not budget
    assert pool.overruns == []


# Capabilities.

@pytest.mark.parametrize("cap_mb, want", [(None, 1 << 30), (16, 16 << 20)])
def test_capabilities_give_a_gib_unset_and_the_cap_when_set(
        request, monkeypatch, cap_mb, want):
    """The largest adapter matrix a client may send is 1 GiB with no cap,
    and the cap when server.image.adapter_cache_mb sets one, whatever the
    store holds."""
    pool = _pool(monkeypatch, 10 * GB)
    _mount(request, _PoolHost({}, pool),
           ImageServerCfg(adapter_cache_mb=cap_mb))
    c = _client()
    before = c.capabilities()["limits"]["max_adapter_bytes"]
    c.put(np.ones((1, 1 << 16), np.float32), kind="adapters")
    after = c.capabilities()["limits"]["max_adapter_bytes"]
    assert before == after == want


# Uploads.

_UPLOAD_503 = ("the LoRA matrices of this upload need 1.0 GB of memory, "
               "and 0.6 GB is free beside the models in use (flux2-klein); "
               "retry when one is idle")


def test_an_upload_over_the_memory_free_is_refused_before_its_body_is_read(
        request, monkeypatch):
    """A 1 GB upload beside a model in use that leaves 0.6 GB is a 503
    with both figures, answered from its Content-Length: no byte of the
    body is read and the store holds nothing more."""
    pool = _pool(monkeypatch, 9_600_000_000)
    _model(pool, "flux2-klein", 9 * GB, busy=True)
    _mount(request, _PoolHost({}, pool))
    status, doc, read = _asgi(
        "PUT", f"{P}/adapters/{'ab' * 32}", [b"x" * (1 << 20)] * 4,
        {"content-type": wire.CONTENT_TYPE, "content-length": 10**9})
    assert (status, doc["error"], read) == (503, _UPLOAD_503, [])
    assert _store() == {} and steps.STATE.adapters.incoming == 0


def test_a_register_over_the_memory_free_is_refused_before_its_file_is_read(
        request, monkeypatch, tmp_path):
    """A registered adapter file is booked at its size before it is
    opened: a 1 GB file beside a model in use that leaves 0.6 GB is a 503
    that opens nothing."""
    path = tmp_path / "big.safetensors"
    with open(path, "wb") as f:
        f.truncate(10**9)
    pool = _pool(monkeypatch, 9_600_000_000)
    _model(pool, "flux2-klein", 9 * GB, busy=True)
    _mount(request, _PoolHost({}, pool), model_dirs=(str(tmp_path),))
    opened = []
    monkeypatch.setattr(routes, "open", lambda *a, **k: opened.append(a),
                        raising=False)
    body = json.dumps({"path": str(path)}).encode()
    status, doc, _read = _asgi(
        "POST", f"{P}/adapters/register", [body],
        {"content-type": "application/json", "content-length": len(body)})
    assert (status, doc["error"], opened) == (503, _UPLOAD_503, [])
    assert _store() == {} and steps.STATE.adapters.incoming == 0


def test_an_upload_evicts_an_idle_model_at_its_store_turn(request,
                                                           monkeypatch):
    """An upload that does not fit beside an idle model gets its room at
    a store turn on the image worker, which unloads the model as a cold
    load does, and is kept. A 503 carries Retry-After."""
    a = np.ones((1, 1 << 16), np.float32)
    pool = _pool(monkeypatch, 9 * GB + a.nbytes // 2)
    idle = _model(pool, "qwen3-32b", 9 * GB, busy=False)
    _mount(request, _PoolHost({}, pool))
    h = _client().put(a, kind="adapters")
    assert idle not in pool._entries.values()
    assert _store() == {h: a.nbytes}
    _books_hold(pool)


def test_an_upload_refusal_carries_retry_after(request, monkeypatch):
    pool = _pool(monkeypatch, 9_600_000_000)
    _model(pool, "flux2-klein", 9 * GB, busy=True)
    _mount(request, _PoolHost({}, pool))
    r = TestClient(_APP.app, base_url="http://127.0.0.1").put(
        f"{P}/adapters/{'ab' * 32}", content=b"x" * 16,
        headers={**V, "content-type": wire.CONTENT_TYPE,
                 "content-length": str(10**9)})
    assert r.status_code == 503 and r.headers["retry-after"] == "1"
    assert r.json()["error"] == _UPLOAD_503


def test_a_body_with_no_length_is_booked_in_steps_and_refused_mid_read(
        request, monkeypatch):
    """A body with no Content-Length is booked a step at a time as it is
    read (64 MiB; 1 MiB here). One under the memory free is kept and its
    booking settles at its bytes. One over it is refused at the step that
    cannot be booked, before the rest of it is read."""
    assert steps.BODY_STEP == 64 << 20
    monkeypatch.setattr(steps, "BODY_STEP", 1 << 20)
    pool = _pool(monkeypatch, GB + (5 << 19))
    _model(pool, "flux2-klein", GB, busy=True)
    _mount(request, _PoolHost({}, pool))

    def put(a):
        raw = wire.write({"t": a})
        chunks = [raw[i:i + (1 << 18)] for i in range(0, len(raw), 1 << 18)]
        return _asgi("PUT", f"{P}/adapters/{array_hash(a)}", chunks,
                     {"content-type": wire.CONTENT_TYPE}) + (len(raw),)

    # 4 MiB beside 2.5 MiB free: the third step is refused.
    status, doc, read, size = put(np.ones((1, 1 << 20), np.float32))
    assert (status, doc["error"]) == (
        503, "the LoRA matrices of this upload need 0.1 GB of memory, and "
        "0.0 GB is free beside the models in use (flux2-klein); retry when "
        "one is idle")
    assert 2 << 20 < sum(read) <= (2 << 20) + (1 << 18) < size
    assert _store() == {}
    _books_hold(pool)
    small = np.ones((1, 3 << 17), np.float32)       # 1.5 MiB
    status, doc, read, size = put(small)
    assert (status, doc, sum(read)) == (200, {"hash": array_hash(small)},
                                        size)
    assert _store() == {array_hash(small): small.nbytes}
    _books_hold(pool)


# Steps.

def _stepped(request, monkeypatch, tmp_path):
    """Entry "own" with a client's table defined and stepped once: the
    client's LoRA stacks on the entry's own, so the step builds its table.
    Returns the pool, the entry, the client, the table id, the rows, the
    noise and the first step's output."""
    pool, e, _p, ((c, rows, sync),) = _own_entry(request, monkeypatch,
                                                 tmp_path, 1)
    tid = sync()[1]
    x = noise_from_seed(3, SHAPE)
    first = c.step("own", None, x, [0.75], rows, patch_table=tid)
    return pool, e, c, tid, rows, x, first


def _own_bytes(pipe) -> int:
    return sum(pipe.adapters.nbytes(k) for k in pipe.adapters.kept)


def _between(monkeypatch, fn) -> None:
    """``fn`` on the image worker once, right after a step's route has
    checked it, as another job would run there before the step's own."""
    real, ran = routes._hopeless, []

    def then(pool, owner, n, subject):
        real(pool, owner, n, subject)
        if not ran:
            ran.append(n)
            image_service.WORKER.submit(fn).result()

    monkeypatch.setattr(routes, "_hopeless", then)


def _crowd(pool) -> None:
    """A model in use takes the memory free and a byte more, as one whose
    plan was priced again at its run: the pool is past its budget, so a
    turn is refused whatever it could shed."""
    free = pool._budget - _held(pool)
    _model(pool, "flux2-klein", free + 1, busy=True)


@pytest.mark.parametrize("room", [True, False])
def test_a_step_whose_pairs_another_job_shed_books_them_at_its_turn(
        request, monkeypatch, tmp_path, room):
    """Another job sheds a table's pairs and its built table after the
    step's route found them held. The step books the pairs and the
    factors at its turns on the worker and steps as before, with the
    books equal to what the caches hold. With no room for the pairs at
    the turn it answers 503 and makes no pair."""
    pool, e, c, tid, rows, x, first = _stepped(request, monkeypatch,
                                               tmp_path)
    pipe = e.pipeline.pipe
    pairs, built = pipe.adapters.bytes, pipe.tables.bytes
    assert pairs > _own_bytes(pipe) and built > 0

    def shed():
        # Another job's turn sheds them and its settle sets the ledger.
        image_service.shed_lora(e, pipe.adapters.bytes + pipe.tables.bytes)
        pool.lora_settle(e, "another job")
        if not room:
            _crowd(pool)

    _between(monkeypatch, shed)
    if room:
        again = c.step("own", None, x, [0.75], rows, patch_table=tid)
        assert np.array_equal(again, first)
        assert pipe.adapters.bytes == pairs and 0 < pipe.tables.bytes <= built
        _books_hold(pool, e)
        return
    with pytest.raises(StepsClientError) as refused:
        c.step("own", None, x, [0.75], rows, patch_table=tid)
    assert refused.value.status == 503
    assert refused.value.body["error"] == (
        "the LoRA factors of this step need 0.1 GB of memory, and 0.0 GB is "
        "free beside the models in use (flux2-klein); retry when one is "
        "idle")
    assert (pipe.adapters.bytes, pipe.tables.bytes) == (_own_bytes(pipe), 0)
    _books_hold(pool, e, budget=False)


@pytest.mark.parametrize("room", [True, False])
def test_a_step_whose_built_table_another_job_shed_books_it_at_its_turn(
        request, monkeypatch, tmp_path, room):
    """Another job's claim sheds a held table's built factors after the
    step's route priced them as built. The step's turn books them again
    from the dry run, and the step steps as before. With no room for them
    it answers 503 and builds nothing."""
    pool, e, c, tid, rows, x, first = _stepped(request, monkeypatch,
                                               tmp_path)
    pipe = e.pipeline.pipe
    built = pipe.tables.bytes
    assert built > 0

    def shed():
        image_service.shed_lora(e, pipe.tables.bytes)
        pool.lora_settle(e, "another job")
        assert pipe.tables.bytes == 0 and pipe.adapters.bytes > 0
        if not room:
            _crowd(pool)

    _between(monkeypatch, shed)
    if room:
        again = c.step("own", None, x, [0.75], rows, patch_table=tid)
        assert np.array_equal(again, first) and 0 < pipe.tables.bytes <= built
        _books_hold(pool, e)
        return
    with pytest.raises(StepsClientError) as refused:
        c.step("own", None, x, [0.75], rows, patch_table=tid)
    assert refused.value.status == 503
    assert pipe.tables.bytes == 0 and pipe.adapters.bytes > 0
    _books_hold(pool, e, budget=False)


def test_a_turn_claims_other_jobs_bytes_and_never_its_own_pairs(
        request, monkeypatch, tmp_path):
    """The built tables of clients A and B go after A's step's route
    priced A's as built, and a model in use takes most of the memory
    free. A's turn claims what it lacks from B's pairs and then from the
    uploads, never from A's own pairs, which its shed holds: the step
    steps as before and the books equal what the caches hold."""
    pool, e, p, packs = _own_entry(request, monkeypatch, tmp_path, 2)
    x = noise_from_seed(5, SHAPE)
    firsts = []
    for c, rows, sync in packs:
        tid = sync()[1]
        firsts.append((tid, c.step("own", None, x, [0.5], rows,
                                   patch_table=tid)))
    _set_budget(pool, _held(pool))
    core = steps.STATE.tables_of("own", p)[firsts[0][0]]
    a_pairs = image_service.own_bytes(p, {
        k for _m, _b, ps in p.patch_tables[core]["entries"] for k, _s, _a in ps})

    def drop():
        built = sum(p.tables.nbytes(k) for k in p.tables.keys(
            lambda k: isinstance(k, tuple) and k[0] == core))
        image_service.shed_lora(e, p.tables.bytes)
        pool.lora_settle(e, "another job")
        assert p.tables.bytes == 0
        # A's turn lacks B's pairs and half of what the uploads hold
        # beside them.
        others = e.lora.held - a_pairs
        short = others + (built - others) // 2
        assert others < short < built
        free = pool._budget - _held(pool)
        _model(pool, "flux2-klein", free - built + short, busy=True)

    _between(monkeypatch, drop)
    c, rows, _sync = packs[0]
    tid, first = firsts[0]
    assert np.array_equal(c.step("own", None, x, [0.5], rows,
                                 patch_table=tid), first)
    _books_hold(pool, e)


# Review 2's sweeps, with the pool.

def _own_pairs(request, monkeypatch, tmp_path):
    """Entry "own" with a LoRA of its own in an open pool. Returns the
    pool, the entry, the pipeline and its own pairs, (module, band, pair)
    each."""
    pool = _pool(monkeypatch, 4 * GB)
    host = _mount(request, _PoolHost({}, pool))
    p, ad, _c, _body = _header_alpha_own(host, tmp_path)
    # The footprint's own term, as _load sets it.
    host.served["own"].lora_own_term = 2 * (tmp_path /
                                            "style.safetensors").stat().st_size
    e = host.entry("own")
    return pool, e, p, [(m, b, p.adapters.peek(k)) for m, b, _a, k in ad.dit]


def _own_entry(request, monkeypatch, tmp_path, n: int):
    """Entry "own" with a LoRA of its own in an open pool, and ``n``
    clients each under a LoRA the size of the entry's, as the pack syncs
    them. Returns the pool, the entry, the pipeline and the packs."""
    pool, e, p, pairs = _own_pairs(request, monkeypatch, tmp_path)
    packs = []
    for scale in (2.0, 3.0, 4.0)[:n]:
        c = _client()
        packs.append((c, _rows(c, _features()), _pack_sync(c, pairs, scale)))
    return pool, e, p, packs


def _sweep(packs, x, steps_each: int, *, synced=None):
    """Each pack syncs, unless ``synced`` gives its table id, then samples
    ``steps_each`` steps, one request a turn across the packs. A 409 on
    the table syncs it again, three times at most as the pack does.
    Returns each pack's outputs and the error text of each 409 it met."""
    outs = [[] for _ in packs]
    errors = [[] for _ in packs]

    def run(i, c, rows, sync):
        tid = (synced or {}).get(i) or sync()[1]
        yield
        for _ in range(steps_each):
            for _try in range(4):
                try:
                    outs[i].append(c.step("own", None, x, [0.5], rows,
                                          patch_table=tid))
                    break
                except MissingTensors as e:
                    assert e.missing == [tid] and _try < 3
                    errors[i].append(e.body["error"])
                    yield
                    tid = sync()[1]
                    yield
            yield
    runs = [run(i, *pk) for i, pk in enumerate(packs)]
    while runs:
        for r in list(runs):
            try:
                next(r)
            except StopIteration:
                runs.remove(r)
    return outs, errors


@pytest.mark.parametrize("n", [2, 3])
def test_clients_whose_loras_fit_in_memory_step_without_a_409(
        request, monkeypatch, tmp_path, n):
    """Two and three ComfyUIs sample on one entry at once with memory for
    all their LoRAs: no 409, each steps the same every time, and the
    books equal what the caches hold."""
    pool, e, p, packs = _own_entry(request, monkeypatch, tmp_path, n)
    outs, errors = _sweep(packs, noise_from_seed(5, SHAPE), 3)
    assert errors == [[]] * n
    for out in outs:
        assert len(out) == 3 and all(np.array_equal(y, out[0]) for y in out)
    assert len({out[0].tobytes() for out in outs}) == n
    _books_hold(pool, e)


def _one_lora(pool, e, packs, x) -> tuple:
    """The first pack's sync and step at the open budget, and what it
    books: its uploads and its entry bytes."""
    c, rows, sync = packs[0]
    tid = sync()[1]
    first = c.step("own", None, x, [0.5], rows, patch_table=tid)
    return tid, first, steps.STATE.adapters.live(), e.lora.held


def _count_sheds(monkeypatch) -> list:
    """The entry sheds of the claims from here on."""
    sheds, real = [], image_service.shed_lora

    def shed(entry, nbytes, hold=(), own=False):
        sheds.append(nbytes)
        return real(entry, nbytes, hold, own)

    monkeypatch.setattr(image_service, "shed_lora", shed)
    return sheds


def test_two_clients_at_a_budget_for_one_lora_take_turns(request,
                                                         monkeypatch,
                                                         tmp_path):
    """With memory for both clients' uploads and one LoRA's pairs and
    tables, each define claims the other's, and each step defines its
    table again in its own turn from the matrices the server keeps: no
    409, and each steps the same every time."""
    pool, e, p, packs = _own_entry(request, monkeypatch, tmp_path, 2)
    x = noise_from_seed(5, SHAPE)
    tid, first, uploads, lora = _one_lora(pool, e, packs, x)
    _set_budget(pool, _held(pool) + uploads + lora // 2)
    sheds = _count_sheds(monkeypatch)
    outs, errors = _sweep(packs, x, 3, synced={0: tid, 1: packs[1][2]()[1]})
    assert errors == [[], []] and sheds
    assert all(np.array_equal(y, first) for y in outs[0])
    for out in outs:
        assert len(out) == 3 and all(np.array_equal(y, out[0]) for y in out)
    _books_hold(pool, e)


def test_when_the_uploads_give_way_the_409_names_the_memory(request,
                                                            monkeypatch,
                                                            tmp_path):
    """Three clients with memory for one client's uploads and LoRA: each
    sync sheds the others' pairs and tables, and then their matrices. A
    step whose pairs and matrices both went meets a 409 whose text gives
    the bytes of matrices not held, and syncs again. Each still steps the
    same every time, within the pack's tries."""
    pool, e, p, packs = _own_entry(request, monkeypatch, tmp_path, 3)
    x = noise_from_seed(5, SHAPE)
    tid, first, uploads, lora = _one_lora(pool, e, packs, x)
    _set_budget(pool, _held(pool))
    outs, errors = _sweep(packs, x, 3, synced={0: tid})
    assert any(errors) and all(
        err.endswith(". 0.1 GB of LoRA matrices are not held: the server "
                     "let them go for memory")
        for errs in errors for err in errs)
    assert all(np.array_equal(y, first) for y in outs[0])
    for out in outs:
        assert len(out) == 3 and all(np.array_equal(y, out[0]) for y in out)
    _books_hold(pool, e)


# A define's own matrices in the uploads.

def _sync_retried(c, mats, entries, between=None) -> tuple:
    """The pack's sync_table of a table of ``entries``: upload
    ``mats``, run ``between``, then define, and after a 409 upload the
    matrices it names again and define once more. Returns the table id or
    None, and the status and text of each refusal."""
    c._uploaded.difference_update(mats)
    for a in mats.values():
        c.put(a, kind="adapters")
    if between is not None:
        between()
    log = []
    for _try in range(2):
        try:
            return c.define_table("own", {"entries": entries}), log
        except MissingTensors as err:
            log.append((409, err.body["error"]))
            for h in err.missing:
                if h in mats:
                    c._uploaded.discard(h)
                    c.put(mats[h], kind="adapters")
        except StepsClientError as err:
            log.append((err.status, err.body["error"]))
            return None, log
    return None, log


@pytest.mark.parametrize("case", ["alone", "idle_model", "older_upload",
                                  "newer_upload"])
def test_a_define_short_by_part_of_its_matrices_never_sheds_them(
        request, monkeypatch, tmp_path, case):
    """A client's define on a server short by a quarter of the bytes of
    the matrices it just uploaded, synced as the pack does. The matrices
    are the define's own keys of the uploads, which its claim never
    sheds: alone it is refused with the memory figures and the matrices
    stay, beside an idle model it unloads the model and defines, and
    beside another upload, sent before or after its own, it sheds that
    one and defines. No case meets a 409."""
    pool, e, p, pairs = _own_pairs(request, monkeypatch, tmp_path)
    steps.STATE.adapters.clear()
    c, other = _client(), _client()
    mats, entries = _pack_mats(pairs, 2.0)
    m_bytes = sum(a.nbytes for a in mats.values())
    short = m_bytes // 4
    # Another client's matrix of the bytes the define lacks.
    stale = np.ones((1, short // 4 + 64), np.float32)
    if case == "older_upload":
        other.put(stale, kind="adapters")
    idle = _model(pool, "idle-chat", 2 * short, busy=False) \
        if case == "idle_model" else None

    def between():
        if case == "newer_upload":
            other.put(stale, kind="adapters")
        # The pairs the define makes are the matrices' bytes again.
        _set_budget(pool, _held(pool) + m_bytes - short)

    tid, log = _sync_retried(c, mats, entries, between)
    assert all(h in _store() for h in mats)
    if case == "alone":
        assert tid is None and log == [(
            503, "the LoRA pairs of this patch table need 0.1 GB of memory, "
            "and 0.0 GB is free; unload a model or retry when the requests "
            "in flight finish")]
        _books_hold(pool, e)
        return
    assert tid is not None and log == []
    if idle is not None:
        assert idle not in pool._entries.values()
    else:
        assert array_hash(stale) not in _store()
    _books_hold(pool, e)


def test_a_step_defining_its_table_again_never_sheds_its_matrices(
        request, monkeypatch, tmp_path):
    """A step whose table and pairs another job let go defines the table
    again in its turn from the matrices the server keeps, on a server
    short by a quarter of their bytes beside an idle model. The matrices
    are the step's own keys of the uploads until its define: its route
    unloads the idle model, and the step steps as before with no 409."""
    pool, e, p, pairs = _own_pairs(request, monkeypatch, tmp_path)
    steps.STATE.adapters.clear()
    c = _client()
    rows = _rows(c, _features())
    tid = _pack_sync(c, pairs, 2.0)()[1]
    x = noise_from_seed(3, SHAPE)
    first = c.step("own", None, x, [0.75], rows, patch_table=tid)
    # What the step holds once it ran: its pairs and its built table.
    n = e.lora.held
    image_service.shed_lora(e, p.adapters.bytes + p.tables.bytes)
    pool.lora_settle(e, "another job")
    core = steps.STATE.tables_of("own", p)[tid]
    assert not routes._held(p, core)
    short = sum(_store().values()) // 4
    idle = _model(pool, "idle-chat", 2 * short, busy=False)
    _set_budget(pool, _held(pool) + n - short)
    again = c.step("own", None, x, [0.75], rows, patch_table=tid)
    assert np.array_equal(again, first)
    assert idle not in pool._entries.values()
    _books_hold(pool, e)


# T14 (a) at the routes: the event loop never waits on the pool's lock.

# Seconds a lock holder waits for the event loop before it lets go.
LOOP_WAIT = 5


def _loop_client(loops: list):
    """A steps client whose requests each record the event loop that
    serves them in ``loops``."""
    async def app(scope, receive, send):
        loops.append(asyncio.get_running_loop())
        await _APP.app(scope, receive, send)
    return _client(http=TestClient(app, base_url="http://127.0.0.1"))


def _locked_at(monkeypatch, pool, name: str, loops: list) -> list:
    """Wrap the route helper ``name``: as it starts, a thread takes the
    pool's lock and asks the request's event loop to let it go. The list
    returned gets True when the loop did, and False when it did not
    within LOOP_WAIT, as when the helper waits on the lock on the loop."""
    real, out = getattr(routes, name), []

    def wrapped(*a, **k):
        locked, free = threading.Event(), threading.Event()
        loop = loops[-1]

        def holder():
            with pool._lock:
                locked.set()
                loop.call_soon_threadsafe(free.set)
                out.append(free.wait(LOOP_WAIT))
        threading.Thread(target=holder, daemon=True).start()
        assert locked.wait(LOOP_WAIT)
        return real(*a, **k)

    monkeypatch.setattr(routes, name, wrapped)
    return out


def test_a_steps_route_checks_its_lora_without_the_pools_lock(
        request, monkeypatch, tmp_path):
    """Another thread holds the pool's lock as a step's route checks its
    LoRA bytes against the budget. The request's event loop runs on: the
    check reads the pool's figures alone."""
    pool, e, p, pairs = _own_pairs(request, monkeypatch, tmp_path)
    loops: list = []
    c = _loop_client(loops)
    rows = _rows(c, _features())
    tid = _pack_sync(c, pairs, 2.0)()[1]
    x = noise_from_seed(3, SHAPE)
    free = _locked_at(monkeypatch, pool, "_hopeless", loops)
    c.step("own", None, x, [0.75], rows, patch_table=tid)
    assert free == [True], "the route's check waited on the pool's lock"


def test_a_steps_409_names_its_memory_without_the_pools_lock(
        request, monkeypatch, tmp_path):
    """A step whose table's pairs and matrices the server let go meets a
    409 with a memory sentence, written while another thread holds the
    pool's lock: it takes none."""
    pool, e, p, pairs = _own_pairs(request, monkeypatch, tmp_path)
    c = _client()
    rows = _rows(c, _features())
    tid = _pack_sync(c, pairs, 2.0)()[1]
    x = noise_from_seed(3, SHAPE)
    c.step("own", None, x, [0.75], rows, patch_table=tid)
    image_service.shed_lora(e, p.adapters.bytes + p.tables.bytes)
    steps.STATE.adapters.clear()
    out: list = []
    with pool._lock:
        t = threading.Thread(target=lambda: out.append(
            routes._table_hint(e.pipeline, tid)), daemon=True)
        t.start()
        t.join(LOOP_WAIT)
        assert not t.is_alive(), "the 409's sentence waited on the lock"
    with pytest.raises(MissingTensors) as missing:
        c.step("own", None, x, [0.75], rows, patch_table=tid)
    assert missing.value.missing == [tid]
    assert missing.value.body["error"].endswith(out[0])
    assert missing.value.body["error"] == (
        "1 tensor(s) or table(s) are not held by the server; upload them "
        "and retry. 0.1 GB of LoRA matrices are not held: the server let "
        "them go for memory")


# A route's estimate reads no pipeline.

def test_a_steps_estimate_is_its_matrices_whatever_the_pipeline_holds(
        request, monkeypatch, tmp_path):
    """A step's route counts the bytes of its table's matrices, the
    entry's own left out, whether the pipeline holds the pairs or a shed
    let them go: it reads no cache. Over the budget less the entry and
    the pinned models it is a 413 with the figures."""
    pool, e, c, tid, rows, x, first = _stepped(request, monkeypatch,
                                               tmp_path)
    pipe = e.pipeline.pipe
    mats = sum(steps.STATE.tables_of("own", pipe).sizes(tid).values())
    assert mats > 0
    assert routes._step_estimate(e.pipeline, tid) == mats
    image_service.shed_lora(e, pipe.adapters.bytes + pipe.tables.bytes)
    assert routes._step_estimate(e.pipeline, tid) == mats
    again = c.step("own", None, x, [0.75], rows, patch_table=tid)
    assert np.array_equal(again, first)
    _books_hold(pool, e)
    _set_budget(pool, e.footprint + mats - 1)
    with pytest.raises(StepsClientError) as refused:
        c.step("own", None, x, [0.75], rows, patch_table=tid)
    assert refused.value.status == 413
    assert refused.value.body["error"].startswith(
        "the LoRA factors of this step need 0.1 GB of memory, over the ")
    assert refused.value.body["error"].endswith(
        " GB the server's memory budget leaves beside 'own' and the pinned "
        "models")
