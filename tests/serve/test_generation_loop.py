"""The engine's batch loop lets go of a request's inputs once the request is
in the batch: an idle server must not keep the last request's images or
audio rows alive until the next request comes. It builds the batch with the
server's logprob cap and a neutral default row, and inserts each request
with its own row settings. The prompt embedding reads the same settings,
so a token embedding adapter embeds at the row's strengths. No model is
loaded."""

from __future__ import annotations

import gc
import importlib
import queue
import threading
import types
import weakref

import mlx.core as mx
import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")

_GEN = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")


class _Inputs:
    """Stands in for a request's images or encoded audio."""


class _Batch:
    unprocessed_prompts: list = []
    has_work = False
    made: list = []

    def __init__(self, *a, **kw):
        self.uid = 0
        self.kw = kw
        self.inserts = []
        self.made.append(self)

    def insert(self, prompts, **kw):
        self.uid += 1
        self.inserts.append(kw.get("row_settings"))
        return [self.uid]

    def close(self):
        pass


def _response_generator(monkeypatch):
    monkeypatch.setattr(_GEN, "BatchGenerator", _Batch)
    monkeypatch.setattr(_Batch, "made", [])
    monkeypatch.setattr(_GEN, "is_diffusion_model", lambda m: False)
    monkeypatch.setattr(_GEN, "_ServerTokenStreamer", lambda *a: None)
    monkeypatch.setattr(_GEN, "make_streaming_detokenizer", lambda p: None)
    rg = object.__new__(_GEN.ResponseGenerator)
    rg._stop = False
    rg.requests = queue.Queue()
    rg._cancel_lock = threading.Lock()
    rg._cancelled = set()
    rg._ready = threading.Event()
    rg._load_error = None
    rg.model = types.SimpleNamespace(language_model=None)
    rg.processor = rg.tokenizer = rg.config = None
    rg.stop_tokens = rg.apc_manager = rg.draft_model = rg.draft_kind = None
    rg.kv_bits = rg.kv_group_size = rg.kv_quant_scheme = None
    rg.quantized_kv_start = rg.top_logprobs_k = None
    rg._initialize_model = lambda: None
    rg._make_sampler = lambda args: None
    rg._make_logits_processors = lambda args, ids: []
    rg._effective_prefill_step_size = lambda: 2048
    rg._gpu_embed = lambda raw, images, apc_semantic_hash=None, row_settings=None: (
        mx.array([[1, 2]]), {"inputs_embeds": raw["audio_clips"]})

    def step(batch_gen, active):
        for uid in list(active):
            active.pop(uid)["rqueue"].put(None)

    rg._step = step
    return rg


def _request(rg, args, raw=None):
    rq: queue.Queue = queue.Queue()
    rg.requests.put(_GEN.QueuedGenerationRequest(
        rqueue=rq, raw_inputs=raw or {"input_ids": None, "audio_clips": []},
        prompt_tokens=2, args=args))
    assert isinstance(rq.get(timeout=5), _GEN.GenerationContext)
    assert rq.get(timeout=5) is None


def _stop(rg, thread):
    rg._stop = True
    rg.requests.put(None)
    thread.join(timeout=5)


def test_an_idle_loop_holds_no_finished_requests_inputs(monkeypatch):
    rg = _response_generator(monkeypatch)
    finished = threading.Event()
    idle = threading.Event()
    collect = rg._collect_pending_requests

    def collect_pending(**kw):
        if finished.is_set():
            idle.set()
        return collect(**{**kw, "idle_timeout": 0.01})

    rg._collect_pending_requests = collect_pending
    thread = threading.Thread(target=rg._run_impl, daemon=True)
    thread.start()
    try:
        inputs = _Inputs()
        alive = weakref.ref(inputs)
        rq: queue.Queue = queue.Queue()
        rg.requests.put(_GEN.QueuedGenerationRequest(
            rqueue=rq, raw_inputs={"input_ids": None, "audio_clips": [inputs]},
            prompt_tokens=2, args=_GEN.GenerationArguments(max_tokens=1)))
        del inputs
        assert isinstance(rq.get(timeout=5), _GEN.GenerationContext)
        assert rq.get(timeout=5) is None
        finished.set()
        assert idle.wait(timeout=5)
        gc.collect()
        assert alive() is None
    finally:
        _stop(rg, thread)


class _Sampler:
    def __init__(self, temperature):
        self.temperature = temperature
        self._kq_row_seeds = {}


def test_the_loop_passes_the_cap_and_each_requests_row_settings(monkeypatch):
    from gmlx.serve.row_sampling import GREEDY_KEY, RowSettings, sampler_key

    rg = _response_generator(monkeypatch)
    rg.top_logprobs_k = 7
    rg._make_sampler = lambda args: (
        _Sampler(args.temperature) if args.temperature else None)
    thread = threading.Thread(target=rg._run_impl, daemon=True)
    thread.start()
    sampled = _GEN.GenerationArguments(
        max_tokens=1, temperature=0.7, seed=5, logprobs=True, top_logprobs=3)
    try:
        _request(rg, _GEN.GenerationArguments(max_tokens=1, temperature=0.0))
        _request(rg, sampled)
    finally:
        _stop(rg, thread)

    # The first request asked for no logprobs, yet its batch takes the cap.
    for batch in _Batch.made:
        assert batch.kw["logprobs_cap"] == 7
        assert batch.kw["default_row"] == RowSettings()
    first, second = (rows for b in _Batch.made for rows in b.inserts)
    assert first == [RowSettings(key=GREEDY_KEY)]
    (row,) = second
    assert row.sampler.temperature == 0.7
    assert (row.seed, row.logprobs, row.top_k) == (5, True, 3)
    assert row.key == sampler_key(sampled)


def _lora_args(scales, salt=None):
    import numpy as np

    from gmlx.serve.lora_request import RowLora

    args = _GEN.GenerationArguments(max_tokens=1)
    args._kq_lora = RowLora(tuple(scales), np.asarray(scales, np.float32),
                            salt, "m")
    return args


def test_an_adapted_row_without_identities_fails_its_request(monkeypatch):
    from gmlx.serve.lora_request import AdapterIdentityMissing

    rg = _response_generator(monkeypatch)
    thread = threading.Thread(target=rg._run_impl, daemon=True)
    thread.start()
    try:
        rq: queue.Queue = queue.Queue()
        rg.requests.put(_GEN.QueuedGenerationRequest(
            rqueue=rq, raw_inputs={"input_ids": None, "audio_clips": []},
            prompt_tokens=2, args=_lora_args((1.0,))))
        got = rq.get(timeout=5)
        assert isinstance(got, AdapterIdentityMissing)
        assert str(got) == ("model 'm' has no loaded adapter identities; this is "
                            "a gmlx defect")
        # A base row on the same copy needs no identity.
        _request(rg, _lora_args((0.0,), salt=0))
    finally:
        _stop(rg, thread)
    rows = [rows for b in _Batch.made for rows in b.inserts]
    assert rows[-1][0].lora.salt == 0


def test_an_adapted_row_is_salted_from_its_copys_identities(monkeypatch):
    from types import SimpleNamespace

    from gmlx import lora_rows

    rg = _response_generator(monkeypatch)
    ids = (SimpleNamespace(sha256="a" * 64),)
    object.__setattr__(rg.model, "gmlx_adapter_identities", ids)
    thread = threading.Thread(target=rg._run_impl, daemon=True)
    thread.start()
    try:
        _request(rg, _lora_args((0.5,)))
    finally:
        _stop(rg, thread)
    (row,) = [rows for b in _Batch.made for rows in b.inserts][-1]
    assert row.lora.salt == lora_rows.row_salt(ids, (0.5,)) != 0


# the prompt embedding publishes the row's adapter scales


def _embedding_model():
    import mlx.nn as nn
    import numpy as np
    from types import SimpleNamespace

    from gmlx._vendor.mlx_vlm.models.base import InputEmbeddingsFeatures
    from gmlx.load.modules import LoRAKQuantEmbedding

    base = nn.Embedding(10, 4)
    base.weight = mx.array((np.arange(40, dtype=np.float32).reshape(10, 4) % 5) * 0.5)
    a = mx.array((np.arange(20, dtype=np.float32).reshape(10, 2) % 3) - 1.0)
    b = mx.array((np.arange(8, dtype=np.float32).reshape(4, 2) % 4) * 0.25)
    wrap = LoRAKQuantEmbedding(base, a, b, 0.5)

    def get_input_embeddings(input_ids, pixel_values=None, mask=None, **kw):
        return InputEmbeddingsFeatures(inputs_embeds=wrap(input_ids))

    model = SimpleNamespace(language_model=None, get_input_embeddings=get_input_embeddings)
    object.__setattr__(model, "gmlx_adapter_identities", (SimpleNamespace(sha256="a" * 64),))
    return model, wrap, base


class _EmbedsBatch(_Batch):
    embeds: list = []

    def insert(self, prompts, **kw):
        embeds = kw["prompt_kwargs"][0]["inputs_embeds"]
        mx.eval(embeds)                  # on the loop thread's stream
        self.embeds.append(embeds)
        return super().insert(prompts, **kw)


def _embed_request(rg, args, ids):
    rq: queue.Queue = queue.Queue()
    rg.requests.put(_GEN.QueuedGenerationRequest(
        rqueue=rq, raw_inputs={"input_ids": ids}, prompt_tokens=int(ids.shape[1]), args=args))
    return rq


@pytest.fixture
def rows_mode():
    from gmlx import lora_rows

    lora_rows.configure("rows", 1)
    yield
    lora_rows.configure("static", 1)


def test_the_prompt_embedding_runs_at_the_rows_strength(monkeypatch, rows_mode):
    from gmlx import lora_rows

    rg = _response_generator(monkeypatch)
    monkeypatch.setattr(_GEN, "BatchGenerator", _EmbedsBatch)
    monkeypatch.setattr(_EmbedsBatch, "embeds", [])
    rg.model, wrap, base = _embedding_model()
    rg.vision_cache = None
    real = _GEN.ResponseGenerator._gpu_embed
    read = []

    def gpu_embed(*a, **kw):
        read.append(kw.get("row_settings"))
        return real(rg, *a, **kw)

    rg._gpu_embed = gpu_embed
    ids = mx.array([[1, 4, 7]])
    thread = threading.Thread(target=rg._run_impl, daemon=True)
    thread.start()
    try:
        for scales in ((1.0,), (0.0,)):
            rq = _embed_request(rg, _lora_args(scales), ids)
            assert isinstance(rq.get(timeout=5), _GEN.GenerationContext)
            assert rq.get(timeout=5) is None
    finally:
        _stop(rg, thread)
    inserted = [rows[0] for b in _Batch.made for rows in b.inserts]
    assert len(inserted) == len(read) == 2
    assert all(r is not None and i is r for i, r in zip(inserted, read))
    adapted, bare = _EmbedsBatch.embeds
    lora_rows.configure("static", 1)
    assert mx.array_equal(adapted, wrap(ids))
    assert mx.array_equal(bare, base(ids))
    assert not mx.array_equal(adapted, bare)


def test_a_row_without_adapter_scales_fails_only_its_request(monkeypatch, rows_mode):
    from gmlx import lora_rows

    rg = _response_generator(monkeypatch)
    rg.model, _wrap, _base = _embedding_model()
    rg.vision_cache = None
    rg._gpu_embed = lambda *a, **kw: _GEN.ResponseGenerator._gpu_embed(rg, *a, **kw)
    ids = mx.array([[2, 3]])
    thread = threading.Thread(target=rg._run_impl, daemon=True)
    thread.start()
    try:
        missing = _embed_request(rg, _GEN.GenerationArguments(max_tokens=1), ids)
        assert isinstance(missing.get(timeout=5), lora_rows.LoraRowsError)
        rq = _embed_request(rg, _lora_args((1.0,)), ids)
        assert isinstance(rq.get(timeout=5), _GEN.GenerationContext)
        assert rq.get(timeout=5) is None
    finally:
        _stop(rg, thread)


def test_the_context_carries_the_ids_the_worker_inserted(monkeypatch):
    rg = _response_generator(monkeypatch)
    inserted = []
    insert = _Batch.insert

    def recording(self, prompts, **kw):
        inserted.append(list(prompts[0]))
        return insert(self, prompts, **kw)

    monkeypatch.setattr(_Batch, "insert", recording)
    rg._gpu_embed = lambda raw, images, apc_semantic_hash=None, row_settings=None: (
        mx.array([[7, 8, 9]]), {})
    thread = threading.Thread(target=rg._run_impl, daemon=True)
    thread.start()
    try:
        rq: queue.Queue = queue.Queue()
        rg.requests.put(_GEN.QueuedGenerationRequest(
            rqueue=rq, raw_inputs={"input_ids": None}, prompt_tokens=3,
            args=_GEN.GenerationArguments(max_tokens=1)))
        ctx = rq.get(timeout=5)
        assert rq.get(timeout=5) is None
    finally:
        _stop(rg, thread)
    assert inserted == [[7, 8, 9]]
    assert ctx.prompt_ids == [7, 8, 9]


def test_each_new_batch_numbers_on_from_the_last(monkeypatch):
    """A uid names one request for the server's life, so a cancel that
    arrives after the batch was replaced names no new request."""
    rg = _response_generator(monkeypatch)

    def insert(self, prompts, **kw):
        uid = self.uid_count
        self.uid_count += 1
        return [uid]

    monkeypatch.setattr(_Batch, "insert", insert)
    thread = threading.Thread(target=rg._run_impl, daemon=True)
    thread.start()
    uids = []
    try:
        for _ in range(2):
            rq: queue.Queue = queue.Queue()
            rg.requests.put(_GEN.QueuedGenerationRequest(
                rqueue=rq, raw_inputs={"input_ids": None, "audio_clips": []},
                prompt_tokens=2, args=_GEN.GenerationArguments(max_tokens=1)))
            uids.append(rq.get(timeout=5).uid)
            assert rq.get(timeout=5) is None
    finally:
        _stop(rg, thread)
    assert len(_Batch.made) == 2
    assert uids == [0, 1]


class _Cache:
    """Stands in for a batch's KV cache."""


class _Fault(RuntimeError):
    """The engine's error, weakly referable."""


class _FailingBatch(_Batch):
    """A batch whose step fails once armed, from a frame that holds the
    batch. Its text prompts stay pending, so an embedding request drains
    them through a step first."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.cache = _Cache()
        self.unprocessed_prompts = []
        self.armed = False

    @property
    def has_pending_prompts(self):
        return bool(self.unprocessed_prompts)

    def insert(self, prompts, **kw):
        self.unprocessed_prompts.append(
            (None, prompts[0], None, kw["prompt_kwargs"][0]))
        return super().insert(prompts, **kw)

    def step(self):
        if self.armed:
            raise _Fault("[metal::malloc] Attempting to allocate 1 bytes")


def test_a_gpu_thread_error_frees_the_batch_by_refcount(monkeypatch):
    """Seven rows decode and an eighth waits for its context when the
    engine fails. Every request gets the error: the seven from their
    token streams, the eighth from generate. With the collector off and
    no epilogue collect, the batch and the error are freed once the
    requests and the loop let go of them."""
    rg = _response_generator(monkeypatch)
    monkeypatch.setattr(_GEN, "BatchGenerator", _FailingBatch)
    # Log records keep the frames; the error must be the only holder.
    monkeypatch.setattr(_GEN.logger, "disabled", True)
    collected = threading.Event()
    monkeypatch.setattr(_GEN, "gc", types.SimpleNamespace(
        collect=collected.set))
    rg._make_thinking_budget_criteria = lambda args, ids: None
    rg._preprocess_request = lambda prompt, images, audio, videos: (
        {"input_ids": [1, 2], "embeds": audio} if audio
        else {"input_ids": [1, 2]})
    batch = []

    def gpu_embed(raw, images, apc_semantic_hash=None, row_settings=None):
        if "embeds" not in raw:
            return mx.array([[1, 2]]), {}
        batch[0]().armed = True
        return mx.array([[1, 2]]), {"inputs_embeds": raw["embeds"]}

    rg._gpu_embed = gpu_embed
    rg._step = lambda batch_gen, active: batch_gen.step()
    idle = threading.Event()
    collect = rg._collect_pending_requests

    def collect_pending(**kw):
        if collected.is_set():
            idle.set()
        return collect(**{**kw, "idle_timeout": 0.01})

    rg._collect_pending_requests = collect_pending
    started: queue.Queue = queue.Queue()
    errors, faults = [], []

    def request(audio=None):
        try:
            _ctx, tokens = rg.generate(
                "p", audio=audio, args=_GEN.GenerationArguments(max_tokens=4))
            started.put(True)
            for _ in tokens:
                pass
        except _Fault as e:
            errors.append(str(e))
            faults.append(weakref.ref(e))

    gc.disable()
    loop = threading.Thread(target=rg._run_impl, daemon=True)
    loop.start()
    try:
        rows = [threading.Thread(target=request) for _ in range(7)]
        for t in rows:
            t.start()
        for _ in rows:
            assert started.get(timeout=5)
        (made,) = _Batch.made
        _Batch.made.clear()
        batch.append(weakref.ref(made))
        cache = weakref.ref(made.cache)
        del made
        rows.append(threading.Thread(target=request, args=(["clip"],)))
        rows[-1].start()
        for t in rows:
            t.join(timeout=5)
        assert idle.wait(timeout=5)
        assert len(errors) == 8 and len(set(errors)) == 1
        assert batch[0]() is None and cache() is None
        assert [f() for f in faults] == [None] * 8
    finally:
        gc.enable()
        _stop(rg, loop)


def test_an_idle_worker_waits_for_peers_only_while_one_is_arriving(monkeypatch):
    """The wait that lets a burst land in one batch runs only while
    generate() is preparing another request. A lone request starts at
    once."""
    rg = object.__new__(_GEN.ResponseGenerator)
    rg._stop = False
    rg.requests = queue.Queue()
    rg._kq_arriving = 0
    rg._kq_arriving_lock = threading.Lock()
    slept = []
    monkeypatch.setattr(_GEN.time, "sleep", slept.append)

    rg.requests.put("first")
    assert rg._collect_pending_requests(
        active=False, coalesce_s=0.005) == (["first"], False)
    assert slept == []

    rg.requests.put("first")
    rg._kq_arriving = 1
    assert rg._collect_pending_requests(
        active=False, coalesce_s=0.005) == (["first"], False)
    assert slept == [0.005]


def test_a_request_counts_as_arriving_until_it_is_queued(monkeypatch):
    """generate() counts a request from its start until its queue put
    returns, so a worker never reads it as gone before it can see it; a
    request that fails in preparation leaves the count as it was."""
    rg = object.__new__(_GEN.ResponseGenerator)
    rg._kq_arriving = 0
    rg._kq_arriving_lock = threading.Lock()
    rg.draft_model = rg.apc_mode = None
    rg.wait_until_ready = lambda: None
    rg._make_thinking_budget_criteria = lambda args, ids: None
    seen = []

    def preprocess(prompt, images=None, audio=None, videos=None):
        seen.append(rg._kq_arriving)
        if prompt == "bad":
            raise ValueError("bad prompt")
        return {"input_ids": [1, 2, 3]}

    rg._preprocess_request = preprocess

    class _Requests:
        def put(self, request):
            seen.append(rg._kq_arriving)
            request.rqueue.put(_GEN.GenerationContext(uid=0, prompt_tokens=3))

    rg.requests = _Requests()
    ctx, _tokens = rg.generate("hello", args=_GEN.GenerationArguments(max_tokens=1))
    assert ctx.prompt_tokens == 3
    assert seen == [1, 1] and rg._kq_arriving == 0
    with pytest.raises(ValueError):
        rg.generate("bad", args=_GEN.GenerationArguments(max_tokens=1))
    assert rg._kq_arriving == 0


# --- a held prompt cache store (gmlx.spec.speculative) ---------------------

def _hold(monkeypatch, key):
    """A held store under ``key`` that records its run."""
    spec = importlib.import_module("gmlx.spec.speculative")
    order: list = []
    ran = threading.Event()

    def job():
        order.append("store")
        ran.set()

    monkeypatch.setattr(spec, "_round_log_note", lambda note: None)
    # The worker thread runs a store this test thread holds for it.
    monkeypatch.setattr(spec, "_owner", lambda: "worker")
    monkeypatch.setattr(spec, "_held_stores", [spec._HeldStore(job, key, 0)])
    return order, ran


def test_an_idle_worker_runs_a_held_store(monkeypatch):
    monkeypatch.setenv("GMLX_SPEC_STORE_IDLE_MS", "1")
    rg = _response_generator(monkeypatch)
    order, ran = _hold(monkeypatch, [9])
    thread = threading.Thread(target=rg._run, daemon=True)
    thread.start()
    try:
        assert ran.wait(5) and order == ["store"]
    finally:
        _stop(rg, thread)


@pytest.mark.parametrize("key, order", [
    ([1], ["store", "insert"]),   # the request continues the stored sequence
    ([9], ["insert", "store"]),   # another conversation starts first
])
def test_a_request_that_can_use_a_held_store_waits_for_it(monkeypatch, key, order):
    monkeypatch.setenv("GMLX_SPEC_STORE_IDLE_MS", "1")
    rg = _response_generator(monkeypatch)
    seen, ran = _hold(monkeypatch, key)
    insert = _Batch.insert

    def recording_insert(self, prompts, **kw):
        seen.append("insert")
        return insert(self, prompts, **kw)

    monkeypatch.setattr(_Batch, "insert", recording_insert)
    rq: queue.Queue = queue.Queue()
    rg.requests.put(_GEN.QueuedGenerationRequest(
        rqueue=rq, raw_inputs={"input_ids": None, "audio_clips": []},
        prompt_tokens=2, args=_GEN.GenerationArguments(max_tokens=1)))
    thread = threading.Thread(target=rg._run, daemon=True)
    thread.start()
    try:
        assert isinstance(rq.get(timeout=5), _GEN.GenerationContext)
        assert ran.wait(5) and seen == order
    finally:
        _stop(rg, thread)


def test_a_stopping_worker_runs_a_held_store(monkeypatch):
    monkeypatch.setenv("GMLX_SPEC_STORE_IDLE_MS", "60000")
    rg = _response_generator(monkeypatch)
    rg._stop = True
    order, ran = _hold(monkeypatch, [9])
    thread = threading.Thread(target=rg._run, daemon=True)
    thread.start()
    thread.join(timeout=5)
    assert order == ["store"]


def test_a_step_runs_what_it_queued_after_its_responses(monkeypatch):
    spec = importlib.import_module("gmlx.spec.speculative")
    order: list = []

    def body(self, batch_gen, active, gen_kwargs=None):
        spec.step_end_queue().append(lambda: order.append("store"))
        order.append("first token")

    monkeypatch.setattr(_GEN.ResponseGenerator, "_step_body", body)
    _GEN.ResponseGenerator._step(types.SimpleNamespace(), None, {})
    assert order == ["first token", "store"]
    assert spec.step_end_queue() is None


def test_a_held_store_runs_a_step_after_every_rows_first_token(monkeypatch):
    order, ran = _hold(monkeypatch, [9])

    def body(self, batch_gen, active, gen_kwargs=None):
        for info in active.values():
            info.setdefault("kq_first_step", self._kq_step_no)

    monkeypatch.setattr(_GEN.ResponseGenerator, "_step_body", body)
    stub = types.SimpleNamespace()
    active = {1: {}}
    _GEN.ResponseGenerator._step(stub, None, active)
    assert order == []
    _GEN.ResponseGenerator._step(stub, None, active)
    assert order == ["store"]
