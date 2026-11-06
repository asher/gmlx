"""The engine's batch loop lets go of a request's inputs once the request is
in the batch: an idle server must not keep the last request's images or
audio rows alive until the next request comes. No model is loaded."""

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

    def __init__(self, *a, **kw):
        self.uid = 0

    def insert(self, prompts, **kw):
        self.uid += 1
        return [self.uid]

    def close(self):
        pass


def test_an_idle_loop_holds_no_finished_requests_inputs(monkeypatch):
    monkeypatch.setattr(_GEN, "BatchGenerator", _Batch)
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
    rg._gpu_embed = lambda raw, images, apc_semantic_hash=None: (
        mx.array([[1, 2]]), {"inputs_embeds": raw["audio_clips"]})

    def step(batch_gen, active):
        for uid in list(active):
            active.pop(uid)["rqueue"].put(None)

    rg._step = step
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
        rg._stop = True
        rg.requests.put(None)
        thread.join(timeout=5)
