"""The video step's LoRA bytes, booked with the residency pool: a tiny
MiniMax-H3 entry with a LoRA of its own runs encode and step with its
books equal to what its caches hold. The encode lets the DiT and every
pair go; the own pairs load again with the DiT, within the footprint,
and the step's turns book a client's table and the own table's excess
again."""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")
mx = pytest.importorskip("mlx.core")

from gmlx.serve import video_service as vs  # noqa: E402
from gmlx.serve.image_steps_client import StepsClientError  # noqa: E402
from gmlx.serve.lora_books import STORE  # noqa: E402
from gmlx.serve.residency import IMAGE_KIND, _BusyHold  # noqa: E402
from test_images_api_lora import _books_hold  # noqa: E402
from test_step_routes_lora import GB, _crowd, _pool, _store  # noqa: E402
from test_video_steps import (FRAMES, SCALE, SHIFTS, SIZE,  # noqa: E402
                              _clip, _client,
                              _Host, _lora_file, _mount, _on_worker,
                              _own_pipe, _rel, _served, _stub_encode,
                              _table_body)


class _PoolHost(_Host):
    """The tiny H3 entry as an entry of ``pool`` of ``fp`` bytes."""

    def __init__(self, served, pool, fp: int = GB):
        super().__init__(served)
        self.pool, self.fp = pool, int(fp)

    def lora_pool(self):
        return self.pool

    def entry(self):
        spec = self.served.spec
        e = self.pool.acquire(
            spec.path, None, IMAGE_KIND, loaded_as=spec.id,
            build_spec=SimpleNamespace(fp=self.fp, stream=None,
                                       path=spec.path, pipe=self.served))
        self.pool.release(e)
        return e

    def acquire(self, model):
        if (model or "h3") != "h3":
            return super().acquire(model)
        e = self.entry()
        with self.pool._lock:
            e.busy += 1
        return _BusyHold(self.pool, e)


def _bf16_lora(path, seed: int):
    """The LoRA of ``_adapter`` ``seed`` written in bfloat16, as most
    published H3 LoRAs are."""
    from test_video_h3_lora import _adapter, _comfy_keys

    mx.save_safetensors(str(path), {
        k: mx.array(v).astype(mx.bfloat16)
        for k, v in _comfy_keys(_adapter(seed)).items()})
    return path


def _entry(request, monkeypatch, tmp_path, *, own=(5,), bf16=False,
           budget: int = 4 * GB):
    """The tiny H3 entry "h3" with the LoRAs of ``_adapter`` of each seed
    in ``own`` as its own, in float32 or in bfloat16, priced by the
    entry's plan, in a pool of ``budget`` bytes, with the encoder stood
    in."""
    from gmlx.video import memfit

    pipe, weights = _own_pipe()
    served = _served(pipe)
    paths = [_bf16_lora(tmp_path / f"own{s}.safetensors", s) if bf16
             else _lora_file(tmp_path, s, f"own{s}.safetensors")
             for s in own]
    served.spec.adapters = tuple((str(p), 1.0) for p in paths)
    plan = dataclasses.replace(
        served.plan, loras=tuple(p.stat().st_size for p in paths),
        lora_dtypes=tuple(memfit.lora_dtype(str(p)) for p in paths))
    clip = vs.plan_clip(plan, frames=FRAMES, width=SIZE, height=SIZE,
                        text=6, decode=False, budget=1)
    served.plan = dataclasses.replace(
        plan, lora_term=int(clip.terms["denoise"]["lora"]))
    _on_worker(vs._load_adapters, served)
    pool = _pool(monkeypatch, budget)
    host = _mount(_PoolHost(served, pool), request)
    _stub_encode(monkeypatch, weights)
    e = host.entry()
    with pool._lock:
        assert pool.books.rec(e).own_term == served.lora_own_term == (
            served.plan.lora_term) > 0
    c = _client()
    video, audio, feats = _clip()
    return SimpleNamespace(pool=pool, e=e, pipe=pipe, served=served, c=c,
                           video=video, audio=audio,
                           row={"cond": c.put(feats)})


def _step(t, **kw):
    return t.c.av_step("h3", None, t.video, t.audio, 600.0, t.row,
                       shifts=SHIFTS, audio_scale=SCALE, **kw)


def _held_never_negative(t) -> None:
    with t.pool._lock:
        t.pool.books.drain()
        assert t.pool.books.held(t.e) >= 0
        assert t.pool.books.held(STORE) == sum(_store().values())


def test_a_video_entrys_encodes_and_steps_book_its_request_keys_alone(
        request, monkeypatch, tmp_path):
    """A client's table on an entry with its own LoRA runs step, encode,
    step, encode, step. After each, the ledger holds the request keys the
    caches hold and the excess of the own pairs and default table, the
    own keys are never in it, held is never below zero, and each step
    gives the first one's output (G1)."""
    from test_video_h3_lora import _adapter

    t = _entry(request, monkeypatch, tmp_path)
    tid = t.c.define_table("h3", _table_body(t.c, _adapter(6)))
    first = _step(t, patch_table=tid)
    _books_hold(t.pool, t.e)
    _held_never_negative(t)
    _held, _pending, ledger = t.pool.lora_held(t.e)
    assert ledger
    for _ in range(2):
        t.c.encode("h3", "a cat")
        assert t.pipe.adapters.bytes == t.pipe.tables.bytes == 0
        assert t.pool.lora_held(t.e) == (0, 0, {})
        _books_hold(t.pool, t.e)
        _held_never_negative(t)
        again = _step(t, patch_table=tid)
        assert all(_rel(a, b) < 1e-6 for a, b in zip(again, first))
        # The entry keeps the refined rows of the cond, so the token
        # refiner's leaves build no factors again.
        got = t.pool.lora_held(t.e)[2]
        assert got.keys() == ledger.keys()
        assert all(0 < got[k] <= ledger[k] for k in got)
        _books_hold(t.pool, t.e)
        _held_never_negative(t)


@pytest.mark.parametrize("room", [True, False])
def test_a_video_step_after_an_encode_books_the_own_tables_excess_again(
        request, monkeypatch, tmp_path, room):
    """A step on no table runs the entry's own LoRAs, two bfloat16 files
    stacked. Their pairs and default table past the plan's term are booked
    as the excess, which the encode returns as it lets them go. The next
    step loads the own pairs again within the footprint and books the
    excess again at its turn, with the same output. With the memory taken
    by then it answers 503 and builds no table."""
    t = _entry(request, monkeypatch, tmp_path, own=(5, 6), bf16=True)
    first = _step(t)
    excess = t.pool.books.excess(t.e)
    assert excess > 0
    _books_hold(t.pool, t.e)
    t.c.encode("h3", "a cat")
    assert t.pool.lora_held(t.e) == (0, 0, {})
    assert t.pool.books.excess(t.e) == 0
    _books_hold(t.pool, t.e)
    if not room:
        with t.pool._lock:
            t.e.busy += 1
        _crowd(t.pool)
        t.pool.release(t.e)
        with pytest.raises(StepsClientError) as e:
            _step(t)
        assert e.value.status == 503
        assert e.value.body["error"].startswith(
            "the LoRA factors of this step need ")
        assert t.pipe.tables.bytes == 0
        _books_hold(t.pool, t.e, budget=False)
        _held_never_negative(t)
        return
    again = _step(t)
    assert all(_rel(a, b) < 1e-6 for a, b in zip(again, first))
    # The refined rows are kept: the token refiner's leaves build no
    # factors again.
    assert 0 < t.pool.books.excess(t.e) <= excess
    _books_hold(t.pool, t.e)
    _held_never_negative(t)
