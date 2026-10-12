#!/usr/bin/env python3
"""The image job of the serve routes over a stub pipeline: the stop event
ends the sampler, and each image of a request takes its own seed. The
sampler and the VAE are stubs, so nothing heavy runs."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from gmlx.image import api  # noqa: E402
from gmlx.image.registry import get_family  # noqa: E402
from gmlx.serve import image_service as svc  # noqa: E402


def _served():
    pipe = SimpleNamespace(
        family=get_family("z_image"), variant="turbo",
        defaults=SimpleNamespace(steps=4, strength=0.6), encoder=None,
        vae_scale=8, dit=SimpleNamespace(cfg=SimpleNamespace(patch=2)),
        sigmas=lambda steps, n: [1.0 - i / steps for i in range(steps + 1)])
    return SimpleNamespace(
        pipe=pipe, adapters=(),
        spec=SimpleNamespace(id="zit", path="/zit.gguf", defaults={},
                             encoder=None, encoder_model=None),
        plan=SimpleNamespace(keep_encoder=True, cfg=1.0, size=(64, 64)))


@pytest.fixture
def stubs(monkeypatch):
    rec = SimpleNamespace(seeds=[], stop_at=None)
    cond = SimpleNamespace(features=mx.zeros((5, 4)), image_slots=(),
                           refs=(), ref_latents=())
    monkeypatch.setattr(svc, "_encode", lambda *a, **k: (cond, None))
    monkeypatch.setattr(api, "plan_request",
                        lambda *a, **k: SimpleNamespace(release_encoder=False))

    def sample(pipe, cond, *, seed, steps, should_stop, on_step, **kw):
        rec.seeds.append(seed)
        x = mx.full((1, 16, 8, 8), float(seed))
        for i in range(steps):
            if rec.stop_at == i:
                rec.stop.set()
            if should_stop():
                raise api.SampleStopped(i, x)
            on_step(i, x)
        return x

    def vae_decode(pipe, x, *, alpha, plan):
        return mx.full((1, 8, 8, 3), (float(x.reshape(-1)[0].item()) % 10) / 10)

    monkeypatch.setattr(api, "sample", sample)
    monkeypatch.setattr(api, "vae_decode", vae_decode)
    return rec


def test_a_set_stop_event_ends_the_sampler_between_steps(stubs):
    stop = threading.Event()
    stubs.stop, stubs.stop_at = stop, 1
    req = svc.parse_request("generations", {"prompt": "x"})
    with pytest.raises(svc.JobStopped) as e:
        svc.run_job(_served(), req, size=(64, 64), budget=None, stop=stop)
    assert e.value.timed_out is False


def test_each_image_of_a_request_takes_the_next_seed(stubs):
    req = svc.parse_request("generations", {"prompt": "x", "n": 2,
                                            "seed": 5})
    out = svc.run_job(_served(), req, size=(64, 64), budget=None,
                      stop=threading.Event())
    assert stubs.seeds == [5, 6] and out.seeds == [5, 6]
    assert len(out.images) == 2 and out.images[0] != out.images[1]
