#!/usr/bin/env python3
"""The image job of the serve routes over a stub pipeline: the stop event
ends the sampler, and each image of a request takes its own seed. The
sampler and the VAE are stubs, so nothing heavy runs."""
from __future__ import annotations

import json
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


def _flux3_served():
    served = _served()
    served.pipe.family = get_family("flux3_klein")
    served.pipe.variant = "distilled"
    served.pipe.size_multiple = 16
    served.pipe.vae_scale, served.pipe.dit.cfg.patch = 16, 1
    served.spec.encoder_mmproj = None
    return served


def _flux3_job(monkeypatch, fields, mask=None, emit=None):
    """An unsized FLUX.3 edit of one 100x60 image through the job, whose
    VAE decodes to the canvas it is given."""
    import numpy as np

    monkeypatch.setattr(api, "vae_decode", lambda pipe, x, *, alpha, plan:
                        mx.zeros((1, h, w, 3)))
    monkeypatch.setattr(api, "vae_encode", lambda pipe, img, *, plan:
                        mx.zeros((1, 32, h // 16, w // 16)))
    monkeypatch.setattr(api, "encode_references", lambda pipe, c, plan: c)
    served = _flux3_served()
    req = svc.parse_request("edits", fields)
    req.images = [np.full((60, 100, 3), 9, np.uint8)]
    req.mask = mask
    svc.check_family(req, served)
    w, h = svc.target_size(req, served)
    out = svc.run_job(served, req, size=(w, h), budget=None,
                      stop=threading.Event(), emit=emit)
    return (w, h), out


def test_a_masked_unsized_flux3_edit_keeps_the_size_of_its_image(
        stubs, monkeypatch):
    """A mask sizes the edit at the first image on the stride and blends
    its output into that image, so the canvas of the padded reference
    crops nothing."""
    import io

    import numpy as np
    from PIL import Image

    mask = np.zeros((60, 100, 4), np.uint8)
    size, out = _flux3_job(monkeypatch, {"prompt": "make it red"},
                           mask=mask)
    assert size == (96, 48) and out.size == size
    with Image.open(io.BytesIO(out.images[0])) as im:
        assert im.size == size


def test_the_partial_images_of_a_cropped_flux3_edit_are_cropped(
        stubs, monkeypatch):
    """Each partial image of a stream comes out at the size of the final
    one, cropped from the canvas of the padded reference."""
    import io

    from PIL import Image

    sent = []
    size, out = _flux3_job(
        monkeypatch, {"prompt": "make it red", "stream": "true",
                      "partial_images": "2"},
        emit=lambda i, data: sent.append(data))
    assert size != (100, 60) and out.size == (100, 60)
    assert len(sent) == 2
    for data in [*sent, *out.images]:
        with Image.open(io.BytesIO(data)) as im:
            assert im.size == (100, 60)


@pytest.mark.parametrize("route,size,want", [
    ("edits", None, (100, 60)), ("variations", None, (100, 60)),
    ("edits", "128x64", (128, 64))])
def test_a_flux3_edit_runs_its_prepared_references_and_crops_the_pad(
        stubs, monkeypatch, route, size, want):
    """The job conditions on the references check_family prepared, and an
    output on the canvas of one padded reference is cropped back to the
    image's box unless the request names its size. A variation takes the
    fixed instruction."""
    import io

    import numpy as np
    from PIL import Image

    seen = SimpleNamespace(refs=None, prompt=None, decoded=None)
    cond = SimpleNamespace(features=mx.zeros((5, 4)), image_slots=(),
                           refs=(), ref_latents=())

    def encode(served, prompt, neg, refs, borrow):
        seen.refs, seen.prompt = refs, prompt
        return cond, None

    def vae_decode(pipe, x, *, alpha, plan):
        seen.decoded = (w, h)
        return mx.zeros((1, h, w, 3))

    monkeypatch.setattr(svc, "_encode", encode)
    monkeypatch.setattr(api, "vae_decode", vae_decode)
    monkeypatch.setattr(api, "encode_references", lambda pipe, c, plan: c)
    served = _flux3_served()
    fields = {"prompt": "" if route == "variations" else "make it red"}
    if size:
        fields["size"] = size
    req = svc.parse_request(route, fields)
    req.images = [np.full((60, 100, 3), 9, np.uint8)]
    svc.check_family(req, served)
    w, h = svc.target_size(req, served)
    out = svc.run_job(served, req, size=(w, h), budget=None,
                      stop=threading.Event(), embed_metadata=True)
    assert seen.refs == tuple(req.refs) and seen.decoded == (w, h)
    assert seen.prompt == ("make it red" if route == "edits"
                           else svc.VARIATION_PROMPT)
    assert out.size == want
    with Image.open(io.BytesIO(out.images[0])) as im:
        assert im.size == want
        # The metadata records the size written, not the canvas.
        assert json.loads(im.text["gmlx"])["size"] == "%dx%d" % want
