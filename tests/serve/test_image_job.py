#!/usr/bin/env python3
"""The image job of the serve routes over a stub pipeline: the stop event
ends the sampler, and each image of a request takes its own seed. An
edit's schedule counts the image's tokens alone, and the negative row of
a guided edit takes the references. The sampler and the VAE are stubs,
so nothing heavy runs."""
from __future__ import annotations

import dataclasses
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
        plan=SimpleNamespace(keep_encoder=True, cfg=1.0, size=(64, 64)),
        facts=svc.DitFacts(act_dtype=mx.bfloat16, vae_dtype=None,
                           size_multiple=16, flow_map=False, ref_cache=False,
                           blocks=None))


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


@pytest.mark.parametrize("fields, revised, texts", [
    ({"prompt": "a cat", "negative_prompt": "blurry"}, None, ["a cat"]),
    ({"prompt": "a cat", "negative_prompt": "blurry", "guidance": 4.0},
     None, ["a cat", "blurry"]),
    ({"prompt": "a cat", "guidance": 4.0}, "a cat on a mat",
     ["a cat on a mat", ""]),
], ids=["unguided", "guided", "upsampled"])
def test_the_job_makes_room_for_the_texts_it_encodes(monkeypatch, fields,
                                                     revised, texts):
    """Before its encode a job makes room for the texts the encoder reads:
    the prompt, or the upsampler's rewrite, and under guidance the
    negative prompt."""
    seen = []

    class Room(Exception):
        pass

    def make_room(served, cfg, size, refs, init, budget, prompts=(),
                  int8=None):
        seen.append(list(prompts))
        raise Room

    monkeypatch.setattr(svc, "_make_room", make_room)
    req = svc.parse_request("generations", fields)
    req.revised_prompt = revised
    with pytest.raises(Room):
        svc.run_job(_served(), req, size=(64, 64), budget=None,
                    stop=threading.Event())
    assert seen == [texts]


def test_each_image_of_a_request_takes_the_next_seed(stubs):
    req = svc.parse_request("generations", {"prompt": "x", "n": 2,
                                            "seed": 5})
    out = svc.run_job(_served(), req, size=(64, 64), budget=None,
                      stop=threading.Event())
    assert stubs.seeds == [5, 6] and out.seeds == [5, 6]
    assert len(out.images) == 2 and out.images[0] != out.images[1]


def test_each_request_runs_its_own_int8_attention(stubs, monkeypatch):
    """A request that names int8 attention runs its steps with it and
    records it in the PNG, and the next request, naming none, runs the
    entry's dense attention again."""
    import io

    import mlx_kquant as kq
    from PIL import Image

    for name in ("sdpa_qk8_inputs", "sdpa_qk8"):
        monkeypatch.setattr(kq, name, getattr(kq, name, lambda *a, **k: None),
                            raising=False)
    monkeypatch.setattr(kq, "nax_available", lambda: True)
    served = _served()
    served.pipe.family, served.pipe.variant = get_family("h3_image"), "pruned"
    served.pipe.dit.attention = "dense"
    served.plan.int8 = "off"
    seen, real = [], api.sample
    monkeypatch.setattr(api, "sample", lambda pipe, cond, **kw: (
        seen.append(pipe.dit.attention), real(pipe, cond, **kw))[1])
    metas = []
    for fields in ({"int8": "attention"}, {}):
        req = svc.parse_request("generations", {"prompt": "x", **fields})
        svc.check_family(req, served)
        out = svc.run_job(served, req, size=(64, 64), budget=None,
                          stop=threading.Event(), embed_metadata=True)
        with Image.open(io.BytesIO(out.images[0])) as im:
            metas.append(json.loads(im.text["gmlx"]))
    assert seen == ["int8", "dense"]
    assert metas[0]["int8"] == "attention" and "int8" not in metas[1]


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


def _flux2_job(monkeypatch, variant, fields, shape=(256, 256, 3)):
    """A FLUX.2 klein edit of one image through check_family, target_size
    and the job, with the encode of the serve path and stub primitives
    under it. Returns the sizes the schedule was asked for, the encodes,
    and the conditionings the sampler got."""
    import dataclasses

    import numpy as np

    from gmlx.image.conditioning import Conditioning

    rec = SimpleNamespace(sigmas=[], encoded=[], sampled=[])
    served = _served()
    pipe = served.pipe
    pipe.family = get_family("flux2_klein")
    pipe.variant = variant
    pipe.size_multiple = 16
    pipe.vae_scale, pipe.dit.cfg.patch = 16, 1
    pipe.sigmas = lambda steps, n: rec.sigmas.append((steps, n)) or [
        1.0 - i / steps for i in range(steps + 1)]
    served.spec.encoder_mmproj = None
    served.plan.cfg = 4.0 if variant == "base" else 1.0

    def encode_text(pipe, prompt, *, images=(), ref_latents=True):
        rec.encoded.append((prompt, tuple(images)))
        return Conditioning(mx.zeros((5, 4)), refs=tuple(images))

    def encode_references(pipe, c, plan=None):
        return dataclasses.replace(c, ref_latents=tuple(
            mx.zeros((128, r.height // 16, r.width // 16)) for r in c.refs))

    def sample(pipe, cond, *, neg, cfg, height, width, seed, steps,
               should_stop, on_step, **kw):
        rec.sampled.append((cond, neg, cfg))
        return mx.zeros((1, 128, height // 16, width // 16))

    monkeypatch.setattr(api, "encode_text", encode_text)
    monkeypatch.setattr(api, "encode_references", encode_references)
    monkeypatch.setattr(api, "sample", sample)
    monkeypatch.setattr(api, "plan_request",
                        lambda *a, **k: SimpleNamespace(release_encoder=False))
    monkeypatch.setattr(api, "vae_decode", lambda pipe, x, *, alpha, plan:
                        mx.zeros((1, x.shape[2] * 16, x.shape[3] * 16, 3)))
    req = svc.parse_request("edits", fields)
    req.images = [np.full(shape, 9, np.uint8)]
    svc.check_family(req, served)
    size = svc.target_size(req, served)
    out = svc.run_job(served, req, size=size, budget=None,
                      stop=threading.Event())
    return rec, req, size, out


def test_a_flux2_edit_schedules_by_the_image_tokens_alone(monkeypatch):
    """A 256 x 256 reference holds 256 tokens beside the 16 of a 64 x 64
    image: the schedule reads the 16."""
    rec, req, size, out = _flux2_job(
        monkeypatch, "distilled", {"prompt": "make it red", "size": "64x64",
                                   "steps": "4"})
    assert size == (64, 64)
    (cond, _, _), = rec.sampled
    assert [z.shape for z in cond.ref_latents] == [(128, 16, 16)]
    assert rec.sigmas == [(4, 16)]
    assert out.usage["output_tokens"] == 16


def test_the_negative_row_of_a_guided_flux2_base_edit_takes_the_references(
        monkeypatch):
    """Base guides the prompt's row by the empty prompt's, and both rows
    carry the references, as the reference's denoise_cfg gives img_cond_seq
    to both."""
    rec, req, _, _ = _flux2_job(monkeypatch, "base",
                                {"prompt": "make it red"})
    refs = tuple(req.refs)
    assert rec.encoded == [("make it red", refs), ("", refs)]
    (cond, neg, cfg), = rec.sampled
    assert cfg == 4.0 and neg is not None
    assert neg.refs == cond.refs == refs
    assert [z.shape for z in neg.ref_latents] == [
        z.shape for z in cond.ref_latents] == [(128, 16, 16)]


@pytest.mark.parametrize("budget_gib, shed, levers", [
    (14, True, ("tiled decode",)), (16, True, ()), (32, False, ())])
def test_a_request_after_a_kv_edit_encodes_with_its_states_priced(
        monkeypatch, budget_gib, shed, levers):
    """A FLUX.2 klein 9B KV edit with four 1024 pixel references leaves
    the K and V of its references held, 8 GiB by the family's own state
    size. The next request runs its encode with them only when its plan
    counts them and fits: on a budget that does not hold them beside it,
    they go before the encode. They never change the levers the request
    takes, which are those it takes with nothing held."""
    from gmlx.image.conditioning import Conditioning
    from gmlx.image.memfit import GiB

    dit = SimpleNamespace(cfg=SimpleNamespace(patch=1, dim=4096, depth=8,
                                              depth_single=24),
                          act_dtype=mx.bfloat16)
    pipe = api.ImagePipeline(dit=dit, family=get_family("flux2_klein_9b_kv"),
                             variant="distilled",
                             weights={"dit": int(9.3 * GiB)})
    edit = Conditioning(mx.zeros((512, 3 * 4096)), "edit", ref_latents=tuple(
        mx.full((128, 64, 64), float(i)) for i in range(4)))
    held = pipe.family.step.state_size(dit, edit)
    assert held == 512 * 4096 * 2 + 8 * GiB
    pipe.captions.put(api._state_key(pipe, edit, api._table_id(pipe, None),
                                     (64, 64)), None, held, shed=False)
    seen, plans = [], []
    text = Conditioning(mx.zeros((512, 3 * 4096)), "next")

    def encode(*a, **k):
        seen.append(pipe.captions.bytes)
        return text, None

    real_plan = api.plan_request
    calls = []

    def plan_request(*a, **k):
        calls.append((a, k))
        plans.append(real_plan(*a, **k))
        return plans[-1]

    monkeypatch.setattr(svc, "_encode", encode)
    monkeypatch.setattr(api, "plan_request", plan_request)
    monkeypatch.setattr(api, "sample", lambda *a, **k: mx.zeros(
        (1, 128, 64, 64)))
    monkeypatch.setattr(api, "vae_decode", lambda *a, **k: mx.zeros(
        (1, 1024, 1024, 3)))
    served = SimpleNamespace(
        pipe=pipe, adapters=(),
        spec=SimpleNamespace(id="kv", path="/kv.gguf", defaults={},
                             encoder=None, encoder_model=None),
        plan=SimpleNamespace(keep_encoder=True, cfg=1.0, size=(1024, 1024)),
        facts=svc.dit_facts(pipe))
    req = svc.parse_request("generations", {"prompt": "x"})
    svc.run_job(served, req, size=(1024, 1024), budget=budget_gib * GiB,
                stop=threading.Event())
    (plan,) = plans
    assert seen == [0 if shed else held]
    assert plan.terms["prefix_held"] == seen[0] and plan.fits
    assert plan.phases["encode"] >= seen[0] + plan.terms["dit"]
    pipe.captions.clear()
    (a, k), = calls
    assert plan.levers == real_plan(*a, **k).levers == levers


def _h3_served(tmp_path, monkeypatch, entry=None):
    """An H3-Image entry whose one adapter is a file gmlx's table knows as
    pDMD-2steps, with the entry's schedule fields ``entry``."""
    from gmlx.image.lora import schedule as sc

    path = tmp_path / "pdmd.safetensors"
    path.write_bytes(b"pdmd")
    monkeypatch.setattr(sc, "KNOWN", {sc.file_sha256(str(path)): sc.Known(
        "pDMD-2steps", "linoyts/MiniMax-H3-Image-pDMD-2steps",
        sc.AdapterSchedule(sigmas=(1.0, 12 / 13, 0.0), heads="in_order",
                           size=(1024, 1024)))})
    served = _served()
    served.pipe.family, served.pipe.variant = get_family("h3_image"), "pruned"
    served.adapters = [(SimpleNamespace(path=str(path)), 1.0)]
    served.spec.adapter_schedules = (
        () if entry is None else ((str(path), entry),))
    served.plan.size = (2048, 2048)
    served.facts = dataclasses.replace(served.facts, head_bank=6)
    return served


def test_an_adapters_schedule_runs_on_serve_and_the_request_wins(
        stubs, monkeypatch, tmp_path):
    """The entry's pDMD runs its two steps with heads in order at its
    1024x1024, and says so in the PNG; a request's sigmas, steps or size
    win, and at strength 0 the adapter sets nothing."""
    import io

    from PIL import Image

    served = _h3_served(tmp_path, monkeypatch)
    # The table of a request that names an adapter reads no file here.
    monkeypatch.setattr(svc, "_adapter_table", lambda *a, **k: (None, []))
    seen = []
    monkeypatch.setattr(api, "sample", lambda pipe, cond, **kw: (
        seen.append(kw), mx.zeros((1, 16, 8, 8)))[1])
    runs = [({}, (1024, 1024), ((1.0, 12 / 13, 0.0), "in_order", 2)),
            ({"sigmas": [1, 0.5, 0]}, (1024, 1024),
             ((1.0, 0.5, 0.0), "in_order", 2)),
            ({"sigmas": "1, 0.5, 0", "size": "2048x2048"}, (2048, 2048),
             ((1.0, 0.5, 0.0), "in_order", 2)),
            ({"steps": 6}, (1024, 1024), (None, None, 6)),
            ({"adapters": [{"id": "pdmd.safetensors", "strength": 0}]},
             (2048, 2048), (None, None, 4))]
    for fields, size, want in runs:
        req = svc.parse_request("generations", {"prompt": "x", **fields})
        svc.check_family(req, served)
        svc.request_schedule(req, served, [])
        assert svc.target_size(req, served) == size, fields
        out = svc.run_job(served, req, size=size, budget=None,
                          stop=threading.Event(), embed_metadata=True)
        kw = seen[-1]
        assert (kw["sigmas"], kw["heads"], kw["steps"]) == want, fields
        with Image.open(io.BytesIO(out.images[0])) as im:
            meta = json.loads(im.text["gmlx"])
        assert (meta.get("sigmas"), meta.get("heads")) == (
            None if want[0] is None else list(want[0]), want[1])


def test_an_entrys_schedule_wins_and_a_bad_one_is_refused(
        stubs, monkeypatch, tmp_path):
    """The entry's fields replace the table's row; a request whose steps
    disagree with its sigmas, and heads in order past the bank, get 400."""
    served = _h3_served(tmp_path, monkeypatch,
                        entry={"sigmas": [1.0, 0.75, 0.5, 0.0]})
    req = svc.parse_request("generations", {"prompt": "x"})
    svc.request_schedule(req, served, [])
    assert (req.schedule.sigmas, req.schedule.heads, req.schedule.size) == (
        (1.0, 0.75, 0.5, 0.0), "in_order", None)
    assert "(its entry in the config)" in req.schedule.line
    with pytest.raises(svc.ImageRequestError, match="steps 2 disagrees "
                       "with the 3 steps of sigmas"):
        svc.parse_request("generations", {"prompt": "x", "steps": 2,
                                          "sigmas": [1, 0.6, 0.3, 0]})
    with pytest.raises(svc.ImageRequestError, match="sigmas must not "
                       "increase"):
        svc.parse_request("generations", {"prompt": "x", "sigmas": [0, 1]})
    req = svc.parse_request("generations", {
        "prompt": "x", "sigmas": [1, .9, .8, .7, .6, .5, .4, 0]})
    svc.check_family(req, served)
    with pytest.raises(svc.ImageRequestError, match="bank holds 6 heads, "
                       "not the 7 steps"):
        svc.request_schedule(req, served, [])
