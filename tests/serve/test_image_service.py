#!/usr/bin/env python3
"""Image models in the serve pool: config family names and default sizes.
CPU-only; the loads and the request path run in the image e2e."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from gmlx.serve import image_service


def test_family_names_match_on_letters_and_digits():
    from gmlx.image.registry import get_family

    z, q = get_family("z_image"), get_family("qwen_image21")
    assert image_service.family_matches("z-image", z, "turbo")
    assert image_service.family_matches("z-image-turbo", z, "turbo")
    assert image_service.family_matches("qwen-image-2.1", q, "base")
    assert image_service.family_matches("qwen-image-2.1-turbo", q, "turbo")
    assert not image_service.family_matches("qwen-image-2.1-turbo", q, "base")
    assert not image_service.family_matches("qwen-image-2.1", z, "turbo")
    assert not image_service.family_matches("z-image-base", z, "turbo")


@pytest.mark.parametrize("size,want", [(None, (1024, 1024)),
                                       ("768x512", (768, 512))])
def test_default_size_takes_the_entry_then_the_family(size, want):
    spec = SimpleNamespace(defaults={"size": size} if size else {})
    fam = SimpleNamespace(size=(1024, 1024))
    assert image_service.default_size(spec, fam) == want


def test_request_fields_map_and_aliases_reach_the_default():
    req = image_service.parse_request("generations", {
        "prompt": " a cat ", "model": "gpt-image-1", "n": "3", "size": "768x512",
        "quality": "standard", "output_compression": "20", "stream": "false",
        "adapters": '[{"id": "eyes", "strength": 0.5}, "plain"]'})
    assert (req.prompt, req.model, req.n, req.size) == ("a cat", "", 3, (768, 512))
    assert req.quality == "medium" and req.compression == 20
    assert req.adapters == [("eyes", 0.5), ("plain", 1.0)]
    assert "output_compression" in image_service.parse_request(
        "generations", {"prompt": "x", "output_compression": 5}).ignored
    for name in ("", "default", "dall-e-3", "chatgpt-image-latest"):
        assert image_service.default_model_field(name) == ""
    assert image_service.default_model_field("zit") == "zit"


def test_png_keeps_alpha_and_jpeg_and_webp_take_compression():
    import io

    import numpy as np
    from PIL import Image

    px = np.zeros((16, 24, 4), np.uint8)
    px[..., 3] = 128
    im = Image.open(io.BytesIO(image_service.encode_output(px, "png")))
    assert im.mode == "RGBA" and im.size == (24, 16)
    rgb = np.random.default_rng(0).integers(0, 255, (64, 64, 3), np.uint8)
    # output_compression is the quality, as OpenAI's is: 100 is the
    # default and keeps the most.
    hi = image_service.encode_output(rgb, "jpeg", 100)
    lo = image_service.encode_output(rgb, "jpeg", 10)
    assert len(lo) < len(hi) and Image.open(io.BytesIO(hi)).format == "JPEG"
    assert image_service.encode_output(rgb, "jpeg", None) == hi
    assert Image.open(io.BytesIO(
        image_service.encode_output(px, "webp", 50))).format == "WEBP"


def test_png_metadata_rides_only_when_asked():
    import io

    import numpy as np
    from PIL import Image

    px = np.zeros((8, 8, 3), np.uint8)
    plain = Image.open(io.BytesIO(image_service.encode_output(px, "png")))
    meta = Image.open(io.BytesIO(
        image_service.encode_output(px, "png", metadata={"seed": 3})))
    assert "gmlx" not in plain.info and '"seed": 3' in meta.info["gmlx"]


def test_a_mask_edits_where_its_alpha_is_clear():
    import numpy as np

    px = np.full((4, 4, 4), 255, np.uint8)
    px[:2, :, 3] = 0
    m = image_service.mask_from_image(px, (4, 4))
    assert m[:2].tolist() == [[1.0] * 4] * 2 and m[2:].sum() == 0
    with pytest.raises(image_service.ImageRequestError):
        image_service.mask_from_image(px[..., :3], (4, 4))


def test_decode_refuses_what_is_not_an_image():
    with pytest.raises(image_service.ImageRequestError) as e:
        image_service.decode_image(b"not an image", "image")
    assert e.value.status == 400 and e.value.param == "image"


def test_partials_fall_evenly_inside_the_run():
    assert image_service.partial_steps(8, 3) == [1, 3, 5]
    assert image_service.partial_steps(8, 1) == [3]
    assert image_service.partial_steps(8, 0) == []
    assert image_service.partial_steps(2, 3) == []
    assert all(1 <= i <= 38 for i in image_service.partial_steps(40, 3))


def test_the_file_store_expires_and_caps_oldest_first(tmp_path):
    import os

    now = [1000.0]
    store = image_service.FileStore(str(tmp_path / "f"), ttl_s=60,
                                    max_bytes=10, time_fn=lambda: now[0])
    a = store.put(b"aaaa", "png")
    os.utime(store.path(a, "png"), (999, 999))
    b = store.put(b"bbbb", "png")
    os.utime(store.path(b, "png"), (1000, 1000))
    assert len(a) == image_service.FileStore.TOKEN_CHARS
    assert store.path(a, "png") and store.path(b, "png")
    c = store.put(b"cccc", "png")          # 12 bytes over a cap of 10
    assert store.path(a, "png") is None
    assert store.path(b, "png") and store.path(c, "png")
    now[0] = 1000.0 + 61
    assert store.path(b, "png") is None
    assert store.path(c, "jpeg") is None and store.path("../x", "png") is None
    assert store.sweep() >= 1


def test_quality_steps_come_from_the_family_table():
    from gmlx.image.registry import get_family

    z, q = get_family("z_image"), get_family("qwen_image21")
    assert image_service.quality_steps(z, "turbo", "high") == 8
    assert image_service.quality_steps(z, "turbo", "low") == 4
    assert image_service.quality_steps(q, "base", "medium") == 28
    assert image_service.quality_steps(q, "base", "auto") is None
    # Turbo runs its fixed schedule at every level, and a file that
    # matches no variant takes base's table.
    assert [image_service.quality_steps(q, "turbo", lv)
            for lv in ("low", "medium", "high")] == [8, 8, 8]
    assert image_service.quality_steps(q, "unknown", "low") == 16


GB = 1024**3


def _spec(**kw):
    base = dict(id="zit", path="/dit.gguf", family=None, variant=None,
                defaults={},
                adapters=(), vae="/vae.gguf", encoder="/enc.gguf",
                encoder_model=None, encoder_mmproj=None,
                encoder_release="auto")
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def plan_stubs(monkeypatch):
    """entry_plan over stubbed headers: with the encoder an image peaks at
    10 GB, 8 of them in the encode phase; without it, at 6 GB."""
    from gmlx.image import api, dtypes, loader, memfit
    from gmlx.image.registry import get_family
    from gmlx.serve import capacity

    fam = get_family("z_image")
    state = SimpleNamespace(budget=100 * GB, checked=[], planned=[])
    monkeypatch.setattr(loader, "identify_dit",
                        lambda path, **k: SimpleNamespace(family=fam,
                                                     variant="turbo"))
    monkeypatch.setattr(api, "sampler_defaults",
                        lambda *a: SimpleNamespace(size=(1024, 1024), cfg=1.0))
    monkeypatch.setattr(api, "_header_bytes", lambda path, kind: 1)
    monkeypatch.setattr(dtypes, "image_activation",
                        lambda name: SimpleNamespace(dit="bf16", vae="bf16"))

    def plan(dit, enc, vae, mmproj=None, **kw):
        state.planned.append(enc)
        if enc:
            return SimpleNamespace(peak=10 * GB, phases={"encode": 8 * GB})
        return SimpleNamespace(peak=6 * GB, phases={"encode": 0})

    monkeypatch.setattr(memfit, "plan_memory", plan)
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: state.budget)
    monkeypatch.setattr(capacity, "overcommit", lambda: False)
    monkeypatch.setattr(image_service, "check_parts",
                        lambda spec, fam: state.checked.append(spec.id))
    monkeypatch.setattr(image_service, "_load_figures", lambda spec, act: {
        "peak": 0, "text_peak": 0, "parts": (1, 2), "tower_parts": (3, 2)})
    return state


def test_an_entry_of_unknown_krea_2_weights_needs_its_variant(
        plan_stubs, monkeypatch):
    from gmlx.image import loader
    from gmlx.image.fingerprint import ImageModelError
    from gmlx.image.registry import get_family

    asked = []
    monkeypatch.setattr(loader, "identify_dit", lambda path, variant=None: (
        asked.append(variant) or SimpleNamespace(
            family=get_family("krea2"), variant=variant or "unknown")))
    with pytest.raises(ImageModelError,
                       match="image model 'k2': .*--variant turbo"):
        image_service.entry_plan(_spec(id="k2"))
    assert image_service.entry_plan(_spec(id="k2", variant="turbo"))
    assert asked == [None, "turbo"]


def test_a_released_encoder_books_the_encode_phase(plan_stubs):
    plan = image_service.entry_plan(_spec(encoder_release="always"))
    assert plan.keep_encoder is False and plan.footprint == 8 * GB


@pytest.mark.parametrize("budget_gb,keeps,booked", [(100, True, 10),
                                                    (11, False, 8)])
def test_auto_keeps_the_encoder_while_the_entry_fits_its_share(
        plan_stubs, budget_gb, keeps, booked):
    plan_stubs.budget = budget_gb * GB
    plan = image_service.entry_plan(_spec())
    assert plan.keep_encoder is keeps and plan.footprint == booked * GB


def _modes(monkeypatch, keep=10, release=8, alone=6, staged=5,
           caches=False):
    """plan_memory by how serve runs the entry, in GB: the encoder kept,
    the encoder released after each encode (its encode phase and the
    image without it), and the staged order. With ``caches`` the booked
    caches add to each figure."""
    from gmlx.image import memfit

    seen = []

    def plan(dit, enc, vae, mmproj=None, **kw):
        seen.append(kw)
        extra = (kw["cache_bytes"] + sum(kw["adapters"])) if caches else 0
        if kw.get("load_parts") is not None:
            assert kw["encoder_owned"] and not mmproj   # text, no tower
            return SimpleNamespace(peak=staged * GB + extra, phases={},
                                   staged_load=True)
        if enc:
            return SimpleNamespace(peak=keep * GB + extra,
                                   phases={"encode": release * GB + extra})
        return SimpleNamespace(peak=alone * GB + extra, phases={})

    monkeypatch.setattr(memfit, "plan_memory", plan)
    return seen


@pytest.mark.parametrize("budget_gb,release,staged,booked", [
    (100, "auto", False, 10), (9, "auto", False, 8), (9, "always", False, 8),
    (7, "auto", True, 5), (7, "always", True, 5)])
def test_an_entry_whose_encoder_does_not_fit_beside_the_dit_is_staged(
        plan_stubs, monkeypatch, budget_gb, release, staged, booked):
    _modes(monkeypatch)
    plan_stubs.budget = budget_gb * GB
    plan = image_service.entry_plan(_spec(encoder_release=release))
    assert plan.staged is staged and plan.footprint == booked * GB
    assert plan.cache_share == 1.0
    if staged:
        assert plan.keep_encoder is False and plan.load_parts == (1, 2)
        assert plan.tower_parts == (3, 2)


def test_the_caches_of_a_tight_entry_take_the_room_its_image_leaves(
        plan_stubs, monkeypatch):
    _modes(monkeypatch, caches=True)
    b = image_service._CACHE_BYTES
    full = 2 * b["tensor"] + b["steps"] + 2 * b["adapter"]
    plan_stubs.budget = 5 * GB + full // 2
    plan = image_service.entry_plan(_spec())
    assert plan.staged and 0.4 < plan.cache_share <= 0.5
    assert 5 * GB < plan.footprint <= plan_stubs.budget


@pytest.mark.parametrize("release,budget_gb,hint", [
    ("auto", 4, "use a smaller quant of the DiT or of the text encoder"),
    ("always", 4, "use a smaller quant of the DiT or of the text encoder"),
    ("never", 7, "set encoder_release: auto")])
def test_a_refusal_names_a_change_that_makes_the_entry_fit(
        plan_stubs, monkeypatch, release, budget_gb, hint):
    from gmlx.image import memfit

    _modes(monkeypatch)
    plan_stubs.budget = budget_gb * GB
    with pytest.raises(memfit.ImageMemoryError) as e:
        image_service.entry_plan(_spec(encoder_release=release))
    assert hint in str(e.value)
    assert "encoder_release: always" not in str(e.value)


def test_a_turbo_entry_takes_only_its_own_step_count(plan_stubs, monkeypatch):
    from gmlx.image import loader
    from gmlx.image.fingerprint import ImageModelError
    from gmlx.image.registry import get_family

    q = get_family("qwen_image21")
    monkeypatch.setattr(loader, "identify_dit", lambda path, **k: SimpleNamespace(
        family=q, variant="turbo"))
    plan = image_service.entry_plan(_spec(defaults={"steps": 8}))
    assert plan.variant == "turbo"
    with pytest.raises(ImageModelError, match="defaults.steps: Qwen-Image "
                       "2.1 turbo runs 8 steps"):
        image_service.entry_plan(_spec(defaults={"steps": 20}))
    monkeypatch.setattr(loader, "identify_dit", lambda path, **k: SimpleNamespace(
        family=q, variant="unknown"))
    assert image_service.entry_plan(_spec(defaults={"steps": 20}))


@pytest.mark.parametrize("variant,steps,refused", [
    ("turbo", 8, False), ("turbo", None, False), ("turbo", 20, True),
    ("base", 20, False), ("unknown", 20, False)])
def test_a_request_on_turbo_takes_only_its_own_step_count(variant, steps,
                                                          refused):
    from gmlx.image.registry import get_family

    q = get_family("qwen_image21")
    served = SimpleNamespace(pipe=SimpleNamespace(family=q, variant=variant),
                             spec=SimpleNamespace(id="qi"))
    req = image_service.parse_request("generations", {
        "prompt": "p", **({"steps": steps} if steps else {})})
    if not refused:
        image_service.check_family(req, served)
        return
    with pytest.raises(image_service.ImageRequestError,
                       match="runs 8 steps") as e:
        image_service.check_family(req, served)
    assert e.value.param == "steps"


def test_an_entry_books_both_caches_at_twice_their_bound(plan_stubs,
                                                         monkeypatch):
    from gmlx.image import memfit

    seen = []
    plan = memfit.plan_memory

    def spy(*a, **kw):
        seen.append((kw["adapters"], kw["cache_bytes"]))
        return plan(*a, **kw)

    monkeypatch.setattr(memfit, "plan_memory", spy)
    try:
        for steps_api in (False, True):
            image_service.configure_caches(SimpleNamespace(
                tensor_cache_mb=3, adapter_cache_mb=5, steps_api=steps_api))
            image_service.entry_plan(_spec())
    finally:
        image_service.configure_caches(SimpleNamespace(tensor_cache_mb=64,
                                                       adapter_cache_mb=256))
    # With the step routes on, the entry also books their two caches.
    assert seen[0] == ([2 * (5 << 20)], 2 * (3 << 20))
    assert seen[-1] == ([2 * (5 << 20)], 2 * (3 << 20) + (8 << 20))


def test_a_served_pipeline_caches_hold_at_most_their_bound():
    from gmlx.image.conditioning import ByteLRU as _Lru

    pipe = SimpleNamespace(conditionings=_Lru(1 << 40),
                           reference_latents=_Lru(1 << 40),
                           adapters=_Lru(1 << 40), tables=_Lru(1 << 40))
    image_service.configure_caches(SimpleNamespace(tensor_cache_mb=1,
                                                   adapter_cache_mb=2))
    try:
        image_service.bound_caches(pipe, configured=7)
    finally:
        image_service.configure_caches(SimpleNamespace(tensor_cache_mb=64,
                                                       adapter_cache_mb=256))
    assert pipe.conditionings.max_bytes == 1 << 20
    assert pipe.reference_latents.max_bytes == 1 << 20
    assert pipe.adapters.max_bytes == (2 << 20) + 7
    assert pipe.tables.max_bytes == (2 << 20) + 7


def test_the_default_cache_bounds_are_small():
    from gmlx.config import ImageServerCfg

    cfg = ImageServerCfg()
    assert (cfg.tensor_cache_mb, cfg.adapter_cache_mb) == (64, 256)
    assert image_service.cache_bytes() == (64 << 20, 256 << 20)


def test_the_parts_are_checked_before_the_entry_is_priced(plan_stubs, monkeypatch):
    image_service.entry_plan(_spec())
    assert plan_stubs.checked == ["zit"]


def test_a_vae_or_encoder_of_another_family_is_refused_from_its_header(monkeypatch):
    from gmlx.image import text_encoder
    from gmlx.image.registry import get_family
    from gmlx.image.text_encoder import EncoderMismatchError
    from gmlx.image.vae import VAEError, reader

    fam = get_family("z_image")
    monkeypatch.setattr(reader, "read_header", lambda path: path)

    def identify(header, family):
        if header == "/qwen-vae.gguf":
            raise VAEError("holds the qwen_image21 VAE")

    monkeypatch.setattr(reader, "identify", identify)
    monkeypatch.setattr(text_encoder, "_header_view", lambda path: path)
    monkeypatch.setattr(text_encoder, "encoder_spec", lambda f: "SPEC")

    def check_encoder(view, spec):
        if view == "/wrong-enc.gguf":
            raise EncoderMismatchError("not Qwen3-4B")

    monkeypatch.setattr(text_encoder, "check_encoder_file", check_encoder)
    monkeypatch.setattr(text_encoder, "check_mmproj_file", check_encoder)
    image_service.check_parts(_spec(), fam)
    image_service.check_parts(_spec(encoder=None, encoder_model="qwen",
                                    encoder_mmproj="/wrong-enc.gguf"), fam)
    with pytest.raises(VAEError):
        image_service.check_parts(_spec(vae="/qwen-vae.gguf"), fam)
    with pytest.raises(EncoderMismatchError):
        image_service.check_parts(_spec(encoder="/wrong-enc.gguf"), fam)
    with pytest.raises(EncoderMismatchError):
        image_service.check_parts(_spec(encoder_mmproj="/wrong-enc.gguf"), fam)


class _Encoder:
    def __init__(self):
        self.released = 0

    def release(self):
        self.released += 1
        return True


class _Cache(dict):
    max_bytes = None


def _pipe(encoder=None):
    return SimpleNamespace(encoder=encoder, conditionings=_Cache(),
                           captions={}, modulations={}, adapters=_Cache(),
                           tables=_Cache(), reference_latents=_Cache(),
                           default_table=None)


def test_a_release_drops_the_owned_encoder():
    enc = _Encoder()
    served = image_service.ServedImage(pipe=_pipe(enc), spec=_spec(), plan=None)
    image_service._release([served])
    assert enc.released == 1 and served.pipe is None


@pytest.fixture
def load_stubs(monkeypatch):
    from gmlx.image import api

    calls = []

    def load(path, **kw):
        calls.append(kw)
        return _pipe(_Encoder() if kw.get("text_encoder") else None)

    monkeypatch.setattr(api, "load", load)
    monkeypatch.setattr(image_service, "entry_plan", lambda spec: _plan())
    monkeypatch.setattr(image_service, "_SHARES", {})
    return calls


def _plan(**kw):
    base = dict(footprint=0, keep_encoder=True, family="z_image",
                variant="turbo", size=(1024, 1024), cfg=1.0)
    return image_service.EntryPlan(**(base | kw))


def test_a_loaded_entry_bounds_its_caches(load_stubs):
    image_service.configure_caches(SimpleNamespace(tensor_cache_mb=1,
                                                   adapter_cache_mb=2))
    try:
        served = image_service._load(_spec(encoder=None))
    finally:
        image_service.configure_caches(SimpleNamespace(tensor_cache_mb=64,
                                                       adapter_cache_mb=256))
    pipe = served.pipe
    assert pipe.conditionings.max_bytes == 1 << 20
    assert pipe.reference_latents.max_bytes == 1 << 20
    assert pipe.adapters.max_bytes == pipe.tables.max_bytes == 2 << 20


def test_the_fit_of_a_step_counts_what_the_step_caches_hold(monkeypatch):
    from gmlx.image import memfit
    from gmlx.serve import image_steps

    seen = []
    monkeypatch.setattr(memfit, "plan_memory", lambda **kw: seen.append(
        kw["cache_bytes"]) or SimpleNamespace(fits=kw["height"] < 512))
    monkeypatch.setattr(image_steps.STATE.tensors, "_bytes", 5)
    monkeypatch.setattr(image_steps.STATE.adapters, "_bytes", 7)
    held = SimpleNamespace(bytes=100)
    pipe = SimpleNamespace(encoder=None, dit=SimpleNamespace(act_dtype="bf16"),
                           weights={"dit": 1, "vae": 1}, adapters=held,
                           tables=held, family=SimpleNamespace(name="z_image"),
                           vae=None, conditionings=held, modulations=held)
    served = SimpleNamespace(pipe=pipe, plan=SimpleNamespace(keep_encoder=True))
    assert image_service.size_fits(served, 64, 64, 1 << 30)
    assert not image_service.size_fits(served, 512, 512, 1 << 30)
    assert image_service.size_fits(served, 4096, 4096, None)
    assert seen == [212, 212]


def test_a_staged_request_is_priced_with_its_encoder_loaded_alone():
    held = SimpleNamespace(bytes=0)
    pipe = SimpleNamespace(encoder=None, dit=SimpleNamespace(act_dtype="bf16"),
                           weights={"dit": 1, "vae": 1}, adapters=held,
                           tables=held, family=SimpleNamespace(name="z_image"),
                           vae=None, conditionings=held, modulations=held)
    served = SimpleNamespace(pipe=pipe, plan=_plan(
        staged=True, keep_encoder=False, encoder_bytes=5, mmproj_bytes=7,
        load_parts=(1, 2), tower_parts=(3, 2)))
    text = image_service._fit_args(pipe, served, 1 << 30, 1.0)
    edit = image_service._fit_args(pipe, served, 1 << 30, 1.0, tower=True)
    assert (text["encoder"], text["mmproj"], text["load_parts"]) == (
        5, 0, (1, 2))
    assert (edit["encoder"], edit["mmproj"], edit["load_parts"]) == (
        5, 7, (3, 2))
    assert text["encoder_owned"] and edit["encoder_owned"]


def test_a_shared_encoder_loads_no_mmproj_of_its_own(load_stubs):
    image_service._load(_spec(encoder=None, encoder_model="qwen",
                              encoder_mmproj="/chat-mmproj.gguf"))
    image_service._load(_spec(encoder_mmproj="/own-mmproj.gguf"))
    assert [c["mmproj"] for c in load_stubs] == [None, "/own-mmproj.gguf"]


def test_a_staged_entry_loads_no_encoder_and_shares_its_caches(
        load_stubs, monkeypatch):
    from gmlx.serve import image_steps

    monkeypatch.setattr(image_service, "entry_plan", lambda spec: _plan(
        staged=True, keep_encoder=False, cache_share=0.25))
    scaled = []
    monkeypatch.setattr(image_steps.STATE, "scale_caches", scaled.append)
    served = image_service._load(_spec(encoder_mmproj="/own-mmproj.gguf"))
    assert load_stubs[-1]["text_encoder"] is None
    assert load_stubs[-1]["mmproj"] is None
    tensor, adapter = image_service.cache_bytes()
    assert served.pipe.conditionings.max_bytes == tensor // 4
    assert served.pipe.adapters.max_bytes == adapter // 4
    image_service._release([served])
    assert scaled == [0.25, 1.0]


@pytest.fixture
def staged_stubs(monkeypatch):
    """A staged entry over stubbed loads: ``events`` records each load,
    drop, encode and release, with what is resident at an encode."""
    import mlx.core as mx

    from gmlx.image import api, dtypes, loader, text_encoder, vae
    from gmlx.image.lora import install

    events, cached = [], set()

    class Enc:
        owned = True

        def __init__(self):
            self.released = False

        def release(self):
            events.append("release")
            self.released = True
            return True

    def load_encoder(path, spec, log=None, mmproj=None):
        events.append(("encoder", mmproj))
        return Enc()

    def encode_text(pipe, prompt, images=(), ref_latents=True):
        events.append(("encode", prompt, pipe.dit, pipe.vae))
        if prompt == "boom":
            raise RuntimeError("encode failed")
        cached.add(prompt)
        return SimpleNamespace(features=mx.zeros((1,)))

    monkeypatch.setattr(text_encoder, "load_encoder", load_encoder)
    monkeypatch.setattr(text_encoder, "encoder_spec", lambda fam: "SPEC")
    monkeypatch.setattr(loader, "load_dit", lambda path, **k: (
        events.append("dit") or "DIT2", None))
    monkeypatch.setattr(vae, "load_vae", lambda path, **k: (
        events.append("vae") or "VAE2"))
    monkeypatch.setattr(dtypes, "image_activation",
                        lambda name: SimpleNamespace(dit="bf16", vae="bf16"))
    monkeypatch.setattr(install, "drop_built",
                        lambda pipe: events.append("drop"))
    monkeypatch.setattr(install, "rewrap", lambda pipe: events.append("wrap"))
    monkeypatch.setattr(api, "encode_text", encode_text)
    monkeypatch.setattr(api, "conditioning_cached",
                        lambda pipe, p, images=(): p in cached)
    pipe = SimpleNamespace(dit="DIT", vae="VAE", encoder=None,
                           family=SimpleNamespace(name="z_image"))
    served = image_service.ServedImage(
        pipe=pipe, spec=_spec(encoder_mmproj="/tower.gguf"),
        plan=_plan(staged=True, keep_encoder=False))
    return SimpleNamespace(served=served, pipe=pipe, events=events)


def test_a_staged_encode_holds_the_encoder_without_the_dit(staged_stubs):
    st = staged_stubs
    image_service._encode(st.served, "a", None, ())
    assert st.events == ["drop", ("encoder", None), ("encode", "a", None, None),
                         "release", "dit", "wrap", "vae"]
    assert (st.pipe.dit, st.pipe.vae) == ("DIT2", "VAE2")
    # A prompt in the cache runs on the released encoder's tokenizer.
    del st.events[:]
    image_service._encode(st.served, "a", None, ())
    assert st.events == [("encode", "a", "DIT2", "VAE2")]
    # Reference images load the encoder with its vision tower.
    del st.events[:]
    image_service._encode(st.served, "b", "a", ("REF",))
    assert st.events[:2] == ["drop", ("encoder", "/tower.gguf")]


def test_a_failed_staged_encode_leaves_the_dit_loaded(staged_stubs):
    st = staged_stubs
    with pytest.raises(RuntimeError, match="encode failed"):
        image_service._encode(st.served, "boom", None, ())
    assert st.events[-4:] == ["release", "dit", "wrap", "vae"]
    assert st.pipe.encoder.released and st.pipe.dit == "DIT2"


def test_a_failed_adapter_frees_the_loaded_pipeline(load_stubs, monkeypatch):
    from gmlx.image import lora

    pipes = []
    real_release = image_service._release
    monkeypatch.setattr(image_service, "_release",
                        lambda box: pipes.append(box[0].pipe.encoder)
                        or real_release(box))

    def bad_adapter(pipe, path):
        raise ValueError("not a LoRA")

    monkeypatch.setattr(lora, "load_adapter", bad_adapter)
    with pytest.raises(ValueError, match="not a LoRA"):
        image_service._load(_spec(adapters=(("/a.safetensors", 1.0),)))
    assert len(pipes) == 1 and pipes[0].released == 1


def test_a_failed_vae_load_releases_the_encoder_it_loaded(monkeypatch):
    from gmlx.image import api, dtypes, loader, text_encoder, vae
    from gmlx.image.registry import get_family

    fam = get_family("z_image")
    ident = SimpleNamespace(family=fam, variant="turbo")
    enc = _Encoder()
    monkeypatch.setattr(api, "_header_bytes", lambda path, kind: 0)
    monkeypatch.setattr(loader, "identify_dit", lambda path, variant=None: ident)
    monkeypatch.setattr(loader, "load_dit", lambda *a, **k: (
        "DIT", SimpleNamespace(identified=ident)))
    monkeypatch.setattr(dtypes, "image_activation",
                        lambda name: SimpleNamespace(dit="bf16", vae="bf16"))
    monkeypatch.setattr(text_encoder, "encoder_spec", lambda f: "SPEC")
    monkeypatch.setattr(text_encoder, "load_encoder", lambda *a, **k: enc)

    def bad_vae(*a, **k):
        raise vae.VAEError("holds the qwen_image21 VAE")

    monkeypatch.setattr(vae, "load_vae", bad_vae)
    with pytest.raises(vae.VAEError):
        api.load("/dit.gguf", text_encoder="/enc.gguf", vae="/vae.gguf",
                 memory="off", log=lambda *a: None)
    assert enc.released == 1


# Request limits, the fit search and the family checks.

@pytest.mark.parametrize("size", ["4194304x16", "2049x2048", "4112x64",
                                  "16777216x16"])
def test_a_size_over_the_pixel_or_side_limit_is_refused_by_name(size):
    with pytest.raises(image_service.ImageRequestError,
                       match="limit of 4194304 pixels") as e:
        image_service.parse_size(size)
    assert e.value.param == "size"


@pytest.mark.parametrize("size", ["2048x2048", "4096x1024", "16x16"])
def test_a_size_inside_the_limits_is_taken(size):
    w, h = (int(v) for v in size.split("x"))
    assert image_service.parse_size(size) == (w, h)


def _fit_served(monkeypatch, fits_px: int):
    """A served entry whose plan fits sizes of at most ``fits_px`` pixels,
    counting the plans it makes."""
    from gmlx.image import memfit

    calls = []

    def plan_memory(*, height, width, _hint=True, **k):
        calls.append((width, height))
        ok = width * height <= fits_px
        if not ok and _hint:
            raise memfit.ImageMemoryError("no room")
        return SimpleNamespace(fits=ok)

    monkeypatch.setattr(memfit, "plan_memory", plan_memory)
    nb = SimpleNamespace(bytes=0)
    pipe = SimpleNamespace(
        encoder=None, dit=SimpleNamespace(act_dtype="bfloat16"),
        weights={"dit": 1, "vae": 1}, adapters=nb, tables=nb,
        family=SimpleNamespace(name="z_image", edit=None, title="Z-Image"),
        vae=None, conditionings=nb, modulations=nb, size_multiple=16)
    served = SimpleNamespace(pipe=pipe, plan=SimpleNamespace(
        keep_encoder=True, cfg=1.0), spec=SimpleNamespace(id="zit"))
    return served, calls


def test_the_fit_check_names_the_largest_fitting_size_in_few_plans(
        monkeypatch):
    served, calls = _fit_served(monkeypatch, 1024 * 512)
    req = image_service.parse_request("generations", {
        "prompt": "x", "size": "4096x1024"})
    with pytest.raises(image_service.ImageRequestError,
                       match="largest size of its shape that fits is "
                             "1456x352") as e:
        image_service.fit_check(req, served, req.size, 1)
    assert e.value.param == "size"
    # A bisection over the longer side, not a plan per stride.
    assert len(calls) <= 12, len(calls)
    calls.clear()
    image_service.fit_check(req, served, (512, 512), 1)
    assert len(calls) == 1


def test_the_fit_check_passes_without_a_budget(monkeypatch):
    served, calls = _fit_served(monkeypatch, 0)
    req = image_service.parse_request("generations", {"prompt": "x"})
    image_service.fit_check(req, served, (1024, 1024), None)
    assert calls == []


def _edit_served(family: str, mmproj=True):
    from gmlx.image.registry import get_family

    fam = get_family(family)
    return SimpleNamespace(
        pipe=SimpleNamespace(family=fam, variant="base"),
        spec=SimpleNamespace(id="m", encoder_model=None,
                             encoder_mmproj="/mm.gguf" if mmproj else None))


def _edit_req(n: int):
    import numpy as np

    req = image_service.parse_request("edits", {"prompt": "x"})
    req.images = [np.zeros((32, 32, 3), np.uint8)] * n
    return req


def test_an_edit_takes_at_most_the_family_reference_count():
    served = _edit_served("qwen_image21")
    image_service.check_family(_edit_req(10), served)
    with pytest.raises(image_service.ImageRequestError,
                       match="takes at most 10 images, and the request has "
                             "11") as e:
        image_service.check_family(_edit_req(11), served)
    assert e.value.param == "image"


def test_z_image_edits_one_image_at_a_time():
    served = _edit_served("z_image")
    image_service.check_family(_edit_req(1), served)
    with pytest.raises(image_service.ImageRequestError,
                       match="Z-Image edits one image at a time"):
        image_service.check_family(_edit_req(2), served)


def test_the_server_refuses_the_references_of_flux3_klein_by_name():
    """FLUX.3 klein prepares the references of a request together, which
    the server's path of one image at a time does not; its edits run in
    the CLI."""
    from gmlx.image.registry import get_family

    assert image_service.serves_edits(get_family("qwen_image21"))
    assert not image_service.serves_edits(get_family("flux3_klein"))
    assert not image_service.serves_edits(get_family("z_image"))
    served = _edit_served("flux3_klein")
    with pytest.raises(image_service.ImageRequestError,
                       match="the server takes no FLUX.3 klein reference "
                             "images; gmlx image generate --ref runs "
                             "them") as e:
        image_service.check_family(_edit_req(1), served)
    assert e.value.param == "image"
    gen = image_service.parse_request("generations", {"prompt": "x"})
    image_service.check_family(gen, served)


@pytest.mark.parametrize("aid", ["/abs/a.safetensors", "~/a.safetensors",
                                 "../a.safetensors", "x/../../a.safetensors"])
def test_an_adapter_id_is_never_an_absolute_or_escaping_path(tmp_path, aid):
    root = tmp_path / "models"
    root.mkdir()
    with pytest.raises(image_service.ImageRequestError, match="never an "
                       "absolute path"):
        image_service._adapter_under_roots(aid, [str(root)])


def test_an_adapter_id_resolves_only_inside_a_model_folder(tmp_path):
    root = tmp_path / "models"
    (root / "lora").mkdir(parents=True)
    (root / "lora" / "a.safetensors").write_bytes(b"x")
    outside = tmp_path / "b.safetensors"
    outside.write_bytes(b"x")
    (root / "lora" / "b.safetensors").symlink_to(outside)
    got = image_service._adapter_under_roots("lora/a.safetensors", [str(root)])
    assert got == str((root / "lora" / "a.safetensors").resolve())
    with pytest.raises(image_service.ImageRequestError) as e:
        image_service._adapter_under_roots("lora/b.safetensors", [str(root)])
    assert e.value.status == 404


@pytest.mark.parametrize("aid", ["style.safetensors", "style",
                                 "lora/style.safetensors"])
def test_a_request_naming_an_entry_adapter_sets_its_strength_once(
        tmp_path, monkeypatch, aid):
    from gmlx.image import api, lora

    root = tmp_path / "models"
    (root / "lora").mkdir(parents=True)
    path = root / "lora" / "style.safetensors"
    path.write_bytes(b"x")
    entry = SimpleNamespace(path=str(path), text=("t",))
    other = SimpleNamespace(path=str(root / "lora" / "b.safetensors"), text=())
    monkeypatch.setattr(lora, "load_adapter", lambda pipe, p: (
        entry if os.path.realpath(p) == os.path.realpath(path) else other))
    monkeypatch.setattr(lora, "table_entries", lambda rows: list(rows))
    monkeypatch.setattr(api, "define_patch_table", lambda pipe, rows: rows)
    (root / "lora" / "b.safetensors").write_bytes(b"x")
    served = SimpleNamespace(pipe=object(), adapters=[(entry, 1.0)])
    table, shown = image_service._adapter_table(
        served, [(aid, 0.4), ("lora/b.safetensors", 0.7)], [str(root)])
    assert table == [(entry, 0.4), (other, 0.7)]
    assert shown == [{"id": aid, "strength": 0.4},
                     {"id": "lora/b.safetensors", "strength": 0.7}]
    assert served.adapters == [(entry, 1.0)]


@pytest.mark.parametrize("scheduler", ["karras", "simple", "beta"])
def test_a_scheduler_other_than_the_variant_own_is_refused(scheduler):
    with pytest.raises(image_service.ImageRequestError,
                       match="scheduler takes only default") as e:
        image_service.parse_request("generations", {
            "prompt": "x", "scheduler": scheduler})
    assert e.value.param == "scheduler"
    assert image_service.parse_request("generations", {
        "prompt": "x", "scheduler": "default"})


def test_a_size_off_the_family_stride_is_refused():
    served = SimpleNamespace(pipe=SimpleNamespace(size_multiple=16),
                             plan=SimpleNamespace(size=(1024, 1024)))
    req = image_service.parse_request("generations", {
        "prompt": "x", "size": "1000x1024"})
    with pytest.raises(image_service.ImageRequestError,
                       match="multiples of 16") as e:
        image_service.target_size(req, served)
    assert e.value.param == "size"


def test_a_large_edit_image_gives_the_largest_size_of_its_shape():
    import numpy as np

    served = SimpleNamespace(pipe=SimpleNamespace(size_multiple=16),
                             plan=SimpleNamespace(size=(1024, 1024)))
    req = image_service.parse_request("edits", {"prompt": "x"})
    req.images = [np.zeros((3000, 6000, 3), np.uint8)]
    w, h = image_service.target_size(req, served)
    assert w * h <= image_service.MAX_PIXELS and max(w, h) <= 4096
    assert (w % 16, h % 16) == (0, 0) and abs(w / h - 2) < 0.02


def test_a_url_file_over_the_byte_cap_keeps_its_link(tmp_path):
    store = image_service.FileStore(str(tmp_path / "f"), ttl_s=60,
                                    max_bytes=10)
    old = store.put(b"y" * 8, "png")
    tok = store.put(b"x" * 64, "png")
    assert store.path(tok, "png") is not None
    # The older file goes to make room, not the new one.
    assert store.path(old, "png") is None


def test_each_server_port_keeps_its_own_url_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    a = image_service.default_store_root(8080)
    b = image_service.default_store_root(8081)
    assert a != b and os.path.dirname(a) == os.path.dirname(b)
    assert a == str(tmp_path / "gmlx" / "images" / "8080")
