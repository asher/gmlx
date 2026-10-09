#!/usr/bin/env python3
"""Image models in the serve pool: config family names and default sizes.
CPU-only; the loads and the request path run in the image e2e."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from gmlx.serve import image_service


def test_family_names_match_on_letters_and_digits():
    from gmlx.image.registry import get_family

    z, q = get_family("z_image"), get_family("qwen_image21")
    assert image_service.family_matches("z-image", z, "turbo")
    assert image_service.family_matches("z-image-turbo", z, "turbo")
    assert image_service.family_matches("qwen-image-2.1", q, "default")
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
    hi = image_service.encode_output(rgb, "jpeg", 0)
    lo = image_service.encode_output(rgb, "jpeg", 90)
    assert len(lo) < len(hi) and Image.open(io.BytesIO(hi)).format == "JPEG"
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
    assert image_service.quality_steps(q, "default", "medium") == 28
    assert image_service.quality_steps(q, "default", "auto") is None


GB = 1024**3


def _spec(**kw):
    base = dict(id="zit", path="/dit.gguf", family=None, defaults={},
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
                        lambda path: SimpleNamespace(family=fam,
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
    return state


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


def _pipe(encoder=None):
    return SimpleNamespace(encoder=encoder, conditionings={}, captions={},
                           modulations={}, adapters={}, tables={},
                           reference_latents={}, default_table=None)


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
    monkeypatch.setattr(image_service, "entry_plan", lambda spec: "PLAN")
    return calls


def test_a_shared_encoder_loads_no_mmproj_of_its_own(load_stubs):
    image_service._load(_spec(encoder=None, encoder_model="qwen",
                              encoder_mmproj="/chat-mmproj.gguf"))
    image_service._load(_spec(encoder_mmproj="/own-mmproj.gguf"))
    assert [c["mmproj"] for c in load_stubs] == [None, "/own-mmproj.gguf"]


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
