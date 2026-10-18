#!/usr/bin/env python3
"""Prompt upsampling on the image routes, over stub models: the chat turn
the server sends its upsampler, and a rerun of a prompt with a new seed,
which writes no new caption and loads no encoder. The negative prompt
runs as typed. The chat engine, the encoder and the sampler are stubs."""
from __future__ import annotations

import io
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from gmlx.image import api  # noqa: E402
from gmlx.image.registry import get_family  # noqa: E402
from gmlx.image.upsample import system_prompt  # noqa: E402
from gmlx.serve import image_service as svc  # noqa: E402
from gmlx.serve import image_upsample  # noqa: E402

CAPTION = "A red bicycle leans against a weathered brick wall."


@pytest.fixture
def server(monkeypatch):
    """A staged FLUX.3 klein Base entry with an upsampler. ``events``
    records the chat turns, the encoder loads and each prompt encoded;
    ``cached`` is the conditioning cache."""
    import importlib

    from gmlx.image import dtypes, loader, text_encoder, vae
    from gmlx.image.lora import install
    from gmlx.serve import bridge_vlm, image_steps

    fam = get_family("flux3_klein")
    rec = SimpleNamespace(events=[], requests=[], messages=[], seeds=[],
                          too_long=False)
    cached: set = set()
    spec = SimpleNamespace(
        id="flux3", path="/flux3.gguf", family=None, variant=None,
        defaults={}, adapters=(), vae="/vae.gguf", encoder="/enc.gguf",
        encoder_model=None, encoder_mmproj=None, encoder_release="auto",
        upsampler="writer", upsample=True, upsample_prompt=None)
    monkeypatch.setattr(bridge_vlm, "resolve_image_request",
                        lambda field: (spec.path, spec))
    writer = SimpleNamespace(load_signature=lambda: ("/writer.gguf",),
                             adapter=None, chat_template_kwargs={},
                             thinking=None)
    monkeypatch.setattr(bridge_vlm, "resolve_request_model",
                        lambda field, **k: ("/writer.gguf", writer))
    monkeypatch.setattr(image_steps, "family_of", lambda s: (fam, "base"))
    monkeypatch.setattr(image_upsample, "CAPTIONS",
                        image_upsample._Captions(4))

    # The chat engine of the upsampler.
    app = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")
    openai = importlib.import_module("gmlx._vendor.mlx_vlm.server.openai")
    runtime = importlib.import_module("gmlx._vendor.mlx_vlm.server.runtime")

    def build_gen_args(request, processor, tenant_id=None):
        rec.requests.append(request)
        return SimpleNamespace(logprobs=True,
                               to_template_kwargs=lambda: {})

    def apply_chat_template(processor, config, messages, **kw):
        rec.messages.append(messages)
        return "PROMPT"

    class Engine:
        def generate(self, prompt, args):
            rec.events.append("chat")
            if rec.too_long:
                gen = importlib.import_module(
                    "gmlx._vendor.mlx_vlm.server.generation")
                raise gen.PromptTooLongError("Request needs 13536 context "
                                             "tokens, but MAX_KV_SIZE is 8192.")
            toks = [SimpleNamespace(text=t, finish_reason=None)
                    for t in ("  ", CAPTION, "\n")]
            toks[-1].finish_reason = "stop"
            return SimpleNamespace(prompt_tokens=9), iter(toks)

    monkeypatch.setattr(app, "get_cached_model",
                        lambda model: (None, "PROC", "CFG"))
    monkeypatch.setattr(app, "_build_gen_args", build_gen_args)
    monkeypatch.setattr(openai, "apply_chat_template", apply_chat_template)
    monkeypatch.setattr(runtime, "runtime",
                        SimpleNamespace(response_generator=Engine()))

    # The staged image entry.
    class Enc:
        owned = True

        def __init__(self):
            self.released = False

        def release(self):
            self.released = True
            return True

    def load_encoder(path, spec, log=None, mmproj=None):
        rec.events.append("encoder")
        return Enc()

    def encode_text(pipe, prompt, images=(), ref_latents=True):
        rec.events.append(("encode", prompt))
        cached.add(prompt)
        return SimpleNamespace(features=mx.zeros((5, 4)), image_slots=(),
                               refs=(), ref_latents=())

    dit = SimpleNamespace(cfg=SimpleNamespace(patch=1))
    monkeypatch.setattr(text_encoder, "load_encoder", load_encoder)
    monkeypatch.setattr(text_encoder, "encoder_spec", lambda fam: "SPEC")
    monkeypatch.setattr(loader, "load_dit", lambda path, **k: (dit, None))
    monkeypatch.setattr(vae, "load_vae", lambda path, **k: "VAE")
    monkeypatch.setattr(dtypes, "image_activation",
                        lambda name: SimpleNamespace(dit="bf16", vae="bf16"))
    monkeypatch.setattr(install, "drop_built", lambda pipe: None)
    monkeypatch.setattr(install, "rewrap", lambda pipe: None)
    monkeypatch.setattr(api, "encode_text", encode_text)
    monkeypatch.setattr(api, "conditioning_cached",
                        lambda pipe, p, images=(): p in cached)
    monkeypatch.setattr(api, "plan_request",
                        lambda *a, **k: SimpleNamespace(release_encoder=False))
    monkeypatch.setattr(api, "image_sigmas",
                        lambda pipe, steps, h, w: np.array([1.0, 0.5, 0.0]))

    def sample(pipe, cond, *, seed, **kw):
        rec.seeds.append(seed)
        return mx.zeros((1, 128, 4, 4))

    monkeypatch.setattr(api, "sample", sample)
    monkeypatch.setattr(api, "vae_decode", lambda pipe, x, *, alpha, plan:
                        mx.zeros((1, 64, 64, 3)))
    pipe = SimpleNamespace(
        family=fam, variant="base", dit=dit, vae="VAE", encoder=None,
        defaults=SimpleNamespace(steps=2, strength=0.6), vae_scale=16)
    served = svc.ServedImage(
        pipe=pipe, spec=spec,
        plan=SimpleNamespace(staged=True, keep_encoder=False, cfg=4.0))
    rec.served = served
    rec.spec = spec
    return rec


def _request(server, fields):
    """One request as the images route runs it: the rewrite, then the
    job."""
    req = svc.parse_request("generations", fields)
    req.revised_prompt = image_upsample.revise(
        req.model, req.prompt, req.prompt_upsampling,
        req.route != "generations")
    return req, svc.run_job(server.served, req, size=(64, 64), budget=None,
                            stop=threading.Event(), embed_metadata=True)


def test_the_server_sends_bfls_turn_and_encodes_the_caption(server):
    req, out = _request(server, {"prompt": "a red bicycle", "seed": 1})
    assert req.revised_prompt == CAPTION
    (sent,) = server.requests
    assert sent.model == "writer" and sent.stream is False
    assert (sent.temperature, sent.max_tokens, sent.presence_penalty,
            sent.thinking) == (0.0, 3584, 1.5, "off")
    fam = get_family("flux3_klein")
    assert server.messages == [[
        {"role": "system", "content": system_prompt(fam)},
        {"role": "user", "content": "a red bicycle"}]]
    from PIL import Image

    with Image.open(io.BytesIO(out.images[0])) as im:
        meta = json.loads(im.text["gmlx"])
    assert meta["prompt"] == "a red bicycle"
    assert meta["revised_prompt"] == CAPTION


def test_a_rerun_with_a_new_seed_writes_no_caption_and_loads_no_encoder(
        server):
    fields = {"prompt": "a red bicycle", "guidance": "4",
              "negative_prompt": "blurry"}
    _request(server, {**fields, "seed": 1})
    assert server.events == ["chat", "encoder", ("encode", CAPTION),
                             ("encode", "blurry")]
    del server.events[:]
    req, _ = _request(server, {**fields, "seed": 2})
    assert req.revised_prompt == CAPTION
    assert server.events == [("encode", CAPTION), ("encode", "blurry")]
    assert server.seeds == [1, 2]


def test_the_entrys_prompt_file_replaces_the_shipped_one_and_its_captions(
        server, tmp_path):
    """Captions are keyed by the file's content: an edit to it writes a
    new caption, and the old prompt's caption is not reused."""
    own = tmp_path / "rewrite.txt"
    own.write_text("Describe the scene in one long sentence.\n")
    server.spec.upsample_prompt = str(own)
    _request(server, {"prompt": "a red bicycle", "seed": 1})
    _request(server, {"prompt": "a red bicycle", "seed": 2})
    own.write_text("Describe the scene in two long sentences.\n")
    _request(server, {"prompt": "a red bicycle", "seed": 3})
    assert server.events.count("chat") == 2
    assert [m[0]["content"] for m in server.messages] == [
        "Describe the scene in one long sentence.",
        "Describe the scene in two long sentences."]
    server.spec.upsample_prompt = None
    _request(server, {"prompt": "a red bicycle", "seed": 4})
    assert server.messages[-1][0]["content"] == system_prompt(
        get_family("flux3_klein"))


def test_an_upsampler_with_too_small_a_context_is_a_server_error(server):
    """The typed prompt is not the cause, so the answer is no 400 on it."""
    server.too_long = True
    with pytest.raises(svc.ImageRequestError) as e:
        _request(server, {"prompt": "a red bicycle", "seed": 1})
    assert (e.value.status, e.value.err_type, e.value.param) == (
        500, "server_error", None)
    assert "the upsampler 'writer' has no room in its context" in str(e.value)


def test_an_edit_ignores_prompt_upsampling_on_every_family():
    import dataclasses

    spec = SimpleNamespace(id="qwen", upsampler=None, upsample=True)
    fam = dataclasses.replace(get_family("qwen_image21"), upsample=None)
    assert image_upsample.upsampler_for(spec, fam, True, True) is None
    with pytest.raises(svc.ImageRequestError, match="takes no prompt"):
        image_upsample.upsampler_for(spec, fam, True, False)
