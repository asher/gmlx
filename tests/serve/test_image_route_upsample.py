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
                          too_long=False, media=[])
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
        def generate(self, prompt, args, **media):
            rec.events.append("chat")
            rec.media.append(media)
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

    dit = SimpleNamespace(cfg=SimpleNamespace(patch=1), act_dtype="bf16")
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
        defaults=SimpleNamespace(steps=2, strength=0.6), vae_scale=16,
        size_multiple=16)
    served = svc.ServedImage(
        pipe=pipe, spec=spec,
        plan=SimpleNamespace(staged=True, keep_encoder=False, cfg=4.0),
        facts=svc.dit_facts(pipe))
    rec.served = served
    rec.spec = spec
    rec.writer = writer
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


def test_upsampler_request_blocks_profile_budget_message(server):
    """Both budget fields are set to null, so no profile of the chat model
    caps or messages the caption's thinking."""
    _request(server, {"prompt": "a red bicycle", "seed": 1})
    (sent,) = server.requests
    for field in ("thinking_budget", "thinking_budget_message"):
        assert field in sent.model_fields_set
        assert getattr(sent, field) is None


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


H3_CAPTION = "A blue door in a whitewashed wall."


@pytest.fixture
def h3(server, monkeypatch, tmp_path):
    """The server's entry as MiniMax-H3-Image, with MiniMax's prompt files
    in stand-in text and an upsampler entry with its vision tower."""
    from gmlx.serve import image_steps

    root = tmp_path / "pe" / "prompt"
    for rel, text in {"t2i/system_prompt.txt": "Expand prompts.\n",
                      "t2i/task_prompt.txt": "Prompt: {}",
                      "edit/task_prompt.txt": "Edit task."}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    (root / "manifest.json").write_text(json.dumps({
        "t2i": {"files": {"prompt/t2i/system_prompt.txt": {},
                          "prompt/t2i/task_prompt.txt": {}}},
        "edit": {"files": {"prompt/edit/task_prompt.txt": {}}}}))
    fam = get_family("h3_image")
    monkeypatch.setattr(image_steps, "family_of", lambda s: (fam, "base"))
    server.spec.upsample_prompt = str(root / "manifest.json")
    server.writer.mmproj = "/writer-mmproj.gguf"
    prompt_utils = __import__("gmlx._vendor.mlx_vlm.prompt_utils",
                              fromlist=["get_chat_template"])

    def get_chat_template(processor, messages, add_generation_prompt, **kw):
        server.messages.append(messages)
        return "PICTURES PROMPT"

    monkeypatch.setattr(prompt_utils, "get_chat_template", get_chat_template)

    class Engine:
        def generate(self, prompt, args, **media):
            server.events.append("chat")
            server.media.append((prompt, media))
            tok = SimpleNamespace(text='</think>{"expanded_caption": "'
                                  + H3_CAPTION + '"}', finish_reason="stop")
            return SimpleNamespace(prompt_tokens=9), iter([tok])

    runtime = __import__("gmlx._vendor.mlx_vlm.server.runtime",
                         fromlist=["runtime"])
    monkeypatch.setattr(runtime, "runtime",
                        SimpleNamespace(response_generator=Engine()))
    return server


def _pictures():
    return [np.full((32, 64, 3), 7, np.uint8),
            np.full((48, 48, 4), 9, np.uint8)]


def test_minimax_h3_image_edit_sends_its_pictures_to_the_upsamplers_tower(h3):
    """The edit's rewrite renders MiniMax's edit turn with each picture in
    place, sends the pictures in order to the chat engine, samples at
    MiniMax's settings, and is seeded by the image's seed."""
    caption = image_upsample.revise("h3", "paint the door blue", True, True,
                                    pictures=_pictures(), seed=5,
                                    canvas=(1024, 768))
    assert caption == H3_CAPTION
    (sent,) = h3.requests
    assert (sent.temperature, sent.top_p, sent.top_k, sent.min_p,
            sent.max_tokens, sent.thinking, sent.seed) == (
        1.0, 0.95, 20, 0.0, 4096, "on", 5)
    ((msg,),) = h3.messages
    assert msg["role"] == "user" and msg["content"][0]["text"] == "Edit task."
    assert [c["text"] for c in msg["content"] if "Media" in c.get("text", "")
            ] == ["Reference Media:\n- <Picture 1> [aspect_ratio 2:1]",
                  "Reference Media:\n- <Picture 2> [aspect_ratio 1:1]"]
    assert msg["content"][-1]["text"].count("Selection:\n4:3\n") == 1
    ((prompt, media),) = h3.media
    assert prompt == "PICTURES PROMPT"
    assert [im.size for im in media["images"]] == [(64, 32), (48, 48)]
    assert [im.mode for im in media["images"]] == ["RGB", "RGB"]


def test_minimax_h3_image_captions_rerun_only_for_their_seed_and_pictures(h3):
    """A sampled caption is kept by its seed and pictures: the same seed
    and pictures reuse it, another seed or picture writes a new one, and
    a rewrite without a seed is drawn afresh and not kept."""
    def run(seed, pictures=None):
        image_upsample.revise("h3", "paint the door blue", True, True,
                              pictures=pictures or _pictures(), seed=seed)

    run(5)
    run(5)
    assert h3.events.count("chat") == 1
    run(6)
    other = _pictures()
    other[0][0, 0] = 8
    run(6, other)
    assert h3.events.count("chat") == 3
    run(None)
    run(None)
    assert h3.events.count("chat") == 5
    assert h3.requests[-1].seed is not None


def test_minimax_h3_image_without_its_upsamplers_tower_is_a_server_error(h3):
    h3.writer.mmproj = None
    with pytest.raises(svc.ImageRequestError) as e:
        image_upsample.revise("h3", "paint the door blue", True, True,
                              pictures=_pictures(), seed=5)
    assert e.value.status == 500
    assert ("the MiniMax-H3-Image upsampler sees the reference images through "
            "its vision tower, and the models: entry 'writer' has no mmproj"
            ) in str(e.value)
    assert "chat" not in h3.events
    # A prompt without pictures runs the text turn without the tower.
    assert image_upsample.revise("h3", "a red bicycle", True, False,
                                 seed=5) == H3_CAPTION
    assert h3.messages[-1] == [
        {"role": "system", "content": "Expand prompts."},
        {"role": "user", "content": "Prompt: a red bicycle"}]
    assert h3.media[-1][1] == {}


def test_an_edit_on_a_family_whose_upsampler_sees_no_pictures_runs_as_typed(
        server):
    """FLUX.3 klein's rewrite of an edit is skipped, pictures given or
    not, and its chat turn is unchanged."""
    assert image_upsample.revise("flux3", "swap them", True, True,
                                 pictures=_pictures(), seed=5) is None
    assert server.events == [] and server.requests == []
    image_upsample.revise("flux3", "a red bicycle", True, False, seed=5)
    (sent,) = server.requests
    assert (sent.top_p, sent.top_k, sent.min_p) == (1.0, 0, 0.0)
    assert "seed" not in sent.model_fields_set
    assert server.media == [{}]
