"""Prompt upsampling of FLUX.3 klein: the chat turn is BFL's for a prompt
without references, from the system prompt gmlx ships, and ``gmlx image
generate`` releases the upsampler before the encoder loads and encodes
the caption in place of the typed prompt, but never the negative. The
chat model and the image pipeline are stubs."""
from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import weakref
from importlib import resources
from types import SimpleNamespace

import pytest

from gmlx.image import upsample
from gmlx.image.registry import get_family

PROMPTS_SHA256 = (
    "361bb6e081d50e54a3dee9abf26327436a9386c83465f881c7848989232028be")
CAPTION = "A red bicycle leans against a weathered brick wall."


def test_the_chat_turn_is_bfls_t2i_turn():
    fam = get_family("flux3_klein")
    data = (resources.files("gmlx.image.families.flux3_klein")
            / "pup_prompts_external_vlm.json").read_bytes()
    assert hashlib.sha256(data).hexdigest() == PROMPTS_SHA256
    req = upsample.chat_request(fam, "a red bicycle")
    assert req.messages == (
        {"role": "system", "content": json.loads(data)["t2i"]},
        {"role": "user", "content": "a red bicycle"})
    assert (req.temperature, req.max_tokens, req.presence_penalty,
            req.thinking) == (0.0, 3584, 0.0, "off")


def test_an_empty_reply_is_an_error_not_the_typed_prompt():
    fam = get_family("flux3_klein")
    assert upsample.upsample(fam, "x", lambda req: f"  {CAPTION}\n") == CAPTION
    with pytest.raises(upsample.UpsampleError, match="empty caption"):
        upsample.upsample(fam, "x", lambda req: " \n")


@pytest.fixture
def cli(tmp_path, monkeypatch):
    """``gmlx image generate`` on FLUX.3 klein Base over stub loads.
    ``events`` records the upsampler's load and turn, the image load
    with whether the upsampler was alive then, and each prompt encoded."""
    mx = pytest.importorskip("mlx.core")

    import gmlx.gen.generation as generation
    import gmlx.load.loader as loader
    import gmlx.load.loadlog as loadlog
    import gmlx.serve.memory as memory
    from gmlx.commands import image_preflight, tool_preflight
    from gmlx.image import api

    files = {}
    for name in ("dit.gguf", "enc.gguf", "vae.gguf", "writer.gguf"):
        p = tmp_path / name
        p.write_bytes(b"x")
        files[name] = str(p)
    fam = get_family("flux3_klein")
    rec = SimpleNamespace(events=[], turns=[], model=None)

    class Model:
        pass

    def load_model(path):
        model = Model()
        rec.model = weakref.ref(model)
        rec.events.append("upsampler")
        tok = SimpleNamespace(
            chat_template="{% if enable_thinking %}{% endif %}")
        return model, None, tok

    def generate(model, tok, prompt, **kw):
        rec.events.append("chat")
        rec.turns.append({"prompt": prompt, **kw})
        return f" {CAPTION}\n"

    class Enc:
        owned, released = True, False

        def release(self):
            self.released = True
            return True

    def load(dit, **kw):
        gc.collect()
        rec.events.append(("image", rec.model is not None
                           and rec.model() is not None))
        return SimpleNamespace(encoder=Enc(), variant="base",
                               load_seconds={}, family=fam)

    def encode_text(pipe, prompt, images=(), ref_latents=True):
        rec.events.append(("encode", prompt))
        return SimpleNamespace(features=mx.zeros((5, 4)), image_slots=())

    monkeypatch.setattr(loader, "load_model", load_model)
    monkeypatch.setattr(generation, "generate", generate)
    monkeypatch.setattr(loadlog, "load_ui",
                        lambda verbose, path: contextlib.nullcontext())
    monkeypatch.setattr(memory, "resolve_cache_limit",
                        lambda *a: (None, "test"))
    monkeypatch.setattr(tool_preflight, "check_or_exit", lambda *a, **k: None)
    monkeypatch.setattr(image_preflight, "identify_dit",
                        lambda p, **k: SimpleNamespace(
                            family=fam, variant="base", config={}))
    monkeypatch.setattr(image_preflight, "check", lambda *a, **k: None)
    monkeypatch.setattr(api, "load", load)
    monkeypatch.setattr(api, "encode_text", encode_text)
    monkeypatch.setattr(api, "plan_request", lambda *a, **k: SimpleNamespace(
        release_encoder=False, levers=()))
    monkeypatch.setattr(api, "sample", lambda *a, **k: mx.zeros((1, 128, 2, 2)))
    monkeypatch.setattr(api, "vae_decode", lambda pipe, x, alpha=False,
                        plan=None: mx.zeros((1, 32, 32, 3)))
    monkeypatch.chdir(tmp_path)
    rec.base = ["image", "generate", files["dit.gguf"], "--encoder",
                files["enc.gguf"], "--vae", files["vae.gguf"], "--size",
                "32x32", "--steps", "2", "--guidance", "4", "--seed", "3"]
    rec.files = files
    return rec


def test_the_cli_releases_the_upsampler_before_the_encoder_loads(
        cli, tmp_path, capsys):
    from PIL import Image

    from gmlx.commands.cli import umbrella_main

    out = tmp_path / "a.png"
    assert umbrella_main(cli.base + [
        "--prompt", "a red bicycle", "--negative-prompt", "blurry",
        "--upsampler", cli.files["writer.gguf"], "--output", str(out),
        "--json"]) == 0
    assert cli.events == ["upsampler", "chat", ("image", False),
                          ("encode", CAPTION), ("encode", "blurry")]
    (turn,) = cli.turns
    fam = get_family("flux3_klein")
    assert turn["prompt"] == "a red bicycle"
    assert turn["system_prompt"] == upsample.system_prompt(fam)
    assert (turn["temp"], turn["max_tokens"], turn["presence_penalty"],
            turn["template_kwargs"]) == (0.0, 3584, 0.0,
                                         {"enable_thinking": False})
    record = json.loads(capsys.readouterr().out)
    assert record["settings"]["revised_prompt"] == CAPTION
    assert record["model"]["upsampler"] == "writer.gguf"
    with Image.open(out) as im:
        meta = json.loads(im.text["gmlx"])
    assert (meta["prompt"], meta["revised_prompt"]) == ("a red bicycle",
                                                        CAPTION)


def test_without_an_upsampler_the_cli_warns_and_runs_the_prompt_as_typed(
        cli, tmp_path, capsys):
    from gmlx.commands.cli import umbrella_main

    assert umbrella_main(cli.base + ["--prompt", "a red bicycle",
                                     "--output", str(tmp_path / "a.png")]) == 0
    assert "set --upsampler to a Qwen3.5-9B GGUF" in capsys.readouterr().out
    assert ("encode", "a red bicycle") in cli.events
    del cli.events[:]
    assert umbrella_main(cli.base + ["--prompt", "a red bicycle",
                                     "--no-upsample",
                                     "--output", str(tmp_path / "b.png")]) == 0
    assert "expects" not in capsys.readouterr().out
    assert "upsampler" not in cli.events


def test_an_img2img_prompt_runs_as_typed_and_loads_no_upsampler(
        cli, tmp_path, monkeypatch, capsys):
    import mlx.core as mx
    from PIL import Image

    from gmlx.commands.cli import umbrella_main
    from gmlx.image import api

    init = tmp_path / "init.png"
    Image.new("RGB", (32, 32), (90, 120, 200)).save(init)
    monkeypatch.setattr(api, "vae_encode", lambda pipe, x, plan=None:
                        mx.zeros((1, 128, 2, 2)))
    assert umbrella_main(cli.base + [
        "--prompt", "the same scene at night", "--init-image", str(init),
        "--upsampler", cli.files["writer.gguf"],
        "--output", str(tmp_path / "a.png")]) == 0
    assert "--init-image runs as typed" in capsys.readouterr().out
    assert "upsampler" not in cli.events
    assert ("encode", "the same scene at night") in cli.events
