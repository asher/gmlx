"""Prompt upsampling: the chat turn is the source's for a prompt without
references, from the system prompt gmlx ships, the caption is what the
family's parser reads in a reply that thinks first, and ``gmlx image
generate`` releases the upsampler before the encoder loads and encodes
the caption in place of the typed prompt, but never the negative.
Qwen-Image 2.1 ships no system prompt and takes the user's. The chat
model and the image pipeline are stubs."""
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

CAPTION = "A red bicycle leans against a weathered brick wall."


# Each shipped system prompt, pinned by the sha256 of the file its source
# publishes, and the sampling of the turn. The FLUX.2 klein rows of every
# size share BFL's text to image system message.
@pytest.mark.parametrize("family, package, name, sha256, key, sampling", [
    ("flux3_klein", "flux3_klein", "pup_prompts_external_vlm.json",
     "361bb6e081d50e54a3dee9abf26327436a9386c83465f881c7848989232028be",
     "t2i", (0.0, 3584, 0.0, "off")),
    ("flux2_klein", "flux2_klein", "system_message_upsampling_t2i.txt",
     "0ad9a42c565716b0a8eb82a442dd4f6cc82d583eb3474f82ccb610c1a29233c7",
     None, (0.0, 1024, 0.0, "off")),
    ("flux2_klein_9b_kv", "flux2_klein", "system_message_upsampling_t2i.txt",
     "0ad9a42c565716b0a8eb82a442dd4f6cc82d583eb3474f82ccb610c1a29233c7",
     None, (0.0, 1024, 0.0, "off")),
    ("krea2", "krea2", "expansion.txt",
     "2a5c24a3a83e9d415d4679a4e1853ddf056bd0329f3af315b27ed72b6ad9c2f0",
     None, (0.0, 2048, 0.0, "off")),
    # The typed prompt goes in the {prompt} slot, and the user turn is a
    # space, as the Space's app.py sends them.
    ("z_image", "z_image", "pe_prompt_template.txt",
     "36bdd888b953e9b7a8237dab55bee145a44e96e92e8f6e81a6e4d433a401a0b6",
     None, (0.0, 2048, 0.0, "off")),
])
def test_the_chat_turn_is_the_sources_t2i_turn(family, package, name, sha256,
                                                key, sampling):
    fam = get_family(family)
    data = (resources.files(f"gmlx.image.families.{package}") / name
            ).read_bytes()
    assert hashlib.sha256(data).hexdigest() == sha256
    system = json.loads(data)[key] if key else data.decode("utf-8")
    user = "a red bicycle"
    if "{prompt}" in system:
        system, user = system.replace("{prompt}", user), " "
    req = upsample.chat_request(fam, "a red bicycle")
    assert req.messages == (
        {"role": "system", "content": system},
        {"role": "user", "content": user})
    assert (req.temperature, req.max_tokens, req.presence_penalty,
            req.thinking) == sampling


KREA_REPLY = """The user wants a red bicycle. Subject: a bicycle, red, \
nothing else implied. Mood: quiet, everyday. Style options: documentary \
photo, gouache illustration, 3D render. A photo serves a plain request \
best. Composition: side view, the whole frame of the bike, a wall behind \
it for context.
</think>

Expanded prompt:

A red city bicycle with a black leather saddle and a wicker basket \
stands on its kickstand against a weathered brick wall, photographed \
from the side in soft late afternoon light. The paint of the frame is \
slightly faded, and a long shadow falls across the cobblestones.
"""


def test_krea_2_takes_the_paragraph_after_the_thinking_block():
    fam = get_family("krea2")
    got = upsample.upsample(fam, "a red bicycle", lambda req: KREA_REPLY)
    assert got.startswith("A red city bicycle with a black leather saddle")
    assert got.endswith("falls across the cobblestones.")
    # Reasoning written out of the thinking block is not the caption,
    # though it runs longer.
    reasoning = ("Step 1: the subject is a bicycle, red, with nothing else "
                 "implied by the request.\nStep 2: a documentary photo "
                 "serves a plain request best, so the style is a photo.\n"
                 "Step 3: a side view shows the whole frame of the bike.")
    assert upsample.upsample(
        fam, "a red bicycle", lambda req:
        f"{reasoning}\n\nA red bicycle against a brick wall.\n") == (
        "A red bicycle against a brick wall.")
    # A rule after the caption is not one.
    assert upsample.upsample(fam, "a red bicycle", lambda req:
                             "A red bicycle.\n\n---\n") == "A red bicycle."
    # A reply that runs out of tokens while it thinks holds no paragraph.
    with pytest.raises(upsample.UpsampleError,
                       match="no caption within 2048 tokens"):
        upsample.upsample(fam, "a red bicycle", lambda req:
                          "<think>\nThe user wants a red bicycle. Style")


QWEN_REPLY = """The prompt is short. Draft: {"rewritten_prompt": "a bike"} \
is too thin; add the setting and the light.
</think>

```json
{"rewritten_prompt": "A red bicycle with a wicker basket leans against a \
weathered brick wall in soft afternoon light.", "wh_ratio": "3:2", \
"ratio_follow": ""}
```
"""


def test_qwen_image_takes_rewritten_prompt_from_the_json_after_thinking():
    fam = get_family("qwen_image21")
    rewrite = "own prompt: answer with JSON"
    got = upsample.upsample(fam, "a red bicycle", lambda req: QWEN_REPLY,
                            system=rewrite)
    assert got == ("A red bicycle with a wicker basket leans against a "
                   "weathered brick wall in soft afternoon light.")
    # A reply with no JSON object is the caption as written.
    assert upsample.upsample(
        fam, "a red bicycle", lambda req: "</think>\nA red bicycle.\n",
        system=rewrite) == "A red bicycle."


@pytest.mark.parametrize("answer", [
    '{"rewritten_prompt": "A red bicycle leans against a wall, and a red',
    '{"rewritten_prompt": "", "wh_ratio": "3:2"}',
    '{"rewritten_prompt": null}',
    '{"prompt": "A red bicycle."}',
    "{'rewritten_prompt': 'A red bicycle.'}"])
def test_a_qwen_image_object_without_the_field_is_an_error(answer):
    # A reply cut off in its JSON never reaches the encoder as raw JSON.
    fam = get_family("qwen_image21")
    with pytest.raises(upsample.UpsampleError,
                       match="a reply with no caption in it"):
        upsample.upsample(fam, "a red bicycle", lambda req:
                          f"</think>\n{answer}", system="own prompt")


def test_qwen_image_reads_a_rewritten_prompt_with_unescaped_quotes():
    fam = get_family("qwen_image21")
    reply = ('</think>\n```json\n{"rewritten_prompt": "a sign that reads '
             '"OPEN" on a door,\\nin red", "wh_ratio": "3:2", '
             '"ratio_follow": ""}\n```\n')
    got = upsample.upsample(fam, "a sign", lambda req: reply,
                            system="own prompt")
    assert got == 'a sign that reads "OPEN" on a door,\nin red'
    emoji = '{"rewritten_prompt": "a "SALE" sign \\ud83d\\ude00", "x": 1}'
    assert upsample.upsample(fam, "a sign", lambda req: emoji,
                             system="own prompt") == 'a "SALE" sign \U0001f600'
    last = '{"rewritten_prompt": "a sign that reads "OPEN" on a door"}'
    assert upsample.upsample(fam, "a sign", lambda req: last,
                             system="own prompt") == (
        'a sign that reads "OPEN" on a door')


def test_a_thinking_block_in_another_spelling_is_skipped():
    fam = get_family("flux3_klein")
    reply = f"<mm:think>\nThe user wants a bike.\n</mm:think>\n{CAPTION}\n"
    assert upsample.upsample(fam, "x", lambda req: reply) == CAPTION
    with pytest.raises(upsample.UpsampleError, match="no caption within"):
        upsample.upsample(fam, "x", lambda req: "<mm:think>\nThe user")


def test_z_image_takes_revised_prompt_or_the_reply_as_written():
    fam = get_family("z_image")
    block = '```json\n{"revised_prompt": "A red bicycle."}\n```'
    assert upsample.upsample(fam, "x", lambda req: block) == "A red bicycle."
    # As the Space reads it, a reply whose braces hold no revised_prompt
    # is the caption.
    sign = "A neon sign that reads {OPEN} above a red bicycle."
    assert upsample.upsample(fam, "x", lambda req: sign) == sign


@pytest.mark.parametrize("reply", [
    "<|channel>thought\nThe user wants a bike.<channel|>{caption}",
    "<|channel|>analysis<|message|>The user wants a bike.<|end|>"
    "<|start|>assistant<|channel|>final<|message|>{caption}<|return|>"])
def test_a_reply_in_a_channel_format_gives_its_answer(reply):
    fam = get_family("flux3_klein")
    assert upsample.upsample(fam, "x", lambda req: reply.format(
        caption=CAPTION)) == CAPTION


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
    # ``fam`` is the family the DiT identifies as, and ``reply`` what the
    # upsampler writes.
    rec = SimpleNamespace(events=[], turns=[], model=None,
                          fam=get_family("flux3_klein"), reply=f" {CAPTION}\n")

    class Model:
        pass

    def load_model(path):
        model = Model()
        rec.model = weakref.ref(model)
        rec.events.append("upsampler")
        tok = SimpleNamespace(
            chat_template="{% if enable_thinking %}{% endif %}",
            apply_chat_template=lambda messages, **kw: (
                "<think>\n" if kw.get("enable_thinking") else ""))
        return model, None, tok

    def generate(model, tok, prompt, **kw):
        rec.events.append("chat")
        rec.turns.append({"prompt": prompt, **kw})
        return rec.reply

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
                               load_seconds={}, family=rec.fam)

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
                            family=rec.fam, variant="base", config={}))
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


@pytest.mark.parametrize("given", [None, "missing.txt"])
def test_qwen_image_without_a_prompt_file_refuses_and_names_qwens(
        cli, tmp_path, capsys, given):
    from gmlx.commands.cli import umbrella_main

    cli.fam = get_family("qwen_image21")
    # Without an upsampler, its prompt runs as typed with no warning.
    assert umbrella_main(cli.base + ["--prompt", "a red bicycle",
                                     "--output", str(tmp_path / "a.png")]) == 0
    assert "expects" not in capsys.readouterr().out
    args = ["--prompt", "a red bicycle", "--upsampler",
            cli.files["writer.gguf"], "--output", str(tmp_path / "b.png")]
    if given:
        args += ["--upsample-prompt", str(tmp_path / given)]
    assert umbrella_main(cli.base + args) == 2
    err = capsys.readouterr().err
    assert "set upsample_prompt on its image_models entry" in err
    assert ("system_prompt.txt in the Hugging Face repository "
            "Qwen/Qwen-Image-2.1-PE-T2I, under the Qwen Research License, "
            "for non-commercial use only, and `gmlx pull qwen-image` "
            "fetches it") in err
    assert ("no such file" in err) == bool(given)
    assert "upsampler" not in cli.events


def test_qwen_image_runs_the_users_prompt_file_and_encodes_its_rewrite(
        cli, tmp_path):
    from gmlx.commands.cli import umbrella_main

    cli.fam = get_family("qwen_image21")
    cli.reply = QWEN_REPLY
    own = tmp_path / "rewrite.txt"
    own.write_text("Rewrite the prompt. Think, then answer with JSON.\n")
    assert umbrella_main(cli.base + [
        "--prompt", "a red bicycle", "--upsampler", cli.files["writer.gguf"],
        "--upsample-prompt", str(own),
        "--output", str(tmp_path / "a.png")]) == 0
    (turn,) = cli.turns
    assert turn["system_prompt"] == ("Rewrite the prompt. Think, then answer "
                                     "with JSON.")
    assert (turn["presence_penalty"], turn["template_kwargs"]) == (
        0.0, {"enable_thinking": False})
    assert ("encode", "A red bicycle with a wicker basket leans against a "
            "weathered brick wall in soft afternoon light.") in cli.events
