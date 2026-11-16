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
     "t2i", (0.0, 3584, 1.5, "off")),
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
    # gmlx's own caption prompt, from the OpenMOSS prompt-enhancer skill.
    ("moss_image", "moss_image", "caption_prompt.txt",
     "389ce41e0d9f4e11df7f0c078cef7a4af772b7312b19fc7d637e5808345cbba1",
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


# MiniMax-H3-Image: MiniMax's prompts, which gmlx neither ships nor
# copies into a test, laid out as its PE repository lays them out, here
# with stand-in text.
H3_SYSTEM = "You expand image prompts.\n\n"
H3_TASK = "Expand the prompt below.\n\nPrompt: {}\n"
H3_EDIT = "Write the edited image's caption.\n"


@pytest.fixture
def h3_prompts(tmp_path):
    """MiniMax's prompt/manifest.json and the files it lists, with
    stand-in text."""
    root = tmp_path / "pe"
    files = {"prompt/t2i/system_prompt.txt": H3_SYSTEM,
             "prompt/t2i/task_prompt.txt": H3_TASK,
             "prompt/t2i/request_params.json": "{}",
             "prompt/edit/task_prompt.txt": H3_EDIT}
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    listed = {task: {"files": {rel: {"sha256": "0" * 64} for rel in files
                               if rel.startswith(f"prompt/{task}/")}}
              for task in ("t2i", "edit")}
    (root / "prompt" / "manifest.json").write_text(json.dumps(listed))
    return root / "prompt" / "manifest.json"


def _picture(w, h, fill=0, channels=3):
    import numpy as np

    return upsample.Picture.of(np.full((h, w, channels), fill, np.uint8))


def test_minimax_h3_image_takes_minimaxs_t2i_turn_and_sampling(h3_prompts):
    """The system prompt without its trailing newlines, the task prompt
    with the typed prompt in its slot, and MiniMax's request settings:
    temperature 1, top-p 0.95, top-k 20, thinking on, 4096 tokens. The
    image's seed seeds the turn."""
    fam = get_family("h3_image")
    system = upsample.system_prompt(fam, str(h3_prompts))
    req = upsample.chat_request(fam, "a red bicycle", system, seed=7)
    assert req.messages == (
        {"role": "system", "content": "You expand image prompts."},
        {"role": "user", "content": "Expand the prompt below.\n\nPrompt: "
                                    "a red bicycle\n"})
    assert (req.temperature, req.top_p, req.top_k, req.min_p,
            req.presence_penalty, req.max_tokens, req.thinking) == (
        1.0, 0.95, 20, 0.0, 0.0, 4096, "on")
    assert (req.seed, req.pictures) == (7, ())
    assert upsample.upsample(
        fam, "a red bicycle", lambda r: '</think>\n{"expanded_caption": '
        '"A red bicycle.", "aspect_ratio": "3:2"}', system=system) == (
        "A red bicycle.")


def test_minimax_h3_image_with_references_takes_minimaxs_edit_turn(
        h3_prompts):
    """One user turn of parts: the editing task prompt, the picture count,
    each picture's label and shape before it, then the instruction as a
    JSON string and the aspect ratio the request selects."""
    fam = get_family("h3_image")
    system = upsample.system_prompt(fam, str(h3_prompts))
    pics = (_picture(768, 1024), _picture(512, 512, channels=4))
    req = upsample.chat_request(fam, 'paint the "door" blue', system,
                                seed=3, pictures=pics)
    tail = ("\nUser Editing Instruction (JSON string):\n"
            '"paint the \\"door\\" blue"\n\n'
            "User Aspect Ratio Selection:\nadaptive\n\n"
            "Apply the instructions above to the supplied inputs and "
            "reference images. Output only the required two-field JSON "
            "object.")
    assert req.messages == ({"role": "user", "content": [
        {"type": "text", "text": H3_EDIT},
        {"type": "text", "text": "Reference Image Count:\n2 input image(s), "
                                 "labeled <Picture 1> through <Picture 2>."},
        {"type": "text", "text": "Reference Media:\n- <Picture 1> "
                                 "[aspect_ratio 3:4]"},
        {"type": "image"},
        {"type": "text", "text": "Reference Media:\n- <Picture 2> "
                                 "[aspect_ratio 1:1]"},
        {"type": "image"},
        {"type": "text", "text": tail}]},)
    assert req.pictures == pics and req.system is None
    # A request that names its size selects its shape.
    sized = upsample.chat_request(fam, "x", system, pictures=pics,
                                  canvas=(1344, 768))
    assert sized.messages[0]["content"][-1]["text"].endswith(
        "Selection:\n7:4\n\nApply the instructions above to the supplied "
        "inputs and reference images. Output only the required two-field "
        "JSON object.")
    # Without references the turn is the t2i one.
    assert upsample.chat_request(fam, "x", system).messages[0]["role"] == \
        "system"


def test_minimax_h3_image_refuses_a_prompt_file_that_is_not_minimaxs(
        h3_prompts, tmp_path):
    fam = get_family("h3_image")
    with pytest.raises(upsample.UpsampleError, match="no such file. "
                       "MiniMax-H3-Image ships no upsampler prompt"):
        upsample.system_prompt(fam, str(tmp_path / "missing.json"))
    with pytest.raises(upsample.UpsampleError, match="`gmlx pull "
                       "minimax-h3-image` fetches them"):
        upsample.system_prompt(fam)
    other = tmp_path / "own.txt"
    other.write_text("Rewrite the prompt.\n")
    with pytest.raises(upsample.UpsampleError, match="is not MiniMax's "
                       "prompt/manifest.json, which lists t2i's "
                       "system_prompt.txt"):
        upsample.system_prompt(fam, str(other))
    listed = json.loads(h3_prompts.read_text())
    del listed["edit"]
    h3_prompts.write_text(json.dumps(listed))
    with pytest.raises(upsample.UpsampleError, match="lists edit's "
                       "task_prompt.txt"):
        upsample.system_prompt(fam, str(h3_prompts))


def test_minimax_h3_image_captions_are_keyed_by_seed_pictures_and_canvas(
        h3_prompts):
    """A sampled caption depends on its seed, and one that saw pictures
    on their pixels and the requested size; a sampled turn without a seed
    is never kept."""
    fam = get_family("h3_image")
    system = upsample.system_prompt(fam, str(h3_prompts))
    a, b = _picture(64, 32), _picture(64, 32, fill=1)

    def key(**kw):
        return upsample.cache_key("up", fam, "x", system, **kw)

    assert key() is None
    assert key(seed=1) != key(seed=2)
    assert key(seed=1, pictures=(a,)) != key(seed=1, pictures=(b,))
    assert key(seed=1, pictures=(a,)) == key(seed=1,
                                             pictures=(_picture(64, 32),))
    assert key(seed=1, pictures=(a,)) != key(seed=1, pictures=(_picture(
        32, 64),))
    assert key(seed=1, pictures=(a,)) != key(seed=1, pictures=(a,),
                                             canvas=(64, 32))


def test_every_other_familys_upsampler_sees_no_pictures_and_keeps_its_turn():
    """Only MiniMax-H3-Image's upsampler takes pictures. Every other
    family's turn is greedy, unseeded and text alone, and its caption key
    leaves out the seed, as before."""
    from gmlx.image.registry import families

    others = [f for f in families()
              if f.upsample is not None and f.name != "h3_image"]
    assert len(others) >= 6
    for fam in others:
        assert not upsample.takes_pictures(fam)
        system = "Rewrite the prompt."
        req = upsample.chat_request(fam, "a red bicycle", system, seed=5)
        assert (req.seed, req.pictures, req.temperature) == (None, (), 0.0)
        assert (req.top_p, req.top_k, req.min_p) == (1.0, 0, 0.0)
        assert all(isinstance(m["content"], str) for m in req.messages)
        assert upsample.cache_key("up", fam, "x", system, seed=1) == \
            upsample.cache_key("up", fam, "x", system, seed=2) == (
                "up", fam.upsample, hashlib.sha256(
                    system.encode()).hexdigest(), "x")
        with pytest.raises(ValueError, match="sees no reference images"):
            upsample.chat_request(fam, "x", system,
                                  pictures=(_picture(32, 32),))
    assert upsample.takes_pictures(get_family("h3_image"))


def test_the_minimax_h3_image_recipe_pins_the_revision_the_row_names():
    from gmlx.image import catalog
    from gmlx.image.families.h3_image import upsample as h3

    r = catalog.recipe("minimax-h3-image")
    assert (r.prompt_repo, r.prompt_revision, r.prompt_file) == (
        h3.PE_REPO, h3.PE_REVISION, h3.MANIFEST)
    assert get_family("h3_image").upsample.revision == h3.PE_REVISION
    assert sorted(f for f, _ in r.prompt_more) == [
        "prompt/edit/task_prompt.txt", "prompt/t2i/request_params.json",
        "prompt/t2i/system_prompt.txt", "prompt/t2i/task_prompt.txt"]


def test_minimax_h3_image_prompt_files_on_disk_are_the_pinned_ones(gguf_dir):
    """MiniMax's files where pull lays them, under KQUANT_TEST_GGUF_DIR:
    each is at the recipe's pin and the manifest's sha256, and the turn
    built from them has the system prompt without its trailing newlines
    and the typed prompt in the task's slot."""
    from pathlib import Path

    from gmlx.image import catalog

    r = catalog.recipe("minimax-h3-image")
    found = [m for m in sorted(Path(gguf_dir).rglob("manifest.json"))
             if m.parent.name == "prompt"
             and catalog.sha256_file(str(m)) == r.prompt_sha256]
    if not found:
        pytest.skip("MiniMax's prompt/manifest.json is not under "
                    "KQUANT_TEST_GGUF_DIR")
    manifest = found[0]
    root = manifest.parent.parent
    listed = {rel: v["sha256"] for task in json.loads(manifest.read_text())
              .values() for rel, v in task["files"].items()}
    assert listed == dict(r.prompt_more)
    for rel, sha in r.prompt_more:
        assert catalog.sha256_file(str(root / rel)) == sha
    fam = get_family("h3_image")
    req = upsample.chat_request(
        fam, "a red bicycle", upsample.system_prompt(fam, str(manifest)))
    system = (root / "prompt/t2i/system_prompt.txt").read_text()
    task = (root / "prompt/t2i/task_prompt.txt").read_text()
    assert req.messages == ({"role": "system", "content": system.rstrip("\n")},
                            {"role": "user",
                             "content": task.format("a red bicycle")})


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


def test_moss_image_takes_the_caption_from_its_json_or_the_reply_as_written():
    """The caption keeps its paragraphs and the text the image shows;
    a reply with no object is the caption as written."""
    fam = get_family("moss_image")
    caption = ('A bakery storefront under a painted sign that reads "FRESH '
               'BREAD DAILY".\n\nLoaves fill the window in warm light.')
    reply = json.dumps({"caption": caption})
    assert upsample.upsample(fam, "x", lambda req: reply) == caption
    fenced = "```json\n" + reply + "\n```"
    assert upsample.upsample(fam, "x", lambda req: fenced) == caption
    assert upsample.upsample(fam, "x", lambda req: CAPTION) == CAPTION
    with pytest.raises(upsample.UpsampleError):
        upsample.upsample(fam, "x", lambda req: '{"layout": []}')

@pytest.mark.parametrize("reply", [
    "<|channel>thought\nThe user wants a bike.<channel|>{caption}",
    "<|channel|>analysis<|message|>The user wants a bike.<|end|>"
    "<|start|>assistant<|channel|>final<|message|>{caption}<|return|>"])
def test_a_reply_in_a_channel_format_gives_its_answer(reply):
    fam = get_family("flux3_klein")
    assert upsample.upsample(fam, "x", lambda req: reply.format(
        caption=CAPTION)) == CAPTION


@pytest.mark.parametrize("reply", [
    "Sure, here is a rewrite.\n<think>\nThe user wants",
    "A t-shirt printed with the text <think> in bold.",
    "A poster reading <|channel|> in bold letters..."])
def test_a_reply_that_ends_in_reasoning_is_an_error(reply):
    fam = get_family("flux3_klein")
    with pytest.raises(upsample.UpsampleError, match="no caption within"):
        upsample.upsample(fam, "x", lambda req: reply)


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
                          fam=get_family("flux3_klein"), reply=f" {CAPTION}\n",
                          rendered=None)

    class Model:
        pass

    def load_model(path):
        model = Model()
        rec.model = weakref.ref(model)
        rec.events.append("upsampler")
        tok = SimpleNamespace(
            chat_template="{% if enable_thinking %}{% endif %}",
            apply_chat_template=lambda messages, **kw: rec.rendered or (
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
            turn["template_kwargs"]) == (0.0, 3584, 1.5,
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


def test_an_atem_reply_that_starts_mid_header_gives_its_caption(
        cli, tmp_path):
    from gmlx.commands.cli import umbrella_main

    # With thinking off, the prompt ends at the header the reply finishes.
    cli.rendered = "<|start|>user<|message|>a red bicycle<|eot|><|start|>assistant"
    cli.reply = f" to=user<|message|>{CAPTION}<|eot|>"
    assert umbrella_main(cli.base + [
        "--prompt", "a red bicycle", "--upsampler", cli.files["writer.gguf"],
        "--output", str(tmp_path / "a.png")]) == 0
    assert ("encode", CAPTION) in cli.events


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
        1.5, {"enable_thinking": False})
    assert ("encode", "A red bicycle with a wicker basket leans against a "
            "weathered brick wall in soft afternoon light.") in cli.events


H3_TEMPLATE = (
    "{% if enable_thinking is defined %}{% endif %}"
    "{%- for m in messages -%}<|im_start|>{{ m.role }}\n"
    "{% if m.content is string %}{{ m.content }}{% else %}"
    "{%- for c in m.content %}{% if c.type == 'image' %}"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "{%- else %}{{ c.text }}{% endif %}{% endfor %}{% endif %}"
    "<|im_end|>\n{% endfor %}"
    "{%- if add_generation_prompt %}<|im_start|>assistant\n<think>\n"
    "{% endif %}")


@pytest.fixture
def h3_cli(cli, tmp_path, monkeypatch, h3_prompts):
    """The cli fixture on MiniMax-H3-Image with two references and a
    vision upsampler: ``vlm`` records the tower's load, the rendered
    prompt and the pictures the turn sends, and ``priced`` the files the
    cache limit priced and the context the preflight checked."""
    import jinja2
    from PIL import Image

    import gmlx._vendor.mlx_vlm as mlx_vlm
    import gmlx.load.vlm as vlm
    import gmlx.serve.memory as memory
    from gmlx.commands import cli as cli_mod
    from gmlx.commands import tool_preflight

    cli.fam = get_family("h3_image")
    cli.reply = ('</think>\n{"expanded_caption": "' + CAPTION
                 + '", "target_aspect_ratio": "1:1"}')
    template = jinja2.Environment().from_string(H3_TEMPLATE)
    rec = SimpleNamespace(loads=[], turns=[], priced=[], ctx=[])

    def apply_chat_template(messages, tokenize=False, **kw):
        return template.render(messages=messages, **kw)

    processor = SimpleNamespace(
        chat_template=H3_TEMPLATE, apply_chat_template=apply_chat_template,
        tokenizer=SimpleNamespace(chat_template=H3_TEMPLATE))

    def load_vlm_model(path, mmproj, **kw):
        rec.loads.append((path, mmproj))
        cli.events.append("upsampler+tower")
        return object(), None, processor

    def vlm_generate(model, proc, prompt, image=(), **kw):
        cli.events.append("chat")
        rec.turns.append({"prompt": prompt, "image": image, **kw})
        return SimpleNamespace(text=cli.reply)

    monkeypatch.setattr(vlm, "load_vlm_model", load_vlm_model)
    monkeypatch.setattr(mlx_vlm, "generate", vlm_generate)
    monkeypatch.setattr(cli_mod, "unfit_request", lambda *a: None)
    from gmlx.image import api

    monkeypatch.setattr(api, "encode_references", lambda pipe, c, plan=None: c)
    monkeypatch.setattr(memory, "resolve_cache_limit",
                        lambda r, files, ws: (rec.priced.append(files)
                                              or (None, "test")))
    monkeypatch.setattr(tool_preflight, "check_or_exit",
                        lambda p, ctx_tokens=None: rec.ctx.append(ctx_tokens))
    refs = []
    for i, (w, h) in enumerate([(768, 1024), (512, 512)]):
        refs.append(tmp_path / f"ref{i}.png")
        Image.new("RGB", (w, h), (40 * i, 90, 200)).save(refs[-1])
    tower = tmp_path / "mmproj.gguf"
    tower.write_bytes(b"x")
    rec.tower = str(tower)
    # The base runs its distilled 6 steps; with no --size, the image takes
    # its size from the references.
    rec.base = [a for a in cli.base if a not in ("--size", "32x32")]
    rec.base[rec.base.index("--steps") + 1] = "6"
    rec.args = ["--prompt", "paint the door blue", "--ref", str(refs[0]),
                "--ref", str(refs[1]), "--encoder-mmproj", str(tower),
                "--upsampler", cli.files["writer.gguf"],
                "--upsample-prompt", str(h3_prompts),
                "--output", str(tmp_path / "a.png")]
    cli.h3 = rec
    return cli


def test_minimax_h3_image_cli_shows_its_upsampler_the_references(h3_cli):
    """The upsampler loads with its vision tower, which the cache limit
    prices beside it, and its turn renders each picture in place, sends
    the pictures in order, samples at MiniMax's settings, and is seeded by
    the image's seed. The caption is encoded."""
    from gmlx.commands.cli import umbrella_main

    rec = h3_cli.h3
    assert umbrella_main(rec.base + rec.args + [
        "--upsampler-mmproj", rec.tower]) == 0
    assert rec.loads == [(h3_cli.files["writer.gguf"], rec.tower)]
    assert rec.priced[0] == [h3_cli.files["writer.gguf"], rec.tower]
    (turn,) = rec.turns
    prompt = turn["prompt"]
    assert prompt.startswith("<|im_start|>user\n" + H3_EDIT
                             + "Reference Image Count:\n2 input image(s)")
    assert prompt.index("<Picture 1> [aspect_ratio 3:4]<|vision_start|>") < \
        prompt.index("<Picture 2> [aspect_ratio 1:1]<|vision_start|>")
    assert "User Aspect Ratio Selection:\nadaptive" in prompt
    assert [im.size for im in turn["image"]] == [(768, 1024), (512, 512)]
    assert (turn["temperature"], turn["top_p"], turn["top_k"], turn["min_p"],
            turn["max_tokens"]) == (1.0, 0.95, 20, 0.0, 4096)
    # About the turn's text over 3, the vision tokens of a 768x1024 and a
    # 512x512 picture, and the reply.
    assert rec.ctx[0] > 4096 + 24 * 32 + 16 * 16
    assert ("encode", CAPTION) in h3_cli.events
    assert "upsampler" not in h3_cli.events


def test_minimax_h3_image_cli_without_the_upsamplers_tower_refuses(
        h3_cli, capsys):
    from gmlx.commands.cli import umbrella_main

    assert umbrella_main(h3_cli.h3.base + h3_cli.h3.args) == 2
    assert ("--ref: the MiniMax-H3-Image upsampler sees the references "
            "through its vision tower: pass --upsampler-mmproj"
            ) in capsys.readouterr().err
    assert h3_cli.h3.loads == [] and "chat" not in h3_cli.events
    # Without references its upsampler runs on the text model alone.
    assert umbrella_main(h3_cli.h3.base + [
        "--prompt", "a red bicycle", "--upsampler", h3_cli.files[
            "writer.gguf"], "--upsample-prompt", h3_cli.h3.args[
            h3_cli.h3.args.index("--upsample-prompt") + 1],
        "--output", "b.png"]) == 0
    (turn,) = h3_cli.turns
    assert turn["system_prompt"] == "You expand image prompts."
    assert (turn["temp"], turn["top_p"], turn["top_k"],
            turn["template_kwargs"]) == (1.0, 0.95, 20,
                                         {"enable_thinking": True})
    assert h3_cli.h3.loads == []
