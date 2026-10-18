#!/usr/bin/env python3
"""End-to-end test of the OpenAI image routes with the openai client.

Launches the real server (``python -m gmlx.serve.server --config``) with a
small chat model and the image families, then drives it with the
``openai`` package: generations in both response formats, a stream with
partial images, edits, variations, a cancelled request and the refusals a
client sees. ``--with-chat`` keeps a chat stream decoding while each family
generates, and ``--memguard`` runs ``memguard.py`` beside the server.
``--measure`` instead times chat decode and a 1024x1024 image, each
alone and beside the other, and the time to the first byte of a streamed
image. ``--upsample`` gives FLUX.3 klein and FLUX.2 klein the
``--upsampler`` chat model and runs the prompt upsampling checks in place
of their family checks: through ``gmlx image generate`` on the entry
before the server starts, then the images route and the step contract's
encode. Writes a Markdown and a JSON report.

    python tests/e2e/run_image_api_e2e.py --out DIR [--with-chat] [--memguard]
    python tests/e2e/run_image_api_e2e.py --measure --out DIR
    python tests/e2e/run_image_api_e2e.py --family flux --upsample --out DIR
    python tests/e2e/run_image_api_e2e.py --family flux2 --upsample --out DIR

Not ``test_``-prefixed: it loads real models on the GPU.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server_proc import ServerProc, free_port  # noqa: E402

_GGUF = os.path.expanduser(os.environ.get("KQUANT_TEST_GGUF_DIR", "~/llm/gguf"))
_TEST = os.path.expanduser("~/llm/gguf-test")
DEFAULTS = {
    "chat": os.path.join(_TEST, "gemma-4-E2B-it-UD-Q6_K_XL.gguf"),
    "z_dit": os.path.join(_GGUF, "unsloth__Z-Image-Turbo-GGUF",
                          "z-image-turbo-Q4_K_M.gguf"),
    "z_encoder": os.path.join(_GGUF, "unsloth__Qwen3-4B-GGUF",
                              "Qwen3-4B-Q4_K_M.gguf"),
    "z_vae": os.path.join(_GGUF, "Tongyi-MAI__Z-Image-Turbo",
                          "z-image-vae-BF16.gguf"),
    "q_dit": os.path.join(_GGUF, "unsloth__Qwen-Image-2.1-GGUF",
                          "qwen-image-2.1-Q4_K_M.gguf"),
    "q_encoder": os.path.join(_GGUF, "unsloth__Qwen3-VL-8B-Instruct-GGUF",
                              "Qwen3-VL-8B-Instruct-Q4_K_M.gguf"),
    "q_mmproj": os.path.join(_GGUF, "unsloth__Qwen3-VL-8B-Instruct-GGUF",
                             "mmproj-BF16.gguf"),
    "q_vae": os.path.join(_GGUF, "Qwen__Qwen-Image-2.1",
                          "qwen-image-2.1-vae-F32.gguf"),
    "f_dit": os.path.join(_GGUF, "realrebelai__FLUX-3-Image-Klein_GGUFs",
                          "DISTILLED", "flux3_image_klein_sd-Q4_K_M.gguf"),
    "f_encoder": os.path.join(_GGUF, "unsloth__Qwen3-VL-4B-Instruct-GGUF",
                              "Qwen3-VL-4B-Instruct-Q8_0.gguf"),
    "f_vae": os.path.join(_GGUF, "black-forest-labs__FLUX.2-klein-4B",
                          "flux2-vae-BF16.gguf"),
    "f2_dit": os.path.join(_GGUF, "unsloth__FLUX.2-klein-4B-GGUF",
                           "flux-2-klein-4b-Q4_K_M.gguf"),
    "f2_encoder": os.path.join(_GGUF, "unsloth__Qwen3-4B-GGUF",
                               "Qwen3-4B-Q8_0.gguf"),
    "f2_vae": os.path.join(_GGUF, "black-forest-labs__FLUX.2-klein-4B",
                           "flux2-vae-BF16.gguf"),
    "upsampler": os.path.join(_GGUF, "unsloth__Qwen3.5-9B-MTP-GGUF",
                              "Qwen3.5-9B-Q6_K.gguf"),
}
FAMILIES = (("z", "z-image-turbo"), ("qwen", "qwen-image"),
            ("flux", "flux-3-klein"), ("flux2", "flux-2-klein"))
# The families --upsample gives the upsampler, by --family choice.
UPSAMPLED = ("flux", "flux2")
# The --family choices: each family, both of the first two, and all four.
CHOICES = {"z": ("z",), "qwen": ("qwen",), "flux": ("flux",),
           "flux2": ("flux2",), "both": ("z", "qwen"),
           "all": ("z", "qwen", "flux", "flux2")}
# The DEFAULTS keys of each family's files.
PREFIX = {"z": "z_", "qwen": "q_", "flux": "f_", "flux2": "f2_"}
# How to make a file a pull packs rather than downloads.
_PACK = ("packs the FLUX.2 VAE there, or pass the GGUF that gmlx image "
         "pack-vae wrote")
MAKE = {"f_vae": f"gmlx pull flux-3-klein {_PACK}",
        "f2_vae": f"gmlx pull flux-2-klein {_PACK}"}


def missing_files(a) -> list:
    """A line for each file of the run that is not there, before the
    server starts."""
    keys = ["chat"] + [k for k in DEFAULTS for f in CHOICES[a.family]
                       if k.startswith(PREFIX[f])]
    if a.upsample:
        keys.append("upsampler")
    out = []
    for k in keys:
        path = getattr(a, k)
        if not os.path.isfile(path):
            hint = f": {MAKE[k]}" if k in MAKE else ""
            out.append(f"--{k.replace('_', '-')} {path} is not a "
                       f"file{hint}")
    return out
PROMPT = "a red lighthouse on a rocky coast at dusk, photograph"


def _config(a) -> dict:
    image_models = {}
    models = {"chat": {"path": a.chat}}
    families = CHOICES[a.family]
    if "z" in families:
        encoder = a.z_encoder
        if a.shared_encoder:
            models["qwen3-4b"] = {"path": a.z_encoder}
            encoder = "qwen3-4b"
        image_models["z-image-turbo"] = {
            "path": a.z_dit, "encoder": encoder, "vae": a.z_vae}
    if "qwen" in families:
        image_models["qwen-image"] = {
            "path": a.q_dit, "encoder": a.q_encoder,
            "encoder_mmproj": a.q_mmproj, "vae": a.q_vae}
    if "flux" in families:
        # The encoder reads the prompt alone: no vision tower.
        image_models["flux-3-klein"] = {
            "path": a.f_dit, "encoder": a.f_encoder, "vae": a.f_vae}
    if "flux2" in families:
        # The encoder reads the prompt alone: no vision tower.
        image_models["flux-2-klein"] = {
            "path": a.f2_dit, "encoder": a.f2_encoder, "vae": a.f2_vae}
    if a.upsample:
        models["upsampler"] = {"path": a.upsampler}
        for family, model in FAMILIES:
            if family in UPSAMPLED and family in families:
                image_models[model]["upsampler"] = "upsampler"
    return {"models": models,
            "image_models": image_models,
            "server": {"defaults": {"model": "chat",
                                    "image_model": next(iter(image_models))},
                       "image": {"url_ttl_s": 600,
                                 "embed_metadata": bool(a.upsample)}}}


def _decode(b64: str):
    from PIL import Image

    return Image.open(io.BytesIO(base64.b64decode(b64)))


class Report:
    def __init__(self):
        self.rows: list = []

    def check(self, name: str, fn):
        t0 = time.perf_counter()
        try:
            detail = fn() or ""
            ok = True
        except Exception as e:  # noqa: BLE001 - every failure is a row
            detail, ok = f"{type(e).__name__}: {e}", False
        secs = time.perf_counter() - t0
        self.rows.append({"name": name, "ok": ok, "seconds": round(secs, 2),
                          "detail": str(detail)[:400]})
        print(f"[{'ok' if ok else 'FAIL'}] {name} ({secs:.1f}s) {detail}",
              flush=True)
        return ok


def _chat_stream(base: str, max_tokens: int, out: dict) -> None:
    """One streamed chat completion; fills ``out`` with tokens, seconds of
    decode after the first token, and the text."""
    from openai import OpenAI

    client = OpenAI(base_url=base + "/v1", api_key="none", timeout=600)
    t0 = time.perf_counter()
    first = None
    n = 0
    text = []
    stream = client.chat.completions.create(
        model="chat", stream=True, max_tokens=max_tokens, temperature=0,
        extra_body={"ignore_eos": True},
        messages=[{"role": "user", "content":
                   "Write a long story about a lighthouse keeper."}])
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if first is None:
                first = time.perf_counter()
            n += 1
            text.append(chunk.choices[0].delta.content)
    end = time.perf_counter()
    out.update(tokens=n, ttft=(first or end) - t0,
               decode_s=end - (first or end), text="".join(text))


def family_checks(rep: Report, base: str, model: str, family: str, log_path: str,
                  with_chat: bool) -> None:
    import httpx
    from openai import BadRequestError, OpenAI

    client = OpenAI(base_url=base + "/v1", api_key="none", timeout=900)
    size = "512x512"
    made = {}
    chat = {}
    chat_thread = None
    if with_chat:
        chat_thread = threading.Thread(target=_chat_stream, args=(base, 400, chat))
        chat_thread.start()

    def gen_b64():
        r = client.images.generate(model=model, prompt=PROMPT, size=size, n=2,
                                   extra_body={"seed": 11})
        ims = [_decode(d.b64_json) for d in r.data]
        assert [im.size for im in ims] == [(512, 512)] * 2, [im.size for im in ims]
        assert ims[0].tobytes() != ims[1].tobytes(), "two seeds gave one image"
        assert r.usage and r.usage.output_tokens > 0
        made["png"] = base64.b64decode(r.data[0].b64_json)
        return f"usage {r.usage.total_tokens} tokens"

    def gen_url():
        r = client.images.generate(model=model, prompt=PROMPT, size=size,
                                   response_format="url", output_format="jpeg",
                                   extra_body={"seed": 12})
        url = r.data[0].url
        got = httpx.get(url, timeout=30)
        from PIL import Image

        im = Image.open(io.BytesIO(got.content))
        assert got.status_code == 200 and im.format == "JPEG", got.status_code
        assert "?" not in url and url.rsplit(".", 1)[-1] == "jpeg"
        return url.rsplit("/", 1)[-1][:12] + "..."

    def stream():
        t0 = time.perf_counter()
        first = None
        types = []
        events = client.images.generate(model=model, prompt=PROMPT, size=size,
                                        stream=True, partial_images=2,
                                        extra_body={"seed": 13})
        for ev in events:
            if first is None:
                first = time.perf_counter() - t0
            types.append(ev.type)
            last = ev
        assert types == ["image_generation.partial_image"] * 2 + [
            "image_generation.completed"], types
        assert _decode(last.b64_json).size == (512, 512)
        return f"first event {first:.2f}s, {types}"

    def edit():
        src = made["png"]
        if family in ("qwen", "flux", "flux2"):
            r = client.images.edit(model=model, image=[("a.png", src, "image/png"),
                                                       ("b.png", src, "image/png")],
                                   prompt="put the two lighthouses side by side",
                                   size=size)
        else:
            r = client.images.edit(model=model, image=("a.png", src, "image/png"),
                                   prompt="the same lighthouse at night",
                                   size=size, extra_body={"strength": 0.7})
        assert _decode(r.data[0].b64_json).size == (512, 512)

    def edit_unsized():
        # FLUX.3 klein pads one reference off the 16 pixel grid to its
        # canvas and crops the output back to the image's own size. FLUX.2
        # klein crops the reference to the grid, and the output keeps the
        # cropped size.
        from PIL import Image

        buf = io.BytesIO()
        Image.open(io.BytesIO(made["png"])).crop((0, 0, 500, 300)).save(
            buf, format="PNG")
        r = client.images.edit(model=model, image=("a.png", buf.getvalue(),
                                                   "image/png"),
                               prompt="the same lighthouse at night")
        got = _decode(r.data[0].b64_json).size
        assert got == ((496, 288) if family == "flux2" else (500, 300)), got
        return f"{got[0]}x{got[1]}"

    def edit_stream():
        src = made["png"]
        types = [ev.type for ev in client.images.edit(
            model=model, image=("a.png", src, "image/png"),
            prompt="the same lighthouse in snow", size=size, stream=True,
            partial_images=1)]
        assert types == ["image_edit.partial_image", "image_edit.completed"], types

    def variation():
        r = client.images.create_variation(model=model, image=("a.png", made["png"],
                                                               "image/png"),
                                           n=1, size=size)
        assert _decode(r.data[0].b64_json).size == (512, 512)

    def cancel():
        before = open(log_path).read().count("cancelled")
        try:
            httpx.post(base + "/v1/images/generations", timeout=2.0, json={
                "model": model, "prompt": PROMPT, "size": "1024x1024",
                "steps": 60})
        except httpx.TimeoutException:
            pass
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if open(log_path).read().count("cancelled") > before:
                break
            time.sleep(0.5)
        else:
            raise AssertionError("no cancelled line in the server log")
        r = client.images.generate(model=model, prompt=PROMPT, size="256x256")
        assert r.data[0].b64_json
        return "the job stopped and the next request ran"

    def refusals():
        for kw in ({"model": "chat"}, {"size": "500x500"},
                   {"extra_body": {"output_path": "/tmp/x.png"}}):
            try:
                client.images.generate(**{"model": model, "prompt": PROMPT, **kw})
            except BadRequestError:
                continue
            raise AssertionError(f"no 400 for {kw}")
        if family in ("flux", "flux2"):
            # The most references either klein takes.
            src = ("a.png", made["png"], "image/png")
            try:
                client.images.edit(model=model, image=[src] * 11, prompt="x")
                raise AssertionError("no 400 for 11 references")
            except BadRequestError as e:
                assert "at most 10 images" in str(e), e

    tag = family
    rep.check(f"{tag} generations b64 n=2", gen_b64)
    rep.check(f"{tag} generations url jpeg", gen_url)
    rep.check(f"{tag} stream partial_images=2", stream)
    rep.check(f"{tag} edits", edit)
    if family in ("flux", "flux2"):
        rep.check(f"{tag} edits without a size", edit_unsized)
    rep.check(f"{tag} edits stream", edit_stream)
    rep.check(f"{tag} variations", variation)
    rep.check(f"{tag} cancellation", cancel)
    rep.check(f"{tag} refusals", refusals)
    if chat_thread is not None:
        def chat_done():
            chat_thread.join(900)
            assert chat.get("tokens", 0) > 100, chat
            return f"{chat['tokens']} tokens beside the image jobs"
        rep.check(f"{tag} chat decode in flight", chat_done)


def cli_upsample_check(rep: Report, a, cfg_path: str, model: str,
                       out: dict) -> None:
    """``gmlx image generate`` on the entry, which loads its upsampler,
    writes the caption and releases the upsampler before the image model
    loads."""
    def run():
        png = os.path.join(a.out, f"cli-{model}.png")
        got = subprocess.run(
            [a.python, "-c", "import sys; from gmlx.commands.cli import "
             "umbrella_main; sys.exit(umbrella_main(sys.argv[1:]))",
             "image", "generate", model, "--config", cfg_path, "--prompt",
             PROMPT, "--size", "512x512", "--seed", "21", "--output", png,
             "--json"], capture_output=True, text=True, timeout=1800)
        with open(os.path.join(a.out, f"cli-{model}.log"), "w") as f:
            f.write(got.stderr)
        assert got.returncode == 0, got.stderr[-2000:]
        rec = json.loads(got.stdout)
        caption = rec["settings"]["revised_prompt"]
        assert caption and caption != PROMPT
        assert rec["model"]["upsampler"] == os.path.basename(a.upsampler)
        from PIL import Image

        meta = json.loads(Image.open(png).text["gmlx"])
        assert meta["revised_prompt"] == caption
        line = next(ln for ln in got.stderr.splitlines()
                    if "prompt upsampled by" in ln)
        out.update(cli_caption=caption, cli_line=line)
        return line

    rep.check(f"{model} upsample image generate", run)


def upsample_checks(rep: Report, base: str, model: str, log_path: str,
                    out: dict) -> None:
    """Prompt upsampling on the images route and the step encode: the
    caption comes back as revised_prompt and lands in the PNG, a rerun
    with a new seed writes no new caption, and an edit and the off switch
    run the prompt as typed."""
    import httpx
    from openai import OpenAI

    from gmlx.serve.image_steps_client import StepsClient, http_transport

    client = OpenAI(base_url=base + "/v1", api_key="none", timeout=900)
    size = "512x512"
    made: dict = {}

    start = open(log_path).read().count("prompt upsampled by")

    def rewrites() -> int:
        """The captions the server wrote in these checks."""
        return open(log_path).read().count("prompt upsampled by") - start

    def first():
        t0 = time.perf_counter()
        r = client.images.generate(model=model, prompt=PROMPT, size=size,
                                   extra_body={"seed": 21})
        secs = time.perf_counter() - t0
        caption = r.data[0].revised_prompt
        assert caption and caption != PROMPT, caption
        assert rewrites() == 1, rewrites()
        png = base64.b64decode(r.data[0].b64_json)
        meta = json.loads(_decode(r.data[0].b64_json).text["gmlx"])
        assert (meta["prompt"], meta["revised_prompt"]) == (PROMPT, caption)
        made.update(caption=caption, png=png, first_s=secs)
        out.update(prompt=PROMPT, caption=caption, first_request_s=secs,
                   cli_caption_equal=out.get("cli_caption") == caption)
        return f"{len(caption.split())} words in {secs:.1f}s"

    def rerun():
        t0 = time.perf_counter()
        r = client.images.generate(model=model, prompt=PROMPT, size=size,
                                   extra_body={"seed": 22})
        secs = time.perf_counter() - t0
        assert r.data[0].revised_prompt == made["caption"]
        assert rewrites() == 1, "the rerun wrote a new caption"
        assert base64.b64decode(r.data[0].b64_json) != made["png"]
        out["rerun_s"] = secs
        return f"same caption, {secs:.1f}s against {made['first_s']:.1f}s"

    def stream():
        body = None
        with httpx.stream("POST", base + "/v1/images/generations", timeout=900,
                          json={"model": model, "prompt": PROMPT, "size": size,
                                "stream": True, "seed": 23}) as r:
            for line in r.iter_lines():
                if line.startswith("data: ") and "completed" in line:
                    body = json.loads(line[6:])
        assert body and body["revised_prompt"] == made["caption"], body

    def encode_step():
        steps = StepsClient(http_transport(base))
        fields, got = steps.encode(model, PROMPT)
        assert fields.get("revised_prompt") == made["caption"], fields
        assert fields["cond_typed"] != fields["cond"], fields
        assert got["cond_typed"].shape[-1] == got["cond"].shape[-1]
        assert rewrites() == 1, rewrites()
        return f"cond {fields['cond'][:12]}, {fields['num_tokens']} tokens"

    def off():
        r = client.images.generate(model=model, prompt=PROMPT, size=size,
                                   extra_body={"seed": 24,
                                               "prompt_upsampling": False})
        assert r.data[0].revised_prompt is None
        assert rewrites() == 1, rewrites()

    def edit():
        r = client.images.edit(model=model, image=("a.png", made["png"],
                                                   "image/png"),
                               prompt="the same scene in snow", size=size)
        assert r.data[0].revised_prompt is None
        assert rewrites() == 1, rewrites()

    for name, fn in (("generations", first),
                     ("rerun with a new seed", rerun),
                     ("stream completed event", stream),
                     ("step encode", encode_step),
                     ("prompt_upsampling false", off),
                     ("edit runs as typed", edit)):
        rep.check(f"{model} upsample {name}", fn)


def _chat_window(base: str, max_tokens: int, state: dict) -> None:
    """A streamed chat completion that records the time of each content
    chunk in ``state["times"]`` until ``state["stop"]`` is set."""
    from openai import OpenAI

    client = OpenAI(base_url=base + "/v1", api_key="none", timeout=900)
    stream = client.chat.completions.create(
        model="chat", stream=True, max_tokens=max_tokens, temperature=0,
        extra_body={"ignore_eos": True},
        messages=[{"role": "user", "content":
                   "Write a long story about a lighthouse keeper."}])
    try:
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                state["times"].append(time.perf_counter())
            if state["stop"].is_set():
                break
    finally:
        stream.close()


def _median(values: list) -> float:
    v = sorted(values)
    return v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2


def measure(base: str, out: dict, rounds: int = 2) -> None:
    """Chat decode alone, an image alone, and the two beside each other:
    a chat stream decodes while one 1024x1024 image runs, and the chat
    rate counts the tokens inside the image's time."""
    from openai import OpenAI

    client = OpenAI(base_url=base + "/v1", api_key="none", timeout=900)

    def one_image(seed: int) -> float:
        t0 = time.perf_counter()
        client.images.generate(prompt=PROMPT, size="1024x1024",
                               extra_body={"seed": seed})
        return time.perf_counter() - t0

    # Warm both models and their kernels at the measured shapes.
    one_image(0)
    warm = {}
    _chat_stream(base, 64, warm)
    alone = {}
    _chat_stream(base, 512, alone)
    image_alone = [one_image(1 + i) for i in range(rounds)]
    image_beside, chat_beside = [], []
    for i in range(rounds):
        state = {"times": [], "stop": threading.Event()}
        th = threading.Thread(target=_chat_window, args=(base, 8000, state))
        th.start()
        deadline = time.monotonic() + 120
        while len(state["times"]) < 32 and time.monotonic() < deadline:
            time.sleep(0.05)
        t0 = time.perf_counter()
        one_image(10 + i)
        t1 = time.perf_counter()
        state["stop"].set()
        th.join(300)
        inside = sum(1 for t in state["times"] if t0 <= t <= t1)
        image_beside.append(t1 - t0)
        chat_beside.append(inside / (t1 - t0))
    import httpx

    def first_byte(partials: int) -> tuple:
        """Seconds to the first body byte, to the first byte of the final
        image event, and to the end, of one streamed request."""
        t0 = time.perf_counter()
        first = final = None
        with httpx.stream("POST", base + "/v1/images/generations", timeout=900,
                          json={"prompt": PROMPT, "size": "1024x1024",
                                "stream": True,
                                "partial_images": partials}) as r:
            for chunk in r.iter_bytes():
                now = time.perf_counter() - t0
                if first is None and chunk:
                    first = now
                if final is None and b"image_generation.completed" in chunk:
                    final = now
        return first, final, time.perf_counter() - t0

    plain = first_byte(0)
    partial = first_byte(1)
    out.update(
        chat_alone_tps=alone["tokens"] / alone["decode_s"],
        chat_alone_tokens=alone["tokens"],
        image_alone_s=_median(image_alone),
        image_beside_chat_s=_median(image_beside),
        chat_beside_image_tps=_median(chat_beside),
        image_alone_runs=[round(v, 3) for v in image_alone],
        image_beside_runs=[round(v, 3) for v in image_beside],
        chat_beside_runs=[round(v, 2) for v in chat_beside],
        chat_share=os.environ.get("GMLX_IMAGE_CHAT_SHARE", "default"),
        stream_first_byte_s=plain[0], stream_total_s=plain[2],
        stream_partial1_first_byte_s=partial[0],
        stream_partial1_final_s=partial[1], stream_partial1_total_s=partial[2])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--family", choices=tuple(CHOICES), default="both",
                    help="both is Z-Image and Qwen-Image, all adds FLUX.3 "
                         "klein and FLUX.2 klein")
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), dest=k, default=v)
    ap.add_argument("--with-chat", action="store_true",
                    help="keep a chat stream decoding while each family runs")
    ap.add_argument("--shared-encoder", action="store_true",
                    help="serve the Z-Image encoder as a chat model and share it")
    ap.add_argument("--memguard", action="store_true",
                    help="run memguard.py beside the server")
    ap.add_argument("--upsample", action="store_true",
                    help="give FLUX.3 klein and FLUX.2 klein the --upsampler "
                         "chat model and run the prompt upsampling checks in "
                         "place of their family checks")
    ap.add_argument("--measure", action="store_true",
                    help="time chat decode beside an image job and the first "
                         "byte of a streamed image, with Z-Image only")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()
    if a.measure:
        a.family = "z"
    missing = missing_files(a)
    if missing:
        print("\n".join(missing), file=sys.stderr)
        return 2
    os.makedirs(a.out, exist_ok=True)
    import yaml

    cfg_path = os.path.join(a.out, "gmlx.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(_config(a), f, sort_keys=False)
    port = free_port()
    guard = None
    if a.memguard:
        # Only this run's server: other servers on the machine are not ours.
        guard = subprocess.Popen([
            a.python, os.path.join(os.path.dirname(__file__), "memguard.py"),
            "--verbose", "--log", os.path.join(a.out, "memguard.log"),
            "--pattern", rf"gmlx\.serve\.server .*--port {port}\b"])
    log_path = os.path.join(a.out, "server.log")
    rep = Report()
    result: dict = {"config": _config(a)}
    result["upsample"] = {}
    for family, model in FAMILIES:
        if a.upsample and family in UPSAMPLED and family in CHOICES[a.family]:
            cli_upsample_check(rep, a, cfg_path, model,
                               result["upsample"].setdefault(model, {}))
    try:
        with ServerProc(["--config", cfg_path], log_path=log_path, port=port,
                        python=a.python) as srv:
            srv.wait_ready(timeout=300)
            base = srv.base_url
            if a.measure:
                m: dict = {}
                rep.check("measurements", lambda: measure(base, m) or json.dumps(
                    {k: round(v, 3) if isinstance(v, float) else v
                     for k, v in m.items() if not isinstance(v, list)}))
                result["measurements"] = m
            else:
                for family, model in FAMILIES:
                    if family not in CHOICES[a.family]:
                        continue
                    if family in UPSAMPLED and a.upsample:
                        upsample_checks(rep, base, model, log_path,
                                        result["upsample"][model])
                        continue
                    family_checks(rep, base, model, family, log_path,
                                  a.with_chat)
    finally:
        if guard is not None:
            guard.terminate()
    result["checks"] = rep.rows
    ok = all(r["ok"] for r in rep.rows)
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(a.out, "report.md"), "w") as f:
        f.write("# Image API end to end\n\n| Check | Result | Seconds | Detail |\n"
                "|-------|--------|---------|--------|\n")
        for r in rep.rows:
            f.write(f"| {r['name']} | {'ok' if r['ok'] else 'FAIL'} | "
                    f"{r['seconds']} | {r['detail'].replace('|', '/')} |\n")
    print(f"RESULT: {sum(r['ok'] for r in rep.rows)}/{len(rep.rows)} checks passed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
