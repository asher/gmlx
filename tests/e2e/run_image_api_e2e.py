#!/usr/bin/env python3
"""End-to-end test of the OpenAI image routes with the openai client.

Launches the real server (``python -m gmlx.serve.server --config``) with a
small chat model and the two image families, then drives it with the
``openai`` package: generations in both response formats, a stream with
partial images, edits, variations, a cancelled request and the refusals a
client sees. ``--with-chat`` keeps a chat stream decoding while each family
generates, and ``--memguard`` runs ``memguard.py`` beside the server.
``--measure`` instead times chat decode with and without an image job
beside it, and the time to the first byte of a streamed image. Writes a
Markdown and a JSON report.

    python tests/e2e/run_image_api_e2e.py --out DIR [--with-chat] [--memguard]
    python tests/e2e/run_image_api_e2e.py --measure --out DIR

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
}
PROMPT = "a red lighthouse on a rocky coast at dusk, photograph"


def _config(a) -> dict:
    image_models = {}
    models = {"chat": {"path": a.chat}}
    if a.family in ("z", "both"):
        encoder = a.z_encoder
        if a.shared_encoder:
            models["qwen3-4b"] = {"path": a.z_encoder}
            encoder = "qwen3-4b"
        image_models["z-image-turbo"] = {
            "path": a.z_dit, "encoder": encoder, "vae": a.z_vae}
    if a.family in ("qwen", "both"):
        image_models["qwen-image"] = {
            "path": a.q_dit, "encoder": a.q_encoder,
            "encoder_mmproj": a.q_mmproj, "vae": a.q_vae}
    return {"models": models,
            "image_models": image_models,
            "server": {"defaults": {"model": "chat",
                                    "image_model": next(iter(image_models))},
                       "image": {"url_ttl_s": 600}}}


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
        if family == "qwen":
            r = client.images.edit(model=model, image=[("a.png", src, "image/png"),
                                                       ("b.png", src, "image/png")],
                                   prompt="put the two lighthouses side by side",
                                   size=size)
        else:
            r = client.images.edit(model=model, image=("a.png", src, "image/png"),
                                   prompt="the same lighthouse at night",
                                   size=size, extra_body={"strength": 0.7})
        assert _decode(r.data[0].b64_json).size == (512, 512)

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

    tag = family
    rep.check(f"{tag} generations b64 n=2", gen_b64)
    rep.check(f"{tag} generations url jpeg", gen_url)
    rep.check(f"{tag} stream partial_images=2", stream)
    rep.check(f"{tag} edits", edit)
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


def measure(base: str, out: dict) -> None:
    from openai import OpenAI

    client = OpenAI(base_url=base + "/v1", api_key="none", timeout=900)
    # Warm both models and their kernels at the measured shapes.
    client.images.generate(prompt=PROMPT, size="1024x1024")
    warm = {}
    _chat_stream(base, 64, warm)
    alone = {}
    _chat_stream(base, 512, alone)
    beside = {}
    job = threading.Thread(target=lambda: [client.images.generate(
        prompt=PROMPT, size="1024x1024", n=4, extra_body={"seed": i})
        for i in range(3)])
    job.start()
    time.sleep(1.0)
    _chat_stream(base, 512, beside)
    job.join()
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
        chat_beside_image_tps=beside["tokens"] / beside["decode_s"],
        chat_alone_tokens=alone["tokens"], chat_beside_tokens=beside["tokens"],
        stream_first_byte_s=plain[0], stream_total_s=plain[2],
        stream_partial1_first_byte_s=partial[0],
        stream_partial1_final_s=partial[1], stream_partial1_total_s=partial[2])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--family", choices=("z", "qwen", "both"), default="both")
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), dest=k, default=v)
    ap.add_argument("--with-chat", action="store_true",
                    help="keep a chat stream decoding while each family runs")
    ap.add_argument("--shared-encoder", action="store_true",
                    help="serve the Z-Image encoder as a chat model and share it")
    ap.add_argument("--memguard", action="store_true",
                    help="run memguard.py beside the server")
    ap.add_argument("--measure", action="store_true",
                    help="time chat decode beside an image job and the first "
                         "byte of a streamed image, with Z-Image only")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()
    if a.measure:
        a.family = "z"
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
    try:
        with ServerProc(["--config", cfg_path], log_path=log_path, port=port,
                        python=a.python) as srv:
            srv.wait_ready(timeout=300)
            base = srv.base_url
            if a.measure:
                m: dict = {}
                rep.check("measurements", lambda: measure(base, m) or json.dumps(
                    {k: round(v, 3) if isinstance(v, float) else v
                     for k, v in m.items()}))
                result["measurements"] = m
            else:
                for family, model in (("z", "z-image-turbo"), ("qwen", "qwen-image")):
                    if a.family in (family, "both"):
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
