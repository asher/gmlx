#!/usr/bin/env python3
"""End-to-end test of the image step routes with the reference client.

Launches the real server (``python -m gmlx.serve.server --config``) with
the image models of both families, then drives the routes under
``/internal/image-steps/v1`` with ``gmlx.serve.image_steps_client``: the
handshake, the status and manifest of each model, a prompt encoded by the
server (with two reference images for Qwen-Image 2.1), and a sampling run
of one step request per sigma, with Euler and guidance in numpy on the
client. After the server stops, the DiT loads in this process and
``sample`` runs from the same noise, sigmas and conditioning tensors; the
final latents must be equal bit for bit. Writes a Markdown and a JSON
report with the median transport time per step.

    python tests/e2e/run_image_steps_e2e.py --out DIR [--family z|qwen|both]

Not ``test_``-prefixed: it loads real models on the GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server_proc import ServerProc, free_port  # noqa: E402

_GGUF = os.path.expanduser(os.environ.get("KQUANT_TEST_GGUF_DIR", "~/llm/gguf"))
DEFAULTS = {
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
EDIT = "the two objects side by side on a wooden table"


def _config(a) -> dict:
    image_models = {}
    if a.family in ("z", "both"):
        image_models["z-image-turbo"] = {
            "path": a.z_dit, "encoder": a.z_encoder, "vae": a.z_vae}
    if a.family in ("qwen", "both"):
        image_models["qwen-image"] = {
            "path": a.q_dit, "encoder": a.q_encoder,
            "encoder_mmproj": a.q_mmproj, "vae": a.q_vae}
    return {"image_models": image_models,
            "server": {"defaults": {"image_model": next(iter(image_models))}}}


def _image(seed: int, size: int = 256):
    """A synthetic RGB reference image, u8 (H, W, 3)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    px = np.stack([(x * 255 // size), (y * 255 // size),
                   np.full_like(x, 60 + 120 * seed % 255)], -1)
    return (px + rng.integers(0, 20, px.shape)).clip(0, 255).astype(np.uint8)


def _sigmas(family: str, variant: str, sampling, steps: int, tokens: int):
    """The schedule the client samples on: the variant's fixed sigmas from
    the status, else the family's schedule of ``steps``."""
    from gmlx.image.registry import get_family

    if sampling and sampling.get("sigmas"):
        return np.asarray(sampling["sigmas"], np.float32)
    fam = get_family(family)
    return np.asarray(fam.schedule(steps, tokens=tokens, variant=variant),
                      np.float32)


class Report:
    def __init__(self):
        self.rows: list = []

    def check(self, name, fn):
        t0 = time.monotonic()
        try:
            detail = fn() or ""
            ok = True
        except Exception as e:  # noqa: BLE001 - every failure is a row
            detail, ok = f"{type(e).__name__}: {e}", False
        self.rows.append({"name": name, "ok": ok, "detail": str(detail),
                          "seconds": round(time.monotonic() - t0, 2)})
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}", flush=True)
        return ok


def run_family(rep, client, model: str, family: str, a, keep: dict) -> None:
    from gmlx.image.samplers import noise_from_seed

    st = client.status(model)
    rep.check(f"{family} status", lambda: (
        st["family"], st["variant"], st["comfy"]["model_class"],
        st["latent"]) if st["family"] else 1 / 0)
    man = client.manifest(model)
    rep.check(f"{family} manifest", lambda: f"{man['count']} tensors, arch "
              f"{man['arch']}, inline {sorted(man['inline'])}")
    lat = st["latent"]
    h = w = a.size
    shape = (1, lat["channels"], h // lat["stride"], w // lat["stride"])
    tokens = (h // (lat["stride"] * lat["patch"])) * (
        w // (lat["stride"] * lat["patch"]))
    sampling = st.get("sampling") or {}
    steps = a.steps
    cfg = float(sampling.get("cfg") or 1.0)
    sigmas = _sigmas(st["family"], st["variant"], sampling, steps, tokens)
    images = [_image(1), _image(2)] if family == "qwen" else []
    tensors: dict = {}

    def encode(text):
        fields, got = client.encode(model, text, images)
        tensors[fields["cond"]] = got["cond"]
        for i, hsh in enumerate(fields["ref_latents"]):
            tensors[hsh] = got[f"ref_{i}"]
        return fields

    enc = encode(EDIT if images else PROMPT)
    rep.check(f"{family} encode", lambda: f"cond {enc['cond'][:12]}, "
              f"{enc['num_tokens']} tokens, refs {len(enc['ref_latents'])}, "
              f"slots {len(enc['image_slots'])}")
    keys = ("cond", "num_tokens", "ref_latents", "image_slots")
    rows = [{k: enc[k] for k in keys}]
    neg_rows = None
    if cfg != 1.0:
        neg = encode("")
        neg_rows = [{k: neg[k] for k in keys}]
    noise = noise_from_seed(a.seed, shape)
    run = client.begin(model)
    first = len(client.records)
    t0 = time.monotonic()
    try:
        x = client.sample(model, noise, sigmas, rows, cfg=cfg,
                          neg_rows=neg_rows, run=run)
    finally:
        client.end(run)
    recs = client.records[first:]
    over = [r.overhead_ms for r in recs]
    rep.check(f"{family} sample through the routes", lambda: (
        f"{len(sigmas) - 1} steps {h}x{w} cfg {cfg} in "
        f"{time.monotonic() - t0:.1f} s; forward median "
        f"{statistics.median(r.forward_ms for r in recs):.1f} ms; transport "
        f"median {statistics.median(over):.2f} ms; prefix "
        f"{[r.prefix_cache for r in recs]}"))
    keep[family] = {"model": model, "x": x, "noise": noise, "sigmas": sigmas,
                    "tensors": tensors,
                    "cfg": cfg, "rows": rows, "neg_rows": neg_rows,
                    "size": (h, w), "transport_ms": over,
                    "forward_ms": [r.forward_ms for r in recs],
                    "prefix_cache": [r.prefix_cache for r in recs]}


def direct(rep, family: str, k: dict, dit: str) -> None:
    import mlx.core as mx

    from gmlx.image import api

    from gmlx.serve.image_steps import SLOT_FAMILIES, slots_from_wire

    pipe = api.load(dit, log=lambda *x: None)
    slotted = pipe.family.name in SLOT_FAMILIES

    def cond(row, tensors, key):
        # The wire's slots are ComfyUI's; the core takes its own layout.
        x = mx.array(tensors[row["cond"]][0])
        refs = tuple(mx.array(tensors[r][0]) for r in row["ref_latents"])
        slots = tuple(row["image_slots"])
        if refs and slotted:
            x, slots = slots_from_wire(x, slots,
                                       [z.shape[-2:] for z in refs])
        return api.Conditioning(x, key, ref_latents=refs, image_slots=slots)

    t = k["tensors"]
    c = cond(k["rows"][0], t, "direct")
    n = cond(k["neg_rows"][0], t, "direct-neg") if k["neg_rows"] else None
    h, w = k["size"]
    x = api.sample(pipe, c, neg=n, cfg=k["cfg"], height=h, width=w,
                   noise=mx.array(k["noise"]), sigmas=k["sigmas"])
    want = np.array(x)
    diff = float(np.abs(want - k["x"]).max())

    def same():
        if not np.array_equal(want, k["x"]):
            raise AssertionError(f"latents differ, max abs {diff:.3g}")
        return f"equal bit for bit, {want.shape}"

    rep.check(f"{family} latents equal direct sample", same)
    del pipe
    mx.clear_cache()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--family", choices=("z", "qwen", "both"), default="both")
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), dest=k, default=v)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=8,
                    help="steps for a variant without fixed sigmas")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import yaml

    from gmlx.serve.image_steps_client import StepsClient, http_transport

    cfg_path = os.path.join(a.out, "gmlx.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(_config(a), f, sort_keys=False)
    rep = Report()
    keep: dict = {}
    models = [(f, m) for f, m in (("z", "z-image-turbo"), ("qwen",
                                                            "qwen-image"))
              if a.family in (f, "both")]
    with ServerProc(["--config", cfg_path], port=free_port(),
                    log_path=os.path.join(a.out, "server.log"),
                    python=a.python) as srv:
        srv.wait_ready(timeout=300)
        client = StepsClient(http_transport(srv.base_url))
        caps = client.capabilities()
        rep.check("capabilities", lambda: f"contract {caps['contract']}, "
                  f"families {caps['families']}")
        for family, model in models:
            try:
                run_family(rep, client, model, family, a, keep)
            except Exception as e:  # noqa: BLE001 - a family failure is a row
                rep.rows.append({"name": f"{family} run", "ok": False,
                                 "detail": f"{type(e).__name__}: {e}",
                                 "seconds": 0})
                print(f"[FAIL] {family} run: {e}", flush=True)
    for family, model in models:
        if family in keep:
            dit = a.z_dit if family == "z" else a.q_dit
            try:
                direct(rep, family, keep[family], dit)
            except Exception as e:  # noqa: BLE001 - a failure is a row
                rep.rows.append({"name": f"{family} direct run", "ok": False,
                                 "detail": f"{type(e).__name__}: {e}",
                                 "seconds": 0})
    result = {"config": _config(a), "checks": rep.rows,
              "transport": {f: {"median_ms": statistics.median(
                  k["transport_ms"]), "per_step_ms": k["transport_ms"],
                  "forward_ms": k["forward_ms"],
                  "prefix_cache": k["prefix_cache"]}
                  for f, k in keep.items()}}
    ok = all(r["ok"] for r in rep.rows)
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(a.out, "report.md"), "w") as f:
        f.write("# Image step routes end to end\n\n| Check | Result | Seconds "
                "| Detail |\n|-------|--------|---------|--------|\n")
        for r in rep.rows:
            f.write(f"| {r['name']} | {'ok' if r['ok'] else 'FAIL'} | "
                    f"{r['seconds']} | {r['detail'].replace('|', '/')} |\n")
    print(f"RESULT: {sum(r['ok'] for r in rep.rows)}/{len(rep.rows)} checks "
          "passed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
