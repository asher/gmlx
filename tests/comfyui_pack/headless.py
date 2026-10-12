#!/usr/bin/env python3
"""Headless ComfyUI with the pack, against the stub server.

ComfyUI runs on the cpu device with a base directory of its own, so nothing
is written in the checkout: the pack is linked into that directory's
custom_nodes and is the only custom node loaded. Each case queues a graph
through the HTTP API, follows the websocket and reads the stub's log.

Usage: headless.py --comfy DIR --out DIR --label NAME [--step-delay-ms 40]
Writes headless-NAME.json and headless-NAME-comfyui.log in --out.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import uuid

import aiohttp
import numpy as np

from support import PACK, Stub, manifest, write_zimage_lora

# API-format copies of the example workflows, which the UI files convert to.
WORKFLOWS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflows")
LINK = "comfyui-gmlx"
LORA = "z-image-turbo-style.safetensors"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def workflow(name: str) -> dict:
    """An example workflow, whose loaders keep the default address the
    templates carry, so they reach the stub through GMLX_BASE_URL."""
    with open(os.path.join(WORKFLOWS, name)) as f:
        wf = json.load(f)
    for node in wf.values():
        node.pop("_meta", None)
        if node["class_type"].startswith("Gmlx"):
            assert node["inputs"]["server_url"] == "http://127.0.0.1:8080", name
            node["inputs"]["start_server"] = False
    return wf


def zimage(seed=42, steps=4, size=512) -> dict:
    wf = workflow("z-image-turbo-lora-t2i.json")
    wf["4"]["inputs"]["lora_name"] = LORA
    wf["7"]["inputs"].update(width=size, height=size)
    wf["8"]["inputs"].update(seed=seed, steps=steps)
    wf["11"] = {"class_type": "SaveLatent", "inputs": {"samples": ["8", 0],
                                                       "filename_prefix": f"latents/zimage-{seed}"}}
    return wf


def insert_model_node(wf: dict, node_id: str, class_type: str, inputs: dict, sampler="8",
                      source=("4", 0)) -> dict:
    wf = copy.deepcopy(wf)
    wf[node_id] = {"class_type": class_type, "inputs": {"model": list(source), **inputs}}
    wf[sampler]["inputs"]["model"] = [node_id, 0]
    return wf


def flow_timesteps(shift: float, steps: int) -> list:
    """ComfyUI's simple schedule on a discrete flow model with multiplier 1:
    the timesteps the DiT sees."""
    t = np.arange(1, 1001, dtype=np.float32) / np.float32(1000)
    sig = (shift * t / (1 + (shift - 1) * t)).astype(np.float32)
    ss = len(sig) / steps
    return [float(sig[-(1 + int(x * ss))]) for x in range(steps)]


class Comfy:
    def __init__(self, comfy_dir: str, work: str, env: dict, log_path: str):
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.log_path = log_path
        self.log = open(log_path, "w")
        cmd = [sys.executable, "main.py", "--cpu", "--listen", "127.0.0.1", "--port", str(self.port),
               "--base-directory", work, "--database-url",
               "sqlite:///" + os.path.join(work, "user", "comfyui.db"),
               "--disable-all-custom-nodes", "--whitelist-custom-nodes", LINK,
               "--preview-method", "latent2rgb", "--disable-auto-launch"]
        self.proc = subprocess.Popen(cmd, cwd=comfy_dir, env=env, stdout=self.log,
                                     stderr=subprocess.STDOUT)

    def tail(self, n=40) -> str:
        with open(self.log_path) as f:
            return "".join(f.readlines()[-n:])

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


class Session:
    def __init__(self, comfy: Comfy, http: aiohttp.ClientSession):
        self.c, self.http = comfy, http
        self.client_id = uuid.uuid4().hex
        self.ws = None

    async def ready(self, timeout=180.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.c.proc.poll() is not None:
                raise RuntimeError("ComfyUI exited:\n" + self.c.tail())
            try:
                async with self.http.get(self.c.url + "/object_info/GmlxModelLoader") as r:
                    if r.status == 200 and "GmlxModelLoader" in await r.json():
                        break
            except aiohttp.ClientError:
                pass
            await asyncio.sleep(0.5)
        else:
            raise RuntimeError("ComfyUI did not list the gmlx nodes:\n" + self.c.tail())
        self.ws = await self.http.ws_connect(f"{self.c.url.replace('http', 'ws')}/ws?clientId={self.client_id}")

    async def run(self, prompt: dict, on_progress=None, timeout=300.0) -> dict:
        async with self.http.post(self.c.url + "/prompt",
                                  json={"prompt": prompt, "client_id": self.client_id}) as r:
            doc = await r.json()
            if r.status != 200:
                return {"status": "rejected", "error": doc}
        pid = doc["prompt_id"]
        res = {"prompt_id": pid, "progress": 0, "previews": 0, "messages": [], "status": None}
        t0 = time.monotonic()
        while res["status"] is None:
            msg = await self.ws.receive(timeout=timeout)
            if msg.type == aiohttp.WSMsgType.BINARY:
                res["previews"] += 1
                continue
            if msg.type != aiohttp.WSMsgType.TEXT:
                raise RuntimeError(f"websocket closed: {msg.type}")
            d = json.loads(msg.data)
            data = d.get("data") or {}
            if data.get("prompt_id") not in (pid, None):
                continue
            kind = d.get("type")
            res["messages"].append(kind)
            if kind == "progress":
                res["progress"] += 1
                if on_progress is not None:
                    await on_progress(res["progress"], data)
            elif kind == "execution_success":
                res["status"] = "success"
            elif kind == "execution_error":
                res["status"] = "error"
                res["error"] = {k: data.get(k) for k in ("node_id", "node_type", "exception_type",
                                                          "exception_message")}
            elif kind == "execution_interrupted":
                res["status"] = "interrupted"
                res["interrupted_at"] = time.monotonic()
        res["wall_s"] = round(time.monotonic() - t0, 3)
        res["messages"] = sorted(set(res["messages"]))
        return res

    async def interrupt(self):
        async with self.http.post(self.c.url + "/interrupt", json={}) as r:
            await r.read()


def latent_sha(work: str, prefix: str) -> str | None:
    from safetensors.numpy import load_file
    d = os.path.join(work, "output", os.path.dirname(prefix))
    stem = os.path.basename(prefix)
    files = sorted(f for f in os.listdir(d) if f.startswith(stem) and f.endswith(".latent")) \
        if os.path.isdir(d) else []
    if not files:
        return None
    t = load_file(os.path.join(d, files[-1]))["latent_tensor"]
    return hashlib.sha256(str(t.dtype).encode() + json.dumps(list(t.shape)).encode() + t.tobytes()).hexdigest()


def kinds(log: list, kind: str) -> list:
    return [e for e in log if e["kind"] == kind]


class Cases:
    def __init__(self, s: Session, stub: Stub, stub2: Stub, work: str, targets: int):
        self.s, self.stub, self.stub2, self.work, self.targets = s, stub, stub2, work, targets
        self.results: list = []

    def check(self, name: str, ok: bool, detail):
        self.results[-1]["checks"].append({"check": name, "ok": bool(ok), "detail": detail})

    async def case(self, name: str, fn):
        self.results.append({"case": name, "checks": []})
        self.stub.reset()
        try:
            await fn()
        except Exception:  # noqa: BLE001: a broken case is recorded and the rest still run
            self.check("ran", False, traceback.format_exc()[-2000:])
        self.results[-1]["ok"] = all(c["ok"] for c in self.results[-1]["checks"])

    async def t2i(self):
        steps = 4
        r = await self.s.run(zimage(seed=42, steps=steps))
        log = self.stub.log()
        self.check("success", r["status"] == "success", r.get("error"))
        self.check("progress per step", r["progress"] == steps, r["progress"])
        self.check("latent previews", r["previews"] >= 1, r["previews"])
        self.check("one run", [e["kind"] for e in log if e["kind"].startswith("run_")] == ["run_begin", "run_end"],
                   [e["kind"] for e in log if e["kind"].startswith("run_")])
        self.check("steps", len(kinds(log, "step")) == steps, len(kinds(log, "step")))
        self.check("one table put", len(kinds(log, "put_table")) == 1, len(kinds(log, "put_table")))
        self.check("adapter uploads", len(kinds(log, "put_adapter")) == 2 * self.targets,
                   len(kinds(log, "put_adapter")))
        self.check("steps name the table", {e["patch_table"] for e in kinds(log, "step")}
                   == {kinds(log, "put_table")[0]["id"] if kinds(log, "put_table") else None}, None)
        self.check("decoded", len(kinds(log, "vae_decode")) == 1, len(kinds(log, "vae_decode")))
        self.latent_t2i = latent_sha(self.work, "latents/zimage-42")
        self.check("latent saved", self.latent_t2i is not None, self.latent_t2i)
        self.timesteps_default = [e["timestep"][0] for e in kinds(log, "step")]
        want = flow_timesteps(3.0, steps)
        self.check("timesteps at shift 3", np.allclose(self.timesteps_default, want, atol=1e-5),
                   {"got": self.timesteps_default, "want": want})
        self.results[-1]["latent_sha256"] = self.latent_t2i
        self.results[-1]["wall_s"] = r["wall_s"]

    async def rerun(self):
        r = await self.s.run(zimage(seed=43))
        log = self.stub.log()
        self.check("success", r["status"] == "success", r.get("error"))
        self.check("no table put", len(kinds(log, "put_table")) == 0, len(kinds(log, "put_table")))
        self.check("no adapter upload", len(kinds(log, "put_adapter")) == 0, len(kinds(log, "put_adapter")))
        self.check("no encode (cached)", len(kinds(log, "encode")) == 0, len(kinds(log, "encode")))
        self.check("one run", len(kinds(log, "run_begin")) == 1 and len(kinds(log, "run_end")) == 1, None)

    async def strength_change(self):
        wf = zimage(seed=44)
        wf["4"]["inputs"]["strength_model"] = 0.5
        r = await self.s.run(wf)
        log = self.stub.log()
        self.check("success", r["status"] == "success", r.get("error"))
        self.check("one new table", len(kinds(log, "put_table")) == 1, len(kinds(log, "put_table")))
        self.check("adapters not uploaded again", len(kinds(log, "put_adapter")) == 0,
                   len(kinds(log, "put_adapter")))
        self.check("strength 0.5", [e["strengths"] for e in kinds(log, "put_table")] == [[0.5]],
                   [e["strengths"] for e in kinds(log, "put_table")])

    async def server_restart(self):
        self.stub.forget()
        self.stub.reset()
        r = await self.s.run(zimage(seed=45))
        log = self.stub.log()
        self.check("success", r["status"] == "success", r.get("error"))
        self.check("a 409 was answered", len(kinds(log, "step_missing")) >= 1, len(kinds(log, "step_missing")))
        self.check("conditioning uploaded again", len(kinds(log, "put_tensor")) >= 1, len(kinds(log, "put_tensor")))
        self.check("table defined again", len(kinds(log, "put_table")) == 1, len(kinds(log, "put_table")))
        self.check("adapters uploaded again", len(kinds(log, "put_adapter")) == 2 * self.targets,
                   len(kinds(log, "put_adapter")))
        self.check("every step served", len(kinds(log, "step")) == 4, len(kinds(log, "step")))

    async def auraflow(self):
        steps = 4
        wf = insert_model_node(zimage(seed=42, steps=steps), "20", "ModelSamplingAuraFlow", {"shift": 6.0})
        r = await self.s.run(wf)
        log = self.stub.log()
        got = [e["timestep"][0] for e in kinds(log, "step")]
        want = flow_timesteps(6.0, steps)
        self.check("success", r["status"] == "success", r.get("error"))
        self.check("timesteps at shift 6", np.allclose(got, want, atol=1e-5), {"got": got, "want": want})
        self.check("no table put (object patch only)", len(kinds(log, "put_table")) == 0,
                   len(kinds(log, "put_table")))

    async def interrupt(self):
        sent = {}

        async def on_progress(n, data):
            if n == 2 and "t" not in sent:
                sent["t"] = time.monotonic()
                await self.s.interrupt()
        r = await self.s.run(zimage(seed=46, steps=40), on_progress=on_progress)
        await asyncio.sleep(0.5)
        log = self.stub.log()
        self.check("interrupted", r["status"] == "interrupted", r["status"])
        lat = (r.get("interrupted_at", 0) - sent.get("t", 0)) if sent else None
        self.check("stopped within a second", lat is not None and 0 <= lat < 1.0, lat)
        self.check("server told", len(kinds(log, "interrupt")) >= 1, len(kinds(log, "interrupt")))
        self.check("run ended", len(kinds(log, "run_end")) == 1, len(kinds(log, "run_end")))
        self.check("stopped early", len(kinds(log, "step")) < 40, len(kinds(log, "step")))
        self.results[-1]["step_interrupted"] = len(kinds(log, "step_interrupted"))
        r2 = await self.s.run(zimage(seed=47))
        self.check("next prompt runs", r2["status"] == "success", r2.get("error"))

    async def refusal(self, class_type, inputs, at, words):
        wf = insert_model_node(zimage(seed=48), "30", class_type, inputs)
        r = await self.s.run(wf)
        err = r.get("error") or {}
        self.check("error", r["status"] == "error", r["status"])
        self.check(f"on node {at}", str(err.get("node_id")) == at, err.get("node_id"))
        msg = str(err.get("exception_message"))
        self.check("names it", all(w in msg for w in words), msg)
        n = len(kinds(self.stub.log(), "step"))
        self.check("no step served" if at == "30" else "stopped at the first step", n == 0 if at == "30" else n <= 1, n)

    async def version(self):
        wf = zimage(seed=49)
        wf["1"]["inputs"]["server_url"] = self.stub2.url
        r = await self.s.run(wf)
        err = r.get("error") or {}
        self.check("error on the loader", r["status"] == "error" and str(err.get("node_id")) == "1", err)
        self.check("says to restart ComfyUI", "Restart ComfyUI"
                   in str(err.get("exception_message")), err.get("exception_message"))

    async def qwen_edit(self):
        wf = workflow("qwen-image-2.1-edit-two-references.json")
        for loader in ("1", "2", "3"):
            wf[loader]["inputs"]["model_name"] = "qwen-image-2.1"
        wf["6"]["inputs"]["resolution"] = 512
        wf["7"]["inputs"]["steps"] = 3
        wf["10"] = {"class_type": "SaveLatent", "inputs": {"samples": ["7", 0],
                                                           "filename_prefix": "latents/qwen-edit"}}
        r = await self.s.run(wf)
        log = self.stub.log()
        self.check("success", r["status"] == "success", r.get("error"))
        enc = kinds(log, "encode")
        self.check("two encodes with two images", [e["images"] for e in enc] == [2, 2], [e["images"] for e in enc])
        self.check("two reference latents from the VAE", len(kinds(log, "vae_encode")) == 2,
                   len(kinds(log, "vae_encode")))
        steps = kinds(log, "step")
        self.check("rows carry the references", {tuple(e["refs"]) for e in steps} == {(2, 2)},
                   [e["refs"] for e in steps])
        self.check("rows carry the image slots", {json.dumps(e["image_slots"]) for e in steps}
                   == {json.dumps([[2, 5], [2, 5]])}, [e["image_slots"] for e in steps][:1])
        self.check("rgba decode", [e["alpha"] for e in kinds(log, "vae_decode")] == [True],
                   [e["alpha"] for e in kinds(log, "vae_decode")])
        self.latent_qwen = latent_sha(self.work, "latents/qwen-edit")
        self.check("latent saved", self.latent_qwen is not None, self.latent_qwen)
        self.results[-1]["latent_sha256"] = self.latent_qwen
        self.results[-1]["x_shape"] = steps[0]["x_shape"] if steps else None

    async def turbo_sigmas(self):
        wf = {
            "1": {"class_type": "GmlxModelLoader", "inputs": {"model_name": "qwen-image-2.1-turbo",
                                                              "server_url": "", "start_server": False}},
            "2": {"class_type": "GmlxCLIPLoader", "inputs": {"model_name": "qwen-image-2.1-turbo",
                                                             "server_url": "", "start_server": False}},
            "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "a fox"}},
            "4": {"class_type": "GmlxSigmas", "inputs": {"model": ["1", 0], "steps": 0, "denoise": 1.0}},
            "5": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "6": {"class_type": "EmptySD3LatentImage",
                  "inputs": {"width": 256, "height": 256, "batch_size": 1}},
            "7": {"class_type": "SamplerCustom", "inputs": {"model": ["1", 0], "add_noise": True, "noise_seed": 5,
                                                            "cfg": ["4", 1], "positive": ["3", 0],
                                                            "negative": ["3", 0], "sampler": ["5", 0],
                                                            "sigmas": ["4", 0], "latent_image": ["6", 0]}},
            "8": {"class_type": "SaveLatent", "inputs": {"samples": ["7", 0], "filename_prefix": "latents/turbo"}},
        }
        r = await self.s.run(wf)
        log = self.stub.log()
        got = [e["timestep"][0] for e in kinds(log, "step")]
        want = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]
        self.check("success", r["status"] == "success", r.get("error"))
        self.check("samples on the fixed sigmas", np.allclose(got, want, atol=1e-5), got)


async def amain(a) -> dict:
    os.makedirs(a.out, exist_ok=True)
    work = tempfile.mkdtemp(prefix="gmlx-headless-")
    os.makedirs(os.path.join(work, "custom_nodes"))
    os.symlink(PACK, os.path.join(work, "custom_nodes", LINK))
    for d in ("models/loras", "input", "output", "user", "temp"):
        os.makedirs(os.path.join(work, d), exist_ok=True)
    targets = write_zimage_lora(os.path.join(work, "models", "loras", LORA), manifest("z_image"))
    from PIL import Image
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (768, 512, 3), dtype=np.uint8)).save(
        os.path.join(work, "input", "reference-1.png"))
    Image.fromarray(rng.integers(0, 255, (512, 512, 4), dtype=np.uint8), "RGBA").save(
        os.path.join(work, "input", "reference-2.png"))
    stub = Stub("--step-delay-ms", str(a.step_delay_ms))
    stub2 = Stub("--contract", "2.0")
    env = dict(os.environ, GMLX_BASE_URL=stub.url, GMLX_COMFY_SETTINGS=os.path.join(work, "none.json"),
               PYTHONUNBUFFERED="1")
    env.pop("GMLX_API_KEY", None)
    comfy = Comfy(a.comfy, work, env, os.path.join(a.out, f"headless-{a.label}-comfyui.log"))
    out = {"label": a.label, "comfy": a.comfy, "cases": [], "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        async with aiohttp.ClientSession() as http:
            s = Session(comfy, http)
            t0 = time.monotonic()
            await s.ready()
            out["startup_s"] = round(time.monotonic() - t0, 2)
            async with http.get(comfy.url + "/system_stats") as r:
                out["system"] = (await r.json()).get("system", {})
            c = Cases(s, stub, stub2, work, targets)
            await c.case("zimage t2i with a stock LoraLoader", c.t2i)
            await c.case("same graph, new seed: nothing uploaded again", c.rerun)
            await c.case("new LoRA strength: one new table", c.strength_change)
            await c.case("server dropped everything: 409 and upload again", c.server_restart)
            await c.case("ModelSamplingAuraFlow changes the timesteps", c.auraflow)
            await c.case("interrupt", c.interrupt)
            await c.case("ScaleROPE refused on its node",
                         lambda: c.refusal("ScaleROPE", {"scale_x": 2.0, "shift_x": 0.0, "scale_y": 1.0,
                                                         "shift_y": 0.0, "scale_t": 1.0, "shift_t": 0.0},
                                           "30", ["RoPE"]))
            await c.case("PatchModelAddDownscale refused on its node",
                         lambda: c.refusal("PatchModelAddDownscale",
                                           {"block_number": 3, "downscale_factor": 2.0, "start_percent": 0.0,
                                            "end_percent": 0.35, "downscale_after_skip": True,
                                            "downscale_method": "bicubic", "upscale_method": "bicubic"},
                                           "30", ["model patch"]))
            await c.case("PAG refused at the sampler, naming PerturbedAttentionGuidance",
                         lambda: c.refusal("PerturbedAttentionGuidance", {"scale": 3.0}, "8",
                                           ["patches_replace", "PerturbedAttentionGuidance"]))
            await c.case("version mismatch names the side to upgrade", c.version)
            await c.case("qwen-image-2.1 edit with two references", c.qwen_edit)
            await c.case("fixed sigmas through GmlxSigmas and SamplerCustom", c.turbo_sigmas)
            out["cases"] = c.results
    except Exception:  # noqa: BLE001
        out["fatal"] = traceback.format_exc()
    finally:
        comfy.close()
        stub.close()
        stub2.close()
        shutil.rmtree(work, ignore_errors=True)
    out["passed"] = sum(1 for x in out["cases"] if x.get("ok"))
    out["failed"] = [x["case"] for x in out["cases"] if not x.get("ok")]
    with open(os.path.join(a.out, f"headless-{a.label}.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--step-delay-ms", type=float, default=40.0)
    a = ap.parse_args()
    out = asyncio.run(amain(a))
    for c in out["cases"]:
        print(("ok   " if c.get("ok") else "FAIL ") + c["case"])
        for k in c["checks"]:
            if not k["ok"]:
                print(f"     {k['check']}: {str(k['detail'])[:600]}")
    if out.get("fatal"):
        print(out["fatal"])
    sys.exit(0 if not out["failed"] and not out.get("fatal") else 1)


if __name__ == "__main__":
    main()
