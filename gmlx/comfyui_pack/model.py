# SPDX-License-Identifier: Apache-2.0
"""A stock ComfyUI MODEL whose DiT forward runs on the gmlx server.

The loader fetches the model's manifest, builds a state dict of empty
tensors on the meta device, lets ComfyUI's own ``model_config_from_unet``
detect the model, checks the class against the server's answer, and builds
the model on the meta device without loading weights. Its
``diffusion_model.forward`` then becomes one step contract call, so
``unet_config`` and the LoRA key maps are those of any other loader of the
same file.

The forward takes what ComfyUI hands its DiT and the server, under the
``comfy`` convention, returns what ComfyUI expects back.
"""

from __future__ import annotations

import base64
import collections
import logging
import time

import numpy as np
import torch

import comfy.model_detection
import comfy.model_management
import comfy.patcher_extension
import comfy.utils

from .client import (GmlxError, GmlxInterrupted, GmlxMissing, GmlxRefusal, StepClient,
                     set_interrupt_check, tensor_hash)
from .patcher import CPU, GmlxBinding, GmlxModelPatcher

log = logging.getLogger("gmlx")

# Cancel in ComfyUI stops any wait on the server: a step, a load, an encode,
# a VAE call or the start of the server.
set_interrupt_check(comfy.model_management.throw_exception_if_processing_interrupted)

_DTYPES = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16,
           "quant": torch.bfloat16}

# transformer_options keys and how the forward treats them.
FORWARDED = {"sample_sigmas", "sigmas", "cond_or_uncond", "uuids", "qwen_image21_cache"}
REFUSED = {"patches", "patches_replace", "rope_options"}
# Keys ComfyUI sets on every call that carry no behavior inside the forward.
BENIGN = {"prefetch_dynamic_vbars", "wrappers", "callbacks", "hooks", "transformer_index",
          "block_index", "model_dtype", "sampler_calc_cond_batch_function", "total_blocks",
          "block_type", "activations_shape_list", "patches_point", "optimized_attention_override"}
# DiT inputs the step contract carries. Any other input with a value is refused.
INPUTS = {"num_tokens", "attention_mask", "ref_latents", "image_slots", "ref_latents_method"}
# Stock nodes whose effect shows only inside the forward, named by the walk-back.
IN_FORWARD_CLASSES = {"PerturbedAttentionGuidance", "SkipLayerGuidanceDiT", "SkipLayerGuidanceSD3",
                      "SkipLayerGuidanceDiTSimple", "ControlNetApply", "ControlNetApplyAdvanced",
                      "ControlNetApplySD3", "SetUnionControlNetType", "QwenImageDiffsynthControlnet",
                      "TomePatchModel", "FreeU", "FreeU_V2", "ScaleROPE"}

LAST_PROMPT: dict = {}
STRICT_KEYS = False


def manifest_state_dict(manifest: dict, inline: bool = True) -> dict:
    """Empty tensors on the meta device by manifest name, plus the few
    tensors ComfyUI's detection reads by value."""
    sd = {}
    for t in manifest["tensors"]:
        sd[t["name"]] = torch.empty(t["shape"], dtype=_DTYPES.get(t["dtype"], torch.bfloat16),
                                    device="meta")
    if inline:
        for name, v in (manifest.get("inline") or {}).items():
            a = np.frombuffer(base64.b64decode(v["b64"]), dtype="<f4").reshape(v["shape"])
            sd[name] = torch.from_numpy(a.copy())
    return sd


def build_weightless(manifest: dict, model_class: str):
    """``(model, model_config)`` built on the meta device from a manifest,
    after checking that ComfyUI detects ``model_class``."""
    sd = manifest_state_dict(manifest)
    prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
    stripped = comfy.utils.state_dict_prefix_replace(sd, {prefix: ""}, filter_keys=True)
    if stripped:
        sd = stripped
    cfg = comfy.model_detection.model_config_from_unet(sd, "")
    if cfg is None:
        raise GmlxError("gmlx: ComfyUI detects no model in the server's manifest; this "
                        "ComfyUI may be older than the model")
    detected = type(cfg).__name__
    if detected != model_class:
        raise GmlxError(f"gmlx: ComfyUI detects {detected} where the server serves "
                        f"{model_class}; update ComfyUI or the gmlx server")
    dtypes = list(cfg.supported_inference_dtypes)
    dtype = torch.float32 if torch.float32 in dtypes else dtypes[0]
    cfg.set_inference_dtype(dtype, None, device=CPU)
    model = cfg.get_model(sd, "", device=torch.device("meta"))
    model.device = CPU
    return model, cfg


def walk_back(prompt: dict, node_id: str, classes: set) -> list:
    """Nodes of ``classes`` upstream of ``node_id`` in a submitted graph."""
    seen, stack, hits = set(), [str(node_id)], []
    while stack:
        n = stack.pop()
        if n in seen or n not in prompt:
            continue
        seen.add(n)
        if prompt[n].get("class_type") in classes:
            hits.append(n)
        for v in (prompt[n].get("inputs") or {}).values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                stack.append(v[0])
    return sorted(hits, key=lambda s: (len(s), s))


def register_prompt_handler() -> bool:
    """Keep the last submitted graph, so a forward-time refusal can name the
    node upstream of the sampler that caused it."""
    try:
        import server
        inst = getattr(server.PromptServer, "instance", None)
        if inst is None:
            return False

        def on_prompt(json_data):
            LAST_PROMPT.clear()
            LAST_PROMPT.update(json_data if isinstance(json_data, dict) else {})
            return json_data

        inst.add_on_prompt_handler(on_prompt)
        return True
    except Exception:  # noqa: BLE001: best effort; the refusal still names the cause
        return False


def _upstream_names() -> list:
    try:
        import server
        sid = getattr(server.PromptServer.instance, "last_node_id", None)
    except Exception:  # noqa: BLE001
        return []
    prompt = LAST_PROMPT.get("prompt") or {}
    if sid is None or not prompt:
        return []
    return [f"{n} ({prompt[n]['class_type']})" for n in walk_back(prompt, sid, IN_FORWARD_CLASSES)]


def _refuse(what: str):
    hits = _upstream_names()
    where = f"; upstream node(s): {', '.join(hits)}" if hits else ""
    raise GmlxRefusal(f"gmlx: {what} reached the DiT forward and is not supported on a "
                      f"gmlx model{where}")


class RemoteForward:
    """Replaces ``diffusion_model.forward`` with one ``/step`` call."""

    def __init__(self, base_model, binding: GmlxBinding):
        self.base_model = base_model
        self.b = binding
        self.warned: set = set()
        self.memo: dict = {}
        self.calls = collections.deque(maxlen=512)

    def _check(self, control, to: dict, kwargs: dict):
        if control is not None:
            _refuse("ControlNet control")
        for k in sorted(REFUSED):
            if to.get(k):
                _refuse(f"transformer_options[{k!r}]")
        if comfy.patcher_extension.get_all_wrappers(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, to):
            _refuse("a DIFFUSION_MODEL wrapper")
        for k in sorted(set(to) - FORWARDED - REFUSED - BENIGN):
            if STRICT_KEYS:
                _refuse(f"transformer_options[{k!r}]")
            if k not in self.warned:
                self.warned.add(k)
                log.warning("gmlx: transformer_options[%r] is not read by a gmlx model", k)
        for k, v in kwargs.items():
            if k in INPUTS or v is None or (isinstance(v, (list, tuple, dict)) and not v):
                continue
            _refuse(f"the DiT input {k!r}")
        if kwargs.get("ref_latents") and not self.b.edit_capable:
            _refuse(f"reference latents on {self.b.model_id!r}, which takes none")
        cache = to.get("qwen_image21_cache") or {}
        if cache.get("dtype", "default") not in ("default", None):
            raise GmlxRefusal(f"gmlx: a {cache['dtype']} prefix cache is not supported on a "
                              "gmlx model; use dtype default")
        return cache.get("device") != "off"

    def _hash(self, key, t: torch.Tensor, held: dict) -> str:
        h = self.memo.get(key) if key is not None else None
        if h is None:
            h = tensor_hash(t)
            if key is not None:
                self.memo[key] = h
        held[h] = t
        return h

    def rows(self, x, context, to: dict, kwargs: dict, run, prefix_cache: bool):
        """The step's rows and the tensors they name by hash."""
        B = x.shape[0]
        if context is None:
            raise GmlxError("gmlx: the DiT forward got no conditioning")
        if context.shape[0] == 1 and B > 1:
            context = context.expand(B, *context.shape[1:])
        if context.shape[0] != B:
            raise GmlxError(f"gmlx: {context.shape[0]} conditioning rows for {B} latents")
        coru = list(to.get("cond_or_uncond") or [0])
        uuids = list(to.get("uuids") or [None] * len(coru))
        per = B // len(coru) if B % len(coru) == 0 else B
        refs = list(kwargs.get("ref_latents") or [])
        slots = kwargs.get("image_slots")
        held: dict = {}
        rows = []
        for i in range(B):
            ci = min(i // per, len(coru) - 1)
            u = uuids[ci] if ci < len(uuids) else None
            base = None if (u is None or run is None) else (run, str(u), i % per)
            c = context[i:i + 1].float()
            # Every row of the context goes, as the stock DiTs of both families read
            # neither the attention mask nor num_tokens: a padded stock conditioning
            # keeps its padding rows there too.
            row = {"cond": self._hash(None if base is None else base + ("cond", tuple(c.shape)),
                                      c, held),
                   "num_tokens": int(c.shape[1])}
            if refs:
                hs = []
                for j, r in enumerate(refs):
                    rr = (r[i:i + 1] if r.shape[0] == B else r[:1]).float()
                    key = None if base is None else base + ("ref", j, tuple(rr.shape))
                    hs.append(self._hash(key, rr, held))
                row["ref_latents"] = hs
            if slots:
                row["image_slots"] = [int(s) for s in slots]
            if not prefix_cache:
                row["prefix_cache"] = False
            rows.append(row)
        return rows, held

    def _send(self, x, timesteps, fields):
        client: StepClient = self.b.client

        def stop():
            try:
                client.interrupt(fields.get("run"))
            except GmlxError as e:
                log.warning("gmlx: the interrupt did not reach the server: %s", e)

        return client.step(x, timesteps, fields, on_interrupt=stop)

    def __call__(self, x, timesteps, context=None, control=None, transformer_options={}, **kwargs):
        t0 = time.perf_counter()
        to = transformer_options or {}
        prefix_cache = self._check(control, to, kwargs)
        # A latent format with a frame axis, such as Krea 2's, gives one frame.
        framed = x.ndim == 5 and x.shape[2] == 1
        if framed:
            x = x[:, :, 0]
        if x.ndim != 4:
            raise GmlxError(f"gmlx: latents of shape {tuple(x.shape)} are not (B, C, h, w)")
        patcher = getattr(self.base_model, "current_patcher", None)
        run = getattr(patcher, "gmlx_run", None)
        table = getattr(patcher, "gmlx_table_id", None)
        if patcher is not None and getattr(patcher, "gmlx", None) is not None:
            table = patcher.sync_patches()
        rows, held = self.rows(x, context, to, kwargs, run, prefix_cache)
        client = self.b.client
        for h, t in held.items():
            client.ensure_tensor(t, h)
        fields = {"model": self.b.model_id, "run": run, "convention": "comfy",
                  "patch_table": table, "rows": rows}
        xs = x.float()
        ts = timesteps.float().reshape(-1)
        t1 = time.perf_counter()
        try:
            out, hdrs = self._send(xs, ts, fields)
        except GmlxMissing as e:
            client.stats["missing_retries"] += 1
            missing = set(e.missing)
            for h in sorted(missing & set(held)):
                client.put_tensor(h, held[h])
            if table is not None and table in missing:
                table = patcher.sync_patches(force=True)
                fields["patch_table"] = table
            out, hdrs = self._send(xs, ts, fields)
        except GmlxInterrupted:
            comfy.model_management.throw_exception_if_processing_interrupted()
            raise
        t2 = time.perf_counter()
        out = out.to(dtype=x.dtype, device=x.device)
        t3 = time.perf_counter()
        self.calls.append({"rows": len(rows), "wall_ms": (t3 - t0) * 1000,
                           "prepare_ms": (t1 - t0) * 1000, "http_ms": (t2 - t1) * 1000,
                           "server_ms": float(hdrs.get("x-gmlx-server-ms", "nan")),
                           "forward_ms": float(hdrs.get("x-gmlx-forward-ms", "nan")),
                           "prefix_cache": hdrs.get("x-gmlx-prefix-cache"),
                           "table": table, "run": run})
        return out.unsqueeze(2) if framed else out


def load_model(client: StepClient, model_id: str) -> GmlxModelPatcher:
    """The gmlx MODEL of ``model_id`` on ``client``'s server."""
    status = client.status(model_id)
    facts = status.get("comfy") or {}
    if not facts.get("model_class"):
        raise GmlxError(f"gmlx: the server gives no ComfyUI model class for {model_id!r}")
    manifest = client.manifest(model_id)
    model, _cfg = build_weightless(manifest, facts["model_class"])
    binding = GmlxBinding(client, model_id, status, int(manifest.get("count") or 0))
    model.diffusion_model.forward = RemoteForward(model, binding)
    patcher = GmlxModelPatcher(model, load_device=CPU, offload_device=CPU)
    patcher.gmlx = binding
    return patcher
