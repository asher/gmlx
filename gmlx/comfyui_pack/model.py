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
import functools
import logging
import re
import time

import numpy as np
import torch

import comfy.model_detection
import comfy.model_management
import comfy.patcher_extension
import comfy.supported_models
import comfy.utils

from .client import (GmlxError, GmlxInterrupted, GmlxMissing, GmlxRefusal, StepClient,
                     set_interrupt_check, tensor_hash)
from .clip import TYPED, TypedPrompt
from .compat import comfyui_version
from .moss_image import PACK_MODELS
from .patcher import CPU, KV_METHOD, KV_PATCHES, GmlxBinding, GmlxModelPatcher

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
                      "TomePatchModel", "FreeU", "FreeU_V2", "ScaleROPE",
                      "FluxKontextMultiReferenceLatentMethod"}

STRICT_KEYS = False
# The node whose skips of single blocks on the negative pass a step takes
# as its skip set, on the families whose DiT skips blocks.
SLG_NODE = "SkipLayerGuidanceDiTSimple"
SKIP_FAMILIES = frozenset({"moss_image"})
# The block replacement that node installs, which returns its input.
SLG_SKIP = f"{SLG_NODE}.execute.<locals>.skip"


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
    after checking that ComfyUI detects ``model_class``, or by the pack's
    own class of that name."""
    if model_class in PACK_MODELS:
        cfg = PACK_MODELS[model_class](manifest)
        return _built(cfg, {})
    if not hasattr(comfy.supported_models, model_class):
        raise GmlxError(f"gmlx: ComfyUI {comfyui_version()} has no {model_class} model, which "
                        f"the server serves {manifest.get('model')!r} as; update ComfyUI")
    sd = manifest_state_dict(manifest)
    prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
    stripped = comfy.utils.state_dict_prefix_replace(sd, {prefix: ""}, filter_keys=True)
    if stripped:
        sd = stripped
    cfg = comfy.model_detection.model_config_from_unet(sd, "")
    if cfg is None:
        raise GmlxError(f"gmlx: ComfyUI {comfyui_version()} detects no model in the server's "
                        f"manifest, where the server serves {model_class}; this ComfyUI may "
                        "be older than the model")
    detected = type(cfg).__name__
    if detected != model_class:
        raise GmlxError(f"gmlx: ComfyUI detects {detected} where the server serves "
                        f"{model_class}; update ComfyUI or the gmlx server")
    if cfg.unet_config.get("fused_qkv"):
        raise GmlxRefusal(f"gmlx: the server's {model_class} DiT holds the fused qkv layout "
                          "(single_blocks.0.qkv.weight), which a gmlx model does not take; "
                          "serve a file that stores q_proj, k_proj and v_proj apart")
    return _built(cfg, sd)


def _built(cfg, sd):
    dtypes = list(cfg.supported_inference_dtypes)
    dtype = torch.float32 if torch.float32 in dtypes else dtypes[0]
    cfg.set_inference_dtype(dtype, None, device=CPU)
    model = cfg.get_model(sd, "", device=torch.device("meta"))
    model.device = CPU
    if hasattr(model.diffusion_model, "preprocess_conds"):
        _conditioning_on_server(model)
    _typed_for_edits_and_negatives(model)
    return model, cfg


def _typed_for_edits_and_negatives(model) -> None:
    """When the server upsampled a prompt, an edit, whose conditioning
    carries reference latents, and a negative sample on the prompt as
    typed."""
    extra_conds = model.extra_conds

    @functools.wraps(extra_conds)
    def pick_prompt(**kwargs):
        tp = kwargs.pop(TYPED, None)
        if isinstance(tp, TypedPrompt) and tp.holds(kwargs.get("cross_attn")) and (
                kwargs.get("reference_latents") or kwargs.get("prompt_type") == "negative"):
            kwargs["cross_attn"] = tp.typed
        return extra_conds(**kwargs)
    model.extra_conds = pick_prompt


def _conditioning_on_server(model) -> None:
    """FLUX.3: ComfyUI runs the DiT's conditioning blocks once in
    ``extra_conds`` when the DiT is no flow map. Those blocks run on the
    server, so the model's ``extra_conds`` hands the step the conditioning
    as it does for a flow map, and a call of the blocks in ComfyUI is
    refused by name."""
    dm = model.diffusion_model

    def preprocess_conds(*args, **kwargs):
        raise GmlxRefusal(f"gmlx: ComfyUI ran the conditioning blocks of the "
                          f"{type(dm).__name__} DiT outside its forward, which a gmlx model "
                          "runs on the server")
    dm.preprocess_conds = preprocess_conds
    if getattr(dm, "flow_map", True):
        return
    extra_conds = model.extra_conds

    @functools.wraps(extra_conds)
    def keep_conditioning(**kwargs):
        dm.flow_map = True
        try:
            return extra_conds(**kwargs)
        finally:
            dm.flow_map = False
    model.extra_conds = keep_conditioning


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


def _running() -> tuple:
    """``(graph, node id)`` of the prompt that runs: the queue's running
    item of the server's last prompt id, else its only running item, and
    the node the server says runs. ``({}, None)`` when none is known. A
    prompt queued behind it is not read: ComfyUI hands the on_prompt
    handlers each prompt when it is queued, not when it runs."""
    try:
        import server
        inst = server.PromptServer.instance
        running, _ = inst.prompt_queue.get_current_queue_volatile()
        pid = getattr(inst, "last_prompt_id", None)
        sid = getattr(inst, "last_node_id", None)
    except Exception:  # noqa: BLE001: best effort; the refusal still names the cause
        return {}, None
    items = [it for it in running if it[1] == pid] or (running if len(running) == 1 else [])
    graph = items[0][2] if items else None
    if not isinstance(graph, dict) or sid is None:
        return {}, None
    return graph, sid


def _upstream_names() -> list:
    prompt, sid = _running()
    if not prompt:
        return []
    return [f"{n} ({prompt[n]['class_type']})" for n in walk_back(prompt, sid, IN_FORWARD_CLASSES)]


def _layers(text) -> list:
    """The block numbers of a layers widget: every run of digits."""
    return [int(v) for v in re.findall(r"\d+", str(text))]


def _slg_inputs() -> tuple:
    """``(node id, inputs)`` of the SkipLayerGuidanceDiTSimple node upstream
    of the running sampler. Refuses none and two that differ."""
    prompt, sid = _running()
    hits = walk_back(prompt, sid, {SLG_NODE}) if prompt else []
    if not hits:
        _refuse("transformer_options['patches_replace'] on single blocks, from a node other "
                f"than {SLG_NODE},")
    inputs = [prompt[n].get("inputs") or {} for n in hits]
    if any(i != inputs[0] for i in inputs[1:]):
        raise GmlxRefusal(f"gmlx: the {SLG_NODE} nodes {', '.join(hits)} skip other blocks; "
                          "a gmlx model takes one skip set a step")
    return hits[0], inputs[0]


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
        # A flow map DiT, FLUX.3's distilled one, reads the schedule the
        # sampler runs on, which each step carries.
        self.flow_map = bool(getattr(base_model.diffusion_model, "flow_map", False))
        params = getattr(base_model.diffusion_model, "params", None)
        # ComfyUI's Flux DiT reads the guidance input only through a guidance
        # embedder, which FLUX.2 klein has none of, and Flux.extra_conds
        # hands it on every call.
        self.reads_guidance = bool(getattr(params, "guidance_embed", True))
        # The reference rule of a DiT that reads ref_latents_method, which
        # the server's DiT follows; None for a DiT that reads no method.
        self.ref_method = getattr(params, "default_ref_method", None)
        self.warned: set = set()
        self.memo: dict = {}
        self.calls = collections.deque(maxlen=512)

    def _check(self, control, to: dict, kwargs: dict, kv: bool = False):
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
            if k == "guidance" and not self.reads_guidance:
                continue
            _refuse(f"the DiT input {k!r}")
        if kwargs.get("ref_latents") and not self.b.edit_capable:
            _refuse(f"reference latents on {self.b.model_id!r}, which takes none")
        # The method in force: the call's, else the DiT's default, which the
        # Flux KV Cache node sets for its sampling.
        params = getattr(self.base_model.diffusion_model, "params", None)
        method = kwargs.get("ref_latents_method") or getattr(params, "default_ref_method", None)
        want = KV_METHOD[1] if kv else self.ref_method
        if kwargs.get("ref_latents") and self.ref_method is not None and method != want:
            hits = _upstream_names()
            where = f"; upstream node(s): {', '.join(hits)}" if hits else ""
            under = " under the Flux KV Cache node" if kv else ""
            raise GmlxRefusal(f"gmlx: {self.b.model_id!r} places its reference latents by the "
                              f"{want!r} method{under}, and this call asks for {method!r}{where}")
        cache = to.get("qwen_image21_cache") or {}
        if cache.get("dtype", "default") not in ("default", None):
            raise GmlxRefusal(f"gmlx: a {cache['dtype']} prefix cache is not supported on a "
                              "gmlx model; use dtype default")
        return cache.get("device") != "off"

    def _skip(self, to: dict) -> tuple:
        """The blocks the rows leave out, from the single-block skips that
        SkipLayerGuidanceDiTSimple puts on the negative pass, checked
        against the node's widgets. Empty when the call carries no block
        replacements, or on a family whose DiT skips none, which the check
        refuses."""
        pr = to.get("patches_replace")
        if not pr or self.b.family not in SKIP_FAMILIES:
            return ()
        dit = (pr.get("dit") or {}) if isinstance(pr, dict) else {}
        if set(pr) - {"dit"} or not dit or any(
                not (isinstance(k, tuple) and len(k) == 2
                     and k[0] in ("single_block", "double_block"))
                or getattr(f, "__qualname__", None) != SLG_SKIP for k, f in dit.items()):
            _refuse("a transformer_options['patches_replace'] entry other than the "
                    f"block skips of {SLG_NODE}")
        blocks = sorted(int(k[1]) for k in dit if k[0] == "single_block")
        node, inputs = _slg_inputs()
        where = f"the {SLG_NODE} node {node}"
        widgets = ("double_layers", "single_layers", "start_percent", "end_percent")
        linked = [w for w in widgets if isinstance(inputs.get(w), list)]
        if linked:
            raise GmlxRefusal(f"gmlx: {where} takes {', '.join(linked)} from another node; "
                              "a gmlx model reads them as widgets")
        double = sorted({*_layers(inputs.get("double_layers", "")),
                         *(int(k[1]) for k in dit if k[0] == "double_block")})
        if double:
            raise GmlxRefusal(f"gmlx: {where} skips double_layers {double}; the MOSS-Image "
                              "DiT has single blocks only, so leave double_layers empty")
        start = float(inputs.get("start_percent", 0.0))
        end = float(inputs.get("end_percent", 1.0))
        if (start, end) != (0.0, 1.0):
            raise GmlxRefusal(f"gmlx: {where} skips its blocks from {start} to {end} of the "
                              "run; a gmlx model skips them over the whole run, at "
                              "start_percent 0 and end_percent 1")
        if sorted(set(_layers(inputs.get("single_layers", "")))) != blocks:
            raise GmlxRefusal(f"gmlx: this call skips single blocks {blocks}, which are not "
                              f"the single_layers of {where}")
        if any(c != 1 for c in to.get("cond_or_uncond") or [0]):
            raise GmlxRefusal(f"gmlx: {where} skips blocks of a positive row; a gmlx model "
                              "skips them on the negative pass alone")
        caps = getattr(self.b.client, "capabilities", None) or {}
        if "skip_blocks" not in (caps.get("features") or ()):
            raise GmlxRefusal("gmlx: this gmlx server skips no blocks in a step; update it")
        n = int(getattr(self.base_model.diffusion_model, "blocks", 0) or 0)
        bad = [b for b in blocks if not 0 <= b < n]
        if bad:
            raise GmlxRefusal(f"gmlx: {where} skips blocks {bad}, and the DiT has blocks 0 to "
                              f"{n - 1}")
        return tuple(blocks)

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

    def _send(self, x, timesteps, fields, sigmas=None):
        client: StepClient = self.b.client

        def stop():
            try:
                client.interrupt(fields.get("run"))
            except GmlxError as e:
                log.warning("gmlx: the interrupt did not reach the server: %s", e)

        return client.step(x, timesteps, fields, on_interrupt=stop, sample_sigmas=sigmas)

    def _schedule(self, to: dict):
        """The sampler's schedule a flow map step carries, else None."""
        if not self.flow_map:
            return None
        s = to.get("sample_sigmas")
        if s is None:
            raise GmlxError(f"gmlx: {self.b.model_id!r} is a flow map DiT, which reads the "
                            "schedule the sampler runs on, and this call carries no "
                            "transformer_options['sample_sigmas']")
        return torch.as_tensor(s).detach().float().reshape(-1).cpu()

    def __call__(self, x, timesteps, context=None, control=None, transformer_options={}, **kwargs):
        t0 = time.perf_counter()
        to = transformer_options or {}
        patcher = getattr(self.base_model, "current_patcher", None)
        kv = getattr(patcher, "gmlx_ref_cache", frozenset())
        if kv and kv != KV_PATCHES:
            _refuse("part of the Flux KV Cache node's patches")
        skip = self._skip(to)
        checked = {k: v for k, v in to.items() if k != "patches_replace"} if skip else to
        prefix_cache = self._check(control, checked, kwargs, bool(kv))
        sigmas = self._schedule(to)
        # A latent format with a frame axis, such as Krea 2's, gives one frame.
        framed = x.ndim == 5 and x.shape[2] == 1
        if framed:
            x = x[:, :, 0]
        if x.ndim != 4:
            raise GmlxError(f"gmlx: latents of shape {tuple(x.shape)} are not (B, C, h, w)")
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
        if kv and kwargs.get("ref_latents"):
            # The server keeps the references' K and V of the run's first step.
            fields["ref_cache"] = True
        if skip:
            fields["skip"] = list(skip)
        xs = x.float()
        ts = timesteps.float().reshape(-1)
        t1 = time.perf_counter()
        try:
            out, hdrs = self._send(xs, ts, fields, sigmas)
        except GmlxMissing as e:
            client.stats["missing_retries"] += 1
            missing = set(e.missing)
            for h in sorted(missing & set(held)):
                client.put_tensor(h, held[h])
            if table is not None and table in missing:
                table = patcher.sync_patches(force=True)
                fields["patch_table"] = table
            out, hdrs = self._send(xs, ts, fields, sigmas)
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
