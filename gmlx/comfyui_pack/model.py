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

from . import compat, discovery
from .client import (INT8_FEATURE, GmlxError, GmlxInterrupted, GmlxMissing, GmlxRefusal,
                     StepClient,
                     set_interrupt_check, tensor_hash)
from .clip import TYPED, TypedPrompt
from .compat import comfyui_version
from .moss_image import PACK_MODELS
from .patcher import (CPU, FASTH3, H3_NO_REFS, KV_METHOD, KV_PATCHES, GmlxBinding,
                      GmlxModelPatcher, tag_spans)

log = logging.getLogger("gmlx")

# Cancel in ComfyUI stops any wait on the server: a step, a load, an encode,
# a VAE call or the start of the server.
set_interrupt_check(comfy.model_management.throw_exception_if_processing_interrupted)

_DTYPES = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16,
           "quant": torch.bfloat16}

# The shifts MiniMaxH3SigmaShift sets, which an AV step carries.
H3_SHIFTS = {"video": "minimax_h3_sigma_shift_video", "audio": "minimax_h3_sigma_shift_audio"}
# The minimax_payload keys of MiniMaxH3.extra_conds the forward reads, each
# with a default when absent: what each carries, and the extra_conds input
# it is written from, None for one the model computes.
H3_PAYLOAD = {"audio_scale": ("the audio scale", None),
              "keyframes": ("the keyframes and guides", "minimax_keyframes"),
              "refs": ("the references", "minimax_refs"),
              "text_token_tags": ("the vision tokens of the prompt", "minimax_token_tags"),
              "seed": ("the seed of the keyframes' noise", "seed")}
# The noise mix of the anchor rows the server runs, ComfyUI's defaults:
# minimax_payload key, ComfyUI's constant, the server's value. No stock
# node sets the keys.
H3_COND_AUG = {"video": ("visual_cond_noise_aug", "VISUAL_COND_TIMESTEP", 0.999),
               "audio": ("audio_cond_noise_aug", "AUDIO_COND_TIMESTEP", 1.0)}
# The seed the load's probe of extra_conds passes, not the default 0.
PROBE_SEED = 7
# transformer_options keys and how the forward treats them.
FORWARDED = {"sample_sigmas", "sigmas", "cond_or_uncond", "uuids", "qwen_image21_cache",
             *H3_SHIFTS.values()}
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
# ComfyUI's class of the model a step with a video and an audio stream runs.
H3_CLASS = "MiniMaxH3"
# The node whose skips of single blocks on the negative pass a step takes
# as its skip set, on the families whose DiT skips blocks.
SLG_NODE = "SkipLayerGuidanceDiTSimple"
SKIP_FAMILIES = frozenset({"moss_image"})
# The sends again of a step the server answers 409, after the first.
STEP_RETRIES = 3
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
    missing = compat.first_missing(compat.H3_NAMES) if model_class == H3_CLASS else None
    if missing is not None:
        raise GmlxError(f"gmlx: the gmlx nodes run MiniMax-H3 through {missing}, which ComfyUI "
                        f"{comfyui_version()} does not have; update ComfyUI")
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
    if type(cfg).__name__ == H3_CLASS:
        # MiniMax-H3's extra_conds runs the refiner on the encoder features
        # once a sample; the server runs it in each step, on those features.
        model.diffusion_model.preprocess_text_embeds = lambda text_states: text_states
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
        # A model with a video and an audio stream takes ComfyUI's unpacked
        # [video, audio] list and returns one.
        self.av = bool(binding.status.get("audio"))
        self.warned: set = set()
        self.memo: dict = {}
        self.calls = collections.deque(maxlen=512)
        # The run and shifts _trained_shifts last warned of.
        self._shifts_warned = None

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

    def _send_held(self, send, fields: dict, held: dict, patcher):
        """``send()``, sent again after a 409 with what it names put again:
        the step's tensors, and its patch table synced again. An upload of
        another client can make the server drop them again before the step
        reaches it, so a step is sent STEP_RETRIES times more at most, and
        the 409 past that raises with the server's message."""
        client: StepClient = self.b.client
        tries, sync = 0, False
        while True:
            try:
                if sync:
                    fields["patch_table"] = patcher.sync_patches(force=True)
                    sync = False
                return send()
            except GmlxMissing as e:
                if tries == STEP_RETRIES:
                    raise
                tries += 1
                client.stats["missing_retries"] += 1
                missing = set(e.missing)
                for h in sorted(missing & set(held)):
                    client.put_tensor(h, held[h])
                sync = sync or fields["patch_table"] in missing

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

    def _h3_conditions(self, payload: dict, kwargs: dict) -> None:
        """Refuse what MiniMax-H3 conditions on that a gmlx model does not
        run: references, tags other than text and vision, an anchor noise
        mix other than ComfyUI's default, keyframes and guides on FastH3,
        and the noise masks of the latent."""
        if payload.get("refs"):
            raise GmlxRefusal("gmlx: references (minimax_refs, from MiniMaxH3ReferenceToVideo) "
                              f"reached the DiT forward; {H3_NO_REFS}")
        tags = payload.get("text_token_tags")
        if tags is not None and not bool(torch.isin(torch.as_tensor(tags),
                                                    torch.tensor([0, 1])).all()):
            raise GmlxRefusal("gmlx: the prompt's minimax_token_tags hold tags other than text "
                              "(1) and vision (0), which no stock node makes; a gmlx model runs "
                              "the text and the keyframes' vision rows of a prompt")
        for stream, (key, _name, value) in H3_COND_AUG.items():
            if key in payload and float(payload[key]) != value:
                art = "an" if stream == "audio" else "a"
                raise GmlxRefusal(f"gmlx: {art} {stream} anchor noise mix of {float(payload[key]):g} "
                                  f"(minimax_payload[{key!r}], which no stock node sets) reached "
                                  f"the DiT forward; a gmlx model mixes at ComfyUI's {value:g}")
        anchors = len(_h3_keyframes(payload))
        vision = tags is not None and bool((torch.as_tensor(tags) == 0).any())
        if self.b.status.get("variant") == FASTH3 and (anchors or vision):
            held = (f"{anchors} keyframe{'s' if anchors > 1 else ''} or guide"
                    f"{'s' if anchors > 1 else ''}" if anchors else "keyframes in its prompt")
            raise GmlxRefusal(f"gmlx: {self.b.model_id!r} is FastH3, which runs text to video "
                              f"only, and this run has {held}; run it without keyframes or "
                              "guides, or on h3")
        for k in ("denoise_mask", "audio_denoise_mask"):
            if kwargs.pop(k, None) is not None:
                raise GmlxRefusal(f"gmlx: a noise mask on the MiniMax-H3 latent reached the DiT "
                                  f"forward as {k}; a gmlx model samples the whole clip, so "
                                  "remove the latent's noise mask")

    def _trained_shifts(self, to: dict, shifts: dict) -> None:
        """Refuse a run on the entry's trained sigmas, or on any run of
        them in order, as a split of the schedule samples, at shifts other
        than the entry's. The forward derives the audio sigma from the
        video one at the shifts in force, which are 12 and 3 without
        ModelSamplingMiniMaxH3, so the audio would sample off its trained
        sigmas with no error. A call at other shifts that carries no
        schedule is refused too, naming the key it lacks. A schedule of
        its own at other shifts, as BasicScheduler gives, runs, with one
        warning a run."""
        sampling = self.b.status.get("sampling") or {}
        fixed = sampling.get("sigmas")
        if not fixed:
            return
        v, a = float(sampling["shift"]), float(sampling["audio_shift"])
        if (shifts["video"], shifts["audio"]) == (v, a):
            return
        why = (f"gmlx: {self.b.model_id!r} samples on its trained sigmas at the shifts {v:g} "
               f"and {a:g}, and this run is at {shifts['video']:g} and {shifts['audio']:g}")
        fix = f"put ModelSamplingMiniMaxH3 on the model at {v:g} and {a:g}"
        ran = to.get("sample_sigmas")
        if ran is None:
            raise GmlxError(f"{why} with no transformer_options['sample_sigmas'] to tell its "
                            f"schedule from them; {fix}")
        ran = torch.as_tensor(ran).detach().float().reshape(-1).cpu()
        table = torch.tensor([float(s) for s in fixed], dtype=torch.float32)
        n = len(ran)
        if any(torch.equal(ran, table[i:i + n]) for i in range(len(table) - n + 1)):
            raise GmlxRefusal(f"{why}; {fix}")
        run = getattr(getattr(self.base_model, "current_patcher", None), "gmlx_run", None)
        key = (run, shifts["video"], shifts["audio"])
        if self._shifts_warned != key:
            self._shifts_warned = key
            log.warning("%s on a schedule of its own; to sample as trained, %s and sample on "
                        "gmlx Sigmas", why, fix)

    def _h3_anchors(self, payload: dict, rows: int, base, held: dict) -> tuple:
        """(anchors, spans) of an AV step: each keyframe's frame and its
        latents by hash, a video latent with the noise ComfyUI's forward
        mixes into its rows (a CPU torch generator restarted at the
        payload's seed for each one, drawn in the rows' shape), and the
        vision spans of the prompt's ``rows`` tags."""
        keyframes = _h3_keyframes(payload)
        tags = payload.get("text_token_tags")
        spans = tag_spans(tags) if tags is not None else []
        if tags is not None and torch.as_tensor(tags).numel() != rows:
            raise GmlxError(f"gmlx: the prompt's minimax_token_tags hold "
                            f"{torch.as_tensor(tags).numel()} tags for its {rows} rows; encode "
                            "the prompt and its keyframes with one text encode node")
        if keyframes:
            caps = getattr(self.b.client, "capabilities", None) or {}
            if "anchors" not in (caps.get("features") or ()):
                raise GmlxRefusal("gmlx: this gmlx server runs no keyframes or guides; update it")
        pt, ph, pw = (int(p) for p in self.base_model.diffusion_model.patch_size)
        seed = int(payload.get("seed", 0))

        def put(i, kind, t):
            key = None if base is None else base + ("anchor", i, kind, tuple(t.shape), seed)
            return self._hash(key, t, held)
        anchors = []
        for i, kf in enumerate(keyframes):
            a = {"frame": int(kf["resolved_frame_index"])}
            if kf.get("latent") is not None:
                z = kf["latent"].detach().float().cpu()
                _, ch, t, h, w = z.shape
                gen = torch.Generator("cpu").manual_seed(seed)
                noise = torch.randn((t // pt * (h // ph) * (w // pw), ch * pt * ph * pw),
                                    generator=gen, dtype=torch.float32)
                a.update(video=put(i, "video", z), noise=put(i, "noise", noise))
            if kf.get("audio_latent") is not None:
                a["audio"] = put(i, "audio", kf["audio_latent"].detach().float().cpu())
            anchors.append(a)
        return anchors, spans

    def _int8(self, fields: dict) -> None:
        """The model loader's int8 setting on each AV step, which the server
        runs over its entry's; ``entry`` sends nothing. A server from before
        the int8 setting runs the entry's, with one warning a model when the
        loader asks for a value other than off."""
        value = self.b.int8
        if value == "entry":
            return
        caps = getattr(self.b.client, "capabilities", None) or {}
        if INT8_FEATURE in (caps.get("features") or ()):
            fields["int8"] = value
        elif value != "off" and "int8" not in self.warned:
            self.warned.add("int8")
            log.warning("gmlx: this gmlx server has no int8 setting, so %r runs as its "
                        "entry says; update it", self.b.model_id)

    def _av_call(self, t0, x, timesteps, context, control, to: dict, kwargs: dict):
        """One AV step: the video and audio latents as the sampler carries
        them, the audio scale and the shifts, which the server's step reads
        as ComfyUI's MiniMax-H3 forward does."""
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            shape = tuple(x.shape) if hasattr(x, "shape") else type(x).__name__
            raise GmlxError(f"gmlx: {self.b.model_id!r} samples a video and an audio latent, "
                            f"and this call carries {shape}; start from "
                            "EmptyMiniMaxH3LatentAV, MiniMaxH3ImageToVideo or "
                            "GmlxH3ImageToVideo")
        payload = kwargs.pop("minimax_payload", None) or {}
        self._h3_conditions(payload, kwargs)
        self._check(control, to, kwargs)
        dm = self.base_model.diffusion_model
        shifts = {"video": float(to.get(H3_SHIFTS["video"], dm.sigma_shift_video)),
                  "audio": float(to.get(H3_SHIFTS["audio"], dm.sigma_shift_audio))}
        self._trained_shifts(to, shifts)
        # One clip a step: the server refuses a batch, as ComfyUI's forward does.
        video, audio = x
        if context is None:
            raise GmlxError("gmlx: the DiT forward got no conditioning")
        patcher = getattr(self.base_model, "current_patcher", None)
        run = getattr(patcher, "gmlx_run", None)
        table = getattr(patcher, "gmlx_table_id", None)
        if patcher is not None and getattr(patcher, "gmlx", None) is not None:
            table = patcher.sync_patches()
        uuids = list(to.get("uuids") or [None])
        base = None if (uuids[0] is None or run is None) else (run, str(uuids[0]), 0)
        c = context[:1].float()
        held: dict = {}
        row = {"cond": self._hash(None if base is None else base + ("cond", tuple(c.shape)),
                                  c, held)}
        anchors, spans = self._h3_anchors(payload, int(c.shape[1]), base, held)
        client = self.b.client
        for h, t in held.items():
            client.ensure_tensor(t, h)
        fields = {"model": self.b.model_id, "run": run, "patch_table": table, "rows": [row],
                  "shifts": shifts, "audio_scale": float(payload.get("audio_scale", 1.0))}
        if anchors:
            fields["anchors"] = anchors
        if spans:
            fields["spans"] = spans
        self._int8(fields)
        vs, au = video.float(), audio.float()
        ts = timesteps.float().reshape(-1)

        def send():
            def stop():
                try:
                    client.interrupt(run)
                except GmlxError as e:
                    log.warning("gmlx: the interrupt did not reach the server: %s", e)
            return client.av_step(vs, au, ts, fields, on_interrupt=stop)

        t1 = time.perf_counter()
        try:
            out, hdrs = self._send_held(send, fields, held, patcher)
        except GmlxInterrupted:
            comfy.model_management.throw_exception_if_processing_interrupted()
            raise
        table = fields["patch_table"]
        t2 = time.perf_counter()
        out = [o.to(dtype=i.dtype, device=i.device) for o, i in zip(out, (video, audio))]
        t3 = time.perf_counter()
        self.calls.append({"rows": 1, "wall_ms": (t3 - t0) * 1000,
                           "prepare_ms": (t1 - t0) * 1000, "http_ms": (t2 - t1) * 1000,
                           "server_ms": float(hdrs.get("x-gmlx-server-ms", "nan")),
                           "forward_ms": float(hdrs.get("x-gmlx-forward-ms", "nan")),
                           "prefix_cache": hdrs.get("x-gmlx-prefix-cache"),
                           "table": table, "run": run})
        return out

    def __call__(self, x, timesteps, context=None, control=None, transformer_options={}, **kwargs):
        t0 = time.perf_counter()
        to = transformer_options or {}
        if self.av:
            return self._av_call(t0, x, timesteps, context, control, to, kwargs)
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
            out, hdrs = self._send_held(lambda: self._send(xs, ts, fields, sigmas), fields,
                                        held, patcher)
        except GmlxInterrupted:
            comfy.model_management.throw_exception_if_processing_interrupted()
            raise
        table = fields["patch_table"]
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


def load_model(client: StepClient, model_id: str, int8: str = "entry") -> GmlxModelPatcher:
    """The gmlx MODEL of ``model_id`` on ``client``'s server, whose MiniMax-H3
    steps run at the int8 setting ``int8``, or the entry's. A value whose
    linears differ from the entry's is refused here when the status names the
    entry's setting, else at the first step."""
    status = client.status(model_id)
    facts = status.get("comfy") or {}
    if not facts.get("model_class"):
        raise GmlxError(f"gmlx: the server gives no ComfyUI model class for {model_id!r}")
    if int8 not in ("entry", "off") and not status.get("audio"):
        raise GmlxRefusal(f"gmlx: int8 {int8} runs on MiniMax-H3, and {model_id!r} is an "
                          "image model; set int8 to entry")
    why = discovery.int8_refusal(status.get("int8") or "entry", int8)
    if why:
        raise GmlxRefusal(f"gmlx: {why}")
    manifest = client.manifest(model_id)
    model, _cfg = build_weightless(manifest, facts["model_class"])
    binding = GmlxBinding(client, model_id, status, int(manifest.get("count") or 0),
                          int8=int8)
    model.diffusion_model.forward = RemoteForward(model, binding)
    patcher = GmlxModelPatcher(model, load_device=CPU, offload_device=CPU)
    patcher.gmlx = binding
    if facts["model_class"] == H3_CLASS:
        _h3_keys(patcher)
    return patcher


# The keys of a MiniMax-H3 keyframe the AV step reads.
H3_KEYFRAME = ("resolved_frame_index", "latent", "audio_latent")


def _h3_keyframes(payload: dict) -> list:
    """The keyframes of ``payload`` that anchor latents. A keyframe that
    holds no latent anchors nothing, as in ComfyUI's forward. Refuses one
    that names none of the keys the step reads, or a latent with no
    frame, which a renamed key would give."""
    out = []
    for kf in payload.get("keyframes") or ():
        if not isinstance(kf, dict) or not any(k in kf for k in H3_KEYFRAME) or (
                "resolved_frame_index" not in kf
                and any(kf.get(k) is not None for k in H3_KEYFRAME[1:])):
            from .version import COMFYUI_FLOOR, COMFYUI_TESTED

            keys = sorted(kf) if isinstance(kf, dict) else type(kf).__name__
            raise GmlxError(f"gmlx: a MiniMax-H3 keyframe holds {keys}, and the gmlx nodes "
                            f"read {', '.join(H3_KEYFRAME)}; the gmlx nodes are tested on "
                            f"ComfyUI {COMFYUI_FLOOR} to {COMFYUI_TESTED}")
        if kf.get("latent") is not None or kf.get("audio_latent") is not None:
            out.append(kf)
    return out


def _h3_keys(patcher) -> None:
    """Refuse a ComfyUI whose MiniMax-H3 nodes no longer write a key the
    forward reads with a default, so a renamed key cannot run a clip at the
    wrong shift or scale: the shifts ModelSamplingMiniMaxH3 sets, and the
    minimax_payload keys of the model's extra_conds."""
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift

    from .version import COMFYUI_FLOOR, COMFYUI_TESTED

    tested = f"the gmlx nodes are tested on ComfyUI {COMFYUI_FLOOR} to {COMFYUI_TESTED}"

    def refuse(writer: str, key: str, what: str):
        raise GmlxError(f"gmlx: ComfyUI {comfyui_version()}'s {writer} does not write {key}, "
                        f"which the gmlx nodes read {what} from; {tested}")

    def probe(writer: str, call, given: str):
        try:
            return call()
        except Exception as e:  # noqa: BLE001: a changed node or model is named
            raise GmlxError(f"gmlx: ComfyUI {comfyui_version()}'s {writer} fails on {given}, "
                            f"which the gmlx nodes check it with ({type(e).__name__}: {e}); "
                            f"{tested}") from e

    def shifted():
        out = MiniMaxH3SigmaShift.execute(model=patcher, shift_video=10.0, shift_audio=3.0)
        m = (out.result if hasattr(out, "result") else out)[0]
        return m.model_options.get("transformer_options") or {}

    to = probe("ModelSamplingMiniMaxH3", shifted, "a model and shifts of 10 and 3")
    for stream, key in H3_SHIFTS.items():
        if key not in to:
            refuse("ModelSamplingMiniMaxH3", f"transformer_options[{key!r}]",
                   f"the {stream} shift")
    conds = probe("MiniMaxH3 model", lambda: patcher.model.extra_conds(
        minimax_token_tags=torch.ones(1, dtype=torch.long), minimax_keyframes=[{}],
        minimax_refs=[{}], seed=PROBE_SEED, device=torch.device("cpu")),
        "the extra_conds of a text prompt")
    held = getattr(conds.get("minimax_payload"), "cond", None)
    if not isinstance(held, dict):
        refuse("MiniMaxH3 model", "minimax_payload", "the conditioning beside the prompt")
    for key, (what, given) in H3_PAYLOAD.items():
        # The seed is written with a default, so the probe's own must come back.
        if key not in held or (key == "seed" and held[key] != PROBE_SEED):
            src = f" from {given}" if given else ""
            refuse("MiniMaxH3 model", f"minimax_payload[{key!r}]{src}", what)
    from comfy_extras.nodes_minimax_h3 import EmptyMiniMaxH3LatentAV, MiniMaxH3AddGuide

    class _Encoder:
        """A VAE that encodes a frame to zeros, for the probe of a guide."""
        def encode(self, frames):
            return torch.zeros(1, 24, 1, frames.shape[1] // 16, frames.shape[2] // 16)

    class _AudioEncoder:
        """An audio VAE that encodes a sound to one latent of zeros, at the
        rate the probe sends, so the node resamples nothing."""
        audio_sample_rate = 32000

        def encode(self, wave):
            return torch.zeros(1, 32, 2, 1)

    def guided(**given):
        out = EmptyMiniMaxH3LatentAV.execute(width=64, height=64, length=5)
        latent = (out.result if hasattr(out, "result") else out)[0]
        out = MiniMaxH3AddGuide.execute([[torch.zeros(1, 1, 8), {}]], latent, 0, **given)
        pos = (out.result if hasattr(out, "result") else out)[0]
        return (pos[0][1].get("minimax_keyframes") or [None])[-1]

    kf = probe("MiniMaxH3AddGuide", lambda: guided(
        vae=_Encoder(), image=torch.zeros(1, 64, 64, 3)), "a 64x64 image at frame 0")
    for key, what in (("resolved_frame_index", "a guide's frame"),
                      ("latent", "a guide's latents")):
        if not isinstance(kf, dict) or key not in kf:
            refuse("MiniMaxH3AddGuide", f"a keyframe's {key!r}", what)
    kf = probe("MiniMaxH3AddGuide", lambda: guided(
        audio_vae=_AudioEncoder(),
        audio={"waveform": torch.zeros(1, 2, 800), "sample_rate": 32000}),
        "a sound at frame 0")
    if not isinstance(kf, dict) or "audio_latent" not in kf:
        refuse("MiniMaxH3AddGuide", "a keyframe's 'audio_latent'", "a guide's sound latents")
    import comfy.ldm.minimax.model as h3_model

    for stream, (_key, name, value) in H3_COND_AUG.items():
        if getattr(h3_model, name, None) != value:
            raise GmlxError(f"gmlx: ComfyUI {comfyui_version()} mixes the {stream} anchor rows "
                            f"at {getattr(h3_model, name, None)} ({name}), and a gmlx model at "
                            f"{value:g}; {tested}")
