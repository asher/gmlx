"""MTP (multi-token-prediction) speculative-decode load path.

``load_mtp_model`` / ``load_vlm_mtp_model`` load a target model plus its MTP
drafter (native next-N head, gemma4 assistant GGUF, or the deepseek4 MTP
block) and install the verify-path runtime patches the speculative round
needs.
"""

from __future__ import annotations

import os
import sys

import mlx.core as mx
import mlx.nn as nn

import mlx_kquant as kq

from . import loadlog
from .dtypes import activation_dtype, activation_dtype_name
from .envflags import env_int
from .gdn_patches import (
    _needs_tiled_v_patch,
    _patch_dense_head_verify,
    _patch_gated_delta_tiled_v,
    _patch_mlxvlm_gated_delta_tiled_v,
)
from .gguf_meta import first_nonzero_int, read_int
from .loader import (
    _FP32_KEEP_BY_MODEL_TYPE,
    _active_now,
    _install_and_load,
    _resolve_chat_template,
    build_model,
    load_gguf_wire_bytes,
    materialize_module_arrays,
    model_is_moe,
    print_inventory,
    remap_arrays,
    remap_gemma4_assistant_arrays,
    remap_mtp_arrays,
    weights_source_key,
)
from .native_fp import _strip_weight
from .populate import maybe_populate_for_load
from .populate import wait_for as wait_for_populate
from .preflight import preflight
from .qwen35_gdn import prepare_gdn
from .qwen35_owned import is_owned_language_model
from .transforms import coalesce_split_experts


# Per-family MTP batch-width caps, keyed on model_type (NEVER isinstance:
# HyV3MTPDrafter subclasses QwenMTPDrafter, so a class check would hand hy_v3
# the uncapped dense-qwen default). Cap = speculate only while the live decode
# batch is this wide; 0 = uncapped. Measured knees, 2026-07 batch campaign:
# qwen dense nextn wins through c4 (worst corner 0.97); the gemma assistant
# wins c1/c2 on the dense 31B (1.83/1.80) and knees at B>=3. Limit = hard
# ceiling a config/env value can never cross, for drafters that raise above it.
#
# These are the DENSE-target defaults. Routed-expert targets are capped to 1 by
# `model_is_moe` regardless of family (see `_stamp_mtp_width_cap`): qwen MoE
# loses at B=2 (0.78x aggregate, d14k max_tokens 1024), and the structural
# check generalizes that to every MoE arch instead of needing a row here per
# checkpoint. The MoE rows below are kept explicit so a target that never
# reaches the structural check still lands on the measured value rather than
# the fallback.
_MTP_WIDTH_CAP_BY_MODEL_TYPE = {
    "qwen3_5": 0,
    "qwen3_5_text": 0,
    "qwen3_5_moe": 1,
    "qwen3_5_moe_text": 1,
    "gemma4_assistant": 2,
}
# B=1-only drafters: DeepseekV4MTPDrafter.reset and HyV3MTPDrafter.make_cache
# both raise on batched left_padding (hy_v3's inject_rows raises outright).
_MTP_WIDTH_LIMIT_BY_MODEL_TYPE = {
    "hy_v3": 1,
    "deepseek_v4": 1,
    "deepseek4": 1,
    "muse_glimmer": 1,
    # Qwen4ExpMTPDrafter.make_cache raises on batched left_padding.
    "qwen4_exp": 1,
}
# Unknown arch: cap conservatively rather than opting a new family into the
# losing regime. Uncapped is earned by measurement, not inherited by default.
_MTP_WIDTH_CAP_FALLBACK = 2

# Drafted depth per DFlash round: the GGUF's trained block, capped at 16.
# Verify cost on the 30B target is flat from 8 to 16 rows (the kquant split-K
# tile holds 16 rows in one MMA row-tile). Row 17 starts a second row-tile and
# costs ~55% more.
_MUSE_GLIMMER_DFLASH_BLOCK_DEFAULT = 16
# Drafted depth per round by dflash container. None drafts the GGUF's trained
# block (DFlash2: Qwen3.8 8, Muse 16); --draft-block-size moves it below that.
_DFLASH_BLOCK_DEFAULT = {
    "muse_glimmer": _MUSE_GLIMMER_DFLASH_BLOCK_DEFAULT,
    "dflash2": None,
}


def _drafter_block_depths(native_total, preferred_total=None) -> tuple[int, int]:
    """Return (deepest block the drafter can produce, depth drafted per round).

    The runtime depth is the family's preferred depth, bounded by the ceiling.
    --draft-block-size moves it at run time.
    """
    native_total = int(native_total)
    preferred = min(int(preferred_total or native_total), native_total)
    return native_total, max(2, min(preferred, native_total))


def _stamp_mtp_width_cap(drafter, model_type: str, *, target=None,
                         hard_limit: int | None = None,
                         log=loadlog.verbose_print):
    """Stamp mtp_width_cap / mtp_width_limit for the runtime gate.

    Call on the raw drafter BEFORE any DrafterAdapter wrap: the adapter
    forwards attribute reads but a setattr would land on the wrapper.
    A routed-expert ``target`` caps at 1 whatever its family default says.
    ``hard_limit`` is a drafter-imposed ceiling (the B=1-only DFlash
    drafters) that replaces the family's table row.
    MLX_VLM_GGUF_SPEC_WIDTH_CAP (per-model config, load-window env) overrides
    both and is itself clamped by the hard limit.
    """
    if hard_limit is not None:
        limit = int(hard_limit)
    else:
        limit = _MTP_WIDTH_LIMIT_BY_MODEL_TYPE.get(model_type, 0)
    cap = _MTP_WIDTH_CAP_BY_MODEL_TYPE.get(model_type)
    if cap is None:
        cap = limit if limit >= 1 else _MTP_WIDTH_CAP_FALLBACK
        if hard_limit is None and model_type not in _MTP_WIDTH_LIMIT_BY_MODEL_TYPE:
            log(f"[mtp] width cap: model_type {model_type!r} unmapped, "
                f"defaulting to {cap} (uncapped is measurement-earned)")
    # MoE verify multiplies the expert union each drafted position touches, and
    # both measured MoE families lose the trade above B=1. Structural, so a new
    # MoE arch inherits it without a table row; dense stays family-defaulted.
    if cap != 1 and target is not None and model_is_moe(target):
        log(f"[mtp] width cap: routed-expert target -> 1 "
            f"(family default was {cap or 'uncapped'})")
        cap = 1
    raw = os.environ.get("MLX_VLM_GGUF_SPEC_WIDTH_CAP", "")
    if raw:
        try:
            cap = max(0, int(raw))
        except ValueError:
            log(f"[mtp] width cap: MLX_VLM_GGUF_SPEC_WIDTH_CAP={raw!r} is not "
                f"an int; keeping {cap}")
    if limit >= 1 and (cap == 0 or cap > limit):
        log(f"[mtp] width cap: clamped {cap} -> {limit} ({model_type} drafter "
            f"is batch-width {limit} only)")
        cap = limit
    try:
        drafter.mtp_width_cap = cap
        drafter.mtp_width_limit = limit
    except AttributeError:
        pass  # slotted/frozen drafter forbids ad-hoc attrs
    log(f"[mtp] width cap: {cap or 'uncapped'}"
        + (f" (hard limit {limit})" if limit else ""))
    return drafter


def _load_mtp_drafter(
    arrays: dict,
    kquant_meta: dict,
    arch: str,
    config_dict: dict,
    target,
    *,
    n_head: int | None = None,
    n_head_kv: int | None = None,
    log=loadlog.verbose_print,
    source_key: tuple | None = None,
):
    """Build + load + bind the native-head MTP drafter (seam 4).

    The drafter weights live in the GGUF's own MTP block (block index
    ``num_hidden_layers``, the ``nextn.*`` extras + a full decoder layer). Build
    via ``Qwen3_5MTPConfig.from_dict`` (the only path that sets
    ``mtp_num_hidden_layers``), remap that block onto the drafter param tree,
    swap its quantized leaves, load with ``sanitize=False`` (the drafter's own
    ``sanitize`` adds +1 to GGUF norms unconditionally - seam 2), then ``bind``
    the target embeddings + LM head.
    """
    import importlib

    num_mtp = int(config_dict.get("mtp_num_hidden_layers", 1))
    first_mtp_block = int(config_dict["num_hidden_layers"])
    model_type = config_dict.get("model_type", "")

    if model_type == "hy_v3":
        from .hy_v3_model import ModelArgs
        from .hy_v3_mtp import HyV3MTPConfig, HyV3MTPDrafter

        drafter = HyV3MTPDrafter(
            HyV3MTPConfig(
                text_config=ModelArgs.from_dict(config_dict),
                # < 2 would make the owned decode loop exit after one token
                # (round size min(budget, native) <= 1 ends generation).
                block_size=max(2, env_int("GMLX_HY3_MTP_BLOCK", 2)),
            )
        )
        log(
            f"[mtp] drafter: HyV3MTPDrafter layer_idx={first_mtp_block} "
            f"block_size={drafter.config.block_size}"
        )
    else:
        cfg_mod = importlib.import_module(
            "mlx_vlm.speculative.drafters.qwen3_5_mtp.config"
        )
        drafter_mod = importlib.import_module(
            "mlx_vlm.speculative.drafters.qwen3_5_mtp.qwen3_5_mtp"
        )

        mtp_config = cfg_mod.Qwen3_5MTPConfig.from_dict(
            {
                "model_type": "qwen3_5_mtp",
                "text_config": dict(config_dict),
            }
        )
        # Owned native-head drafter (decode-time-only KV) vs mlx-vlm's. Same
        # weight tree, so the remap/install below is identical either way.
        # Default owned; GMLX_OWNED_MTP_DRAFTER=0 falls back for A/B.
        if os.environ.get("GMLX_OWNED_MTP_DRAFTER", "1") != "0":
            from gmlx.mtp_drafter import QwenMTPDrafter

            drafter = QwenMTPDrafter(mtp_config)
            log("[mtp] drafter: owned QwenMTPDrafter (decode-time-only KV)")
        else:
            drafter = drafter_mod.Qwen3_5MTPDraftModel(mtp_config)
            log("[mtp] drafter: mlx-vlm Qwen3_5MTPDraftModel")

    d_weights, d_meta, d_stats = remap_mtp_arrays(
        arrays,
        kquant_meta,
        arch,
        first_mtp_block=first_mtp_block,
        num_mtp_layers=num_mtp,
        n_head=n_head,
        n_head_kv=n_head_kv,
    )
    log(f"[mtp] drafter remap (block {first_mtp_block}+): {d_stats}")

    _install_and_load(
        drafter,
        d_weights,
        d_meta,
        log=log,
        sanitize=False,
        fp32_keep=_FP32_KEEP_BY_MODEL_TYPE.get(model_type, ()),
        source_key=source_key,
    )
    drafter.bind(target)
    from .drafter_protocol import validate_drafter
    validate_drafter(drafter)
    log("[mtp] drafter bound to target embeddings + LM head")
    _patch_draft_head_quantized(drafter)
    _stamp_mtp_width_cap(drafter, model_type, target=target, log=log)
    return drafter


def _mtp_dbg(msg: str) -> None:
    """Bind/decode-time MTP notices; opt in with GMLX_MTP_DEBUG=1 so they
    never land on stdout mid-reply (they otherwise corrupt streamed output)."""
    if os.environ.get("GMLX_MTP_DEBUG", "0") not in ("", "0"):
        print(msg, file=sys.stderr, flush=True)


def _patch_draft_head_quantized(drafter) -> None:
    """Swap the drafter's lm_head for a q8_0-encoded draft-side copy.

    The draft proposal only needs argmax/sample fidelity; verify keeps the
    target's own head, so emitted tokens are unchanged (lossless). On targets
    whose output.weight is float (F16/BF16), the draft-head GEMV is
    bandwidth-bound, so halving its bytes roughly halves the draft phase.
    Targets with an already-quantized head are left alone.
    GMLX_DRAFT_HEAD=f16 restores the stock head; =q4 trades further
    bytes for acceptance risk (measurement-gated).
    """
    mode = os.environ.get("GMLX_DRAFT_HEAD", "q8")
    if mode == "f16":
        return
    codec = {"q8": "q8_0", "q4": "q4_0"}.get(mode)
    if codec is None:
        return
    orig_bind = drafter.bind
    quantized = {}

    def _bind_with_quant_head(target_model):
        out = orig_bind(target_model)
        head = getattr(drafter, "_lm_head_fn", None)
        w = getattr(head, "weight", None)
        if w is None or w.dtype == mx.uint8:
            _mtp_dbg(f"[mtp] drafter head: skip (head={type(head).__name__}, "
                     f"w={'none' if w is None else w.dtype})")
            return out
        if int(w.shape[-1]) % 32 != 0 or getattr(head, "bias", None) is not None:
            _mtp_dbg(f"[mtp] drafter head: skip (K={int(w.shape[-1])}, "
                     f"bias={getattr(head, 'bias', None) is not None})")
            return out
        key = id(w)
        if key not in quantized:
            wq, sc = kq.quantize(w, codec)
            mx.eval(wq, sc)
            # Keep w itself in the entry: the id() key is only stable while
            # the weight is alive, so pin it against id reuse.
            quantized[key] = (w, wq, sc)
            _mtp_dbg(f"[mtp] drafter head: {codec} draft-side copy "
                     f"({w.nbytes / 1e9:.2f} -> {wq.nbytes / 1e9:.2f} GB)")
        _, wq, sc = quantized[key]
        drafter._lm_head_fn = lambda h: kq.quantized_matmul(h, wq, sc, codec)
        return out

    drafter.bind = _bind_with_quant_head


def _load_gemma4_assistant_drafter(
    draft_gguf_path: str, target, *, zero_copy: bool = True, log=loadlog.verbose_print
):
    """Build + load + bind the gemma4 assistant drafter from a companion GGUF.

    The drafter is a separate small dense gemma4 model (the ``--draft-gguf``
    companion, structurally like ``--mmproj``): load its wire bytes, synth a
    ``Gemma4AssistantConfig`` from its own metadata, build the mlx-vlm
    ``Gemma4AssistantDraftModel``, remap + kquant-swap its leaves, then ``bind``
    the target's input embedding + LM head. One class serves dense and MoE
    targets; the bridge is ``backbone_hidden_size`` (the target hidden), which
    mlx-vlm's ``validate_drafter_compatibility`` checks against the target.
    gemma4's drafter ``sanitize`` adds no norm offset, but we load with
    ``sanitize=False`` anyway (GGUF norms are already raw - seam 2).
    """
    import importlib

    active_before = _active_now()
    arrays, kquant_meta, d_arch, meta, tensor_shapes = load_gguf_wire_bytes(
        draft_gguf_path, zero_copy=zero_copy
    )
    log(
        f"[mtp] drafter gguf ({d_arch}): {len(arrays)} arrays, "
        f"{len(kquant_meta)} kquant"
    )

    from .config_synth import synthesize_gemma4_assistant_config

    drafter_cfg = synthesize_gemma4_assistant_config(meta, tensor_shapes)
    tc = drafter_cfg["text_config"]
    log(
        f"[mtp] drafter: gemma4_assistant backbone_hidden="
        f"{drafter_cfg['backbone_hidden_size']} layers={tc['num_hidden_layers']} "
        f"block_size={drafter_cfg['block_size']}"
    )

    cfg_mod = importlib.import_module(
        "mlx_vlm.speculative.drafters.gemma4_assistant.config"
    )
    drafter_mod = importlib.import_module(
        "mlx_vlm.speculative.drafters.gemma4_assistant.gemma4_assistant"
    )
    dcfg = cfg_mod.Gemma4AssistantConfig.from_dict(drafter_cfg)
    drafter = drafter_mod.Gemma4AssistantDraftModel(dcfg)

    d_weights, d_meta, d_stats = remap_gemma4_assistant_arrays(arrays, kquant_meta)
    log(f"[mtp] drafter remap: {d_stats}")

    _install_and_load(drafter, d_weights, d_meta, log=log, sanitize=False,
                      source_key=weights_source_key(draft_gguf_path),
                      active_before=active_before)
    # Ordered-embeddings drafters (E2B/E4B) route the LM head through a
    # MaskedEmbedder that reads embed_tokens.weight as a [vocab, hidden] float
    # matrix (gathers candidate rows then a dense matmul). A kquant wire-byte
    # embed table has row width = bytes-per-row (e.g. 272 for a 256-dim Q8_0 row),
    # not hidden, so dequantize it to the activation dtype before bind() - bind
    # closes over embed_tokens.weight for the head. The centroids Linear stays
    # kquant (it's a plain matmul). The table is small (vocab x hidden_size), so
    # a dense 16-bit copy is cheap.
    if drafter_cfg.get("use_ordered_embeddings"):
        from .modules import KQuantEmbedding

        emb = drafter.model.embed_tokens
        if isinstance(emb, KQuantEmbedding):
            w = kq.dequantize(emb["weight"], emb["scales"], emb.kquant_type).astype(
                activation_dtype()
            )
            new_emb = nn.Embedding(emb.num_embeddings, emb.dims)
            new_emb.weight = w
            drafter.model.embed_tokens = new_emb
            mx.eval(new_emb.weight)
            log(
                f"[mtp] drafter embed_tokens -> {activation_dtype_name()} for "
                f"ordered-embeddings head ({w.shape[0]}x{w.shape[1]})"
            )
    drafter.bind(target)

    from .drafter_protocol import DrafterAdapter, validate_drafter
    # Stamp before the wrap: DrafterAdapter forwards attribute reads, but a
    # setattr on the adapter would never reach the inner drafter.
    _stamp_mtp_width_cap(drafter, "gemma4_assistant", target=target, log=log)
    drafter = DrafterAdapter(drafter)
    validate_drafter(drafter)

    log("[mtp] drafter bound to target embeddings + LM head")
    return drafter


# The closed mtp.0.* tensor set of a deepseek4_mtp_support GGUF (verified
# against the real dump, 32 tensors). Weight-bearing entries (".weight" on
# both sides; kquant .scales siblings follow automatically):
_DEEPSEEK4_MTP_MAP = {
    "attn_q_a": "block.attn.wq_a.weight",
    "attn_q_a_norm": "block.attn.q_norm.weight",
    "attn_q_b": "block.attn.wq_b.weight",
    "attn_kv": "block.attn.wkv.weight",
    "attn_kv_a_norm": "block.attn.kv_norm.weight",
    "attn_output_a": "block.attn.wo_a.weight",  # 2D->3D MultiLinear reshape
    "attn_output_b": "block.attn.wo_b.weight",
    "attn_norm": "block.attn_norm.weight",
    "ffn_norm": "block.ffn_norm.weight",
    "ffn_gate_inp": "block.ffn.gate.weight",
    "ffn_gate_exps": "block.ffn.switch_mlp.gate_proj.weight",
    "ffn_up_exps": "block.ffn.switch_mlp.up_proj.weight",
    "ffn_down_exps": "block.ffn.switch_mlp.down_proj.weight",
    "ffn_gate_shexp": "block.ffn.shared_experts.gate_proj.weight",
    "ffn_up_shexp": "block.ffn.shared_experts.up_proj.weight",
    "ffn_down_shexp": "block.ffn.shared_experts.down_proj.weight",
    "e_proj": "e_proj.weight",
    "h_proj": "h_proj.weight",
    "enorm": "enorm.weight",
    "hnorm": "hnorm.weight",
    "norm": "norm.weight",
}
# Raw fp32 params (no ".weight" on the drafter side; fp32-pinned through the
# bf16 cast by _FP32_KEEP_BY_MODEL_TYPE["deepseek_v4"]):
_DEEPSEEK4_MTP_RAW = {
    "attn_sinks": "block.attn.attn_sink",
    "exp_probs_b.bias": "block.ffn.gate.e_score_correction_bias",
    "hc_attn_fn": "block.attn_hc.fn",
    "hc_attn_base": "block.attn_hc.base",
    "hc_attn_scale": "block.attn_hc.scale",
    "hc_ffn_fn": "block.ffn_hc.fn",
    "hc_ffn_base": "block.ffn_hc.base",
    "hc_ffn_scale": "block.ffn_hc.scale",
    "hc_head_fn": "hc_head.fn",
    "hc_head_base": "hc_head.base",
    "hc_head_scale": "hc_head.scale",
}


def remap_deepseek4_mtp_arrays(
    arrays: dict, kquant_meta: dict, *, o_groups: int, o_lora_rank: int
):
    """Remap a ``deepseek4_mtp_support`` GGUF onto the DeepseekV4MTPDrafter
    param tree. Self-contained (like ``remap_gemma4_assistant_arrays``): the
    tensor set is closed, so any unknown ``mtp.0.*`` name is a hard error
    (converter drift must surface at load, not as an unfilled param).

    ``attn_output_a`` arrives 2D ``[o_groups*o_lora_rank, in]`` and is
    reshaped to the 3D MultiLinear layout on the wire bytes and scales alike
    (row-major kquant rows are untouched; pure leading-dim split) -- the same
    transform the vendored ``Model.sanitize`` applies on the target, which
    the drafter (sanitize=False, its names are final) doesn't run.
    """
    hf_weights: dict[str, mx.array] = {}
    hf_kquant_meta: dict[str, str] = {}
    stats = {"mapped": 0}
    for name, arr in arrays.items():
        if name.endswith(".scales") or name.endswith(".biases"):
            continue
        if not name.startswith("mtp.0."):
            raise RuntimeError(
                f"deepseek4 MTP remap: unexpected non-mtp.0 tensor {name!r} "
                f"(the drafter tensor set is closed)"
            )
        rest = name[len("mtp.0.") :]
        base = rest[: -len(".weight")] if rest.endswith(".weight") else rest
        raw_target = _DEEPSEEK4_MTP_RAW.get(base)
        if raw_target is not None:
            hf_weights[raw_target] = arr
            stats["mapped"] += 1
            continue
        target = _DEEPSEEK4_MTP_MAP.get(base)
        if target is None:
            raise RuntimeError(
                f"deepseek4 MTP remap: unknown tensor {name!r} (converter drift?)"
            )
        codec = kquant_meta.get(name)
        scales = (
            arrays.get(_strip_weight(name) + ".scales") if codec is not None else None
        )
        if base == "attn_output_a":
            arr = arr.reshape(o_groups, o_lora_rank, -1)
            # Same ndim guard as the vendored Model.sanitize: codecs with
            # inline scales (q8_0) carry a size-1 .scales placeholder.
            if scales is not None and scales.ndim == 2:
                scales = scales.reshape(o_groups, o_lora_rank, -1)
        hf_weights[target] = arr
        if codec is not None:
            hf_weights[_strip_weight(target) + ".scales"] = scales
            hf_kquant_meta[target] = codec
        stats["mapped"] += 1
    return hf_weights, hf_kquant_meta, stats


def _load_deepseek4_mtp_drafter(
    draft_gguf_path: str,
    target,
    target_config_dict: dict,
    *,
    zero_copy: bool = True,
    log=loadlog.verbose_print,
):
    """Build + load + bind the DeepSeek-V4-Flash MTP drafter from its
    companion GGUF (arch ``deepseek4_mtp_support``, one full V4 block +
    fusion projections under ``mtp.0.*``). Mirrors the gemma4 assistant
    loader shape; the block config is the target's config with
    ``compress_ratios`` post-init extended by the MTP layer's ratio 0
    (``ModelArgs.__post_init__`` truncates to num_hidden_layers)."""
    active_before = _active_now()
    arrays, kquant_meta, d_arch, _meta, _shapes = load_gguf_wire_bytes(
        draft_gguf_path, zero_copy=zero_copy
    )
    if d_arch == "dflash":
        container = dflash_container(arrays)
        if container != "dspark":
            raise ValueError(
                f"{draft_gguf_path}: this dflash GGUF holds the {container} "
                f"drafter, which a deepseek_v4 target cannot drive"
            )
        arrays, kquant_meta, _meta = normalize_dflash_arrays(
            arrays, kquant_meta, _meta
        )
        d_arch = "deepseek4-dspark"
    if d_arch == "deepseek4-dspark":
        return _load_deepseek4_dspark_drafter(
            draft_gguf_path,
            target,
            target_config_dict,
            arrays=arrays,
            kquant_meta=kquant_meta,
            meta=_meta,
            active_before=active_before,
            log=log,
        )
    if d_arch != "deepseek4_mtp_support":
        raise ValueError(
            f"{draft_gguf_path}: expected a deepseek4-dspark, dflash, or "
            f"deepseek4_mtp_support drafter GGUF for a deepseek_v4 target, "
            f"got arch {d_arch!r}"
        )
    log(
        f"[mtp] drafter gguf ({d_arch}): {len(arrays)} arrays, "
        f"{len(kquant_meta)} kquant"
    )

    from .deepseek_v4_model import ModelArgs, ensure_registered
    from .deepseek_v4_mtp import DeepseekV4MTPConfig, DeepseekV4MTPDrafter

    ensure_registered()
    args = ModelArgs.from_dict(target_config_dict)
    args.compress_ratios = list(args.compress_ratios) + [0]
    drafter = DeepseekV4MTPDrafter(
        DeepseekV4MTPConfig(
            text=args, block_size=env_int("GMLX_DSV4_MTP_BLOCK", 4)
        )
    )
    log(
        f"[mtp] drafter: deepseek4 MTP block layer_idx={args.num_hidden_layers} "
        f"window={args.sliding_window} block_size={drafter.config.block_size}"
    )

    d_weights, d_meta, d_stats = remap_deepseek4_mtp_arrays(
        arrays, kquant_meta, o_groups=args.o_groups, o_lora_rank=args.o_lora_rank
    )
    log(f"[mtp] drafter remap: {d_stats}")

    _install_and_load(
        drafter,
        d_weights,
        d_meta,
        log=log,
        sanitize=False,
        fp32_keep=_FP32_KEEP_BY_MODEL_TYPE["deepseek_v4"],
        source_key=weights_source_key(draft_gguf_path),
        active_before=active_before,
    )
    drafter.bind(target)

    from .drafter_protocol import validate_drafter

    validate_drafter(drafter)
    log("[mtp] drafter bound to target embeddings + LM head")
    _patch_draft_head_quantized(drafter)
    _stamp_mtp_width_cap(drafter, "deepseek_v4", target=target, log=log)
    return drafter


def _load_qwen4exp_mtp_drafter(
    draft_gguf_path: str,
    target,
    target_config_dict: dict,
    *,
    zero_copy: bool = True,
    log=loadlog.verbose_print,
):
    """Build + load + bind the Qwen3.8-Flash-Next MTP drafter from its
    companion GGUF (arch ``qwen4exp-mtp``: the HF ``mtp.*`` tree, tensor
    names already in the drafter's layout under ``mtp.``)."""
    active_before = _active_now()
    arrays, kquant_meta, d_arch, meta, _shapes = load_gguf_wire_bytes(
        draft_gguf_path, zero_copy=zero_copy
    )
    from .qwen4_exp_model import ModelArgs, ensure_registered
    from .qwen4_exp_mtp import (
        MTP_ARCH,
        Qwen4ExpMTPConfig,
        Qwen4ExpMTPDrafter,
        remap_qwen4exp_mtp_arrays,
    )

    if d_arch != MTP_ARCH:
        raise ValueError(
            f"{draft_gguf_path}: expected a {MTP_ARCH} drafter GGUF for a "
            f"qwen4_exp target, got arch {d_arch!r}"
        )
    log(f"[mtp] drafter gguf ({d_arch}): {len(arrays)} arrays, "
        f"{len(kquant_meta)} kquant")

    ensure_registered()
    args = ModelArgs.from_dict(target_config_dict)
    ratio = int(meta.get(f"{MTP_ARCH}.attention.compress_ratio", 4) or 0)
    drafter = Qwen4ExpMTPDrafter(Qwen4ExpMTPConfig(
        text=args, block_size=env_int("GMLX_Q4_MTP_BLOCK", 4),
        compress_ratio=ratio))
    log(f"[mtp] drafter: qwen4exp MTP layer, QSA ratio={ratio} "
        f"block_size={drafter.config.block_size}")

    d_weights, d_meta, d_stats = remap_qwen4exp_mtp_arrays(arrays, kquant_meta)
    log(f"[mtp] drafter remap: {d_stats}")
    _install_and_load(
        drafter,
        d_weights,
        d_meta,
        log=log,
        sanitize=False,
        fp32_keep=_FP32_KEEP_BY_MODEL_TYPE["qwen4_exp"],
        source_key=weights_source_key(draft_gguf_path),
        active_before=active_before,
    )
    drafter.bind(target)

    from .drafter_protocol import validate_drafter

    validate_drafter(drafter)
    log("[mtp] drafter bound to target embeddings + LM head")
    _patch_draft_head_quantized(drafter)
    _stamp_mtp_width_cap(drafter, "qwen4_exp", target=target, log=log)
    return drafter


# The closed per-stage tensor set of a deepseek4-dspark GGUF (81 tensors for
# 3 stages, verified against both the antirez DSpark-support sidecar and the
# scripts/convert_dspark_sidecar.py output). Stage entries land under
# ``stages.{k}.``; the stage-0 fusion and final-stage head entries are
# drafter-level.
_DSPARK_STAGE_MAP = {
    "attn_q_a": "block.attn.wq_a.weight",
    "attn_q_a_norm": "block.attn.q_norm.weight",
    "attn_q_b": "block.attn.wq_b.weight",
    "attn_kv": "block.attn.wkv.weight",
    "attn_kv_a_norm": "block.attn.kv_norm.weight",
    "attn_output_a": "block.attn.wo_a.weight",  # 2D->3D MultiLinear reshape
    "attn_output_b": "block.attn.wo_b.weight",
    "attn_norm": "block.attn_norm.weight",
    "ffn_norm": "block.ffn_norm.weight",
    "ffn_gate_inp": "block.ffn.gate.weight",
    "ffn_gate_exps": "block.ffn.switch_mlp.gate_proj.weight",
    "ffn_up_exps": "block.ffn.switch_mlp.up_proj.weight",
    "ffn_down_exps": "block.ffn.switch_mlp.down_proj.weight",
    "ffn_gate_shexp": "block.ffn.shared_experts.gate_proj.weight",
    "ffn_up_shexp": "block.ffn.shared_experts.up_proj.weight",
    "ffn_down_shexp": "block.ffn.shared_experts.down_proj.weight",
}
_DSPARK_STAGE_RAW = {
    "attn_sinks": "block.attn.attn_sink",
    "exp_probs_b.bias": "block.ffn.gate.e_score_correction_bias",
    "hc_attn_fn": "block.attn_hc.fn",
    "hc_attn_base": "block.attn_hc.base",
    "hc_attn_scale": "block.attn_hc.scale",
    "hc_ffn_fn": "block.ffn_hc.fn",
    "hc_ffn_base": "block.ffn_hc.base",
    "hc_ffn_scale": "block.ffn_hc.scale",
}
_DSPARK_TOP_MAP = {
    "main_proj": "main_proj.weight",
    "main_norm": "main_norm.weight",
    "norm": "norm.weight",
    "markov_head.markov_w1": "markov_w1.weight",
    "markov_head.markov_w2": "markov_w2.weight",
    "confidence_head.proj": "confidence_proj.weight",
}
_DSPARK_TOP_RAW = {
    "hc_head_fn": "hc_head.fn",
    "hc_head_base": "hc_head.base",
    "hc_head_scale": "hc_head.scale",
}


# llama.cpp packages the same DSpark drafter under arch "dflash" (its
# convert_hf_to_gguf --dspark output, e.g. the unsloth release): per-stage
# tensors under blk.{k}.* with the same leaf names, and drafter-level tensors
# renamed through its root map. normalize_dflash_arrays translates that
# container back to the deepseek4-dspark namespace so one remap serves both.
# Drafter-level base name -> dspark base name; {L} is the last stage index
# (stage placement only matters for the remap's per-stage bookkeeping:
# main_proj/main_norm are stage-0 entries, the head tensors final-stage).
_DFLASH_ROOT_MAP = {
    "fc": "mtp.0.main_proj",
    "enc.output_norm": "mtp.0.main_norm",
    "output_norm": "mtp.{L}.norm",
    "markov_w1": "mtp.{L}.markov_head.markov_w1",
    "markov_w2": "mtp.{L}.markov_head.markov_w2",
    "conf_proj": "mtp.{L}.confidence_head.proj",
    "output_hc_fn": "mtp.{L}.hc_head_fn",
    "output_hc_base": "mtp.{L}.hc_head_base",
    "output_hc_scale": "mtp.{L}.hc_head_scale",
}
_DFLASH_SUFFIXES = (".weight", ".scales", ".biases", ".bias")


def _dflash_rename(name: str, last_stage: int) -> str:
    """The deepseek4-dspark name for one dflash tensor entry."""
    if name.startswith("blk."):
        return "mtp." + name[len("blk."):]
    base, suffix = name, ""
    for s in _DFLASH_SUFFIXES:
        if name.endswith(s):
            base, suffix = name[: -len(s)], s
            break
    mapped = _DFLASH_ROOT_MAP.get(base)
    if mapped is None:
        raise RuntimeError(
            f"dflash normalize: unknown tensor {name!r} "
            f"(the drafter tensor set is closed)"
        )
    return mapped.replace("{L}", str(last_stage)) + suffix


def dflash_container(arrays: dict) -> str:
    """Which drafter a llama.cpp ``dflash`` GGUF actually holds.

    The arch tag is shared: llama.cpp packages the DeepSeek-V4 DSpark drafter,
    the Muse Glimmer DFlash one and the DFlash 2 drafters under ``dflash``,
    and picks its graph on header keys. Tensor presence is the equivalent
    split here - DSpark carries the markov/confidence heads and MLA's
    ``attn_q_a``, DFlash 2 the candidate selector, Muse Glimmer plain
    ``attn_q`` with per-head QK-norms and no hyper-connections.
    """
    if any(n.startswith(("markov_w1", "markov_w2", "conf_proj", "output_hc_"))
           or ".attn_q_a" in n for n in arrays):
        return "dspark"
    if any(n.startswith("selector_hidden") for n in arrays):
        return "dflash2"
    if any(".attn_q_norm" in n for n in arrays):
        return "muse_glimmer"
    raise RuntimeError(
        "dflash GGUF matches no known drafter container (expected DSpark's "
        "markov/confidence heads, DFlash 2's selector, or Muse Glimmer's "
        "attn_q_norm)"
    )


# The closed tensor set of a DFlash drafter, onto gmlx's DFlashDrafter tree.
# Per-block leaves (blk.{i}.<key> -> layers.{i}.<value>):
_DFLASH_BLK = {
    "attn_norm": "input_layernorm.weight",
    "attn_q": "self_attn.q_proj.weight",
    "attn_k": "self_attn.k_proj.weight",
    "attn_v": "self_attn.v_proj.weight",
    "attn_output": "self_attn.o_proj.weight",
    "attn_q_norm": "self_attn.q_norm.weight",
    "attn_k_norm": "self_attn.k_norm.weight",
    "ffn_norm": "post_attention_layernorm.weight",
    "ffn_gate": "mlp.gate_proj.weight",
    "ffn_up": "mlp.up_proj.weight",
    "ffn_down": "mlp.down_proj.weight",
}
# DFlash 2 adds the grouped dynamic convolutions around both sublayers. The
# F32 base kernel has no ``.weight`` suffix on either side.
_DFLASH2_BLK = {
    "attn_conv_base": "attention_conv.base_kernel",
    "attn_conv_proj": "attention_conv.kernel_projection.weight",
    "ffn_conv_base": "mlp_conv.base_kernel",
    "ffn_conv_proj": "mlp_conv.kernel_projection.weight",
}
# Drafter-level leaves. ``enc.output_norm`` closes the encoder that fuses the
# target captures (llama.cpp's dflash graph<true>); ``output_norm`` is the
# decoder's final norm before the borrowed LM head.
_DFLASH_ROOT = {
    "fc": "fc.weight",
    "enc.output_norm": "hidden_norm.weight",
    "output_norm": "norm.weight",
}
_DFLASH2_ROOT = {
    "selector_hidden": "candidate_selector.hidden_projection.weight",
    "selector_predecessor": "candidate_selector.predecessor_codebook.weight",
    "selector_successor": "candidate_selector.successor_codebook.weight",
}
_DFLASH_CONTAINER_MAPS = {
    "muse_glimmer": (_DFLASH_BLK, _DFLASH_ROOT),
    "dflash2": ({**_DFLASH_BLK, **_DFLASH2_BLK}, {**_DFLASH_ROOT, **_DFLASH2_ROOT}),
}
# Which target families a container can drive. A load-time check; the
# arch-tag filter (arch_table.drafter_serves) is discovery's pairing-time one.
_DFLASH_CONTAINER_TARGETS = {
    "dflash2": ("qwen3_5", "qwen3_5_text", "muse_glimmer"),
    "muse_glimmer": ("muse_glimmer",),
}


def remap_dflash_arrays(arrays: dict, kquant_meta: dict, container: str):
    """Remap a ``dflash`` GGUF of ``container`` onto the drafter param tree.
    Closed tensor set: unknown names are hard errors (converter drift must
    surface at load, not as an unfilled param)."""
    blk_map, root_map = _DFLASH_CONTAINER_MAPS[container]
    hf_weights: dict[str, mx.array] = {}
    hf_kquant_meta: dict[str, str] = {}
    stats = {"mapped": 0}
    for name, arr in arrays.items():
        if name.endswith((".scales", ".biases")):
            continue
        base = name[: -len(".weight")] if name.endswith(".weight") else name
        if base.startswith("blk."):
            _, idx, leaf = base.split(".", 2)
            target = blk_map.get(leaf)
            if target is not None:
                target = f"layers.{idx}.{target}"
        else:
            target = root_map.get(base)
        if target is None:
            raise RuntimeError(
                f"{container} dflash remap: unknown tensor {name!r} "
                f"(the drafter tensor set is closed)"
            )
        hf_weights[target] = arr
        codec = kquant_meta.get(name)
        if codec is not None:
            hf_weights[_strip_weight(target) + ".scales"] = arrays.get(
                _strip_weight(name) + ".scales")
            hf_kquant_meta[target] = codec
        stats["mapped"] += 1
    return hf_weights, hf_kquant_meta, stats


def remap_muse_glimmer_dflash_arrays(arrays: dict, kquant_meta: dict):
    return remap_dflash_arrays(arrays, kquant_meta, "muse_glimmer")


def _dflash_config_from_meta(
    draft_gguf_path: str,
    meta: dict,
    target_config_dict: dict,
    container: str,
    *,
    arrays: dict,
    shapes: dict | None = None,
):
    """DFlashConfig for a ``dflash`` GGUF against its target, plus the
    capture layer ids. Drafter header keys win over the target's for the
    logit tail; ``dflash.attention.causal`` absent reads as non-causal
    (llama.cpp's reading of the same header)."""
    from .dflash_drafter import DFlashConfig

    family = target_config_dict.get("model_type")
    allowed = _DFLASH_CONTAINER_TARGETS[container]
    if family is not None and family not in allowed:
        raise ValueError(
            f"{draft_gguf_path}: the {container} drafter serves "
            f"{'/'.join(allowed)} targets, not {family}"
        )
    layers = meta.get("dflash.target_layers")
    block_size = meta.get("dflash.block_size")
    mask_token_id = meta.get("tokenizer.ggml.mask_token_id")
    if not layers or block_size is None or mask_token_id is None:
        raise ValueError(
            f"{draft_gguf_path}: dflash.target_layers / dflash.block_size / "
            f"tokenizer.ggml.mask_token_id missing - re-run the converter"
        )
    # llama.cpp indexes the residual ENTERING a layer, so the converter writes
    # the HF ids (layer outputs) one higher. Undo that: the capture seam takes
    # layer-output indices.
    layer_ids = tuple(int(i) - 1 for i in layers)
    n_target_layers = int(target_config_dict["num_hidden_layers"])
    if list(layer_ids) != sorted(set(layer_ids)) or not (
        0 <= layer_ids[0] and layer_ids[-1] < n_target_layers
    ):
        raise ValueError(
            f"{draft_gguf_path}: dflash.target_layers {layers} must be "
            f"strictly increasing and within [1, {n_target_layers}]"
        )
    hidden = int(target_config_dict["hidden_size"])
    declared = meta.get("dflash.embedding_length")
    if declared is not None and int(declared) != hidden:
        raise ValueError(
            f"{draft_gguf_path}: dflash.embedding_length {declared} != "
            f"target hidden_size {hidden}"
        )
    vocab = int(target_config_dict["vocab_size"])
    if container == "dflash2" and shapes:
        for name in ("selector_predecessor.weight", "selector_successor.weight"):
            ne = shapes.get(name)
            if ne is not None and int(ne[-1]) != vocab:
                raise ValueError(
                    f"{draft_gguf_path}: {name} has {ne[-1]} rows, the "
                    f"target vocab is {vocab}"
                )
    n_layers = 1 + max(
        int(n.split(".")[1]) for n in arrays if n.startswith("blk."))
    pattern = meta.get("dflash.attention.sliding_window_pattern") or ()
    layer_types = [
        "sliding_attention" if bool(t) else "full_attention" for t in pattern
    ] or ["full_attention"] * n_layers
    window = int(meta.get("dflash.attention.sliding_window") or 0) or None
    native_total, block_total = _drafter_block_depths(
        block_size, _DFLASH_BLOCK_DEFAULT[container])
    rope_scaling = None
    rs_type = meta.get("dflash.rope.scaling.type")
    if rs_type and rs_type != "none":
        rope_scaling = {"type": rs_type, "rope_type": rs_type}
        factor = meta.get("dflash.rope.scaling.factor")
        if factor is not None:
            rope_scaling["factor"] = float(factor)
    causal = meta.get("dflash.attention.causal")
    softcap = meta.get("dflash.final_logit_softcapping",
                       target_config_dict.get("final_logit_softcapping"))
    config = DFlashConfig(
        hidden_size=hidden,
        intermediate_size=int(meta["dflash.feed_forward_length"]),
        num_hidden_layers=n_layers,
        num_attention_heads=int(meta["dflash.attention.head_count"]),
        num_key_value_heads=int(meta["dflash.attention.head_count_kv"]),
        head_dim=int(meta["dflash.attention.key_length"]),
        rms_norm_eps=float(meta["dflash.attention.layer_norm_rms_epsilon"]),
        vocab_size=vocab,
        max_position_embeddings=int(meta.get("dflash.context_length")
                                    or target_config_dict["max_position_embeddings"]),
        rope_theta=float(meta["dflash.rope.freq_base"]),
        rope_scaling=rope_scaling,
        tie_word_embeddings=False,
        block_size=block_total,
        native_block_size=native_total,
        mask_token_id=int(mask_token_id),
        target_layer_ids=list(layer_ids),
        num_target_layers=n_target_layers,
        layer_types=layer_types,
        sliding_window=window,
        is_causal=None if causal is None else bool(causal),
        final_logit_softcapping=softcap or None,
        output_multiplier=float(meta.get(
            "dflash.logit_scale", target_config_dict.get("output_multiplier", 1.0))),
        input_embedding_scale=float(meta.get("dflash.embedding_scale", 1.0)),
        conv_kernel_size=int(meta.get("dflash.conv_kernel_size") or 0),
        conv_group_size=int(meta.get("dflash.conv_group_size") or 0),
        selector_rank=int(meta.get("dflash.selector_rank") or 0),
        selector_top_k=int(meta.get("dflash.selector_top_k") or 0),
    )
    return config, layer_ids


def _wire_dflash_capture(target, layer_ids) -> None:
    """Arm the target's packed-hidden capture for ``layer_ids``."""
    lm = getattr(target, "language_model", target)
    if not callable(getattr(lm, "set_dflash_capture", None)):
        raise RuntimeError(
            "DFlash drafters need a target carrying the _dflash_capture seam "
            f"(muse_glimmer, owned qwen3_5); got {type(lm).__name__}"
        )
    lm.set_dflash_capture(layer_ids)


def _load_muse_glimmer_dflash_drafter(
    draft_gguf_path: str,
    target,
    target_config_dict: dict,
    *,
    arrays: dict,
    kquant_meta: dict,
    meta: dict,
    shapes: dict | None = None,
    active_before: float | None = None,
    log=loadlog.verbose_print,
):
    """Build + load + bind the Muse Glimmer DFlash drafter, and wire the
    target's ``_dflash_capture`` so every engine-facing hidden carries the
    five captured residuals."""
    from .muse_glimmer_dflash import MuseGlimmerDFlashDrafter

    config, layer_ids = _dflash_config_from_meta(
        draft_gguf_path, meta, target_config_dict, "muse_glimmer",
        arrays=arrays, shapes=shapes)
    drafter = MuseGlimmerDFlashDrafter(config)
    log(
        f"[mtp] drafter: muse-glimmer dflash layers={config.num_hidden_layers} "
        f"targets={layer_ids} block_total={config.block_size} "
        f"window={config.sliding_window} causal={bool(config.is_causal)}"
    )

    d_weights, d_meta, d_stats = remap_dflash_arrays(
        arrays, kquant_meta, "muse_glimmer")
    log(f"[mtp] drafter remap: {d_stats}")
    _install_and_load(
        drafter,
        d_weights,
        d_meta,
        log=log,
        sanitize=False,
        source_key=weights_source_key(draft_gguf_path),
        active_before=active_before,
    )
    drafter.bind(target)
    _wire_dflash_capture(target, layer_ids)

    from .drafter_protocol import validate_drafter

    validate_drafter(drafter)
    # No draft-side head quantization: Muse Glimmer GGUFs ship a quantized
    # output.weight, which _patch_draft_head_quantized leaves alone anyway.
    log("[mtp] dflash drafter bound; target capture layers wired")
    _stamp_mtp_width_cap(drafter, "muse_glimmer", target=target,
                         hard_limit=1, log=log)
    return drafter


def _load_dflash2_drafter(
    draft_gguf_path: str,
    target,
    target_config_dict: dict,
    *,
    arrays: dict,
    kquant_meta: dict,
    meta: dict,
    shapes: dict | None = None,
    active_before: float | None = None,
    log=loadlog.verbose_print,
):
    """Build + load + bind a DFlash 2 drafter (conv-wrapped draft layers plus
    the candidate selector) and wire the target's ``_dflash_capture``."""
    from .dflash_drafter import DFlash2Drafter

    config, layer_ids = _dflash_config_from_meta(
        draft_gguf_path, meta, target_config_dict, "dflash2",
        arrays=arrays, shapes=shapes)
    drafter = DFlash2Drafter(config)
    log(
        f"[mtp] drafter: dflash2 layers={config.num_hidden_layers} "
        f"targets={layer_ids} block_total={config.block_size} "
        f"(native {config.native_block_size}) window={config.sliding_window} "
        f"causal={bool(config.is_causal)} selector top_k={config.selector_top_k} "
        f"rank={config.selector_rank} conv={config.conv_kernel_size}x"
        f"{config.conv_group_size} logit_scale={config.output_multiplier} "
        f"softcap={config.final_logit_softcapping}"
    )

    d_weights, d_meta, d_stats = remap_dflash_arrays(arrays, kquant_meta, "dflash2")
    log(f"[mtp] drafter remap: {d_stats}")
    _install_and_load(
        drafter,
        d_weights,
        d_meta,
        log=log,
        sanitize=False,
        source_key=weights_source_key(draft_gguf_path),
        active_before=active_before,
    )
    drafter.bind(target)
    _wire_dflash_capture(target, layer_ids)

    from .drafter_protocol import validate_drafter

    validate_drafter(drafter)
    log("[mtp] dflash2 drafter bound; target capture layers wired")
    _stamp_mtp_width_cap(
        drafter, str(target_config_dict.get("model_type") or "dflash2"),
        target=target, hard_limit=1, log=log)
    return drafter


def _load_dflash_drafter(
    draft_gguf_path: str,
    target,
    target_config_dict: dict,
    *,
    zero_copy: bool = True,
    log=loadlog.verbose_print,
):
    """Load a ``dflash`` companion drafter: Muse Glimmer's DFlash or a DFlash
    2 drafter, by container. DSpark's ``dflash`` GGUFs load through the
    deepseek_v4 path."""
    active_before = _active_now()
    arrays, kquant_meta, d_arch, meta, shapes = load_gguf_wire_bytes(
        draft_gguf_path, zero_copy=zero_copy
    )
    if d_arch != "dflash":
        raise ValueError(
            f"{draft_gguf_path}: expected a dflash drafter GGUF, got arch "
            f"{d_arch!r}"
        )
    container = dflash_container(arrays)
    log(f"[mtp] drafter gguf ({d_arch}/{container}): {len(arrays)} arrays, "
        f"{len(kquant_meta)} kquant")
    loaders = {
        "dflash2": _load_dflash2_drafter,
        "muse_glimmer": _load_muse_glimmer_dflash_drafter,
    }
    loader = loaders.get(container)
    if loader is None:
        raise ValueError(
            f"{draft_gguf_path}: this dflash GGUF holds the {container} "
            f"drafter, which loads only against a deepseek_v4 target"
        )
    return loader(
        draft_gguf_path,
        target,
        target_config_dict,
        arrays=arrays,
        kquant_meta=kquant_meta,
        meta=meta,
        shapes=shapes,
        active_before=active_before,
        log=log,
    )


def _drafter_header_arch(draft_gguf_path: str) -> str | None:
    """The companion GGUF's ``general.architecture`` from a header-only read."""
    from .discovery import header_meta

    meta = header_meta(draft_gguf_path)
    return meta.get("arch") if meta else None


def _assistant_kind(model_type: str | None, draft_gguf_path: str) -> str:
    """Which companion loader a ``--draft-gguf`` takes: ``deepseek4`` for a
    deepseek_v4 target (its drafters share the ``dflash`` tag with DSpark),
    ``dflash`` for a muse_glimmer target or any ``dflash`` header, else
    ``gemma4``. Each loader validates the pairing it is handed."""
    if model_type == "deepseek_v4":
        return "deepseek4"
    if model_type == "qwen4_exp":
        return "qwen4exp"
    if model_type == "muse_glimmer" or _drafter_header_arch(draft_gguf_path) == "dflash":
        return "dflash"
    return "gemma4"


def normalize_dflash_arrays(arrays: dict, kquant_meta: dict, meta: dict):
    """Translate a llama.cpp ``dflash`` GGUF (tensor names and metadata) to
    the ``deepseek4-dspark`` namespace. Returns ``(arrays, kquant_meta,
    meta)`` ready for :func:`_load_deepseek4_dspark_drafter`."""
    stages = {
        int(n.split(".")[1]) for n in arrays if n.startswith("blk.")
    }
    if not stages:
        raise RuntimeError("dflash normalize: no blk.* stage tensors found")
    last_stage = max(stages)
    n_arrays = {
        _dflash_rename(name, last_stage): arr for name, arr in arrays.items()
    }
    n_kquant = {
        _dflash_rename(name, last_stage): codec
        for name, codec in kquant_meta.items()
    }
    n_meta = dict(meta)
    block_size = meta.get("dflash.block_size")
    if block_size is not None:
        n_meta["dspark.block_size"] = block_size
    # llama.cpp's converter writes the capture layers shifted +1 (its layer 0
    # is the embedding) and carries the noise token as the tokenizer mask
    # token; undo both. Verified against the source config ([40, 41, 42] for
    # a 43-layer target) and conversion/deepseek.py's add_target_layers.
    layers = meta.get("dflash.target_layers")
    if layers is not None:
        n_meta["dspark.target_layer_ids"] = [int(i) - 1 for i in layers]
    mask = meta.get("tokenizer.ggml.mask_token_id")
    if mask is not None:
        n_meta["dspark.noise_token_id"] = int(mask)
    w1 = n_arrays.get(f"mtp.{last_stage}.markov_head.markov_w1.weight")
    if w1 is not None and "dspark.markov_rank" not in n_meta:
        n_meta["dspark.markov_rank"] = int(min(w1.shape))
    return n_arrays, n_kquant, n_meta


def remap_deepseek4_dspark_arrays(
    arrays: dict,
    kquant_meta: dict,
    *,
    n_stages: int,
    o_groups: int,
    o_lora_rank: int,
):
    """Remap a ``deepseek4-dspark`` GGUF onto the DeepseekV4DSparkDrafter
    param tree. Closed tensor set: unknown ``mtp.*`` names are hard errors
    (converter drift must surface at load, not as an unfilled param)."""
    hf_weights: dict[str, mx.array] = {}
    hf_kquant_meta: dict[str, str] = {}
    stats = {"mapped": 0}
    for name, arr in arrays.items():
        if name.endswith(".scales") or name.endswith(".biases"):
            continue
        parts = name.split(".", 2)
        if len(parts) != 3 or parts[0] != "mtp" or not parts[1].isdigit():
            raise RuntimeError(
                f"deepseek4 DSpark remap: unexpected tensor {name!r} "
                f"(the drafter tensor set is closed)"
            )
        stage = int(parts[1])
        if stage >= n_stages:
            raise RuntimeError(
                f"deepseek4 DSpark remap: {name!r} exceeds stage count "
                f"{n_stages}"
            )
        rest = parts[2]
        base = rest[: -len(".weight")] if rest.endswith(".weight") else rest
        top_raw = _DSPARK_TOP_RAW.get(base)
        if top_raw is not None:
            hf_weights[top_raw] = arr
            stats["mapped"] += 1
            continue
        stage_raw = _DSPARK_STAGE_RAW.get(base)
        if stage_raw is not None:
            hf_weights[f"stages.{stage}.{stage_raw}"] = arr
            stats["mapped"] += 1
            continue
        top = _DSPARK_TOP_MAP.get(base)
        if top is not None:
            target = top
        else:
            mapped = _DSPARK_STAGE_MAP.get(base)
            if mapped is None:
                raise RuntimeError(
                    f"deepseek4 DSpark remap: unknown tensor {name!r} "
                    f"(converter drift?)"
                )
            target = f"stages.{stage}.{mapped}"
        codec = kquant_meta.get(name)
        scales = (
            arrays.get(_strip_weight(name) + ".scales") if codec is not None else None
        )
        if base == "attn_output_a":
            arr = arr.reshape(o_groups, o_lora_rank, -1)
            if scales is not None and scales.ndim == 2:
                scales = scales.reshape(o_groups, o_lora_rank, -1)
        hf_weights[target] = arr
        if codec is not None:
            hf_weights[_strip_weight(target) + ".scales"] = scales
            hf_kquant_meta[target] = codec
        stats["mapped"] += 1
    return hf_weights, hf_kquant_meta, stats


def _dspark_meta(meta: dict, key: str, default=None):
    """dspark kv with the ds4 alias order (deepseek4.dspark.X first)."""
    for k in (
        f"deepseek4.dspark.{key}",
        f"deepseek4.dspark_{key}",
        f"dspark.{key}",
    ):
        if k in meta:
            return meta[k]
    return default


def _load_deepseek4_dspark_drafter(
    draft_gguf_path: str,
    target,
    target_config_dict: dict,
    *,
    arrays: dict,
    kquant_meta: dict,
    meta: dict,
    active_before: float | None = None,
    log=loadlog.verbose_print,
):
    """Build + load + bind the DSpark drafter from its companion GGUF (arch
    ``deepseek4-dspark``: 3 chained V4 blocks under ``mtp.{k}.*`` reading
    target hiddens from ``dspark.target_layer_ids``, plus markov/confidence
    heads). Also wires the target's ``_dspark_capture`` so every
    engine-facing hidden carries the capture pack."""
    from .deepseek_v4_dspark import (
        DeepseekV4DSparkConfig,
        DeepseekV4DSparkDrafter,
    )
    from .deepseek_v4_model import ModelArgs, ensure_registered

    ensure_registered()
    n_stages = 1 + max(
        int(n.split(".")[1]) for n in arrays if n.startswith("mtp.")
    )
    draft_len = int(_dspark_meta(meta, "block_size", 5))
    layer_ids = tuple(
        int(i) for i in (_dspark_meta(meta, "target_layer_ids") or ())
    )
    noise_token_id = _dspark_meta(meta, "noise_token_id")
    if not layer_ids or noise_token_id is None:
        raise ValueError(
            f"{draft_gguf_path}: dspark.target_layer_ids / "
            f"dspark.noise_token_id metadata missing - re-run the converter"
        )
    args = ModelArgs.from_dict(target_config_dict)
    n = int(args.num_hidden_layers)
    if list(layer_ids) != sorted(set(layer_ids)) or layer_ids[-1] >= n:
        raise ValueError(
            f"{draft_gguf_path}: dspark.target_layer_ids {layer_ids} must be "
            f"strictly increasing and < {n}"
        )
    args.compress_ratios = list(args.compress_ratios) + [0] * n_stages
    native_total, block_total = _drafter_block_depths(draft_len + 1)
    drafter = DeepseekV4DSparkDrafter(
        DeepseekV4DSparkConfig(
            text=args,
            n_stages=n_stages,
            draft_len=draft_len,
            noise_token_id=int(noise_token_id),
            target_layer_ids=layer_ids,
            markov_rank=int(_dspark_meta(meta, "markov_rank", 256)),
            block_size=block_total,
            native_block_size=native_total,
        )
    )
    log(
        f"[mtp] drafter: dspark stages={n_stages} draft_len={draft_len} "
        f"targets={layer_ids} block_total={block_total} "
        f"window={args.sliding_window}"
    )

    d_weights, d_meta, d_stats = remap_deepseek4_dspark_arrays(
        arrays,
        kquant_meta,
        n_stages=n_stages,
        o_groups=args.o_groups,
        o_lora_rank=args.o_lora_rank,
    )
    log(f"[mtp] drafter remap: {d_stats}")

    # Default the drafter's native-fp (mxfp4) experts to zero-copy wire: the
    # packed de-interleave would pin ~9.6 GiB of anonymous memory for
    # sparsely-touched MoE experts, and the auto gate never picks wire for a
    # 10 GiB GGUF on its own. Wire's measured decode penalty (-5%) is noise
    # on a drafter that is a small slice of each verify round. An explicit
    # GMLX_NATIVE_FP (or a kq build without the codecs) wins.
    import mlx_kquant as kq

    from .native_fp import NATIVE_FP_CODECS

    fp_codecs = {c for c in d_meta.values() if c in NATIVE_FP_CODECS}
    force_wire = (
        "GMLX_NATIVE_FP" not in os.environ
        and fp_codecs
        and fp_codecs <= set(kq.codecs())
    )
    if force_wire:
        os.environ["GMLX_NATIVE_FP"] = "wire"
    try:
        _install_and_load(
            drafter,
            d_weights,
            d_meta,
            log=log,
            sanitize=False,
            fp32_keep=_FP32_KEEP_BY_MODEL_TYPE["deepseek_v4"]
            + ("confidence_proj.",),
            source_key=weights_source_key(draft_gguf_path),
            active_before=active_before,
        )
    finally:
        if force_wire:
            del os.environ["GMLX_NATIVE_FP"]
    drafter.bind(target)

    lm = getattr(target, "language_model", target)
    if not hasattr(lm, "_dspark_capture"):
        raise RuntimeError(
            "DSpark drafter needs a DeepseekV4SpecLM target (with the "
            f"_dspark_capture seam); got {type(lm).__name__}"
        )
    lm._dspark_capture = layer_ids

    from .drafter_protocol import validate_drafter

    validate_drafter(drafter)
    log("[mtp] dspark drafter bound; target capture layers wired")
    _patch_draft_head_quantized(drafter)
    _stamp_mtp_width_cap(drafter, "deepseek_v4", target=target, log=log)
    return drafter


@loadlog.seeds
def load_mtp_model(
    gguf_path: str,
    *,
    arch: str | None = None,
    draft_gguf_path: str | None = None,
    chat_template: str | None = None,
    zero_copy: bool = True,
    verbose: bool = False,
    wire: bool = True,
):
    """Load an MTP target+drafter pair: the text target on mlx-vlm classes plus
    a drafter. Two drafter-acquisition shapes:

    - **Native-head** (``draft_gguf_path=None``): qwen3.5/3.6, drafter extracted
      from the target GGUF's own MTP block (``nextn.*``).
    - **Assistant** (``draft_gguf_path`` given): gemma4, drafter is a separate
      companion GGUF (``Gemma4AssistantDraftModel``), structurally like mmproj.

    Returns ``(model, drafter, config, tokenizer)`` where ``model`` is an
    ``MTPTextTarget`` exposing the mlx-vlm ``LanguageModel`` as ``.language_model``.
    Drives mlx-vlm's MTP engine (``run_speculative_rounds`` /
    ``run_speculative_server_rounds``). The plain-text ``load_model`` path is
    untouched; this is a separate entry so the proven text load stays
    byte-identical.

    ``wire=False`` skips the sticky wired limit for callers that will install
    expert streaming: ``mx.set_wired_limit`` actively wires the resident
    buffer set, and on an over-RAM zero-copy target that wires the whole
    model's mmap views - a fast march straight through the free-page floor.
    The streaming installers manage their own (arena) wiring.
    """

    _log = loadlog.verbose_print

    loadlog.stage("reading gguf metadata")
    pf = preflight(gguf_path, arch=arch)
    arch = pf.arch
    loadlog.fact("arch", arch)
    loadlog.fact_file_size(pf.shards)
    _log(f"[arch] {arch}")

    maybe_populate_for_load(pf.shards, log=_log)

    loadlog.stage("reading tensors")
    active_before = _active_now()
    arrays, kquant_meta, _arch_meta, meta, tensor_shapes = load_gguf_wire_bytes(
        gguf_path, zero_copy=zero_copy, shards=pf.shards
    )
    arrays, kquant_meta, _n = coalesce_split_experts(arrays, kquant_meta)
    _log(f"[gguf] {len(arrays)} arrays, {len(kquant_meta)} kquant")

    from .config_synth import synthesize_config

    config_dict = synthesize_config(meta, tensor_shapes)
    assistant = draft_gguf_path is not None
    if not assistant and config_dict.get("model_type") == "deepseek_v4":
        # DeepSeek-V4 ships its MTP head as a companion GGUF (arch
        # deepseek4_mtp_support), never as in-GGUF nextn tensors; the
        # native-head extraction below is qwen-shaped and cannot serve it,
        # even though the V4 metadata advertises mtp_num_hidden_layers.
        from . import arch_table
        from .discovery import find_mtp_companion

        draft_gguf_path = find_mtp_companion(
            gguf_path, arch_table.drafter_arches("deepseek_v4"))
        if draft_gguf_path is None:
            raise ValueError(
                "deepseek_v4 MTP needs its companion drafter GGUF (arch "
                "deepseek4-dspark or deepseek4_mtp_support); none found next "
                f"to {gguf_path} - pass --draft-gguf <path>."
            )
        assistant = True
        loadlog.fact("mtp_companion", os.path.basename(draft_gguf_path))
        _log(f"[mtp] companion drafter autodetected: {draft_gguf_path}")
    if not assistant and config_dict.get("model_type") == "qwen4_exp":
        # Qwen3.8-Flash-Next's head lives in the HF safetensors only; the
        # companion GGUF (arch qwen4exp-mtp) is the drafter.
        from . import arch_table
        from .discovery import find_mtp_companion

        draft_gguf_path = find_mtp_companion(
            gguf_path, arch_table.drafter_arches("qwen4_exp"))
        if draft_gguf_path is None:
            raise ValueError(
                "qwen4_exp MTP needs its companion drafter GGUF (arch "
                "qwen4exp-mtp, built from the HF mtp.* tensors); none found "
                f"next to {gguf_path} - pass --draft-gguf <path>."
            )
        assistant = True
        loadlog.fact("mtp_companion", os.path.basename(draft_gguf_path))
        _log(f"[mtp] companion drafter autodetected: {draft_gguf_path}")
    if not assistant and config_dict.get("model_type") == "muse_glimmer":
        # Muse Glimmer's drafter is likewise a companion GGUF (arch dflash),
        # never an in-file nextn block.
        from . import arch_table
        from .discovery import find_mtp_companion

        draft_gguf_path = find_mtp_companion(
            gguf_path, arch_table.drafter_arches("muse_glimmer"))
        if draft_gguf_path is None:
            raise ValueError(
                "muse_glimmer MTP needs its companion DFlash drafter GGUF "
                f"(arch dflash); none found next to {gguf_path} - pass "
                "--draft-gguf <path>."
            )
        assistant = True
        loadlog.fact("mtp_companion", os.path.basename(draft_gguf_path))
        _log(f"[mtp] companion drafter autodetected: {draft_gguf_path}")
    if not assistant and int(config_dict.get("mtp_num_hidden_layers", 0)) < 1:
        raise ValueError(
            f"{gguf_path}: no native MTP head "
            f"({arch}.nextn_predict_layers absent / 0) - pass draft_gguf_path "
            f"for assistant-shape MTP (gemma4), or use a native-head GGUF"
        )

    n_head = read_int(meta, f"{arch}.attention.head_count")
    n_head_kv = first_nonzero_int(meta, f"{arch}.attention.head_count_kv")

    # 1. target text weights -> mlx-vlm LanguageModel (sanitize=False, seam 2).
    loadlog.stage("remapping tensors")
    owned_names: set[str] = set()
    hf_weights, hf_kquant_meta, stats = remap_arrays(
        arrays,
        kquant_meta,
        arch,
        n_head=n_head,
        n_head_kv=n_head_kv,
        owned_names=owned_names,
    )
    from collections import Counter

    loadlog.fact("codecs", Counter(hf_kquant_meta.values()))
    if loadlog.is_verbose():
        print_inventory(arch, kquant_meta, hf_kquant_meta, stats)

    loadlog.stage("building model")
    model, config = build_model(config_dict, mtp=True)
    loadlog.fact("model_type", config.get("model_type"))

    # 2. tiled-V fixup for asymmetric K/V heads - both mlx-lm (transitive) and
    #    mlx-vlm's own gated_delta (the MTP target / state-capture paths).
    #    Owned trees never route through the vlm module, so the vlm rebind
    #    is stock-only. Gate on the built tree, not the config.
    if _needs_tiled_v_patch(config):
        _patch_gated_delta_tiled_v()
        if not is_owned_language_model(model):
            _patch_mlxvlm_gated_delta_tiled_v()

    # deepseek_v4 needs sanitize=True (the vendored Model.sanitize does the
    # wo_a 2D->3D MultiLinear reshape, same as the plain-text load path) and
    # the fp32 pins (HC tables / sinks / router -- see _FP32_KEEP_BY_MODEL_TYPE).
    # hy_v3's sanitize strips the in-GGUF MTP block (model.layers.80.*) from
    # the trunk weights, same as its plain-text load path. mlx-vlm targets
    # keep sanitize=False (seam 2: GGUF norms already raw).
    _mt = config_dict.get("model_type")
    _install_and_load(
        model.language_model,
        hf_weights,
        hf_kquant_meta,
        log=_log,
        sanitize=(_mt in ("deepseek_v4", "hy_v3")),
        no_alias=owned_names,
        fp32_keep=_FP32_KEEP_BY_MODEL_TYPE.get(_mt, ()),
        source_key=weights_source_key(*pf.shards),
        active_before=active_before,
    )

    # 2b. fused gated-delta verify kernel. The multi-position verify forward is the
    #     MTP round's roofline; fusing conv+silu+rmsnorm+scan-with-states+gated-norm
    #     into one launch removes the serial per-stage chain between those ops. This
    #     fires only on the verify branch (gdn_sink set, S>1), never on S=1 decode.
    #     Enabled for both MoE and dense gated-delta: on the dense hybrid the chain
    #     is ~70% of the gdn layer's per-position cost at verify (the matmuls do not
    #     hide it - that earlier "dense=wash" read was the S=1 decode regime, which
    #     this path never touches), so fusing wins the verify forward (measured
    #     ~7% at M=5, ~15% at M=8 on the 27B hybrid) and is token-lossless vs the
    #     stock verify. Decode fusion stays MoE-only (a genuine wash at S=1).
    #     GMLX_FUSED_GDN=0 kills it.
    if config_dict.get("model_type") in (
        "qwen3_5_moe",
        "qwen3_5_moe_text",
        "qwen3_5",
        "qwen3_5_text",
    ):
        # Owned trees carry the fused routes natively; prepare_gdn arms
        # them post-load. The GMLX_QWEN_OWNED=0 fallback stays bare stock
        # plus the tiled-V rebind above.
        if is_owned_language_model(model):
            prepare_gdn(model)
        _patch_dense_head_verify(model)
    elif config_dict.get("model_type") == "qwen4_exp":
        # Vendored tree: arm the fused GDN decode + verify routes.
        from .qwen4_exp_model import prepare_runtime

        counts = prepare_runtime(model.language_model)
        _log(f"[patch] qwen4_exp: fused GDN decode on {counts['gdn_fused']} "
             f"layers, verify on {counts['gdn_fused_verify']}, b/a cat on "
             f"{counts['gdn_ba_cat']}")
    elif config_dict.get("model_type") in ("gemma4", "gemma4_text"):
        # gemma4 MTP target (assistant drafter): none of the qwen verify
        # levers apply here, so none are installed.
        # - dense-head verify is categorically inapplicable: gemma4 ties
        #   the head to the quantized embedding (embed_tokens.as_linear;
        #   no lm_head attr), and the q6_k head already runs kq's fast
        #   verify path (measured 445-490 GB/s at verify M, ~0.4 ms/round
        #   of headroom at most).
        # - the qwen verify levers live in qwen3_5 modules gemma4 never
        #   routes through; its verify-attention seam is the open front
        #   (verify is 89.7% of the round on the 31B).
        pass

    # 3. drafter - native-head (extracted from this GGUF's MTP block) or
    #    assistant (a separate companion GGUF). Seam 4.
    loadlog.stage("loading drafter")
    loadlog.fact("drafter", "assistant" if assistant else "native-head")
    if assistant:
        if int(config_dict.get("mtp_num_hidden_layers", 0)) >= 1:
            _log(f"[mtp] native MTP head present; using external drafter "
                 f"{os.path.basename(draft_gguf_path)} (pass --native-mtp to "
                 f"use the head)")
        kind = _assistant_kind(_mt, draft_gguf_path)
        if kind == "deepseek4":
            drafter = _load_deepseek4_mtp_drafter(
                draft_gguf_path, model, config_dict, zero_copy=zero_copy, log=_log
            )
        elif kind == "qwen4exp":
            drafter = _load_qwen4exp_mtp_drafter(
                draft_gguf_path, model, config_dict, zero_copy=zero_copy, log=_log
            )
        elif kind == "dflash":
            drafter = _load_dflash_drafter(
                draft_gguf_path, model, config_dict, zero_copy=zero_copy, log=_log
            )
        else:
            drafter = _load_gemma4_assistant_drafter(
                draft_gguf_path, model, zero_copy=zero_copy, log=_log
            )
    else:
        drafter = _load_mtp_drafter(
            arrays,
            kquant_meta,
            arch,
            config_dict,
            model,
            n_head=n_head,
            n_head_kv=n_head_kv,
            source_key=weights_source_key(*pf.shards),
            log=_log,
        )

    # 4. tokenizer (synthesized; multi-EOS wrapped) - same as the text path.
    loadlog.stage("building tokenizer")
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    from .tokenizer import load_tokenizer_from_gguf

    template_override = _resolve_chat_template(chat_template)
    raw_tokenizer = load_tokenizer_from_gguf(
        meta, arch, chat_template_override=template_override
    )
    eos_ids = getattr(raw_tokenizer, "_gguf_eos_token_ids", None)
    tokenizer = TokenizerWrapper(raw_tokenizer, eos_token_ids=eos_ids)

    materialize_module_arrays(model, drafter)
    if wire:
        _wire_big_model(model)
    wait_for_populate(pf.shards, log=_log)

    return model, drafter, config, tokenizer


def _install_stock_qwen35_verify_patches(model) -> None:
    """Full verify patch set for a stock-built qwen3.5/3.6 MTP target.

    ``load_vlm_mtp_model``'s target comes out of mlx_vlm.utils
    construction, which never consults the owned-class selector, so it
    gets the patched regime the text path ran before the owned forwards
    landed. The ``GMLX_QWEN_OWNED=0`` text fallback does not take this
    path: bare stock plus tiled-V is its debugging contract.
    """
    from .gdn_patches import (
        _patch_batched_verify_sdpa,
        _patch_bf16_verify_linear,
        _patch_gated_delta_fused_verify,
    )
    from .qwen35_verify_fold import install_qwen35_verify_fold
    from .ragged_decode import install_unified_ragged_plan

    _patch_gated_delta_fused_verify(model)
    _patch_batched_verify_sdpa()
    install_qwen35_verify_fold()
    install_unified_ragged_plan()
    _patch_bf16_verify_linear()
    loadlog.verbose_print(
        "[build] qwen3.5 stock verify patch set installed (vlm mtp target)"
    )


@loadlog.seeds
def load_vlm_mtp_model(
    gguf_path: str,
    mmproj_path: str,
    *,
    arch: str | None = None,
    draft_gguf_path: str | None = None,
    chat_template: str | None = None,
    hf_source: str | None = None,
    zero_copy: bool = True,
    verbose: bool = False,
):
    """Load a VLM target + MTP drafter for TEXT-ONLY speculative decoding.

    Same MTP engine as ``load_mtp_model``, but the target is a full mlx-vlm VLM
    (K-quant LLM GGUF + float mmproj) whose ``.language_model`` already carries
    the ``speculative_*`` hooks (gemma4, qwen3_5/qwen3_5_moe). Text-only requests
    run through the MTP rounds (which only touch ``.language_model`` + caches);
    image requests stay on the plain VLM path. Returns
    ``(model, drafter, config, tokenizer, processor)``.

    Two drafter shapes, same as ``load_mtp_model``: a gemma4 assistant
    (``draft_gguf_path`` given) or a qwen3.5/3.6 native head (nextn block re-read
    from the LLM GGUF; ``load_vlm_model`` discards the raw arrays). The native path
    also adds the two MTP-only gated-delta patches ``load_vlm_model`` omits
    (mlx-vlm-side tiled-V for the state-capture paths + the fused verify kernel).
    """

    _log = loadlog.verbose_print

    from mlx_lm.tokenizer_utils import TokenizerWrapper

    from .vlm import load_vlm_model

    # 1. target VLM - .language_model is the hook-bearing text class.
    model, config, processor, raw_tokenizer = load_vlm_model(
        gguf_path,
        mmproj_path,
        arch=arch,
        hf_source=hf_source,
        zero_copy=zero_copy,
        verbose=verbose,
        return_tokenizer=True,
    )

    # Template override, same contract as load_mtp_model. The processor
    # snapshotted the tokenizer's template at construction, so set both.
    template_override = _resolve_chat_template(chat_template)
    if template_override is not None:
        raw_tokenizer.chat_template = template_override
        if hasattr(processor, "chat_template"):
            processor.chat_template = template_override

    # 2. the MTP engine drives model.language_model; fail loud if a mlx-vlm bump
    #    drops a hook rather than corrupting decode (mirrors _build_mtp_target).
    lm = getattr(model, "language_model", model)
    missing = [
        h
        for h in ("speculative_logits_from_hidden", "rollback_speculative_cache")
        if not hasattr(lm, h)
    ]
    if missing:
        raise RuntimeError(
            f"VLM language_model {type(lm).__name__} lacks MTP hooks {missing}; "
            "this VLM arch can't run text-only MTP"
        )

    # 3. drafter - assistant (a --draft-gguf companion; gemma4, or a
    #    muse-glimmer dflash) or native-head (nextn block inside the LLM GGUF;
    #    qwen3.5/3.6).
    loadlog.stage("loading drafter")
    loadlog.fact("drafter", "assistant" if draft_gguf_path else "native-head")
    if draft_gguf_path:
        if int((config.get("text_config") or {}).get("mtp_num_hidden_layers", 0)) >= 1:
            _log(f"[mtp] native MTP head present; using external drafter "
                 f"{os.path.basename(draft_gguf_path)} (pass --native-mtp to "
                 f"use the head)")
        if _assistant_kind(config.get("model_type"), draft_gguf_path) == "dflash":
            drafter = _load_dflash_drafter(
                draft_gguf_path, model, config["text_config"],
                zero_copy=zero_copy, log=_log
            )
        else:
            drafter = _load_gemma4_assistant_drafter(
                draft_gguf_path, model, zero_copy=zero_copy, log=_log
            )
    else:
        # Native head: load_vlm_model already loaded the target and applied the
        # mlx-lm tiled-V patch, but it discards the raw GGUF arrays the drafter's
        # nextn block needs. Re-read the LLM wire bytes (mmap, cheap) for the block,
        # then add the two MTP-only gated-delta patches load_vlm_model omits (both
        # are forward-time rebinds, so applying them after the target load is safe):
        #   - mlx-vlm-side tiled-V for the MTP state-capture ops/kernels, and
        #   - the fused gated-delta verify kernel (the MTP round's roofline).
        from .config_synth import synthesize_config

        pf = preflight(gguf_path, arch=arch)
        arch_r = pf.arch
        arrays, kquant_meta, _arch_meta, meta, tensor_shapes = load_gguf_wire_bytes(
            gguf_path, zero_copy=zero_copy, shards=pf.shards
        )
        arrays, kquant_meta, _n = coalesce_split_experts(arrays, kquant_meta)
        config_dict = synthesize_config(meta, tensor_shapes)
        if int(config_dict.get("mtp_num_hidden_layers", 0)) < 1:
            raise ValueError(
                f"{gguf_path}: no native MTP head (nextn) and no --draft-gguf - "
                "this VLM can't run text-only MTP (pass --draft-gguf for a gemma4 "
                "assistant drafter, or use a native-head qwen3.5/3.6 LLM GGUF)"
            )
        n_head = read_int(meta, f"{arch_r}.attention.head_count")
        n_head_kv = first_nonzero_int(meta, f"{arch_r}.attention.head_count_kv")
        if _needs_tiled_v_patch(config_dict):
            _patch_gated_delta_tiled_v()  # idempotent; load_vlm_model already ran it
            if not is_owned_language_model(model):
                _patch_mlxvlm_gated_delta_tiled_v()
        if config_dict.get("model_type") in (
            "qwen3_5_moe",
            "qwen3_5_moe_text",
            "qwen3_5",
            "qwen3_5_text",
        ):
            # Gate on the built tree: mlx_vlm.utils construction never
            # consults the owned-class selector, so this is stock today.
            # The owned branch covers ownership reaching vlm builds.
            if is_owned_language_model(model):
                prepare_gdn(model)
            else:
                _install_stock_qwen35_verify_patches(model)
            _patch_dense_head_verify(model)
        drafter = _load_mtp_drafter(
            arrays,
            kquant_meta,
            arch_r,
            config_dict,
            model,
            n_head=n_head,
            n_head_kv=n_head_kv,
            source_key=weights_source_key(*pf.shards),
            log=_log,
        )

    # 5. wrap the raw GGUF tokenizer (multi-EOS) for generate_speculative.
    loadlog.stage("building tokenizer")
    eos_ids = getattr(raw_tokenizer, "_gguf_eos_token_ids", None)
    tokenizer = TokenizerWrapper(raw_tokenizer, eos_token_ids=eos_ids)

    materialize_module_arrays(model, drafter)
    _wire_big_model(model)

    return model, drafter, config, tokenizer, processor


def _wire_big_model(model) -> None:
    """Sticky wired limit for MTP processes (mlx-lm's ``wired_limit`` policy,
    applied once at load): when the weight bytes crowd the recommended GPU
    working set, wire it so per-token decode doesn't re-page the weight
    buffers. mlx-lm's ``stream_generate`` self-wires around generation, but
    the MTP engines (mlx-vlm ``generate_step``, the owned speculative round)
    run outside that context -- unwired, an 87 GB target decodes at ~0.5
    tok/s vs ~20 wired."""
    try:
        if not mx.metal.is_available():
            return
        max_rec_size = int(
            mx.device_info()["max_recommended_working_set_size"]
        )
        # Unconditional, like mlx-lm's wired_limit (its 0.9x threshold only
        # gates a warning): the limit is a cap, not an allocation.
        mx.set_wired_limit(max_rec_size)
        loadlog.verbose_print(
            f"[wire] wired limit set to {max_rec_size / 2**30:.1f} GiB "
            "(sticky; MTP engines run outside mlx-lm's wired_limit context)"
        )
    except Exception as exc:  # pragma: no cover - platform-dependent
        loadlog.verbose_print(f"[wire] wired-limit skipped: {exc}")
