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
from .envflags import env_int
from .gdn_patches import (
    _needs_tiled_v_patch,
    _patch_batched_verify_sdpa,
    _patch_bf16_verify_linear,
    _patch_dense_head_verify,
    _patch_gated_delta_fused_verify,
    _patch_gated_delta_tiled_v,
    _patch_mlxvlm_gated_delta_tiled_v,
)
from .gguf_meta import first_nonzero_int, read_int
from .loader import (
    _FP32_KEEP_BY_MODEL_TYPE,
    _install_and_load,
    _resolve_chat_template,
    build_model,
    load_gguf_wire_bytes,
    print_inventory,
    remap_arrays,
    remap_gemma4_assistant_arrays,
    remap_mtp_arrays,
)
from .native_fp import _strip_weight
from .populate import maybe_populate_for_load
from .populate import wait_for as wait_for_populate
from .preflight import preflight
from .transforms import coalesce_split_experts


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
    )
    drafter.bind(target)
    from .drafter_protocol import validate_drafter
    validate_drafter(drafter)
    log("[mtp] drafter bound to target embeddings + LM head")
    _patch_draft_head_quantized(drafter)
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

    _install_and_load(drafter, d_weights, d_meta, log=log, sanitize=False)
    # Ordered-embeddings drafters (E2B/E4B) route the LM head through a
    # MaskedEmbedder that reads embed_tokens.weight as a [vocab, hidden] float
    # matrix (gathers candidate rows then a dense matmul). A kquant wire-byte
    # embed table has row width = bytes-per-row (e.g. 272 for a 256-dim Q8_0 row),
    # not hidden, so dequantize it to bf16 before bind() - bind closes over
    # embed_tokens.weight for the head. The centroids Linear stays kquant (it's a
    # plain matmul). The table is small (vocab x hidden_size), so bf16 is cheap.
    if drafter_cfg.get("use_ordered_embeddings"):
        from .modules import KQuantEmbedding

        emb = drafter.model.embed_tokens
        if isinstance(emb, KQuantEmbedding):
            w = kq.dequantize(emb["weight"], emb["scales"], emb.kquant_type).astype(
                mx.bfloat16
            )
            new_emb = nn.Embedding(emb.num_embeddings, emb.dims)
            new_emb.weight = w
            drafter.model.embed_tokens = new_emb
            mx.eval(new_emb.weight)
            log(
                f"[mtp] drafter embed_tokens -> bf16 for ordered-embeddings "
                f"head ({w.shape[0]}x{w.shape[1]})"
            )
    drafter.bind(target)

    from .drafter_protocol import DrafterAdapter, validate_drafter
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
    arrays, kquant_meta, d_arch, _meta, _shapes = load_gguf_wire_bytes(
        draft_gguf_path, zero_copy=zero_copy
    )
    if d_arch != "deepseek4_mtp_support":
        raise ValueError(
            f"{draft_gguf_path}: expected a deepseek4_mtp_support drafter "
            f"GGUF for a deepseek_v4 target, got arch {d_arch!r}"
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
    )
    drafter.bind(target)

    from .drafter_protocol import validate_drafter

    validate_drafter(drafter)
    log("[mtp] drafter bound to target embeddings + LM head")
    _patch_draft_head_quantized(drafter)
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
    _log(f"[arch] {arch}")

    maybe_populate_for_load(pf.shards, log=_log)

    loadlog.stage("reading tensors")
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
        from .discovery import find_mtp_companion

        draft_gguf_path = find_mtp_companion(gguf_path)
        if draft_gguf_path is None:
            raise ValueError(
                "deepseek_v4 MTP needs its companion drafter GGUF (arch "
                "deepseek4_mtp_support); none found next to "
                f"{gguf_path} - pass --draft-gguf <path>."
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
    if _needs_tiled_v_patch(config):
        _patch_gated_delta_tiled_v()
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
        _patch_gated_delta_fused_verify(model)
        _patch_dense_head_verify(model)
        _patch_batched_verify_sdpa()
        _patch_bf16_verify_linear()
    elif config_dict.get("model_type") in ("gemma4", "gemma4_text"):
        # gemma4 MTP target (assistant drafter): no GDN, but the generic
        # verify-width levers should apply -- the M-stationary GEMV-ext
        # dense head alone is ~2 ms/round at M=4 on the 31B (2.8 GB bf16
        # head read at 68% of peak on stock mx.matmul vs ~95% on the ext
        # kernel). All three currently no-op here: their applicability
        # checks match the qwen3_5 target classes only. Adapting them to
        # the gemma4 LanguageModel class is the open MTP front (verify is
        # 89.7% of the round and +37% over the M=1 step on the 31B).
        _patch_dense_head_verify(model)
        _patch_batched_verify_sdpa()
        _patch_bf16_verify_linear()

    # 3. drafter - native-head (extracted from this GGUF's MTP block) or
    #    assistant (a separate companion GGUF). Seam 4.
    loadlog.stage("loading drafter")
    loadlog.fact("drafter", "assistant" if assistant else "native-head")
    if assistant:
        if _mt == "deepseek_v4":
            drafter = _load_deepseek4_mtp_drafter(
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

    if wire:
        _wire_big_model(model)
    wait_for_populate(pf.shards, log=_log)

    return model, drafter, config, tokenizer


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

    # 3. drafter - assistant (a --draft-gguf companion; gemma4) or native-head
    #    (nextn block inside the LLM GGUF; qwen3.5/3.6).
    loadlog.stage("loading drafter")
    loadlog.fact("drafter", "assistant" if draft_gguf_path else "native-head")
    if draft_gguf_path:
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
            _patch_mlxvlm_gated_delta_tiled_v()
        if config_dict.get("model_type") in (
            "qwen3_5_moe",
            "qwen3_5_moe_text",
            "qwen3_5",
            "qwen3_5_text",
        ):
            _patch_gated_delta_fused_verify(model)
            _patch_dense_head_verify(model)
            _patch_batched_verify_sdpa()
            _patch_bf16_verify_linear()
        drafter = _load_mtp_drafter(
            arrays,
            kquant_meta,
            arch_r,
            config_dict,
            model,
            n_head=n_head,
            n_head_kv=n_head_kv,
            log=_log,
        )

    # 5. wrap the raw GGUF tokenizer (multi-EOS) for generate_speculative.
    loadlog.stage("building tokenizer")
    eos_ids = getattr(raw_tokenizer, "_gguf_eos_token_ids", None)
    tokenizer = TokenizerWrapper(raw_tokenizer, eos_token_ids=eos_ids)

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
