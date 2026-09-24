"""Tiny random-weight models of every owned architecture, loaded the way
the GGUF loader leaves them: K-quant leaves, the loader's float cast and
its post-load installs. Also the tools a training-step test needs around
them: GGUF base tensor names, eval-mode logits and a record/replay of the
selections a forward makes."""
from __future__ import annotations

import contextlib
import io
import re
import re._parser as sre
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

# Some model modules build their Metal kernels at import and skip them on a
# CPU default device, so a CPU-pinned test that imports one first would
# leave it without them. Import them at collection, on the default device.
import gmlx.models.glm5_next.model  # noqa: F401
import gmlx.models.hy_v4.model  # noqa: F401


def _qwen35(**over):
    cfg = dict(
        model_type="qwen3_5", hidden_size=256, intermediate_size=512, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64, rms_norm_eps=1e-6,
        vocab_size=512, max_position_embeddings=4096, linear_num_value_heads=4,
        linear_num_key_heads=2, linear_key_head_dim=64, linear_value_head_dim=64,
        linear_conv_kernel_dim=4, tie_word_embeddings=False, full_attention_interval=4,
        kv_head_layout="tiled",
        rope_parameters={"type": "default", "mrope_section": [3, 3, 2], "rope_theta": 1e7,
                         "partial_rotary_factor": 0.25})
    cfg.update(over)
    return cfg


_ROPE = {"rope_theta": 10000.0, "rope_type": "default"}

CONFIGS = {
    "qwen3_moe": dict(
        model_type="qwen3_moe", hidden_size=256, num_hidden_layers=2, intermediate_size=512,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64, num_experts=8,
        num_experts_per_tok=4, decoder_sparse_step=1, mlp_only_layers=[],
        moe_intermediate_size=256, rms_norm_eps=1e-6, vocab_size=512, rope_theta=1e6,
        tie_word_embeddings=False, max_position_embeddings=4096, norm_topk_prob=True),
    "muse_glimmer": dict(
        model_type="muse_glimmer", hidden_size=256, intermediate_size=512, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64, vocab_size=512,
        layer_types=["sliding_attention"] * 3 + ["full_attention"], sliding_window=512,
        final_logit_softcapping=20.0, output_multiplier=0.19611613, post_norm_eps=1e-8,
        rms_norm_eps=1e-6, rope_parameters=_ROPE, rope_theta=10000.0,
        max_position_embeddings=1024, tie_word_embeddings=False),
    "hy_v3": dict(
        model_type="hy_v3", hidden_size=256, intermediate_size=512, expert_hidden_dim=256,
        num_hidden_layers=3, first_k_dense_replace=1, num_attention_heads=4,
        num_key_value_heads=2, head_dim=64, num_experts=8, num_experts_per_tok=4,
        num_shared_experts=1, moe_router_enable_expert_bias=True, moe_router_use_sigmoid=True,
        route_norm=True, router_scaling_factor=2.826, qk_norm=True, enable_lm_head_fp32=False,
        mtp_num_hidden_layers=1, num_nextn_predict_layers=1, rms_norm_eps=1e-6,
        rope_parameters=_ROPE, rope_theta=10000.0, max_position_embeddings=1024,
        tie_word_embeddings=False, vocab_size=512),
    "glm5_next": dict(
        model_type="glm5_next", vocab_size=512, hidden_size=256, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=1, intermediate_size=512, rms_norm_eps=1e-5,
        layer_types=["linear_attention", "full_attention"] * 2,
        kda_head_dim=64, ssm_conv_kernel=4, kda_gate_lower_bound=-5.0,
        q_lora_rank=256, kv_lora_rank=256, qk_nope_head_dim=64, qk_rope_head_dim=0,
        v_head_dim=64, index_n_heads=2, index_head_dim=64, index_topk=8, index_kpool=4,
        index_knorm_eps=1e-6, n_routed_experts=8, num_experts_per_tok=4,
        moe_intermediate_size=256, n_shared_experts=1, first_k_dense_replace=1,
        routed_scaling_factor=2.5, norm_topk_prob=True, scoring_func="sigmoid",
        swiglu_limit=10.0, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
        max_position_embeddings=4096, tie_word_embeddings=False),
    "kimi_k3": dict(
        model_type="kimi_k3", vocab_size=512, hidden_size=256, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=1, intermediate_size=512, rms_norm_eps=1e-5,
        layer_types=["linear_attention", "full_attention"] * 2,
        kda_head_dim=64, ssm_conv_kernel=4, kda_gate_lower_bound=-5.0,
        q_lora_rank=256, kv_lora_rank=256, qk_nope_head_dim=64, qk_rope_head_dim=32,
        v_head_dim=64, num_experts=8, num_experts_per_tok=4, moe_intermediate_size=256,
        num_shared_experts=1, first_k_dense_replace=1, routed_scaling_factor=1.0,
        moe_renormalize=True, routed_expert_hidden_size=256, has_routed_norm=True,
        situ_beta=4.0, situ_linear_beta=25.0, attn_res_block_size=2,
        max_position_embeddings=4096, tie_word_embeddings=False),
    "hy_v4": dict(
        model_type="hy_v4", vocab_size=512, hidden_size=256, intermediate_size=512,
        moe_intermediate_size=256, num_hidden_layers=3, num_attention_heads=4,
        q_lora_rank=256, kv_lora_rank=256, qk_nope_head_dim=64, qk_rope_head_dim=32,
        v_head_dim=64, n_routed_experts=8, num_experts_per_tok=4, n_shared_experts=1,
        first_k_dense_replace=1, rms_norm_eps=1e-6, hc_mult=4, hc_eps=1e-6,
        hc_magnitude=2.0, index_n_heads=2, index_head_dim=64, index_topk=8,
        index_is_full=[1, 0, 1], swiglu_limit=10.0, routed_scaling_factor=2.827),
    "minimax_m3": dict(
        model_type="minimax_m3", hidden_size=256, intermediate_size=256,
        dense_intermediate_size=512, shared_intermediate_size=256, num_attention_heads=4,
        num_key_value_heads=2, num_hidden_layers=3, num_local_experts=8,
        num_experts_per_tok=4, rms_norm_eps=1e-6, rope_theta=5e6, rotary_dim=32,
        vocab_size=512, head_dim=64, mlp_layer_types=["dense", "sparse", "sparse"],
        use_sparse_attention=True, sparse_index_dim=64, sparse_num_index_heads=2,
        sparse_topk_blocks=2, sparse_block_size=8, sparse_local_block=1),
    "qwen4_exp": dict(
        model_type="qwen4_exp", hidden_size=256, intermediate_size=512, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64, full_attention_interval=4,
        compress_ratios=[0, 0, 0, 4], indexer_budget=8, indexer_head_dim=64,
        indexer_n_heads=2, hc_count=4, hc_lowrank=32, kv_head_layout="tiled",
        linear_conv_kernel_dim=4, linear_key_head_dim=64, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_value_head_dim=64, num_experts=8,
        num_experts_per_tok=4, moe_intermediate_size=256, shared_expert_intermediate_size=256,
        norm_topk_prob=True, mrope_section=[3, 3, 2], partial_rotary_factor=0.25,
        ple_conv_kernel=4, ple_embed_dim=8, ple_eos_token_id=1, ple_image_token_id=2,
        ple_head_offsets=list(range(0, 800, 50)), ple_head_vocab_sizes=[50] * 16,
        ple_heads_per_ngram=8, ple_layer_ids=[1],
        ple_layer_multipliers=[23703573157769, 20109073645365, 8052911324071],
        ple_ngram_size=3, ple_table_rows=800, rms_norm_eps=1e-6,
        rope_parameters={"mrope_section": [3, 3, 2], "partial_rotary_factor": 0.25,
                         "rope_theta": 1e7, "type": "default"},
        rope_theta=1e7, max_position_embeddings=1024, tie_word_embeddings=False,
        vocab_size=512),
    "qwen3_next": dict(
        model_type="qwen3_next", hidden_size=256, intermediate_size=512, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64, full_attention_interval=4,
        gdn_split_layout=True, kv_head_layout="grouped", linear_conv_kernel_dim=4,
        linear_key_head_dim=64, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_value_head_dim=64, decoder_sparse_step=1, mlp_only_layers=[], num_experts=8,
        num_experts_per_tok=4, moe_intermediate_size=256, shared_expert_intermediate_size=256,
        norm_topk_prob=True, partial_rotary_factor=0.25, rms_norm_eps=1e-6,
        rope_theta=10000.0, max_position_embeddings=1024, tie_word_embeddings=False,
        vocab_size=512),
    "qwen3_5": _qwen35(),
    "qwen3_5_moe": _qwen35(
        model_type="qwen3_5_moe", num_experts=8, num_experts_per_tok=4, decoder_sparse_step=1,
        moe_intermediate_size=256, shared_expert_intermediate_size=256, norm_topk_prob=True),
    "qwen3_5_hadamard": _qwen35(),
}

# ssm_alpha and ssm_beta ship float in the GGUFs, so the loader can
# concatenate them for the fused decode.
_GDN_BA = (r"in_proj_[ab]$",)


@dataclass(frozen=True)
class Case:
    gguf_arch: str
    float_patterns: tuple = ()
    hadamard: bool = False


CASES = {
    "qwen3_moe": Case("qwen3moe"),
    "muse_glimmer": Case("muse-glimmer"),
    "hy_v3": Case("hy_v3", (r"router\.gate$",)),
    "glm5_next": Case("glm5next"),
    "kimi_k3": Case("kimi-k3"),
    "hy_v4": Case("hyv4"),
    "minimax_m3": Case("minimax-m3", (r"block_sparse_moe\.gate$",)),
    "qwen4_exp": Case("qwen4exp", _GDN_BA),
    "qwen3_next": Case("qwen3next"),
    "qwen3_5": Case("qwen35", _GDN_BA),
    "qwen3_5_moe": Case("qwen35moe", _GDN_BA),
    "qwen3_5_hadamard": Case("qwen35", hadamard=True),
}


@contextlib.contextmanager
def process_patches():
    """Undo on exit the module-global patches a load applies: the tiled-V
    gated-delta rewrite would otherwise reach every later load."""
    from mlx_lm.models import gated_delta as gd

    saved = dict(vars(gd))
    try:
        yield
    finally:
        for k in [k for k in vars(gd) if k not in saved]:
            delattr(gd, k)
        for k, v in saved.items():
            if vars(gd).get(k) is not v:
                setattr(gd, k, v)


def _randomize_zero_params(model, seed=0, std=0.05):
    """Small noise in every all-zero float parameter, so a zero-initialized
    weight cannot fake a zero gradient."""
    r = np.random.default_rng(seed)
    upd = []
    for k, v in tree_flatten(model.parameters()):
        if mx.issubdtype(v.dtype, mx.floating) and v.size and float(mx.abs(v).max()) == 0.0:
            noise = r.standard_normal(v.shape).astype(np.float32) * std
            upd.append((k, mx.array(noise).astype(v.dtype)))
    if upd:
        model.load_weights(upd, strict=False)


def _quantize(model, config, float_patterns):
    """K-quantize the way a GGUF quantizer leaves the model: routers and the
    loader's fp32 keep list stay float, q4_k where the input width takes
    it, q8_0 on multiples of 32. Every other path stays float."""
    import mlx_kquant.convert as conv
    from mlx_kquant.recipes import TensorRole, classify_path

    from gmlx.load.loader import _FP32_KEEP_BY_MODEL_TYPE

    keep = _FP32_KEEP_BY_MODEL_TYPE.get(config["model_type"], ())
    over = {}
    for p, m in model.named_modules():
        if not hasattr(m, "to_quantized") or "weight" not in m:
            continue
        if classify_path(p) == TensorRole.MOE_ROUTER or any(re.search(f, p) for f in float_patterns):
            continue
        if any(k in "." + p + ".weight" for k in keep):
            continue
        in_dims = m.weight.shape[-1]
        if in_dims % 256 == 0:
            over[p] = "q4_k"
        elif in_dims % 32 == 0:
            over[p] = "q8_0"

    # quantize_model applies overrides to classified paths only
    orig = conv.classify_tensors
    conv.classify_tensors = lambda m: {p: classify_path(p) or TensorRole.FFN for p in over}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            model, _ = conv.quantize_model(model, config, default_codec="q8_0", overrides=over)
    finally:
        conv.classify_tensors = orig
    return model


def _cast_like_loader(model, model_type):
    from gmlx.load.dtypes import activation_dtype
    from gmlx.load.loader import _FP32_KEEP_BY_MODEL_TYPE

    act = activation_dtype()
    keep = _FP32_KEEP_BY_MODEL_TYPE.get(model_type, ())
    upd = []
    for k, v in tree_flatten(model.parameters()):
        if v.dtype not in (mx.float32, mx.float16):
            continue
        if any(s in k for s in keep):
            if v.dtype != mx.float32:
                upd.append((k, v.astype(mx.float32)))
        elif v.dtype != act:
            upd.append((k, v.astype(act)))
    if upd:
        model.load_weights(upd, strict=False)


_FOLD_NAMES = ["output.weight"] + [
    f"blk.{i}.{n}.weight" for i in range(3)
    for n in ("attn_qkv", "attn_gate", "ssm_out", "ffn_gate", "ffn_up", "ffn_down")] + [
    f"blk.3.{n}.weight" for n in ("attn_q", "attn_k", "attn_v", "attn_output",
                                   "ffn_gate", "ffn_up", "ffn_down")]


def _hadamard_fold(model):
    """The loader's fold for a qwen35 GGUF carrying a Hadamard header."""
    from gmlx.load.hadamard import hadamard_targets_for
    from gmlx.load.hadamard_modules import install_hadamard_modules
    from gmlx.upstream.hadamard_share import install_hadamard_sharing

    r = np.random.default_rng(7)
    widths = (256, 512)
    shapes = {"output.weight": (256, 512), "token_embd.weight": (256, 512)}
    for n in _FOLD_NAMES[1:]:
        shapes[n] = (512 if n.split(".")[2] == "ffn_down" else 256, 0)
    meta = {
        "general.architecture": "qwen35",
        "qwen35.ssm.time_step_rank": 4, "qwen35.ssm.group_count": 2,
        "prism.hadamard.version": 1, "prism.hadamard.block_size": 256,
        "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
        "prism.hadamard.axis": "input-last-dimension", "prism.hadamard.sign_mode": "explicit",
        "prism.hadamard.weight_names": _FOLD_NAMES,
        "prism.hadamard.inverse_weight_names": ["token_embd.weight"],
        "prism.hadamard.gdn_v_grouped": True,
        "prism.hadamard.sign_widths": list(widths),
        "prism.hadamard.sign_values": np.concatenate(
            [r.choice(np.array([-1, 1]), size=w) for w in widths]).tolist(),
    }
    targets = hadamard_targets_for(meta, "qwen35", shapes, target_prefix="language_model")
    folded = install_hadamard_modules(model, targets)
    assert folded > 0
    install_hadamard_sharing(model)


def _post_load_installs(model, config):
    """The loader's installs after load_weights, in its order."""
    from gmlx.load.invariant_linear import install_batch_invariant_linears
    from gmlx.load.modules import install_fused_moe_glu, install_hyv3_shexp_fold
    from gmlx.upstream.occupancy_fuse import install_occupancy_fuse
    from gmlx.upstream.qkv_fuse import install_fused_qkv

    install_fused_moe_glu(model)
    install_batch_invariant_linears(model)
    install_hyv3_shexp_fold(model)
    install_fused_qkv(model)
    install_occupancy_fuse(model)
    model.eval()
    mt = config["model_type"]
    if mt in ("qwen3_5", "qwen3_5_moe"):
        from gmlx.load.loader import _patch_gated_delta_fused_decode
        _patch_gated_delta_fused_decode(model)
    if mt == "qwen4_exp":
        from gmlx.models.qwen4_exp.model import prepare_runtime
        prepare_runtime(model)


def build(name):
    """(model, config, case) for one architecture, as the loader leaves it.
    Run inside ``process_patches``."""
    from gmlx.load.loader import build_model
    from gmlx.upstream.gdn_patches import (
        _needs_tiled_v_patch,
        _patch_gated_delta_tiled_v,
        _patch_qwen3next_split_gdn,
    )

    case = CASES[name]
    mx.random.seed(0)
    model, config = build_model(dict(CONFIGS[name]))
    if config.get("gdn_split_layout"):
        _patch_qwen3next_split_gdn(model)
    if _needs_tiled_v_patch(config):
        _patch_gated_delta_tiled_v()
    mx.eval(model.parameters())
    _randomize_zero_params(model)
    model = _quantize(model, config, case.float_patterns)
    _cast_like_loader(model, config["model_type"])
    if case.hadamard:
        _hadamard_fold(model)
    _post_load_installs(model, config)
    mx.eval(model.parameters())
    return model, config, case


def text_root(model):
    return getattr(model, "language_model", model)


def num_layers(config):
    return config.get("num_hidden_layers") or config["text_config"]["num_hidden_layers"]


def vocab_size(config):
    return config.get("vocab_size") or config["text_config"]["vocab_size"]


# GGUF base tensor names


def _expand(parsed, bid):
    """Every string a parsed override regex matches, digit runs set to bid."""
    outs = [""]
    for op, av in parsed:
        op = str(op)
        if op == "LITERAL":
            outs = [o + chr(av) for o in outs]
        elif op == "SUBPATTERN":
            subs = _expand(av[-1], bid)
            outs = [o + s for o in outs for s in subs]
        elif op in ("MAX_REPEAT", "MIN_REPEAT"):
            lo, hi, sub = av
            inner = [str(o) for o, _ in sub]
            if inner == ["IN"]:
                outs = [o + str(bid) for o in outs]
            elif lo == 0 and hi == 1:
                subs = [""] + _expand(sub, bid)
                outs = [o + s for o in outs for s in subs]
            else:
                raise ValueError(f"repeat over {inner}")
        elif op == "IN":
            items = [(str(o), a) for o, a in av]
            if items and items[0][0] == "CATEGORY":
                outs = [o + str(bid) for o in outs]
            else:
                chars = [chr(a) for o, a in items if o == "LITERAL"]
                outs = [o + c for o in outs for c in chars]
        elif op == "BRANCH":
            subs = [s for b in av[1] for s in _expand(b, bid)]
            outs = [o + s for o in outs for s in subs]
        elif op != "AT":
            raise ValueError(op)
    return outs


def base_names(model, arch, n_layers):
    """The GGUF names a base file of ``arch`` would carry for the modules of
    ``model``: gguf-py's templates plus the per-arch override patterns,
    kept where parse_gguf_name maps them onto a module."""
    from gguf.constants import TENSOR_NAMES

    from gmlx.load import remap

    alias = remap.ARCH_ALIAS.get(arch)
    bids = range(n_layers + 2)
    cands = set()
    for tmpl in TENSOR_NAMES.values():
        if "{bid}" in tmpl:
            cands.update(tmpl.format(bid=b) + ".weight" for b in bids)
        else:
            cands.add(tmpl + ".weight")
    pats = [e[0] for e in remap.ARCH_PRIORITY_OVERRIDES.get(alias, [])]
    pats += [e[0] for e in remap.EXTRA_OVERRIDES.get(alias, [])]
    for pat in pats:
        parsed = sre.parse(pat.pattern)
        for b in bids:
            for s in _expand(parsed, b):
                cands.add(s if s.endswith(".weight") else s + ".weight")
    paths = {p for p, _ in model.named_modules()}
    chosen = {}
    for name in sorted(cands):
        dec = remap.parse_gguf_name(arch, name)
        if dec.kind != remap.RemapDecision.KIND_MAP or not (dec.hf_name or "").endswith(".weight"):
            continue
        path = dec.hf_name[: -len(".weight")]
        if path in paths:
            chosen.setdefault(path, name)
    return sorted(set(chosen.values()))


# Eval-mode logits with replayable selections


class Selections:
    """Records the ids every selection of one forward picks (mx.argpartition,
    argsort, argmax, argmin and the kq top-k kernels) per function, and
    replays them in a later forward: where the later forward picks other
    ids the recorded output stands in. Two forwards that differ by bf16
    rounding then route every token the same way. Compilation is off
    inside so every call reaches the wrappers."""

    NAMES = ("argpartition", "argsort", "argmax", "argmin")
    KQ = ("moe_router_topk", "dsa_topk_indices")

    def __init__(self):
        self.logs = {}
        self.pos = {}
        self.mode = None
        self.mismatch = 0

    def __enter__(self):
        import mlx_kquant as kq

        self._orig = [(mx, n, getattr(mx, n)) for n in self.NAMES]
        self._orig += [(kq, n, getattr(kq, n)) for n in self.KQ if hasattr(kq, n)]
        for mod, n, f in self._orig:
            setattr(mod, n, self._wrap(n, f))
        mx.disable_compile()
        return self

    def __exit__(self, *exc):
        for mod, n, f in self._orig:
            setattr(mod, n, f)
        mx.enable_compile()

    def start(self, mode):
        self.mode = mode
        self.pos = {}

    def aligned(self):
        return self.pos.keys() == self.logs.keys() and all(
            self.pos[n] == len(v) for n, v in self.logs.items())

    def _wrap(self, name, fn):
        def ids(o):
            return o[0] if isinstance(o, (tuple, list)) else o

        def call(*args, **kwargs):
            out = fn(*args, **kwargs)
            if self.mode == "record":
                mx.eval(out)
                self.logs.setdefault(name, []).append(out)
            elif self.mode == "replay":
                i = self.pos.get(name, 0)
                self.pos[name] = i + 1
                log = self.logs.get(name, [])
                if i >= len(log) or ids(log[i]).shape != ids(out).shape:
                    self.mismatch += 1
                elif not mx.array_equal(ids(log[i]), ids(out)).item():
                    return log[i]
            return out
        return call


def logits_of(out):
    """The logits of a forward that may return an output object."""
    return out if isinstance(out, mx.array) else out.logits


def eval_logits(model, tokens, sel=None, mode=None):
    """Eval-mode float32 logits of a full forward, and of a one-token
    cached decode after a prefill of the rest."""
    from mlx_lm.models.cache import make_prompt_cache

    model.eval()
    if sel is not None:
        sel.start(mode)
    full = logits_of(model(tokens)).astype(mx.float32)
    cache = make_prompt_cache(model)
    model(tokens[:, :-1], cache=cache)
    dec = logits_of(model(tokens[:, -1:], cache=cache)).astype(mx.float32)
    mx.eval(full, dec)
    if sel is not None:
        sel.mode = None
    return full, dec


def rel(a, b):
    """max |a - b| over max |a|."""
    return float(mx.abs(a - b).max() / mx.maximum(mx.abs(a).max(), 1e-12))
