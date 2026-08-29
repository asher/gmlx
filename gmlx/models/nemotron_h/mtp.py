# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Nemotron-H MoE MTP: speculative target wrapper + native-head drafter.

Nemotron-3.5-Lightning GGUFs carry the MTP head in-file as block
``num_hidden_layers`` (52): the four ``nextn.*`` extras (``eh_proj`` /
``enorm`` / ``hnorm`` / ``shared_head_norm``) plus a combined block that folds
a dense NoPE attention sub-layer and a MoE sub-layer behind plain residuals -
unlike the trunk, whose blocks are single-mixer. Head forward (llama.cpp
``src/models/nemotron-h-moe.cpp`` graph_mtp, the reference runtime for this
GGUF; upstream runs it via ``--spec-type draft-mtp``):

    x = eh_proj(concat[enorm(embed(ids)), hnorm(h)])
    x = x + attn(attn_norm(x))
    x = x + moe(post_attention_norm(x)) + shexp(post_attention_norm(x))
    logits = lm_head(shared_head_norm(x))

Two conventions differ from the GLM/DeepSeek nextn families:

- ``h`` is the trunk's POST-final-norm hidden (llama.cpp seeds ``t_h_nextn``
  after ``output_norm``), so the target hooks hand the engine the backbone
  output as-is and ``speculative_logits_from_hidden`` is a bare ``lm_head``.
  The rollout likewise chains the head's own post-``shared_head_norm``
  output (llama.cpp feeds the drafter context's ``t_h_nextn`` back in).
- ``shared_head_norm`` is a full LayerNorm (llama.cpp builds it LLM_NORM,
  not LLM_NORM_RMS), weight-only.

The trunk stays the stock mlx-lm ``nemotron_h`` classes; the vendored
backbone subclass only adds the verify-time Mamba2 sink. Verify rollback is
two-part (the glm5_next recurrent precedent): the attention KV leaves trim,
and the Mamba2 layers replay the accepted prefix from the recorded
pre-verify (conv, ssm) state - ``ArraysCache`` state entries are replaced,
not mutated, so the recorded pre-arrays stay valid.

Correctness is drafter-independent: the verify walk emits the target's own
greedy/sampled tokens, so the drafter affects speed (acceptance), never
output. The losslessness gate is the greedy A/B vs plain decode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models import nemotron_h as nh
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

from gmlx.envflags import env_bool
from gmlx.spec.mtp_drafter import QwenMTPDrafter


@dataclass
class NemotronHMTPConfig:
    """Drafter config. ``text_config`` is the target's stock ModelArgs; the
    MTP layer is a combined attention+MoE block (dense NoPE attention).
    ``block_size`` is the block TOTAL (drafts + bonus)."""

    text_config: Any
    block_size: int = 3


@dataclass
class _SpecOutput:
    """Duck-typed output for the owned engine's ``return_hidden`` calls."""

    logits: mx.array
    hidden_states: List[mx.array]
    shared_kv_states: dict = field(default_factory=dict)
    gdn_states: Optional[list] = None


class NemotronHSpecBackbone(nh.NemotronHModel):
    """Stock backbone + an optional verify-time Mamba2 rollback sink.

    The loop is the stock ``NemotronHModel.__call__`` with one addition:
    when ``mamba_sink`` is given, each Mamba block records its cache, the
    pre-update (conv, ssm) state arrays, and its own input, so
    :func:`rollback_verify_sink` can rewind to the accepted prefix.
    """

    @property
    def embed_tokens(self):
        # mlx-vlm-facing alias (MTPTextTarget.get_input_embeddings and the
        # drafter's bind walk expect ``.model.embed_tokens``).
        return self.embeddings

    def __call__(
        self,
        inputs,
        cache: Optional[Any] = None,
        mamba_sink: Optional[list] = None,
    ):
        hidden_states = self.embeddings(inputs)

        if cache is None:
            cache = [None] * len(self.layers)
        attn_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])

        cache_counter = 0
        for layer in self.layers:
            if layer.block_type == "M" or layer.block_type == "*":
                c = cache[cache_counter]
                cache_counter += 1
            else:
                c = None

            if layer.block_type == "*":
                mask = attn_mask
            else:
                mask = ssm_mask

            if mamba_sink is not None and layer.block_type == "M" and c is not None:
                # The mixer replaces c[0]/c[1] (conv, ssm) rather than
                # mutating them, so holding the pre-call arrays is a
                # zero-copy snapshot.
                mamba_sink.append({
                    "layer": layer, "cache": c,
                    "pre": (c[0], c[1]),
                    "inputs": hidden_states, "mask": mask,
                })
                hidden_states = _mamba_stepwise(layer, hidden_states, mask, c)
            elif mamba_sink is not None and layer.block_type == "*" and c is not None:
                hidden_states = _attn_stepwise(layer, hidden_states, c)
            elif mamba_sink is not None and layer.block_type == "E":
                hidden_states = _moe_rowgate(layer, hidden_states)
            else:
                hidden_states = layer(hidden_states, mask=mask, cache=c)

        return self.norm_f(hidden_states)


def _mamba_stepwise(layer, x: mx.array, mask, cache) -> mx.array:
    """Mamba block one position at a time: ``ssm_update`` runs T=1 on the
    sequential kernel and T>1 through the segsum scan, whose rounding
    differs. Stepping keeps verify and replay bit-identical to decode."""
    T = int(x.shape[1])
    if T <= 1:
        return layer(x, mask=mask, cache=cache)
    outs = []
    for t in range(T):
        m = mask[:, t:t + 1] if isinstance(mask, mx.array) else mask
        outs.append(layer(x[:, t:t + 1], mask=m, cache=cache))
    return mx.concatenate(outs, axis=1)


def _attn_stepwise(layer, x: mx.array, cache) -> mx.array:
    """Attention block one position at a time: SDPA's qL=1 and qL>1 kernels
    round differently. A qL=1 causal step over full history needs no mask;
    rollback stays trim-based."""
    T = int(x.shape[1])
    if T <= 1:
        return layer(x, mask=None, cache=cache)
    outs = []
    for t in range(T):
        outs.append(layer(x[:, t:t + 1], mask=None, cache=cache))
    return mx.concatenate(outs, axis=1)


def _moe_rowgate(layer, x: mx.array) -> mx.array:
    """MoE block with the router gate run per row: the gate matmul's
    M=1-vs-M>1 rounding shifts scores enough to flip near-tie argmaxes
    downstream. switch_mlp and shared_experts are M-invariant and stay
    batched (stock ``NemotronHMoE.__call__`` body otherwise)."""
    moe = layer.mixer
    if getattr(moe, "moe_latent_size", None) is not None:
        outs = [layer(x[:, t:t + 1]) for t in range(int(x.shape[1]))]
        return mx.concatenate(outs, axis=1)
    xn = layer.norm(x)
    pairs = [moe.gate(xn[:, t:t + 1]) for t in range(int(xn.shape[1]))]
    inds = mx.concatenate([p[0] for p in pairs], axis=1)
    scores = mx.concatenate([p[1] for p in pairs], axis=1)
    y = moe.switch_mlp(xn, inds)
    y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
    if moe.config.n_shared_experts is not None:
        y = y + moe.shared_experts(xn)
    return x + y


def rollback_verify_sink(sink: list, n: int) -> None:
    """Rewind the Mamba2 caches after an MTP verify forward over S positions
    to the state after its first ``n`` (the accepted prefix): restore the
    recorded pre-verify (conv, ssm) state, then replay the block over the
    accepted prefix. O(n <= block) per layer; the attention KV leaves are
    trimmed by the caller. The replay steps position-wise like the verify
    (see :func:`_mamba_stepwise`), so the rewound state is bit-identical to
    a plain decode over the accepted tokens."""
    for e in sink:
        cache = e["cache"]
        cache[0], cache[1] = e["pre"]
        mask = e["mask"]
        _mamba_stepwise(e["layer"], e["inputs"][:, :n], mask, cache)


class NemotronHSpecLM(nh.Model):
    """Stock nemotron_h ``Model`` + the ``speculative_*`` hooks the owned MTP
    engine probes on its target. The backbone is swapped for the sink-aware
    subclass; numerics are byte-identical to stock."""

    def __init__(self, args):
        super().__init__(args)
        self.backbone = NemotronHSpecBackbone(args)

    @property
    def model(self):
        # mlx-vlm-facing alias: MTPTextTarget and the drafter bind walk read
        # ``language_model.model.embed_tokens``.
        return self.backbone

    # Match mlx-lm's prefill split (chunk n-1, step the last token at T=1):
    # a full-prompt chunk puts the last token through the segsum scan and
    # the Mamba2 state diverges from plain decode.
    prefill_split_last = True

    def chunked_prefill_policy(self, **kwargs):
        # The drafter teacher-forces from the retained per-chunk hiddens,
        # so chunked prefill is always safe for this target.
        return True

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
        n_to_process: Optional[int] = None,
        return_hidden: bool = False,
        return_shared_kv: bool = False,
        **kwargs,
    ):
        # mlx-vlm's chunked prefill calls language_model(inputs=ids, ...) by
        # keyword; the token ids are authoritative (a GGUF text target has
        # no vision features) and shared_kv is never used.
        del inputs_embeds, n_to_process, kwargs
        hidden = self.backbone(inputs, cache=cache)
        logits = self.lm_head(hidden)
        if not (return_hidden or return_shared_kv):
            from mlx_vlm.models.base import LanguageModelOutput

            return LanguageModelOutput(logits=logits)
        return _SpecOutput(logits=logits, hidden_states=[hidden])

    # --- speculative hooks (owned engine contract) -------------------------
    # ``hidden_states`` is the POST-final-norm backbone output - exactly what
    # the drafter's ``hnorm`` consumes (llama.cpp's t_h_nextn seed) - so the
    # from-hidden hooks apply no further norm.

    def speculative_logits_from_hidden(self, hidden: mx.array) -> mx.array:
        return self.lm_head(hidden)

    def speculative_argmax_from_hidden(self, hidden: mx.array) -> mx.array:
        return mx.argmax(self.speculative_logits_from_hidden(hidden), axis=-1)

    def speculative_verify_hidden(self, verify_input: mx.array, prompt_cache):
        """The single verify forward (qL = drafts + 1): backbone only, with
        the Mamba2 rollback sink recorded."""
        sink: list = []
        hidden = self.backbone(verify_input, cache=prompt_cache, mamba_sink=sink)
        return hidden, {}, sink

    def rollback_speculative_cache(
        self, prompt_cache, gdn_states, accepted, block_size: int
    ) -> None:
        """Rewind every layer cache to the accepted prefix of the verify
        block: trim the attention KV leaves (two-phase - all trimmable before
        any mutation, since the shared attention mask derives from one
        layer's offset), then replay the Mamba2 layers from the sink."""
        if isinstance(accepted, mx.array):
            accepted = int(accepted.reshape(-1)[0].item())
        accepted = int(accepted)
        rejected = int(block_size) - accepted - 1
        if rejected <= 0:
            return
        sink = gdn_states or []
        sink_ids = {id(e["cache"]) for e in sink}
        leaves = [
            c for c in prompt_cache
            if c is not None and id(c) not in sink_ids
        ]
        refused = [
            type(leaf).__name__ for leaf in leaves if not leaf.is_trimmable()
        ]
        if refused:
            raise RuntimeError(
                f"nemotron_h MTP rollback: untrimmable cache leaves {refused} "
                f"(rejected={rejected})")
        for leaf in leaves:
            if leaf.trim(rejected) != rejected:
                raise RuntimeError(
                    f"nemotron_h MTP rollback: {type(leaf).__name__}.trim"
                    f"({rejected}) refused after is_trimmable() - cache "
                    f"state is now inconsistent")
        if sink:
            rollback_verify_sink(sink, accepted + 1)


class NemotronHMTPLayer(nn.Module):
    """The MTP decoder layer: a dense NoPE attention sub-layer and a MoE
    sub-layer behind plain residuals (llama.cpp graph_mtp), built from the
    stock trunk modules."""

    def __init__(self, args):
        super().__init__()
        self.self_attn = nh.NemotronHAttention(args)
        self.mlp = nh.NemotronHMoE(args)
        self.input_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.layer_norm_epsilon)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.layer_norm_epsilon)

    def __call__(self, x: mx.array, mask, cache) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class NemotronHMTPDrafter(QwenMTPDrafter):
    """Qwen drafter algorithm over the Nemotron MTP block (full-prompt KV,
    teacher-forced at prefill; decode-time-only trim/accept). Dense NoPE
    attention attends its whole KV, so prompt seeding is uncapped."""

    prefer_requested_block_size = False
    cap_at_configured_depth = True
    # The plain-KV clone path is untested for this head; no sidecar snapshots.
    supports_kv_sidecar = False
    # CLI entry points must route to the owned engine: mlx-vlm's stock MTP
    # round doesn't know the nemotron_h target hooks or the Mamba2 sink.
    requires_owned_engine = True

    def __init__(self, config: NemotronHMTPConfig):
        nn.Module.__init__(self)
        self.config = config
        from gmlx.spec.drafter_protocol import native_block_size

        self._native_block_size = (
            native_block_size(config) or int(config.block_size))
        args = config.text_config

        hidden_size = args.hidden_size
        eps = args.layer_norm_epsilon
        self.fc = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.pre_fc_norm_embedding = nn.RMSNorm(hidden_size, eps=eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(hidden_size, eps=eps)
        self.layers = [NemotronHMTPLayer(args)]
        # llama.cpp builds shared_head_norm as LLM_NORM (full LayerNorm),
        # not RMS; the GGUF ships weight only.
        self.norm = nn.LayerNorm(hidden_size, eps=eps, bias=False)

        # llama.cpp's draft-mtp chains the drafter context's own t_h_nextn
        # (the post-shared_head_norm output) as the next rollout hidden -
        # consistent with the seed, which is the trunk's post-final-norm
        # hidden. GMLX_MTP_POSTNORM_FEED=0 A/Bs the pre-norm feed.
        self._postnorm_feed = env_bool("GMLX_MTP_POSTNORM_FEED", True)

        # Bound to the target at reset(): the head shares embeddings + head.
        self._input_embed = None
        self._input_embed_scale: float = 1.0
        self._lm_head_fn = None

        # Decode-time-only state: own KV + the precomputed next-round seed.
        self._cache: List[Any] = []
        self._seed_token: Optional[mx.array] = None
        self._seed_hidden: Optional[mx.array] = None
        self._round_appended = 0

        self.accept_lens: List[float] = []
        self.draft_lens: List[int] = []

    def make_cache(self, left_padding: Optional[List[int]] = None) -> List[Any]:
        if left_padding is not None:
            raise NotImplementedError(
                "NemotronHMTPDrafter is B=1 only (v1): the MoE target caps "
                "speculation width at 1 anyway")
        from mlx_lm.models.cache import KVCache as _KVCache

        from gmlx.cache.compat import construction_cache_module

        kv_cls = getattr(construction_cache_module(), "KVCache", _KVCache)
        return [kv_cls() for _ in self.layers]

    def inject_rows(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "NemotronHMTPDrafter is B=1 only (v1): no batched-row injection")

    def _forward(self, tokens: mx.array, hidden: mx.array,
                 cache: Optional[List[Any]] = None) -> mx.array:
        """Run the head over (tokens, target post-final-norm hidden); return
        the PRE-shared-head-norm output. NoPE attention: the mask offset
        comes from the head's own KV length, there is no rope frame. cache
        overrides the head's own list (seed streaming), never swaps it."""
        embed = self._input_embed(tokens.astype(mx.int32))
        h = mx.concatenate(
            [self.pre_fc_norm_embedding(embed),
             self.pre_fc_norm_hidden(hidden)],
            axis=-1,
        )
        h = self.fc(h)
        caches = self._cache if cache is None else cache
        c = caches[0] if caches else None
        mask = create_attention_mask(h, c)
        return self.layers[0](h, mask, c)


# Native-head remap: the closed tensor set of the GGUF MTP block (16
# weight-bearing entries at blk.{num_hidden_layers}; verified against the
# Nemotron-3.5-Lightning dump). ``token_embd`` / ``output`` / ``output_norm``
# only exist in a standalone companion GGUF and are skipped - the drafter
# binds the target's embeddings + LM head at reset. ``ffn_latent_down/up``
# cover a latent-MoE variant (Nemotron-3-Super shape), absent on Lightning.
_NEMOTRON_MTP_MAP = {
    "nextn.eh_proj": "fc.weight",
    "nextn.enorm": "pre_fc_norm_embedding.weight",
    "nextn.hnorm": "pre_fc_norm_hidden.weight",
    "nextn.shared_head_norm": "norm.weight",
    "attn_norm": "layers.0.input_layernorm.weight",
    "attn_q": "layers.0.self_attn.q_proj.weight",
    "attn_k": "layers.0.self_attn.k_proj.weight",
    "attn_v": "layers.0.self_attn.v_proj.weight",
    "attn_output": "layers.0.self_attn.o_proj.weight",
    "post_attention_norm": "layers.0.post_attention_layernorm.weight",
    "ffn_gate_inp": "layers.0.mlp.gate.weight",
    "ffn_up_exps": "layers.0.mlp.switch_mlp.fc1.weight",
    "ffn_down_exps": "layers.0.mlp.switch_mlp.fc2.weight",
    "ffn_latent_down": "layers.0.mlp.fc1_latent_proj.weight",
    "ffn_latent_up": "layers.0.mlp.fc2_latent_proj.weight",
    "ffn_up_shexp": "layers.0.mlp.shared_experts.up_proj.weight",
    "ffn_down_shexp": "layers.0.mlp.shared_experts.down_proj.weight",
}
# Raw fp32 params (no ".weight" suffix on the GGUF side).
_NEMOTRON_MTP_RAW = {
    "exp_probs_b.bias": "layers.0.mlp.gate.e_score_correction_bias",
}
# Companion-GGUF globals the drafter shares from the target instead.
_NEMOTRON_MTP_SHARED = ("token_embd.weight", "output.weight",
                        "output_norm.weight")


def remap_nemotron_mtp_arrays(
    arrays: dict, kquant_meta: dict, *,
    first_mtp_block: int, n_head: int, n_head_kv: int,
):
    """Remap the Nemotron MTP block onto the ``NemotronHMTPDrafter`` param
    tree. Self-contained and closed: an unknown ``blk.{N}.*`` tensor in the
    MTP block is a hard error (converter drift must surface at load, not as
    an unfilled param). Q/K undo the llama.cpp rope permute exactly like the
    trunk remap (the converter permutes every attention block, the MTP block
    included, even though the model is NoPE)."""
    from gmlx.load.transforms import qk_permute_wire

    def _strip_weight(name: str) -> str:
        return name[: -len(".weight")] if name.endswith(".weight") else name

    hf_weights: dict[str, mx.array] = {}
    hf_kquant_meta: dict[str, str] = {}
    stats = {"mapped": 0, "skipped": 0, "qk_permute_applied": 0}
    prefix = f"blk.{int(first_mtp_block)}."
    for name, arr in arrays.items():
        if name.endswith(".scales") or name.endswith(".biases"):
            continue
        if not name.startswith(prefix):
            if name in _NEMOTRON_MTP_SHARED:
                stats["skipped"] += 1
            continue
        rest = name[len(prefix):]
        raw_target = _NEMOTRON_MTP_RAW.get(rest)
        if raw_target is not None:
            hf_weights[raw_target] = arr
            stats["mapped"] += 1
            continue
        base = _strip_weight(rest)
        target = _NEMOTRON_MTP_MAP.get(base)
        if target is None:
            raise RuntimeError(
                f"nemotron MTP remap: unknown tensor {name!r} "
                f"(converter drift?)")
        codec = kquant_meta.get(name)
        if base in ("attn_q", "attn_k"):
            nh_for = n_head_kv if base == "attn_k" else n_head
            hf_weights[target] = qk_permute_wire(arr, nh_for)
            stats["qk_permute_applied"] += 1
        else:
            hf_weights[target] = arr
        if codec is not None:
            hf_weights[_strip_weight(target) + ".scales"] = arrays[
                _strip_weight(name) + ".scales"]
            hf_kquant_meta[target] = codec
        stats["mapped"] += 1
    return hf_weights, hf_kquant_meta, stats
