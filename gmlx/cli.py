"""``gmlx run`` - generate text from, or benchmark, a GGUF K-quant model.

Loads any text-only GGUF directly (no conversion) into a kquant-native mlx-lm
model and either generates a completion, runs a prefill/decode benchmark, or
prints a load inventory (``--report-only``).
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def _prog() -> str:
    """The command name shown in help/usage text - always ``gmlx`` (the console
    script; also the fallback when run as ``python -m gmlx``)."""
    return "gmlx"


def _parse_int_list(s: str, *, flag: str) -> list[int]:
    try:
        out = [int(x) for x in s.split(",") if x.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"{flag}: not a comma-separated int list: {s!r} ({e})"
        )
    if not out:
        raise argparse.ArgumentTypeError(f"{flag}: empty list")
    return out


def add_verbosity_arg(ap: argparse.ArgumentParser) -> None:
    """Add ``-v/--verbose``. Shared by ``run`` and ``chat``."""
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Full load diagnostics instead of the progress spinner.",
    )


def mass_share(value) -> float:
    """Argparse type (and config normalizer helper) for a gate-mass share:
    a float in (0, 1]. Rejects at parse time so a bad value fails before the
    model load, not after it."""
    try:
        p = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"not a number: {value!r}")
    if not 0.0 < p <= 1.0:
        raise argparse.ArgumentTypeError(
            f"must be a mass share in (0, 1], got {value}")
    return p


def add_moe_expert_args(ap: argparse.ArgumentParser) -> None:
    """Add the lossy MoE fan-out flags. Shared by ``run`` and ``chat``; they
    only act on streamed MoE layers (--stream-experts / --stream-cpu)."""
    grp = ap.add_argument_group("MoE experts (lossy speed levers)")
    grp.add_argument(
        "--moe-experts",
        type=int,
        default=None,
        metavar="K",
        help="Lossy: cap the router at K experts per token on the "
        "streamed MoE layers (--stream-experts / --stream-cpu) - fewer "
        "experts means fewer expert bytes per prefill chunk / decode "
        "token, at a quality cost (outputs differ from the trained "
        "router). Composes with --moe-expert-mass.",
    )
    lossy = grp.add_mutually_exclusive_group()
    lossy.add_argument(
        "--moe-expert-mass",
        type=mass_share,
        default=None,
        metavar="P",
        help="Lossy: adaptive experts-per-token on the streamed MoE "
        "layers - each token keeps only the smallest set of its routed "
        "experts covering share P (0 < P <= 1) of the router's gate "
        "mass, so confident tokens read fewer expert bytes. Run "
        "--moe-expert-probe first to size P.",
    )
    lossy.add_argument(
        "--moe-expert-probe",
        action="store_true",
        help="Lossless probe for --moe-expert-mass: run at the trained "
        "fan-out while recording how many experts each token needed at "
        "candidate P values; prints decode and prefill tables at exit.",
    )
    grp.add_argument(
        "--moe-miss-shed",
        type=mass_share,
        default=None,
        metavar="P",
        help="Lossy: at decode, drop routed experts that would demand-miss "
        "the expert arena, lowest scores first, keeping at least share P "
        "(0 < P <= 1) of each token's gate mass. Targets IO stalls "
        "directly; resident experts are never dropped. Composes with "
        "--moe-expert-mass.",
    )
    grp.add_argument(
        "--moe-layer-shed",
        type=float,
        default=None,
        metavar="P",
        help="Lossy: at decode, skip each streamed MoE layer's routed "
        "experts with probability P (0 < P < 1) per token; the shared "
        "expert still runs on shed layers.",
    )


def add_speculative_args(ap: argparse.ArgumentParser) -> None:
    """Add the MTP speculative-decoding flags. Shared by ``run`` and ``chat`` so
    the two parsers can't drift: this group was once ``run``-only, so a config's
    ``speculative: true`` was silently dropped on ``chat`` (``_apply_resolved_to_args``
    skips a setting whose dest the verb's parser never defined). Composing both
    parsers from one builder keeps the surface - and the config plumbing - in sync."""
    ap = ap.add_argument_group("speculative decoding (MTP)")
    ap.add_argument(
        "--speculative",
        "--mtp",
        action="store_true",
        default=None,  # None = unset (auto native-head detect); True/False = explicit
        help="Force MTP speculative decoding on. Native-head models "
        "(qwen3.5/3.6 'nextn') need no companion; gemma4 needs "
        "a --draft-gguf assistant. Native-head GGUFs auto-enable MTP even "
        "without this flag; pass it to force the path when a sampler flag "
        "would otherwise keep plain decoding (those flags are then dropped).",
    )
    ap.add_argument(
        "--no-speculative",
        "--no-mtp",
        action="store_true",
        help="Disable MTP speculative decoding. Overrides the native-head "
        "auto-enable and config 'speculative: true'; forces plain decoding.",
    )
    ap.add_argument(
        "--draft-gguf",
        default=None,
        metavar="PATH",
        help="Separate assistant-drafter GGUF (gemma4 two-GGUF MTP "
        "shape); implies --speculative. Native-head models "
        "(qwen3.5/3.6) need no companion.",
    )
    ap.add_argument(
        "--draft-block-size",
        type=int,
        default=None,
        metavar="N",
        help="Override the MTP draft block size.",
    )
    ap.add_argument(
        "--stochastic-mtp",
        action="store_true",
        help="Accept MTP drafts by p/q rejection sampling instead of exact "
        "match on sampled runs: output follows the same distribution as "
        "non-speculative sampling but is not token-identical to it, and "
        "acceptance (so decode speed) rises at temp > 0. Greedy decoding "
        "is unaffected and stays token-identical.",
    )


def add_config_profile_args(ap: argparse.ArgumentParser) -> None:
    """Add the config-resolution flags. Shared by ``run`` and ``chat``."""
    ap.add_argument(
        "--config",
        default=None,
        metavar="FILE",
        help="Server config to resolve a model name against when the positional "
        "isn't a file on disk (default: the first existing default config). The "
        "matched model's path + sampling/template/load settings are applied.",
    )
    ap.add_argument(
        "--profile",
        default=None,
        metavar="NAME",
        help="Sampling profile: a built-in intent (coding, creative, instruct, "
        "reasoning-low/-medium/-high) resolved for the model's family, or - "
        "for a config-named model - any profile from the config. `model@NAME` "
        "is the inline spelling. See `gmlx profiles`.",
    )
    ap.add_argument(
        "--no-family-defaults",
        action="store_true",
        help="Don't seed the model family's recommended sampling (model-card "
        "defaults) for flags you didn't pass; keep argparse defaults "
        "(temp 0.0 = greedy). Env: GMLX_NO_FAMILY_DEFAULTS=1.",
    )


def add_load_args(ap: argparse.ArgumentParser) -> None:
    """Add the loader/template flags. Shared by ``run`` and ``chat``."""
    ap.add_argument(
        "--arch", default=None, metavar="ARCH", help="Override architecture detection."
    )
    ap.add_argument(
        "--chat-template",
        default=None,
        metavar="STR|PATH",
        help="Inline Jinja chat template, or a path to a "
        ".jinja/.txt file, replacing the GGUF's template.",
    )
    ap.add_argument(
        "--chat-template-config",
        default=None,
        metavar="JSON",
        help="JSON of extra chat-template kwargs, e.g. '{\"enable_thinking\": false}'.",
    )
    ap.add_argument(
        "--no-remap",
        action="store_true",
        help="Skip GGUF->HF name remap (raw GGUF names).",
    )
    ap.add_argument(
        "--no-zero-copy",
        action="store_true",
        help="memcpy tensors out of the mmap instead of viewing.",
    )


def add_sampling_args(ap: argparse.ArgumentParser) -> None:
    """Add the sampler flags. Shared by ``run`` and ``chat`` (chat also adjusts
    them live via the matching /commands)."""
    ap = ap.add_argument_group("sampling")
    ap.add_argument(
        "--temp",
        type=float,
        default=0.0,
        metavar="T",
        help="Sampling temperature (default 0.0 = greedy).",
    )
    ap.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        metavar="P",
        help="Nucleus sampling: keep the smallest token set with "
        "cumulative probability top-p (default 0.95; 1 = off).",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=0,
        metavar="N",
        help="Sample only from the k most likely tokens (default 0 = off).",
    )
    ap.add_argument(
        "--min-p",
        type=float,
        default=0.05,
        metavar="P",
        help="Drop tokens below min-p x the top token's "
        "probability (default 0.05; 0 = off).",
    )
    ap.add_argument(
        "--repetition-penalty",
        type=float,
        default=0.0,
        metavar="X",
        help="Penalize recently generated tokens, e.g. 1.1 (default 0 = off).",
    )
    ap.add_argument(
        "--presence-penalty",
        type=float,
        default=0.0,
        metavar="X",
        help="Flat penalty on tokens already generated "
        "(OpenAI semantics; default 0 = off).",
    )
    ap.add_argument(
        "--frequency-penalty",
        type=float,
        default=0.0,
        metavar="X",
        help="Penalty scaled by each token's count so far "
        "(OpenAI semantics; default 0 = off).",
    )
    ap.add_argument(
        "--repetition-context-size",
        type=int,
        default=20,
        metavar="N",
        help="How many recent tokens the repetition penalty considers (default 20).",
    )
    ap.add_argument(
        "--xtc-probability",
        type=float,
        default=0.0,
        metavar="P",
        help="XTC sampling probability (default 0 = off). "
        "Text-only - ignored with --mmproj.",
    )
    ap.add_argument(
        "--xtc-threshold",
        type=float,
        default=0.0,
        metavar="T",
        help="XTC sampling threshold (default 0.0). Text-only - ignored with --mmproj.",
    )
    ap.add_argument(
        "--logit-bias",
        default=None,
        metavar="JSON",
        help="JSON token-id->bias map, e.g. '{\"128001\": -100}'.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="PRNG seed for reproducible sampling.",
    )


def add_kv_cache_args(ap: argparse.ArgumentParser) -> None:
    """Add the KV-cache + prefill memory knobs. Shared by ``run`` and ``chat``."""
    ap = ap.add_argument_group("KV cache & prefill memory")
    ap.add_argument(
        "--max-kv-size",
        type=int,
        default=None,
        metavar="N",
        help="Cap the KV cache size (rotating cache above it).",
    )
    ap.add_argument(
        "--kv-bits",
        type=int,
        default=None,
        metavar="N",
        help="Quantize the KV cache to this many bits (e.g. 8 or 4) - "
        "2-4x cache memory cut at long context; off by default.",
    )
    ap.add_argument(
        "--kv-group-size",
        type=int,
        default=64,
        metavar="N",
        help="KV-cache quantization group size (default 64).",
    )
    ap.add_argument(
        "--quantized-kv-start",
        type=int,
        default=0,
        metavar="N",
        help="Token position from which the KV cache is "
        "quantized (default 0 = from the start).",
    )
    ap.add_argument(
        "--prefill-step-size",
        type=int,
        default=None,
        metavar="N",
        help="Prefill chunk size in tokens - lower it to cap "
        "peak memory on long prompts (default 2048; streaming "
        "--stream-experts / --stream-cpu models default to 8192).",
    )


def add_vlm_shared_args(ap: argparse.ArgumentParser) -> None:
    """Add the VLM/thinking knobs both verbs expose. Shared by ``run`` and ``chat``."""
    ap.add_argument(
        "--resize-shape",
        default=None,
        metavar="N|WxH",
        help="Resize images before encoding (VLM mode), e.g. "
        "448 or 672x448 - controls soft-token count.",
    )
    ap.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        metavar="N",
        help="Cap thinking tokens: force </think> after ~N reasoning tokens "
        "(text + VLM modes; no-op when thinking is disabled).",
    )


def add_placement_args(ap: argparse.ArgumentParser) -> None:
    """Add the execution-placement knobs (the disk-streaming placements, the
    lossy MoE fan-out group, and the feeders). Shared by ``run`` and ``chat``."""
    grp = ap.add_argument_group("model placement (bigger-than-RAM streaming)")
    placement = grp.add_mutually_exclusive_group()
    placement.add_argument(
        "--stream-experts",
        action="store_true",
        help="Run an over-RAM MoE by streaming its experts from disk: "
        "the every-token layers - attention, norms, routers, shared "
        "experts, embeddings - and the KV cache stay on GPU; the "
        "routed-expert stacks (~90%% of MoE bytes) stay file-backed, "
        "with decode served from a wired popularity-managed GPU "
        "expert arena (misses read from the GGUF) and the CPU "
        "stream as fallback.",
    )
    placement.add_argument(
        "--stream-cpu",
        action="store_true",
        help="Run the whole model on the CPU device, every weight "
        "streamed: weights stay mmap-backed, so the page cache "
        "streams them from disk on demand. Loads models the GPU "
        "wired limit can't hold, and on Apple Silicon decoding "
        "the every-token layers on CPU too beats handing every "
        "layer across the GPU<->CPU boundary. Expert prefetch + "
        "wide prefill chunks stay on for MoE models past the "
        "wired budget.",
    )
    add_moe_expert_args(ap)
    grp.add_argument(
        "--prefill-feeder",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Faster prompt processing for streaming models "
        "(--stream-experts / --stream-cpu past the wired budget): "
        "stage each prefill layer's expert stacks straight from the "
        "GGUF into GPU-visible ring slots instead of through the "
        "page cache. Default on; --no-prefill-feeder falls back to "
        "page-cache prefetch.",
    )
    grp.add_argument(
        "--decode-feeder",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Faster decode for --stream-experts models: keep the "
        "most-used experts in a wired, popularity-managed GPU arena "
        "and read only the misses from the GGUF at SSD queue depth. "
        "Default on for --stream-experts (needs the every-token "
        "layers on GPU, so never under --stream-cpu); "
        "GMLX_DECODE_ARENA_GB caps the arena size.",
    )


# MTP / speculative auto-enable (shared by run + chat)
# Native-head GGUFs (qwen3.5/3.6 'nextn') auto-enable MTP speculative decoding so
# the win is on by default. It stays a pure speed win: auto engages only when the
# user set nothing the MTP verify walk can't honor (otherwise it defers to plain
# decoding so the requested flag still takes effect). Explicit --speculative/--mtp
# forces it (dropping those flags with a warning); --no-speculative/--no-mtp forces
# plain decoding. serve/discovery resolve the same notion via 'speculative: auto'.


def mtp_dropped_run_flags(args) -> list[str]:
    """run flags the MTP verify walk can't honor (generate_speculative takes only
    temp/top-p/top-k/min-p, plus a baked system prompt). Drives the drop-with-warning
    on the MTP path: MTP stays on, --no-mtp honors these via plain decoding."""
    pairs = (
        ("--stop", args.stop is not None),
        ("--logit-bias", args.logit_bias is not None),
        ("--repetition-penalty", args.repetition_penalty != 0.0),
        ("--presence-penalty", args.presence_penalty != 0.0),
        ("--frequency-penalty", args.frequency_penalty != 0.0),
        ("--xtc-probability", args.xtc_probability != 0.0),
        # --kv-bits/--kv-group-size are handled on the MTP path itself
        # (pooled packing where the arch has pools, an accurate note where
        # it doesn't), so they are no longer listed here.
        ("--quantized-kv-start", args.quantized_kv_start != 0),
        ("--max-kv-size", args.max_kv_size is not None),
        ("--over-generation", args.over_generation != 0),
        ("--inject-critique", args.inject_critique is not None),
        ("--thinking-budget", args.thinking_budget is not None),
        ("--prefill-step-size", args.prefill_step_size is not None),
    )
    return [name for name, on in pairs if on]


def mtp_dropped_chat_flags(args) -> list[str]:
    """chat MTP-incompatible flags. Narrower than run: the chat MTP path keeps
    --system-prompt (baked into the templated turn) and --stop (post-hoc stream
    filter), so only the sampler/KV knobs below are lost."""
    pairs = (
        ("--logit-bias", args.logit_bias is not None),
        ("--repetition-penalty", args.repetition_penalty != 0.0),
        ("--presence-penalty", args.presence_penalty != 0.0),
        ("--frequency-penalty", args.frequency_penalty != 0.0),
        ("--xtc-probability", args.xtc_probability != 0.0),
        # --kv-bits/--kv-group-size: handled on the MTP path (see
        # mtp_dropped_run_flags).
        ("--quantized-kv-start", args.quantized_kv_start != 0),
        ("--max-kv-size", args.max_kv_size is not None),
    )
    return [name for name, on in pairs if on]


def _mtp_hard_incompatible(args) -> str | None:
    """A flag that can't run on the text-only MTP path at all (not merely dropped):
    auto defers, and an explicit --speculative plus one of these errors upstream.

    ``--stream-experts`` is NOT here: streaming composes with MTP (the
    decode-feeder arena serves any call <= its token cap, which covers the
    verify widths, and the drafter block stays resident) - the run/bench
    entry points apply placement after ``load_mtp_model``. ``--stream-cpu``
    stays blocked: the verify forward on the CPU stream is untested."""
    for name, on in (
        ("--mmproj", getattr(args, "mmproj", None) is not None),
        ("--adapter", getattr(args, "adapter", None) is not None),
        ("--stream-cpu", getattr(args, "stream_cpu", False)),
        ("--moe-experts", getattr(args, "moe_experts", None) is not None),
        ("--moe-expert-mass", getattr(args, "moe_expert_mass", None) is not None),
        ("--moe-expert-probe", getattr(args, "moe_expert_probe", False)),
        ("--moe-miss-shed", getattr(args, "moe_miss_shed", None) is not None),
        ("--moe-layer-shed", getattr(args, "moe_layer_shed", None) is not None),
    ):
        if on:
            return name
    return None


def _has_native_mtp_head(gguf_path: str) -> bool:
    """Header-only peek (cheap on multi-GB files): does this GGUF carry a native MTP
    head ('nextn' layers)? Rides discovery's stat-validated header cache, so this
    and the family-defaults probe share one read; the verdict matches
    serve/discovery exactly."""
    try:
        from .discovery import header_meta

        meta = header_meta(gguf_path)
    except Exception:
        return False
    return bool(meta and meta.get("kind") == "model" and meta.get("mtp"))


def resolve_speculative(args, gguf_path: str) -> tuple[bool, str]:
    """Decide whether run/chat takes the MTP path, plus a note to print.

    Precedence: --no-speculative/--no-mtp (off) > explicit --speculative/--mtp or
    --draft-gguf (on) > auto. Auto enables MTP iff the GGUF has a native head and no
    hard-incompatible flag (--mmproj/--adapter/--stream-cpu/--moe-*) is set; sampler
    flags the MTP walk can't honor are dropped with a warning at generation (--no-mtp
    honors them via plain decoding), not deferred. The note is empty when the user
    was explicit."""
    if getattr(args, "no_speculative", False):
        return False, ""
    spec = getattr(args, "speculative", None)
    if spec or getattr(args, "draft_gguf", None):
        return True, ""
    if spec is False:
        # Explicit opt-out from a config model ('speculative: false'); not auto.
        return False, ""
    # auto from here (spec is None): a native head always enables MTP (sticky).
    # Flags the verify walk can't honor are dropped with a warning at generation
    # (set --no-mtp to honor them via plain decoding) -- not deferred, so a
    # habitual penalty never silently disables MTP. Only the hard-incompatible
    # flags above (--mmproj/--adapter/--stream-cpu/...) force plain decoding.
    if _mtp_hard_incompatible(args):
        return False, ""
    if getattr(args, "stream_experts", False):
        # Streaming decode: auto-MTP stays off (measured wash at typical
        # no-think acceptance, and verify widths add miss IO); explicit
        # --speculative/--mtp above still opts in.
        return False, ""
    if not _has_native_mtp_head(gguf_path):
        companion = _deepseek4_mtp_companion(gguf_path)
        if companion is None:
            return False, ""
        return True, (
            f"[mtp] companion MTP drafter detected "
            f"({os.path.basename(companion)}) -> speculative decoding on "
            "(--no-mtp to disable)"
        )
    return True, (
        "[mtp] native MTP head detected -> speculative decoding on "
        "(--no-mtp to disable)"
    )


def _deepseek4_mtp_companion(gguf_path: str) -> str | None:
    """Companion MTP drafter GGUF for a deepseek4 target, if one sits next to
    it (auto enable; the loader re-resolves the same path when
    draft_gguf_path is not given). Header-cache peeks only."""
    try:
        from .discovery import find_mtp_companion, header_meta

        meta = header_meta(gguf_path)
        if not meta or meta.get("arch") != "deepseek4":
            return None
        return find_mtp_companion(gguf_path)
    except Exception:
        return None


def _vlm_mtp_drafter_available(args) -> bool:
    """Whether a ``--mmproj`` (VLM) load should also build an MTP drafter so
    text-only requests take the fast path (image/audio requests stay on plain VLM).

    A drafter is available from a ``--draft-gguf`` assistant (gemma4) or a native
    MTP head in the LLM GGUF (qwen3.5/3.6 ``nextn``). ``--no-mtp`` / config
    ``speculative: false`` opts out. Unlike :func:`resolve_speculative`, ``--mmproj``
    is expected here (it's the VLM path), not a blocker - the adapter/cpu*/offload
    conflicts are already rejected before dispatch, so only the drafter source and
    the on/off toggle matter."""
    if getattr(args, "no_speculative", False):
        return False
    if getattr(args, "speculative", None) is False:
        return False  # explicit config opt-out
    if getattr(args, "draft_gguf", None):
        return True  # gemma4 assistant companion
    return _has_native_mtp_head(args.gguf)


def _build_parser(prog: str = "gmlx run") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=prog,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("gguf", help="Path to the GGUF file (sharded ok).")
    prompt_group = ap.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--prompt",
        default="Hello, world!",
        metavar="STR",
        help="Generation prompt (default: 'Hello, world!').",
    )
    prompt_group.add_argument(
        "--prompt-file",
        default="",
        metavar="PATH",
        help="Read the prompt from a file (mutually exclusive with --prompt).",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Decode-token cap; default: generate until the model stops. "
        "Pass N to cap the reply.",
    )
    add_sampling_args(ap)
    ap.add_argument(
        "--stop",
        action="append",
        default=None,
        metavar="STR",
        help="Stop sequence - generation ends (trimmed) when it "
        "appears. Repeatable. Text-only - ignored with "
        "--mmproj.",
    )
    ap.add_argument(
        "--system-prompt",
        default=None,
        metavar="STR",
        help="System message for the chat template.",
    )
    add_kv_cache_args(ap)
    add_vlm_shared_args(ap)

    exp = ap.add_argument_group("experimental")
    exp.add_argument(
        "--over-generation",
        type=int,
        default=0,
        metavar="N",
        help="Force N continuation tokens past the stop token, or cap an "
        "injected critique reply (default 0 = off). Experimental.",
    )
    exp.add_argument(
        "--inject-critique",
        nargs="?",
        const="List any bugs or issues in the code you just wrote.",
        default=None,
        metavar="TEXT",
        help="After the stop, inject a follow-up turn and answer it from the "
        "cache; bare flag asks for bugs. Experimental.",
    )
    exp.add_argument(
        "--inject-no-thinking",
        action="store_true",
        help="Render the injected critique with enable_thinking=False "
        "(effective only where the template gates thinking by it). Experimental.",
    )
    exp.add_argument(
        "--over-temp",
        type=float,
        default=None,
        help="Sampling temperature for the continuation (default: --temp). "
        "Experimental.",
    )
    exp.add_argument(
        "--over-top-p",
        type=float,
        default=None,
        help="top-p for the continuation (default: --top-p). Experimental.",
    )
    exp.add_argument(
        "--over-top-k",
        type=int,
        default=None,
        metavar="N",
        help="top-k for the continuation (default: --top-k). Experimental.",
    )
    exp.add_argument(
        "--over-min-p",
        type=float,
        default=None,
        help="min-p for the continuation (default: --min-p). Experimental.",
    )
    exp.add_argument(
        "--over-generation-log",
        default=None,
        metavar="PATH",
        help="Append a JSONL probe record per run for analysis. Experimental.",
    )
    exp.add_argument(
        "--over-label",
        default=None,
        metavar="TAG",
        help="Tag the probe record to pair free vs injected runs in the "
        "rollup (defaults to pairing by prompt). Experimental.",
    )

    add_config_profile_args(ap)
    ap.add_argument(
        "--hf-source",
        default=None,
        metavar="ID|DIR",
        help="Load config (and, with --mmproj, the image processor + "
        "chat template) from this HF id / local dir instead of "
        "synthesizing it from GGUF metadata. Optional override.",
    )
    ap.add_argument(
        "--mmproj",
        default=None,
        metavar="PATH",
        help="Vision projector GGUF (general.architecture=clip). "
        "Pairs a float vision tower with the K-quant LLM GGUF "
        "to load a vision-language model. The image processor + "
        "chat template are synthesized from the GGUFs; "
        "--hf-source overrides that if needed.",
    )
    ap.add_argument(
        "--image",
        default=None,
        metavar="PATH|URL",
        help="Image to prepend to the prompt (VLM mode; repeatable "
        "by passing a comma-separated list).",
    )
    ap.add_argument(
        "--audio",
        default=None,
        metavar="PATH|URL",
        help="Audio to prepend to the prompt (omni mmproj with an "
        "audio tower; repeatable via a comma-separated list).",
    )
    ap.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Pass the prompt verbatim, even if a chat template "
        "exists (for base / non-instruct models).",
    )
    add_load_args(ap)
    add_placement_args(ap)
    ap.add_argument(
        "--adapter",
        default=None,
        metavar="PATH",
        help="GGUF LoRA adapter applied live over the base at load - "
        "base stays K-quant, no merge. Text path only (not "
        "--mmproj / --speculative).",
    )

    add_speculative_args(ap)
    add_verbosity_arg(ap)

    ap.add_argument(
        "--report-only",
        action="store_true",
        help="Load + print inventory (and, with --chat-template / "
        "default, the rendered prompt); skip model build.",
    )
    ap.add_argument(
        "--bench",
        default="",
        metavar="LIST",
        help="Comma-separated prompt lengths (e.g. 512,4096,16384); "
        "prints a prefill/decode tok/s table.",
    )
    ap.add_argument(
        "--bench-depths",
        default="",
        metavar="LIST",
        help="Comma-separated context depths (e.g. 0,4096,16384); "
        "reports tg (decode tok/s) AT each depth as its own "
        "column. With --speculative, also accept-rate + speedup.",
    )
    ap.add_argument(
        "--bench-runs",
        type=int,
        default=2,
        metavar="N",
        help="Timed runs per length; best (max tps) reported (default 2).",
    )
    ap.add_argument(
        "--bench-decode-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Decode tokens per bench run (--bench default 32, "
        "--bench-depths default 128).",
    )
    ap.add_argument(
        "--bench-temp",
        type=float,
        default=0.0,
        metavar="T",
        help="Sampling temperature for speculative bench (default 0.0 = greedy).",
    )
    ap.add_argument(
        "--bench-chat-dataset",
        default="",
        metavar="DATASET",
        help="HF chat dataset for bench prompts (e.g. "
        "HuggingFaceH4/ultrachat_200k or dataset_id:split). "
        "Chat-templated prompts give representative MTP acceptance "
        "on instruct models; without this, synthetic prompts are used.",
    )
    ap.add_argument(
        "--bench-chat-seed",
        type=int,
        default=42,
        metavar="N",
        help="Seed for the per-depth chat-prompt slice (default 42). The "
        "k-th prompt at a depth is a pure function of (seed, depth, k), so "
        "identical seeds give content-identical cells across runs and "
        "depth lists; vary the seed to re-check a decision on a "
        "different slice.",
    )
    return ap


# --max-tokens unset means "until the model stops": the generation loops still
# need a finite bound, so unset resolves to a cap no real reply reaches.
_UNCAPPED_MAX_TOKENS = 1 << 25

# The denoising canvas is allocated at max_tokens, so "until the model stops"
# has no meaning for diffusion models - uncapped falls back to this bounded
# canvas (shared by run and chat so both verbs behave identically).
_DIFFUSION_MAX_TOKENS = 2048


def resolve_max_tokens(args) -> None:
    """Resolve the tri-state ``--max-tokens`` (unset/0 = until EOS) into an
    effective cap, remembering whether the user (or a config) capped it so the
    generation paths can say when the cap - not the model - ended a reply.
    Runs after the config / family-defaults overlays (either may seed a cap)."""
    capped = args.max_tokens is not None and args.max_tokens > 0
    args._max_tokens_capped = capped
    if not capped:
        args.max_tokens = _UNCAPPED_MAX_TOKENS


def max_tokens_label(args) -> str:
    """``max_tokens=`` banner value: the cap, or ``until-eos`` when uncapped."""
    return (str(args.max_tokens)
            if getattr(args, "_max_tokens_capped", True) else "until-eos")


def warn_cap_hit(args, n_tokens) -> None:
    """One stderr line when a reply ended at the --max-tokens cap, not EOS."""
    if (getattr(args, "_max_tokens_capped", False)
            and n_tokens is not None and n_tokens >= args.max_tokens):
        print(
            f"note: reply stopped at the --max-tokens cap ({args.max_tokens}); "
            "raise it or omit the flag to generate until the model stops.",
            file=sys.stderr,
        )


def _report_error(e) -> int:
    """One stderr line per failure, skipped when loadlog already printed the
    merged load-failure line for this exception (_gmlx_reported)."""
    if not getattr(e, "_gmlx_reported", False):
        print(f"error: {e}", file=sys.stderr)
    return 1


def print_family_note(args) -> None:
    """Print the family-defaults note deferred by :func:`apply_family_defaults`
    - called after a successful load so the note never trails a load failure."""
    note = getattr(args, "_family_note", None)
    if note:
        args._family_note = None
        print(note)


def _report_only(args) -> int:
    """Load wire bytes + remap, print the inventory and the rendered prompt."""
    from .gguf_meta import first_nonzero_int, read_int
    from .loader import (
        _resolve_chat_template,
        load_gguf_wire_bytes,
        print_inventory,
        remap_arrays,
    )

    # Codec preflight so an IQ / unsupported-codec GGUF refuses cleanly here
    # instead of crashing kq.load_gguf. The arch gate is *skipped* - report-only
    # should still inventory an arch the loader can't yet build.
    from .arch_table import UnsupportedArchError
    from .preflight import preflight

    try:
        preflight(args.gguf, arch=args.arch)
    except UnsupportedArchError:
        pass  # UnsupportedCodecError still propagates to main()'s handler

    t0 = time.perf_counter()
    arrays, kquant_meta, arch_meta, meta, _shapes = load_gguf_wire_bytes(
        args.gguf, zero_copy=not args.no_zero_copy
    )
    print(
        f"[gguf] {len(arrays)} arrays, {len(kquant_meta)} kquant "
        f"({time.perf_counter() - t0:.2f}s)"
    )
    arch = args.arch or arch_meta
    if arch is None:
        print("error: could not determine arch; pass --arch", file=sys.stderr)
        return 1
    print(f"[arch] {arch}")
    n_head = read_int(meta, f"{arch}.attention.head_count")
    n_head_kv = first_nonzero_int(meta, f"{arch}.attention.head_count_kv")
    _, hf_kquant_meta, stats = remap_arrays(
        arrays,
        kquant_meta,
        arch,
        no_remap=args.no_remap,
        n_head=n_head,
        n_head_kv=n_head_kv,
    )
    print_inventory(arch, kquant_meta, hf_kquant_meta, stats)

    # Render the prompt the way generate would, so --report-only can preview a
    # --chat-template override without building the model.
    if not args.no_chat_template:
        from .tokenizer import load_tokenizer_from_gguf

        override = _resolve_chat_template(args.chat_template)
        tok = load_tokenizer_from_gguf(meta, arch, chat_template_override=override)
        if tok.chat_template is not None:
            rendered = tok.apply_chat_template(
                [{"role": "user", "content": args.prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            print("\n=== rendered prompt ===")
            print(rendered)
    return 0


def _run_bench(args) -> int:
    from . import loadlog
    from .benchmarks import bench
    from .loader import load_model, preset_native_fp_wire_env

    # Fail fast on a malformed length list - before the multi-GB load.
    try:
        lengths = _parse_int_list(args.bench, flag="--bench")
    except argparse.ArgumentTypeError as e:
        print(str(e), file=sys.stderr)
        return 2
    preset_native_fp_wire_env(args)
    with loadlog.load_ui(args.verbose, args.gguf):
        model, _config, tok = load_model(
            args.gguf,
            arch=args.arch,
            hf_source=args.hf_source,
            chat_template=args.chat_template,
            no_remap=args.no_remap,
            zero_copy=not args.no_zero_copy,
            verbose=args.verbose,
        )
    print_family_note(args)
    _apply_placement(args, model)
    decode_tokens = args.bench_decode_tokens or 32
    print(
        f"[bench] lengths={lengths} runs={args.bench_runs} "
        f"decode_tokens={decode_tokens}"
    )
    results = bench(
        model,
        tok,
        lengths,
        decode_tokens=decode_tokens,
        runs=args.bench_runs,
        prefill_step_size=args.prefill_step_size,
    )
    print(f"\n{'prompt_len':>10} {'prefill_tps':>12} {'decode_tps':>11}")
    for L in lengths:
        r = results[L]
        print(f"{L:>10} {r['prefill_tps']:>12.1f} {r['decode_tps']:>11.1f}")
    return 0


def _run_bench_depths(args) -> int:
    """``tg@depth`` benchmark: decode tok/s measured at each context depth,
    with an optional MTP speculative A/B (accept-rate + speedup) per depth."""
    from .benchmarks import _ChatPromptSource, _load_chat_dataset, bench_tg_depth
    from .chat import parse_template_config
    from .loader import load_model, preset_native_fp_wire_env
    from .mtp_load import load_mtp_model

    try:
        depths = _parse_int_list(args.bench_depths, flag="--bench-depths")
    except argparse.ArgumentTypeError as e:
        print(str(e), file=sys.stderr)
        return 2
    decode_tokens = args.bench_decode_tokens or 128
    # bool(): --speculative defaults to the auto sentinel None, which would
    # otherwise render as "speculative=None" in the banner below.
    speculative = bool(args.speculative) and not args.no_speculative

    from . import loadlog

    # load the chat corpus BEFORE the model: dataset prepare can spike
    # several GB, which an over-RAM wired load leaves no headroom for
    convs = None
    chat_ds = getattr(args, "bench_chat_dataset", "") or ""
    if chat_ds:
        if ":" in chat_ds:
            ds_id, ds_split = chat_ds.rsplit(":", 1)
        else:
            ds_id, ds_split = chat_ds, "train_sft"
        print(f"[bench] loading chat dataset {ds_id}:{ds_split} ...")
        convs = _load_chat_dataset(ds_id, ds_split)
        print(f"[bench] {len(convs)} conversations loaded")

    preset_native_fp_wire_env(args)
    drafter = None
    if speculative:
        with loadlog.load_ui(args.verbose, args.gguf):
            model, drafter, _config, tok = load_mtp_model(
                args.gguf,
                arch=args.arch,
                draft_gguf_path=args.draft_gguf,
                chat_template=args.chat_template,
                zero_copy=not args.no_zero_copy,
                verbose=args.verbose,
                wire=not getattr(args, "stream_experts", False),
            )
    else:
        with loadlog.load_ui(args.verbose, args.gguf):
            model, _config, tok = load_model(
                args.gguf,
                arch=args.arch,
                hf_source=args.hf_source,
                chat_template=args.chat_template,
                no_remap=args.no_remap,
                zero_copy=not args.no_zero_copy,
                verbose=args.verbose,
            )
    _apply_placement(args, getattr(model, "language_model", model))
    print_family_note(args)

    prompt_source = None
    if convs is not None:
        seed = int(getattr(args, "bench_chat_seed", 42))
        tkw = parse_template_config(args.chat_template_config)
        prompt_source = _ChatPromptSource(
            convs, tok, seed=seed, template_kwargs=tkw)
        print(f"[bench] prompt slice seed {seed}"
              + (f" template_config={tkw}" if tkw else ""))

    corpus_label = chat_ds if chat_ds else "synthetic"
    print(
        f"[bench] depths={depths} runs={args.bench_runs} "
        f"decode_tokens={decode_tokens} speculative={speculative} "
        f"corpus={corpus_label}"
    )
    results = bench_tg_depth(
        model,
        tok,
        depths,
        decode_tokens=decode_tokens,
        runs=args.bench_runs,
        drafter=drafter,
        draft_block_size=args.draft_block_size,
        prefill_step_size=args.prefill_step_size,
        temp=args.bench_temp,
        prompt_source=prompt_source,
    )

    if speculative:
        print(
            f"\n{'depth':>8} {'prefill_tps':>12} {'tg_tps':>9} "
            f"{'spec_tps':>9} {'accept':>12} {'mean_acc':>9} "
            f"{'rounds':>7} {'speedup':>8}"
        )
        for D in depths:
            r = results[D]
            dn = r.get("draft_n", 0)
            da = r.get("draft_n_accepted", 0)
            pct = 100.0 * da / dn if dn else 0.0
            accept_str = f"{da}/{dn} {pct:.0f}%" if dn else f"{r['accept_rate'] * 100:.1f}%"
            print(
                f"{D:>8} {r['prefill_tps']:>12.1f} {r['tg_tps']:>9.1f} "
                f"{r['spec_tps']:>9.1f} {accept_str:>12} "
                f"{r['mean_accept_len']:>9.2f} "
                f"{r.get('rounds', 0):>7} {r['speedup']:>7.2f}x"
            )
    else:
        print(f"\n{'depth':>8} {'prefill_tps':>12} {'tg_tps':>9}")
        for D in depths:
            r = results[D]
            print(f"{D:>8} {r['prefill_tps']:>12.1f} {r['tg_tps']:>9.1f}")
    return 0


def _apply_placement(args, model) -> None:
    """Apply the requested execution placement (text paths only).

    ``--stream-experts`` keeps the every-token layers + KV cache on the GPU and
    streams only the routed experts from disk; ``--stream-cpu`` runs the whole
    model on the CPU device with the streaming-expert machinery engaged.
    """
    stream_cpu = getattr(args, "stream_cpu", False)
    stream_experts = getattr(args, "stream_experts", False)
    if not (stream_cpu or stream_experts):
        used = [
            name
            for name, on in (
                ("--moe-experts", getattr(args, "moe_experts", None) is not None),
                (
                    "--moe-expert-mass",
                    getattr(args, "moe_expert_mass", None) is not None,
                ),
                ("--moe-expert-probe", getattr(args, "moe_expert_probe", False)),
                (
                    "--moe-miss-shed",
                    getattr(args, "moe_miss_shed", None) is not None,
                ),
                (
                    "--moe-layer-shed",
                    getattr(args, "moe_layer_shed", None) is not None,
                ),
            )
            if on
        ]
        if used:
            print(
                f"[stream] {'/'.join(used)} ignored: needs --stream-experts "
                "or --stream-cpu (they only apply to streamed MoE layers)"
            )
        return

    gguf_path = getattr(args, "gguf", None)
    feeders = dict(
        feeder_prefill=getattr(args, "prefill_feeder", None),
        feeder_decode=getattr(args, "decode_feeder", None),
    )
    if stream_cpu:
        from .loader import configure_stream_cpu

        n, _ = configure_stream_cpu(model, gguf_path=gguf_path, **feeders)
        if n == 0:
            print(
                "[stream] note: no MoE expert stacks found - running this "
                "(dense?) model on the CPU device"
            )
    else:
        from .loader import install_expert_streaming

        n, _ = install_expert_streaming(model, gguf_path=gguf_path, **feeders)
        if n == 0:
            print(
                "[stream] warning: no MoE expert stacks found - "
                "--stream-experts has no effect on this (dense?) model"
            )
            return

    if getattr(args, "moe_experts", None) is not None:
        from .loader import install_moe_experts_override

        install_moe_experts_override(model, args.moe_experts)
    if getattr(args, "moe_expert_mass", None) is not None:
        from .moe_experts import install_moe_expert_mass

        install_moe_expert_mass(model, args.moe_expert_mass)
    elif getattr(args, "moe_expert_probe", False):
        from .moe_experts import install_moe_expert_probe

        install_moe_expert_probe(model)
    if getattr(args, "moe_miss_shed", None) is not None:
        from .moe_experts import install_moe_miss_shed

        install_moe_miss_shed(model, args.moe_miss_shed)
    if getattr(args, "moe_layer_shed", None) is not None:
        from .moe_experts import install_moe_layer_shed

        install_moe_layer_shed(model, args.moe_layer_shed)


def _run_generate(args) -> int:
    from .chat import parse_logit_bias, parse_template_config
    from .generation import generate
    from .loader import load_model, preset_native_fp_wire_env

    # Parse before the model load so a JSON typo fails fast.
    template_kwargs = parse_template_config(args.chat_template_config)
    logit_bias = parse_logit_bias(args.logit_bias)
    preset_native_fp_wire_env(args)

    if args.seed is not None:
        import mlx.core as mx

        mx.random.seed(args.seed)

    use_mtp, mtp_note = resolve_speculative(args, args.gguf)
    if mtp_note:
        print(mtp_note)
    if use_mtp:
        # generate_speculative takes only temp/top_p/top_k/min_p (+ a baked system
        # prompt). Sticky auto and explicit --mtp both land here; any other sampler
        # flag the verify walk has no hook for (penalties, bias, stop, xtc, KV) is
        # dropped with a warning -- --no-mtp is the escape to honor it via plain
        # decoding.
        dropped = mtp_dropped_run_flags(args)
        if dropped:
            print(
                f"warning: {', '.join(dropped)} not applied on the MTP path "
                f"(set --no-mtp to apply via plain decoding)",
                file=sys.stderr,
            )
        from . import loadlog
        from .generation import generate_speculative
        from .mtp_load import load_mtp_model

        with loadlog.load_ui(args.verbose, args.gguf):
            model, drafter, _config, tok = load_mtp_model(
                args.gguf,
                arch=args.arch,
                draft_gguf_path=args.draft_gguf,
                chat_template=args.chat_template,
                zero_copy=not args.no_zero_copy,
                verbose=args.verbose,
                wire=not args.stream_experts,
            )
        # Streaming placement applies to the target trunk only (the drafter
        # block is small and stays resident); the verify calls ride the
        # decode-feeder arena like any small chunk.
        _apply_placement(args, getattr(model, "language_model", model))
        print_family_note(args)
        print(
            f"[generate] MTP speculative: max_tokens={max_tokens_label(args)} "
            f"temp={args.temp}\n"
        )
        stats = generate_speculative(
            model,
            drafter,
            tok,
            args.prompt,
            max_tokens=args.max_tokens,
            temp=args.temp,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            draft_block_size=args.draft_block_size,
            apply_chat_template=not args.no_chat_template,
            system_prompt=args.system_prompt,
            template_kwargs=template_kwargs,
            verbose=True,
            kv_bits=args.kv_bits,
            kv_group_size=args.kv_group_size,
        )
        print(
            f"\n[mtp] {stats['tokens']} tok @ {stats['decode_tps']:.1f} tok/s "
            f"| accept_rate={stats['accept_rate'] * 100:.1f}% "
            f"mean_accept_len={stats['mean_accept_len']:.2f} "
            f"rounds={stats['rounds']}"
        )
        warn_cap_hit(args, stats.get("tokens"))
        return 0

    from . import loadlog

    with loadlog.load_ui(args.verbose, args.gguf):
        model, config, tok = load_model(
            args.gguf,
            arch=args.arch,
            hf_source=args.hf_source,
            chat_template=args.chat_template,
            no_remap=args.no_remap,
            zero_copy=not args.no_zero_copy,
            verbose=args.verbose,
        )
    print_family_note(args)
    _apply_placement(args, model)

    from .diffusion import is_diffusion_model
    if is_diffusion_model(model) and not args._max_tokens_capped:
        # Bounded canvas fallback, shown in the banner; see the constant.
        args.max_tokens = _DIFFUSION_MAX_TOKENS
        args._max_tokens_capped = True

    if args.adapter:
        from .adapter import apply_gguf_adapter
        from .discovery import header_meta

        adapter = os.path.abspath(os.path.expanduser(args.adapter))
        # Base arch gates the adapter up front: a mismatched family fails with
        # "adapter arch X != base arch Y", not a missing-targets install error.
        base_arch = args.arch or (header_meta(args.gguf) or {}).get("arch")
        n = apply_gguf_adapter(model, config, adapter, base_arch=base_arch)
        print(f"[adapter] applied {n}-module GGUF LoRA from {adapter}")

    # Cap-vs-EOS is not reported back on this path (generate returns text), so
    # no cap-hit note here; the MTP/VLM paths and chat print one.
    print(f"[generate] max_tokens={max_tokens_label(args)} temp={args.temp}\n")
    generate(
        model,
        tok,
        args.prompt,
        max_tokens=args.max_tokens,
        temp=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        xtc_probability=args.xtc_probability,
        xtc_threshold=args.xtc_threshold,
        repetition_penalty=args.repetition_penalty,
        repetition_context_size=args.repetition_context_size,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        logit_bias=logit_bias,
        stop=args.stop,
        system_prompt=args.system_prompt,
        template_kwargs=template_kwargs,
        max_kv_size=args.max_kv_size,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        quantized_kv_start=args.quantized_kv_start,
        prefill_step_size=args.prefill_step_size,
        thinking_budget=args.thinking_budget,
        apply_chat_template=not args.no_chat_template,
        prefill_progress=sys.stdout.isatty() and not args.verbose,
        over_generation=args.over_generation,
        inject_critique=args.inject_critique,
        inject_no_thinking=args.inject_no_thinking,
        over_temp=args.over_temp,
        over_top_p=args.over_top_p,
        over_top_k=args.over_top_k,
        over_min_p=args.over_min_p,
        over_generation_log=args.over_generation_log,
        over_label=args.over_label,
        verbose=True,
    )
    return 0


def _run_vlm(args) -> int:
    """Image+text generation through a two-GGUF VLM (K-quant LLM + float mmproj).

    Loads both GGUFs into a single mlx-vlm Model and hands the image request to
    mlx-vlm's own generate path (preprocessing + soft-token expansion + decode).
    The sampling surface mirrors the text path where mlx-vlm's generate supports
    it; --stop / --xtc-* stay text-only (mlx-vlm has no seam for them here).
    """
    from .chat import parse_logit_bias, parse_template_config
    from .vlm import load_vlm_model

    # Parse before the model load so a JSON typo fails fast.
    template_kwargs = parse_template_config(args.chat_template_config)
    logit_bias = parse_logit_bias(args.logit_bias)
    if args.seed is not None:
        import mlx.core as mx

        mx.random.seed(args.seed)

    images = [s for s in (args.image or "").split(",") if s.strip()]
    audios = [s for s in (args.audio or "").split(",") if s.strip()]
    from . import loadlog

    with loadlog.load_ui(args.verbose, args.gguf):
        model, config, processor = load_vlm_model(
            args.gguf,
            args.mmproj,
            hf_source=args.hf_source,
            arch=args.arch,
            zero_copy=not args.no_zero_copy,
            verbose=args.verbose,
        )
    print_family_note(args)

    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    prompt = args.prompt
    if not args.no_chat_template:
        messages = (
            [{"role": "system", "content": args.system_prompt}]
            if args.system_prompt
            else []
        )
        messages.append({"role": "user", "content": prompt})
        prompt = apply_chat_template(
            processor,
            config,
            messages,
            num_images=len(images),
            num_audios=len(audios),
            **template_kwargs,
        )

    from .chat import parse_resize_shape

    rep = args.repetition_penalty
    extra = {
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "repetition_penalty": None if rep in (0.0, 1.0) else rep,
        "repetition_context_size": args.repetition_context_size,
        "presence_penalty": args.presence_penalty or None,
        "frequency_penalty": args.frequency_penalty or None,
        "logit_bias": logit_bias,
    }
    if args.resize_shape:
        extra["resize_shape"] = parse_resize_shape(args.resize_shape)
    if args.thinking_budget is not None:
        extra["thinking_budget"] = args.thinking_budget
    if args.kv_bits is not None:
        extra.update(
            kv_bits=args.kv_bits,
            kv_group_size=args.kv_group_size,
            quantized_kv_start=args.quantized_kv_start,
        )
    if args.max_kv_size is not None:
        extra["max_kv_size"] = args.max_kv_size
    if args.prefill_step_size is not None:
        extra["prefill_step_size"] = args.prefill_step_size

    print(
        f"[generate] max_tokens={max_tokens_label(args)} temp={args.temp} "
        f"images={len(images)} audios={len(audios)}\n"
    )
    result = generate(
        model,
        processor,
        prompt,
        image=images or None,
        audio=audios or None,
        max_tokens=args.max_tokens,
        temperature=args.temp,
        verbose=True,
        **extra,
    )
    if getattr(result, "finish_reason", None) == "length":
        warn_cap_hit(args, getattr(result, "generation_tokens", None))
    return 0


def _run_vlm_mtp(args) -> int:
    """Text-only MTP speculative decoding on a loaded VLM (gemma4 + assistant drafter).

    A ``--mmproj`` VLM serves text-only requests in place: the request runs through
    mlx-vlm's MTP rounds (which only touch ``.language_model`` + caches), so it gets
    the speculative speedup. Image/audio requests are routed to ``_run_vlm`` upstream;
    the drafter is simply unused for those.
    """
    from .chat import parse_template_config
    from .generation import generate_speculative
    from .mtp_load import load_vlm_mtp_model

    if args.seed is not None:
        import mlx.core as mx

        mx.random.seed(args.seed)

    print("[mtp] VLM loaded, text-only request -> MTP speculative decoding")
    dropped = mtp_dropped_run_flags(args)
    if dropped:
        print(
            f"warning: {', '.join(dropped)} not applied on the MTP path "
            f"(set --no-mtp to apply via plain decoding)",
            file=sys.stderr,
        )
    from . import loadlog

    with loadlog.load_ui(args.verbose, args.gguf):
        model, drafter, _config, tok, _processor = load_vlm_mtp_model(
            args.gguf,
            args.mmproj,
            arch=args.arch,
            draft_gguf_path=args.draft_gguf,
            chat_template=args.chat_template,
            zero_copy=not args.no_zero_copy,
            verbose=args.verbose,
        )
    print_family_note(args)
    print(
        f"[generate] VLM text-only MTP: max_tokens={max_tokens_label(args)} "
        f"temp={args.temp}\n"
    )
    stats = generate_speculative(
        model,
        drafter,
        tok,
        args.prompt,
        max_tokens=args.max_tokens,
        temp=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        draft_block_size=args.draft_block_size,
        apply_chat_template=not args.no_chat_template,
        system_prompt=args.system_prompt,
        template_kwargs=parse_template_config(args.chat_template_config),
        verbose=True,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
    )
    print(
        f"\n[mtp] {stats['tokens']} tok @ {stats['decode_tps']:.1f} tok/s "
        f"| accept_rate={stats['accept_rate'] * 100:.1f}% "
        f"mean_accept_len={stats['mean_accept_len']:.2f} "
        f"rounds={stats['rounds']}"
    )
    warn_cap_hit(args, stats.get("tokens"))
    return 0


# Config-by-name resolution (run / chat) - a positional that isn't an on-disk file
# (nor a remote ref) is looked up in the server config by id/alias, and that model's
# resolved path + merged sampling/template/load settings fill in for unset flags.
# Config sampling key -> CLI args attribute (run + chat share this surface).
_CFG_SAMPLING_TO_ARG = {
    "temperature": "temp",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "max_tokens": "max_tokens",
    "repetition_penalty": "repetition_penalty",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "repetition_context_size": "repetition_context_size",
    "seed": "seed",
    "stop": "stop",
    "xtc_probability": "xtc_probability",
    "xtc_threshold": "xtc_threshold",
}
_CFG_LOAD_TO_ARG = {
    "kv_bits": "kv_bits",
    "kv_group_size": "kv_group_size",
    "max_kv_size": "max_kv_size",
    "quantized_kv_start": "quantized_kv_start",
}


def _explicit_dests(parser, argv) -> set:
    """The argparse dests the user passed explicitly in ``argv`` - so config values
    never clobber a flag the user typed. Matches each action's option strings against
    argv, handling the ``--flag=value`` form and the unique-prefix abbreviations
    argparse accepts by default (an ambiguous prefix would have failed the parse
    before the overlay runs, so a prefix match here is unambiguous)."""
    names = [a.split("=", 1)[0] for a in (argv or []) if a.startswith("-")]
    provided = set()
    for action in parser._actions:
        for opt in action.option_strings:
            if any(n == opt or (len(n) > 2 and n.startswith("--")
                                and opt.startswith(n)) for n in names):
                provided.add(action.dest)
                break
    return provided


def _apply_sampling_to_args(args, sampling, explicit: set,
                            chat_template_kwargs=None) -> list[str]:
    """Overlay a sampling group (plus optional chat-template kwargs like
    ``enable_thinking`` / gpt-oss ``reasoning_effort``) onto CLI ``args``,
    filling only what the user didn't pass explicitly. The sampling half of the
    config overlay, split out so the family-defaults layer can reuse it without
    the overlay's load/companion/speculative handling (which would clobber the
    bare-path MTP auto sentinel). Returns the applied labels."""
    applied: list[str] = []

    def _set(attr, value, label):
        if value is None or attr in explicit or not hasattr(args, attr):
            return
        setattr(args, attr, value)
        applied.append(label)

    sampling = sampling or {}
    for k, v in sampling.items():
        if k == "enable_thinking":
            continue                       # folded into chat-template kwargs below
        attr = _CFG_SAMPLING_TO_ARG.get(k)
        if attr:
            _set(attr, v, k)
    # enable_thinking + arbitrary template variables ride the chat-template
    # kwargs JSON, not the sampler.
    ctk = dict(chat_template_kwargs or {})
    et = sampling.get("enable_thinking")
    if et is not None:
        ctk["enable_thinking"] = et        # the sampling key is authoritative
    if (ctk and "chat_template_config" not in explicit
            and hasattr(args, "chat_template_config")
            and not args.chat_template_config):
        import json

        args.chat_template_config = json.dumps(ctk)
        applied.extend(sorted(ctk))
    return applied


def _apply_resolved_to_args(args, rm, explicit: set) -> list[str]:
    """Overlay a config :class:`ResolvedModel`'s settings onto the CLI ``args``,
    filling only what the user didn't pass explicitly (explicit flags win). Returns the
    list of applied setting labels for a one-line note. Guarded by ``hasattr`` so it
    works for both the ``run`` and (narrower) ``chat`` arg surfaces."""
    applied = _apply_sampling_to_args(args, rm.sampling, explicit,
                                      rm.chat_template_kwargs)

    def _set(attr, value, label):
        if value is None or attr in explicit or not hasattr(args, attr):
            return
        setattr(args, attr, value)
        applied.append(label)

    for k, v in (rm.load or {}).items():
        attr = _CFG_LOAD_TO_ARG.get(k)
        if attr:
            _set(attr, v, k)
    _set("system_prompt", rm.system, "system")
    _set("chat_template", rm.chat_template, "chat_template")
    _set("mmproj", rm.mmproj, "mmproj")
    _set("adapter", rm.adapter, "adapter")
    _set("draft_gguf", rm.draft_gguf, "draft_gguf")
    if hasattr(args, "speculative") and "speculative" not in explicit:
        # Config is authoritative for a configured model id: write the resolved bool
        # (True or False) so the native-head auto never second-guesses a config
        # opt-out. discovery already turns 'auto' -> True for MTP models; a bare file
        # path skips this overlay entirely and stays auto-eligible.
        args.speculative = bool(rm.speculative)
        if rm.speculative:
            applied.append("speculative")
    if (rm.stream and "stream_cpu" not in explicit
            and "stream_experts" not in explicit):
        if rm.stream == "cpu" and hasattr(args, "stream_cpu"):
            args.stream_cpu = True
            applied.append("stream-cpu")
        elif rm.stream == "experts" and hasattr(args, "stream_experts"):
            args.stream_experts = True
            applied.append("stream-experts")
    if (getattr(rm, "moe_expert_mass", None) is not None
            and "moe_expert_mass" not in explicit
            and hasattr(args, "moe_expert_mass")
            # an explicit --moe-expert-probe wins over the config's mass
            # (they are mutually exclusive: the probe must stay lossless)
            and not getattr(args, "moe_expert_probe", False)):
        args.moe_expert_mass = rm.moe_expert_mass
        applied.append("moe-expert-mass")
    for dest, label in (("moe_experts", "moe-experts"),
                        ("moe_miss_shed", "moe-miss-shed"),
                        ("moe_layer_shed", "moe-layer-shed")):
        if (getattr(rm, dest, None) is not None and dest not in explicit
                and hasattr(args, dest)):
            setattr(args, dest, getattr(rm, dest))
            applied.append(label)
    return applied


def split_path_intent(args) -> None:
    """Support ``path@intent`` addressing for a bare GGUF file: when the
    positional doesn't exist as-is, ends with an ``@<built-in intent>`` suffix,
    and the head does exist on disk, strip the suffix into ``args.profile``.
    Strictly gated on head-exists, so files whose names contain ``@`` (checked
    whole, first) and ``hf:...@rev`` refs are unaffected - the same last-``@``
    rule as config addressing."""
    raw = getattr(args, "gguf", None)
    if not raw or "@" not in raw or not hasattr(args, "profile"):
        return
    if raw.startswith(("hf:", "http://", "https://")):
        return
    if os.path.exists(os.path.expanduser(raw)):
        return
    head, tail = raw.rsplit("@", 1)
    from . import profiles as fam
    if tail in fam.BUILTIN_INTENTS and os.path.exists(os.path.expanduser(head)):
        args.gguf = head
        args.profile = args.profile or tail


def apply_family_defaults(args, parser, argv) -> int | None:
    """Seed a bare GGUF path's model-card family defaults (profiles.py) onto
    the CLI args - the same lowest layer the server applies - filling only
    flags the user didn't pass. ``--profile`` picks a built-in intent
    (``coding`` / ``creative`` / ...). Skipped when the model came from the
    config (the family layer already rode :func:`config.resolve_model`) or with
    ``--no-family-defaults`` / ``GMLX_NO_FAMILY_DEFAULTS=1``. Returns an
    exit code for an unknown ``--profile``, else ``None``."""
    if getattr(args, "_config_resolved", False):
        return None
    intent = getattr(args, "profile", None)
    if (getattr(args, "no_family_defaults", False)
            or os.environ.get("GMLX_NO_FAMILY_DEFAULTS")):
        if intent:
            print("error: --profile is a family-defaults feature; drop "
                  "--no-family-defaults (or the env opt-out) to use it",
                  file=sys.stderr)
            return 2
        return None
    from . import profiles as fam
    from .discovery import header_meta

    if intent is not None and intent not in fam.BUILTIN_INTENTS:
        print(f"error: unknown profile {intent!r}; built-in intents: "
              f"{', '.join(sorted(fam.BUILTIN_INTENTS))} (see `gmlx profiles`)",
              file=sys.stderr)
        return 2
    meta = header_meta(args.gguf)      # cached; the MTP auto probe shares it
    family = (fam.detect_family(meta.get("arch"), meta.get("name"))
              if meta else "default")
    groups = fam.groups_for(family, intent)
    label = fam.FAMILIES.get(family, {}).get("label", family)
    # An intent the family defines no delta for resolves to the base defaults;
    # say so instead of a banner claiming the intent was applied.
    has_delta = bool(intent) and bool(fam.family_intents(family).get(intent))
    if intent and not has_delta:
        print(f"[family] note: {label} has no @{intent} tuning - the family "
              f"base defaults apply", file=sys.stderr)
    explicit = _explicit_dests(parser, argv)
    applied = _apply_sampling_to_args(args, groups.get("sampling"), explicit,
                                      groups.get("chat_template_kwargs"))
    if applied:
        suffix = f" @{intent}" if has_delta else ""
        # Deferred to print_family_note() after a successful load: the banner
        # must never trail a load failure claiming defaults were applied.
        args._family_note = (
            f"[family] {label}{suffix} defaults: "
            f"{', '.join(sorted(applied))}  (--no-family-defaults to disable)")
    return None


def maybe_load_from_config(args, parser, argv) -> int | None:
    """If the positional model isn't an on-disk file (nor a remote ref), resolve it as a
    server-config model id/alias and overlay that model's settings onto ``args`` (and
    set ``args.gguf`` to its resolved path - including hf-cache-resident GGUFs). Returns
    an int return code on a config error, else ``None`` (whether or not a match was
    found - an unmatched name falls through to the caller's file-miss error). Sets
    ``args._config_resolved`` on a match so the bare-path family-defaults layer
    knows to stand down."""
    raw = args.gguf
    if os.path.exists(os.path.expanduser(raw)):
        return None
    if raw.startswith(("hf:", "http://", "https://")):
        return None
    # Only a bare name (no path separator, not a .gguf filename) is treated as a
    # config model id/alias; an obvious path is left to the file-miss error so a
    # typo'd path never silently reads the default config.
    if "/" in raw or os.sep in raw or raw.lower().endswith(".gguf"):
        return None
    from . import config as cfgmod

    try:
        cfg, cfg_path = cfgmod.load_cli_config(getattr(args, "config", None))
        if cfg is None:
            return None
        # Family detection before resolution, so the family base layer (and
        # family-resolved @intents) shape the overlay exactly like the server.
        from .discovery import fill_families
        fill_families(cfg)
        rm = cfgmod.resolve_cli_model(
            raw, cfg, request_profile=getattr(args, "profile", None))
    except cfgmod.ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if rm is None:
        return None
    explicit = _explicit_dests(parser, argv)
    applied = _apply_resolved_to_args(args, rm, explicit)
    args.gguf = rm.path
    args._config_resolved = True
    print(f"[config] '{raw}' -> {rm.path}  (from {cfg_path})")
    if applied:
        print(f"[config] applied: {', '.join(sorted(applied))}")
    return None


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = _build_parser(prog or f"{_prog()} run")
    args = parser.parse_args(argv)
    split_path_intent(args)
    rc = maybe_load_from_config(args, parser, argv)
    if rc is not None:
        return rc
    if args.draft_gguf:
        args.speculative = True  # same implication as `serve`
    if args.stochastic_mtp:
        from .speculative import set_stoch_accept

        set_stoch_accept(True)
    if args.adapter and (args.mmproj or args.speculative):
        which = "--mmproj" if args.mmproj else "--speculative"
        print(
            f"error: --adapter (live GGUF LoRA) on a {which} base is not "
            f"supported yet.",
            file=sys.stderr,
        )
        return 2
    if args.speculative and args.stream_cpu:
        # --stream-experts composes with MTP (streaming placement after
        # load_mtp_model; auto-MTP still defers under streaming, explicit
        # --speculative opts in). CPU-stream verify stays unsupported.
        print(
            "error: --stream-cpu on a --speculative/MTP base is not "
            "supported yet.",
            file=sys.stderr,
        )
        return 2
    if args.mmproj:
        # The VLM path has no bench/report/offload plumbing - refuse rather
        # than silently dispatch to plain generation.
        unsupported = [
            flag
            for flag, on in (
                ("--bench", bool(args.bench)),
                ("--bench-depths", bool(args.bench_depths)),
                ("--report-only", args.report_only),
                ("--stream-experts", args.stream_experts),
                ("--stream-cpu", args.stream_cpu),
            )
            if on
        ]
        if unsupported:
            print(
                f"error: {', '.join(unsupported)} not supported with --mmproj",
                file=sys.stderr,
            )
            return 2
        ignored = [
            flag
            for flag, on in (
                ("--stop", args.stop is not None),
                ("--xtc-probability", args.xtc_probability != 0.0),
                ("--xtc-threshold", args.xtc_threshold != 0.0),
            )
            if on
        ]
        if ignored:
            print(
                f"warning: {', '.join(ignored)} ignored in VLM mode (text-only)",
                file=sys.stderr,
            )
    # Validate the paths cheaply before any load machinery (or its heavy
    # imports) runs, so a path typo (or a remote ref pasted at the wrong verb)
    # gets one clear line in milliseconds.
    gguf = os.path.expanduser(args.gguf)
    if not os.path.exists(gguf):
        if args.gguf.startswith(("hf:", "http://", "https://")):
            hint = (
                " (remote refs work with `gmlx validate` / "
                "`gmlx pull`; run needs a local file)"
            )
        else:
            hint = (
                " (not a file, and no matching model id/alias in your config - "
                "see `gmlx list` or your config's models:)"
            )
        print(f"error: no such file: {args.gguf}{hint}", file=sys.stderr)
        return 2
    args.gguf = gguf
    rc = apply_family_defaults(args, parser, argv)
    if rc is not None:
        return rc
    # After the config/family overlays: either may have seeded a cap.
    resolve_max_tokens(args)
    for flag, val in (
        ("--mmproj", args.mmproj),
        ("--draft-gguf", args.draft_gguf),
        ("--adapter", args.adapter),
    ):
        if val and not os.path.exists(os.path.expanduser(val)):
            print(f"error: {flag}: no such file: {val}", file=sys.stderr)
            return 2
    from .arch_table import UnsupportedArchError
    from .preflight import UnsupportedCodecError

    # Resolve --prompt-file here so it applies in every mode (generate, VLM,
    # bench prompts, --report-only template preview), not just plain generate.
    if args.prompt_file:
        try:
            with open(os.path.expanduser(args.prompt_file), "r") as f:
                args.prompt = f.read()
        except OSError as e:
            print(f"error: --prompt-file: {e}", file=sys.stderr)
            return 2
        print(f"[prompt] loaded {len(args.prompt)} chars from {args.prompt_file}")
    try:
        if args.report_only:
            return _report_only(args)
        if args.mmproj:
            # A loaded VLM serves text-only requests in place: route a text-only
            # request through the MTP path when a drafter is available (gemma4
            # --draft-gguf, or a qwen3.5/3.6 native head), else plain VLM
            # generation. Image/audio -> VLM (the drafter idles that request).
            images = [s for s in (args.image or "").split(",") if s.strip()]
            audios = [s for s in (args.audio or "").split(",") if s.strip()]
            if not images and not audios and _vlm_mtp_drafter_available(args):
                return _run_vlm_mtp(args)
            return _run_vlm(args)
        if args.bench_depths:
            return _run_bench_depths(args)
        if args.bench:
            return _run_bench(args)
        return _run_generate(args)
    except (UnsupportedCodecError, UnsupportedArchError) as e:
        return _report_error(e)          # skips loadlog's merged line
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except (ImportError, ValueError, OSError, RuntimeError) as e:
        # OSError covers FileNotFoundError plus network/HF errors from
        # --hf-source; RuntimeError covers the loader's refusals and Metal
        # errors (e.g. out of GPU memory) - all written to stand alone, so a
        # traceback adds nothing. A load failure already printed loadlog's
        # merged line; _report_error doesn't stack a second one.
        return _report_error(e)
    except Exception as e:
        # UnsupportedVLMError, without importing transformers-heavy .vlm on
        # the text-only path: if .vlm was never imported, the exception
        # cannot be one of its.
        vlm_mod = sys.modules.get("gmlx.vlm")
        if vlm_mod is not None and isinstance(e, vlm_mod.UnsupportedVLMError):
            return _report_error(e)
        raise


# `gmlx` umbrella dispatcher - the console script.
# Verbs route to the per-area entry points (run == this module's `main`; serve /
# init / sync-models / launch / stop / restart / status / logs / service ==
# server.main; validate / pull / list / ps / profiles == manage; chat == chat;
# train == train).
# The macOS menu bar is `gmlx launch menubar`, not a top-level verb.
_VERBS = (
    "run",
    "chat",
    "talk",
    "serve",
    "init",
    "sync-models",
    "launch",
    "stop",
    "restart",
    "status",
    "logs",
    "service",
    "validate",
    "pull",
    "rm",
    "list",
    "ps",
    "profiles",
    "doctor",
    "train",
    "completion",
)

# Convenience verb aliases, canonicalized before dispatch.
_VERB_ALIASES = {"ls": "list"}


def _print_umbrella_help(prog: str = "gmlx") -> None:
    print(
        f"{prog} - a local inference platform for Apple Silicon: run, chat "
        "with, serve, and fine-tune GGUF models\n\n"
        f"usage: {prog} <command> [options]\n\n"
        "commands:\n"
        "  run          generate text from, benchmark, or inspect a GGUF\n"
        "  chat         interactive chat REPL on a GGUF (multi-turn, KV-cached)\n"
        "  talk         voice chat with a served model (wake word, STT, TTS)\n"
        "  serve        run the batched multi-model OpenAI/Anthropic server\n"
        "  init         scaffold a starter server config\n"
        "  sync-models  reconcile a config's models with disk / the hf cache\n"
        "  launch       point a coding harness at a running server (also: the\n"
        "               `launch menubar` macOS status-bar monitor)\n"
        "  stop         stop a backgrounded server (serve detaches by default)\n"
        "  restart      restart a backgrounded server with its original arguments\n"
        "  status       show whether a backgrounded server is running\n"
        "  logs         print or follow a backgrounded server's log\n"
        "  service      install/uninstall a launchd LaunchAgent (start at login)\n"
        "  validate     check a local or remote GGUF will load (no full download)\n"
        "  pull         validate a remote GGUF, then download it into your model dir\n"
        "  rm           delete a model's GGUF files and its config entry\n"
        "  list (ls)    list the models your server config defines (ids/aliases)\n"
        "  ps           show the models resident in a running server\n"
        "  profiles     show the per-family sampling defaults + @intents "
        "(add an id to resolve one model)\n"
        "  doctor       check the runtime, config, models, and services in one pass\n"
        "  train        finetune a LoRA adapter on a GGUF base (writes a GGUF adapter)\n"
        "  completion   print a shell completion script (zsh, bash, fish)\n\n"
        f"first run:  {prog} init   (scaffold a config) ->  {prog} serve   "
        f"(start the server) ->  {prog} launch <harness>\n"
        f"or one-shot:  {prog} run <model.gguf | id> --prompt \"...\"\n\n"
        f"run `{prog} <command> --help` for a command's options; "
        f"`{prog} --version` prints the installed version."
    )


def umbrella_main(argv: list[str] | None = None) -> int:
    """Dispatch ``gmlx <verb> ...`` to the matching entry point."""
    import warnings

    # Validation warnings (unrecognized config keys, ...) are user-facing on
    # the CLI: print the message, not the library source path and line.
    warnings.formatwarning = \
        lambda msg, cat, fn, ln, line=None: f"warning: {msg}\n"
    # transformers advises that PyTorch is missing on import - it is missing by
    # design here (all numerics run on MLX), so the advisory is pure noise.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    # Set when this process was exec'd through the renamed stub (procname.py);
    # getpath already consumed it, and it must not leak into children that run
    # their own Python (MCP tool servers via uvx/npx, ...).
    os.environ.pop("PYTHONEXECUTABLE", None)
    # A deleted cwd makes every relative-path abspath() die with a bare
    # "[Errno 2] No such file or directory" (no filename) deep in a verb.
    # Fail up front with an actionable message instead.
    try:
        os.getcwd()
    except OSError:
        print("error: current working directory no longer exists - "
              "cd to an existing directory and retry", file=sys.stderr)
        return 1
    argv = list(sys.argv[1:] if argv is None else argv)
    prog = _prog()
    if argv and argv[0] == "__complete":
        # Hidden: the per-TAB callback the emitted shell-completion script forwards to.
        from .completion import cmd_complete

        return cmd_complete(argv[1:])
    if not argv or argv[0] in ("-h", "--help", "help"):
        if argv and argv[0] == "help" and len(argv) > 1:
            sub = _VERB_ALIASES.get(argv[1], argv[1])
            if sub in _VERBS:              # `help serve` == `serve --help`
                return umbrella_main([sub, "--help"])
        _print_umbrella_help(prog)
        return 0
    if argv[0] in ("-V", "--version", "version"):
        import gmlx

        print(f"{prog} {gmlx.__version__}")
        return 0
    verb, rest = argv[0], argv[1:]
    verb = _VERB_ALIASES.get(verb, verb)
    if verb not in _VERBS:
        if verb.endswith(".gguf") or os.path.exists(verb):
            print(f"error: unknown command {verb!r}.\n\n"
                  f"did you mean: {prog} run {verb} ...?\n", file=sys.stderr)
            _print_umbrella_help(prog)
            return 2
        import difflib

        close = difflib.get_close_matches(
            verb, [*_VERBS, *_VERB_ALIASES], n=1, cutoff=0.6)
        if close:
            # With a likely typo in hand, two lines beat burying the
            # suggestion under the full command table.
            print(f"error: unknown command {verb!r}. did you mean "
                  f"{close[0]!r}?", file=sys.stderr)
            return 2
        print(f"error: unknown command {verb!r}.\n", file=sys.stderr)
        _print_umbrella_help(prog)
        return 2
    if verb != "doctor":  # doctor must run on a broken env to diagnose it
        from .upstream_seams import check_upstream_versions
        try:
            check_upstream_versions()
        except RuntimeError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    try:
        if verb == "run":
            return main(rest, prog=f"{prog} run")
        if verb == "chat":
            from .chat import cmd_chat

            return cmd_chat(rest, prog=f"{prog} chat")
        if verb == "talk":
            from .talk import cmd_talk

            return cmd_talk(rest, prog=f"{prog} talk")
        if verb in ("serve", "init", "sync-models", "launch", "stop", "restart",
                    "status", "logs", "service"):
            from .server import main as server_main

            return server_main(
                rest if verb == "serve" else [verb, *rest], prog=f"{prog} {verb}"
            )
        if verb == "train":
            from .train import cmd_train

            return cmd_train(rest, prog=f"{prog} train")
        if verb == "completion":
            from .completion import cmd_completion

            return cmd_completion(rest, prog=f"{prog} completion")
        if verb == "doctor":
            from .doctor import cmd_doctor

            return cmd_doctor(rest, prog=f"{prog} doctor")
        from . import manage

        if verb == "validate":
            return manage.cmd_validate(rest, prog=f"{prog} validate")
        if verb == "rm":
            return manage.cmd_rm(rest, prog=f"{prog} rm")
        if verb == "list":
            return manage.cmd_list(rest, prog=f"{prog} list")
        if verb == "ps":
            return manage.cmd_ps(rest, prog=f"{prog} ps")
        if verb == "profiles":
            return manage.cmd_profiles(rest, prog=f"{prog} profiles")
        return manage.cmd_pull(rest, prog=f"{prog} pull")
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except (ImportError, ValueError, OSError, RuntimeError) as e:
        return _report_error(e)          # skips loadlog's merged line
    except Exception as e:
        # The loaders raise their own refusal types (UnsupportedArchError,
        # UnsupportedCodecError, UnsupportedVLMError, ...) - `run` formats
        # them itself, but verbs like `chat` rely on this backstop. Their
        # messages are written to stand alone; a traceback adds nothing.
        from .arch_table import UnsupportedArchError
        from .preflight import UnsupportedCodecError

        if isinstance(e, (UnsupportedArchError, UnsupportedCodecError)):
            return _report_error(e)
        vlm_mod = sys.modules.get("gmlx.vlm")
        if vlm_mod is not None and isinstance(e, vlm_mod.UnsupportedVLMError):
            return _report_error(e)
        # Truly unexpected: a raw traceback helps a bug report but buries the
        # user. One line + next steps; --verbose / GMLX_DEBUG re-raises.
        if os.environ.get("GMLX_DEBUG") or "-v" in argv or "--verbose" in argv:
            raise
        print(f"error: unexpected failure: {type(e).__name__}: {e}",
              file=sys.stderr)
        print("run `gmlx doctor` to check your setup; set GMLX_DEBUG=1 "
              "for the full traceback", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
