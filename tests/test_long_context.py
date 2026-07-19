#!/usr/bin/env python3
"""Long-context integration: attention correctness only shows up at length.

Short-prompt/short-decode parity is necessary but NOT sufficient - a wrong
attention scale, rope base, or KV stride can stay coherent for a few hundred
tokens and then corrupt once the sequence is long enough for the error to
dominate. So every loadable arch is exercised at >=16k tokens, two ways:

  * ``test_long_decode_integrity`` (needs only KQUANT_TEST_GGUF_DIR) - generate
    a long greedy continuation (EOS suppressed, so an instruct model that wants
    to stop early still exercises attention to full depth) and assert the *bug
    signatures*: every token id in range, every step's logprob finite (no NaN),
    and no single token spammed for a huge run. Greedy *semantic* looping on a
    small model is expected and is NOT failed here - the target is corruption,
    not repetition.

  * ``test_long_prefill_parity`` (also needs KQUANT_LLAMACPP_BIN) - feed a long
    prompt, greedy-decode, and require the continuation to agree (as text, to
    sidestep tokenizer-boundary artifacts) with llama.cpp on the same prompt. A
    real attention bug diverges in the first few characters.

Both are ``integration`` + ``slow`` and skip unless the env points at real
models (see ``conftest``). Select one arch with ``-k qwen2`` and/or shrink the
length with ``KQUANT_LONGCTX_TOKENS=4096`` for a quick run; the defaults sweep
every arch whose GGUF is present at >=16k and can take minutes on large models.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile

import pytest

import mlx.core as mx  # noqa: E402

# Loadable arches to sweep; each auto-skips when no matching GGUF is present.
CANDIDATE_ARCHES = [
    "qwen2", "qwen2moe", "gemma", "gemma2", "gemma3", "phi3",
    "gemma4", "glm4", "qwen35", "qwen35moe", "mistral3", "llama",
    "nemotron_h_moe", "deepseek2", "mixtral", "glm4moe", "gpt-oss",
    "seed_oss", "smollm3", "granite", "ernie4_5-moe", "minimax-m2", "minimax-m3",
    "hunyuan-moe", "granitehybrid", "falcon-h1", "qwen3next", "hy_v3",
]

TARGET = int(os.environ.get("KQUANT_LONGCTX_TOKENS", "16384"))

# Arches whose mlx-lm model has a known upper-context limitation vs the GGUF
# reference, so token-for-token parity is only expected up to this length:
#   gemma2 - mlx-lm's gemma2 has no sliding-window attention; output is exact
#   only up to the 4096 window and diverges (coherently) above it. Upstream
#   mlx-lm limitation, not a loader defect - cap the parity comparison so the
#   test stays meaningful instead of asserting a gap we don't own.
PARITY_WINDOW = {"gemma2": 4096}

# Arches where greedy TEXT parity vs llama.cpp carries no loader signal:
#   gpt-oss - harmony-format model; a raw completion prompt is out-of-
#   distribution, so the greedy distribution is flat and sub-ulp numeric
#   differences flip tokens. Measured: llama.cpp and gmlx each produce
#   coherent but DIFFERENT replies at 2k/4k, agree by chance at 8k, and both
#   collapse to identical "..." spam at 16k - a coin flip at every depth.
#   gpt-oss loader/numerics are covered by the batch-parity and kernel
#   bit-exactness tests instead.
PARITY_SKIP = {
    "gpt-oss": "harmony-format model: raw-completion greedy text parity is "
               "a coin flip (flat OOD distribution); no loader signal",
}

_SEED = ("The history of science is a long and winding story, full of false "
         "starts, lucky accidents, and the slow accumulation of careful "
         "measurement. ")

# Needle-recall probe: a unique fact planted ~100 tokens into the prompt with
# a cloze tail that forces its retrieval across the full prompt depth. The
# repeated-seed continuation alone only requires the last ~40 tokens plus a
# correct rope frame at the 16k offset, so a bug that corrupts MID-context
# attention while leaving the local tail intact would pass the agreement
# check; retrieving the code cannot be done from the tail. Cloze (not a
# question) so a raw completion prompt stays in-distribution for base models,
# and the greedy continuation stays peaked (tie-resistant, like the rest of
# the prompt design).
_NEEDLE = "QZ-4812"
_NEEDLE_SENTENCE = (f"For the record, the vault access code written in the "
                    f"expedition logbook is {_NEEDLE}. ")
_NEEDLE_QUERY = ("\nAs noted near the beginning of this document, the vault "
                 "access code written in the expedition logbook is")

pytestmark = [pytest.mark.integration, pytest.mark.slow]


# helpers
def _require(gguf_index, arch):
    paths = gguf_index.get(arch)
    if not paths:
        pytest.skip(f"no {arch!r} GGUF under KQUANT_TEST_GGUF_DIR")
    return paths[0]


def _load(path):
    from gmlx import load_model
    return load_model(path, verbose=False)


def _ctx(config) -> int:
    return int(config.get("max_position_embeddings") or 4096)


def _greedy(model, tok, ids, n, logits_processors=None):
    """Greedy-decode ``n`` tokens; return (token_ids, all_finite, max_run).

    Decodes through ``stream_generate`` -- the same self-wiring entry the
    product paths use -- NOT raw ``generate_step``, which runs outside the
    wired_limit context: unwired, a near-RAM model decodes at ~0.5 tok/s vs
    ~20 wired, turning this 16k sweep into a 9-hour crawl (Hy3 87 GB)."""
    from mlx_lm.generate import stream_generate

    kwargs = {"max_tokens": n}
    if logits_processors:
        kwargs["logits_processors"] = logits_processors
    out, finite, max_run, run, prev = [], True, 0, 0, None
    for r in stream_generate(model, tok, mx.array(ids), **kwargs):
        t = int(r.token)
        out.append(t)
        if not bool(mx.isfinite(r.logprobs[t])):
            finite = False
        run = run + 1 if t == prev else 1
        max_run = max(max_run, run)
        prev = t
    return out, finite, max_run


def _eos_ids(tok):
    return set(getattr(tok, "eos_token_ids", None)
               or ([tok.eos_token_id] if tok.eos_token_id is not None else []))


def _max_run(ids) -> int:
    """Longest run of a single repeated token id."""
    max_run = run = 0
    prev = None
    for t in ids:
        run = run + 1 if t == prev else 1
        max_run = max(max_run, run)
        prev = t
    return max_run


# Single-token-spam is only a *corruption* signal in the early, in-distribution
# region: a rope/KV bug at depth collapses to one token within a few hundred
# steps (or trips finite/in-range first). A healthy tiny model force-decoded for
# thousands of tokens past where it wanted to stop eventually loops benignly, so
# the spam check is scoped to this window while finite/in-range cover full depth.
_SPAM_WINDOW = 2048


def _build_prompt(tok, n_tokens):
    """A natural ``n_tokens``-long prompt, rebuilt from a decoded token slice so
    mlx-kquant and llama.cpp tokenize the same text.

    The seed is repeated with a distinct section header each time so the prompt
    never becomes exactly periodic. A bare ``text *= 2`` repetition drives the
    model to a maximal-uncertainty point where two continuations' logits tie
    EXACTLY (hunyuan-A13B at 16k: '0' vs '1', gap 0.0000 nats) and greedy text
    parity then diverges on the tie-break - an artifact, not an attention bug
    (forcing llama.cpp's pick reproduced its continuation char-for-char).

    Encodes with ``add_special_tokens=False`` so the content tokens stay BOS-free
    on add_bos tokenizers - BOS is owned by the parity test's explicit n_prompt
    logic, not folded into the truncation/decode/re-encode round-trip.

    The needle sentence lands right after section 1 (~100 tokens in) and the
    cloze query is appended after truncation, so retrieval spans the whole
    prompt (see the _NEEDLE note)."""
    parts, i = [f"Section 1. {_SEED}", _NEEDLE_SENTENCE], 2
    while len(tok.encode("".join(parts), add_special_tokens=False)) < n_tokens + 8:
        parts.extend(f"Section {i + j}. {_SEED}" for j in range(32))
        i += 32
    n_query = len(tok.encode(_NEEDLE_QUERY, add_special_tokens=False))
    filler = tok.encode("".join(parts), add_special_tokens=False)
    decoded = tok.decode(filler[:max(n_tokens - n_query, 256)]) + _NEEDLE_QUERY
    return decoded, tok.encode(decoded, add_special_tokens=False)


def _llama_complete(binary, model_path, prompt_text, n, ctx):
    """Greedy llama.cpp continuation + reported prompt-token count."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(prompt_text)
        pf = f.name
    try:
        # KQUANT_LLAMACPP_NGL=0 keeps the reference on CPU for models past the
        # Metal wired budget (e.g. MiniMax-M3 UD-IQ3_XXS at 148 GB).
        ngl = os.environ.get("KQUANT_LLAMACPP_NGL", "99")
        cmd = [binary, "-m", model_path, "-f", pf, "-no-cnv",
               "--temp", "0", "-n", str(n), "-ngl", ngl, "-c", str(ctx),
               "--no-warmup", "--no-display-prompt", "--simple-io", "-s", "0"]
        r = subprocess.run(
            cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL,
            timeout=int(os.environ.get("KQUANT_LLAMACPP_TIMEOUT", "1200")))
    finally:
        os.unlink(pf)
    m = re.search(r"prompt eval time.*?/\s*(\d+)\s*tokens", r.stderr)
    return r.stdout, (int(m.group(1)) if m else None)


# tests
@pytest.mark.parametrize("arch", CANDIDATE_ARCHES)
def test_long_decode_integrity(arch, gguf_index):
    path = _require(gguf_index, arch)
    model, config, tok = _load(path)
    vocab = int(config["vocab_size"])
    n = min(TARGET, _ctx(config) - 8)

    # Suppress EOS so the decode actually reaches depth ``n``. An instruct model
    # given a complete-looking prompt emits EOS within a few dozen tokens, and
    # force-decoding *past* EOS produces out-of-distribution single-token spam
    # that is the model's, not a loader bug (and that llama.cpp never shows - it
    # halts at EOS). With EOS masked the decode runs to depth and the signals
    # below stay meaningful: a real attention bug at length yields NaN /
    # out-of-range / single-token collapse, while benign looping repeats
    # *phrases* so the single-token run stays small.
    eos_ids = _eos_ids(tok)

    def _suppress_eos(_tokens, logits):
        for e in eos_ids:
            logits[:, e] = -float("inf")
        return logits

    procs = [_suppress_eos] if eos_ids else None
    seed_ids = tok.encode(_SEED)
    # Belt and braces with the logit mask above: stream_generate also STOPS
    # at an EOS token id, and a NaN-corrupted argmax could land on one and
    # end the run before depth; clearing the EOG set keeps it decoding.
    from gmlx.overgen import suppressed_eos
    with suppressed_eos(tok):
        ids, finite, _ = _greedy(model, tok, seed_ids, n, logits_processors=procs)

    assert len(ids) > 0
    assert all(0 <= t < vocab for t in ids), "token id out of range"
    assert finite, "non-finite logprob during long decode (NaN/corruption)"
    # A NaN/spam collapse pins one token for hundreds of steps from early on;
    # semantic looping repeats *phrases*, so the single-token run stays modest.
    # Scoped to the in-distribution window (see _SPAM_WINDOW); finite/in-range
    # above already cover NaN/garbage to full depth.
    early = ids[:min(len(ids), _SPAM_WINDOW)]
    max_run = _max_run(early)
    assert max_run < 256, (
        f"single token spammed {max_run}x in first {len(early)} tokens "
        f"- degeneration")


@pytest.mark.parametrize("arch", CANDIDATE_ARCHES)
def test_long_prefill_parity(arch, gguf_index, llamacpp_bin):
    from mlx_lm.generate import stream_generate

    if arch in PARITY_SKIP:
        pytest.skip(f"{arch}: {PARITY_SKIP[arch]}")
    path = _require(gguf_index, arch)
    model, config, tok = _load(path)

    ceiling = min(TARGET, _ctx(config) - 32, PARITY_WINDOW.get(arch, 1 << 30))
    prompt_text, ids = _build_prompt(tok, ceiling)

    # Resident-memory discipline. llama.cpp (-ngl 99) wires its own full copy
    # of the model, so release ours - and MLX's buffer cache - before launching
    # it; two wired copies of a 40-80 GB model is how a 128 GB box earns a
    # watchdog panic. The zero-copy reload afterwards is seconds. And cap -c at
    # what the run actually needs: llama.cpp pre-allocates the ENTIRE -c KV
    # cache up front, and a native-512k-context model (seed_oss: 256 KiB/token)
    # at full -c is a 128 GiB allocation before the first token.
    del model
    mx.clear_cache()
    llama_ctx = min(_ctx(config), len(ids) + 24 + 64)
    llama_cont, n_prompt = _llama_complete(
        llamacpp_bin, path, prompt_text, n=24, ctx=llama_ctx)
    model, _, _ = _load(path)

    # llama.cpp prepends BOS by default; match it when it did (SPM archs) and
    # leave qwen2-style (add_bos_token=False) tokenizers alone.
    if (n_prompt is not None and n_prompt == len(ids) + 1
            and tok.bos_token_id is not None):
        ids = [tok.bos_token_id] + ids

    # Tripwire: the parity contract is RAW completion on BOTH sides (-no-cnv
    # disables llama.cpp's chat template; our generate_step gets bare ids).
    # If llama.cpp ever tokenizes the prompt to a different count - a chat
    # template applied despite -no-cnv (thinking-template models are the
    # suspects), a tokenizer mismatch, or context truncation - the text
    # comparison below would diverge for a non-loader reason. Fail labeled.
    if n_prompt is not None and n_prompt != len(ids):
        pytest.fail(
            f"{arch}: llama.cpp saw {n_prompt} prompt tokens vs our "
            f"{len(ids)} - chat template applied in -no-cnv mode, tokenizer "
            f"mismatch, or truncation; parity would be meaningless. Check "
            f"llama.cpp's stderr/template handling before blaming the loader.")

    # Greedy-decode through the self-wiring stream_generate (see _greedy). It
    # stops at the first EOS without yielding it, so the comparison ends where
    # llama.cpp stops. Without that, a long prompt the model wants to end
    # makes mlx keep emitting tokens past EOS while llama.cpp halts, diverging
    # on rendering (<eos> vs "[end of text]") rather than on attention.
    mlx_ids = [int(r.token)
               for r in stream_generate(model, tok, mx.array(ids), max_tokens=24)]
    mlx_text = tok.decode(mlx_ids)

    # llama.cpp prints "[end of text]" when it greedily emits EOG; strip it so
    # only the generated continuation remains.
    llama_cont = re.sub(r"\s*\[end of text\]\s*$", "", llama_cont)

    # Normalize leading whitespace: SPM tokenizers fold the leading space into
    # the first token (U+2581), so llama.cpp emits it while mlx's mid-sequence
    # decode may not - a boundary artifact, not a divergence. A real attention bug
    # diverges in the actual content, which survives this strip.
    mlx_n, llama_n = mlx_text.strip(), llama_cont.strip()
    common = os.path.commonprefix([mlx_n, llama_n])
    note = "" if arch not in PARITY_WINDOW else f" (capped at window {ceiling})"
    msg = (f"{arch}: prefill@{ceiling} diverged immediately{note}\n"
           f"  mlx  : {mlx_n[:80]!r}\n  llama: {llama_n[:80]!r}")
    print(f"[needle] llama={_NEEDLE in llama_n} mlx={_NEEDLE in mlx_n} "
          f"common={common[:24]!r} (reference-gated; see -rP)")
    # Use llama.cpp as the reference: require agreement for its full
    # continuation (when short, e.g. the model ends right after completing a
    # phrase) or at least the first 16 chars (when long). A real attention bug
    # diverges in the first few chars -> tiny shared prefix -> fail. Both ending
    # immediately at EOS (llama_n == "") is itself agreement and passes.
    # A shared prefix containing the needle is ALSO agreement: the cloze
    # answer is the informative content, and the "document" effectively ends
    # there - past it the distribution flattens and two correct engines
    # legally tie-flip (measured on Hy3 @16k: both emit 'QZ-4812.' then
    # free-wheel differently), the artifact regime the repeated-seed body is
    # designed to avoid.
    assert _NEEDLE in common or len(common) >= min(16, len(llama_n)), msg
    # Needle recall, reference-gated: when llama.cpp retrieves the planted
    # code from ~100 tokens deep, we must too - this is the check with
    # full-depth discriminative power (prefix agreement alone is satisfiable
    # from the tail window). When the reference itself fails retrieval
    # (model/quant too weak), parity falls back to agreement only rather
    # than failing the sweep for a non-loader reason.
    if _NEEDLE in llama_n:
        assert _NEEDLE in mlx_n, (
            f"{arch}: llama.cpp retrieved the needle {_NEEDLE!r} from depth "
            f"but mlx did not - mid-context attention defect\n"
            f"  mlx  : {mlx_n[:80]!r}\n  llama: {llama_n[:80]!r}")
