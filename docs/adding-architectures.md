# Adding a GGUF architecture

What it takes for a new model family to become a supported architecture, and
the acceptance gate every family clears before its row appears in the
[coverage matrix](arch-coverage.md).

Precondition: a GGUF arch is only reachable if the installed mlx-lm defines a
`class Model` for the corresponding `model_type`. If mlx-lm doesn't implement
the architecture, support is blocked upstream. gmlx never reimplements
model math.

## The shape of the work

The engine is architecture-generic and data-driven: the load pipeline and the
module-swap machinery are never edited per arch. A new family adds:

- a tensor-name map from the GGUF's naming to the mlx-lm model class's
  parameter paths,
- a config synthesizer that reconstructs the exact `ModelArgs` the model class
  wants from the GGUF's key-value metadata (and, where the metadata is lossy,
  from tensor shapes),
- an architecture-table row that the CLI, preflight, and coverage matrix
  derive from,

and, when the family diverges from the canonical layouts, the parts that make
it diverge: per-tensor remap overrides, wire-byte transforms for fused or
permuted weight layouts, occasionally a new module class or a
tokenizer-classifier branch.

That last list is where the effort lives, and it varies widely. A clean
Llama-layout family can resolve with near-zero per-arch code in an afternoon.
Hybrids and exotic layouts (SSM mixes, MLA attention, MoE variants with biased
projections, fused expert tensors, new float formats) are real engineering
with real debugging time. Don't judge the work by the shortest case. Vision
and audio towers are a parallel track with the same gate philosophy
([vlm.md](vlm.md) lists what's supported).

## Why the gate is strict

The characteristic failure modes of a mis-ported architecture are silent. A
wrong rope layout or a bias landing on a quantized weight slot still produces
fluent, plausible text on short prompts; the damage only surfaces deep into a
long context. That is why fluent generation does not count as done, and why
the parity bar sits at 16k tokens. The same standard cuts the other way too:
when every public GGUF of a family is broken upstream, the loader gates the
family off by name with the reason (the current `gemma3n` case) rather than
load cleanly into wrong weights.

## The acceptance gate (all must pass)

An architecture is not done when it generates fluent text. It's done when:

- Strict load: `load_model` builds, swaps, and `load_weights` leaves no
  parameter unfilled. The loader's unfilled-params warning must be empty.
- Coherent short generation: a chat model answers "capital of France?" with
  Paris in 20 greedy tokens.
- No-loop: ~300 greedy tokens with no n-gram (window 8) repeating 4+ times.
- 16k long-context parity vs llama.cpp:
  `tests/test_long_context.py::test_long_prefill_parity`. A >=16k-token prompt,
  greedy-decoded, agrees as text with llama.cpp on the same file. Short-prompt
  parity is necessary but not sufficient: rope, KV-cache, GQA-layout, and
  permute bugs only surface at depth. Prepend BOS for `add_bos_token=True` archs
  and match llama.cpp's prompt token count, so a tokenization delta isn't
  misread as a model bug. If the installed mlx-lm has a known context limitation
  for the family (e.g. a missing sliding-window implementation), cap the
  comparison window and document it in the arch notes.
- Degeneration check: `test_long_decode_integrity`. A long EOS-suppressed
  greedy decode with every token id in range, every step's logprob finite, and
  no single-token spam. Semantic looping on a tiny model is expected. NaNs and
  out-of-range ids are not.
- Bench sanity: prefill/decode throughput on one real model, compared
  against llama.cpp on the same file. A large unexplained deficit is usually a
  contiguity or layout bug, not "MLX being slow."
- Route check at depth: run a >=16k-context decode (and an MTP round if the
  family has a draft head) with `GMLX_SDPA_DEBUG=1` and confirm attention
  lands on a fused route (`gqa_decode`/`fa_decode`/`fa_verify`/`verify_gemm`/
  `sdpa_vector`), not `stock`. A new family's head geometry (head_dim, GQA
  ratio, verify fold width) can silently miss every eligibility gate and pay a
  materialized-scores penalty that only shows at depth. `GMLX_ROUTE_LOG=1`
  prints per-route call counts at exit; a one-shot warning also fires if a
  verify-shaped causal call at depth falls to stock. For MTP families, check
  the verify branch with `GMLX_MTP_DEBUG=1` (`[mtp] verify branch: ...`).
  Serve perf claims must be certified in the actual server process (the
  round profile works there: `GMLX_ROUND_PROFILE=1` +
  `GMLX_ROUND_LOG=/tmp/rounds.tsv`), not an in-process harness.
- Repo gates green: `pytest` (the CPU tier) and
  `scripts/check-coverage.py --check --strict`, with `docs/arch-coverage.md`
  regenerated.

## Smoke commands

```sh
# What resolves, what skips, which codecs, without running the model
gmlx run model.gguf --report-only

# Coherence
gmlx run model.gguf --prompt "What is the capital of France?" --max-tokens 20

# Long-context parity + decode integrity (needs a real GGUF + llama.cpp;
# without KQUANT_LLAMACPP_BIN the parity half skips and only integrity runs)
KQUANT_TEST_GGUF_DIR=~/models KQUANT_LLAMACPP_BIN=/path/to/llama-completion \
  pytest tests/test_long_context.py -k <arch>

# Coverage table stays truthful
python scripts/check-coverage.py --check --strict
```

## Requesting or contributing a family

Missing a family you care about? Open an issue with a link to the GGUF (or its
Hugging Face repo) and the model's `general.architecture` string;
`gmlx validate <ref>` prints it without downloading the file. Contributions
are welcome: a new-architecture PR is expected to pass the acceptance gate
above, add a config-synth fixture test, and regenerate the coverage matrix.
