# Adding a GGUF architecture

What is required for a new model family to become a supported architecture, and
the acceptance gate every family clears before its row appears in the
[coverage matrix](../arch-coverage.md).

Precondition: a GGUF arch needs a model class for its `model_type`. That class
normally comes from the installed mlx-lm or mlx-vlm, and gmlx supplies only the
tensor map and the config.

A few families have no upstream class at all (kimi-k3, muse-glimmer). gmlx
vendors the model math for those, in its own module, inserted into the upstream
namespace so a later upstream implementation takes precedence. Vendoring is the exception.
It is justified only when the family cannot otherwise be supported, and it adds
two obligations: numeric parity against llama.cpp, and a collision check that
reports the collision once upstream publishes its own class.

## What the work involves

The engine is architecture-generic and data-driven: the load pipeline and the
module-swap code are never edited per arch. A new family adds:

- a tensor-name map from the GGUF's naming to the mlx-lm model class's
  parameter paths,
- a config synthesizer that reconstructs the exact `ModelArgs` the model class
  expects from the GGUF's key-value metadata (and, where the metadata is lossy,
  from tensor shapes),
- an architecture-table row that the CLI, preflight, and coverage matrix
  derive from,

and, when the family diverges from the canonical layouts, the parts that make
it diverge: per-tensor remap overrides, wire-byte transforms for fused or
permuted weight layouts, occasionally a new module class or a
tokenizer-classifier branch.

That last list is where most of the effort goes, and it varies widely. A clean
Llama-layout family can be supported with near-zero per-arch code in a few hours.
Hybrids and exotic layouts (SSM mixes, MLA attention, MoE variants with biased
projections, fused expert tensors, new float formats) take significant engineering
and debugging time. Do not estimate the work from the simplest case. Vision
and audio towers are a separate track with the same gate rules
([vlm.md](../vlm.md) lists what is supported).

## Why the gate is strict

The characteristic failure modes of a mis-ported architecture are silent. A
wrong rope layout or a bias assigned to a quantized weight slot still produces
fluent, plausible text on short prompts; the error only appears far into a
long context. That is why fluent generation does not count as done, and why
the parity requirement is 16k tokens. The same standard also applies in reverse:
when every public GGUF of a family is broken upstream, the loader gates the
family off by name with the reason (the current `gemma3n` case) rather than
load cleanly into wrong weights.

## The acceptance gate (all must pass)

An architecture is done when:

- Strict load: `load_model` builds, swaps, and `load_weights` leaves no
  parameter unfilled. The loader's unfilled-params warning must be empty.
- Coherent short generation: a chat model answers "capital of France?" with
  Paris in 20 greedy tokens.
- No-loop: ~300 greedy tokens with no n-gram (window 8) repeating 4+ times.
- 16k long-context parity vs llama.cpp:
  `tests/gen/test_long_context.py::test_long_prefill_parity`. A >=16k-token prompt,
  greedy-decoded, agrees as text with llama.cpp on the same file. Short-prompt
  parity is necessary but not sufficient: rope, KV-cache, GQA-layout, and
  permute bugs only surface at depth. Prepend BOS for `add_bos_token=True` archs
  and match llama.cpp's prompt token count, so a tokenization delta is not
  misread as a model bug. If the installed mlx-lm has a known context limitation
  for the family (e.g. a missing sliding-window implementation), cap the
  comparison window and document it in the arch notes.
- Degeneration check: `test_long_decode_integrity`. A long EOS-suppressed
  greedy decode with every token id in range, every step's logprob finite, and
  no single-token repetition. Semantic looping on a tiny model is expected. NaNs and
  out-of-range ids are not.
- Bench sanity: prefill/decode throughput on one real model, compared
  against llama.cpp on the same file. A large unexplained deficit is usually a
  contiguity or layout bug, not MLX itself.
- Route check at depth: run a >=16k-context decode (and an MTP round if the
  family has a draft head) with `GMLX_SDPA_DEBUG=1` and confirm attention
  uses a fused route (`gqa_decode`/`fa_decode`/`fa_verify`/`verify_gemm`/
  `sdpa_vector`), not `stock`. A new family's head geometry (head_dim, GQA
  ratio, verify fold width) can silently miss every eligibility gate and incur a
  materialized-scores penalty that only shows at depth. `GMLX_ROUTE_LOG=1`
  prints per-route call counts at exit; a one-shot warning also fires if a
  verify-shaped causal call at depth falls back to stock. For MTP families, check
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
  pytest tests/gen/test_long_context.py -k <arch>

# Coverage table stays truthful
python scripts/check-coverage.py --check --strict
```

The tiers these tests run in, and how to select a GGUF-gated tier, are in
[testing.md](testing.md).

## Requesting or contributing a family

To request a family, open an issue with a link to the GGUF (or its
Hugging Face repo) and the model's `general.architecture` string;
`gmlx validate <ref>` prints it without downloading the file. Contributions
are welcome: a new-architecture PR is expected to pass the acceptance gate
above, add a config-synth fixture test, and regenerate the coverage matrix.
