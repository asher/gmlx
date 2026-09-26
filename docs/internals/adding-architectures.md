# Adding a GGUF architecture

A new model family becomes a supported architecture once gmlx can map its
tensors and config, and it clears an acceptance gate before its row
appears in [Supported architectures](../arch-coverage.md).

Each GGUF architecture needs a model class for its `model_type`. The class
normally comes from the installed mlx-lm or mlx-vlm, so gmlx supplies only
the tensor map and the config.

Some families, kimi-k3 and muse-glimmer among them, have no upstream class
at all. gmlx vendors the model math for those in its own module, inserted
into the upstream namespace so that a later upstream implementation takes
precedence. Vendoring is the exception, justified only when the family
cannot otherwise be supported, because it adds two obligations. The
vendored math must match llama.cpp numerically, and a collision check must
report once upstream publishes its own class.

## What the work involves

The engine is architecture-generic and data-driven, so neither the load
pipeline nor the module-swap code is edited per arch. A new family adds
three things:

- A tensor-name map from the GGUF's naming to the parameter paths of the
  mlx-lm model class.
- A config synthesizer that rebuilds the exact `ModelArgs` the model class
  expects, from the GGUF's key-value metadata or, where the metadata is
  lossy, from tensor shapes.
- An architecture-table row, from which the CLI, preflight and the coverage
  matrix derive.

A family that diverges from the canonical layouts also adds the parts that
make it diverge. These are per-tensor remap overrides, wire-byte transforms
for fused or permuted weight layouts, and occasionally a new module class
or a tokenizer-classifier branch. That is where the effort goes. A clean
Llama-layout family needs almost no per-arch code and a few hours of work.
Hybrids and unusual layouts take much more engineering and debugging time.
Examples are state-space layers mixed with attention, the compressed-KV
[MLA](../glossary.md#mla) attention of the DeepSeek lineage, MoE variants
with biased projections, fused expert tensors and new float formats.
Estimate the work from the hardest part of the family, not the simplest
case.

Vision and audio towers are a separate track with the same gate rules, and
[Vision and audio](../vlm.md) lists what is supported.

## Why the gate is strict

The characteristic failure modes of a mis-ported architecture are silent.
A wrong rope layout or a bias assigned to a quantized weight slot still
produces fluent, plausible text on short prompts. The error appears only
far into a long context, which is why fluent generation does not count as
done and why parity is required at 16k tokens. The same standard works in
reverse. When every public GGUF of a family is broken upstream, as with
`gemma3n`, the loader gates the family off by name with the reason instead
of loading cleanly into wrong weights.

## The acceptance gate

An architecture is done when all of the following pass.

- Strict load. `load_model` builds and swaps, and `load_weights` leaves no
  parameter unfilled. The loader's unfilled-params warning must be empty.
- Coherent short generation. A chat model answers "capital of France?" with
  Paris in 20 greedy tokens.
- No looping. About 300 greedy tokens contain no 8-token n-gram repeated
  four or more times.
- Long-context parity against llama.cpp at 16k, from
  `tests/gen/test_long_context.py::test_long_prefill_parity`. A prompt of
  16k tokens or more, greedy-decoded, agrees as text with llama.cpp on the
  same file. Short-prompt parity is not enough, because rope, KV-cache,
  grouped-query (GQA) head-layout and permute bugs surface only at depth.
- Degeneration check, from `test_long_decode_integrity`. A long
  EOS-suppressed greedy decode keeps each token id in range and each step's
  logprob finite, with no single-token repetition. Semantic looping on a tiny
  model is expected. NaNs and out-of-range ids are not.
- Bench sanity. Prefill and decode throughput on one real model are
  compared against llama.cpp on the same file. A large unexplained deficit is usually
  a contiguity or layout bug, not MLX itself.
- Route check at depth. Run a decode at 16k context or more, plus an MTP
  round if the family has a draft head, with `GMLX_SDPA_DEBUG=1`, and
  confirm attention uses one of the fused routes, `gqa_decode`, `fa_decode`,
  `fa_verify`, `verify_gemm` or `sdpa_vector`, rather than `stock`. A new
  family's head geometry can miss the eligibility gates without any error
  and pay a materialized-scores penalty that only shows at depth.
  `GMLX_ROUTE_LOG=1` prints per-route call counts at exit, and a one-shot
  warning fires if a verify-shaped causal call at depth falls back to stock.
  For MTP families, `GMLX_MTP_DEBUG=1` logs a line starting
  `[mtp] verify branch:` per round.
- Repo gates green. The CPU tier of `pytest` passes with a new fixture
  case for the family in `tests/load/test_config_synth.py`, and
  `scripts/check-coverage.py --check --strict` passes with
  `docs/arch-coverage.md` regenerated from the new table row.

Those two tests, the config-synth fixture and the parity run, are the
required deliverables of a new-architecture PR alongside the code.

## Smoke commands

```sh
# What resolves, what skips, which codecs, without running the model
gmlx run model.gguf --report-only

# Coherence
gmlx run model.gguf --prompt "What is the capital of France?" --max-tokens 20

# Long-context parity and decode integrity. Needs a real GGUF and llama.cpp.
# Without KQUANT_LLAMACPP_BIN the parity half skips and only integrity runs.
KQUANT_TEST_GGUF_DIR=~/models KQUANT_LLAMACPP_BIN=/path/to/llama-completion \
  pytest tests/gen/test_long_context.py -k <arch>

# Coverage table stays truthful
python scripts/check-coverage.py --check --strict
```

Two details keep the parity run valid. Prepend BOS for archs with
`add_bos_token=True` and match llama.cpp's prompt token count, or a
tokenization delta is misread as a model bug. If the installed mlx-lm has a
known context limitation for the family, such as a missing sliding-window
implementation, cap the comparison window and record that in the arch
notes. The tiers these tests run in and how to select a GGUF-gated tier are
in [Testing](testing.md).

## Requesting or contributing a family

To request a family, open an issue with a link to the GGUF or its Hugging
Face repo and the model's `general.architecture` string, which
`gmlx validate <ref>` prints without downloading the file. Contributions
are welcome, and a new-architecture PR is reviewed against
[the acceptance gate](#the-acceptance-gate).
