# Adding a GGUF architecture

What is required for a new model family to become a supported architecture,
and the acceptance gate a family clears before its row appears in the
[coverage matrix](../arch-coverage.md).

A GGUF arch first needs a model class for its `model_type`. That class
normally comes from the installed mlx-lm or mlx-vlm, and gmlx supplies only
the tensor map and the config.

A few families, kimi-k3 and muse-glimmer among them, have no upstream class
at all. gmlx vendors the model math for those in its own module, inserted
into the upstream namespace so a later upstream implementation takes
precedence. Vendoring is the exception. It is justified only when the family
cannot otherwise be supported, and it adds two obligations. The vendored math
must match llama.cpp numerically, and a collision check must report once
upstream publishes its own class.

## What the work involves

The engine is architecture-generic and data-driven, and neither the load
pipeline nor the module-swap code is edited per arch. A new family adds
three things:

- a tensor-name map from the GGUF's naming to the mlx-lm model class's
  parameter paths,
- a config synthesizer that reconstructs the exact `ModelArgs` the model
  class expects from the GGUF's key-value metadata, or from tensor shapes
  where the metadata is lossy,
- an architecture-table row that the CLI, preflight and coverage matrix
  derive from.

A family that diverges from the canonical layouts also adds the parts that
make it diverge. Those are per-tensor remap overrides, wire-byte transforms
for fused or permuted weight layouts, and occasionally a new module class or
a tokenizer-classifier branch.

That last list is where most of the effort goes, and it varies widely. A
clean Llama-layout family can be supported with almost no per-arch code in a
few hours. Hybrids and exotic layouts take significant engineering and
debugging time. SSM mixes, MLA attention, MoE variants with biased
projections, fused expert tensors and new float formats all fall in that
group. Do not estimate the work from the simplest case. Vision and audio
towers are a separate track with the same gate rules, and
[vlm.md](../vlm.md) lists what is supported.

## Why the gate is strict

The characteristic failure modes of a mis-ported architecture are silent. A
wrong rope layout or a bias assigned to a quantized weight slot still
produces fluent, plausible text on short prompts. The error only appears far
into a long context. That is why fluent generation does not count as done,
and why the parity requirement is 16k tokens. The standard also applies in
reverse. When every public GGUF of a family is broken upstream, as with
`gemma3n` today, the loader gates the family off by name with the reason
instead of loading cleanly into wrong weights.

## The acceptance gate

An architecture is done when all of the following pass.

- Strict load. `load_model` builds, swaps, and `load_weights` leaves no
  parameter unfilled. The loader's unfilled-params warning must be empty.
- Coherent short generation. A chat model answers "capital of France?" with
  Paris in 20 greedy tokens.
- No looping. About 300 greedy tokens contain no 8-token n-gram repeated
  four or more times.
- Long-context parity against llama.cpp at 16k, from
  `tests/gen/test_long_context.py::test_long_prefill_parity`. A prompt of
  16k tokens or more, greedy-decoded, agrees as text with llama.cpp on the
  same file. Short-prompt parity is necessary but not sufficient, because
  rope, KV-cache, GQA-layout and permute bugs only surface at depth. Prepend
  BOS for archs with `add_bos_token=True` and match llama.cpp's prompt token
  count. Otherwise a tokenization delta is misread as a model bug. If the
  installed mlx-lm has a known context limitation for the family, such as a
  missing sliding-window implementation, cap the comparison window and
  document it in the arch notes.
- Degeneration check, from `test_long_decode_integrity`. A long
  EOS-suppressed greedy decode keeps each token id in range and each step's
  logprob finite, with no single-token repetition. Semantic looping on a tiny
  model is expected. NaNs and out-of-range ids are not.
- Bench sanity. Prefill and decode throughput on one real model, compared
  against llama.cpp on the same file. A large unexplained deficit is usually
  a contiguity or layout bug, not MLX itself.
- Route check at depth. Run a decode at 16k context or more, plus an MTP
  round if the family has a draft head, with `GMLX_SDPA_DEBUG=1`. Confirm
  attention uses a fused route and not `stock`. The fused routes are
  `gqa_decode`, `fa_decode`, `fa_verify`, `verify_gemm` and `sdpa_vector`. A
  new family's head geometry can silently miss the eligibility gates and pay
  a materialized-scores penalty that only shows at depth. `GMLX_ROUTE_LOG=1`
  prints per-route call counts at exit, and a one-shot warning fires if a
  verify-shaped causal call at depth falls back to stock. For MTP families,
  check the verify branch with `GMLX_MTP_DEBUG=1`, which logs a line starting
  `[mtp] verify branch:`. Serve performance claims must be certified in the
  actual server process and not in an in-process harness. The round profile
  works there through `GMLX_ROUND_PROFILE=1` with `GMLX_ROUND_LOG` set to a
  TSV path.
- Repo gates green. The CPU tier of `pytest` passes, and
  `scripts/check-coverage.py --check --strict` passes with
  `docs/arch-coverage.md` regenerated.

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

The tiers these tests run in, and how to select a GGUF-gated tier, are in
[testing.md](testing.md).

## Requesting or contributing a family

To request a family, open an issue with a link to the GGUF or its Hugging
Face repo and the model's `general.architecture` string. `gmlx validate <ref>`
prints that string without downloading the file. Contributions are welcome.
A new-architecture PR is expected to pass the acceptance gate, add a
config-synth fixture test, and regenerate the coverage matrix.
