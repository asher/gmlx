# Adding a GGUF architecture

A new model family becomes a supported architecture once gmlx can map its
tensors and config, and it must clear an acceptance gate before its row
appears in [Supported architectures](../arch-coverage.md).

## Where the model class comes from

Each GGUF architecture needs a model class for its `model_type`. The set
of classes that mlx-lm and mlx-vlm ship changes with every release, so
check the installed packages first. When one of them has the class, gmlx
supplies only the tensor map and the config, and the `backend` field of
the architecture-table row names the package.

When neither package has the class, vendor the model into `gmlx/models/`
and add it to `_VENDORED_MLX_LM_MODULES` in `gmlx/load/arch_table.py`. A
module grafted into mlx-vlm, such as a multimodal model or a tool parser,
goes in `VENDORED_MLX_VLM_MODULES` in `gmlx/upstream/seams.py` instead.
The graft puts the copy in the upstream namespace, and an installed
release that ships the same module wins over it.
`vendored_upstream_collisions` in `gmlx/upstream/seams.py` reports each
vendored module that upstream now ships, which is the signal to drop the
copy. A vendored model must match llama.cpp numerically, the same as any
other.

## What the work involves

The engine is architecture-generic and data-driven, so neither the load
pipeline nor the module-swap code is edited per architecture. A new family
adds three things:

- A tensor-name map from the GGUF naming to the parameter paths of the
  model class.
- A config synthesizer that rebuilds the exact `ModelArgs` that the model
  class expects, from the GGUF metadata or, where the metadata is lossy,
  from tensor shapes.
- An architecture-table row, from which the CLI, preflight and the
  coverage table derive.

A family that diverges from the canonical layouts also adds the parts that
make it diverge. These are per-tensor remap overrides, wire-byte
transforms for fused or permuted weight layouts, and sometimes a new
module class or a tokenizer branch. That is where the effort goes. A
Llama-layout family needs almost no per-architecture code. Hybrids,
compressed-KV attention, fused expert tensors and new float formats take
much more work, so estimate from the hardest part of the family. Vision
and audio towers are a separate track with the same gate.

## Why the gate is strict

A mis-ported architecture fails silently. A wrong rope layout or a bias
assigned to a quantized weight slot still produces fluent text on short
prompts, and the error appears only far into a long context. Fluent
generation therefore does not count as done, and parity is required at 16k
tokens. When every public GGUF of a family is broken, the loader gates the
family off by name with the reason instead of loading it into wrong
weights.

## The acceptance gate

An architecture is done when all of the following pass.

- Strict load: `load_model` builds and swaps, and no parameter is left
  unfilled.
- Coherent short generation: A chat model answers a simple factual
  question correctly in a few greedy tokens.
- No looping: A few hundred greedy tokens contain no repeating n-gram.
- Long-context parity: `test_long_prefill_parity` in
  `tests/gen/test_long_context.py` agrees as text with llama.cpp on the
  same file, at 16k tokens or more.
- Degeneration check: In `test_long_decode_integrity`, a long
  EOS-suppressed greedy decode keeps each token id in range and each
  logprob finite.
- Bench sanity: Prefill and decode throughput on one real model are close
  to llama.cpp on the same file. A large unexplained deficit is usually a
  contiguity or layout bug.
- Route check at depth: A decode at 16k context or more takes a fused
  attention route under `GMLX_SDPA_DEBUG=1`, not `stock`. A new head
  geometry can miss the eligibility gates without an error and pay a
  penalty that shows only at depth. A one-shot warning fires when a
  verify-shaped call at depth falls back to stock. For MTP families, look
  for a line starting `[mtp] verify branch:` under `GMLX_MTP_DEBUG=1`.
  [Debug switches](debug-switches.md) describes these switches and
  `GMLX_ROUTE_LOG=1`.
- Repo gates: The CPU tier passes with a new fixture case for the family
  in `tests/load/test_config_synth.py`, and
  `scripts/check-coverage.py --check --strict` passes with
  `docs/arch-coverage.md` regenerated.

The config-synth fixture and the parity run are the required deliverables
of a new-architecture pull request, beside the code.

## Smoke commands

```sh
# What resolves, what skips, which codecs, without running the model
gmlx run model.gguf --report-only

# Coherence
gmlx run model.gguf --prompt "What is the capital of France?" --max-tokens 20

# Long-context parity and decode integrity. Needs a real GGUF and llama.cpp.
KQUANT_TEST_GGUF_DIR=~/models KQUANT_LLAMACPP_BIN=/path/to/llama-completion \
  pytest tests/gen/test_long_context.py -k <arch>

# Coverage table stays truthful
python scripts/check-coverage.py --check --strict
```

For a valid parity run, prepend BOS where the tokenizer asks for it and
match the prompt token count of llama.cpp, or a tokenization difference
reads as a model bug. When the upstream class has a known limit for the
family, cap the comparison window and record the limit in the notes of its
table row. [Testing](testing.md) describes the test tiers.

## Requesting or contributing a family

To request a family, open an issue with a link to the GGUF or its Hugging
Face repo and the model's `general.architecture` string, which
`gmlx validate <ref>` prints without downloading the file. A
new-architecture pull request is reviewed against
[the acceptance gate](#the-acceptance-gate).
