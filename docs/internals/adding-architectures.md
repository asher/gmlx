# Adding a GGUF architecture

A new model family becomes a supported architecture once gmlx can map its
tensors and config, and it must clear an acceptance gate before its row
appears in [Supported architectures](../arch-coverage.md).

## Where the model class comes from

Each GGUF architecture needs a model class for its `model_type`. Check
the copies of mlx-lm and mlx-vlm in `gmlx/_vendor` first. When one of them
has the class, gmlx supplies only the tensor map and the config, and the
architecture-table row's `backend` field names the package.

When neither package has the class, add the model to `gmlx/models/` and
give its `model_type` a row in `gmlx/models/owned.py`: `OWNED_LM_MODELS`
for a text model, `OWNED_VLM_MODELS` for a multimodal one. The vendored
loaders look the type up there before their own model packages, so the
gmlx class is the one that loads. An upstream class built for the Hugging
Face checkpoint does not match the GGUF tensor map and has no speculative
hooks, so the gmlx class must always win.
`tests/load/test_owned_model_routing.py` resolves every row through the
vendored loaders and fails when a type returns another class. A tool parser
registers with `register_tool_parser` in `gmlx/upstream/tool_parsers.py`
when its model loads. An owned model must match llama.cpp numerically, the
same as any other.

## What the work involves

The engine is architecture-generic and data-driven, so neither the load
pipeline nor the module-swap code is edited per architecture. A new family
adds three things:

- A tensor-name map translates the GGUF names to the model class's
  parameter paths.
- A config synthesizer rebuilds the exact `ModelArgs` that the model class
  expects, from the GGUF metadata or, where the metadata is lossy, from
  tensor shapes.
- An architecture-table row is the source from which the CLI, preflight
  and the coverage table derive.

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

## KV cache copies at decode depth

Serve builds decode step t+1 before step t has finished. Step t+1 appends
to each layer's K and V while the attention of step t may still read them.
When a command buffer that has not run yet still holds a layer's buffer,
the slice update copies the whole buffer instead of writing in place. The
output stays correct, but decode slows as the context grows.

A deep model splits each step into many command buffers and does not hit
this. A step that fits in a few command buffers does: a drafter or MTP
head with a few layers and its own cache, a model of fewer than about 8
layers, or an architecture whose step is a few large fused kernels.
`qwen4exp` is such an architecture.

To check, decode at 16k and at 128k tokens twice: once evaluating each
step before building the next, and once pipelined as serve runs it.
Compare `mx.get_peak_memory()` of the two. When the pipelined peak is
higher by about one layer's K and V buffer, the appends copy. The fix is
to store the cache in two segments, a base and a small tail that takes the
decode appends, as `QSAKVCache` in `gmlx/models/qwen4_exp/model.py` does.
Where it costs less, evaluate each step instead.

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
match llama.cpp's prompt token count, or a tokenization difference
reads as a model bug. When the upstream class has a known limit for the
family, cap the comparison window and record the limit in its table row's
notes. [Testing](testing.md) describes the test tiers.

## Requesting or contributing a family

To request a family, open an issue with a link to the GGUF or its Hugging
Face repo and the model's `general.architecture` string, which
`gmlx validate <ref>` prints without downloading the file. A
new-architecture pull request is reviewed against
[the acceptance gate](#the-acceptance-gate).
