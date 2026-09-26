# Internals

These pages describe how gmlx works inside, for people who change its
code. The user guides start at the [documentation home](../README.md), and
the development setup is in the [contributing guide](../../CONTRIBUTING.md).

## How it works

- [Serving architecture](serving-architecture.md): the path from GGUF bytes
  to a streamed response, and the scheduling policy gmlx adds to mlx-vlm
- [Speculative batching](speculative-batching.md): how speculative decoding
  and continuous batching run together
- [Prompt cache internals](prompt-cache.md): the cache tiers per
  architecture, reuse counters and switches
- [Structured reads](structured-reads.md): how `/v1/systemone` answers
  questions with one denoise step, and its parity with the vLLM example
- [Hadamard-folded GGUFs](hadamard-fold.md): weights stored under a
  Hadamard rotation, the header contract and rotation sharing
- [Distillation internals](distill.md): the memory arithmetic of the
  teacher pass and the training head

## Working on gmlx

- [Adding a GGUF architecture](adding-architectures.md): what supporting a
  model family involves, and its acceptance gate
- [Testing](testing.md): the test tiers, GPU-gated runs and the end-to-end
  harnesses
- [Upgrading mlx-vlm, mlx-lm and mlx](upstream-upgrades.md): moving the
  pinned upstream versions
- [Debug switches](debug-switches.md): environment variables for isolating
  defects

## Measurements

- [Streaming measurements](streaming-measurements.md): the data behind the
  lossless and lossy settings of streamed models
- [Structured read measurements](structured-read-measurements.md): prefill,
  read, sample, step and thought timings on DiffusionGemma
