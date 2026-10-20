# Internals

gmlx internals cover the serving path, the caches, speculative batching,
model support and the tooling for tests and upstream upgrades. The user
guides start at the [documentation home](../README.md), and the development
setup is in the [contributing guide](../../CONTRIBUTING.md).

## How it works

- [Serving architecture](serving-architecture.md): The path of a request
  through the server, and the scheduling policy gmlx adds to mlx-vlm.
- [Speculative batching](speculative-batching.md): How speculative decoding
  and continuous batching run together.
- [Prompt cache internals](prompt-cache.md): The cache tier of each
  architecture, the cache layers and the reuse counters.
- [Structured reads](structured-reads.md): How `/v1/systemone` answers
  questions from a denoise read, and its parity with the vLLM example.
- [Letter readout](letter-readout.md): How `/v1/systemone` answers on
  autoregressive models, and its parity with OpenJev's helper.
- [Hadamard-folded GGUFs](hadamard-fold.md): Weights stored under a Hadamard
  rotation, and where gmlx undoes it.
- [Distillation internals](distill.md): How the teacher pass and the
  training head bound their memory, and the costs of the worked run.
- [Image serving](image-serving.md): How image jobs share the GPU with chat
  replies, and how the server plans their memory.
- [Image step routes](image-steps.md): The contract that lets a sampler in
  another program, such as ComfyUI, run an image model one step at a time.

## Working on gmlx

- [Adding a GGUF architecture](adding-architectures.md): What supporting a
  model family involves, and its acceptance gate.
- [Testing](testing.md): The test tiers, GGUF-gated runs and the end-to-end
  harnesses.
- [CI and the GPU gate](ci.md): What a pull request needs to merge, the GPU
  gate run on a Mac, and what each CI leg runs.
- [Upstream releases](upstream-upgrades.md): The copy of mlx-vlm and
  mlx-lm that gmlx carries, how it takes upstream changes, and moving mlx.
- [Debug switches](debug-switches.md): Environment variables for isolating
  defects, including the prompt cache switches.

## Measurements

- [Streaming measurements](streaming-measurements.md): The data behind the
  lossless and lossy settings of streamed models.
- [Structured read measurements](structured-read-measurements.md): Prefill,
  read, sample, step and thought timings on DiffusionGemma, and the letter
  readout's timings and accuracy on OpenJev.
