# gmlx

gmlx runs GGUF models on Apple Silicon. It runs the K-quant and IQ-quant
files that the community publishes, unchanged, on Metal kernels for Apple's
MLX framework. One command, `gmlx`, chats with a model in the terminal,
serves it over OpenAI and Anthropic compatible APIs, connects coding agents
to it and talks to it by voice.

These pages are the starting points:

- To run your first model, follow the [Quickstart](quickstart.md).
- To serve models to apps and agents, read
  [Configuration](config.md) and [Agents and chat apps](launch.md).
- To look up a flag, a key or an endpoint, use the
  [CLI reference](cli.md), [Configuration](config.md) or the
  [HTTP API](api.md).
- To work on gmlx itself, start with [Internals](internals/README.md).

## Getting started

- [Installation](installation.md): Homebrew, uv and pip, the optional
  features, upgrading and removal.
- [Quickstart](quickstart.md): A first model, the server, a request and a
  connected client.
- [Migrating from other tools](migrating.md): What carries over from
  llama.cpp, Ollama and LM Studio.

## Serving

- [Configuration](config.md): `gmlx.yaml`, its models, its profiles and how a
  request gets its settings.
- [Agents and chat apps](launch.md): Claude Code, other coding agents and Open
  WebUI, set up by `gmlx launch`.
- [Container mode](launch-container.md): A client in an Apple container that
  sees only the folders you share.
- [Custom container images](container-images.md): Packages, Containerfiles,
  ready-made images and services for container mode.
- [Container security](container-security.md): What a client in a container
  can still reach on the Mac and the server, and the limits of a session.
- [Menu bar app](menubar.md): Server status and controls in the macOS menu
  bar.
- [Speech, embeddings and rerank](services.md): The services a server can host
  beside chat models.
- [RAG pipelines](rag.md): Retrieval with the embeddings and rerank services.
- [Structured decisions](decisions.md): A probability for each answer to a
  fixed set of questions.

## Clients

- [Chat](chat.md): The terminal chat client, its commands, sessions and
  themes.
- [Voice chat](talk.md): Talking to a model with `gmlx talk`.
- [Assistant](assistant.md): Tools and long-term memory for chat, voice and
  served models.

## Models

- [Supported architectures](arch-coverage.md): The GGUF architectures gmlx
  loads, with their caveats.
- [Vision and audio](vlm.md): Multimodal models and their `mmproj` files.
- [Models larger than memory](streaming.md): Mixture-of-experts models that
  stream their experts from disk.
- [LoRA adapters](lora.md): Training an adapter and serving several on one
  base model.
- [Distillation](distill.md): Teaching a small model a document or a larger
  model's behavior.

## Performance

- [Performance tuning](performance.md): What makes a model fast, measuring,
  and choosing a quant.
- [Speculative decoding](speculative-decoding.md): Faster decoding with a
  drafter, with the same output.
- [Prompt cache](prompt-cache.md): Skipping prefill for prompts the server has
  seen.
- [Concurrent requests](concurrency.md): Batching, admission pacing and shared
  prompts.
- [Memory and the KV cache](memory.md): How much memory a model and its
  context take.
- [KV cache quantization](kv-quantization.md): Storing the context in fewer
  bits.
- [Benchmarks](benchmarks.md): gmlx against llama.cpp on the same files, with
  the method.

## Help

- [Troubleshooting](troubleshooting.md): `gmlx doctor`, common failures, and
  where gmlx keeps its files.
- [Glossary](glossary.md): The terms these pages use.

## Reference

- [CLI reference](cli.md): Every command and flag.
- [Configuration](config.md): Every key of `gmlx.yaml`.
- [Family defaults](family-defaults.md): The sampling defaults and intents of
  each model family.
- [HTTP API](api.md): The endpoints and request features.
- [Environment variables](env-vars.md): The variables a user can set.
- [Python API](python.md): Using gmlx from Python.

## Development

- [Internals](internals/README.md): How gmlx works inside.
- [Adding a GGUF architecture](internals/adding-architectures.md): What
  supporting a new model family involves.
- [Contributing](../CONTRIBUTING.md): Development setup, tests and commit
  style.
- [Changelog](../CHANGELOG.md): What changed in each release.
