# gmlx

gmlx runs GGUF models on Apple Silicon. It runs the K-quant and IQ-quant
files that the community publishes, unchanged, on Metal kernels for Apple's
MLX framework. One command, `gmlx`, chats with a model in the terminal,
serves it over OpenAI and Anthropic compatible APIs, connects coding agents
to it and talks to it by voice.

Where to start depends on what you want:

- To run your first model, follow the [Quickstart](quickstart.md).
- To serve models to apps and agents, read
  [Configuration](config.md) and [Agents and chat apps](launch.md).
- To look up a flag, a key or an endpoint, use the
  [CLI reference](cli.md), [Configuration](config.md) or the
  [HTTP API](api.md).
- To work on gmlx itself, start with [Internals](internals/README.md).

## Getting started

- [Installation](installation.md): Homebrew, uv and pip, the optional
  features, upgrading and removal
- [Quickstart](quickstart.md): a first model, the server, a request and a
  connected client
- [Migrating from other tools](migrating.md): what carries over from
  llama.cpp, Ollama and LM Studio

## Serving

- [Configuration](config.md): `gmlx.yaml`, its models, its profiles and
  how a request gets its settings
- [Agents and chat apps](launch.md): Claude Code, other coding agents and
  Open WebUI, set up by `gmlx launch`
- [Menu bar app](menubar.md): server status and controls in the macOS menu
  bar
- [Speech, embeddings and rerank](services.md): the services a server can
  host beside chat models
- [RAG pipelines](rag.md): retrieval with the embeddings and rerank
  services
- [Structured decisions](decisions.md): a probability for each answer to a
  fixed set of questions

## Clients

- [Chat](chat.md): the terminal chat client, its commands, sessions and
  themes
- [Voice](talk.md): talking to a model with `gmlx talk`
- [Assistant](assistant.md): tools and long-term memory for chat, voice
  and served models

## Models

- [Supported architectures](arch-coverage.md): the GGUF architectures gmlx
  loads, with their caveats
- [Vision and audio](vlm.md): multimodal models and their `mmproj` files
- [Models larger than memory](streaming.md): mixture-of-experts models that
  stream their experts from disk
- [LoRA adapters](lora.md): training an adapter and serving several on one
  base model
- [Distillation](distill.md): teaching a small model a document or a
  larger model's behavior

## Performance and help

- [Performance tuning](performance.md): the speed features and what each
  setting costs
- [Benchmarks](benchmarks.md): gmlx against llama.cpp on the same files,
  with the method
- [Troubleshooting](troubleshooting.md): `gmlx doctor`, common failures,
  and where gmlx keeps its files
- [Glossary](glossary.md): the terms these pages use

## Reference

- [CLI reference](cli.md): every command and flag
- [Configuration](config.md#server): every key of `gmlx.yaml`
- [Family defaults](family-defaults.md): the sampling defaults and intents
  of each model family
- [HTTP API](api.md): the endpoints and request features
- [Environment variables](env-vars.md): the variables a user can set
- [Python API](python.md): using gmlx from Python

## Development

- [Internals](internals/README.md): how gmlx works, for contributors
- [Adding a GGUF architecture](internals/adding-architectures.md): what
  supporting a new model family involves
- [Contributing](../CONTRIBUTING.md): development setup, tests and commit
  style
- [Changelog](../CHANGELOG.md): what changed in each release
