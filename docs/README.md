# gmlx documentation

gmlx runs GGUF models on Apple Silicon. A single command chats, serves,
connects an agent or talks by voice. This index groups the pages by what
you want to do. The [project README](../README.md) is the overview.

Start with [getting-started.md](getting-started.md), which takes you from
install to a served model with a client connected. Look up flags in
[cli.md](cli.md), config keys in [server-config.md](server-config.md) and
the HTTP API in [api.md](api.md). The [glossary](glossary.md) defines the
terms the rest of the docs use.

## Learn

| Page | Contains |
|------|----------|
| [getting-started.md](getting-started.md) | install, a first model, model picks per machine size, the server, a client |
| [glossary.md](glossary.md) | the terms these docs use, from GGUF and quant to arena and governor |
| [migrating.md](migrating.md) | what transfers from llama.cpp, Ollama and LM Studio and what maps to what |

## Do

| Task | Page |
|------|------|
| give Claude Code or another coding agent a local model and keep it resident | [launch.md](launch.md) |
| use the chat REPL's commands, sessions and themes | [chat.md](chat.md) |
| control the server from the menu bar | [menubar.md](menubar.md) |
| talk to a model by voice | [talk.md](talk.md) |
| give a model tools and long-term memory | [assistant.md](assistant.md) |
| build a local RAG pipeline with embeddings and rerank | [rag.md](rag.md) |
| run a vision or audio model | [vlm.md](vlm.md) |
| fine-tune with LoRA and serve several adapters on one base | [lora.md](lora.md) |
| run a 200B MoE on a 64 GB Mac and pick a lossy setting | [streaming.md](streaming.md) |
| make it faster and know what each setting costs | [performance.md](performance.md) |
| fix something that broke | `gmlx doctor`, then [troubleshooting.md](troubleshooting.md) |

## Reference

| Page | Contains |
|------|----------|
| [cli.md](cli.md) | each verb and flag, with defaults and exit codes |
| [server-config.md](server-config.md) | each key of the YAML config, precedence, profiles, residency |
| [api.md](api.md) | endpoints, addressing a model, tools, structured output, logprobs, vision, limits |
| [services.md](services.md) | the speech-to-text, text-to-speech, embeddings and rerank services |
| [env-vars.md](env-vars.md) | the environment variables a user can set |
| [python.md](python.md) | the Python API: load, generate, bench, preflight |
| [arch-coverage.md](arch-coverage.md) | the generated table of supported architectures and their caveats |
| [benchmarks.md](benchmarks.md) | the generated scorecard against llama.cpp, with method |

## Internals

| Page | Contains |
|------|----------|
| [internals/adding-architectures.md](internals/adding-architectures.md) | what adding a model family involves and the tests that certify it |
| [internals/README.md](internals/README.md) | the serving architecture, speculative batching, the prompt cache, testing, upstream upgrades, debug switches |

[CONTRIBUTING.md](../CONTRIBUTING.md) has the development setup and
[CHANGELOG.md](../CHANGELOG.md) records what was released when.
