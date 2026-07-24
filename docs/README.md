# gmlx documentation

gmlx is a local inference platform for Apple Silicon; the
[project README](../README.md) is the overview. This page routes you to the
right document for the task at hand.

## If you want to...

| Task | Read |
|------|------|
| Install and chat with a first model | [getting-started.md](getting-started.md) |
| Switch over from llama.cpp, Ollama, or LM Studio | [migrating.md](migrating.md) |
| Chat in your browser instead of a terminal | [getting-started.md](getting-started.md#chat-in-your-browser) |
| Know whether a model fits your Mac | [getting-started.md](getting-started.md#pick-a-model-for-your-mac) |
| Use Claude Code, opencode, goose, or Open WebUI on a local model | [launch.md](launch.md) |
| Stand up the server and write its config | [getting-started.md](getting-started.md#set-up-the-server), then [server-config.md](server-config.md) |
| Talk to a model by voice | [talk.md](talk.md) |
| Give a model tools and long-term memory | [assistant.md](assistant.md) |
| Build a local RAG pipeline (embeddings + rerank) | [rag.md](rag.md) |
| Run a vision or audio-input model | [vlm.md](vlm.md) |
| Fine-tune with LoRA on a quantized base | [lora.md](lora.md) |
| Make it faster | [performance.md](performance.md) |
| Run a MoE model bigger than RAM | [streaming.md](streaming.md) |
| Fix something that broke | run `gmlx doctor`, then [troubleshooting.md](troubleshooting.md) |
| Look up a flag or config key | [cli.md](cli.md), [server-config.md](server-config.md) |
| Load and generate from your own Python | [python.md](python.md) |
| Check whether an architecture is supported | [arch-coverage.md](arch-coverage.md) |
| Add support for a new architecture | [adding-architectures.md](adding-architectures.md) |

## Start here

[getting-started.md](getting-started.md) takes you from nothing to a running model:
install, download a GGUF, chat, stand up the server, connect a client. Each step
ends in something usable, so you can stop wherever your needs are met. It also
carries the model-picking table by machine size and the KV-cache arithmetic for
judging whether a model fits.

## Guides

Task-oriented walkthroughs. Each one assumes the install from getting-started and
nothing else.

[launch.md](launch.md) covers `gmlx launch`: one command that points an external
tool at your server, starting the server first if it is down. Per-client sections
for Claude Code, opencode, pi, omp, hermes, goose, aichat, elia, and Open WebUI
spell out exactly what gets written where and why your dotfiles are never touched,
plus the menu-bar app and launch troubleshooting.

[talk.md](talk.md) is the voice loop: wake word, speech-to-text, spoken replies
streamed sentence by sentence. Two worked examples, one taking you from an empty
config to a spoken reply and one wiring the built-in assistant for voice, plus
the full configuration reference.

[assistant.md](assistant.md) covers the built-in tool-loop assistant shared by
voice, the chat REPL, and the server: the `assistant:` block, MCP tool servers,
long-term memory, `gmlx chat --assistant`, and served assistant ids with their
routing contract and security model.

[rag.md](rag.md) stands up a fully local retrieval stack on the server's
`/v1/embeddings` and `/v1/rerank` endpoints, shows the exact request and response
shapes, and wires Open WebUI's document RAG to it.

[vlm.md](vlm.md) covers vision and audio input: pairing a model GGUF with its
mmproj file, image and audio messages over the API, and which families support
what.

[lora.md](lora.md) trains a LoRA adapter directly on a quantized GGUF base with
`gmlx train`, saves it as a GGUF adapter, and applies it live at load; adapters
interoperate with llama.cpp.

[performance.md](performance.md) explains what actually determines speed on Apple
Silicon and what each lever buys: quant choice (uniform K-quant files decode
markedly faster than heavily mixed ones), speculative decoding, the prompt cache,
and KV-cache quantization and sizing. Includes how to run your own benchmarks.

[streaming.md](streaming.md) covers MoE models bigger than RAM: the two
placements, the feeder paths and lookahead prestage, GPU keep-warm, and the
lossy levers with their measured decision procedure.

[benchmarks.md](benchmarks.md) is the generated fleet scorecard behind the
performance claims: gmlx vs llama.cpp on the same GGUF, per-model charts and
tables at KV depths from 512 to 200k+, with methodology and exact weight
provenance.

[migrating.md](migrating.md) maps llama.cpp flags, Ollama conventions, and LM
Studio libraries onto their gmlx equivalents, and says plainly what does not
carry over.

[troubleshooting.md](troubleshooting.md) lists the failures new setups actually
hit, each with the diagnostic and the fix.

## Reference

[cli.md](cli.md) documents every verb and flag of the `gmlx` command, one section
per verb, with the semantics that do not fit in `--help`.

[server-config.md](server-config.md) is the full server reference: the YAML config
file and every key in it, all HTTP endpoints, API capabilities (tools, structured
output, logprobs, vision), model residency, and the speech and embedding services.

[python.md](python.md) documents the stable Python surface: `load_model`,
`generate`, `bench`, preflight and its errors, and the mlx-lm server bridge.

[arch-coverage.md](arch-coverage.md) is the generated architecture matrix: every
mapped GGUF architecture, its loadability, and validation status. Regenerated by
script; do not edit it by hand.

## Internals and contributing

[serving-architecture.md](serving-architecture.md) explains how the pieces
compose: loader, engine, batching, and the HTTP layers.

[adding-architectures.md](adding-architectures.md) is what adding a model
family involves and the acceptance gate an architecture clears to be listed
as supported.

[testing.md](testing.md) describes the test tiers, the end-to-end harnesses, and
the manual voice-loop pass.

[CONTRIBUTING.md](../CONTRIBUTING.md) has the development setup and expectations
for pull requests; [CHANGELOG.md](../CHANGELOG.md) records what shipped when.
