# gmlx

[![CI build
status](https://github.com/asher/gmlx/actions/workflows/test.yml/badge.svg)](https://github.com/asher/gmlx/actions/workflows/test.yml)
[![License: BSL
1.1](https://img.shields.io/badge/License-BSL%201.1-blue.svg)](https://github.com/asher/gmlx/blob/main/LICENSE)

The fastest way to run GGUF models on Apple Silicon.

gmlx is a local inference platform. You can chat with an open model in the
terminal or your browser, serve it over OpenAI and Anthropic compatible APIs,
connect a coding agent to it, talk to it by voice, build a local RAG stack on
it and fine-tune it with LoRA.

It runs the community's K-quant and IQ-quant GGUF builds exactly as
published. Those formats are the most accurate open quants at a given file
size, and the companion project
[mlx-kquant](https://github.com/asher/mlx-kquant) supplies the Metal kernels
that run them natively on Apple's [MLX](https://github.com/ml-explore/mlx)
framework.

On the same file gmlx benchmarks faster than llama.cpp, and the gap is widest
at the long contexts that coding agents and long sessions use. A
mixture-of-experts model bigger than RAM still runs, by streaming its experts
from disk.

If you are coming from llama.cpp, Ollama or LM Studio,
[Migrating from other tools](https://github.com/asher/gmlx/blob/main/docs/migrating.md) says what carries over.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/asher/gmlx/main/docs/assets/perf/fleet-ratio-dark.svg">
  <img src="https://raw.githubusercontent.com/asher/gmlx/main/docs/assets/perf/fleet-ratio.svg" alt="gmlx against llama.cpp: throughput speedup by KV depth">
</picture>

Higher is faster, and depth is the number of tokens already in the context.
The per-model charts and the method behind them are in
[Benchmarks](https://github.com/asher/gmlx/blob/main/docs/benchmarks.md).

![gmlx chat with a 27B model answering through a running server, with live
tokens per
second](https://raw.githubusercontent.com/asher/gmlx/main/docs/assets/demo.gif)

The recording runs at true speed, with a 27B model resident in a local server
answering at 46 tokens per second.

## Quickstart

gmlx needs an Apple Silicon Mac with macOS 26.2 or newer. Intel Macs and
Linux are not supported. Install with Homebrew, then create a configuration
file and download a model:

```sh
brew install asher/gmlx/gmlx

gmlx init --models-dir ~/models
gmlx pull hf:unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K.gguf
gmlx run  qwen3.8-27b-ud-q6 --prompt "Explain entropy in one paragraph."
gmlx chat qwen3.8-27b-ud-q6
gmlx serve

curl localhost:8080/v1/chat/completions -d \
  '{"model": "qwen3.8-27b-ud-q6", "messages": [{"role": "user", "content": "hi"}]}'
gmlx launch pi --model qwen3.8-27b-ud-q6
```

`gmlx init` writes the configuration file. `pull` downloads into the folder
that the file names and adds the model under the id `qwen3.8-27b-ud-q6`,
which every command then accepts in place of a path. `serve` starts the
server in the background on port 8080, and `launch` connects the pi coding
agent to it. Run `gmlx init` with no flags for a wizard that scans the
folders where you already keep models.

To choose the optional features yourself, install with
`uv tool install "gmlx[all]"` and add `brew install ffmpeg` for voice.
[Installation](https://github.com/asher/gmlx/blob/main/docs/installation.md) covers both routes.

The model above is 20.5 GB. A model needs memory for about its file size
plus the KV cache of the conversation, and the Quickstart
[suggests models](https://github.com/asher/gmlx/blob/main/docs/quickstart.md#choosing-a-model)
for each memory size. Upgrade with `brew upgrade gmlx`, or `uv tool upgrade
gmlx` for a uv install. To remove gmlx, follow
[Removing gmlx](https://github.com/asher/gmlx/blob/main/docs/installation.md#removing-gmlx).

### A GGUF with no setup

Any GGUF file also runs, chats and serves by its path, with no
configuration file:

```sh
gmlx run   Qwen3-4B-Q4_K_M.gguf --prompt "Explain entropy in one paragraph."
gmlx chat  Qwen3-4B-Q4_K_M.gguf
gmlx serve Qwen3-4B-Q4_K_M.gguf
```

A server started with one file names the model after the file without its
quant, here `qwen3-4b`, and uses it for a request that names no model.

## What you get

### Run and chat

`run` generates, benchmarks or prints the load plan of one file. `chat` is a
multi-turn terminal client with markdown rendering, sessions, live sampling
changes and image input. Both start from each model family's
[recommended sampling](https://github.com/asher/gmlx/blob/main/docs/family-defaults.md),
and every flag is listed under its verb in the
[CLI reference](https://github.com/asher/gmlx/blob/main/docs/cli.md).

### Find and download models

`validate` reads only a remote file's header to tell you whether it will load
and fit, and it can list the quants in a repo so you can pick one before
downloading anything. `pull` then fetches sharded files, resumes an
interrupted download and registers the result in your config.

### Serve an API

One port serves OpenAI Chat Completions, OpenAI Responses and Anthropic
Messages, all streaming, with tool calling, structured output, logprobs and
vision messages. Concurrent requests decode together, a new prompt's prefill
is paced so that live replies keep streaming, and a prompt cache skips
repeated prefixes. The server binds loopback by default, requires a static
key for anything wider and never contacts Hugging Face to satisfy a request.
The endpoints are documented in the [HTTP API](https://github.com/asher/gmlx/blob/main/docs/api.md), and
the file that configures them in
[Configuration](https://github.com/asher/gmlx/blob/main/docs/config.md).

A served DiffusionGemma model also answers the Jev decision API at
`/v1/systemone`. A request asks a fixed set of yes or no, choice and score
questions about a state, and each answer comes back as a probability for
every option.
[Structured decisions](https://github.com/asher/gmlx/blob/main/docs/decisions.md)
shows how to write the questions and act on the answers.

### Connect coding agents and chat apps

`gmlx launch pi --model qwen3.8-27b-ud-q6@coding` writes the tool's
configuration so that it uses the server, starts the server first if it is
not running, and then runs the tool. It works for the common coding agents, two terminal chat clients
and two browser apps, Open WebUI and DeepSeek Harness, each listed with its
quirks in [Agents and chat apps](https://github.com/asher/gmlx/blob/main/docs/launch.md).
A menu bar app shows what is resident, and `gmlx service install` keeps the
server running from login.

### Voice chat and the assistant

With `gmlx talk` a wake phrase opens the mic, Whisper transcribes, and the
reply is spoken as it streams, as [Voice chat](https://github.com/asher/gmlx/blob/main/docs/talk.md)
describes. The
built-in [assistant](https://github.com/asher/gmlx/blob/main/docs/assistant.md)
adds MCP tools and long-term memory to a voice session, to
`chat --assistant`, and to assistant ids that the server exposes as models.

### Embeddings, reranking and speech

The server also exposes `/v1/embeddings`, `/v1/rerank`,
`/v1/audio/transcriptions` and `/v1/audio/speech`, which together give a
client like Open WebUI a local RAG and voice stack. The services are
described in
[Speech, embeddings and rerank](https://github.com/asher/gmlx/blob/main/docs/services.md) and
the RAG setup in [RAG pipelines](https://github.com/asher/gmlx/blob/main/docs/rag.md).

### Fine-tune with LoRA

`train` fine-tunes through the quantized matmul, so a model too large for
memory in fp16 still trains, and it writes the adapter as a GGUF that
llama.cpp reads too. `--adapter` applies it at run, chat or serve, and one
base can serve several adapters at once, which
[LoRA adapters](https://github.com/asher/gmlx/blob/main/docs/lora.md) walks through
end to end.

`distill` teaches a small GGUF what a larger one knows, a document or a
behavior. It trains an adapter on the larger model's outputs without
running the two at once, and
[Distillation](https://github.com/asher/gmlx/blob/main/docs/distill.md)
gives the commands for each case.

## Performance

Because gmlx and llama.cpp run the same file, the comparison is direct. On an
M5 Max, gmlx prefills faster on every model in the fleet at every depth, and
with speculative decoding on both engines it decodes faster at every depth as
well. Absolute numbers scale with the machine's memory bandwidth, so measure
your own with `gmlx run model.gguf --bench 128,512,2048`.

[Performance tuning](https://github.com/asher/gmlx/blob/main/docs/performance.md)
covers the performance features. Speculative decoding uses a model's own
draft head, or a companion drafter on models without one, and `run` and
`chat` turn it on by themselves. The prompt cache skips prefill for the
repeated prefixes of agent workloads, and KV-cache quantization shrinks long
contexts. Disk-streamed execution, described in
[Models larger than memory](https://github.com/asher/gmlx/blob/main/docs/streaming.md),
runs MoE models larger than memory and makes a 200B-class model usable on a
64 GB machine.

The file you choose matters as well. A uniform K-quant decodes faster than a
heavily mixed one at similar quality, and K-quants carry less error per byte
than MLX's native quantization, as
[mlx-kquant](https://github.com/asher/mlx-kquant#why) explains.

A one-minute video shows the same server moving from one chat at full
speculative speed to four concurrent streams and back, with no break in the
live stream:

https://github.com/user-attachments/assets/de5dab84-3155-4cee-aa57-7d0b9c726ec5

## Supported architectures

- Llama, Mistral, Phi-3, SmolLM3, Seed-OSS and ERNIE-4.5
- Qwen 2 through 3.8, dense and MoE, with the hybrid attention families
- Gemma 1 through 4, except the 3n variant, whose GGUFs are broken upstream
- DeepSeek V3, R1, V4-Flash and V4.1-Flash
- GLM 4 through 5.3, Kimi-K3, MiniMax M2 and M3, and gpt-oss
- Hunyuan, Hy3, HY4 and Muse Glimmer
- Granite, Nemotron-H and Falcon-H1

A family appears in the generated [coverage
table](https://github.com/asher/gmlx/blob/main/docs/arch-coverage.md) only
after token-parity certification against llama.cpp at 16k context, and the
table names the caveats where an architecture has any. All 19 K-quant,
legacy and IQ codecs load, plus the MXFP4 and NVFP4 pair and the ternary
STQ1_0, PTQ1_0 and PQ2_0 types. Vision models load
as a GGUF paired with its projector, as
[Vision and audio](https://github.com/asher/gmlx/blob/main/docs/vlm.md) describes, and
[Adding a GGUF architecture](https://github.com/asher/gmlx/blob/main/docs/internals/adding-architectures.md)
explains what adding a family involves.

## Python API

```python
from gmlx import load_model, generate

model, config, tokenizer = load_model("model.gguf")
print(generate(model, tokenizer, "Explain entropy.", max_tokens=128))
```

`load_model` returns a ready-to-run mlx-lm model, the synthesized config and
the tokenizer. The full API, including preflight and the mlx-lm server bridge,
is in the [Python API](https://github.com/asher/gmlx/blob/main/docs/python.md) reference.

## Documentation

The [documentation site](https://asher.github.io/gmlx/) has every page
below for the latest release, with navigation and search.

### Getting started

-
  [Installation](https://github.com/asher/gmlx/blob/main/docs/installation.md):
  Homebrew, uv and pip, the optional features, upgrading and removal.
- [Quickstart](https://github.com/asher/gmlx/blob/main/docs/quickstart.md): A
  first model, the server, a request and a connected client.
- [Migrating from other
  tools](https://github.com/asher/gmlx/blob/main/docs/migrating.md): What
  carries over from llama.cpp, Ollama and LM Studio.

### Serving

- [Configuration](https://github.com/asher/gmlx/blob/main/docs/config.md):
  `gmlx.yaml`, its models, its profiles and how a request gets its settings.
- [Agents and chat
  apps](https://github.com/asher/gmlx/blob/main/docs/launch.md): Claude Code,
  other coding agents and Open WebUI, set up by `gmlx launch`.
- [Menu bar app](https://github.com/asher/gmlx/blob/main/docs/menubar.md):
  Server status and controls in the macOS menu bar.
- [Speech, embeddings and
  rerank](https://github.com/asher/gmlx/blob/main/docs/services.md): The
  services a server can host beside chat models.
- [RAG pipelines](https://github.com/asher/gmlx/blob/main/docs/rag.md):
  Retrieval with the embeddings and rerank services.
- [Structured
  decisions](https://github.com/asher/gmlx/blob/main/docs/decisions.md): A
  probability for each answer to a fixed set of questions.

### Clients

- [Chat](https://github.com/asher/gmlx/blob/main/docs/chat.md): The terminal
  chat client, its commands, sessions and themes.
- [Voice chat](https://github.com/asher/gmlx/blob/main/docs/talk.md): Talking
  to a model with `gmlx talk`.
- [Assistant](https://github.com/asher/gmlx/blob/main/docs/assistant.md):
  Tools and long-term memory for chat, voice and served models.

### Models

- [Supported
  architectures](https://github.com/asher/gmlx/blob/main/docs/arch-coverage.md):
  The GGUF architectures gmlx loads, with their caveats.
- [Vision and audio](https://github.com/asher/gmlx/blob/main/docs/vlm.md):
  Multimodal models and their `mmproj` files.
- [Models larger than
  memory](https://github.com/asher/gmlx/blob/main/docs/streaming.md):
  Mixture-of-experts models that stream their experts from disk.
- [LoRA adapters](https://github.com/asher/gmlx/blob/main/docs/lora.md):
  Training an adapter and serving several on one base model.
- [Distillation](https://github.com/asher/gmlx/blob/main/docs/distill.md):
  Teaching a small model a document or a larger model's behavior.

### Performance

- [Performance
  tuning](https://github.com/asher/gmlx/blob/main/docs/performance.md): What
  makes a model fast, measuring, and choosing a quant.
- [Speculative
  decoding](https://github.com/asher/gmlx/blob/main/docs/speculative-decoding.md):
  Faster decoding with a drafter, with the same output.
- [Prompt
  cache](https://github.com/asher/gmlx/blob/main/docs/prompt-cache.md):
  Skipping prefill for prompts the server has seen.
- [Concurrent
  requests](https://github.com/asher/gmlx/blob/main/docs/concurrency.md):
  Batching, admission pacing and shared prompts.
- [Memory and the KV
  cache](https://github.com/asher/gmlx/blob/main/docs/memory.md): How much
  memory a model and its context take.
- [KV cache
  quantization](https://github.com/asher/gmlx/blob/main/docs/kv-quantization.md):
  Storing the context in fewer bits.
- [Benchmarks](https://github.com/asher/gmlx/blob/main/docs/benchmarks.md):
  gmlx against llama.cpp on the same files, with the method.

### Help

-
  [Troubleshooting](https://github.com/asher/gmlx/blob/main/docs/troubleshooting.md):
  `gmlx doctor`, common failures, and where gmlx keeps its files.
- [Glossary](https://github.com/asher/gmlx/blob/main/docs/glossary.md): The
  terms these pages use.

### Reference

- [CLI reference](https://github.com/asher/gmlx/blob/main/docs/cli.md): Every
  command and flag.
- [Configuration](https://github.com/asher/gmlx/blob/main/docs/config.md):
  Every key of `gmlx.yaml`.
- [Family
  defaults](https://github.com/asher/gmlx/blob/main/docs/family-defaults.md):
  The sampling defaults and intents of each model family.
- [HTTP API](https://github.com/asher/gmlx/blob/main/docs/api.md): The
  endpoints and request features.
- [Environment
  variables](https://github.com/asher/gmlx/blob/main/docs/env-vars.md): The
  variables a user can set.
- [Python API](https://github.com/asher/gmlx/blob/main/docs/python.md): Using
  gmlx from Python.

## Contributing

Pull requests are welcome. Dev setup and the rules are in
the [contributing guide](https://github.com/asher/gmlx/blob/main/CONTRIBUTING.md),
the test tiers in
[Testing](https://github.com/asher/gmlx/blob/main/docs/internals/testing.md),
and the runtime's design in
[Internals](https://github.com/asher/gmlx/blob/main/docs/internals/README.md).

## Acknowledgments

gmlx builds on [llama.cpp and ggml](https://github.com/ggml-org/llama.cpp)
for the GGUF format and the K-quant reference implementations,
[MLX and mlx-lm](https://github.com/ml-explore/mlx-lm) for the runtime and
model implementations, [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) for the
server app, generation step loop and vision towers,
[mlx-whisper](https://pypi.org/project/mlx-whisper/) for speech-to-text, and
[mlx-audio](https://pypi.org/project/mlx-audio/) for text-to-speech.

## License

gmlx is released under the [Business Source License
1.1](https://github.com/asher/gmlx/blob/main/LICENSE), which is
source-available but not open source. You may use, modify and run gmlx for
your own purposes, including commercial work. You may not redistribute or
sublicense it, incorporate it into another product, or offer it as a hosted
service. Each released version converts to the Apache License 2.0 four years
after its release, and downloaded model weights have their own licenses.

The files listed in
[LICENSE-MIT](https://github.com/asher/gmlx/blob/main/LICENSE-MIT) are MIT
licensed and have an SPDX header saying so. Vendored third-party code is
documented in
the [third-party notices](https://github.com/asher/gmlx/blob/main/THIRD_PARTY_NOTICES.md).
