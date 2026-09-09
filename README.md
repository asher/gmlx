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

It runs the community's K-quant and IQ-quant GGUF builds exactly as published,
which at equal file size are the most accurate open quant formats. The
companion project [mlx-kquant](https://github.com/asher/mlx-kquant) supplies
the Metal kernels that run them natively on Apple's
[MLX](https://github.com/ml-explore/mlx) framework, and on the same file gmlx
benchmarks faster than llama.cpp, with the widest gap at the long contexts
that coding agents and long sessions use. A mixture-of-experts model bigger
than RAM still runs, by streaming its experts from disk. If you are coming
from llama.cpp, Ollama or LM Studio,
[migrating.md](https://github.com/asher/gmlx/blob/main/docs/migrating.md)
says what carries over.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/asher/gmlx/main/docs/assets/perf/fleet-ratio-dark.svg">
  <img src="https://raw.githubusercontent.com/asher/gmlx/main/docs/assets/perf/fleet-ratio.svg" alt="gmlx vs llama.cpp: fleet throughput speedup vs KV depth">
</picture>

Higher is faster, and depth is the number of tokens already in the context.
The per-model charts and the method behind them are in
[benchmarks.md](https://github.com/asher/gmlx/blob/main/docs/benchmarks.md).

![gmlx chat with a 27B model answering through a running server, with live
tokens per
second](https://raw.githubusercontent.com/asher/gmlx/main/docs/assets/demo.gif)

The recording runs at true speed: a 27B model resident in a local server,
answering at 46 tokens per second.

## Quickstart

gmlx needs an Apple Silicon Mac running Python 3.11 or newer, which uv will
fetch for you. Intel Macs and Linux are not supported. On macOS 26.2 or newer
the Metal kernels install as a prebuilt wheel, while older versions build them
from source and need full Xcode with its Metal toolchain, as
[troubleshooting.md](https://github.com/asher/gmlx/blob/main/docs/troubleshooting.md#the-install-fails-compiling-the-metal-kernels)
describes. A model needs memory for roughly its file size plus the
conversation's KV cache, and
[getting-started.md](https://github.com/asher/gmlx/blob/main/docs/getting-started.md#pick-a-model-for-your-mac)
suggests models for each machine size.

```sh
uv tool install "gmlx[all]"     # or: pip install "gmlx[all]" into a venv you manage
brew install ffmpeg             # voice and non-wav audio only

mkdir ~/gmlx && cd ~/gmlx
gmlx pull hf:unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf --to .
gmlx run  Qwen3-0.6B-Q4_K_M.gguf --prompt "Explain entropy in one paragraph."
gmlx chat Qwen3-0.6B-Q4_K_M.gguf
gmlx serve Qwen3-0.6B-Q4_K_M.gguf --port 8080

curl localhost:8080/v1/chat/completions -d \
  '{"model": "qwen3-0.6b", "messages": [{"role": "user", "content": "hi"}]}'
gmlx stop
```

Any local `.gguf` runs, chats or serves this way with no other setup. The
served id is the filename without its quant tag, so `qwen3-0.6b` in this
example, whereas ids assigned by `gmlx init` keep the tag. `--to .` downloads
into the current directory, and once a config exists `pull` writes to your
model directory and registers the file there instead. `gmlx[all]` turns on
every optional feature, but the core install already serves, loads vision
models, embeds and runs the menu bar, so `gmlx[chat]` omits only voice and the
assistant. Upgrade with `uv tool upgrade gmlx`. Removing gmlx is described in
[troubleshooting.md](https://github.com/asher/gmlx/blob/main/docs/troubleshooting.md#where-files-are-on-disk).

## Set up with gmlx init

`gmlx init` finds your GGUF files, names them and writes the config that the
other commands read. Run with no arguments it opens a wizard, which scans the
folders you name, including an LM Studio library or a Hugging Face cache, lets
you rename ids and set a default, offers the on-disk prompt cache and the
speech, embedding and rerank services, and then writes
`~/.config/gmlx/gmlx.yaml`.

```sh
gmlx init                 # the wizard
gmlx serve                # finds the config, detaches, returns
gmlx list                 # the model ids it defines
gmlx launch pi            # connect a coding agent to the server
```

After that, every command takes a model id in place of a path. Each wizard
choice also has a flag, so `gmlx init --models-dir ~/models` writes the config
without asking anything. The [getting-started
guide](https://github.com/asher/gmlx/blob/main/docs/getting-started.md) is the
full walkthrough.

## What you get

### Run and chat

`run` generates, benchmarks or prints the load plan of one file, and `chat`
is a multi-turn terminal client with markdown rendering, sessions, live
sampling changes and image input. Sampling defaults come from each model
family's card, and `@intents` such as `model.gguf@creative` switch them for a
single call. The flags are listed in
[cli.md](https://github.com/asher/gmlx/blob/main/docs/cli.md).

### Find and download models

`validate` reads only a remote file's header to tell you whether it will load
and fit, and it can list the quants in a repo so you can pick one before
downloading anything. `pull` then downloads sharded files, resumes an
interrupted download and registers the file in your config.
[getting-started.md](https://github.com/asher/gmlx/blob/main/docs/getting-started.md#pick-a-model-for-your-mac)
suggests models for each machine size.

### Serve an API

One port serves OpenAI Chat Completions, OpenAI Responses and Anthropic
Messages, all streaming, with tool calling, structured output, logprobs and
vision messages. Concurrent requests decode together, a new prompt's prefill
is paced so that live replies keep streaming, and a prompt cache skips
repeated prefixes. The server binds loopback by default, requires a static key
for anything wider and never contacts Hugging Face to satisfy a request. The
endpoints are in [api.md](https://github.com/asher/gmlx/blob/main/docs/api.md)
and the config keys in
[server-config.md](https://github.com/asher/gmlx/blob/main/docs/server-config.md).

### Connect coding agents and chat apps

`gmlx launch pi --model qwen3.6-27b@coding` writes the tool's native config
without touching your dotfiles, starting the server first if it is not
running. pi, opencode, omp, claude-code, hermes, goose, aichat, elia and Open
WebUI are supported. A menu bar app shows what is resident, and `gmlx service
install` keeps the server running from login. Details are in
[launch.md](https://github.com/asher/gmlx/blob/main/docs/launch.md).

### Voice chat and the assistant

`gmlx talk` is hands-free: a wake phrase opens the mic, Whisper transcribes,
and the reply is spoken as it streams. The built-in assistant adds MCP tools
and long-term memory, and the same assistant is behind `chat --assistant` and
the served assistant ids. See
[talk.md](https://github.com/asher/gmlx/blob/main/docs/talk.md) and
[assistant.md](https://github.com/asher/gmlx/blob/main/docs/assistant.md).

### Embeddings, reranking and speech

The server also exposes `/v1/embeddings`, `/v1/rerank`,
`/v1/audio/transcriptions` and `/v1/audio/speech`, which together give
clients like Open WebUI a local RAG and voice stack.
[rag.md](https://github.com/asher/gmlx/blob/main/docs/rag.md) and
[services.md](https://github.com/asher/gmlx/blob/main/docs/services.md)
describe them.

### Fine-tune with LoRA

`train` fine-tunes through the quantized matmul, so a model too large for
memory in fp16 still trains, and it writes the adapter as a GGUF that
llama.cpp reads too. `--adapter` applies it at run, chat or serve, and one
base can serve several adapters at once. The guide is
[lora.md](https://github.com/asher/gmlx/blob/main/docs/lora.md).

## Performance

Because gmlx and llama.cpp run the same file, the comparison is direct. On an
M5 Max, gmlx prefills faster on every model in the fleet at every depth, and
with speculative decoding on both engines it decodes faster at every depth as
well. DeepSeek-V4-Flash is compared against ds4-server, since llama.cpp cannot
run it. Absolute numbers scale with the machine's memory bandwidth, so measure
your own with `gmlx run model.gguf --bench 128,512,2048`.

[performance.md](https://github.com/asher/gmlx/blob/main/docs/performance.md)
covers the performance features. Speculative decoding is automatic on models
with a draft head and available through a companion drafter on others, the
prompt cache skips prefill for the repeated prefixes of agent workloads, and
KV-cache quantization shrinks long contexts. Disk-streamed execution,
described in
[streaming.md](https://github.com/asher/gmlx/blob/main/docs/streaming.md),
runs MoE models larger than memory and makes a 200B-class model usable on a
64 GB machine. The file you choose matters as well: a uniform K-quant decodes
faster than a heavily mixed one at similar quality, and K-quants carry less
error per byte than MLX's native quantization, as
[mlx-kquant](https://github.com/asher/mlx-kquant#why) explains.

A one-minute video shows the same server moving from one chat at full
speculative speed to four concurrent streams and back, with no break in the
live stream:

https://github.com/user-attachments/assets/de5dab84-3155-4cee-aa57-7d0b9c726ec5

## Supported architectures

- Llama, Mistral, Phi-3, SmolLM3, Seed-OSS and ERNIE-4.5
- Qwen 2 through 3.8, dense and MoE, with the hybrid attention families
- Gemma 1 through 4
- DeepSeek V3, R1 and V4-Flash
- GLM 4 through 5.2, Kimi-K3, MiniMax M2 and M3, and gpt-oss
- Hunyuan, Hy3, HY4 and Muse Glimmer
- Granite, Nemotron-H and Falcon-H1

A family appears in the generated [coverage
table](https://github.com/asher/gmlx/blob/main/docs/arch-coverage.md) only
after token-parity certification against llama.cpp at 16k context, and the
table names the one caveat for each architecture. All 19 K-quant, legacy and
IQ codecs load, plus the MXFP4 and NVFP4 pair. Vision models load as a GGUF
paired with its projector, as
[vlm.md](https://github.com/asher/gmlx/blob/main/docs/vlm.md) describes, and
[adding-architectures.md](https://github.com/asher/gmlx/blob/main/docs/internals/adding-architectures.md)
explains what adding a family involves.

## Python API

```python
from gmlx import load_model, generate

model, config, tokenizer = load_model("model.gguf")
print(generate(model, tokenizer, "Explain entropy.", max_tokens=128))
```

`load_model` returns a ready-to-run mlx-lm model, the synthesized config and
the tokenizer. The full API, including preflight and the mlx-lm server bridge,
is in [python.md](https://github.com/asher/gmlx/blob/main/docs/python.md).

## Documentation

- [getting-started.md](https://github.com/asher/gmlx/blob/main/docs/getting-started.md):
  install to a served model with a connected client.
- [cli.md](https://github.com/asher/gmlx/blob/main/docs/cli.md): every verb
  and flag.
- [server-config.md](https://github.com/asher/gmlx/blob/main/docs/server-config.md):
  every key of the YAML config.
- [api.md](https://github.com/asher/gmlx/blob/main/docs/api.md): the endpoints
  and request features.
- [troubleshooting.md](https://github.com/asher/gmlx/blob/main/docs/troubleshooting.md):
  `gmlx doctor` first, then the common failures, where files are on disk and
  how to remove gmlx.
- [migrating.md](https://github.com/asher/gmlx/blob/main/docs/migrating.md):
  what transfers from llama.cpp, Ollama and LM Studio.
- [glossary.md](https://github.com/asher/gmlx/blob/main/docs/glossary.md): the
  terms the docs use, from GGUF and quant to prefill and depth.
- [docs/README.md](https://github.com/asher/gmlx/blob/main/docs/README.md):
  the full index, grouped by what you want to do.

## Contributing

Pull requests are welcome. Dev setup and the rules are in
[CONTRIBUTING.md](https://github.com/asher/gmlx/blob/main/CONTRIBUTING.md),
the test tiers in
[testing.md](https://github.com/asher/gmlx/blob/main/docs/internals/testing.md),
and the runtime's design in
[docs/internals](https://github.com/asher/gmlx/blob/main/docs/internals/README.md).

## Acknowledgments

gmlx builds on [llama.cpp and ggml](https://github.com/ggml-org/llama.cpp)
for the GGUF format and the K-quant reference implementations,
[MLX and mlx-lm](https://github.com/ml-explore/mlx-lm) for the runtime and
model implementations, [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) for the
batching server engine and the vision towers,
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
[THIRD_PARTY_NOTICES.md](https://github.com/asher/gmlx/blob/main/THIRD_PARTY_NOTICES.md).
