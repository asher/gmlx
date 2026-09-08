# gmlx

[![CI build status](https://github.com/asher/gmlx/actions/workflows/test.yml/badge.svg)](https://github.com/asher/gmlx/actions/workflows/test.yml)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-blue.svg)](https://github.com/asher/gmlx/blob/main/LICENSE)

The fastest way to run GGUF models on Apple Silicon.

gmlx is a local inference platform. Chat with an open model in the terminal
or your browser, serve it over OpenAI and Anthropic compatible APIs, connect
your coding agent to it, talk to it by voice, build a local RAG stack on it,
and fine-tune it with LoRA. One command, entirely on your Mac.

It runs the community's K-quant and IQ-quant GGUF builds exactly as
published. At equal file size those are the most accurate open quant formats, and
the companion project [mlx-kquant](https://github.com/asher/mlx-kquant)
supplies the Metal kernels that run them natively on Apple's
[MLX](https://github.com/ml-explore/mlx) framework. On the same file, gmlx
benchmarks faster than llama.cpp, and the gap is widest at the long contexts
that coding agents and long sessions use. A mixture-of-experts model
bigger than RAM still runs, streaming its experts from disk.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/asher/gmlx/main/docs/assets/perf/fleet-ratio-dark.svg">
  <img src="https://raw.githubusercontent.com/asher/gmlx/main/docs/assets/perf/fleet-ratio.svg" alt="gmlx vs llama.cpp: fleet throughput speedup vs KV depth">
</picture>

Higher is faster, and depth is the number of tokens already in the context.
The per-model charts and the method are in
[benchmarks.md](https://github.com/asher/gmlx/blob/main/docs/benchmarks.md).

![gmlx chat: a 27B model answering through a running server, with live tokens per second](https://raw.githubusercontent.com/asher/gmlx/main/docs/assets/demo.gif)

A 27B model, resident in a local server, answering at 46 tokens per second.
Recorded at true speed.

## Quickstart

You need an Apple Silicon Mac. macOS 26 or newer is recommended, because the
Metal kernels then install as a prebuilt wheel; older versions build them
from source, which needs full Xcode with its Metal toolchain
([troubleshooting](https://github.com/asher/gmlx/blob/main/docs/troubleshooting.md#the-install-fails-compiling-the-metal-kernels)).

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
served id comes from the filename. `gmlx[all]` turns on every optional
feature; the core install already serves, loads vision models, embeds and runs the
menu bar, so `gmlx[chat]` omits only voice and the assistant.

## Set up with gmlx init

`gmlx init` finds your GGUF files, names them, and writes the one config
every other command reads. Run with no arguments, it opens a wizard: it scans the folders
you name, an LM Studio library or a Hugging Face cache included, lets you
rename ids and set a default, offers the on-disk prompt cache and the speech,
embedding and rerank services, and writes `~/.config/gmlx/gmlx.yaml`.

```sh
gmlx init                 # the wizard
gmlx serve                # finds the config, detaches, returns
gmlx list                 # the model ids it defines
gmlx launch pi            # connect a coding agent to the server
```

Every command now takes a model id in place of a path, and every wizard
choice has a flag, so `gmlx init --models-dir ~/models` writes the config with no
questions. The
[getting-started guide](https://github.com/asher/gmlx/blob/main/docs/getting-started.md)
is the full walkthrough, with model suggestions per machine size.

## What you get

Run and chat. `run` generates, benchmarks or prints the load plan of one
file. `chat` is a multi-turn terminal client with markdown rendering,
sessions, live sampling changes and image input. Sampling defaults come from
each model family's card, and `@intents` such as `model.gguf@creative` switch
them per call.
[cli.md](https://github.com/asher/gmlx/blob/main/docs/cli.md).

Find and download models. `validate` reads only a remote file's header to
say whether it will load and fit, and lists every quant in a repo.
`pull` downloads sharded, resumable, and registers the file in your config.
[getting-started.md](https://github.com/asher/gmlx/blob/main/docs/getting-started.md#pick-a-model-for-your-mac).

Serve an API. One port serves OpenAI Chat Completions, OpenAI Responses and
Anthropic Messages, all streaming, with tool calling, structured output,
logprobs and vision messages. Concurrent requests decode together, a new
prompt's prefill is paced so live replies keep streaming, and a prompt cache
skips repeated prefixes. The server binds loopback by default, requires a static
key for anything wider, and never contacts Hugging Face to satisfy a
request.
[api.md](https://github.com/asher/gmlx/blob/main/docs/api.md) and
[server-config.md](https://github.com/asher/gmlx/blob/main/docs/server-config.md).

Connect coding agents and chat apps. `gmlx launch pi --model qwen3.6-27b@coding`
writes the tool's native config without modifying your dotfiles and starts
the server if it is not running. Supported: pi, opencode, omp, claude-code, hermes,
goose, aichat, elia and Open WebUI. A menu bar app shows what is resident,
and `gmlx service install` keeps the server running from login.
[launch.md](https://github.com/asher/gmlx/blob/main/docs/launch.md).

Voice chat. `gmlx talk` is hands-free: a wake phrase, Whisper speech-to-text,
and replies spoken as they stream. The built-in assistant adds MCP tools and
long-term memory, and is also used by `chat --assistant` and served assistant ids.
[talk.md](https://github.com/asher/gmlx/blob/main/docs/talk.md) and
[assistant.md](https://github.com/asher/gmlx/blob/main/docs/assistant.md).

Embeddings, reranking and speech. The same server exposes `/v1/embeddings`,
`/v1/rerank`, `/v1/audio/transcriptions` and `/v1/audio/speech`, which
together make a local RAG and voice stack for clients like Open WebUI.
[rag.md](https://github.com/asher/gmlx/blob/main/docs/rag.md) and
[services.md](https://github.com/asher/gmlx/blob/main/docs/services.md).

Fine-tune with LoRA. `train` fine-tunes through the quantized matmul, so a
model too large for memory in fp16 still trains, and writes the adapter as a
GGUF that llama.cpp reads too. `--adapter` applies it at run, chat or serve,
and one base can serve several adapters at once.
[lora.md](https://github.com/asher/gmlx/blob/main/docs/lora.md).

## Performance

gmlx and llama.cpp run the same file, so the comparison is direct. On an M5
Max, gmlx prefills faster on every model in the fleet at every depth, and
with speculative decoding on both engines decodes faster at every depth. Absolute
numbers scale with the machine's memory bandwidth; measure your own with
`gmlx run model.gguf --bench 128,512,2048`.

The performance features, in [performance.md](https://github.com/asher/gmlx/blob/main/docs/performance.md):
speculative decoding, automatic on models with a draft head and available
through a companion drafter on others; the prompt cache for agent workloads;
KV-cache quantization for long contexts; and disk-streamed execution for MoE
models larger than memory, a capacity feature that makes a 200B-class model
usable on a 64 GB machine
([streaming.md](https://github.com/asher/gmlx/blob/main/docs/streaming.md)).
The choice of file also affects speed: a uniform K-quant decodes faster than a
heavily mixed one at similar quality, and K-quants have less error per byte
than MLX's native quantization
([mlx-kquant](https://github.com/asher/mlx-kquant#why)).

A one-minute video of the same server moving from one chat at full
speculative speed to four concurrent streams and back, with no break in the
live stream:

https://github.com/user-attachments/assets/de5dab84-3155-4cee-aa57-7d0b9c726ec5

## Supported architectures

Llama and Mistral; Qwen 2 through 3.8, dense and MoE, with the hybrid
attention families; Gemma 1 through 4; DeepSeek V3, R1 and V4-Flash; GLM 4
through 5.2; gpt-oss; Kimi-K3; MiniMax M2 and M3; Hunyuan, Hy3 and HY4; Muse
Glimmer; Granite; Nemotron-H; Falcon-H1; ERNIE-4.5; Phi-3; Seed-OSS; SmolLM3.
A family is listed in the generated
[coverage table](https://github.com/asher/gmlx/blob/main/docs/arch-coverage.md)
only after token-parity certification against llama.cpp at 16k context, and
the table lists the one caveat for each architecture. All 19 K-quant,
legacy and IQ codecs load, plus the MXFP4 and NVFP4 pair. Vision models load
as a GGUF paired with its projector
([vlm.md](https://github.com/asher/gmlx/blob/main/docs/vlm.md)). Adding a
family is described in
[adding-architectures.md](https://github.com/asher/gmlx/blob/main/docs/internals/adding-architectures.md).

## Python API

```python
from gmlx import load_model, generate, bench

model, config, tokenizer = load_model("model.gguf")
print(generate(model, tokenizer, "Explain entropy.", max_tokens=128))
```

`load_model` returns a ready-to-run mlx-lm model, the synthesized config and
the tokenizer. The full API, including preflight and the mlx-lm server
bridge, is in [python.md](https://github.com/asher/gmlx/blob/main/docs/python.md).

## Documentation

- [getting-started.md](https://github.com/asher/gmlx/blob/main/docs/getting-started.md): install to a served model with a connected client.
- [cli.md](https://github.com/asher/gmlx/blob/main/docs/cli.md): every verb and flag.
- [server-config.md](https://github.com/asher/gmlx/blob/main/docs/server-config.md): every key of the YAML config.
- [api.md](https://github.com/asher/gmlx/blob/main/docs/api.md): the endpoints and request features.
- [docs/README.md](https://github.com/asher/gmlx/blob/main/docs/README.md): the full index, grouped by what you want to do.

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
model implementations, [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) for the batching
server engine and the vision towers,
[mlx-whisper](https://pypi.org/project/mlx-whisper/) for speech-to-text, and
[mlx-audio](https://pypi.org/project/mlx-audio/) for text-to-speech.

## License

[Business Source License 1.1](https://github.com/asher/gmlx/blob/main/LICENSE):
source-available, not open source. You may use, modify and run gmlx for your
own purposes, including commercial work. You may not redistribute or
sublicense it, incorporate it into another product, or offer it as a hosted
service. Each released version converts to the Apache License 2.0 four years
after its release. Downloaded model weights have their own licenses.

The files listed in
[LICENSE-MIT](https://github.com/asher/gmlx/blob/main/LICENSE-MIT) are MIT
licensed and have an SPDX header saying so. Vendored third-party code is
documented in
[THIRD_PARTY_NOTICES.md](https://github.com/asher/gmlx/blob/main/THIRD_PARTY_NOTICES.md).
