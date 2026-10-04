<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/asher/gmlx/main/docs/assets/wordmark-dark.svg">
    <img src="https://raw.githubusercontent.com/asher/gmlx/main/docs/assets/wordmark.svg" alt="gmlx" height="56">
  </picture>
</h1>

[![CI build
status](https://github.com/asher/gmlx/actions/workflows/test.yml/badge.svg)](https://github.com/asher/gmlx/actions/workflows/test.yml)
[![License: BSL
1.1](https://img.shields.io/badge/License-BSL%201.1-blue.svg)](https://github.com/asher/gmlx/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-asher.github.io%2Fgmlx-blue.svg)](https://asher.github.io/gmlx/)

The fastest way to run GGUF models on Apple Silicon.

gmlx is a local inference platform. Chat with an open model in the terminal
or your browser, serve it over OpenAI and Anthropic compatible APIs, connect
a coding agent to it, talk to it by voice, and fine-tune it with LoRA.

It runs the community's K-quant and IQ-quant GGUF files exactly as
published, on Metal kernels from
[mlx-kquant](https://github.com/asher/mlx-kquant) for Apple's
[MLX](https://github.com/ml-explore/mlx). On the same file it prefills
faster than llama.cpp, and with speculative decoding on both engines it
decodes faster too. The gap is widest at the long contexts that coding
agents use. A mixture-of-experts model bigger than RAM still runs, by
streaming its experts from disk.

## Quickstart

gmlx needs an Apple Silicon Mac with macOS 26.2 or newer. Install it with
Homebrew:

```sh
brew install asher/gmlx/gmlx
```

The example model below, Qwen3.8-27B UD-Q6_K, is a 20.5 GB download and
suits a Mac with 64 GB of memory. A model needs about its file size in
memory, plus room for the conversation. On a smaller Mac, pick a model from
[Choosing a model](https://asher.github.io/gmlx/quickstart.html#choosing-a-model)
first.

```sh
gmlx init --models-dir ~/models
gmlx pull hf:unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K.gguf
gmlx run  qwen3.8-27b-ud-q6 --prompt "Explain entropy in one paragraph."
gmlx chat qwen3.8-27b-ud-q6
gmlx serve

curl localhost:8080/v1/chat/completions -d \
  '{"model": "qwen3.8-27b-ud-q6", "messages": [{"role": "user", "content": "hi"}]}'
gmlx launch pi --model qwen3.8-27b-ud-q6
```

`gmlx init` writes the configuration file, and `pull` downloads the model
and adds it under the id `qwen3.8-27b-ud-q6`. `serve` starts the server in
the background on port 8080, and `launch` connects the pi coding agent to
it. Install pi first with `npm install -g @earendil-works/pi-coding-agent`,
or add `--container` to run it in an Apple container with pi installed.

Without Homebrew, install with uv: `uv tool install "gmlx[all]"`, plus
`brew install ffmpeg` for voice. To use gmlx from your own Python
environment, `pip install "gmlx[all]"`. Both commands install every optional
feature.
[Installation](https://asher.github.io/gmlx/installation.html) covers each
route, upgrading and
[removing gmlx](https://asher.github.io/gmlx/installation.html#removing-gmlx).

Any GGUF file also runs, chats and serves by its path, with no
configuration file:

```sh
gmlx chat Qwen3-4B-Q4_K_M.gguf
```

![gmlx chat with a 27B model answering through a running server, with live
tokens per
second](https://raw.githubusercontent.com/asher/gmlx/main/docs/assets/demo.gif)

The recording runs at true speed, with a 27B model in a local server.

## What you get

- A terminal chat with markdown, sessions, images and each family's
  recommended sampling. See [Chat](https://asher.github.io/gmlx/chat.html).
- Remote checks before you download: `gmlx validate` tells you whether a file
  will load and fit. See the [CLI reference](https://asher.github.io/gmlx/cli.html).
- One server for OpenAI Chat Completions, OpenAI Responses and Anthropic
  Messages, with tools, structured output and vision. See the
  [HTTP API](https://asher.github.io/gmlx/api.html).
- Coding agents and chat apps set up in one command, on the Mac or in an
  Apple container that sees only your project. See
  [Agents and chat apps](https://asher.github.io/gmlx/launch.html).
- Voice chat with a wake phrase, and an assistant with tools and memory. See
  [Voice chat](https://asher.github.io/gmlx/talk.html) and
  [Assistant](https://asher.github.io/gmlx/assistant.html).
- Embeddings, rerank, speech-to-text and text-to-speech on the same server,
  for a local RAG and voice stack. See
  [Speech, embeddings and rerank](https://asher.github.io/gmlx/services.html).
- A decision API at `/v1/systemone` that returns a probability for each
  answer to yes-or-no, choice and score questions about a text. See
  [Structured decisions](https://asher.github.io/gmlx/decisions.html).
- LoRA training on the quantized model, and distillation from a larger model.
  See [LoRA adapters](https://asher.github.io/gmlx/lora.html).
- A menu bar app that shows the loaded models and controls the server. See
  [Menu bar app](https://asher.github.io/gmlx/menubar.html).

## Performance

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/asher/gmlx/main/docs/assets/perf/fleet-ratio-dark.svg">
  <img src="https://raw.githubusercontent.com/asher/gmlx/main/docs/assets/perf/fleet-ratio.svg" alt="gmlx against llama.cpp: throughput speedup by KV depth">
</picture>

Higher is faster. Depth is the number of tokens already in the context. See
[Benchmarks](https://asher.github.io/gmlx/benchmarks.html) for each model.

On an M5 Max, gmlx prefills faster than llama.cpp on every benchmarked model
at every depth. With speculative decoding on both engines, it decodes faster
too. Measure your own Mac with `gmlx run model.gguf --bench 128,512,2048`.

Speculative decoding, the prompt cache and KV cache quantization are
described in [Performance tuning](https://asher.github.io/gmlx/performance.html).
A [one-minute video](https://github.com/user-attachments/assets/de5dab84-3155-4cee-aa57-7d0b9c726ec5)
shows one server going from a single chat to four concurrent streams and
back.

## Supported architectures

- Llama, Mistral, Phi-3, SmolLM3, Seed-OSS and ERNIE-4.5
- Qwen 2 through 3.8, dense and MoE, with the hybrid attention families
- Gemma 1 through 4, except the 3n variant, whose GGUFs are broken upstream
- DeepSeek V3, R1, V4-Flash and V4.1-Flash
- GLM 4 through 5.3, Kimi-K3, MiniMax M2 and M3, and gpt-oss
- Hunyuan, Hy3, HY4 and Muse Glimmer
- Granite, Nemotron-H and Falcon-H1

Each family's output is checked against llama.cpp at 16k context. The
[coverage table](https://asher.github.io/gmlx/arch-coverage.html) lists the
caveats. All K-quant, IQ-quant and legacy types load, plus MXFP4, NVFP4 and
the ternary types, and vision models load with their projector.

## Python API

```python
from gmlx import load_model, generate

model, config, tokenizer = load_model("model.gguf")
print(generate(model, tokenizer, "Explain entropy.", max_tokens=128))
```

See the [Python API](https://asher.github.io/gmlx/python.html).

## Documentation

The [documentation site](https://asher.github.io/gmlx/) covers the latest release, with navigation
and search. Good places to start:

- [Installation](https://asher.github.io/gmlx/installation.html): Homebrew, uv and pip, and upgrading.
- [Quickstart](https://asher.github.io/gmlx/quickstart.html): A first model, and choosing one for your Mac.
- [Configuration](https://asher.github.io/gmlx/config.html): Every key of `gmlx.yaml`.
- [Agents and chat apps](https://asher.github.io/gmlx/launch.html): Connecting coding agents and chat apps.
- [HTTP API](https://asher.github.io/gmlx/api.html): The OpenAI and Anthropic endpoints.
- [CLI reference](https://asher.github.io/gmlx/cli.html): Every command and flag.
- [Troubleshooting](https://asher.github.io/gmlx/troubleshooting.html): Common errors and their fixes.
- [All documentation](https://asher.github.io/gmlx/): Every guide and reference page.

## Contributing

Pull requests are welcome. Dev setup and the rules are in the
[contributing guide](https://github.com/asher/gmlx/blob/main/CONTRIBUTING.md),
the test tiers in [Testing](https://asher.github.io/gmlx/internals/testing.html),
and the runtime's design in
[Internals](https://asher.github.io/gmlx/internals/index.html).

## Acknowledgments

gmlx builds on [llama.cpp and ggml](https://github.com/ggml-org/llama.cpp)
for the GGUF format and the K-quant reference implementations,
[MLX and mlx-lm](https://github.com/ml-explore/mlx-lm) for the runtime and
model implementations, and [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) for
the server app, generation step loop and vision towers. Speech uses
[mlx-whisper](https://pypi.org/project/mlx-whisper/) for speech-to-text and
[mlx-audio](https://pypi.org/project/mlx-audio/) for text-to-speech.

## License

gmlx is released under the [Business Source License
1.1](https://github.com/asher/gmlx/blob/main/LICENSE), which is
source-available but not open source. You may use, modify and run gmlx for
your own purposes, including commercial work, and you may redistribute
unmodified copies free of charge. You may not sell gmlx or a derivative of
it, incorporate either into a commercial product or service, or offer
either to others as a hosted service. Each released version converts to
the Apache License 2.0 four years after its release, and downloaded model
weights have their own licenses.

The files listed in
[LICENSE-MIT](https://github.com/asher/gmlx/blob/main/LICENSE-MIT) are MIT
licensed and have an SPDX header saying so. Vendored third-party code is
documented in
the [third-party notices](https://github.com/asher/gmlx/blob/main/THIRD_PARTY_NOTICES.md).
