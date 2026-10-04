# Glossary

This page explains the terms that gmlx prints in its logs, help and errors,
and that these docs use. Each entry links to the page that covers it.

## Adapter and LoRA

A small file of weight changes that adjusts a base model for a task. One
base in memory can serve several. See [LoRA adapters](lora.md).

## APC

Automatic prefix caching, the [prompt cache](#prompt-cache). Its log lines
start with `APC`, such as `APC tier:`.

## Arena

The GPU memory where a streamed MoE model keeps its most used experts. The
log prints its size as `[stream] memory budget:`. See
[Models larger than memory](streaming.md).

## Budget

The memory that a server lets its models use, set by
[`server.budget_gb`](config.md#serverbudget_gb). To fit a new model, the
server unloads the least recently used models that are not pinned or busy.

## Canvas

The block of positions that a diffusion model such as DiffusionGemma writes
its reply into, all at once. See [Structured read](#structured-read).

## Codec

The quantization type of one tensor, such as `Q4_K` or `IQ2_XXS`. A file can
mix codecs, and `gmlx validate` lists them.

## Context and depth

The context is everything in the model's input, in tokens: the conversation,
pasted files and the reply so far. Depth is how many tokens it holds now.

## Custom agent

A program of your own, defined under `launch.agents`, that `gmlx launch`
runs in a container. See [Custom agents](launch-agents.md).

## DFlash

A drafter that proposes a whole block of tokens in one pass. See
[DFlash 2 drafters](speculative-decoding.md#dflash-2-drafters).

## Discovery

The scan that finds GGUF files in a folder, names each model and pairs it
with its mmproj and drafter. `gmlx init` and `gmlx sync-models` use it. See
[Model discovery](config.md#model-discovery).

## Drafter

The small predictor that [speculative decoding](#speculative-decoding-and-mtp)
uses: a [native head](#native-head) or a separate companion GGUF.

## Expert and MoE

A mixture-of-experts (MoE) model is built from many small sub-networks called
experts, and each token uses only a few. `35B-A3B` means 3B active
parameters. A MoE model larger than memory can [stream](#stream) its experts.

## Family defaults

The sampling settings that a model family's publisher recommends, with its
intents. Each request starts from them. See
[Family defaults](family-defaults.md).

## Feeder

The code that moves expert weights for a [streamed](#stream) model. The
prefill feeder stages each layer's experts from the GGUF during prefill.
The decode feeder serves experts from the [arena](#arena) and reads the
misses from disk, and the log prints `[stream] decode feeder arena hit rate:`.
`--prefill-feeder` and `--decode-feeder` turn them off with `--no-`. See
[Models larger than memory](streaming.md).

## GDN

Gated delta net, a recurrent layer in Qwen3.5, 3.6, 3.8 and some other
families. Its state has a fixed size, so these models use less memory at
depth.

## GGUF

The single-file model format that the open-model community publishes on
Hugging Face. gmlx runs it as published. A reference such as
`hf:org/repo/file.gguf` points at a file there, for `gmlx pull`.

## Governor

The server's memory watchdog, under `governor` in `GET /v1/metrics`. As
memory runs short, its band moves from `green` to `yellow`, `orange` and
`red`, and it takes these steps in order:

1. In `yellow`, it stops admitting requests and shrinks the
   [MLX buffer cache](memory.md#the-mlx-buffer-cache).
2. If that is not enough, it halves the prefill chunk.
3. In `orange`, it evicts caches such as the prompt cache.
4. As a last step, it [sheds](#shed) the largest request.

See [Memory](memory.md) and [Capacity and metrics](capacity.md).

## Hadamard fold

Weights stored after a fixed rotation, so they quantize with less error.
`gmlx validate` prints `Hadamard-folded` for such a file.

## Intent and profile

An intent is a family's built-in sampling preset, such as `@coding`. A
profile is a set of settings you write under
[`profiles`](config.md#profiles). Select either with `model@NAME`.

## Keep, pin and idle

A pinned model unloads only by `POST /unload`. A kept model skips the idle
timeout but can still make room in the [budget](#budget). An idle model
unloads after `ttl_s` seconds without a request.

## KV cache

The model's stored attention state for the context, in memory beside the
weights. It grows with the context. See
[KV cache quantization](kv-quantization.md) to shrink it.

## kvarn

The KV cache quantization scheme with the best accuracy for each bit.
gmlx picks it for recurrent and sliding-window models, and
`--kv-quant-scheme kvarn` names it for any model. See
[KV cache quantization](kv-quantization.md).

## Letter readout

How [`/v1/systemone`](decisions.md) answers on models other than
DiffusionGemma: each option gets a letter, and the model's probability for
each letter is the answer. The diagnostics show `"readout": "letters"`.

## MCP

The Model Context Protocol, a standard way for a model to call tools that
other programs provide. The [assistant](assistant.md) supports it.

## MLA

Multi-head latent attention, used by DeepSeek and related families. Its KV
cache is already compressed, so kvarn does not apply. See
[The scheme gmlx picks](kv-quantization.md#the-scheme-gmlx-picks).

## mmproj

A companion GGUF with a vision or audio encoder, which lets its model
accept images or audio. See [Vision and audio](vlm.md#supported-families).

## Native head

A small prediction layer inside a model's own GGUF that drafts tokens for
speculative decoding. The `[load]` line shows `drafter native-head`.

## Prefill and decode

The two phases of a reply. Prefill reads the whole prompt at once, and
decode writes the reply one token at a time. gmlx reports both speeds.

## Preflight

The checks before a model loads: architecture, codecs, shards, and whether
the model and its context fit in memory.

## Prestage

Reading the experts that the router is predicted to select before the
router runs, so the read overlaps with compute. It never changes which
experts run. `--moe-prestage` chooses how it picks the experts. See
[The lossless settings](streaming.md#the-lossless-settings).

## Private home

The home folder that [container mode](launch-container.md) gives a client
in each project, which keeps its settings, logins and history. See
[The private home](container-access.md#the-private-home).

## Project

In container mode, the folder you launch from. Each project gets its own
[private home](#private-home) and session. See
[One session per project](container-sessions.md#one-session-per-project).

## Prompt cache

The server's store of prefilled prompts. A request that starts like an
earlier one skips prefilling the shared part. See
[Prompt cache](prompt-cache.md).

## Quant

A compressed version of a model. The GGUF name suffix, such as `Q4_K_M`,
`IQ2_M`, `Q8_0` or `MXFP4`, gives the type. Fewer bits per weight make a
smaller file that loses more quality.

## Resident

Loaded and ready to answer. Several models can be resident within the
[budget](#budget), and `gmlx ps` lists them.

## Shed

To end a request early to free memory. The [governor](#governor) ends the
largest one with the error type `server_overloaded_shed`.

## Speculative decoding and MTP

A [drafter](#drafter) proposes several tokens, and the model checks them in
one pass, with the same output. gmlx's flags and logs call this MTP. See
[Speculative decoding](speculative-decoding.md).

## Stream

How a model larger than memory runs. `stream: experts` reads the routed
experts from disk, and `stream: cpu` runs the whole model on the CPU. See
[Models larger than memory](streaming.md).

## Structured read

How [`/v1/systemone`](decisions.md) answers on DiffusionGemma. The model
predicts every answer position of a template on its [canvas](#canvas) at
once, limited to each question's labels.

## Thinking model

A model that reasons before it answers, between markers such as `<think>`
and `</think>`. The chat client shows the reasoning under a label.

## Token

The unit that models read and write, about three quarters of an English
word. Speeds are in tokens per second.

## Wired memory

Memory that the GPU has pinned, so macOS cannot page it out. Weights and the
[arena](#arena) are wired.

## Working set

The share of RAM that macOS lets the GPU use. When a model and its context
need more than the working set, gmlx refuses to load it with `cannot fit`.
