# Glossary

gmlx prints these terms in its logs, help and errors, and the rest of the
docs use them in the same sense. The guides link each term here where it
first matters.

## Adapter and LoRA

A LoRA adapter is a small file of low-rank weight changes that adjusts a
base model for a task without replacing its weights. gmlx trains adapters
on the quantized GGUF and serves them with `--adapter` or the `adapter`
key, and one base in memory can serve several adapters.
[LoRA adapters](lora.md) covers training an adapter and serving it.

## APC

APC stands for automatic prefix caching, the name that mlx-vlm uses for
the [prompt cache](#prompt-cache). The server log lines about the prompt
cache start with `APC`, such as `APC tier:`.

## Arena

The arena is the wired region of GPU memory where a streamed MoE model
keeps its most used experts. Decoding reads experts from the arena and
fetches only the misses from disk. The server log prints its size as
`[stream] memory budget:`, and [Models larger than memory](streaming.md)
explains its part in streaming.

## Budget

The budget is the memory that a server lets its resident models use, set
by [`server.budget_gb`](config.md#serverbudget_gb). When a new model does
not fit in the budget, the server unloads the least recently used models
that are not pinned and have no request in progress. A
streamed model has its own budget for the [arena](#arena).

## Canvas

The canvas is the block of token positions that a diffusion model such as
DiffusionGemma writes its reply into. Each denoise step predicts every
position of the canvas at once, instead of one token after another.

## Codec

A codec is the GGUF quantization type of one tensor, such as `Q4_K` or
`IQ2_XXS`. A file mixes codecs across its tensors, and every codec in a
file needs a kernel for the file to load. `gmlx validate` lists the codecs
of a file.

## Context and depth

The context is everything in the model's input, measured in tokens, which
includes the conversation so far, pasted files and the reply in progress.
Depth is how many tokens are already in the context. The benchmark charts
plot speed against depth, because the cost of attention grows with it.

## DFlash

DFlash is a drafter that proposes a whole block of tokens in one pass, by
block diffusion, instead of one token at a time. DFlash 2 drafters exist
for Qwen3.8-27B and Muse-Glimmer-30B, and
[DFlash 2 drafters](speculative-decoding.md#dflash-2-drafters) shows how to
pair one with its model.

## Discovery

Discovery scans a folder for GGUF files, gives each model an id, pairs it
with its mmproj and drafter files, and chooses load settings. `gmlx init`,
`gmlx sync-models` and a `gmlx serve` with no config use it. Its naming
and pairing rules are in [Model discovery](config.md#model-discovery).

## Drafter

A drafter is the small predictor that [speculative
decoding](#speculative-decoding-and-mtp) uses to propose tokens. It is
either a [native head](#native-head) inside the model's own GGUF or a
separate companion GGUF.

## Every-token weights

The every-token weights are the parts of a MoE model that run on every
token, which are attention, norms, routers and shared experts. Streaming
keeps them on the GPU and reads only the routed experts from disk.

## Expert and MoE

A mixture-of-experts model is built from many small sub-networks called
experts, of which each token uses a few, so decoding costs only what the
active fraction costs. The `A3B` in `35B-A3B` means 3B active parameters.
Because most experts are idle on any token, a MoE model larger than memory
can still run when gmlx [streams](#stream) the experts from disk as they
are needed.

## Family defaults

The family defaults are the sampling settings that a model family's
publisher recommends, such as temperature and top-p, together with the
built-in intents of that family. Each request starts from them unless it
sets its own values. [Family defaults](family-defaults.md) gives the
values for each family.

## Feeder

The feeder is the code that moves expert weights for a streamed model.
During prefill it stages each layer's experts from the GGUF into the
[ring](#ring). During decoding it serves experts from the [arena](#arena)
and reads the misses.

## GDN

GDN, or gated delta net, is a recurrent layer kind that Qwen3.5, 3.6, 3.8
and some other families mix with attention layers. A recurrent layer keeps
a state of fixed size instead of a KV cache that grows with the context,
so these models use less memory at depth than their size suggests.

## GGUF

GGUF is the single-file model format that the open-model community
publishes on Hugging Face. Very large models are split into numbered
shards, which gmlx treats as one file. gmlx runs GGUF files as published,
with no conversion.

## Governor

The governor is the server's memory watchdog, and its state is under
`governor` in `GET /v1/metrics`. When memory runs short, it stops
admitting requests and shrinks the [MLX buffer
cache](memory.md#the-mlx-buffer-cache). If that is not enough, it halves
the prefill chunk, then evicts caches such as the prompt cache, and as a
last step it [sheds](#shed) the largest request.

## Hadamard fold

A Hadamard fold stores quantized weights after a fixed rotation of their
input, which spreads large values across each row before quantization. The
model rotates each activation in the same way at run time. `gmlx validate`
prints `Hadamard-folded` for such a file. How gmlx runs one is in
[Hadamard-folded GGUFs](internals/hadamard-fold.md).

## Hugging Face

Hugging Face is the site where the open-model community publishes models.
A reference of the form `hf:org/repo/file.gguf` points there, and `gmlx
pull` downloads it.

## Intent and profile

An intent is a built-in sampling preset from a model family, such as
`@coding`, which works on any model with no config. A profile is a named
set of settings that you write in the config. Both are selected in the
same way, with `model@NAME` or `--profile NAME`. Your own profiles go
under [`profiles`](config.md#profiles) in the config.

## Keep, pin and idle

A resident model can be pinned, kept or idle. A pinned model loads at
start and is never unloaded on its own, only by `POST /unload`. A kept
model is exempt from the idle timeout but can still be unloaded when the
[budget](#budget) needs room. `gmlx launch` and voice sessions keep their
model.
An idle model unloads after `ttl_s` seconds without a request.

## KV cache

The KV cache is the model's stored attention state for the context, kept
in memory beside the weights. It grows with the context, so a model whose
file barely fits leaves no room for long conversations. `--kv-bits 8` or
`--kv-quant-scheme kvarn` compresses it, and [KV cache
quantization](kv-quantization.md) compares the two.

## kvarn

The kvarn scheme quantizes the KV cache with the most accuracy for each
bit. It rotates and scales the cache in records of 128 tokens before it
rounds, so that no token or channel dominates, and it keeps the first
tokens and the newest ones at full precision. `--kv-quant-scheme kvarn`
selects it.

## MCP

MCP, the Model Context Protocol, is a standard way for a model to call
tools that separate programs provide. The built-in
[assistant](assistant.md) supports it.

## MLA

MLA, or multi-head latent attention, is the attention layout of DeepSeek
and the families derived from it. Keys and values are stored as one
compressed vector for each token instead of separate K and V rows. The KV
cache is therefore already small, and kvarn does not apply to it.
[Choosing a scheme by model](kv-quantization.md#choosing-a-scheme-by-model)
tells which MLA models take affine quantization.

## mmproj

An mmproj is a companion GGUF that holds a vision or audio encoder. Paired
with its language model, it makes a model that accepts images or audio.
[Vision and audio](vlm.md#supported-families) lists the model families
that gmlx supports.

## Native head

A native head is a small prediction layer inside a model's own GGUF that
drafts tokens for speculative decoding, as in Qwen3.5, 3.6 and 3.8. `run`
and `chat` turn it on automatically, and the `[load]` summary line shows
`drafter native-head` for a model that uses one.

## Prefill and decode

Prefill and decode are the two phases of a reply. Prefill reads the whole
prompt at once, and decoding generates the reply one token at a time. The
two run at different speeds, so gmlx reports them separately.

## Preflight

Preflight is the set of checks that run before a model loads. The loader
checks the architecture, the codec of each tensor, the shard set and the
file size before it reads any tensor data. `run` and `chat` also refuse a
context that cannot fit in memory, and the server refuses a request whose
prompt cannot fit.

## Prestage

Prestaging reads the experts that the router is predicted to select before
the router runs, so that the read overlaps with compute. It moves bytes
only and never changes which experts run.

## Prompt cache

The prompt cache is the server's store of prefilled prompts. A request
that shares its start with an earlier one, such as a system prompt or the
conversation so far, skips prefilling the shared part.
[Prompt cache](prompt-cache.md) lists its settings and the counters that
show whether reuse works.

## Quant

A quant is a compressed version of a model. The suffix of a GGUF name
gives the approximate bits per weight, such as `Q4_K_M` or `IQ2_M`. Lower
bits make a smaller file that loses more quality.

The families are the K-quants, which store weights in blocks with scales,
the IQ quants, which use codebooks for the smallest files, and legacy
types such as `Q4_0` and `Q8_0`. Newer files also use the 4-bit float
types `MXFP4` and `NVFP4`. The ternary types `STQ1_0` and `PTQ1_0` and the
2-bit `PQ2_0` store each weight in 2 bits or less.

## Resident

A resident model is loaded and ready to answer. Several models can stay
resident at once within the server's [budget](#budget), and `gmlx ps`
lists them.

## Ring

The ring is the set of GPU-visible slots through which the
[feeder](#feeder) stages the experts of a streamed model during prefill,
one layer at a time.

## Runfile

A runfile is the small file that a background server writes under
`~/.cache/gmlx/`, with its process id, address, command line and config.
`gmlx status`, `stop`, `restart` and `logs` find the server through it.

## Shed

To shed a request is to stop it early to free memory. When the
[governor](#governor) has tried everything else, it ends the largest
request with an error of type `server_overloaded_shed`, so that the other
requests keep running.

## Speculative decoding and MTP

A [drafter](#drafter) proposes several tokens, and the model checks them
in one pass. By default the output stays the same, with fewer full
passes. MTP, for multi-token prediction, is the name that the flags and
logs of gmlx give to speculative decoding with any drafter, a
[native head](#native-head) or a companion GGUF.
[Speculative decoding](speculative-decoding.md) describes both.

## Stream

A model larger than memory has two stream placements. With `stream:
experts`, the [every-token weights](#every-token-weights) and the KV cache
stay on the GPU, and the routed experts are read from disk. With `stream:
cpu`, the whole model runs on the CPU from the page cache.

## Structured read

A structured read is how [`/v1/systemone`](decisions.md) answers its
questions, in one denoise step by default. The [canvas](#canvas) holds an answer
template with a random token at each answer position. The model's
prediction at that position, limited to the question's labels, is the
answer. A sample is one such read with its own random tokens, and a
request averages one or more samples.

## Thinking model

A thinking model is trained to reason before it answers, and it writes
that reasoning between markers such as `<think>` and `</think>`. The chat
client shows the reasoning under a label by default.

## Token

A token is the unit that models read and write, about three quarters of an
English word on average. Speeds are given in tokens per second.

## Wired memory

Wired memory is memory that the GPU has pinned, so that macOS cannot page
it out. Weights and the arena are wired, which is why the server budgets
them against the [working set](#working-set) instead of the total RAM.

## Working set

The working set is the share of RAM that macOS lets the GPU use, which
depends on the machine. Memory errors and the `cannot fit` refusal compare
what a model needs with the working set.
