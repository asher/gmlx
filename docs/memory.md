# Memory and the KV cache

A model needs memory for its weights, about the size of its GGUF file, and
for its context's KV cache, which grows with every token. `gmlx validate`
tells you whether a file fits this Mac before you download it:

```sh
gmlx validate hf:unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K.gguf
gmlx run model.gguf --kv-bits 8    # about half the KV cache
```

A MoE model larger than memory can stream its experts from disk, as
[Models larger than memory](streaming.md) describes.

## The weights and the KV cache

The KV cache of a standard dense model takes this much for each token:

```text
bytes per token = 2 (K and V) x layers x kv_heads x head_dim x 2 (bf16)
```

An 8B model with 32 layers, 8 KV heads and a head dimension of 128 uses
128 KB for each token, or 4 GB for a 32K-token context. A 32B dense model
uses 8 GB at 32K. For long-context agent work, the KV cache can be as large
as the weights.

Many families use much less. Sliding-window layers, as in Gemma, stop
growing at the window. Hybrid models, such as Qwen3.5 to Qwen3.8, Granite 4
and Nemotron-H, keep a full KV cache only on their few attention layers, so
Qwen3.6-27B uses 2.1 GB at 32K. [MLA](glossary.md#mla) models, such as the
DeepSeek family, store a compressed cache.

On a running server, the `memory` and `capacity` sections of
`GET /v1/metrics` show the live numbers, and `POST /v1/estimate` tells you
whether a request fits before you send it, as
[Capacity and metrics](capacity.md) describes.

## Settings that limit memory

These steps reduce memory, the cheapest first:

- Quantize the KV cache. `--kv-bits 8` about halves it with almost no
  quality cost, and `--kv-quant-scheme kvarn` keeps that quality at 6 bits
  in less memory. [KV cache quantization](kv-quantization.md) compares the
  two. On the server, the [load keys](config.md#model-loading) set them.
- Limit the context. On `run` and `chat`, `--max-kv-size N` keeps a
  rolling window of the newest N tokens. With `--kv-bits`, gmlx picks kvarn
  for the window, since affine cannot quantize it. On the server,
  [`max_kv_size`](config.md#loadmax_kv_size) caps a request's context
  instead.
- Shrink the prefill chunk. A smaller `--prefill-step-size` lowers the
  memory peak of a long prompt and slows prefill.

Speculative decoding, which some families turn on by default, drops some of
these settings with a warning, as
[Settings that speculation drops](speculative-decoding.md#settings-that-speculation-drops)
lists.

## Several models on one server

The server keeps several models loaded at once, up to
[`server.budget_gb`](config.md#serverbudget_gb), which is 0.8 times the GPU
working set by default. When a new model needs room, the server unloads
the least recently used model that is not pinned.

| State | Set by | When it unloads |
|-------|--------|-----------------|
| Pinned | `pin: true`, `--pin` | Only on `POST /unload` |
| Kept | `POST /v1/keep`, `gmlx launch --model`, a talk session | Only when the budget needs the room |
| Idle or preloaded | Any request, `server.defaults.preload` | After `ttl_s` seconds without a request, or when the budget needs the room |

A model never unloads during a generation. `POST /v1/keep` with
`{"keep": false}` releases a kept model, and `POST /unload` unloads one at
once. A streamed model counts differently, as
[Residency of a streamed model](streaming.md#residency-of-a-streamed-model)
describes.

Two model ids that point to the same GGUF share one loaded copy. A second
copy loads, and takes its own share of the budget, when the ids differ in
how the model is built:

- the `load` or `cache` keys, or `chat_template`
- `mmproj`, `draft_gguf`, `speculative` or `speculative_width_cap`
- `stream` and the streaming keys

A profile that sets `load`, `cache` or `chat_template` also loads its own
copy when a request uses it. Sampling, `system` and `ttl_s` never cause a
second copy. Ids that differ only in `adapter` share one copy, as
[LoRA adapters](lora.md#serving-one-base-with-many-adapters) describes.
To offer one model several ways without doubling its memory, vary only
those, for example with [profiles](config.md#profiles) of sampling
settings. When you do need two
build settings, check that both copies fit `server.budget_gb`.

## The GPU memory limit

macOS lets the GPU wire only a share of RAM, which depends on the machine.
If one dense model and its cache sit right at that limit on a Mac with a
lot of RAM, `sudo sysctl iogpu.wired_limit_mb=<MB>` raises the limit until
the next restart, at your own risk. Leave several GB for macOS.

## The MLX buffer cache

MLX keeps freed GPU buffers in a pool for reuse. This buffer cache is
separate from the KV cache and the prompt cache. At deep contexts it can
grow to tens of GB, and without a limit it can use up the Mac's free memory
and freeze it instead of failing with an error.

The server limits the pool by default, to between 4 and 12 GiB, and logs
the limit on a `[serve] MLX cache limit:` line.
[`server.cache_limit_gb`](config.md#servercache_limit_gb) or
[`GMLX_CACHE_LIMIT_GB`](env-vars.md#runtime) sets it. Set a fixed limit for
benchmarks, so that the runs are comparable.
