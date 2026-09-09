# Glossary

The terms these docs use, in plain words, alphabetically. Each guide links
here on the first use of a term.

Arena. The wired region of GPU memory a streamed MoE model keeps its most
used experts in. Decode reads from the arena and fetches only the misses
from disk. The server log prints its size as `[stream] memory budget:`.
See [streaming.md](streaming.md).

Barge-in. Speaking over the assistant while it is talking. In voice chat
the reply stops and the new utterance is taken.

Context and depth. Everything currently in the model's input, measured in
tokens, which includes the conversation so far, pasted files and the reply
in progress. Depth is how many tokens are already there, and it is the x
axis of the benchmark charts because attention cost grows with it.

Codec. The GGUF quantization type of one tensor, such as `Q4_K` or
`IQ2_XXS`. A file mixes codecs across tensors, and each codec in a file
needs a kernel for the file to load.

Drafter. The small predictor speculative decoding uses to propose tokens.
It is either a head inside the model's own GGUF or a separate companion
GGUF.

Endpointing. Deciding that an utterance has ended, from trailing silence.
The voice chat settings for it are in [talk.md](talk.md).

Every-token layers. The parts of a MoE model that run on every token, which
are attention, norms, routers and shared experts. Streaming keeps them on
the GPU and streams only the routed experts.

Expert and MoE. A mixture-of-experts model is built from many small
sub-networks, of which each token activates a few, so decode costs only
what the active fraction costs. The `A3B` in `35B-A3B` means 3B active
parameters. Because most experts are idle on any token, a MoE bigger than
RAM can still run by streaming experts from disk.

Feeder. The two paths that move expert weights for a streamed model. One, the
prefill feeder, stages each layer's experts directly from the GGUF into
GPU-visible slots. The other, the decode feeder, serves decode from the arena.

GGUF. The single-file model format the open-model ecosystem publishes on
Hugging Face, where a file is a ready-to-run model. Very large models are
split into numbered shards, which gmlx treats as a single file. gmlx runs
GGUFs exactly as published, with no conversion.

Governor. The runtime memory watchdog in the server. It watches the
kernel's free pages and shrinks its registered caches, the arena first,
before the machine swaps. Its state shows at `/v1/metrics`.

Hugging Face. The site the open-model ecosystem publishes on. References of
the form `hf:org/repo/file.gguf` in these docs point there. `gmlx pull`
downloads them.

Intent and profile. An intent is a built-in named sampling preset from a
model family's card, such as `@coding`, addressable on any model with no
config. A profile is a named bundle of settings you write in the server
config. Both are selected the same way, `model@NAME` or `--profile NAME`.

Keep, pin and idle. The three states of a resident model. A pinned model
is never unloaded, while a kept model is exempt from the idle timeout but
unloads under memory pressure, which is what `gmlx launch` and voice
sessions ask for. An idle model unloads after `ttl_s` seconds without a
request.

KV cache. The model's stored attention state for the context, kept in RAM
beside the weights. It grows with context length, which is why a model whose
file barely fits leaves no memory for long conversations. `--kv-bits 8` or
`--kv-quant-scheme kvarn` compresses it.

K-quant and IQ. The two families of GGUF quantization. K-quants such as
`Q4_K_M` group weights with per-block scales. IQ quants such as `IQ2_M` use
learned codebooks for the smallest files. Both are more accurate per byte
than a plain affine quantization.

MCP. The Model Context Protocol, a standard way for a model to call tools
provided by separate programs. gmlx's built-in assistant supports it.

mmproj. A companion GGUF holding a vision or audio tower. Paired with its
language model GGUF it makes a model that accepts image or audio input. See
[vlm.md](vlm.md).

Preflight. The checks the loader runs before reading any tensor bytes,
covering the architecture gate and the codec of each tensor. A file that
fails preflight is refused with the reason.

Prefill and decode. The two phases of answering. Prefill reads the prompt
all at once, while decode generates the reply one token at a time, and the
two have different speeds and are reported separately.

Prestage. Reading experts the router is predicted to select before the
router runs, which overlaps the read with compute. It moves bytes only and
never changes routing.

Prompt cache. The server's store of prefilled prefixes. A request that shares
a prefix with an earlier one, such as a system prompt or the conversation so
far, skips prefilling the shared part. mlx-vlm calls it APC. Its keys are in
[server-config.md](server-config.md#cache-keys).

Quant. A compressed build of a model. The suffix on a GGUF name says roughly
the number of bits per weight. Q4 files are smaller and slightly lossier,
while Q6 or Q8 are bigger and closer to the original.

Resident. A model that is loaded and ready to answer. Several stay resident
at once within the server's budget, and `gmlx ps` lists them.

Ring. The GPU-visible slots the prefill feeder stages expert layers through
on a streamed model, one layer at a time.

Speculative decoding and MTP. A drafter proposes several tokens and the full
model verifies them in one step, giving the same output with fewer full
passes. MTP, multi-token prediction, is the form where the draft head is
included inside the model's own GGUF. gmlx turns it on automatically.

Streaming placements. The two ways to run a model bigger than memory. `stream:
experts` keeps the every-token layers and the KV cache on the GPU and streams
the routed experts from disk. With `stream: cpu` the whole model runs on the
CPU from the page cache.

Thinking model. A model trained to reason before answering, streaming that
text inside markers such as `<think>`. The chat client shows it under a
label by default.

Token. The unit models read and write, about three quarters of an English
word on average. Speeds are quoted in tokens per second.

Wired memory. Memory the GPU has pinned so the kernel cannot page it out.
Weights and the arena are wired, and the server budgets them against the
machine's working set instead of its total RAM.
