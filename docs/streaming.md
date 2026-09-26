# Models larger than memory

gmlx can run a mixture-of-experts model whose file is larger than the memory
of your Mac, by reading its experts from disk as they are needed. Starting
takes one flag, a header check tells whether a model fits, and further
settings make streaming faster, some with a cost in quality.

- [What to expect](#what-to-expect)
- [Quick start](#quick-start)
- [Choosing a placement](#choosing-a-placement)
- [Features with streaming](#features-with-streaming)
- [How big a model can this machine stream](#how-big-a-model-can-this-machine-stream)
- [The lossless settings](#the-lossless-settings)
- [The lossy settings](#the-lossy-settings)
- [Serving a streamed model](#serving-a-streamed-model)
- [Residency of a streamed model](#residency-of-a-streamed-model)

## What to expect

A [MoE](glossary.md#expert-and-moe) model uses only a few of its experts
for each token, so one token reads a small part of the file. gmlx keeps
the parts that every token reads in memory and reads the routed experts
from disk. This lets a model of about 200 billion parameters run on a Mac
with 64 GB.

The speed of decoding then depends on the SSD and the CPU rather than the
GPU, and a few tokens per second is normal. A model that fits in memory
runs several times faster, so use streaming to run a model that does not
fit, not to gain speed.

Keep the GGUF on the internal SSD, because every expert that is not in
memory is read at the random read latency of the drive. An external drive
works, and decoding slows in proportion to its latency.

## Quick start

`gmlx validate <file>` says whether this Mac can stream a file, before you
download it. Then run the model with `--stream-experts`, and for a file in
several parts, name the first part:

```sh
gmlx run GLM-5.2-UD-IQ3_XXS-00001-of-00006.gguf --stream-experts
```

In the [configuration file](config.md), the
[`stream`](config.md#modelsstream) key of the model entry sets the
placement:

```yaml
models:
  glm:
    path: GLM-5.2-UD-IQ3_XXS-00001-of-00006.gguf
    stream: experts
    overrides: {load: {kv_bits: 8}}
```

Generation starts within seconds whatever the size of the file, because
the load maps the file instead of reading it. The decode arena is the
wired memory that holds the most used experts, and it fills from disk
during decoding. At the end, the decode feeder prints how often the
experts were found in the arena. `run` and `chat` print this line with
`-v`, and the server always logs it. For long contexts, a quantized KV
cache with `--kv-bits 8` is the usual addition.

`--stream-experts` streams only a model larger than 90% of the GPU
working set. A smaller model loads into memory as usual, and a dense
model prints that the flag has no effect.

## Choosing a placement

| Placement | What stays on the GPU | What streams | Use it when |
|-----------|-----------------------|--------------|-------------|
| `--stream-experts`, or `stream: experts` | Attention, routers, shared experts and the KV cache stay on the GPU. | The routed experts stream from the arena and the disk. | Use it in most cases, including long contexts, chat and a server with other models. |
| `--stream-cpu`, or `stream: cpu` | Nothing stays on the GPU, and the whole model runs on the CPU from the page cache. | Everything that does not fit in the wired budget streams. | Use it for a model on its own, where one device for everything is simpler. |

`--stream-cpu` streams the experts whatever the size of the model. The
decode feeder is off under it, so the settings that need the feeder do not
apply. Because it moves the default device of the process to the CPU, run
a `stream: cpu` model on a server of its own.

## Features with streaming

| Combination | Result |
|-------------|--------|
| `--stream-experts` with `--mmproj` | `run` and `serve` stream the language model and keep the vision encoder on the GPU. Chat refuses it. |
| `--stream-cpu` with `--mmproj` | The command refuses it, because the CPU placement would move the vision encoder too. |
| `--stream-experts` with speculative decoding | It works on the command line with an explicit `--speculative`. Automatic speculation stays off under streaming. |
| `stream` on a server entry with `speculative` | The server refuses it at load, because it loads the drafter after the placement. |
| A lossy setting with speculative decoding | Automatic speculation turns off, and the setting applies. Chat refuses the combination with an explicit `--speculative`. |

## How big a model can this machine stream

The every-token weights set the limit, not the size of the file.

A MoE GGUF holds two kinds of tensors. The routed experts are read a few
at a time for each token and stream from disk, so their size sets the
speed of decoding but not the fit. The every-token weights are read by
every token and stay in memory. They are attention, shared experts, dense
layers, routers, norms, embeddings and the output head.

The memory [governor](glossary.md#governor) keeps tracked memory under a
ceiling, which is the GPU working set that macOS recommends, less 5%. It
also keeps a reserve of 8 GB or 10% of RAM, whichever is larger, below
physical RAM. Four things share the ceiling, in this order:

1. The every-token weights.
2. The KV room. It holds the KV cache for 32768 tokens, or the trained
   context when that is shorter. It adds room for prefill and an admission
   reserve, each the larger of 2 GB and 5% of the working set.
3. The prefill ring, which is two copies of the expert stacks of the
   largest layer. When the ring does not fit, prefill reads through the
   page cache instead, and the load says so. Decoding is not affected.
4. The decode arena, which takes what is left after a host floor. The
   floor is 5% of RAM, at least 4 GiB, plus 2.5 GiB for the page cache,
   and it keeps the rest of the Mac out of swap. A larger arena finds more
   experts in memory and decodes faster. Below 1 GiB, the decode feeder
   does not start, and decoding reads through the page cache.

A Mac therefore streams a model when the every-token weights and the KV
room fit under the ceiling. Everything else changes only the speed.

### The ceiling by machine size

The default working set is two thirds of RAM below 36 GB and three
quarters from 36 GB up. Sizes are in decimal GB, and the KV room floor is
the prefill room and the admission reserve, before any KV cache.

| RAM | Working set | Ceiling | KV room floor | Left for weights and KV cache |
|---|---|---|---|---|
| 16 | 11.5 | 9.2 | 4.0 | 5.2 |
| 24 | 17.2 | 16.3 | 4.0 | 12.3 |
| 32 | 22.9 | 21.8 | 4.0 | 17.8 |
| 36 | 29.0 | 27.5 | 4.0 | 23.5 |
| 48 | 38.7 | 36.7 | 4.0 | 32.7 |
| 64 | 51.5 | 49.0 | 5.2 | 43.8 |
| 96 | 77.3 | 73.4 | 7.7 | 65.7 |
| 128 | 103.1 | 97.9 | 10.3 | 87.6 |
| 192 | 154.6 | 146.9 | 15.5 | 131.4 |
| 256 | 206.2 | 195.9 | 20.6 | 175.2 |
| 512 | 412.3 | 391.7 | 41.2 | 350.5 |

`sudo sysctl iogpu.wired_limit_mb=<MB>` raises the working set, and the
reserve below physical RAM still applies.

### A worked example

Kimi-K3 UD-Q2_K_XL is an 861 GB file, and its every-token weights are
62.2 GB:

| Group | GB |
|---|---|
| Attention, which is [MLA](glossary.md#mla) on this model | 31.8 |
| Shared experts, 2 per layer | 12.9 |
| Dense FFN and routers | 8.2 |
| Recurrent layers | 6.9 |
| Embeddings and output head | 2.4 |

Its routed experts are 799 GB, with 896 experts in each of 92 layers and
16 read for each token. On a 128 GB Mac with the default working set, the
ceiling is 97.9 GB and the KV room is 11.7 GB. That leaves 24 GB, where the
22.7 GB ring fits and the host floor takes the rest, so there is no arena.
The model streams, and decoding reads each expert through the page cache.
On a 512 GB Mac, the arena is 234 GB, or 29% of the experts.

The quant of the experts does not change the fit. UD-Q4_K_XL is a 1.5 TB
file with the same 62.2 GB of every-token weights, so it streams on the
same Macs, and it decodes slower because the arena holds a smaller share
of its experts. UD-Q8_K_XL has 114.7 GB of every-token weights and fits
under no ceiling below 192 GB.

### What changes the limit

The size of the every-token weights depends mostly on the bits of their
tensors. A quant that keeps attention and the shared experts at Q8
doubles what stays in memory, so choose a quant with smaller non-expert
tensors before a smaller expert quant.

A smaller KV room gives a larger arena and a shorter safe context, so
the two trade against each other. `--kv-bits 8` halves the KV cache part
of the room, and [`GMLX_STREAM_KV_CTX`](env-vars.md#runtime) sets the
number of tokens in the room.

Other models and other processes take from the arena too. The arena is
sized from the RAM that the kernel can reclaim at load. When macOS reports
memory pressure, or the server governor needs room, the arena shrinks by a
quarter of its size at each step, up to three steps. It grows again when
the memory is free.

### The plan that validate prints

`gmlx validate <ref>` prints the plan for a MoE file, on disk or on
Hugging Face, from the header of each part:

```text
  streaming: every-token weights 62.2 GB, routed experts 799.1 GB (92 layers, 896 experts, 16 per token), prefill ring 22.7 GB
    every-token by group: attention 31.8 GB, shared experts 12.9 GB, dense ffn and routers 8.2 GB, ...
    this Mac: 128 GB RAM, ceiling 97.9 GB, KV room 11.7 GB at 32768 tokens, host floor 9.6 GB
    => streams, but no decode arena (0.0 GB left after the ring and the host floor), decode runs from the page cache; expect slow decode
```

For a file that fits in RAM, the verdict says that streaming is optional.
For a file whose every-token weights do not fit, it says that the file
cannot stream and suggests a quant with smaller every-token tensors. The
loader does not refuse such a file, so check the plan first.
`gmlx validate --json` gives the same numbers under `stream`, and
`gmlx doctor` adds a line for up to three `stream: experts` entries of the
configuration file.

Some architectures have large lookup tables that every token reads a few
rows of, such as the engram tables of DeepSeek-V4.1-Flash. gmlx streams
these tables too, and the plan shows them as `streamed tables`, outside
the every-token total. A table too large for a single GPU buffer is read
from the file row by row, and the load names it on an `[install]` line.

A load prints its live budget on a `[stream] memory budget:` line when the
decode feeder starts. It counts only the RAM that is free at that moment,
so its arena can be smaller than the one in the plan.

## The lossless settings

These settings do not change the output, and all of them are on by
default for `stream: experts`, with one exception. Lookahead prestage is
off by default on GLM-5.2, the `glm-dsa` architecture, and
`GMLX_DECODE_LOOKAHEAD=1` turns it on there.
[Streaming measurements](internals/streaming-measurements.md#lossless-setting-measurements)
records the gain of each setting on real models.

| Setting | What it does | How to turn it off |
|---------|--------------|--------------------|
| Prefill feeder | It reads the experts of each layer from the GGUF into GPU memory while the previous layer computes, so each byte is read once. | `--no-prefill-feeder` |
| Decode feeder | It keeps the most used experts of each layer in the arena, and reads only the others from disk. | `--no-decode-feeder` |
| Lookahead prestage | It predicts the experts of the next layer while the current layer computes, and reads missing ones early. It changes which bytes are read, never the routing. | `GMLX_DECODE_LOOKAHEAD=0` |
| Weight pin | It locks the every-token weights in memory, so that macOS cannot evict them between tokens. | `GMLX_PIN_WEIGHTS=0` |
| GPU keep-warm | It runs a tiny kernel between layers, so that the GPU clock stays high through the disk reads. | `GMLX_GPU_KEEPWARM=0` |
| Streamed lookup tables | It streams large lookup tables before the experts, on the architectures that have them. | `GMLX_STREAM_PLE=0` |
| Stack unmap | It releases the GPU mapping of the expert stacks once the feeders read them from the file. | `GMLX_STREAM_UNMAP_STACKS=0` |
| Tail merge | It widens the prefill chunk by up to an eighth, so that a short last chunk joins the ones before it. | `GMLX_STREAM_PREFILL_TAIL_MERGE=0` |

On a short prompt, the prefill feeder reads only the experts that the
router chose, which shortens the time to the first token. Its reads bypass
the page cache, so a prefill does not push the files of other programs out
of memory.

The arena starts warm. While prefill passes through each layer, the most
used experts of the prompt are copied into the arena, so the first decoded
token already finds them. After a prompt too short for the ring, the arena
starts empty and fills within a few dozen tokens. When a multi-token call,
such as the prefill of the next chat turn, routes to more experts than the
arena holds, the feeder splits the call and still serves it from the
arena.

`--stream-fast-disk` sets how aggressively the decode feeder reads ahead.
With `auto`, the default, the feeder tests the drive at load. On a drive of
about 5 GB/s or faster, it reads predicted experts at normal disk priority,
and on a slower drive it reads them at a lower priority. `on` and `off`
force the choice.

GPU keep-warm matters because streamed decoding alternates short GPU work
with pauses for the host and the disk, and the GPU clock drops in each
pause. It uses power only while decoding, and it stops after one second
without work. Turn it off on battery. It does nothing for a model that
fits in RAM.

The weight pin is skipped, with a printed reason, when the every-token
weights are larger than 60% of RAM.

MXFP4 and NVFP4 experts, as in gpt-oss and the DeepSeek-V4-Flash Q4_K_XL
quants, use the same settings, and
[`GMLX_NATIVE_FP`](env-vars.md#runtime) controls how gmlx lays them out.

## The lossy settings

Five settings trade some output quality for decoding speed. None is on by
default, so without the flags and keys, routing is lossless. They act on
streamed MoE layers only. A setting on a model with no `stream` placement
is ignored with a message, and a MoE block type that a setting does not
support is skipped with a message.

A streamed token has three costs. Experts that are not in the arena are
read from disk, which is the largest cost when few experts are found in
the arena. Experts in the arena cost little. Each streamed layer also has
a fixed cost for kernel launches and a host sync, which does not shrink
when fewer experts are used. Each setting reduces a different cost:

| Setting | Flag | Config key | Reduces | Acts on |
|---------|------|------------|---------|---------|
| Expert cap | `--moe-experts K` | [`moe_experts`](config.md#modelsmoe_experts) | It cuts reads and compute, because each token uses K experts. | It acts on prefill and decoding. |
| Expert mass | `--moe-expert-mass P` | [`moe_expert_mass`](config.md#modelsmoe_expert_mass) | It cuts reads and compute, because each token keeps the fewest experts that cover share P of the gate weight. | It acts on prefill and decoding. |
| Miss shed | `--moe-miss-shed P` | [`moe_miss_shed`](config.md#modelsmoe_miss_shed) | It cuts disk waits by dropping only experts that are not in the arena, lowest scores first, while the kept experts cover share P. | It acts on decoding, with the decode feeder. |
| Keeper prestage | `--moe-prestage keepers` | [`moe_prestage`](config.md#modelsmoe_prestage) | It cuts the remaining disk waits, because lookahead reads only the experts that miss shed would keep. | It acts on decoding, with miss shed and lookahead. |
| Layer shed | `--moe-layer-shed P` | [`moe_layer_shed`](config.md#modelsmoe_layer_shed) | It cuts the fixed cost of each layer by skipping its routed experts with probability P. The shared expert still runs. | It acts on decoding. |

The expert cap and expert mass combine, so
`--moe-experts 6 --moe-expert-mass 0.9` uses at most 6 experts and then
drops within those 6. How much expert mass saves depends on the router of
the model. When a few experts carry most of the gate weight, most reads
disappear for a small loss. When the weight is spread evenly, it saves
almost nothing.

Measure before you choose a value. `--moe-expert-probe` keeps the trained
routing and prints two tables at exit, for decoding and for prefill. For
each candidate P, they show how many experts a token keeps and how much
gate weight it drops. Choose P from the decoding table. The probe exists
only on the command line and cannot combine with `--moe-expert-mass`, so
run it once before you put a value in a configuration file.

Choose a setting from the share of experts found in the arena:

1. Read the share from the line that the decode feeder prints at the end.
2. When the share is low, use miss shed, because it costs quality only on
   the reads that would wait for the disk.
3. When the share is high and a few experts carry most of the gate
   weight, use expert mass. When the weight is spread evenly, only layer
   shed reduces what is left.
4. Before you accept the quality cost of layer shed, check that GPU
   keep-warm is on, because a low GPU clock is a large part of the fixed
   cost.

To check a setting, compare its output with a lossless run on the same
prompt, at the same seed and with the sampling that you use.
[Streaming measurements](internals/streaming-measurements.md#certifying-a-setting)
describes a full procedure, and lists the settings that passed it on four
models.

## Serving a streamed model

Send one request at a time to a server that streams a model. The decode
feeder keeps the arena wired and runs lookahead only while each decode
step holds one token. A second request at the same time therefore slows
both, and fewer experts are found in the arena. Prefill is not affected.

The feeder settings are the [`prefill_feeder`](config.md#modelsprefill_feeder)
and [`decode_feeder`](config.md#modelsdecode_feeder) keys beside `stream`.
Each lossy setting has a model key, listed in the table of
[lossy settings](#the-lossy-settings).

## Residency of a streamed model

The server counts a `stream: experts` entry against
[`server.budget_gb`](config.md#serverbudget_gb) as its every-token weights
plus its arena and its prefill ring, and the routed experts stay on disk.
A `stream: cpu` entry counts at the full size of its file.

A streamed model's arena takes what the ceiling leaves, so the model can
use the whole budget by itself. When it does not fit beside pinned or
busy models, the server defers the load with a message that names
[`GMLX_DECODE_ARENA_GB`](env-vars.md#runtime). To keep a second model
loaded beside a streamed one, set that variable, in GiB, so that both
fit the budget. The arena is never larger than the experts, and a value
past the reclaimable RAM is reduced unless `GMLX_DECODE_ARENA_FORCE=1`
is set.

A streamed load also lowers the wired memory limit for the rest
of the process, so a dense model in memory runs without wiring from then
on.
