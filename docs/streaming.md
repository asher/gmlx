# Streaming: MoE models bigger than memory

A MoE model whose file exceeds what the GPU can wire - or exceeds RAM
outright - still runs. gmlx streams the expert weights from disk and
keeps what every token needs resident. MoE decode is what makes this
viable: only the routed experts are read per token, so the per-token
working set is a small slice of the file. The levers in this guide
exist to keep that slice cheap - fed from the SSD ahead of demand,
cached where it repeats, and, strictly opt-in, thinned where the
router can spare it.

Set expectations first. When experts stream from disk, decode is bound
by the SSD and CPU, not the GPU, and single-digit tokens per second is
normal. The feeders raise the constant, not the nature of the bound.
This is a capacity feature that makes a 200B-class MoE usable on a
64 GB machine, not a speed feature, and it is strictly for the
over-budget case. A model that fits in memory runs several times
faster on the normal GPU path.

Demand misses are served at the drive's random-read latency and the
feeders read at its queue depth, so keep the GGUF on the internal NVMe
SSD for the best result. An external drive works, but decode follows
its latency and bandwidth down.

Unless a different machine is named inline, measured numbers in this
guide are from a 14-inch M5 Max MacBook Pro (128 GB). The hardware
scope note under [Reference numbers](performance.md#reference-numbers)
covers scaling to other chips.

This guide covers launching a streamed model, choosing between the two
placements, the lossless levers (the feeders, lookahead prestage, GPU
keep-warm), and the lossy levers with a measured decision procedure.
General performance topics - measuring, quant choice, speculative
decoding, the prompt cache, memory and the KV cache - are in
[performance.md](performance.md).

## Quick start

```sh
# sharded files: point at the first shard
gmlx run GLM-5.2-UD-IQ3_XXS-00001-of-00006.gguf --stream-experts
```

Or served, with the per-model `stream` key:

```yaml
models:
  glm:
    path: ~/models/GLM-5.2-UD-IQ3_XXS-00001-of-00006.gguf
    stream: experts
    overrides: {load: {kv_bits: 8}}
```

The load is a mmap, not a read, so generation starts within seconds
whatever the file size. Prefill stages each layer's experts through the
prefill feeder. Decode starts at the disk's demand rate and improves
over the first few dozen tokens as the expert arena converges on the
model's hot set. The decode feeder's exit stats (printed at `-v` on
`run` and `chat`, always in server logs) show the arena hit rate a
session settled at.

You can stream a text model, or the text tower of a vision model. The
server accepts `stream: experts` on a VLM entry, and puts the placement
on the text tower. The server refuses `stream: cpu` on a VLM entry. It
also refuses `stream` on a speculative entry. On the CLI, MTP composes with
`--stream-experts` (not `--stream-cpu`) but defers by default: auto-MTP
stays off and an explicit `--speculative` opts in. The lossy `--moe-*`
levers below are hard-incompatible with MTP and force plain decoding.
The every-token layers and the KV cache stay resident. They must fit
under the memory ceiling. The rule and the numbers are in
[How big a model can this box stream](#how-big-a-model-can-this-box-stream).
`gmlx validate <file>` prints the verdict for this Mac. A quantized KV
cache (`--kv-bits 8`) is the usual companion at long context.

Two Kimi-K3 samples generated on this path are shown in
[internals/streaming-measurements.md](internals/streaming-measurements.md#what-the-over-budget-case-produces).

## Choosing a placement

Two placements run MoE models whose files exceed what the GPU can wire:

- `--stream-experts` keeps the every-token layers (attention, routers, shared
  experts, KV cache) on the GPU and streams the routed experts, which run on
  the CPU stream. It was historically slower than `--stream-cpu` at short
  context because of the per-layer handoff. The decode feeder (below)
  reverses that, and with a quantized KV cache it keeps its long-context
  advantage, since the large KV stays on GPU.
- `--stream-cpu` runs the whole model on the CPU device, mmap-backed, so the page cache
  streams weights from disk on demand. Past the wired budget the runtime adds
  sequential expert prefetch, advising the kernel a couple of layers ahead so
  prefill reads expert stacks at sequential bandwidth instead of demand-faulting
  them.

With the decode feeder on, `--stream-experts` is the usual choice: it
matches `--stream-cpu` on short generations, pulls ahead once the arena
warms (measured below), and keeps the large KV cache on GPU at depth.
`--stream-cpu` keeps everything on one device. In a server config,
`stream: cpu` switches the whole process to the CPU device, so it suits
a single-model setup rather than mixing with GPU-resident models.

## How big a model can this box stream

The file size does not set the limit. The every-token weights set it.

A MoE GGUF has two kinds of tensors:

- The routed experts. Each token reads a few of them. They stream from
  disk. Their size sets the decode speed. It does not set the fit.
- The every-token weights. Every token reads all of them. They stay
  resident. They are the attention layers, the shared experts, the dense
  layers, the routers, the norms, the embeddings and the output head.

The serve memory governor enforces one ceiling on tracked memory. The
ceiling is Metal's recommended working set less 5%. It never comes
closer to physical RAM than the reserve. The reserve is 8 GB or 10% of
RAM, whichever is larger. Three things share the ceiling:

1. The every-token weights.
2. The KV room. It holds the KV cache for `GMLX_STREAM_KV_CTX` tokens
   (default 32768, capped at the trained context). It also holds a
   prefill transient and an admission reserve. Each of those two is 2 GB
   or 5% of the working set, whichever is larger.
3. The host floor. The ceiling is a share of the Metal working set.
   The rest of the machine is not in it: the page cache and other
   processes. The floor keeps that share of RAM free. It is 5% of RAM,
   at least 4 GB, plus a 2.5 GB page-cache reserve
   (`GMLX_DECODE_RAM_FLOOR_GB`, `GMLX_DECODE_PAGECACHE_GB`).
   Without it a wired arena pushes the rest of the box into swap, and
   a swap storm under wired memory is a watchdog panic.
4. The decode arena. It gets everything the first three leave. The
   arena holds the hot experts. A larger arena gives a higher hit rate.
   A higher hit rate gives faster decode.

The rule: a box streams a model when the every-token weights plus the
KV room fit under the ceiling. The arena is what remains after the
host floor. The arena must be at least 1 GB. Below that the decode feeder does not start, and
decode runs from the page cache.

The prefill ring is a second check. The ring holds two copies of the
largest layer's expert stacks. Its size comes from the model, not the
box. The ring takes its room under the ceiling before the arena, and
keeps it for the process lifetime. The arena is what remains after
the ring and the floor. When the ring does not fit in what the ceiling leaves after
the every-token weights and the KV room, prefill falls back to
page-cache prefetch. The load prints the reason. Decode is not
affected, and the arena gets the whole remainder.

### The ceiling by machine size

These are the stock working sets: two thirds of RAM below 36 GB, three
quarters from 36 GB up. Sizes are decimal GB. The KV room floor is the
transient plus the admission reserve, with no KV cache in it yet.

| RAM | working set | ceiling | KV room floor | left for weights and KV cache |
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

`sudo sysctl iogpu.wired_limit_mb=<MB>` raises the working set. The
reserve still holds 8 GB or 10% back.

### Kimi-K3 as the worked example

Kimi-K3 UD-Q2_K_XL is an 861 GB file. Its every-token weights are
62.2 GB:

| group | GB |
|---|---|
| attention (MLA) | 31.8 |
| shared experts (2 per layer) | 12.9 |
| dense ffn and routers | 8.2 |
| recurrent (KDA) layers | 6.9 |
| embeddings and output head | 2.4 |

The routed experts are 799 GB: 896 experts in each of 92 layers. Each
token reads 16 of them. With a cold arena a token reads about 14 GB of
experts from disk. A warm arena serves the hot share from memory.

On a stock 128 GB box the ceiling is 97.9 GB and the KV room is
11.7 GB. That leaves 24 GB. The 22.7 GB ring fits, and the 9.6 GB
host floor takes the rest. There is no arena. The model streams, and
decode reads every expert through the page cache. On a 96 GB box the
ring does not fit, and the floor takes what is left. On a 512 GB box
the arena is 234 GB, or 29% of the experts.

The quant of the experts does not change the fit. UD-Q4_K_XL is a
1.5 TB file. Its every-token weights are the same 62.2 GB, because that
quant keeps the non-expert tensors at the same bits. It streams on the
same boxes. Its arena holds a smaller share of a larger expert set, so
it decodes slower. UD-Q8_K_XL has 114.7 GB of every-token weights. It
does not fit under the ceiling of any box below 192 GB.

### What moves the limit

- The bits of the every-token tensors. A quant that keeps attention and
  the shared experts at Q8 doubles the resident set. Pick a quant with
  smaller non-expert tensors before you pick a smaller expert quant.
- The working set. `iogpu.wired_limit_mb` raises it. The reserve still
  applies.
- The KV room. `GMLX_STREAM_KV_CTX` sizes it. A smaller room gives a
  larger arena and a shorter safe context. `--kv-bits 8` halves the KV
  cache part of the room.
- SSD bandwidth sets the speed of cold reads. It does not change the
  fit.
- Other resident models. On `gmlx serve` a streamed entry counts its
  every-token weights, its arena and its ring against `server.budget_gb`.
  The load gate also keeps the ring and KV room a resident streamed
  model has not filled: a second model must fit beside them. The auto
  arena fills the budget. Cap it with `GMLX_DECODE_ARENA_GB` to keep a
  second model resident beside it. The streamed load lowers the
  MLX wired limit for the rest of the process. A resident dense model
  runs unwired from then on.
- Other processes. The arena is sized from the RAM that is reclaimable
  at load, less the host floor. When the box later dips under the governor's kernel floor,
  the arena steps down by a quarter and regrows when the RAM returns.

### Ask gmlx

`gmlx validate <ref>` prints the plan for a MoE file. It works on a
local file and on a Hugging Face ref. A remote ref costs one header read
per shard, a few MB each. The block reads:

```text
  streaming: every-token weights 62.2 GB, routed experts 799.1 GB (92 layers, 896 experts, 16 per token), prefill ring 22.7 GB
    every-token by group: attention 31.8 GB, shared experts 12.9 GB, dense ffn and routers 8.2 GB, ...
    this Mac: 128 GB RAM, ceiling 97.9 GB, KV room 11.7 GB at 32768 tokens, host floor 9.6 GB
    => streams, but no decode arena (0.0 GB left after the ring and the host floor), decode runs from the page cache; expect slow decode
```

`gmlx validate --json` carries the same numbers under `stream`.
`gmlx doctor` adds one clause per `stream: experts` entry to its memory
row. A load prints the live budget as `[stream] memory budget:`. That
line includes the reclaimable-RAM clamp, so its arena can be smaller
than the plan's.

## Streamable lookup tables (table-before-experts)

Some architectures carry a large lookup table that every token reads only a
few rows of. Qwen4-Exp's per-layer n-gram embedding table is the shipped
case: 27-54 GB on disk depending on the build, with each decode step
gathering 16 rows (about 1.4 KB). Wiring such a table costs tens of GB to
serve kilobytes per token, so on an over-budget model `--stream-experts`
tries the table first:

- If streaming the table alone brings the resident set under the wired
  budget, the table stays file-backed and its row gathers run on a dedicated
  CPU stream; the experts stay fully resident on GPU. This is the fastest
  over-budget placement: expert reads disappear entirely.
- Otherwise the experts stream too and the table still leaves the wired
  set (compose). Measured on a 169 GB Qwen4-Exp Q6 build: short-context
  decode 8.4 to 12.6-13.4 tok/s and wired peak 106 to 54 GB vs leaving
  the table resident, converging at 16k depth. Cold prefill pays
  page-granular table faults (~10% on a cold 16k prompt) until the table
  shares the prefetch pool. `GMLX_STREAM_PLE_COMPOSE=0` keeps the table
  resident instead.

The selection is automatic and per-architecture (models with no declared
table are untouched). `GMLX_STREAM_PLE=1` forces the table onto the CPU
stream on a model that fits (for overhead measurement); `GMLX_STREAM_PLE=0`
disables the tier. Streamed table output is bit-identical to resident - the
same bytes are gathered and dequantized either way.

## The feeder paths

Streaming models engage two feeder paths by default:

- The prefill feeder (`--no-prefill-feeder` disables) stages each layer's
  expert stacks straight from the GGUF into GPU-visible ring slots while the
  previous layer computes, so every byte makes one trip - the page-cache path
  reads each expert byte twice on a machine that is at memory capacity by
  definition. Short prompts stage only the experts the router actually chose
  instead of whole layers (measured on an M3 Max, 162 GB MiniMax-M2 Q5_K_M: a
  53-token prompt's time-to-first-token dropped from 19.4 s to 11.4 s).
  The two ring slots hold the largest layer's expert stacks. Their size
  comes from the model, not the box. When they do not fit under the
  memory ceiling after the every-token weights and the KV room (a 340
  GB model on a 32 GB machine), the load says so. Prefill then falls
  back to page-cache prefetch by itself. The ring reads bypass the page
  cache (`GMLX_PREFILL_NOCACHE=0` restores buffered reads): a pass reads
  every routed expert once, and through the cache it evicts the rest of
  the box for pages it never reads again. The slots are wired for the
  pass, like the arena, in the room the budget keeps for them: a filled
  slot left unwired is what the kernel compresses first when free RAM
  is gone, and the GPU then decompresses it on every use.
- The decode feeder (`--stream-experts` only; `--no-decode-feeder`
  disables) keeps the most-routed experts of every layer in a wired,
  popularity-managed GPU arena and reads only the misses from the GGUF, at
  SSD queue depth. The arena gets what the memory ceiling leaves after
  the every-token weights, the KV room, the prefill ring and the host
  floor (see
  [How big a model can this box stream](#how-big-a-model-can-this-box-stream)).
  The load then clamps it to the RAM reclaimable at that moment, less
  the same floor.
  `GMLX_DECODE_ARENA_GB` overrides the size outright. The prefill ring
  keeps its own room under the same ceiling, so ring and arena together
  stay under it. The arena starts empty and converges within a few
  dozen tokens. Each layer wires when it first fills. Under system
  memory pressure (another model, a build) it shrinks and keeps its most
  popular experts. It regrows once pressure clears and the governor has
  its room back. A long-running model therefore coexists with a machine
  that is doing other work. `GMLX_DECODE_PRESSURE=0` pins it instead.
  Under `gmlx serve` the governor shrinks it the same way before it
  sheds a request.
  Same model and box as the prefill measurement above: decode went from 2.4
  tok/s on the page-cache path to 4.0 tok/s averaged over a 512-token
  generation (~4.7 steady, ~90% arena hits), against 3.0 tok/s for
  `--stream-cpu`. `--stream-experts` therefore matches `--stream-cpu` on
  short generations and pulls ahead roughly 1.5x once the arena warms,
  before the KV-cache advantage at depth.
- The arena also serves multi-token expert calls whose routed union
  exceeds its slots - the next chat turn's prefill after a decode, or a
  wide speculative verify batch - by halving the chunk along the token axis
  and recursing until each piece fits (`GMLX_ARENA_SPLIT_MAX_TOKENS`, default
  256, caps the size; `0` disables). Without this, those calls fall to a CPU
  page-cache gather that runs at demand-fault speed while most of RAM is
  wired. On Kimi-K3 UD-IQ2_XXS, a 48-token second-turn prefill measured 0.25
  tok/s on the fallthrough and 2.13 tok/s through the split (8.5x). The
  post-turn decode dip disappeared as well, because the reads stay on the
  arena's read pool and its popularity accounting.
- Follow-up turns longer than the split cap go back through the prefill
  feeder's ring, which was released at first decode. The ring rebuilds
  in its own room, so the arena keeps its residents and the decode
  resume starts warm. Only on a box whose free RAM is gone does the
  arena lend the ring its footprint: every layer shrinks eagerly,
  keeping its most popular experts, and the next decode releases the
  ring and regrows the arena layer by layer. Both feeders on is the
  right default for chat and serve.

- No fork beside the arena. A fork of the serve process copies every
  Metal-mapped buffer before the exec, the wired arena included, and a
  copy of the arena is a swap storm. The arena, the ring and the pinned
  weights are marked `VM_INHERIT_NONE`
  at allocation, so a child maps none of them and a fork copies none.
  The serve process reads kernel counters in process
  (`gmlx.serve.kernel_vm`), and the stock APC exact-restore gate is
  rebound to the same read. MLX's own buffers (KV cache, activations)
  still copy, so spawn nothing from a streaming server. Where a spawn is
  unavoidable, `posix_spawn` (Python `subprocess` with an absolute path
  and `close_fds=False`) copies nothing.

Streaming installs also pin the every-token weights (`GMLX_PIN_WEIGHTS=0`
disables): every
non-expert tensor - attention, routers, shared experts, norms, the lm head -
is mlocked so the kernel cannot evict it. Without the pin those weights are
plain file-backed mmap pages, and on a box running at the free-page floor the
kernel evicts them between uses. Each decode token then re-faults the whole
every-token set from disk, which on a large model saturates the SSD before
the experts read a byte. The fault traffic is invisible to the feeder's
stall accounting (it appears as compute time), so the symptom is a decode
rate stuck near `every_token_bytes / ssd_bandwidth` per token no matter the
arena hit rate. Measured on Kimi-K3 UD-IQ2_XXS (662 GB, 62 GB every-token set, M5 Max
128 GB): decode 0.10 -> 0.38 tok/s, prefill 0.62 -> 0.97 tok/s. The pin is
skipped with a printed reason when the every-token set would exceed 60%
of RAM.

A tensor the loader converts is left out of the pin, and the arena is
charged what the copy weighs rather than what the wire does. The pin works
because the runtime views the mlocked GGUF bytes in place; a converted
tensor has its own array instead, so wiring its wire range holds memory
nothing reads again. The match is on element count and dtype width, not on
tensor names, so it covers a head under any name, tied or not.
HY4-preview holds an F32 `output.weight` (2.97 GB) as a 1.49 GB bf16 array:
the pin drops to 19.7 GB from 22.6 GB and the arena reads 3.5% less from
disk per token. `GMLX_PIN_CAST_EXCLUDE=0` pins the wire bytes as before.

Two streaming installs in one process share nothing. The weight pin and the
decode arena are both mlocked, and mlocked pages are never compressed,
swapped or evicted, so they raise no memory pressure and jetsam never
selects a process to kill. A machine with nothing reclaimable stops rather
than recovers. A streaming install therefore reclaims any install whose
model is gone (a released model keeps its arena until a collection breaks
the feeder/module cycle), then charges what is still held against its own
weight pin, which sizes against total RAM. The decode arena needs no such
charge: it sizes against reclaimable RAM, and a pinned page leaves that
count. A second model on a full box degrades to the page-cache path with a
printed reason instead of wiring the machine solid. Release a model with
`gmlx.stream.installs.release(model)` to give the next one the full
budget; the server does this at eviction.

In server configs the placement is the per-model `stream: experts | cpu`
key and the feeder opt-outs are `prefill_feeder: false` /
`decode_feeder: false`.

Send only one request at a time to a server that streams a model. The
engine can batch concurrent requests, but the streaming tier makes decode
faster only when a step contains one token. The wired-memory refresh, the
lookahead prestage, and the GPU-side token path each test the step width,
and each stays off when a step contains more than one token. A second
concurrent request thus puts both requests on the slow path, and the
arena loses the hit rate that it built. This does not affect prefill.

When a larger-than-RAM model is released, its page cache is also released, via
`msync(MS_INVALIDATE)` over the shards - at process exit, or at unload on a
running server (`GMLX_RELEASE_PAGECACHE=0` disables).

## Lookahead prestage

With the decode feeder on, arena misses are also prestaged by lookahead
(`GMLX_DECODE_LOOKAHEAD=0` disables): each MoE layer runs the next MoE layer's router
on its own input and pre-reads the predicted misses on a small dedicated pool
while the current layer computes. The residual changes little between
adjacent sublayers, so the prediction lands. Measured recall of the next
layer's actual top-k is ~78% on GLM-5.2 (@8) and MiniMax-M3 (@4), against
~35% for previous-token routing reuse. Predictions move bytes and nothing
else; routing and outputs are bit-identical. Speculation is kept off
the demand path three ways: prestage reads are submitted only after the
current layer's demand misses have finished, the read threads run at
utility disk-I/O priority so the kernel services demand misses first
(`GMLX_DECODE_LOOKAHEAD_IOPOL=0` restores default priority), and every
layer settles its in-flight prestages before serving. A per-layer rank
gate watches how often each prediction rank actually lands and stops
submitting ranks that measure below `GMLX_DECODE_LOOKAHEAD_MIN_P` (default
`0.5`). Predictions the router then does not route to are cancelled before
they reach the disk when their reads have not started
(`GMLX_DECODE_LOOKAHEAD_CANCEL=0` disables). Together these keep the
wasted-read tax near zero on models where the SSD is the bottleneck.
`GMLX_DECODE_LOOKAHEAD_PROBE=1` prints the per-layer recall table at exit without
issuing reads, the check worth running on a new model family.

## GPU keep-warm (`--gpu-keepwarm`)

Streamed decode has a work pattern the GPU's power management punishes:
sub-millisecond compute bursts separated by host and disk gaps every MoE
layer. The GPU races to idle in each gap, its clocks sag, and the next
burst pays the ramp back up. On identical per-layer work that shows as
3-5x inflation (0.3 ms warm vs up to 4+ ms ramp-inflated). The more
per-token host syncs a model's decode path has, the more of its token time
is ramp rather than work.

`--gpu-keepwarm` (default on for streamed installs; `GMLX_GPU_KEEPWARM=0`
disables) holds clocks up with a tiny heartbeat
kernel (a 256x256 matmul every 0.5 ms) on its own stream from a background
thread. It moves no model bytes and changes no outputs - the win is purely
clock residency. Measured on the production configs of two over-RAM
models, ABBA-alternated medians over 4 reversed rounds of 512-token
generations:

| model | config under test | without | with | lift |
|---|---|---|---|---|
| GLM-5.2 UD-IQ3_XXS (282 GB, 75 streamed layers) | arena capped at 70 GB (`GMLX_DECODE_ARENA_GB=70`) + `--moe-miss-shed 0.85`, lookahead off | 2.51 tok/s | 3.64 tok/s | +45% |
| Hunyuan3 IQ4_XS (159 GB, 79 streamed layers) | `--moe-layer-shed 0.10` + `--moe-miss-shed 0.90` | 4.01 tok/s | 5.29 tok/s | +32% |

The diagnostic signature says whether a given model will benefit. Stall
time and arena hit rate are unchanged by the heartbeat, because the disk
is doing the same work. So when a streamed model's per-token time is
dominated by the eval/sync bucket rather than stalls
(`GMLX_DECODE_PHASE_STATS=1` prints the split), clock sag is a candidate
and keep-warm is the cheap test. Dense in-RAM decode does not have the
gap pattern and gains nothing.

The cost is power, and only while decoding: the heartbeat parks (no GPU
work) after one second without decode activity and wakes on the next
streamed decode call (`GMLX_KEEPWARM_IDLE_S` tunes the window; `0` beats
continuously). An idle server pays nothing, and the first token after an
idle gap pays one clock ramp.

The heartbeat kernel itself is nearly free, and measuring it shows the
mechanism plainly (powermetrics, M5 Max, heartbeat alone on an otherwise
idle box): GPU power 199 mW -> 287 mW, while GPU active residency went
from 58% to 99.8% with the clock still at the 338 MHz floor. The beat
does not raise clocks - it removes idleness. During decode the model's
own bursts set the clock level; the heartbeat keeps the GPU from going
idle in the gaps, so the governor holds that level instead of sagging
and re-ramping every layer. The real power cost is therefore holding
the decode-level P-state through the gaps, which scales with how hard
the workload drives the clock; the ~0.1 W kernel is noise against it.
The trade is more watts for the same output, faster. On battery,
`GMLX_GPU_KEEPWARM=0` takes it back.

## The lossy levers

Four levers trade a bounded amount of output quality for decode speed on
streamed MoE models. None is ever on by default: absent flags and absent
config keys mean lossless routing. All are decode-side - a large prefill
chunk routes to nearly every expert either way - and all act only on
streamed layers.

A streamed decode token pays three distinct costs, and each lever cuts a
different one. Experts that miss the decode arena are read from disk at
demand latency - the dominant cost when the hit rate is low. Experts
already resident in the arena cost only gather compute, which is small.
And every streamed MoE layer pays a fixed per-layer overhead (kernel
launches plus a host synchronization) that does not shrink when fewer
experts are routed.

The router-side levers thin the routed set itself, cutting reads and
compute in proportion. `--moe-experts K` caps the router at a fixed K
experts per token. `--moe-expert-mass P` is the adaptive version and
usually the better trade of the two: each token keeps the smallest set of
its routed experts covering share P of the gate mass, so the dropped mass
is bounded by 1-P and lands on tokens where the router was already
confident. A token whose top 3 experts carry 92% of the mass reads 3
experts at P=0.9, while an uncertain token keeps the full fan-out. How
much expert-mass buys is a property of the model's router. On a
concentrated router it is the strongest lever available: most reads
disappear for a few percent of dropped mass. On a flat router it buys
almost nothing - a 299B model we measured keeps 7.1 of its 8 experts at
P=0.90. Measure rather than guess: `--moe-expert-probe` runs the trained
routing losslessly and prints, per candidate P, the experts kept, the
implied read fraction, and the mass actually dropped, with decode and
prefill tabled separately. Size P against the decode table. The two router levers
compose (`--moe-experts 6 --moe-expert-mass 0.9` caps at 6, then drops
within the 6).

The staging levers act at the decode feeder instead of the router.
`--moe-miss-shed P` drops routed experts that would demand-miss the
arena, lowest scores first, keeping at least share P of the token's gate
mass. The quality budget is spent exactly where the disk stalls are: an
arena-resident or prestage-inflight expert is never dropped, and a shed
expert earns no popularity credit, so the arena keeps its hot set.
It needs the decode feeder and a block that hands router scores to the
expert call. Where it engages, it is the most targeted lever per point of
quality spent, and its payoff scales directly with the miss rate.

`--moe-prestage keepers` attacks miss-shed's residual stalls from the
speculative side. It adds no quality knob of its own; it applies the
policy miss-shed already defines, one layer earlier. In the default
`ranked` mode, lookahead prestages its rank-gated predictions with
guess-grade caution (never evicting a more popular resident), which
caps how many demand misses it can absorb. And because an inflight
read exempts its expert from the shed, ranked lookahead incidentally
rescues some experts the policy would have dropped, at the price of
the read. Keeper mode filters each prediction through the policy
instead: an expert that would be shed if it demand-missed is not read
at all, and the predicted keepers are staged demand-grade, since if
the prediction is right the demand path would do those same reads
synchronously one layer later.
Where prediction recall is good, this converts demand stalls into reads
that overlap compute. The lookahead exit stats (submitted vs adopted)
are the guardrail that the added aggression is landing. Requires
`--moe-miss-shed` to define the policy.

`--moe-layer-shed P` skips a streamed MoE layer's routed experts entirely
with probability P per token (the layer's shared expert still runs). It
is the blunt end of the scale, and the only lever that also cuts the
per-layer overhead - which makes it the one that still pays when the
arena hit rate is high and misses are rare.

In server configs the lossy levers and keeper prestage are the
per-model `moe_experts: K` / `moe_expert_mass: P` / `moe_miss_shed: P`
/ `moe_layer_shed: P` / `moe_prestage: keepers` keys (or the matching
`serve` flags for a single positional model). The probe stays
CLI-only, so size P with a `gmlx run --moe-expert-probe` pass before
pinning a value in a config.

Which to reach for is a measurement, not a doctrine. Run the probe once,
and read the decode feeder's exit stats (arena hit rate; printed by
`run`/`chat` at `-v`, and always in server logs) from a representative
session. The hit rate decides first. A low hit rate - an arena small
relative to the model - makes `--moe-miss-shed` the lead lever whatever
the router looks like. It spends only on calls that would stall, so it
beats expert-mass on cost at equal reads saved, and a probe-attractive
concentration number can still lose to it outright (measured below). At
a healthy hit rate, a concentrated router points to `--moe-expert-mass`,
which removes reads and compute together at minimal dropped mass. A flat
router takes it off the table and leaves the per-layer overhead as the
standing cost, which only `--moe-layer-shed` touches. A large share of
that overhead is clock ramp, though, which the lossless keep-warm above
removes first. Run it before spending quality here.

The per-model case studies, their sample galleries and the certification
method are in
[internals/streaming-measurements.md](internals/streaming-measurements.md).

## Native-fp experts (MXFP4/NVFP4)

Models with MXFP4/NVFP4 expert tensors (gpt-oss, DeepSeek-V4-Flash Q4_K_XL
quants) participate in all of the above on equal terms. By default these
tensors are eagerly repacked into MLX's packed layout at load - fine in RAM,
fatal over it (the repack materializes every expert). `GMLX_NATIVE_FP`
picks the layout: `wire` keeps them as zero-copy GGUF wire bytes served by
mlx-kquant's fp4 kernels (loads in seconds, streams like any k-quant),
`packed` forces the repack, and the default `auto` chooses wire whenever a
streaming placement is requested or the file exceeds ~90% of the wired budget.
Wire mode is a hair slower than packed when the model fits in RAM (gpt-oss
decode ~5% at depth 0, converging at depth; prefill at parity or better).
It is what makes the over-RAM case work at all, and it cuts the load time
from minutes of repack to a mmap.
