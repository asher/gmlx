# Models larger than memory

gmlx can run a mixture-of-experts model whose file is larger than your
Mac's memory. It keeps the parts that every token reads in memory and reads
the routed experts from disk as each token needs them. A model of about 200
billion parameters runs this way on a Mac with 64 GB.

```sh
gmlx validate GLM-5.2-UD-IQ3_XXS-00001-of-00006.gguf          # can this Mac stream it?
gmlx run GLM-5.2-UD-IQ3_XXS-00001-of-00006.gguf --stream-experts
```

For a file in several parts, name the first part. Decoding runs at the
speed of the SSD, and a few tokens per second is normal, so stream only a
model that does not fit. Keep the GGUF on the internal SSD, because an
external drive slows decoding in proportion to its latency.

- [Check whether a model fits](#check-whether-a-model-fits)
- [Run a streamed model](#run-a-streamed-model)
- [When a model cannot stream](#when-a-model-cannot-stream)
- [The lossless settings](#the-lossless-settings)
- [The lossy settings](#the-lossy-settings)
- [Features with streaming](#features-with-streaming)
- [Serving a streamed model](#serving-a-streamed-model)
- [Residency of a streamed model](#residency-of-a-streamed-model)

## Check whether a model fits

`gmlx validate` reads only the GGUF header, so it works on a local file
or on Hugging Face before you download anything. For a MoE file it prints
a streaming plan for this Mac. This plan is for Kimi-K3 UD-Q2_K_XL, an
861 GB file, on a Mac with 128 GB:

```text
  streaming: every-token weights 62.2 GB, routed experts 799.1 GB (92 layers, 896 experts, 16 per token), prefill ring 22.7 GB
    every-token by group: attention 31.8 GB, shared experts 12.9 GB, dense ffn and routers 8.2 GB, ...
    this Mac: 128 GB RAM, ceiling 97.9 GB, KV room 11.7 GB at 32768 tokens, host floor 9.6 GB
    => streams, but no decode arena (0.0 GB left after the ring and the host floor), decode runs from the page cache; expect slow decode
```

The last line is the verdict:

| Verdict | What to do |
|---------|------------|
| `=> streams with --stream-experts (server: stream: experts)` | Run it. The line above gives the decode arena's share of the experts, and a larger share decodes faster. |
| `=> streams, but no decode arena (...); expect slow decode` | Runs, slowly. [When a model cannot stream](#when-a-model-cannot-stream) lists ways to free memory for the arena. |
| `=> cannot stream: every-token weights ... + KV room ... exceed the ... ceiling by ...` | Pick another quant, as [When a model cannot stream](#when-a-model-cannot-stream) describes. |
| `=> the whole file fits in RAM; streaming is optional` | Run it without `--stream-experts`. |

A dense model larger than RAM has no experts to stream and needs a smaller
quant.

`gmlx validate --json` gives the same numbers under `stream`, and
`gmlx doctor` names any `stream: experts` entry in your configuration file
that cannot stream.

## Run a streamed model

Add `--stream-experts` on `run` or `chat`. In the
[configuration file](config.md), set [`stream`](config.md#modelsstream) on
the model entry:

```yaml
models:
  glm:
    path: GLM-5.2-UD-IQ3_XXS-00001-of-00006.gguf
    stream: experts
    overrides: {load: {kv_bits: 8}}
```

Generation starts within seconds whatever the file's size, because the
load maps the file instead of reading it. For long contexts, add a
quantized KV cache with `--kv-bits 8` or `kv_bits: 8`.

The decode arena is wired memory that holds the most used experts. At the
end of a run, gmlx prints how often a token found its experts there, on a
`[stream] decode feeder arena hit rate:` line. `run` and `chat` print it
with `-v`, and the server always logs it.

`--stream-experts` streams only a model larger than 90% of the GPU working
set. A smaller model loads into memory as usual.

| Placement | What stays on the GPU | Use it for |
|-----------|-----------------------|------------|
| `--stream-experts`, `stream: experts` | Attention, routers, shared experts and the KV cache | Most cases, including long contexts, chat and a server with other models |
| `--stream-cpu`, `stream: cpu` | Nothing. The whole model runs on the CPU from the page cache. | A model on its own server. The decode feeder and its settings do not apply. |

## When a model cannot stream

A MoE file holds two kinds of tensors. The routed experts stream from
disk, so their size sets the decode speed but not whether the model fits.
Everything else is read by every token and stays in memory: attention,
shared experts, dense layers, routers, embeddings and the output head.
These every-token weights, plus room for a 32768-token KV cache, must fit
under the memory ceiling, which is close to the GPU working set that macOS
allows.

To make a model fit, or to give it a decode arena:

- Pick a quant with smaller every-token tensors. A smaller expert quant
  does not change the fit. The `every-token by group` line shows which
  tensors are large, and a quant that keeps attention and the shared
  experts at Q8 doubles them.
- Shrink the KV room. `GMLX_STREAM_KV_CTX` sets how many tokens it holds,
  and `--kv-bits 8` halves the KV cache part. The safe context is then
  shorter.
- Raise the GPU limit, as [The GPU memory limit](memory.md#the-gpu-memory-limit)
  describes.
- Quit other models and apps. The arena takes what is left of the RAM at
  load, and it shrinks when macOS reports memory pressure.

```sh
GMLX_STREAM_KV_CTX=16384 gmlx validate model-00001-of-00004.gguf
```

[Streaming measurements](internals/streaming-measurements.md#how-the-memory-ceiling-is-shared)
gives the ceiling for each RAM size and how the memory is divided.

## The lossless settings

Every setting that speeds up streaming without changing the output is on
by default for `stream: experts`. The one exception is lookahead prestage
on GLM-5.2, where it costs more than it saves.
[Streaming measurements](internals/streaming-measurements.md#the-lossless-settings)
lists the settings and how to turn each one off. These two have a flag:

- `--stream-fast-disk {auto,on,off}`, or the
  [`stream_fast_disk`](config.md#modelsstream_fast_disk) key, sets how hard
  decoding reads ahead. `auto` tests the drive at load and reads ahead
  harder on a drive of about 5 GB/s or faster.
- `--no-gpu-keepwarm` turns off a tiny kernel that keeps the GPU clock high
  between disk reads. It costs a few watts while decoding, so turn it off
  on battery.

## The lossy settings

Five settings trade some output quality for decoding speed. None is on by
default, and they act only on a model with a `stream` placement.

| Setting | Flag | Config key | What it does | Acts on |
|---------|------|------------|--------------|---------|
| Expert cap | `--moe-experts K` | [`moe_experts`](config.md#modelsmoe_experts) | Each token uses K experts. | Prefill and decoding |
| Expert-mass | `--moe-expert-mass P` | [`moe_expert_mass`](config.md#modelsmoe_expert_mass) | Each token keeps the fewest experts that cover share P of the gate weight. | Prefill and decoding |
| Miss-shed | `--moe-miss-shed P` | [`moe_miss_shed`](config.md#modelsmoe_miss_shed) | Drops experts that are not in the arena, lowest first, while the rest cover share P. | Decoding |
| Keeper prestage | `--moe-prestage keepers` | [`moe_prestage`](config.md#modelsmoe_prestage) | Reads ahead only the experts that miss-shed would keep. Needs miss-shed. | Decoding |
| Layer-shed | `--moe-layer-shed P` | [`moe_layer_shed`](config.md#modelsmoe_layer_shed) | Skips a layer's routed experts with probability P. The shared expert still runs. | Decoding |

Choose a setting from the arena hit rate, which the
`[stream] decode feeder arena hit rate:` line prints:

1. When the hit rate is low, use miss-shed. It costs quality only on the
   reads that would wait for the disk.
2. When the hit rate is high and a few experts carry most of the gate
   weight, use expert-mass.
3. When the weight is spread evenly, only layer-shed helps. Check first
   that GPU keep-warm is on.

`--moe-expert-probe` shows how much expert-mass would save. It keeps the
trained routing and prints two tables at exit, for decoding and for
prefill, with the experts kept and the gate weight dropped at each P.
Choose P from the decoding table.

```sh
gmlx run model-00001-of-00004.gguf --stream-experts --moe-expert-probe
```

To check a setting, compare its output with a lossless run on the same
prompt, at the same seed and with your sampling settings.
[Streaming measurements](internals/streaming-measurements.md#certifying-a-setting)
has a full procedure and the values that passed it on four models.

## Features with streaming

| Combination | Result |
|-------------|--------|
| A placement with `--mmproj` | [Media with other features](vlm.md#media-with-other-features) gives the result for each placement. |
| `--stream-experts` with `--speculative` | Works. Automatic speculation stays off under streaming. |
| `stream` with `speculative` on a server entry | The model loads streamed, without speculation. |
| A lossy setting with speculative decoding | Automatic speculation turns off. Chat refuses an explicit `--speculative`. |

## Serving a streamed model

Send one request at a time to a server that streams a model. A second
request at the same time slows both, because fewer experts are found in
the arena. Prefill is not affected.

The [`prefill_feeder`](config.md#modelsprefill_feeder) and
[`decode_feeder`](config.md#modelsdecode_feeder) keys turn the two feeders
off for a model.

## Residency of a streamed model

A `stream: experts` entry counts against
[`server.budget_gb`](config.md#serverbudget_gb) as its every-token weights,
its arena and its prefill ring. A `stream: cpu` entry counts at the full
size of its file.

The arena takes what the ceiling leaves, so a streamed model can fill the
budget by itself. To keep a second model loaded beside it, set
[`GMLX_DECODE_ARENA_GB`](env-vars.md#runtime) to an arena size in GiB that
leaves room for both. Streaming also lowers the wired memory limit for the
rest of the process, so a dense model loaded later runs without wiring.
