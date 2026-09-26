# Speculative decoding

Speculative decoding makes a model generate faster without changing its
output. gmlx turns it on by itself for most models that have a drafter,
limits it while many requests share a batch, and can trade exact output
for more speed at a temperature above zero.

A small, fast drafter proposes the next few tokens, and the model checks
all of them in one pass. The model keeps the tokens it agrees with and
adds one of its own, so a round can produce several tokens for about the
cost of one. By default, the output is exactly what the model would write
alone.

- [Turning it on](#turning-it-on)
- [Settings that speculation drops](#settings-that-speculation-drops)
- [How much it gains](#how-much-it-gains)
- [Several requests at once](#several-requests-at-once)
- [DFlash 2 drafters](#dflash-2-drafters)
- [Bonsai drafters](#bonsai-drafters)
- [Stochastic acceptance](#stochastic-acceptance)

## Turning it on

Some GGUF files carry a [native head](glossary.md#native-head), a small
extra layer that drafts tokens, as the Qwen3.5, Qwen3.6 and Qwen3.8
models do. Other families use a separate drafter GGUF, a companion file
such as the gemma-4 assistant drafter or a DFlash 2 drafter.

On `run` and `chat`, speculation turns on by itself for a model with a
native head, and for DeepSeek-V4 when its companion drafter is in the same
folder. It stays off under `--stream-experts`, `--stream-cpu` and the
[lossy MoE settings](streaming.md#the-lossy-settings). With `--adapter`, the adapted model verifies each
draft, so the output matches plain decoding with the adapter. With `--mmproj`, the companion drafters of
DeepSeek-V4, Qwen3.8-Flash-Next and Muse Glimmer also turn it on for text
turns.

`--speculative` turns speculation on with a native head, or with a
companion that the loader finds beside a DeepSeek-V4, Qwen3.8-Flash-Next or
Muse Glimmer model. Other families need `--draft-gguf`, which names a
drafter file, and `--no-mtp` turns speculation off. When `--draft-gguf` or
the `draft_gguf` key names a companion for a model with a native head, the
companion takes precedence, and `--native-mtp` forces the head. A model
with a native head prints the name of any companion in its folder, and
uses that companion only with `--draft-gguf`.

The server enables speculation for a model through its
[`speculative`](config.md#modelsspeculative) key, and
[`draft_gguf`](config.md#modelsdraft_gguf) names the drafter. `gmlx pull`
and a [discover](config.md#model-discovery) scan pair a drafter that they
find with its model.

## Settings that speculation drops

The verification step samples with temperature, top-p, top-k and min-p
only. On `run`, speculation drops `--stop`, `--logit-bias`, the penalties,
the XTC settings, `--max-kv-size`, `--quantized-kv-start`,
`--prefill-step-size`, `--over-generation` and `--inject-critique`, with a
warning for each. On `chat`, speculation keeps the system prompt, `--stop` and
`--prefill-step-size`, and drops the others. `--no-mtp` keeps these settings
and decodes without speculation. A
[multimodal model](vlm.md#media-with-other-features) speculates on text
turns and decodes turns with images or audio without speculation.

## How much it gains

The gain depends on how many drafts the model accepts, and on the depth of
the context. Speculation makes a dense model decode 1.6 to 1.9 times as
fast at short contexts, and keeps a smaller gain deep into long ones. Most
MoE models gain less, and gemma-4-26B-A4B becomes slower at depth, but
Qwen3.8-Flash-Next gains more as the context grows, so measure the gain
before you rely on speculation. The model accepts more drafts on
predictable text, such as code, than on free prose.

[Benchmarks](benchmarks.md) has each model's speedup curves. This command measures your own model at two context depths:

```sh
gmlx run model.gguf --bench-depths "0,4096" --speculative
```

A [quantized KV cache](kv-quantization.md#speed-and-speculative-decoding)
makes the model accept fewer drafts, and 4 bits costs the most. When
speculation is on, keep the KV cache at full precision if you can, and use
8 bits if memory requires quantization.

## Several requests at once

Speculation and batching compete for the same memory bandwidth. Checking a
draft widens each request's weight reads, which costs little while
one stream decodes and much more when several do. The server therefore
applies a width cap to each model. It speculates while the batch is
narrow, decodes without speculation past the cap, and speculates again
when the batch shrinks.

The default cap depends on the drafter and on whether the model routes
experts. [`speculative_width_cap`](config.md#modelsspeculative_width_cap)
lists the defaults and overrides them for a model, and
`gmlx serve --speculative-width-cap` sets the cap for every model.
[Speculative batching](internals/speculative-batching.md) describes how
the batch switches between the two modes.

## DFlash 2 drafters

[DFlash 2](https://inco.ai/blog/dflash2/) is a block-diffusion drafter,
with checkpoints for Qwen3.8-27B and Muse-Glimmer-30B. One drafter pass
proposes a whole block of tokens, and the model checks the block in one
pass. A round therefore costs one small drafter pass and one check, where a
native head runs one drafter pass for each token that it drafts.

Pair the drafter with its model through `--draft-gguf`. A DFlash 2
file's header names its base model, so a discover scan pairs the two even
in different folders. Muse Glimmer finds a drafter in the same folder with
`--speculative`, and by itself when `--mmproj` loads its vision encoder. Qwen3.8-27B keeps its native head until you pass `--draft-gguf`.

The block defaults to the size that the checkpoint was trained with, 8 on
Qwen3.8 and 16 on Muse Glimmer, so a round drafts 7 or 15 tokens.
`--draft-block-size` makes the block smaller. The drafter handles one
sequence at a time, which sets its
[server width cap](config.md#modelsspeculative_width_cap). Acceptance is
exact by default, and `--stochastic-mtp` applies to DFlash 2 as well.

## Bonsai drafters

The community DSpark drafters for Ternary Bonsai 2 27B keep the DFlash
layers and add two heads, a bigram head and a confidence head. The bigram
head adjusts each drafted position by the token before it. The drafters
pair through `--draft-gguf` in the same way, and the loader reports them
as `dflash_dspark`.

These drafters also draft the block's first position, so a drafter with a
block of 7 proposes seven tokens a round.
The confidence head, which cuts a block short, is off unless
[`GMLX_DSPARK_CONF`](internals/debug-switches.md) sets a threshold. The
output is the same either way, because acceptance is exact.

On Bonsai, the Qwen3.8-27B DFlash 2 drafter is faster for most prompts.
The Bonsai-trained drafters lead only on long code output, and on chat
prompts every drafter runs at about the speed of plain decoding. Start
with the Qwen3.8 drafter, and measure the others on your own work.

## Stochastic acceptance

By default, the model accepts a draft only when it matches the token that
the model would choose, which keeps the output identical. At a
temperature above zero, exact matching also limits speed, because a draft
cannot match a sampled token more often than the model's probabilities
allow.

`--stochastic-mtp`, or
[`server.stochastic_mtp: true`](config.md#serverstochastic_mtp), removes
that limit with rejection sampling. The drafter samples its tokens, and
the model accepts each with probability `min(1, p/q)`, where p is the
model's probability of the token and q is the drafter's. This rule keeps
the sampling distribution exact. The output is still a true sample from the
model's distribution, but the tokens are no longer identical to a run
without speculation. Greedy requests do not change.

The gain is largest on low-bit quants and on text where the model is
unsure, because there exact matching gives up the most drafts. Turn it on
when you sample and want more speed.
