# Speculative decoding

Speculative decoding makes a model generate faster without changing its
output. gmlx turns it on by itself for most models that have a drafter,
limits it while many requests share a batch, and can trade exact output
for more speed when sampling.

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
folder. `--draft-gguf` names a drafter file, and `--speculative` turns
speculation on with a companion that the loader finds beside the model.
`--no-mtp` turns speculation off. When `--draft-gguf` or the `draft_gguf`
key names a companion for a model with a native head, the companion wins,
and `--native-mtp` forces the head. A model with a native head prints the
name of a companion that it finds beside it, and uses it only with
`--draft-gguf`.

The server enables speculation for a model through its
[`speculative`](config.md#modelsspeculative) key, and
[`draft_gguf`](config.md#modelsdraft_gguf) names the drafter. `gmlx pull`
and a [discover](config.md#model-discovery) scan pair a drafter that they
find with its model.

## Settings that speculation drops

The verification step samples with temperature, top-p, top-k and min-p
only. On `run`, speculation drops `--stop`, `--logit-bias`, the penalties,
the XTC settings, `--max-kv-size`, `--quantized-kv-start` and
`--prefill-step-size`, with a warning for each. Chat keeps the system
prompt and `--stop`, and drops the others. `--no-mtp` keeps these settings
and decodes without speculation. A
[multimodal model](vlm.md#media-with-other-features) speculates on text
turns and decodes turns with images or audio without speculation.

## How much it gains

The gain depends on how many drafts the model accepts, and on the depth of
the context. Speculation makes a dense model decode 1.6 to 1.9 times as
fast at short contexts, and keeps a smaller gain deep into long ones. MoE models
gain less, and on some of them it becomes a loss at depth, so measure
before you rely on it. Predictable text, such as code, accepts more drafts
than free prose. [Benchmarks](benchmarks.md) has the speedup curves of
each model. This command measures your own model at two context depths:

```sh
gmlx run model.gguf --bench-depths "0,4096" --speculative
```

A [quantized KV cache](kv-quantization.md#speed-and-speculative-decoding)
makes the model accept fewer drafts, and 4 bits costs the most. When
speculation is on, keep the KV cache at full precision if you can, and use
8 bits if memory requires quantization.

## Several requests at once

Speculation and batching compete for the same memory bandwidth. Checking a
draft widens the weight reads of each request, which costs little while
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
pass. A round therefore costs one small pass and one check, instead of a
check for each drafted token.

Pair the drafter with its model through `--draft-gguf`. The header of a
DFlash 2 file names its base model, so a discover scan pairs the two even
in different folders. Muse Glimmer finds a drafter in the same folder by
itself. Qwen3.8-27B keeps its native head until you pass `--draft-gguf`.

The block defaults to the size that the checkpoint was trained with, 8 on
Qwen3.8 and 16 on Muse Glimmer, so a round drafts 7 or 15 tokens.
`--draft-block-size` makes the block smaller. The drafter handles one
sequence at a time, which sets its
[server width cap](config.md#modelsspeculative_width_cap). Acceptance is
exact by default, and `--stochastic-mtp` applies to DFlash 2 as well.

## Bonsai drafters

The community DSpark drafters for Ternary Bonsai 2 27B keep the DFlash
layers and add a bigram head, which adjusts each drafted position by the
token before it, and a confidence head. They pair through `--draft-gguf`
in the same way, and the loader reports them as `dflash_dspark`. The first
position drafts too, so a block-7 drafter proposes seven tokens a round.
The confidence head, which cuts a block short, is off unless
[`GMLX_DSPARK_CONF`](internals/debug-switches.md) sets a threshold. The
output is the same either way, because acceptance is exact.

On Bonsai, the Qwen3.8-27B DFlash 2 drafter is faster for most prompts.
The Bonsai-trained drafters lead only on long code output, and on chat
prompts every drafter runs at about the speed of plain decoding. Start
with the Qwen3.8 drafter, and measure the others on your own work.

## Stochastic acceptance

By default, the model accepts a draft only when it matches the token that
the model would choose, which keeps the output identical. When sampling
at a temperature above zero, this also limits speed, because a draft
cannot match a sampled token more often than the model's probabilities
allow.

`--stochastic-mtp`, or
[`server.stochastic_mtp: true`](config.md#serverstochastic_mtp), removes
that limit with rejection sampling. The drafter samples its tokens, and
the model accepts each with probability `min(1, p/q)`, which keeps the
sampling distribution exact. The output is still a true sample from the
model's distribution, but the tokens are no longer identical to a run
without speculation. Greedy requests do not change.

The gain is largest on low-bit quants and on text where the model is
unsure, because there exact matching gives up the most drafts. Turn it on
when you sample and want more speed.
