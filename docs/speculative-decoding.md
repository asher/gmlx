# Speculative decoding

Speculative decoding makes a model generate faster without changing its
output. A small, fast drafter proposes the next few tokens, and the model
checks all of them in one pass. It keeps the tokens it agrees with and adds
one of its own, so a round can produce several tokens for about the cost of
one.

```sh
gmlx run Qwen3.8-27B-UD-Q6_K.gguf                          # a native head turns it on
gmlx run model.gguf --draft-gguf drafter.gguf              # a separate drafter file
gmlx run model.gguf --bench-depths "0,4096" --speculative  # measure the gain
```

## Turning it on

Some GGUF files carry a [native head](glossary.md#native-head), a small
extra layer that drafts tokens, as the Qwen3.5, Qwen3.6 and Qwen3.8 models
do. `run` and `chat` turn speculation on for these without a flag. DeepSeek-V4,
Qwen3.8-Flash-Next and Muse Glimmer turn it on the same way when their
companion drafter sits in the model's folder, or in an `MTP` folder inside
or beside it. Other families, such as gemma-4 with its assistant drafter,
need `--draft-gguf`.

| Flag | Config key | Effect |
|------|------------|--------|
| `--speculative` | [`speculative: true`](config.md#modelsspeculative) | Turns it on with a native head or a companion that the loader finds. |
| `--draft-gguf PATH` | [`draft_gguf`](config.md#modelsdraft_gguf) | Drafts with this file. Turns speculation on, and wins over a native head. |
| `--native-mtp` | [`native_mtp: true`](config.md#modelsnative_mtp) | Drafts with the native head even when a drafter file is set. |
| `--no-mtp` | `speculative: false` | Turns it off. |

On the server, a model speculates when its `speculative` key is true.
`gmlx init` sets it for models with a head, and `gmlx pull` and a
[discover](config.md#model-discovery) scan pair a drafter that they find
with its model.

Speculation works with some other features and stays off with others:

- `--stream-experts`: off by default, and `--speculative` turns it on.
- `--stream-cpu`: `run` and `chat` refuse `--speculative`.
- The [lossy MoE settings](streaming.md#the-lossy-settings): off, and
  `chat` refuses `--speculative`.
- `--mmproj`: text turns speculate, and turns with images or audio do not.
- `--adapter`: the adapted model checks each draft, so the output matches
  plain decoding with the adapter.

## Settings that speculation drops

In `run` and `chat`, speculation drops a few settings with a warning for
each, such as:

- `--max-kv-size` and `--quantized-kv-start`
- on `run`, also `--stop` and `--prefill-step-size`

Pass `--no-mtp` to keep them.

Logit bias, the penalties and XTC work with speculation. They lower the
share of accepted drafts, XTC the most, so a request that sets them gains
less from speculation.

## How much it gains

The gain depends on how many drafts the model accepts. A dense model
decodes about 1.6 to 1.9 times as fast at short contexts and keeps a
smaller gain deep into long ones. Most MoE models gain less, and some
become slower at depth. Code gains more than free prose, because it is
easier to predict. [Benchmarks](benchmarks.md) has each model's speedup
curves, and `--bench-depths` measures your own model, as in the example
above.

A [quantized KV cache](kv-quantization.md#speed-and-speculative-decoding)
lowers the share of accepted drafts, and 4 bits costs the most. With
speculation on, keep the KV cache at full precision if you can, and use 8
bits if memory requires it.

## Several requests at once

Checking a draft costs little while one request decodes and much more when
several do. The server therefore gives each model a width cap, the number
of requests that speculate together. Past the cap, the oldest requests keep
speculating and the others decode without speculation in the same step,
and when a speculating request finishes, the next oldest takes its place.
This needs a drafter that keeps its own state for each request and a model
that undoes each request's rejected drafts separately. In these cases the
whole batch decodes without speculation past the cap instead, and
speculates again when the batch shrinks:

- Any mixture-of-experts model, because the padded positions of the other
  requests widen the set of experts each step reads.
  `GMLX_MTP_MIXED=1` gives such a model mixed batches anyway.
- The gemma-4 assistant drafter, which reads the model's own KV cache.
- gemma-4, DeepSeek-V4, Nemotron-H and MiniMax-M3 models, and the MTP heads
  of GLM-DSA, Inkling and Hy-V4 models, which accept the same number of
  drafts for every request in a batch.
- A quantized KV cache that cannot roll back each request separately.

| Drafter | Default cap |
|---------|-------------|
| A native head on a dense Qwen model | No cap |
| The gemma-4 assistant drafter, and any family without a default of its own | 2 |
| Any mixture-of-experts model | 1 |
| The heads of Hy3, DeepSeek-V4, Muse Glimmer, Qwen3.8-Flash-Next and GLM-5.3-Flash, and every DFlash drafter | 1, which no setting raises |

[`speculative_width_cap`](config.md#modelsspeculative_width_cap) sets the
cap for one model, and `0` removes it. `gmlx serve --speculative-width-cap N`
sets it for every model and overrides each model's key. [Speculative batching](internals/speculative-batching.md)
describes how the batch switches between these modes.

## DFlash 2 drafters

[DFlash 2](https://inco.ai/blog/dflash2/) is a block-diffusion drafter,
with checkpoints for Qwen3.8-27B and Muse-Glimmer-30B. One drafter pass
proposes a whole block of tokens, where a native head runs one pass for
each token it drafts.

```sh
gmlx run Qwen3.8-27B-UD-Q6_K.gguf --draft-gguf Qwen3.8-27B-DFlash2-Q8_0.gguf
```

A DFlash 2 file's header names its base model, so a discover scan pairs the
two even in different folders. Muse Glimmer also finds a drafter in its own
folder with `--speculative`, or when `--mmproj` loads its vision encoder.
Qwen3.8-27B keeps its native head until you pass `--draft-gguf`.

The block is the size the checkpoint was trained with, 8 on Qwen3.8 and 16
on Muse Glimmer, so a round drafts 7 or 15 tokens. `--draft-block-size`
makes it smaller. The drafter handles one request at a time, so its server
width cap is 1.

gmlx also runs the community DSpark drafters for Ternary Bonsai 2 27B,
which draft a block in one pass too. The loader reports them as
`dflash_dspark`. On Bonsai, the Qwen3.8-27B DFlash 2 drafter is faster for
most prompts, and the Bonsai-trained drafters lead only on long code
output. Start with the Qwen3.8 drafter and measure the others on your own
work.

## Context copy

When the output repeats the prompt, as when the model edits code it was
given or quotes a document, the prompt itself is a good draft. Context copy
looks for the newest generated tokens in the prompt, and when it finds them,
the round checks the prompt tokens that follow in place of the drafter's
draft. The model accepts them by the same rule as any draft, so the output
does not change. The drafter learns the accepted tokens and drafts the next
round as usual.

A copy can run longer than the drafter's block, up to 24 tokens. Copies
that the model accepts whole grow toward that limit. When the model rejects
most of a copy, or a copy round makes tokens more slowly than the drafter's
rounds, copying pauses for a while and starts again shorter. A copy is
at least twice as long as what a drafter round adds on average, so a
drafter that already predicts the text well keeps most rounds. On the server, a
request copies when it starts while no other request decodes, and stops
copying when another request joins it.

[`context_copy: true`](config.md#modelscontext_copy) turns it on for a
model. `GMLX_CONTEXT_COPY=1` turns it on for every model, in `run` and
`chat` too, `GMLX_CONTEXT_COPY=0` turns it off everywhere, and
`GMLX_CONTEXT_COPY_MAX` sets the longest copy. Context copy gains on
grounded answers, edits and rewrites, and does nothing on free prose that
does not repeat the prompt.

## Stochastic acceptance

By default, the model accepts a draft only when it matches the token that
the model would choose, so the output is identical to a run without
speculation. At a temperature above zero, this limits speed, because a
draft cannot match a sampled token more often than the probabilities allow.

`--stochastic-mtp`, or
[`server.stochastic_mtp: true`](config.md#serverstochastic_mtp), accepts
drafts by rejection sampling instead. The output is still a true sample
from the model's distribution, but no longer token-identical to a run
without speculation. Greedy requests do not change. `--stochastic-mtp`
works with DFlash 2 drafters too. The gain is largest on low-bit quants and on text where the model
is unsure. Turn it on when you sample and want more speed.
