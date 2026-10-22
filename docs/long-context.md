# Long context

A model can run past the context it was trained at when its rope is
scaled. gmlx takes llama.cpp's rope scaling flags with llama.cpp's names and
defaults, and the same GGUF with the same flags gives the same rope on both.

## Run a model past its trained context

Qwen3.8 Flash Next is trained at 262,144 tokens. YaRN with factor 2 runs it
to 524,288:

```sh
gmlx run Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144
```

In a test at factor 2 on a 307,170-token prompt, with three codes placed at
10, 50 and 90 percent of the prompt, the model ranked each token of every
code first among text tokens.

`gmlx chat` and `gmlx serve` take the same flags for a model given on the
command line. In a config file, the model's `rope_scaling` block holds them:

```yaml
models:
  qwen3.8-flash-next-512k:
    path: Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
    rope_scaling:
      type: yarn
      factor: 2.0
      original_context: 262144
```

The [rope scaling flags](cli.md#rope-scaling-flags) and the
[`rope_scaling` keys](config.md#rope-scaling) list every setting. Under
YaRN, the context that `gmlx serve` reports in `/v1/models` grows to the
original context times the factor.

Linear scaling does not extend the context gmlx reports and holds requests
to: a prompt or a reply still stops at the trained context. Use YaRN to run
past it.

## Choose a factor

A YaRN factor changes the rotation at every position, so it also changes
the output on short prompts. Model cards that offer YaRN advise turning it
on only for long inputs and picking the smallest factor that fits: factor 2
for twice the trained context, factor 4 for four times. Two `models` ids on
one GGUF can carry different factors, one for short work and one for long
documents. They load as two models, and the prompt cache keeps their KV
apart.

The KV cache grows with the context. At 300,000 tokens Qwen3.8 Flash Next
used about 10 GiB beside its weights. A million tokens take about 28 GiB of
KV and sparse attention index, and the prefill adds a few GB on top, so
check [Capacity](capacity.md) and [Memory](memory.md) before a long run.

## Scaling the GGUF header asks for

Some GGUFs carry rope scaling in their header, such as gpt-oss with YaRN
factor 32 and Gemma 3 4B and larger with linear factor 8. gmlx applies it
with no flags, as llama.cpp does. A flag wins over its header key, and
`--rope-scaling none` turns the header's scaling off.

## Supported architectures

The flags work on these GGUF architectures. On any other, a load with the
flags refuses.

| GGUF arch | Notes |
|-----------|-------|
| `beam` | Only the sliding window layers rotate, and they keep the header's scale, so the flags change only the reported context. |
| `deepseek4`, `deepseek41` | Compressed layers scale. The other layers keep a plain rope, as in llama.cpp's `deepseek4`. |
| `ernie4_5-moe` | |
| `falcon-h1` | |
| `flint` | Only the sliding window layers rotate, and the flags scale them, as in llama.cpp. |
| `gemma` | |
| `gemma2` | Sliding window layers keep the header's scale, as in llama.cpp. |
| `gemma3`, `gemma4` | Sliding window layers keep a plain rope, as in llama.cpp. |
| `glm4`, `glm4moe` | |
| `gpt-oss` | Sliding window layers keep the header's factor 32. |
| `granite`, `granitehybrid` | |
| `hy_v3`, `hyv4` | |
| `k2-horizon` | |
| `llama` | |
| `minimax-m2`, `minimax-m3` | |
| `mistral3` | |
| `muse-glimmer`, `muse-spark` | Only the sliding window layers rotate, and they keep a plain rope, so the flags change only the reported context. |
| `qwen2`, `qwen2moe` | |
| `qwen3`, `qwen3moe`, `qwen3next` | |
| `qwen35`, `qwen35moe`, `qwen3vl`, `qwen3vlmoe` | The vision tower's text model scales too. |
| `qwen4exp` | The sparse attention indexer scales with the attention. |
| `seed_oss` | |
| `smollm3` | |

## Speculative decoding

An MTP head built into the model scales with it. The Gemma 4 assistant
reads the target's KV, so it takes the target's scaling. A DFlash or DSpark
drafter in its own GGUF takes the flags over its own header, with its own
trained context, as llama.cpp does.
