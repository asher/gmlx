# KV cache quantization

Quantizing the KV cache stores a request's context in fewer bits, so a
long context uses less memory. gmlx offers two schemes, affine and kvarn,
and picks one for each model unless you name it. Both cost a little
quality, and on some models a little speed.

```sh
gmlx run model.gguf --kv-bits 8                           # gmlx picks the scheme for this model
gmlx run model.gguf --kv-bits 8 --kv-quant-scheme uniform # affine on any model
gmlx run model.gguf --kv-quant-scheme kvarn               # kvarn at 6 bits, about the quality of affine 8 in less memory
```

On the server, set the same choice with the [load keys](config.md#model-loading):

```yaml
models:
  my-model:
    path: model.gguf
    overrides: {load: {kv_bits: 8}}
```

## The scheme gmlx picks

With `--kv-bits` set and no `--kv-quant-scheme`, gmlx picks the scheme
from the model's cache shape and prints its pick beside the `[kv]` line,
such as `[kv] auto picked affine: full attention, where affine decodes
faster`. How much quantization saves also depends on that shape:

| Cache shape | Families | Cache at 32K in fp16 | Pick |
|---|---|---|---|
| Full attention on all layers | Llama, Mistral, dense Qwen3 | 4 to 8 GB for an 8B to 32B model | Affine, which decodes faster here. Name kvarn at 6 for the quality of affine 8 in less memory, or at 4 when memory is the limit. |
| Recurrent hybrid, one attention layer in four | Qwen3.5, Qwen3.6, Qwen3.8 | About 2 GB at 27B, plus a fixed recurrent state | kvarn, which keeps [prompt caching](prompt-cache.md). Quantize only when the context is the limit, at 64K and up. |
| Sliding-window mix | gemma-4 | The window layers stop growing at the window. | kvarn, which keeps prompt caching. Only the global layers quantize, so the saving is small. |
| MLA latent | DeepSeek-V4 and Kimi K2, and the MLA layers of the hybrids GLM-5.3-Flash and Kimi K3 | Already compressed by the architecture | Affine. Kimi K2 and K3 keep an fp16 cache under either scheme. |
| Attention sinks, head dimension 64 | gpt-oss | Each token adds little cache. | Neither. Affine cannot read the sinks and kvarn needs a head dimension of 128, 256 or 512, so the cache stays fp16. |

A few settings change the pick. A flag that only one scheme reads picks
that scheme: `--kv-group-size` and `--quantized-kv-start` pick affine, and
`--kv-tail-tokens`, `GMLX_KVARN_BITS` and a width of 5 pick kvarn. On a
model kvarn cannot take, the kvarn flags give way to affine, and a width of
5 stops the load with the reason, since affine has no 5-bit width. On `run`
and `chat`, `--max-kv-size` on a full-attention model picks kvarn, because
affine cannot quantize the rolling window.

Under speculative decoding, gmlx picks affine wherever kvarn would decline,
as on a sliding-window model or with a drafter that reads the target's
cache. Speculation drops `--quantized-kv-start`, so that flag does not
steer the pick there. With `--mmproj`, `run` and `chat` use affine, except
a text-only `run` with a drafter, which picks as speculative decoding does.
Name a scheme to override the pick.

## Which layers quantize

The load prints a `[kv]` line that says which layers quantize and why, and
`GET /v1/models` reports the result for each loaded model as `kv_quant`.
Recurrent state, sliding windows and the last layer of a deep stack stay
fp16. When you name kvarn for a model on which no layer can use it, the
model runs fp16 and says why, and so does affine on a model whose attention
cannot read a quantized cache. Named with `--mmproj`, kvarn leaves the
cache of `run` and `chat` in fp16.

## Options

| Name | Default | Meaning |
|------|---------|---------|
| `--kv-bits N` | off, or 6 under kvarn | Width. Affine takes 2, 3, 4, 6 or 8. kvarn takes 2, 3, 4, 5, 6 or 8. |
| `--kv-quant-scheme` | `auto` | `auto` picks for the model. `uniform` is affine. `kvarn` normalizes the values before rounding them. |
| `--kv-group-size` | `64` | Affine group size, 32, 64 or 128. Each group stores an fp16 scale and offset. |
| `--quantized-kv-start N` | `0` | Affine only. Keeps the cache fp16 until it holds N tokens. |
| `--kv-tail-tokens N` | `1024` | kvarn only. The newest N tokens stay fp16. A multiple of 128, or 0. |
| [`GMLX_KVARN_BITS`](env-vars.md#runtime) | none | kvarn only. Separate widths for keys and values, such as `k6v5`. |

Under kvarn, the first 128 tokens and the tail that `--kv-tail-tokens` sets
stay fp16. At 6 bits, the rest takes about 40% of the memory of fp16.

## Quality

kvarn keeps the output closer to that of an fp16 cache than affine
quantization does at every width below 8, by a factor of 3 to 5 at 2 to 4
bits. The two converge at 8 bits. At 6 bits, kvarn matches or nearly
matches affine at 8 in three quarters of the memory. Widths 2 and 3 are
for experiments. The measurements are in
[KV cache fidelity](benchmarks.md#kv-cache-fidelity).

## Speed and speculative decoding

On hybrid models and gemma-4, the fp16, affine and kvarn caches run at
about the same speed. On a dense model whose decoding is limited by the KV
read, kvarn decodes slower than fp16 and affine 8, so choose kvarn for
memory and quality, and affine for the most speed.

A quantized cache lowers the share of accepted drafts in
[speculative decoding](speculative-decoding.md). Under affine, a
speculative model quantizes only while it serves one request. Under kvarn
it stays quantized at any batch size, and a batch drafts at most three
tokens a round. The width caps in
[Several requests at once](speculative-decoding.md#several-requests-at-once)
still apply.

## Credits

The kvarn scheme implements the method of
[arXiv:2606.03458](https://arxiv.org/abs/2606.03458) by Muller, Bich,
Boretti, Chang, Zhuang and Cavigelli, in the record format of
[beellama.cpp](https://github.com/Anbeeld/beellama.cpp). The
[third-party notices](../THIRD_PARTY_NOTICES.md) credit both.
