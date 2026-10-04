# KV cache quantization

Quantizing the KV cache stores a request's context in fewer bits, so a
long context uses less memory. gmlx offers two schemes, affine and kvarn.
Both cost a little quality, and on some models a little speed.

```sh
gmlx run model.gguf --kv-bits 8                    # affine, about half the memory
gmlx run model.gguf --kv-quant-scheme kvarn        # kvarn at 6 bits, the same quality in less memory
```

On the server, set the same choice with the [load keys](config.md#model-loading):

```yaml
models:
  qwen3.8-27b-ud-q6:
    path: Qwen3.8-27B-UD-Q6_K.gguf
    overrides: {load: {kv_quant_scheme: kvarn, kv_bits: 6}}
```

## Choosing a scheme by model

How much a scheme saves depends on how much of the cache grows with the
context, and that depends on the model:

| Cache shape | Families | Cache at 32K in fp16 | What to use |
|---|---|---|---|
| Full attention on all layers | Llama, Mistral, dense Qwen3 | 4 to 8 GB for an 8B to 32B model | `--kv-bits 8`, or kvarn at 6 for the same quality in less memory, or kvarn at 4 when memory is the limit. |
| Recurrent hybrid, one attention layer in four | Qwen3.5, Qwen3.6, Qwen3.8 | About 2 GB at 27B, plus a fixed recurrent state | Quantize only when the context is the limit, at 64K and up. The quality cost is small. |
| Sliding-window mix | gemma-4 | The window layers stop growing at the window. | Either scheme. Only the global layers quantize, so the saving is small. |
| MLA latent | DeepSeek-V4, GLM-5.3, Kimi K2 and K3 | Already compressed by the architecture | Affine on DeepSeek-V4 and GLM-5.3. Kimi K2 and K3 keep an fp16 cache under either scheme. |
| Head dimension 64 | gpt-oss | Each token adds little cache. | Affine. kvarn needs a head dimension of 128, 256 or 512. |

## Which layers quantize

The load prints a `[kv]` line that says which layers quantize and why, and
`GET /v1/models` reports the result for each loaded model as `kv_quant`.
Recurrent state, sliding windows and the last layer of a deep stack stay
fp16. A model on which no layer can use kvarn runs fp16 and says why. It
never falls back to affine. With `--mmproj`, `run` and `chat` do not apply
kvarn.

## The two schemes

| Name | Default | Meaning |
|------|---------|---------|
| `--kv-bits N` | off, or 6 under kvarn | Width. Affine takes 2, 3, 4, 6 or 8. kvarn takes 2, 3, 4, 5, 6 or 8. |
| `--kv-quant-scheme` | `uniform` | `uniform` is affine. `kvarn` normalizes the values before rounding them. |
| `--kv-group-size` | `64` | Affine group size, 32, 64 or 128. Each group stores an fp16 scale and offset. |
| `--quantized-kv-start N` | `0` | Affine only. Keeps the cache fp16 until it holds N tokens. |
| `--kv-tail-tokens N` | `1024` | kvarn only. The newest N tokens stay fp16. A multiple of 128, or 0. |
| [`GMLX_KVARN_BITS`](env-vars.md#runtime) | none | kvarn only. Separate widths for keys and values, such as `k6v5`. |

kvarn rotates each head with a Hadamard transform, which spreads out the
few channels with large values, and stores K and V in scaled records of
128 tokens. The first 128 tokens stay fp16, as does the tail. At 6 bits, a
record takes about 40% of the memory of fp16.

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
