# KV cache quantization

Quantizing the KV cache stores the context of a request in fewer bits, so
a long context uses less memory. gmlx offers two schemes, affine and
kvarn, for the attention layers whose cache grows with the context. Both
cost a little quality, and on some models a little speed.

- [The two schemes](#the-two-schemes)
- [Which layers quantize](#which-layers-quantize)
- [Choosing a scheme by model](#choosing-a-scheme-by-model)
- [Quality](#quality)
- [Speed and speculative decoding](#speed-and-speculative-decoding)

## The two schemes

Affine quantization, with `--kv-bits N` or the
[`kv_bits`](config.md#loadkv_bits) load key, is the quantized KV cache of
mlx-lm. It splits the K and V rows of each token into groups of
`--kv-group-size` values, 64 by default, and stores N-bit codes with an
fp16 scale and offset for each group. The widths are 2, 3, 4, 6 and 8, and
`--quantized-kv-start` keeps the whole cache in fp16 until the context
reaches that many tokens.
`--kv-bits 8` about halves the cache with almost no loss.

The kvarn scheme, with `--kv-quant-scheme kvarn` or
[`kv_quant_scheme: kvarn`](config.md#loadkv_quant_scheme), normalizes the
variance of the values before it rounds them. `--kv-bits` sets the width,
6 by default, from 2, 3, 4, 5, 6 and 8, and
[`GMLX_KVARN_BITS`](env-vars.md#runtime) sets different widths for keys
and values. The scheme rotates each head with a Hadamard transform, which
spreads out the few channels with large values, and stores K and V in
records of 128 tokens that it scales so that no token or channel dominates.
The first 128 tokens stay fp16, as do the newest
[`--kv-tail-tokens`](config.md#loadkv_tail_tokens) tokens, 1024 by default.
At 6 bits, a record takes about 40% of the memory of fp16.

The kvarn scheme implements the method of Muller, Bich, Boretti, Chang,
Zhuang and Cavigelli, [arXiv:2606.03458](https://arxiv.org/abs/2606.03458).
The gmlx cache
follows the record format of
[beellama.cpp](https://github.com/Anbeeld/beellama.cpp), including the
fp16 tail and `--kv-tail-tokens`, and the
[third-party notices](../THIRD_PARTY_NOTICES.md) credit both.

The same policy decides both schemes layer by layer. It prints a `[kv]`
line at load, and `GET /v1/models` reports the result for each loaded
model as `kv_quant`. A model on which no layer can use kvarn runs fp16 and
prints why, and it never falls back to affine.

## Which layers quantize

The shape of each layer's cache decides, not the name of the model.
Attention layers whose cache grows with the context quantize, except the
last layer of a deep stack, which stays fp16 under both schemes. Recurrent
state and sliding windows stay fp16, apart from the rolling
`--max-kv-size` window of `run` and `chat` that
[Settings that limit memory](memory.md#settings-that-limit-memory)
describes.

Head dimensions of 128, 256 and 512 are the only ones that kvarn accepts,
so layers with a head dimension of 64, as in gpt-oss, use affine
quantization only. The scheme also declines [MLA](glossary.md#mla)
models. Affine still packs the pooled cache of DeepSeek-V4 and GLM-5.3, and
Kimi K2 keeps an fp16 cache under either scheme. With `--mmproj`, `run`
and `chat` do not apply kvarn, and the cache stays fp16.

## Choosing a scheme by model

Quantization saves memory in proportion to how much of the cache grows
with the context, and costs quality in proportion to how many layers it
touches. The shape of the cache decides both, as the table shows.

| Cache shape | Families | Cache at 32K in fp16 | What to use |
|---|---|---|---|
| Full attention on all layers | Llama, Mistral, dense Qwen3 | 4 to 8 GB for an 8B to 32B model. | Use `--kv-bits 8`, or kvarn at 6 for the same quality in less memory, and kvarn at 4 when memory is the limit. |
| Recurrent hybrid, one attention layer in four | Qwen3.5, Qwen3.6, Qwen3.8 | About 2 GB at 27B, plus a fixed recurrent state. | Quantize only when the context is the limit, at 64K and up. The quality cost is small, since three layers in four never quantize. |
| Sliding-window mix | gemma-4 | The window layers stop growing at the window. | Either scheme gives a small saving, since only the global layers quantize. |
| MLA latent | DeepSeek-V4, GLM-5.3, Kimi K2 and K3 | The architecture already compresses it. | Use affine on DeepSeek-V4 and GLM-5.3, which pool their cache. Kimi K2 keeps an fp16 cache under either scheme. |
| Head dimension 64 | gpt-oss | Each token adds little cache. | Use affine, since kvarn needs a head dimension of 128, 256 or 512. |

## Quality

The kvarn scheme keeps the output closer to that of an fp16 cache than
affine quantization does at every width below 8, by a factor of 3 to 5 at
2 to 4 bits. The two converge at 8 bits. At 6 bits, kvarn matches or nearly
matches affine at 8 in three quarters of the memory. Widths 2 and 3 are
for experiments. The measurements are in
[KV cache fidelity](benchmarks.md#kv-cache-fidelity).

## Speed and speculative decoding

The effect on speed depends on how much of a decode step reads the KV
cache. On hybrid models and gemma-4, fp16, affine and kvarn run at about
the same speed. On a dense model whose decoding is limited by the KV read,
kvarn decodes slower than fp16 and affine 8, so choose kvarn for memory
and quality, and affine for the most speed on such a model.

A quantized cache lowers the share of accepted drafts in
[speculative decoding](speculative-decoding.md). Under affine
quantization, a speculative model quantizes only while it serves one
request. Under kvarn, it stays quantized at any batch size, and a batch
verifies at most four tokens a row. A drafter with a wider block then
drafts three tokens a round, and keeps that limit while the batch lives.
[`speculative_width_cap`](config.md#modelsspeculative_width_cap) lists the
drafters that stop speculating in any batch.
