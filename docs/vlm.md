# Vision and audio input

This guide is for running a multimodal GGUF, a language model paired with a
vision or audio tower. It covers the file pairing, usage from the CLI and the
server, the supported families and the known defects in community files.

A multimodal model in GGUF is two files: the language model GGUF, quantized
as usual, and a companion `mmproj` GGUF holding the encoder and the
projector. Repositories ship the companion as an `mmproj-*.gguf` sibling in
the same Hugging Face repo, and `gmlx validate` recognizes one and names the
file to pair it with. Vision and audio support is in the base install.

gmlx pairs the two with `--mmproj`. The text tower runs on the K-quant
kernels exactly as in text-only mode, and a quantized encoder's matmuls run
on them too, while float weights stay native. The image processor and chat
template are synthesized from the two files' metadata, with `--hf-source`
overriding only when a file omits something.

## Usage

```sh
# one-shot generation with an image file or URL
gmlx run model.gguf --mmproj mmproj.gguf --image photo.jpg --prompt "What is this?"

# interactive multimodal chat with /image, /audio, or a file dragged into the prompt
gmlx chat model.gguf --mmproj mmproj.gguf

# serve it, as a single model or with mmproj: on each model in the config
gmlx serve model.gguf --mmproj mmproj.gguf --port 8080
```

`--resize-shape` resizes images before encoding, which sets the soft-token
count that dominates prefill cost. Unset, images encode at native
resolution, and a square cap such as `448` is a typical choice when prefill
cost matters.
Audio input, `--audio` on `run` and `/audio` in chat, works where the
companion carries an audio encoder, as with gemma-4 omni and Qwen3-Omni. The
flags are under [gmlx run](cli.md#gmlx-run), the request shape under [Vision
messages](api.md#vision-messages) and the model key under
[models](server-config.md#models).

## Supported families

The companion's `clip.*` metadata names the projector, and where families
share one, the language model's architecture disambiguates. An unsupported
pairing fails at load with both names.

| Family | Projector and arch | Examples | Notes |
|--------|--------------------|----------|-------|
| LLaVA-1.5 | `has_llava_projector` | llava-1.5-7B | pass `--hf-source llava-hf/llava-1.5-7b-hf`, because the image processor is not in the GGUF |
| Pixtral | `pixtral` | Mistral-Small-3.x, Pixtral-12B | vision quality limited by a conversion defect, described under Known GGUF defects |
| Qwen3.5 and 3.6 | `qwen3vl_merger` with `qwen35` or `qwen35moe` | Qwen3.5-VL-9B, Qwen3.6-VL | |
| Qwen3-Omni | `qwen3vl_merger` with `qwen3vlmoe` | Qwen3-Omni | vision and audio, experimental. Text on the thinker tower is reliable |
| gemma-4 omni | `gemma4v`, `gemma4a` | gemma-4-E2B, E4B | vision and audio |
| gemma-4 unified | `gemma4uv` | gemma-4-12B | encoder-free unified embedder |
| Muse Glimmer | `muse-glimmer` | Muse-Glimmer-30B | vision tower and processor implemented in gmlx, so no `--hf-source` is needed |
| Kimi K2.5 and K2.7 | `kimik25` with `deepseek2` | Kimi-K2.5, Kimi-K2.7-Code | over-RAM MoE, needs `--stream-experts` |
| DeepSeek-V4-Flash-Vision-Exp | `deepseek4v` with `deepseek4` | the unsloth UD builds | has extra properties, listed after the table |

Qwen2-VL and Qwen2.5-VL companions, projector `qwen2vl_merger`, are not
supported yet, so the load fails immediately with the family named. On
LLaVA the loader reports two unfilled `post_layernorm` parameters, which is
expected because the conversion omits them and LLaVA never uses them.

DeepSeek-V4-Flash-Vision-Exp has a few extra properties. Image turns run a
single request at a time and need an unquantized KV cache, which limits
`--kv-bits` to text turns. Each image expands to a block of up to 384
tokens that prefills in a single chunk, and the prompt cache keys on those
blocks, so a conversation that repeats its earlier image turns verbatim
hits the cache. Its text output is not token-for-token comparable with the
text-only release.

## Combining with other features

| Combination | Result |
|-------------|--------|
| `--stream-experts` | works. The text tower streams and the vision tower stays on the GPU. Send a single request at a time |
| `--stream-cpu` | refused, because that placement would move the vision tower to the CPU too |
| `--speculative` | works when a drafter is available, whether a native head, a `--draft-gguf` companion, or an autodetected one. Text turns speculate and media turns decode plain |
| `--adapter` | refused, because live LoRA is text-path only |

The bare language model GGUF still loads and runs as a plain text model
without its companion. Because the serve chat endpoint renders all images
of a conversation on its last user message, a follow-up turn after an image
turn re-prefills from the moved block, whereas `chat` pins each image to
the turn that sent it and keeps the prefix.

## Known GGUF defects

Some community companion files are mis-converted upstream, independent of
this loader. The sign is that llama.cpp's multimodal CLI produces the same
degraded output from the same file, while the native weights of the same
checkpoint render correctly.

Pixtral companions carry corrupted vision attention q and k projections
from a RoPE layout mismatch in the conversion. There is no exact
loader-side inverse, so GGUF Pixtral vision quality is limited until a
re-converted companion appears, although the text tower is unaffected.
