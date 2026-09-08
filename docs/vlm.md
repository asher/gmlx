# Vision and audio input

This guide is for running a multimodal GGUF: a language model paired with a
vision or audio tower. It covers the file pairing, usage from the CLI and the
server, the supported families, and the known defects in community files.

A multimodal model in GGUF is two files: the language model GGUF, quantized
as usual, and a companion `mmproj` GGUF holding the encoder and the projector.
Repositories ship the companion as an `mmproj-*.gguf` sibling in the same
Hugging Face repo, and `gmlx validate` recognizes one and names the file to
pair it with. Vision and audio support is in the base install.

gmlx pairs the two with `--mmproj`. The text tower runs on the K-quant
kernels exactly as in text-only mode, a quantized encoder's matmuls run on
them too while float weights stay native, and the image processor and chat
template are synthesized from the two files' metadata. `--hf-source`
overrides only when a file omits something.

## Usage

```sh
# one-shot generation with an image (or a URL)
gmlx run model.gguf --mmproj mmproj.gguf --image photo.jpg --prompt "What is this?"

# interactive multimodal chat: /image, /audio, or drag a file into the prompt
gmlx chat model.gguf --mmproj mmproj.gguf

# serve it (single model, or `mmproj:` per model in the config)
gmlx serve model.gguf --mmproj mmproj.gguf --port 8080
```

`--resize-shape` resizes images before encoding and so sets the soft-token
count that dominates prefill cost. Unset, images encode at native resolution;
a square cap such as `448` is a typical choice when prefill cost matters.
Audio input, `--audio` on `run` and `/audio` in chat, works where the
companion carries an audio encoder (gemma-4 omni, Qwen3-Omni). The flags are
under [gmlx run](cli.md#gmlx-run), the request shape under
[Vision messages](api.md#vision-messages), and the per-model key under
[models](server-config.md#models).

## Supported families

The companion's `clip.*` metadata names the projector, and the language
model's architecture disambiguates families that share one. An unsupported
pairing fails at load with both names.

| Family | Projector and arch | Examples | Notes |
|--------|--------------------|----------|-------|
| LLaVA-1.5 | `has_llava_projector` | llava-1.5-7B | pass `--hf-source llava-hf/llava-1.5-7b-hf`; the image processor is not in the GGUF |
| Pixtral | `pixtral` | Mistral-Small-3.x, Pixtral-12B | vision quality capped by a conversion defect, see below |
| Qwen3.5 and 3.6 | `qwen3vl_merger` with `qwen35` or `qwen35moe` | Qwen3.5-VL-9B, Qwen3.6-VL | |
| Qwen3-Omni | `qwen3vl_merger` with `qwen3vlmoe` | Qwen3-Omni | vision and audio; treat as experimental, text on the thinker tower is solid |
| gemma-4 omni | `gemma4v`, `gemma4a` | gemma-4-E2B, E4B | vision and audio |
| gemma-4 unified | `gemma4uv` | gemma-4-12B | encoder-free unified embedder |
| Muse Glimmer | `muse-glimmer` | Muse-Glimmer-30B | vision tower and processor implemented in gmlx; no `--hf-source` needed |
| Kimi K2.5 and K2.7 | `kimik25` with `deepseek2` | Kimi-K2.5, Kimi-K2.7-Code | over-RAM MoE, needs `--stream-experts` |
| DeepSeek-V4-Flash-Vision-Exp | `deepseek4v` with `deepseek4` | the unsloth UD builds | see the notes below |

Qwen2-VL and Qwen2.5-VL companions (`qwen2vl_merger`) are not supported yet;
the load fails up front with the family named. On LLaVA the loader reports
two unfilled `post_layernorm` parameters, which is expected: the conversion
omits them and LLaVA never uses them.

DeepSeek-V4-Flash-Vision-Exp has a few properties of its own. Image turns run
one request at a time and need an unquantized KV cache, so `--kv-bits` applies
to text turns only. Each image expands to a block of up to 384 tokens that
prefills in one chunk, and the prompt cache keys on those blocks, so a
conversation that repeats its earlier image turns verbatim hits the cache.
Its text output is not token-for-token comparable with the text-only release.

## Combining with other features

| Combination | Result |
|-------------|--------|
| `--stream-experts` | works; the text tower streams and the vision tower stays on the GPU. Send one request at a time |
| `--stream-cpu` | refused; that placement would move the vision tower to the CPU too |
| `--speculative` | works when a drafter is available: a native head, a `--draft-gguf` companion, or an autodetected one. Text turns speculate; media turns decode plain |
| `--adapter` | refused; live LoRA is text-path only |

The bare language model GGUF still loads and runs as a plain text model
without its companion. The serve chat endpoint renders every image of a
conversation on its last user message, so a follow-up turn after an image
turn re-prefills from the moved block; `chat` pins each image to its own turn
and keeps the prefix.

## Known GGUF defects

Some community companion files are mis-converted upstream, independent of
this loader. The tell is that llama.cpp's own multimodal CLI produces the same
degraded output from the same file while the native weights of the same
checkpoint render correctly.

Pixtral companions carry mangled vision attention q and k projections from a
RoPE layout mismatch in the conversion. There is no clean loader-side inverse,
so GGUF Pixtral vision quality is capped until a re-converted companion
appears. The text tower is unaffected.
