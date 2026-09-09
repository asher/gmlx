# Vision and audio input

This guide is for running a multimodal GGUF, a language model paired with a
vision or audio tower. It covers the file pairing, usage from the CLI and the
server, the supported families and the known defects in community files.

A multimodal model in GGUF is two files. The language model is quantized as
usual, and a companion `mmproj` GGUF holds the encoder and the projector.
Hugging Face repos ship the companion as an `mmproj-*.gguf` sibling of the
language model, and `gmlx validate` recognizes one and names the file it
pairs with. Vision and audio support is in the base install.

`--mmproj` pairs the two. The text tower runs on the K-quant kernels exactly
as in text-only mode, and so do the matmuls of a quantized encoder, while
float encoder weights stay native. The image processor and chat template are
built from the metadata of the two files, and `--hf-source` fills in only
what a file omits.

## Usage

```sh
# one-shot generation with an image file or URL
gmlx run model.gguf --mmproj mmproj.gguf --image photo.jpg --prompt "What is this?"

# interactive chat with /image, /audio, or a file dragged into the prompt
gmlx chat model.gguf --mmproj mmproj.gguf

# serve one model with its companion. A config pairs them with mmproj: per model
gmlx serve model.gguf --mmproj mmproj.gguf --port 8080
```

The flags are under [gmlx run](cli.md#gmlx-run), the request shape under
[Vision messages](api.md#vision-messages) and the per-model key under
[models](server-config.md#models).

Images encode at native resolution unless `--resize-shape` shrinks them
first. The size after resizing decides how many soft tokens an image expands
to, and those tokens dominate prefill cost, so a square cap such as `448` is
the usual choice when prefill time matters more than fine detail.

Audio works the same way on a companion that carries an audio encoder, as
with gemma-4 omni and Qwen3-Omni. `--audio` on `run` and `/audio` in chat
attach a clip to the next turn.

## Supported families

The companion's `clip.*` metadata names the projector. Where several
families share a projector, the language model's architecture tells them
apart, and an unsupported pairing fails at load with the projector and
architecture it found in the error.

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
| DeepSeek-V4-Flash-Vision-Exp | `deepseek4v` with `deepseek4` | the unsloth UD builds | differs in a few ways, described below |

Qwen2-VL and Qwen2.5-VL companions, projector `qwen2vl_merger`, are not
supported yet. That load fails immediately and the error names the family.
On LLaVA the loader reports two unfilled `post_layernorm` parameters, which
is expected: the conversion omits them and LLaVA never uses them.

DeepSeek-V4-Flash-Vision-Exp differs from the other families in three ways.
Image turns need an unquantized KV cache, so `--kv-bits` applies to text
turns only, and the server runs image turns one at a time. Each image
expands to a block of up to 384 tokens that prefills as one chunk, and the
prompt cache keys on those blocks, which means a conversation that repeats
its earlier image turns verbatim hits the cache. Its text output is also not
token-for-token comparable with the text-only release.

## Combining with other features

| Combination | Result |
|-------------|--------|
| `--stream-experts` | works with one request in flight. The text tower streams and the vision tower stays on the GPU |
| `--stream-cpu` | refused, because that placement would move the vision tower to the CPU too |
| `--speculative` | works with any drafter: a native head, a `--draft-gguf` companion or an autodetected one. Text turns speculate and media turns decode plain |
| `--adapter` | refused, because live LoRA is text-path only |

The bare language model GGUF still loads and runs as a plain text model
without its companion.

The two front ends place media differently. `chat` keeps each image on the
turn that sent it, so a later question about an earlier image is answered
against the right history, but once media enters a conversation every turn
re-prefills the whole transcript and re-encodes the media, because the
KV-cached fast path is text-only. The serve chat endpoint instead renders all
of a conversation's images on its last user message, so a follow-up after an
image turn misses the prompt cache from the point where the images moved.

## Known GGUF defects

Some community companion files are mis-converted upstream, independent of
this loader. You can recognize one because llama.cpp's multimodal CLI produces the same
degraded output from the same file, while the native weights of the same
checkpoint render correctly.

Pixtral companions carry corrupted vision attention q and k projections
from a RoPE layout mismatch in the conversion. There is no exact
loader-side inverse, so GGUF Pixtral vision quality is limited until a
re-converted companion appears. The text tower is unaffected.
