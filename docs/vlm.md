# Vision and audio

A multimodal GGUF model reads images or audio as well as text. It comes as
two files that gmlx pairs, and it takes media from the command line, from
chat and from the server. Most features of the language model still work
when a turn carries media.

## The two files

A multimodal model in GGUF is two files. The language model GGUF is
quantized as usual, and a companion `mmproj` GGUF holds the encoder that
turns images or audio into tokens for the language model. Hugging Face
repositories ship the companion as an `mmproj-*.gguf` file beside the
language model. Download both in one command, where the second name is a
file in the same repository:

```sh
gmlx pull hf:unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K.gguf mmproj-BF16.gguf
```

When both files land in a folder of
[`server.model_dirs`](config.md#servermodel_dirs), `pull` registers the
model with its companion and prints `(vlm)` after the id. For a model that
is already registered, add the [`models.*.mmproj`](config.md#modelsmmproj)
key to its entry. [`discover[].pair_mmproj`](config.md#discoverpair_mmproj)
pairs the files that a folder scan finds.

`--mmproj` pairs the two files for one command. `gmlx validate` recognizes
a companion and says to pair it with its language model. The language
model alone still runs as a text model, and media support needs no extra.

gmlx builds the image processor and the chat template from the metadata
of the two files. For a family whose files leave something out,
`--hf-source` names a Hugging Face repository, and the whole processor,
including the tokenizer and chat template, then comes from that
repository.

## Sending media

```sh
# One reply about an image file or URL.
gmlx run model.gguf --mmproj mmproj.gguf --image photo.jpg --prompt "What is this?"

# A chat, with /image, /audio, or a file dragged into the prompt.
gmlx chat model.gguf --mmproj mmproj.gguf

# A server for one model and its companion.
gmlx serve model.gguf --mmproj mmproj.gguf
```

`run` takes local files and URLs, and chat takes local files. Image files
can be PNG, JPEG, GIF, WebP, BMP, TIFF or HEIC, and audio files can be WAV,
MP3, FLAC, M4A, Ogg, Opus or AIFF. A request to the server carries images
as [Vision messages](api.md#vision-messages) describes. The flags are
listed under [gmlx run](cli.md#gmlx-run).

Each family resizes images to its own limits before encoding. The size
sets how many tokens an image becomes, and those tokens make up most of
the time to read a prompt with images. `--resize-shape`, such as `448` or
`672x448`, shrinks images first, which helps when speed matters more than
fine detail.

Audio works in the same way on a companion with an audio encoder, such as
gemma-4 E-series and Qwen3-Omni. `--audio` on `run` and `/audio` in chat
attach a clip to the next turn.

## Supported families

The metadata of the companion names its projector. When several families
share a projector, the architecture of the language model tells them
apart. A pairing that gmlx does not support fails at load, and the error
names the projector and architecture that it found.

| Family | Projector and architecture | Examples | Notes |
|--------|----------------------------|----------|-------|
| LLaVA-1.5 | `has_llava_projector` | llava-1.5-7B | Needs `--hf-source llava-hf/llava-1.5-7b-hf`, because the GGUF has no image processor. |
| Pixtral | `pixtral` | Mistral-Small-3.x, Pixtral-12B | Vision quality is poor, because the published companion files were converted with a defect. |
| Qwen3.5 and Qwen3.6 | `qwen3vl_merger` with `qwen35` or `qwen35moe` | Qwen3.5-VL-9B, Qwen3.6-VL | |
| Qwen3.8-Flash-Next | `qwen3vl_merger` with `qwen4exp` | Qwen3.8-Flash-Next | |
| Qwen3-Omni | `qwen3vl_merger` with `qwen3vlmoe` | Qwen3-Omni | Vision and audio. |
| gemma-4 E-series | `gemma4v`, `gemma4a` | gemma-4-E2B, gemma-4-E4B | Vision and audio. |
| gemma-4 unified | `gemma4uv`, `gemma4ua` | gemma-4-12B | Vision, and audio when the companion has it. |
| GLM-5.3-Flash | `glm5next` | GLM-5.3-Flash | |
| Muse Glimmer | `muse-glimmer` | Muse-Glimmer-30B | |
| Kimi K2.5 and K2.7 | `kimik25` with `deepseek2` | Kimi-K2.5, Kimi-K2.7-Code | Larger than RAM on most Macs, so it needs `--stream-experts`. |
| DeepSeek-V4-Flash-Vision-Exp | `deepseek4v` with `deepseek4` | The unsloth UD builds | See the notes below the table. |
| DeepSeek-V4.1-Flash-Vision | `deepseek4-vision` with `deepseek41` | The antirez encoder GGUF | |

Qwen2-VL and Qwen2.5-VL companions, with the projector `qwen2vl_merger`,
are not supported, and the load fails with a message that names the
family. On LLaVA, the loader reports two unfilled `post_layernorm`
parameters, which is expected, because LLaVA does not use them.

On DeepSeek-V4-Flash-Vision-Exp, image turns need a KV cache without
quantization, so `--kv-bits` applies to text turns only, and the server
runs image turns one at a time. Each image becomes a block of up to 384
tokens. The prompt cache recognizes these blocks, so a conversation that
repeats its earlier image turns unchanged reuses the cache. On V4.1, each
image becomes up to 1024 tokens, and image turns have no such limits.

## Media with other features

Most features of the language model work with media. These combinations
have limits:

| Combination | Result |
|-------------|--------|
| [Speculative decoding](speculative-decoding.md) | Text turns use speculation, and turns with media decode without it. It needs a drafter, which is a native head or a `--draft-gguf` companion. |
| [`--stream-experts`](streaming.md) | `run` and `serve` stream the language model and keep the vision encoder on the GPU. Chat refuses it, and a served model cannot combine it with speculation. |
| `--stream-cpu` | Refused, because it would move the vision encoder to the CPU too. |
| The lossy MoE settings, such as `--moe-experts` | Chat refuses them with `--mmproj`. |
| `--adapter` | Refused, because a LoRA adapter applies only to text models. |
| `--stop`, `--xtc-probability`, `--xtc-threshold` | `run` ignores them with a warning. |

Chat and the server place media differently in a conversation. Chat keeps
each image with the turn that sent it, so a later question about an
earlier image reads the right history. Once a conversation holds media,
though, chat reads the whole conversation and encodes the media again on
every turn. The server puts all the images of a conversation on its last
user message, so a follow-up question after an image misses the prompt
cache from the point where the images moved.
