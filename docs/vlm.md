# Multimodal (vision-language) GGUF support

A multimodal model in GGUF is two files:

1. the LLM GGUF, a normal text architecture, K-quantized as usual; and
2. a separate `mmproj` GGUF (`general.architecture = "clip"`) holding the
   vision (and/or audio) encoder plus the cross-modal projector. It ships float
   or Q8_0. A quantized mmproj's encoder matmuls run on the K-quant kernels like
   the text tower, while float weights stay native.

`gmlx` pairs them with `--mmproj` and loads an [mlx-vlm](https://github.com/Blaizzy/mlx-vlm)
`Model`: the text tower runs on the K-quant kernels exactly as in text-only mode, the
encoders run in float, and the image processor + chat template (including the
per-family image/audio marker tokens) are synthesized from the two GGUFs' metadata.
`--hf-source` overrides only when a file omits something.

Vision and audio support is included in the base install, no extra needed.

Where mmproj files come from: llama.cpp-style multimodal GGUF repos ship them
as `mmproj-*.gguf` siblings of the LLM GGUF in the same Hugging Face repo.
`gmlx validate` recognizes one as a companion and tells you to pair it with
its LLM GGUF via `--mmproj`.

## Usage

```sh
# one-shot generation with an image (or a URL)
gmlx run model.gguf --mmproj mmproj.gguf --image photo.jpg --prompt "What is this?"

# interactive multimodal chat: /image, /audio, or just drag a file into the prompt
gmlx chat model.gguf --mmproj mmproj.gguf

# serve it (single model, or `mmproj:` per model in the YAML config)
gmlx serve model.gguf --mmproj mmproj.gguf --port 8080
```

`--resize-shape N|WxH` resizes images before encoding, setting the soft-token count
that dominates prefill cost. Unset, images encode at native resolution. A
square cap like `448` (or an explicit `672x448`) is a typical choice when prefill
cost matters. Audio input (`--audio`, `/audio`) works where the
mmproj carries an audio encoder (gemma-4 omni, Qwen3-Omni). Vision-only mmprojs
reject it. Full flag reference: [docs/cli.md](cli.md); server config:
[docs/server-config.md](server-config.md).

## Supported families

The mmproj's `clip.*` metadata names the projector; the LLM arch disambiguates
families that share one. An unsupported pairing fails loudly at load with both names.

| Family | Projector / arch | Examples |
|--------|------------------|----------|
| LLaVA-1.5 | `has_llava_projector` | llava-1.5-7B (processor needs `--hf-source`, see below) |
| Pixtral | `pixtral` | Mistral-Small-3.x, Pixtral-12B (see defect note below) |
| Qwen3.5 / 3.6 | `qwen3vl_merger` + `qwen35`/`qwen35moe` | Qwen3.5-VL-9B, Qwen3.6-VL (dense + MoE) |
| Qwen3-Omni | `qwen3vl_merger` + `qwen3vlmoe` | Qwen3-Omni (vision + audio) |
| gemma-4 omni | `gemma4v`/`gemma4a` | gemma-4-E2B / E4B (vision + audio) |
| gemma-4 unified | `gemma4uv` | gemma-4-12B (encoder-free unified embedder) |
| Muse Glimmer | `muse-glimmer` + `muse-glimmer` | Muse-Glimmer-30B |
| Kimi K2.5 / K2.7 | `kimik25` + `deepseek2` | Kimi-K2.5, Kimi-K2.7-Code |

Muse Glimmer's vision tower and image processor are implemented in gmlx. Neither
mlx-vlm nor the installed transformers ships the family, so the preprocessing
ports llama.cpp's `mtmd_image_preprocessor_muse_glimmer`. `--hf-source` is not
needed.

Kimi K2.x pairs its MoonViT mmproj with the `deepseek2` text tower. The
converter rewrites the vision Q/K into the split 2-D RoPE layout llama.cpp's
`build_rope_2d` consumes; the remap puts them back into the interleaved layout
MoonViT itself uses, so a mis-decoded mmproj shows up as confidently wrong image
descriptions rather than as a load error. GLM-5.2-V ships the same vision
encoder under a different text arch and is refused by name. Because K2.x is an
over-RAM MoE, `--stream-experts` composes with `--mmproj` here: placement runs
on the text tower after the load and the vision tower stays resident.

Qwen2-VL / Qwen2.5-VL mmprojs (`qwen2vl_merger`) are not supported yet. The
load fails up front with the family named. LLaVA's image processor isn't
synthesized from the GGUF. Pass the checkpoint's HF id (e.g.
`--hf-source llava-hf/llava-1.5-7b-hf`) so the processor loads from there. On
LLaVA the loader reports two unfilled parameters
(`vision_tower.[...].post_layernorm.{weight,bias}`). This is expected: llama.cpp's
mmproj conversion omits CLIP's `post_ln`, and LLaVA never uses it. Features
come from the raw penultimate layer (`vision_feature_layer = -2`), while
`post_layernorm` only touches the final pooled output.

## Known upstream conversion defects

Some community mmproj GGUFs are mis-converted upstream (in llama.cpp's
`convert_hf_to_gguf.py --mmproj`), independent of this loader. The tell is that
both this loader and llama.cpp's own `llama-mtmd-cli` produce degraded vision
output from the same file, while the native (HF- or MLX-converted) weights of
the same checkpoint render correctly, so the defect lives in the GGUF, not the
consumer.

- Pixtral (`projector_type = pixtral`): the vision attention `q`/`k`
  projection weights are mangled by a RoPE-permutation mismatch in the mmproj
  conversion. Pixtral's ViT uses 2-D RoPE, and the conversion's q/k layout doesn't
  match it. The corruption is isolated to `v.blk.N.attn_q` / `attn_k` across
  every block. `attn_v` / `attn_out`, the FFN, every norm, the patch conv, and
  the projector are faithful at cosine >= 0.99 vs the native weights, which is
  how the defect was localized. The standard 1-D RoPE un-permute only partially
  realigns q/k, so there is no clean loader-side inverse, and llama.cpp's `mtmd`
  shows the same degradation on the same file. This is separate from the
  already-merged `image_std` mean/std fix (llama.cpp #13208). Current community
  mmproj files carry that fix. GGUF Pixtral vision quality is capped until a
  re-converted mmproj appears. The text tower is unaffected.

## Caveats

- A multimodal request needs the mmproj at load. The bare LLM GGUF still loads and
  runs as a plain text model (the vision side is simply absent).
- Adapters (`--adapter`) don't combine with `--mmproj` yet -- live GGUF LoRA is
  text-path-only and errors loudly.
- Speculative decoding (`--speculative`) *does* combine with `--mmproj` when a drafter
  is available -- a native MTP head (e.g. Qwen3.5/3.6) or `--draft-gguf`. Text-only
  turns speculate; media turns fall back to plain decode. With `--mmproj` but no
  drafter source, `--speculative` still errors loudly (it suggests `--draft-gguf`).
- Qwen3-Omni multimodal generation rides mlx-vlm's `qwen3_omni_moe` path, which we
  have found unreliable in stock mlx-vlm. Treat vision/audio input on Omni as
  experimental. Text generation on the Omni thinker tower is solid.
