# Image generation

gmlx makes images from a text prompt, and edits images you give it, with
the Z-Image-Turbo, Qwen-Image 2.1, Krea 2 Turbo, FLUX.2 klein and FLUX.3
klein models. You can run them from the command line, call them from an
app through the OpenAI image API, or use them in [ComfyUI](comfyui.md).
Image generation came with gmlx 0.5 and needs a Mac with an M3 chip or
newer and 16 GB of memory or more.
[An M1 or M2](troubleshooting.md#needs-an-m3-or-newer) runs only
Z-Image-Turbo, in float16, when you allow it.

## Choose a model

Every model runs on 16 GB. `pull` downloads the quant that suits your
Mac's memory, so the download is larger on a Mac with more. The image size
is the one you get when you ask for none.

| Model | Download, 16 GB Mac | Download, 24 GB Mac | Image size | License |
|-------|---------------------|---------------------|------------|---------|
| `z-image-turbo` | 8.8 GB | 10.9 GB | 1024x1024 | Apache 2.0, any use |
| `qwen-image-turbo` | 14.4 GB | 17.5 GB | 1024x1024 | Research and evaluation only |
| `qwen-image` | 14.4 GB | 17.6 GB | 1024x1024 | Research and evaluation only |
| `krea-2-turbo` | 9.8 GB | 14.1 GB | 512x512 on 16 GB, else 1024x1024 | Free commercial use under $1M of yearly revenue |
| `flux-2-klein` | 6.6 GB | 8.1 GB | 1024x1024 | Apache 2.0, any use |
| `flux-2-klein-base` | 6.6 GB | 8.1 GB | 1024x1024 | Apache 2.0, any use |
| `flux-3-klein` | 7.3 GB | 9.8 GB | 1024x1024 | Non-commercial, and commercial use of the images under $5M of yearly revenue |
| `flux-3-klein-base` | 7.3 GB | 9.8 GB | 1024x1024 | Non-commercial, and commercial use of the images under $5M of yearly revenue |

Start with Z-Image-Turbo: it is the fastest, as
[Benchmarks](benchmarks.md#image-generation) shows, and its license allows
any use. [Models and quants](#models-and-quants) lists the
[steps](glossary.md#steps) each model runs and what it edits from.

## Make your first image

Create a configuration with [`gmlx init`](cli.md#gmlx-init) if you have
none, then download the model and make an image:

```sh
gmlx pull z-image-turbo
gmlx image generate z-image-turbo --prompt "a lighthouse at dusk" --seed 1 --output lighthouse.png
```

An image model is three files: the [DiT](glossary.md#dit) that makes the
image, the [text encoder](glossary.md#text-encoder) that reads the prompt,
and the [VAE](glossary.md#vae) that turns the result into pixels. `pull`
shows what it will download and the license, then downloads the three and
adds the model to your configuration. On a 16 GB Mac it prints:

```text
pull z-image-turbo (Z-Image-Turbo): DiT Q4_K_M, encoder Q8_0, the default for a 16 GB Mac
  dit      z-image-turbo-Q4_K_M.gguf                  4.7 GB  download
  encoder  Qwen3-4B-Q8_0.gguf                         4.0 GB  download
  vae      vae/diffusion_pytorch_model.safetensors    0.2 GB  download, packed to z-image-vae-BF16.gguf
  ...
  the three parts: 8.8 GB, which fit this Mac's 16 GB RAM
  license: Z-Image-Turbo apache-2.0 (https://huggingface.co/Tongyi-MAI/Z-Image-Turbo); encoder apache-2.0
...
registered z-image-turbo in /Users/you/.config/gmlx/gmlx.yaml (the image default)
try it: gmlx image generate z-image-turbo --prompt "a red bicycle" --output out.png
```

`generate` writes the image to the `--output` file. Without `--output` it
writes `gmlx-<time>-<seed>.png` in the current folder. The PNG keeps the
prompt, the seed and the settings it was made with. Each `generate` loads
the model before its first step, which takes a while on every run.

[`gmlx image generate`](cli.md#image-generate) has the flags for size,
steps, edits from an image and several images at once. When a command
stops with an error, [Troubleshooting](troubleshooting.md#image-generation)
has the fix.

- [Call it from an app](#call-it-from-an-app)
- [Use a LoRA](#use-a-lora)
- [ComfyUI](#comfyui)
- [Models and quants](#models-and-quants)
- [Licenses](#licenses)

## Call it from an app

`gmlx serve` answers the OpenAI image routes with the image models of
your configuration. Start it, and keep it running in its own terminal:

```sh
gmlx serve
```

A request names the model and the prompt. The answer holds the image as
base64, which this command decodes into a PNG:

```sh
curl -s localhost:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model": "z-image-turbo", "prompt": "a lighthouse at dusk"}' \
  | jq -r '.data[0].b64_json' | base64 -d > lighthouse.png
```

The `openai` Python package, which `pip install openai` installs, works the
same way. gmlx's own fields, such as `seed` and `steps`, go in `extra_body`:

```python
import base64

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
result = client.images.generate(model="z-image-turbo",
                                prompt="a lighthouse at dusk",
                                extra_body={"seed": 1, "steps": 8})
with open("lighthouse.png", "wb") as f:
    f.write(base64.b64decode(result.data[0].b64_json))
```

Leave out `size` to get the model's default size, which fits your Mac.
When the server has an API key, pass it as `api_key`, or to curl as
`-H "Authorization: Bearer KEY"`.

The model loads on its first request and shares the memory budget with
the chat models. When a chat model and an image model do not fit side by
side, the server unloads one to load the other, so the first request after
a switch waits for the load. [Image API](image-api.md) lists every field,
edits from your own images, streaming and the errors.

[`gmlx launch open-webui`](launch.md#open-webui) starts Open WebUI with its
image generation turned on, the image button in its chat box, using the
server's default image model.

## Use a LoRA

A LoRA is a small file that changes a model's style or teaches it a
subject. Hugging Face and Civitai host many. Take one made for the same
model you run, such as a Z-Image-Turbo LoRA for `z-image-turbo`, as a
`.safetensors` file:

```sh
gmlx image generate z-image-turbo --prompt "a lighthouse at dusk" \
    --adapter ~/models/loras/watercolor.safetensors:0.8 --output lighthouse.png
```

The number after the colon is the strength, 1 when you leave it out.
Repeat `--adapter` to stack LoRAs. The command applies both halves of a
LoRA that also trains the text encoder.

On the server, the LoRAs in the entry's
[`adapters`](config.md#image_modelsadapters) apply to every request, both
halves. A request adds its own in `adapters`. Each id is a path inside a
model folder, such as `loras/watercolor.safetensors` for
`~/models/loras/watercolor.safetensors`, or the file name of an entry
adapter, which sets that adapter's strength for the request:

```json
{"model": "z-image-turbo", "prompt": "a lighthouse at dusk",
 "adapters": [{"id": "loras/watercolor.safetensors", "strength": 0.8}]}
```

An absolute path gets 400, and so does a request LoRA with text encoder
layers. List that one in the entry's `adapters`, where both halves apply.
LoHa, LoKr, DoRA and the other kinds that are not plain LoRA pairs are
refused by name, as is a LoRA of another model. ComfyUI takes LoRAs as
[LoRAs](comfyui.md#loras) describes.

## ComfyUI

[ComfyUI](comfyui.md) builds image workflows from nodes in a browser. The
gmlx node pack runs these image models in it, on the gmlx server.

## Models and quants

| Name | Model | Steps | Edits | Text encoder |
|------|-------|-------|-------|--------------|
| `z-image-turbo` | Z-Image-Turbo | 8 | From one image | Qwen3-4B |
| `qwen-image` | Qwen-Image 2.1 | 40 | From up to 10 reference images | Qwen3-VL-8B-Instruct, with its mmproj |
| `qwen-image-turbo` | Qwen-Image 2.1 Turbo | 8 | From up to 10 reference images | Qwen3-VL-8B-Instruct, with its mmproj |
| `krea-2-turbo` | Krea 2 Turbo | 8 | From one image | Qwen3-VL-4B-Instruct |
| `flux-2-klein` | FLUX.2 klein 4B | 4, with guidance 1 only | From up to 10 reference images | Qwen3-4B |
| `flux-2-klein-base` | FLUX.2 klein Base 4B | 50 | From up to 10 reference images | Qwen3-4B |
| `flux-3-klein` | FLUX.3 klein | 16, with the `euler` sampler and guidance 1 only | From up to 10 reference images | Qwen3-VL-4B-Instruct |
| `flux-3-klein-base` | FLUX.3 klein Base | 50 | From up to 10 reference images | Qwen3-VL-4B-Instruct |

FLUX.3 klein is distilled to sample without guidance, and with `euler`
alone, so a request or a flag that sets another guidance or sampler is
refused. FLUX.3 klein Base takes both. FLUX.3 klein reads its reference
images without the encoder's vision tower, so it edits with the encoder
file alone. FLUX.3 klein and Krea 2
Turbo use the same Qwen3-VL-4B-Instruct file, and `pull` downloads it once
for both.

FLUX.2 klein is distilled to sample without guidance too, so a request or
a flag that sets another guidance is refused, and it samples with `euler`
or `uni_pc_bh2`. FLUX.2 klein Base takes guidance. FLUX.2 klein also
edits with the encoder file alone, since its encoder reads the prompt and
the VAE encodes the reference images. FLUX.2 klein and Z-Image-Turbo use
the same Qwen3-4B file, and FLUX.2 klein and FLUX.3 klein decode with the
same VAE, so `pull` downloads each once. Of FLUX.2 klein, gmlx runs the 4B
models, and it refuses a 9B GGUF by name.

`pull` takes the DiT at the quant for your Mac's memory, and the text
encoder at Q8_0. A Mac takes the column of the largest class its memory
holds. `--quant` and `--encoder-quant` pick others, and
[`gmlx pull`](cli.md#gmlx-pull) lists every flag.

| Model | 16 GB | 24 GB | 32 GB | 64 GB or more |
|-------|-------|-------|-------|---------------|
| `z-image-turbo` | Q4_K_M | Q8_0 | BF16 | BF16 |
| `qwen-image` | Q4_K_M | Q8_0 | Q8_0 | F16 |
| `qwen-image-turbo` | AD-Q4_K | Q8_0 | BF16 | BF16 |
| `krea-2-turbo` | Q3_K_M | Q6_K | Q8_0 | Q8_0 |
| `flux-2-klein` | Q4_K_M | Q8_0 | Q8_0 | Q8_0 |
| `flux-2-klein-base` | Q4_K_M | Q8_0 | Q8_0 | Q8_0 |
| `flux-3-klein` | Q4_K_M | Q8_0 | Q8_0 | Q8_0 |
| `flux-3-klein-base` | Q4_K_M | Q8_0 | Q8_0 | Q8_0 |

No Krea 2 quant fits a 1024x1024 image on 16 GB, so a pull of the 16 GB
default sets the entry's default size to 512x512, and a pull with another
`--quant` sets none. Of Krea 2, gmlx runs Turbo, and it refuses a Krea 2
Raw GGUF by name.

## Licenses

| Model | License | Use |
|-------|---------|-----|
| Z-Image-Turbo | [Apache 2.0](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | Any, under the license's terms |
| Qwen-Image 2.1 | [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE) | Research and evaluation only. Commercial use needs a license from Qwen. |
| Qwen-Image 2.1 Turbo | [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/LICENSE) | Research and evaluation only. Commercial use needs a license from Qwen. |
| Krea 2 Turbo | [Krea 2 Community License](https://cdn.jsdelivr.net/gh/krea-ai/krea-2@db3984fbc6e13b34c0064990fc2d95ac64d00058/assets/hf_samples/LICENSE.pdf) | Free commercial use for a company, affiliates counted, under $1M of revenue in the last twelve months. A deployment must filter what it makes. |
| FLUX.2 klein 4B, FLUX.2 klein Base 4B | [Apache 2.0](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/main/LICENSE.md) | Any, under the license's terms |
| FLUX.3 klein, FLUX.3 klein Base | [FLUX Kommunity License](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md) | Non-commercial, and commercial use of the images under $5M of yearly revenue, on the terms below |
| Qwen3-4B, Qwen3-VL-8B-Instruct, Qwen3-VL-4B-Instruct | Apache 2.0 | Any, under the license's terms |

Under the [FLUX Kommunity License](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md),
a user whose yearly revenue, affiliates counted, is under $5M may use
FLUX.3 klein images commercially with content filters or a review of each
image for unlawful or infringing content before it is shared, and with a
disclosure that the image is made with AI where the law requires one.
Other commercial use of FLUX.3 klein needs a license from Black Forest
Labs. gmlx has no content filter of its own, so a deployment that a
license asks to filter adds its own.

The GGUF files are quantized by unsloth, the Qwen-Image 2.1 Turbo files by
AtomicChat, the Krea 2 Turbo files by Abiray and the FLUX.3 klein files by
realrebelai, under the same licenses. The Krea 2 VAE comes from Comfy-Org's
Krea 2 repository. The FLUX.2 VAE, which FLUX.2 klein and FLUX.3 klein
decode with, comes from Black Forest Labs' FLUX.2 klein 4B repository,
under Apache 2.0.
