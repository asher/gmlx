# Image generation

gmlx makes images from a text prompt, and edits images you give it, with
the Z-Image-Turbo, Qwen-Image 2.1, Krea 2 Turbo, MOSS-Image, FLUX.2 klein
and FLUX.3 klein models. You can run them from the command line, call them from an
app through the OpenAI image API, or use them in [ComfyUI](comfyui.md).
Image generation came with gmlx 0.5 and needs a Mac with an M3 chip or
newer and 16 GB of memory or more.
[An M1 or M2](troubleshooting.md#needs-an-m3-or-newer) runs only
Z-Image-Turbo, in float16, when you allow it.

## Choose a model

Every model runs on 16 GB but the FLUX.2 klein 9B models, which need
24 GB. `pull` downloads the quant that suits your Mac's memory, so
the download is larger on a Mac with more. The image size is the one you
get when you ask for none.

| Model | Download, 16 GB Mac | Download, 24 GB Mac | Image size | License |
|-------|---------------------|---------------------|------------|---------|
| `z-image-turbo` | 8.8 GB | 10.9 GB | 1024x1024 | Apache 2.0, any use |
| `qwen-image-turbo` | 14.4 GB | 17.5 GB | 1024x1024 | Research and evaluation only |
| `qwen-image` | 14.4 GB | 17.6 GB | 1024x1024 | Research and evaluation only |
| `krea-2-turbo` | 9.8 GB | 14.1 GB | 512x512 on 16 GB, else 1024x1024 | Free commercial use under $1M of yearly revenue |
| `moss-image` | 10.5 GB | 15.5 GB | 1024x1024 | Apache 2.0, any use |
| `moss-image-base` | 10.5 GB | 15.5 GB | 1024x1024 | Apache 2.0, any use |
| `flux-2-klein` | 6.6 GB | 8.1 GB | 1024x1024 | Apache 2.0, any use |
| `flux-2-klein-base` | 6.6 GB | 8.1 GB | 1024x1024 | Apache 2.0, any use |
| `flux-2-klein-9b` | Needs 24 GB | 13.8 GB | 1024x1024 | Non-commercial only |
| `flux-2-klein-base-9b` | Needs 24 GB | 13.8 GB | 1024x1024 | Non-commercial only |
| `flux-2-klein-9b-kv` | Needs 24 GB | 13.6 GB | 1024x1024 | Non-commercial only |
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
- [Prompt upsampling](#prompt-upsampling)
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

## Prompt upsampling

Prompt upsampling turns your prompt into a long, detailed caption before
the text encoder reads it: a chat model of your choice rewrites it with a
system prompt written for that image model, and the image follows the
caption. FLUX.3 klein and FLUX.3 klein Base were trained on such captions,
not on the short prompts people type, and a prompt sent as typed gives
weaker images. On the other families the rewrite is optional, and helps
most with short prompts and with text in the image.

| Family | System prompt |
|---|---|
| FLUX.3 klein, FLUX.3 klein Base | Ships with gmlx, from Black Forest Labs |
| FLUX.2 klein, all sizes and variants | Ships with gmlx, from Black Forest Labs |
| Krea 2 | Ships with gmlx, from Krea |
| Z-Image, Z-Image-Turbo | Ships with gmlx, from the Z-Image-Turbo demo of Tongyi-MAI |
| Qwen-Image 2.1, Qwen-Image 2.1 Turbo | Your own file, as [Qwen-Image 2.1](#qwen-image-21-rewriter-prompt) describes |

Qwen3.5-9B writes captions close to those of BFL's own FLUX.3 upsampler,
and its Q6_K GGUF is a good choice for every family. Any chat model of your
configuration can write them. Add it under `models`, and name it as the
`upsampler` of the image model's entry:

```yaml
models:
  qwen3.5-9b:
    path: Qwen3.5-9B-Q6_K.gguf
image_models:
  flux-3-klein:
    path: flux3_image_klein_sd-Q4_K_M.gguf
    encoder: Qwen3-VL-4B-Instruct-Q8_0.gguf
    vae: flux2-vae-BF16.gguf
    upsampler: qwen3.5-9b
```

With an upsampler set, each prompt is rewritten:

- `gmlx image generate flux-3-klein` loads the chat model, writes the
  caption, and unloads the chat model before the image model loads.
  `--upsampler` names a chat model, a GGUF or a `models` id, for one run.
- `gmlx serve` writes the caption with the chat model, which loads and
  unloads as it does for a chat request, and returns the caption as
  `revised_prompt` with the image. A request with the same prompt and
  another seed reuses the caption. With the
  [prompt cache](config.md#prompt-cache) on, a new prompt reuses the
  system prompt's prefill. The first caption also loads the chat model,
  which `pin: true` on its `models` entry keeps loaded.
- The ComfyUI pack rewrites the prompt of its text encode node, and
  ComfyUI's log shows the caption.

The chat model writes the same caption for the same prompt. The PNG keeps
the caption beside your prompt. A prompt with reference images or an
init image runs as typed, and the negative prompt is never rewritten. In
ComfyUI the text encode node does not see the references of an edit, so
the server also encodes the prompt as typed, and the pack samples an edit
and a negative prompt on that encoding.

To run a prompt as typed, pass `--no-upsample`, send
`"prompt_upsampling": false` in the request, or set `upsample: false` on
the entry, which a request with `"prompt_upsampling": true` turns back on.
Without an upsampler, FLUX.3 klein runs each prompt as typed and prints a
warning that names the `upsampler` setting and Qwen3.5-9B, unless the
entry has `upsample: false`. The other families run it as typed without a
warning.

### Your own system prompt

`upsample_prompt` on an entry names a text file whose system prompt the
upsampler takes in place of the family's, and `--upsample-prompt FILE`
does the same for one `gmlx image generate` run. A relative path is found
in `model_dirs`, as a model path is. The server reads the file for each
request, so an edit to it takes effect on the next prompt, and captions
written with the old file are not reused.

```yaml
image_models:
  krea-2-turbo:
    path: Krea-2-Turbo-Q6_K.gguf
    encoder: Qwen3-VL-4B-Instruct-Q8_0.gguf
    vae: qwen-image-vae-BF16.gguf
    upsampler: qwen3.5-9b
    upsample_prompt: prompts/krea-2-short.txt
```

The reply is read without its reasoning, as the family's own system prompt
asks for it: the caption as written on FLUX.3 klein and FLUX.2 klein, the
last paragraph on Krea 2, `rewritten_prompt` of a JSON object on
Qwen-Image 2.1, or the reply as written when it has no `{`, and the
caption as written, or `revised_prompt` of a JSON block, on Z-Image. A
Qwen-Image reply with a `{` but no `rewritten_prompt` in it, such as one
cut off at the token limit, is an error. On Z-Image, a `{prompt}` in your
file is replaced by the typed prompt. Your file should ask for the same
form.

### Qwen-Image 2.1 rewriter prompt

Qwen publishes the system prompt of its text to image rewriter as
`system_prompt.txt` in the Hugging Face repository
`Qwen/Qwen-Image-2.1-PE-T2I`, under the Qwen Research License, which
allows non-commercial use only. gmlx does not ship it.
`gmlx pull qwen-image` downloads it next to the model's other files, says
its license, and names it as the entry's `upsample_prompt`, so setting an
`upsampler` is all that is left to do. A file of that name already there
that is not Qwen's, such as your own prompt, stays as it is, and the pull
says to move or delete it. A new entry then goes without it, and an entry
that already names the file keeps naming it, as the warning says. Without
the file, a Qwen-Image request with an upsampler set is refused with a
message that says where to get it. The rewrite keeps the image size you ask for; the aspect ratio in
Qwen's reply is not used.

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
| `moss-image` | MOSS-Image | 50 | From one image | Qwen3.5-9B |
| `moss-image-base` | MOSS-Image Base | 50 | From one image | Qwen3.5-9B |
| `flux-2-klein` | FLUX.2 klein 4B | 4, with guidance 1 only | From up to 10 reference images | Qwen3-4B |
| `flux-2-klein-base` | FLUX.2 klein Base 4B | 50 | From up to 10 reference images | Qwen3-4B |
| `flux-2-klein-9b` | FLUX.2 klein 9B, non-commercial | 4, with guidance 1 only | From up to 10 reference images | Qwen3-8B |
| `flux-2-klein-base-9b` | FLUX.2 klein Base 9B, non-commercial | 50 | From up to 10 reference images | Qwen3-8B |
| `flux-2-klein-9b-kv` | FLUX.2 klein 9B KV, non-commercial | 4, with guidance 1 only | From up to 10 reference images | Qwen3-8B |
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
the VAE encodes the reference images. FLUX.2 klein 4B and Z-Image-Turbo
use the same Qwen3-4B file, and FLUX.2 klein at both sizes and FLUX.3
klein decode with the same VAE, so `pull` downloads each once. The 9B
models run the 4B models' steps and edit rules, with Qwen3-8B as the text
encoder, and their license allows non-commercial use only, as
[Licenses](#licenses) says.

FLUX.2 klein 9B KV is a version of FLUX.2 klein 9B made for faster edits.
Its reference images attend to each other alone, at time zero, so the
first step of an edit computes their keys and values in every block, and
the later steps reuse them and run the image and the prompt alone. A
1024x1024 edit from two 1024x1024 references takes about half the time it
takes on FLUX.2 klein 9B. Without reference images it runs as FLUX.2
klein 9B does.

MOSS-Image is OpenMOSS's fine-tuned model, and MOSS-Image Base the model
it is tuned from. Both run 50 steps at guidance 3, and leave block 22 out
of the negative pass, as OpenMOSS runs them. They read the prompt with
Qwen3.5-9B and edit from an init image, not from reference images.

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
| `moss-image` | Q6_K | Q8_0 | Q8_0 | Q8_0 |
| `moss-image-base` | Q6_K | Q8_0 | Q8_0 | Q8_0 |
| `flux-2-klein` | Q4_K_M | Q8_0 | Q8_0 | Q8_0 |
| `flux-2-klein-base` | Q4_K_M | Q8_0 | Q8_0 | Q8_0 |
| `flux-2-klein-9b` | None fits | Q4_K_M | Q8_0 | Q8_0 |
| `flux-2-klein-base-9b` | None fits | Q4_K_M | Q8_0 | Q8_0 |
| `flux-2-klein-9b-kv` | None fits | Q4_K_M | Q8_0 | Q8_0 |
| `flux-3-klein` | Q4_K_M | Q8_0 | Q8_0 | Q8_0 |
| `flux-3-klein-base` | Q4_K_M | Q8_0 | Q8_0 | Q8_0 |

No Krea 2 quant fits a 1024x1024 image on 16 GB, so a pull of the 16 GB
default sets the entry's default size to 512x512, and a pull with another
`--quant` sets none. Of Krea 2, gmlx runs Turbo, and it refuses a Krea 2
Raw GGUF by name.

On 16 GB a MOSS-Image pull takes the encoder at Q4_K_M, since the Q8_0
encoder leaves too little memory for the DiT there.

No FLUX.2 klein 9B DiT that `pull` takes makes an image on 16 GB, so
there a pull of a 9B model downloads nothing and names 24 GB, the
smallest Mac that runs one. With `--quant` it takes that DiT, and
`image generate` and the server refuse each request with the memory it
needs. On 24 GB the Q4_K_M DiT makes a
1024x1024 image from up to four 1024x1024 references, and a Q8_0 one from
one at most. FLUX.2 klein 9B KV keeps the keys and values of its
references, so on 24 GB its Q4_K_M DiT takes up to two 1024x1024
references and its Q8_0 DiT none, and on 32 GB its Q8_0 DiT takes two.

## Licenses

| Model | License | Use |
|-------|---------|-----|
| Z-Image-Turbo | [Apache 2.0](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | Any, under the license's terms |
| Qwen-Image 2.1 | [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE) | Research and evaluation only. Commercial use needs a license from Qwen. |
| Qwen-Image 2.1 Turbo | [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/LICENSE) | Research and evaluation only. Commercial use needs a license from Qwen. |
| Krea 2 Turbo | [Krea 2 Community License](https://cdn.jsdelivr.net/gh/krea-ai/krea-2@db3984fbc6e13b34c0064990fc2d95ac64d00058/assets/hf_samples/LICENSE.pdf) | Free commercial use for a company, affiliates counted, under $1M of revenue in the last twelve months. A deployment must filter what it makes. |
| MOSS-Image, MOSS-Image Base | [Apache 2.0](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/blob/main/LICENSE) | Any, under the license's terms |
| FLUX.2 klein 4B, FLUX.2 klein Base 4B | [Apache 2.0](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/main/LICENSE.md) | Any, under the license's terms |
| FLUX.2 klein 9B, FLUX.2 klein Base 9B | [FLUX Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/blob/main/LICENSE.md) | Non-commercial use only, on the terms below |
| FLUX.2 klein 9B KV | [FLUX Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv/blob/main/LICENSE) | Non-commercial use only, on the terms below |
| FLUX.3 klein, FLUX.3 klein Base | [FLUX Kommunity License](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md) | Non-commercial, and commercial use of the images under $5M of yearly revenue, on the terms below |
| Qwen3-4B, Qwen3-8B, Qwen3-VL-8B-Instruct, Qwen3-VL-4B-Instruct, Qwen3.5-9B | Apache 2.0 | Any, under the license's terms |

Under the [FLUX Kommunity License](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md),
a user whose yearly revenue, affiliates counted, is under $5M may use
FLUX.3 klein images commercially with content filters or a review of each
image for unlawful or infringing content before it is shared, and with a
disclosure that the image is made with AI where the law requires one.
Other commercial use of FLUX.3 klein needs a license from Black Forest
Labs.

Under the [FLUX Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/blob/main/LICENSE.md),
FLUX.2 klein 9B, FLUX.2 klein Base 9B and FLUX.2 klein 9B KV may be used
for non-commercial purposes only, as the license defines them, with
content filters or a review of each image for unlawful or infringing
content before it is shared, and with a disclosure that the image is made
with AI where the law requires one. Commercial use of them needs a license
from Black Forest Labs.

The system prompts gmlx ships for [prompt upsampling](#prompt-upsampling)
are under Apache 2.0, from Black Forest Labs' flux-3-image-klein and flux2
repositories, Krea's krea-2 repository and Tongyi-MAI's Z-Image-Turbo
Space. Qwen's rewriter prompt, which
`gmlx pull qwen-image` downloads, is under the Qwen Research License, for
non-commercial use only.

gmlx has no content filter of its own, so a deployment that a license
asks to filter adds its own.

The GGUF files are quantized by unsloth, the Qwen-Image 2.1 Turbo files by
AtomicChat, the Krea 2 Turbo files by Abiray, the FLUX.2 klein 9B KV files
by QuantStack and the FLUX.3 klein and MOSS-Image files by realrebelai,
under the same licenses. The Krea 2 VAE comes from Comfy-Org's Krea 2
repository, and the MOSS-Image VAE from OpenMOSS-Team's MOSS-Image-SFT
repository, under Apache 2.0. The
FLUX.2 VAE, which FLUX.2 klein at both sizes and FLUX.3 klein decode with,
comes from Black Forest Labs' FLUX.2 klein 4B repository, under Apache
2.0. The 9B repositories hold the same file.
