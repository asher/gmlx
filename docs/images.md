# Image generation

gmlx makes images from Z-Image-Turbo, Qwen-Image 2.1 and Krea 2 GGUF
files: on the command line, on the server's OpenAI image routes, and in
ComfyUI. With a config from [`gmlx init`](cli.md#gmlx-init), one command
downloads a model and adds it to the config, and the next one draws:

```sh
gmlx pull z-image-turbo
gmlx image generate z-image-turbo --prompt "a lighthouse at dusk" --seed 1 --output lighthouse.png
```

On a Mac with 128 GB, `pull` takes the BF16 DiT and the Q8_0 text
encoder, and the two commands print:

```text
pull z-image-turbo (Z-Image-Turbo): DiT BF16, encoder Q8_0, the default for 64 GB and up
  dit      z-image-turbo-BF16.gguf                   11.5 GB  download
  encoder  Qwen3-4B-Q8_0.gguf                         4.0 GB  download
  vae      vae/diffusion_pytorch_model.safetensors    0.2 GB  download, packed to z-image-vae-BF16.gguf
  download 15.6 GB, 248.2 GB free at ~/models
  the three parts: 15.6 GB, which fit this Mac's 128 GB RAM
  license: Z-Image-Turbo apache-2.0 (https://huggingface.co/Tongyi-MAI/Z-Image-Turbo); encoder apache-2.0
downloading unsloth/Z-Image-Turbo-GGUF/z-image-turbo-BF16.gguf
  dit: ~/models/unsloth__Z-Image-Turbo-GGUF/z-image-turbo-BF16.gguf
downloading unsloth/Qwen3-4B-GGUF/Qwen3-4B-Q8_0.gguf
  encoder: ~/models/unsloth__Qwen3-4B-GGUF/Qwen3-4B-Q8_0.gguf
downloading Tongyi-MAI/Z-Image-Turbo/vae/diffusion_pytorch_model.safetensors
downloading Tongyi-MAI/Z-Image-Turbo/vae/config.json
  vae: ~/models/Tongyi-MAI__Z-Image-Turbo/z-image-vae-BF16.gguf
registered z-image-turbo in ~/.config/gmlx/gmlx.yaml (the image default)
try it: gmlx image generate z-image-turbo --prompt "a red bicycle" --output out.png

[image] Z-Image (turbo), layout default, 453 tensors
[install] replaced 0 leaves with kquant modules
[dtype] cast 242 float params (norms etc.) to bf16
[load_weights] loaded 453 / 453 model parameters
[verify] zero-copy views OK (453 params)
[image] text encoder qwen3-4b, tap after layer 35 of 36
[image] wrote lighthouse.png (seed 1, 9.91 s)
```

- [Models](#models)
- [Pull a model](#pull-a-model)
- [Which quant for your Mac](#which-quant-for-your-mac)
- [Generate on the command line](#generate-on-the-command-line)
- [Setting up a model](#setting-up-a-model)
- [Routes](#routes)
- [Request fields](#request-fields)
- [Edits and variations](#edits-and-variations)
- [URL responses](#url-responses)
- [Streaming](#streaming)
- [Memory, queueing and errors](#memory-queueing-and-errors)
- [Step routes](#step-routes)
- [Licenses](#licenses)

## Models

An image model is three files: a DiT GGUF that draws, the text encoder
GGUF that reads the prompt, and a VAE that turns the DiT's latents into
pixels. `gmlx pull` takes a model by its name:

| Name | Model | Steps | Edits | Text encoder |
|------|-------|-------|-------|--------------|
| `z-image-turbo` | Z-Image-Turbo | 8 | From one image | Qwen3-4B |
| `qwen-image` | Qwen-Image 2.1 | 40 | From up to 10 reference images | Qwen3-VL-8B-Instruct, with its mmproj |
| `qwen-image-turbo` | Qwen-Image 2.1 Turbo | 8 | From up to 10 reference images | Qwen3-VL-8B-Instruct, with its mmproj |
| `krea-2-turbo` | Krea 2 Turbo | 8 | From one image | Qwen3-VL-4B-Instruct |
| `krea-2` | Krea 2 Raw | 52 | From one image | Qwen3-VL-4B-Instruct |

The Qwen-Image 2.1 models are for non-commercial use only, and Krea 2 is
free for commercial use only under $1M of annual revenue and 50 seats, as
[Licenses](#licenses) describes. [Benchmarks](benchmarks.md#image-generation)
compares the speed of the first three with ComfyUI and stable-diffusion.cpp,
and of Krea 2 Turbo with ComfyUI.

## Pull a model

```sh
gmlx pull qwen-image-turbo
```

`pull` prints its plan, then downloads. The plan names each part, its
size, whether it is on disk already, the space the download takes beside
the free space, and the model's license. The VAE comes as a safetensors
file, which `pull` checks against its published digest and packs to a
GGUF. The new entry goes under [`image_models`](config.md#image-models),
and the first image model becomes the server's default image model.

A text encoder or VAE already in a model folder is used where it is when
its weights match the model's. Of several encoders, `pull` takes the
quant for your Mac first, then a chat model, then the file nearest the
DiT, and the plan names the quant it takes. A chat model that is the
text encoder is shared by its id, so the server holds its weights once.
For the Qwen-Image models the chat model must have its `mmproj` vision
tower; without one, the entry names the encoder file and the tower that
`pull` adds.
These flags change what `pull` takes:

| Flag | Effect |
|------|--------|
| `--quant QUANT` | Download the DiT at this quant. |
| `--encoder-quant QUANT` | Download the text encoder at this quant, even when another quant is on disk. |
| `--only dit,encoder,vae` | Download only these parts. The entry is added once all of them are on disk. |
| `--keep-source` | Keep the safetensors VAE beside its GGUF. |
| `--to DIR` | Download into this folder. |
| `--no-register` | Download without adding an entry. |

The [`gmlx pull`](cli.md#gmlx-pull) reference has every flag. Model
discovery finds image files too: a DiT in a scanned folder with its text
encoder and VAE becomes an image model, and a DiT without them is reported
with the `gmlx pull ... --only encoder,vae` command that completes it.

`gmlx list --image` lists the image models, `gmlx rm ID` removes one with
the files no other model uses, and `gmlx doctor` checks that each DiT has
its family's text encoder and VAE.

## Which quant for your Mac

Without `--quant`, `pull` picks the quants for the memory of the Mac it
runs on:

| Model | 16 GB | 24 GB | 32 GB | 64 GB or more |
|-------|-------|-------|-------|---------------|
| `z-image-turbo` | Q4_K_M | Q8_0 | BF16 | BF16 |
| `qwen-image` | Q4_K_M | Q8_0 | Q8_0 | F16 |
| `qwen-image-turbo` | AD-Q4_K | Q8_0 | BF16 | BF16 |
| `krea-2-turbo` | Q3_K_M | Q6_K | Q8_0 | Q8_0 |
| `krea-2` | Q3_K_M | Q6_K | Q8_0 | Q8_0 |

The table gives the DiT quant, and the text encoder is Q8_0 on every
class. A Mac takes the column of the largest class its RAM holds, and a
Mac under 16 GB takes the 16 GB column.

Each pick is the quant whose images come closest to full precision among
those that fit the memory of that class, both in `gmlx image generate` and
in `gmlx serve`. When the text encoder and the DiT do not fit side by side,
as on 16 GB, both load the encoder for each new prompt, encode, release it,
and then load the DiT and the VAE. A prompt encoded before is taken from the
cache, and nothing loads for it. On 16 GB `serve` also keeps smaller caches
of prompts and request LoRAs. Pull with `--encoder-quant q4_k_m` to leave
more room for other work.

Krea 2 has a 12B DiT, and no quant of it fits a 1024x1024 image on
16 GB. There Q3_K_M, the smallest, makes images up to 512x512. A pull of
that default writes `defaults: {size: 512x512}` into the entry it
registers, and its plan says so. `gmlx image generate` and `serve` then
make 512x512 images from a prompt that names no size. A pull with another
`--quant` writes no size.

`--quant` and `--encoder-quant` pick other quants:

```sh
gmlx pull qwen-image-turbo --quant ad-q5_k --encoder-quant q4_k_m
```

## Generate on the command line

`gmlx image generate` takes the id of an `image_models` entry, or the DiT
GGUF with `--encoder` and `--vae`:

```sh
gmlx image generate z-image-turbo --prompt "a red bicycle" --size 1024x1024 --output bicycle.png
gmlx image generate qwen-image-turbo --prompt "make it a watercolor" --ref photo.jpg --output watercolor.png
gmlx image generate z-image-turbo --prompt "a red bicycle" --init-image sketch.png --strength 0.6 --output bicycle.png
```

A PNG keeps the prompt, seed and settings in a text chunk. Before it
loads, `generate` checks that the image fits in memory, and refuses with
what does not fit. The
[`gmlx image`](cli.md#gmlx-image) reference has the flags, and `gmlx image
bench` times a model on this Mac.

## Setting up a model

The server answers the OpenAI image routes with the image models of its
configuration file. `gmlx pull` adds an entry, and an entry written by hand
under [`image_models`](config.md#image-models) works the same way. Its id is
the `model` of a request:

```sh
curl localhost:8080/v1/images/generations -d '{
  "model": "z-image-turbo",
  "prompt": "a lighthouse at dusk",
  "size": "1024x1024"
}'
```

An OpenAI client, such as the `openai` Python package, works against it
unchanged:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
result = client.images.generate(prompt="a lighthouse at dusk",
                                size="1024x1024")
```

An image model loads on its first request and shares the memory budget,
the idle timeout and `/unload` with the chat models, as
[Image models](config.md#image-models) describes. Without an image model
in the file the routes answer 404 with type `model_not_found`.
[Open WebUI](launch.md#open-webui) draws with the server's default image
model, and [ComfyUI](comfyui.md) runs its own graphs on the server's image
models.

## Routes

| Route | Body | Purpose |
|-------|------|---------|
| `POST /v1/images/generations` | JSON | Images from a prompt |
| `POST /v1/images/edits` | Multipart form or JSON | Images from a prompt and one or more images |
| `POST /v1/images/variations` | Multipart form | Images like the one given |
| `GET /v1/images/files/{name}` | None | An image that a request with `response_format: url` wrote |

Each route also answers without the `/v1` prefix. Each request writes one
line to the server log.

## Request fields

| Field | Meaning |
|-------|---------|
| `prompt` | What to draw. Required on generations and edits. |
| `model` | An image model id, or an OpenAI model name as below. A name that is not an image model gets 404, and a chat model id gets 400 of type `wrong_model_kind`. |
| `n` | Images to make, 1 to 10. They run one after another, at seeds `seed`, `seed` plus 1 and so on. |
| `size` | `WxH` or `auto`, the model's default size. Sides are multiples of the model's size step and at most 4096, and an image at most 2048x2048 pixels. |
| `quality` | `low`, `medium`, `high` or `auto`, each a step count of the model's family. `standard` reads as `medium`, and `hd` as `high`. |
| `background` | `transparent`, `opaque` or `auto`. Transparency needs a model whose VAE decodes an alpha channel, such as Qwen-Image 2.1, and png or webp. |
| `output_format` | `png`, the default, `jpeg` or `webp`. |
| `output_compression` | The jpeg and webp quality, 0 to 100, where 100, the default, keeps the most detail. Ignored for png. |
| `response_format` | `b64_json`, the default, or `url`. |
| `stream` | Send the result as server-sent events, as [Streaming](#streaming) shows. Takes `n` of 1. |
| `partial_images` | 0 to 3 images of the work in progress before the final one, in a stream. |
| `negative_prompt` | What to steer away from, read when `guidance` is above 1. |
| `steps` | Sampler steps, 1 to 200, in place of `quality`. |
| `seed` | The seed of the first image. A random one without it. |
| `guidance` | Classifier-free guidance, 1 to 30. 1 turns guidance off. |
| `sampler` | `euler` or `uni_pc_bh2`. |
| `scheduler` | `default` only: the model's own schedule. |
| `strength` | How far an edit or a variation departs from its image, above 0 and at most 1. |
| `adapters` | LoRA adapters for this request, as a list of `{"id", "strength"}`. The ids are described below the table. |
| `image`, `image[]`, `images` | The images of an edit or a variation, as below. |
| `mask` | An image with an alpha channel: an edit changes the pixels where it is fully transparent. |

An empty `model`, `default`, and a name that starts with `gpt-image-`,
`dall-e-` or `chatgpt-image` take the default image model: the one that
[`server.defaults.image_model`](config.md#serverdefaultsimage_model) names,
or the only one. A `quality` of `high` runs the family's own step count,
and `auto` the steps of the entry's `defaults`, or else the family's own.
An adapter id is the file name of an adapter in the model's
`adapters`, or a path inside a [model folder](config.md#servermodel_dirs).

`moderation`, `style`, `input_fidelity` and `user` are accepted and
ignored, and the log names them in one warning line, as it names any field
a route does not read. `output_path`, `output_dir` and
`response_format: "path"` get 400: the server writes no file a request
names. An adapter id that is an absolute path gets 400.

## Edits and variations

An edit takes its images as files of a multipart form, `image` for one and
`image[]` for several, which is how the `openai` package sends them. A
JSON body takes `images` as a list of `{"image_url": "data:image/..."}`,
or of base64 image data. A `file_id` gets 400, and the server never reads
a local path or fetches a URL from a JSON body. Images are PNG, JPEG, WebP or GIF.

| Model | Edits | Variations |
|-------|-------|------------|
| Z-Image | One image, which the result starts from at `strength`, 0.6 by default. More than one image gets 400. | The image with an empty prompt, at the default `strength` |
| Qwen-Image 2.1 | Up to 10 images, which the model sees as references. The image model needs `encoder_mmproj`. | The image as a reference, with the instruction to make a variation of it |
| Krea 2 | One image, which the result starts from at `strength`, 0.6 by default. More than one image gets 400. | The image with an empty prompt, at the default `strength` |

Without `size` the result takes the size of the first image, cut down to
the model's size step, and a larger image gives the largest size of its
shape within the size limits. With a `mask` the edit starts from the first image
at `strength` 1, changes only the masked region, and keeps the other
pixels as they were.

## URL responses

With `response_format: url` each image is written to a file under the
server's cache folder, and the answer holds a link to
`/v1/images/files/{token}.{ext}`. The token is 32 random bytes and is the
only thing that grants access: the file route needs no API key, so that an
`img` tag can show the image. Files expire after
[`image.url_ttl_s`](config.md#image-requests), 60 minutes by default, and
past [`image.url_max_bytes`](config.md#image-requests) the oldest go
first. The link uses the host the request was sent to.

## Streaming

With `stream: true` the answer is server-sent events. Generations send
`image_generation.partial_image` events, then one
`image_generation.completed` event; edits send `image_edit.partial_image`
and `image_edit.completed`. Each event repeats its type inside its data,
and the stream closes after the last event. A partial image is a full
decode of the model's estimate of the final image at that step, and costs
one VAE decode. An error during the stream is sent as a data object with an
`error` key, which the `openai` package raises.

```python
stream = client.images.generate(prompt="a lighthouse at dusk", stream=True,
                                partial_images=2)
for event in stream:
    print(event.type)
```

## Memory, queueing and errors

Chat stays responsive while an image renders. Image jobs run one at a
time on their own thread, and while a chat reply streams, an image job
pauses after each block of its model so chat steps run in between. The
reply keeps streaming at a steady pace, somewhat slower than alone, and
the image slows only while chat is streaming: with no chat reply in
progress, an image runs at its full speed.

By default chat gets half of the GPU while both run. On an M5 Max, a
gemma-4-E2B reply beside a 1024x1024 Z-Image-Turbo image streams at about
74 tokens per second, against about 157 alone, and the image takes about
1.8 times as long for the part of its run that chat shares. [`GMLX_IMAGE_CHAT_SHARE`](env-vars.md#server)
sets chat's share, from 0 to 0.9 with a default of 0.5, and `off` runs
each image step whole, which leaves chat waiting for each step to end.

Before a request queues, the server plans its memory at
its size beside the models that are loaded. A size that does not fit gets
400 naming the largest size of the same shape that does. An edit with more
reference images than fit gets 400 naming the count and the reference size
that fit.

| Condition | Response |
|-----------|----------|
| More requests wait than [`image.max_queue`](config.md#image-requests) | 429 with `Retry-After` |
| The client disconnects | The job stops at its next step, and the log line marks the request `cancelled` |
| The job waits for its turn, or for one step, longer than the token queue timeout | 504, and the log line marks the request `timeout` |
| No image model is configured | 404 `model_not_found` |
| The model's file is gone | 404 `model_file_missing` |
| The model's VAE, encoder or mmproj is not of its family | 400 `image_model_mismatch` |

The usage of an answer counts the encoder tokens of the prompts as text
tokens, the tokens of reference images as image tokens, and the latent
tokens of each image as output tokens.

The Responses route does not run the `image_generation` tool. A request
that sends it gets a warning line in the log, and the tool is dropped.

## Step routes

The step routes run one denoise step at a time for a client whose sampler
runs elsewhere, such as the [ComfyUI node pack](comfyui.md). They live under
`/internal/image-steps/v1` and are an internal interface, which
[Image step routes](internals/image-steps.md) describes.

The routes answer loopback peers only. [`image.steps_remote`](config.md#serverimagesteps_remote)
serves other peers that the API key admits, and
[`image.steps_api`](config.md#serverimagesteps_api) set to `false` removes
the routes. The steps of all clients queue on the image thread with the
jobs of the OpenAI routes, and share the GPU with chat in the same way.

## Licenses

| Model | License | Use |
|-------|---------|-----|
| Z-Image-Turbo | [Apache 2.0](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | Any, under the license's terms |
| Qwen-Image 2.1 | [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE) | Research and evaluation only. Commercial use needs a separate license from Qwen. |
| Qwen-Image 2.1 Turbo | [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/LICENSE) | Research and evaluation only. Commercial use needs a separate license from Qwen. |
| Krea 2 Turbo | [Krea 2 Community License](https://cdn.jsdelivr.net/gh/krea-ai/krea-2@db3984fbc6e13b34c0064990fc2d95ac64d00058/assets/hf_samples/LICENSE.pdf) | Free commercial use only under $1M of annual revenue and 50 seats. A deployment must filter the content it makes. |
| Krea 2 Raw | [Krea 2 Community License](https://cdn.jsdelivr.net/gh/krea-ai/krea-2@db3984fbc6e13b34c0064990fc2d95ac64d00058/assets/hf_samples/LICENSE.pdf) | Free commercial use only under $1M of annual revenue and 50 seats. A deployment must filter the content it makes. |
| Qwen3-4B, Qwen3-VL-8B-Instruct, the text encoders | Apache 2.0 | Any, under the license's terms |
| Qwen3-VL-4B-Instruct, the Krea 2 text encoder | Apache 2.0 | Any, under the license's terms |

`gmlx pull` prints the license of each model before it downloads. The
GGUF files are quantized by unsloth, and the Qwen-Image 2.1 Turbo files by
AtomicChat, under the same licenses. The Krea 2 Turbo files are quantized by
Abiray and the Krea 2 Raw files by realrebelai, and the Krea 2 VAE file comes
from Comfy-Org's Krea 2 repository.
