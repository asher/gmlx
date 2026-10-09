# Image generation API

The server answers the OpenAI image routes with the image models of its
configuration file. An OpenAI client, such as the `openai` Python package,
works against it unchanged:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
result = client.images.generate(prompt="a lighthouse at dusk",
                                size="1024x1024")
```

- [Setting up a model](#setting-up-a-model)
- [Routes](#routes)
- [Request fields](#request-fields)
- [Edits and variations](#edits-and-variations)
- [URL responses](#url-responses)
- [Streaming](#streaming)
- [Memory, queueing and errors](#memory-queueing-and-errors)
- [Step routes](#step-routes)

## Setting up a model

Add the model under [`image_models`](config.md#image-models) in the
configuration file and reload the server. Its id is the `model` of a
request:

```sh
curl localhost:8080/v1/images/generations -d '{
  "model": "z-image-turbo",
  "prompt": "a lighthouse at dusk",
  "size": "1024x1024"
}'
```

An image model loads on its first request and shares the memory budget,
the idle timeout and `/unload` with the chat models, as
[Image models](config.md#image-models) describes. Without an image model
in the file the routes answer 404 with type `model_not_found`.

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
runs elsewhere, such as the ComfyUI node pack. They live under
`/internal/image-steps/v1` and are an internal interface: their payloads
can change with each contract version, and the version a client speaks
goes in the `X-Gmlx-Steps-Contract` header of every request but
`capabilities`. A major version the server does not speak gets 400 that
names the side to upgrade.

The routes answer loopback peers only. [`image.steps_remote`](config.md#serverimagesteps_remote)
serves other peers that the API key admits, and
[`image.steps_api`](config.md#serverimagesteps_api) set to `false` removes
the routes.

Tensors travel as a safetensors container with content type
`application/vnd.gmlx.tensors+safetensors`, in F32, F16, BF16, U8 or
BOOL, and the request fields of a container travel as JSON under
`__metadata__["gmlx"]`. Other bodies are JSON. A tensor's hash is the
sha256 of its dtype name, its shape as JSON and its bytes, and a client
uploads a conditioning or a LoRA matrix once by that hash.

| Route | Purpose |
|-------|---------|
| `GET capabilities` | The contract version, families, wire dtypes and limits |
| `GET models`, `GET models/{model}/status`, `GET models/{model}/manifest` | The image models; one model's family, latent format, memory and sampling; its DiT tensor names and shapes |
| `POST models/load`, `POST models/unload` | Load or unload a model |
| `POST encode` | A prompt, with reference images for an edit, as a conditioning; the server keeps its tensors by hash. `refs: false` leaves out the VAE latents of the reference images, for a client that encodes them itself |
| `PUT` or `HEAD tensors/{hash}`, `PUT` or `HEAD adapters/{hash}` | Upload a tensor or a LoRA matrix, or ask whether the server holds it |
| `POST adapters/register` | Read a LoRA file under a model folder of the server |
| `PUT patch-tables/{id}?model=ID` | Define a set of LoRA patches; its id is the sha256 of its JSON with sorted keys and no spaces |
| `POST step` | The model output for latents at a timestep, one row per conditioning |
| `POST vae/decode`, `POST vae/encode` | VAE decode and encode in the VAE's own latent space |
| `POST runs/begin`, `POST runs/{run}/end`, `POST interrupt` | Hold a model for a sampling run, end the run, stop its step in flight |

A step names its conditionings and its patch table by hash. When the
server does not hold one, the step gets 409 with a `missing` list, and
the client uploads those and sends the step again. A run keeps the
prefix state of each of its conditionings between steps, and the
`X-Gmlx-Prefix-Cache` header of a step answers `hit` or `miss`.
`X-Gmlx-Forward-Ms` is the time of the model forward, so a client's wall
time less it is the transport time. A run that sits idle for 120 seconds
ends.

The steps of all clients queue on the image thread with the jobs of the
OpenAI routes, and share the GPU with chat in the same way.

