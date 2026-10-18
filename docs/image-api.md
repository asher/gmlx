# Image API

`gmlx serve` answers the OpenAI image routes with the image models of its
configuration. This page lists the routes, the fields a request takes and
the errors it can get. [Call it from an app](images.md#call-it-from-an-app)
shows a first request:

```sh
curl -s localhost:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model": "z-image-turbo", "prompt": "a lighthouse at dusk", "seed": 1}' \
  | jq -r '.data[0].b64_json' | base64 -d > lighthouse.png
```

- [Routes](#routes)
- [Request fields](#request-fields)
- [Edits and variations](#edits-and-variations)
- [URL responses](#url-responses)
- [Streaming](#streaming)
- [Errors](#errors)

## Routes

| Route | Body | Makes |
|-------|------|-------|
| `POST /v1/images/generations` | JSON | Images from a prompt |
| `POST /v1/images/edits` | Multipart form or JSON | Images from a prompt and one or more images |
| `POST /v1/images/variations` | Multipart form | Images like the one given |
| `GET /v1/images/files/{name}` | None | An image that a request with `response_format: url` wrote |

Each route also answers without the `/v1` prefix. In `/v1/models` an image
model has `"image": true`, its `family` and `variant`, and `edit: true` when
it edits from reference images: FLUX.2 klein and FLUX.3 klein always, and
Qwen-Image 2.1 when the entry has its encoder's vision tower. It carries
`resident`, `pinned` and `default`, where `default` marks the default image
model, and no context figures.

## Request fields

| Field | Meaning |
|-------|---------|
| `prompt` | What to make. Required on generations and edits. |
| `model` | An image model id. `default`, no model, and a name that starts with `gpt-image-`, `dall-e-` or `chatgpt-image` take the default image model. |
| `n` | Images to make, 1 to 10, at seeds `seed`, `seed` plus 1 and so on. |
| `size` | `WxH`, or `auto` for the entry's default size. Sides are multiples of the model's size step, at most 4096, and an image at most 2048x2048 pixels. |
| `quality` | `low`, `medium`, `high` or `auto`, each a step count. `standard` reads as `medium`, and `hd` as `high`. |
| `steps` | [Steps](glossary.md#steps), 1 to 200, in place of `quality`. |
| `seed` | The seed of the first image, random without it. |
| `guidance` | [Guidance](glossary.md#guidance), 1 to 30. 1 turns it off. |
| `negative_prompt` | What to steer away from, read when `guidance` is above 1. It is never rewritten. |
| `prompt_upsampling` | `false` runs the prompt as typed on an entry with an [upsampler](images.md#prompt-upsampling), and `true` rewrites it on one with `upsample: false`. |
| `sampler` | [Sampler](glossary.md#sampler), `euler` or `uni_pc_bh2`. |
| `scheduler` | `default` only, the model's own schedule. |
| `strength` | How far an edit or a variation departs from its image, from 0.01 to 1. |
| `adapters` | LoRAs for this request, as a list of `{"id", "strength"}`, as [Use a LoRA](images.md#use-a-lora) describes. |
| `background` | `transparent`, `opaque` or `auto`. Transparency needs Qwen-Image 2.1 and png or webp. |
| `output_format` | `png`, the default, `jpeg` or `webp`. |
| `output_compression` | The jpeg and webp quality, 0 to 100, 100 by default. A png ignores it. |
| `response_format` | `b64_json`, the default, or `url`. |
| `stream` | Send the result as server-sent events. Takes `n` of 1. |
| `partial_images` | 0 to 3 images of the work in progress before the final one, in a stream. |
| `image`, `image[]`, `images` | The images of an edit or a variation. |
| `mask` | An image with an alpha channel. An edit changes the pixels whose alpha is below 128. |

The default image model is the one that
[`server.defaults.image_model`](config.md#serverdefaultsimage_model) names,
or the only one. Another name that is no image model gets 404. A `quality`
of `high` runs the model's own steps, and `auto` the steps of the entry's
`defaults`, or else the model's own. FLUX.2 klein and FLUX.3 klein run
their default steps at every `quality`.

`moderation`, `style`, `input_fidelity` and `user` are accepted and
ignored, and the log names them in a warning. The server writes no file
that a request names, so `output_path`, `output_dir` and `response_format:
"path"` get 400.

On an entry with an [upsampler](images.md#prompt-upsampling), each
image of the answer carries the caption the prompt was rewritten to as
`revised_prompt`, as does a stream's `completed` event. An edit or a
variation runs its prompt as typed and ignores `prompt_upsampling`.

## Edits and variations

An edit takes its images as files of a multipart form, `image` for one and
`image[]` for several, which is how the `openai` package sends them. A JSON
body takes `images` as a list of `{"image_url": "data:image/..."}`, or of
base64 image data. The server never reads a local path or fetches a URL
from a JSON body, and a `file_id` gets 400. Images are PNG, JPEG, WebP or
GIF.

| Model | Edits | Variations |
|-------|-------|------------|
| Z-Image, Krea 2, MOSS-Image | One image, which the result starts from at `strength`, 0.6 by default. More images get 400. | The image with an empty prompt, at the default `strength` |
| Qwen-Image 2.1 | Up to 10 reference images, when the entry has `encoder_mmproj` | The image as a reference, with an instruction to make a variation of it |
| FLUX.2 klein | Up to 10 reference images, with no vision tower | The image as a reference, with an instruction to make a variation of it |
| FLUX.3 klein | Up to 10 reference images, with no vision tower | The image as a reference, with an instruction to make a variation of it |

Without `size`, the result takes the size of the first image, cut down to
the model's size step and the size limits. With a `mask`, the edit starts
from the first image at `strength` 1 and changes only the masked region.

FLUX.3 klein scales a single reference down to at most the area of
2048x2048 pixels, and pads it out to sides that are multiples of 16. Of
several, it scales each down to at most the area of 1024x1024 and crops it
to sides that are multiples of 16. Without `size` the result has the size
of the first reference as prepared, and the result of a single reference is
cropped back to the scaled image. A size past the limits of 4096 pixels a
side and 2048x2048 pixels is scaled into them at its shape, so a first
reference of 8192x128 pixels among several gives a 4096x64 result. When a
side scales to less than 16 pixels, as from a first reference of 20000x30
pixels among several, the request gets 400 and needs a `size`.

FLUX.2 klein scales a single reference down to at most the area of
2024x2024 pixels, and each of several to at most the area of 1024x1024.
It then crops each to sides that are multiples of 16, around the center.
A reference needs sides of at least 64 pixels, and neither side more than
8 times the other. Without `size` the result has the size of the first
reference as prepared, scaled into the same limits, and is not cropped
back. A 500x300 image gives a 496x288 result, and an 8000x1000 image,
prepared at 5712x704, gives a 4096x496 result.

## URL responses

With `response_format: url` each image is written to a file under the
server's cache folder, and the answer links to
`/v1/images/files/{token}.{ext}` on the host the request was sent to. The
file route needs no API key, so that an `img` tag can show the image. The
token is 32 random bytes, and only a client that has the link can read the
file. Files expire after [`image.url_ttl_s`](config.md#image-requests), and
past [`image.url_max_bytes`](config.md#image-requests) the oldest go first.

## Streaming

With `stream: true` the answer is server-sent events. A generation sends
`image_generation.partial_image` events, then one
`image_generation.completed` event. An edit sends `image_edit.partial_image`
and `image_edit.completed`. Each partial image shows the whole image as the
model sees it at that step, and costs one VAE decode. An error during the
stream arrives as a data object with an `error` key, which the `openai`
package raises:

```python
stream = client.images.generate(model="z-image-turbo", prompt="a lighthouse at dusk",
                                stream=True, partial_images=2)
for event in stream:
    print(event.type)
```

## Errors

| Condition | Response |
|-----------|----------|
| No image model in the configuration | 404 `model_not_found` |
| A chat model id on an image route | 400 `wrong_model_kind` |
| The model's file is gone | 404 `model_file_missing` |
| The entry's VAE, encoder or mmproj is of another model | 400 `image_model_mismatch` |
| A size that does not fit in memory beside the loaded models | 400 naming the largest size of its shape that fits |
| More reference images than fit in memory | 400 naming the count and the reference size that fit |
| One of several FLUX.3 klein references is under 16 pixels on a side | 400 naming the image |
| A FLUX.2 klein reference is under 64 pixels on a side, or one side is more than 8 times the other | 400 naming the image |
| `prompt_upsampling: true` on an entry without an upsampler | 400 naming the `upsampler` setting |
| The upsampler writes an empty caption, or a reply its family's parser finds no caption in | 500 `server_error` |
| The entry's `upsample_prompt` file is missing, or a Qwen-Image 2.1 entry with an upsampler has none | 500 `server_error` saying where to get the file |
| The upsampler's context has no room for its system prompt and caption | 500 `server_error` naming the upsampler |
| More requests waiting than [`image.max_queue`](config.md#image-requests) | 429 with `Retry-After` |
| The client disconnects | The job stops at its next step, and the log marks the request `cancelled` |
| The job waits for its turn, or for one step, longer than [`token_queue_timeout_s`](config.md#servertoken_queue_timeout_s) | 504, and the log marks the request `timeout` |

[Troubleshooting](troubleshooting.md#image-generation) has the fix for
each message. The Responses route does not run the `image_generation`
tool: it drops the tool and logs a warning.
