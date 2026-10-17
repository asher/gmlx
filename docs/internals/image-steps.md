# Image step routes

The step routes let a sampler that runs in another program, such as the
[ComfyUI node pack](../comfyui.md), call an image model one step at a time.
The server holds the weights, encodes prompts, and runs each DiT forward
and the VAE. The client keeps the noise, the schedule, the guidance and the
sampler. The user side is in [ComfyUI](../comfyui.md).

## The contract

The routes live under `/internal/image-steps/v1`. Every request but
`GET capabilities` carries the contract version the client speaks, as
`X-Gmlx-Steps-Contract: MAJOR.MINOR`. The server speaks 1.0. A request
without the header, or with another major version, gets 400, and the
message names the side to upgrade. A client reads `capabilities` first and
uses what it lists:

```sh
curl -s localhost:8080/internal/image-steps/v1/capabilities
```

The answer gives the contract and the oldest client it serves, the gmlx
version, the wire container and dtypes, the families, the step conventions,
the patch kinds, the features, and the limits: the largest tensor and
adapter upload, the most rows in one step, and the most runs held at once.

## A sampling run

A client that samples one image makes these calls:

1. `POST models/load` loads the model, and `GET models/{model}/manifest`
   gives its DiT tensor names and shapes, which a LoRA's keys map onto.
2. `POST encode` turns the prompt, and any reference images, into a
   conditioning. The answer names it by hash, and the server keeps it.
3. Each LoRA matrix goes up once: `HEAD adapters/{hash}` asks whether the
   server holds it, and `PUT adapters/{hash}` uploads it.
   `PUT patch-tables/{id}?model=ID` then names the set of patches and their
   strengths.
4. `POST runs/begin` holds the model for the run.
5. `POST step` runs once a step, with the latents, the timestep, the
   conditioning hashes and the patch table id.
6. `POST vae/decode` turns the final latents into pixels.
7. `POST runs/{run}/end` lets the model go.

A conditioning that the client made itself goes up the same way:
`HEAD tensors/{hash}` answers 200 when the server holds the tensor and 404
when it does not, and `PUT tensors/{hash}` uploads it. The server checks
that the body hashes to the name in the path.

## Routes

| Route | Purpose |
|-------|---------|
| `GET capabilities` | The contract version, families, wire dtypes and limits |
| `GET models`, `GET models/{model}/status`, `GET models/{model}/manifest` | The image models and the default one; one model's family, latent format, memory and sampling; its DiT tensor names and shapes |
| `POST models/load`, `POST models/unload` | Load or unload a model |
| `POST encode` | A prompt, with reference images for an edit, as a conditioning; the server keeps its tensors by hash |
| `HEAD tensors/{hash}`, `PUT tensors/{hash}` | Ask whether the server holds a tensor, or upload it |
| `HEAD adapters/{hash}`, `PUT adapters/{hash}` | Ask whether the server holds a LoRA matrix, or upload it |
| `POST adapters/register` | Read a LoRA file under a model folder of the server |
| `PUT patch-tables/{id}?model=ID` | Define a set of LoRA patches; its id is the sha256 of its JSON with sorted keys and no spaces |
| `POST step` | The model output for latents at a timestep, one row per conditioning |
| `POST vae/decode`, `POST vae/encode` | VAE decode and encode in the VAE's own latent space |
| `POST runs/begin`, `POST runs/{run}/end`, `POST interrupt` | Hold a model for a sampling run, end the run, stop its step in flight |

A model's status gives, under `sampling.schedule`, the rule and the
constants of its schedule when that schedule follows the image size,
as on FLUX.2 klein and Qwen-Image 2.1, and null on any other.

## Bodies and hashes

Tensors travel as a safetensors container with content type
`application/vnd.gmlx.tensors+safetensors`, in F32, F16, BF16, U8 or
BOOL, and the request fields of a container travel as JSON under
`__metadata__["gmlx"]`. Other bodies are JSON.

A tensor's hash is the sha256 of three parts in order: its safetensors
dtype name, such as `F32`; its shape as a JSON list with a space after each
comma, such as `[1, 16, 128]`; and its bytes. A client uploads a
conditioning or a LoRA matrix once by that hash.

## Conditionings and references

A conditioning with reference images takes ComfyUI's form: its tensor
holds the text rows, and `image_slots` names, for each reference image,
the text row that its latents go in before. A step row that names more
reference latents than slots puts the others after the text.

An encode keeps each reference image at its own size, so a client that
sizes its references gives the encoder and its own VAE the same pixels.
Qwen-Image 2.1 takes references of 256x256 to 1024x1024 pixels in area,
with sides a multiple of 32, and an image outside that area gets 400. An
encode with `refs: false` leaves out the VAE latents of its reference
images, for a client that encodes them with a VAE of its own.

## Steps and runs

A step names its conditionings and its patch table by hash. When the
server does not hold one, the step gets 409 with a `missing` list, and
the client uploads those and sends the step again. A run keeps the
prefix state of each of its conditionings between steps, and the
`X-Gmlx-Prefix-Cache` header of a step answers `hit` or `miss`.
`X-Gmlx-Forward-Ms` is the time of the model forward, so a client's wall
time less it is the transport time.

A step with `ref_cache: true`, on a server whose features list
`ref_cache`, runs as ComfyUI's `Flux KV Cache` node does on FLUX.2 klein
9B KV. On the first step of the run, each row with reference latents runs
its references at time zero and keeps their keys and values in every
block, and the step returns its output. The keys and values of a
reference depend on the latents of the row, so each row keeps its own,
under its conditioning and its place in the step's rows. The later steps
of the run run the image and the text of each row alone, and read the
keys and values of that row. The header answers `hit` on those steps, and
the end of the run drops the keys and values. A step with `ref_cache` and
no run gets 400, as does one on a DiT that keeps no such cache.

A run that sits idle for 120 seconds ends within a few seconds after, and
lets the model go. The server holds at most 32 runs, and a begin past that
gets 429. An interrupt names its run and takes effect at the next step:
the step running when it lands finishes on the GPU, gets 409 with code
`interrupted`, and every later step of the run gets the same. An end
does the same to a step in flight. An interrupt that names no run stops
nothing.

## Limits

A step, a VAE decode and a VAE encode take the size limits of the OpenAI
routes: an image over 2048x2048 pixels or with a side over 4096 gets 413,
as does a VAE pass of more than 16 images. One whose memory plan does not
fit beside the models loaded gets 400. The VAE routes decode and encode
the images of a batch one at a time. `capabilities` gives the largest
tensor and adapter upload, which are the
[`image.tensor_cache_mb`](../config.md#serverimagetensor_cache_mb) and
[`image.adapter_cache_mb`](../config.md#serverimageadapter_cache_mb) caches.
Each image model books those two caches while the step routes are on.

## Access

The routes answer loopback peers only.
[`image.steps_remote`](../config.md#serverimagesteps_remote) serves other
peers that the API key admits, and
[`image.steps_api`](../config.md#serverimagesteps_api) set to `false`
removes the routes. The steps of all clients queue on the image thread with
the jobs of the OpenAI routes, and share the GPU with chat in the same way.

## Timing the step routes

`gmlx image bench --via-steps` runs each step through these routes on a
loopback port of its own process. The prompt is encoded in the process and
uploaded by hash, each step is one request, and the latents move between
steps in the bench with its sampler. Its steps take the `reference`
convention, which embeds the time as `sample` does. The ComfyUI pack's
steps take `comfy`, which embeds the time as ComfyUI's forward does, and
for FLUX.2 klein that is the fp32 time.

The record's transport is `steps`. It holds the median per step of the
request's wall time less the forward time the server reports, and
`equal_to_direct`, whether the final latents of the last run equal those of
`sample` at the same seed. Each run also lists its transport time per step.
