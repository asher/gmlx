# Image serving

This page describes how `gmlx serve` runs image jobs beside chat replies
and plans their memory. The user side is in
[Image generation](../images.md) and the [Image API](../image-api.md).

## Image jobs beside chat

Image jobs run one at a time on their own thread. While a chat reply
streams, an image job pauses after each block of its model so chat steps
run in between. The reply keeps streaming at a steady pace, somewhat
slower than alone, and the image slows only while chat is streaming. With
no chat reply in progress, an image runs at its full speed.

By default chat gets half of the GPU while both run.
[`GMLX_IMAGE_CHAT_SHARE`](../env-vars.md#server) sets chat's share, from 0
to 0.9. At 0, chat runs one step after each block of the image model. `off`
runs each image step whole, which leaves chat waiting for each step to end.

## Memory plan of a request

Before a request queues, the server plans its memory at its size beside
the models that are loaded. The plan counts the weights, the step, the VAE
decode and the caches of the image model. A request that does not fit is
refused before it queues, with the [error](../image-api.md#errors) that
names what fits.

An entry whose text encoder and DiT do not fit side by side loads the
encoder for each new prompt, as [Models and
quants](../images.md#models-and-quants) describes. A prompt encoded before
comes from the cache, and nothing loads for it. On a 16 GB Mac the server
keeps smaller caches of prompts and request LoRAs.

## Step routes

The [image step routes](image-steps.md) run one denoise step at a time for
a sampler in another program, such as the ComfyUI node pack. Their steps
queue with the other image jobs, as [Access](image-steps.md#access)
describes.
