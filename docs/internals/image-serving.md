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
The share holds for image jobs. A video clip pauses more often, and gives
chat more, as [Video jobs](#video-jobs) describes.

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
keeps a smaller cache of prompts. A request's LoRAs are booked when its
job runs on the image thread, before their files are read, and a request
whose LoRAs do not fit is refused with the memory they need.

## Step routes

The [image step routes](image-steps.md) run one denoise step at a time for
a sampler in another program, such as the ComfyUI node pack. Their steps
queue with the other image jobs, as [Access](image-steps.md#access)
describes.

## Video jobs

A video model's work runs on a thread of its own, the video worker, so a
clip's steps never queue behind an image job, and an image never waits
for a clip. The step routes on a video model run there too.

A job of [`POST /v1/videos`](../video-api.md) passes through two threads.
The job thread runs the jobs one at a time, in order, and runs no model.
It reads the keyframes and counts a guide video's frames, has the
upsampler write the caption, and loads the video model. The video worker
then encodes the prompt and the anchors, takes the job's LoRA bytes,
samples the clip and decodes it into the MP4. A guide video is read
through ffmpeg a few seconds at a time when its whole encode does not
fit, each part priced before it is read.

Chat replies keep streaming while a clip runs. The sampler pauses for
chat after each attention and each MLP chunk of a step, and the decode
after each piece it decodes, about 250 times in a step of a 768x768
clip. Each pause waits through chat steps until one ends past the
share's budget, and through one step at least. At the default share the
budget sets the wait, and the wait runs on to the end of the chat step
in flight, about 29 ms a pause against about 24 ms of clip work. At
[`GMLX_IMAGE_CHAT_SHARE`](../env-vars.md#server) `0` the one step is the
whole wait, about 9 ms. Beside a chat reply that streams without a
break, a 768x768 clip's steps take about 2.7 times as long as alone at
the default share, with the pauses 55% of the denoise, and about 1.6
times at `0`, with the pauses 29%.

With no chat reply in progress, a pause only checks for a cancel. The
sampler evaluates on the row budget `gmlx video generate` uses, which is
at each pause on a clip of 32,768 rows a step or more and every few
pauses on a smaller clip.

The request that queues a job plans its clip from the model's file
headers against the memory the pinned models leave, and books nothing.
The video worker checks the memory of each phase beside the models
loaded when the job runs, and books the job's LoRA bytes there, through
the path a model load takes. When models in use hold the memory, the job
waits on its own thread and runs the same phase again, the encoded
prompt kept.
