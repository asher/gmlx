# Video API

`gmlx serve` answers the OpenAI video routes with the video models of its
configuration. This page lists the routes, the fields a request takes, the
job it makes and the errors it can get, for an app or a script that makes
clips through the server. A first clip:

```sh
curl -s localhost:8080/v1/videos -H "Content-Type: application/json" \
  -d '{"model": "h3", "prompt": "waves on a pebble beach at dawn", "seconds": "5", "size": "768x768", "seed": 7}'
# {"id": "video_3f...", "object": "video", "status": "queued", ...}
curl -s localhost:8080/v1/videos/video_3f...
curl -s -o beach.mp4 localhost:8080/v1/videos/video_3f.../content
```

A clip takes minutes, so `POST /v1/videos` answers at once with a job,
which runs in the background. You read the job until it is `completed`,
then download its MP4.

A server whose configuration has a [video model](config.md#video-models)
serves these routes. They need ffmpeg, which writes the MP4, as
[Video generation](video.md#make-your-first-clip) describes.

- [Routes](#routes)
- [Request fields](#request-fields)
- [Keyframes and guides](#keyframes-and-guides)
- [The job](#the-job)
- [The job store](#the-job-store)
- [Errors](#errors)

## Routes

| Route | Body | Does |
|-------|------|------|
| `POST /v1/videos` | JSON or multipart form | Queues a job that makes a clip, and answers with the job |
| `GET /v1/videos/{id}` | None | The job: its status, progress and error |
| `GET /v1/videos/{id}/content` | None | The MP4 of a completed job |
| `DELETE /v1/videos/{id}` | None | Cancels a queued or running job, or removes a finished one and its MP4 |

## Request fields

| Field | Meaning |
|-------|---------|
| `prompt` | What the clip shows and sounds like. Required. |
| `model` | A `video_models` id. It may be left out when the configuration has one video model. |
| `seconds` | The length, up to 15, as a number or a string such as `"8"`, rounded up to [whole frames](video.md#clip-length-size-and-memory). Unset, the entry's, else 5. |
| `size` | `WxH` in multiples of 32. Unset, the entry's `defaults.size`, else `1344x768`. |
| `seed` | The seed of the noise, an integer under 2^32, random without it. |
| `prompt_upsampling` | `false` runs the prompt as typed on an entry with an [upsampler](video.md#prompt-upsampling), and `true` rewrites it on one with `upsample: false`. |
| `first_frame`, `input_reference` | An image the clip starts at. `input_reference` is the OpenAI name of the same field. |
| `last_frame` | An image the clip ends at. |
| `guides` | Images, videos and sounds anchored at frames of the clip, in a multipart form. |
| `guide_no_audio` | `true` anchors guide videos without their sound. |

The steps and the shifts come from the entry, as on the command line. A
`references` list asks for MiniMax-H3's Ref2VA mode, which the server does
not run, and gets 400. Other fields, `adapters` among them, are ignored,
and the log names them in a warning.

## Keyframes and guides

A JSON body takes `first_frame` and `last_frame` as image data URLs,
`data:image/png;base64,...`. The server never reads a local path or
fetches a URL. Guides need a multipart form. Each guide is a file field,
and the `guides` field lists them as JSON, each with its field name, the
frame it starts at, from the end when below 0, and its kind, `video` for
an image or a video with its sound, or `audio` for a sound alone:

```sh
curl -s localhost:8080/v1/videos \
  -F model=h3 -F prompt="the fox turns and runs into the snow" \
  -F first_frame=@fox.png -F guide_0=@steps.wav \
  -F 'guides=[{"file": "guide_0", "frame": 48, "kind": "audio"}]'
```

The anchors work as `--first-frame`, `--last-frame`, `--guide` and
`--guide-audio` do, as [Keyframes and guides](video.md#keyframes-and-guides)
describes, and take the memory listed there. A guide video is read a few
seconds at a time when its whole encode does not fit beside the models
loaded. FastH3 runs text to video only, so it refuses them.

## The job

The job is the OpenAI video object, with a `gmlx` part:

```json
{
  "id": "video_3f...",
  "object": "video",
  "model": "h3",
  "status": "in_progress",
  "progress": 47,
  "created_at": 1762400000,
  "completed_at": null,
  "expires_at": 1762486400,
  "seconds": "5.17",
  "size": "768x768",
  "prompt": "waves on a pebble beach at dawn",
  "error": null,
  "gmlx": {"frames": 124, "seed": 7, "steps": 8, "step": 4,
           "phase": "denoising", "revised_prompt": null, "eta_s": 210}
}
```

`status` runs from `queued` to `in_progress`, and ends `completed`,
`failed` or `cancelled`. `seconds` is the clip's length, its frames over 24.
`progress` is 0 until the prompt encode, 5 from it, counts the steps to 90
and the decode after them, and is 100 when the MP4 is written.
`gmlx.phase` names the work now running, `gmlx.step` the steps done, and
`gmlx.eta_s` the seconds the job expects to take still. `revised_prompt`
holds the upsampler's caption.

Jobs run one at a time, in order, on a thread of their own, so a clip
never holds up the image routes. Chat replies keep streaming while a clip
runs, since the clip pauses for them. On an M5 Max, beside a chat reply
that streams without a break, the steps of a 768x768 clip of 22 frames
took about 2.7 times as long as alone at the default share, and about
1.6 times at
[`GMLX_IMAGE_CHAT_SHARE`](env-vars.md#server) `0`, as
[Video jobs](internals/image-serving.md#video-jobs) describes.
When the memory a job needs is held by models in use, the job stays
`in_progress` with `gmlx.phase` `waiting for memory`, and tries again
once they finish. A failed job has `error` with its `code`, `message` and
the HTTP `status` the request would have had. A job still queued when
it expires never runs. It fails with `timed_out` and keeps its record as
any failed job does.

`DELETE` on a queued job cancels it at once. On a running job it answers
at once, and the job stops at its next step or decoded piece. A cancelled
job keeps its record until it expires, and a second `DELETE` removes it.

## The job store

The jobs, their inputs and their MP4s live under
`~/.cache/gmlx/videos/<port>/`. A finished job and its MP4 last
[`video.job_ttl_s`](config.md#video-jobs) seconds from when it finishes, a
day by default, and then read as 404. Past
[`video.store_max_bytes`](config.md#video-jobs) the oldest MP4s go first,
and their jobs with them. The files of a job's form go to the store as
they arrive, and so do the images of a JSON body's keyframes. A form
with a Content-Length takes its room before its first byte, and one sent
with no length takes it 64 MiB at a time as its bytes come. A job whose
files do not fit in the room left gets 413, and the message says how
much of the store the uploads arriving now hold. A server that restarts
fails the jobs it had queued or running with `server_restarted` and
status 500, and keeps the ones that ended.

## Errors

| Condition | Response |
|-----------|----------|
| The server cannot run ffmpeg | 501 naming ffmpeg and `brew install ffmpeg` |
| More jobs queued than [`video.max_queue`](config.md#video-jobs) | 429 `queue_full`, with a `Retry-After` of the median job time less the running job's time, at least 2, or 60 before the first clip ends |
| No `model` with several video models, or an unknown one | 400 naming the video models, or 404 `model_not_found` |
| The entry's file is no video DiT, its chip, an adapter or its defaults are refused, or its default clip does not fit the working budget | 400 `video_model_mismatch` |
| `seconds` past 15, or a `size` that is no multiple of 32 | 400 naming the field |
| A clip too large for the memory left beside the pinned models | 413 with the plan's phases and what to change |
| A JSON body that its read and parse cannot hold in the memory free | 413 naming both figures, before the body is read |
| A form's text fields past the memory free | 413 naming the memory free, once the text read passes it |
| A form's files larger than the room left in the job store | 413 naming both figures, before the files are read |
| A JSON keyframe larger than the room left in the job store | 413 naming the keyframe and both figures |
| A keyframe or guide image over 64 Mi pixels | 400 naming the field |
| A guide video whose frames are over 64 Mi pixels | The job fails with 400 naming the guide, before a frame is read |
| A guide video that does not fit in the clip from its frame | The job fails with 400 naming the guide |
| A guide video whose encode does not fit beside the models loaded, even a few seconds at a time | The job fails with 400 naming the guide: "... does not fit in memory beside the models now loaded: no window fits; the clip needs NEED, and FREE is free" |
| `references` | 400 naming Ref2VA |
| `prompt_upsampling: true` on an entry without an upsampler | 400 naming the `upsampler` setting |
| An unknown or expired job | 404 `not_found` |
| The content of a job not completed | 409 `not_ready` |
| A clip that does not fit beside the models loaded when it runs | The job fails with 400, as [this clip does not fit in memory](troubleshooting.md#this-clip-does-not-fit-in-memory) describes |
| Memory held by models in use past the job's expiry | The job fails with `timed_out` |
| A job still queued at its expiry | The job fails with `timed_out`, "the job was still queued when it expired" |
| The upsampler gives no caption for a job whose prompt it rewrites | The job fails with 500 `upsample_failed` |
| The server restarted during the job | The job fails with 500 `server_restarted` |

[Troubleshooting](troubleshooting.md#video-jobs) has the fix for each
message.
