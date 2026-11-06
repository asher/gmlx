# Video generation

gmlx makes short video clips with sound from a text prompt, with the
MiniMax-H3 model. A clip runs up to 15 seconds at 24 frames a second, and
the model makes the sound with the picture: speech, music and the noises
of the scene. A clip can also start or end at an image, or follow an
image, a video or a sound at a frame you pick. Video runs from the
command line, and in ComfyUI through the
[gmlx nodes](comfyui.md#make-a-video).

Video needs a Mac with an M3 chip or newer, 32 GB of memory or more,
ffmpeg, and about 40 GB of disk for the model's files.
[FastH3](#fasth3), a faster version of the model, needs an M5 chip or
newer.

- [Make your first clip](#make-your-first-clip)
- [Choose a recipe](#choose-a-recipe)
- [FastH3](#fasth3)
- [Clip length, size and memory](#clip-length-size-and-memory)
- [Dialogue](#dialogue)
- [Keyframes and guides](#keyframes-and-guides)
- [Use the preview decoder](#use-the-preview-decoder)
- [Frames without ffmpeg](#frames-without-ffmpeg)
- [Use a LoRA](#use-a-lora)
- [Licenses](#licenses)

## Make your first clip

gmlx writes the MP4 through ffmpeg. The Homebrew install of gmlx brings
it. With uv or pip, install it yourself:

```sh
brew install ffmpeg
```

Create a configuration with [`gmlx init`](cli.md#gmlx-init) if you have
none, then download the model:

```sh
gmlx video pull h3
```

`pull` lists the seven files it needs, where each one goes, the download
size and the licenses, then asks before it downloads:

```text
h3: MiniMax-H3 Turbo 8-step
  DiT              minimax_h3_fl2va_pruned-Q4_K.gguf
                   download 11.4 GB: /Users/you/models/unsloth__MiniMax-H3-GGUF/minimax_h3_fl2va_pruned-Q4_K.gguf
  ...
to download: 40.2 GB, 191.3 GB free
license: minimax-h3-community-license-agreement (https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
  ...
download? [y/N]
```

A file already in one of your model folders is used where it is, unless
it is a larger copy too big for your Mac's memory. When the downloads
finish, `pull` adds the model to your configuration as `h3`.
Then make a clip:

```sh
gmlx video generate h3 --size 768x768 --seed 7 --output musician.mp4 \
    --prompt "A street musician in a gray coat plays an acoustic guitar under a stone bridge on a rainy evening. Rain falls on the cobblestones, people with umbrellas walk past, and the warm guitar melody echoes under the arch."
```

`generate` prints the clip's settings, one line per step and one per
decoded piece of the clip:

```text
[video] MiniMax-H3 768x768, 124 frames, 8 steps at shift 6, seed 7
...
step 1 of 8, 49 s, about 345 s left
...
decoded 124 of 124 frames
wrote musician.mp4 in 427 s
```

Without `--output`, the clip goes to `gmlx-<time>-<seed>.mp4` in the
current folder. Without `--seed`, each run picks a seed and prints it.
Pass it back with `--seed` to make the same clip again.
[`gmlx video generate`](cli.md#video-generate) lists every flag.

## Choose a recipe

`pull` takes one of four recipes. The three `h3` recipes run the same
model files, so pulling a second one downloads at most one more LoRA.
`fasth3` runs a DiT file of its own with the same text encoder and VAEs.

| Recipe | Steps | Use |
|--------|-------|-----|
| `h3` | 8 | The default, with lightx2v's Turbo 8-step [LoRA](#licenses). |
| `h3-fast` | 4 | Drafts in about half the time on the Turbo 4-step [LoRA](#licenses), with softer motion. |
| `h3-base` | 20 | The model with no LoRA. Slower, and it fits larger clips. |
| `fasth3` | 8 | [FastH3](#fasth3), on an M5 or newer. Each step takes about half the time of an `h3` step. |

The recipe's name is the model's name in your configuration:

```sh
gmlx video pull h3-fast
gmlx video generate h3-fast --prompt "..."
```

On an M5 Max, an `h3` clip takes:

| Clip | Time |
|------|------|
| 768x768, 5 s | about 7 minutes |
| 1344x768, 5 s | about 17 minutes |
| 1344x768, 15 s | about 2 hours |

A Mac with fewer GPU cores takes longer, and a base M4 or M5 several
times as long.

## FastH3

```sh
gmlx video pull fasth3
gmlx video generate fasth3 --seed 7 --output harbor.mp4 \
    --prompt "A fishing boat comes into a small harbor at dawn. Gulls circle over the mast, the engine hums, and waves slap against the stone pier."
```

FastH3 is FastVideo's version of MiniMax-H3, trained to make a clip in 8
steps with video sparse attention: each part of the video attends in
full to the parts that matter most to it, and to a pooled summary of the
rest, so a step takes about half the time of an `h3` step. That
attention runs on the neural accelerators of the M5's GPU, so FastH3
needs an M5 chip or newer.

`pull` downloads vantagewithai's 13.6 GB GGUF of FastH3 and uses the text
encoder and the VAEs of the `h3` recipes, so with one of those pulled it
downloads that one file. The recipe has no LoRA. At 8 steps FastH3
samples the 8 noise levels it was trained at, at video shift 10:

```text
[video] MiniMax-H3 1344x768, 124 frames, 8 steps at shift 10, seed 7
```

With another `--steps` count it samples evenly spaced levels instead.

On an M5 Max, a 1344x768 clip of 5 seconds takes about 9 minutes with
FastH3, against about 17 with `h3`. That clip needs 36 GB of memory,
since sparse attention holds more than `h3`'s attention at that size. On
a 32 GB Mac, add `--size 768x768`: FastH3 makes 768x768 clips of 5
seconds there.

On an M3 or M4, `pull`, `generate` and `bench` stop before they download
or load anything:

```text
error: MiniMax-H3 fasth3 runs video sparse attention, whose kernel needs a GPU with NAX units (M5 or newer), and this Mac's GPU has none
```

In ComfyUI, the gmlx nodes run FastH3 through the `fasth3-t2v` workflow,
as [FastH3 in ComfyUI](comfyui.md#fasth3) shows.

## Clip length, size and memory

`--seconds` sets the length, up to 15, and `--size` the width and
height, in multiples of 32. A clip is 1344x768 for 5 seconds when you set
neither:

```sh
gmlx video generate h3 --size 768x1344 --seconds 10 --prompt "..."
```

This portrait clip needs 48 GB with `h3`, 36 GB with `h3-base`, or
64 GB with `fasth3`.

The Turbo LoRAs were made for a short side of 768, as in 1344x768,
768x1344 and 768x768. A longer or larger clip takes more memory, and its
time grows faster than its length: a 15 second clip takes about seven
times as long as a 5 second one at the same size.

The largest clip each Mac makes:

| Mac memory | `h3` and `h3-fast` | `h3-base` | `fasth3` |
|------------|--------------------|-----------|----------|
| 32 GB | 1344x768, 5 s | 1344x768, 5 s, or 768x768 up to 10 s | 768x768, 5 s |
| 36 GB | 1344x768, 5 s, or 768x768 up to 10 s | 1344x768 up to 10 s, or 768x768 up to 15 s | 1344x768, 5 s |
| 48 GB | 1344x768 up to 10 s, or 768x768 up to 15 s | 1344x768 up to 15 s | 1344x768, 5 s, or 768x768 up to 15 s |
| 64 GB or more | 1344x768 up to 15 s | 1344x768 up to 15 s | 1344x768 up to 15 s |

[Keyframes and guides](#keyframes-and-guides) take more memory. The
smallest Mac that makes a 5 second `h3` clip with them on the command
line, which holds no caches, for a prompt of up to 1,024 tokens beside its
images:

| Clip | 1344x768 | 768x768 |
|------|----------|---------|
| Text alone | 32 GB | 32 GB |
| A first frame or a last frame | 32 GB | 32 GB |
| A first frame and a last frame | 36 GB | 32 GB |
| An image guide | 32 GB | 32 GB |
| A 5-frame guide video with its sound | 32 GB | 32 GB |
| A 22-frame guide video with its sound | 36 GB | 32 GB |
| 5 seconds of guide sound | 32 GB | 32 GB |
| A 362-frame guide video with its sound, in a 15 second clip | 96 GB | 64 GB |

Through ComfyUI a step also holds the server's caches, so
[Make a video](comfyui.md#make-a-video) says what runs there.

A longer prompt takes more, and a clip that does not fit stops before it
loads a model, as [`this clip does not fit in
memory`](troubleshooting.md#this-clip-does-not-fit-in-memory) describes.

[`gmlx video bench`](cli.md#video-bench) measures the time and memory of
a few clip shapes on your Mac.

## Dialogue

Write each spoken line inside `<d>` and `</d>`, with its language in
brackets:

```sh
gmlx video generate h3 --prompt "A man in a grey raincoat stops under a streetlight and says <d>[English] We should have left an hour ago.</d> The woman beside him laughs and answers <d>[English] Then we run.</d>"
```

For a long prompt, put it in a file and pass `--prompt-file`.

## Keyframes and guides

`--first-frame` starts the clip at an image, and `--last-frame` ends it
at one:

```sh
gmlx video generate h3 --size 768x768 --first-frame fox.png --output fox.mp4 \
    --prompt "The red fox in the snow lifts its head, turns and runs off between the pines."
```

Give each image the clip's shape. The
[`video generate`](cli.md#video-generate) flags say how one of another
shape is fitted. The prompt reads both images through the text encoder's
vision tower, the `mmproj` file that `pull` downloads with the model, so
write what happens in the clip.

`--guide` anchors an image, or a video with its sound, at a frame of the
clip, and `--guide-audio` anchors a sound. The frame goes after an `@`,
counted from 0, or from the end of the clip when it is negative:

```sh
gmlx video generate h3 --guide street.png@60 --guide-audio rain.wav --prompt "..."
```

A guide video keeps 5, 22, 39 or more of its first frames, 17 k + 5 for
the largest k it holds, or its first frame alone when it has fewer than
5, and its sound unless you pass `--guide-no-audio`. Those frames must fit
in the clip from the guide's frame on.

To continue a clip, cut its last 22 frames with ffmpeg and guide a new
clip with them at frame 0:

```sh
ffmpeg -sseof -0.92 -i part1.mp4 part1-end.mp4
gmlx video generate h3 --guide part1-end.mp4@0 --output part2.mp4 --prompt "..."
```

`part2.mp4` starts with those 22 frames and their sound, so cut its first
0.92 seconds before you join the two. The memory the 22-frame guide
needs is in [Clip length, size and memory](#clip-length-size-and-memory).

The seed also sets the noise that gmlx mixes into each keyframe and
guide, so the same seed and the same files make the same clip again.
ComfyUI draws that noise another way, so the same seed there makes
another clip.

gmlx reads guide videos and sounds through ffmpeg, and images too when
ffmpeg is installed. Without it, gmlx reads images through Pillow, and a
JPEG can then decode a few levels of color away from ComfyUI's reading of
it.

FastH3 runs text to video only, so `fasth3` refuses keyframes and guides.

## Use the preview decoder

`--preview` decodes the frames with TAEH3, a small decoder that `pull`
downloads with the model, and gives softer detail. The full decoder
takes a small part of a clip's time, so the preview saves little of it.
For a fast draft, use `h3-fast`, which runs half the steps of `h3`:

```sh
gmlx video generate h3-fast --preview --seed 3 --prompt "..."
```

## Frames without ffmpeg

`--frames-dir` writes the clip as PNG frames and an `audio.wav` file.
Alone, it writes the clip without ffmpeg, though guide videos and sounds
are still read through it:

```sh
gmlx video generate h3 --frames-dir clip-frames --prompt "..."
```

With `--output` too, you get both the MP4 and the frames.

## Use a LoRA

`--adapter` applies a MiniMax-H3 LoRA on top of the recipe's, with an
optional strength after a colon:

```sh
gmlx video generate h3 --adapter ~/models/loras/style.safetensors:0.8 --prompt "..."
```

Repeat the flag to stack LoRAs. A LoRA takes memory about the size of
its file when it runs alone and its file is bf16, as the recipes' LoRAs
are, and about twice its size otherwise. An `--adapter` on `h3` or
`h3-fast` stacks on the recipe's LoRA, so both take twice. On a 32 GB
Mac, `gmlx video generate` has room in a 5 second clip at 768x768 for a
bf16 LoRA file of up to 5 GB beside `h3-base`, and up to 800 MB beside
`h3`. The [`adapters`](config.md#video_modelsadapters) key of the
model's entry applies LoRAs to every clip. A `gmlx serve` entry keeps
its DiT and LoRAs loaded through the decode, so on a 32 GB Mac an entry
whose default clip is 768x768 for 5 seconds has room in its `adapters`
for a bf16 LoRA file of up to 3 GB beside `h3-base`, and none beside
`h3`. A LoRA a request sends is booked when its step runs, in the memory
free beside the loaded entry: about 1.4 GB beside `h3-base`, and 1.2 GB
beside `h3`, where the LoRA stacks on the recipe's. One that does not
fit is refused with the memory it needs, as the
[LoRA refusal](troubleshooting.md#a-lora-is-refused-for-memory)
describes.

## Licenses

`pull` prints each file's license before it downloads. Read the MiniMax
H3 license before you use the clips.

| Files | License | Use |
|-------|---------|-----|
| MiniMax-H3, its video and audio VAEs | [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) | On the terms below |
| FastH3, in [vantagewithai's GGUF](https://huggingface.co/vantagewithai/FastVideo-FastH3-8-Step-V2-ComfyUI-GGUF) of FastVideo's release | The same license, as [FastVideo's card](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2) states | On the terms below |
| Qwen3-VL-32B-Instruct, the text encoder | Apache 2.0 | Any, under the license's terms |
| TAEH3, the preview decoder | MIT | Any, under the license's terms |
| lightx2v's Turbo LoRAs | None stated on the [model card](https://huggingface.co/lightx2v/Minimax-h3-Turbo) | See below |

Under the MiniMax H3 Community License Agreement:

- It grants nothing in the United States, the European Union, the United
  Kingdom or South Korea without a separate license from MiniMax.
- Use above $20 million of yearly revenue needs MiniMax's written
  authorization.
- A commercial user interface must show MiniMax H3.
- A hosted service needs terms of use and safeguards for its users.
- Clips you post publicly must be marked as made by a machine.

gmlx downloads a Turbo LoRA from its model card's repository only when
you pull `h3` or `h3-fast`, verifies the file against the revision the
recipe pins, and redistributes none of it. How you use the file is your
responsibility. `h3-base` and `fasth3` run with no LoRA, so you can make
clips without it.

gmlx has no content filter of its own.
