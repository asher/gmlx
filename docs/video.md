# Video generation

gmlx makes short video clips with sound from a text prompt, with the
MiniMax-H3 model. A clip runs up to 15 seconds at 24 frames a second, and
the model makes the sound with the picture: speech, music and the noises
of the scene. Video is a preview in this release and runs from the
command line only.

Video needs a Mac with an M3 chip or newer, 32 GB of memory or more,
ffmpeg, and about 40 GB of disk for the model's files.

- [Make your first clip](#make-your-first-clip)
- [Choose a recipe](#choose-a-recipe)
- [Clip length, size and memory](#clip-length-size-and-memory)
- [Dialogue](#dialogue)
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

On a 32 or 36 GB Mac, keep `--size 768x768` in every command on this
page. The default size, 1344x768, needs 48 GB with `h3`.

Without `--output`, the clip goes to `gmlx-<time>-<seed>.mp4` in the
current folder. Without `--seed`, each run picks a seed and prints it.
Pass it back with `--seed` to make the same clip again.
[`gmlx video generate`](cli.md#video-generate) lists every flag.

## Choose a recipe

`pull` takes one of three recipes. All three run the same model files, so
pulling a second one downloads at most one more LoRA.

| Recipe | Steps | Use |
|--------|-------|-----|
| `h3` | 8 | The default, with lightx2v's Turbo 8-step LoRA. |
| `h3-fast` | 4 | Drafts in about half the time, with softer motion. |
| `h3-base` | 20 | The model with no LoRA. Slower, and it fits larger clips. |

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

## Clip length, size and memory

`--seconds` sets the length, up to 15, and `--size` the width and
height, in multiples of 32. A clip is 1344x768 for 5 seconds when you set
neither:

```sh
gmlx video generate h3 --size 768x768 --seconds 10 --prompt "..."
```

This clip needs 48 GB with `h3`. A 32 GB Mac makes it with `h3-base`.

The Turbo LoRAs were made for a short side of 768, as in 1344x768,
768x1344 and 768x768. A longer or larger clip takes more memory, and its
time grows faster than its length: a 15 second clip takes about seven
times as long as a 5 second one at the same size.

The largest clip each Mac makes:

| Mac memory | `h3` and `h3-fast` | `h3-base` |
|------------|--------------------|-----------|
| 32 GB, 36 GB | 768x768, 5 s | 1344x768, 5 s, or 768x768 up to 10 s |
| 48 GB | 1344x768, 5 s, or 768x768 up to 15 s | 1344x768 up to 10 s, or 768x768 up to 15 s |
| 64 GB or more | 1344x768 up to 15 s | 1344x768 up to 15 s |

[`gmlx video bench`](cli.md#video-bench) measures the time and memory of
a few clip shapes on your Mac. On a 32 or 36 GB Mac, pass
`--shapes 512x512x22,768x768x124`.

## Dialogue

Write each spoken line inside `<d>` and `</d>`, with its language in
brackets:

```sh
gmlx video generate h3 --prompt "A man in a grey raincoat stops under a streetlight and says <d>[English] We should have left an hour ago.</d> The woman beside him laughs and answers <d>[English] Then we run.</d>"
```

For a long prompt, put it in a file and pass `--prompt-file`.

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
Alone, it needs no ffmpeg:

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

Repeat the flag to stack LoRAs. Each one takes memory, about twice the
size of its file. On a 32 GB Mac, a 5 second clip at 768x768 has room
for a LoRA file of up to 2 GB beside `h3-base`, and up to 250 MB beside
`h3`. The
[`adapters`](config.md#video_modelsadapters) key of the model's entry
applies LoRAs to every clip.

## Licenses

`pull` prints each file's license before it downloads. Read the MiniMax
H3 license before you use the clips.

| Files | License | Use |
|-------|---------|-----|
| MiniMax-H3, its video and audio VAEs | [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) | On the terms below |
| Qwen3-VL-32B-Instruct, the text encoder | Apache 2.0 | Any, under the license's terms |
| TAEH3, the preview decoder | MIT | Any, under the license's terms |
| lightx2v's Turbo LoRAs | No license stated | The [model card](https://huggingface.co/lightx2v/Minimax-h3-Turbo) states none |

Under the MiniMax H3 Community License Agreement:

- It grants nothing in the United States, the European Union, the United
  Kingdom or South Korea without a separate license from MiniMax.
- Use above $20 million of yearly revenue needs MiniMax's written
  authorization.
- A commercial user interface must show MiniMax H3.
- A hosted service needs terms of use and safeguards for its users.
- Clips you post publicly must be marked as made by a machine.

gmlx has no content filter of its own.
