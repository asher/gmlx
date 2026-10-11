# ComfyUI

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) is a free program for
making images that you use in a browser. You build a workflow in it from
nodes, boxes that each do one part of the work: load the model, read the
prompt, run the [steps](glossary.md#steps), turn the result into pixels and
save the image. The gmlx nodes, a node pack that ships with gmlx, add
loaders that run the gmlx [image models](images.md) on the gmlx server, so
ComfyUI's own nodes work with them. Like the rest of image generation, it
needs a Mac with an M3 chip or newer.

- [Get ComfyUI](#get-comfyui)
- [Put the gmlx nodes into ComfyUI](#put-the-gmlx-nodes-into-comfyui)
- [Comfy Desktop](#comfy-desktop)
- [Make your first image](#make-your-first-image)
- [The nodes you use](#the-nodes-you-use)
- [The example workflows](#the-example-workflows)
- [LoRAs](#loras)
- [Upgrades](#upgrades)
- [Remove the gmlx nodes](#remove-the-gmlx-nodes)
- [Launch options](#launch-options)
- [What the gmlx nodes refuse](#what-the-gmlx-nodes-refuse)

## Get ComfyUI

ComfyUI is a separate program with its own Python environment. gmlx does
not install it. Get it one of two ways.

Comfy Desktop is ComfyUI as a Mac app. Download it from
[comfy.org/download](https://comfy.org/download), open it and create an
install. Desktop starts ComfyUI itself, so you start the app, not
`gmlx launch comfyui`.

A ComfyUI folder is a clone of ComfyUI with its environment in a `.venv`
folder, which `gmlx launch comfyui` starts for you. These commands put it
in `~/ComfyUI`, where `launch` looks for it, using
[uv](https://docs.astral.sh/uv/) for Python:

```sh
brew install uv
git clone https://github.com/comfyanonymous/ComfyUI ~/ComfyUI
cd ~/ComfyUI
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
```

ComfyUI's requirements include PyTorch, so the environment takes about
1.7 GB. The gmlx nodes need ComfyUI 0.37 or newer and no Python packages
beyond ComfyUI's own.

## Put the gmlx nodes into ComfyUI

```sh
gmlx comfyui install
```

`install` finds ComfyUI in these places:

| ComfyUI | Where its `custom_nodes` folder is |
|---------|------------------------------------|
| A clone | `~/ComfyUI/custom_nodes` |
| comfy-cli | `~/Documents/comfy/ComfyUI/custom_nodes`, or in the default or recent workspace that comfy-cli's config names |
| Comfy Desktop | `ComfyUI/custom_nodes` in each install's folder, which is in `~/ComfyUI-Installs` unless you moved it |
| The old ComfyUI desktop app | The `custom_nodes` folder in its base folder, by default `~/Documents/ComfyUI` |

With more than one, it lists them and asks for `--comfyui-dir`. Run it once
for each ComfyUI that should have the nodes:

```sh
gmlx comfyui install --comfyui-dir ~/ComfyUI
gmlx comfyui install --comfyui-dir ~/ComfyUI-Installs/Stable
```

It reads ComfyUI's version from `comfyui_version.py` and refuses a ComfyUI
older than 0.37, with the steps that update that kind of ComfyUI. Then it
copies the gmlx nodes to `~/.local/share/gmlx/comfyui` and links
`custom_nodes/comfyui-gmlx` to that copy, so every ComfyUI shares one copy.
A folder of that name that is not gmlx's link, such as a clone of the old
ComfyUI-gmlx repository, moves into `custom_nodes/.disabled`, where ComfyUI
skips it. Last, it records in `~/.config/gmlx/comfyui.json` the folders,
the path of the `gmlx` command, how gmlx was installed and your default
image model, which a new loader node starts on. Restart ComfyUI if it is
running, so it loads the nodes.

`gmlx comfyui status` shows where the nodes are, the version of each
ComfyUI and whether it is newer than the one this gmlx was tested on.
`gmlx doctor` checks the same. [gmlx comfyui](cli.md#gmlx-comfyui) lists
the flags.

## Comfy Desktop

`gmlx comfyui install` reads Comfy Desktop's list of installs, so it finds
an install wherever you put it. After it runs, restart the app, or start
it, from the Dock.

gmlx manages the gmlx nodes, and they update when gmlx does. They do not
show in Desktop's list of custom nodes or in its snapshots, and restoring a
snapshot leaves them in place. ComfyUI-Manager lists them as an unknown
pack and cannot update them, which is expected.

An install that Desktop migrated from the old desktop app still loads its
custom nodes from the old app's folder, so `install` links the nodes there.
A Desktop build that carries a signed policy file can refuse custom nodes
that its policy does not list, and `install` says so when it finds one.

Desktop starts ComfyUI without your shell's `PATH`, so the nodes run the
`gmlx` command whose path `install` recorded. After you reinstall gmlx in
another place, run `gmlx comfyui install` again.

## Make your first image

Download Z-Image-Turbo, then start ComfyUI:

```sh
gmlx pull z-image-turbo
gmlx launch comfyui
```

`pull` adds the model to your configuration, as
[Make your first image](images.md#make-your-first-image) describes.
`launch` puts the gmlx nodes into ComfyUI, starts the gmlx server when none
is running, then starts ComfyUI. Among its lines it prints:

```text
[launch] installed the gmlx nodes in ~/ComfyUI/custom_nodes/comfyui-gmlx (a ComfyUI folder, ComfyUI 0.39.2)
[launch] comfyui -> http://127.0.0.1:8080/v1  (1 model(s), default z-image-turbo)
[launch] ComfyUI runs at http://localhost:8188, which you open in a browser, from /Users/you/ComfyUI.
```

With Comfy Desktop, run `gmlx comfyui install` in place of `launch`, then
start the app. The loaders start the gmlx server when none is running.

Open ComfyUI in the browser, or in the app. In ComfyUI's template browser,
open `z-image-turbo-t2i` from the `comfyui-gmlx` templates, or drag
`custom_nodes/comfyui-gmlx/example_workflows/z-image-turbo-t2i.json` onto
the page. Press Run. The first run waits while the server loads the model.

The image shows in the `Save Image` node, and ComfyUI saves it as
`gmlx/z-image-turbo_00001_.png` in its `output` folder, counting up with
each run. To make another image, change the text of the `Positive prompt`
node and press Run again.

Stop ComfyUI with Ctrl+C in the terminal where `launch` runs. When `launch`
stops with an error, [Troubleshooting](troubleshooting.md#comfyui) has the
fix.

## The nodes you use

The first workflow has these nodes, under the titles they show:

| Node | What it does |
|------|--------------|
| `gmlx Model Loader` | Loads the [DiT](glossary.md#dit) of the image model that `model_name` names, an id from `gmlx list --image`. |
| `gmlx CLIP Loader` | Loads the model's [text encoder](glossary.md#text-encoder), which ComfyUI calls CLIP. |
| `gmlx VAE Loader` | Loads the model's [VAE](glossary.md#vae). |
| `Positive prompt` | What to make. |
| `Negative prompt` | What to steer away from. It has no effect at `cfg` 1. |
| `Empty latent` | The image's `width` and `height`. |
| `KSampler` | The `seed`, the `steps`, the `cfg` ([guidance](glossary.md#guidance)) and the [sampler](glossary.md#sampler). |
| `VAE Decode` | Turns the result into pixels. |
| `Save Image` | Shows the image and saves it in `output`, named by `filename_prefix`. |

The prompt nodes are ComfyUI's `CLIP Text Encode (Prompt)`, and the latent
node is its `EmptySD3LatentImage`. ComfyUI sees each gmlx model as its own
model class, so its stock nodes for that model work too, such as
`ModelSamplingAuraFlow` and `TextEncodeQwenImage21`.

There are two more gmlx nodes. `gmlx Sigmas` gives the fixed schedule of a
distilled model such as Qwen-Image 2.1 Turbo, for `SamplerCustom` or
`SamplerCustomAdvanced`, though `KSampler` works on every model.
`gmlx Server Status` shows whether a model is loaded, the memory it holds
and the gmlx version of the nodes.

The sampler, the noise, the guidance and the schedule run in ComfyUI. Each
call to the model is one request to the gmlx server, which runs the DiT,
encodes the text and runs the VAE. The tiled VAE nodes call the same
route, since the server tiles large images itself. ComfyUI's interrupt
stops the step in progress on the server.

## The example workflows

ComfyUI's template browser lists three workflows under `comfyui-gmlx`. Their
files are in `custom_nodes/comfyui-gmlx/example_workflows`.

| Workflow | What it does | Needs |
|----------|--------------|-------|
| `z-image-turbo-t2i` | Z-Image-Turbo text to image at 1024x1024 in 8 steps | Nothing beyond the model |
| `z-image-turbo-lora-t2i` | The same through a stock `LoraLoader` | A Z-Image LoRA named `z-image-turbo-style.safetensors` in `models/loras`, or another picked in the node |
| `qwen-image-2.1-edit-two-references` | A Qwen-Image 2.1 edit from two reference images, through `TextEncodeQwenImage21` | `gmlx pull qwen-image`, and `reference-1.png` and `reference-2.png` in ComfyUI's `input` folder |

Each workflow has a note with the commands that install the gmlx nodes and
pull its model, so a shared workflow tells its reader what it needs.

## LoRAs

Put a LoRA's `.safetensors` file in ComfyUI's `models/loras` folder. A file
added while ComfyUI runs shows in the lists after a restart.
[Use a LoRA](images.md#use-a-lora) says which LoRAs fit a model. The
`z-image-turbo-lora-t2i` workflow has a `Load LoRA` node to pick it in.

To add a LoRA to another workflow, double-click an empty part of the canvas,
search for `Load LoRA (Model and CLIP)`, which is `LoraLoader`, and add it.
Connect the MODEL and CLIP outputs of the gmlx loaders to its inputs, and
its outputs to the nodes that the loaders fed before. The gmlx nodes send
each LoRA to the server once, so a change of strength costs one small
request.

`LoraLoader` works at any `strength_model`. The text encoder half of a
LoRA is not applied in ComfyUI, so set `strength_clip` to 0. To apply both
halves, use the LoRA with [`gmlx image generate`](images.md#use-a-lora).

## Upgrades

The gmlx nodes come with gmlx, so an upgrade of gmlx upgrades them. The
next `gmlx serve`, `gmlx launch comfyui` or `gmlx comfyui install` replaces
the copy in one step, and the server prints a line when it does. A ComfyUI
that was running keeps the old nodes until you restart it. Until then,
`gmlx Server Status` says that the server and the nodes are from two gmlx
versions.

ComfyUI updates on its own schedule, and a new release can change a part
of ComfyUI that the nodes use. Then the nodes still show in your workflows
with `(not loaded)` after their names, and running one gives an error that
names the change, the newest ComfyUI this gmlx was tested on and the
command that upgrades gmlx, such as `brew upgrade gmlx`.

## Remove the gmlx nodes

```sh
gmlx comfyui uninstall
```

`uninstall` removes the `comfyui-gmlx` link from each ComfyUI, the copy in
`~/.local/share/gmlx/comfyui` and the record. With `--comfyui-dir` it
removes the nodes from that ComfyUI only. A folder that it moved into
`custom_nodes/.disabled` stays there.

If gmlx is already gone, the nodes say that gmlx is not installed and name
the two things to delete: the `comfyui-gmlx` link in `custom_nodes` and
`~/.local/share/gmlx/comfyui`.

## Launch options

```sh
gmlx launch comfyui --model qwen-image-turbo
gmlx launch comfyui --comfyui-dir ~/src/ComfyUI
gmlx launch comfyui -- --use-pytorch-cross-attention
```

`--model` picks the model the loaders show first. `--comfyui-dir` names
another ComfyUI folder, and arguments after `--` go to ComfyUI. Without
it, `launch` uses the folder it used last, then `~/ComfyUI`. It puts the
gmlx nodes into that folder as `gmlx comfyui install` does, then runs
ComfyUI's `main.py` on the Python of the folder's `.venv` or `venv`.
ComfyUI opens on port 8188, or 8189 when gmlx uses 8188, and listens only
on `127.0.0.1`.

`launch` tells the gmlx nodes where the server is through the environment:

| Variable | Set to |
|----------|--------|
| `GMLX_BASE_URL` | The server's address |
| `GMLX_API_KEY` | The server's API key, when it has one |
| `GMLX_DEFAULT_IMAGE_MODEL` | `--model`, or the server's default image model |

A ComfyUI started another way, such as Comfy Desktop, finds the server
through the loader's `server_url` input, then `GMLX_BASE_URL`, then
`base_url` in `~/.config/gmlx/comfyui.json` or the file that
`GMLX_COMFY_SETTINGS` names. After those it tries the address
`gmlx status` reports, then `http://127.0.0.1:8080`. With their
`start_server` input on, the default, the loaders start `gmlx serve` when
no server answers.

## What the gmlx nodes refuse

Anything that would run inside the DiT in ComfyUI is refused with an error
that names it:

- Model patches and block replacements, such as `FreeU`,
  `PatchModelAddDownscale`, `SkipLayerGuidanceDiT`,
  `PerturbedAttentionGuidance` and `TomePatchModel`.
- Attention overrides, RoPE options such as `ScaleROPE`, weight wrappers
  and diffusion model wrappers.
- ControlNet.
- Hook keyframes and scheduled CLIP hooks.
- A quantized Qwen-Image 2.1 prefix cache: `QwenImage21Cache` with a dtype
  other than the default.
- LoHa, LoKr, GLoRA, OFT, BOFT and DoRA files, LoRAs with a mid tensor,
  model merges, and any node that scales the model's own weights.
- Saving a gmlx model, text encoder or VAE with the checkpoint save nodes,
  since ComfyUI holds no weights for them.

A refusal from a node's own setup shows on that node. A refusal during
sampling shows on the sampler and names the node upstream that caused it.
