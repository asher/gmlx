# ComfyUI

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) is a free program for
making images that you use in a browser. You build a workflow in it from
nodes, boxes that each do one part of the work: load the model, read the
prompt, run the [steps](glossary.md#steps), turn the result into pixels and
save the image. The ComfyUI-gmlx node pack adds loaders that run the gmlx
[image models](images.md) on the gmlx server, so ComfyUI's own nodes work
with them. Like the rest of image generation, it needs a Mac with an M3
chip or newer.

- [Install ComfyUI and the pack](#install-comfyui-and-the-pack)
- [Make your first image](#make-your-first-image)
- [The nodes you use](#the-nodes-you-use)
- [The example workflows](#the-example-workflows)
- [LoRAs](#loras)
- [Launch options](#launch-options)
- [What the pack refuses](#what-the-pack-refuses)

## Install ComfyUI and the pack

ComfyUI is a separate program with its own Python environment.
`gmlx launch comfyui` runs a ComfyUI folder that you cloned, with that
environment in a `.venv` or `venv` folder inside it. These commands
install ComfyUI in `~/ComfyUI`, where `launch` looks for it, using
[uv](https://docs.astral.sh/uv/) for Python, then install the pack:

```sh
brew install uv
git clone https://github.com/comfyanonymous/ComfyUI ~/ComfyUI
cd ~/ComfyUI
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
git clone https://github.com/asher/ComfyUI-gmlx custom_nodes/ComfyUI-gmlx
```

ComfyUI's requirements include PyTorch, so the environment takes about
1.7 GB. The pack needs ComfyUI 0.37 or newer and no Python packages beyond
ComfyUI's own. At startup it checks the ComfyUI internals it uses, and when
one is missing, ComfyUI's log names it and the ComfyUI version.

## Make your first image

Download Z-Image-Turbo, then start ComfyUI:

```sh
gmlx pull z-image-turbo
gmlx launch comfyui
```

`pull` adds the model to your configuration, as
[Make your first image](images.md#make-your-first-image) describes.
`launch` starts the gmlx server when none is running, then starts ComfyUI
with the pack. Among its lines it prints:

```text
[launch] comfyui -> http://127.0.0.1:8080/v1  (1 model(s), default z-image-turbo)
[launch] ComfyUI runs at http://localhost:8188, which you open in a browser, from /Users/you/ComfyUI.
```

Open `http://localhost:8188` in a browser. In ComfyUI's Workflow menu,
choose Open and pick
`~/ComfyUI/custom_nodes/ComfyUI-gmlx/workflows/z-image-turbo-t2i.json`, or
drag that file onto the page. Press Run. The first run waits while the
server loads the model.

The image shows in the `Save Image` node, and ComfyUI saves it as
`~/ComfyUI/output/gmlx/z-image-turbo_00001_.png`, counting up with each
run. To make another image, change the text of the `Positive prompt` node
and press Run again.

Stop ComfyUI with Ctrl+C in the terminal where `launch` runs.

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

The pack has two more nodes. `gmlx Sigmas` gives the fixed schedule of a
distilled model such as Qwen-Image 2.1 Turbo, for `SamplerCustom` or
`SamplerCustomAdvanced`, though `KSampler` works on every model.
`gmlx Server Status` shows whether a model is loaded and the memory it
holds.

The sampler, the noise, the guidance and the schedule run in ComfyUI. Each
call to the model is one request to the gmlx server, which runs the DiT,
encodes the text and runs the VAE. The tiled VAE nodes call the same
route, since the server tiles large images itself. ComfyUI's interrupt
stops the step in progress on the server.

## The example workflows

The pack's `workflows` folder holds three workflows:

| Workflow | What it does | Needs |
|----------|--------------|-------|
| `z-image-turbo-t2i.json` | Z-Image-Turbo text to image at 1024x1024 in 8 steps | Nothing beyond the model |
| `z-image-turbo-lora-t2i.json` | The same through a stock `LoraLoader` | A Z-Image LoRA named `z-image-turbo-style.safetensors` in `models/loras`, or another picked in the node |
| `qwen-image-2.1-edit-two-references.json` | A Qwen-Image 2.1 edit from two reference images, through `TextEncodeQwenImage21` | `gmlx pull qwen-image`, and `reference-1.png` and `reference-2.png` in ComfyUI's `input` folder |

## LoRAs

Put a LoRA's `.safetensors` file in `~/ComfyUI/models/loras`. A file added
while ComfyUI runs shows in the lists after a restart.
[Use a LoRA](images.md#use-a-lora) says which LoRAs fit a model. The
`z-image-turbo-lora-t2i.json` workflow has a `Load LoRA` node to pick it in.

To add a LoRA to another workflow, double-click an empty part of the canvas,
search for `Load LoRA (Model and CLIP)`, which is `LoraLoader`, and add it.
Connect the MODEL and CLIP outputs of the gmlx loaders to its inputs, and
its outputs to the nodes that the loaders fed before. The pack sends each
LoRA to the server once, so a change of strength costs one small request.

`LoraLoader` works at any `strength_model`. The text encoder half of a
LoRA is not applied in ComfyUI, so set `strength_clip` to 0. To apply both
halves, use the LoRA with [`gmlx image generate`](images.md#use-a-lora).

## Launch options

```sh
gmlx launch comfyui --model qwen-image-turbo
gmlx launch comfyui --comfyui-dir ~/src/ComfyUI
gmlx launch comfyui -- --use-pytorch-cross-attention
```

`--model` picks the model the loaders show first. `--comfyui-dir` names
another ComfyUI folder, and arguments after `--` go to ComfyUI. `launch`
runs ComfyUI's `main.py` on the Python of that folder's `.venv` or `venv`.
ComfyUI opens on port 8188, or 8189 when gmlx uses 8188, and listens only
on `127.0.0.1`.

`launch` tells the pack where the server is through the environment:

| Variable | Set to |
|----------|--------|
| `GMLX_BASE_URL` | The server's address |
| `GMLX_API_KEY` | The server's API key, when it has one |
| `GMLX_DEFAULT_IMAGE_MODEL` | `--model`, or the server's default image model |

A ComfyUI started another way finds the server through the loader's
`server_url` input, then `GMLX_BASE_URL`, then `base_url` in
`~/.config/gmlx/comfyui.json` or the file that `GMLX_COMFY_SETTINGS`
names. After those it tries the address `gmlx status` reports, then
`http://127.0.0.1:8080`. With their `start_server` input on, the default,
the loaders start `gmlx serve` when no server answers.

## What the pack refuses

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
