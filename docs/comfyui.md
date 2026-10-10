# ComfyUI

The gmlx node pack runs the image models of a gmlx server inside ComfyUI
graphs. Its loaders give ComfyUI a model, a text encoder and a VAE, so the
stock sampler, LoRA, conditioning, latent and image nodes work with them,
while the weights stay on the server. With `gmlx serve` running, pull an
image model, install the pack, then launch ComfyUI against the server:

```sh
gmlx pull z-image-turbo
git clone https://github.com/asher/ComfyUI-gmlx ~/ComfyUI/custom_nodes/ComfyUI-gmlx
gmlx launch comfyui
```

`gmlx launch comfyui` prints where ComfyUI runs, then ComfyUI starts with
the pack loaded:

```text
[launch] comfyui -> http://127.0.0.1:8080/v1  (1 model(s), default z-image-turbo)
[launch] ComfyUI runs at http://localhost:8188, which you open in a browser, from ~/ComfyUI.
...
[INFO]    0.0 seconds: ~/ComfyUI/custom_nodes/ComfyUI-gmlx
...
[INFO] To see the GUI go to: http://127.0.0.1:8188
```

- [Install](#install)
- [Launch](#launch)
- [Build a graph](#build-a-graph)
- [The example workflows](#the-example-workflows)
- [What the pack refuses](#what-the-pack-refuses)
- [Comparing with ComfyUI's own nodes](#comparing-with-comfyuis-own-nodes)

## Install

The pack needs ComfyUI 0.37 or newer, installed with its own Python
environment in a `.venv` or `venv` folder, as ComfyUI's manual install
describes. `gmlx launch comfyui` looks for it in `~/ComfyUI`. The pack
needs no Python packages beyond those ComfyUI already has. Clone it into
ComfyUI's `custom_nodes` folder and restart ComfyUI:

```sh
cd ~/ComfyUI/custom_nodes
git clone https://github.com/asher/ComfyUI-gmlx
```

At startup the pack checks the ComfyUI internals it uses. When one is
missing, ComfyUI's log names it and the ComfyUI version.

## Launch

```sh
gmlx launch comfyui
gmlx launch comfyui --model qwen-image-turbo
gmlx launch comfyui --comfyui-dir ~/src/ComfyUI
```

`launch` starts the gmlx server when none answers, then runs ComfyUI's
`main.py` on the interpreter of the ComfyUI folder's `.venv` or `venv`, so
ComfyUI keeps its own packages. ComfyUI opens on port 8188, or 8189 when
gmlx uses 8188, and listens only on `127.0.0.1`. Arguments after `--` go
to ComfyUI.

The pack reads the server address, the API key and the model its loaders
show first from the environment, which `launch` sets:

| Variable | Set to |
|----------|--------|
| `GMLX_BASE_URL` | The server's address |
| `GMLX_API_KEY` | The server's API key, when it has one |
| `GMLX_DEFAULT_IMAGE_MODEL` | `--model`, or the server's default image model |

A ComfyUI started another way finds the server through the loader's
`server_url` input, then `GMLX_BASE_URL`, then `gmlx status`, then
`http://127.0.0.1:8080`. `launch` exits 2 without starting anything when
the pack is not installed, and prints the clone command.

## Build a graph

1. Add `gmlx Model Loader`, `gmlx CLIP Loader` and `gmlx VAE Loader`, and set
   `model_name` on each to an image model id from `gmlx list --image`.
2. Connect them to stock nodes as you would a checkpoint: `CLIPTextEncode`,
   `KSampler`, `VAEDecode`, `LoraLoader`, `ModelSamplingAuraFlow` and so on.
3. Queue the prompt.

The loaders take Z-Image, Qwen-Image 2.1 and Krea 2 models. ComfyUI
detects each one as its own model class, so its stock nodes for that
model work too.

The sampler, the noise, the guidance and the schedule run in ComfyUI. Each
model call is one request to the server, which runs the DiT with the LoRAs
applied, encodes the text, and runs the VAE. The tiled VAE nodes call the
same route, since the server tiles large images itself. ComfyUI's
interrupt stops the step in flight on the server.

`gmlx Sigmas` gives the fixed list of sigmas that a distilled variant, such
as Qwen-Image 2.1 Turbo, samples on, and its guidance. Connect it to
`SamplerCustom` or `SamplerCustomAdvanced`. `KSampler` with its own
scheduler works on every variant, on ComfyUI's schedule. `gmlx Server
Status` shows whether a model is loaded and the memory it holds.

Stock `LoraLoader` and `LoraLoaderModelOnly` work. Each LoRA matrix goes to
the server once, so a change of strength or another stack of LoRAs costs
one small request. The text encoder part of a LoRA is not applied, so set
`strength_clip` to 0.

## The example workflows

The pack's `workflows` folder holds two graphs in ComfyUI's API format.
Open them from the workflow menu, or queue them with the `/prompt` route.

| Workflow | What it does | Needs |
|----------|--------------|-------|
| `z-image-turbo-lora-t2i.json` | Z-Image-Turbo text to image at 1024x1024 in 8 steps, through a stock `LoraLoader` | A Z-Image LoRA named `z-image-turbo-style.safetensors` in `models/loras`, or another one picked in the node |
| `qwen-image-2.1-edit-two-references.json` | A Qwen-Image 2.1 edit from two reference images, through the stock `TextEncodeQwenImage21` node | `reference-1.png` and `reference-2.png` in ComfyUI's `input` folder, or other images picked in the node |

## What the pack refuses

Anything that would run inside the DiT forward in ComfyUI is refused with
an error that names it:

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
  model merges, and a patch `strength_model` other than 1.
- Saving a gmlx model, text encoder or VAE with the checkpoint save nodes,
  since ComfyUI holds no weights for them.

A refusal from a node's own setup shows on that node. A refusal during
sampling shows on the sampler and names the node upstream that caused it.

## Comparing with ComfyUI's own nodes

On a Mac, start ComfyUI with `--use-pytorch-cross-attention` when you
compare a gmlx model with ComfyUI's own Qwen-Image 2.1 nodes:

```sh
gmlx launch comfyui -- --use-pytorch-cross-attention
```

ComfyUI's default attention on a Mac adds the text mask of Qwen-Image 2.1
to the attention logits as a small bias, so text rows also see the later
text tokens, and ComfyUI's own model departs from the reference
implementation. PyTorch attention applies the mask as a mask.
