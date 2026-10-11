# ComfyUI reference

This page has the details of the gmlx nodes for ComfyUI: what
`gmlx launch comfyui` does with its options, how the nodes find the server,
how they share the work with ComfyUI and what they refuse. The guide is
[ComfyUI](comfyui.md).

## Launch options

```sh
gmlx launch comfyui --model qwen-image-turbo
gmlx launch comfyui --comfyui-dir ~/src/ComfyUI
gmlx launch comfyui -- --use-pytorch-cross-attention
```

`--model` picks the model the loaders show first. `--comfyui-dir` names
another ComfyUI folder, and arguments after `--` go to ComfyUI. Without
`--comfyui-dir`, `launch` uses the folder it used last, then `~/ComfyUI`,
then the ComfyUI that `gmlx comfyui install` finds, where a ComfyUI it can
start comes before Comfy Desktop. It puts the gmlx nodes into that ComfyUI
as `gmlx comfyui install` does, then runs ComfyUI's `main.py` on the Python
of the folder's `.venv` or `venv`. ComfyUI opens on port 8188, or 8189 when
gmlx uses 8188, and listens only on `127.0.0.1`.

`launch` tells the gmlx nodes where the server is through the environment:

| Variable | Set to |
|----------|--------|
| `GMLX_BASE_URL` | The server's address |
| `GMLX_API_KEY` | The server's API key, when it has one |
| `GMLX_DEFAULT_IMAGE_MODEL` | `--model`, or the server's default image model |

## How the nodes find the server

A ComfyUI started another way, such as Comfy Desktop, finds the server
through the loader's `server_url` input, then `GMLX_BASE_URL`, then
`base_url` in `~/.config/gmlx/comfyui.json` or the file that
`GMLX_COMFY_SETTINGS` names. After those it tries the address
`gmlx status` reports, then `http://127.0.0.1:8080`. With their
`start_server` input on, the default, the loaders start `gmlx serve` when
no server answers.

Desktop starts ComfyUI without your shell's `PATH`, so the nodes run the
`gmlx` command whose path `gmlx comfyui install` recorded. After you
reinstall gmlx in another place, run `gmlx comfyui install` again.

## How the work is shared

ComfyUI sees each gmlx model as its own model class, so its stock nodes
for that model work too, such as `ModelSamplingAuraFlow` and
`TextEncodeQwenImage21`. The prompt nodes of the example workflows are
ComfyUI's `CLIP Text Encode (Prompt)`, and the latent node is its
`EmptySD3LatentImage`.

The sampler, the noise, the guidance and the schedule run in ComfyUI. Each
call to the model is one request to the gmlx server, which runs the DiT,
encodes the text and runs the VAE. The tiled VAE nodes call the same
route, since the server tiles large images itself. ComfyUI's interrupt
stops the step in progress on the server.

## Comfy Desktop installs

An install that Desktop migrated from the old desktop app still loads its
custom nodes from the old app's folder, so `gmlx comfyui install` links
the nodes there. A Desktop build that carries a signed policy file can
refuse custom nodes that its policy does not list, and `install` says so
when it finds one.

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
