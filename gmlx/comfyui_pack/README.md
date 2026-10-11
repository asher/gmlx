# gmlx nodes for ComfyUI

This ComfyUI node pack runs image model steps on a local gmlx server on Apple Silicon while the sampler stays in ComfyUI.
It ships inside gmlx and has gmlx's version.
The loaders give a stock MODEL, CLIP and VAE, so the stock sampler, LoRA, conditioning, latent and image nodes work with them.
The weights stay on the gmlx server, which holds the GGUF file once for ComfyUI and for its own API.

Supported model families:

- Z-Image (Z-Image-Turbo), text to image.
- Qwen-Image 2.1 and Qwen-Image 2.1 Turbo, text to image and editing with up to 10 reference images.
- Krea 2 Turbo, text to image.

## Install

Requirements:

- ComfyUI 0.37 or newer.
- gmlx, with image models in its config.

Put the pack into ComfyUI with gmlx, then restart ComfyUI:

```
gmlx comfyui install
```

The command finds ComfyUI, copies the pack to a folder gmlx owns and links it into ComfyUI's `custom_nodes` folder as `comfyui-gmlx`.
gmlx manages the pack, and a gmlx upgrade updates the copy. ComfyUI-Manager lists the pack as unknown and cannot update it, and Comfy Desktop does not list it.
The [ComfyUI guide](https://asher.github.io/gmlx/comfyui.html) covers the install, Comfy Desktop, the upgrades and the removal.

The pack needs no Python packages beyond those ComfyUI already has.
At startup it checks the ComfyUI internals it uses. When one is missing, the nodes load as `(not loaded)` stand-ins, and running one names the missing internal, the ComfyUI version, the newest ComfyUI this gmlx was tested on and the command that upgrades gmlx.

## Use

1. Add the image model to the gmlx config and run `gmlx serve`. The loaders also start `gmlx serve` when no server answers and `start_server` is on.
2. Add `gmlx Model Loader`, `gmlx CLIP Loader` and `gmlx VAE Loader`, and pick the model in `model_name` on each. The list holds the image models of the gmlx server, the default first, or the models `gmlx pull` downloads when it has none. A model the server does not have stops the run with the `gmlx pull` command that adds it.
3. Connect them to stock nodes as you would any checkpoint: `CLIPTextEncode`, `KSampler`, `VAEDecode`, `LoraLoader`, `ModelSamplingAuraFlow` and so on.

The `example_workflows` folder has three graphs, which ComfyUI's template browser lists under the pack:

- `z-image-turbo-t2i.json`: Z-Image-Turbo text to image.
- `z-image-turbo-lora-t2i.json`: Z-Image-Turbo text to image through a stock `LoraLoader`. Put a Z-Image LoRA named `z-image-turbo-style.safetensors` in `models/loras`, or pick another one in the node.
- `qwen-image-2.1-edit-two-references.json`: Qwen-Image 2.1 editing with two reference images through the stock `TextEncodeQwenImage21` node. Put `reference-1.png` and `reference-2.png` in ComfyUI's `input` folder, or pick other images.

### Server address

Each loader has a `server_url` input, which a new node fills with the first of:

1. the `GMLX_BASE_URL` environment variable;
2. `base_url` in `~/.config/gmlx/comfyui.json` (or the file `GMLX_COMFY_SETTINGS` names);
3. the address `gmlx status --json` reports, the port in the gmlx config when no server runs;
4. `http://127.0.0.1:8080`.

An empty `server_url`, or a `127.0.0.1` or `localhost` address that does not answer, uses that address when the node runs.
When `gmlx serve` fails to start, the run stops with what it printed, and Cancel stops the wait for a starting server.

When the server has an API key, set `GMLX_API_KEY`, or `api_key` in the settings file.
The pack and the server check that they speak the same step contract. On a mismatch, or when they come from two gmlx versions, the error or the `gmlx Server Status` node says whether to restart ComfyUI or the server.
The loaders start the server with the `gmlx` command that `gmlx comfyui install` recorded, then one on `PATH`. With neither, they say gmlx is not installed and name the two paths that remove the pack.

### LoRA

Stock `LoraLoader` and `LoraLoaderModelOnly` work. The pack sends each LoRA's matrices to the server once and then refers to them by hash, so changing a strength or stacking LoRAs costs one small request.
Plain LoRA pairs are applied, including those ComfyUI maps onto a band of a fused weight.
LoHa, LoKr, GLoRA, OFT, BOFT, DoRA, LoRAs with a mid tensor, model merges and `strength_model` values other than 1 on a patch are refused by name.
The text encoder part of a LoRA is not applied; set `strength_clip` to 0.

### Fixed schedules

Some variants, such as Qwen-Image 2.1 Turbo, are distilled to a fixed list of sigmas.
`gmlx Sigmas` outputs that list and the variant's cfg; connect them to `SamplerCustom` or `SamplerCustomAdvanced` with any stock sampler.
For a variant without a fixed list, it gives ComfyUI's simple schedule over the variant's step count.
`KSampler` with its own scheduler still works on every variant; it samples on ComfyUI's schedule instead.

### What runs where

The sampler, noise, cfg, guiders and schedules run in ComfyUI. Each model call is one request: the server runs the DiT forward with the LoRA table applied and returns the output ComfyUI expects.
Text encoding and VAE encode and decode run on the server too. The VAE tiles large images itself, so the tiled VAE nodes call the same route.

Anything that would run inside the DiT forward in ComfyUI is refused with an error that names it:

- model patches and block replacements (`FreeU`, `PatchModelAddDownscale`, `SkipLayerGuidanceDiT`, `PerturbedAttentionGuidance`, `TomePatchModel` and similar);
- attention overrides, RoPE options (`ScaleROPE`), weight wrappers and diffusion model wrappers;
- ControlNet;
- hook keyframes and scheduled CLIP hooks;
- a quantized Qwen-Image 2.1 prefix cache (`QwenImage21Cache` with a dtype other than default).

A refusal from a node's own setup is reported on that node. A refusal during sampling is reported on the sampler and names the node upstream that caused it.
Saving a gmlx MODEL, CLIP or VAE with the checkpoint save nodes is refused, since ComfyUI holds no weights for them.

### Interrupt

ComfyUI's interrupt stops the step in flight on the server within about a tenth of a second, and the next prompt runs normally.

## Tests

The pack's tests live in gmlx's `tests/comfyui_pack` folder and run against a stub server in `stub_server.py` there, with no gmlx server and no model files. With a ComfyUI checkout and its Python, from the gmlx checkout:

```
COMFYUI_DIR=/path/to/ComfyUI /path/to/ComfyUI/.venv/bin/python tests/comfyui_pack/run_tests.py --comfy /path/to/ComfyUI --label main --out /tmp/gmlx-tests
```

That runs the wire, client and in-process ComfyUI suites on the cpu device, then a headless ComfyUI with the pack against the stub.
`--mps` instead runs the suite that keeps a stock model resident on the MPS device while a gmlx model loads and samples.

### Parity with the stock loaders

Three checks hold a gmlx MODEL against ComfyUI's own model on the same graph, noise, sigmas and conditioning: ComfyUI-GGUF for Z-Image Turbo and for Krea 2 Turbo, and ComfyUI's Qwen-Image 2.1 model on a GGUF with its arch key set. They need the model files and both loaders, so they are not part of `run_tests.py`.

The step check is the gate. The stock model in bf16 samples and records, at each step, the latent it is fed and the denoised output it returns. The gmlx MODEL and the stock model in fp32 are then fed the same latents and sigmas. At every step the relative L2 of the gmlx output from the stock bf16 output must be at most 1.5 times that of the stock fp32 output, or at most 2e-2 where that is larger. A fixed 2e-2 cannot hold on its own: the stock fp32 output is 0.11 from the stock bf16 output at the first step of Z-Image Turbo.

| Model | Graph | gmlx from stock bf16, worst step | stock fp32 from stock bf16 | Worst ratio | Result |
|---|---|---|---|---|---|
| Z-Image Turbo | 1024x1024, 8 steps, stock LoRA loader | 0.115 | 0.111 | 1.14 | pass |
| Qwen-Image 2.1 | 512x512, 30 steps, cfg 4, two references | 0.106 | 0.113 | 1.27 | pass |
| Krea 2 Turbo | 1024x1024, 8 steps, cfg 1 | 0.055 | 0.095 | 0.72 | pass |

On Z-Image Turbo the gmlx output is closer to the stock fp32 output than the stock bf16 output is, at every step. The Qwen-Image 2.1 and Krea 2 checks run ComfyUI with `--use-pytorch-cross-attention`. Without it, ComfyUI's attention on the Mac adds the text mask as a bias, so the stock model departs from the reference implementation at high sigma, and the ratio at the first step is about 3.

The final image check is reported beside it: the gmlx image must be at least as close to the stock fp32 image as the stock bf16 image is, within 6 dB.

| Model | gmlx to stock fp32 | stock bf16 to stock fp32 | Result |
|---|---|---|---|
| Z-Image Turbo, 1024x1024, 8 steps | 25.25 dB | 20.30 dB | pass |
| Qwen-Image 2.1, 256x256, 30 steps, cfg 4, two references | 19.99 dB | 16.68 dB | pass |
| Krea 2 Turbo, 1024x1024, 8 steps | 21.61 dB | 22.10 dB | pass |

A fault planted in the Z-Image step convention shows why the step check is the gate. With the velocity sign dropped, the worst step ratio is 188 and the final image reads 8.35 dB. With the timestep read 3% low, the worst step ratio is 7.1, while the final image reads 13.49 dB against a bound of 14.30 dB and fails by only 0.8 dB, so a smaller timestep fault would pass the final image check.

## License

Apache-2.0. The pack imports ComfyUI at runtime and copies none of its code.
