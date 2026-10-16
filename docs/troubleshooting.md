# Troubleshooting

Run [gmlx doctor](cli.md#gmlx-doctor) first. It checks each part of a
working setup and names the fix for each check that fails. For a coding
agent or chat app that does not connect, read its entry under
[The clients](launch.md#the-clients).

- [Installing](#installing)
- [Downloading and loading models](#downloading-and-loading-models)
- [Starting the server](#starting-the-server)
- [Requests](#requests)
- [Memory](#memory)
- [Image generation](#image-generation)
- [ComfyUI](#comfyui)
- [Container mode](#container-mode)
- [Custom agents](#custom-agents)
- [Voice](#voice)
- [Distillation](#distillation)
- [Logs and files](#logs-and-files)

## Installing

### The install fails on macOS before 26.2

The mlx-kquant kernels need macOS 26.2 or newer, and `gmlx doctor` warns on
an older version. Update macOS in System Settings, then install gmlx again.

### `gmlx: command not found` in a new terminal

A [pip](installation.md#pip) install lives in a Python venv, and each new
terminal starts with that venv inactive. Activate it again:

```sh
source <install dir>/.venv/bin/activate
```

Homebrew and `uv tool install` keep `gmlx` on PATH in every terminal.

## Downloading and loading models

### `not enough disk space`, or a download stopped partway

Run the same `gmlx pull` again. It resumes from the bytes already on disk.

- Not enough disk space: free some, pass `--to DIR` for another volume, or
  pass `--force` to skip the check.
- `stale partial download`: the repo replaced the file. Delete the `.part`
  file that the message names, and pull again.
- A flaky or slow host: raise `GMLX_PULL_RETRIES` or `GMLX_PULL_TIMEOUT`
  ([Command environment variables](env-vars.md#commands)).
- A load says `incomplete split GGUF` or `truncated GGUF`: pull again. A
  pull skips a file that exists, so delete a truncated file first.

### A gated or private repo will not download

Hugging Face answers 401 or 403. Accept the repo's terms on its Hugging Face
page, and check that your token has access to it. gmlx sends `HF_TOKEN`, or
the token that `hf auth login` stored.

### `GGUF architecture 'X' is not supported`

gmlx has no code for this model architecture.
[Supported architectures](arch-coverage.md) lists the ones it runs, and
`gmlx validate` shows a file's architecture before you download it.

### A tensor type is marked `<- no kernel`

The file uses a tensor type that gmlx cannot run, usually the plain ternary
`TQ1_0` or `TQ2_0`. Pick another quant from the same repo.
`gmlx validate hf:<org>/<repo>` lists the files without downloading, and a
uniform K-quant [decodes fastest](performance.md#choosing-a-quant-for-speed).

### A Hadamard-folded file refuses to load

gmlx runs a Hadamard-folded file only as a dense Qwen3.5, 3.6 or 3.8 model
at fold version 1, never as a drafter. Pick an unfolded quant. Before you
download, look for a `weights: Hadamard-folded` line in `gmlx validate`,
because its `loadable` verdict does not cover the fold.

## Starting the server

### `gmlx` no longer reads `./gmlx.yaml`

Move the file to `~/.config/gmlx/gmlx.yaml`. gmlx does not read a config
from the current folder, because a client in a container could write one
there. When a login item still points at the old file, the `login start`
row of `gmlx doctor` gives the commands that fix it.

### `gmlx status` reports 0 models served

The config lists no models, or the server skipped every entry. `gmlx logs`
shows a line for each skip:

- `[server] skipping model`: the entry's file is missing.
- `[server] model_dirs root missing`: a folder in
  [`server.model_dirs`](config.md#servermodel_dirs) does not exist. Use
  absolute paths.

### `binding 0.0.0.0 exposes this server beyond localhost`

A server that the network can reach needs a key in
[`server.api_key`](config.md#serverapi_key).

### `binding 0.0.0.0 exposes the assistant tool loop beyond localhost`

With [served assistants](assistant.md#served-assistants), anyone with the
API key could run tools on the Mac. gmlx refuses to bind a host other than
localhost unless you set
[`server.assistant_allow_remote`](config.md#serverassistant_allow_remote).

### Port 8080 is already in use

`gmlx status` shows whether a gmlx server holds the port, and `gmlx stop`
stops it. A server from `gmlx service install` starts again at login until
`gmlx service uninstall`. For another program, `lsof -i :8080` names it, or
serve on another port with `--port 8081`.

### The first request after startup is slow

That request loaded the model. To load a model at startup,
[pin](config.md#modelspin) it or list it in
[`server.defaults.preload`](config.md#serverdefaultspreload). A slow first
turn on a long prompt is prefill instead, which the
[prompt cache](prompt-cache.md) shortens on later turns.

## Requests

### 404 `model_not_found`

The server never downloads a model on a request. Use an id from
`gmlx list`, which the error also gives in `available_models`, or fetch the
model with `gmlx pull`. A file that `--to` saved outside your model folders
needs `gmlx sync-models --models-dir DIR` or a [`models`](config.md#models)
entry.

A 400 `unknown_profile` means that a suffix such as `@coding` names no
intent or profile. The error lists the valid names.

### 404 `model_file_missing`

The model's GGUF was deleted, moved or renamed, and the log shows
`[server] skipping model '<id>'`. Restore the file, and the id works again
with no restart. When the file is gone for good, `gmlx sync-models` removes
its entry.

A missing [`server.embeddings`](config.md#serverembeddings) or
[`server.rerank`](config.md#serverrerank) file turns that service off. Once
the file is back, reload the server or run `gmlx restart`.

### `the chat template of <file> drops message text`

Replies ignore what you sent, and a request with a system prompt can fail
with 500. For models that also take images, the server sends each message
as a list of parts, and this template renders only plain strings. Replace
it, usually with the base model's template, through `--chat-template` on
[gmlx serve](cli.md#gmlx-serve) or
[`chat_template`](config.md#profileschat_template) in a profile or in the
model's `overrides`.

### Requests get 503

A 503 carries a `Retry-After` header, and its error type says why:

| Type | Cause | What to do |
|------|-------|------------|
| `server_overloaded` | More requests wait than the queue cap allows. | Retry after the delay, or raise `GMLX_QUEUE_DEPTH_CAP`. |
| `model_load_deferred` | The model cannot load beside the busy resident models. | Retry after the delay, or keep fewer models resident. |

[Limits and back-pressure](api.md#limits-and-back-pressure) lists every
limit a request can hit.

### A request with media gets 400 or 413

The server takes media only inline or from its media folder, as
[Media in requests](api.md#media-in-requests) describes. `gmlx logs` shows
the message, with the command that copies a file into the media folder.

### `server_overloaded_shed`

A streamed reply ends early with an error of type `server_overloaded_shed`,
code `row_shed` and `finish_reason: shed`. The memory governor stopped this
request to keep the other requests running. Send it again, and read
[Memory](#memory) if it happens often.

### A web page or browser extension gets 403 or a CORS error

The page's origin is not in
[`server.cors_origins`](config.md#servercors_origins). `gmlx logs` names the
origin. Add it to `server.cors_origins` and run `gmlx restart`.

An extension's origin holds an ID that the browser gives it:

- Chrome and Edge show it on the extension's card in `chrome://extensions`
  or `edge://extensions`, with Developer mode on.
- Firefox shows it as the Internal UUID in `about:debugging`, under This
  Firefox.
- Safari changes it at every start, so list `safari-web-extension://*`.

Do not list a website's origin for an extension, because every page on that
site could then call the server. A page opened from disk sends
`Origin: null`, so serve it from `http://localhost:<port>` instead.

## Memory

### `cannot fit:`, `out of GPU memory mid-run`, or the Mac swaps

The weights plus the KV cache need more memory than the GPU may use.
[Memory and the KV cache](memory.md) shows how to estimate both. To fix it:

- Use a [quantized KV cache](kv-quantization.md).
- Use a smaller context or a smaller quant.
- [Stream](streaming.md) a MoE model that is larger than memory.
- On a server with several models, lower
  [`server.budget_gb`](config.md#serverbudget_gb) or
  [`server.max_models`](config.md#servermax_models).

To try anyway on `run` and `chat`, set
[`GMLX_TOOL_PREFLIGHT=0`](env-vars.md#commands).

## Image generation

### `image request does not fit`

The image at this size needs more memory than gmlx may use, and the message
names the largest part. Try these in order:

- A smaller image, such as `--size 768x768` on `gmlx image generate`, or
  `"size": "768x768"` in a request.
- A smaller DiT quant. `gmlx pull z-image-turbo --quant Q3_K_M` downloads
  it and adds it as a new model, `z-image-turbo-q3-k-m`.
- A smaller text encoder as well, with `--encoder-quant Q4_K_M` on that
  pull.

`GMLX_OVERCOMMIT=1` skips the check. The run then takes more memory than
macOS can give it, so the Mac swaps and slows down. On a 16 GB Mac it can
stop responding until it restarts.

### `VAE decode of ... needs ... GiB, and ... GiB is free`

The last step, which turns the result into pixels, does not fit in the
memory that is free at this size. A `VAE encode` message is the same check
for the image an edit starts from. Make a smaller image, or quit apps that
hold a lot of memory. `GMLX_OVERCOMMIT=1` overrides it, with the risk the
entry above describes.

### `The ... weights of ... GiB exceed the working budget of ... GiB`

The model's files alone are larger than the memory gmlx may use, which
happens with a quant meant for a Mac with more memory. Pull the model
without `--quant` to get the quant for this Mac, or a smaller one, as
[`image request does not fit`](#image-request-does-not-fit) shows.

### `size WxH does not fit in memory beside the models now loaded`

The server answers 400 when an image of that size does not fit beside the
chat models it has loaded. The message names the largest size of the same
shape that fits. Send that size, leave out `size` to get the model's
default, or unload chat models with [`POST /unload`](api.md#endpoints).

### `needs an M3 or newer`

`gmlx pull` refuses an image model on an M1 or M2 Mac before it downloads
anything, and `--force` downloads the files anyway. The models run in
bfloat16, which M1 and M2 chips lack. On those Macs `gmlx image generate`
and the server refuse Qwen-Image 2.1, Krea 2, FLUX.2 klein and FLUX.3
klein with `needs bfloat16, which the M1 chip lacks`. Z-Image-Turbo can
run in float16, as the next entry describes.

### `runs in float16 on the M1 chip, which misses the reference checks`

Z-Image-Turbo can run in float16 on an M1 or M2, where the other image
models cannot. Its images then differ from those of an M3 or newer, because
float16 misses the reference checks. With
[`GMLX_IMAGE_ALLOW_FP16=1`](env-vars.md#runtime) set, `gmlx pull`,
`gmlx image generate` and the server run it that way.

### `no such file, and no image model of that id`

`gmlx image generate` takes the id of an image model in your configuration,
or the path of a DiT file. The message lists the ids the configuration has,
and `gmlx list --image` lists them too. `gmlx pull z-image-turbo` adds one.

### `--encoder and --vae are required with a DiT path`

Given the path of a DiT file, `generate` also needs the text encoder and VAE
files. Use the id that `gmlx pull` added instead, which brings both.

### `no image model is configured on this server`

The server has no image model, and answers 404 `model_not_found`.
`gmlx pull z-image-turbo` downloads one and adds it to the running server.

### `is not an image model` or `is an image model`

The request sent a chat model to an image route, or an image model to a
chat route, and got 400 `wrong_model_kind`. The message lists the image
models, or names the route the model takes.

### 400 `image_model_mismatch`

The text encoder, VAE or mmproj of an image model's entry belongs to
another model, and the message names the part. Set the entry's `encoder`,
`vae` or `encoder_mmproj` to the file made for this model, or pull the
model by name, which takes the parts that match.

### `is not the ... VAE, and a download would land on it`

`gmlx pull` found another file where it would save one of the model's parts.
Move or delete that file, and pull again.

### `has sha256 ..., and the ... VAE at ... has ...`

The VAE that `gmlx pull` downloaded is not the published file. Delete the
file that the message names, and pull again.

### `is a flow map, which samples without guidance: drop cfg ...` or `is a flow map, which samples with euler only, not ...`

FLUX.3 klein is distilled to sample without guidance and with the `euler`
sampler alone. Leave out `--guidance` and `--sampler`, the request's
`guidance` and `sampler`, and the entry's `defaults.guidance` and
`defaults.sampler`, or set them to 1 and `euler`. FLUX.3 klein Base takes
other values of both.

### `distilled DiT samples without guidance: drop guidance ...`

FLUX.2 klein is distilled to sample without guidance. Leave out
`--guidance`, the request's `guidance` and the entry's
`defaults.guidance`, or set them to 1. FLUX.2 klein Base takes guidance.

### `gmlx image does not apply` or `is not a ... adapter`

The LoRA is of a kind gmlx does not apply, or it was made for another
model. [Use a LoRA](images.md#use-a-lora) says which LoRAs work.

### `gmlx applies them and ComfyUI drops them`

The LoRA loads in `gmlx image`, and ComfyUI's LoraLoader skips its keys,
so ComfyUI makes the image without it. A FLUX.3 klein LoRA trained with
diffusers is one: ComfyUI reads FLUX.3 keys only in the GGUF names, under
`lora_unet_` or `diffusion_model.`. A FLUX.2 klein LoRA in diffusers names
that trains the modulation layers or the time embedder is another:
ComfyUI has no diffusers name for those layers, so it drops those pairs
and applies the rest.

### `has text encoder layers; set it on the image model's adapters instead`

A LoRA in a request applies to the DiT only, so the server refuses one that
also trains the text encoder. List it in the image model's
[`adapters`](config.md#image_modelsadapters), where both halves apply.

### `never an absolute path` or `is not a configured adapter or a file in a model folder`

A LoRA in a request is named by a path inside one of your model folders,
such as `loras/watercolor.safetensors`, or by the file name of one in the
entry's `adapters`. Move the file into a model folder and use its path
there.

## ComfyUI

### `no ComfyUI found`

`gmlx comfyui install` and `gmlx launch comfyui` found no ComfyUI in the
places that [Put the gmlx nodes into ComfyUI](comfyui.md#put-the-gmlx-nodes-into-comfyui)
lists. Get ComfyUI as [Get ComfyUI](comfyui.md#get-comfyui) shows, or pass
`--comfyui-dir` with the folder that holds it.

### `found ... ComfyUI installs`

More than one ComfyUI is on this Mac. Run the command again with
`--comfyui-dir` and one of the folders it lists, once for each ComfyUI that
should have the gmlx nodes.

### `is not a ComfyUI folder`

The folder that `--comfyui-dir` names has neither ComfyUI's `main.py` nor a
`custom_nodes` folder. Pass the folder you cloned ComfyUI into, or a Comfy
Desktop install folder.

### `comfyui.json is not a JSON object`

The settings file that the gmlx nodes read, `~/.config/gmlx/comfyui.json`
or the file that `GMLX_COMFY_SETTINGS` names, does not parse, often after
a hand edit left a trailing comma. It may hold your own `base_url` and
`api_key`, so `gmlx comfyui` leaves it as it is. Fix the file, or move it
aside, and run the command again.

### `has no Python environment of its own`

`gmlx launch comfyui` runs ComfyUI on the Python of a `.venv` or `venv`
folder inside it. Create one with the commands the message prints.

### `is older than ..., the oldest the gmlx nodes run on`

The gmlx nodes need a newer ComfyUI. The message gives the steps for that
kind of ComfyUI, such as `git -C ~/ComfyUI pull` for a clone. Update it,
then run `gmlx comfyui install` again. The old ComfyUI desktop app cannot
update that far, so install Comfy Desktop in its place.

### `is older than ..., which the nodes need`

`gmlx comfyui status` and `gmlx doctor` found a ComfyUI that has the gmlx
nodes but is older than they need, such as after a gmlx upgrade raised the
oldest ComfyUI they run on. The nodes may show as `(not loaded)` there.
Update that ComfyUI as the entry above describes, then restart it.

### `the ComfyUI version in ... is unknown`

The ComfyUI folder has no `comfyui_version.py`, so `install` cannot check
the version. The nodes are installed, and they load on ComfyUI 0.37 or
newer.

### `is newer than ..., the newest this gmlx was tested on`

The nodes are installed and usually work. When a change in that ComfyUI
breaks them, they show as `(not loaded)`, as the entry below describes.

### `carries a signed policy`

A Comfy Desktop build with a signed policy can refuse custom nodes that the
policy does not list. If the gmlx nodes do not show in ComfyUI, ask whoever
manages the build to allow them, or use a Comfy Desktop install without a
policy.

### `the folder that was there moved to`

`custom_nodes` held a folder named `comfyui-gmlx`, such as a clone of the
old ComfyUI-gmlx repository. `install` moved it into `custom_nodes/.disabled`,
where ComfyUI skips it. Delete it when you no longer need it.

### `runs its own ComfyUI, so launch does not start it`

The ComfyUI that `launch` found is a Comfy Desktop install or the old
desktop app. `launch` put the gmlx nodes into it. Start the app, or restart
it, from the Dock.

### `has no main.py, so launch cannot start ComfyUI from it`

`launch` put the gmlx nodes into that folder, but it holds no ComfyUI to
start. Start that ComfyUI the way you installed it, or pass
`--comfyui-dir` with the folder that holds `main.py`.

### `do not load on ComfyUI`, and the gmlx nodes show `(not loaded)`

A ComfyUI update changed a part of ComfyUI that this version of the nodes
uses. Running one of them gives this error, which names the change and the
command that upgrades gmlx, such as `brew upgrade gmlx`. Upgrade, run
`gmlx comfyui install` and restart ComfyUI. When the error says the nodes
need a newer ComfyUI, update ComfyUI in its place and restart it. The
terminal or log of ComfyUI holds the full error.

### `the server speaks step contract`, `the server names no step contract` or `the server needs a client of step contract`

The server and the gmlx nodes come from gmlx versions too far apart to
work together. The message names the step to take. When the server is the
older one, restart it with `gmlx restart`, and upgrade gmlx if it stays
older. When the nodes are older, restart ComfyUI so it loads the nodes of
the gmlx the server runs, and run `gmlx comfyui install` if they stay
older.

### `the server runs gmlx ... and these nodes are from gmlx ...`

The server and the nodes come from two gmlx versions. When the server is
newer, restart ComfyUI so it loads the new nodes, and run
`gmlx comfyui install` if they stay old. When the nodes are newer, run
`gmlx restart`.

### `the gmlx server at ... has no image model`

The loader names a model that the server's config does not have, such as
the model of an example workflow you have not pulled. Run the `gmlx pull`
command that the message gives, then run the workflow again. When the
message says the server has that model under another id, pick that id in
the loader's `model_name` list. To use a model you already have, pick it
in the list.

### `gmlx serve` failed (exit ...)

The loader started `gmlx serve` because no server answered, and the server
stopped before it was ready. The message ends with what `gmlx serve`
printed and the end of the server's log, which name the cause. A message
that says the server stopped before it answered has the same cause. Fix
that, then run the workflow again, or start the server yourself with
`gmlx serve` to see the whole output.

### `a server answers at ..., but it is not a gmlx image server`

Another server, such as llama-server, listens at the address the loader
uses and has no gmlx image step routes, or its answer to the handshake is
not a gmlx one. Set the loader's `server_url` to the gmlx server's
address, which `gmlx status` shows. When the address is a gmlx server,
its `server.image.steps_api` is off: turn it on and run `gmlx restart`.

### `the capabilities handshake was refused (401)` or `(403)`

A gmlx server answered and refused the nodes. With 401, the server has an
API key: set `GMLX_API_KEY`, or `api_key` in the nodes' settings file.
With 403, the server takes step requests from its own Mac only: run
ComfyUI there, or set `server.image.steps_remote` in the server's config.

### `gmlx is not installed`

The nodes found neither the `gmlx` command that `install` recorded nor one
on `PATH`, or the one they found does not run. Install gmlx again and run
`gmlx comfyui install`, or remove the nodes by deleting the two paths the
message names: the `comfyui-gmlx` link in `custom_nodes` and
`~/.local/share/gmlx/comfyui`.

### `the pack copy is missing` or `the pack copy is from gmlx`

`gmlx comfyui status` and `gmlx doctor` found the copy of the nodes gone or
from another gmlx version than the one that runs. Run
`gmlx comfyui install`.

### `folder missing; run gmlx comfyui install`

A ComfyUI that the record lists is gone: deleted, moved, or on a drive that
is not mounted. A drive that comes back needs nothing, since its link
points at the copy that gmlx keeps current. For a ComfyUI that is gone for
good, run `gmlx comfyui install`. It removes the folder from the record and
says so, then installs into the other ComfyUI folders it has or finds.
When it finds no other ComfyUI, it stops with `no ComfyUI found`, and when
it finds several, it lists them. The record is clean either way, so status
and `gmlx doctor` stop naming the folder. Running
`gmlx comfyui uninstall --comfyui-dir DIR` with its path removes it from
the record without installing anything. If the folder comes back, run
`gmlx comfyui install --comfyui-dir DIR` to record it again.

### `missing; run gmlx comfyui install`, `elsewhere; run gmlx comfyui install` or `folder; run gmlx comfyui install`

The `comfyui-gmlx` entry in that ComfyUI's `custom_nodes` is gone
(`missing`), links to something other than the gmlx copy (`elsewhere`), or
is a real folder (`folder`), such as one a hand copy put there.
`gmlx comfyui install` puts the link back and moves such a folder into
`custom_nodes/.disabled`.

### ComfyUI-Manager says `Not a git repository`

ComfyUI-Manager lists the gmlx nodes as an unknown pack, and updating them
there fails, since they are a link to the copy that gmlx keeps, not a git
clone. They update with gmlx: upgrade gmlx, and the next `gmlx serve`,
`gmlx launch comfyui` or `gmlx comfyui install` brings the new nodes.
Restart ComfyUI after that.

### `could not refresh the ComfyUI node pack`

The server could not replace the copy of the nodes, and it started anyway.
The message names the cause, such as a full disk. Fix it, then run
`gmlx comfyui install`.

### `replaced the gmlx nodes from gmlx ... with those of the older gmlx`

An older gmlx ran `install` or a server start after a newer one, so the
nodes are now the older version. Run the newer gmlx, or upgrade.

### `the server offers no image model`

`gmlx launch comfyui` found a server with no image model.
`gmlx pull z-image-turbo` downloads one and adds it to the running server.

## Container mode

### A launch stops on a malformed launch block

The `launch` block of your config does not load, and the message names the
file and the problem. Fix the block, or pass `--no-container` to run the
client on the Mac meanwhile. The server ignores a broken `launch` block and
loads the rest of the file.

### Launch refuses a mount through a symbolic link

A `--mount` or [`mounts`](config.md#launchcontainermounts) path is, or passes
through, a symbolic link. Write the real path that the message gives
instead, for example `/private/tmp/x` for `/tmp/x`.

### No Mac port is free for a browser app

`no Mac port from 3100 to 3199 is free` means other projects keep every
browser app port. The message names projects you used longest ago, each with
the `--remove-home` command that frees its port. Run one, or stop another
program that uses a port, and launch again.

### Launch says Apple container has no Linux kernel

Apple container needs a Linux kernel, about 700 MB, before any container can
start. A launch in a terminal asks to download it. A launch from a script
has no terminal to ask in, so it stops and names one command to run:

- `container system start --enable-kernel-install` when the container
  service is not running;
- `container system kernel set --recommended` when it runs.

Run it, then launch again. If you answered no, or the download failed, the
next launch in a terminal asks again.

### A container command gave no answer

The container service is stuck. Run `container system stop`, then
`container system start`, and try again.

### A leftover container of another session keeps running

A launch that was killed left its container running, and the message gives
the `container stop` command. The container holds its memory until it
stops, so run that command. `gmlx launch --list` lists these containers too.

### A volume is in use

[A volume serves one container at a time](container-access.md#volumes).
Stop the other session first. To give each project its own volume, list the
volume under the client rather than directly under `launch.container`.

### Apple's image builder cannot start without Rosetta

Apple container starts its image builder with Rosetta by default, and this
Mac has no Rosetta. `launch` turns that setting off in
`~/.config/container/config.toml`. Run `container system stop` so the
service reads the change, and launch again.

### Launch refuses to build while the builder forwards your SSH agent

The image builder was started with your SSH agent, which every build could
use. Run `container builder stop`, and the next launch starts a builder
without it.

### The image build fails

One step of the image build failed, and the build output above the message
shows which. For a package in
[`packages`](config.md#launchcontainerclientspackages), check that Debian 13
has a package by that name. For a step in your own Containerfile, fix the
file. Otherwise launch again with `--rebuild`.

### The image build cannot reach the network

A VPN that sends all traffic through its tunnel is the usual cause. The Mac
stays online, but containers get no connection out. Disconnect the VPN, or
turn on its setting that allows local network access, and launch again. A
running client also has no internet while the VPN is on, but it still
reaches the gmlx server.

### An image has no `linux/arm64` variant

Container mode runs only Linux on arm64 images. Use an arm64 or
multi-platform tag, or build one with
[`build`](config.md#launchcontainerclientsbuild).

### A command is not in the image

The launch stops with one of these messages:

```text
[launch] <name> is not on the image's PATH (<search path>). Install it in the image, or set launch.container.clients.<client>.command.
[launch] <path> is not an executable file in this image. The image needs the command it runs.
```

The image lacks the client or the command in
[`command`](config.md#launchcontainerclientscommand). Install it in the
image, as [Custom container images](container-images.md) shows, or fix the
`command` list. The second form is for a command written as a path. Other
messages name a specific problem:

| Message | Fix |
|---------|-----|
| `has no execute bit` | Add `RUN chmod 755 <file>` to the Containerfile. |
| `names X in its #! line, which is not in the image` | Install the interpreter, or change the script's `#!` line. |
| `ends in a carriage return` | Convert the script to Unix line endings, for example with `dos2unix`. |
| `env receives it as one command name` | Write `#!/usr/bin/env -S tool --flag`. |
| `Exec format error` | Install an arm64 build, or add a `#!` line to the script. |
| `its #! interpreter or its program loader is not in the image` | The program is built for another system, such as glibc in a musl image. |

### Launch will not follow a file in the private home

The client can put links in its [private home](container-access.md#the-private-home),
so `launch` never follows one there. Delete the path the message names, or
remove the home with `gmlx launch <client> --remove-home`, and launch again.

### A pasted file path reaches the client unchanged

`launch` left the path as it was. The session log,
`~/.cache/gmlx/launch/last-<client>-<project>.log`, has a `paste:` line with
the reason. Folders, links, files in protected folders and paths inside
other text stay as they are, as
[Pasting files and images](container-access.md#pasting-files-and-images)
lists. If macOS denied your terminal app access to the file, allow it in
Privacy & Security in System Settings and paste again.

### The clipboard image paste fails in a container

The client read the clipboard without a press of its paste key first. Press
Ctrl-V, or Alt-V in hermes, in the session's terminal, and paste within 10
seconds. A `--shell` or a joined copy started without a terminal has no
clipboard access.

### A container launch waits with no output

macOS is asking whether the container may read a protected folder, such as
`~/Documents`. Look for the prompt behind other windows, or launch from a
folder outside the protected places.

### A request from a container gets 403 peer_not_allowed

An Apple container localhost domain sends container traffic to the Mac's
`127.0.0.1`, and the server refuses it. Launch sessions need no such domain,
so remove it with `sudo container system dns delete <domain>`.

### The Mac runs out of file handles

A session that reads a very large shared folder holds a Mac file handle for
each file. Stop the session to release them, and share a narrower folder.

### Postgres refuses the data folder on a share

Every file in a share belongs to root in the container, which Postgres does
not accept. Put the data on a volume, as the
[Postgres](container-recipes.md#postgres) recipe shows.

### Packages installed in `--shell` are gone at the next launch

Only the project folder, the private home and volumes are kept. Add the
package to the image with [`packages`](container-images.md#extra-packages).

## Custom agents

### An agent name is refused

The name breaks a rule of [`launch.agents`](config.md#launchagents), such as
being a client's name or longer than 32 characters. The message names the
rule. Rename the agent.

### An agent cannot start because the container has no command

The `command` does not match a script that uv installed. Make
[`command`](config.md#launchagentscommand) match an entry in the project's
`[project.scripts]`. If the project is not a package, uv installs no
scripts, so add a `[build-system]` table, as `uv init --package` writes it.

### A source elsewhere fails with No module named

The project is not a package, so Python cannot import it from another
folder. Add a `[build-system]` table and a `[project.scripts]` entry, as
`uv init --package` writes them, or launch from the project folder.

### A read-only source fails with Read-only file system

The message names an `.egg-info` folder:

```text
could not create 'src/<name>.egg-info': Read-only file system
```

The build backend, usually setuptools, writes this folder into the
read-only source. Switch the project's `[build-system]` to hatchling or
uv_build, or launch from the source folder.

### uv says the lockfile needs to be updated

The agent stops with one of these messages:

```text
The lockfile at `uv.lock` needs to be updated, but `UV_LOCKED=1` was provided
Unable to find lockfile at `uv.lock`, but `UV_LOCKED=1` was provided
```

The [source folder](launch-agents.md#the-source-folder) is read-only, so uv
cannot write `uv.lock`. Launch the agent once from its source folder, where
uv can write the lock, then launch it as before. You can also run `uv lock`
in a `--shell` opened from the source folder.

### An agent's first launch fails under network none

uv cannot install the environment without a network. Launch once with
`--network default` in each project folder, then turn the network off. See
[Offline launches](launch-agents.md#offline-launches).

## Voice

### The mic never works in talk

macOS gives the microphone to the terminal app that you run `gmlx talk`
from. In System Settings, open Privacy and Security, then Microphone, and
turn on your terminal app. If you dismissed the prompt long ago, turn the
entry off and on again. `gmlx talk --list-devices` shows whether any input
device is visible.

### Transcription or speech fails because ffmpeg is not found

Transcription, or speech in mp3, flac or opus, answers 500. Install ffmpeg,
and send the request again with no restart:

```sh
brew install ffmpeg
```

The server never runs an ffmpeg from a folder that a container session
shares or shared read-write, and the log names each folder it skips. Once
you trust a folder that only an earlier session shared, run
`gmlx launch --forget-share PATH`, as
[The share history](container-security.md#the-share-history) explains.

## Distillation

### A distill step fails or the adapter learns nothing

[Distillation troubleshooting](distill-troubleshooting.md)
covers each step, and [gmlx distill](distill-reference.md) lists the exit
codes.

## Logs and files

### Where the logs are

```sh
gmlx logs -f                # Follows the background server's log.
gmlx serve --print-config   # Prints the full config the server would run with.
```

Each finished request logs a line with the model, the token counts and the
timing. `gmlx ps` lists the resident models.

### Where files are on disk

Paths under `~/.cache` and `~/.local/share` follow `XDG_CACHE_HOME` and
`XDG_DATA_HOME` when they are set.

| Path | Contents |
|------|----------|
| `~/.config/gmlx/gmlx.yaml`, `~/.gmlx.yaml` | The config, as [Where gmlx looks](config.md#where-gmlx-looks) describes. |
| `~/.config/gmlx/` | Client configs that `gmlx launch` writes. |
| `~/.config/gmlx/comfyui.json` | The ComfyUI settings and what `gmlx comfyui install` recorded. |
| `~/.local/share/gmlx/comfyui/` | The copy of the gmlx nodes that each ComfyUI's `custom_nodes/comfyui-gmlx` links to. |
| `~/.pi/agent/`, `~/.omp/agent/`, `~/.config/goose/config.yaml`, `~/.hermes/config.yaml` | Client files that `gmlx launch` adds its settings to. Delete those settings to remove gmlx. |
| `~/.cache/gmlx/` | Server runfiles and logs, chat input history and the GGUF header cache. |
| `~/.cache/gmlx/apc/` | The prompt cache, when the disk tier is on and has no `path` of its own. |
| `~/.cache/gmlx/media/` | Media files that requests name, as [Media in requests](api.md#media-in-requests) describes. |
| `~/.cache/gmlx/images/` | The images behind `url` answers of the [image routes](images.md), a folder per server port. They expire after `server.image.url_ttl_s`. |
| `~/.cache/gmlx/talk/` | The wake-word and voice-activity models of `gmlx talk`. |
| `~/.cache/huggingface/` | The files that `hf:` references resolve from. |
| `~/.local/share/gmlx/chats/` | Saved chat sessions. |
| `~/.local/share/gmlx/assistant-memory.db` | The assistant's memory, with an `assistant-<id>.db` beside it for each served assistant. |
| `~/Library/Application Support/gmlx/` | The menu bar app bundle. |
| `~/Library/LaunchAgents/com.gmlx.*.plist` | Login items from `gmlx service install`. |
| `~/.open-webui/` | Open WebUI chat history. |
| `~/.local/share/gmlx/launch/` | Container mode's private homes, browser app ports, locks and records. |
| `~/.cache/gmlx/launch/` | Container mode's session logs, detached session output and session folders. |
| Your model folders | The GGUFs that `pull` downloads. |

[Removing gmlx](installation.md#removing-gmlx) gives the steps that remove
gmlx and these files.
