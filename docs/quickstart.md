# Quickstart

With gmlx [installed](installation.md), create a configuration file and
download a model:

```sh
gmlx init --models-dir ~/models
gmlx pull qwen3.8-27b
```

`gmlx init` writes a [configuration file](config.md) that lists your models
and the folders that hold them. Run it with no flags for a wizard that scans
the folders where you already keep models. `gmlx pull` downloads Qwen3.8-27B
at the quality tier that suits this Mac, with image input and faster
replies, a 23 GB download on a 64 GB Mac. It registers the model under the
name `qwen3.8-27b`, which every command accepts in place of a path. On a
smaller Mac, [Choosing a model](#choosing-a-model) names a better fit.

- [Running a model](#running-a-model)
- [Serving models](#serving-models)
- [Sending requests](#sending-requests)
- [Choosing a model](#choosing-a-model)
- [Connecting a client](#connecting-a-client)
- [Next steps](#next-steps)

## Running a model

Give the model a prompt, or chat with it in the terminal:

```sh
gmlx run qwen3.8-27b --prompt "Explain entropy in one paragraph."
gmlx chat qwen3.8-27b
```

In the chat, type `/help` for the commands, press Esc to stop a reply, and
type `/exit` to quit. Both commands also take the path of a GGUF file, with
no configuration file. [Chat](chat.md) describes the rest.

Both start from the sampling values that the model's publisher recommends.
An intent after the id picks the values for one kind of task:

```sh
gmlx chat qwen3.8-27b@instruct
```

`gmlx profiles` lists the intents of each family.

## Serving models

A server keeps models loaded and answers requests from any app:

```sh
gmlx serve
```

The server starts in the background on port 8080, and on a Mac desktop the
[menu bar app](menubar.md) opens with it. A model you pull while it runs is
available at once. These commands manage it:

| Command | Result |
|---------|--------|
| `gmlx list` | Lists the model ids in the configuration file |
| `gmlx status` | Shows the server's process id, uptime and URL |
| `gmlx ps` | Lists the loaded models |
| `gmlx logs -n 20 -f` | Shows the last 20 log lines and follows the log |
| `gmlx stop` | Stops the server |

## Sending requests

The server answers the OpenAI Chat Completions, OpenAI Responses and
Anthropic Messages APIs on one port. A request names its model by id:

```sh
curl localhost:8080/v1/chat/completions -d '{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "Explain entropy in one paragraph."}]
}'
```

Add `"stream": true` to stream the reply. Intents work here too, as in
`"model": "qwen3.8-27b@coding"`. The OpenAI Python client works
unchanged:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
reply = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "Explain entropy in one paragraph."}],
)
print(reply.choices[0].message.content)
```

Tool calls, structured output, log probabilities and images also work, as
the [HTTP API](api.md) describes.

## Choosing a model

`gmlx pull` with no name opens a picker of the models that fit this Mac,
with the recommended ones first. Each shows its size, whether it reads
images, and how much context it leaves room for. Pick one, or pull it by
name and chat with it:

```sh
gmlx pull
gmlx pull gemma-4-12b
gmlx chat gemma-4-12b
```

The first recommendation at each memory size:

| Mac memory | Model | Command |
|------------|-------|---------|
| 8 GB | Ternary Bonsai 2 9B | `gmlx pull ternary-bonsai-2-9b` |
| 16 GB | gemma-4-12B | `gmlx pull gemma-4-12b` |
| 24 GB | gemma-4-26B-A4B | `gmlx pull gemma-4-26b-a4b` |
| 32, 48, 64 or 96 GB | Qwen3.8-27B | `gmlx pull qwen3.8-27b` |
| 128, 192 or 256 GB | Qwen3.8-Flash-Next | `gmlx pull qwen3.8-flash-next` |
| 512 GB | DeepSeek-V4.1-Flash | `gmlx pull deepseek-v4.1-flash-1017` |

`gmlx pull --list` prints every recommended model with its license and a
`gmlx pull` command, and the [pull reference](cli.md#gmlx-pull) lists them
all.

A model needs memory for about its file size, plus its KV cache, which
grows with the conversation. [Memory and the KV cache](memory.md) shows how
to estimate and shrink it. A mixture-of-experts model larger than memory
can still run from the SSD, as [Models larger than memory](streaming.md)
describes.

Any GGUF on Hugging Face pulls by reference. To see which files of a
repository fit your Mac before you download one:

```sh
gmlx validate hf:unsloth/Qwen3-4B-GGUF
```

A gated repository needs a Hugging Face token, as
[Troubleshooting](troubleshooting.md#a-gated-or-private-repo-will-not-download)
describes. Models you already have from LM Studio or llama.cpp run as they
are, and [Migrating from other tools](migrating.md) says what else carries
over.

## Connecting a client

`gmlx launch` connects an app to your server. pi is a coding agent that you
install separately. These commands install it and connect it:

```sh
npm install -g @earendil-works/pi-coding-agent
gmlx launch pi --model qwen3.8-27b
```

`launch` starts the server if it is not running, adds the server to pi's
settings, and starts pi. The same command connects the other coding agents
and chat apps, as [Agents and chat apps](launch.md) describes.

To run pi in an Apple container instead, add `--container`:

```sh
gmlx launch pi --container --model qwen3.8-27b
```

The container sees only the current folder, and gmlx installs pi in it for
you. This needs [Apple container](installation.md#apple-container), and
[Container mode](launch-container.md) covers the rest.

## Next steps

- [Configuration](config.md): profiles, aliases, the prompt cache and memory
  limits.
- [Voice chat](talk.md): `gmlx talk` answers spoken questions aloud.
- [Image generation](images.md): images from a text prompt, on the command
  line, from an app or in ComfyUI.
- [Menu bar app](menubar.md): the server's status, models and voice
  sessions in the macOS menu bar.
- [Performance tuning](performance.md): speculative decoding, KV cache
  quantization and other speed settings.

When something fails, run `gmlx doctor`. It checks the runtime, the
configuration file, the model paths and the services, and gives a fix for
each problem. [Troubleshooting](troubleshooting.md) covers common failures.
