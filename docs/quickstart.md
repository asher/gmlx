# Quickstart

With gmlx [installed](installation.md), create a configuration file and
download a model. This model is 20.5 GB and suits a Mac with 64 GB of
memory. On a smaller Mac, pick one from
[Choosing a model](#choosing-a-model) first.

```sh
gmlx init --models-dir ~/models
gmlx pull hf:unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K.gguf
```

`gmlx init` writes a [configuration file](config.md) that lists your models
and the folders that hold them. Run it with no flags for a wizard that scans
the folders where you already keep models. `gmlx pull` downloads into the
first folder and adds the model under the id `qwen3.8-27b-ud-q6`, which every
command accepts in place of a path.

- [Running a model](#running-a-model)
- [Serving models](#serving-models)
- [Sending requests](#sending-requests)
- [Choosing a model](#choosing-a-model)
- [Connecting a client](#connecting-a-client)
- [Next steps](#next-steps)

## Running a model

Give the model a prompt, or chat with it in the terminal:

```sh
gmlx run qwen3.8-27b-ud-q6 --prompt "Explain entropy in one paragraph."
gmlx chat qwen3.8-27b-ud-q6
```

After the reply, `run` prints the prompt and generation speeds and the peak
memory. In the chat, type `/help` for the commands, press Esc to stop a
reply, and type `/exit` to quit. Both commands also take the path of a GGUF
file, with no configuration file. [Chat](chat.md) describes the rest.

Both start from the sampling values that the model's publisher recommends.
An intent after the id, as in `qwen3.8-27b-ud-q6@instruct`, picks the values
for one kind of task. `gmlx profiles` lists the intents of each family.

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
  "model": "qwen3.8-27b-ud-q6",
  "messages": [{"role": "user", "content": "Explain entropy in one paragraph."}]
}'
```

Add `"stream": true` to stream the reply. Intents work here too, as in
`"model": "qwen3.8-27b-ud-q6@coding"`. The OpenAI Python client works
unchanged:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
reply = client.chat.completions.create(
    model="qwen3.8-27b-ud-q6",
    messages=[{"role": "user", "content": "Explain entropy in one paragraph."}],
)
print(reply.choices[0].message.content)
```

Tool calls, structured output, log probabilities and images also work, as
the [HTTP API](api.md) describes.

## Choosing a model

The suffix of a GGUF file name, such as `Q6_K`, is its
[quant](glossary.md#quant), which trades file size against accuracy. These
instruct models fit each memory size with room for a long conversation:

| Mac memory | Model | Notes |
|------------|-------|-------|
| 16 GB | Qwen3-4B, Q4_K_M, 2.3 GB | Fast and capable for its size |
| 32 GB | Qwen3.5-9B, Q6_K from `unsloth/Qwen3.5-9B-MTP-GGUF`, 7.2 GB | MTP head turns on speculative decoding by itself |
| 64 GB | Qwen3.8-27B, UD-Q6_K, 20.5 GB | Strong at chat, code and tool calls |
| 96 GB or more | Qwen3.6-35B-A3B, UD-Q6_K, 27 GB, or gpt-oss-120b, MXFP4, 59 GB | Mixture-of-experts: large-model quality at small-model speed |

A model needs memory for about its file size, plus its KV cache, which
grows with the conversation and can reach the size of the weights.
[Memory and the KV cache](memory.md) shows how to estimate and shrink it. A
mixture-of-experts model larger than memory can still run, as
[Models larger than memory](streaming.md) describes.

To see which files of a repository fit your Mac before you download one:

```sh
gmlx validate hf:unsloth/Qwen3-4B-GGUF
```

A gated repository needs a Hugging Face token, as
[Troubleshooting](troubleshooting.md#a-gated-or-private-repo-will-not-download)
describes. Models you already have from LM Studio or llama.cpp run as they
are, and [Migrating from other tools](migrating.md) says what else carries
over.

## Connecting a client

`gmlx launch` connects an app to your server. This command connects pi, a
coding agent:

```sh
gmlx launch pi --model qwen3.8-27b-ud-q6
```

`launch` starts the server if it is not running, adds the server to pi's
settings, and starts pi. The same command connects the other coding agents
and chat apps, as [Agents and chat apps](launch.md) describes.

Add `--container` to run pi in an Apple container that sees only the current
folder, with pi installed for you. This needs
[Apple container](installation.md#apple-container), and
[Container mode](launch-container.md) covers the rest.

## Next steps

- [Configuration](config.md): profiles, aliases, the prompt cache and memory
  limits.
- [Voice chat](talk.md): `gmlx talk` answers spoken questions aloud.
- [Menu bar app](menubar.md): the login item that keeps the server running.
- [Performance tuning](performance.md): speculative decoding, KV cache
  quantization and other speed settings.

When something fails, run `gmlx doctor`. It checks the runtime, the
configuration file, the model paths and the services, and gives a fix for
each problem. [Troubleshooting](troubleshooting.md) covers common failures.
