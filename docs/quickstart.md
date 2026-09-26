# Quickstart

gmlx runs GGUF files as they are, with no conversion step. With gmlx
[installed](installation.md), create a configuration file and download a
model into it:

```sh
gmlx init --models-dir ~/models
gmlx pull hf:unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K.gguf
```

`gmlx init` writes a [configuration file](config.md) that lists your models
and the folders that hold them. The `~/models` folder can be empty or not
exist yet.
Once the file exists, `gmlx pull` downloads into its first folder and adds
the model to the file, with an id made from the file name. Here the id is
`qwen3.8-27b-ud-q6`, and every gmlx command accepts it in place of a path.
A server that is already running reads the file again, so a pulled model is
available at once.

This model is 20.5 GB. On a Mac with less memory, choose a smaller model
from [Choosing a model](#choosing-a-model). To answer questions in a
wizard instead, run `gmlx init` with no flags. The wizard scans the
folders that you already keep models in, lets you rename the models, and
turns on optional services such as speech and embeddings.

## Running a model

Give the model a prompt, or chat with it in the terminal:

```sh
gmlx run qwen3.8-27b-ud-q6 --prompt "Explain entropy in one paragraph."
gmlx chat qwen3.8-27b-ud-q6
```

After the reply, `run` prints the prompt and generation speeds in tokens
per second and the peak memory. The chat keeps its KV cache between
turns, so each turn processes only the new message. In the chat, type
`/help` for the commands, press Esc to stop a reply, and type `/exit` to
quit. Both commands also accept the path of a GGUF file, which needs no
configuration file. [Chat](chat.md) describes the rest.

Both commands start from the sampling values that the model's publisher
recommends, which gmlx keeps as [family defaults](family-defaults.md). An
intent such as `@coding` or `@creative` after the id selects the
publisher's values for that kind of task, as in
`gmlx chat qwen3.8-27b-ud-q6@creative`. `gmlx profiles` prints the values
of every intent for each family.

## Serving models

A server keeps models loaded and answers requests from any app:

```sh
gmlx serve
```

`gmlx serve` finds the configuration file, starts the server in the
background on port 8080, and returns. On a Mac desktop it also opens the
[menu bar app](menubar.md), which shows the loaded models. These commands
manage the server:

| Command | Result |
|---------|--------|
| `gmlx list` | Lists the model ids in the file. |
| `gmlx status` | Shows the process id, uptime and URL of the server. |
| `gmlx ps` | Lists the loaded models. |
| `gmlx logs -n 20 -f` | Shows the last 20 lines of the log and follows it. |
| `gmlx stop` | Stops the server. |

## Sending requests

The server answers the OpenAI, Anthropic Messages and OpenAI Responses APIs
on one port. A request names its model by id:

```sh
curl localhost:8080/v1/chat/completions -d '{
  "model": "qwen3.8-27b-ud-q6",
  "messages": [{"role": "user", "content": "Explain entropy in one paragraph."}]
}'
```

Add `"stream": true` to receive the reply as server-sent events. An intent
works here too, as in `"model": "qwen3.8-27b-ud-q6@coding"`. The OpenAI
Python client works with no changes:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
reply = client.chat.completions.create(
    model="qwen3.8-27b-ud-q6",
    messages=[{"role": "user", "content": "Explain entropy in one paragraph."}],
)
print(reply.choices[0].message.content)
```

Tool calls, structured output, log probabilities and images in messages
also work. For the endpoints and the request fields that each one accepts,
see the [HTTP API](api.md).

## Choosing a model

The suffix of a GGUF file name, such as `Q6_K`, is its
[quant](glossary.md#quant), which trades file size against accuracy. These
instruct models fit each memory size with room left for a long
conversation:

| Mac memory | Model | Notes |
|------------|-------|-------|
| 16 GB | Qwen3-4B, Q4_K_M, 2.5 GB | It is fast and capable for its size. |
| 32 GB | Qwen3.5-9B, Q6_K, 8 GB | Its MTP head turns on speculative decoding by itself. |
| 64 GB | Qwen3.8-27B, UD-Q6_K, 20.5 GB | It is strong at chat, code and tool calls. |
| 96 GB or more | Qwen3.6-35B-A3B, Q6_K, 29 GB, or gpt-oss-120b, MXFP4, 63 GB | Both are mixture-of-experts models, with large-model quality at small-model speed. |

A loaded model needs memory for its weights, about its file size, and for
its KV cache, which grows with the conversation. In a long session the KV
cache can grow as large as the weights.
[Memory and the KV cache](memory.md) explains how
to estimate it and how to make it smaller. A mixture-of-experts model that
is larger than memory can still run, as
[Models larger than memory](streaming.md) describes.

To see the files of a repository before you download one, run
`gmlx validate`:

```sh
gmlx validate hf:unsloth/Qwen3-4B-GGUF
```

For a repository, `validate` lists its GGUF files and, when it knows their
sizes, which of them fit in the memory of your Mac. For a single file, it
reads only the header and says whether gmlx can load it. A gated
repository needs a Hugging Face token, as
[Troubleshooting](troubleshooting.md#a-gated-or-private-repo-will-not-download)
describes.

Models that you already have from LM Studio or llama.cpp run as they are,
and [Migrating from other tools](migrating.md) says what else carries over.

## Connecting a client

`gmlx launch` connects an app to your server. This command connects pi, a
coding agent:

```sh
gmlx launch pi --model qwen3.8-27b-ud-q6
```

`launch` starts the server if it is not running, and adds a provider for
the server to the settings of pi without changing the providers that are
already there. It asks the server to load the model and keep it loaded
through the idle timeout, and then it starts pi. The same command connects
the other coding agents and chat apps, including Open WebUI in the
browser, as [Agents and chat apps](launch.md) describes.

## Next steps

- [Configuration](config.md) explains profiles, aliases, the prompt cache
  and the memory limits of the server.
- [Voice chat](talk.md) sets up `gmlx talk`, which answers spoken questions
  aloud.
- [Menu bar app](menubar.md) describes the login item, which keeps the
  server and the menu bar app running.
- [Performance tuning](performance.md) explains speculative decoding, KV
  cache quantization and the other speed settings.

When something fails, run `gmlx doctor`. It checks the runtime, the
configuration file, the model paths and the services, and gives the fix
for each problem it finds. [Troubleshooting](troubleshooting.md) covers
the common failures.
