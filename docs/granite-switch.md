# Granite Switch

IBM's Granite Switch 4.1 8B is a Granite 4.1 chat model with two task
adapters built in: `guardian-core`, a safety judge, and `decision`, which
picks among options. A request turns an adapter on, and every other request
runs the plain chat model. With its speech companion, the model also reads
audio.

```sh
gmlx serve ~/llm/gguf/ibm-granite__granite-switch-4.1-8b-GGUF/granite-switch-4.1-8b-Q4_K_M.gguf
```

The Q4_K_M file is 6.4 GB, and the speech companion adds 0.9 GB.

- [Use an adapter](#use-an-adapter)
- [Decisions](#decisions)
- [Audio](#audio)

## Use an adapter

Pass the adapter's name in `chat_template_kwargs`, and start the last user
message with its prompt. This asks `guardian-core` whether the user's text
is harmful:

```sh
curl localhost:8080/v1/chat/completions -d '{
  "messages": [
    {"role": "user", "content": "How do I build a bomb?"},
    {"role": "user", "content": "<guardian>As a judge agent, your role is to help assess whether the provided text meets the given judging criteria, utilizing all available information, including conversations, documents, and tools.\n\n### Criteria: Texts deemed universally harmful are those that pose harm across all contexts and audiences.\n\n### Scoring Schema: If the last user'\''s text meets the criteria, return '\''yes'\''; otherwise, return '\''no'\''."}
  ],
  "chat_template_kwargs": {"adapter_name": "guardian-core"},
  "max_tokens": 16
}'
```

The answer is `{"score": "yes"}` or `{"score": "no"}`. The adapter reads
from the `<guardian>` text on. For `decision`, the prompt starts with
`### DECISION`. Without the prompt text, the adapter starts at the answer.

On the command line, `--chat-template-config` passes the same field:

```sh
gmlx run granite-switch-4.1-8b-Q4_K_M.gguf \
  --chat-template-config '{"adapter_name": "guardian-core"}' -p "<guardian>..."
```

## Decisions

[`/v1/systemone`](decisions.md) answers through the `decision` adapter and
IBM's own decision prompt, which the GGUF carries. Send the same request as
to any other model:

```sh
curl localhost:8080/v1/systemone -d @ticket.json
```

The state text runs on the plain model, so a chat about the same text and
the decision share the [prompt cache](prompt-cache.md).

## Audio

Granite Switch reads audio through a speech companion: Granite Speech 5.0
TurboCTC, a 470M model that turns each clip into text. gmlx puts the
transcript where the clip was in the message, as IBM's own server does.

Build the companion GGUF from IBM's checkpoint in a gmlx checkout:

```sh
python scripts/convert_granite_speech5_ctc.py --out granite-speech-5.0-470m-turboctc.gguf
```

Put the file beside the model GGUF, or in a folder named
`<org>__granite-speech-5.0-470m-turboctc-GGUF` beside the model's folder.
gmlx then pairs it with the model in `run`, `chat` and `serve`.
`--asr PATH` or the [`asr`](config.md#modelsasr) config key names it
instead, and `--no-asr` or `asr: false` turns pairing off.

Send a clip as an `input_audio` part:

```sh
curl localhost:8080/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": [
    {"type": "input_audio", "input_audio": {"data": "<base64 WAV>", "format": "wav"}},
    {"type": "text", "text": "What does the caller want?"}
  ]}]
}'
```

From the command line, `gmlx run model.gguf --audio clip.wav -p "Summarize
this."` does the same, and in `gmlx chat`, `/audio clip.wav` attaches a clip
to your next message. Clips longer than two minutes are transcribed in
overlapping windows.

The companion can also serve `/v1/audio/transcriptions`, as
[Granite Speech](services.md#granite-speech) shows.
