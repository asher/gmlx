# LFM3

LFM3 is Liquid AI's family of small models. The largest, LFM3-Omni-2B,
reads text, images, video and audio, and answers in text or in speech. It
runs a whole voice conversation by itself, with no separate speech
models.

```sh
gmlx pull lfm3-omni-2b
gmlx chat lfm3-omni-2b
```

`gmlx pull` has four models of the family:

| Name | Reads | Answers in | Context |
|------|-------|------------|---------|
| `lfm3-omni-2b` | Text, images, video and audio | Text or speech | 32,768 tokens |
| `lfm3-audio-2b` | Text and audio | Text or speech | 32,768 tokens |
| `lfm3-vl-2b` | Text, images and video | Text | 128,000 tokens |
| `lfm3-1.7b` | Text | Text | 262,144 tokens |

Each pull downloads the model and, for all but `lfm3-1.7b`, its projector
file, and registers the two under the model's name. The projector holds
the vision tower, the audio encoder and the speech decoder, so a pull
with `--no-vision` leaves a text model.

The models are under the LFM Open License v1.0. It licenses commercial
use only to organizations with less than $10 million in annual revenue.

`d1-2b` and `d1-omni-2b` are LFM3 models that answer
[structured decisions](decisions.md#images-video-and-sound) only, not chat.

- [A voice conversation](#a-voice-conversation)
- [Spoken replies in chat](#spoken-replies-in-chat)
- [Speech from the command line](#speech-from-the-command-line)
- [Images, video and audio](#images-video-and-audio)
- [Speech on the server](#speech-on-the-server)
- [Transcription and translation](#transcription-and-translation)
- [Voices and languages](#voices-and-languages)
- [Memory](#memory)
- [Limits](#limits)

## A voice conversation

[`gmlx talk`](talk.md) runs on `lfm3-omni-2b` or `lfm3-audio-2b` alone.
The model transcribes what you say, hears your voice, and speaks its
reply as it writes it:

```sh
gmlx talk lfm3-omni-2b
```

The server needs no `stt` and no `tts` service for this.
[One model for the whole turn](talk.md#one-model-for-the-whole-turn)
describes the session.

## Spoken replies in chat

In `gmlx chat`, `/speak on` has the model speak each reply while the text
prints:

```text
>> /speak on
[chat] replies are spoken and played
```

| Command | Effect |
|---------|--------|
| `/speak on` | Plays each reply on the default output device. It needs the `talk` extra. |
| `/speak save DIR` | Writes each reply to the next `reply-NNN.wav` in `DIR`. It works with or without `/speak on`. |
| `/speak off` | Goes back to text replies |
| `/speak` | Shows the setting |

The conversation keeps each spoken reply as its text. `/image` and
`/audio` attach a file to your next message, as in any multimodal chat.

## Speech from the command line

`--speech-out` saves the reply as speech and prints its text:

```sh
gmlx run lfm3-omni-2b --speech-out reply.wav --prompt "Why is the sky blue?"
```

```text
The sky is blue because the atmosphere scatters blue light from the sun more than other colors.
==========
Speech: 5.76 s saved to reply.wav
```

With `--tts`, the model reads the prompt aloud and gives no answer:

```sh
gmlx run lfm3-omni-2b --tts --speech-out morning.wav --voice male --prompt "Good morning."
```

[gmlx run](cli.md#gmlx-run) lists the speech flags.

## Images, video and audio

```sh
gmlx run lfm3-omni-2b --image chart.png --prompt "What does this chart show?"
gmlx run lfm3-omni-2b --audio memo.wav --prompt "What does the speaker ask for?"
```

On the server, the models read `image_url`, `video_url` and `input_audio`
parts on `/v1/chat/completions`, as
[Media in requests](api.md#media-in-requests) shows. Each video and each
clip goes where its message put it. Of a video, the model reads the
frames and not the sound.

To transcribe a clip with `run`, give the model card's instruction as the
system prompt, name the language, and send an empty prompt:

```sh
gmlx run lfm3-omni-2b --audio memo.wav --system-prompt "Perform ASR in English." --prompt ""
```

## Speech on the server

A server that serves `lfm3-omni-2b` or `lfm3-audio-2b` answers
`/v1/audio/speech` with it. It says so at start:

```text
[server] tts: lfm3-omni-2b  (POST /v1/audio/speech, the served model itself)
```

```sh
curl localhost:8080/v1/audio/speech -H 'content-type: application/json' \
  -d '{"model": "tts-1", "input": "Hello from LFM3.", "voice": "female"}' -o hello.mp3
```

| Name | Default | Meaning |
|------|---------|---------|
| `input` | Required | The text to speak. The server removes markdown and emoji first. |
| `voice` | `female` | `female` or `male` |
| `language` | Read from the text | The language of the text, as a code or a name such as `fr` or `French` |
| `response_format` | `mp3` | `mp3`, `wav`, `flac`, `opus` or `pcm`. Every format except `wav` and `pcm` needs ffmpeg. |
| `stream_format` | `audio` | `sse` streams OpenAI `speech.audio.delta` events of 16-bit PCM at 24 kHz |
| `seed` | None | An integer that makes the speech repeat |

[`server.tts`](services.md#a-served-model-that-speaks) decides which
model answers when the server has more than one.

A chat completion returns speech when the request asks for audio:

```sh
curl localhost:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "lfm3-omni-2b",
  "modalities": ["text", "audio"],
  "audio": {"voice": "female", "format": "wav"},
  "messages": [{"role": "user", "content": "Why is the sky blue?"}]
}'
```

```jsonc
{"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": null,
   "audio": {"id": "audio_bf92a727...", "data": "<base64 WAV>", "expires_at": 1794421968,
             "transcript": "The sky is blue because the atmosphere scatters blue light ..."}}}],
 "usage": {"prompt_tokens": 35, "completion_tokens": 96, "total_tokens": 131,
           "completion_tokens_details": {"audio_tokens": 74}}}
```

| Name | Default | Meaning |
|------|---------|---------|
| `audio.voice` | `female` | `female` or `male` |
| `audio.format` | `wav`, or `pcm16` in a stream | `wav`, `mp3`, `flac`, `opus` or `pcm16`. A stream takes `pcm16` only. |
| `audio.language` | Read from the text | The language of a reply that the model writes first and then speaks |

With `"stream": true`, each chunk's `delta.audio` carries `transcript`
text or `data`, base64 16-bit PCM at 24 kHz.

The model speaks in one of two ways:

- With the `female` voice and no tools, it speaks as it writes. The text
  comes only as the transcript, and `content` is null.
- With the `male` voice, or with `tools` in the request, it writes the
  reply as text first and then speaks it. `content` holds the text, and
  the audio starts after the text ends. A tool call has no audio.

A reply that speaks as it writes counts its speech toward `max_tokens`,
at 12.5 tokens for each second. In a reply written first, `max_tokens`
limits the text, and the whole text is spoken.
`completion_tokens_details.audio_tokens` reports the speech. To continue the
conversation, send the earlier reply back as its `audio.id`, which the
server keeps for an hour, or as its transcript. `/v1/responses` returns
text only.

## Transcription and translation

The same two models answer the OpenAI speech-to-text routes. The server
needs ffmpeg on its PATH (`brew install ffmpeg`) to decode the upload.

```sh
curl localhost:8080/v1/audio/transcriptions -F file=@memo.wav -F model=whisper-1 -F language=en
# {"text":"What is the capital of France?"}
curl localhost:8080/v1/audio/translations -F file=@memo-fr.wav -F model=whisper-1
```

Without `language`, the model finds the language itself. A translation
is always into English.

`response_format` is `json` or `text`. These models give no word times
and no speakers, so the other formats, `timestamp_granularities[]` and
`diarization` get a 400. [`server.stt`](services.md#a-served-model-that-hears-audio)
decides which model answers when the server has more than one.

## Voices and languages

The voices are `female` and `male`.

A request can name the language of the text to speak with one of these
codes, or with the language's English name: `ar`, `da`, `de`, `el`, `en`,
`en-gb`, `en-us`, `es`, `fi`, `fr`, `he`, `it`, `ja`, `ko`, `nl`, `pl`,
`pt`, `sv`, `tr`, `zh`. Without one, the model reads the language from the
text.

LFM3-Omni-2B lists 23 languages, and LFM3-Audio-2B 18. A transcription
request with a `language` that the model does not list gets a 400 that
names the ones it does.

## Memory

On every Mac, each quality tier of `lfm3-omni-2b` and `lfm3-audio-2b`
holds the full 32,768-token context for one request at a time.

On an 8 GB Mac, `gmlx pull lfm3-omni-2b` picks the `balanced` tier, which
also leaves room for an image or a spoken reply beside a full context.
The `xhigh` tier fits there with almost none. From 16 GB, the pull picks
`xhigh`.

`gmlx pull <name> --list` shows the room each tier leaves on your Mac.
On an M1 or M2, the pull takes the projector in F16 and not BF16, because
those chips have no native bfloat16.

## Limits

| Limit | Value |
|-------|-------|
| Context of `lfm3-omni-2b` and `lfm3-audio-2b` | 32,768 tokens |
| Audio input | 12.5 tokens for each second, so 43 minutes fill the context |
| An image | At most 4,096 tokens |
| A video | Two frames for each second: 4 to 384 frames on `lfm3-omni-2b`, 4 to 768 on `lfm3-vl-2b`. The model reads no sound. |
| Speech output | 24 kHz mono, 12.5 tokens for each second |
| Speech with other features | No speculative decoding, no `logprobs` and no `response_format` |

Speech speed on an 8 GB M1 or M2 is not measured.
