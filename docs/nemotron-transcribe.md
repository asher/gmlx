# Nemotron 3.5 Transcribe

NVIDIA's Nemotron 3.5 Transcribe is a speech model. It hears audio through
its own speech encoder, and it writes a transcript, translates the speech,
or answers a question about it. In `gmlx serve` it also answers the
OpenAI transcription and translation routes, with word times and speakers.

```sh
gmlx pull nemotron-3.5-transcribe
gmlx run nemotron-3.5-transcribe --audio meeting.wav
```

`gmlx pull` downloads two files from
`AtomicChat/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF`, and registers the
model under the name `nemotron-3.5-transcribe` with the encoder as its
`mmproj`. `nemotron-transcribe` is a shorter name for the pull. The license
is OpenMDW-1.1.

| File | Name | Size |
|------|------|------|
| Speech decoder, 30B-A4B | `NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-AD-NVFP4.gguf` | 19.7 GB |
| Speech encoder | `NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-Encoder-F16.gguf` | 1.7 GB |

The encoder header names the decoder checkpoint that it belongs to. When
the two files are in the same folder, `run`, `chat`, `serve` and a
[discover](config.md#model-discovery) scan pair them with no `mmproj` key.
`--mmproj` names the encoder for a file elsewhere.

- [Transcribe a recording](#transcribe-a-recording)
- [Ask about a recording or translate it](#ask-about-a-recording-or-translate-it)
- [Chat about a recording](#chat-about-a-recording)
- [Send audio to the server](#send-audio-to-the-server)
- [Transcription and translation routes](#transcription-and-translation-routes)
- [Speakers and word times](#speakers-and-word-times)
- [A voice session on one model](#a-voice-session-on-one-model)
- [Speculative decoding](#speculative-decoding)
- [Limits](#limits)

## Transcribe a recording

Pass the recording with `--audio` and no prompt:

```sh
gmlx run nemotron-3.5-transcribe --audio meeting.wav
```

`run` then uses the model card's transcription instruction, greedy
decoding, no thinking, and a token budget that fits the length of the
recording. It says so on stderr:

```text
[asr] transcribing: the model card's instruction, greedy, no thinking (--prompt to ask something else)
```

A flag that you pass, such as `--temp` or `--max-tokens`, keeps its value.
[Sending media](vlm.md#sending-media) lists the audio formats.

## Ask about a recording or translate it

`--prompt` replaces the transcription instruction with your own:

```sh
gmlx run nemotron-3.5-transcribe --audio interview.wav \
  --prompt "How many people speak in this recording?"
```

With a prompt of your own, the model uses its family's sampling values
and thinks before it answers. `--thinking off` turns thinking off.

To translate, use the model card's translation instruction and name the
language:

```sh
gmlx run nemotron-3.5-transcribe --audio interview.wav --thinking off \
  --prompt "Please translate the given speech to English."
```

To label the speakers in a transcript, use the card's speaker instruction:

```sh
gmlx run nemotron-3.5-transcribe --audio meeting.wav --thinking off --temp 0 \
  --prompt "Transcribe this audio with punctuation and capitalization. Include all spoken words and verbal fillers. Write numbers in their standard written form. Put speaker tags <spk:0> <spk:1> <spk:2> <spk:3> on the transcription."
```

The transcript then marks each change of speaker with a tag:

```text
<spk:0> He had written a number of books himself, ... <spk:1> Their churches have disappeared, ...
```

## Chat about a recording

```sh
gmlx chat nemotron-3.5-transcribe
```

`/audio meeting.wav` attaches a clip to your next message. Press Enter on
an empty line to transcribe it with the card's instruction, greedy and
without thinking, or type a question about it. Later turns can ask about
the same clip. With speculation on, audio turns speculate as text turns
do.

A conversation holds at most six clips, and it must fit in the model's
65,536-token context. When it does not, chat says so and does not send the
message. `/reset` starts a new conversation. The model reads no images.

## Send audio to the server

The model reads `input_audio` parts on `/v1/chat/completions` and
`/v1/responses`:

```sh
curl localhost:8080/v1/chat/completions -d '{
  "model": "nemotron-3.5-transcribe@instruct",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "Summarize this call in two sentences."},
    {"type": "input_audio", "input_audio": {"data": "<base64 WAV>", "format": "wav"}}
  ]}]
}'
```

The `@instruct` intent decodes greedily without thinking. Without it, the
model thinks first. Each clip goes where the message put it. The server
keeps recent encoded clips, so a follow-up turn about the same clip skips
the encoder. With the card's speaker instruction, the reply holds
`<spk:N>` speaker tags as written.

A chat completion can also return the time of each word, as
[Word timestamps in chat](api.md#word-timestamps-in-chat) shows.

## Transcription and translation routes

The server answers `/v1/audio/transcriptions` and `/v1/audio/translations`
with this model, with no Whisper and no `stt` extra. The server needs
ffmpeg on its PATH (`brew install ffmpeg`) to decode the upload.

```sh
curl localhost:8080/v1/audio/transcriptions -F file=@meeting.wav -F model=whisper-1
curl localhost:8080/v1/audio/translations -F file=@interview.ogg -F model=whisper-1
```

A translation is always into English. The server picks the model at
start and names it by its id:

```text
[server] stt: nvidia-nemotron-3.5-transcribe-30b-a4b-ad  (POST /v1/audio/transcriptions, the served model itself)
```

[`server.stt`](config.md#serverstt) decides which model answers:

| Value | Meaning |
|-------|---------|
| Unset | The default model when it hears audio, else the one configured model that does. With two or more and neither the default, no model is picked. |
| `nemotron-3.5-transcribe` | The configured model with this id or alias. A model that hears no audio is a config error. |
| `false` | No speech-to-text. The routes answer 404. |

A request whose `model` is `whisper-1`, `default`,
`gpt-4o-transcribe-diarize` or empty goes to the `server.stt` model. A
request that names a served model which hears audio goes to that model.

The routes take these form fields:

| Name | Default | Meaning |
|------|---------|---------|
| `response_format` | `json` | `json`, `text`, `verbose_json`, `srt` or `vtt`, and `diarized_json` on transcriptions |
| `language` | None | Transcriptions only. A code such as `en` or `pt-BR` adds a language hint, and a code the model has no hint for adds none. |
| `diarization` | `false` | `true` labels the speakers, as [Speakers and word times](#speakers-and-word-times) shows |
| `timestamp_granularities[]` | None | `word` adds `words` to `verbose_json`. Segments are always there. |
| `temperature` | `0` | Sampling temperature, 0 to 2 |
| `prompt` | None | Ignored |
| `chunking_strategy` | None | Ignored |
| `known_speaker_names[]`, `known_speaker_references[]` | None | A 400: the server does not enroll speakers |

A recording of near-digital silence, or one where the encoder hears no
speaker, gets an empty transcript. The model's event and style tags never
appear in the text. Transcriptions and translations do not stream.

## Speakers and word times

`diarization=true` switches to the card's speaker instruction and labels
each speaker. It works with `verbose_json`, `srt`, `vtt` and
`diarized_json`, and `json` or `text` get a 400:

```sh
curl localhost:8080/v1/audio/transcriptions -F file=@meeting.wav -F model=whisper-1 \
  -F response_format=verbose_json -F diarization=true -F 'timestamp_granularities[]=word'
```

```jsonc
{"task": "transcribe", "language": null, "duration": 166.144, "text": "He had written a number of books himself, ...",
 "segments": [{"id": 0, "start": 0.36, "end": 14.096, "text": "He had written a number of books himself, ... et cetera.", "speaker": 1}, ...],
 "words": [{"word": "He", "start": 0.36, "end": 0.524, "speaker": 1}, ...]}
```

Times are in seconds. Each format names the speakers in its own way:

| Format | Speakers |
|--------|----------|
| `verbose_json` | `speaker`, a number from 1, on each segment and word. Without `diarization`, the key is left out. |
| `srt` | `Speaker 1: ` before the first cue of each turn |
| `vtt` | A voice tag on each cue, such as `<v Speaker 1>` |
| `diarized_json` | OpenAI's diarized shape: one segment per turn, with speakers `A`, `B` and on in order of first speech |

`response_format=diarized_json` turns diarization on by itself, and
`timestamp_granularities[]` with it gets a 400. The speaker tags never
appear in the text of any format.

A transcript that the aligner cannot time still returns with a 200. In
`verbose_json` and `diarized_json`, `words` and `segments` are empty and
`alignment_error` names the reason. In `srt` and `vtt`, one cue spans the
whole recording. A Thai transcript gets no word times.

The model writes at most four speaker tags, `<spk:0>` to `<spk:3>`, so a
recording with more than four speakers gets at most four labels.

## A voice session on one model

[`gmlx talk`](talk.md) can use this model to hear you and to answer. The
server needs only a text-to-speech service beside it:

```yaml
server:
  tts: kokoro

talk:
  model: nemotron-3.5-transcribe@instruct
```

```sh
gmlx talk
```

With `server.stt` unset and this model the only one that hears audio, it
transcribes what you say, and `talk.model` makes it answer too.

## Speculative decoding

The decoder file carries a [native head](glossary.md#native-head) that
drafts 7 tokens a round, in `run`, `chat` and `serve`, on audio turns as
on text turns. `--no-mtp` turns it off, `--draft-block-size` sets a
smaller block, and the `GMLX_NEM_MTP_BLOCK` environment variable sets the
block of every Nemotron-H head, drafts plus one.

On a Mac with less than 48 GB, `gmlx pull` registers the model with
`speculative: false`, because the head takes memory that a long recording
needs. From 48 GB up, the pull leaves speculation on.
[Speculative decoding](speculative-decoding.md) covers the rest.

## Limits

| Limit | Value |
|-------|-------|
| Clips in one request or chat conversation | 6. A seventh is refused, with a hint to start a new conversation. |
| Audio in one request | 3600 seconds in all |
| Context | 65,536 tokens. Each second of audio is 12.5 tokens, so an hour is 45,000. |
| Speakers | 4 labels |
| Word times in chat | One clip and `n` of 1 |

The server checks memory before the encoder runs, and refuses with a 400
a request that does not fit. Word times hold extra memory until the words
are timed, so on a 32 GB Mac a long recording can fit as plain text and
not with word times. Message text that holds one of the model's control
tokens also gets a 400.
