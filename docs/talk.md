# Voice chat

This guide is for talking to a served model by voice with `gmlx talk`: say
the wake phrase, speak, and the reply is spoken back as it streams. It covers
setup, a worked example, the listening modes, the in-session keys, the config
block, and what sets the latency.

`talk` is a client of the gmlx server. The whole loop runs against the
server's own endpoints, transcription in, a streamed chat turn, and speech
out sentence by sentence, so the speech models and the language model share
one GPU under the server's arbitration. Expect 1.3 to 2.2 seconds from the
end of your speech to the first spoken audio.

- [Setup](#setup)
- [Worked example](#worked-example)
- [Modes](#modes)
- [Keys and slash commands](#keys-and-slash-commands)
- [The assistant by voice](#the-assistant-by-voice)
- [Configuration reference](#configuration-reference)
- [Remote server and scripting](#remote-server-and-scripting)
- [Latency and interruption](#latency-and-interruption)

## Setup

```sh
uv tool install "gmlx[talk]"     # or: pip install "gmlx[talk]"
brew install ffmpeg              # audio decoding for Whisper
```

The server needs both speech services in its config:

```yaml
server:
  stt: whisper-turbo
  tts: kokoro
```

`gmlx init` offers both and adds a voice-chat step, for the voice, wake
phrase and listen mode, whenever you configure the two together. If
something is missing at startup, `gmlx talk` prints the exact lines to add.
The services themselves are described in [services.md](services.md).

On first run two small files download into `~/.cache/gmlx/talk/`, the
keyword-spotting bundle and the voice-activity model, a few MB together.
macOS then asks for microphone permission once. The prompt names your
terminal rather than gmlx, because macOS grants the mic to the app you
launched from; allow it. Voice sessions started from the
[menu bar](menubar.md#voice-sessions) prompt as gmlx instead. If the prompt
was denied, re-enable it under System Settings, Privacy and Security,
Microphone ([troubleshooting](troubleshooting.md#the-mic-never-works-in-talk)).

## Worked example

Starting from a machine with a config and one served model:

```sh
gmlx init            # rerun the wizard; add STT and TTS and take the voice-chat step
gmlx talk
```

`talk` starts the server if it is down, waits for it, checks the speech
services, and listens:

```text
listening for "hey assistant"  (say it; a rising chime confirms)
you: what's a good name for a gray cat?
assistant: How about Ash? It suits a gray coat, and it's short enough
that the cat might actually learn it.
listening for "hey assistant"
```

Space stops the assistant mid-sentence, and typing at any time sends a text
message instead of speaking. Try voices live:

```text
/voice            # list the server's voices
/voice bf_emma    # switch mid-session
/wake okay computer
```

The wake phrase is plain text with no training: the keyword spotter is an
open-vocabulary transducer, so any phrase is spelled into tokens at startup.
Continuous listening costs well under one percent of a CPU core. If the wake
engine is not installed, `talk` falls back to open-mic mode with an install
hint.

## Modes

| Mode | Mic behavior |
|------|--------------|
| `wake` (default) | listens for the wake phrase, then captures one utterance; the phrase also interrupts a reply in progress |
| `vad` | open mic: any speech starts a turn |
| `ptt` | push-to-talk: Space starts and ends a capture |
| `text` | no mic: a typed prompt whose replies are still spoken |

## Keys and slash commands

While running, Space stops speech or drives push-to-talk, Esc cancels the
current turn, `m` mutes the mic, and `q` quits. Typing any printable
character drops into line input.

| Command | Effect |
|---------|--------|
| `/voice [name]` | list the server's voices, or switch |
| `/speed <0.25-4>` | speech speed |
| `/mode wake|vad|ptt|text` | switch listening mode |
| `/wake [phrase]` | show or change the wake phrase |
| `/mute` | toggle the mic |
| `/system <prompt>` | set the spoken persona |
| `/reset` | clear the conversation |
| `/memory` | list stored memories; `forget ID` and `clear` manage them (assistant brain only) |
| `/devices` | list audio devices |
| `/help`, `/quit` | |

## The assistant by voice

`talk.brain: assistant` upgrades the turn engine from plain chat to the
built-in [assistant](assistant.md): the model can call tools mid-turn and the
conversation gains long-term memory. Tools come from MCP servers you
configure, and memory is a local store built on the server's own embeddings.
This example wires two MCP servers that run locally with no API keys, the
reference filesystem server (needs Node) and the reference fetch server
(needs uv), and turns memory on, which needs `embeddings:` on the server:

```sh
uv tool install "gmlx[talk,assistant]"     # or: pip install "gmlx[talk,assistant]"
```

```yaml
server:
  model_dirs: [~/models]
  stt: whisper-turbo
  tts: kokoro
  embeddings: qwen3-embed-0.6b     # required for memory
  rerank: qwen3-rerank-0.6b        # optional: reorders recalled memories
  defaults:
    model: qwen3.6-27b

models:
  qwen3.6-27b:
    path: Qwen3.6-27B-Q6_K.gguf    # relative to model_dirs

talk:
  model: qwen3.6-27b@instruct
  brain: assistant

assistant:
  mcp:
    - name: files
      command: [npx, -y, "@modelcontextprotocol/server-filesystem", "~/notes"]
    - name: web
      command: [uvx, mcp-server-fetch]
  memory:
    enabled: true
```

Tool calling needs a model that is competent at it; Qwen3.6-27B is the
recommended class. A session then looks like this, with tool activity in the
status line and only the answer spoken:

```text
listening for "hey assistant"
you: which of my notes mentions the tax deadline?
  using search_files
  using read_text_file
assistant: Your note taxes-2026.md mentions it. The filing deadline you
wrote down is April 15th, with the extension window to October 15th.

you: remember that my sister Ana's birthday is March 12th.
assistant: Noted. Ana's birthday is March 12th.
```

Quit, relaunch later, ask when your sister's birthday is, and the assistant
answers from memory. What it stored is a distilled fact, not a transcript,
extracted in the background after the turn so it adds no latency. The store
is shared with `gmlx chat --assistant`, and `/memory` inspects it from inside
a session. The rules, the on-disk location and the security model are in
[assistant.md](assistant.md#memory).

Two things to know. Tool rounds cost time, a model turn plus the call each,
so multi-tool answers are slower than plain chat. And a barge-in still
interrupts cleanly: the loop commits what you heard and never leaves a
half-finished tool round in the history.

## Configuration reference

Everything lives in a top-level `talk:` block of the same YAML the server
reads. It configures the client, so it is not under `server:`. Most keys have
a flag mirror under [gmlx talk](cli.md#gmlx-talk); `vad.pre_roll_ms` and
`push_to_talk_modifier` are config-only. Precedence is defaults, then YAML,
then flags.

```yaml
talk:
  model: qwen3.6-27b@instruct   # id[@profile]; default: the server's default model
  voice: af_heart               # a Kokoro preset or qwen3-tts speaker
  speed: 1.0
  system: null                  # spoken persona; omit the key for the default speakable-output
                                #   prompt. A literal null (or "") sets no persona, not the default.
  language: null                # whisper language hint
  max_tokens: null              # reply cap; unset = until the model stops
  mode: wake                    # wake | vad | ptt | text
  wake_word: "hey assistant"    # any text phrase
  wake_threshold: 0.3           # higher = fewer false fires
  vad:
    threshold: 0.6              # silero speech probability
    silence_ms: 550             # pause length that ends an utterance
    min_speech_ms: 300          # shorter captures are dropped
    pre_roll_ms: 400            # audio kept from before speech onset
  input_device: null            # sounddevice name substring or index
  output_device: null
  chime: true                   # sounds on wake and turn end
  brain: chat                   # chat | assistant
```

The menu bar app runs the same loop without a terminal, and can bind a
tap-to-talk hotkey; both are in [menubar.md](menubar.md#voice-sessions).

## Remote server and scripting

`--base-url http://host:8080/v1`, with `--api-key` if the server has one,
points the client at a server elsewhere. Speech-to-text and text-to-speech
then run on that machine; only the mic and speaker are local. Without
`--base-url`, `talk` targets the managed local server and starts it when down
(`--no-start` disables that).

`--once` runs a single ask-and-answer exchange and exits, skipping the wake
gate, which suits scripting and smoke-testing a setup.

## Latency and interruption

End of speech to first audio is typically 1.3 to 2.2 seconds, made of the
endpointer's silence hangover (550 ms), Whisper turbo (300 to 500 ms), the
model's first sentence, and Kokoro synthesis (150 to 300 ms). Replies are
chunked at sentence boundaries and synthesized one sentence ahead of
playback, so long answers speak continuously.

| Tuning | Trade |
|--------|-------|
| `vad.silence_ms` down to about 400 | a snappier turn against more mid-sentence cutoffs |
| `stt: whisper-turbo-q4` | shaves the transcription step |
| `max_tokens` around 512 | keeps answers conversational |

In wake mode the wake phrase itself barges in: the keyword spotter stays live
while the assistant transcribes, thinks and speaks, so saying the phrase
mid-reply stops playback, cancels the turn and opens the mic. A stop phrase
after it ("stop", "cancel", "never mind") acknowledges and goes back to sleep
instead of starting a turn. Space and Esc do the same from the keyboard
within about 150 ms.

Only wake-phrase scoring runs during a reply. Full transcription of the open
mic stays gated, since playback would otherwise be re-transcribed, so `vad`
and `ptt` modes are half-duplex and keyboard-interrupt only. There is no
protection against the assistant speaking the wake phrase: if a reply quotes
it aloud, the spotter hears it through the speakers. Pick a phrase the model
is unlikely to say. Whisper's stock hallucinations on silence and noise are
filtered by a minimum-speech and energy floor before transcription and a
known-phrase check after, so noise does not become a turn.

Contributors: the manual smoke checklist for this loop is in
[internals/testing.md](internals/testing.md#voice-loop-manual-pass).
