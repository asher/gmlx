# Voice chat

This guide is for talking to a served model by voice with `gmlx talk`. You
say the wake phrase and speak, and the reply is spoken back as it streams.
The guide covers setup, a worked example, the listening modes, the
in-session keys, the config block, and what sets the latency.

`talk` is a client of the gmlx server. The whole loop runs against the
server's endpoints. Transcription goes in, a chat turn streams back, and
speech comes out sentence by sentence. The speech models and the language
model therefore share a GPU under the server's arbitration. Expect 1.3 to
2.2 seconds from the end of your speech to the first spoken audio.

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

On first run two small files download into `~/.cache/gmlx/talk/`. They are
the keyword-spotting bundle and the voice-activity model, a few MB
together. macOS then asks for microphone permission once. Allow it. The
prompt names your terminal instead of gmlx, because macOS grants the mic to
the app you launched from. Voice sessions started from the
[menu bar](menubar.md#voice-sessions) prompt as gmlx instead. If the prompt
was denied, re-enable it under System Settings, Privacy and Security,
Microphone.
[troubleshooting.md](troubleshooting.md#the-mic-never-works-in-talk) has
the steps.

## Worked example

Start from a machine with a config and a served model:

```sh
gmlx init            # rerun the wizard, add STT and TTS, take the voice-chat step
gmlx talk
```

`talk` starts the server if it is down, waits for it, checks the speech
services, and listens:

```text
listening for "hey assistant"  (say it, and a rising chime confirms)
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

The wake phrase is plain text with no training. Because the keyword spotter
is an open-vocabulary transducer, any phrase is spelled into tokens at
startup.
Continuous listening costs well under one percent of a CPU core. If the wake
engine is not installed, `talk` falls back to open-mic mode with an install
hint.

## Modes

| Mode | Mic behavior |
|------|--------------|
| `wake`, the default | listens for the wake phrase, then captures an utterance. The phrase also interrupts a reply in progress |
| `vad` | open mic, where any speech starts a turn |
| `ptt` | push-to-talk, where Space starts and ends a capture |
| `text` | no mic, a typed prompt whose replies are still spoken |

## Keys and slash commands

While running, Space stops speech or drives push-to-talk, Esc cancels the
current turn, `m` mutes the mic, and `q` quits. Typing any printable
character switches to line input.

| Command | Effect |
|---------|--------|
| `/voice [name]` | list the server's voices, or switch |
| `/speed <0.25-4>` | speech speed |
| `/mode wake|vad|ptt|text` | switch listening mode |
| `/wake [phrase]` | show or change the wake phrase |
| `/mute` | toggle the mic |
| `/system <prompt>` | set the spoken persona |
| `/reset` | clear the conversation |
| `/memory` | list stored memories, with `forget ID` and `clear` to manage them, on the assistant brain only |
| `/devices` | list audio devices |
| `/help`, `/quit` | show the commands, or quit |

## The assistant by voice

`talk.brain: assistant` switches the turn engine from plain chat to the
built-in [assistant](assistant.md). The model can then call tools mid-turn,
and the conversation gains long-term memory. Tools come from MCP servers
you configure. Memory is a local store built on the server's embeddings.
This example configures two MCP servers that run locally with no API keys,
and turns memory on. The reference filesystem server needs Node, the
reference fetch server needs uv, and memory needs `embeddings:` on the
server:

```sh
uv tool install "gmlx[talk,assistant]"     # or: pip install "gmlx[talk,assistant]"
```

```yaml
server:
  model_dirs: [~/models]
  stt: whisper-turbo
  tts: kokoro
  embeddings: qwen3-embed-0.6b     # required for memory
  rerank: qwen3-rerank-0.6b        # optional, reorders recalled memories
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

Tool calling needs a model that is competent at it. Qwen3.6-27B is the
recommended class. A session then looks like this, with tool activity in
the status line and only the answer spoken:

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

Quit, relaunch later, and ask when your sister's birthday is. The
assistant answers from memory. What it stored is an extracted fact, not a
transcript. Extraction runs in the background after the turn and adds no
latency. The store is shared with `gmlx chat --assistant`, and `/memory`
inspects it from inside a session. For the rules, the on-disk location and
the security model, read [assistant.md](assistant.md#memory).

Tool rounds cost time, a model turn plus the call for each round, and
multi-tool answers are slower than plain chat. A barge-in during a tool
round is still handled correctly. The loop commits what you heard and never
leaves a half-finished tool round in the history.

## Configuration reference

All keys sit in a top-level `talk:` block of the YAML the server reads.
The block configures the client and is therefore not under `server:`. Most keys
have a matching flag under [gmlx talk](cli.md#gmlx-talk).
`vad.pre_roll_ms` and `push_to_talk_modifier` are config-only. Precedence
is defaults, then YAML, then flags.

```yaml
talk:
  model: qwen3.6-27b@instruct   # id[@profile], default the server's default model
  voice: af_heart               # a Kokoro preset or qwen3-tts speaker
  speed: 1.0
  system: null                  # spoken persona. Omit the key for the default speakable-output
                                #   prompt. A literal null or "" sets no persona, not the default.
  language: null                # whisper language hint
  max_tokens: null              # reply cap, unset means until the model stops
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

The [menu bar app](menubar.md#voice-sessions) runs the same loop without a
terminal and can bind a tap-to-talk hotkey.

## Remote server and scripting

`--base-url http://host:8080/v1`, with `--api-key` if the server has one,
points the client at a server elsewhere. Speech-to-text and text-to-speech
then run on that machine. Only the mic and speaker are local. Without
`--base-url`, `talk` targets the managed local server and starts it when
down. `--no-start` disables that.

`--once` runs a single ask-and-answer exchange and exits, skipping the wake
gate, which suits scripting and smoke-testing a setup.

## Latency and interruption

End of speech to first audio is typically 1.3 to 2.2 seconds. That is the
sum of the endpointer's 550 ms silence hangover, 300 to 500 ms of Whisper
turbo, the model's first sentence, and 150 to 300 ms of Kokoro synthesis.
Replies are chunked at sentence boundaries and synthesized a sentence ahead
of playback, which keeps long answers speaking continuously.

| Tuning | Trade |
|--------|-------|
| `vad.silence_ms` down to about 400 | a faster turn at the cost of more mid-sentence cutoffs |
| `stt: whisper-turbo-q4` | shortens the transcription step |
| `max_tokens` around 512 | keeps answers short |

In wake mode the wake phrase itself interrupts a reply. The keyword
spotter stays live while the assistant transcribes, thinks and speaks.
Saying the phrase mid-reply stops playback, cancels the turn and opens the
mic. A stop phrase after it, such as "stop", "cancel" or "never mind",
acknowledges and returns to waiting for the wake phrase instead of starting
a turn. Space and Esc do the same from the keyboard within about 150 ms.

Only wake-phrase scoring runs during a reply. Full transcription of the
open mic stays gated, since playback would otherwise be re-transcribed.
`vad` and `ptt` modes are therefore half-duplex and keyboard-interrupt
only. There is no protection against the assistant speaking the wake
phrase. If a reply quotes it aloud, the spotter detects it through the
speakers. Pick a phrase the model is unlikely to say. Whisper's known
hallucinations on silence and noise are filtered by a minimum-speech and
energy floor before transcription and a known-phrase check after, and noise
does not become a turn.

The manual smoke checklist for this loop, for contributors, is in
[internals/testing.md](internals/testing.md#voice-loop-manual-pass).
