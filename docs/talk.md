# Voice chat: `gmlx talk`

Hands-free voice chat with any model your server serves: say the wake phrase, speak,
and the reply is spoken back as it streams. `talk` is a client of the gmlx
server. The whole loop runs against the existing OpenAI-compatible endpoints (mic to
wake word to endpointing, `/v1/audio/transcriptions` for speech-to-text, a streamed
`/v1/chat/completions` turn, sentence-buffered `/v1/audio/speech` back out), so the
speech models and the LLM share one Metal device under the server's arbitration.

Expect about 1.3 to 2.2 seconds from the end of your speech to the first spoken
audio. The [latency section](#latency-and-interruption) breaks that down and shows
what tuning can and cannot buy.

## Quick setup

```sh
pip install 'gmlx[talk]'    # client audio + wake word, includes server STT/TTS deps
brew install ffmpeg             # Whisper's audio decoding
```

The `talk` extra runs on any supported Python, 3.11 through 3.14.

The server needs both audio services in its config:

```yaml
server:
  stt: whisper-turbo
  tts: kokoro
```

`gmlx init` offers both, and adds a voice-chat step (voice, wake phrase, listen
mode) whenever you configure the two together. If something is missing at startup,
`gmlx talk` prints the exact lines to add.

On first run, two small files download into `~/.cache/gmlx/talk/`: the
sherpa-onnx keyword-spotting bundle and the silero voice-activity model, a few MB
together. macOS asks for microphone permission once, and the prompt names
your *terminal* (Terminal, iTerm2, your IDE), not gmlx, because macOS grants
the mic to the app you launched from. Allow it. (Voice sessions started from
the menu-bar login item are the exception: those prompt as "gmlx" - see
[menu-bar voice sessions](menubar.md#voice-sessions).) If it was denied,
nothing is stuck: re-enable in System Settings > Privacy & Security >
Microphone ([troubleshooting](troubleshooting.md#the-mic-never-works-in-talk)).

## Worked example: voice chat from zero

Starting from a machine with a config and one served model (see
[getting-started.md](getting-started.md) to get there):

```sh
pip install 'gmlx[talk]'
brew install ffmpeg
gmlx init            # rerun the wizard; add STT + TTS and take the voice-chat step
gmlx talk
```

`talk` starts the server if it is down, waits for it, checks the audio services, and
listens. A session looks like this:

```text
listening for "hey assistant"  (say it; a rising chime confirms)
you: what's a good name for a gray cat?
assistant: How about Ash? It suits a gray coat, and it's short enough
that the cat might actually learn it.
listening for "hey assistant"
```

Useful while it runs: Space stops the assistant mid-sentence, and typing at any
time sends a text message instead of speaking. The full set is in
[keys and slash commands](#keys-and-slash-commands). Try voices live:

```text
/voice            # list the server's voices
/voice bf_emma    # switch mid-session
/wake okay computer
```

The wake phrase is plain text, no training: the keyword spotter is an
open-vocabulary transducer, so any phrase is spelled into tokens at startup.
Continuous listening costs well under one percent of one CPU core. If the wake
engine is not installed, `talk` degrades to open-mic mode with a pip hint rather
than failing.

## Worked example: the assistant

`talk.brain: assistant` upgrades the turn engine from plain chat to the built-in
[assistant](assistant.md). The model can call tools mid-turn (the standard OpenAI
tool loop, run against this same server), and the conversation gains long-term
memory. Tools come from MCP servers you configure; memory is a local store built
on the server's own embeddings. This section is the voice-flavored tour.
[assistant.md](assistant.md) is the full reference, and the same engine also
drives `gmlx chat --assistant` and served assistant ids.

This example wires two MCP servers that run locally with no API keys: the reference
filesystem server (needs Node) and the reference fetch server (needs uv). It also
enables memory, which requires the server to have `embeddings:` configured
(`rerank:` is optional but improves recall ordering).

Install the extras:

```sh
pip install 'gmlx[talk,assistant]'   # adds the MCP SDK
```

The complete config:

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

Tool calling needs a model that is competent at it. The config's Qwen3.6-27B is
the recommended class ([model picks](assistant.md#the-tool-loop)). Then:

```sh
gmlx talk
```

A representative session (tool activity appears in the status line; only the answer
is spoken):

```text
listening for "hey assistant"
you: which of my notes mentions the tax deadline?
  using search_files
  using read_text_file
assistant: Your note taxes-2026.md mentions it. The filing deadline you
wrote down is April 15th, with the extension window to October 15th.

you: fetch the MLX github page and tell me what the latest release says.
  using fetch
assistant: The latest release listed is 0.31.2. The notes highlight new
fast attention paths and several quantization fixes.

you: remember that my sister Ana's birthday is March 12th.
assistant: Noted. Ana's birthday is March 12th.
```

Quit, relaunch `gmlx talk` later, and ask:

```text
you: when is my sister's birthday?
assistant: Ana's birthday is March 12th.
```

What got stored is a distilled fact ("sister Ana, birthday March 12"), not a
transcript. The distillation runs in the background after each turn, off the
voice path, so it adds no latency. The rules (at most three durable facts,
restatement replaces) are in [assistant.md#memory](assistant.md#memory).

Two honest caveats. Tool rounds cost time - a model turn plus the call, each -
so multi-tool answers are noticeably slower than plain chat. And a barge-in
still interrupts cleanly: the loop commits what you heard and never leaves a
half-finished tool round in the history.

`/memory` inspects and edits the store from inside a session
([keys and slash commands](#keys-and-slash-commands)). The menu-bar voice
session exposes the same store through its "Show memory" and "Clear memory"
items. The store is shared with `gmlx chat --assistant` - details and the
on-disk location in [assistant.md#memory](assistant.md#memory).

## Modes

| Mode | Mic behavior |
|------|--------------|
| `wake` (default) | Listens for the wake phrase, then captures one utterance. The phrase also interrupts a reply in progress ([details](#latency-and-interruption)). |
| `vad` | Open mic. Any speech starts a turn. |
| `ptt` | Push-to-talk. Space starts and ends a capture. |
| `text` | No mic. A typed REPL whose replies are still spoken. |

## Keys and slash commands

While running: Space stops speech (barge-in) or drives push-to-talk, Esc cancels the
current turn, `m` mutes the mic, `q` quits. Typing any printable character drops
into line input. Commands: `/voice [name]` (bare `/voice` lists the server's
voices), `/speed <0.25-4>`, `/mode wake|vad|ptt|text`, `/wake [phrase]`, `/mute`,
`/system <prompt>`, `/reset` (clear the conversation), `/memory` (list stored
memories; `forget ID` and `clear` manage them), `/devices`, `/help`, `/quit`.

## Configuration reference

Everything lives in a top-level `talk:` block of the same YAML the server reads. It
configures the client, so it is not under `server:`. Most keys have a flag mirror
(`vad.pre_roll_ms` and `push_to_talk_modifier` are config-only). Precedence is
defaults, then YAML, then flags. `--list-devices` and `--list-voices`
enumerate the device and voice values.

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
  chime: true                   # earcons on wake and turn end
  brain: chat                   # chat | assistant
```

## Assistant reference

The assistant itself -- the top-level `assistant:` block with tool servers,
memory settings, and their defaults -- is documented in
[assistant.md](assistant.md). It is shared with `gmlx chat --assistant` and
`server.assistants`, so it does not live under `talk:`. Talk uses it whenever
`talk.brain: assistant` is set, and everything there applies as-is: MCP
tool-name prefixing, the degrade-to-warning behavior when a tool server or the
`[assistant]` extra is missing, and the memory lifecycle
([assistant.md#memory](assistant.md#memory)).

## Menu bar voice sessions

The menu bar app can run a voice session without a terminal, and can bind a
tap-to-talk hotkey. Both are described in [menubar.md](menubar.md#voice-sessions).

## Remote server and scripting

`--base-url http://host:8080/v1` (with `--api-key` if the server has one) points the
client at a server elsewhere. STT and TTS then run on that machine. Only the mic and
speaker are local. Without `--base-url`, `talk` targets the managed local server and
starts it when down (`--no-start` disables that).

`--once` runs a single ask-and-answer exchange and exits, skipping the wake gate.
Useful for scripting and for smoke-testing a setup.

## Latency and interruption

End of speech to first audio is typically 1.3 to 2.2 seconds: the endpointer's
silence hangover (550 ms), Whisper turbo (roughly 300 to 500 ms), the LLM's first
sentence, and Kokoro synthesis (roughly 150 to 300 ms). Replies are chunked at
sentence boundaries and synthesized one sentence ahead of playback, so long answers
speak continuously. Tuning options: `vad.silence_ms` down to about 400 trades a
snappier turn for more mid-sentence cutoffs, `whisper-turbo-q4` shaves the STT step,
and a short `max_tokens` (e.g. 512) keeps answers conversational.

In wake mode the wake phrase itself barges in: the keyword spotter stays live
while the assistant transcribes, thinks, and speaks, so saying the wake phrase
mid-reply stops playback, cancels the turn, and opens the mic for the next
utterance. Saying just a stop phrase after that ("stop", "cancel", "never
mind", and the like) acknowledges and goes back to sleep instead of starting a
turn -- so "<wake phrase>, stop" kills a long reply by voice. Space and Esc do
the same from the keyboard, and stop playback within about 150 ms.

Only wake-phrase scoring runs during a reply. Full transcription of the open
mic stays gated (playback would otherwise be re-transcribed), so `vad` and
`ptt` modes remain half-duplex, keyboard-interrupt only. One caveat: there is
no protection against the assistant *speaking* the wake phrase -- if a reply
quotes it aloud, the spotter hears it through the speakers and the assistant
interrupts itself. Pick a wake phrase it is unlikely to say. Full acoustic echo
cancellation, which would fix that and allow talking over the assistant in any
mode, needs the OS voice-processing audio unit and is on the roadmap.

Whisper hallucinates stock phrases ("thank you") on silence and noise. `talk`
filters these with a minimum-speech and energy floor before transcription and a
known-ghost check after, so noise does not become a turn.

Contributors: the manual smoke checklist for this loop lives in
[testing.md](internals/testing.md#voice-loop-manual-pass).
