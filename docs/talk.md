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

The `talk` extra needs Python 3.11-3.13 (on 3.14, the Kokoro voice's
phoneme stack has no wheels yet - configure the `qwen3-tts` model instead).

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
together. macOS asks for microphone permission once - and the prompt names
your *terminal* (Terminal, iTerm2, your IDE), not gmlx, because macOS grants
the mic to the app you launched from. Allow it. (Voice sessions started from
the menu-bar login item are the exception: those prompt as "gmlx" - see
[menu-bar voice sessions](#menu-bar-voice-sessions).) If it was denied,
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

Useful controls while it runs: Space stops the assistant mid-sentence (and is
push-to-talk in `ptt` mode), Esc cancels the current turn, `m` mutes the mic, `q`
quits. Start typing at any time to send a text message instead of speaking; lines
starting with `/` are commands. Try voices live:

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
on the server's own embeddings. This section is the voice-flavored tour;
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

Tool calling needs a model that is competent at it. The Qwen3.6-27B class is a good
fit on a 48 GB or larger machine; Qwen3.5-9B is a workable floor on 32 GB, with more
tool fumbles. Then:

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
transcript. After each turn, a background request asks the chat model to boil the
exchange down to at most three durable facts, or none, so small talk leaves no
residue. A new fact that restates an existing one replaces it. Extraction runs off
the voice path and adds no latency.

Two honest caveats. Each tool round adds seconds (a full model turn plus the tool
call), so multi-tool answers are noticeably slower than plain chat. And a barge-in
still interrupts cleanly: the loop commits what you heard and never leaves a
half-finished tool round in the history.

To inspect or edit memory from inside a session, `/memory` lists the stored
facts with their ids, `/memory forget ID` removes one, and `/memory clear yes`
removes them all. The menu-bar app's voice session exposes the same store:
"Show memory" prints the list into the transcript panel, and "Clear memory"
wipes it after a confirmation dialog. The store itself is a plain sqlite file
at `~/.local/share/gmlx/assistant-memory.db` if you want to look deeper.
It is shared with `gmlx chat --assistant`: a fact taught by voice recalls
in the text REPL, and vice versa.

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
(`vad.pre_roll_ms` and `push_to_talk_modifier` are config-only); precedence is
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
[assistant.md](assistant.md); it is shared with `gmlx chat --assistant` and
`server.assistants`, so it does not live under `talk:`. Talk uses it whenever
`talk.brain: assistant` is set, and everything there applies as-is: MCP
tool-name prefixing, the degrade-to-warning behavior when a tool server or the
`[assistant]` extra is missing, and the memory lifecycle
([assistant.md#memory](assistant.md#memory)).

## Menu bar voice sessions

When the tracked server advertises STT and TTS, the macOS menu bar app
(`gmlx launch menubar`, raised automatically by a background `serve`) shows a
"Talk to <model>" item, named after `talk.model` or the server's default model.
Clicking it starts a voice session inside the menu bar app, no terminal window: the
bar icon changes to show the state (a microphone while listening, a thought bubble
while the model thinks, a speaker while it talks, a muted-speaker while the mic is
off), and the menu offers Stop speaking, Mute mic, and End voice chat. Show
transcript opens a floating panel with the running conversation text. A Volume
slider under the session controls scales the voice (and the chimes) relative
to the system output volume - it applies mid-sentence while dragging, and the
setting persists across sessions. Mic input has no gain control on purpose:
software input gain would shift the endpointing and wake-word thresholds and
clip loud speech; use the macOS Sound settings input level instead.

All settings come from the YAML `talk:` block; push-to-talk and text modes fall
back to wake mode, since there is no keyboard. A "Talk in a terminal" item opens
`gmlx talk` in iTerm2 when it is running, otherwise the default terminal handler.
No AppleScript is involved, so there is no automation-permission popup.

Starting a voice session (either surface) holds its model resident on the
server for the session's lifetime - loaded and warmed up front, exempt from
the idle reaper, released (not evicted) when the session ends - so an open
mic never sits in front of an unloaded model.

### Tap-to-talk hotkey

The menu bar can bind a global tap-to-talk combo that works from any app:
a "Tap-to-talk with Globe + Space" toggle (Space pressed while the Globe/fn
key is held). Firing it always drives toward an open mic, whatever the
session is doing: no session running starts one; idle opens the mic (wake
chime); tapping again while listening dismisses; mid-capture it ends the
utterance immediately; and while the assistant is transcribing, thinking,
or speaking it barges in and listens. A tap while muted unmutes first -
pressing the key is explicit intent to talk.

Keyboards without a Globe key (most non-Apple desktop keyboards) pick a
different modifier - `gmlx init` asks for it in the voice-chat step, or set
it in the config's `talk:` block:

```yaml
talk:
  push_to_talk_modifier: globe   # or: right-command | right-option | control
```

The menu item shows whichever combo is active. The right-side variants are
deliberate - left Cmd+Space is Spotlight and left Option+Space is a common
launcher bind, while the right-side keys are nearly always free.

The hotkey swallows the Space keystroke so a space is not typed into the
focused app, which requires an active event tap - *Accessibility*
permission, requested only when you first enable the hotkey (never at
launch). While armed, every keystroke in the login session passes through
the tap (the overhead is small, and the menu bar process does no
inference). Holding Globe as a modifier suppresses macOS's own Globe-key
action, so your "Press Globe key to" setting (emoji, dictation, input
source) keeps working for bare presses - nothing to reconfigure.

(A bare double-press of Globe was considered and dropped: current macOS
routes the solo Globe press to the system shortcut handler without posting
an event that session event taps can see - only raw HID sees it - so
double-press stays available for the system's own dictation shortcut.)

The choice persists across launches. On startup the app re-arms it only
after a silent permission check - if the grant is missing (denied, or
silently dropped by an app-stub re-sign after an interpreter upgrade) a
"not active - needs permission" note appears under the toggle and nothing
prompts until you flip it again. Granting access in System Settings while
the bar is running is picked up within a few seconds and the hotkey arms
itself. If arming still fails right after a grant (some macOS versions bind
TCC grants only to a freshly launched process), the app says to quit and
reopen the menu bar.

Permission prompts attribute to "gmlx" when the bar runs as the launchd
login item (`gmlx service install`). A bar launched from a terminal
(`gmlx launch menubar`) runs under the terminal's TCC identity instead, so
grants then attach to the terminal app, not gmlx.

## Remote server and scripting

`--base-url http://host:8080/v1` (with `--api-key` if the server has one) points the
client at a server elsewhere. STT and TTS then run on that machine; only the mic and
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

Only wake-phrase scoring runs during a reply; full transcription of the open
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
[testing.md](testing.md#voice-loop-manual-pass).
