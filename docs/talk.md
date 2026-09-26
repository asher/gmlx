# Voice chat

`gmlx talk` is a voice client for the models that the gmlx server serves. It
listens for a wake phrase, sends what you say to the model and speaks the
reply as it streams. You can interrupt it, tune it and script it.

The client records your speech, and the server turns it into text, streams
the reply of the model, and turns each part of the reply into speech while
the model is still writing. Speech recognition, speech synthesis and the
model all run in the server, so a remote server does the work while the
microphone and speakers stay on your Mac.

- [Setting up](#setting-up)
- [A session](#a-session)
- [Listening modes](#listening-modes)
- [Keys and commands](#keys-and-commands)
- [Interrupting a reply](#interrupting-a-reply)
- [The assistant by voice](#the-assistant-by-voice)
- [Settings](#settings)
- [A remote server and scripts](#a-remote-server-and-scripts)

## Setting up

1. Install the `talk` extra, which includes the audio, wake phrase and
   speech packages. A Homebrew install already has it, and
   [Optional features](installation.md#optional-features) shows how to add
   it to a uv or pip install. Without it, `gmlx talk` exits with
   `error: voice chat requires the optional talk extra`.
2. Turn on the [speech services](services.md) of the server:

    ```yaml
    server:
      stt: whisper-turbo
      tts: kokoro
    ```

3. Run `gmlx init` to choose the voice, the wake phrase, the listening mode
   and the hotkey of the menu bar app. The wizard asks these questions
   whenever it sets up both speech services, and it offers to install the
   `talk` extra when it is missing.
4. Run `gmlx restart` so that the server starts the services, and then run
   `gmlx talk`.

When a speech service is off, `gmlx talk` prints the lines to add to the
configuration file. The first session downloads a 15 MB wake phrase model
and a 0.6 MB speech detection model into the cache folder that
[Where files are on disk](troubleshooting.md#where-files-are-on-disk)
lists. macOS then asks once for permission to use the microphone, in the
name of your terminal. [Permissions](menubar.md#permissions) explains why,
and [The mic never works in talk](troubleshooting.md#the-mic-never-works-in-talk)
shows how to allow it later.

## A session

`gmlx talk` starts the server if it is down, and then asks it to keep the
model loaded until the session ends. With no model named, it uses
[`talk.model`](config.md#talkmodel), then the default model of the server,
then the only model that the server serves. The first line names the
model, the voice and the mode:

```text
[talk] qwen3.8-27b-ud-q6, voice default, mode wake, say "hey assistant"  (q quits, / commands)
you: what's a good name for a gray cat?
  How about Ash?
  It suits a gray coat, and it is short enough for the cat to learn.
```

Each thing you say prints after `you:`, and each part of the reply prints
indented as it is spoken. A status line at the bottom shows whether the
session is idle, listening, capturing, transcribing, thinking, speaking or
muted. A rising chime means that the microphone is open, and a falling
chime means that it has closed.

To type a message instead, start typing. The first key opens a line
editor, and Enter sends the line. The reply is spoken as usual. A line
that you type while the model is still answering is dropped.

## Listening modes

The mode sets how a turn starts:

| Mode | How a turn starts |
|------|-------------------|
| `wake` | You say the wake phrase, and then speak. This is the default. |
| `vad` | Any speech starts a turn. |
| `ptt` | Space opens the microphone. The turn ends after a pause, or when you press Space again. |
| `text` | You type every message, and the replies are spoken. This mode needs only the text-to-speech service. |

Any English text works as the wake phrase, with no training needed. The
default is "hey assistant", and `/wake` changes it during a session. Choose a
phrase that the model is unlikely to say, because the client also hears
the speakers. When the wake phrase model cannot load, the client prints
why and uses `vad` mode instead.

An utterance ends after 550 milliseconds of silence by default, which
[`talk.vad.silence_ms`](config.md#talkvadsilence_ms) changes. In `ptt`
mode, pressing Space before you speak closes the microphone again.

## Keys and commands

These keys work while the status line shows:

| Key | Effect |
|-----|--------|
| Space or Esc | Stops the reply while the model transcribes, thinks or speaks. In `ptt` mode, Space also opens and closes the microphone. |
| `m` | Mutes or unmutes the microphone, at any time. |
| `q` or Ctrl-D | Quits, when no reply is in progress. |
| Any other letter | Opens the line editor with that letter. |

Because `q`, `m` and Space act at once, a typed message cannot start with
them. A line that starts with `/` runs a command:

| Command | Effect |
|---------|--------|
| `/voice [name]` | Lists the voices of the server, or switches to one. |
| `/speed <number>` | Sets the speed of speech. The server accepts 0.25 to 4. |
| `/mode wake\|vad\|ptt\|text` | Switches the listening mode. |
| `/wake [phrase]` | Shows or changes the wake phrase. |
| `/mute` | Mutes or unmutes the microphone. |
| `/system [text]` | Sets the system prompt, or clears it with no text. Either way, the conversation starts again. |
| `/reset` | Starts the conversation again. |
| `/memory` | Lists the newest 20 memories of the [assistant](assistant.md#memory). `/memory forget ID` removes one, and `/memory clear yes` removes all. |
| `/devices` | Lists the audio devices. |
| `/help` | Lists the commands. |
| `/quit`, `/exit`, `/q` | Quits. |

## Interrupting a reply

Space or Esc stops the speech within about 150 milliseconds and cancels
the turn. In `wake` mode, the wake phrase also interrupts, because the
client keeps listening for it while the model answers. The microphone
then opens for your next turn. If you say a stop phrase such as "stop",
"cancel" or "never mind" instead, the client goes back to waiting for the
wake phrase.

The `vad` and `ptt` modes do not listen while the model answers, because
the microphone would pick up the reply and transcribe it. In these modes,
interrupt from the keyboard.

Whisper sometimes produces words such as "thank you" from silence or
noise. The client drops utterances that are too short or too quiet, and it
discards a clip of 1.5 seconds or less when its text is one of these
known phrases. Noise therefore does not start a turn.

## The assistant by voice

With `talk.brain: assistant`, the [assistant](assistant.md) answers
instead of the plain model. It can call tools from MCP servers during a
turn, and it can remember facts across sessions. This example adds two
MCP servers that need no API keys, and turns on memory:

```yaml
server:
  stt: whisper-turbo
  tts: kokoro
  embeddings: qwen3-embed-0.6b     # Required for memory.
  rerank: qwen3-rerank-0.6b        # Optional. Reorders recalled memories.

talk:
  model: qwen3.8-27b-ud-q6@instruct
  brain: assistant

assistant:
  mcp:
    - name: files
      command: [npx, -y, "@modelcontextprotocol/server-filesystem", "/Users/me/notes"]
    - name: web
      command: [uvx, mcp-server-fetch]
  memory:
    enabled: true
```

The filesystem server needs Node, and the fetch server needs uv. The
status line names each tool as the assistant uses it, and only the answer
is spoken. After the turn, the assistant stores what it learned in the
background, so memory does not delay the spoken reply. A turn that needs
several tool calls takes longer than a plain reply. When you interrupt a
tool call, the conversation keeps what you heard and drops the unfinished
call. [Assistant](assistant.md) covers choosing a model, the tools and the
memory store.

## Settings

The [`talk`](config.md#voice) block of the configuration file sets the
model, the voice, the mode, the wake phrase and the speech detection
thresholds. Most of its keys have a flag under
[`gmlx talk`](cli.md#gmlx-talk), and a flag wins over the file. The
[menu bar app](menubar.md#voice-sessions) runs the same session without a
terminal and adds a tap-to-talk hotkey.

To shorten the wait for a reply, change these settings:

| Setting | Effect |
|---------|--------|
| A lower [`talk.vad.silence_ms`](config.md#talkvadsilence_ms), such as 400 | The turn ends sooner after you stop, and a pause in a sentence ends it more often. |
| `stt: whisper-turbo-q4` | Transcription is faster. |
| A lower [`talk.max_tokens`](config.md#talkmax_tokens), such as 512 | Replies are shorter. |

## A remote server and scripts

`--base-url http://host:8080/v1` points the client at another server, and
`--api-key` gives its key. Speech recognition and synthesis then run on
that machine. Without `--base-url`, the client uses the server of your
configuration file and starts it when it is down, unless you pass
`--no-start`.

`--once` answers one utterance and exits. It skips the wake phrase, which
makes it useful in scripts and for checking a setup.
