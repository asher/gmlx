# Menu bar app

The menu bar app shows your gmlx server in the macOS menu bar. From it you
can see the loaded models, start and stop the server, edit the configuration
file and hold voice sessions.

A background `gmlx serve`, or a `gmlx launch` that starts the server, opens
the app for you. To open it yourself:

```sh
gmlx launch menubar
```

The app appears as `gmlx` and a dot in the menu bar. When the server exits
unexpectedly or stops responding, the app posts a macOS notification.

## What the menu shows

The dot is filled while the server is up, ringed while it generates, empty
while it is down, and half filled when the server needs an API key that the
app does not have.

Under the title, rows show the server's process id and port, and how many
requests are generating and queued. The Loaded models submenu lists each
model with its size, a `[default]`, `[pinned]` or `[kept]` marker, and the
time until an idle model unloads. Select a model to unload it.

| Item | Action |
|------|--------|
| Start server | Starts the server again. Shown only while it is down. |
| Stop server | Stops a background server |
| Restart server | Restarts the server |
| Reload config | Makes the server read its configuration file again |
| Copy server URL | Copies the server's address |
| Open logs | Shows recent log lines from the server and the app, also while the server is down |
| Quit | Quits the app. The server keeps running. |

## Starting and stopping the app

| Command | Result |
|---------|--------|
| `gmlx launch menubar` | Starts the app in the background |
| `gmlx launch menubar --foreground` | Runs the app in this terminal |
| `gmlx launch menubar --stop` | Quits a background app |
| `gmlx serve --no-menubar` | Starts the server without the app |
| `gmlx service install` | Starts the app and the server at login |

Set [`server.menubar: false`](config.md#servermenubar) to never open the app
with the server. An app that a server opened quits when the last server it
manages stops. Only one app runs at a time, and it needs a desktop session,
so it does not start over SSH.

The app follows the main server. `--url`, `--host` or `--port` makes it
track one server, and `--api-key` supplies the key when the app cannot read
the server's configuration file. The flags are under
[`gmlx launch menubar`](cli.md#launch-menubar).

## Editing the configuration

Edit config opens the server's configuration file in a panel:

- Validate checks the draft the way `gmlx serve` would, including model
  paths.
- Save writes the file, as [Changing the file](config.md#changing-the-file)
  describes.
- Save & Reload saves the draft and makes the server read it.
- Revert discards the draft.
- Open in Editor opens the file in your default text editor.

## Voice sessions

When the server runs both speech services from
[Speech, embeddings and rerank](services.md), the menu gains a Talk to
item, named after [`talk.model`](config.md#talkmodel) or the default model.
It starts a voice session inside the app. Talk in a terminal opens
[`gmlx talk`](talk.md) in iTerm2 when it runs, otherwise in your default
terminal app.

During a session the menu bar shows a microphone while it listens, a thought
bubble while the model thinks, and a speaker while it talks. The menu offers
Stop speaking, Mute mic, a volume slider, Show transcript and End voice
chat. With the [assistant](assistant.md) memory on, it also offers Show
memory and Clear memory.

Settings come from the [`talk`](config.md#voice) block of the configuration
file. The `ptt` and `text` modes need a keyboard, so the app runs them as
`wake`. To change the microphone level, use the macOS Sound settings.

An app started from a terminal asks for the microphone in the terminal's
name, and the login item from [`gmlx service install`](cli.md#gmlx-service)
asks in the name of gmlx. To change it later, open System Settings, Privacy
and Security, Microphone.

### Tap-to-talk hotkey

The menu item `Tap-to-talk with <key> + Space` turns on a shortcut that
works in any app: hold the Globe key and press Space. The item shows the key
as a symbol. It needs the `talk` extra.

| Session state | A tap |
|---------------|-------|
| No session | Starts one |
| Idle | Opens the microphone |
| Listening | Closes the microphone. In `vad` mode it does nothing. |
| Capturing an utterance | Ends the utterance |
| Transcribing, thinking or speaking | Interrupts the reply and opens the microphone |

A tap also unmutes a muted microphone. On a keyboard without a Globe key,
choose another key in `gmlx init`, or set it in the configuration file:

```yaml
# doctest: build
talk:
  push_to_talk_modifier: right-command   # or: globe, right-option, control
```

The hotkey needs the Accessibility permission, which the app asks for when
you turn the hotkey on. It goes to the same app as the microphone, under
Privacy and Security, Accessibility. If the item then says to quit and
reopen the menu bar, do that.
