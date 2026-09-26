# Menu bar app

The menu bar app shows the state of a gmlx server in the macOS menu bar. It
lists the loaded models, starts and stops the server, edits the
configuration file, and runs voice sessions. `gmlx serve` opens it when the
server starts in the background on a Mac desktop, so you rarely start it
yourself.

## What the menu shows

The title is `gmlx` followed by a dot that shows the state of the server.
The dot is filled while the server is up, ringed while it is generating,
empty while it is down, and half filled when the server needs an API key
that the app does not have.

Under the title, rows show the process id and port of the server, and how
many requests are generating and queued. The Loaded models submenu lists each
loaded model with its size, a marker when it is the default, pinned or
kept, and the time until an idle model unloads. Selecting a model unloads
it.

The server items act on the server that the app tracks:

| Item | Action |
|------|--------|
| Start server | Starts the server again. It appears only while the server is down. |
| Stop server | Stops a server that runs in the background. A login-item server shows how to remove it instead. |
| Restart server | Restarts the server. |
| Reload config | Tells the server to read its configuration file again. |
| Copy server URL | Copies the address of the server. |
| Open logs | Opens a panel with the recent log lines of the server and the app. |
| Quit | Quits the app. The server keeps running. |

When the server exits unexpectedly or stops responding, the app posts a
macOS notification. A stop or restart from the menu posts none.

## Starting and stopping

A background `gmlx serve`, and a `gmlx launch` that starts the server,
open the app. `gmlx serve --no-menubar` or
[`server.menubar: false`](config.md#servermenubar) prevents that. An app
opened this way quits when the last server it manages stops. To keep the
app and the server running from login, install them as a login item with
[`gmlx service install`](cli.md#gmlx-service).

To start the app yourself, run `gmlx launch menubar`. It detaches like
`gmlx serve`, and `--foreground` keeps it in the terminal instead. Only one
app runs at a time. A second `gmlx serve` leaves the running app alone, and
a `--foreground` start while one runs exits with an error.
`gmlx launch menubar --stop` quits a detached app. The app needs a desktop
session, so it does not start over SSH.

With no target, the app tracks the main server and follows it as servers
start and stop. `--url`, `--host` or `--port` makes it track one server.
The app reads the API key from the configuration file of the server, and
`--api-key` supplies it for a server whose file the app cannot read. The
flags are listed under [`gmlx launch menubar`](cli.md#launch-menubar).

## Editing the configuration

Edit config opens the configuration file of the server in a panel:

- Validate checks the draft the way `gmlx serve` would, and checks that
  each model path exists.
- Save writes the file. If the file changed on disk while you edited it,
  the first Save refuses. Save writes an invalid draft too, and reports
  what is wrong with it.
- Save & Reload validates the draft, saves it and tells the server to read
  it again.
- Revert discards the draft.
- Open in Editor opens the file in your default text editor.

## Voice sessions

When the server runs both speech services of
[Speech, embeddings and rerank](services.md), the menu gains a
Talk to item, named after [`talk.model`](config.md#talkmodel) or the
default model of the server. It starts a voice session inside the app,
with no terminal window. The menu bar icon shows a microphone while the
session listens, a thought bubble while the model thinks, and a speaker
while it talks.

During a session, the menu offers Stop speaking, Mute mic, a volume
slider, Show transcript, which opens the conversation in a panel, and End
voice chat. With the [assistant](assistant.md) brain and memory turned on,
it also offers Show memory and Clear memory. The volume persists between
sessions. The app has no microphone gain control, because a software gain
would change when speech counts as speech, so set the input level in the
Sound settings of macOS instead.

The session reads its settings from the [`talk`](config.md#voice) block of
the configuration file. The `ptt` and `text` modes need a keyboard, so the
app uses `wake` mode for them. Talk in a terminal opens
[`gmlx talk`](talk.md) in iTerm2 when it is running, and otherwise in the
default terminal app.

A session asks the server to keep its model loaded. The load runs in the
background, so the microphone opens without waiting for it.

### Tap-to-talk hotkey

The app can turn on a hotkey that works in any app. By default the hotkey
is Space pressed while the Globe key is held. A tap unmutes the microphone
if it was muted, and then acts on the session:

| Session state | A tap |
|---------------|-------|
| No session | Starts one. |
| Idle | Opens the microphone. |
| Listening | Closes the microphone. In `vad` mode, listening is the idle state, and a tap does nothing. |
| Capturing an utterance | Ends the utterance. |
| Transcribing, thinking or speaking | Interrupts, and opens the microphone. |

On a keyboard without a Globe key, choose another key in `gmlx init`, or
set it in the configuration file:

```yaml
talk:
  push_to_talk_modifier: right-command   # or: globe, right-option, control
```

The right-hand keys are offered because Command and Space opens Spotlight,
and Option and Space is a common launcher shortcut. Your Globe key setting
in macOS still works for a plain press of Globe.

With the `talk` extra installed, the menu shows the hotkey item. To keep
Space from reaching the app in front, the hotkey needs Accessibility
permission. The app asks for the permission each time you turn on the
hotkey without it, and never at startup. The hotkey stays on across
restarts, and the app turns it back on at startup only when the permission
is present. Until then, the item says that it needs permission. After you
grant the permission in System Settings, the app notices within a few
seconds. If the item then says to
quit and reopen the menu bar, do that, because some macOS versions apply a
permission only to a newly started app.

## Permissions

macOS gives a microphone or Accessibility permission to the app that
started the process that asks for it. An app or a `gmlx talk` session
started from a terminal therefore asks in the terminal's name, and the
permission applies to the terminal. When the app runs as the login item
that [`gmlx service install`](cli.md#gmlx-service) creates, the request
names gmlx. To change a permission later, use Privacy and Security in
System Settings, under Microphone or Accessibility.
