# The menu bar app

`gmlx launch menubar` runs a macOS status-bar item that watches a server. This
page covers what it shows, how it is started and stopped, and its voice
session and tap-to-talk hotkey.

## What it does

`gmlx launch menubar` puts a small macOS status-bar item up for a backgrounded
server: up/down state (the dot fills in while requests are generating or queued),
the resident models (size, default/pinned/kept markers, an eviction countdown on
idle ones; click one to unload it), reload-config, restart, stop, copy-URL, and
open-logs, all over the existing HTTP endpoints. If a tracked server dies, it
posts a macOS notification (an intentional stop or restart does not). A background
`gmlx serve` raises it for you on a macOS GUI session, so you rarely run it by hand.
Disable that with `--no-menubar` or `server.menubar: false`. To keep it (and the
server) across reboots, install it as a login item with `gmlx service install`
(see [gmlx service](cli.md#gmlx-service)). That also makes macOS permission
prompts attribute to gmlx instead of your terminal.

"Edit config" opens the server's YAML in a floating editor panel. Validate runs
the draft through the server's own config parser, so the verdict - a typo'd key,
a bad value, a model path that doesn't resolve - is exactly what `gmlx serve`
would say, caught before the server ever sees the file. Save writes atomically
and refuses (once) if the file changed on disk while you were editing; Save &
Reload validates first, then saves and triggers the running server's config
reload in one step. Open in Editor hands the file to your default text editor
instead. The item is there whenever the bar knows a config: the tracked
server's own file or, with everything stopped, the default config location.
Fixing the config is usually why you are there.

Like `serve`, it detaches by default. Pass `-f` / `--foreground` to run the event
loop in place, and `--stop` to quit a detached monitor from the CLI. One menu bar
per machine, deduplicated via a pidfile: a second `serve` on any port, or a manual
`launch menubar`, is a no-op. With no explicit target it tracks the primary server
(the single managed one, else `127.0.0.1:8080`), following it as servers come and
go. Pass `--url`, `--host`, or `--port` to pin it to one.

It reads the API key from the managed server's own `server.api_key`, or takes
`--api-key` for a server whose config it cannot see. A key-protected server it has no
key for shows as up with a key-required note, never as down. macOS only (it needs
`rumps`, a default dependency there). On Linux or over SSH it prints a one-line
notice. `--interval S` sets the poll interval (default 4 seconds).

When the tracked server also advertises STT and TTS, the menu gains a voice-chat
item that runs a full talk session inside the menu bar app, no terminal needed. See
[talk.md](#voice-sessions).

## Voice sessions

When the tracked server advertises STT and TTS, the macOS menu bar app
(`gmlx launch menubar`, raised automatically by a background `serve`) shows a
"Talk to <model>" item, named after `talk.model` or the server's default model.
Clicking it starts a voice session inside the menu bar app, no terminal window.
The bar icon changes to show the state (a microphone while listening, a thought
bubble while the model thinks, a speaker while it talks, a muted-speaker while
the mic is off), and the menu offers Stop speaking, Mute mic, and End voice
chat. Show transcript opens a floating panel with the running conversation
text. A Volume slider under the session controls scales the voice (and the
chimes) relative to the system output volume. It applies mid-sentence while
dragging, and the setting persists across sessions. Mic input has no gain
control on purpose: software input gain would shift the endpointing and
wake-word thresholds and clip loud speech. Use the macOS Sound settings input
level instead.

All settings come from the YAML `talk:` block. Push-to-talk and text modes fall
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

(A bare double-press of Globe was considered and dropped. Current macOS
routes the solo Globe press to the system shortcut handler without posting
an event that session event taps can see - only raw HID sees it - so
double-press stays available for the system's own dictation shortcut.)

The choice persists across launches. On startup the app re-arms it only
after a silent permission check. If the grant is missing (denied, or
silently dropped by an app-stub re-sign after an interpreter upgrade), a
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
