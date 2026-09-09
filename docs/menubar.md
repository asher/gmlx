# The menu bar app

`gmlx launch menubar` runs a macOS status-bar item that monitors a server.
This page covers what it shows, how it starts and stops, its config editor,
its voice session with the tap-to-talk hotkey, and how macOS permissions
attach to it.

- [What it does](#what-it-does)
- [Editing the config](#editing-the-config)
- [Voice sessions](#voice-sessions)
- [Permissions](#permissions)

## What it does

The item shows the server's state, with the dot filled while requests are
generating or queued. Below it are the resident models with their size,
their default, pinned or kept markers and an eviction countdown on idle
ones, and clicking a model unloads it. The menu's server actions are Start
server, Stop server, Restart server, Reload config, Copy server URL, Open
logs and Quit, all through the server's endpoints. If a tracked server exits
unexpectedly the item posts a macOS notification, while an intentional stop
or restart does not.

A background `gmlx serve` starts the item on a macOS desktop, so you rarely
run it manually, and `--no-menubar` or `server.menubar: false` disables
that. To keep the item and the server across reboots, install them as a
login item with [gmlx service install](cli.md#gmlx-service).

Like `serve`, the item detaches by default. `--foreground` runs the event
loop in place and `--stop` quits a detached item. Only one item runs per
machine: a second `serve` finds the running one and leaves it alone, and a
manual `--foreground` launch while one is running exits with an error. With
no explicit target the item tracks the primary server, following it as
servers start and stop, while `--url`, `--host` or `--port` restricts it to
one server. It reads the API key from the managed server's config, or takes
`--api-key` for a server whose config it cannot see, and a key-protected
server it has no key for shows as up with a key-required note. The flags
are listed under [launch menubar](cli.md#launch-menubar).

## Editing the config

"Edit config" opens the server's YAML in a floating editor panel. Validate
runs the draft through the server's config parser, so the verdict is what
`gmlx serve` would say, before the server reads the file. Save writes
atomically and refuses once if the file changed on disk while you were
editing, and "Save & Reload" validates, saves and triggers the running
server's reload in a single step. Open in Editor opens the file in your
default text editor instead. The item appears whenever the menu bar has a
config.

## Voice sessions

When the tracked server advertises speech-to-text and text-to-speech, the
menu gains a "Talk to <model>" item, named after `talk.model` or the
server's default model. Clicking it starts a voice session inside the menu
bar app with no terminal window. The bar icon shows a microphone while
listening, a thought bubble while the model thinks and a speaker while it
talks. The session's menu offers Stop speaking, Mute mic, a Volume slider,
Show transcript, which opens a floating panel with the running
conversation, and End voice chat. With the assistant brain and memory on it
also offers Show memory and Clear memory. The volume setting persists
across sessions. Mic input has no gain control, so use the input level in
macOS Sound settings.

All settings come from the `talk:` block described in [talk.md](talk.md),
although push-to-talk and text modes fall back to wake mode since there is
no keyboard. A "Talk in a terminal" item opens `gmlx talk` in iTerm2 when it
is running and otherwise in the default terminal handler, with no
AppleScript involved.

A voice session started from either surface asks the server to
[keep](glossary.md) its model resident until the session
ends, and the model is loaded and warmed before the mic opens.

### Tap-to-talk hotkey

The menu bar can bind a global tap-to-talk combo that works from any app,
Space pressed while the Globe key is held. Pressing it always leads to an
open mic, whatever the session state: with no session running it starts
one, when the session is idle it opens the mic, tapping again while
listening dismisses, mid-capture it ends the utterance, and while the
assistant is transcribing, thinking or speaking it barges in and listens. A
tap while muted unmutes first.

On keyboards without a Globe key, choose a different modifier. `gmlx init`
asks for it in the voice-chat step, or set it in the config:

```yaml
talk:
  push_to_talk_modifier: globe   # or: right-command | right-option | control
```

The right-side variants are offered because left Cmd+Space is Spotlight
and left Option+Space is a common launcher binding. Holding Globe as a
modifier suppresses macOS's own Globe-key action only for the combo, so
your "Press Globe key to" setting keeps working for bare presses.

Suppressing the Space keystroke, which stops a space being typed into the
focused app, requires an active event tap and therefore Accessibility
permission, which is requested only when you first enable the hotkey. While
armed, all keystrokes in the login session pass through the tap, although
the overhead is small since the menu bar process does no inference.

The choice persists across launches, and on startup the app re-arms it only
after a silent permission check. If the grant is missing, a "needs
permission" note appears under the toggle and nothing prompts until you
toggle it again. Granting access in System Settings while the bar is running
is detected within a few seconds. If arming still fails right after a
grant, the app says to quit and reopen the menu bar, since some macOS
versions bind grants only to a freshly launched process.

## Permissions

macOS attaches a microphone or Accessibility grant to the app that launched
the process asking for it. A bar or a `gmlx talk` session started from a
terminal therefore prompts in the terminal's name, and the grant applies to
the terminal app. When the bar runs as the login item that
[gmlx service install](cli.md#gmlx-service) creates, prompts attribute to
gmlx and the grants follow it. If a grant was denied, re-enable it under
System Settings, Privacy and Security, Microphone or Accessibility.
