# The menu bar app

`gmlx launch menubar` runs a macOS status-bar item that watches a server.
This page covers what it shows, how it starts and stops, its config editor,
and its voice session with the tap-to-talk hotkey.

## What it does

The item shows the server's state, with the dot filled while requests are
generating or queued, and the resident models with their size, their
default, pinned or kept markers and an eviction countdown on idle ones.
Clicking a model unloads it. The menu offers reload config, restart, stop,
copy URL and open logs, all over the server's own endpoints. If a tracked
server dies it posts a macOS notification; an intentional stop or restart
does not.

A background `gmlx serve` raises the item for you on a macOS desktop, so you
rarely run it by hand. `--no-menubar` or `server.menubar: false` disables
that. To keep it and the server across reboots, install them as a login item
with `gmlx service install` ([gmlx service](cli.md#gmlx-service)), which
also makes macOS permission prompts attribute to gmlx instead of your
terminal.

Like `serve`, it detaches by default; `--foreground` runs the event loop in
place and `--stop` quits a detached item. One item runs per machine, so a
second `serve` or a manual launch is a no-op. With no explicit target it
tracks the primary server, following it as servers come and go, and `--url`,
`--host` or `--port` pins it to one. It reads the API key from the managed
server's config, or takes `--api-key` for a server whose config it cannot
see; a key-protected server it has no key for shows as up with a
key-required note. The flags are under
[launch menubar](cli.md#launch-menubar).

## Editing the config

"Edit config" opens the server's YAML in a floating editor panel. Validate
runs the draft through the server's own config parser, so the verdict is
exactly what `gmlx serve` would say, caught before the server sees the file.
Save writes atomically and refuses once if the file changed on disk while
you were editing. Save and Reload validates, saves and triggers the running
server's reload in one step, and Open in Editor hands the file to your
default text editor instead. The item is there whenever the bar knows a
config.

## Voice sessions

When the tracked server advertises speech-to-text and text-to-speech, the
menu gains a "Talk to <model>" item, named after `talk.model` or the
server's default model. Clicking it starts a voice session inside the menu
bar app with no terminal window. The bar icon shows the state, a microphone
while listening, a thought bubble while the model thinks, a speaker while it
talks, and the menu offers Stop speaking, Mute mic, End voice chat and Show
transcript, which opens a floating panel with the running conversation. A
volume slider scales the voice and the chimes relative to the system output
and persists across sessions. Mic input has no gain control, since software
gain would shift the endpointing and wake-word thresholds; use the macOS
Sound settings input level instead.

All settings come from the `talk:` block ([talk.md](talk.md)). Push-to-talk
and text modes fall back to wake mode, since there is no keyboard. A "Talk in
a terminal" item opens `gmlx talk` in iTerm2 when it is running, otherwise
the default terminal handler, with no AppleScript involved.

Starting a voice session from either surface holds its model resident on the
server for the session's lifetime, loaded and warmed up front and exempt from
the idle timeout, so an open mic never sits in front of an unloaded model.
The hold is released, not evicted, when the session ends.

### Tap-to-talk hotkey

The menu bar can bind a global tap-to-talk combo that works from any app,
Space pressed while the Globe key is held. Firing it always drives toward an
open mic, whatever the session is doing: with no session running it starts
one, idle opens the mic, tapping again while listening dismisses, mid-capture
it ends the utterance, and while the assistant is transcribing, thinking or
speaking it barges in and listens. A tap while muted unmutes first.

Keyboards without a Globe key pick a different modifier. `gmlx init` asks
for it in the voice-chat step, or set it in the config:

```yaml
talk:
  push_to_talk_modifier: globe   # or: right-command | right-option | control
```

The right-side variants are offered because left Cmd+Space is Spotlight and
left Option+Space is a common launcher bind. Holding Globe as a modifier
suppresses macOS's own Globe-key action only for the combo, so your "Press
Globe key to" setting keeps working for bare presses.

Swallowing the Space keystroke, so that no space lands in the
focused app, requires an active event tap and therefore Accessibility
permission, requested only when you first enable the hotkey. While armed,
every keystroke in the login session passes through the tap; the overhead is
small and the menu bar process does no inference.

The choice persists across launches. On startup the app re-arms it only after
a silent permission check. If the grant is missing, a "needs permission" note
appears under the toggle and nothing prompts until you flip it again.
Granting access in System Settings while the bar is running is picked up
within a few seconds. If arming still fails right after a grant, the app says
to quit and reopen the menu bar, since some macOS versions bind grants only
to a freshly launched process. Permission prompts attribute to gmlx when the
bar runs as the login item; a bar launched from a terminal runs under the
terminal's identity, so grants then attach to the terminal app.
