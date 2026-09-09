# The menu bar app

`gmlx launch menubar` runs a macOS status-bar item that monitors a server.
This page covers what it shows, how it starts and stops, its config editor,
and its voice session with the tap-to-talk hotkey.

## What it does

The item shows the server's state. The dot is filled while requests are
generating or queued. Below it are the resident models with their size,
their default, pinned or kept markers, and an eviction countdown on idle
ones. Clicking a model unloads it. The menu offers reload config, restart,
stop, copy URL and open logs, all through the server's endpoints. If a
tracked server exits unexpectedly the item posts a macOS notification. An
intentional stop or restart does not.

A background `gmlx serve` starts the item on a macOS desktop, and you
rarely run it manually. `--no-menubar` or `server.menubar: false` disables that.
To keep the item and the server across reboots, install them as a login
item with [gmlx service install](cli.md#gmlx-service). Running as a login
item also makes macOS permission prompts attribute to gmlx instead of your
terminal.

Like `serve`, the item detaches by default. `--foreground` runs the event
loop in place, and `--stop` quits a detached item. Only one item runs on a
machine. A second `serve` or a manual launch is a no-op. With no explicit
target it tracks the primary server, following it as servers start and
stop. `--url`, `--host` or `--port` restricts it to one server. It reads
the API key from the managed server's config, or takes `--api-key` for a
server whose config it cannot see. A key-protected server it has no key for
shows as up with a key-required note. The flags are listed under
[launch menubar](cli.md#launch-menubar).

## Editing the config

"Edit config" opens the server's YAML in a floating editor panel. Validate
runs the draft through the server's config parser, and the verdict is what
`gmlx serve` would say, before the server reads the file. Save writes
atomically. It refuses once if the file changed on disk while you were
editing. Save and Reload validates, saves and triggers the running server's
reload in a single step. Open in Editor opens the file in your default text
editor instead. The item appears whenever the menu bar has a config.

## Voice sessions

When the tracked server advertises speech-to-text and text-to-speech, the
menu gains a "Talk to <model>" item, named after `talk.model` or the
server's default model. Clicking it starts a voice session inside the menu
bar app with no terminal window. The bar icon shows a microphone while
listening, a thought bubble while the model thinks, and a speaker while it
talks. The menu offers Stop speaking, Mute mic, End voice chat and Show
transcript. Show transcript opens a floating panel with the running
conversation. A volume slider scales the voice and the chimes relative to
the system output and persists across sessions. Mic input has no gain
control, because software gain would shift the endpointing and wake-word
thresholds. Use the input level in macOS Sound settings instead.

All settings come from the `talk:` block described in [talk.md](talk.md).
Push-to-talk and text modes fall back to wake mode, since there is no
keyboard. A "Talk in a terminal" item opens `gmlx talk` in iTerm2 when it
is running, and otherwise in the default terminal handler. No AppleScript
is involved.

Starting a voice session from either surface holds its model resident on
the server for the session's lifetime. The model is loaded and warmed up in
advance and exempt from the idle timeout, and the mic is never open while
the model is unloaded. When the session ends the hold is released and the
model is not evicted.

### Tap-to-talk hotkey

The menu bar can bind a global tap-to-talk combo that works from any app.
The combo is Space pressed while the Globe key is held. Pressing it always
leads to an open mic, whatever the session state. With no session running
it starts one. When the session is idle it opens the mic. Tapping again
while listening dismisses. Mid-capture it ends the utterance. While the
assistant is transcribing, thinking or speaking it barges in and listens. A
tap while muted unmutes first.

On keyboards without a Globe key, choose a different modifier. `gmlx init` asks
for it in the voice-chat step, or set it in the config:

```yaml
talk:
  push_to_talk_modifier: globe   # or: right-command | right-option | control
```

The right-side variants are offered because left Cmd+Space is Spotlight and
left Option+Space is a common launcher binding. Holding Globe as a modifier
suppresses macOS's own Globe-key action only for the combo. Your "Press
Globe key to" setting keeps working for bare presses.

Suppressing the Space keystroke, which stops a space being typed into the
focused app, requires an active event tap and therefore Accessibility
permission. The permission is requested only when you first enable the
hotkey. While armed, all keystrokes in the login session pass through the
tap. The overhead is small, and the menu bar process does no inference.

The choice persists across launches. On startup the app re-arms it only after
a silent permission check. If the grant is missing, a "needs permission" note
appears under the toggle and nothing prompts until you toggle it again.
Granting access in System Settings while the bar is running is detected
within a few seconds. If arming still fails right after a grant, the app says
to quit and reopen the menu bar, since some macOS versions bind grants only
to a freshly launched process. Permission prompts attribute to gmlx when
the bar runs as the login item. A bar launched from a terminal runs under
the terminal's identity, and grants then apply to the terminal app.
