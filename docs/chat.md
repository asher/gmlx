# The chat client

`gmlx chat` is the interactive terminal client for a local model or a
served one. This page covers its commands, sessions, rendering and themes.
The flags are under [gmlx chat](cli.md#gmlx-chat).

- [Commands](#commands)
- [Editing and history](#editing-and-history)
- [Sampling at runtime](#sampling-at-runtime)
- [Undo, retry and sessions](#undo-retry-and-sessions)
- [Shell output and media](#shell-output-and-media)
- [Reasoning and rendering](#reasoning-and-rendering)
- [Themes](#themes)

## Commands

| Command | Effect |
|---------|--------|
| `/help` | list every command |
| `/exit`, `/quit`, Ctrl-D | quit; Ctrl-C at an idle prompt does too |
| `/reset`, `/clear` | restart the conversation; `/clear` also wipes the screen |
| `/system [text|off]` | show or set the system prompt, which restarts the conversation |
| `/retry`, `/undo` | regenerate the last reply, or remove the last exchange |
| `/temp`, `/top-p`, `/top-k`, `/min-p`, `/max-tokens` and the other sampling commands | adjust the next reply; `/sampling` shows the current values |
| `/thinking [on|off|adaptive|default]` | toggle the model's own reasoning per turn |
| `/thinking-budget [N|off]` | cap a thinking model's reasoning tokens per reply |
| `/reasoning show|hide|raw` | how thinking is displayed |
| `/render rich|lite|plain` | the markdown renderer |
| `/theme NAME [cb]` | the color theme, with an optional colorblind modifier |
| `/model`, `/stats` | the loaded model's card, and the session totals; in server mode `/model <id>` switches the served id |
| `/history [on|off|clear]` | prompt history persistence |
| `/save [name]`, `/sessions`, `/load-session <name|N>`, `/export [file.md]` | session persistence |
| `/load <file>` | prefill the next prompt from a text file |
| `/! <command>` | run a shell command and stage its output for the next message; `/drop` discards staged blocks |
| `/image <file>`, `/audio <file>` | stage media for the next message on a multimodal model |
| `/copy` | copy the last answer to the clipboard, thinking stripped |
| `/memory` | inspect the assistant's memory, with `--assistant` ([assistant.md](assistant.md#memory)) |

Esc or Ctrl-C during a reply cancels it and returns to the prompt. The
partial reply stays in the KV cache, so `/retry` regenerates it and `/reset`
clears it.

## Editing and history

Arrow keys, Ctrl-A and Ctrl-E edit the line, and up-arrow history persists
across sessions under `$XDG_CACHE_HOME/gmlx/`. `--no-history` keeps a
session ephemeral. With the `chat` extra installed, completion menus appear as
you type a command, history offers ghost suggestions accepted with the
right-arrow key, a bottom toolbar shows the live sampling settings, staged
blocks, context fill and the last reply's speed, multi-line paste is handled
correctly, and Alt-Enter inserts a newline without submitting. Without the
extra, readline provides line editing and Tab completion.

Tab completes command names, the arguments of `/history` and `/reasoning`,
file paths after `/load`, `/image`, `/audio` and `/!`, and served ids after
`/model`.

## Sampling at runtime

Every sampling command adjusts the next reply, and each is also a startup
flag, so a model card's full sampling recommendation fits on the command
line. `/max-tokens 0` removes the per-reply cap so replies run until the
model stops. Bare `run` and `chat` already start from the model family's
card defaults, and an `@intent` suffix on the model switches to another
preset ([profiles](server-config.md#profiles)).

## Undo, retry and sessions

`/retry` and `/undo` rewind the persistent KV cache to the turn's
checkpoint, so nothing re-prefills, restore the pre-turn state including
the system prompt and media markers, and work after a cancelled reply. A
rotating cache that has wrapped its window cannot rewind past the evicted
boundary; the next message re-prefills, or use `/reset`.

Every chat autosaves after each turn as JSON under
`$XDG_DATA_HOME/gmlx/chats`, and `/reset` rotates to a fresh file so old
conversations are preserved. `--no-autosave` opts out. `/load-session` restores
settings and transcript at once, with the KV replay deferred to your next
message, and `--resume` resumes the model's latest session at startup.
`/export` writes a markdown transcript with thinking in collapsed blocks.

In server mode, `/model` lists the served ids and `/model <id>` switches the
id the next turn is sent to while keeping the transcript. The server re-reads
the conversation under the new id, so a base and its adapters, which share
one loaded model, can be compared mid-conversation
([lora.md](lora.md#serving-one-base-with-many-adapters)).

## Shell output and media

`/! <command>` runs a shell command and stages its output as a fenced block
with the command as header and the exit status as footer, attached to your
next message so your question and the output are sent in one turn. The prompt
shows `(+n) >> ` while blocks are staged, several can be staged at once, Enter on an empty
prompt sends them alone, and `/drop` discards them. Long output is
middle-truncated at about 16 KB, stdin is closed so interactive commands
cannot block the client, and Ctrl-C interrupts the command, not the session.

With `--mmproj`, `/image` and `/audio` stage media the same way, and
dragging a file from Finder into the terminal also works. Media markers stay
attached to the turn that sent them, so follow-ups reference earlier images
correctly. Media turns re-prefill the conversation each time; the KV-cached
fast path is text-only. On a model with a drafter, text-only turns keep
speculative decoding on and media turns fall back to the plain stream
([vlm.md](vlm.md)).

## Reasoning and rendering

For thinking models the chain of thought is stripped of its control markers
and streamed in the theme's thinking style inside a framed block that closes
with a line showing how long the model thought and how many tokens it used.
`--reasoning hide` drops the thinking and prints only the answer, and
`--reasoning raw` passes everything through verbatim, for when a model's
markers segment incorrectly. Ctrl-O toggles expand and collapse live during a reply.
The stored conversation keeps the raw text in every mode, so display never
changes what the model receives next turn.

Replies render as styled markdown while they stream. Completed blocks are
printed permanently with native scrollback intact, and only the in-progress
block repaints in place. `rich` mode, the default on a color terminal with
`rich` installed, adds tables and syntax-highlighted code fences; `lite` is a
zero-dependency ANSI fallback; `plain` is raw text and the automatic
behavior on a non-TTY or under `NO_COLOR`.

Speculative decoding is on automatically for models with a native head, and
`--draft-gguf` pairs a companion drafter; the reply streams the same way and
the KV cache is reused across turns exactly like the plain path
([performance.md](performance.md#mtp-speculative-decoding)). On that path
sampling is temperature, top-p, top-k and min-p only, since the verify step
has no penalty or bias hooks.

## Themes

`--theme` and `/theme NAME [cb]` pick a palette: `dark` (default, follows the
terminal's own colors), `light`, `dark-hc`, `nord`, `dracula`,
`solarized-dark` and `gruvbox`. The `cb` modifier, or `--colorblind`, swaps
every accent onto the colorblind-safe Okabe-Ito palette and works with all
themes.

Two top-level config keys configure the client, not the server. `theme:`
names the theme every chat starts with, and `themes:` defines custom themes
usable anywhere a built-in is:

```yaml
theme: my-black

themes:
  my-black:                        # a user name shadows a built-in
    extends: dark                  # unspecified slots inherit from here (default dark)
    thinking: {italic: true, fg16: 94}
    heading:  {bold: true, rgb: "#88c0d0"}   # rgb: "#rrggbb" or [r, g, b]
    stat:     {fg16: 90}
    code_theme: nord               # pygments style for rich code fences
```

| Key | Values |
|-----|--------|
| slots | `thinking`, `heading`, `bold`, `italic`, `inline_code`, `code_block`, `code_border`, `bullet`, `blockquote`, `link`, `hr`, `stat`, `info`, `error` |
| style keys per slot | `bold`, `dim`, `italic`, `underline` (booleans); `fg16` (ANSI code 30-37 or 90-97); `rgb` (truecolor, 256-color fallback, overrides `fg16`) |
| meta keys | `extends`, `code_theme`, `code_theme_cb`, `ptk_toolbar` |

A malformed theme definition prints a warning at chat startup and is
skipped; the rest still register.
