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
| `/help` | list the commands |
| `/exit`, `/quit`, Ctrl-D | quit, as does Ctrl-C at an idle prompt |
| `/reset`, `/clear` | restart the conversation. `/clear` also wipes the screen |
| `/system [text|off]` | show or set the system prompt, which restarts the conversation |
| `/retry`, `/undo` | regenerate the last reply, or remove the last exchange |
| `/temp`, `/top-p`, `/top-k`, `/min-p`, `/max-tokens` and the other sampling commands | adjust the next reply, and `/sampling` shows the current values |
| `/thinking [on|off|adaptive|default]` | toggle the model's own reasoning per turn |
| `/thinking-budget [N|off]` | cap a thinking model's reasoning tokens per reply |
| `/reasoning show|hide|raw` | how thinking is displayed |
| `/render rich|lite|plain` | the markdown renderer |
| `/theme NAME [cb]` | the color theme, with an optional colorblind modifier |
| `/model`, `/stats` | the loaded model's card, and the session totals. In server mode `/model <id>` switches the served id |
| `/history [on|off|clear]` | prompt history persistence |
| `/save [name]`, `/sessions`, `/load-session <name|N>`, `/export [file.md]` | session persistence |
| `/load <file>` | prefill the next prompt from a text file |
| `/! <command>` | run a shell command and stage its output for the next message. `/drop` discards staged blocks |
| `/image <file>`, `/audio <file>` | stage media for the next message on a multimodal model |
| `/copy` | copy the last answer to the clipboard, thinking stripped |
| `/memory` | inspect the [assistant's memory](assistant.md#memory) when running with `--assistant` |

Esc or Ctrl-C during a reply cancels it and returns to the prompt. The
partial reply stays in the KV cache, where `/retry` regenerates it and
`/reset` clears it.

## Editing and history

Arrow keys, Ctrl-A and Ctrl-E edit the line, and up-arrow history persists
across sessions under `$XDG_CACHE_HOME/gmlx/` unless `--no-history` keeps a
session ephemeral. The `chat` extra adds more: completion menus appear as
you type a command, history offers ghost suggestions accepted with the
right-arrow key, and a bottom toolbar shows the live sampling settings,
staged blocks, context fill and the last reply's speed. Multi-line paste is
handled correctly, and Alt-Enter inserts a newline without submitting.
Without the extra, readline provides line editing and Tab completion.

Tab completes command names, the arguments of `/history` and `/reasoning`,
file paths after `/load`, `/image`, `/audio` and `/!`, plus served ids after
`/model`.

## Sampling at runtime

Sampling commands adjust the next reply. Each one is also a startup flag,
which means a model card's full sampling recommendation fits on the command
line. `/max-tokens 0` removes the cap on reply length, so replies run until
the model stops. Bare `run` and `chat` already start from the model family's
card defaults, and an `@intent` suffix on the model switches to another
[preset](server-config.md#profiles).

## Undo, retry and sessions

`/retry` and `/undo` rewind the persistent KV cache to the turn's
checkpoint, so nothing re-prefills. They restore the pre-turn state,
including the system prompt and media markers, and work after a cancelled
reply. A rotating cache that has wrapped its window cannot rewind past the
evicted boundary, in which case the next message re-prefills, or you can use
`/reset`.

A chat autosaves after each turn as JSON under `$XDG_DATA_HOME/gmlx/chats`.
`/reset` rotates to a fresh file, which preserves old conversations.
`--no-autosave` opts out. To restore settings and transcript at once, use
`/load-session`, which defers the KV replay to your next message. At startup,
`--resume` resumes the model's latest session. A markdown transcript with
thinking in collapsed blocks comes from `/export`.

In server mode, `/model` lists the served ids and `/model <id>` switches
the id the next turn is sent to while keeping the transcript, which the
server re-reads under the new id. That lets you compare a base and its
adapters mid-conversation, since they share a loaded model, as
[lora.md](lora.md#serving-one-base-with-many-adapters) describes.

## Shell output and media

`/! <command>` runs a shell command and stages its output as a fenced
block, with the command as header and the exit status as footer. The block
is attached to your next message, so your question and the output are sent
in a single turn. While blocks are staged the prompt shows `(+n) >> `, and
several can be staged at once. Enter on an empty prompt sends them alone and
`/drop` discards them. Long output is middle-truncated at about 16 KB. stdin
is closed so that interactive commands cannot block the client, and Ctrl-C
interrupts the command rather than the session.

With `--mmproj`, `/image` and `/audio` stage media in the same way, and
dragging a file from Finder into the terminal also works. Media markers stay
attached to the turn that sent them, which lets follow-ups reference earlier
images correctly. Because the KV-cached fast path is text-only, media turns
re-prefill the conversation each time, and on a model with a drafter,
text-only turns keep speculative decoding on while media turns fall back to
the plain stream. [vlm.md](vlm.md) covers multimodal models.

## Reasoning and rendering

For thinking models, the chain of thought is stripped of its control
markers and streamed in the theme's thinking style inside a framed block,
which closes with a line showing how long the model thought and how many
tokens it used. `--reasoning hide` drops the thinking and prints only the
answer, and `--reasoning raw` passes everything through verbatim, for a
model whose markers segment incorrectly. Ctrl-O toggles expand and collapse
live during a reply. The stored conversation keeps the raw text in all
modes, so display never changes what the model receives next turn.

Replies render as styled markdown while they stream. Completed blocks are
printed permanently with native scrollback intact, and only the in-progress
block repaints in place. `rich` mode is the default on a color terminal with
`rich` installed and adds tables and syntax-highlighted code fences, `lite`
is a zero-dependency ANSI fallback, and `plain` is raw text, the automatic
choice on a non-TTY or under `NO_COLOR`.

[Speculative decoding](performance.md#mtp-speculative-decoding) is on
automatically for models with a native head, and `--draft-gguf` pairs a
companion drafter. The reply streams as before, with the KV cache reused
across turns exactly as on the plain path, although sampling is limited to
temperature, top-p, top-k and min-p because the verify step has no penalty
or bias hooks.

## Themes

`--theme` and `/theme NAME [cb]` pick a palette from `dark`, `light`,
`dark-hc`, `nord`, `dracula`, `solarized-dark` and `gruvbox`, where `dark`
is the default and follows the terminal's colors. The `cb` modifier, or
`--colorblind`, swaps the accents onto the colorblind-safe Okabe-Ito palette
and works with all themes.

Two top-level config keys configure the client, not the server. `theme:` names
the theme each chat starts with. `themes:` defines custom themes, usable
anywhere a built-in one is:

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
| style keys in a slot | booleans `bold`, `dim`, `italic` and `underline`, `fg16` as an ANSI code 30-37 or 90-97, `rgb` as truecolor with a 256-color fallback that overrides `fg16` |
| meta keys | `extends`, `code_theme`, `code_theme_cb`, `ptk_toolbar` |

A malformed theme definition prints a warning at chat startup and is
skipped, while the rest still register.
