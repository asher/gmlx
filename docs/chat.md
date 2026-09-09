# The chat client

`gmlx chat` is the interactive terminal client. This page covers its
commands, sessions, rendering and themes. The flags are under
[gmlx chat](cli.md#gmlx-chat).

The client runs in one of three modes. By default it loads the GGUF you
name in-process and generates locally. With `--server` it becomes a client
of the managed server instead, sending each turn to a served model id and
starting the server if it is down. That mode also engages on its own when
the config's server is already running and serves the model you asked for,
and `--local` forces an in-process load anyway. `--assistant` is the server
mode with the [tool loop and memory](assistant.md#text-chat) added.

The `chat` extra is optional and installs prompt_toolkit and rich:

```sh
uv tool install "gmlx[chat]"     # or: pip install "gmlx[chat]"
```

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
| `/temp`, `/top-p`, `/top-k`, `/min-p`, `/max-tokens` and the other sampling commands | adjust the next reply. `/sampling` shows the current values |
| `/thinking [on|off|adaptive|default]` | toggle the model's own reasoning per turn |
| `/thinking-budget [N|off]` | cap a thinking model's reasoning tokens per reply |
| `/reasoning show|hide|raw` | how thinking is displayed |
| `/render rich|lite|plain` | the markdown renderer |
| `/theme NAME [cb]` | the color theme, with an optional colorblind modifier |
| `/model`, `/stats` | the loaded model's card, and the session totals. In server mode `/model <id>` switches the served id |
| `/adapter [on|off|SCALE]` | switch the `--adapter` LoRA off and on, or scale it, for the next turns |
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
session ephemeral. The `chat` extra improves the prompt in three ways. A
completion menu opens as you type a command. A matching earlier line is
suggested in grey ahead of the cursor, and the right-arrow key accepts it.
A bottom toolbar shows the live sampling settings, staged blocks, context
fill and the last reply's speed. Multi-line paste is handled correctly, and
Alt-Enter inserts a newline without submitting. Without the extra, readline
provides line editing and Tab completion.

Tab completes command names, and after a command it completes the argument:

| After | Tab offers |
|-------|------------|
| `/history`, `/reasoning`, `/render`, `/thinking` | that command's values |
| `/thinking-budget` | `off` |
| `/load-session` | saved session names |
| `/theme` | theme names, then the `cb` modifier |
| `/load`, `/image`, `/audio`, `/export`, `/!` | file paths |
| `/model` | served ids, in server mode |

## Sampling at runtime

The sampling commands are `/temp`, `/top-p`, `/top-k`, `/min-p`,
`/max-tokens`, `/xtc-probability`, `/xtc-threshold`, `/repetition-penalty`,
`/repetition-context-size`, `/presence-penalty` and `/frequency-penalty`.
Each changes the next reply and each is also a startup flag, so a model
card's full sampling recommendation fits on the command line. `/max-tokens
0` removes the cap on reply length and lets replies run until the model
stops. A bare `chat` starts from the model family's card defaults, and an
`@profile` suffix on the model, such as `model.gguf@creative`, starts from
another [profile](server-config.md#profiles) instead.

[Speculative decoding](performance.md#mtp-speculative-decoding) is on
automatically for models with a native head, and `--draft-gguf` pairs a
companion drafter. The only visible difference is in sampling: while a
drafter is active only temperature, top-p, top-k and min-p apply, because
the verification step cannot apply penalties or biases.

## Undo, retry and sessions

`/retry` and `/undo` rewind the persistent KV cache to the turn's
checkpoint, so nothing re-prefills. They restore the pre-turn state,
including the system prompt and media markers, and work after a cancelled
reply. A rotating cache that has wrapped its window cannot rewind past the
evicted boundary. In that case the next message re-prefills, or `/reset`
starts over.

A chat autosaves after each turn as JSON under `$XDG_DATA_HOME/gmlx/chats`,
and `/reset` rotates to a fresh file so old conversations survive.
`--no-autosave` turns this off. A saved session restores its settings and
transcript together: `--resume` at startup brings back the model's latest
one, or a named one, and `/load-session` does the same from inside a chat.
Either way the KV cache is replayed on your next message rather than at
load time. `/export` writes a markdown transcript with thinking in
collapsed blocks.

In server mode, `/model` lists the served ids and `/model <id>` switches
the id the next turn is sent to. The transcript is kept, and the server
re-reads it under the new id. A base and its adapters share one loaded
model, so switching between them is instant, which is how
[lora.md](lora.md#serving-one-base-with-many-adapters) compares adapters
mid-conversation.

## Shell output and media

`/! <command>` runs a shell command and stages its output as a fenced
block, with the command as header and the exit status as footer. The block
is attached to your next message, so your question and the output arrive
in one turn. Several blocks can be staged at once, and the prompt shows
`(+n) >> ` while any are waiting. Enter on an empty prompt sends them alone,
and `/drop` discards them. Output longer than about 16 KB is truncated in the
middle. The command's stdin is closed so an interactive program cannot
block the client, and Ctrl-C interrupts the command rather than the
session.

With `--mmproj`, `/image` and `/audio` stage media the same way, and so
does dragging a file from Finder into the terminal. Each attachment stays on
the turn that sent it, so a follow-up can refer to an earlier image. What a
media turn costs, and how it interacts with speculative decoding, is
described in [vlm.md](vlm.md#combining-with-other-features).

## Reasoning and rendering

For thinking models, the chain of thought is stripped of its control
markers and streamed in the theme's thinking style inside a framed block,
which closes with a line showing how long the model thought and how many
tokens it used. Ctrl-O expands or collapses the block live during a reply,
and Ctrl-T ends the thinking early so the answer starts now. `--reasoning
hide` shows only the answer, and `--reasoning raw` passes the whole stream
through verbatim, which helps when a model's thinking markers are not
recognized and the block would otherwise cut in the wrong place. The stored
conversation keeps the raw text in every mode, so display never changes
what the model receives next turn.

Replies render as styled markdown while they stream. Completed blocks are
printed permanently with native scrollback intact, and only the in-progress
block repaints in place. The renderer is chosen automatically and `/render`
overrides it:

| Mode | When it is chosen | What it adds |
|------|-------------------|--------------|
| `rich` | the default on a color terminal with the `chat` extra installed | tables and syntax-highlighted code fences |
| `lite` | a color terminal without the extra | ANSI styling with no dependencies |
| `plain` | a non-TTY, or `NO_COLOR` set | raw text |

## Themes

`--theme` and `/theme NAME [cb]` pick a palette from `dark`, `light`,
`dark-hc`, `nord`, `dracula`, `solarized-dark` and `gruvbox`. `dark` is
the default and follows the terminal's own colors. The `cb` modifier, or
`--colorblind`, swaps the accents onto the colorblind-safe Okabe-Ito palette
and works with every theme.

Two top-level config keys configure the client rather than the server.
`theme:` names the theme each chat starts with, and `themes:` defines
custom themes, usable anywhere a built-in one is:

```yaml
theme: my-black

themes:
  my-black:                        # a custom theme with a built-in's name replaces it
    extends: dark                  # slots you leave out come from this theme
    thinking: {italic: true, fg16: 94}
    heading:  {bold: true, rgb: "#88c0d0"}   # rgb takes "#rrggbb" or [r, g, b]
    stat:     {fg16: 90}
    code_theme: nord               # pygments style for rich code fences
```

A theme is a set of slots, one per kind of text, each holding a style.

| Key | Values |
|-----|--------|
| slots | `thinking`, `heading`, `bold`, `italic`, `inline_code`, `code_block`, `code_border`, `bullet`, `blockquote`, `link`, `hr`, `stat`, `info`, `error` |
| style keys in a slot | the booleans `bold`, `dim`, `italic` and `underline`, plus `fg16` (an ANSI color code 30-37 or 90-97) and `rgb` (a truecolor value) |
| meta keys | `extends`, `code_theme`, `code_theme_cb`, `ptk_toolbar` |

`rgb` is used on a 256-color or better terminal, reduced to the nearest of
256 colors when truecolor is unavailable, and `fg16` is the fallback on a
16-color terminal. `code_theme_cb` is the pygments style used under the
colorblind modifier, and `ptk_toolbar` is the prompt_toolkit style string
for the bottom toolbar. A malformed theme definition prints a warning at
chat startup and is skipped, and the rest still register.
