# Chat

`gmlx chat` is the interactive terminal client for talking to a model. It
loads the model itself or sends turns to a running server, saves sessions,
and renders replies as markdown in a color theme. The flags are listed
under [gmlx chat](cli.md#gmlx-chat).

- [Where the model runs](#where-the-model-runs)
- [Commands](#commands)
- [Editing and history](#editing-and-history)
- [Sampling at runtime](#sampling-at-runtime)
- [Undo, retry and sessions](#undo-retry-and-sessions)
- [Shell output and media](#shell-output-and-media)
- [Reasoning and rendering](#reasoning-and-rendering)
- [Themes](#themes)

## Where the model runs

By default, chat loads the GGUF that you name and generates in its own
process. With `--server`, it sends each turn to a model that the gmlx
server serves, and it starts the server if the server is down. With
`--assistant`, it works like `--server` and adds the assistant's
[tool loop and long-term memory](assistant.md#text-chat).

Chat also uses the server without `--server` when your configured
server is running and serves the id that you name, or when you
name no model. A GGUF path always loads in the chat process, because the
file on disk can be newer than what the server holds, and chat prints the
served id when the server already serves that file.

Any of `--base-url`, `--host`, `--port`, `--api-key` and `--no-start`
selects `--server` on its own. A flag that changes how the model loads keeps
the model in the chat process, and `--local` forces an in-process load in
every case.

In server mode, the server owns the model and its chat template. Chat
refuses `--adapter`, `--mmproj` and the chat template flags. It ignores
the other load flags and prints a note that names them. The
model's served [profile](config.md#profiles) sets the sampling, and
chat sends only the settings that you change. `/image`, `/audio`,
`/thinking-budget` and Ctrl-T work only on a model loaded in the chat
process.

The optional `chat` extra adds a completion menu, a toolbar and styled
markdown. To install it, follow
[Optional features](installation.md#optional-features).

## Commands

A line that starts with `/` runs one of these commands:

| Command | Effect |
|---------|--------|
| `/help` | It lists the commands. An unknown command also lists them. |
| `/exit`, `/quit`, Ctrl-D | Each one quits chat. Ctrl-C at an empty prompt also quits. |
| `/reset`, `/clear` | Both start the conversation again. `/clear` also clears the screen. |
| `/system [text\|off]` | It shows or sets the system prompt. Setting it starts the conversation again. |
| `/retry`, `/undo` | `/retry` generates the last reply again, and `/undo` removes the last exchange. |
| `/temp`, `/top-p` and the other sampling commands | Each one changes the sampling of the next replies. `/sampling` shows the current values. |
| `/thinking [on\|off\|adaptive\|default]` | It turns a thinking model's reasoning on or off for the next turns. |
| `/thinking-budget [N\|off]` | It limits each reply's reasoning tokens. |
| `/reasoning show\|hide\|raw` | It sets how reasoning is shown. |
| `/render rich\|lite\|plain` | It sets the markdown renderer. |
| `/theme NAME [cb]` | It sets the color theme. `cb` adds colorblind-safe accents. |
| `/model`, `/stats` | `/model` shows the model card, and `/stats` shows the session totals. In server mode, `/model <id>` switches the served id. |
| `/adapter [on\|off\|SCALE]` | It turns the `--adapter` LoRA off or on, or scales it, for the next turns. With no argument, it shows the scale. |
| `/history [on\|off\|clear]` | It turns saving of prompt history on or off, or clears the history. |
| `/save [name]`, `/sessions`, `/load-session <name\|N>` | These commands save, list and restore sessions. |
| `/export [file.md]` | It writes the conversation as markdown, by default to the session name in the current folder. |
| `/load <file>` | It puts a file's text in the prompt for you to edit and send. |
| `/! <command>` | It runs a shell command and attaches its output to the next message. `/drop` discards what is attached. |
| `/image <file>`, `/audio <file>` | Each one attaches a local file to the next message on a multimodal model. |
| `/copy` | It copies the last answer, without its reasoning, to the clipboard. |
| `/memory` | It shows or edits the [assistant's memory](assistant.md#memory) in `--assistant` mode. |

Esc or Ctrl-C during a reply cancels it and returns to the prompt. The
partial reply stays in the conversation, so `/retry` generates it again and
`/undo` removes it.

## Editing and history

The arrow keys, Ctrl-A and Ctrl-E edit the line. The up arrow recalls
earlier prompts, which chat saves across sessions in
`$XDG_CACHE_HOME/gmlx/chat_history`, or in `chat_history.ptk` beside it
with the `chat` extra. `--no-history` keeps a session out of the file, and
`/history off` stops saving mid-session.

With the `chat` extra, a completion menu opens as you type a command, and
an earlier prompt that matches your text appears in grey ahead of the
cursor, where the right arrow accepts it. A toolbar at the bottom shows the
sampling settings, attached items, how full the context is and the last
reply's speed.

The extra also keeps the line breaks of pasted text, and Alt-Enter
inserts a newline. Shift-Enter does the same when your terminal sends
ESC CR for it. Without the extra, readline edits the line and completes
with Tab.

Tab completes command names. After a command, it completes the argument:

| After | Tab offers |
|-------|------------|
| `/history`, `/reasoning`, `/render`, `/thinking` | Tab offers that command's values. |
| `/thinking-budget` | Tab offers `off`. |
| `/load-session` | Tab offers the names of saved sessions. |
| `/theme` | Tab offers theme names, and then `cb`. |
| `/load`, `/image`, `/audio`, `/export`, `/!` | Tab offers file paths. |
| `/model` | In server mode, Tab offers the served ids. |

## Sampling at runtime

The sampling commands are `/temp`, `/top-p`, `/top-k`, `/min-p`,
`/max-tokens`, `/xtc-probability`, `/xtc-threshold`, `/repetition-penalty`,
`/repetition-context-size`, `/presence-penalty` and `/frequency-penalty`.
Each one changes the next replies and has a startup flag with the same
name, so the sampling that a model card recommends fits on the command
line. `/max-tokens 0` removes the limit on the length of a reply.

Chat starts from the model's [family defaults](family-defaults.md).
An intent suffix on a GGUF path, such as `model.gguf@creative`, starts
from that intent of the family defaults instead. A configured id also
takes a [profile](config.md#profiles) of the configuration file after the
`@`.

Models with an MTP head use
[speculative decoding](speculative-decoding.md) automatically, and
`--draft-gguf` pairs a separate drafter. While a drafter is active, only
temperature, top-p, top-k and min-p apply. Chat warns at startup and drops
the penalties, logit bias and XTC settings, as well as `--max-kv-size` and
`--quantized-kv-start`.

## Undo, retry and sessions

`/retry` and `/undo` return the KV cache to where it was before the last
turn, so the earlier conversation is not read again. They restore the
system prompt and attached media too, and they work after a cancelled
reply. `/retry` refuses while items are attached for the next message, so
send them or `/drop` them first.

The cache cannot go back in place for a recurrent hybrid model, for a
sliding window that has moved past the turn, or for turns restored from a
saved session. In these cases, chat prints that it rebuilt the cache, and
the earlier conversation is read again with your next message.

Chat saves the session as JSON in `$XDG_DATA_HOME/gmlx/chats` after each
turn, and `--no-autosave` turns that off. `/reset` starts a new file, so
the old conversation stays saved.

A saved session restores its settings and its conversation together.
`--resume` at startup restores the model's latest session, or a named
one, and `/load-session` does the same from inside a chat. Both refuse a
session that was recorded with another model. The restored conversation
is read into the KV cache with your next message, not at load time.

In server mode, `/model` lists the served ids, and `/model <id>` sends the
next turns to another id. The conversation is kept, and the server reads it
again under the new id. A base model and its adapters share one loaded
model, so switching between them is fast.
[LoRA adapters](lora.md#serving-one-base-with-many-adapters) uses this to
compare adapters in one conversation.

## Shell output and media

`/! <command>` runs a shell command and attaches its output to your next
message as a fenced block, with the command above it and the exit status
below it. Your question and the output then arrive in one turn. You can
attach several blocks, and the prompt shows `(+n) >> ` while any are
waiting. Enter on an empty prompt sends the blocks alone.

Output longer than about 16,000 characters is cut in the middle. The
command gets no input, so an interactive program cannot leave the chat
waiting, and Ctrl-C stops the command instead of the chat.

With `--mmproj`, `/image` and `/audio` attach media the same way, and so
does dragging a file from Finder into the terminal. Each attachment stays
with the turn that sent it, so a later question can refer to an earlier
image. The supported models and file types are listed in
[Vision and audio](vlm.md).

## Reasoning and rendering

For a thinking model, chat shows the reasoning in a frame in the theme's
thinking style. The frame closes with a line that says how long the
model thought and how many tokens it used. Ctrl-O folds or unfolds the
frame during a reply, and Ctrl-T ends the reasoning so that the answer
starts.

`--reasoning hide` shows a spinner while the model thinks, and then the
same closing line. `--reasoning raw` prints the output unchanged, which
helps when chat does not recognize a model's reasoning markers. The
conversation keeps the full output in every mode, so the display never
changes what the model reads in the next turn.

Replies render as markdown while they stream. Finished blocks stay in the
terminal's scrollback, and only the block in progress is drawn again.
Chat chooses the renderer, and `/render` or `--render` overrides it:

| Mode | Chosen when | What it shows |
|------|-------------|---------------|
| `rich` | The terminal has color and the `chat` extra is installed. | It shows tables and code blocks with syntax colors. |
| `lite` | The terminal has color and the extra is missing. | It shows markdown styles and needs no extra packages. |
| `plain` | Output is not a terminal, or `NO_COLOR` is set. | It shows the raw text. |

## Themes

`--theme` and `/theme` choose from the built-in themes `dark`, `light`,
`dark-hc`, `nord`, `dracula`, `solarized-dark` and `gruvbox`. The default
is `dark`, which uses your terminal's colors. The `cb` modifier, or
`--colorblind`, changes the accent colors to the colorblind-safe Okabe-Ito
palette and works with every theme.

The [`theme`](config.md#theme) key of the configuration file sets the
theme that each chat starts with, and [`themes`](config.md#themes) defines
your own:

```yaml
theme: my-black

themes:
  my-black:
    extends: dark                  # Slots that you leave out come from this theme.
    thinking: {italic: true, fg16: 94}
    heading:  {bold: true, rgb: "#88c0d0"}
    stat:     {fg16: 90}
    code_theme: nord               # The pygments style of rich code blocks.
```

A theme with the name of a built-in theme replaces it. A theme is a set of
slots, one for each kind of text, named `thinking`, `heading`, `bold`,
`italic`, `inline_code`, `code_block`, `code_border`, `bullet`,
`blockquote`, `link`, `hr`, `stat`, `info` and `error`. Beside its slots, a
theme takes `extends`, `code_theme`, `code_theme_cb` for the pygments style
under the colorblind modifier, and `ptk_toolbar` for the toolbar's
prompt_toolkit style.

Each slot holds a style, which takes the booleans `bold`, `dim`, `italic`
and `underline`, and the colors `fg16` and `rgb`. `rgb` takes `"#rrggbb"`
or `[r, g, b]`. A terminal with 256 colors or more uses it, reduced to the
nearest of 256 colors when the terminal has no true color. `fg16` is an
ANSI color code from 30 to 37 or 90 to 97, which a 16-color terminal uses
instead.

Chat prints a warning at startup for a theme that it cannot read, skips
that theme and loads the others.
