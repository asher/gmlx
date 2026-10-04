# Chat

`gmlx chat` is a terminal chat with a model. It saves each conversation, and
renders replies as markdown while they stream:

```sh
gmlx chat model.gguf                   # load a GGUF file in this process
gmlx chat --server qwen3.8-27b-ud-q6   # chat with a model the server serves
gmlx chat model.gguf --resume          # continue this model's last chat
```

Type `/help` in a chat for the commands. Esc or Ctrl-C stops a reply, and
Ctrl-D quits. The flags are in the [CLI reference](cli.md#gmlx-chat).

The optional `chat` extra adds a completion menu, a toolbar and styled
markdown. [Optional features](installation.md#optional-features) shows how
to install it.

- [Where the model runs](#where-the-model-runs)
- [Commands](#commands)
- [Sampling](#sampling)
- [Undo, retry and sessions](#undo-retry-and-sessions)
- [Shell output and media](#shell-output-and-media)
- [Reasoning and rendering](#reasoning-and-rendering)
- [Themes](#themes)

## Where the model runs

Chat loads the model in its own process, or sends each turn to the gmlx
server:

- A GGUF path always loads in the chat process.
- A configured model id goes to the server when the server is running and
  serves that id. Otherwise chat loads the model itself.
- With no model, chat uses the default model of a server that is already
  running.
- `--server` sends each turn to the server, and starts the server if it is
  down. `--host`, `--port` and `--api-key` do the same. `--base-url` and
  `--no-start` use only a server that is already running.
- `--assistant` works like `--server` and adds the assistant's
  [tools and memory](assistant.md#text-chat).
- `--local` always loads the model in the chat process.

With a server model, the server owns the model and its chat template, so
chat refuses `--adapter`, `--mmproj` and the chat template flags. The
model's served [profile](config.md#profiles) sets the sampling. `/image`,
`/audio`, `/thinking-budget` and Ctrl-T need a model loaded in the chat
process.

## Commands

| Command | Effect |
|---------|--------|
| `/help` | Lists the commands |
| `/exit`, `/quit` | Quits |
| `/reset`, `/clear` | Starts the conversation again. `/clear` also clears the screen. |
| `/system [text\|off]` | Shows or sets the system prompt, and starts again |
| `/retry`, `/undo` | Generates the last reply again, or removes the last exchange |
| `/temp`, `/top-p`, ... | Changes the sampling, as [Sampling](#sampling) lists. `/sampling` shows the values. |
| `/thinking [on\|off\|adaptive\|default]` | Turns a thinking model's reasoning on or off |
| `/thinking-budget [N\|off]` | Limits each reply's reasoning tokens |
| `/reasoning show\|hide\|raw` | How reasoning is shown |
| `/render rich\|lite\|plain` | The markdown renderer |
| `/theme NAME [cb]` | The color theme. `cb` adds colorblind-safe accents. |
| `/model`, `/stats` | The model card and the session totals. With a server model, `/model <id>` switches models. |
| `/adapter [on\|off\|SCALE]` | Turns the `--adapter` LoRA on or off, or scales it |
| `/history [on\|off\|clear]` | Prompt history saving |
| `/save [name]`, `/sessions`, `/load-session <name\|N>` | Saves, lists and restores sessions |
| `/export [file.md]` | Writes the conversation as markdown |
| `/load <file>` | Puts a file's text in the prompt to edit and send |
| `/! <command>` | Runs a shell command and attaches its output to the next message |
| `/drop` | Discards what is attached |
| `/image <file>`, `/audio <file>` | Attaches a file to the next message on a multimodal model |
| `/copy` | Copies the last answer, without its reasoning |
| `/memory` | Shows or edits the [assistant's memory](assistant.md#memory), with `--assistant` |

The up arrow recalls earlier prompts, saved across sessions in
`~/.cache/gmlx/chat_history` (`chat_history.ptk` with the `chat` extra).
`--no-history` keeps a session out of it. With the `chat` extra,
Alt-Enter inserts a newline, and a multi-line paste keeps its line breaks.

Tab completes command names. After a command, it completes the argument:

| After | Tab offers |
|-------|------------|
| `/history`, `/reasoning`, `/render`, `/thinking` | That command's values |
| `/thinking-budget` | `off` |
| `/load-session` | The names of saved sessions |
| `/theme` | Theme names, and then `cb` |
| `/load`, `/image`, `/audio`, `/export`, `/!` | File paths |
| `/model` | The served ids, with a server model |

## Sampling

The sampling commands are `/temp`, `/top-p`, `/top-k`, `/min-p`,
`/max-tokens`, `/xtc-probability`, `/xtc-threshold`, `/repetition-penalty`,
`/repetition-context-size`, `/presence-penalty` and `/frequency-penalty`.
Each has a startup flag with the same name, so the sampling a model card
recommends fits on the command line:

```sh
gmlx chat model.gguf --temp 0.7 --top-p 0.8 --top-k 20
```

Chat starts from the model's [family defaults](family-defaults.md). A suffix
picks an intent, as in `model.gguf@creative`, and a configured model id
takes a [profile](config.md#profiles) from your config file after the `@`.
`/max-tokens 0` removes the reply length limit.

Models with an MTP head use
[speculative decoding](speculative-decoding.md) on their own. While it is
on, only temperature, top-p, top-k and min-p apply, and chat warns at start
about the settings it drops.

## Undo, retry and sessions

`/retry` and `/undo` rewind the KV cache to before the last turn, so the
earlier conversation is not read again. A cancelled reply stays in the
conversation, so `/retry` generates it again and `/undo` removes it.
For some models, chat prints that it rebuilt the cache, and the next message
takes longer.

Chat saves each session in `~/.local/share/gmlx/chats` after every turn.
`--no-autosave` turns this off, and `/reset` starts a new file.

- `--resume` restores the model's latest session, and `--resume NAME` a
  named one.
- `/sessions` lists the saved sessions, and `/load-session` restores one
  from inside a chat.
- A session recorded with another model is refused.

The first message after a restore takes longer, because chat reads the
restored conversation into the model with it.

With a server model, `/model` lists the served ids, and `/model <id>` sends
the next turns to another one and keeps the conversation. A base model and
its adapters share one loaded model, so switching between them is fast.
[LoRA adapters](lora.md#serving-one-base-with-many-adapters) uses this to
compare adapters in one conversation.

## Shell output and media

`/! <command>` runs a shell command and attaches its output to your next
message, with the command and its exit status. For example, type
`/! git diff --stat`, then "Write a commit message for this change." Your
question and the output go in one turn.

While output waits, the prompt shows `(+1) >> `. You can attach several
outputs, and Enter on an empty prompt sends them alone. Very long output is
cut in the middle. The command gets no input, so an interactive program
cannot hang the chat.

With `--mmproj`, `/image` and `/audio` attach media the same way, and so
does dragging a file from Finder into the terminal. A later question can
refer to an earlier image. [Vision and audio](vlm.md) lists the models and
file types.

## Reasoning and rendering

For a thinking model, chat shows the reasoning in a frame, closed by a line
with how long the model thought. Ctrl-O folds or unfolds the frame during a
reply, and Ctrl-T ends the reasoning so the answer starts.

`--reasoning hide` shows a spinner instead. `--reasoning raw` prints the
output unchanged, which helps when chat does not recognize a model's
reasoning markers. The model always reads its full output in the next turn.

Chat picks a renderer, and `--render` or `/render` overrides it:

| Mode | Chosen when | Shows |
|------|-------------|-------|
| `rich` | Color terminal, `chat` extra installed | Tables, and code blocks with syntax colors |
| `lite` | Color terminal, no extra | Markdown styles |
| `plain` | Not a terminal, or `NO_COLOR` set | Raw text |

## Themes

The built-in themes are `dark` (the default, in your terminal's colors),
`light`, `dark-hc`, `nord`, `dracula`, `solarized-dark` and `gruvbox`.
`--colorblind`, or `cb` after `/theme`, switches the accents to the
colorblind-safe Okabe-Ito palette.

The [`theme`](config.md#theme) key of the config file sets the starting
theme, and [`themes`](config.md#themes) defines your own:

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

A theme has one slot for each kind of text: `thinking`, `heading`, `bold`,
`italic`, `inline_code`, `code_block`, `code_border`, `bullet`,
`blockquote`, `link`, `hr`, `stat`, `info` and `error`. It also takes
`code_theme_cb`, the pygments style with `cb`, and `ptk_toolbar`, the
toolbar's prompt_toolkit style. A theme with a built-in name replaces it.

Each slot takes the booleans `bold`, `dim`, `italic` and `underline`, and
two colors:

- `rgb`, as `"#rrggbb"` or `[r, g, b]`, for terminals with 256 colors or
  more
- `fg16`, an ANSI code from 30 to 37 or 90 to 97, for 16-color terminals

A theme that chat cannot read prints a warning at start and is skipped.
