# The chat REPL

`gmlx chat` is the interactive terminal client. This page covers its slash
commands, sessions, rendering and themes; the flags are in
[cli.md](cli.md#gmlx-chat).

## Commands and keys

`/exit` (or Ctrl-D) quits, `/reset` restarts the conversation, `/help` lists
every command. The terminal is upgraded on top:

- Line editing and history: arrow keys, Ctrl-A/E, and up-arrow history
  persisted across sessions (`$XDG_CACHE_HOME/gmlx/chat_history[.ptk]`).
  `--no-history` keeps the session ephemeral; `/history [on|off|clear]`
  controls it at runtime.
- Upgraded editor: with `pip install 'gmlx[chat]'` (prompt_toolkit),
  completion menus pop as you type a `/command`, history offers fish-style
  ghost suggestions (accept with the right-arrow key), a bottom toolbar shows
  the live sampling settings, staged-block count, context fill, and the last
  reply's tok/s, multi-line paste is handled cleanly, and Alt-Enter inserts a
  newline without submitting (Shift-Enter too, in terminals that send
  `ESC CR` for it: iTerm2, VS Code, Windows Terminal). Without it, readline
  provides line editing and Tab completion.
- Runtime sampling: `/temp` `/top-p` `/top-k` `/min-p` `/max-tokens`
  `/xtc-probability` `/xtc-threshold` `/repetition-penalty`
  `/repetition-context-size` `/presence-penalty` `/frequency-penalty <value>`
  adjust the next reply; `/sampling` shows current values (`/max-tokens 0`
  removes the per-reply cap - replies then run until the model stops). All
  are also startup flags, so a model card's full sampling recommendation fits
  on the command line.
- `/retry` and `/undo` regenerate the last reply or remove the last exchange
  entirely. Both rewind the persistent KV cache to the turn's checkpoint (no
  re-prefill), restore the pre-turn state (system prompt, media markers), and
  work after an Esc-canceled reply. A rotating cache (`--max-kv-size`) that
  has wrapped its window can't rewind; `/reset` then.
- `/model` and `/stats` print the loaded model's card (arch, params, codecs,
  size, context, drafter, adapter) and the running session totals (turns,
  tokens, average tok/s, MTP acceptance). In server mode `/model` lists the
  served ids and `/model <id>` switches the id the next turn is sent to,
  keeping the transcript: the server re-reads the conversation under the
  new id, so a base and its adapters (which share one loaded model) can be
  compared mid-conversation. Tab completes the served ids.
- `/system [text|off]` shows or sets the system prompt at runtime (setting
  restarts the conversation).
- `/thinking [on|off|adaptive|default]` flips the model's own reasoning
  switch per turn, mapped onto its template spelling (`default` restores the
  template's default; same mapping as `--thinking`).
- `/thinking-budget [N|off]` caps a thinking model's reasoning tokens per
  reply, adjustable mid-session.
- `/copy` copies the last answer to the clipboard, thinking stripped
  (pbcopy / xclip / wl-copy, falling back to the OSC 52 terminal escape).
- `/load <file>` prefills the next prompt from a text file (edit it, then
  Enter sends it). Tab completes `/command` names, `/history` and
  `/reasoning` arguments, and file paths after `/load`, `/image`, `/audio`,
  and `/!`.
- `/! <command>` runs a shell command and stages its output (fenced block
  with a `$ command` header and an `[exit N in T]` footer) so it is attached
  to your next message; your question and the evidence land in one turn. The
  prompt shows `(+n) >> ` while blocks are staged, and several `/!` stack.
  Enter on an empty prompt sends them alone; `/drop` discards. Long output is
  middle-truncated (about 16 KB kept), stdin is `/dev/null` so interactive
  commands can't wedge the REPL, and Ctrl-C interrupts the command, not the
  session.
- Multimodal (`--mmproj <projector.gguf>`): `/image <file>` and
  `/audio <file>` stage media for the next message exactly like `/!` stages
  text, and dragging a file from Finder into the terminal also works; the
  pasted path is recognized and staged. Media markers stay pinned to the turn
  that sent them, so follow-ups reference earlier images correctly. VLM turns
  re-prefill the conversation each time; the KV-cached fast path is text-only
  for now.
- MTP speculative decoding (auto for native heads; `--no-mtp` to disable): a
  native-head model (qwen3.5/3.6/3.8 `nextn`) drafts and verifies multiple
  tokens per step for a decode speedup; gemma4 and muse-glimmer need a
  `--draft-gguf` assistant, and a DFlash 2 `--draft-gguf` drafter serves
  qwen3.8 and muse-glimmer (`--native-mtp` keeps the head). The reply streams the same way and ends with the same `tok/s`
  stat, and the persistent KV cache is reused across turns exactly like the
  text path. Not combinable with `--adapter` / `--stream-*`. Sampling is
  temperature/top-p/top-k/min-p only; the MTP verify walk has no penalty/bias
  hooks, so the other `/` sampling commands don't apply on this path.
  - VLM + MTP: a `--mmproj` VLM with a drafter (a `--draft-gguf` assistant
    for gemma4 or muse-glimmer, or a native `nextn` head for qwen3.5/3.6)
    keeps MTP on for text-only turns (the fast path above) while `/image` /
    `/audio` turns fall back to the plain VLM stream. The first media turn
    upgrades the session to the VLM path for the rest of the conversation,
    since the text tokenizer can't render a history that holds image markers.
    The prior text turns are carried into that re-prefill so nothing is lost.
- Reasoning display: for thinking models (Qwen3/DeepSeek-R1/GLM `<think>`,
  gpt-oss harmony channels, Gemma `<|channel>thought`, Muse Glimmer's ATEM
  `to=self` channel), the chain-of-thought is stripped of its control markers
  and streamed in the theme's thinking style (italic bright blue under the
  default `dark` theme) inside a gutter-framed block that closes with a payoff
  line showing how long the model thought and how many tokens it spent; the
  final answer follows in normal weight. `--reasoning hide` collapses the
  reasoning to a single live spinner that resolves to the same payoff, so you
  see it working without reading it. Ctrl-O toggles expand and collapse live during a reply (and
  persists as the default for the next). `--reasoning raw` / `/reasoning raw`
  passes everything through verbatim (the old behavior, for when a model's
  markers segment oddly). The stored conversation keeps the raw text in every
  mode, so display never changes what the model sees next turn.
- Markdown rendering: replies render as styled markdown while they stream.
  Completed blocks are printed permanently (native scrollback intact) and
  only the in-progress block repaints in place. `rich` mode (default when
  `rich` is installed on a color terminal) adds tables and syntax-highlighted
  code fences; `lite` is a zero-dependency ANSI fallback; `plain` is raw text
  (and the automatic non-TTY/`NO_COLOR` behavior). `--render` sets the mode,
  `/render` switches it live; `--reasoning raw` bypasses rendering entirely.
- Color themes: `--theme` / `/theme NAME [cb]` pick a palette: `dark`
  (default; follows the terminal's own colors), `light`, `dark-hc`
  (high-contrast), `nord`, `dracula`, `solarized-dark`, `gruvbox`. The `cb`
  modifier (or `--colorblind`) swaps every accent onto the colorblind-safe
  Okabe-Ito palette and works with all themes. A top-level `theme:` in
  `gmlx.yaml` sets the default, and a `themes:` section defines custom
  palettes (see [server-config.md](#chat-themes-theme--themes)).
- Session persistence: every chat autosaves after each turn (schema-v1 JSON
  under `$XDG_DATA_HOME/gmlx/chats`; `--no-autosave` opts out, and
  `/reset` rotates to a fresh file so old conversations survive). `/save
  [name]` saves explicitly, `/sessions` lists what's stored, `/load-session
  <name|N>` restores one (settings and transcript immediately; the KV replay
  is deferred, and the history prefills with your next message), and
  `--resume` picks up the model's latest session at startup. `/export
  [file.md]` writes a markdown transcript with thinking in collapsed
  `<details>` blocks.
- `/reset` / `/clear` restart the conversation; `/clear` also wipes the
  screen.
- Esc or Ctrl-C during a reply cancels it and returns to the prompt (the
  partial reply stays in the KV cache; `/retry` regenerates it, `/reset`
  clears it). Ctrl-C at an idle prompt, Ctrl-D, `/exit`, or `/quit` exits.

---

## Chat themes (`theme:` / `themes:`)

Like `talk:`, these configure a client - `gmlx chat` - not the server. A
top-level `theme:` names the theme every chat starts with (`--theme` and
`/theme` still override), and `themes:` defines custom themes usable anywhere
a built-in is:

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

Slots: `thinking`, `heading`, `bold`, `italic`, `inline_code`, `code_block`,
`code_border`, `bullet`, `blockquote`, `link`, `hr`, `stat`, `info`, `error`.
Style keys per slot: `bold`, `dim`, `italic`, `underline` (booleans), `fg16`
(an ANSI code 30-37/90-97, follows the terminal palette), `rgb` (truecolor
with automatic 256-color fallback; wins over `fg16` on capable terminals).
Meta keys: `extends`, `code_theme`, `code_theme_cb`, `ptk_toolbar`. The
colorblind modifier (`--colorblind` / `/theme NAME cb`) applies to user themes
the same way it does to built-ins. A malformed theme definition prints a
warning at chat startup and is skipped; the rest still register.

---
