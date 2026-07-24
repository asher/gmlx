# gmlx CLI reference

`gmlx` is an umbrella command with one verb per task:

| Command | Purpose |
|---------|---------|
| [`gmlx run`](#gmlx-run) | Generate text from, benchmark, or inspect a single GGUF. |
| [`gmlx chat`](#gmlx-chat) | Interactive chat REPL on a GGUF: multi-turn, KV-cached. |
| [`gmlx talk`](#gmlx-talk) | Voice chat with a served model: wake word, whisper STT, spoken streamed replies. |
| [`gmlx serve`](#gmlx-serve) | Run the batched, multi-model OpenAI/Anthropic server. |
| [`gmlx train`](#gmlx-train) | LoRA-finetune a GGUF base; emit a GGUF adapter. |
| [`gmlx init`](#init-scaffold-a-config) | Scaffold a starter server config. |
| [`gmlx sync-models`](#sync-models-reconcile-a-config-with-whats-on-disk) | Reconcile a config's models with disk or the hf cache. |
| [`gmlx launch`](#launch-connect-a-coding-agent-or-chat-app) | Point a coding harness or agent (hermes / goose), chat TUI (aichat / elia), or web app (open-webui) at a server, starting one if it is down. `launch menubar` raises the macOS status-bar monitor. |
| [`gmlx stop` / `restart` / `status` / `logs`](#background-mode--server-lifecycle) | Manage a backgrounded server (`serve` detaches by default). |
| [`gmlx service`](#service-run-at-login-launchd-macos) | Install/uninstall a launchd LaunchAgent (start at login, macOS). |
| [`gmlx doctor`](#gmlx-doctor) | Check the runtime, config, model paths, and services in one pass. |
| [`gmlx validate`](#gmlx-validate) | Check a local or remote GGUF will load, without a full download. |
| [`gmlx pull`](#gmlx-pull) | Validate a remote GGUF, then download it into your model dir. |
| [`gmlx rm`](#gmlx-rm) | Delete a model's GGUF files and its config entry. |
| [`gmlx list`](#gmlx-list) (`ls`) | List the models your server config defines (ids, paths, aliases, default). |
| [`gmlx ps`](#gmlx-ps) | Show the models resident in a running server. |
| [`gmlx profiles`](#gmlx-profiles) | Show per-family sampling defaults and `@intents`; resolve one model's sampling. |
| [`gmlx completion`](#gmlx-completion) | Print a shell completion script (zsh, bash, fish). |

`gmlx` is the installed command; every action is a subcommand (`gmlx run`,
`gmlx serve`, `gmlx init`, and so on). The project, package, and command are
all named `gmlx`.

Every verb operates on the GGUF file itself: the file on disk is the model.
Every flag is also visible via `--help`.

```sh
gmlx --help              # list the verbs
gmlx <verb> --help       # a verb's options
```

---

## `gmlx run`

Load a GGUF and either generate a completion, run a benchmark, or print a
load inventory.

```sh
gmlx run GGUF [options]
```

```sh
# generate
gmlx run model.gguf --prompt "Explain entropy." --max-tokens 128

# a bare path seeds the family's model-card sampling; explicit flags always win
gmlx run model.gguf --prompt "Write a haiku." --temp 0.8 --top-p 0.95

# prefill/decode throughput sweep at several prompt lengths
gmlx run model.gguf --bench 512,4096,16384 --bench-runs 3

# decode tok/s at increasing context depths
gmlx run model.gguf --bench-depths 0,4096,16384,32768

# inspect the load plan (arch, codecs, remap, rendered prompt) without running
gmlx run model.gguf --report-only
```

### Generation

Sampling flags you don't pass are seeded from the model's family model-card
defaults (see [Family defaults](#family-defaults-intent-and---profile)); the
table's defaults apply as-is only with `--no-family-defaults`, or where the
family sets no value.

| Flag | Default | Meaning |
|------|---------|---------|
| `gguf` (positional) | - | Path to the GGUF (sharded ok). |
| `--prompt STR` | `Hello, world!` | Generation prompt. |
| `--prompt-file PATH` | - | Read the prompt from a file (mutually exclusive with `--prompt`). Applies in every mode: generate, VLM, bench, `--report-only`. |
| `--max-tokens N` | until EOS | Decode-token cap; unset, generation runs until the model stops (diffusion models fall back to a bounded 2048-token canvas). Pass N to cap the reply; the MTP and VLM paths note on stderr when the cap - not the model - ended it (the plain text path cannot tell the two apart). |
| `--temp F` | `0.0` | Sampling temperature; `0.0` = greedy. Unset, the family default applies instead. |
| `--top-p F` | `0.95` | Nucleus sampling. |
| `--top-k N` | `0` | Top-k (`0` = off). |
| `--min-p F` | `0.05` | Min-p sampling. |
| `--repetition-penalty F` | `0.0` | Repetition penalty (`0.0` = off). |
| `--presence-penalty F` | `0.0` | Presence penalty (`0.0` = off). |
| `--frequency-penalty F` | `0.0` | Frequency penalty (`0.0` = off). |
| `--seed N` | - | PRNG seed for reproducible sampling. |
| `-v` / `--verbose` | off | Full load diagnostics instead of the progress spinner. |
| `--system-prompt STR` | - | System message for the chat template. |
| `--chat-template-config JSON` | - | Extra chat-template kwargs, e.g. `'{"enable_thinking": false}'`. |
| `--xtc-probability F` / `--xtc-threshold F` | `0.0` | XTC sampling (text path). |
| `--repetition-context-size N` | `20` | How many recent tokens the repetition penalty considers. |
| `--logit-bias JSON` | - | Token-id to bias map, e.g. `'{"128001": -100}'`. |
| `--stop STR` | - | Stop sequence; generation ends (trimmed) when it appears. Repeatable. |
| `--max-kv-size N` | - | Cap the KV cache (rotating cache above it). |
| `--kv-bits N` / `--kv-group-size N` / `--quantized-kv-start N` | off / `64` / `0` | Quantize the KV cache (mlx-lm `QuantizedKVCache`): cuts cache memory 2-4x for long contexts. |
| `--prefill-step-size N` | `2048` / `8192` streaming | Prefill chunk size; lower it to cap peak memory on long prompts. Over-RAM streaming `--stream-cpu` / `--stream-experts` models default to `8192`: each chunk re-streams the expert lane from disk, so fewer, bigger chunks prefill faster. Applies to `run`, `--bench`, and `--bench-depths`. |

### Multimodal (VLM)

Supported families, usage, and caveats: [docs/vlm.md](vlm.md).

| Flag | Meaning |
|------|---------|
| `--mmproj PATH` | Float vision/audio projector GGUF (`general.architecture=clip`); pairs with the K-quant LLM GGUF to load a VLM. The image processor and chat template are synthesized from the GGUFs. |
| `--image PATH\|URL` | Image(s) to prepend to the prompt (comma-separated for several). |
| `--audio PATH\|URL` | Audio to prepend (comma-separated for several). Requires an omni mmproj with an audio tower. |
| `--resize-shape N\|WxH` | Resize images before encoding (e.g. `448` or `672x448`). Controls the soft-token count, a big prefill-cost lever. |
| `--thinking-budget N` | Cap thinking tokens (VLM generate path). |

`run --mmproj` honours the full sampling surface from the table above
(`--top-p`/`--top-k`/`--min-p`, the repetition/presence/frequency penalties,
`--logit-bias`, `--seed`, `--system-prompt`, `--chat-template-config`) plus the
KV-cache and prefill flags. `--stop` and `--xtc-*` remain text-path-only, since
mlx-vlm's generate path has no seam for them; passing them with `--mmproj`
prints a one-line `ignored in VLM mode (text-only)` warning. Flags with no VLM
plumbing at all (`--bench`, `--bench-depths`, `--report-only`, `--stream-cpu`,
`--stream-experts`) error with `--mmproj` (exit `2`) instead of silently
falling back to plain text generation.

A text-only request under `--mmproj` runs through the MTP speculative path
whenever a drafter is available: a `--draft-gguf` assistant (gemma4) or a
native `nextn` head in the LLM GGUF (qwen3.5/3.6, no companion needed). The
verify walk only touches the language model, so a resident VLM gets the decode
speedup on text turns, token-identical to the same model's text-only MTP. An
image or audio request uses the plain VLM path; the drafter is idle that turn.

### Speculative / MTP

Native-head GGUFs (qwen3.5/3.6 `nextn`) auto-enable MTP speculative decoding,
no flag needed, and you'll see `[mtp] native MTP head detected -> speculative
decoding on`. Auto only engages when it changes nothing you asked for: if you
set a flag the verify walk can't honour (below), it defers to plain decoding
and says so. `--speculative`/`--mtp` forces the path on (dropping those flags
with a warning); `--no-speculative`/`--no-mtp` forces it off.

| Flag | Meaning |
|------|---------|
| `--speculative` / `--mtp` | Force MTP speculative decoding on. Native-head models (qwen3.5/3.6 `nextn`) need no companion; gemma4 needs `--draft-gguf`. Native heads are auto-enabled without this. Use it to force the path when a sampler flag would otherwise defer. |
| `--no-speculative` / `--no-mtp` | Disable MTP. Overrides the native-head auto-enable and config `speculative: true`. |
| `--draft-gguf PATH` | Separate assistant-drafter GGUF (gemma4 two-GGUF MTP shape); implies `--speculative` (same as `serve`). |
| `--draft-block-size N` | Override the MTP draft block size. |

Speculative generation takes only `--temp`/`--top-p`/`--top-k`/`--min-p` plus a
baked `--system-prompt`; mlx-vlm's verify walk has no stop/bias/penalty/KV
hooks. A native head is sticky: MTP auto-enables and stays on. Flags the verify
walk can't honour (`--stop`, `--logit-bias`, the
repetition/presence/frequency penalties, `--xtc-probability`, `--max-kv-size`,
the `--kv-*` flags) are dropped, each named in a warning, never silently. To
apply one of those flags, pass `--no-mtp` to decode on the plain path, which
honours it exactly. Only hard-incompatible flags (`--mmproj`, `--adapter`,
`--stream-cpu`/`--stream-experts`, `--moe-experts`, `--moe-expert-mass`,
`--moe-expert-probe`) make auto-enable step aside to plain decoding. `chat` also keeps `--stop` via a post-hoc stream filter.

### Adapter (LoRA)

| Flag | Meaning |
|------|---------|
| `--adapter PATH` | Apply a GGUF LoRA adapter live over the base at load; the base stays K-quant, no merge, no requant. Text path only (not `--mmproj` / `--speculative`). Train one with [`gmlx train`](#gmlx-train). |

```sh
gmlx run model.gguf --adapter pirate-lora.gguf --prompt "What's the weather like today?"
```

### Loading

| Flag | Meaning |
|------|---------|
| `--arch NAME` | Override architecture detection. |
| `--config FILE` | Server config to resolve the positional against when it isn't a file on disk (default: the first existing default config). See [Resolving a model from a config](#resolving-a-model-from-a-config). |
| `--profile NAME` | Apply a built-in intent (`coding`, `instruct`, `creative`, `reasoning-low\|-medium\|-high`) or, on the config path, any user profile. Same as a `@NAME` suffix on the positional. See [Family defaults](#family-defaults-intent-and---profile). |
| `--no-family-defaults` | Don't seed the family's model-card sampling on a bare-path run (env: `GMLX_NO_FAMILY_DEFAULTS=1`). |
| `--hf-source ID\|DIR` | Load config (and, with `--mmproj`, the image processor and chat template) from this HF id / local dir instead of synthesizing from GGUF metadata. |
| `--chat-template STR\|PATH` | Inline Jinja template, or a path to a `.jinja`/`.txt` file, replacing the GGUF's. |
| `--no-chat-template` | Pass the prompt verbatim (base / non-instruct models). |
| `--no-remap` | Skip the GGUF-to-HF name remap (raw GGUF names). |
| `--no-zero-copy` | `memcpy` tensors out of the mmap instead of viewing them. |
| `--stream-cpu` | Run the whole model on the CPU device: the over-RAM MoE path kept entirely on one device. Weights stay mmap-backed so the page cache streams them from disk; past the GPU wired budget the runtime adds sequential expert prefetch (`GMLX_STREAM_PREFETCH=0` disables) and a wider default prefill chunk (8192). When and why: [streaming.md](streaming.md). |
| `--stream-experts` | Routed-expert stacks stream from disk while the every-token layers (attention, norms, routers, shared experts) and the KV cache stay on GPU. With the decode feeder (default) it matches `--stream-cpu` on short generations and pulls ahead as the arena warms; a quantized KV cache (`--kv-bits`) extends the win to long context. Mutually exclusive with `--stream-cpu`. Details: [streaming.md](streaming.md). |
| `--moe-experts K` | Lossy: cap the router at K experts per token on the streamed MoE layers (`--stream-experts` / `--stream-cpu`). Scoring and weight renormalization run unchanged at the new k, but outputs differ from the trained model by design. Composes with `--moe-expert-mass`. |
| `--moe-expert-mass P` | Lossy, adaptive: per token, keep only the smallest set of routed experts covering share P (0 < P <= 1) of the router's gate mass - confident tokens read fewer expert bytes, uncertain tokens keep the full fan-out. Choosing P: [streaming.md](streaming.md). |
| `--moe-expert-probe` | Lossless companion to `--moe-expert-mass`: run at the trained fan-out while recording how many experts each token needed at candidate P values; prints decode and prefill tables (experts/token, expert-read fraction, dropped mass per P) at exit - size P from the decode table. |
| `--[no-]prefill-feeder` | Streaming models (`--stream-cpu` / `--stream-experts` past the wired budget): stage each prefill layer's expert stacks straight from the GGUF into GPU-visible ring slots instead of through the page cache - one trip per byte, and short prompts stage only the experts the router chose. Default on; `--no-prefill-feeder` falls back to page-cache prefetch. Details: [streaming.md](streaming.md). |
| `--[no-]decode-feeder` | Streaming `--stream-experts` models: serve decode from a wired, popularity-managed GPU expert arena; misses read from the GGUF at SSD queue depth. Default on for `--stream-experts` (needs the every-token layers on GPU, so never under `--stream-cpu`); `GMLX_DECODE_ARENA_GB` caps the arena size. Details: [streaming.md](streaming.md). |

### Resolving a model from a config

The `gguf` positional is normally a path. When it isn't a file on disk (a bare
name, no `/`, no `.gguf`, not an `hf:`/`http(s):` ref), it's looked up as a
model id or alias in your server config, the same `models:`/`aliases:` blocks
[`serve`](#gmlx-serve) uses (see [server-config.md](server-config.md)). On a
match, the model's resolved path and its merged profile/override settings are
overlaid onto the run: sampling (`temp`, `top-p`, `top-k`, `min-p`, penalties,
`max-tokens`, `stop`, `seed`, XTC), KV-cache
(`kv-bits`/`kv-group-size`/`max-kv-size`/`quantized-kv-start`), `system`,
`chat_template`, `enable_thinking`, `mmproj`, `adapter`, speculative/draft, and
`stream` placement. Flags you pass explicitly always win; the config only
fills what you left at its default. Models defined as `hf:` paths resolve to
their local Hugging Face cache file (offline; `gmlx pull` them first). The
config is the first existing default location unless `--config FILE` points
elsewhere.

```sh
# 'coder' is a model id (or alias) in your config; its path + sampling/template apply
gmlx run coder --prompt "Refactor this loop."

# explicit --temp overrides the profile's; everything else still comes from the config
gmlx run coder --prompt "Be creative." --temp 1.0
```

A name that matches nothing falls through to the normal file-miss error, so a
typo'd path never silently reads a config. A config `name@profile` with an
unknown profile fails listing the available ones. On a bare path only the
built-in intents are recognized after `@`; anything else is treated as part of
the filename.

#### Family defaults, `@intent`, and `--profile`

Sampling is model-aware even with no config at all: a bare-path run reads the
GGUF header, detects the model's family, and seeds its model-card recommended
defaults. Qwen3.6, Gemma, and gpt-oss each publish different numbers. The
applied values print on one `[family]` line, and explicit flags always win.
The built-in intents (`coding`, `instruct`, `creative`,
`reasoning-low|-medium|-high`) are addressable on paths and config ids alike,
via an `@intent` suffix or `--profile NAME`, which also takes any user profile
on the config path. `gmlx profiles` prints the full table; see
[server-config.md](server-config.md#profiles-sampling-profiles-and-built-in-intents).

```sh
gmlx run model.gguf -p "..."                    # family base defaults applied
gmlx run model.gguf@creative -p "..."           # the family's creative point
gmlx chat model.gguf --profile reasoning-high   # gpt-oss reasoning effort
gmlx launch pi --model qwen3.6-27b@coding       # a coding harness on the coding point
gmlx run model.gguf --temp 0 -p "..."           # explicit flag wins (greedy)
gmlx run model.gguf --no-family-defaults -p "..."   # opt out (or GMLX_NO_FAMILY_DEFAULTS=1)
```

### Inspect & benchmark

| Flag | Default | Meaning |
|------|---------|---------|
| `--report-only` | - | Load and print the inventory (and rendered prompt); skip the model build. |
| `--bench LIST` | - | Comma-separated prompt lengths (e.g. `512,4096,16384`); prints a prefill/decode tok/s table. |
| `--bench-depths LIST` | - | Comma-separated context depths; prints decode tok/s at each depth (with `--speculative`, also accept rate and speedup). |
| `--bench-runs N` | `2` | Timed runs per length; best (max tok/s) reported. |
| `--bench-decode-tokens N` | `32`/`128` | Decode tokens per bench run (`--bench` default 32, `--bench-depths` default 128). |
| `--bench-temp T` | `0.0` | Sampling temperature for speculative bench runs (`0.0` = greedy). |
| `--bench-chat-dataset DATASET` | - | HF chat dataset for bench prompts (`id` or `id:split`); chat-templated prompts give representative MTP acceptance on instruct models. Default: synthetic prompts. |

Exit code `0` = success, `1` = load refusal (unsupported codec/arch), `2` =
usage/file errors (bad flag combination, missing file), `130` = interrupted.

---

## `gmlx chat`

Interactive chat REPL on a GGUF. The model loads once (same loader path as
`run`) and the conversation runs over a persistent prompt cache, so each turn
prefills only the new message. Each reply ends with a one-line stat: prompt and
decode tok/s, MTP acceptance when speculating, and context fill
(`ctx 4.1k/32k`).

```sh
gmlx chat model.gguf --temp 0.7 --system-prompt "You are terse."
gmlx chat --assistant                 # the tool-loop assistant on the managed server
```

With `--assistant` the REPL loads nothing locally: turns run through the
built-in tool-loop assistant on the managed (auto-started) server, with MCP
tools and long-term memory from the config's `assistant:` block. The
positional becomes a served model id (or is omitted for the server default).
`/memory` lists and edits the stored facts. The terminal experience is
unchanged; local-load flags do not apply. Full contract:
[assistant.md](assistant.md#text-chat-gmlx-chat---assistant).

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
  tokens, average tok/s, MTP acceptance).
- `/system [text|off]` shows or sets the system prompt at runtime (setting
  restarts the conversation).
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
  native-head model (qwen3.5/3.6 `nextn`) drafts and verifies multiple tokens
  per step for a decode speedup; gemma4 needs a `--draft-gguf` assistant. The
  reply streams the same way and ends with the same `tok/s` stat, and the
  persistent KV cache is reused across turns exactly like the text path. Not
  combinable with `--adapter` / `--stream-*`. Sampling is
  temperature/top-p/top-k/min-p only; the MTP verify walk has no penalty/bias
  hooks, so the other `/` sampling commands don't apply on this path.
  - VLM + MTP: a `--mmproj` VLM with a drafter (a `--draft-gguf` assistant
    for gemma4, or a native `nextn` head for qwen3.5/3.6) keeps MTP on for
    text-only turns (the fast path above) while `/image` / `/audio` turns
    fall back to the plain VLM stream. The first media turn upgrades the
    session to the VLM path for the rest of the conversation, since the text
    tokenizer can't render a history that holds image markers. The prior text
    turns are carried into that re-prefill so nothing is lost.
- Reasoning display: for thinking models (Qwen3/DeepSeek-R1/GLM `<think>`,
  gpt-oss harmony channels, Gemma `<|channel>thought`), the chain-of-thought
  is stripped of its control markers and streamed dimmed inside a
  gutter-framed block that closes with a payoff line showing how long the
  model thought and how many tokens it spent; the final answer follows in
  normal weight. `--reasoning hide` collapses the reasoning to a single live
  spinner that resolves to the same payoff, so you see it working without
  reading it. Ctrl-O toggles expand and collapse live during a reply (and
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
  Okabe-Ito palette and works with all themes.
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

| Flag | Default | Meaning |
|------|---------|---------|
| `gguf` (positional) | - | Path to the GGUF (sharded ok) or a config model id; with `--assistant`, a served model id (optional: server default). |
| `--assistant` | - | Chat through the built-in tool-loop assistant on the managed server: MCP tools + long-term memory from the `assistant:` block ([assistant.md](assistant.md)). Local-load flags don't apply. |
| `--base-url URL` / `--host` / `--port` / `--api-key` | managed server | Assistant mode: target server (as in [`talk`](#gmlx-talk)). |
| `--no-start` / `--start-timeout S` | - / `180` | Assistant mode: never auto-start the server / auto-start wait. |
| `--max-tokens N` | until EOS | Per-reply decode-token cap; default `0` = each reply runs until the model stops (diffusion models fall back to a bounded 2048-token canvas; in `--assistant` mode `0` defers to the server's own default). Pass N to cap (adjustable via `/max-tokens`; `0` removes the cap, and a note says when the cap ended a reply). |
| `--temp` / `--top-p` / `--top-k` / `--min-p` | family default | Sampling; unset flags seed from the model's [family defaults](#family-defaults-intent-and---profile) (`0.0`/`0.95`/`0`/`0.05` under `--no-family-defaults`). All adjustable in-chat. |
| `--xtc-probability` / `--xtc-threshold` | `0.0` | XTC sampling (text path, adjustable in-chat). |
| `--repetition-penalty` / `--presence-penalty` / `--frequency-penalty` | `0.0` | Penalties (`0` = off, adjustable in-chat). |
| `--repetition-context-size N` | `20` | Penalty lookback window (adjustable in-chat). |
| `--logit-bias JSON` | - | Token-id to bias map. |
| `--stop STR` | - | Stop sequence (trimmed; repeatable). |
| `--kv-bits` / `--kv-group-size` / `--quantized-kv-start` | off / `64` / `0` | KV-cache quantization: cuts cache memory 2-4x for long chats. |
| `--prefill-step-size N` | `2048` / `8192` streaming | Prefill chunk size (peak-memory cap for `/load`-ed long prompts). Streaming `--stream-cpu` / `--stream-experts` models default to `8192`, same as `run`. |
| `--stream-cpu` / `--stream-experts` / `--moe-experts` / `--moe-expert-mass` / `--moe-expert-probe` | - | Execution placement and lossy MoE fan-out, same as [`run`](#loading), including larger-than-RAM streaming `--stream-cpu` chat (text path only). |
| `--resize-shape N\|WxH` / `--thinking-budget N` | - | Image resolution (soft-token count, VLM mode) / thinking-token cap (text + VLM; adjustable via `/thinking-budget`). |
| `--reasoning {show,hide,raw}` | `show` | How a thinking model's reasoning is displayed: gutter-framed with payoff, collapsed spinner, or verbatim. Live toggle: Ctrl-O or `/reasoning`. |
| `--render {auto,plain,lite,rich}` | `auto` | Reply markdown rendering (`auto`: rich when installed on a color TTY). Switch live with `/render`. |
| `--theme NAME` | `dark` | Color theme (`dark`, `light`, `dark-hc`, `nord`, `dracula`, `solarized-dark`, `gruvbox`). Switch live with `/theme`. |
| `--colorblind` | - | Colorblind-friendly accents (Okabe-Ito) for any theme. |
| `--seed N` | - | PRNG seed for reproducible sampling. |
| `-v` / `--verbose` | off | Full load diagnostics instead of the progress spinner. |
| `--system-prompt STR` | - | System message, sent on the first turn (and after each reset; adjustable via `/system`). |
| `--chat-template-config JSON` | - | Extra chat-template kwargs, e.g. `'{"enable_thinking": false}'`. |
| `--max-kv-size N` | - | Cap the KV cache (rotating cache above it). |
| `--no-history` | - | Don't read or write the prompt-history file. |
| `--no-autosave` | - | Don't autosave the session after each turn. |
| `--resume [NAME]` | - | Resume a saved session (default: this model's latest). |
| `--adapter PATH` | - | GGUF LoRA adapter applied live over the base (text path only). |
| `--speculative`/`--mtp` / `--draft-gguf PATH` / `--draft-block-size N` / `--no-speculative`/`--no-mtp` | - | MTP speculative decoding, same surface as [`run`](#speculative--mtp): native-head qwen3.5/3.6 needs no companion, gemma4 needs `--draft-gguf`. With `--mmproj` (a VLM with a drafter), text-only turns take the MTP path and media turns fall back to VLM. Not combinable with `--adapter`/`--stream-*`. |
| `--mmproj PATH` | - | Vision/audio projector GGUF; enables multimodal chat (`/image`, `/audio`, drag-and-drop). |
| `--arch` / `--hf-source` / `--chat-template` / `--no-remap` / `--no-zero-copy` | - | Loading flags, same as [`run`](#loading). |
| `--no-chat-template` | - | Send each turn verbatim, applying no chat template (base / non-instruct GGUFs that carry no template). Plain-text models only. |
| `--config FILE` | - | Resolve the positional against a server config when it isn't a file; same id/alias lookup and settings overlay as [`run`](#resolving-a-model-from-a-config). |
| `--profile NAME` / `--no-family-defaults` | - | Intent/profile selection and family-defaults opt-out, same as [`run`](#family-defaults-intent-and---profile). |

A base GGUF with no chat template refuses to start; pass one with
`--chat-template`, or `--no-chat-template` to send turns verbatim.

`gmlx chat coder` works the same as `run`: a bare name that isn't a file is
resolved from the config, and that model's sampling / `system` /
`chat_template` settings seed the session (still adjustable in-chat). See
[Resolving a model from a config](#resolving-a-model-from-a-config).

---

## `gmlx talk`

Voice chat with a served model: say the wake phrase, speak, and the reply
streams back as speech. It is a client of the server's existing endpoints
(STT + chat + TTS), so it needs `stt:` and `tts:` in the server config and
`pip install 'gmlx[talk]'`; it starts the background server when it's
down. The full guide, covering setup, the `talk:` config block, latency
tuning, and the in-session keys and `/commands`, is [talk.md](talk.md).

```sh
gmlx talk                                     # wake mode: listens for "hey assistant"
gmlx talk --wake-word "okay computer" --voice bf_emma
gmlx talk --mode vad                          # open mic: any speech starts a turn
gmlx talk --once                              # one exchange, then exit
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--model ID[@profile]` | server default | Chat model to talk to. |
| `--voice NAME` / `--list-voices` | server default | TTS voice / list the server's voices. |
| `--speed X` | `1.0` | Speech speed (0.25-4). |
| `--mode {wake,vad,ptt,text}` | `wake` | Listening mode (`--once` skips the wake gate: speak immediately, one exchange). |
| `--wake-word PHRASE` / `--wake-threshold X` | `hey assistant` / `0.5` | Any text phrase, no training; higher threshold = fewer false fires. |
| `--vad-threshold` / `--vad-silence-ms` / `--min-speech-ms` | `0.6` / `550` / `300` | Endpointing knobs. |
| `--input-device` / `--output-device` / `--list-devices` | system default | Audio devices (name substring or index). |
| `--system TEXT` / `--language L` / `--max-tokens N` | - / - / `512` | Spoken persona / whisper hint / reply cap. |
| `--no-chime` | - | Disable the wake/idle earcons. |
| `--brain {chat,assistant}` | config `talk.brain`, else `chat` | Turn engine: plain chat, or the built-in assistant with MCP tools and long-term memory ([assistant.md](assistant.md); tools need `pip install 'gmlx[assistant]'`). |
| `--base-url URL` / `--host` / `--port` / `--api-key` | managed server | Target server (a remote `--base-url` runs STT/TTS there). |
| `--no-start` / `--start-timeout S` | - / `180` | Don't autostart the server / autostart wait. |
| `--config PATH` | default locations | YAML with the `talk:` block (flags override it). |

---

## `gmlx serve`

The platform's server: continuously batched, multi-model, OpenAI/Anthropic-
compatible (text + VLM + MTP), serving a config of named models and profiles. Every model's sampling starts from its family's
model-card defaults, and a request can address `id@intent`; see
[server-config.md profiles](server-config.md#profiles-sampling-profiles-and-built-in-intents).
Config-defined assistant ids (`server.assistants:`) answer through the
built-in server-side tool loop ([assistant.md](assistant.md#served-assistants)).
The full config surface and the start-mode semantics live in
[server-config.md](server-config.md); this is the command-line summary.

```sh
gmlx serve       [model.gguf] [options]
gmlx init                                   # guided wizard (bare, on a terminal)
gmlx init        (--models-dir DIR | --from-hf-cache) [--port N] [--disk-cache [GB]] [--with-stt|--with-tts|--with-embeddings [MODEL]|--with-rerank [MODEL]] [--idle-ttl T] [--request-timeout T] [--out FILE] [-r] [--force]
gmlx sync-models [--config FILE] [--models-dir DIR] [--from-hf-cache] [--no-recursive] [--dry-run]
gmlx launch      <harness> [options]
```

```sh
# single model (pinned, id derived from the filename)
gmlx serve model-Q4_K_M.gguf

# a config of named models + profiles
gmlx serve --config ~/.config/gmlx/gmlx.yaml

# discovery-scan a directory, no config file
gmlx serve --models-dir ~/models --recursive

# a single VLM (LLM GGUF + float mmproj)
gmlx serve gemma-4-E4B-it-Q6_K.gguf --mmproj mmproj-gemma-4-E4B-it-bf16.gguf

# a single native-head MTP model
gmlx serve Qwen3.6-27B-Q4_K_S.gguf --speculative
```

### Serve flags

| Flag | Meaning |
|------|---------|
| `model` (positional) | A single GGUF to serve (sharded ok). Mutually exclusive with `--config`/`--models-dir`. |
| `--config FILE` | Serve a YAML config (named models + profiles). Enables `POST /v1/reload`. |
| `--print-config` | Resolve the effective config for the chosen start mode (config / discovery / single model), print it as YAML with every key and default filled in, and exit without serving. Use it to introspect the schema or sanity-check a config. |
| `--models-dir DIR` | Serve a header-only discovery scan of a directory (repeatable). |
| `-r`, `--recursive` / `--no-recursive` | Recurse into subdirectories when discovering (default: shallow). |
| `--hf-cache` (alias `--from-hf-cache`) | Resolve named hf repo ids from the local hf cache only (never the network). Off means no HF access at all. |
| `--mmproj PATH` | Float mmproj GGUF; makes a single positional model a VLM. |
| `--hf-source REPO` | Processor/config override for a single VLM model (rarely needed). |
| `--speculative` | Serve a single positional model with MTP (native-head qwen3.5/3.6; gemma4 also needs `--draft-gguf`). |
| `--draft-gguf PATH` | Companion drafter GGUF for assistant-shape MTP (gemma4); implies `--speculative`. |
| `--draft-block-size N` | MTP draft tokens per round (analogous to llama-server `--spec-draft-n-max`). Default: the drafter's own block size. Also via `GMLX_DRAFT_BLOCK_SIZE`. |
| `--adapter PATH` | GGUF LoRA adapter applied live over a single positional model at load (text only, not `--mmproj`/`--speculative`). In config mode set `adapter:` per model instead. |
| `--stream-cpu` | Run a single positional model entirely on the CPU device: the over-RAM MoE path, same semantics as [`run --stream-cpu`](#loading). In config mode set `stream: cpu` per model instead; see [server-config.md](server-config.md#models). |
| `--stream-experts` | Routed-expert stacks stream from disk while the every-token layers and KV cache stay on GPU; the decode feeder (default) serves decode from a wired expert arena and makes this the faster placement once warm. Config mode: `stream: experts`. Mutually exclusive with `--stream-cpu`. |
| `--moe-expert-mass P` | Lossy, adaptive experts-per-token for a single positional model on the streamed MoE layers (`--stream-experts` / `--stream-cpu`), same semantics as [`run --moe-expert-mass`](#loading). Size P with a `gmlx run --moe-expert-probe` pass first. Config mode: `moe_expert_mass: P` per model; see [server-config.md](server-config.md#models). |
| `--chat-template STR\|PATH` | Inline Jinja, or a `.jinja`/`.txt` path, replacing a single positional model's GGUF template. In config mode set it per profile/model (`chat_template:` / `overrides: {chat_template: ...}`) instead; see [server-config.md](server-config.md). |
| `--host ADDR` | Bind address (default from config or `127.0.0.1`). A non-loopback host refuses to start without `server.api_key` unless `--no-auth` opts out. |
| `--port N` | Port (default from config or `8080`). |
| `-f`, `--foreground` | Run the server attached to this terminal (blocking) instead of the default detached background start. The background start returns at once (so `gmlx launch` works in the same shell), lands a runfile and log under `~/.cache/gmlx/`, and on a macOS GUI session also raises the [menu-bar monitor](#launch-menubar). Manage it with [`stop`/`restart`/`status`/`logs`](#background-mode--server-lifecycle). |
| `--no-menubar` | Don't auto-start the macOS menu-bar monitor with a background server (same effect as `server.menubar: false`). |
| `--log FILE` | Log file for a background server (default `~/.cache/gmlx/server-<host>-<port>.log`). Each background start rotates the previous run's log to `<file>.1`; a launchd agent's append-mode log is rotated at each `gmlx service install` once it passes 10 MB (launchd holds the file open between installs). |
| `--log-level LEVEL` | Server log verbosity: `critical`, `error`, `warning`, `info` (default), `debug`, or `trace`. `debug`/`trace` add uvicorn's connection-level detail; survives background/launchd relaunches. |
| `--start-timeout S` | How long a background start waits for the child to become ready before returning with a "still starting" note (default `40`). |
| `--budget-gb F` | Resident weight-byte budget across all models (default 0.8x the GPU recommended working set). |
| `--max-models N` | Optional secondary cap on resident model count. |
| `--pin ID_OR_PATH` | Pin a model so it is never evicted (repeatable). |
| `--max-tokens N` | Server default max completion tokens. |
| `--ignore-eos` | Never stop on EOS; decode every request to `max_tokens` (forced-length throughput benchmarking; mirrors llama-server `--ignore-eos`). Also via `GMLX_IGNORE_EOS=1`. |
| `--no-auth` | Serve a non-loopback bind without an API key: an explicit opt-out (config: `server.no_auth: true`) for setups that authenticate in front (mTLS, reverse proxy). Loopback binds never need it. |
| `--stt [MODEL]` | Speech-to-text: serve `POST /v1/audio/transcriptions` via mlx-whisper (`pip install 'gmlx[stt]'`, ffmpeg on PATH). MODEL = an alias (`whisper-turbo`, `whisper-turbo-q4`, `whisper-large/medium/small/base/tiny`), any MLX-whisper HF repo, or a local dir. Bare `--stt` uses `whisper-turbo`. Overrides config `server.stt:`; see [server-config.md](server-config.md#speech-to-text-stt). |
| `--tts [MODEL]` | Text-to-speech: serve `POST /v1/audio/speech` via mlx-audio (`pip install 'gmlx[tts]'`, ffmpeg on PATH for non-wav). MODEL = an alias (`kokoro`, `kokoro-8bit/4bit`, `qwen3-tts`), any MLX-audio HF repo, or a local dir. Bare `--tts` uses `kokoro`. Overrides config `server.tts:`; see [server-config.md](server-config.md#text-to-speech-tts). |
| `--embeddings [MODEL]` | Text embeddings: serve `POST /v1/embeddings` (no extra needed; mlx-embeddings is a core dependency). Bare `--embeddings` uses `qwen3-embed-0.6b`; the MODEL forms are listed below the table. This is the local RAG embedder `gmlx launch open-webui` points at. Overrides config `server.embeddings:`; see [server-config.md](server-config.md#text-embeddings-embeddings). |
| `--rerank [MODEL]` | Reranking: serve `POST /v1/rerank` (Cohere/Jina shape) from a Qwen3-Reranker GGUF (no extra needed). MODEL = an alias (`qwen3-rerank-0.6b`/`-4b`/`-8b`), a `*.gguf`, or `hf:.../*.gguf`. Bare `--rerank` uses `qwen3-rerank-0.6b`. This is the RAG second stage `gmlx launch open-webui` points at. Overrides config `server.rerank:`; see [server-config.md](server-config.md#reranking-rerank). |

For `--embeddings MODEL`, MODEL is either a GGUF embedder (decoder-LM alias
`qwen3-embed-0.6b`/`-4b`/`-8b`, encoder alias `embeddinggemma-gguf`, a
`*.gguf`, or `hf:.../*.gguf`) or an mlx-embeddings safetensors encoder (alias
`embeddinggemma`/`arctic-l`/`nomic-embed`/`bge-m3`, any MLX-embeddings HF
repo, or a local dir).

VLM and MTP coexist: a resident VLM serves text-only requests through the MTP
speculative path and image/audio requests through the plain VLM forward
(gemma4 needs `--draft-gguf`; qwen3.5/3.6 use the native `nextn` head, no
companion). In a config this is a model with both `mmproj:` and
`speculative: true`; discovery (`speculative: auto`) turns it on automatically
for a paired-mmproj model whose GGUF has a native head. `/v1/models` flags
such a model `[mtp, vlm]`. `--adapter` on a VLM is still unsupported.

The API key is config-only: the server reads its key from `server.api_key` in
the config, and nowhere else. There is no `serve --api-key` flag and no
server-side `GMLX_API_KEY` fallback. That is a deliberate simplification:
one key, in one file, readable by the lifecycle tools and the menu bar, never
stored in a runfile. Loopback binds need no key; a non-loopback bind refuses
to start without `server.api_key` unless `--no-auth`. Clients (`ps`, `status`,
`launch`, `menubar`) still take `--api-key` to present a key. See
[server-config.md](server-config.md).

### Background mode & server lifecycle

`serve` detaches by default and returns at once, so there is no second shell
to run [`launch`](#launch-connect-a-coding-agent-or-chat-app) in; pass `-f` /
`--foreground` to stay attached. State (a runfile and a log) lives under
`~/.cache/gmlx/`, keyed by host and port, so several binds coexist. On a
macOS GUI session the background start also raises the
[menu-bar monitor](#launch-menubar) (disable with `--no-menubar` or
`server.menubar: false`). With one managed server, the verbs below need no
`--host/--port`; with several, pass them.

```sh
gmlx serve --config ~/.config/gmlx/gmlx.yaml   # detaches + returns (add -f to stay attached)
gmlx launch claude-code          # same shell (and starts the server if it isn't up)
gmlx status                      # pid, uptime, url (no API key needed; uses /health)
gmlx ps                          # resident models (needs the key on a key-protected server)
gmlx logs -n 40 -f               # tail the server log; -f follows
gmlx restart                     # relaunch with the original arguments
gmlx stop                        # SIGTERM the process group, then SIGKILL after --timeout
```

| Command | Meaning |
|---------|---------|
| `gmlx stop [--host H --port P] [--timeout S]` | Stop a backgrounded server: SIGTERM the whole process group, then SIGKILL after `--timeout` (default `15`, cutting any in-flight generation). Verifies the pid is ours before signalling; a stale runfile is just cleared. |
| `gmlx restart [...] [--start-timeout S]` | Stop, then relaunch with the runfile's recorded arguments (absolute `--config`, so it relaunches faithfully from any directory). |
| `gmlx status [--json]` | Process-layer status: pid, uptime, url, log, how it's managed. No API key (uses the auth-exempt `/health`). Resident-model detail lives in `ps`. |
| `gmlx logs [-n N] [-f] [--clear]` | Print the last `N` lines of the server log (`-n`/`--lines`). `-f`/`--follow` follows (`tail -f`); `--clear` truncates the file (keeps it). |

Each completed generation request logs one timestamped line with the endpoint,
model, token counts, and timing (`ttft`, `prefill`/`decode` tok/s, `total`):

```text
[req] 2026-06-15 16:07:42 /chat/completions qwen3-0.6b prompt=19 gen=3 ttft=0.47s prefill=45t/s decode=172.6t/s total=0.51s
```

Other endpoints keep a standard, timestamped access line. The menu bar's
once-a-second `/health` and `/v1/metrics` polls are filtered out, so the log
stays readable. The timing is read from mlx-vlm's own per-request metrics (the
same numbers `/v1/metrics` aggregates), so this adds no measurement overhead.

#### `service`: run at login (launchd, macOS)

`gmlx service install` registers a launchd LaunchAgent for the *menu bar*:
it comes up at every login, and (unless `--no-autostart`) starts the server
too when one isn't already up. It takes the same options as `serve`; the
server it starts now is the one it will bring back at each login. macOS-only.

```sh
gmlx service install --config ~/.config/gmlx/gmlx.yaml   # start now + at login
gmlx service status                                            # launchd state
gmlx service uninstall                                         # unload + remove (idempotent)
```

Running the menu bar under launchd (rather than from a terminal) also makes
macOS attribute its permission prompts - microphone for voice chat,
Accessibility for the tap-to-talk hotkey, notifications - to "gmlx" instead
of your terminal app. The entry in System Settings' Login Items likewise
shows gmlx (the agent launches through a small script inside gmlx.app,
never a bare `python`).

The server itself stays an ordinary background process: if it crashes, the
menu bar posts a notification and offers one-click Start (it does not
supervise). Autostart runs once per login - a server you deliberately stop
stays stopped until the next login. Stop the server with `gmlx stop` as
usual; quit the menu bar from its own menu (`service uninstall` removes the
login item).

`--headless` installs the pre-menubar shape instead: a per-port agent that
runs `serve` directly, restarts it on crash (`--no-keepalive` opts out), and
needs no GUI session - for SSH-only boxes. A headless-managed server is
stopped with `service uninstall` (not `stop`). The two modes refuse to share
a host:port.

## `init`: scaffold a config

Run bare on a terminal, `gmlx init` opens a guided wizard: it scans your
model directories, lets you rename or drop the auto-named models and set a
default and aliases, then walks through the on-disk prompt cache, the optional
speech-to-text / text-to-speech / embeddings services (offering to
`pip`-install a missing extra), and the residency limits. It previews the YAML
before it writes. `-i` forces the wizard even alongside flags;
`--no-interactive` (or `--help`) takes the flag-driven path.

Auto-named ids carry the quant codec in compact form (`qwen3-0.6b-q4`,
`llama-3.2-1b-instruct-iq3`); when two quants of one model share that compact
form (e.g. `Q4_K_S` and `Q4_K_M` both give `q4`), the full codec is used on
both (`...-q4-k-s` / `...-q4-k-m`) so the id is never ambiguous. Rename any of
them in the wizard or edit `models:` afterward.

Flag-driven, `init` discovers the GGUFs under `--models-dir` (repeatable)
and/or the local Hugging Face cache (`--from-hf-cache`) and writes a starter
YAML; it is the only mode that writes a file. It streams per-file scan
progress to stderr, writes to `~/.config/gmlx/gmlx.yaml` by default
(override with `--out`), and refuses to overwrite without `--force`. Every
wizard choice except the voice-chat step has a flag below, so a script does
the same without the prompts (a `talk:` block is edited in by hand).

```sh
gmlx init                                    # guided wizard (on a terminal)
gmlx init --models-dir ~/models            # -> ~/.config/gmlx/gmlx.yaml
gmlx init --models-dir ~/models -r --out ./gmlx.yaml   # project-local instead
gmlx init --from-hf-cache                    # models you already have in the hf cache
gmlx init --models-dir ~/models --with-stt --with-embeddings --with-rerank   # wire services
```

| Flag | Meaning |
|------|---------|
| `--models-dir DIR` | Directory of GGUFs to scan (repeatable). Required unless `--from-hf-cache`. |
| `--from-hf-cache` (alias `--hf-cache`) | Also scan the local hf cache; add cache GGUFs as portable `hf:<org>/<repo>/<file>` entries and set `server.hf_cache`. Resolved from the cache, never downloaded. |
| `--disk-cache [GB]` | Enable the on-disk prompt cache in the generated config: a shared prompt prefix is reused across requests and restarts (persisted under `~/.cache/gmlx/apc`), skipping the prefill recompute on a hit. Without the flag the same block is written commented out, ready to flip on. Bare `--disk-cache` caps the store at 50 GB per model; pass a size to override (`--disk-cache 100`). The wizard asks the size as a follow-up. |
| `-i`, `--interactive` | Run the wizard even with flags present (they pre-seed the answers). |
| `--no-interactive` | Never run the wizard; scaffold from the flags as given. |
| `--with-stt [MODEL]` | Configure speech-to-text (`server.stt`). Bare uses the default alias (`whisper-turbo`); a value sets the model (alias / HF repo id / local path). |
| `--with-tts [MODEL]` | Configure text-to-speech (`server.tts`). Bare uses `kokoro`. |
| `--with-embeddings [MODEL]` | Configure text embeddings (`server.embeddings`). Bare uses `qwen3-embed-0.6b`; takes a `qwen3-embed-*` / `embeddinggemma-gguf` alias, a `*.gguf` / `hf:.../*.gguf` ref, or an mlx-embeddings encoder alias (`embeddinggemma` and friends) / repo / dir. No extra needed (mlx-embeddings is core). The wizard adds a quant follow-up and auto-pickup of an embedder GGUF already on disk. |
| `--with-rerank [MODEL]` | Configure reranking (`server.rerank`). Bare uses `qwen3-rerank-0.6b`; a `qwen3-rerank-*` alias / `*.gguf` / `hf:.../*.gguf` (a Qwen3-Reranker GGUF, no extra needed). In the wizard the quant defaults to the embedder's chosen rung. |
| `--install` / `--no-install` | After writing, `pip`-install the missing extras for the chosen `--with-*` services / don't offer to install extras in the wizard. |
| `--default-model ID` | Set `server.defaults.model` (used when a request omits `model`). |
| `--port N` | Set `server.port` (default 8080). |
| `--idle-ttl SECONDS\|none` | Set `server.defaults.ttl_s`: idle auto-unload. `none` keeps models resident (evict manually or under LRU pressure). |
| `--request-timeout DURATION\|none` | Set `server.token_queue_timeout_s`: give up if no new token arrives for this long (e.g. `10m`, `1h`). `none` waits forever. |
| `--out FILE` | Where to write the config (default `~/.config/gmlx/gmlx.yaml`). |
| `-r`, `--recursive` / `--no-recursive` | Recurse into subdirectories when scanning `--models-dir` (default: shallow). |
| `--force` | Overwrite an existing config. |
| `--no-reload` | Don't `SIGHUP` a server already running this config to pick up the change. |

Pointing `init` at an empty directory is fine: it writes a valid zero-model
config and tells you to `pull` GGUFs in (or drop them) and run `sync-models`
to add them.

If a `--config` server is already running the config you just (re)wrote,
`init` SIGHUPs it so it re-reads the file and re-registers its models without
a restart (resident models stay warm); pass `--no-reload` to skip that. See
[Reloading the config](server-config.md#reloading-the-config).

## `sync-models`: reconcile a config with what's on disk

Re-scans the config's model dirs and updates the `models:` block to match
disk: configured models that still exist are left untouched (comments and
hand-edits preserved), models whose file is gone are dropped, and
newly-discovered GGUFs are added. Use it after dropping or `pull`-ing files
into your model dir; it's the incremental counterpart to `init`. Operates on
the first existing default config unless `--config` is given. Scanning
recurses by default, because `pull` nests downloads under
`<dir>/<org>__<repo>/`.

With `--from-hf-cache` (or a config already carrying `server.hf_cache: true`)
it also reconciles cache-resident GGUFs, adding new `hf:` entries and dropping
ones that are no longer cached, and flips `server.hf_cache` on for them.

Removal only happens when the entry could actually be checked: if the hf cache
is unreadable, or a whole `model_dirs` root is missing (an unmounted disk, a
different shell environment), the entries it covers are kept and reported as
`keep: (unverified - ...)` with a warning, never dropped. Absolute-path
entries whose file is gone are dropped like any other.

```sh
gmlx pull hf:unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_S.gguf  # into your model dir
gmlx sync-models                              # adds the just-pulled model to the config
gmlx sync-models --from-hf-cache              # also pick up models in the hf cache
gmlx sync-models --config ~/.config/gmlx/gmlx.yaml --dry-run    # preview only
```

| Flag | Meaning |
|------|---------|
| `--config FILE` | Config to sync (default: the first existing default location). |
| `--models-dir DIR` | Dirs to scan, overriding the config's `server.model_dirs` (repeatable). |
| `--from-hf-cache` (alias `--hf-cache`) | Also reconcile cache-resident GGUFs (implied when the config has `hf_cache: true`). |
| `-r`, `--recursive` / `--no-recursive` | Recurse into subdirectories (default: deep, since `pull` nests under `<dir>/<org>__<repo>/`). |
| `--dry-run` | Show the add/remove plan without writing the config. |
| `--no-reload` | Don't `SIGHUP` a server already running this config to pick up the change. |

When `sync-models` actually rewrites the config (not a `--dry-run`, and
something changed), a `--config` server already running that file is SIGHUP'd
so it re-reads it and re-registers its models without a restart; resident
models stay warm. Pass `--no-reload` to skip it. See
[Reloading the config](server-config.md#reloading-the-config).

## `launch`: connect a coding agent or chat app

Configures and execs an external tool against a gmlx server: a coding
harness (claude-code, opencode, pi, omp), an agent runtime (hermes, goose), a
terminal chat client (aichat, elia), or the Open WebUI browser app. It writes
the tool's native config without touching your dotfiles, auto-starts a server
in the background if none is reachable, and never installs the tool itself.
Behavior, per-client details, and troubleshooting: [launch.md](launch.md).

```sh
gmlx launch opencode                         # uses the server's default-marked model
gmlx launch pi --model qwen3.6-27b@coding    # a served id, on the family's coding sampling
gmlx launch claude-code --model qwen3.6-27b  # Anthropic surface (/v1/messages)
gmlx launch open-webui                       # browser chat app on :3000
gmlx launch omp --config-only                # write the config, print the run command
```

| Flag | Meaning |
|------|---------|
| `--model ID[@profile]` | Served model (and optional profile) the tool uses; also kept resident through the idle-TTL reaper. Default: the server's default-marked model. |
| `--base-url URL` | Target an explicit server (never auto-started). |
| `--host H` / `--port P` | Managed-server target (default: the single managed server if there's one, else the config's, else `127.0.0.1:8080`). |
| `--api-key KEY` | Key for a key-protected server, placed in the tool's native slot. |
| `--provider-id NAME` | Provider id written into the tool's config. |
| `--config-path PATH` | Where the tool config is written (default under `~/.config/gmlx`; open-webui: its `DATA_DIR`). |
| `--config-only` | Write the config and print the run command; do not exec. |
| `--no-start` | Never auto-start a server; error if it is down. |
| `--start-timeout S` | Cap the auto-start readiness wait (default `0` = unbounded). |
| `--no-keep` | Do not keep `--model` resident. |

Exit codes: `0` tool launched or server ready; `1` server down (with
`--no-start` / `--base-url`), launchd server mid-restart, or the auto-started
process died; `2` no config or a malformed config; `130` Ctrl-C during the
start wait.

### `launch menubar`

`gmlx launch menubar` puts a small status-bar item up for a backgrounded
server: up/down/busy state, resident models (size, default/pinned markers,
eviction countdown; click to unload), reload-config, restart, stop, copy-URL,
and open-logs, plus a notification if the server dies. "Edit config" opens the
server's YAML in a floating editor that validates with the server's own parser
and can save-and-reload in one step. A background `serve`
raises it automatically on a macOS GUI session (disable with `--no-menubar` or
`server.menubar: false`); `gmlx launch menubar --stop` quits a detached
monitor from the CLI.
Flags, targeting, and details:
[launch.md](launch.md#the-menu-bar-app-launch-menubar).

---

## `gmlx train`

LoRA-finetune a K-quant GGUF base and write the adapter as a GGUF: GGUF in,
GGUF out, no safetensors, no merge. gmlx runs the training on mlx-lm's LoRA
tuner; the adapter's gradient flows through the frozen quant matmul via the kquant op's
`vjp`, so the base carries no float copy and no optimizer state (you can
finetune a model you couldn't hold in fp16). The emitted adapter round-trips
straight back into [`run --adapter`](#adapter-lora) /
[`serve --adapter`](#serve-flags). Full walkthrough with a worked example:
[docs/lora.md](lora.md). LoRA only: mlx-lm's DoRA dispatch doesn't consult the
kquant base.

```sh
# finetune the top 8 layers, 150 iters, write a GGUF adapter
gmlx train base-Q8_0.gguf --data ./my-data --adapter-out my-lora.gguf

# then serve (or run) the base with the adapter attached at load, no merge
gmlx serve base-Q8_0.gguf --adapter my-lora.gguf
gmlx run   base-Q8_0.gguf --adapter my-lora.gguf --prompt "..."
```

`--data` is a directory of `train.jsonl` / `valid.jsonl` (or an HF dataset id)
in any format mlx-lm's LoRA trainer accepts: chat (`{"messages": [...]}`),
prompt/completion, or plain text. The adapter targets the attention and MLP
projections of the top `--num-layers` transformer blocks.

| Flag | Default | Meaning |
|------|---------|---------|
| `model` (positional) | - | Path to the base GGUF (sharded ok), or a server-config model id/alias when `--config` is set (or a default config exists); same id resolution as [`run`](#resolving-a-model-from-a-config). |
| `--config FILE` | - | Server config to resolve the base model name against when it isn't a file on disk (default: the first existing default config). |
| `--data PATH\|ID` | - | Dataset dir (`train.jsonl`/`valid.jsonl`) or HF dataset id. Required. |
| `--adapter-out PATH` | - | Output path for the trained `.gguf` adapter. Required. |
| `--iters N` | `150` | Training iterations. |
| `--batch-size N` | `4` | Batch size. |
| `--num-layers N` | `8` | Number of top transformer layers to adapt. |
| `--rank N` | `8` | LoRA rank. |
| `--scale F` | `20.0` | LoRA scale (`alpha = scale x rank`, recovered on load). |
| `--dropout F` | `0.0` | LoRA dropout. |
| `--learning-rate F` | `1e-4` | Adam learning rate. |
| `--max-seq-length N` | `2048` | Max training sequence length. |
| `--val-batches N` | `25` | Validation batches per eval. |
| `--steps-per-report N` | `10` | Train-loss report interval. |
| `--steps-per-eval N` | `200` | Validation interval. |
| `--seed N` | `0` | RNG seed. |
| `--hf-source ID` | - | HF repo id for tokenizer/config fallback (rarely needed; the GGUF synthesizes both). |

---

## `gmlx doctor`

One pass over everything a working setup needs, with a PASS/WARN/FAIL line
per check and the fix named for anything that fails. It checks: the runtime
(mlx, mlx-kquant, mlx-lm importable, Metal available, versions), the compiled
kernels, the config (parse errors and warnings), every configured model's
paths (all shards, mmproj/draft/adapter companions), any background server
(including stale run files), free disk space, and the HF token. Rows for
optional features (installed launchd agents and their load state, extras
installed, ffmpeg, MCP tool binaries, served assistants exposed beyond
loopback) appear only when your setup uses them.
It never touches the network and finishes in about a second.

```sh
gmlx doctor                          # check the default config
gmlx doctor --deep                   # also header-read every configured GGUF
gmlx doctor --json
```

| Flag | Meaning |
|------|---------|
| `--config FILE` | Config to check (default: the bare-start search path). |
| `--deep` | Also read each configured model's GGUF header (catches codec/arch problems before a load). |
| `--json` | Emit `{version, checks, ok}` as JSON. |

Exit codes: `0` when nothing failed (warnings are fine), `1` when any check
failed, `2` usage error.

## `gmlx validate`

Check that a GGUF will load before you commit to a multi-GB download. A local
file is classified by reading its header; a remote ref is checked by
range-reading just the GGUF header (a few MB), never the weights. Both report
the architecture (and whether the installed `mlx-lm` implements it) plus the
per-tensor codec histogram, and agree because they reuse the same codec gate
as the loader. Exit code `0` = loadable, `1` = not, `2` = couldn't resolve or
read the ref (no such file, an ambiguous folder ref, an unreadable header).

An mmproj file (`general.architecture = "clip"`, a VLM's vision/audio
projector) is recognized as a companion, not judged as a standalone model: the
verdict says to pair it with its LLM GGUF via `--mmproj`, the exit code is
`0`, and `pull` downloads it without `--force`.

```sh
# a local file (all shards)
gmlx validate ~/models/Qwen3.6-27B-Q4_K_S.gguf

# a remote file on Hugging Face: hf:<org>/<repo>/<path/to/file.gguf>[@revision]
gmlx validate hf:unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_S.gguf

# a folder: one GGUF model inside auto-resolves; several are listed to pick from
gmlx validate hf:unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF/UD-Q5_K_M

# a bare repo: lists every quant variant as a ready-to-paste ref
gmlx validate hf:unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF

# paste a Hugging Face web link straight from the browser (blob / tree / resolve)
gmlx validate https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/blob/main/Qwen3.6-27B-Q4_K_S.gguf

# any direct URL, or a machine-readable verdict
gmlx validate https://example.com/models/model-Q6_K.gguf
gmlx validate hf:org/repo/model.gguf --json
```

A ref can be:

| Form | Example |
|------|---------|
| local path | `~/models/model.gguf` |
| `hf:` file | `hf:org/repo/path/to/file.gguf` (optionally `@<revision>`, default `main`) |
| `hf:` folder | `hf:org/repo/UD-Q5_K_M`: lists the GGUFs inside; one model auto-resolves |
| `hf:` repo | `hf:org/repo`: lists every quant variant as a pickable ref |
| HF web URL | `https://huggingface.co/org/repo/blob/main/file.gguf` (or `/tree/...`, `/resolve/...`), normalized to the raw file or folder |
| direct URL | `https://host/path/file.gguf` |

A folder or repo ref is resolved by listing it: a single GGUF model (one file
or one split set) is validated automatically (a `[resolved]` line is printed);
several models produce a clean error listing each as an `hf:` ref you can
copy. A pasted huggingface.co page link (`/blob/...`, `/tree/...`) is
rewritten to the underlying file or folder, so you don't have to hunt for the
`/resolve/...` raw link.

| Flag | Meaning |
|------|---------|
| `ref` (positional) | Local path, `hf:<org>/<repo>/<file.gguf>[@rev]`, or an `http(s)://` URL. |
| `--arch NAME` | Override architecture detection. |
| `--hf-source ID` | Treat the arch as loadable with this config override (matches `run --hf-source`). |
| `--max-mb N` | Cap the remote header range-read (MB; default 128). |
| `--json` | Emit the verdict as JSON instead of the report. |

The verdict names the offending codec when a file won't load. A ternary
BitNet GGUF, for example, reports `=> NOT LOADABLE: unsupported codecs
(TQ1_0 x160)` so you can pick a supported quant instead. The K-quant, legacy,
and IQ families all load.

Split GGUFs (`...-00001-of-000NN.gguf`) are handled as one model: point
`validate` at any shard and it range-reads every shard's header and unions the
result (the report notes the shard count). This is necessary, not just
convenient. A split file keeps its architecture in the first (often
tensor-free) shard and spreads its tensors across the rest, so a codec used by
even a single tensor can hide in a later shard; checking only the first shard
would wrongly report it loadable.

## `gmlx pull`

Run the same remote header check and, only if it passes, download the GGUF
(all shards of a split file) into your model library. It writes a plain file,
not the Hugging Face blob cache, so you get a model you can point `run` /
`serve` at directly. A file that lands under a `model_dirs` root is also
registered in your config on the spot, with the same id derivation and mmproj
pairing as `sync-models`, comments preserved. A running server picks it up
immediately, so a pull is requestable as soon as it lands. `--no-register` opts out;
`gmlx sync-models` remains the bulk reconcile.

```sh
# download into your model dir, under <model-dir>/unsloth__Qwen3.6-27B-GGUF/
gmlx pull hf:unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_S.gguf

# download into an explicit directory instead (no <org>__<repo> nesting)
gmlx pull hf:org/repo/model.gguf --to ~/models

# a sharded model downloads every shard automatically
gmlx pull hf:org/repo/model-00001-of-00005.gguf

# fetch several files from one repo at once: a model plus its mmproj, or two quants.
# the first ref names the repo; bare filenames after it resolve in that same repo.
gmlx pull hf:org/gemma-3-27b-GGUF/gemma-3-27b-Q4_K_M.gguf mmproj-F16.gguf
```

By default `pull` lands files in the first `model_dirs` root from your server
config (searched in the
[standard locations](server-config.md#default-config-locations), or pass
`--config FILE`), nesting hf downloads under `<dir>/<org>__<repo>/` so
siblings stay grouped and `sync-models` / recursive discovery find them. With
no config, `pull` errors and tells you to run `init` (or pass `--to DIR`).
`--to DIR` overrides this and writes straight into `DIR`, with no
`<org>__<repo>` subdir.

Pass several refs to fetch them in one run. Multipart (split) GGUFs always
expand to every shard automatically. To grab extra files from the same repo
(an mmproj companion, a second quant), name them after the first ref as bare
filenames; they resolve in that ref's repo (and subfolder).

Interrupted downloads resume rather than restart: each file streams to a
sibling `.part` and is renamed into place only on completion, so re-running
`pull` continues where it left off (and skips files already finished).

If the header check says a file won't load, `pull` refuses it unless you pass
`--force`. Set `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) for gated/private
repos.

Before downloading, `pull` also checks that the destination volume has room
for every shard (bytes already landed in `.part` resume files count toward
the total). If it does not, the pull is refused with both numbers; use
`--to DIR` to target another volume, or `--force` to try anyway.

| Flag | Default | Meaning |
|------|---------|---------|
| `refs` (positional, 1+) | - | `hf:<org>/<repo>/<file.gguf>[@rev]` or an `http(s)://` URL (a local path is rejected; it's already on disk). Extra bare filenames resolve in the first ref's repo. |
| `--to DIR` (alias `--out`) | model dir | Download into `DIR` exactly (no `<org>__<repo>` nesting). Default: the config's first `model_dirs` root, nesting hf files under `<dir>/<org>__<repo>/`. |
| `--config FILE` | - | Config to read `model_dirs` from for the default destination (default: search the standard locations). |
| `--force` | - | Download even if the header check says it won't load or the disk-space check fails. |
| `--hf-source ID` | - | Treat the arch as loadable with this config override. |
| `--max-mb N` | `128` | Cap the remote header range-read (MB). |
| `--json` | - | Emit each verdict as JSON before downloading. |
| `--no-register` | - | Don't add the downloaded file(s) to the server config's models. |

## `gmlx rm`

The inverse of `pull`: delete a model's GGUF file(s) from disk and remove its
entry from the server config in one step. All shards of a split file go, along
with any `.part` resume leftovers and the model's mmproj/draft/adapter
companions, unless another configured (or discovered) model still references
the same file, in which case that file is kept and a note says so. Aliases
pointing at the removed id are dropped, and `server.defaults.model` is cleared
if it named it. The config edit is a round-trip rewrite: comments and
hand-edits elsewhere in the file survive.

```sh
gmlx rm old-model                    # shows the plan, then asks
gmlx rm old-model --yes              # no prompt
gmlx rm old-model --keep-files       # drop the config entry, keep the files
```

Before deleting anything, `rm` prints every file with its size and what will
happen to the config, then asks for confirmation (`--yes` skips this; without
a terminal, `--yes` is required). A model that only appears via a `discover:`
scan has no config entry; its files are deleted and it disappears from
`gmlx list` on its own. Models whose `path` is an `hf:` cache ref lose only
the config entry; the Hugging Face cache manages its own blobs.

| Flag | Meaning |
|------|---------|
| `ID` (positional) | Model id (or alias) from the config, or a discovered model's id (see `gmlx list`). |
| `--config FILE` | Config to read (default: the bare-start search path). |
| `--keep-files` | Remove only the config entry; leave files on disk. |
| `--yes` | Skip the confirmation prompt. |
| `--json` | Emit the removal result as JSON (requires `--yes`). |

Exit codes: `0` removed, `1` declined at the prompt or a file could not be
deleted, `2` usage error / unknown id / no config.

## `gmlx list`

Aliased as `gmlx ls`. List the models your server config defines: the ids
you can actually address in a request, not a directory of files. It reads the
config (the bare-start search path, or `--config FILE`) and prints each model
with its source path, flags (`vlm`, `mtp`, `lora`, `stream-cpu`/`stream-experts`,
`pinned`), and profile. The default model is marked `*`. Both the explicit
`models:` entries and anything a `discover:` scan would add are shown
(discovered ones tagged `[discovered]`), followed by the `aliases:` table and
the default model: the same set `serve` exposes at `/v1/models`, viewable
offline. To list GGUF files on disk, use your shell; `gmlx init` /
`sync-models` fold them into a config.

```sh
gmlx list                            # the bare-start config (e.g. ~/.config/gmlx/gmlx.yaml)
gmlx list --config ./gmlx.yaml
gmlx list --json
```

| Flag | Meaning |
|------|---------|
| `--config FILE` | Config to read (default: the bare-start search path). |
| `--json` | Emit `{config, models, aliases, default_model}` as JSON instead of the table. |

Exit code `2` if no config is found (or an explicit `--config` path is
missing); the message points you at `gmlx init`.

## `gmlx ps`

Show the models resident in a running gmlx server: it reads the server's
`GET /v1/metrics` snapshot and tables each
resident entry (ids, footprint, idle time, TTL, pinned) with the model path
on a second line. The target defaults like `status`/`stop`: the single
managed server if there's one, else the config's host/port, else
`http://127.0.0.1:8080`; the output names the server it probed. `/v1/metrics` requires the API key when the server runs
with one; pass the same key via `--api-key` (or the `GMLX_API_KEY` env
var). On a `401`, `ps` prints exactly that hint. Exit code `1` if no server
is reachable.

```sh
gmlx ps
gmlx ps --url http://192.168.4.20:8080 --api-key sk-local-...
gmlx ps --json
```

| Flag | Meaning |
|------|---------|
| `--url URL` | Server base URL (default: resolved as above; a trailing `/v1` is stripped). |
| `--host H` / `--port P` | Managed-server target (alternative to `--url`). |
| `--api-key KEY` | API key for a key-protected server (default: the `GMLX_API_KEY` env var). |
| `--json` | Emit the resident-model list as JSON instead of the table. |

## `gmlx profiles`

Show the built-in per-family sampling table (each family's model-card
defaults plus its addressable `@intents`), followed by your config's user
profiles (flagging any that shadow a built-in) and each configured model's
detected or declared family. With a model id (or alias), print that model's
fully resolved sampling for its base and every addressable profile, along
with the config layers that shaped the merge (rule, model profile, per-model
tweaks, overrides). Read-only. The table form works with no config at all.
See
[server-config.md](server-config.md#profiles-sampling-profiles-and-built-in-intents).

```sh
gmlx profiles                        # the family table + user profiles + model families
gmlx profiles qwen3.6-27b            # one model, resolved per intent
gmlx profiles qwen3.6-27b --json
```

| Flag | Meaning |
|------|---------|
| `--config FILE` | Config to read (default: the bare-start search path). |
| `--json` | Emit the table / resolution as JSON instead of text. |

---

## `gmlx completion`

Print a shell completion script for `zsh`, `bash`, or `fish`.

```sh
# zsh: eval at shell start (add to ~/.zshrc):
eval "$(gmlx completion zsh)"
# ...or drop a function file onto your fpath:
mkdir -p ~/.zfunc && gmlx completion zsh > ~/.zfunc/_gmlx
# then, before `compinit` in ~/.zshrc:  fpath+=(~/.zfunc)

# bash: eval at shell start (add to ~/.bashrc):
eval "$(gmlx completion bash)"
# ...or, with the bash-completion package installed:
gmlx completion bash > ~/.local/share/bash-completion/completions/gmlx

# fish: source at shell start (add to ~/.config/fish/config.fish):
gmlx completion fish | source
# ...or drop it where fish autoloads completions:
gmlx completion fish > ~/.config/fish/completions/gmlx.fish
```

Completion is live, not baked: the emitted script is a thin shim that calls a
hidden `gmlx __complete` on every TAB, so candidates always match the
installed version and your config. It completes:

- verbs (`run`, `serve`, `launch`, ...) and the `ls` alias.
- flags for the verb being typed (read from that verb's own `--help`).
- model ids and aliases from your server config for `run` / `chat` / `serve`
  (the `--config FILE` already on the line is honoured, else the default
  config).
- harness names (plus `menubar`) for `launch`, and `install` / `uninstall` /
  `status` for `service`.
- live host / port / URL for `--host` / `--port` / `--url` / `--base-url`,
  read from the servers you've backgrounded, so completing `stop --port
  <TAB>` offers the port of the running server.
- file paths (via the shell's own path completion) for path-valued flags and
  positionals.

No regeneration is needed after an upgrade; re-running the script only
matters if you move where it's installed.

---

## Environment variables

When the same setting is reachable more than one way, the precedence is
**flag > config key > environment variable** - a flag you typed always wins,
and a config key beats an exported env var. The named exception:
`GMLX_CACHE_LIMIT_GB` wins over the `server.cache_limit_gb` config key, so a
benchmark run can pin the MLX buffer-cache limit without editing the config
(see [server-config.md](server-config.md)).

This is the supported set:

| Variable | Meaning |
|----------|---------|
| `GMLX_STREAM_GPU_TOKENS` | `--stream-experts` prefill staging threshold: offloaded expert calls with at least this many tokens run on the GPU stream (same zero-copy buffers; prefill is a GEMM workload the CPU loses badly). Default `32`; `0` keeps every expert call on CPU (conservative for models far larger than RAM). |
| `GMLX_STREAM_PREFETCH=0` | Disable sequential expert prefetch for streaming-mode (over-wired-budget) `--stream-cpu` / `--stream-experts` models. Default on: prefill-sized expert calls advise the kernel (`F_RDADVISE`) two layers ahead and pace the lazy graph per layer, reading expert stacks at sequential bandwidth instead of demand-faulting. |
| `GMLX_DECODE_ARENA_GB` | Decode-feeder arena size override in GB (see `--decode-feeder`). Default: sized from the GPU working-set budget, a physical-RAM fraction (`GMLX_DECODE_ARENA_RAM_FRAC`, default `0.6`), and the memory reclaimable at load, minus the non-expert weights and a KV reserve (`GMLX_DECODE_KV_RESERVE_GB`, default `8`). |
| `GMLX_ARENA_STAGE_MAX_TOKENS` | Largest expert call served router-aware (decode-feeder arena or partial ring staging) instead of whole-layer staging. Default `64`; above it a chunk routes to nearly every expert anyway. |
| `GMLX_DECODE_PRESSURE` | Set `0` to keep the decode-feeder arena at its sized capacity regardless of system memory pressure. Default on: the arena shrinks (keeping its most popular experts) when the kernel reports pressure and regrows once pressure clears and reclaimable RAM returns. |
| `GMLX_DECODE_LOOKAHEAD=0` | Disable lookahead expert prestage on the decode feeder. Default on: each MoE layer runs the *next* MoE layer's router on its own input (the residual moves little between adjacent sublayers, so recall is far above previous-token reuse) and pre-reads the predicted arena misses while the current layer computes. Lossless - predictions move bytes, never routing. `GMLX_DECODE_LOOKAHEAD_K` caps the ranked predictions considered per call (default `6`); `GMLX_DECODE_LOOKAHEAD_WORKERS` sizes the dedicated read pool (default `6`); `GMLX_DECODE_LOOKAHEAD_NORM` picks the prediction input (`ratio` default, `raw` skips the norm-gain rescale); `GMLX_DECODE_LOOKAHEAD_MIN_P` sets the per-rank reliability floor below which a prediction rank stops being submitted (default `0.5`); `GMLX_DECODE_LOOKAHEAD_CANCEL=0` keeps unrouted predictions reading to completion instead of cancelling the unstarted ones at settle; `GMLX_DECODE_LOOKAHEAD_IOPOL=0` runs the read pool at default disk-I/O priority instead of the utility tier. |
| `GMLX_DECODE_LOOKAHEAD_PROBE` | Lossless recall probe for the lookahead predictor: records predicted-vs-actual routing per layer (plus a previous-token baseline) and prints a table at exit, issuing no reads. Worth a run on a new model family to see whether the prestage pays there. |
| `GMLX_DECODE_PAGECACHE_GB` | Page-cache reserve inside the decode-arena RAM floor (default `2.5`). Buffered read paths (prefill feeder, CPU-mmap fallback) collapse when the cache is starved; the floor also clamps an oversized `GMLX_DECODE_ARENA_GB` (`GMLX_DECODE_ARENA_FORCE=1` restores the unclamped override). |
| `GMLX_CACHE_LIMIT_GB` | MLX buffer-cache limit for `serve`, in GiB (wins over `server.cache_limit_gb`). Negative or `off`/`none`/`unlimited` forces an unbounded cache; `0` disables buffer caching. See [performance.md](performance.md). |
| `GMLX_NATIVE_FP` | Layout for MXFP4/NVFP4 expert tensors: `wire` (zero-copy GGUF wire bytes, loads in seconds), `packed` (eager repack into MLX's layout), or the default `auto` (wire when a streaming placement is requested or the file nears the wired budget). See [streaming.md](streaming.md). |
| `GMLX_ROPE_FACTORS=0` | Expert escape hatch for rope scaling: disables the `rope_freqs` factors patch that rebuilds Llama-3.1-style per-dim rope scaling from GGUF metadata. Set `0` only to rule the patch out when debugging long-context degradation. |
| `GMLX_FUSED_GDN=0` | Disable the fused gated-delta Metal kernels used by the Qwen3.5/3.6 hybrid architectures. The fusion is a numerics-affecting runtime patch; set `0` first when debugging those archs to rule it out. |
| `GMLX_NO_FAMILY_DEFAULTS` | Disable the family model-card sampling defaults on bare-path `run` / `chat` (same as `--no-family-defaults`). |
| `GMLX_DRAFT_BLOCK_SIZE` | MTP draft tokens per round for `serve` (same as `--draft-block-size`). |
| `GMLX_IGNORE_EOS=1` | `serve`: never stop on EOS; decode every request to `max_tokens` (same as `--ignore-eos`; forced-length throughput benchmarking). |
| `GMLX_API_KEY` | Client-side default key for `ps` (sent to `/v1/metrics`) when `--api-key` isn't passed. Not a `serve` source; the server reads its key only from `server.api_key` in the config. |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | Hugging Face auth for `validate` / `pull` on gated or private repos. |
| `XDG_CACHE_HOME` | Where `chat` keeps its prompt history (`$XDG_CACHE_HOME/gmlx/chat_history[.ptk]`), and where backgrounded servers keep their runfiles and logs (`$XDG_CACHE_HOME/gmlx/`). |

`PREFILL_STEP_SIZE` (no `GMLX_` prefix) is upstream mlx-vlm's own variable -
the serve prefill chunk size; prefer `--prefill-step-size` or
`server.prefill_step_size`. Server-side switches for the speculative prompt
cache (`GMLX_SPEC_APC*`) and logprobs (`TOP_LOGPROBS_K`) are documented in
the [server config reference](server-config.md).

Anything not in this table (or in the server config reference's tables) is
internal and unstable - it may change meaning or disappear between releases.

---

See also: the [server config reference](server-config.md) (config surface,
start modes, endpoints, architecture diagrams) and the
[serving architecture deep-dive](serving-architecture.md) (the mlx-vlm
adoption mechanics).
