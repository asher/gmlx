# Plan: capacity-aware subagents for pi on gmlx

Status: draft for review. The server side is built (0.4.1, unreleased): sections 4.1, 4.2, 11.1 and 11.2 items 1, 2, 3 and 5. Leases (11.2 item 4) wait for evidence per that item. The pi extension itself (sections 5-9) is not started.

A pi extension that dispatches subagents against a gmlx server, sizing
the parallel fan-out from what the server says it can hold right now
(`/v1/metrics`: decode width, in-flight streams, queue depth, governor
band, and the depth-by-width capacity table) instead of a fixed pool.
Plus the small server-side additions that make the client's job
deterministic.

## Contents

1. [Goal and non-goals](#1-goal-and-non-goals)
2. [What exists today](#2-what-exists-today)
3. [Architecture](#3-architecture)
4. [Server-side changes in gmlx](#4-server-side-changes-in-gmlx)
5. [Extension design](#5-extension-design)
6. [Edge cases and decisions](#6-edge-cases-and-decisions)
7. [Observability](#7-observability)
8. [Testing](#8-testing)
9. [Phases](#9-phases)
10. [Open questions](#10-open-questions)

## 1. Goal and non-goals

**Goal.** When a pi session on a gmlx model calls the `subagent` tool
with N parallel tasks, the tasks start at a width the server can hold
without shedding, queueing toward a timeout, or degrading every stream;
tasks the server cannot take yet wait client-side and start as capacity
frees; the parent model and the user both see why the fan-out is the
width it is.

**Non-goals for v1.**

- Scheduling across multiple pi sessions or machines. One extension
  instance schedules its own children; other clients on the server are
  observed as load, not coordinated with.
- Choosing a model per task on quality grounds. Agent definitions name
  their model as today.
- Replacing pi's own retry, compaction, or tool machinery in the child.
- Priority or preemption. Children the server admits run to completion;
  the governor's shed policy is the only preemption and it stays
  server-side.

## 2. What exists today

### 2.1 pi (0.69.0, installed at `/opt/homebrew/lib/node_modules/@mariozechner/pi-coding-agent`)

**Reference subagent extension**
(`examples/extensions/subagent/`). The shape this plan forks:

- Agents are `~/.pi/agent/agents/*.md` (or project `.pi/agents/`) with
  frontmatter `name`, `description`, `tools`, `model`; the body becomes
  an appended system prompt (`agents.ts:52-67`).
- One tool, three modes: `single` (`agent` + `task`), `parallel`
  (`tasks[]`), `chain` (`chain[]` with a `{previous}` placeholder).
- Each task is a child process:
  `pi --mode json -p --no-session [--model X] [--tools a,b]
  --append-system-prompt <tmpfile> "Task: ..."` (`index.ts:265-301`).
  The extension parses the child's JSON event stream (`message_end`
  events carry messages and usage) and streams progress to the parent's
  tool-call renderer.
- Parallelism is a fixed pool: `MAX_CONCURRENCY = 4`,
  `MAX_PARALLEL_TASKS = 8` (`index.ts:27-28`), drained by
  `mapWithConcurrencyLimit` (`index.ts:190-208`). This is the part that
  becomes capacity-aware.
- Abort propagates from the parent's `signal` to `proc.kill()`.

**Extension API surface the dispatcher uses** (`docs/extensions.md`):

| Need | API |
|---|---|
| Find the server | `ctx.model` is a pi-ai `Model` with `provider`, `id`, `baseUrl`, `contextWindow`, `maxTokens`, `headers` (`pi-ai/dist/types.d.ts:346`). |
| Resolve another model | `ctx.modelRegistry.find(provider, id)`; `--model provider/id` on the child CLI. |
| Abort | `ctx.signal` during a tool call; `AbortSignal` into `fetch` and the child spawn. |
| Live status | `ctx.ui.setStatus(key, text)` (footer), `ctx.ui.setWidget(key, lines)` (above the editor). Both are no-ops in `-p`/JSON mode and RPC-forwarded in RPC mode. |
| User commands | `pi.registerCommand("capacity", ...)`. |
| Child isolation | `--no-session`, `--tools`, `--no-extensions`, `-e <path>`, `--append-system-prompt`, `--thinking`. |
| Packaging | `~/.pi/agent/extensions/<dir>/index.ts` auto-discovers; `settings.json` `extensions: [path]` points at a directory elsewhere; `pi install <path>` for a pi package (`docs/packages.md`). |

**pi's own retry** (`dist/core/agent-session.js:1919-1985`,
`docs/settings.md` "Retry"): on an assistant message whose error text
matches `/overloaded|...|429|500|502|503|504|service.?unavailable|
connection.?refused|.../i`, the session retries with exponential backoff
`baseDelayMs * 2^(attempt-1)` (defaults 2 s, 4 s, 8 s; `maxRetries` 3).
The `openai-completions` provider does **not** read `Retry-After`; only
the Gemini CLI provider does. Context-overflow errors are not retried
(compaction handles them). This matters twice below: a child hitting a
gmlx 503 or shed retries on its own schedule, and a child whose prompt
cannot fit fails permanently.

**What pi sends.** The `openai-completions` provider always sends
`max_tokens` (or `max_completion_tokens`) equal to the model's
`maxTokens` (`pi-ai/dist/providers/openai-completions.js:346-350`).
gmlx's `launch pi` writes only model ids into `~/.pi/agent/models.json`
(`gmlx/launch.py:352-375`), so every gmlx model gets pi's defaults:
`contextWindow` 128000, `maxTokens` 16384.

### 2.2 gmlx (0.4.0)

Everything a dispatcher needs is on the authed `GET /v1/metrics`,
assembled in `gmlx/server_patches/routes.py:230-283` under the `server`
key:

| Key | Contents | Source |
|---|---|---|
| `resident_models[]` | `ids`, `model_path`, `busy` (in-flight refcount, held from admission to the end of the token stream), `footprint_bytes`, `idle_s`, `ttl_s`, `pinned`, `kept` | `routes.py:197`, `residency.py:174` |
| `request_queue_depth` | requests waiting in the server queue | stock mlx-vlm snapshot (`mlx_vlm/server/app.py:161`) |
| `governor` | `band` (green/yellow/orange/red), `band_changed_at`, `yellow_entries`, `orange_evictions`, `orange_retires`, `red_failures`, `victim_repeats`, `sheds_suppressed`, `last_action`, `last_recovered_bytes`, `kernel_reclaimable_bytes`, `kernel_floor`, `kernel_floor_reds`, `enabled` | `governor.py:122-136, 205` |
| `capacity` | `weight_bytes`, `working_set_bytes`, `budget_bytes`, `reserve_bytes`, `max_buffer_length`, `resource_limit`, `trained_ctx`, `max_ctx` (by width 1/2/4/8/16/32), `max_width_at_depth` (at 4096/16384/65536), `overcommit` | `capacity.py:212-224` |
| `memory` | `active_bytes`, `cache_bytes`, `headroom_bytes` | `routes.py:244` |
| `residency` | `budget_bytes`, `resident_bytes` | `routes.py:235` |
| `admission` / `freshness` / `queue` | `deferrals`, `holds`, `rejections` with `last_*_reason` | the three gate modules |

`capacity` is only present for GGUF-served models with a derived table
(HF fall-through logs "no table" and omits the key). `governor.enabled`
is false under `GMLX_GOVERNOR=0`. `/health` is unauthenticated;
`/v1/metrics` needs the Bearer key when `server.api_key` is set and is
on the observability silent-path list (`observability.py:31`), so a 1 Hz
poll costs nothing in the request log.

**Not exposed today:** the effective decode width,
`decode_batch()` = `min(GMLX_DECODE_BATCH default 8, capacity frontier
width)` (`decode_batch.py`), and the queue cap, default 2 x decode width
(`queue_cap.py`). Section 4 adds them.

**Back-pressure on the wire** (what a child pi process sees):

| Condition | Response | pi child behavior |
|---|---|---|
| Waiting queue at cap | HTTP 503, JSON `{error: {message, queue_depth, queue_cap}}`, `Retry-After` 2-60 s (drain estimate) | retries 2/4/8 s ignoring `Retry-After`; fails after 3 |
| Prompt cannot fit with the batch drained | HTTP 400, message "request cannot fit: prompt [+ max_tokens N] needs an estimated X GB ... only Y GB remains ... (prompt_tokens=T)" (`mem_preflight.py:176`) | not retryable; child ends with the error |
| Governor sheds the row mid-stream | terminal SSE `data: {"error": {"type": "server_overloaded_shed", "code": "row_shed", "prompt_len", "delivered"}, "finish_reason": "shed"}` (`request_flow.py:176-190`); non-stream: HTTP 500 | message contains "overloaded" so pi retries 2/4/8 s; fails after 3 |
| Model not resident, cannot fit to load | HTTP 400 with the numbers (residency build gate) | not retryable |

Two 0.4.0 behaviors the dispatcher leans on:

- **Fresh gate** (`GMLX_APC_FRESH_WAIT_MS`, default 500): sibling
  requests that arrive together share one cold prefill of the common
  prefix; later ones start warm. Children share pi's base system prompt
  and any shared skills, so dispatching a batch as a burst is cheaper
  than trickling.
- **Cancel frees promptly**: a cancelled row in a speculative batch
  releases its memory immediately, so killing a child on abort returns
  its slot within a tick.

## 3. Architecture

```
parent pi session --(tool call: subagent)--> gmlx-subagent extension
                                                  |
                     +----------------------------+--------------------------+
                     | CapacityProbe              | Scheduler                 |
                     |  GET /v1/metrics (1 Hz     |  task queue -> admission   |
                     |  while tasks pending)      |  policy -> spawn children  |
                     |  -> Snapshot                |  -> collect results        |
                     +--------------+-------------+-------------+------------+
                                    |                            |
                              gmlx serve  <---- child pi (-p --mode json) x N
                              /v1/chat/completions
```

Three pure pieces and one effectful one:

- **Snapshot** (data): the subset of `/v1/metrics` the policy reads,
  plus a timestamp, plus "unknown" markers for every optional key.
- **Policy** (pure function): `(snapshot, running, pending, config) ->
  {allow: number, reason: string}`. Unit-tested without a server.
- **Scheduler** (state machine): owns the task queue, running set,
  retry ledger; calls Policy before each start; reacts to child
  events.
- **Probe** (I/O): fetches metrics, handles auth and failure, exposes
  the latest Snapshot with staleness.

The child process contract is unchanged from the reference extension,
with two additions: a depth guard env var and an optional per-child
context cap (section 5.6).

## 4. Server-side changes in gmlx

Small, independent of the extension, and useful to any client.

### 4.1 `concurrency` section on `/v1/metrics`

Add next to the other sections in `routes.py`:

```json
"concurrency": {
  "decode_batch": 8,
  "queue_cap": 16,
  "in_flight": 3,
  "waiting": 0
}
```

- `decode_batch` from `decode_batch.decode_batch()`; `queue_cap` from
  `queue_cap._cap()` (0 when disabled); `in_flight` = sum of resident
  `busy`; `waiting` = the queue-cap census `_waiting_depth()` (server
  queue plus unadmitted prompt candidates), which is the number the cap
  actually compares against and is a superset of `request_queue_depth`.
- Pure read; shape-guarded like the others (`try/except` so a missing
  engine never breaks the snapshot).
- Test: `tests/test_server_patches.py` gets a case asserting the keys
  and that `decode_batch` respects `GMLX_DECODE_BATCH`.

Without this, a client can only guess width from
`capacity.max_width_at_depth` and hope the default is 8.

### 4.2 Context window on `/v1/models` and in `launch pi`

`_entry()` in `routes.py:44-58` gains `context_length` (the GGUF's
trained context, what `capacity.trained_ctx` reads) and, when a table
exists, `max_context_at_width_1`. `launch pi` then writes
`contextWindow` and a sane `maxTokens` per model into `models.json`
instead of leaving pi's 128k/16k defaults. Two reasons:

- pi's compaction threshold keys off `contextWindow`; a 32k model
  advertised as 128k never compacts and overflows the server instead.
- pi always pins `max_tokens`, and gmlx's preflight prices a pinned
  `max_tokens` into the KV estimate (`mem_preflight.py:160-171`). A
  16384 default inflates every child's need by 16k tokens of KV. A
  `maxTokens` of 4096-8192 for subagent-class work is plenty and frees
  real headroom.

This is a `launch.py` change plus a `probe_models` field, with its
existing tests extended.

Built: `/v1/models` carries `context_length` and `max_context_at_width_1`;
`launch pi` writes `contextWindow` (the smaller of the two) and
`maxTokens` (a quarter of the window, capped at 8192, floor 1024).

### 4.3 Ship the extension with gmlx (phase 3)

The extension is TypeScript; gmlx is Python. Ship the sources as package
data under `gmlx/harness/pi/gmlx-subagent/` and have
`gmlx launch pi --subagents` (or a `launch pi` default, decided in
phase 3) add the directory to `~/.pi/agent/settings.json`
`extensions: [...]`, the same merge style `launch pi` already uses for
`models.json`. Agent `.md` files ship alongside and merge into
`~/.pi/agent/agents/` only when absent (never overwrite a user's edit).
`pi -e <dir>` works for trying it without installing.

## 5. Extension design

Directory: `~/.pi/agent/extensions/gmlx-subagent/` during development
(symlink into the gmlx checkout), `gmlx/harness/pi/gmlx-subagent/` once
shipped.

```
gmlx-subagent/
+-- index.ts        tool registration, renderers, /capacity command
+-- agents.ts       agent discovery (from the reference, + context_need)
+-- probe.ts        CapacityProbe
+-- policy.ts       pure admission policy
+-- scheduler.ts    queue, running set, retry ledger, child lifecycle
+-- child.ts        spawn + JSON event parsing (from the reference)
+-- README.md
+-- agents/         scout.md, planner.md, reviewer.md, worker.md
```

### 5.1 Tool schema

The reference `subagent` tool, unchanged in modes and parameters, plus:

| Field | Where | Meaning |
|---|---|---|
| `context_need` | agent frontmatter, optional | tokens the agent typically needs (prompt + working set). Default by role: scout 16k, planner 24k, reviewer 32k, worker 64k. |
| `timeout_s` | task item, optional | wall-clock cap per child; kill and report on expiry. Default none. |
| `max_tokens` | agent frontmatter, optional | per-child generation cap. pi has no per-process `maxTokens` flag, so v1 cannot apply it; it takes effect once the child-side companion (5.6) exists. Until then the model's `models.json` value applies (4.2). |

`MAX_PARALLEL_TASKS` stays as a sanity cap (raise to 16) so a runaway
parent cannot enqueue hundreds; the real bound is the policy.

### 5.2 Server discovery and auth

- The target server is derived from the model each task will use:
  `agent.model` if set (resolved via `ctx.modelRegistry.find`),
  otherwise `ctx.model`. Root = `baseUrl` with a trailing `/v1`
  stripped.
- Tasks are grouped by root. Each root gets its own Probe and Policy
  state; a root whose provider `api` is not `openai-completions` or
  whose `/v1/metrics` does not return a gmlx shape (no `server` key)
  is a **foreign server** and falls back to the static pool
  (`MAX_CONCURRENCY` 4) with a one-line notice in the tool result.
- Bearer key: the provider's `apiKey` from `models.json` via the model
  registry (this is what `launch pi` wrote; a placeholder equal to the
  provider id when the server has no key, which gmlx ignores). If pi
  exposes command-resolved keys through the registry, use that path so
  `!cmd` keys work; verify during phase 2.
- 401 from `/v1/metrics` means "up, key required": treat as foreign
  (static pool) and say so in the notice, never as down.

### 5.3 CapacityProbe

- Polls `/v1/metrics` at 1 Hz only while the tool has pending or
  running tasks; idle otherwise. Timeout 1.5 s, `AbortSignal` from the
  tool call.
- Produces a Snapshot:

```ts
interface Snapshot {
  at: number;                         // monotonic ms
  reachable: boolean;
  width: number | null;               // concurrency.decode_batch, else derived
  queueCap: number | null;
  inFlight: number;                   // concurrency.in_flight, else sum  busy
  waiting: number;                    // concurrency.waiting, else request_queue_depth
  band: "green"|"yellow"|"orange"|"red"|"unknown";
  governorEnabled: boolean;
  maxCtxByWidth: Record<number, number> | null;   // capacity.max_ctx
  overcommit: boolean;
  headroomBytes: number | null;
  residentIds: Set<string>;
  residencyFreeBytes: number | null;  // residency.budget - resident
  counters: { rejections, deferrals, holds, redFailures, orangeRetires };
}
```

- Width derivation when `concurrency` is absent (server older than 4.1):
  `min(8, capacity.max_width_at_depth[16384] || 8)`; mark
  `widthDerived: true` so the UI can say "assumed".
- Staleness: a snapshot older than 3 s is treated as `band: "unknown"`
  and `inFlight` unknown; the policy then admits at most one new child
  per 3 s (slow start) rather than freezing.

### 5.4 Policy

Evaluated before every child start. Inputs: Snapshot, the scheduler's
`running` (children spawned, whether or not the server has admitted
them yet) and `pendingStarts` (spawned in the last grace window), the
task's `context_need`, and config. Output: `allow` (how many more may
start now) and a `reason` string for the UI.

```
if !reachable                      -> allow 0, "server unreachable"
if band in {orange, red}           -> allow 0, "governor <band>"
if waiting > 0                     -> allow 0, "server queue non-empty"

slots   = width - inFlight - pendingStarts       (foreign load counts; see 6.3)
if band == yellow                  -> slots = min(slots, 1)
if band == unknown (stale)         -> slots = min(slots, 1) and rate-limit 1 per 3 s

need    = context_need + maxTokens(child) + promptTokens(task)   (see 5.6 for maxTokens)
depthW  = largest w in maxCtxByWidth with maxCtxByWidth[w] >= need
          (skip when overcommit or no table)
allow   = max(0, min(slots, depthW - runningOnThisRoot))
```

Notes:

- `depthW` is the term a fixed pool cannot express: on a near-capacity
  model, eight 64k-context workers do not fit even when eight slots are
  free, and the table says so before a shed does.
- `pendingStarts` closes the gap between spawning a child and its first
  request landing (pi startup is ~1-2 s, longer with many skills). A
  spawned child counts as occupying a slot until the child's first
  `message_start` for an assistant turn arrives or 10 s pass, whichever
  is first. Without this, a burst of eight starts would all see
  `inFlight: 0`.
- Config overrides (`~/.pi/agent/gmlx-subagent.json`, all optional):
  `maxConcurrency` (hard cap on top of the policy), `minBand`
  (`green` default; `yellow` lets yellow admit freely), `reserveSlots`
  (slots to leave for other clients, default 0), `staticPool` (force the
  reference behavior). Environment `GMLX_SUBAGENT_STATIC=1` mirrors
  `staticPool` for a quick escape hatch.

### 5.5 Scheduler

- Parallel mode: tasks enter a FIFO queue. A loop runs: probe snapshot
  -> policy -> start up to `allow` tasks **at once** (burst, for the
  fresh gate) -> wait for the earlier of (a child event that changes
  state, the next probe tick) -> repeat. Terminates when the queue is
  empty and nothing is running, or on abort.
- Chain mode: one step at a time, but each step still passes the
  policy gate (a chain step launched into a red server is still a bad
  idea). `{previous}` substitution as in the reference.
- Single mode: same as a one-element parallel.
- Retry ledger, per task: `attempts`, `lastError`, `classification`.
  Classification from the child's final assistant `errorMessage` and
  exit code:

| Class | Match | Action |
|---|---|---|
| `capacity-transient` | 503 body (`queue_cap` key), `server_overloaded_shed` / `row_shed`, connection refused while `/health` was up within 10 s | requeue at the head; do not restart until the policy allows again **and** at least `Retry-After` (from the 503 body's drain estimate, default 5 s) has elapsed; max 2 requeues, then report |
| `capacity-permanent` | 400 "request cannot fit" | do not retry; report with the server's numbers and a hint (smaller task, smaller `context_need`, lower `maxTokens`) |
| `server-down` | `/health` unreachable for > 15 s | fail the remaining queue with one message; running children are left to pi's own retry until they exit |
| `task-error` | anything else non-zero | report; no retry (same as the reference) |

- The child retries 503/shed on its own (2/4/8 s) before it ever exits,
  so the extension's requeue is the second line, not the first. This
  is acceptable in v1; section 6.6 discusses tightening it.
- Abort: `ctx.signal` -> SIGTERM every child, SIGKILL after 5 s, mark
  remaining tasks `aborted`. gmlx frees each cancelled row on
  disconnect.
- Timeout: `timeout_s` per task -> same kill path, class `timeout`.

### 5.6 Child process contract

Same invocation as the reference, plus:

- `PI_SUBAGENT_DEPTH=<n+1>` in the child environment. The extension
  reads it at load: at depth >= 1 it registers the tool but the policy
  refuses with "nested subagents disabled (depth N)" unless the config
  sets `maxDepth` > 1. This blocks accidental recursion when the child
  discovers the same extension directory (it does by default; the
  reference has the same exposure). `--no-extensions` is the blunt
  alternative and is offered as a config switch, at the cost of the
  child losing every other extension.
- `--thinking` passthrough from agent frontmatter (`thinking: off` for
  scouts is a real saving on a thinking model; gmlx maps it per model).
- Optional child-side companion (phase 4): a tiny extension loaded with
  `-e` that reads `GMLX_CHILD_CTX_CAP` and `GMLX_CHILD_MAX_TOKENS`, sets
  the model's `contextWindow`/`maxTokens` for that process via
  `ctx.modelRegistry` override, and compacts early. Until it exists,
  `maxTokens(child)` in the policy is the model's configured value from
  `models.json` (section 4.2 makes that sane), and children rely on
  their own compaction.

### 5.7 Tool result and renderers

- The final tool result keeps the reference layout (per-task output,
  usage, stop reasons) and adds a one-line **capacity summary** at the
  top of `content` so the parent model learns the shape of what
  happened: `gmlx: width 8, admitted 3/5 immediately, 2 waited (yellow
  14 s), 1 requeued after shed, 0 failed`. Models reason better about
  "the server was constrained" than about a silent wait.
- `details` carries the structured version (per-task timeline:
  queued-at, started-at, first-token-at, ended-at, class, attempts) for
  the renderer and for tests.
- Renderer: the reference's collapsed/expanded views, with a queued
  state (`(waiting) waiting: governor yellow`) alongside running/done/failed.

### 5.8 UI outside the tool call

- `ctx.ui.setStatus("gmlx", "gmlx * green - 3/8 - 2 waiting")` while
  the tool runs; cleared on completion. Glyph by band.
- `ctx.ui.setWidget("gmlx", [...])` listing running agents with
  elapsed time and the reason any are waiting.
- `/capacity` command: prints the snapshot in human units (band,
  width, in-flight, waiting, headroom, `max_ctx` by width, resident
  models) and the current config; works when idle too.

## 6. Edge cases and decisions

### 6.1 No capacity table (`capacity` absent)

HF fall-through loads and any future path without a header-derived
table. Decision: policy skips the depth term and uses width only
(`concurrency.decode_batch` or the derived default). UI says "no
capacity table; width only".

### 6.2 Overcommit (`capacity.overcommit: true`)

The user chose to run past the ceilings deliberately (over-RAM decode
program). Decision: same as 6.1, width only; the governor band still
gates.

### 6.3 Other clients on the same server

The menu bar, a second pi session, Open WebUI, a `gmlx chat` all show
up in `busy`. Decision: never assume ownership. `slots = width -
inFlight` counts them, `reserveSlots` lets a user leave room on
purpose, and the tool never cancels or evicts anything. Consequence: if
another client holds every slot, the fan-out waits; the status line
says "8/8 in flight (not ours)".

### 6.4 The parent's own stream

At tool-execution time the parent is not generating (its `busy` hold
was released when its assistant turn ended), so no slot is reserved for
it. When the tool returns, the parent's next turn needs one slot; by
then every child has exited, so it is free. Chain mode holds the same
property between steps.

### 6.5 Startup race

Covered by `pendingStarts` (5.4). Additional guard: if the number of
`busy` never rises after a burst within 10 s while children are alive,
assume the children are stuck before their first request (auth prompt,
extension error) and surface their stderr rather than starting more.

### 6.6 Governor band changes while children run

- green -> yellow: stop new starts (or 1 at a time per `minBand`).
  Running children continue; the server throttles them.
- -> orange: no new starts. Server may evict caches or retire a row
  (APC replay makes a retire a warm re-prefill for that child; the
  child sees a longer time-to-first-token, nothing else).
- -> red: no new starts. The server may shed the largest row; that
  child sees `finish_reason: shed`, pi retries it 2/4/8 s. Those retries
  land while the server is still red, which is wasteful but bounded
  (three attempts, prefix warm via APC). Decision for v1: accept.
  Tightening option for later: run children with `retry.enabled:
  false` in a child-specific settings overlay so the extension owns
  every retry and can wait for green. pi has no per-process settings
  flag today; this needs either a companion extension (5.6) or an
  upstream flag.
- Band flapping: the governor has its own dwell (`GMLX_GOV_MIN_DWELL_S`).
  The scheduler adds hysteresis: after leaving green, require 2
  consecutive green snapshots before bursting more than one start.

### 6.7 Queue cap hit anyway

The policy never starts a child while `waiting > 0`, so in steady state
the cap is not reached by our children. It can still be hit by a burst
that races another client. The child sees 503 and retries on its own
schedule (2/4/8 s), ignoring the drain estimate. Decision: v1 accepts
pi's schedule; the extension reads the 503 body's `Retry-After` from the
child's error text when it eventually fails and uses it for the requeue
delay. Filed as a question for upstream: honor `Retry-After` in the
`openai-completions` provider.

### 6.8 Prompt cannot fit (400 preflight)

Not retryable and retrying does not help: the estimate already assumes
the batch drained. Decision: report with the numbers and a hint. If the
task's `promptTokens` estimate was far below what the server measured
(`prompt_tokens=T` in the message), record the ratio so the next
estimate is corrected within the same tool call.

### 6.9 `context_need` estimation

Frontmatter value plus an estimate of the task text (bytes / 4) plus a
fixed 6k allowance for pi's base system prompt and skills (measuring it
would itself cost a request; the correction in 6.8 refines it from the
server's measured `prompt_tokens`). Under-estimates cost a preflight
400 or a shed; over-estimates cost parallelism.
Decision: lean high (defaults in 5.1 are generous) and log the measured
`prompt_tokens` per task in `details` so users can tune frontmatter.

### 6.10 Child needs a model that is not resident

Loading a second model competes for the residency budget and takes the
same headroom check a request takes. Decision: a task whose model is
not in `residentIds` is held until the root has `inFlight == 0` and
`band == green`, then started alone; once the model shows resident,
normal scheduling resumes. If `residencyFreeBytes` is known and smaller
than the model's file size (from `/v1/models` `created`/size when
available, else unknown), report instead of holding forever. The
tool-result summary names the hold reason.

### 6.11 Same task, different roots

Tasks may target different gmlx servers (a second Mac). Each root has
its own probe and policy; the scheduler interleaves. A foreign root
(not gmlx) uses the static pool. No cross-root balancing in v1.

### 6.12 Server restarts or dies mid-run

Children see connection errors and retry on their own; the probe sees
`/health` down. Decision (5.5 `server-down`): after 15 s unreachable,
fail the remaining queue with one message and let running children
exhaust their own retries. When `/health` returns within the window,
continue normally; resident models may have changed, so `residentIds`
is re-read before the next start (6.10 applies).

### 6.13 Metrics shape drift

Every key is optional in the Probe; a missing key degrades to the
matching fallback (width derived, band unknown, table absent). A
non-gmlx JSON body (something else on the port) is foreign. The Probe
never throws into the scheduler.

### 6.14 Abort during a burst

Children spawned but not yet admitted are killed like the rest; the
server sees a disconnect before or after admission and frees either
way. The tool result marks them `aborted` with no retry.

### 6.15 Depth guard vs. legitimate nesting

`maxDepth` config (default 1). A planner that wants to run scouts is a
real pattern; at depth 2 the grandchildren share the same server and
the same policy, so the math still holds, but the parent's `running`
count does not see them. Decision: allow with config, count them as
foreign load (they show up in `busy`), document the caveat.

### 6.16 Speculative decoding and MTP

Under pressure the governor shrinks speculative width; nothing for the
client to do. Acceptance-rate drops with batch width are a throughput
concern, not a capacity one, and out of scope.

### 6.17 Print/JSON/RPC parents

The parent itself may be headless (`-p`, RPC). `ctx.ui.*` are no-ops or
forwarded; the tool result carries everything a headless caller needs.
No behavior branches on `ctx.hasUI`.

## 7. Observability

- Every scheduling decision logs one line at debug level through the
  tool's `onUpdate` details and to `~/.pi/agent/gmlx-subagent.log` when
  `GMLX_SUBAGENT_LOG=1`: timestamp, snapshot summary, allow, reason,
  task id. Enough to reconstruct a run.
- `details` in the tool result is the machine-readable timeline
  (5.7); tests assert on it.
- Server side, nothing new: the existing counters (`queue.rejections`,
  `governor.red_failures`, `freshness.holds`) diffed across a run tell
  whether the dispatcher caused pressure. The e2e test asserts they
  stay flat.

## 8. Testing

**gmlx (pytest).**

- `concurrency` section present, keys typed, `GMLX_DECODE_BATCH`
  respected, absent engine tolerated.
- `/v1/models` `context_length` populated for a GGUF entry.
- `launch pi` writes `contextWindow`/`maxTokens` and preserves an
  existing user value (merge style, never clobber).

**Extension (vitest or node:test, no server).**

- Policy table tests: a matrix over band x waiting x inFlight x table
  presence x `context_need`, asserting `allow` and `reason`. This is
  the piece most worth pinning down.
- Scheduler with a fake probe and fake child factory: burst size,
  `pendingStarts` accounting, requeue ordering, retry caps,
  abort/timeout kill paths, chain gating, depth guard.
- Error classification from real captured error strings (the 503 body,
  the shed SSE, the preflight 400 message, a connection refused).

**End to end (manual first, scripted under `bench/` or `tests/e2e/`).**

- Small model, `GMLX_DECODE_BATCH=2`, six scout tasks: assert never
  more than two `busy` from our children, zero `queue.rejections`, all
  six complete, tool summary reports waits.
- Force yellow with `GMLX_GOV_FORCE=yellow@...` (the governor's force
  hook, `governor.py:152`): assert one-at-a-time admission.
- Force red mid-run: assert no new starts, a shed child requeues once
  and completes after green.
- Kill the server mid-run: assert the `server-down` path and a clean
  tool result.

## 9. Phases

1. **Server**: 4.1 `concurrency` on metrics; 4.2 `context_length` on
   `/v1/models` and `launch pi` writing `contextWindow`/`maxTokens`.
   Small PR, tests included. Unblocks the client without guesswork.
2. **Extension v1**: fork the reference; Probe, Policy, Scheduler with
   parallel/chain/single; retry ledger; status line and `/capacity`;
   depth guard. Developed as a symlinked user extension against a live
   server. Unit tests for Policy and Scheduler.
3. **Ship**: package data + `launch pi` merge; agents installed when
   absent; README; e2e script.
4. **Refinements** (each optional, driven by what phase 2 shows):
   child-side companion for per-process context cap and retry
   ownership; upstream `Retry-After` in pi's `openai-completions`
   provider; a `/v1/capacity/plan` endpoint that answers "can I run w
   streams at depth d right now" so the policy math moves server-side
   and other harnesses get it for free.

## 10. Open questions

1. **Retry ownership.** Is it better to let children keep pi's own
   retry (simple, bounded, but blind to the band) or to take it over
   (needs a companion extension or an upstream per-process setting)?
   v1 says keep; revisit with data from the e2e runs.
2. **`launch pi` default.** Should `launch pi` install the extension by
   default once it ships, or behind `--subagents`? Installing an
   extension writes to the user's pi settings; the merge style already
   does that for `models.json`, so a default is defensible, but a tool
   that spawns processes deserves an opt-in the first time.
3. **Width for the depth term.** `max_ctx[w]` is the frontier for `w`
   equal streams at the same depth. Children have different depths;
   using the largest `context_need` among running plus the candidate
   is conservative. Is a per-stream cost model (sum of needs against
   `budget_bytes`) worth the extra client-side math, or is that the
   argument for the server-side plan endpoint in phase 4?
4. **`decode_batch` for multiple resident models.** Is the decode
   batch one engine-wide bound or per resident model's generator? The
   policy assumes engine-wide (conservative). Confirm in
   `residency.py` before phase 2.
5. **Keys via command.** Does pi's model registry expose a resolved key
   for `!cmd`-style `apiKey` entries to extensions? If not, the probe
   needs its own resolution or a documented limitation.

## 11. Server-side: closing the gaps against llama.cpp `/slots`, and going past it

Added 2026-08-25 after comparing `/v1/metrics` with llama.cpp's `/slots`
(`docs/server-config.md` documents the shipped surface).

### 11.1 Shipped (0.4.1, unreleased)

| Gap vs `/slots` | What landed |
|---|---|
| No live per-request view | `server.requests[]` on `/v1/metrics`: one row per queued / prefilling / decoding request with id, model, state, queue position, prompt and generated tokens, max_tokens, elapsed, TTFT, live decode tok/s, APC tier and warm tokens, speculative acceptance. Published from the engine tick, rate-limited to 4 Hz (`gmlx/live_requests.py`). |
| No keyless readiness probe | `GET /health?ready=1`: 200, or 503 + `Retry-After` with a one-word reason (`pressure` / `queue` / `busy`). Stays auth-exempt; adds nothing else to the liveness body. |
| No Prometheus export | `GET /metrics?format=prometheus` (or `Accept: text/plain` / OpenMetrics): the JSON snapshot flattened to `gmlx_*` gauges and counters with `model` / `width` / `depth` / `band` labels. |
| No explicit cache control | `POST /v1/cache/reset` now takes `{"model": id}` to clear one resident model's prefix cache, and with no body clears every resident model's (the stock handler reached only the request context's). |
| Deferred requests had no ETA | `server.queue.waiting`, `server.queue.cap`, `server.queue.eta_s` (the same drain estimate a 503 carries), and `position` on each queued row of `requests[]`. |
| Effective width not exposed | `server.concurrency.{decode_batch, queue_cap, in_flight, waiting}` (section 4.1 of this plan). |

Effect on this plan: the `pendingStarts` heuristic (5.4) can be replaced
by reading `requests[]` (a spawned child shows up as a queued or prefill
row within a tick of its first request); `width` no longer needs
deriving; the Probe can poll `/health?ready=1` keylessly for the coarse
signal and `/v1/metrics` for the numbers.

### 11.2 Going past `/slots` (items 1, 2, 3, 5 built; ordered by leverage)

`/slots` is a rectangle carved at boot (slot count x per-slot context).
gmlx's batch is a governed frontier. Each item below makes that frontier
queryable rather than only observable.

1. **Dry-run admission.** The memory preflight already tokenizes the
   prompt and prices its KV against the drained budget, then either
   admits or throws a 400. Expose the same computation as a query:
   `dry_run: true` on `/v1/chat/completions` (or `POST /v1/estimate`)
   returning `prompt_tokens`, `need_bytes`, `fits_now`, `fits_drained`,
   `warm_tokens` (how much of the prefix APC already holds) and
   `est_ttft_s`. A harness decides to compact before sending instead of
   after being refused, and the `context_need` estimation problem in
   section 6.9 disappears: ask the server. Reuses `mem_preflight.py`
   and `apc_lookup_plan`; the highest-leverage item on this list.
2. **Capacity plan endpoint.** `GET /v1/capacity/plan?width=w&depth=d`
   -> `{ok, max_width_at_depth, max_depth_at_width, band}`. The policy
   function of section 5.4 run where the numbers live. Once it exists
   the pi extension is a loop around it, and llama.cpp could answer the
   same route from `n_ctx` / `np` in a few lines, which is what makes it
   a shared contract rather than a gmlx feature.
3. **Locality as a signal.** `warm_tokens` in the dry-run and in
   `requests[]` tells a client which server holds its prefix and how
   much of it. Across two Macs that is the routing key; on one Mac it
   tells the parent that a child costs a 2k prefill, not a 20k one.
   llama.cpp routes to the warm slot internally and tells nobody.
4. **Leases.** A burst of siblings races itself. A dry-run that
   optionally reserves (`reserve: true` -> `lease_id`, TTL a few
   seconds, consumed by the real request carrying it) closes the race
   without client-side accounting. The one item that is new machinery
   rather than exposure; build it only if the e2e runs in section 8
   show the race biting.
5. **Rates, not just counts.** `requests[].decode_tok_s` is live now;
   an aggregate decode rate on the snapshot and `est_ttft_s` from the
   dry-run let a client compute its own ETA and let the menu bar show
   "8 streams, 71 tok/s aggregate".

Suggested order: 1, then 2 (they share the preflight pricing), 3 comes
with 1 for free, 5 is small, 4 waits for evidence.

Built (0.4.1, unreleased): `POST /v1/estimate` and `dry_run: true` on chat
completions (item 1, with `warm_tokens` / `cache_tier` for item 3),
`GET /v1/capacity/plan?width=W&depth=D` (item 2, returning `ok`,
`max_context_at_width`, `max_width_at_depth`, `band`, `slots`,
`admit_now`, `reason`), and `server.rates` on `/v1/metrics` plus
`est_ttft_s` on the estimate (item 5). All in `gmlx/estimate.py`; the
routes in `gmlx/server_patches/capacity_routes.py`. Leases (item 4) are
not built: the multi-model e2e (`tests/e2e/run_capacity_multi_e2e.py`)
shows bursts of eight overshooting the cap by a few requests without a
503, which the racy census already tolerates by design; build leases
when a run shows sibling starts actually being refused.
