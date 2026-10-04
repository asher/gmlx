# Capacity and metrics

A running server reports how much work it can take. A load balancer, or a
harness that fans out subagents, reads these routes to decide how many
requests to send and when to send them:

```sh
curl 'localhost:8080/v1/capacity/plan?width=4&depth=32768'   # can 4 streams of 32K tokens start now?
curl 'localhost:8080/health?ready=1'                          # can the server take a request now?
curl localhost:8080/v1/metrics                                # the full live snapshot
```

Every route except `/health` needs the API key when the server has one.

- [Plan a fan-out](#plan-a-fan-out)
- [Estimate one request](#estimate-one-request)
- [Readiness checks](#readiness-checks)
- [Context windows in the model list](#context-windows-in-the-model-list)
- [Live metrics](#live-metrics)
- [Prometheus](#prometheus)

## Plan a fan-out

`GET /v1/capacity/plan?width=W&depth=D` asks whether `W` streams of `D`
tokens each fit in memory, and whether they can start now. The values below
are examples:

```json
{"width": 4, "depth": 32768, "ok": true,
 "max_context_at_width": 65536, "max_width_at_depth": 8,
 "model": "qwen3.8-27b-ud-q6", "band": "green",
 "decode_batch": 8, "in_flight": 1, "waiting": 0, "slots": 7,
 "admit_now": true, "reason": "ok"}
```

- `ok` is the memory answer, from the capacity table the server builds at
  start. It is `null` without a table or with `GMLX_OVERCOMMIT=1`.
- `admit_now` is the timing answer. It needs the memory
  [governor](glossary.md#governor) below orange, nothing waiting, and `W`
  free decode slots. Under a yellow band, only one slot counts as free.
- `reason` names the first check that fails, such as `governor orange`,
  `2 waiting for a slot` or `3 free slot(s) of 8, need 4`.

## Estimate one request

`POST /v1/estimate` takes a chat-completions body and says whether it fits
on a resident model, without running it. `"dry_run": true` on
`/v1/chat/completions` returns the same estimate.

| Field | Meaning |
|-------|---------|
| `prompt_tokens`, `warm_tokens`, `cache_tier` | Prompt length, and how much of it the prompt cache already holds, on which tier. Route on this across machines. |
| `need_bytes` | Memory for prefill and the prompt's KV, plus `max_tokens` more when the body sets it. |
| `fits_now`, `fits_drained` | Whether `need_bytes` fits in free memory now, and once running requests finish. |
| `context_ok`, `context_limit` | Whether the prompt fits the context window, and that window. |
| `est_ttft_s` | Estimated time to the first token. |
| `resident` | `false` when the model is not loaded. The estimate never loads a model. |

A body with images, audio or video is rendered but not estimated. A dry run
that names a [served assistant](assistant.md#served-assistants) gets a 400.

## Readiness checks

`GET /health` answers `{"status": "healthy", "pid": N}` while the process
runs, with no API key. Add `?ready=1` for a load balancer: the server
answers 200 with `"ready": true`, or 503 with a `Retry-After` header and a
one-word `reason`:

| `reason` | Meaning |
|----------|---------|
| `pressure` | The memory governor is orange or red. |
| `queue` | Requests are waiting for a slot. |
| `busy` | Every model's engine is at its decode width. |

## Context windows in the model list

Each `GET /v1/models` entry carries two context figures. A harness sizes its
context window from the smaller of the two:

- `context_length`, the GGUF's trained window, or
  [`max_kv_size`](config.md#loadmax_kv_size) when that is smaller.
- `max_context_at_width_1`, how much of the window fits in memory for one
  stream. It is `null` except for the model the capacity table was built
  for, and then use `context_length`.

A resident model with [KV quantization](kv-quantization.md) adds a
`kv_quant` object. Its `verdict` is `full`, `partial` or `dropped`.

## Live metrics

`GET /v1/metrics` returns a snapshot of the server under the `server` key.
This is a trimmed example with example values. A real snapshot has more
sections and fields.

```json
{
  "server": {
    "concurrency": {"decode_batch": 8, "queue_cap": 16, "in_flight": 2, "waiting": 0},
    "queue": {"waiting": 0, "cap": 16, "eta_s": 0, "rejections": 0, "last_reject_reason": null},
    "governor": {"band": "green", "enabled": true},
    "memory": {"active_bytes": 24696061952, "cache_bytes": 1073741824, "headroom_bytes": 60129542144},
    "rates": {"decode_tok_s": 41.3, "decode_streams": 2, "prefill_tok_s_recent": 612.0,
              "decode_tok_s_recent": 21.8, "decode_tok_s_lifetime": 22.4},
    "requests": [
      {"id": "11f4c2d90", "uid": 3, "model": "qwen3.8-27b-ud-q6", "state": "decode",
       "prompt_tokens": 1834, "generated": 212, "max_tokens": 4096,
       "elapsed_s": 11.42, "ttft_s": 1.71, "decode_tok_s": 21.8,
       "cache": {"tier": "block", "warm_tokens": 1536}, "speculative": null}
    ]
  }
}
```

| Section | What it tells you |
|---------|-------------------|
| `concurrency` | Decode width, queue cap, streams generating now, and requests waiting. |
| `queue` | Waiting count, its cap, the `Retry-After` a rejected request would get now, and rejections so far. |
| `requests` | One row per request, queued rows first. `state` is `queued`, `prefill` or `decode`. |
| `resident_models` | Each loaded model, with its own `in_flight`. Each model decodes on its own engine, so compare it with `decode_batch`. |
| `governor` | The memory band, `green`, `yellow`, `orange` or `red`, and what the governor has done. |
| `memory` | Active and cached bytes, free working set, and for a [streamed](streaming.md) model its [arena](glossary.md#arena). |
| `capacity` | The capacity table that the plan route reads. |
| `rates` | Decode rate now, means over recent requests, and the lifetime mean. |

In a request row, `cache.tier` is where its prefix hit in the
[prompt cache](prompt-cache.md): `exact`, `block`, `ckpt`, `anchor` or
`miss`. `speculative` holds the [drafter's](glossary.md#drafter) rounds and
accept rate, or `null` without a drafter. A section whose probe fails
reports `null` fields, and the rest of the snapshot still answers.

## Prometheus

`/v1/metrics?format=prometheus` returns the same snapshot as Prometheus
text:

```sh
curl 'localhost:8080/v1/metrics?format=prometheus'
```

The sections become gauges such as `gmlx_concurrency_in_flight`,
`gmlx_queue_eta_s`, `gmlx_governor_band` with a `band` label, and
`gmlx_capacity_max_ctx` with a `width` label. Per-model series carry a
`model` label. `requests` contributes only its count.
