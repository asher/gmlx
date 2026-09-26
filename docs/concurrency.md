# Concurrent requests

The gmlx server decodes the requests of several clients together. This
page explains what batching gains, how the server admits a new request
without stalling the others, and how it speeds up requests that share a
prompt.

- [Batched decoding](#batched-decoding)
- [Admitting a new request](#admitting-a-new-request)
- [Shared prompts](#shared-prompts)

## Batched decoding

The server decodes all active requests of a model as one batch. Decoding
is limited by memory bandwidth, and a batched step reads the weights once
for all requests. Total throughput therefore rises with the number of
clients, and each request slows by less than its share. The gain falls as
the contexts grow, because each request does its own attention work.
[Benchmarks](benchmarks.md) has measurements.

A model with speculative decoding speculates only while the batch is
narrow, as [Several requests at once](speculative-decoding.md#several-requests-at-once)
describes. A model that streams from disk should get one request at a
time, as [Serving a streamed model](streaming.md#serving-a-streamed-model)
explains.

## Admitting a new request

The prompt of a new request must prefill while other requests are
decoding. Prefill runs in chunks of 2048 tokens, and at a deep context one
chunk can take as much GPU time as hundreds of decode steps. A scheduler
that alternates one decode step with one chunk therefore lets a long
prompt stall the streams that are already running. When chunks are short,
slowing admission only delays the new request and narrows the batch. Two
settings handle the two cases:

| Symptom | Setting | Default | Effect |
|---------|---------|---------|--------|
| The running streams stall while a long prompt arrives. | [`server.decode_prefill_ratio`](config.md#serverdecode_prefill_ratio) | `auto` | Slows admission only when a running stream would drop below half of its batched speed. A number fixes the ratio, and `0` alternates strictly. |
| A stream pauses during a long prefill chunk. | [`server.prefill_tick_ms`](config.md#serverprefill_tick_ms) | `500` | Halves each chunk until its expected time fits the budget. `0` suits batch jobs where only total throughput matters. |

With pacing, a new request waits at most about twice as long for its
first token as it would without pacing. Prefill runs at full speed when
nothing is decoding, so one client alone sees no difference under any
setting. The server reads both settings at start, so change them and run
`gmlx restart`.

## Shared prompts

Requests often share the start of their prompt, such as a common system
prompt or histories restored from the [prompt cache](prompt-cache.md).
The server finds the shared part from the token ids of the requests, and
decodes such a batch with a cascade kernel that reads the shared part once
for the whole batch. The gain grows with the length of the shared part and
the number of requests. The cascade is exact and on by default, and
[`GMLX_CASCADE_SDPA=0`](env-vars.md#runtime) turns it off.
