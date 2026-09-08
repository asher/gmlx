# Prompt cache internals

How the prompt cache tiers are chosen per architecture, the counters that
show whether reuse is working, and the environment switches that tune or
disable each layer. Operators read [performance.md](../performance.md); this
page is for contributors.

### Which tier serves which architecture

APC routes each model by its cache shape, once, at load; the server logs
the routing (`APC tier: ...`) so a silent mis-route is visible. Reuse
works on every family - the tiers differ in storage layout, not in
whether hits happen:

| Cache shape | Example archs | Tier |
|-------------|---------------|------|
| plain KV, dense or MoE | llama, qwen2/3, qwen3moe, glm4(-moe), deepseek2/v3, phi3, granite, hunyuan, minimax-m2, gemma2 | block |
| hybrid GDN (recurrent + KV layers) | qwen3.5/3.6 (incl. MoE), qwen3-next, kimi-k3, nemotron-h, granitemoehybrid | ckpt |
| sliding-window attention | gemma3, gemma-4 (incl. E4B/3n), gpt-oss, SWA llama | ckpt |
| CacheList / pure recurrent | falcon-h1, mamba2, rwkv7, plamo2, deepseek-v3.2, deepseek4 | exact |
| MSA indexer armed | minimax-m3 indexer GGUFs | none (loud log; the indexless GGUF serves via the block tier) |

### Checkpoint-tier counters

`GET /v1/cache/stats` carries the `ckpt_*` fields for every model, but
they only move on checkpoint-tier architectures (the hybrid/SWA rows
above) -- all-zero on a block- or exact-tier model is normal, not a
fault. On ckpt-tier models they answer one question: is prefix reuse
working? Read reuse health from `ckpt_*`, not from ratios built on the
stock fields: checkpoint lookups bump the shared `lookups_hit` /
`matched_tokens` on success but record nothing on a miss, and the
token totals include window snapshots that can never be shared, so
aggregate hit rates skew on these models. (`disk_writes` counts write
operations: one per exact-format entry -- checkpoint skeletons and
drafter sidecars included -- and one per block for block shards.)

| Field | What it tells you |
|-------|-------------------|
| `ckpt_stores` | Prefixes saved for reuse. On a ckpt-tier model, stuck at zero after a few requests means nothing is being cached; the server logs a one-time warning when that happens. |
| `ckpt_hits`, `ckpt_matched_tokens` | Requests that warm-started from a saved prefix, and the prompt tokens they skipped. This is the value the tier delivers: on a repeat-heavy workload, matched tokens should approach total prompt tokens. |
| `ckpt_declines` | Saves the server skipped, grouped by reason (the same reason strings appear in the server log). Occasional entries are normal; every request piling into one reason means reuse is off for that traffic shape - include this map when filing an issue. |
| `ckpt_missed_adoptions` | Requests that matched a saved prefix but could not use it. Stays 0 in healthy operation; growth is a bug signal and trips a one-time warning. |
| `ckpt_pool_evictions` | Saved prefixes discarded to make room for new ones. Normal on long sessions, fastest on sliding-window models (gemma-family), whose snapshots are large and cannot be deduplicated. Raise `num_blocks` if you want deeper history to stay warm. |
| `ckpt_skeleton_writes` | Saves mirrored to the disk tier for warm restarts. Zero with `disk` on means a restart will start cold. |
| `sidecar_writes` | Draft-model cache entries saved next to their target entries (speculative decoding only). |
| `retire_fallback_full` | Finished requests whose generated tokens were saved in the slower whole-sequence form because no cheaper snapshot was available. Occasional is fine. |

The server also watches for a dead tier and warns once per model:
`GMLX_APC_CKPT_TRIPWIRE` (default 5) completed requests with zero stores,
or that many unusable matches with zero hits. Either warning means prefix
reuse is not working for that model - file an issue with the
`/v1/cache/stats` snapshot.

## Environment switches

| Variable | Meaning |
|----------|---------|
| `GMLX_SPEC_APC` | `0` turns every speculative cache layer off at once: lookups, stores, sidecar, checkpoint tier. The master switch. |
| `GMLX_SPEC_APC_RETIRE` | `0` turns off just the retirement store. |
| `GMLX_SPEC_APC_SIDECAR` | `0` turns off just the drafter-KV sidecar. |
| `GMLX_SPEC_APC_CKPT` | `0` turns off just the hybrid checkpoint tier (exact full clones return). |
| `GMLX_SPEC_APC_ENTRIES` | Prefix-layer LRU entries (default `4`). |
| `GMLX_SPEC_APC_SIDECAR_ENTRIES` | Drafter-sidecar LRU entries (default `8`). |
| `GMLX_SPEC_APC_BUDGET_MB` | Byte budget for the in-memory prefix layer, in MB (default `8192`). |
| `GMLX_SPEC_APC_SIDECAR_BUDGET_MB` | Byte budget for the drafter-sidecar LRU, in MB (default `512`). |
| `GMLX_APC_STORE_EVAL_CHUNK` | Blocks evaluated per step in the post-prefill store (default `32`); bounds the prefill-thread stall on long prompts. |
| `GMLX_APC_CKPT_INTERVAL` | Prefill checkpoint interval in tokens (default `4096`, snapped to the chunk grid; `0` = final checkpoint only). |
| `GMLX_APC_CKPT_REPLAY` | `0` disables the replay checkpoint (identical resends prefill cold again). |
| `GMLX_APC_CKPT_REPLAY_MIN` | Minimum prompt tokens before a replay checkpoint is saved on recurrent (GDN) models (default `1024`; short prompts re-prefill cheaply and are not worth the >100 MB state snapshot). |
| `GMLX_APC_CKPT_TURN` | `0` disables the turn checkpoint (next-turn reuse falls back to the interval grid). |
| `GMLX_APC_CKPT_SYS` | `0` disables the system-prompt anchor on both tiers (sibling requests sharing a system prompt prefill the shared prefix cold; on hybrid models they also fall back to the interval grid). |
| `GMLX_APC_CKPT_SYS_MIN` | Minimum tokens of shared system prefix before an anchor is saved (default `256`; raised to `GMLX_APC_CKPT_REPLAY_MIN` on recurrent models). A shorter shared prefix re-prefills in milliseconds and is not worth a record. |
| `GMLX_APC_ANCHOR_ENTRIES` | Exact-tier anchor LRU entries (default `4`). Exact-mode models (deepseek-v4-class pooling stacks) keep their system-prompt anchors here as whole-prefix clones, out of reach of the count-capped upstream exact LRU that every request's own store would churn. |
| `GMLX_APC_ANCHOR_BUDGET_MB` | Byte budget for the exact-tier anchor LRU, in MB (default `4096`). A deep shared prefix on a pooling stack clones to GBs; newest always survives. |
| `GMLX_APC_CKPT_TRIPWIRE` | Requests before the dead-tier tripwires warn (default `5`; `0` silences both). |
| `GMLX_APC_CKPT_RECORDS` | Checkpoint-record LRU entries (default `32`). |
| `GMLX_APC_CKPT_BUDGET_MB` | Byte budget for checkpoint-record payload (recurrent states + KV tails), in MB (default `4096`). A GDN record can carry >100 MB of state and each request saves several checkpoints, so expect resident memory to grow toward this budget on hybrid models under sustained multi-turn traffic; lower it if 4 GB of cache is too much for your machine. |
| `GMLX_APC_DECODE_CKPT` | Decode-time snapshot interval in generated tokens on hybrid models, anchored to the prompt end (default `512`; `0` off; widens automatically with context). |
| `GMLX_APC_RETIRE_LCP` | `0` keys retirement on the forwarded ids instead of the predicted next-turn render (also disables decode-time snapshots, which key on the prediction). |
| `GMLX_APC_FRESH_WAIT_MS` | Hold ceiling for the freshness admission gate, in ms (default `500`; `0` disables the gate). Sibling requests that arrive together admit one formation apart instead of together and cold: the first request prefills and stores the shared prefix, and the held siblings then admit warm. A sibling held past the ceiling admits cold. |
| `GMLX_APC_FRESH_MIN` | Minimum uncovered shared-prefix tokens before the gate holds a sibling (default `256`). Below the floor the duplicate prefill costs less than the wait. |
| `GMLX_FAITHFUL_HISTORY` | `0` restores mlx-vlm's stock chat-history rebuild, which drops `reasoning_content` from non-tool assistant messages before the template sees it (see `chat_template_kwargs`). |
