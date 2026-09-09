# Prompt cache internals

How the prompt cache tiers are chosen per architecture, the counters that
show whether reuse is working, and the environment switches that tune or
disable each layer. Operators read
[performance.md](../performance.md#the-prompt-cache); this page is for
contributors. Upstream calls the cache APC, and the switches use that name.

## Which tier serves which architecture

The cache routes each model by its cache shape once, at load, and the
server logs the routing as `APC tier:` so a silent mis-route is visible.
Reuse works on every family; the tiers differ in storage layout, not in
whether hits happen.

| Cache shape | Example archs | Tier |
|-------------|---------------|------|
| plain KV, dense or MoE | llama, qwen2/3, qwen3moe, glm4(-moe), deepseek2/v3, phi3, granite, hunyuan, minimax-m2, gemma2 | block |
| hybrid GDN (recurrent + KV layers) | qwen3.5/3.6 (incl. MoE), qwen3-next, kimi-k3, nemotron-h, granitemoehybrid | ckpt |
| sliding-window attention | gemma3, gemma-4 (incl. E4B/3n), gpt-oss, SWA llama | ckpt |
| CacheList / pure recurrent | falcon-h1, mamba2, rwkv7, plamo2, deepseek-v3.2, deepseek4 | exact |
| MSA indexer armed | minimax-m3 indexer GGUFs | none (logged warning; the indexless GGUF serves via the block tier) |

## Checkpoint-tier counters

`GET /v1/cache/stats` carries the `ckpt_*` fields for every model, but
they only change on checkpoint-tier architectures. All-zero on a block-tier
or exact-tier model is normal. On checkpoint-tier models they answer one
question: is prefix reuse working? Judge reuse from these fields
rather than from ratios built on the stock ones, because checkpoint lookups
increment the shared hit counters on success but record nothing on a miss, and
the token totals include window snapshots that can never be shared.
`disk_writes` counts write operations: one per exact-format entry,
checkpoint skeletons and drafter sidecars included, and one per block for
block shards.

| Field | What it tells you |
|-------|-------------------|
| `ckpt_stores` | Prefixes saved for reuse. Still zero after a few requests on a checkpoint-tier model means nothing is being cached; the server logs a one-time warning. |
| `ckpt_hits`, `ckpt_matched_tokens` | Requests that warm-started from a saved prefix, and the prompt tokens they skipped. On repeat-heavy traffic matched tokens approach total prompt tokens. |
| `ckpt_declines` | Saves the server skipped, by reason. Occasional entries are normal; every request counted under one reason means reuse is not working for that traffic pattern. |
| `ckpt_missed_adoptions` | Requests that matched a saved prefix but could not use it. Stays 0 in healthy operation; growth indicates a bug and triggers a one-time warning. |
| `ckpt_pool_evictions` | Saved prefixes discarded to make room. Normal on long sessions, fastest on sliding-window models. Raise `num_blocks` to keep more history cached. |
| `ckpt_skeleton_writes` | Saves mirrored to the disk tier for warm restarts. Zero with `disk` on means a restart will start cold. |
| `sidecar_writes` | Draft-model cache entries saved next to their target entries (speculative decoding only). |
| `retire_fallback_full` | Finished requests whose generated tokens were saved in the slower whole-sequence form because no smaller snapshot was available. Occasional entries are normal. |

The server also checks for a tier that is not storing or hitting and warns once per model:
`GMLX_APC_CKPT_TRIPWIRE` (default 5) completed requests with zero stores,
or that many unusable matches with zero hits. Either warning means prefix
reuse is not working for that model. File an issue with the
`/v1/cache/stats` snapshot.

## Under kvarn KV

Tier routing is the same as under fp16 KV, with one change: dense models use
the exact tier, since the 16-token block tier cannot split kvarn's 128-token
records. Checkpoint-shaped stacks (every hybrid-GDN and sliding-window family
in the reuse table, when the attention head_dim is 128, 256 or 512) keep full
checkpoint-tier reuse and store kvarn records. The attention payload is inline
in the record rather than in pool blocks, so `GMLX_APC_CKPT_BUDGET_MB` bounds
the tier's memory and `APC_NUM_BLOCKS` matters little. Entries and disk
skeletons are keyed to the kvarn width and tail, so a config change or a
restart that switches between stock and kvarn misses instead of reading a
stale format. Cascade shared-prefix decode is off under kvarn and logs that
once. Speculative rollback into a sealed record reopens it from its codes,
one lossy round trip; rows still in the fp16 tail roll back exactly.

## Environment switches

| Variable | Meaning |
|----------|---------|
| `GMLX_SPEC_APC` | `0` turns every speculative cache layer off at once: lookups, stores, sidecar, checkpoint tier. The master switch. |
| `GMLX_SPEC_APC_RETIRE` | `0` turns off just the retirement store. |
| `GMLX_SPEC_APC_SIDECAR` | `0` turns off just the drafter-KV sidecar. |
| `GMLX_SPEC_APC_CKPT` | `0` turns off just the hybrid checkpoint tier (the exact full-clone path is used instead). |
| `GMLX_SPEC_APC_ENTRIES` | Prefix-layer LRU entries (default `4`). |
| `GMLX_SPEC_APC_SIDECAR_ENTRIES` | Drafter-sidecar LRU entries (default `8`). |
| `GMLX_SPEC_APC_BUDGET_MB` | Byte budget for the in-memory prefix layer, in MB (default `8192`). |
| `GMLX_SPEC_APC_SIDECAR_BUDGET_MB` | Byte budget for the drafter-sidecar LRU, in MB (default `512`). |
| `GMLX_APC_STORE_EVAL_CHUNK` | Blocks evaluated per step in the post-prefill store (default `32`); bounds the prefill-thread stall on long prompts. |
| `GMLX_APC_CKPT_INTERVAL` | Prefill checkpoint interval in tokens (default `4096`, rounded to the chunk grid; `0` = final checkpoint only). |
| `GMLX_APC_CKPT_REPLAY` | `0` disables the replay checkpoint (identical resends prefill cold again). |
| `GMLX_APC_CKPT_REPLAY_MIN` | Minimum prompt tokens before a replay checkpoint is saved on recurrent models (default `1024`); shorter prompts re-prefill quickly. |
| `GMLX_APC_CKPT_TURN` | `0` disables the turn checkpoint (next-turn reuse falls back to the interval grid). |
| `GMLX_APC_CKPT_SYS` | `0` disables the system-prompt anchor on both tiers. Sibling requests sharing a system prompt then prefill the shared prefix without cache reuse. |
| `GMLX_APC_CKPT_SYS_MIN` | Minimum tokens of shared system prefix before an anchor is saved (default `256`, raised to the replay minimum on recurrent models). |
| `GMLX_APC_ANCHOR_ENTRIES` | Exact-tier anchor LRU entries (default `4`), where exact-mode models keep system-prompt anchors as whole-prefix clones. |
| `GMLX_APC_ANCHOR_BUDGET_MB` | Byte budget for the exact-tier anchor LRU, in MB (default `4096`). A long shared prefix on a pooling stack clones to GBs; the newest entry is never evicted. |
| `GMLX_APC_CKPT_TRIPWIRE` | Completed requests before the not-storing and not-hitting checks warn (default `5`; `0` silences both). |
| `GMLX_APC_CKPT_RECORDS` | Checkpoint-record LRU entries (default `32`). |
| `GMLX_APC_CKPT_BUDGET_MB` | Byte budget for checkpoint payload, in MB (default `4096`). Resident memory grows toward it on hybrid models under multi-turn traffic. |
| `GMLX_APC_DECODE_CKPT` | Decode-time snapshot interval in generated tokens on hybrid models, anchored to the prompt end (default `512`; `0` off; widens automatically with context). |
| `GMLX_APC_RETIRE_LCP` | `0` keys retirement on the forwarded ids instead of the predicted next-turn render (also disables decode-time snapshots, which key on the prediction). |
| `GMLX_APC_FRESH_WAIT_MS` | Maximum hold for the freshness admission gate, in ms (default `500`; `0` disables). Siblings arriving together admit one at a time; the rest reuse the prefix. |
| `GMLX_APC_FRESH_MIN` | Minimum uncovered shared-prefix tokens before the gate holds a sibling (default `256`). Below the minimum the duplicate prefill takes less time than the wait. |
| `GMLX_FAITHFUL_HISTORY` | `0` restores mlx-vlm's stock chat-history rebuild, which drops `reasoning_content` from non-tool assistant messages before the template sees it. |
