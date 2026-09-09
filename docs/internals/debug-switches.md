# Debug switches

Environment variables that change how gmlx builds or routes a model so a
defect can be isolated. None is a tuning setting: each is read at load or per
call, reduces performance or disables a fix, and exists for A/B runs while
debugging. User-facing variables are in [../env-vars.md](../env-vars.md).

| Variable | Meaning |
|----------|---------|
| `GMLX_KVARN=0` | Disable `--kv-quant-scheme kvarn` at cache build; the scheme is dropped with that reason and the model runs fp16 KV. |
| `GMLX_KVARN_SDPA=0` | Route kvarn decode through the materialize path instead of the fused record kernels. Differs at fp16 rounding only; set first when debugging kvarn. |
| `GMLX_KVARN_FA=0` | Keep kvarn MTP verify rounds on the vector decode kernel instead of the matrix-unit kernel. Same numerics to fp16 rounding; widths above 4 materialize. |
| `GMLX_DECODE_LOOKAHEAD_PROBE=1` | Record predicted versus actual expert routing per layer and print the recall table at exit, issuing no reads. Run it on a new model family. |
| `GMLX_ROPE_FACTORS=0` | Disable the patch that rebuilds Llama-3.1-style per-dimension rope scaling from GGUF metadata. Set only to rule it out when debugging long context. |
| `GMLX_SPARSE_ARCHS` | Extra architecture modules the sparse attention route may apply to, comma separated, for a quality gate on a new arch. Default: the llama family only. |
| `GMLX_FUSED_GDN=0` | Disable the fused gated-delta Metal kernels the Qwen3.5 and 3.6 hybrids use. The fusion affects numerics, so set this first when debugging those archs. |
| `GMLX_QWEN_OWNED=0` | Build Qwen3.5 and 3.6 text MTP targets on stock mlx-vlm classes. Disables every performance patch and restores two stock defects. Multimodal targets ignore it. |
| `GMLX_GEMMA_OWNED=0` | Build gemma-4 text MTP targets on stock mlx-vlm classes. Numerics are unchanged either way. Multimodal targets ignore it. |
| `GMLX_SDPA_DEBUG=1` | Log which attention route each layer took, so a wrong route on a new architecture shows in the log. |
| `GMLX_ROUTE_LOG=1` | Print per-route attention call counts at process exit. |
| `GMLX_MTP_DEBUG=1` | Log the MTP verify branch per round. |
| `GMLX_ROUND_PROFILE=1` | Profile each speculative round; `GMLX_ROUND_LOG=/path.tsv` writes the rounds to a TSV file. Works in the server process. |
| `GMLX_DECODE_PHASE_STATS=1` | Print a streamed decode's per-token split between disk stalls and the eval and sync bucket at exit. A clock frequency drop shows as a large sync bucket. |
| `GMLX_PIN_CAST_EXCLUDE=0` | Pin the file bytes of every every-token tensor on a streamed model, including tensors the loader converts at load. Default on: converted tensors are left out. |
| `GMLX_STREAM_PLE_COMPOSE=0` | Keep a streamable lookup table resident when the experts also stream, instead of streaming both. |
| `GMLX_RELEASE_PAGECACHE=0` | Keep a released over-RAM model's pages in the page cache at exit or unload instead of invalidating them. |
