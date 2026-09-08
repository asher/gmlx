# Debug switches

Environment variables that change how gmlx builds or routes a model so a
defect can be isolated. None is a tuning knob: each is read at load or per
call, costs performance or loses a fix, and exists for A/B runs while
debugging. User-facing variables are in [../env-vars.md](../env-vars.md).

| Variable | Meaning |
|----------|---------|
| `GMLX_DECODE_LOOKAHEAD_PROBE` | Lossless recall probe for the lookahead predictor: records predicted-vs-actual routing per layer (plus a previous-token baseline) and prints a table at exit, issuing no reads. Worth a run on a new model family to see whether the prestage pays there. |
| `GMLX_ROPE_FACTORS=0` | Expert escape hatch for rope scaling: disables the `rope_freqs` factors patch that rebuilds Llama-3.1-style per-dim rope scaling from GGUF metadata. Set `0` only to rule the patch out when debugging long-context degradation. |
| `GMLX_SPARSE_ARCHS` | Extra architecture modules the sparse route may claim, comma-separated (for running a quality gate on a new arch). Default: only quality-gated archs (llama-family) are ever claimed. |
| `GMLX_FUSED_GDN=0` | Disable the fused gated-delta Metal kernels used by the Qwen3.5/3.6 hybrid architectures. The fusion is a numerics-affecting runtime patch; set `0` first when debugging those archs to rule it out. |
| `GMLX_QWEN_OWNED=0` | Build qwen3.5/3.6 text MTP targets on genuinely stock mlx-vlm classes instead of the owned forwards. The only install the fallback keeps is the tiled-V rebind (GGUF weight-order correctness); it loses every performance patch (fused GDN kernels, ragged decode kernels, verify fold, batched-verify SDPA, bf16 verify GEMV) and the two stock defects the owned path fixes come back: left-padded single-row batches attend their pad tokens, and an empty-sequence row in batched serve crashes (the old guard patch no longer installs). Multimodal MTP targets (LLM GGUF + mmproj) never take this flag's path: their trees are always built stock by mlx-vlm construction and always run the full patched regime. Read at load; a debugging A/B, not a tuning knob. |
| `GMLX_GEMMA_OWNED=0` | Build gemma4 text MTP targets on the stock mlx-vlm classes instead of the owned mask builder and attention. The fallback keeps the full patch regime (nosync mask/offset bodies, hd512 batched row route), so numerics are unchanged either way; the owned classes carry the same semantics natively. Multimodal targets (LLM GGUF + mmproj) are always built stock by mlx-vlm construction and always run the patched regime. Read at load; a debugging A/B, not a tuning knob. |
| `GMLX_SDPA_DEBUG=1` | Log which attention route each layer took, so a wrong route on a new architecture shows in the log. |
| `GMLX_ROUTE_LOG=1` | Log kernel route decisions per call. |
| `GMLX_MTP_DEBUG=1` | Log the MTP verify branch per round (`[mtp] verify branch: ...`). |
| `GMLX_ROUND_PROFILE=1` | Profile each speculative round; `GMLX_ROUND_LOG=/path.tsv` writes the rounds to a TSV file. |
