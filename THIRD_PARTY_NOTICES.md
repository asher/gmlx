# Third-party notices

gmlx (Business Source License 1.1, see [LICENSE](LICENSE)) vendors or
derives code from the following projects. Full license texts, with per-file
provenance, live in [`licenses/`](licenses/).

| Project | License | What was used |
|---|---|---|
| [mlx-lm](https://github.com/ml-explore/mlx-lm) | MIT | Vendored model/cache classes from unmerged PRs: `deepseek_v4_model.py` + `deepseek_v4_hyper_connection.py` + `deepseek_v4_cache.py` (PR #1192), `minimax_m3_model.py` (PR #1401), `hy_v3_model.py` (PR #1485). The installed package is also a runtime dependency. |
| [omlx](https://github.com/jundot/omlx) | Apache-2.0 | DeepSeek-V4-Flash MTP port (`deepseek_v4_mtp.py`), the rotating-cache MTP undo wrap in `deepseek_v4_cache.py`, and the native-kernel dispatch pattern in `deepseek_v4_model.py`. |
| [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) | MIT | Pre-tokenizer split regex patterns in `tokenizer.py`; GGUF metadata and quantization-type conventions. |
| [ds4.c (dwarfstar)](licenses/ds4.c-LICENSE) | MIT | QAT round-trip and sink-attention reference semantics reimplemented in `deepseek_v4_model.py` (cited per function in comments). |
| [misaki](https://github.com/hexgrad/misaki) | Apache-2.0 | Whole package vendored at `gmlx/_vendor/misaki` (commit `fba1236`): Kokoro's grapheme-to-phoneme front-end, used by the TTS service until upstream's next PyPI release. |

The [mlx-kquant](https://github.com/asher/mlx-kquant) dependency ships its own
third-party notices (llama.cpp/ggml, gguf-tools, MLX, omlx) inside the wheel
under `mlx_kquant/licenses/`.
