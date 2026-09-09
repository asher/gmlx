# Third-party notices

gmlx (Business Source License 1.1, see [LICENSE](LICENSE)) vendors or
derives code from the following projects. Full license texts, with per-file
provenance, live in [`licenses/`](licenses/).

| Project | License | What was used |
|---|---|---|
| [mlx-lm](https://github.com/ml-explore/mlx-lm) | MIT | Vendored model/cache classes from unmerged PRs: `gmlx/models/deepseek_v4/model.py` + `gmlx/models/deepseek_v4/hyper_connection.py` + `gmlx/models/deepseek_v4/cache.py` (PR #1192), `gmlx/models/minimax_m3.py` (PR #1401), `gmlx/models/hy_v3/model.py` (PR #1485); `gmlx/models/kimi_k3.py` builds on the released `kimi_linear` module. The installed package is also a runtime dependency. |
| [omlx](https://github.com/jundot/omlx) | Apache-2.0 | DeepSeek-V4-Flash MTP port (`gmlx/models/deepseek_v4/mtp.py`), the rotating-cache MTP undo wrap in `gmlx/models/deepseek_v4/cache.py`, and the native-kernel dispatch pattern in `gmlx/models/deepseek_v4/model.py`. |
| [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) | MIT | Pre-tokenizer split regex patterns in `gmlx/load/tokenizer.py`; GGUF metadata and quantization-type conventions; the Kimi-K3 chat-template fixture `tests/fixtures/kimi_k3_template.jinja` (PR #26185). |
| [ds4.c (dwarfstar)](licenses/ds4.c-LICENSE) | MIT | QAT round-trip and sink-attention reference semantics reimplemented in `gmlx/models/deepseek_v4/model.py` (cited per function in comments); the DSpark drafter runtime reimplemented in `gmlx/models/deepseek_v4/dspark.py`; the `deepseek4-dspark` sidecar layout (`gguf-tools/deepseek4-quantize.c` and the published DSpark-support sidecar) emitted by `scripts/convert_dspark_sidecar.py`. |
| [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | MIT | Reference DSparkBlock forward semantics (`inference/model.py`) reimplemented in `gmlx/models/deepseek_v4/dspark.py` (license text in [`licenses/deepseek-v4-LICENSE`](licenses/deepseek-v4-LICENSE)). |
| [beellama.cpp](https://github.com/Anbeeld/beellama.cpp) | MIT | The KVarN KV-cache record format and cache semantics (128-token rotated records with three axis vectors, fp16 sink and precision tail, `kvarnN` widths, `--kv-tail-tokens`) reimplemented in `gmlx/cache/kvarn_cache.py` and `gmlx/cache/kvarn_sdpa.py`. The method is Huawei's KVarN ([arXiv:2606.03458](https://arxiv.org/abs/2606.03458), reference at [huawei-csl/KVarN](https://github.com/huawei-csl/KVarN), Apache-2.0; no code from it is used). The kernels live in mlx-kquant, whose notices cover its own beellama.cpp port. |
| [misaki](https://github.com/hexgrad/misaki) | Apache-2.0 | Whole package vendored at `gmlx/_vendor/misaki` (commit `fba1236`): Kokoro's grapheme-to-phoneme front-end, used by the TTS service until upstream's next PyPI release. |

The [mlx-kquant](https://github.com/asher/mlx-kquant) dependency ships its own
third-party notices (llama.cpp/ggml, gguf-tools, MLX, omlx) inside the wheel
under `mlx_kquant/licenses/`.
