"""EmbeddingGemma 2 through POST /v1/embeddings (run_embeddings) against
reference vectors that sentence-transformers recorded in float32 from
google/embeddinggemma-2 (scripts/embedder_oracle/embeddinggemma2_reference.py).

Integration tier. ``KQUANT_TEST_GGUF_DIR`` supplies the GGUFs, with their
projectors in the same folder, and ``GMLX_TEST_MLX_DIR`` the mlx-community
builds as ``mlx-community__<name>``. An absent build skips.

Each case is one request of one item: text, with and without a prompt
name, near 6,000 tokens; images at the default budget and at 560; a 10 s
clip, a 34 s 44.1 kHz stereo clip cut at 30 s and whole, three clips in one
item; a 3 s video, a 40 s video (the 32-frame cap) and a 3 s HD video
tagged BT.709; an interleaved item; and a conversation. Each response holds the reference's token count, and
its vector holds a minimum cosine to the reference vector.

A quantized build is held to a reference recorded from a float32 copy of
its own weights (scripts/embedder_oracle/embeddinggemma2_dequant.py), so
its floor measures gmlx and not the quantization. The bf16 build is held to
google's weights.

Floors, from the worst cosines on M3 GPU and on the CPU:

- Text holds v1's floors on the GPU. On the CPU two builds sit lower: the
  UD-Q6_K_XL text at 0.99766, where the CPU Q6_K matmul is less exact in
  every dtype, and the bf16 build at 0.99906, where MLX's CPU bfloat16
  matmul is. The reference itself in bfloat16 is at 0.99979.
- Media sit at 0.9998 and above, with ffmpeg on PATH. ffmpeg decodes the
  same frames as torchcodec. Without ffmpeg, OpenCV reads the videos: its
  frames differ by a few levels, and by tens on BT.709 video, so the video
  cases hold 0.995.

The F16 projector case runs the float32 path that M1 and M2 take.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import host_box
import numpy as np
import pytest

pytestmark = pytest.mark.integration

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "embeddinggemma2_reference"

# Minimum cosine per build: text on the GPU, text on the CPU, media.
_FLOORS = {
    "Q8_0": (0.9985, 0.9985, 0.999),
    "UD-Q6_K_XL": (0.9985, 0.997, 0.999),
    "8bit": (0.999, 0.999, 0.999),
    "bf16": (0.9995, 0.999, 0.999),
}
# Video read by OpenCV, when ffmpeg is not on PATH.
_OPENCV_VIDEO_FLOOR = 0.995


def _has_video(case) -> bool:
    parts = case.get("parts") or [p for m in case.get("messages", [])
                                  if isinstance(m["content"], list) for p in m["content"]]
    return any(p["type"] == "video" for p in parts)


def _cases():
    return json.loads((_FIXTURE / "cases.json").read_text())


def _reference(build: str):
    name = "reference.npz" if build == "bf16" else f"reference-{build.lower()}.npz"
    with np.load(_FIXTURE / name) as z:
        return ({k[2:]: z[k].astype(np.float64) for k in z.files if k.startswith("v.")},
                {k[2:]: int(z[k]) for k in z.files if k.startswith("n.")})


def _part(p):
    if p["type"] == "text":
        return p
    data = base64.b64encode((_FIXTURE / "media" / p["file"]).read_bytes()).decode()
    return {"type": p["type"], "base64": data}


def _body(case) -> dict:
    if "messages" in case:
        body = {"messages": [
            {"role": m["role"], "content": m["content"] if isinstance(m["content"], str)
             else [_part(p) for p in m["content"]]} for m in case["messages"]]}
    elif len(case["parts"]) == 1 and case["parts"][0]["type"] == "text":
        body = {"input": case["parts"][0]["text"]}
    else:
        body = {"input": [[_part(p) for p in case["parts"]]]}
    if case.get("prompt_name"):
        body["prompt_name"] = case["prompt_name"]
    if case.get("mm"):
        body["mm_processor_kwargs"] = case["mm"]
    return body


def _check(configured, media, build):
    import gmlx.serve.embeddings as emb

    import mlx.core as mx

    vectors, tokens = _reference(build)
    from gmlx.serve import media_programs

    gpu, cpu, media_floor = _FLOORS[build]
    text_floor = cpu if mx.default_device() == mx.cpu else gpu
    ffmpeg = bool(media_programs.find("ffmpeg") and media_programs.find("ffprobe"))
    worst: dict[str, float] = {}
    misses = []
    try:
        for case in _cases():
            out = emb.run_embeddings(configured_model=configured, body=_body(case),
                                     media=media)
            i = case["id"]
            assert out["usage"]["prompt_tokens"] == tokens[i], i
            v = np.asarray(out["data"][0]["embedding"], dtype=np.float64)
            assert abs(np.linalg.norm(v) - 1.0) < 1e-4, i
            r = vectors[i]
            cos = float(v @ r / np.linalg.norm(r))
            kind = "text" if i.startswith("t_") else "media"
            print(f"{i:<20} {cos:.6f}")
            worst[kind] = min(worst.get(kind, 1.0), cos)
            floor = text_floor if kind == "text" else media_floor
            if not ffmpeg and _has_video(case):
                floor = _OPENCV_VIDEO_FLOOR
            if cos < floor:
                misses.append(f"{i}: cosine {cos:.6f} < {floor}")
    finally:
        emb.release().result()
    print("worst", {k: round(v, 6) for k, v in worst.items()})
    assert not misses, "; ".join(misses)


def _gguf_folder(gguf_index):
    paths = gguf_index.get("gemma-embedding2")
    if not paths:
        pytest.skip("no gemma-embedding2 GGUF under KQUANT_TEST_GGUF_DIR")
    return paths


@pytest.mark.parametrize("quant", ["Q8_0", "UD-Q6_K_XL"])
@pytest.mark.parametrize("projector", ["mmproj-BF16.gguf", "mmproj-F16.gguf"])
def test_gguf_matches_the_reference(quant, projector, gguf_index, monkeypatch):
    import gmlx.serve.embeddings as emb
    from gmlx.load import dtypes

    path = next((p for p in _gguf_folder(gguf_index)
                 if Path(p).name == f"embeddinggemma-2-{quant}.gguf"), None)
    if path is None:
        pytest.skip(f"no embeddinggemma-2-{quant}.gguf under KQUANT_TEST_GGUF_DIR")
    mmproj = Path(path).with_name(projector)
    if not mmproj.is_file():
        pytest.skip(f"no {projector} beside {Path(path).name}")
    if projector == "mmproj-F16.gguf":
        # The float32 path of M1 and M2.
        monkeypatch.setenv(dtypes.ENV_VAR, "float16")
    media = emb.resolve_media(path, str(mmproj))
    assert media.path == str(mmproj), media.reason
    _check(path, media, quant)


@pytest.fixture
@host_box.gate
def mlx_dir() -> Path:
    d = os.environ.get("GMLX_TEST_MLX_DIR")
    if not d:
        pytest.skip("set GMLX_TEST_MLX_DIR to a folder of MLX encoder checkpoints")
    p = Path(d).expanduser()
    if not p.is_dir():
        pytest.skip(f"GMLX_TEST_MLX_DIR={d!r} is not a directory")
    return p


@pytest.mark.parametrize("build", ["8bit", "bf16"])
def test_mlx_build_matches_the_reference(build, mlx_dir):
    import gmlx.serve.embeddings as emb

    path = mlx_dir / f"mlx-community__embeddinggemma-2-{build}"
    if not path.is_dir():
        pytest.skip(f"no {path.name} under GMLX_TEST_MLX_DIR")
    media = emb.resolve_media(str(path))
    assert media.reason is None, media.reason
    _check(str(path), media, build)
