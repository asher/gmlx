"""Record the EmbeddingGemma 2 reference vectors that
tests/models/test_embeddinggemma2_reference.py holds gmlx to.

Runs sentence-transformers on google/embeddinggemma-2 in float32 on the CPU,
in an environment with torch, torchvision, torchcodec, transformers and
sentence-transformers, and never imports gmlx. Writes cases.json (each
case's parts, prompt name and processor options) and reference.npz (each
case's vector and token count) to
tests/fixtures/embeddinggemma2_reference/.

    python -I scripts/embedder_oracle/embeddinggemma2_reference.py \\
        <google/embeddinggemma-2 folder> <fixture folder> [<weights.npz> <name>]

With a weights file from embeddinggemma2_dequant.py, the model runs those
weights in place of google's and the vectors go to reference-<name>.npz,
the reference of that quantized build.

Media parts name files in the fixture's media/ folder, written by
embeddinggemma2_media.py.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path


def _long_text() -> str:
    words = ("river mountain library engine garden window market signal "
             "harvest bridge winter lantern copper meadow harbor orchard "
             "valley thunder canvas pebble compass violin glacier forest "
             "ladder kettle island candle meadowlark tunnel mirror quarry "
             "falcon basket marble saddle cottage blanket chimney feather "
             "planet journey station theory measure pattern history season "
             "student teacher village captain doctor farmer painter writer "
             "quickly slowly gently loudly quietly carefully suddenly often "
             "builds carries finds opens follows watches crosses gathers "
             "under over beside behind through across toward beyond").split()
    rng = random.Random(0)
    out = []
    for p in range(120):
        sents = []
        for _ in range(rng.randint(3, 6)):
            n = rng.randint(6, 14)
            s = " ".join(rng.choice(words) for _ in range(n))
            sents.append(s[0].upper() + s[1:] + ".")
        out.append(f"Section {p + 1}. " + " ".join(sents))
    return "\n\n".join(out)


def _t(text):
    return {"type": "text", "text": text}


def _m(kind, name):
    return {"type": kind, "file": name}


def cases() -> list[dict]:
    texts = {
        "short0": "Hello, world.",
        "short1": "What is the capital of France?",
        "short2": "The quick brown fox jumps over the lazy dog.",
        "multi0": "\u4eca\u5929\u5929\u6c14\u5f88\u597d\uff0c\u6211\u4eec\u53bb\u516c\u56ed\u6563\u6b65\u5427\u3002",
        "multi4": "\u00bfD\u00f3nde est\u00e1 la biblioteca m\u00e1s cercana a la estaci\u00f3n de tren?",
    }
    out = [{"id": f"t_{k}", "parts": [_t(v)]} for k, v in texts.items()]
    out += [
        {"id": "t_query", "parts": [_t(texts["short1"])], "prompt_name": "query"},
        {"id": "t_document", "parts": [_t(texts["short2"])], "prompt_name": "document"},
        {"id": "t_long", "parts": [_t(_long_text())]},
        {"id": "i_circle", "parts": [_m("image", "red_circle.png")]},
        {"id": "i_stripes_document", "parts": [_m("image", "stripes_triangle.png")],
         "prompt_name": "document"},
        {"id": "i_stripes_560", "parts": [_m("image", "stripes_triangle.png")],
         "mm": {"images_kwargs": {"max_soft_tokens": 560}}},
        {"id": "a_arpeggio", "parts": [_m("audio", "arpeggio_10s.flac")]},
        {"id": "a_melody_cut", "parts": [_m("audio", "melody_34s.flac")]},
        {"id": "a_melody_whole", "parts": [_m("audio", "melody_34s.flac")],
         "mm": {"audio_kwargs": {"truncation": False}}},
        {"id": "a_three_clips", "parts": [_m("audio", "beep_1s.flac"),
                                          _m("audio", "hum_4s.flac"),
                                          _m("audio", "arpeggio_10s.flac")]},
        {"id": "v_ball", "parts": [_m("video", "ball_3s.mp4")]},
        {"id": "v_scenes", "parts": [_m("video", "scenes_40s.mp4")]},
        {"id": "v_hd709", "parts": [_m("video", "square_hd709_3s.mp4")]},
        {"id": "x_interleaved", "parts": [_t("A photo of "), _m("image", "red_circle.png"),
                                          _t(" next to the sound "),
                                          _m("audio", "beep_1s.flac"), _t(".")]},
        {"id": "m_conversation", "messages": [
            {"role": "system", "content": "title: none | text: "},
            {"role": "user", "content": [_t("a red ball rolling"),
                                         _m("video", "ball_3s.mp4")]}]},
    ]
    return out


_ST_KWARGS = {"images_kwargs": "image", "videos_kwargs": "video", "audio_kwargs": "audio"}


def _st_part(p, media: Path):
    if p["type"] == "text":
        return p
    return {"type": p["type"], "path": str(media / p["file"])}


def _st_input(case, media: Path):
    if "messages" in case:
        return [{"role": m["role"],
                 "content": m["content"] if isinstance(m["content"], str)
                 else [_st_part(p, media) for p in m["content"]]}
                for m in case["messages"]]
    parts = case["parts"]
    if len(parts) == 1 and parts[0]["type"] == "text":
        return parts[0]["text"]
    if len(parts) == 1:
        return str(media / parts[0]["file"])
    return [{"role": "user", "content": [_st_part(p, media) for p in parts]}]


def _load_weights(model, path: str) -> None:
    import numpy as np
    import torch

    state = model[0].model.state_dict()
    with np.load(path) as z:
        for k in z.files:
            hits = [s for s in state if s == k or s.endswith("." + k)]
            if len(hits) != 1:
                raise SystemExit(f"{k}: {len(hits)} matches in the model")
            want = state[hits[0]]
            v = torch.from_numpy(z[k]).reshape(want.shape).to(want.dtype)
            want.copy_(v)
    print(f"loaded {len(z.files)} tensors from {path}", flush=True)


def main(model_dir: str, fixture: str, weights: str | None = None,
         name: str | None = None) -> None:
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(0)
    out = Path(fixture)
    media = out / "media"
    model = SentenceTransformer(model_dir, device="cpu",
                                model_kwargs={"dtype": torch.float32})
    if weights:
        with torch.no_grad():
            _load_weights(model, weights)
    all_cases = cases()
    vectors, tokens = {}, {}
    for case in all_cases:
        inp = _st_input(case, media)
        kw = {}
        if case.get("prompt_name"):
            kw["prompt_name"] = case["prompt_name"]
        if case.get("mm"):
            kw["processing_kwargs"] = {_ST_KWARGS[k]: v for k, v in case["mm"].items()}
        features = model.preprocess([inp], prompt=model.prompts.get(
            case.get("prompt_name") or "", None), **({"processing_kwargs": kw["processing_kwargs"]}
                                                    if "processing_kwargs" in kw else {}))
        tokens[case["id"]] = int(features["input_ids"].shape[-1])
        v = model.encode([inp], convert_to_numpy=True, **kw)[0]
        vectors[case["id"]] = v.astype(np.float32)
        print(f"{case['id']:<20} tokens {tokens[case['id']]:>5}  norm {np.linalg.norm(v):.6f}",
              flush=True)
    if not weights:
        (out / "cases.json").write_text(
            json.dumps(all_cases, indent=1, ensure_ascii=True) + "\n")
    np.savez(out / (f"reference-{name}.npz" if weights else "reference.npz"),
             **{f"v.{k}": v for k, v in vectors.items()},
             **{f"n.{k}": np.int64(n) for k, n in tokens.items()})


if __name__ == "__main__":
    if len(sys.argv) not in (3, 5):
        sys.exit(__doc__)
    main(*sys.argv[1:])
