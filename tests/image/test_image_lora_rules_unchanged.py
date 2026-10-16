"""The reader's ComfyUI rules for the families whose maps set no
``comfy_stems``: every spelling of every target and every other layout's
name of each map under every prefix the reader strips, resolved on its
own, and every LoRA file under GMLX_TEST_LORA_DIR resolved under every
map, give the targets, refusals and warnings that the reader gave before
``comfy_stems`` existed. lora_rules_digest.json records the sha256 of
those results, written from the reader of the commit before the field.
The spellings run on the CPU; the files half skips without the
directory."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path

import pytest

from image_fixtures import fixture_view, load_json

DIGEST = "lora_rules_digest.json"
ENV = "GMLX_TEST_LORA_DIR"
# Each family's map on the config of a header fixture, and Qwen-Image's
# on the layout that stores its MLP pair apart too.
MAPS = {"z_image": ("z_image_turbo_q4_k_m.header.json", {}),
        "qwen_image21": ("qwen_image21_q4_k_m.header.json", {}),
        "qwen_image21 unfused": ("qwen_image21_q4_k_m.header.json",
                                 {"mlp_layout": "unfused"}),
        "krea2": ("krea2_turbo_q4_k_m.header.json", {}),
        "flux3_klein": ("flux3_klein_q4_k_m.header.json", {}),
        "flux3_klein base": ("flux3_klein_base_q4_k_m.header.json", {})}
PREFIXES = ("lora_unet_", "lycoris_", "lora_transformer_",
            "diffusion_model.", "model.diffusion_model.", "transformer.",
            "", "unet.", "base_model.model.", "text_encoders.")


def keymaps() -> dict:
    from gmlx.image import fingerprint as fp

    out = {}
    for label, (header, over) in MAPS.items():
        ident = fp.identify(fixture_view(header))
        out[label] = ident.family.lora_map(dict(ident.config, **over))
    return out


def _outcome(fn, path: str, name: str) -> list:
    """The targets and warnings of ``fn()``, or the refusal, with the
    file's path as ``name``."""
    from gmlx.image.lora import reader

    try:
        res = fn()
    except reader.AdapterError as e:
        return ["refused", type(e).__name__, str(e).replace(path, name)]
    rows = sorted([t.module, list(t.band) if t.band else None, t.pair.stem,
                   t.pair.rank, t.pair.alpha] for t in res.dit + res.text)
    return [rows, [w.replace(path, name) for w in res.warnings],
            sorted(res.unmatched)]


def _digest(cases: list) -> dict:
    h = hashlib.sha256()
    for case in cases:
        h.update(json.dumps(case, sort_keys=True).encode() + b"\n")
    return {"cases": len(cases), "sha256": h.hexdigest(),
            "refused": sum(c[1][0] == "refused" for c in cases),
            "warned": sum(c[1][0] != "refused" and bool(c[1][1])
                          for c in cases)}


def spelling_digests() -> dict:
    """For each map: every stem of its targets, its other layouts' names
    and the first targets of every other map, under every prefix, as one
    pair on its own."""
    import mlx.core as mx

    from gmlx.image.lora import reader

    maps = keymaps()
    zeros = reader.RawPair("", mx.zeros((1, 4)), mx.zeros((4, 1)), None)
    out = {}
    for label, km in maps.items():
        names = sorted({*km.targets, *km.absent, *km.text_targets,
                        *(n for other in maps.values()
                          for n in sorted(other.targets)[:24])})
        cases = []
        for name in names:
            for prefix in PREFIXES:
                stem = prefix + (name.replace(".", "_")
                                 if prefix.endswith("_") else name)
                raw = reader.RawLora("a.safetensors", {
                    stem: dataclasses.replace(zeros, stem=stem)}, {})
                for allow in (False, True):
                    cases.append([stem, _outcome(
                        lambda: reader.resolve(raw, km,
                                               allow_unmatched=allow),
                        "a.safetensors", "a.safetensors"), allow])
        out[label] = _digest(cases)
    return out


def lora_files(root) -> list:
    return sorted(p for p in Path(root).expanduser().rglob("*.safetensors")
                  if p.is_file())


def file_digests(root) -> dict:
    """For each file under ``root``, by its path there, and each map: the
    file resolved with the reader's defaults and with unmatched keys
    allowed."""
    from gmlx.image.lora import reader

    maps = keymaps()
    root = Path(root).expanduser()
    out = {}
    for p in lora_files(root):
        path, name = str(p), str(p.relative_to(root))
        try:
            raw = reader.read_lora(path, allow_unmatched=True)
        except reader.AdapterError as e:
            out[name] = {"size": p.stat().st_size,
                         "read": str(e).replace(path, name)}
            continue
        out[name] = {
            "size": p.stat().st_size,
            "maps": {label: _digest([[label, _outcome(
                lambda: reader.resolve(raw, km, allow_unmatched=allow),
                path, name), allow] for allow in (False, True)])
                for label, km in maps.items()}}
    return out


def test_no_older_map_sets_comfy_stems():
    for label, km in keymaps().items():
        assert km.comfy_stems is None and km.comfy_reads == "", label


def test_every_spelling_resolves_as_before():
    want = load_json(DIGEST)["spellings"]
    got = spelling_digests()
    assert set(got) == set(want)
    for label in want:
        assert got[label] == want[label], label
    maps = keymaps()
    assert all(got[label]["warned"] for label, km in maps.items()
               if km.comfy_prefixes)


def test_every_lora_file_resolves_as_before():
    root = os.environ.get(ENV)
    if not root:
        pytest.skip(f"set {ENV} to the directory of the community LoRA "
                    "files")
    want = load_json(DIGEST)["files"]
    got = file_digests(root)
    seen = [n for n in got if n in want and got[n].get("size") ==
            want[n].get("size")]
    if not seen:
        pytest.skip(f"none of the recorded files is under {ENV}")
    for name in seen:
        assert got[name] == want[name], name
    print(f"lora rules: {len(seen)} files under every older map as "
          f"before: {', '.join(seen)}")
