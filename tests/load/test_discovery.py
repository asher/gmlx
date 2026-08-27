#!/usr/bin/env python3
"""GGUF discovery: header classification, deterministic id derivation, directory
scan + mmproj pairing, and the YAML scaffold. Pure CPU - verdict logic runs on
plain metadata dicts and the scan monkeypatches ``classify_gguf``, so no GGUF
files, no GPU, no model load."""
from __future__ import annotations

import os

import pytest

pytest.importorskip("yaml")

from gmlx import discovery as disc  # noqa: E402
from gmlx.config import DiscoverSpec, ModelCfg, build_config  # noqa: E402


# _classify_meta - the header verdict, exercised on plain dicts
def _classify(meta, name):
    return disc._classify_meta(meta, basename=name, path=f"/d/{name}")


def test_clip_arch_is_mmproj():
    c = _classify({"general.architecture": "clip"}, "mmproj-x-bf16.gguf")
    assert c.kind == "mmproj"


def test_mmproj_filename_is_mmproj_even_without_clip_arch():
    c = _classify({"general.architecture": "gemma4"}, "mmproj-gemma-bf16.gguf")
    assert c.kind == "mmproj"


def test_native_head_mtp_model():
    c = _classify({"general.architecture": "qwen35",
                   "qwen35.nextn_predict_layers": 1},
                  "Qwen3.6-27B-Q4_K_S.gguf")
    assert c.kind == "model"
    assert c.mtp is True
    assert c.loadable is True                    # qwen35 is a supported arch


def test_plain_text_model_not_mtp():
    c = _classify({"general.architecture": "qwen3",
                   "qwen3.nextn_predict_layers": 0}, "qwen3-0.6b-Q8_0.gguf")
    assert c.kind == "model"
    assert c.mtp is False


def test_assistant_arch_is_drafter():
    c = _classify({"general.architecture": "gemma4_assistant"},
                  "gemma-4-31B-it-assistant.Q8_0.gguf")
    assert c.kind == "drafter"


def test_mtp_support_arch_is_drafter():
    """llama.cpp writes the DeepSeek-V4 MTP module as `<arch>_mtp_support`."""
    c = _classify({"general.architecture": "deepseek4_mtp_support"},
                  "DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf")
    assert c.kind == "drafter"


def test_dflash_arch_is_drafter():
    """llama.cpp packages the DSpark drafter under arch `dflash` (the unsloth
    release); it must classify as a companion, not an unsupported model."""
    c = _classify({"general.architecture": "dflash"},
                  "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf")
    assert c.kind == "drafter"


def test_companion_prefers_native_dspark_over_dflash(tmp_path, monkeypatch):
    """With both containers next to the target, the gmlx-native sidecar wins."""
    target = tmp_path / "model.gguf"
    for name in ("a-dflash.gguf", "b-dspark.gguf", "model.gguf"):
        (tmp_path / name).write_bytes(b"GGUF")
    metas = {str(tmp_path / "a-dflash.gguf"): {"arch": "dflash"},
             str(tmp_path / "b-dspark.gguf"): {"arch": "deepseek4-dspark"},
             str(target): {"arch": "deepseek4"}}
    monkeypatch.setattr(disc, "header_meta", lambda p: metas.get(str(p)))
    assert disc.find_mtp_companion(str(target)) == str(tmp_path / "b-dspark.gguf")


def test_companion_accepts_dflash_alone(tmp_path, monkeypatch):
    target = tmp_path / "model.gguf"
    for name in ("dspark-q8.gguf", "model.gguf"):
        (tmp_path / name).write_bytes(b"GGUF")
    metas = {str(tmp_path / "dspark-q8.gguf"): {"arch": "dflash"},
             str(target): {"arch": "deepseek4"}}
    monkeypatch.setattr(disc, "header_meta", lambda p: metas.get(str(p)))
    assert disc.find_mtp_companion(str(target)) == str(tmp_path / "dspark-q8.gguf")


def test_backbone_field_implies_drafter():
    """A future/unknown drafter arch is still caught by its target-backbone field."""
    c = _classify({"general.architecture": "gemma4-weird-draft",
                   "gemma4-weird-draft.embedding_length_out": 5120},
                  "drafter.Q8_0.gguf")
    assert c.kind == "drafter"


def test_unsupported_arch_model_marked_unloadable():
    c = _classify({"general.architecture": "mamba2"}, "mamba-Q4_K.gguf")
    assert c.kind == "model"
    assert c.loadable is False                   # arch builds no model


def test_missing_arch_is_unloadable_model():
    c = _classify({}, "mystery.gguf")
    assert c.kind == "model" and c.loadable is False


# id derivation + quant tag
@pytest.mark.parametrize("name,expect_id,expect_q", [
    ("Qwen3.6-27B-Q4_K_S.gguf", "qwen3.6-27b", "Q4_K_S"),
    ("google_gemma-4-31B-it-Q6_K_L.gguf", "google-gemma-4-31b-it", "Q6_K_L"),
    ("gemma-4-31B-it-assistant.Q8_0.gguf", "gemma-4-31b-it", "Q8_0"),
    ("mmproj-gemma-4-E4B-it-bf16.gguf", "gemma-4-e4b-it", "BF16"),
    ("Qwen3.6-27B-Q4_K.gguf", "qwen3.6-27b", "Q4_K"),
    ("Llama-3.1-8B-Instruct-UD-Q4_K_XL.gguf", "llama-3.1-8b-instruct", "Q4_K_XL"),
    # imatrix provenance markers are stripped (not part of the model name), so a
    # model's i1 quants share its base id instead of splitting onto `...instruct.i1`.
    ("Llama-3.2-1B-Instruct.i1-IQ2_M.gguf", "llama-3.2-1b-instruct", "IQ2_M"),
    ("Some-Model.imatrix-Q4_K_M.gguf", "some-model", "Q4_K_M"),
])
def test_derive_id_and_quant(name, expect_id, expect_q):
    mid, q = disc.derive_id(name)
    assert mid == expect_id
    assert q == expect_q


# embedding / reranker classification - not generative chat models
@pytest.mark.parametrize("arch,name,kind", [
    ("qwen3", "Qwen3-Embedding-0.6B-Q8_0.gguf", "embedding"),
    ("qwen3", "Qwen3-Reranker-0.6B-Q4_K_M.gguf", "reranker"),
    ("bert", "bge-small-en-v1.5-bf16.gguf", "embedding"),     # encoder arch
])
def test_embedding_reranker_classified(arch, name, kind):
    c = _classify({"general.architecture": arch}, name)
    assert c.kind == kind


def test_pooling_type_implies_embedding():
    c = _classify({"general.architecture": "qwen3", "qwen3.pooling_type": 3}, "x.gguf")
    assert c.kind == "embedding"


def test_pooling_type_zero_is_a_normal_model():
    c = _classify({"general.architecture": "qwen3", "qwen3.pooling_type": 0}, "x.gguf")
    assert c.kind == "model"


def test_adapter_general_type_is_adapter_kind():
    # A trained LoRA adapter carries its BASE model's arch; without the
    # general.type check it would classify as a loadable chat model.
    c = _classify({"general.architecture": "qwen3", "general.type": "adapter",
                   "adapter.type": "lora"}, "my-lora.gguf")
    assert c.kind == "adapter"
    assert c.loadable is False


def test_adapter_type_key_alone_is_adapter_kind():
    c = _classify({"general.architecture": "qwen3", "adapter.type": "lora"},
                  "trained-lora.gguf")
    assert c.kind == "adapter"


def test_gemma_embedding_arch_classified_as_embedding():
    # EmbeddingGemma classifies as an embedder by arch alone (no pooling_type, a
    # filename without an *embed* token) -- it's in _ENCODER_ARCHES.
    c = _classify({"general.architecture": "gemma-embedding"}, "model-Q8_0.gguf")
    assert c.kind == "embedding"


def test_derive_id_is_deterministic():
    assert disc.derive_id("Foo-Bar-Q5_K_M.gguf") == disc.derive_id("Foo-Bar-Q5_K_M.gguf")


def test_quant_tag_none_when_absent():
    assert disc.quant_tag("some-model.gguf") is None


def test_sharded_name_collapses_to_one_id():
    a, _ = disc.derive_id("BigModel-Q6_K-00001-of-00005.gguf")
    b, _ = disc.derive_id("BigModel-Q6_K.gguf")
    assert a == b == "bigmodel"


# scan_dirs - monkeypatched classify, real empty files on disk
def _write(root, *names):
    for n in names:
        (root / n).write_bytes(b"GGUF")
    return root


@pytest.fixture
def fake_classify(monkeypatch):
    """Classify by basename heuristic so the scan needs no real GGUF headers."""
    def _f(path):
        import os
        name = os.path.basename(path).lower()
        meta = {"general.architecture": "qwen35"}
        if name.startswith("mmproj"):
            meta = {"general.architecture": "clip"}
        elif "assistant" in name:
            # the drafter carries the hidden size of the target it drafts for
            meta = {"general.architecture": "gemma4_assistant",
                    "gemma4_assistant.embedding_length_out": 5376}
        elif "gemma" in name:
            meta = {"general.architecture": "gemma4",
                    "gemma4.embedding_length": 5376}
        elif "nomtp" in name:
            meta = {"general.architecture": "qwen3"}
        elif "bad" in name:
            meta = {"general.architecture": "mamba2"}     # unsupported
        else:
            meta["qwen35.nextn_predict_layers"] = 1       # native-head MTP
        return disc._classify_meta(meta, basename=os.path.basename(path), path=path)

    monkeypatch.setattr(disc, "classify_gguf", _f)
    return _f


def _scan(root, **spec_kw):
    spec = DiscoverSpec(dir=str(root), recursive=spec_kw.pop("recursive", False),
                        pair_mmproj=spec_kw.pop("pair_mmproj", True),
                        speculative=spec_kw.pop("speculative", "auto"))
    return disc.scan_dirs([spec], [str(root)])


def test_scan_emits_model_with_speculative(tmp_path, fake_classify):
    root = _write(tmp_path, "Qwen3.6-27B-Q4_K_S.gguf")
    models = _scan(root)
    assert len(models) == 1
    assert models[0].id == "qwen3.6-27b-q4"        # quant codec always in the id
    assert models[0].speculative is True          # native-head MTP, speculative=auto


def test_scan_speculative_false_disables(tmp_path, fake_classify):
    root = _write(tmp_path, "Qwen3.6-27B-Q4_K_S.gguf")
    models = _scan(root, speculative=False)
    assert models[0].speculative is False


def test_scan_pairs_sibling_mmproj(tmp_path, fake_classify):
    root = _write(tmp_path, "qwen3.6-27b-Q4_K_S.gguf",
                  "mmproj-qwen3.6-27b-bf16.gguf")
    models = _scan(root)
    assert len(models) == 1                       # mmproj is a companion, not a model
    assert models[0].mmproj.endswith("mmproj-qwen3.6-27b-bf16.gguf")


def test_scan_skips_unsupported_arch(tmp_path, fake_classify):
    root = _write(tmp_path, "good-Q4_K_S.gguf", "bad-arch-Q4_K.gguf")
    ids = {m.id for m in _scan(root)}
    assert "good-q4" in ids
    assert not any("bad" in i for i in ids)       # unsupported arch dropped


def test_scan_drafter_not_standalone(tmp_path, fake_classify):
    root = _write(tmp_path, "gemma-4-31B-it-Q6_K.gguf",
                  "gemma-4-31B-it-assistant.Q8_0.gguf")
    models = _scan(root)
    assert len(models) == 1                       # the assistant is not a model
    assert models[0].id == "gemma-4-31b-it-q6"
    # the assistant pairs into the model it serves, not into an entry of its own
    assert models[0].draft_gguf.endswith("gemma-4-31B-it-assistant.Q8_0.gguf")
    assert models[0].speculative is True


# drafter pairing - a sibling drafter becomes the `draft_gguf` of its target
def _drafter_classify(monkeypatch, *, target_arch="muse-glimmer",
                      target_embd=6656, draft_embd=6656):
    """Classify a `dflash*` file as a drafter and every other file as a
    target of ``target_arch``, with the hidden size each one declares."""
    def _f(path):
        name = os.path.basename(path)
        if name.lower().startswith("dflash"):
            meta = {"general.architecture": "dflash",
                    "dflash.embedding_length": draft_embd}
        else:
            meta = {"general.architecture": target_arch,
                    f"{target_arch}.embedding_length": target_embd}
        return disc._classify_meta(meta, basename=name, path=path)

    monkeypatch.setattr(disc, "classify_gguf", _f)


def test_scan_pairs_sibling_dflash_drafter(tmp_path, monkeypatch):
    _drafter_classify(monkeypatch)
    root = _write(tmp_path, "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf",
                  "dflash-Muse-Glimmer-30B-Q4_K_M.gguf")
    models = _scan(root)
    assert len(models) == 1
    assert models[0].draft_gguf.endswith("dflash-Muse-Glimmer-30B-Q4_K_M.gguf")
    assert models[0].speculative is True


def test_drafter_pairs_into_every_quant_of_its_target(tmp_path, monkeypatch):
    # Two quants of one target share the drafter: each entry can serve.
    _drafter_classify(monkeypatch)
    root = _write(tmp_path, "Muse-Glimmer-30B-Q4_K_M.gguf",
                  "Muse-Glimmer-30B-Q6_K.gguf",
                  "dflash-Muse-Glimmer-30B-Q4_K_M.gguf")
    models = _scan(root)
    assert len(models) == 2
    assert all(m.draft_gguf and m.speculative for m in models)


def test_drafter_unpaired_on_hidden_size_mismatch(tmp_path, monkeypatch, capsys):
    # llama.cpp writes the DeepSeek DSpark sidecar under the same `dflash` arch,
    # so the arch alone must not pair it with a muse-glimmer target.
    _drafter_classify(monkeypatch, target_embd=4096, draft_embd=6656)
    root = _write(tmp_path, "Muse-Glimmer-30B-Q4_K_M.gguf",
                  "dflash-DeepSeek-V4-Q4_K_M.gguf")
    models = _scan(root)
    assert models[0].draft_gguf is None
    assert models[0].speculative is False
    assert "assistant drafter (configure via" in capsys.readouterr().err


def test_drafter_unpaired_when_speculative_false(tmp_path, monkeypatch):
    _drafter_classify(monkeypatch)
    root = _write(tmp_path, "Muse-Glimmer-30B-Q4_K_M.gguf",
                  "dflash-Muse-Glimmer-30B-Q4_K_M.gguf")
    models = _scan(root, speculative=False)
    assert models[0].draft_gguf is None
    assert models[0].speculative is False


def test_drafter_unpaired_when_arch_has_no_mtp_class(tmp_path, monkeypatch):
    # `llama` builds a model, but the loader has no MTP target class for it,
    # so `speculative` would fail at build time.
    _drafter_classify(monkeypatch, target_arch="llama", target_embd=6656)
    root = _write(tmp_path, "Some-Llama-8B-Q4_K_M.gguf",
                  "dflash-Some-Llama-8B-Q4_K_M.gguf")
    models = _scan(root)
    assert models[0].draft_gguf is None


def test_generic_drafter_name_unpaired_when_several_candidates(
        tmp_path, monkeypatch):
    # A filename with no model name in it carries no signal, and two targets
    # of equal hidden size are both plausible.
    _drafter_classify(monkeypatch)
    root = _write(tmp_path, "Muse-Glimmer-30B-Q4_K_M.gguf",
                  "Other-Muse-Model-Q4_K_M.gguf", "dflash-Q4_K_M.gguf")
    models = _scan(root)
    assert all(m.draft_gguf is None for m in models)


def test_drafter_pairs_into_a_model_already_in_the_config(tmp_path, monkeypatch):
    # The scan skips a configured file, so the drafter would have no candidate
    # without `known_models`. The pairing goes to stats for the caller to write.
    _drafter_classify(monkeypatch)
    root = _write(tmp_path, "Muse-Glimmer-30B-Q4_K_M.gguf",
                  "dflash-Muse-Glimmer-30B-Q4_K_M.gguf")
    target = str(root / "Muse-Glimmer-30B-Q4_K_M.gguf")
    mc = ModelCfg(id="muse-30b", path=target)
    spec = DiscoverSpec(dir=str(root), recursive=False, pair_mmproj=True,
                        speculative="auto")
    stats = {}
    models = disc.scan_dirs([spec], [str(root)], known_paths={target},
                            known_models={target: mc}, stats=stats)
    assert models == []                          # the target stays configured
    assert stats["draft_pairs"] == {
        "muse-30b": str(root / "dflash-Muse-Glimmer-30B-Q4_K_M.gguf")}
    assert mc.draft_gguf and mc.speculative is True


def test_drafter_without_a_hidden_size_pairs_with_its_lone_target(
        tmp_path, monkeypatch):
    # A DSpark sidecar declares no hidden size, and its filename shares too
    # little with the target quant to pass the name test. The arch table plus
    # a single candidate still settle it (llama.cpp's own sidecar layout).
    def _f(path):
        name = os.path.basename(path)
        meta = ({"general.architecture": "deepseek4-dspark"}
                if "DSpark" in name else {"general.architecture": "deepseek4"})
        return disc._classify_meta(meta, basename=name, path=path)

    monkeypatch.setattr(disc, "classify_gguf", _f)
    root = _write(tmp_path, "DeepSeek-V4-Flash-0731-UD-IQ3_XXS.gguf",
                  "DeepSeek-V4-Flash-0731-DSpark-MXFP4-Q8_0.gguf")
    models = _scan(root)
    assert len(models) == 1
    assert models[0].draft_gguf.endswith("DSpark-MXFP4-Q8_0.gguf")


def test_drafter_unpaired_across_model_families(tmp_path, fake_classify):
    # A gemma4 assistant cannot draft for a qwen target: the arch table says
    # which model types each drafter arch serves.
    root = _write(tmp_path, "Qwen3.6-27B-Q4_K_S.gguf",
                  "gemma-4-31B-it-assistant.Q8_0.gguf")
    by_id = {m.id: m for m in _scan(root)}
    assert by_id["qwen3.6-27b-q4"].draft_gguf is None


def test_streamed_model_gets_no_drafter():
    # MTP needs a fully resident base, thus an over-RAM entry stays plain.
    mc = ModelCfg(id="m-q4", path="/m/M-Q4_K_M.gguf", stream="experts")
    target = disc.ClassifiedGguf("/m/M-Q4_K_M.gguf", "model", "muse-glimmer",
                                 False, "Q4_K_M", True, n_embd=6656)
    drafter = disc.ClassifiedGguf("/m/dflash-M-Q4_K_M.gguf", "drafter", "dflash",
                                  False, "Q4_K_M", False, n_embd=6656)
    assert disc._drafter_targets(drafter, [(mc, target)]) == []


def test_scan_adapter_not_standalone(tmp_path, monkeypatch, capsys):
    # The documented `adapter:` shape wants the adapter GGUF under model_dirs,
    # so the scan must announce-and-drop it, never register it as a chat model.
    def _f(path):
        name = os.path.basename(path).lower()
        meta = {"general.architecture": "qwen3"}
        if "lora" in name:
            meta["general.type"] = "adapter"
        return disc._classify_meta(meta, basename=os.path.basename(path),
                                    path=path)

    monkeypatch.setattr(disc, "classify_gguf", _f)
    root = _write(tmp_path, "base-Q4_K_M.gguf", "my-lora.gguf")
    models = _scan(root)
    assert [m.id for m in models] == ["base-q4"]
    assert "LoRA adapter - not a chat model" in capsys.readouterr().err


def test_scan_compact_codec_when_unique(tmp_path, fake_classify):
    # Distinct base codecs in one base -> each keeps the compact form, no clash.
    root = _write(tmp_path, "Llama-3.2-1B-Instruct-IQ2_M.gguf",
                  "Llama-3.2-1B-Instruct-IQ3_M.gguf",
                  "Llama-3.2-1B-Instruct-IQ4_M.gguf")
    ids = {m.id for m in _scan(root)}
    assert ids == {"llama-3.2-1b-instruct-iq2", "llama-3.2-1b-instruct-iq3",
                   "llama-3.2-1b-instruct-iq4"}


def test_scan_collision_uses_full_codec(tmp_path, fake_classify):
    # Two quants sharing a compact codec (both Q4*) -> full codec on BOTH, none bare.
    root = _write(tmp_path, "Qwen3.6-27B-Q4_K_S.gguf", "Qwen3.6-27B-Q4_K.gguf")
    ids = {m.id for m in _scan(root)}
    assert ids == {"qwen3.6-27b-q4-k-s", "qwen3.6-27b-q4-k"}
    assert "qwen3.6-27b" not in ids               # no asymmetric bare member


def test_scan_skips_non_first_shards(tmp_path, fake_classify):
    root = _write(tmp_path, "Big-Q6_K-00001-of-00002.gguf",
                  "Big-Q6_K-00002-of-00002.gguf")
    models = _scan(root)
    assert len(models) == 1                       # only the first shard is a model
    assert models[0].id == "big-q6"


def test_scan_dedupes_against_known(tmp_path, fake_classify):
    root = _write(tmp_path, "Qwen3.6-27B-Q4_K_S.gguf")
    spec = DiscoverSpec(dir=str(root))
    known_path = str(root / "Qwen3.6-27B-Q4_K_S.gguf")
    models = disc.scan_dirs([spec], [str(root)], known_paths={known_path})
    assert models == []                           # already configured -> skipped


def test_scan_dedupes_overlapping_roots(tmp_path, fake_classify):
    """`discover: [{dir: null}, {dir: <sub>}]` over overlapping roots used to
    register the same GGUF twice (x-q4 and x-q4-2), double-counting residency."""
    sub = tmp_path / "sub"
    sub.mkdir()
    _write(sub, "Qwen3.6-27B-Q4_K_S.gguf")
    specs = [DiscoverSpec(dir=str(tmp_path), recursive=True),
             DiscoverSpec(dir=str(sub), recursive=False)]
    models = disc.scan_dirs(specs, [str(tmp_path)])
    assert len(models) == 1
    assert len({m.path for m in models}) == 1


def test_scan_dedupes_repeated_model_dirs(tmp_path, fake_classify):
    """A `dir: null` spec resolving to overlapping model_dirs dedupes too."""
    sub = tmp_path / "sub"
    sub.mkdir()
    _write(sub, "Qwen3.6-27B-Q4_K_S.gguf")
    spec = DiscoverSpec(dir=None, recursive=True)
    models = disc.scan_dirs([spec], [str(tmp_path), str(sub)])
    assert len(models) == 1


def test_scan_multi_model_mmproj_prefix_match(tmp_path, fake_classify):
    root = _write(tmp_path, "alpha-vision-Q4_K.gguf", "beta-Q4_K.gguf",
                  "mmproj-alpha-vision-bf16.gguf")
    by_id = {m.id: m for m in _scan(root)}
    assert by_id["alpha-vision-q4"].mmproj is not None   # prefix-matched to alpha
    assert by_id["beta-q4"].mmproj is None


def test_scan_mmproj_not_paired_on_weak_prefix(tmp_path, fake_classify):
    # A named projector must really match - a bare "qwen" family overlap between a
    # Qwen2-VL projector and a Qwen3 chat model is not enough (the mis-pair bug).
    root = _write(tmp_path, "qwen3-chat-Q4_K_M.gguf", "gemma-foo-Q4_K_M.gguf",
                  "mmproj-Qwen2-VL-2B-Instruct-f16.gguf")
    by_id = {m.id: m for m in _scan(root)}
    assert all(m.mmproj is None for m in by_id.values())


def test_scan_excludes_embedding_and_reranker(tmp_path, fake_classify):
    root = _write(tmp_path, "Qwen3-Embedding-0.6B-Q4_K_M.gguf",
                  "Qwen3-Reranker-0.6B-Q4_K_M.gguf", "real-chat-Q4_K_M.gguf")
    ids = {m.id for m in _scan(root)}
    assert ids == {"real-chat-q4"}                  # embedder/reranker not chat models


def test_scan_embedding_with_stray_mmproj_no_pairing(tmp_path, fake_classify):
    # The original bug: a lone embedding GGUF beside an unrelated projector.
    root = _write(tmp_path, "Qwen3-Embedding-0.6B-Q4_K_M.gguf",
                  "mmproj-Qwen2-VL-2B-Instruct-f16.gguf")
    assert _scan(root) == []                         # embedder excluded; nothing to pair


# scaffold_yaml - parses back through the loader
def test_scaffold_round_trips_through_build_config():
    import yaml
    models = [
        ModelCfg(id="qwen3.6-27b", path="/models/qwen3.6-27b/m-Q4_K_S.gguf",
                 speculative=True),
        ModelCfg(id="gemma-e4b-vlm", path="/models/gemma/llm-Q6_K.gguf",
                 mmproj="/models/gemma/mmproj-bf16.gguf"),
    ]
    text = disc.scaffold_yaml(models, model_dirs=["/models"])
    cfg = build_config(yaml.safe_load(text))
    assert set(cfg.models) == {"qwen3.6-27b", "gemma-e4b-vlm"}
    assert cfg.models["qwen3.6-27b"].speculative is True
    assert cfg.models["gemma-e4b-vlm"].mmproj.endswith("mmproj-bf16.gguf")
    # Paths rendered relative to the model_dirs root.
    assert cfg.models["qwen3.6-27b"].path == "qwen3.6-27b/m-Q4_K_S.gguf"


def test_scaffold_writes_a_paired_drafter():
    import yaml
    models = [ModelCfg(id="muse-glimmer-30b-q4",
                       path="/models/muse/m-Q4_K_M.gguf",
                       draft_gguf="/models/muse/dflash-m-Q4_K_M.gguf",
                       speculative=True)]
    text = disc.scaffold_yaml(models, model_dirs=["/models"])
    cfg = build_config(yaml.safe_load(text))
    mc = cfg.models["muse-glimmer-30b-q4"]
    assert mc.draft_gguf == "muse/dflash-m-Q4_K_M.gguf"
    assert mc.speculative is False        # the drafter key turns it on at load
    # no live `speculative` key: the reference block keeps the only mention
    assert not [ln for ln in text.splitlines() if ln.strip() == "speculative: true"]


def test_scaffold_anchors_relative_model_dirs(tmp_path, monkeypatch):
    # A cwd-relative root would resolve against the SERVER's cwd later (launchd
    # runs at /), silently serving zero models - the scaffold anchors it. `~`
    # and `$VAR` forms stay verbatim (they expand at load, portably).
    monkeypatch.chdir(tmp_path)
    text = disc.scaffold_yaml([], model_dirs=["models", "~/m", "/abs/dir"])
    roots = [ln.strip()[2:] for ln in text.splitlines()
             if ln.strip().startswith("- ")]
    assert roots[0] == os.path.join(os.getcwd(), "models")
    assert os.path.isabs(roots[0])
    assert roots[1] == "~/m"
    assert roots[2] == "/abs/dir"


def test_scaffold_empty_is_valid():
    import yaml
    text = disc.scaffold_yaml([], model_dirs=["~/llm/gguf"])
    cfg = build_config(yaml.safe_load(text))     # must not raise
    assert cfg.models == {}
    # No generated profile / defaults.profile - the family base layer is the
    # default sampling now; the profiles section is documentation only.
    assert cfg.profiles == {}
    assert cfg.defaults.profile is None
    # Default scaffold ships the prompt cache on (disk tier off).
    assert cfg.cache["enabled"] is True


def test_scaffold_family_comments_above_entry():
    """Detected families render as a comment line above the entry with the
    model-card base sampling; an unknown family renders no comment."""
    models = [
        ModelCfg(id="qw", path="/m/qw.gguf", family="qwen3.6"),
        ModelCfg(id="gm", path="/m/gm.gguf", family="gemma"),
        ModelCfg(id="mystery", path="/m/x.gguf"),
    ]
    text = disc.scaffold_yaml(models, model_dirs=["/m"])
    assert "  # sampling (qwen3.6): t=1 top_p=0.95 top_k=20\n  qw:\n" in text
    assert "  # sampling (gemma): t=1 top_p=0.95 top_k=64\n  gm:\n" in text
    assert "  mystery:\n" in text                # no family -> bare key


def _uncomment_hints(text: str) -> str:
    """Strip the leading `# ` from every commented option-example line (a line
    whose content looks like `key: ...` or a `- ` sequence item), leaving prose
    and placeholder examples (`<id>`, `{...}`) commented - the transformation a
    user applies when enabling a documented knob."""
    import re
    key_rx = re.compile(r"^ *(- |[A-Za-z0-9_.-]+: ?)")
    out = []
    for line in text.splitlines():
        stripped = line.lstrip(" ")
        indent = line[: len(line) - len(stripped)]
        if stripped.startswith("# "):
            content = stripped[2:]
            if "<" not in content and "{...}" not in content \
                    and key_rx.match(content):
                out.append(indent + content)
                continue
        out.append(line)
    return "\n".join(out)


def test_scaffold_hints_uncomment_and_parse():
    """Every commented option example in the scaffold is real: uncommenting
    them all still parses + validates through build_config."""
    import yaml
    models = [ModelCfg(id="qw", path="/m/qw.gguf", family="qwen3.6")]
    text = _uncomment_hints(disc.scaffold_yaml(models, model_dirs=["/m"]))
    cfg = build_config(yaml.safe_load(text))     # must not raise
    # The uncommented hints landed as live keys.
    assert cfg.api_key == "change-me"
    assert cfg.budget_gb == 96
    assert cfg.max_models == 2
    assert cfg.family_defaults is True
    assert cfg.token_queue_timeout_s == 600
    assert cfg.cache["enabled"] is True
    assert set(cfg.profiles) == {"brief", "my-coding", "narrator"}
    assert cfg.profiles["my-coding"].extends == "coding"
    assert cfg.rules[0].match == "*-coder-*"


def test_scaffold_emits_rerank_when_set():
    import yaml
    text = disc.scaffold_yaml([], model_dirs=["/m"],
                              embeddings="qwen3-embed-0.6b",
                              rerank="qwen3-rerank-0.6b")
    cfg = build_config(yaml.safe_load(text))
    assert cfg.embeddings == "qwen3-embed-0.6b"
    assert cfg.rerank == "qwen3-rerank-0.6b"


def test_find_retrieval_models_filters_to_decoder_archs(monkeypatch):
    """Only decoder-arch (supported) embedder / reranker GGUFs are returned; chat
    models and encoder-arch retrieval GGUFs are dropped."""
    from gmlx.discovery import ClassifiedGguf
    files = {
        "/m/qwen3-embed.gguf":
            ClassifiedGguf("/m/qwen3-embed.gguf", "embedding", "qwen3", False, "Q8_0", False),
        "/m/qwen3-rerank.gguf":
            ClassifiedGguf("/m/qwen3-rerank.gguf", "reranker", "qwen3", False, "Q8_0", False),
        "/m/bge-m3.gguf":            # encoder arch - not decoder-servable
            ClassifiedGguf("/m/bge-m3.gguf", "embedding", "bert", False, "Q8_0", False),
        "/m/chat.gguf":              # a plain chat model - not retrieval
            ClassifiedGguf("/m/chat.gguf", "model", "llama", False, "Q4_K_M", True),
    }
    monkeypatch.setattr(disc, "_iter_gguf_files", lambda root, recursive: list(files))
    monkeypatch.setattr(disc, "classify_gguf", lambda p: files[p])
    monkeypatch.setattr(disc, "supported_arches", lambda: {"qwen3", "llama"})

    emb, rr = disc.find_retrieval_models(["/m"])
    assert [c.path for c in emb] == ["/m/qwen3-embed.gguf"]
    assert [c.path for c in rr] == ["/m/qwen3-rerank.gguf"]


def test_find_retrieval_models_surfaces_supported_encoder(monkeypatch):
    """An encoder embedder we CAN build (gemma-embedding, now in supported_arches)
    is surfaced; an encoder we can't (bert) is still dropped."""
    from gmlx.discovery import ClassifiedGguf
    files = {
        "/m/egemma.gguf":
            ClassifiedGguf("/m/egemma.gguf", "embedding", "gemma-embedding",
                           False, "Q8_0", False),
        "/m/bge-m3.gguf":            # encoder arch we can't build -> dropped
            ClassifiedGguf("/m/bge-m3.gguf", "embedding", "bert", False, "Q8_0", False),
    }
    monkeypatch.setattr(disc, "_iter_gguf_files", lambda root, recursive: list(files))
    monkeypatch.setattr(disc, "classify_gguf", lambda p: files[p])
    monkeypatch.setattr(disc, "supported_arches", lambda: {"gemma-embedding"})

    emb, rr = disc.find_retrieval_models(["/m"])
    assert [c.path for c in emb] == ["/m/egemma.gguf"]
    assert rr == []


def test_scaffold_disk_cache_writes_active_block():
    """`init --disk-cache` emits a live cache block (enabled + SSD tier), not comments."""
    import yaml
    text = disc.scaffold_yaml([], model_dirs=["~/llm/gguf"], disk_cache=True)
    cfg = build_config(yaml.safe_load(text))     # must not raise
    assert cfg.cache["enabled"] is True
    assert cfg.cache["disk"]["path"].endswith("gmlx/apc")


def test_scaffold_default_enables_cache_disk_off():
    """The default scaffold ships the prompt cache on, with the disk tier an
    explicit `disk: false` (no commented-out cache block)."""
    import yaml
    text = disc.scaffold_yaml([], model_dirs=["~/llm/gguf"])
    cfg = build_config(yaml.safe_load(text))
    assert cfg.cache["enabled"] is True
    assert cfg.cache["disk"] == {"path": None}   # normalized `disk: false`


# model_to_entry - the per-model dict `sync-models` splices into a config
def test_model_to_entry_relative_path_only():
    mc = ModelCfg(id="qwen", path="/models/sub/qwen-Q4_K_S.gguf")
    assert disc.model_to_entry(mc, ["/models"]) == {"path": "sub/qwen-Q4_K_S.gguf"}


def test_model_to_entry_includes_mmproj_and_speculative():
    mc = ModelCfg(id="g", path="/models/llm-Q6_K.gguf",
                  mmproj="/models/mmproj-bf16.gguf", speculative=True)
    assert disc.model_to_entry(mc, ["/models"]) == {
        "path": "llm-Q6_K.gguf",
        "mmproj": "mmproj-bf16.gguf",
        "speculative": True,
    }


def test_model_to_entry_draft_gguf_implies_speculative():
    # `draft_gguf` turns speculative decoding on by itself, so the entry keeps
    # no second `speculative` key.
    mc = ModelCfg(id="g", path="/models/llm-Q6_K.gguf",
                  draft_gguf="/models/dflash-llm-Q4_K_M.gguf", speculative=True)
    assert disc.model_to_entry(mc, ["/models"]) == {
        "path": "llm-Q6_K.gguf",
        "draft_gguf": "dflash-llm-Q4_K_M.gguf",
    }


def test_model_to_entry_absolute_when_outside_model_dirs():
    mc = ModelCfg(id="x", path="/elsewhere/x-Q4_0.gguf")
    assert disc.model_to_entry(mc, ["/models"]) == {"path": "/elsewhere/x-Q4_0.gguf"}


def test_model_to_entry_hf_ref_passthrough():
    mc = ModelCfg(id="x", path="hf:org/repo/x-Q4_K_M.gguf")
    assert disc.model_to_entry(mc, ["/models"]) == {
        "path": "hf:org/repo/x-Q4_K_M.gguf"}


# scan_hf_cache - emit portable hf: entries from a (monkeypatched) HF cache
def _repo(repo_id, files, *, refs=("main",), repo_type="model"):
    # SimpleNamespace is unhashable, so model the cache structure with plain lists
    # (the scanner only iterates repos / revisions / files - never set-membership).
    from types import SimpleNamespace
    rev = SimpleNamespace(
        refs=frozenset(refs), commit_hash="cafef00d", last_modified=1.0,
        files=[SimpleNamespace(file_name=fn, file_path=f"/cache/{repo_id}/{fn}")
               for fn in files])
    return SimpleNamespace(repo_id=repo_id, repo_type=repo_type, revisions=[rev])


def _fake_scan(monkeypatch, repos):
    import huggingface_hub
    from types import SimpleNamespace
    monkeypatch.setattr(huggingface_hub, "scan_cache_dir",
                        lambda: SimpleNamespace(repos=list(repos)))

    def _classify(path):
        import os
        name = os.path.basename(path).lower()
        meta = ({"general.architecture": "clip"} if name.startswith("mmproj")
                else {"general.architecture": "qwen35"})    # supported, non-MTP
        return disc._classify_meta(meta, basename=os.path.basename(path), path=path)

    monkeypatch.setattr(disc, "classify_gguf", _classify)


def test_scan_hf_cache_emits_hf_ref(monkeypatch):
    _fake_scan(monkeypatch, [_repo("org/Model-GGUF", ["model-Q4_K_M.gguf"])])
    models = disc.scan_hf_cache()
    assert len(models) == 1
    assert models[0].id == "model-q4"
    assert models[0].path == "hf:org/Model-GGUF/model-Q4_K_M.gguf"


def test_scan_hf_cache_pairs_mmproj(monkeypatch):
    _fake_scan(monkeypatch, [
        _repo("org/VLM-GGUF", ["llm-Q4_K_M.gguf", "mmproj-F16.gguf"])])
    models = disc.scan_hf_cache()
    assert len(models) == 1                         # mmproj is a companion
    assert models[0].path == "hf:org/VLM-GGUF/llm-Q4_K_M.gguf"
    assert models[0].mmproj == "hf:org/VLM-GGUF/mmproj-F16.gguf"


def test_scan_hf_cache_non_main_revision_suffix(monkeypatch):
    _fake_scan(monkeypatch, [
        _repo("org/Old-GGUF", ["old-Q4_K_M.gguf"], refs=("v1",))])
    models = disc.scan_hf_cache()
    assert models[0].path == "hf:org/Old-GGUF/old-Q4_K_M.gguf@v1"


def test_scan_hf_cache_first_shard_only(monkeypatch):
    _fake_scan(monkeypatch, [_repo("org/Big-GGUF", [
        "big-Q6_K-00001-of-00002.gguf", "big-Q6_K-00002-of-00002.gguf"])])
    models = disc.scan_hf_cache()
    assert len(models) == 1
    assert models[0].path == "hf:org/Big-GGUF/big-Q6_K-00001-of-00002.gguf"


def test_scan_hf_cache_dedupes_known_refs(monkeypatch):
    _fake_scan(monkeypatch, [_repo("org/Model-GGUF", ["model-Q4_K_M.gguf"])])
    models = disc.scan_hf_cache(
        known_refs={"hf:org/Model-GGUF/model-Q4_K_M.gguf"})
    assert models == []


def test_scan_hf_cache_skips_non_model_repos(monkeypatch):
    _fake_scan(monkeypatch, [
        _repo("org/Dataset", ["d-Q4_K_M.gguf"], repo_type="dataset")])
    assert disc.scan_hf_cache() == []


# header_meta - the two-tier (memo + JSON) header cache behind family
# detection and run/chat MTP auto-detection.
@pytest.fixture()
def _hm(monkeypatch, tmp_path):
    """Isolated caches: fresh memo/disk state, XDG cache under tmp_path, and a
    counting _read_header stub returning a gemma4 model header."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(disc, "_HEADER_MEMO", {})
    monkeypatch.setattr(disc, "_HEADER_DISK", None)
    calls = {"n": 0}

    def _fake_read(path):
        calls["n"] += 1
        c = disc._classify_meta(
            {"general.architecture": "gemma4"},
            basename=os.path.basename(path), path=path)
        return c, "Gemma 4 12B It", {}

    monkeypatch.setattr(disc, "_read_header", _fake_read)
    return calls


def test_header_meta_missing_file_is_silent(tmp_path, capsys):
    assert disc.header_meta(str(tmp_path / "absent.gguf")) is None
    assert capsys.readouterr().err == ""


def test_header_meta_reads_and_memoizes(_hm, tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"x")
    meta = disc.header_meta(str(f))
    assert meta == {"arch": "gemma4", "name": "Gemma 4 12B It",
                    "kind": "model", "mtp": False, "sampling": {},
                    "drafter_kind": None}
    disc.header_meta(str(f))
    assert _hm["n"] == 1                         # second hit is the memo


def test_header_meta_disk_cache_survives_new_process(_hm, tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"x")
    disc.header_meta(str(f))
    assert _hm["n"] == 1
    # Simulate a new process: drop the memo, keep the JSON on disk.
    disc._HEADER_MEMO.clear()
    disc._HEADER_DISK = None
    meta = disc.header_meta(str(f))
    assert meta["arch"] == "gemma4"
    assert _hm["n"] == 1                         # stat-only, no re-read


def test_header_meta_invalidates_on_mtime_size_change(_hm, tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"x")
    disc.header_meta(str(f))
    disc._HEADER_MEMO.clear()
    disc._HEADER_DISK = None
    f.write_bytes(b"xy")                         # size change
    disc.header_meta(str(f))
    assert _hm["n"] == 2


def test_header_meta_unreadable_is_silent_none(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(disc, "_HEADER_MEMO", {})
    monkeypatch.setattr(disc, "_HEADER_DISK", None)

    def _boom(path):
        raise ValueError("not a gguf")

    monkeypatch.setattr(disc, "_read_header", _boom)
    f = tmp_path / "junk.gguf"
    f.write_bytes(b"junk")
    assert disc.header_meta(str(f)) is None
    assert capsys.readouterr().err == ""


def test_header_meta_corrupt_disk_cache_ignored(_hm, tmp_path):
    p = tmp_path / "xdg" / "gmlx" / "header-meta.json"
    p.parent.mkdir(parents=True)
    p.write_text("{not json")
    f = tmp_path / "m.gguf"
    f.write_bytes(b"x")
    assert disc.header_meta(str(f))["arch"] == "gemma4"


def test_scan_pairs_llamacpp_default_mmproj_name(tmp_path, fake_classify):
    """llama.cpp's conversion default mmproj-model-f16.gguf leaves a residual
    core of 'model' - a generic projector, pairing to the sole model."""
    root = _write(tmp_path, "llava-v1.6-7b-Q4_K_S.gguf",
                  "mmproj-model-f16.gguf")
    models = _scan(root)
    assert len(models) == 1
    assert models[0].mmproj.endswith("mmproj-model-f16.gguf")


# MTP wiring gate + memory-fit placement (over-RAM models)
def test_expert_count_sets_moe():
    c = _classify({"general.architecture": "qwen35moe",
                   "qwen35moe.expert_count": 128}, "m-Q4_K_M.gguf")
    assert c.moe is True


def test_dense_model_not_moe():
    c = _classify({"general.architecture": "qwen35"}, "m-Q4_K_M.gguf")
    assert c.moe is False


def _classify_as(arch, extra=None):
    """A classify_gguf stand-in returning one fixed arch (plus extra KV)."""
    def _f(path):
        meta = {"general.architecture": arch, **(extra or {})}
        return disc._classify_meta(meta, basename=os.path.basename(path),
                                   path=path)
    return _f


def test_scan_mtp_head_on_unwired_arch_stays_plain(tmp_path, monkeypatch, capsys):
    # GLM-5.2: an MTP/nextn head in the GGUF, but no MTP target class wired for
    # glm-dsa -> `speculative: true` would fail at load, so it must stay off.
    monkeypatch.setattr(disc, "classify_gguf", _classify_as(
        "glm-dsa", {"glm-dsa.nextn_predict_layers": 1,
                    "glm-dsa.expert_count": 160}))
    root = _write(tmp_path, "GLM-5.2-UD-IQ3_XXS.gguf")
    models = _scan(root)
    assert models[0].speculative is False
    assert "no MTP target wired" in capsys.readouterr().err


def test_scan_wired_mtp_arch_keeps_speculative(tmp_path, monkeypatch):
    monkeypatch.setattr(disc, "classify_gguf", _classify_as(
        "hy_v3", {"hy_v3.nextn_predict_layers": 1, "hy_v3.expert_count": 64}))
    root = _write(tmp_path, "Hy3-IQ4_XS.gguf")
    models = _scan(root)
    assert models[0].speculative is True
    assert models[0].stream is None               # tiny file: fits in RAM


def test_scan_overram_moe_streams_experts_and_drops_mtp(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(disc, "classify_gguf", _classify_as(
        "hy_v3", {"hy_v3.nextn_predict_layers": 1, "hy_v3.expert_count": 64}))
    monkeypatch.setattr(disc, "_physmem_bytes", lambda: 128 * 10**9)
    monkeypatch.setattr(disc, "_gguf_set_bytes", lambda p: 200 * 10**9)
    root = _write(tmp_path, "Hy3-IQ4_XS.gguf")
    m = _scan(root)[0]
    assert m.stream == "experts"
    assert m.speculative is False                 # MTP needs a resident base
    err = capsys.readouterr().err
    assert "over-RAM MoE" in err and "speculative off" in err


def test_scan_overram_dense_gets_hint_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(disc, "classify_gguf", _classify_as("llama"))
    monkeypatch.setattr(disc, "_physmem_bytes", lambda: 64 * 10**9)
    monkeypatch.setattr(disc, "_gguf_set_bytes", lambda p: 100 * 10**9)
    root = _write(tmp_path, "Big-Dense-Q8_0.gguf")
    m = _scan(root)[0]
    assert m.stream is None                       # dense: streaming stays opt-in
    assert "stream: cpu" in capsys.readouterr().err


def test_gguf_set_bytes_sums_shards(tmp_path):
    (tmp_path / "m-Q4-00001-of-00002.gguf").write_bytes(b"x" * 10)
    (tmp_path / "m-Q4-00002-of-00002.gguf").write_bytes(b"y" * 7)
    assert disc._gguf_set_bytes(str(tmp_path / "m-Q4-00001-of-00002.gguf")) == 17


def test_model_to_entry_includes_stream():
    mc = ModelCfg(id="x", path="/models/x.gguf", stream="experts")
    assert disc.model_to_entry(mc, ["/models"])["stream"] == "experts"


def test_scaffold_stream_round_trips_through_build_config():
    import yaml
    models = [ModelCfg(id="hy3-iq4", path="/models/Hy3-IQ4_XS.gguf",
                       stream="experts")]
    text = disc.scaffold_yaml(models, model_dirs=["/models"])
    cfg = build_config(yaml.safe_load(text))
    assert cfg.models["hy3-iq4"].stream == "experts"


# GGUF-embedded model-card sampling (general.sampling.*)
def test_read_sampling_maps_header_keys():
    kv = {"general.sampling.temp": 0.7, "general.sampling.top_p": 0.8,
          "general.sampling.top_k": 20}
    assert disc._read_sampling(kv) == {"temperature": 0.7, "top_p": 0.8,
                                       "top_k": 20}


def test_read_sampling_absent_is_empty():
    assert disc._read_sampling({"general.architecture": "qwen3"}) == {}


def test_family_comment_merges_gguf_sampling(monkeypatch):
    monkeypatch.setattr(disc, "header_sampling",
                        lambda p: {"temperature": 0.7, "top_p": 0.8})
    mc = ModelCfg(id="x", path="/m/x.gguf", family="qwen3")
    assert disc.family_comment(mc) == \
        "sampling (qwen3 + gguf): t=0.7 top_p=0.8 top_k=20"


def test_family_comment_gguf_only_without_family(monkeypatch):
    monkeypatch.setattr(disc, "header_sampling", lambda p: {"temperature": 0.9})
    mc = ModelCfg(id="x", path="/m/x.gguf")
    assert disc.family_comment(mc) == "sampling (gguf): t=0.9"


def test_read_sampling_normalizes_disabled_sentinels():
    kv = {"general.sampling.temp": 0.9, "general.sampling.top_p": 1.0,
          "general.sampling.top_k": -1, "general.sampling.repeat_penalty": 1.0}
    assert disc._read_sampling(kv) == {"temperature": 0.9, "top_p": 0.0,
                                       "top_k": 0}


# muse-glimmer: the per-family arch tuple, and the "dflash" id marker


def test_muse_glimmer_companion_is_found_by_arch_tuple(tmp_path, monkeypatch):
    """A muse-glimmer target asks for ``dflash`` only - the deepseek4 arches
    are not in its tuple, so a dspark sidecar next door is not picked up."""
    target = tmp_path / "Muse-Glimmer-30B-Q6_K_L.gguf"
    for name in ("dflash-kquant.gguf", target.name):
        (tmp_path / name).write_bytes(b"GGUF")
    metas = {str(tmp_path / "dflash-kquant.gguf"): {"arch": "dflash"},
             str(target): {"arch": "muse-glimmer"}}
    monkeypatch.setattr(disc, "header_meta", lambda p: metas.get(str(p)))
    assert disc.find_mtp_companion(str(target), ("dflash",)) == str(
        tmp_path / "dflash-kquant.gguf")


def test_muse_glimmer_companion_ignores_a_native_dspark_sidecar(
        tmp_path, monkeypatch):
    target = tmp_path / "Muse-Glimmer-30B-Q6_K_L.gguf"
    for name in ("dspark-sidecar.gguf", target.name):
        (tmp_path / name).write_bytes(b"GGUF")
    metas = {str(tmp_path / "dspark-sidecar.gguf"): {"arch": "deepseek4-dspark"},
             str(target): {"arch": "muse-glimmer"}}
    monkeypatch.setattr(disc, "header_meta", lambda p: metas.get(str(p)))
    assert disc.find_mtp_companion(str(target), ("dflash",)) is None


def test_dflash_is_an_id_marker():
    """Without this the drafter quant splits the model id and a dflash sidecar
    is mistaken for a separate model."""
    assert "dflash" in disc._ID_MARKERS


def test_model_classification_carries_general_name_for_family_refinement():
    """Kimi K2.x and DeepSeek share the 'deepseek2' arch, so the generated
    config's family is only right if the classifier hands general.name to
    detect_family. Without it every K2 GGUF would inherit DeepSeek's card."""
    c = _classify({"general.architecture": "deepseek2",
                   "general.name": "Kimi-K2.7-Code",
                   "deepseek2.expert_count": 384}, "kimi-k2.7-UD-Q2_K_XL.gguf")
    assert c.kind == "model"
    assert c.name == "Kimi-K2.7-Code"

    from gmlx.profiles import detect_family
    assert detect_family(c.arch, c.name) == "kimi-k2"
