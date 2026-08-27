"""Discovery pairs DFlash 2 drafters by the base model their header declares,
across directories; same-directory filename pairing is unchanged."""

import os

import pytest

from gmlx import arch_table
from gmlx import discovery as disc
from gmlx.config import DiscoverSpec

# Header facts per file stem, mirroring the on-disk layout
# (~/llm/gguf/<publisher>__<repo>/). The v1 Muse drafters declare only a
# general.name; the meta-models target declares no basename at all.
HEADERS = {
    "Qwen3.8-27B-UD-Q6_K_XL": {
        "general.architecture": "qwen35", "general.basename": "Qwen3.8-27B",
        "general.name": "Qwen3.8-27B", "general.base_model.0.name": "Qwen3.8 27B",
        "qwen35.nextn_predict_layers": 1, "qwen35.embedding_length": 5120},
    "Qwen3.8-27B-DFlash2-Q8_0": {
        "general.architecture": "dflash", "general.basename": "Qwen3.8-27B",
        "general.size_label": "1.9B", "general.base_model.0.name": "Qwen3.8 27B",
        "general.name": "Qwen3.8-27B-DFlash2", "dflash.selector_top_k": 16,
        "dflash.embedding_length": 5120},
    "Muse-Glimmer-30B-Q6_K_L": {
        "general.architecture": "muse-glimmer", "general.basename": "Muse-Glimmer",
        "general.size_label": "30B", "general.name": "Muse Glimmer 30B",
        "muse-glimmer.embedding_length": 6656},
    "dflash-kquant": {
        "general.architecture": "dflash", "general.name": "Hf_Museglimmer",
        "dflash.embedding_length": 6656},
    "Muse-Glimmer-30B-DFlash2-Q8_0": {
        "general.architecture": "dflash", "general.basename": "Muse-Glimmer-30B",
        "general.base_model.0.name": "Muse Glimmer 30B",
        "general.name": "Muse-Glimmer-30B-DFlash2", "dflash.selector_top_k": 16,
        "dflash.embedding_length": 6656},
    "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M": {
        "general.architecture": "muse-glimmer", "general.name": "Muse Glimmer Hf",
        "general.size_label": "28B", "muse-glimmer.embedding_length": 6656},
    "dflash-Muse-Glimmer-30B-Q4_K_M": {
        "general.architecture": "dflash", "general.name": "Hf_Museglimmer",
        "dflash.embedding_length": 6656},
    "Qwen3.5-27B-DFlash-Q8_0": {
        "general.architecture": "dflash", "general.basename": "Qwen3.5-27B",
        "general.base_model.0.name": "Qwen3.5 27B",
        "general.name": "Qwen3.5-27B-DFlash", "dflash.embedding_length": 5120},
}
LAYOUT = {
    "unsloth__Qwen3.8-27B-GGUF": ["Qwen3.8-27B-UD-Q6_K_XL"],
    "incoai__Qwen3.8-27B-DFlash2-GGUF": ["Qwen3.8-27B-DFlash2-Q8_0"],
    "bartowski__Muse-Glimmer-30B-GGUF": ["Muse-Glimmer-30B-Q6_K_L", "dflash-kquant"],
    "incoai__Muse-Glimmer-30B-DFlash2-GGUF": ["Muse-Glimmer-30B-DFlash2-Q8_0"],
    "meta-models__Muse-Glimmer-30B-GGUF": [
        "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M", "dflash-Muse-Glimmer-30B-Q4_K_M"],
    "zlab__Qwen3.5-27B-DFlash-GGUF": ["Qwen3.5-27B-DFlash-Q8_0"],
}


@pytest.fixture
def fake_classify(monkeypatch):
    def _f(path):
        stem = os.path.basename(path)[:-len(".gguf")]
        return disc._classify_meta(dict(HEADERS[stem]), basename=os.path.basename(path),
                                   path=path)

    monkeypatch.setattr(disc, "classify_gguf", _f)
    monkeypatch.setattr(disc, "_fit_in_memory", lambda mc, c: None)
    return _f


def _layout(root, prefix=None):
    """Write the mirrored tree; ``prefix`` maps dir -> lexical prefix to flip
    the scan order."""
    for d, stems in LAYOUT.items():
        sub = root / ((prefix or {}).get(d, "") + d)
        sub.mkdir()
        for stem in stems:
            (sub / f"{stem}.gguf").write_bytes(b"x")
    return root


def _scan(root):
    spec = DiscoverSpec(dir=str(root), recursive=True, pair_mmproj=True,
                        speculative="auto")
    return {m.id: m for m in disc.scan_dirs([spec], [str(root)])}


def _base(path):
    return os.path.basename(path) if path else None


def test_dflash2_marker_and_declared_ids():
    assert disc.derive_id("Qwen3.8-27B-DFlash2-Q8_0.gguf") == ("qwen3.8-27b", "Q8_0")
    assert disc._declared_base_ids(HEADERS["Qwen3.8-27B-DFlash2-Q8_0"]) == ("qwen3.8-27b",)
    assert disc._declared_base_ids(HEADERS["Muse-Glimmer-30B-Q6_K_L"]) == ("muse-glimmer-30b",)
    assert disc._declared_base_ids(HEADERS["Muse-Glimmer-30B-KQuant-17GB-Q4_K_M"]) == ()
    assert disc._declared_base_ids(HEADERS["dflash-kquant"]) == ()
    assert disc._SIZE_TOKEN_RE.search("Qwen3.8-27B")
    assert disc._SIZE_TOKEN_RE.search("Qwen3.6-35B-A3B")
    assert not disc._SIZE_TOKEN_RE.search("Muse-Glimmer")


def test_classification_carries_kind_and_declarations():
    c = disc._classify_meta(HEADERS["Qwen3.8-27B-DFlash2-Q8_0"], basename="d.gguf", path="/d.gguf")
    assert (c.kind, c.drafter_kind, c.declared, c.n_embd) == ("drafter", "dflash2", ("qwen3.8-27b",), 5120)
    v1 = disc._classify_meta(HEADERS["dflash-kquant"], basename="d.gguf", path="/d.gguf")
    assert (v1.drafter_kind, v1.declared) == ("dflash", ())
    t = disc._classify_meta(HEADERS["Qwen3.8-27B-UD-Q6_K_XL"], basename="t.gguf", path="/t.gguf")
    assert (t.kind, t.mtp, t.declared) == ("model", True, ("qwen3.8-27b",))
    assert disc._target_ids(t) == {"qwen3.8-27b"}
    meta_models = disc._classify_meta(HEADERS["Muse-Glimmer-30B-KQuant-17GB-Q4_K_M"],
                                      basename="m.gguf", path="/m.gguf")
    assert disc._target_ids(meta_models) == {"muse-glimmer-hf"}


def test_arch_table_lists_dflash_for_qwen3_5():
    assert arch_table.drafter_serves("dflash", "qwen35") is True
    assert arch_table.drafter_serves("dflash", "muse-glimmer") is True
    assert arch_table.drafter_serves("dflash", "gemma4") is False
    assert "dflash" in arch_table.drafter_arches("qwen3_5")


@pytest.mark.parametrize("prefix", [None, {"incoai__Qwen3.8-27B-DFlash2-GGUF": "zz-",
                                           "incoai__Muse-Glimmer-30B-DFlash2-GGUF": "aa-",
                                           "bartowski__Muse-Glimmer-30B-GGUF": "zz-"}])
def test_mirrored_layout_pairs_across_directories(tmp_path, fake_classify, capsys, prefix):
    models = _scan(_layout(tmp_path, prefix))
    err = capsys.readouterr().err
    assert set(models) == {"qwen3.8-27b-ud-q6", "muse-glimmer-30b-q6",
                           "muse-glimmer-30b-kquant-17gb-q4"}
    qwen = models["qwen3.8-27b-ud-q6"]
    assert _base(qwen.draft_gguf) == "Qwen3.8-27B-DFlash2-Q8_0.gguf"
    assert qwen.speculative is True
    bartowski = models["muse-glimmer-30b-q6"]
    assert _base(bartowski.draft_gguf) == "Muse-Glimmer-30B-DFlash2-Q8_0.gguf"
    assert "superseded for muse-glimmer-30b-q6" in err
    assert "dflash-kquant.gguf -> " in err
    meta_models = models["muse-glimmer-30b-kquant-17gb-q4"]
    assert _base(meta_models.draft_gguf) == "dflash-Muse-Glimmer-30B-Q4_K_M.gguf"
    assert "Qwen3.5-27B-DFlash-Q8_0.gguf" in err
    assert "configure via draft_gguf" in err
    assert err.count("configure via draft_gguf") == 1


def test_v1_drafter_without_declarations_pairs_as_before(tmp_path, fake_classify):
    layout = {k: v for k, v in LAYOUT.items() if "DFlash2" not in k and "zlab" not in k}
    for d, stems in layout.items():
        (tmp_path / d).mkdir()
        for stem in stems:
            (tmp_path / d / f"{stem}.gguf").write_bytes(b"x")
    models = _scan(tmp_path)
    assert _base(models["muse-glimmer-30b-q6"].draft_gguf) == "dflash-kquant.gguf"
    assert _base(models["muse-glimmer-30b-kquant-17gb-q4"].draft_gguf) == "dflash-Muse-Glimmer-30B-Q4_K_M.gguf"
    assert models["qwen3.8-27b-ud-q6"].draft_gguf is None
    assert models["qwen3.8-27b-ud-q6"].speculative is True


def test_declared_mismatch_vetoes_even_a_lone_candidate():
    target = disc._classify_meta(HEADERS["Qwen3.8-27B-UD-Q6_K_XL"],
                                 basename="Qwen3.8-27B-UD-Q6_K_XL.gguf",
                                 path="/m/Qwen3.8-27B-UD-Q6_K_XL.gguf")
    v1 = disc._classify_meta(HEADERS["Qwen3.5-27B-DFlash-Q8_0"],
                             basename="Qwen3.5-27B-DFlash-Q8_0.gguf",
                             path="/m/Qwen3.5-27B-DFlash-Q8_0.gguf")
    from gmlx.config import ModelCfg
    mc = ModelCfg(id="qwen3.8-27b-ud-q6", path=target.path)
    assert disc._drafter_targets(v1, [(mc, target)]) == []
    d2 = disc._classify_meta(HEADERS["Qwen3.8-27B-DFlash2-Q8_0"],
                             basename="Qwen3.8-27B-DFlash2-Q8_0.gguf",
                             path="/m/Qwen3.8-27B-DFlash2-Q8_0.gguf")
    assert disc._drafter_targets(d2, [(mc, target)]) == [mc]
    # an undeclared target falls through to the filename rule
    bare = disc._classify_meta({"general.architecture": "qwen35",
                                "qwen35.nextn_predict_layers": 1,
                                "qwen35.embedding_length": 5120},
                               basename="Qwen3.8-27B-Q4_K_M.gguf",
                               path="/m/Qwen3.8-27B-Q4_K_M.gguf")
    mc2 = ModelCfg(id="qwen3.8-27b-q4", path=bare.path)
    assert disc._drafter_targets(v1, [(mc2, bare)]) == [mc2]
    assert disc._drafter_targets(v1, [(mc2, bare)], declared_only=True) == []


def test_dflash2_supersedes_only_a_non_dflash2_incumbent():
    from gmlx.config import ModelCfg
    target = disc._classify_meta(HEADERS["Muse-Glimmer-30B-Q6_K_L"],
                                 basename="Muse-Glimmer-30B-Q6_K_L.gguf",
                                 path="/m/Muse-Glimmer-30B-Q6_K_L.gguf")
    d2 = disc._classify_meta(HEADERS["Muse-Glimmer-30B-DFlash2-Q8_0"],
                             basename="Muse-Glimmer-30B-DFlash2-Q8_0.gguf",
                             path="/d/Muse-Glimmer-30B-DFlash2-Q8_0.gguf")
    kinds = {"/m/dflash-kquant.gguf": "dflash", "/d/other-DFlash2.gguf": "dflash2"}
    mc = ModelCfg(id="muse-glimmer-30b-q6", path=target.path, draft_gguf="/m/dflash-kquant.gguf")
    assert disc._drafter_targets(d2, [(mc, target)], declared_only=True, drafter_kinds=kinds) == [mc]
    mc.draft_gguf = "/d/other-DFlash2.gguf"
    assert disc._drafter_targets(d2, [(mc, target)], declared_only=True, drafter_kinds=kinds) == []
    v1 = disc._classify_meta(HEADERS["dflash-kquant"], basename="dflash-kquant.gguf",
                             path="/m/dflash-kquant.gguf")
    mc.draft_gguf = "/d/other-DFlash2.gguf"
    assert disc._drafter_targets(v1, [(mc, target)], drafter_kinds=kinds) == []
