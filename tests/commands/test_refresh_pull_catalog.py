"""The catalog refresh script, offline through its three network seams."""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gmlx.load import catalog

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "refresh_pull_catalog.py"
HOST = "https://huggingface.co"
VOCAB = 32


@pytest.fixture(scope="module")
def refresh():
    spec = importlib.util.spec_from_file_location("_refresh_pull_catalog", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _meta(arch="qwen2", **extra) -> dict:
    m = {
        "general.architecture": arch,
        "general.name": "Model",
        "general.size_label": "4B",
        f"{arch}.embedding_length": 64,
        f"{arch}.block_count": 2,
        f"{arch}.attention.head_count": 4,
        f"{arch}.attention.head_count_kv": 2,
        f"{arch}.feed_forward_length": 128,
        f"{arch}.context_length": 1024,
        f"{arch}.attention.layer_norm_rms_epsilon": 1e-6,
        f"{arch}.attention.key_length": 16,
        f"{arch}.rope.freq_base": 10000.0,
        "tokenizer.ggml.tokens": ["t"] * VOCAB,
    }
    m.update(extra)
    return m


def _gguf(tmp_path, name: str, meta: dict, tensors: dict | None = None) -> bytes:
    """A real GGUF with ``meta`` and zero ``tensors`` (name -> shape)."""
    from gguf import GGMLQuantizationType as GT
    from gguf import GGUFWriter

    path = tmp_path / name.replace("/", "__")
    w = GGUFWriter(str(path), meta["general.architecture"])
    for k, v in meta.items():
        if k == "general.architecture":
            continue
        if isinstance(v, bool):
            w.add_bool(k, v)
        elif isinstance(v, int):
            (w.add_int32 if v < 0 else w.add_uint32)(k, v)
        elif isinstance(v, float):
            w.add_float32(k, v)
        elif isinstance(v, str):
            w.add_string(k, v)
        else:
            w.add_array(k, list(v))
    tensors = tensors if tensors is not None else {"token_embd.weight": (VOCAB, 64)}
    for tname, shape in tensors.items():
        w.add_tensor(tname, np.zeros(shape, dtype=np.float32), raw_dtype=GT.F32)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path.read_bytes()


class Hub:
    """Fake Hugging Face: repos of files and cards, served through the seams."""

    def __init__(self, refresh, monkeypatch):
        self.repos: dict = {}
        self.cards: dict = {}
        self.fetched: list = []
        monkeypatch.setattr(refresh, "hf_list_dir", self.list_dir)
        monkeypatch.setattr(refresh, "fetch_header_bytes", self.fetch)
        monkeypatch.setattr(refresh, "_hf_api_get", self.api_get)
        monkeypatch.setattr(refresh, "_url_ok", lambda url: True)

    def add(self, repo, files: dict, *, license="apache-2.0", license_name="", license_link="",
            gated=False, sha="abcdef1234567890", base=""):
        self.repos[repo] = dict(files)
        self.cards[repo] = {"cardData": {"license": license, "license_name": license_name,
                                         "license_link": license_link, "base_model": base},
                            "gated": gated, "sha": sha, "lastModified": "2026-10-23T00:00:00"}

    def list_dir(self, repo, path, revision="main", **kw):
        if repo not in self.repos:
            from gmlx.load.remote import RemoteError
            raise RemoteError(f"not found on Hugging Face: {repo}")
        return [(p, "file", len(b)) for p, b in self.repos[repo].items()]

    def fetch(self, url, **kw):
        m = re.match(rf"{re.escape(HOST)}/(.+?)/resolve/([^/]+)/(.+)$", url)
        repo, _rev, path = m.groups()
        self.fetched.append((repo, path))
        data = self.repos[repo][path]
        return data, len(data)

    def api_get(self, url):
        repo = url.rsplit("/api/models/", 1)[1]
        return self.cards[repo]


def _seed(name="dense", repo="org/dense", preferred="high", **kw):
    return catalog.Seed(name, name.title(), repo, preferred, "Good at things.", "Quick.", **kw)


@pytest.fixture
def dense_repo(refresh, monkeypatch, tmp_path):
    hub = Hub(refresh, monkeypatch)
    meta = _meta()
    files = {f"Model-{q}.gguf": _gguf(tmp_path, f"Model-{q}.gguf", meta)
             for q in ("UD-IQ3_XXS", "Q4_0", "Q4_K_M", "Q6_K", "Q8_0")}
    files["Model-UD-IQ2_M.gguf"] = _gguf(tmp_path, "Model-UD-IQ2_M.gguf", meta)
    files["mmproj-F16.gguf"] = _gguf(
        tmp_path, "mmproj-F16.gguf",
        {"general.architecture": "clip", "clip.has_llava_projector": True, "general.type": "mmproj"},
        {"mm.0.weight": (8, 8)})
    files["mmproj-BF16.gguf"] = files["mmproj-F16.gguf"]
    hub.add("org/dense", files)
    return hub


# -- what one entry records ----------------------------------------------------------

def test_dense_entry_records_tiers_costs_and_the_tower(refresh, dense_repo, monkeypatch):
    from gmlx.serve.capacity import boot_costs

    seed = _seed(mmproj="mmproj-F16.gguf")
    facts, problems, notes = refresh.record_entry(seed, None, None)
    assert problems == []
    assert facts["revision"] == "abcdef1234567890" and facts["arch"] == "qwen2"
    assert facts["params"] == "4B" and facts["moe"] is False and facts["native_mtp"] is False
    assert facts["n_params"] == VOCAB * 64   # the one tensor the fixture writes
    assert facts["license"] == "apache-2.0" and facts["license_url"] == f"{HOST}/org/dense"
    assert {q: t for q, t, *_ in facts["rungs"]} == {
        "Q4_0": "fastest", "Q4_K_M": "balanced", "Q6_K": "high", "Q8_0": "xhigh",
        "UD-IQ3_XXS": "", "UD-IQ2_M": ""}
    for quant, tier, path, nbytes, text, table, streams_from, head in facts["rungs"]:
        assert path == f"Model-{quant}.gguf"
        assert nbytes == len(dense_repo.repos["org/dense"][path])
        assert 0 < text <= nbytes and table == 0 and streams_from is None and head is False
    assert facts["mmproj_bytes"] == len(dense_repo.repos["org/dense"]["mmproj-F16.gguf"])
    assert facts["drafter"] is None
    # The cost row rebuilds to boot_costs' entries, in both speculation states.
    scans = refresh.scans_for("org/dense", "Model-Q6_K.gguf", facts["revision"], None)
    costs = catalog.kv_costs(catalog.KvCost(facts["kv_cost"]))
    for env in ({}, {"MLX_VLM_GGUF_SPECULATIVE": "1"}):
        _cfg, want, heads = boot_costs(None, env=env, scans=scans)
        assert sorted(map(repr, costs)) == sorted(map(repr, want))
        assert facts["heads"] == heads == 4
    assert facts["trained_ctx"] == 1024
    # The row does not depend on the shell's cache switches.
    monkeypatch.setenv("KV_BITS", "4")
    monkeypatch.setenv("MLX_VLM_GGUF_SPECULATIVE", "1")
    monkeypatch.setenv("GMLX_STREAM_PLE", "0")
    again, _p, _n = refresh.record_entry(seed, None, None)
    assert again == facts
    assert "KV_BITS" not in os.environ
    # The unlisted mmproj-BF16 beside the listed F16 is quiet; the model
    # built from the facts reads them back.
    m = catalog._build(seed, facts)
    assert m.mmproj_bytes == facts["mmproj_bytes"] and catalog.tiers(m)[0].tier == "fastest"
    assert [r.quant for r in catalog.compact_candidates(m)] == ["UD-IQ2_M", "UD-IQ3_XXS"] or \
        sorted(r.quant for r in catalog.compact_candidates(m)) == ["UD-IQ2_M", "UD-IQ3_XXS"]


def test_rope_scaling_leaves_the_trained_context(refresh, monkeypatch, tmp_path):
    hub = Hub(refresh, monkeypatch)
    meta = _meta(**{"qwen2.rope.scaling.type": "yarn", "qwen2.rope.scaling.factor": 4.0,
                    "qwen2.rope.scaling.original_context_length": 1024})
    hub.add("org/yarn", {"M-Q4_K_M.gguf": _gguf(tmp_path, "y.gguf", meta)})
    facts, problems, _ = refresh.record_entry(_seed(repo="org/yarn", preferred="balanced"), None, None)
    assert problems == [] and facts["trained_ctx"] == 1024


def test_table_bytes_come_from_model_plan(refresh, monkeypatch, tmp_path):
    import gmlx.stream.table_stream as ts
    hub = Hub(refresh, monkeypatch)
    monkeypatch.setattr(ts, "streamable_table_names",
                        lambda arch: re.compile(r"ple\.table\.weight") if arch == "qwen2" else None)
    buf = _gguf(tmp_path, "t.gguf", _meta(), {"token_embd.weight": (VOCAB, 64),
                                              "ple.table.weight": (1000, 64)})
    hub.add("org/table", {"M-Q4_K_M.gguf": buf})
    facts, problems, _ = refresh.record_entry(_seed(repo="org/table", preferred="balanced"), None, None)
    assert problems == []
    (_q, _t, _p, nbytes, text, table, *_rest), = facts["rungs"]
    assert table == 1000 * 64 * 4 and text == (1000 * 64 + VOCAB * 64) * 4 < nbytes


def test_streams_from_counts_tables_from_file_as_usable(refresh, monkeypatch):
    import gmlx.stream.plan as sp
    ladder = {8: "too_big", 16: "page_cache", 24: "streams", 32: "streams",
              48: "tables_from_file", 64: "resident"}
    ladder.update({c: "resident" for c in catalog.MEMORY_CLASSES if c not in ladder})
    monkeypatch.setattr(sp, "box_plan",
                        lambda m, *, ram_bytes, ws_bytes, max_buffer_bytes: SimpleNamespace(
                            verdict=ladder[ram_bytes >> 30]))
    streams_from, got = refresh._streams_from(SimpleNamespace())
    assert streams_from == 24 and got[48] == "tables_from_file"
    ladder[48] = "page_cache"
    assert refresh._streams_from(SimpleNamespace())[0] == 64
    ladder.update({c: "too_big" for c in catalog.MEMORY_CLASSES})
    assert refresh._streams_from(SimpleNamespace())[0] is None


# -- tier resolution and the listing --------------------------------------------------------

def test_tier_resolution_prefers_the_uniform_file(refresh):
    groups = refresh.model_files({
        "M-UD-Q6_K.gguf": 10, "M-UD-Q6_K_XL.gguf": 11, "M-UD-Q4_K_XL.gguf": 8,
        "M-UD-IQ4_XS.gguf": 7, "M-UD-Q8_K_XL.gguf": 12, "M-UD-IQ2_M.gguf": 3,
        "mmproj-F16.gguf": 1, "mtp-M.gguf": 1, "MTP/mtp-M-Q8_0.gguf": 1, "M-BF16.gguf": 20,
        "UD-Q2_K_XL/M-UD-Q2_K_XL-00001-of-00002.gguf": 2,
        "UD-Q2_K_XL/M-UD-Q2_K_XL-00002-of-00002.gguf": 2})
    assert set(groups) == {"M-UD-Q6_K.gguf", "M-UD-Q6_K_XL.gguf", "M-UD-Q4_K_XL.gguf",
                           "M-UD-IQ4_XS.gguf", "M-UD-Q8_K_XL.gguf", "M-UD-IQ2_M.gguf",
                           "UD-Q2_K_XL/M-UD-Q2_K_XL-00001-of-00002.gguf"}
    assert groups["UD-Q2_K_XL/M-UD-Q2_K_XL-00001-of-00002.gguf"] == (
        "UD-Q2_K_XL", True, ["UD-Q2_K_XL/M-UD-Q2_K_XL-00001-of-00002.gguf",
                             "UD-Q2_K_XL/M-UD-Q2_K_XL-00002-of-00002.gguf"], 4)
    assert refresh.resolve_tier(groups, "high", False)[:2] == ("M-UD-Q6_K.gguf", "UD-Q6_K")
    assert refresh.resolve_tier(groups, "balanced", True) == (
        "M-UD-Q4_K_XL.gguf", "UD-Q4_K_XL", "balanced falls back to UD-Q4_K_XL")
    assert refresh.resolve_tier(groups, "xhigh", False)[1] == "UD-Q8_K_XL"
    assert refresh.resolve_tier(groups, "fastest", True)[1] == "UD-IQ4_XS"
    assert refresh.resolve_tier(groups, "fastest", False)[1] == "UD-IQ4_XS"   # no Q4_0 here
    dense = refresh.model_files({"M-Q4_0.gguf": 5, "M-IQ4_XS.gguf": 4, "M-Q4_K_M.gguf": 6})
    assert refresh.resolve_tier(dense, "fastest", False)[1] == "Q4_0"
    assert refresh.resolve_tier(dense, "high", False) == (None, None, "high: no file for Q6_K or Q6_K_XL")
    assert refresh.sub4("UD-IQ2_M") and refresh.sub4("PQ2_0-MTP-Q8_0") and refresh.sub4("Q3_K_S")
    assert not refresh.sub4("UD-IQ4_XS") and not refresh.sub4("Q4_K_M")
    bonsai = refresh.model_files({"B-PTQ1_0.gguf": 1, "B-PQ2_0-MTP-Q8_0.gguf": 2, "B-DFlash2-Q8_0.gguf": 1})
    assert refresh.file_for_quant(bonsai, "PQ2_0-MTP-Q8_0") == "B-PQ2_0-MTP-Q8_0.gguf"
    assert refresh.file_for_quant(bonsai, "Q8_0") is None
    assert refresh.file_for_quant(bonsai, "PTQ1_0") == "B-PTQ1_0.gguf"


def test_missing_tier_or_quant_and_gated_repo_fail(refresh, dense_repo, monkeypatch, tmp_path):
    hub = dense_repo
    facts, problems, _ = refresh.record_entry(_seed(quants=("UD-IQ5_Z",)), None, None)
    assert any("compact quant UD-IQ5_Z is not in the listing" in p for p in problems)
    hub.add("org/thin", {"M-Q4_K_M.gguf": hub.repos["org/dense"]["Model-Q4_K_M.gguf"]})
    facts, problems, _ = refresh.record_entry(_seed(repo="org/thin", preferred="high"), None, None)
    assert any("preferred tier high has no file" in p for p in problems)
    hub.add("org/gated", dict(hub.repos["org/thin"]), gated="manual")
    _f, problems, _ = refresh.record_entry(_seed(repo="org/gated", preferred="balanced"), None, None)
    assert any("is gated" in p for p in problems)
    with pytest.raises(refresh.Problem):
        hub.add("org/empty", {"README.md": b"x"})
        refresh.record_entry(_seed(repo="org/empty"), None, None)


# -- companions ----------------------------------------------------------------------------

def _sidecar(tmp_path, name, nextn=1):
    meta = _meta(**{"qwen2.nextn_predict_layers": nextn})
    return _gguf(tmp_path, name, meta, {"blk.2.nextn.eh_proj.weight": (8, 8)})


def test_repo_companions_the_entry_lacks_are_flagged(refresh, dense_repo, monkeypatch, tmp_path):
    hub = dense_repo
    hub.repos["org/dense"]["mtp-Model.gguf"] = _sidecar(tmp_path, "mtp-Model.gguf")
    # No tower and no drafter listed: both kinds are flagged once each.
    _f, problems, _ = refresh.record_entry(_seed(), None, None)
    assert [p for p in problems if "vision tower" in p] == [
        "dense: the repo ships a vision tower the entry does not list: mmproj-BF16.gguf, mmproj-F16.gguf"]
    assert [p for p in problems if "ships a drafter" in p] == [
        "dense: the repo ships a drafter the entry does not list: mtp-Model.gguf"]
    # Ignored files are quiet; a listed tower silences the tower flag.
    _f, problems, _ = refresh.record_entry(
        _seed(mmproj="mmproj-F16.gguf", ignored_files=("mtp-Model.gguf",)), None, None)
    assert problems == []
    # A same-repo sidecar that keeps the base arch reads as a drafter.
    monkeypatch.setattr(refresh.arch_table, "mtp_wired", lambda arch: True)
    monkeypatch.setattr(refresh.arch_table, "drafter_serves", lambda d, a: True)
    facts, problems, _ = refresh.record_entry(
        _seed(mmproj="mmproj-F16.gguf", drafter_repo="org/dense", drafter_file="mtp-Model.gguf"),
        None, None)
    assert problems == []
    assert facts["drafter"] == ("qwen2", "abcdef1234567890", len(hub.repos["org/dense"]["mtp-Model.gguf"]))
    # An entry whose every rung has a head is never asked for a drafter.
    hub.add("org/headed", {"H-Q4_K_M.gguf": _gguf(
        tmp_path, "H-Q4_K_M.gguf", _meta(**{"qwen2.nextn_predict_layers": 1}),
        {"token_embd.weight": (VOCAB, 64), "blk.2.nextn.eh_proj.weight": (8, 8)}),
        "mtp-H.gguf": _sidecar(tmp_path, "mtp-H.gguf")})
    facts, problems, notes = refresh.record_entry(_seed(repo="org/headed", preferred="balanced"), None, None)
    assert problems == [] and facts["native_mtp"] is True and facts["rungs"][0][7] is True
    assert facts["rungs"][0][4] < facts["rungs"][0][3]        # text bytes leave the head block out


def test_a_head_on_an_unwired_arch_is_recorded_without_it(refresh, monkeypatch, tmp_path):
    hub = Hub(refresh, monkeypatch)
    hub.add("org/headed", {"H-Q4_K_M.gguf": _gguf(
        tmp_path, "H.gguf", _meta(**{"qwen2.nextn_predict_layers": 1}),
        {"token_embd.weight": (VOCAB, 64), "blk.2.nextn.eh_proj.weight": (8, 8)})})
    facts, problems, notes = refresh.record_entry(_seed(repo="org/headed", preferred="balanced"), None, None)
    assert facts["native_mtp"] is False and facts["rungs"][0][7] is False
    assert any("no wired target" in n for n in notes)


def _drafter(tmp_path, name, base="Other-Model"):
    meta = {"general.architecture": "dflash", "general.name": "d", "dflash.selector_top_k": 8,
            "general.base_model.0.name": base, "dflash.block_count": 1}
    return _gguf(tmp_path, name, meta, {"d.weight": (4, 4)})


def test_drafter_gates(refresh, dense_repo, monkeypatch, tmp_path):
    hub = dense_repo
    hub.repos["org/dense"]["Model-DFlash2-Q8_0.gguf"] = _drafter(tmp_path, "own.gguf")
    hub.add("org/other", {"Other-DFlash2-Q8_0.gguf": _drafter(tmp_path, "other.gguf")}, sha="9999999999")
    monkeypatch.setattr(refresh.arch_table, "mtp_wired", lambda arch: True)
    monkeypatch.setattr(refresh.arch_table, "drafter_serves", lambda d, a: True)
    # Same repo: the publisher pairs it, whatever it declares.
    facts, problems, _ = refresh.record_entry(_seed(
        mmproj="mmproj-F16.gguf", drafter_repo="org/dense", drafter_file="Model-DFlash2-Q8_0.gguf"), None, None)
    assert problems == [] and facts["drafter"][0] == "dflash2" and facts["drafter"][1] == "abcdef1234567890"
    # Another repo whose declared base misses the model: cross_base or a failure.
    other = _seed(mmproj="mmproj-F16.gguf", drafter_repo="org/other", drafter_file="Other-DFlash2-Q8_0.gguf")
    _f, problems, _ = refresh.record_entry(other, None, None)
    assert any("set cross_base" in p for p in problems)
    facts, problems, _ = refresh.record_entry(
        catalog.Seed(**{**other.__dict__, "cross_base": True}), None, None)
    assert problems == [] and facts["drafter"] == ("dflash2", "9999999999", len(hub.repos["org/other"]["Other-DFlash2-Q8_0.gguf"]))
    # A drafter the table says cannot serve the base, and a base with no wired target.
    monkeypatch.setattr(refresh.arch_table, "drafter_serves", lambda d, a: False)
    _f, problems, _ = refresh.record_entry(_seed(
        mmproj="mmproj-F16.gguf", drafter_repo="org/dense", drafter_file="Model-DFlash2-Q8_0.gguf"), None, None)
    assert any("does not serve" in p for p in problems)
    monkeypatch.setattr(refresh.arch_table, "drafter_serves", lambda d, a: None)
    monkeypatch.setattr(refresh.arch_table, "mtp_wired", lambda arch: False)
    _f, problems, _ = refresh.record_entry(_seed(
        mmproj="mmproj-F16.gguf", drafter_repo="org/dense", drafter_file="Model-DFlash2-Q8_0.gguf"), None, None)
    assert any("no wired speculation target" in p for p in problems)
    # A missing drafter file.
    _f, problems, _ = refresh.record_entry(_seed(
        mmproj="mmproj-F16.gguf", drafter_repo="org/dense", drafter_file="nope.gguf"), None, None)
    assert any("drafter nope.gguf is not in org/dense" in p for p in problems)


# -- licenses ---------------------------------------------------------------------------------

def test_other_license_records_its_name_and_a_resolved_link(refresh, monkeypatch, tmp_path):
    hub = Hub(refresh, monkeypatch)
    buf = _gguf(tmp_path, "l.gguf", _meta())
    hub.add("org/other", {"M-Q4_K_M.gguf": buf}, license="other", license_name="qwen-community-1.0",
            license_link="LICENSE", sha="feedbeef00")
    seed = _seed(repo="org/other", preferred="balanced")
    facts, problems, _ = refresh.record_entry(seed, None, None)
    assert facts["license"] == "other" and facts["license_name"] == "qwen-community-1.0"
    assert facts["license_url"] == f"{HOST}/org/other/blob/feedbeef00/LICENSE"
    assert any("needs a license_note" in p for p in problems)
    noted = catalog.Seed(**{**seed.__dict__, "license_note": "Qwen Community License"})
    _f, problems, _ = refresh.record_entry(noted, None, None)
    assert problems == []
    m = catalog._build(noted, facts)
    assert catalog.license_text(m) == "Qwen Community License"
    # No link on the card: the repo's LICENSE file, else the repo page.
    hub.add("org/file", {"M-Q4_K_M.gguf": buf, "LICENSE.md": b"x"}, license="other", license_name="x")
    facts, _p, _ = refresh.record_entry(_seed(repo="org/file", preferred="balanced", license_note="X"), None, None)
    assert facts["license_url"] == f"{HOST}/org/file/blob/abcdef1234567890/LICENSE.md"
    hub.add("org/page", {"M-Q4_K_M.gguf": buf}, license="other", license_name="x",
            license_link="https://example.invalid/license")
    facts, _p, _ = refresh.record_entry(_seed(repo="org/page", preferred="balanced", license_note="X"), None, None)
    assert facts["license_url"] == "https://example.invalid/license"
    # A card link that does not answer falls back to the base model's LICENSE,
    # then its page, then the repo page.
    hub.add("org/dead", {"M-Q4_K_M.gguf": buf}, license="mit", license_link="LICENSE", base="base/Model")
    answers = {f"{HOST}/base/Model/blob/main/LICENSE"}
    monkeypatch.setattr(refresh, "_url_ok", lambda url: url in answers)
    facts, _p, _ = refresh.record_entry(_seed(repo="org/dead", preferred="balanced"), None, None)
    assert facts["license_url"] == f"{HOST}/base/Model/blob/main/LICENSE"
    answers.clear()
    facts, _p, _ = refresh.record_entry(_seed(repo="org/dead", preferred="balanced"), None, None)
    assert facts["license_url"] == f"{HOST}/org/dead"


# -- the whole run ---------------------------------------------------------------------------

def test_main_prints_facts_and_the_walk_and_check_compares(refresh, dense_repo, monkeypatch, capsys):
    seed = _seed(mmproj="mmproj-F16.gguf")
    monkeypatch.setattr(catalog, "_SEED", (seed,))
    monkeypatch.setattr(catalog, "SEEDS", {seed.name: seed})
    monkeypatch.setattr(catalog, "RECOMMENDED", {8: ("dense",), 16: ("dense",)})
    import gmlx.load.catalog_facts as cf
    monkeypatch.setattr(cf, "FACTS", {})
    # One image recipe rides along: its DiT sizes print as the image facts file.
    from gmlx.image import catalog as ic
    import gmlx.image.catalog_facts as icf
    monkeypatch.setattr(ic, "RECIPES", {"pic": _image_recipe(ic)})
    monkeypatch.setattr(ic, "ENCODER_BYTES", {})
    monkeypatch.setattr(icf, "DIT_BYTES", {})
    dense_repo.add("org/pic", {"pic-Q4_K_M.gguf": b"x" * 40, "pic-Q8_0.gguf": b"x" * 80})
    dense_repo.add("org/enc", {"enc-Q4_K_M.gguf": b"e" * 500, "enc-Q8_0.gguf": b"e" * 1000,
                               "mmproj.gguf": b"m" * 10})
    assert refresh.main([]) == 0
    out = capsys.readouterr().out
    assert out.startswith('"""The measured facts') and "FACTS: dict = {" in out
    assert "# ---- gmlx/image/catalog_facts.py ----" in out and "DIT_BYTES: dict = {" in out
    assert "'dense': {" in out and "('Q6_K', 'high', 'Model-Q6_K.gguf'" in out
    # The printed block is the facts file: it parses to the same dict.
    ns: dict = {}
    exec(out.split("Default plans")[0], ns)
    facts = ns["FACTS"]["dense"]
    assert facts["heads"] == 4
    assert ns["DIT_BYTES"] == {"pic": {"Q4_K_M": 40, "Q8_0": 80}}
    monkeypatch.setattr(icf, "DIT_BYTES", ns["DIT_BYTES"])
    walk = out.split("Default plans", 1)[1]
    assert re.search(r"^ +8 GB  dense +high +Q6_K +\S+ [KMG]B  images +resident +\d+K / \d+K$",
                     walk, re.M), walk
    assert re.search(r"^ +16 GB  dense ", walk, re.M)
    # --check against the printed facts passes and prints no facts block.
    monkeypatch.setattr(cf, "FACTS", {"dense": facts})
    assert refresh.main(["--check"]) == 0
    out = capsys.readouterr().out
    assert "FACTS: dict" not in out and "Default plans" in out
    # A changed size fails --check and names the rung; so does a dead license page.
    rungs = list(map(list, facts["rungs"]))
    rungs[0][3] += 1
    monkeypatch.setattr(cf, "FACTS", {"dense": {**facts, "rungs": tuple(map(tuple, rungs))}})
    assert refresh.main(["--check"]) == 1
    err = capsys.readouterr().err
    assert "FAIL: dense: rung now" in err and "FAIL: dense: rung gone" in err
    monkeypatch.setattr(cf, "FACTS", {"dense": facts})
    # A changed DiT size fails --check too.
    monkeypatch.setattr(icf, "DIT_BYTES", {"pic": {"Q4_K_M": 40, "Q8_0": 81}})
    assert refresh.main(["--check"]) == 1
    assert "FAIL: pic: Q8_0 DiT 81 -> 80" in capsys.readouterr().err
    monkeypatch.setattr(icf, "DIT_BYTES", ns["DIT_BYTES"])
    monkeypatch.setattr(refresh, "_url_ok", lambda url: False)
    assert refresh.main(["--check"]) == 1
    assert "license page" in capsys.readouterr().err
    # A newer main than the pinned revision is a note, not a failure.
    monkeypatch.setattr(refresh, "_url_ok", lambda url: True)
    dense_repo.cards["org/dense"]["sha"] = "0000000000000000"
    assert refresh.main(["--check"]) == 0
    err = capsys.readouterr().err
    assert "note: org/dense: main is at 00000000, newer than the pinned abcdef12" in err
    # --only limits the run; an unknown name runs nothing.
    assert refresh.main(["--only", "nothing"]) == 0
    assert "'dense'" in capsys.readouterr().out   # the recorded facts still print


def _image_recipe(ic, **kw):
    base = ic.recipe("z-image-turbo")
    import dataclasses
    return dataclasses.replace(base, name="pic", dit_repo="org/pic", dit_file="pic-{quant}.gguf",
                               dit_quants=("Q4_K_M", "Q8_0"), encoder_repo="org/enc",
                               encoder_file="enc-{quant}.gguf", encoder_quants=("Q4_K_M", "Q8_0"),
                               mmproj_file="mmproj.gguf", encoder_bytes=1000 + 10, **kw)


def test_record_images_reads_the_dit_sizes_and_checks_the_encoders(refresh, monkeypatch):
    from gmlx.image import catalog as ic

    hub = Hub(refresh, monkeypatch)
    r = _image_recipe(ic)
    monkeypatch.setattr(ic, "RECIPES", {"pic": r})
    monkeypatch.setattr(ic, "ENCODER_BYTES", {"enc-Q4_K_M.gguf": 500 + 10})
    hub.add("org/pic", {"pic-Q4_K_M.gguf": b"x" * 40, "pic-Q8_0.gguf": b"x" * 80, "README.md": b"r"})
    hub.add("org/enc", {"enc-Q4_K_M.gguf": b"e" * 500, "enc-Q8_0.gguf": b"e" * 1000,
                        "mmproj.gguf": b"m" * 10})
    dit, problems = refresh.record_images()
    assert dit == {"pic": {"Q4_K_M": 40, "Q8_0": 80}} and problems == []
    # The printed block parses to the same dict.
    text = refresh.image_facts_text(dit)
    ns: dict = {}
    exec(text, ns)
    assert ns["DIT_BYTES"] == dit
    # A missing DiT file and an encoder whose size drifted are problems.
    del hub.repos["org/pic"]["pic-Q8_0.gguf"]
    hub.repos["org/enc"]["enc-Q8_0.gguf"] = b"e" * 999
    dit, problems = refresh.record_images()
    assert dit == {"pic": {"Q4_K_M": 40}}
    assert problems == ["pic: pic-Q8_0.gguf is not in org/pic",
                        "pic: enc-Q8_0.gguf with its tower is 1009 bytes; the image catalog records 1010"]


def test_compare_images_reports_each_changed_size(refresh):
    old = {"pic": {"Q4_K_M": 40, "Q8_0": 80}}
    assert refresh.compare_images(old, {"pic": {"Q4_K_M": 40, "Q8_0": 80}}) == []
    diffs = refresh.compare_images(old, {"pic": {"Q4_K_M": 41}, "new": {"Q8_0": 1}})
    assert diffs == ["pic: Q4_K_M DiT 40 -> 41", "pic: Q8_0 DiT gone", "new: no recorded DiT sizes"]


def test_compare_reports_each_changed_field(refresh):
    old = {"revision": "a", "heads": 4, "kv_cost": (("full", 0, 1.0, 2),), "rungs": (("Q4_0", "fastest", "p", 1, 1, 0, None, False),)}
    assert refresh.compare("x", old, dict(old)) == []
    new = {**old, "heads": 8, "kv_cost": (("full", 0, 2.0, 2),),
           "rungs": (("Q4_0", "fastest", "p", 2, 1, 0, None, False),)}
    diffs = refresh.compare("x", old, new)
    assert "x: heads 4 -> 8" in diffs and "x: kv_cost changed" in diffs
    assert any(d.startswith("x: rung now") for d in diffs) and any(d.startswith("x: rung gone") for d in diffs)
    assert refresh.compare("x", None, new) == ["x: no recorded facts"]


def test_walk_text_shows_each_name_with_and_without_add_ons(refresh, monkeypatch):
    seed = catalog.Seed("w", "W", "org/w", "high", "B.", "Q.", mmproj="mmproj-F16.gguf",
                        drafter_repo="org/w", drafter_file="d.gguf")
    facts = {"revision": "r", "params": "7B", "n_params": 7 * 10**9, "arch": "qwen2",
             "moe": False, "native_mtp": False,
             "license": "mit", "license_url": "", "license_name": "", "mmproj_bytes": 1 << 29,
             "drafter": ("dflash2", "r", 1 << 29), "heads": 32, "trained_ctx": 131072,
             "kv_cost": (("full", 0, 65536.0, 32),),
             "rungs": (("Q4_K_M", "balanced", "m-Q4_K_M.gguf", 4 << 30, 4 << 30, 0, None, False),
                       ("Q6_K", "high", "m-Q6_K.gguf", 6 << 30, 6 << 30, 0, None, False))}
    monkeypatch.setattr(catalog, "RECOMMENDED", {8: ("w", "missing"), 16: ("w",)})
    text = refresh.walk_text({"w": catalog._build(seed, facts)})
    lines = text.splitlines()
    assert lines[0].startswith("Default plans per memory class")
    assert any("missing" in ln and "not in the catalog" in ln for ln in lines)
    # 8 GiB (bar 4.8): high is over, balanced fits only with the drafter off.
    eight = [ln for ln in lines if ln.startswith("   8 GB  w ")][0]
    assert "balanced" in eight and "images" in eight and "faster replies" not in eight
    assert re.search(r"resident +\d+K / \d+K$", eight)
    # 16 GiB (bar 9.6): high with both add-ons.
    sixteen = [ln for ln in lines if ln.startswith("  16 GB  w ")][0]
    assert "high" in sixteen and "images, faster replies" in sixteen and "resident" in sixteen
