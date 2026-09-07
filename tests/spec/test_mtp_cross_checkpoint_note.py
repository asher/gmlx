"""The cross-checkpoint drafter note: a same-arch drafter whose declared
checkpoint is not the target's gets one warning at pairing time."""

import gmlx.spec.mtp_load as ml
from gmlx.load import loadlog


def test_checkpoint_slug_drops_drafter_words():
    assert ml._checkpoint_slug("DeepSeek V4 Flash 0731 DSpark support",
                               drafter=True) == "deepseekv4flash0731"
    assert ml._checkpoint_slug("DeepSeek V4 Flash 0731") == "deepseekv4flash0731"
    assert ml._checkpoint_slug("downloaded_cache_model_DeepSeek V4 Flash "
                               "Vision Exp") == \
        "downloadedcachemodeldeepseekv4flashvisionexp"


def _run(monkeypatch, d_name, t_name):
    seen = []
    monkeypatch.setattr(loadlog, "warn", lambda m: seen.append(m))
    import gmlx.load.discovery as disc
    monkeypatch.setattr(disc, "header_meta",
                        lambda p: {"name": t_name} if t_name else None)
    noted = ml.note_cross_checkpoint_drafter(
        "/x/drafter.gguf", {"general.name": d_name}, "/x/target.gguf")
    return noted, seen


def test_matching_checkpoint_is_silent(monkeypatch):
    noted, seen = _run(monkeypatch, "DeepSeek V4 Flash 0731 DSpark support",
                       "DeepSeek V4 Flash 0731")
    assert not noted and seen == []


def test_other_checkpoint_warns_once(monkeypatch):
    noted, seen = _run(monkeypatch, "DeepSeek V4 Flash 0731 DSpark support",
                       "downloaded_cache_model_DeepSeek V4 Flash Vision Exp")
    assert noted and len(seen) == 1
    assert "acceptance rate" in seen[0] and "0731" in seen[0]


def test_missing_names_are_silent(monkeypatch):
    assert not _run(monkeypatch, "", "DeepSeek V4 Flash 0731")[0]
    assert not _run(monkeypatch, "DeepSeek V4 Flash 0731 DSpark", None)[0]
    assert not ml.note_cross_checkpoint_drafter(
        "/x/d.gguf", {"general.name": "x"}, None)
