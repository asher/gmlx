"""`gmlx pull` with a recommended model's name, with no ref (the picker and
the list), and the flag rules between the forms. The catalog is a fixture
of entries shaped like the real ones; headers, downloads and the picker
are seams, so nothing reaches the network or a terminal."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from gguf import GGUFWriter
from gguf.constants import GGMLQuantizationType as GT

import gmlx.commands.manage as manage
import gmlx.commands.wizard as wizard
import gmlx.load.remote as remote
import gmlx.serve.server as srv
from gmlx.load import catalog
from helpers import banned_words

GiB = 1 << 30


def G(x: float) -> int:
    return int(x * GiB)


def _r(quant, gb, tier="", *, text=None, table=0.0, sf=None):
    return catalog.Rung(quant=quant, bytes=G(gb), path=f"Model-{quant}.gguf", tier=tier,
                        text_bytes=G(text) if text is not None else 0,
                        table_bytes=G(table), streams_from_gib=sf)


def _m(name, rungs, preferred, *, mmproj=0.0, drafter=0.0, moe=False, cross_base=False,
       cost=(("full", 0, 4096.0, 48),), heads=32, ctx=262144):
    return catalog.Model(
        name=name, title=name.title(), repo=f"org/{name}", revision="abc1234",
        rungs=tuple(sorted(rungs, key=lambda r: r.bytes)), preferred=preferred, params="27B",
        blurb="Good at long documents and code.", speed="Quick.", license="apache-2.0",
        license_url="https://example.invalid/LICENSE", moe=moe,
        mmproj="mmproj-F16.gguf" if mmproj else None, mmproj_bytes=G(mmproj),
        drafter=(catalog.Drafter("dflash2", "org/drafter", "def5678", "drafter-Q8_0.gguf",
                                 G(drafter), cross_base) if drafter else None),
        kv_cost=catalog.KvCost(tuple(cost)), heads=heads, trained_ctx=ctx)


# A 27B-shaped dense entry: on a 32 GiB Mac the high tier is over the bar
# and balanced fits with both add-ons.
DENSE = _m("dense", [
    _r("UD-IQ3_S", 11.21), _r("Q4_0", 14.95, "fastest"), _r("UD-Q4_K_M", 15.33, "balanced"),
    _r("UD-Q6_K", 20.47, "high"), _r("Q8_0", 27.05, "xhigh")],
    "high", mmproj=0.86, drafter=1.92)
# A Flash-Next-shaped MoE: every tier streams on a 32 GiB Mac.
FLASHY = _m("flashy", [
    _r("UD-IQ4_XS", 87.25, "fastest", text=87.24, table=26.82, sf=32),
    _r("UD-Q4_K_XL", 103.69, "balanced", text=103.68, table=26.82, sf=32),
    _r("Q8_0", 175.30, "xhigh", text=175.29, table=26.82, sf=64)],
    "balanced", mmproj=0.84, drafter=2.6, moe=True)
# Text only, one tier.
PLAIN = _m("plain", [_r("Q4_K_M", 5.0, "balanced")], "balanced")
# Too big for any stock Mac in the fixture's classes.
HUGE = _m("huge", [_r("Q4_K_M", 300.0, "balanced")], "balanced")
# A tight entry: its tier fits only without the drafter, so the default
# plan has faster replies off.
TIGHT = _m("tight", [_r("Q4_K_M", 17.5, "balanced")], "balanced", mmproj=0.5, drafter=1.5)


class _ScriptIO(wizard.WizardIO):
    def __init__(self, answers):
        super().__init__()
        self.answers = list(answers)
        self.prompts = []

    def _read(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answers.pop(0) if self.answers else ""

    def note(self, msg: str = "") -> None:
        pass


def _mint(path, *, arch="llama") -> bytes:
    w = GGUFWriter(str(path), arch)
    w.add_tensor("plain.f32", np.zeros((4, 16), dtype=np.float32), raw_dtype=GT.F32)
    w.add_tensor("blk.0.attn_q.weight", np.zeros((4, 32), dtype=np.float32), raw_dtype=GT.F32)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    with open(path, "rb") as f:
        return f.read()


_HEADERS = {
    "Model-": {"general.architecture": "qwen35", "general.basename": "Model"},
    "mmproj": {"general.architecture": "clip"},
    "drafter": {"general.architecture": "dflash", "general.basename": "Other",
                "general.base_model.0.name": "Other Base", "dflash.selector_top_k": 16},
}


_REAL_BOX = catalog.box_for_this_mac   # the fixture below replaces it


@pytest.fixture
def cat(monkeypatch, tmp_path):
    """The fixture catalog on a 32 GiB Mac, offline, with a config whose
    model folder is <tmp>/lib. Returns ``(cfg path, lib, downloads)``;
    ``downloads`` collects ``(repo, filename, revision)`` per download."""
    models = {m.name: m for m in (DENSE, FLASHY, PLAIN, HUGE, TIGHT)}
    monkeypatch.setattr(catalog, "MODELS", models)
    monkeypatch.setattr(catalog, "RECOMMENDED", {32: ("dense", "plain"), 8: ("plain",)})
    monkeypatch.setattr(catalog, "box_for_this_mac", lambda: catalog.box_for_class(32))
    monkeypatch.setattr("gmlx.load.memfit.total_ram_bytes", lambda: 32 * GiB)
    monkeypatch.setattr(remote, "hf_list_dir",
                        lambda *a, **k: (_ for _ in ()).throw(remote.RemoteError("offline")))
    payload = _mint(tmp_path / "hdr.gguf")
    monkeypatch.setattr(remote, "http_get_prefix",
                        lambda url, end, *, timeout=30.0: (payload[:end], 1000))
    monkeypatch.setattr(manage, "_disk_free", lambda p: 1 << 40)
    monkeypatch.setattr(manage, "_on_startup_volume", lambda p: False)
    monkeypatch.setattr(manage, "_want_interactive", lambda: False)
    monkeypatch.setattr(srv, "_reload_running", lambda path, skip: None)
    downloads = []

    def fake_download(repo, filename, revision, dest_dir):
        dest = os.path.join(dest_dir, filename)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(b"GGUF")
        downloads.append((repo, filename, revision))
        return dest

    monkeypatch.setattr(manage, "_hf_download", fake_download)

    def classify(path):
        name = os.path.basename(path)
        meta = next((v for k, v in _HEADERS.items() if k in name), None)
        if meta is None:
            return None
        return srv.discovery._classify_meta(dict(meta), basename=name, path=path)

    monkeypatch.setattr(srv.discovery, "classify_gguf", classify)
    lib = tmp_path / "lib"
    lib.mkdir()
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"server:\n  model_dirs:\n    - {lib}\nmodels: {{}}\n")
    return cfg, lib, downloads


def _cfg(path):
    from gmlx.config import load_config
    return load_config(path)


def _tty(monkeypatch, answers=()):
    io = _ScriptIO(answers)
    monkeypatch.setattr(manage, "_want_interactive", lambda: True)
    monkeypatch.setattr(wizard, "make_io", lambda: io)
    return io


def _stream_verdict(monkeypatch, verdict, lines=()):
    """Give the model file's header verdict a streaming box verdict."""
    orig = manage._verdict

    def fake(*a, **k):
        v = orig(*a, **k)
        v["stream"] = {"box": {"verdict": verdict}, "lines": list(lines)}
        return v

    monkeypatch.setattr(manage, "_verdict", fake)


# -- named pulls -----------------------------------------------------------

def test_a_named_pull_downloads_registers_and_names_the_model(cat, capsys):
    cfg, lib, downloads = cat
    assert manage.cmd_pull(["dense", "--config", str(cfg)]) == 0
    assert downloads == [("org/dense", "Model-UD-Q4_K_M.gguf", "abc1234"),
                         ("org/dense", "mmproj-F16.gguf", "abc1234"),
                         ("org/drafter", "drafter-Q8_0.gguf", "def5678")]
    assert (lib / "org__dense" / "Model-UD-Q4_K_M.gguf").exists()
    assert (lib / "org__drafter" / "drafter-Q8_0.gguf").exists()
    c = _cfg(cfg)
    (mid,) = c.models
    assert c.models[mid].mmproj == "org__dense/mmproj-F16.gguf"
    assert c.models[mid].draft_gguf == "org__drafter/drafter-Q8_0.gguf"
    assert c.aliases == {"dense": mid}
    out, err = capsys.readouterr()
    assert out.startswith("Dense  27B  Apache 2.0\ngmlx pull dense\n")
    assert "Fits in memory. Room for about" in out
    assert "image input  0.9 GB\nfaster replies  1.9 GB\n" in out
    assert f"run it:  gmlx chat dense --config {cfg}" in out
    assert "GGUF validation" not in out          # the report prints no box lines
    assert banned_words(out + err) == []
    assert "overrides" not in cfg.read_text()


def test_a_second_tier_repoints_the_name_and_names_the_old_file(cat, capsys):
    cfg, lib, _dl = cat
    assert manage.cmd_pull(["dense", "--config", str(cfg)]) == 0
    capsys.readouterr()
    assert manage.cmd_pull(["dense", "--config", str(cfg), "--quant", "fastest"]) == 0
    c = _cfg(cfg)
    assert set(c.models) == {"model-ud-q4", "model-q4"}
    assert c.aliases == {"dense": "model-q4"}
    out = capsys.readouterr().out
    assert "gmlx pull dense --quant fastest\n" in out
    assert (f"{lib / 'org__dense' / 'Model-UD-Q4_K_M.gguf'} stays on disk; "
            "gmlx rm model-ud-q4 removes it.") in out


def test_the_users_alias_of_the_name_stays_and_the_hint_names_the_id(cat, capsys):
    cfg, lib, _dl = cat
    cfg.write_text(cfg.read_text().replace(
        "models: {}\n", "models:\n  mine:\n    path: old.gguf\n    overrides:\n"
        "      load:\n        kv_bits: 4\naliases:\n  dense: mine\n"))
    (lib / "old.gguf").write_bytes(b"x")
    assert manage.cmd_pull(["dense", "--config", str(cfg)]) == 0
    c = _cfg(cfg)
    assert c.aliases == {"dense": "mine"}
    assert c.models["mine"].overrides == {"load": {"kv_bits": 4}}
    out = capsys.readouterr().out
    assert "dense is already an alias" in out
    assert "run it:  gmlx chat model-ud-q4 --config" in out


def test_a_re_pull_that_adds_only_the_drafter_writes_it_and_hints_short(cat, capsys):
    cfg, lib, downloads = cat
    assert manage.cmd_pull(["dense", "--config", str(cfg), "--no-drafter"]) == 0
    assert len(downloads) == 2 and "draft_gguf" not in cfg.read_text()
    capsys.readouterr()
    assert manage.cmd_pull(["dense", "--config", str(cfg)]) == 0
    assert downloads[-1] == ("org/drafter", "drafter-Q8_0.gguf", "def5678")
    assert _cfg(cfg).models["model-ud-q4"].draft_gguf == "org__drafter/drafter-Q8_0.gguf"
    out = capsys.readouterr().out
    assert "registered drafter-Q8_0.gguf for faster replies on model-ud-q4" in out
    assert "run it:  gmlx chat dense" in out
    assert "\ndownloaded 1 file (2 already on disk):\n" in out
    # Every file on disk: the summary says so and the pull still ends with the name.
    capsys.readouterr()
    assert manage.cmd_pull(["dense", "--config", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "\n3 files already on disk:\n" in out and "downloaded" not in out
    assert "run it:  gmlx chat dense" in out


def test_speculative_false_keeps_the_drafter_out_with_a_note(cat, capsys):
    cfg, lib, _dl = cat
    assert manage.cmd_pull(["dense", "--config", str(cfg), "--no-drafter"]) == 0
    text = cfg.read_text().replace("    mmproj:", "    speculative: false\n    mmproj:")
    cfg.write_text(text)
    assert manage.cmd_pull(["dense", "--config", str(cfg)]) == 0
    assert "draft_gguf" not in cfg.read_text()
    assert ("drafter-Q8_0.gguf is not set for faster replies on model-ud-q4"
            in capsys.readouterr().out)


def test_a_cross_base_drafter_pairs_through_the_catalog(cat, monkeypatch):
    cfg, lib, _dl = cat
    cross = {"dense": _m("dense", DENSE.rungs, "high", mmproj=0.86, drafter=1.92,
                         cross_base=True)}
    monkeypatch.setattr(catalog, "MODELS", cross)
    assert manage.cmd_pull(["dense", "--config", str(cfg)]) == 0
    assert _cfg(cfg).models["model-ud-q4"].draft_gguf == "org__drafter/drafter-Q8_0.gguf"


@pytest.mark.parametrize("body, what", [
    ("models:\n  dense:\n    path: old.gguf\n", "a model"),
    ("models:\n  old:\n    path: old.gguf\nimage_models:\n  dense:\n    path: z.gguf\n"
     "    encoder: old\n    vae: v.gguf\n", "an image model"),
    ("models:\n  old:\n    path: old.gguf\n", "a served assistant"),
])
def test_a_name_that_is_an_id_gets_no_alias(cat, capsys, body, what):
    cfg, lib, _dl = cat
    (lib / "old.gguf").write_bytes(b"x")
    head = f"server:\n  model_dirs:\n    - {lib}\n"
    if what == "a served assistant":
        head += "  assistants:\n    dense:\n      model: old\n"
    cfg.write_text(head + body)
    assert manage.cmd_pull(["dense", "--config", str(cfg)]) == 0
    c = _cfg(cfg)
    assert "dense" not in c.aliases and "model-ud-q4" in c.models
    out = capsys.readouterr().out
    assert f"dense is already {what} in {cfg}; run it as gmlx chat model-ud-q4" in out
    assert "run it:  gmlx chat model-ud-q4" in out


def test_quant_takes_a_tier_or_a_quant_name(cat, monkeypatch, capsys):
    cfg, lib, _dl = cat
    _tty(monkeypatch, ["n"])
    for quant in ("high", "UD-Q6_K", "Q6_K"):
        assert manage.cmd_pull(["dense", "--config", str(cfg), "--quant", quant]) == 130
        assert "gmlx pull dense --quant high\n" in capsys.readouterr().out


def test_toggles_drop_or_add_refs_and_the_command_says_so(cat, capsys):
    cfg, lib, downloads = cat
    assert manage.cmd_pull(["dense", "--config", str(cfg), "--no-vision",
                            "--no-drafter"]) == 0
    assert downloads == [("org/dense", "Model-UD-Q4_K_M.gguf", "abc1234")]
    assert "gmlx pull dense --no-vision --no-drafter\n" in capsys.readouterr().out
    downloads.clear()
    # The tight entry's default leaves faster replies off; --drafter adds it.
    assert manage.cmd_pull(["tight", "--config", str(cfg)]) == 0
    assert [d[1] for d in downloads] == ["Model-Q4_K_M.gguf", "mmproj-F16.gguf"]
    assert "gmlx pull tight\n" in capsys.readouterr().out
    downloads.clear()
    assert manage.cmd_pull(["tight", "--config", str(cfg), "--drafter"]) == 0
    assert [d[1] for d in downloads] == ["Model-Q4_K_M.gguf", "mmproj-F16.gguf",
                                         "drafter-Q8_0.gguf"]
    assert "gmlx pull tight --drafter\n" in capsys.readouterr().out


def test_bad_quant_no_default_and_missing_add_ons_exit_2(cat, capsys):
    cfg, lib, downloads = cat
    assert manage.cmd_pull(["dense", "--config", str(cfg), "--quant", "nope"]) == 2
    err = capsys.readouterr().err
    assert "no quality tier or quant 'nope'" in err
    assert "fastest, balanced, high, xhigh, UD-IQ3_S" in err
    assert manage.cmd_pull(["huge", "--config", str(cfg)]) == 2
    err = capsys.readouterr().err
    assert "huge has no quant that runs on this Mac" in err and "--quant Q4_K_M" in err
    assert manage.cmd_pull(["plain", "--config", str(cfg), "--vision"]) == 2
    assert "plain has no vision tower" in capsys.readouterr().err
    for flag in ("--drafter", "--no-drafter"):
        assert manage.cmd_pull(["plain", "--config", str(cfg), flag]) == 2
        assert f"plain comes without faster replies; drop {flag}" in capsys.readouterr().err
    assert downloads == []


def test_a_dense_quant_over_the_bar_asks_with_a_default_of_no(cat, monkeypatch, capsys):
    cfg, lib, downloads = cat
    # Off a terminal it needs --force.
    assert manage.cmd_pull(["dense", "--config", str(cfg), "--quant", "xhigh"]) == 1
    out, err = capsys.readouterr()
    assert "Larger than this Mac's memory. It may not load." in out
    assert "Pass --force" in err and downloads == []
    # On one, Enter downloads nothing.
    io = _tty(monkeypatch, [""])
    assert manage.cmd_pull(["dense", "--config", str(cfg), "--quant", "xhigh"]) == 130
    assert io.prompts == ["Download anyway? [y/N] "]
    out, err = capsys.readouterr()
    assert "nothing downloaded" in err and downloads == []
    # A yes downloads, with no chat hint for a file that may not load.
    _tty(monkeypatch, ["y"])
    assert manage.cmd_pull(["dense", "--config", str(cfg), "--quant", "xhigh"]) == 0
    out = capsys.readouterr().out
    assert downloads and "run it:" not in out


def test_an_entry_over_the_bar_streams_with_plain_lines(cat, monkeypatch, capsys):
    cfg, lib, downloads = cat
    _stream_verdict(monkeypatch, "streams", ["experts 60 GB", "=> streams"])
    assert manage.cmd_pull(["flashy", "--config", str(cfg)]) == 0
    out, err = capsys.readouterr()
    assert "gmlx pull flashy\n" in out
    assert manage._SLOW_LINE in out
    assert manage._PAGE_CACHE_LINE not in out
    assert "=> streams" not in out and "experts" not in out.replace("stream: experts", "")
    assert [d[1] for d in downloads] == ["Model-UD-IQ4_XS.gguf"]   # no tower, no drafter
    assert "run it:  gmlx chat flashy" in out
    assert banned_words(out + err) == []


def test_a_page_cache_verdict_adds_its_line_and_asks_with_a_default_of_yes(
        cat, monkeypatch, capsys):
    cfg, lib, downloads = cat
    _stream_verdict(monkeypatch, "page_cache")
    assert manage.cmd_pull(["flashy", "--config", str(cfg)]) == 1    # off a terminal
    out, err = capsys.readouterr()
    assert manage._PAGE_CACHE_LINE in out and "Pass --force" in err
    io = _tty(monkeypatch, [""])
    assert manage.cmd_pull(["flashy", "--config", str(cfg)]) == 0
    assert io.prompts == ["Download anyway? [Y/n] "]
    _tty(monkeypatch, ["n"])
    assert manage.cmd_pull(["flashy", "--config", str(cfg)]) == 130
    assert len(downloads) == 1


def test_a_too_big_verdict_refuses_with_the_cannot_stream_line(cat, monkeypatch, capsys):
    cfg, lib, downloads = cat
    _stream_verdict(monkeypatch, "too_big", ["=> cannot stream: 200 GB every token"])
    assert manage.cmd_pull(["flashy", "--config", str(cfg)]) == 1
    err = capsys.readouterr().err
    assert "cannot stream: 200 GB every token; pick a smaller quant" in err
    assert downloads == []
    assert manage.cmd_pull(["flashy", "--config", str(cfg), "--force"]) == 0


def test_vision_on_a_streaming_plan_hints_chat(cat, monkeypatch, capsys):
    cfg, lib, downloads = cat
    _stream_verdict(monkeypatch, "streams")
    assert manage.cmd_pull(["flashy", "--config", str(cfg), "--vision"]) == 0
    out = capsys.readouterr().out
    assert "gmlx pull flashy --vision\n" in out
    assert [d[1] for d in downloads] == ["Model-UD-IQ4_XS.gguf", "mmproj-F16.gguf"]
    assert f"run it:  gmlx chat flashy --config {cfg}" in out


def test_json_prints_the_verdict_once_with_the_plan_and_never_asks(cat, monkeypatch,
                                                                    capsys):
    cfg, lib, downloads = cat
    _stream_verdict(monkeypatch, "page_cache", ["box line"])
    assert manage.cmd_pull(["flashy", "--config", str(cfg), "--json"]) == 1
    out = capsys.readouterr().out
    v = json.loads(out[:out.index("\n}\n") + 2])
    assert v["stream"]["lines"] == ["box line"]
    assert v["plan"]["command"] == "gmlx pull flashy" and v["plan"]["streaming"]
    assert "Pass --force" in v["error"] and downloads == []
    assert out.count('"plan"') == 1
    _tty(monkeypatch, ["n"])      # never consulted
    assert manage.cmd_pull(["flashy", "--config", str(cfg), "--json", "--force"]) == 0
    out = capsys.readouterr().out
    assert out.count('"ref"') == 1 and len(downloads) == 1


def test_list_prints_the_tiers_and_downloads_nothing(cat, capsys):
    cfg, lib, downloads = cat
    assert manage.cmd_pull(["dense", "--list"]) == 0
    out = capsys.readouterr().out
    assert "Quality\n" in out and "room for ~" in out and "default" in out
    assert "needs a " in out        # xhigh cannot run here
    assert "Extras: image input 0.9 GB, faster replies 1.9 GB" in out
    assert manage.cmd_pull(["huge", "--list"]) == 0
    assert "Needs a" in capsys.readouterr().out
    assert manage.cmd_pull(["dense", "--list", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["name"] == "dense" and d["default"]["tier"] == "balanced"
    assert [r["state"] for r in d["rungs"]][-1] == "over"
    assert downloads == []


# -- the flag rules ---------------------------------------------------------

@pytest.mark.parametrize("argv, text", [
    (["--quant", "high"], "--quant goes with a model name"),
    (["--vision"], "--vision goes with a model name"),
    (["--no-drafter"], "--no-drafter goes with a model name"),
    (["--hf-source", "x"], "--hf-source goes with an hf: ref"),
    (["--list", "--to", "/tmp"], "--list downloads nothing, so --to does not go with it"),
    (["--list", "--force", "--no-register"],
     "--list downloads nothing, so --force, --no-register do not go with it"),
    (["--json", "--force"], "--list downloads nothing"),
    (["dense", "--list", "--quant", "high"], "--list prints the quality tiers"),
    (["dense", "--list", "--no-vision"], "--no-vision does not go with it"),
    (["dense", "--list", "--to", "/tmp"], "--to does not go with it"),
    (["dense", "--only", "dit"], "--only goes with an image model name"),
    (["z-image-turbo", "--vision"], "--vision goes with a model name, not an image model name"),
    (["z-image-turbo", "--list", "--quant", "Q8_0"], "--list prints the quants and downloads nothing, so --quant does not go with it"),
    (["hf:org/repo/x.gguf", "--list"], "--list goes alone or with a model name, not an hf: ref"),
    (["hf:org/repo/x.gguf", "--quant", "Q4_0"], "--quant goes with a model name or an image"),
    (["hf:org/repo/x.gguf", "--drafter"], "--drafter goes with a model name, not an hf: ref"),
    (["dense", "hf:org/repo/x.gguf"], "a model name goes alone"),
    (["hf:org/repo/x.gguf", "dense"], "a model name goes alone"),
])
def test_the_flag_rules(cat, capsys, argv, text):
    assert manage.cmd_pull(argv) == 2
    assert text in capsys.readouterr().err


def test_an_unknown_name_points_at_the_list_and_the_closest_name(cat, capsys):
    assert manage.cmd_pull(["dens"]) == 2
    err = capsys.readouterr().err
    assert ("'dens' is not a model name, an image model name, an hf: ref or a URL; "
            "gmlx pull --list shows the names") in err
    assert "Did you mean dense?" in err
    assert manage.cmd_pull(["zzz"]) == 2
    assert "Did you mean" not in capsys.readouterr().err


def test_a_local_file_named_like_a_model_is_a_path(cat, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "dense").write_bytes(b"x")
    assert manage.cmd_pull(["dense"]) == 2
    assert "already on disk" in capsys.readouterr().err


# -- no ref: the list and the picker ------------------------------------------

def test_a_bare_pull_off_a_terminal_lists_every_group(cat, capsys):
    cfg, lib, downloads = cat
    assert manage.cmd_pull([]) == 0
    out = capsys.readouterr().out
    titles = catalog.group_titles(catalog.groups_for(catalog.box_for_this_mac()))
    assert [t for t in titles if t in out] == list(titles)
    assert out.index("Recommended") < out.index("  dense ") < out.index("Also runs in memory")
    assert out.count("gmlx pull ") == 3 and " GB Mac" in out      # the footer's three commands
    assert banned_words(out) == []
    assert manage.cmd_pull(["--list"]) == 0
    assert capsys.readouterr().out == out
    assert manage.cmd_pull(["--list", "-v"]) == 0
    verbose = capsys.readouterr().out
    assert len(verbose.splitlines()) > len(out.splitlines())
    assert "balanced  UD-Q4_K_M" in verbose and "  default" in verbose and "extras:" in verbose
    assert banned_words(verbose) == []
    for argv in (["-v"], ["--verbose", "dense"], ["-v", "hf:org/x/m.gguf"]):
        assert manage.cmd_pull(argv) == 2
        assert capsys.readouterr().err == "error: --verbose goes with --list\n"
    assert downloads == []


def test_a_bare_pull_with_json_lists_every_entry_with_its_group(cat, capsys):
    assert manage.cmd_pull(["--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["ram_bytes"] == 32 * GiB
    by = {m["name"]: m for m in d["models"]}
    assert set(by) == {"dense", "flashy", "plain", "huge", "tight"}
    assert by["dense"]["recommended"] and by["dense"]["group"] == "recommended"
    assert by["flashy"]["group"] == "streams" and by["huge"]["group"] == "too_big"
    assert by["huge"]["default"] is None and by["huge"]["needs_gb"]
    assert by["dense"]["default"]["context_tokens"] > 0
    from gmlx.image import catalog as ic
    im = {e["name"]: e for e in d["image_models"]}
    assert set(im) == set(ic.RECIPES)
    z = im["z-image-turbo"]
    assert z["group"] == "image" and z["needs_gb"] is None and z["image_size"] == "1024x1024"
    assert z["default"]["command"] == "gmlx pull z-image-turbo"
    assert z["default"]["quant"] == ic.default_quants(ic.recipe("z-image-turbo"), 32 * GiB)[0]
    assert {q["quant"] for q in z["quants"]} == set(ic.recipe("z-image-turbo").dit_quants)
    assert all(q["bytes"] > 0 for q in z["quants"])


def test_an_image_name_with_list_prints_its_quants_and_downloads_nothing(cat, capsys):
    _cfg, _lib, downloads = cat
    assert manage.cmd_pull(["z-image-turbo", "--list"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Z-Image-Turbo  1024x1024  Apache 2.0\n")
    assert "\nQuality\n" in out and out.rstrip().endswith("gmlx pull z-image-turbo")
    assert "default" in out and downloads == []
    assert manage.cmd_pull(["z-image-turbo", "--list", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["name"] == "z-image-turbo" and d["group"] == "image"
    assert d["default"]["command"] == "gmlx pull z-image-turbo" and downloads == []


def test_the_list_shows_the_image_models_after_the_chat_models(cat, capsys):
    assert manage.cmd_pull(["--list"]) == 0
    out = capsys.readouterr().out
    assert out.index("Image models\n") > out.index("Recommended for this Mac\n")
    assert "\n  z-image-turbo " in out and banned_words(out) == []


def _fake_pick(monkeypatch, choose):
    """Replace the widget: ``choose(groups)`` gives the Plan, or None."""
    seen = {}

    def fake(groups, **kw):
        seen["groups"] = groups
        seen["kw"] = kw
        return choose(groups)

    monkeypatch.setattr(manage, "_pick", fake)
    return seen


def _default_plan_of(row):
    from gmlx.tui.pick import Plan
    rung = next(r for r in row.rungs if r.default)
    return Plan(row, rung, tuple(t.id for t in row.toggles if t.on))


def test_the_picker_gets_the_chat_groups_then_the_image_group_and_its_choice_runs_the_named_path(
        cat, monkeypatch, capsys):
    cfg, lib, downloads = cat
    _tty(monkeypatch)
    seen = _fake_pick(monkeypatch, lambda groups: _default_plan_of(groups[0].rows[0]))
    assert manage.cmd_pull(["--config", str(cfg)]) == 0
    groups = seen["groups"]
    assert [g.title for g in groups] == ["Recommended for this Mac", "Also runs in memory",
                                         "Runs from the SSD, much slower", "Image models"]
    assert [r.id for r in groups[0].rows] == ["dense", "plain"]
    assert [r.id for r in groups[1].rows] == ["tight"]
    assert [r.id for r in groups[2].rows] == ["flashy"]
    dense = groups[0].rows[0]
    # A tier fits, so no compact rung shows; the tiers over the bar are absent.
    assert [r.tier for r in dense.rungs] == ["fastest", "balanced"]
    assert [t.id for t in dense.toggles] == ["vision", "drafter"]
    assert dense.default_rung == "UD-Q4_K_M" and all(t.on for t in dense.toggles)
    assert seen["kw"]["ram_label"] == "This Mac has 32 GB of memory."
    assert seen["kw"]["explain_line"] == manage.CONTEXT_LINE
    assert seen["kw"]["other_line"] == catalog.OTHER_MODELS_LINE
    label, note = seen["kw"]["rung_label"](dense, dense.rungs[0], ("vision", "drafter"))
    assert label.startswith("room for ~") and note == ""
    flashy = groups[2].rows[0]
    assert seen["kw"]["rung_label"](flashy, flashy.rungs[0], ()) == ("from SSD, slow", "")
    # The footer reads the disk live; the fake download has written 8-byte files by now.
    footer = seen["kw"]["footer"](dense, dense.rungs[1], ("vision",))
    assert footer[0].startswith("Download 16.2 GB (8 B already on disk)   free disk")
    assert footer[1] == "gmlx pull dense --no-drafter"
    assert len(downloads) == 3 and "run it:  gmlx chat dense" in capsys.readouterr().out


def _pick_image(groups, name, quant):
    from gmlx.tui.pick import Plan
    row = next(r for r in groups[-1].rows if r.id == name)
    return Plan(row, next(r for r in row.rungs if r.id == quant), ())


def test_the_footer_fits_80_columns_and_names_the_folder_from_home():
    assert len("  " + manage.CONTEXT_LINE) <= 80
    home = os.path.expanduser("~")
    line = manage._download_line(1 << 30, 0, 100 << 30, os.path.join(home, "models"))
    assert line == "Download 1.0 GB   free disk 100.0 GB at ~/models"
    assert manage._download_line(1 << 30, 1 << 20, 100 << 30, "/Volumes/ext").endswith(
        "(1 MB already on disk)   free disk 100.0 GB at /Volumes/ext")


def test_the_picker_offers_the_image_models_and_runs_the_image_pull(cat, monkeypatch, capsys):
    from gmlx.image import catalog as ic

    cfg, lib, downloads = cat
    _tty(monkeypatch)
    calls = []
    monkeypatch.setattr(manage, "_pull_recipe",
                        lambda a, r: calls.append((r.name, list(a.refs), a.quant)) or 0)
    seen = _fake_pick(monkeypatch, lambda groups: _pick_image(groups, "z-image-turbo", "Q8_0"))
    assert manage.cmd_pull(["--config", str(cfg)]) == 0
    imgs = seen["groups"][-1]
    assert imgs.title == "Image models"
    assert [r.id for r in imgs.rows] == list(ic.RECIPES)     # a 32 GiB Mac makes every image
    z = imgs.rows[0]
    entry = catalog.image_entries(catalog.box_for_class(32))[0][0]
    assert z.params == "1024x1024" and z.toggles == () and z.explain == "" and z.tags == ()
    assert z.use == entry.use and z.license == "Apache 2.0" and z.size_label == catalog.human_gb(entry.download)
    assert [r.tier for r in z.rungs] == [q for q, _n in entry.quants]
    assert all(r.quant == "" and r.state == "resident" and not r.on_disk for r in z.rungs)
    assert z.default_rung == ic.default_quants(ic.recipe("z-image-turbo"), 32 * GiB)[0] == "BF16"
    assert [r.id for r in z.rungs if r.default] == ["BF16"]
    assert seen["kw"]["rung_label"](z, z.rungs[0], ()) == ("", "")
    q8 = next(r for r in z.rungs if r.id == "Q8_0")
    footer = seen["kw"]["footer"](z, q8, ())
    assert footer[0].startswith(f"Download {q8.size_label}   free disk")
    assert footer[1] == "gmlx pull z-image-turbo --quant Q8_0"
    assert seen["kw"]["footer"](z, next(r for r in z.rungs if r.default), ())[1] == "gmlx pull z-image-turbo"
    # The choice ran the image pull path with --quant, and nothing else downloaded.
    assert calls == [("z-image-turbo", ["z-image-turbo"], "Q8_0")] and downloads == []
    # The default quant passes no --quant, so the pull plans as a bare named pull does.
    _fake_pick(monkeypatch, lambda groups: _pick_image(groups, "flux-3-klein", "Q8_0"))
    assert manage.cmd_pull(["--config", str(cfg)]) == 0
    assert calls[-1] == ("flux-3-klein", ["flux-3-klein"], None)


def test_a_picker_cancel_downloads_nothing_and_writes_no_config(cat, monkeypatch, capsys):
    cfg, lib, downloads = cat
    _tty(monkeypatch)
    _fake_pick(monkeypatch, lambda groups: None)
    before = cfg.read_text()
    assert manage.cmd_pull(["--config", str(cfg)]) == 130
    assert "nothing downloaded" in capsys.readouterr().err
    assert downloads == [] and cfg.read_text() == before


def test_a_picker_opens_with_no_network_and_marks_files_on_disk(cat, monkeypatch):
    cfg, lib, _dl = cat
    _tty(monkeypatch)
    (lib / "org__dense").mkdir()
    (lib / "org__dense" / "mmproj-F16.gguf").write_bytes(b"x" * G(0.86))
    seen = _fake_pick(monkeypatch, lambda groups: None)
    monkeypatch.setattr(remote, "http_get_prefix",
                        lambda *a, **k: pytest.fail("no header read before a choice"))
    assert manage.cmd_pull(["--config", str(cfg)]) == 130
    dense = seen["groups"][0].rows[0]
    assert [t.on_disk for t in dense.toggles] == [True, False]


def test_the_header_is_read_once_for_a_picked_plan(cat, monkeypatch):
    cfg, lib, _dl = cat
    _tty(monkeypatch)
    _fake_pick(monkeypatch, lambda groups: _default_plan_of(groups[0].rows[1]))   # plain
    reads = []
    orig = remote.http_get_prefix
    monkeypatch.setattr(remote, "http_get_prefix",
                        lambda url, end, **k: reads.append(url) or orig(url, end, **k))
    assert manage.cmd_pull(["--config", str(cfg)]) == 0
    assert len(reads) == 1


def test_a_disk_refusal_covers_the_whole_plan_once(cat, monkeypatch, capsys):
    cfg, lib, downloads = cat
    monkeypatch.setattr(manage, "_disk_free", lambda p: G(17))
    assert manage.cmd_pull(["dense", "--config", str(cfg)]) == 1
    err = capsys.readouterr().err
    assert err.count("not enough disk space") == 1
    assert "need 18.1 GB, 17.0 GB free" in err and downloads == []
    assert manage.cmd_pull(["dense", "--config", str(cfg), "--force"]) == 0


@pytest.mark.parametrize("free_gb, startup, warns", [
    (20.0, True, True), (20.0, False, False), (40.0, True, False)])
def test_the_swap_warning(cat, monkeypatch, capsys, free_gb, startup, warns):
    cfg, lib, _dl = cat
    monkeypatch.setattr(manage, "_disk_free", lambda p: G(free_gb))
    monkeypatch.setattr(manage, "_on_startup_volume", lambda p: startup)
    assert manage.cmd_pull(["dense", "--config", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert ("macOS needs room for swap" in out) is warns
    if warns:
        assert "Leaves 1.9 GB free." in out


def test_on_startup_volume_judges_a_missing_folder_by_its_ancestor(tmp_path, monkeypatch):
    seen = []
    real_stat = os.stat

    def stat(p, *a, **k):
        seen.append(p)
        return real_stat(p, *a, **k)

    monkeypatch.setattr(os, "stat", stat)
    manage._on_startup_volume(str(tmp_path / "not" / "yet"))
    assert seen[-2:] == [str(tmp_path), "/System/Volumes/Data"]


def test_to_sets_the_folder_alone(cat, tmp_path, capsys):
    cfg, lib, downloads = cat
    other = tmp_path / "other"
    assert manage.cmd_pull(["dense", "--config", str(cfg), "--to", str(other)]) == 0
    assert (other / "Model-UD-Q4_K_M.gguf").exists() and (other / "drafter-Q8_0.gguf").exists()
    out = capsys.readouterr().out
    # Outside model_dirs nothing registers, so the hint names the file.
    assert _cfg(cfg).models == {}
    assert (f"run it:  gmlx chat {other / 'Model-UD-Q4_K_M.gguf'} "
            f"--mmproj {other / 'mmproj-F16.gguf'} "
            f"--draft-gguf {other / 'drafter-Q8_0.gguf'}") in out


def test_no_register_hints_the_file(cat, capsys):
    cfg, lib, _dl = cat
    assert manage.cmd_pull(["dense", "--config", str(cfg), "--no-register",
                            "--no-vision", "--no-drafter"]) == 0
    assert _cfg(cfg).models == {} and "aliases" not in cfg.read_text()
    assert (f"run it:  gmlx chat {lib / 'org__dense' / 'Model-UD-Q4_K_M.gguf'}\n"
            in capsys.readouterr().out)


# -- the destination ----------------------------------------------------------

def _no_default_config(monkeypatch, tmp_path):
    import gmlx.config as cfgmod
    monkeypatch.setattr(cfgmod, "default_config_paths", lambda *a, **k: [])
    monkeypatch.setattr(cfgmod, "default_config_write_path",
                        lambda: tmp_path / "home" / "gmlx.yaml")


def test_no_config_off_a_terminal_keeps_the_error(cat, tmp_path, monkeypatch, capsys):
    _no_default_config(monkeypatch, tmp_path)
    assert manage.cmd_pull(["dense"]) == 2
    assert "needs a server config" in capsys.readouterr().err


def test_no_config_on_a_terminal_asks_for_a_folder_then_writes_the_config(
        cat, tmp_path, monkeypatch, capsys):
    cfg, lib, downloads = cat
    _no_default_config(monkeypatch, tmp_path)
    folder = tmp_path / "models"
    io = _tty(monkeypatch, [str(folder)])
    order = []
    monkeypatch.setattr(manage, "_pick", lambda groups, **kw: order.append("pick")
                        or _default_plan_of(groups[0].rows[1]))
    monkeypatch.setattr(wizard, "_suggested_dir", lambda: "~/suggested")
    assert manage.cmd_pull([]) == 0
    assert io.prompts == ["No model folder is set. Folder for your models? [~/suggested]: "]
    written = tmp_path / "home" / "gmlx.yaml"
    c = _cfg(written)
    assert c.model_dirs == [str(folder)] and set(c.models) == {"model-q4"}
    assert c.aliases == {"plain": "model-q4"}
    assert (folder / "org__plain" / "Model-Q4_K_M.gguf").exists()
    out = capsys.readouterr().out
    assert f"wrote {written}" in out and f"run it:  gmlx chat plain --config {written}" in out


def test_a_cancel_after_the_folder_question_writes_nothing(cat, tmp_path, monkeypatch):
    _no_default_config(monkeypatch, tmp_path)
    _tty(monkeypatch, [str(tmp_path / "models")])
    _fake_pick(monkeypatch, lambda groups: None)
    assert manage.cmd_pull([]) == 130
    assert not (tmp_path / "home").exists()


def test_a_named_config_that_does_not_exist_is_written_there(cat, tmp_path, monkeypatch):
    _no_default_config(monkeypatch, tmp_path)
    folder = tmp_path / "models"
    _tty(monkeypatch, [str(folder)])
    target = tmp_path / "mine" / "cfg.yaml"
    assert manage.cmd_pull(["plain", "--config", str(target)]) == 0
    c = _cfg(target)
    assert c.model_dirs == [str(folder)] and "model-q4" in c.models


def test_a_config_without_model_dirs_gets_the_key(cat, tmp_path, monkeypatch, capsys):
    cfg, lib, _dl = cat
    cfg.write_text("server:\n  port: 8081\nmodels: {}\n")
    folder = tmp_path / "models"
    _tty(monkeypatch, [str(folder)])
    assert manage.cmd_pull(["plain", "--config", str(cfg)]) == 0
    c = _cfg(cfg)
    assert c.model_dirs == [str(folder)] and c.port == 8081 and "model-q4" in c.models
    assert f"set server.model_dirs in {cfg}" in capsys.readouterr().out


def test_no_register_with_no_config_asks_the_folder_and_writes_no_config(
        cat, tmp_path, monkeypatch):
    _no_default_config(monkeypatch, tmp_path)
    folder = tmp_path / "models"
    io = _tty(monkeypatch, [str(folder)])
    assert manage.cmd_pull(["plain", "--no-register"]) == 0
    assert len(io.prompts) == 1
    assert (folder / "org__plain" / "Model-Q4_K_M.gguf").exists()
    assert not (tmp_path / "home").exists()


def test_a_refused_config_path_stops_before_the_picker(cat, tmp_path, monkeypatch, capsys):
    _no_default_config(monkeypatch, tmp_path)
    _tty(monkeypatch)
    _fake_pick(monkeypatch, lambda groups: pytest.fail("the picker must not open"))
    folder = tmp_path / "afolder"
    folder.mkdir()
    assert manage.cmd_pull(["--config", str(folder)]) == 2
    assert "error:" in capsys.readouterr().err


def test_pull_ram_gb_reads_the_catalog_as_another_mac(cat, monkeypatch, capsys):
    monkeypatch.setattr(catalog, "box_for_this_mac", _REAL_BOX)
    monkeypatch.setenv("GMLX_PULL_RAM_GB", "16")
    assert manage.cmd_pull(["--list"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("This Mac has 16 GB of memory.")
    sixteen = catalog.box_for_class(16)
    assert catalog.box_for_this_mac() == sixteen
    assert out.rstrip("\n") == catalog.recommendation_text(
        catalog.groups_for(sixteen), sixteen, images=catalog.image_entries(sixteen)).rstrip("\n")
    assert "32 GB Mac" in out                  # not the fixture's 32 GB listing
    monkeypatch.setenv("GMLX_PULL_RAM_GB", "lots")
    assert manage.cmd_pull(["--list"]) == 2
    assert capsys.readouterr().err == "error: GMLX_PULL_RAM_GB='lots' is not a whole number of GB\n"
    monkeypatch.setenv("GMLX_PULL_RAM_GB", "0")
    assert manage.cmd_pull(["dense", "--list"]) == 2
    assert "not a whole number" in capsys.readouterr().err
