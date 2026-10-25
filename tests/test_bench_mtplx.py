"""The MTPLX arm of bench/serve-bench.py."""

import importlib.util
import io
import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

_BENCH = Path(__file__).resolve().parents[1] / "bench"


def _load(name, path):
    mod_spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(mod_spec)
    # dataclasses look the module up by name while the script runs
    sys.modules[name] = mod
    try:
        mod_spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(name, None)
    return mod


@pytest.fixture(scope="module")
def bench():
    return _load("serve_bench", _BENCH / "serve-bench.py")


# The shape of the published Qwen3.8-27B comparison: two gmlx models on the
# GGUF and one MTPLX pack, each on its own runtime and draft depth.
_MODELS = [
    {"name": "gmlx-dflash-b8", "gguf": "/m/Qwen3.8-27B-Q4_0.gguf",
     "tokenizer": "gguf", "runtimes": ["gmlx"], "mtp_only": True,
     "draft_depths": [7],
     "mtp": {"gmlx": {"draft_gguf": "/m/Qwen3.8-27B-DFlash2-Q4_0.gguf"}}},
    {"name": "mtplx-bare", "mtplx_model": "/m/Qwen3.8-27B-MTPLX-Bare-Speed",
     "tokenizer": "/m/Qwen3.8-27B-Q4_0.gguf", "runtimes": ["mtplx"],
     "mtp_only": True, "draft_depths": [3], "mtp": {"mtplx": {"enabled": True}}},
]


def _specs(bench, tmp_path, **cfg):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"draft_depths": [3], "models": _MODELS, **cfg}))
    args = SimpleNamespace(
        config=str(path), gguf=[], tokenizer=None, chat_dataset=None,
        chat_split=None, chat_max_convs=None, depths=None, concurrency=None,
        draft_depths=None, max_tokens=None, rounds=None, cooldown=None,
        warmup=None, mtp_only=False, requests=8)
    return bench.load_specs(args)


def _plan(bench, specs, cfg, args):
    return {(s.name, rt, label): bench.BUILDERS[rt](args, s, n, 0)
            for s in specs for rt in ("gmlx", "mtplx") if bench.spec_runs_on(s, rt)
            for label, n in bench.arms_for(s, cfg)}


def test_mtplx_arm_serves_the_pack_at_its_draft_depth(bench, tmp_path):
    specs, cfg = _specs(bench, tmp_path)
    args = SimpleNamespace(gmlx_bin="gmlx", mtplx_bin="mtplx", outdir=str(tmp_path),
                           _thinking=cfg["thinking"])
    plan = _plan(bench, specs, cfg, args)

    assert set(plan) == {("gmlx-dflash-b8", "gmlx", "mtp@7"),
                         ("mtplx-bare", "mtplx", "mtp@3")}
    assert plan["gmlx-dflash-b8", "gmlx", "mtp@7"][-2:] == ["--draft-block-size", "8"]
    assert plan["mtplx-bare", "mtplx", "mtp@3"] == [
        "mtplx", "serve", "--model", "/m/Qwen3.8-27B-MTPLX-Bare-Speed",
        "--host", "127.0.0.1", "--port", str(bench.MTPLX_PORT), "--no-auth",
        "--ssd-session-cache", "off", "--no-stats-footer", "--reasoning", "on",
        "--mtp", "--depth", "3"]
    assert bench.server_env("mtplx")[bench.MTPLX_IGNORE_EOS] == "1"

    specs, cfg = _specs(bench, tmp_path, thinking=False)
    args._thinking = cfg["thinking"]
    cmd = _plan(bench, specs, cfg, args)["mtplx-bare", "mtplx", "mtp@3"]
    assert cmd[cmd.index("--reasoning") + 1] == "off"


class _Tok:
    def encode(self, text):
        return text.split()


def _sse(*events):
    return [f"data: {json.dumps(e)}".encode() for e in events] + [b"data: [DONE]"]


def _mtplx_reply(drafted, accepted):
    """An MTPLX stream: thinking, then answer text, then the final chunk with
    usage, timings and mtplx_stats. The streamed text drops the think tags, so
    it re-tokenizes short of the 256 tokens generated."""
    return _sse(
        {"choices": [{"index": 0, "delta": {"reasoning_content": "The user wants"}}]},
        {"choices": [{"index": 0, "delta": {"content": "def add(a, b): return a + b"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
         "usage": {"prompt_tokens": 545, "completion_tokens": 256},
         "mtplx_stats": {"accepted_drafts": accepted, "drafted_tokens": drafted},
         "timings": {"draft_n": drafted, "draft_n_accepted": accepted}})


@pytest.fixture
def server(bench, monkeypatch):
    """No server process: spawn is stubbed and each chat request gets the
    next canned MTPLX reply. Request bodies are recorded."""
    sent, replies = [], []

    class _Response:
        def __init__(self, lines):
            self.lines = lines

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        def iter_lines(self):
            return iter(self.lines)

    def post(url, json=None, stream=False, timeout=None):
        sent.append(json)
        return _Response(replies.pop(0))

    monkeypatch.setattr(bench.requests, "post", post)
    monkeypatch.setattr(bench, "spawn", lambda runtime, cmd, *a, **k:
                        (None, "model", io.StringIO()))
    monkeypatch.setattr(bench, "_assert_model_match", lambda *a: None)
    return SimpleNamespace(sent=sent, replies=replies)


def _code_chats(bench):
    words = "the function returns a sorted list of unique values from the input".split()
    convs = [[{"role": "user", "content": f"Task {i}: " + " ".join(words * 3)},
              {"role": "assistant", "content": f"def task_{i}(xs):\n    "
               + " ".join(words * 4)}] for i in range(8)]
    return bench.ChatCorpus(convs, "code")


def _measure(bench, tmp_path, specs, cfg, runtime, spec_name, arm, n, rnd=0):
    spec = next(s for s in specs if s.name == spec_name)
    args = SimpleNamespace(
        gmlx_bin="gmlx", mtplx_bin="mtplx", outdir=str(tmp_path),
        _thinking=None, _max_conc=1, _corpus=_code_chats(bench),
        ramp_tokens=0, timeout=60.0, startup_timeout=60.0, requests=2, seed=1234)
    results = {}
    bench.measure_block(args, cfg, spec, runtime, arm, n, _Tok(), threading.Lock(),
                        None, results, str(tmp_path), round_idx=rnd)
    return results


def _cfg(cfg):
    cfg.update(depths=[512], concurrency=[1], max_tokens=256, warmup=0,
               cooldown=0.0, cool_to_c=None)
    return cfg


def test_mtplx_reply_scores_its_full_length_and_timings_acceptance(
        bench, tmp_path, server):
    specs, cfg = _specs(bench, tmp_path)
    server.replies += [_mtplx_reply(240, 171), _mtplx_reply(254, 169)]

    results = _measure(bench, tmp_path, specs, _cfg(cfg), "mtplx", "mtplx-bare",
                       "mtp@3", 3)

    samples = results["mtplx-bare", "mtplx", "mtp@3", 512, 1]["samples"]
    assert [s["output_tokens"] for s in samples] == [256, 256]
    assert results["mtplx-bare", "mtplx", "mtp@3"]["round_log_accept"] == [
        {"draft_n": 494, "draft_n_accepted": 340}]


def test_shared_prompts_match_across_models_and_arms(bench, tmp_path, server):
    # The lab key: a public run regenerates the lab run's prompts.
    assert bench.cell_key(1234, 1, "gmlx-dflash-b8", "mtp@7", 4300, 1,
                          shared=True) == "1234:1:4300:1"
    specs, cfg = _specs(bench, tmp_path, shared_prompts=True)
    _cfg(cfg)
    server.replies += [_mtplx_reply(240, 171) for _ in range(6)]

    _measure(bench, tmp_path, specs, cfg, "gmlx", "gmlx-dflash-b8", "mtp@7", 7)
    _measure(bench, tmp_path, specs, cfg, "mtplx", "mtplx-bare", "mtp@3", 3)
    _measure(bench, tmp_path, specs, cfg, "mtplx", "mtplx-bare", "mtp@3", 3, rnd=1)

    gmlx, mtplx, mtplx_r1 = (server.sent[i:i + 2] for i in (0, 2, 4))
    assert [b["messages"] for b in gmlx] == [b["messages"] for b in mtplx]
    assert [b["messages"] for b in mtplx] != [b["messages"] for b in mtplx_r1]


def test_instruction_rows_load_as_one_turn_conversations(bench, monkeypatch):
    rows = [{"instruction": f"Write a function that returns {i}.",
             "response": f"def f():\n    return {i}"} for i in range(6)]

    class _Dataset(list):
        def select(self, idx):
            return _Dataset(self[i] for i in idx)

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(
        load_dataset=lambda name, split: _Dataset(rows)))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")

    chat = bench.load_chat_corpus("ise-uiuc/Magicoder-Evol-Instruct-110K",
                                  "train", 8000)

    assert len(chat) == 6
    assert chat.convs[2] == [
        {"role": "user", "content": "Write a function that returns 2."},
        {"role": "assistant", "content": "def f():\n    return 2"}]


_STOCK_GENERATION = '''import os


def _default_stop_tokens(tokenizer) -> set[int]:
    ids: set[int] = set()
    for attr in ("eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int):
            ids.add(value)
    return ids
'''


def _fake_mtplx_venv(tmp_path, patched):
    """A venv whose mtplx 2.12.2 carries the bundled patch's added lines, or
    is stock."""
    site = tmp_path / "site"
    (site / "mtplx").mkdir(parents=True)
    (site / "mtplx" / "__init__.py").write_text("")
    source = _STOCK_GENERATION
    if patched:
        patch = (_BENCH / "patches" / "mtplx-ignore-eos.patch").read_text()
        added = [ln[1:] for ln in patch.splitlines()
                 if ln.startswith("+") and not ln.startswith("+++")]
        anchor = "    ids: set[int] = set()\n"
        source = source.replace(anchor, anchor + "\n".join(added) + "\n")
    (site / "mtplx" / "generation.py").write_text(source)
    dist = site / "mtplx-2.12.2.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: mtplx\nVersion: 2.12.2\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text(f"#!/bin/sh\nPYTHONPATH='{site}' exec '{sys.executable}' \"$@\"\n")
    mtplx = bin_dir / "mtplx"
    mtplx.write_text(f"#!{python}\nimport sys\n")
    for exe in (python, mtplx):
        os.chmod(exe, 0o755)
    return str(mtplx)


def test_preflight_refuses_an_mtplx_without_the_patch(bench, tmp_path):
    stock = _fake_mtplx_venv(tmp_path / "stock", patched=False)
    with pytest.raises(SystemExit, match="mtplx-ignore-eos.patch"):
        bench.mtplx_preflight(stock)
    assert bench.mtplx_preflight(_fake_mtplx_venv(tmp_path / "ok", patched=True)) == "2.12.2"
